"""Deterministic kernel: eight opcodes + optional RENDER syscall."""

from __future__ import annotations

import hashlib
from typing import Callable

from greenfield.evidence import attach_chain_root, verify_put_evidence
from greenfield.log_util import append_entry
from greenfield.overflow import ColdStore, fact_keys_in_hot, overflow_evict, promote_cold_to_hot, read_slot_with_cold
from greenfield.schema import (
    canonical_storage,
    check_write_once,
    validate_value_type,
)
from greenfield.types import (
    KernelRevert,
    LogEntry,
    MachineState,
    OpCode,
    OpProposal,
    Policy,
    Storage,
    Working,
)


ToolExecutor = Callable[[str, dict, Storage], dict]


class Kernel:
    def __init__(
        self,
        policy: Policy,
        *,
        tool_executor: ToolExecutor | None = None,
    ):
        self.policy = policy
        self.tool_executor = tool_executor

    def genesis(self) -> MachineState:
        return MachineState(
            storage=Storage(),
            working=Working(),
            log=[],
            checkpoints={},
            gas_used=0,
            cold_store=ColdStore(),
            cold_index={},
            cold_hits=0,
            overflow_evictions=0,
        )

    def _charge_gas(self, state: MachineState, op: OpCode) -> None:
        cost = self.policy.gas_cost.get(op.value, 1)
        state.gas_used += cost
        if state.gas_used > self.policy.gas_per_episode:
            raise KernelRevert("gas budget exceeded", op)

    def apply(self, state: MachineState, proposal: OpProposal) -> MachineState:
        op = proposal.op
        args = dict(proposal.args)
        self._charge_gas(state, op)

        if op == OpCode.OBS:
            return self._obs(state, args)
        if op == OpCode.PUT:
            return self._put(state, args)
        if op == OpCode.GET:
            return self._get(state, args)
        if op == OpCode.FOCUS:
            return self._focus(state, args)
        if op == OpCode.RUN:
            return self._run(state, args)
        if op == OpCode.STEP:
            return self._step(state, args)
        if op == OpCode.SEAL:
            return self._seal(state)
        if op == OpCode.REVERT:
            return self._revert(state, args)
        if op == OpCode.RENDER:
            return self._render(state, args)
        if op == OpCode.FORK:
            return self._fork(state, args)
        if op == OpCode.MERGE:
            return self._merge(state, args)
        if op == OpCode.DELEGATE:
            return self._delegate(state, args)
        raise KernelRevert(f"unknown opcode {op}", op)

    def _log(self, state: MachineState, op: OpCode, args: dict) -> LogEntry:
        entry = append_entry(state.log, op, args)
        if self.policy.evidence_merkle:
            attach_chain_root(state.log, entry)
        return entry

    def _obs(self, state: MachineState, args: dict) -> MachineState:
        source = args.get("source", "system")
        payload = args.get("payload", {})
        state.working.percept = {"source": source, "payload": payload}
        self._log(state, OpCode.OBS, {"source": source, "payload": payload})
        return state

    def _put(self, state: MachineState, args: dict) -> MachineState:
        key = str(args["key"])
        value = args["value"]
        evidence_ref = args.get("evidence_ref")
        if evidence_ref is None:
            raise KernelRevert("PUT requires evidence_ref", OpCode.PUT)
        try:
            evidence = verify_put_evidence(
                state.log,
                str(evidence_ref),
                require_merkle=self.policy.evidence_merkle,
            )
        except KeyError as exc:
            raise KernelRevert(str(exc), OpCode.PUT) from exc

        validate_value_type(key, value)
        check_write_once(state.storage, key, self.policy.fact_write_once)
        state.storage.slots[key] = value
        self._log(
            state,
            OpCode.PUT,
            {"key": key, "value": value, "evidence_ref": evidence.entry_hash},
        )
        return state

    def _get(self, state: MachineState, args: dict) -> MachineState:
        key = str(args["key"])
        cold_hash = args.get("cold_hash")
        value, source = read_slot_with_cold(state, key, cold_hash=str(cold_hash) if cold_hash else None)
        if source == "miss":
            if not self.policy.allow_missing_get:
                raise KernelRevert(f"missing slot {key}", OpCode.GET)
        elif source == "cold" and self.policy.promote_cold_on_get:
            promote_cold_to_hot(state, self.policy, key, value)

        state.working.last_read[key] = value
        if key not in state.working.hot:
            state.working.hot.append(key)
        if len(state.working.hot) > self.policy.max_working_hot:
            state.working.hot = state.working.hot[-self.policy.max_working_hot :]
        self._log(
            state,
            OpCode.GET,
            {"key": key, "hit": source != "miss", "source": source, "cold_hash": cold_hash},
        )
        return state

    def _focus(self, state: MachineState, args: dict) -> MachineState:
        keys = [str(k) for k in args.get("keys", [])]
        state.working.hot = [k for k in keys if k in state.storage.slots][: self.policy.max_working_hot]
        self._log(state, OpCode.FOCUS, {"keys": state.working.hot})
        return state

    def _run(self, state: MachineState, args: dict) -> MachineState:
        handle = str(args["handle"])
        run_args = dict(args.get("args", {}))
        tool_key = f"tool.{handle}"
        if tool_key not in state.storage.slots:
            raise KernelRevert(f"unknown tool handle {handle}", OpCode.RUN)
        if self.policy.plan_required_before_run and not state.storage.plan.steps:
            raise KernelRevert("plan empty before RUN", OpCode.RUN)
        if self.tool_executor is None:
            raise KernelRevert("no tool executor configured", OpCode.RUN)

        result = self.tool_executor(handle, run_args, state.storage)
        slot_writes = dict(result.get("slot_writes", {}))
        state.working.pending.update(slot_writes)
        self._log(
            state,
            OpCode.RUN,
            {"handle": handle, "args": run_args, "slot_writes": slot_writes},
        )
        return state

    def _step(self, state: MachineState, args: dict) -> MachineState:
        if not state.storage.plan.steps:
            raise KernelRevert("plan empty", OpCode.STEP)
        delta = int(args.get("delta", 1))
        new_ptr = state.storage.plan.ptr + delta
        if new_ptr < 0 or new_ptr >= len(state.storage.plan.steps):
            raise KernelRevert("plan pointer out of range", OpCode.STEP)
        state.storage.plan.ptr = new_ptr
        self._log(state, OpCode.STEP, {"delta": delta, "ptr": new_ptr})
        return state

    def _flush_pending_puts(self, state: MachineState) -> None:
        if not state.working.pending:
            return
        last_run = next((e for e in reversed(state.log) if e.op == OpCode.RUN), None)
        if last_run is None:
            raise KernelRevert("pending writes without RUN evidence", OpCode.SEAL)
        evidence_ref = last_run.entry_hash
        pending = dict(state.working.pending)
        state.working.pending.clear()
        for key, value in pending.items():
            self._charge_gas(state, OpCode.PUT)
            key_s = str(key)
            validate_value_type(key_s, value)
            check_write_once(state.storage, key_s, self.policy.fact_write_once)
            state.storage.slots[key_s] = value
            self._log(
                state,
                OpCode.PUT,
                {"key": key_s, "value": value, "evidence_ref": evidence_ref},
            )

    def _seal(self, state: MachineState) -> MachineState:
        self._flush_pending_puts(state)
        canon = canonical_storage(state.storage)
        seal_hash = hashlib.sha256(canon.encode()).hexdigest()
        state.storage.meta_seal_hash = seal_hash
        state.storage.meta_epoch += 1
        state.checkpoints[seal_hash] = state.storage.copy()
        state.working.clear()
        self._log(state, OpCode.SEAL, {"seal_hash": seal_hash, "epoch": state.storage.meta_epoch})
        if self.policy.overflow_on_seal:
            overflow_evict(state, self.policy)
        return state

    def _revert(self, state: MachineState, args: dict) -> MachineState:
        target = args.get("to")
        if target is None:
            if not state.storage.meta_seal_hash:
                state.storage = Storage()
            else:
                snap = state.checkpoints.get(state.storage.meta_seal_hash)
                if snap is None:
                    raise KernelRevert("no checkpoint for current seal", OpCode.REVERT)
                state.storage = snap.copy()
        else:
            snap = state.checkpoints.get(str(target))
            if snap is None:
                raise KernelRevert(f"checkpoint not found: {target}", OpCode.REVERT)
            state.storage = snap.copy()
        state.working.clear()
        self._log(state, OpCode.REVERT, {"to": target or state.storage.meta_seal_hash})
        return state

    def _render(self, state: MachineState, args: dict) -> MachineState:
        self._log(state, OpCode.RENDER, args)
        return state

    def _fork(self, state: MachineState, args: dict) -> MachineState:
        if not self.policy.enable_fork:
            raise KernelRevert("FORK disabled by policy", OpCode.FORK)
        if not state.storage.meta_seal_hash:
            raise KernelRevert("FORK requires sealed state", OpCode.FORK)
        branch_id = str(args.get("branch_id", f"branch-{len(state.checkpoints)}"))
        state.checkpoints[branch_id] = state.storage.copy()
        self._log(state, OpCode.FORK, {"branch_id": branch_id, "from": state.storage.meta_seal_hash})
        return state

    def _merge(self, state: MachineState, args: dict) -> MachineState:
        if not self.policy.enable_merge:
            raise KernelRevert("MERGE disabled by policy", OpCode.MERGE)
        branch_id = str(args["branch_id"])
        snap = state.checkpoints.get(branch_id)
        if snap is None:
            raise KernelRevert(f"branch checkpoint not found: {branch_id}", OpCode.MERGE)
        state.storage = snap.copy()
        self._log(state, OpCode.MERGE, {"branch_id": branch_id})
        return state

    def _delegate(self, state: MachineState, args: dict) -> MachineState:
        if not self.policy.enable_delegate:
            raise KernelRevert("DELEGATE disabled by policy", OpCode.DELEGATE)
        handle = str(args.get("handle", "external"))
        payload = dict(args.get("payload", {}))
        self._log(state, OpCode.DELEGATE, {"handle": handle, "payload": payload})
        return state

    def read_slot(self, state: MachineState, key: str):
        return state.storage.slots.get(key)

    def last_obs_hash(self, state: MachineState) -> str | None:
        for entry in reversed(state.log):
            if entry.op == OpCode.OBS:
                return entry.entry_hash
        return None
