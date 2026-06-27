"""Encoders: oracle (labels) and noisy (training stress)."""

from __future__ import annotations

import random

from greenfield.kernel import Kernel
from greenfield.types import EpisodeEvent, Intent, MachineState, OpCode, OpProposal


class OracleEncoder:
    """Rule-based teacher: maps structured events → valid opcode sequences."""

    def propose(self, event: EpisodeEvent, state: MachineState, kernel: Kernel) -> list[OpProposal]:
        obs = OpProposal(
            op=OpCode.OBS,
            args={"source": event.source, "payload": {"intent": event.intent.value, **event.payload}},
        )
        steps: list[OpProposal] = [obs]

        if event.intent == Intent.PLANT:
            key = event.slot_key()
            value = event.slot_value()
            if key is None or value is None:
                return steps
            steps.append(
                OpProposal(
                    op=OpCode.PUT,
                    args={"key": key, "value": str(value), "evidence_ref": "__LAST_OBS__"},
                )
            )
            if event.requires_seal:
                steps.append(OpProposal(op=OpCode.SEAL, args={}))

        elif event.intent == Intent.QUERY:
            key = event.slot_key()
            if key:
                steps.append(OpProposal(op=OpCode.GET, args={"key": key}))
                steps.append(
                    OpProposal(
                        op=OpCode.RENDER,
                        args={"mode": "answer", "keys": [key], "max_tokens": 32},
                    )
                )

        elif event.intent == Intent.CHITCHAT:
            if event.requires_seal:
                steps.append(OpProposal(op=OpCode.SEAL, args={}))

        elif event.intent == Intent.TOOL_PLANT:
            key = event.slot_key()
            value = event.slot_value()
            handle = str(event.payload.get("handle", "plant_fact"))
            plan = event.payload.get("plan", ["run"])
            state.storage.plan.steps = list(plan)
            state.storage.plan.ptr = 0
            steps = [obs]
            steps.append(
                OpProposal(
                    op=OpCode.RUN,
                    args={"handle": handle, "args": {"key": key, "value": str(value)}},
                )
            )
            if event.requires_seal:
                steps.append(OpProposal(op=OpCode.SEAL, args={}))

        elif event.intent == Intent.DISTRACTOR_PUT:
            key = event.slot_key()
            value = event.slot_value()
            if key and value is not None:
                steps.append(
                    OpProposal(
                        op=OpCode.PUT,
                        args={"key": key, "value": str(value), "evidence_ref": "__LAST_OBS__"},
                    )
                )
            if event.requires_seal:
                steps.append(OpProposal(op=OpCode.SEAL, args={}))

        return steps

    def resolve_evidence(self, state: MachineState, kernel: Kernel, proposal: OpProposal) -> OpProposal:
        if proposal.args.get("evidence_ref") == "__LAST_OBS__":
            ref = kernel.last_obs_hash(state)
            if ref is None:
                raise ValueError("no OBS in log for evidence")
            args = dict(proposal.args)
            args["evidence_ref"] = ref
            return OpProposal(op=proposal.op, args=args)
        return proposal

    def materialize(self, event: EpisodeEvent, op: OpCode) -> OpProposal:
        """Build args for a predicted opcode (oracle fills structured fields)."""
        if op == OpCode.OBS:
            return OpProposal(
                op=OpCode.OBS,
                args={"source": event.source, "payload": {"intent": event.intent.value, **event.payload}},
            )
        if op == OpCode.PUT:
            key = event.slot_key()
            value = event.slot_value()
            return OpProposal(
                op=OpCode.PUT,
                args={"key": key or "fact.name", "value": str(value or ""), "evidence_ref": "__LAST_OBS__"},
            )
        if op == OpCode.GET:
            key = event.slot_key() or "fact.name"
            return OpProposal(op=OpCode.GET, args={"key": key})
        if op == OpCode.FOCUS:
            key = event.slot_key()
            keys = [key] if key else []
            return OpProposal(op=OpCode.FOCUS, args={"keys": keys})
        if op == OpCode.RUN:
            handle = str(event.payload.get("handle", "plant_fact"))
            key = event.slot_key()
            value = event.slot_value()
            return OpProposal(
                op=OpCode.RUN,
                args={"handle": handle, "args": {"key": key, "value": str(value)}},
            )
        if op == OpCode.STEP:
            return OpProposal(op=OpCode.STEP, args={"delta": 1})
        if op == OpCode.SEAL:
            return OpProposal(op=OpCode.SEAL, args={})
        if op == OpCode.REVERT:
            return OpProposal(op=OpCode.REVERT, args={})
        if op == OpCode.RENDER:
            key = event.slot_key() or "fact.name"
            return OpProposal(op=OpCode.RENDER, args={"mode": "answer", "keys": [key], "max_tokens": 32})
        raise ValueError(f"unknown opcode {op}")


class NoisyEncoder:
    """Wraps oracle with random invalid ops to measure revert rate."""

    def __init__(self, oracle: OracleEncoder, *, noise_rate: float = 0.15, rng: random.Random | None = None):
        self.oracle = oracle
        self.noise_rate = noise_rate
        self.rng = rng or random.Random()

    def propose(self, event: EpisodeEvent, state: MachineState, kernel: Kernel) -> list[OpProposal]:
        steps = self.oracle.propose(event, state, kernel)
        if not steps or self.rng.random() >= self.noise_rate:
            return steps
        junk = OpProposal(
            op=self.rng.choice([OpCode.PUT, OpCode.RUN, OpCode.STEP]),
            args={
                "key": "fact.name",
                "value": "HALLUCINATED",
                "evidence_ref": "deadbeef",
                "handle": "missing",
                "delta": 99,
            },
        )
        insert_at = self.rng.randint(1, max(1, len(steps)))
        out = list(steps)
        out.insert(insert_at, junk)
        return out
