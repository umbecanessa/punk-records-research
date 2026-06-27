"""E4 overflow: evict hot facts to cold store; GET retrieves by key or hash."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from greenfield.log_util import append_entry
from greenfield.types import LogEntry, MachineState, OpCode, Policy


FACT_PREFIX = "fact."


@dataclass
class ColdRecord:
    record_hash: str
    key: str
    value: Any
    sealed_epoch: int
    log_anchor: str
    log_segment: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ColdStore:
    records: dict[str, ColdRecord] = field(default_factory=dict)
    key_index: dict[str, str] = field(default_factory=dict)

    def archive(
        self,
        *,
        key: str,
        value: Any,
        log_segment: list[LogEntry],
        sealed_epoch: int,
        log_anchor: str,
    ) -> str:
        seg = [
            {"idx": e.idx, "op": e.op.value, "args": e.args, "entry_hash": e.entry_hash}
            for e in log_segment
        ]
        body = {"key": key, "value": value, "epoch": sealed_epoch, "anchor": log_anchor, "seg": seg}
        record_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        rec = ColdRecord(
            record_hash=record_hash,
            key=key,
            value=value,
            sealed_epoch=sealed_epoch,
            log_anchor=log_anchor,
            log_segment=seg,
        )
        self.records[record_hash] = rec
        self.key_index[key] = record_hash
        return record_hash

    def fetch(self, record_hash: str) -> ColdRecord | None:
        return self.records.get(record_hash)

    def fetch_by_key(self, key: str) -> ColdRecord | None:
        h = self.key_index.get(key)
        return self.records.get(h) if h else None


def fact_keys_in_hot(state: MachineState) -> list[str]:
    return sorted(k for k in state.storage.slots if k.startswith(FACT_PREFIX))


def _put_order(state: MachineState) -> dict[str, int]:
    order: dict[str, int] = {}
    for entry in state.log:
        if entry.op == OpCode.PUT and "key" in entry.args:
            key = str(entry.args["key"])
            if key not in order:
                order[key] = entry.idx
    return order


def pick_eviction_victim(state: MachineState, *, protect: set[str]) -> str | None:
    candidates = [k for k in fact_keys_in_hot(state) if k not in protect]
    if not candidates:
        return None
    order = _put_order(state)
    candidates.sort(key=lambda k: order.get(k, 10**9))
    return candidates[0]


def overflow_evict(state: MachineState, policy: Policy) -> list[str]:
    """Move oldest facts over max_hot_fact_slots into cold store."""
    if policy.max_hot_fact_slots <= 0:
        return []

    evicted: list[str] = []
    protect = set(state.working.hot)
    order = _put_order(state)

    while len(fact_keys_in_hot(state)) > policy.max_hot_fact_slots:
        victim = pick_eviction_victim(state, protect=protect)
        if victim is None:
            break
        value = state.storage.slots[victim]
        anchor = state.storage.meta_seal_hash or "genesis"
        seg_start = order.get(victim, 0)
        log_segment = [e for e in state.log if e.idx >= seg_start and e.idx <= seg_start + 8]
        record_hash = state.cold_store.archive(
            key=victim,
            value=value,
            log_segment=log_segment,
            sealed_epoch=state.storage.meta_epoch,
            log_anchor=anchor,
        )
        state.cold_index[victim] = record_hash
        del state.storage.slots[victim]
        evicted.append(victim)
        state.overflow_evictions += 1

    if evicted:
        append_entry(
            state.log,
            OpCode.SEAL,
            {
                "overflow": True,
                "evicted_keys": evicted,
                "cold_hashes": [state.cold_index[k] for k in evicted],
            },
        )
    return evicted


def promote_cold_to_hot(state: MachineState, policy: Policy, key: str, value: Any) -> None:
    """Promote cold fact into hot storage, evicting oldest hot facts if at capacity."""
    if key in state.storage.slots:
        return
    while len(fact_keys_in_hot(state)) >= policy.max_hot_fact_slots:
        victim = pick_eviction_victim(state, protect={key})
        if victim is None:
            break
        value_v = state.storage.slots[victim]
        anchor = state.storage.meta_seal_hash or "genesis"
        order = _put_order(state)
        seg_start = order.get(victim, 0)
        log_segment = [e for e in state.log if e.idx >= seg_start and e.idx <= seg_start + 8]
        record_hash = state.cold_store.archive(
            key=victim,
            value=value_v,
            log_segment=log_segment,
            sealed_epoch=state.storage.meta_epoch,
            log_anchor=anchor,
        )
        state.cold_index[victim] = record_hash
        del state.storage.slots[victim]
        state.overflow_evictions += 1
    if len(fact_keys_in_hot(state)) < policy.max_hot_fact_slots:
        state.storage.slots[key] = value


def read_slot_with_cold(state: MachineState, key: str, *, cold_hash: str | None = None) -> tuple[Any, str]:
    """Returns (value, source) where source is hot|cold|miss."""
    if cold_hash:
        rec = state.cold_store.fetch(cold_hash)
        if rec is None:
            return None, "miss"
        state.cold_hits += 1
        return rec.value, "cold"

    if key in state.storage.slots:
        return state.storage.slots[key], "hot"

    if key in state.cold_index:
        rec = state.cold_store.fetch(state.cold_index[key])
        if rec is not None:
            state.cold_hits += 1
            return rec.value, "cold"
        rec2 = state.cold_store.fetch_by_key(key)
        if rec2 is not None:
            state.cold_hits += 1
            state.cold_index[key] = rec2.record_hash
            return rec2.value, "cold"

    return None, "miss"
