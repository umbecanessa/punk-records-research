"""E5c evidence chain: cumulative merkle over hash-linked log."""

from __future__ import annotations

import hashlib

from greenfield.log_util import GENESIS_HASH, resolve_evidence
from greenfield.types import KernelRevert, LogEntry, OpCode


def chain_root(entries: list[LogEntry]) -> str:
    """Rolling hash over entry hashes (merkle-style audit chain)."""
    root = GENESIS_HASH
    for entry in entries:
        root = hashlib.sha256(f"{root}:{entry.entry_hash}".encode()).hexdigest()
    return root


def prefix_root(entries: list[LogEntry], end_idx: int) -> str:
    if end_idx < 0:
        return GENESIS_HASH
    return chain_root(entries[: end_idx + 1])


def attach_chain_root(entries: list[LogEntry], entry: LogEntry) -> None:
    """Store cumulative chain root on entry args (audit metadata)."""
    entry.args["chain_root"] = prefix_root(entries, entry.idx)


def verify_put_evidence(
    entries: list[LogEntry],
    evidence_ref: str,
    *,
    require_merkle: bool,
) -> LogEntry:
    evidence = resolve_evidence(entries, evidence_ref)
    if evidence.op not in (OpCode.OBS, OpCode.RUN):
        raise KernelRevert("evidence must reference OBS or RUN event", OpCode.PUT)
    if require_merkle:
        expected = prefix_root(entries, evidence.idx)
        stored = evidence.args.get("chain_root")
        if stored != expected:
            raise KernelRevert("evidence chain_root mismatch", OpCode.PUT)
    return evidence
