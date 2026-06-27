"""Hash-linked immutable log + unbuffered console helpers."""

from __future__ import annotations

import hashlib
import json
import sys

from greenfield.types import LogEntry, OpCode

GENESIS_HASH = "0" * 64


def configure_unbuffered() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass


def log(msg: str) -> None:
    print(msg, flush=True)


def _hash_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def append_entry(
    log_entries: list[LogEntry],
    op: OpCode,
    args: dict,
) -> LogEntry:
    prev = log_entries[-1].entry_hash if log_entries else GENESIS_HASH
    idx = len(log_entries)
    body = {"idx": idx, "op": op.value, "args": args, "prev_hash": prev}
    entry_hash = _hash_payload(body)
    entry = LogEntry(idx=idx, op=op, args=args, prev_hash=prev, entry_hash=entry_hash)
    log_entries.append(entry)
    return entry


def resolve_evidence(log_entries: list[LogEntry], evidence_ref: str | int) -> LogEntry:
    if isinstance(evidence_ref, int) or (isinstance(evidence_ref, str) and evidence_ref.isdigit()):
        idx = int(evidence_ref)
        if idx < 0 or idx >= len(log_entries):
            raise KeyError(f"evidence index out of range: {idx}")
        return log_entries[idx]
    for entry in log_entries:
        if entry.entry_hash == evidence_ref or entry.entry_hash.startswith(str(evidence_ref)):
            return entry
    raise KeyError(f"evidence ref not found: {evidence_ref}")
