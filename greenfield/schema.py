"""Storage schema and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from greenfield.types import KernelRevert, OpCode, Storage

FACT_PREFIX = "fact."
USER_PREFIX = "user."
TOOL_PREFIX = "tool."


def slot_type(key: str) -> str:
    if key.startswith(FACT_PREFIX) or key.startswith(USER_PREFIX):
        return "string"
    if key == "goal":
        return "goal"
    if key.startswith(TOOL_PREFIX):
        return "tool_handle"
    if key.startswith("meta."):
        return "meta"
    return "unknown"


def validate_value_type(key: str, value: Any) -> None:
    kind = slot_type(key)
    if kind == "string" and not isinstance(value, str):
        raise KernelRevert(f"expected string for {key}", OpCode.PUT)
    if kind == "goal" and not isinstance(value, dict):
        raise KernelRevert(f"expected goal dict for {key}", OpCode.PUT)
    if kind == "tool_handle" and not isinstance(value, dict):
        raise KernelRevert(f"expected tool handle dict for {key}", OpCode.PUT)
    if kind == "unknown":
        raise KernelRevert(f"unknown slot key {key}", OpCode.PUT)


def check_write_once(storage: Storage, key: str, write_once_keys: list[str]) -> None:
    if key in write_once_keys and key in storage.slots:
        raise KernelRevert(f"write-once violation for {key}", OpCode.PUT)


def canonical_storage(storage: Storage) -> str:
    payload = {
        "slots": dict(sorted(storage.slots.items())),
        "plan": {"steps": storage.plan.steps, "ptr": storage.plan.ptr},
        "meta_epoch": storage.meta_epoch,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_schema(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
