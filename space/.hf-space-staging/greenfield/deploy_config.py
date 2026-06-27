"""Canonical deploy paths — research builds for release, not demos."""

from __future__ import annotations

import json
from pathlib import Path

_STACK_PATH = Path(__file__).resolve().parent / "deploy" / "stack.v0.json"


def load_stack() -> dict:
    return json.loads(_STACK_PATH.read_text(encoding="utf-8"))


_STACK = load_stack()

DEFAULT_ENCODER: str = _STACK["encoder"]
DEFAULT_RENDERER: str = _STACK["renderer"]
DEFAULT_POLICY: str = _STACK["policy"]
DEFAULT_OVERFLOW_POLICY: str = _STACK["overflow_policy"]
DEFAULT_STAGES: str = _STACK["curriculum_stages"]
USE_LEARNED_ARGS: bool = bool(_STACK.get("use_learned_args", True))
USE_LEARNED_VALUES: bool = bool(_STACK.get("use_learned_values", True))
