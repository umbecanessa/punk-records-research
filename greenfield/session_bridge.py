"""E12c — bridge types between greenfield kernel sessions and Lane C carriers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from greenfield.chat_v1 import ChatSessionState


@dataclass
class HybridSessionState:
    """Kernel truth + optional Lane C narrative carrier (future unified runtime)."""

    chat: ChatSessionState
    lane_c_carrier: Any | None = None
    lane_c_block_count: int = 0
    notes: dict = field(default_factory=dict)


def from_chat(session: ChatSessionState) -> HybridSessionState:
    return HybridSessionState(chat=session)
