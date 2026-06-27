"""Memory tiers — LOG / STORAGE / WORKING (no LoRA, no context replay)."""

from __future__ import annotations

from enum import Enum


class MemoryTier(str, Enum):
    """Where an utterance lands after kernel apply."""

    LOG = "log"  # every OBS — audit trail (remember literally everything observed)
    STORAGE = "storage"  # sealed PUT — retrievable facts (consolidated memory)
    WORKING = "working"  # ephemeral percept + hot GET reads
    NONE = "none"  # chitchat with no OBS beyond normal turn (still logged on OBS)


def describe_tiers() -> str:
    return (
        "LOG: all turns OBS+SEAL (full audit). "
        "STORAGE: PUT commits (recall by GET). "
        "WORKING: current percept / last read. "
        "No weight updates at runtime."
    )
