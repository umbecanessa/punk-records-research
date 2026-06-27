"""E12 — retention tiers and dynamic key proposals (stateful memory, no weight updates)."""

from greenfield.memory.dynamic_plant import parse_dynamic_utterance
from greenfield.memory.retention import MemoryTier, describe_tiers

__all__ = ["MemoryTier", "describe_tiers", "parse_dynamic_utterance"]
