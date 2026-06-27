"""E11b — varied answer phrasing for renderer training (same facts, richer surface)."""

from __future__ import annotations

import random

_NAME_VARIANTS = (
    "My name is {val}.",
    "You can call me {val}.",
    "I'm {val}.",
    "The name I gave you is {val}.",
    "Call me {val}.",
)

_CODE_VARIANTS = (
    "The secret code is {val}.",
    "Your passcode is {val}.",
    "The code you stored is {val}.",
    "I have the code as {val}.",
    "Passcode: {val}.",
)

_ITEM_VARIANTS = (
    "Stored value {val}.",
    "You're carrying {val}.",
    "Your item is {val}.",
    "The item on record is {val}.",
    "I noted your item as {val}.",
)

_VARIANTS = {
    "fact.name": _NAME_VARIANTS,
    "fact.code": _CODE_VARIANTS,
    "fact.item0": _ITEM_VARIANTS,
    "fact.item1": _ITEM_VARIANTS,
    "fact.item2": _ITEM_VARIANTS,
}


def reference_variants(key: str, value: str) -> tuple[str, ...]:
    templates = _VARIANTS.get(key)
    if not templates:
        return (f"{value}.",)
    return tuple(t.format(val=value) for t in templates)


def sample_reference(rng: random.Random, key: str, value: str) -> str:
    return rng.choice(reference_variants(key, value))
