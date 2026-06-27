"""E10.1 — open conversational phrasing (GPT-like surface forms, kernel labels)."""

from __future__ import annotations

import random

from greenfield.parser.paraphrase_messy import apply_messy_perturbation
from greenfield.parser.paraphrases import HELDOUT_NAMES, TRAIN_ITEMS, TRAIN_NAMES
from greenfield.types import Intent

# Plant — natural ways people introduce memory (not template-drill).
_OPEN_PLANT_NAME = [
    "please call me {val}",
    "you can call me {val}",
    "everyone calls me {val}",
    "for the record my name is {val}",
    "keep in mind my name is {val}",
    "i go by {val}",
    "just so you know i'm {val}",
    "by the way my name is {val}",
]

_OPEN_PLANT_CODE = [
    "the passcode is {val}",
    "store this code for me: {val}",
    "my passcode is {val}",
    "for later the code is {val}",
    "secret code for me is {val}",
]

_OPEN_PLANT_ITEM = [
    "i'm carrying {val}",
    "the thing i have is {val}",
    "keep track of my item: {val}",
    "note that my item is {val}",
    "don't forget my item is {val}",
]

_OPEN_QUERY_NAME = [
    "remind me what my name is",
    "what did i tell you my name was",
    "do you recall my name",
    "what name did i give you",
    "who did i say i was",
]

_OPEN_QUERY_CODE = [
    "what's the code i gave you",
    "remind me of my passcode",
    "what passcode did i set",
    "do you remember my code",
]

_OPEN_QUERY_ITEM = [
    "what item did i mention",
    "remind me what i'm carrying",
    "what did i say my item was",
    "do you recall my item",
]

_OPEN_CHITCHAT = [
    "how are you doing",
    "good to see you",
    "thanks for helping",
    "that's interesting",
    "sounds good to me",
    "let me think",
    "fair enough",
    "nice one",
]

_OPEN_NAME_PREFIXES = tuple(p.replace("{val}", "") for p in _OPEN_PLANT_NAME if "{val}" in p)
# Hand-maintained prefixes for span extraction (longest first in value_span).
OPEN_NAME_PLANT_PREFIXES = (
    "for the record my name is ",
    "keep in mind my name is ",
    "just so you know i'm ",
    "by the way my name is ",
    "everyone calls me ",
    "please call me ",
    "you can call me ",
    "i go by ",
)

OPEN_CODE_PLANT_PREFIXES = (
    "store this code for me: ",
    "secret code for me is ",
    "for later the code is ",
    "the passcode is ",
    "my passcode is ",
)

OPEN_ITEM_PLANT_PREFIXES = (
    "keep track of my item: ",
    "note that my item is ",
    "don't forget my item is ",
    "the thing i have is ",
    "i'm carrying ",
)

OPEN_QUERY_NAME = frozenset(_OPEN_QUERY_NAME)
OPEN_QUERY_CODE = frozenset(_OPEN_QUERY_CODE)
OPEN_QUERY_ITEM = frozenset(_OPEN_QUERY_ITEM)


def sample_open_paraphrase(
    rng: random.Random,
    *,
    intent: Intent,
    slot: str,
    value: str,
) -> str:
    if intent == Intent.PLANT and slot == "fact.name":
        return rng.choice(_OPEN_PLANT_NAME).format(val=value)
    if intent == Intent.PLANT and slot == "fact.code":
        return rng.choice(_OPEN_PLANT_CODE).format(val=value)
    if intent == Intent.PLANT and slot == "fact.item0":
        return rng.choice(_OPEN_PLANT_ITEM).format(val=value)
    if intent == Intent.QUERY and slot == "fact.name":
        return rng.choice(_OPEN_QUERY_NAME)
    if intent == Intent.QUERY and slot == "fact.code":
        return rng.choice(_OPEN_QUERY_CODE)
    if intent == Intent.QUERY and slot == "fact.item0":
        return rng.choice(_OPEN_QUERY_ITEM)
    if intent == Intent.CHITCHAT:
        return rng.choice(_OPEN_CHITCHAT)
    return f"remember {slot} is {value}"


def generate_open_paraphrase_batch(
    rng: random.Random,
    size: int,
    *,
    messy_fraction: float = 0.4,
) -> list[tuple[str, Intent, str, str]]:
    """Open phrasing rows — held-out names/items for generalization."""
    slots_cycle = [
        (Intent.PLANT, "fact.name"),
        (Intent.QUERY, "fact.name"),
        (Intent.PLANT, "fact.code"),
        (Intent.QUERY, "fact.code"),
        (Intent.PLANT, "fact.item0"),
        (Intent.QUERY, "fact.item0"),
        (Intent.CHITCHAT, "__none__"),
    ]
    rows: list[tuple[str, Intent, str, str]] = []
    for i in range(size):
        intent, slot = slots_cycle[i % len(slots_cycle)]
        if intent == Intent.CHITCHAT:
            text = sample_open_paraphrase(rng, intent=intent, slot=slot, value="")
            if messy_fraction > 0 and rng.random() < max(messy_fraction, 0.75):
                text = apply_messy_perturbation(rng, text)
            rows.append((text, intent, slot, ""))
            continue
        if slot == "fact.name":
            val = rng.choice(TRAIN_NAMES + HELDOUT_NAMES)
        elif slot == "fact.item0":
            val = rng.choice(TRAIN_ITEMS)
        else:
            val = str(rng.randint(1000, 9999))
        text = sample_open_paraphrase(rng, intent=intent, slot=slot, value=val)
        if messy_fraction > 0 and rng.random() < messy_fraction:
            text = apply_messy_perturbation(rng, text)
        rows.append((text, intent, slot, val if intent == Intent.PLANT else ""))
    return rows
