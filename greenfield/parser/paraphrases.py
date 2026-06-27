"""Synthetic NL paraphrases for E7 event-parser training."""

from __future__ import annotations

import random

from greenfield.types import Intent

# Held out of training — used by eval_nl_heldout / validate_release gates.
HELDOUT_NAMES = ["Giuseppe", "Francesca", "Wolfgang", "Anastasia", "Natasha"]

TRAIN_NAMES = [
    "Ada",
    "Lin",
    "Sam",
    "Rin",
    "Umberto",
    "Bob",
    "Amy",
    "Leo",
    "Zoe",
    "Max",
    "Eva",
    "Alex",
    "Nina",
    "Omar",
    "Jade",
    "Ivan",
    "Mira",
    "Noah",
    "Luna",
    "Kira",
    "Enzo",
    "Sara",
    "Luca",
    "Maya",
    "Theo",
    "Rosa",
    "Hugo",
    "Ella",
    "Marc",
    "Lily",
    "Owen",
    "Carlos",
    "Elena",
    "Marco",
    "Sofia",
    "Alberto",
    "Roberto",
    "Alexander",
    "Isabella",
    "Benjamin",
    "Charlotte",
]

TRAIN_ITEMS = [
    "brass key",
    "old map",
    "red gem",
    "silver coin",
    "iron ring",
    "gold orb",
    "copper axe",
    "jade box",
]

_PLANT_NAME = [
    "remember my name is {val}",
    "my name is {val}",
    "call me {val}",
    "i am {val}",
    "i'm {val}",
    "please remember my name is {val}",
]

_QUERY_NAME = [
    "what is my name",
    "what's my name",
    "who am i",
    "tell me my name",
    "do you know my name",
]

_PLANT_CODE = [
    "my code is {val}",
    "remember code {val}",
    "the code is {val}",
]

_QUERY_CODE = [
    "what is my code",
    "what's my code",
    "tell me the code",
]

_PLANT_ITEM = [
    "my item is {val}",
    "remember item {val}",
    "i have item {val}",
    "the item is {val}",
]

_QUERY_ITEM = [
    "what is my item",
    "what's my item",
    "what item do i have",
    "tell me my item",
]

_CHITCHAT = [
    "hello there",
    "hi",
    "hey",
    "good morning",
    "nice weather today",
    "just thinking out loud",
    "hmm ok",
    "thanks",
    "thank you",
    "cool",
    "got it",
    "interesting",
    "not sure",
    "okay",
    "sounds good",
]

PATTERNS: list[tuple[Intent, str]] = [
    (Intent.PLANT, "fact.name"),
    (Intent.QUERY, "fact.name"),
    (Intent.PLANT, "fact.code"),
    (Intent.QUERY, "fact.code"),
    (Intent.PLANT, "fact.item0"),
    (Intent.QUERY, "fact.item0"),
    (Intent.CHITCHAT, "__none__"),
]


def sample_paraphrase(
    rng: random.Random,
    *,
    intent: Intent,
    slot: str,
    value: str,
) -> str:
    if intent == Intent.PLANT and slot == "fact.name":
        tpl = rng.choice(_PLANT_NAME)
        return tpl.format(val=value)
    if intent == Intent.QUERY and slot == "fact.name":
        return rng.choice(_QUERY_NAME) + rng.choice(["", "?", "?"])
    if intent == Intent.PLANT and slot == "fact.code":
        tpl = rng.choice(_PLANT_CODE)
        return tpl.format(val=value)
    if intent == Intent.QUERY and slot == "fact.code":
        return rng.choice(_QUERY_CODE) + rng.choice(["", "?"])
    if intent == Intent.PLANT and slot == "fact.item0":
        tpl = rng.choice(_PLANT_ITEM)
        return tpl.format(val=value)
    if intent == Intent.QUERY and slot == "fact.item0":
        return rng.choice(_QUERY_ITEM) + rng.choice(["", "?", "?"])
    if intent == Intent.CHITCHAT:
        return rng.choice(_CHITCHAT)
    return f"remember {slot} is {value}"


def _value_for_slot(
    rng: random.Random,
    slot: str,
    *,
    names: list[str],
    items: list[str],
) -> str:
    if slot == "fact.name":
        return rng.choice(names)
    if slot == "fact.item0":
        return rng.choice(items)
    return str(rng.randint(1000, 9999))


def generate_paraphrase_batch(
    rng: random.Random,
    size: int,
    *,
    names: list[str] | None = None,
    items: list[str] | None = None,
) -> list[tuple[str, Intent, str, str]]:
    names = names or TRAIN_NAMES
    items = items or TRAIN_ITEMS
    rows: list[tuple[str, Intent, str, str]] = []
    for i in range(size):
        intent, slot = PATTERNS[i % len(PATTERNS)]
        if intent == Intent.CHITCHAT:
            text = sample_paraphrase(rng, intent=intent, slot=slot, value="")
            rows.append((text, intent, slot, ""))
            continue
        val = _value_for_slot(rng, slot, names=names, items=items)
        text = sample_paraphrase(rng, intent=intent, slot=slot, value=val)
        rows.append((text, intent, slot, val))
    return rows


def generate_heldout_name_batch(
    rng: random.Random,
    size: int,
    *,
    names: list[str] | None = None,
) -> list[tuple[str, Intent, str, str]]:
    """Plant/query paraphrases with names never seen in TRAIN_NAMES."""
    heldout = names or HELDOUT_NAMES
    rows: list[tuple[str, Intent, str, str]] = []
    for i in range(size):
        name = heldout[i % len(heldout)]
        if i % 2 == 0:
            intent, slot = Intent.PLANT, "fact.name"
            val = name
        else:
            intent, slot = Intent.QUERY, "fact.name"
            val = ""
        text = sample_paraphrase(rng, intent=intent, slot=slot, value=val)
        rows.append((text, intent, slot, val))
    return rows
