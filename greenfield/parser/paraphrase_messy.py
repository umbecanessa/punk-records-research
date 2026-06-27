"""Messy NL perturbations for E8 generalization eval."""

from __future__ import annotations

import random

from greenfield.parser.paraphrases import generate_paraphrase_batch
from greenfield.types import Intent


def apply_messy_perturbation(rng: random.Random, text: str) -> str:
    """Surface-form noise — kernel labels unchanged."""
    t = text.strip()
    if not t:
        return t
    if rng.random() < 0.45:
        t = t[0].upper() + t[1:]
    if rng.random() < 0.35:
        t = rng.choice(["um, ", "well, ", "so ", "hey — ", ""]) + t
    if rng.random() < 0.25:
        t = t.replace(" my ", " My ").replace(" is ", " IS ", 1) if " is " in t.lower() else t
    if "?" not in t and rng.random() < 0.2:
        t = t + rng.choice([".", "!", ""])
    if rng.random() < 0.15:
        t = t + "  "
    return t.strip()


def generate_messy_paraphrase_batch(
    rng: random.Random,
    size: int,
) -> list[tuple[str, Intent, str, str]]:
    """Clean paraphrases → messy surface forms (same oracle labels)."""
    base = generate_paraphrase_batch(rng, size)
    rows: list[tuple[str, Intent, str, str]] = []
    for text, intent, slot, value in base:
        messy = apply_messy_perturbation(rng, text)
        rows.append((messy, intent, slot, value))
    return rows
