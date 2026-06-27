"""E7c paraphrase corpus coverage."""

from __future__ import annotations

import random

from greenfield.parser.paraphrases import (
    HELDOUT_NAMES,
    TRAIN_NAMES,
    generate_heldout_name_batch,
    generate_paraphrase_batch,
)
from greenfield.types import Intent


def test_train_names_exclude_heldout():
    overlap = set(TRAIN_NAMES) & set(HELDOUT_NAMES)
    assert not overlap


def test_paraphrase_batch_includes_item0():
    rows = generate_paraphrase_batch(random.Random(0), 700)
    slots = {slot for _, _, slot, _ in rows if _ != Intent.CHITCHAT}
    assert "fact.item0" in slots


def test_heldout_batch_uses_unknown_names():
    rows = generate_heldout_name_batch(random.Random(1), 10)
    plants = [val for _, intent, _, val in rows if intent == Intent.PLANT]
    assert all(name in HELDOUT_NAMES for name in plants)
