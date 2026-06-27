"""E7 NL dataset — messy chitchat in training mix."""

from __future__ import annotations

import random

from greenfield.train.dataset_nl import NlParseDataset, augment_surface
from greenfield.train.features import E7_ID_TO_INTENT
from greenfield.types import Intent


def test_augment_surface_perturbs_chitchat():
    rng = random.Random(0)
    changed = 0
    for _ in range(20):
        out = augment_surface(rng, "hello there", Intent.CHITCHAT, messy_fraction=1.0)
        assert out
        if out != "hello there":
            changed += 1
    assert changed >= 1


def test_messy_dataset_keeps_chitchat_labels():
    ds = NlParseDataset(size=200, seed=42, messy_fraction=1.0)
    chitchat = 0
    for i in range(len(ds)):
        _, intent_y, _, _ = ds[i]
        if E7_ID_TO_INTENT[int(intent_y)] == Intent.CHITCHAT:
            chitchat += 1
    assert chitchat >= 20
