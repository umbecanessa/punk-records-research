"""E7 NL parse dataset: utterance features → intent / slot / value chars."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from greenfield.kernel import Kernel
from greenfield.parser.paraphrase_messy import apply_messy_perturbation
from greenfield.parser.paraphrases import generate_paraphrase_batch
from greenfield.runner import load_policy
from greenfield.parser.value_span import extract_plant_value
from greenfield.train.features import e7_intent_id, e7_slot_id, featurize_utterance
from greenfield.train.value_codec import VALUE_CHARS, encode_value
from greenfield.types import Intent


def augment_surface(rng: random.Random, text: str, intent: Intent, *, messy_fraction: float) -> str:
    """Apply E8 messy perturbations — chitchat is noisier (production-like small talk)."""
    if messy_fraction <= 0:
        return text
    p = max(messy_fraction, 0.85) if intent == Intent.CHITCHAT else messy_fraction
    if rng.random() >= p:
        return text
    return apply_messy_perturbation(rng, text)


class NlParseDataset(Dataset):
    """(features, intent_id, slot_id, value_char_ids) from synthetic paraphrases."""

    def __init__(
        self,
        *,
        size: int,
        seed: int,
        stage: str = "B",
        messy_fraction: float = 0.0,
    ):
        self.size = size
        self.seed = seed
        self.stage = stage
        self.messy_fraction = messy_fraction
        self._rows: list[tuple[list[float], int, int, list[int]]] | None = None

    def _build(self) -> list[tuple[list[float], int, int, list[int]]]:
        rng = random.Random(self.seed)
        kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
        state = kernel.genesis()
        batch = generate_paraphrase_batch(rng, self.size)
        rows: list[tuple[list[float], int, int, list[int]]] = []
        for text, intent, slot, value in batch:
            surface = augment_surface(rng, text, intent, messy_fraction=self.messy_fraction)
            fv = featurize_utterance(surface, state, stage=self.stage)
            slot_key = slot if slot in ("fact.name", "fact.code", "fact.item0", "__none__") else "__none__"
            label = value if intent == Intent.PLANT else ""
            guess = extract_plant_value(surface, slot) if intent == Intent.PLANT else ""
            value_ids = encode_value(label or guess) if intent == Intent.PLANT else [0] * VALUE_CHARS
            rows.append(
                (
                    fv.as_list(),
                    e7_intent_id(intent),
                    e7_slot_id(slot_key),
                    value_ids,
                )
            )
        return rows

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if self._rows is None:
            self._rows = self._build()
        x, intent_y, slot_y, value_y = self._rows[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(intent_y, dtype=torch.long),
            torch.tensor(slot_y, dtype=torch.long),
            torch.tensor(value_y, dtype=torch.long),
        )
