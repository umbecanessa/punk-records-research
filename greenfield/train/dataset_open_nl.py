"""E10.1 open phrasing dataset for NlParserModel."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from greenfield.parser.open_phrasing import generate_open_paraphrase_batch
from greenfield.parser.paraphrases import generate_paraphrase_batch
from greenfield.train.features import e7_intent_id, e7_slot_id
from greenfield.train.value_codec import UTTERANCE_LEN, encode_utterance_window
from greenfield.types import Intent


class OpenNlDataset(Dataset):
    """Mix template + open paraphrases (96-char window)."""

    def __init__(self, *, size: int, seed: int, open_fraction: float = 0.6, messy_fraction: float = 0.4):
        self.size = size
        self.seed = seed
        self.open_fraction = open_fraction
        self.messy_fraction = messy_fraction
        self._rows: list[tuple[list[float], int, int]] | None = None

    def _build(self) -> list[tuple[list[float], int, int]]:
        rng = random.Random(self.seed)
        open_n = int(self.size * self.open_fraction)
        template_n = self.size - open_n
        from greenfield.train.dataset_nl import augment_surface

        rows_raw: list[tuple[str, Intent, str, str]] = []
        rows_raw.extend(
            generate_open_paraphrase_batch(rng, open_n, messy_fraction=self.messy_fraction)
        )
        base = generate_paraphrase_batch(rng, template_n)
        for text, intent, slot, value in base:
            surface = augment_surface(rng, text, intent, messy_fraction=self.messy_fraction)
            rows_raw.append((surface, intent, slot, value))
        rng.shuffle(rows_raw)

        rows: list[tuple[list[float], int, int]] = []
        for text, intent, slot, _value in rows_raw:
            utt = encode_utterance_window(text, UTTERANCE_LEN)
            slot_key = slot if slot in ("fact.name", "fact.code", "fact.item0", "__none__") else "__none__"
            rows.append((utt, e7_intent_id(intent), e7_slot_id(slot_key)))
        return rows

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if self._rows is None:
            self._rows = self._build()
        utt, intent_y, slot_y = self._rows[idx]
        return (
            torch.tensor(utt, dtype=torch.float32),
            torch.tensor(intent_y, dtype=torch.long),
            torch.tensor(slot_y, dtype=torch.long),
        )
