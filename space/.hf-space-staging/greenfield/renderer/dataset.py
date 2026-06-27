"""Renderer training dataset — template references, not chat."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from greenfield.renderer.core import MAX_RENDER_LEN, encode_text, encode_value
from greenfield.renderer.templates import reference_text
from greenfield.simulator import sample_world
from greenfield.train.features import SLOT_TO_ID


class RenderDataset(Dataset):
    def __init__(self, *, size: int, seed: int):
        self.size = size
        self.seed = seed
        self._rows: list[tuple[int, str, str]] | None = None

    def _build(self) -> list[tuple[int, str, str]]:
        rng = random.Random(self.seed)
        rows: list[tuple[int, str, str]] = []
        while len(rows) < self.size:
            world = sample_world(rng, num_facts=1 + rng.randint(0, 2))
            for key, val in world.facts.items():
                ref = reference_text(key, str(val))
                rows.append((SLOT_TO_ID.get(key, 0), str(val), ref))
                if len(rows) >= self.size:
                    break
        return rows

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._rows is None:
            self._rows = self._build()
        slot_id, value, ref = self._rows[idx]
        slot = torch.tensor(slot_id, dtype=torch.long)
        vids = torch.tensor(encode_value(value), dtype=torch.long)
        target = torch.tensor(encode_text(ref, MAX_RENDER_LEN), dtype=torch.long)
        return slot, vids, target
