"""Renderer implementations — read storage only, never write."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from greenfield.renderer.templates import reference_text
from greenfield.train.features import SLOT_TO_ID
from greenfield.types import MachineState

PRINTABLE_OFFSET = 32
PRINTABLE_COUNT = 95
PAD_ID = 0
MAX_RENDER_LEN = 48
VALUE_CHARS = 12


def encode_text(text: str, max_len: int = MAX_RENDER_LEN) -> list[int]:
    ids = [PAD_ID] * max_len
    for i, ch in enumerate(text[:max_len]):
        code = ord(ch)
        if PRINTABLE_OFFSET <= code < PRINTABLE_OFFSET + PRINTABLE_COUNT:
            ids[i] = code - PRINTABLE_OFFSET + 1
    return ids


def encode_value(value: str, max_len: int = VALUE_CHARS) -> list[int]:
    return encode_text(value, max_len=max_len)[:max_len]


def decode_text(ids: list[int]) -> str:
    chars = []
    for tid in ids:
        if tid <= 0:
            continue
        chars.append(chr(tid - 1 + PRINTABLE_OFFSET))
    return "".join(chars).strip()


class Renderer(ABC):
    @abstractmethod
    def render(self, state: MachineState, render_args: dict) -> str:
        raise NotImplementedError


class StubRenderer(Renderer):
    def render(self, state: MachineState, render_args: dict) -> str:
        keys = render_args.get("keys", [])
        parts = []
        for key in keys:
            key_s = str(key)
            val = state.storage.slots.get(key_s)
            if val is None:
                val = state.working.last_read.get(key_s)
            if val is not None:
                parts.append(str(val))
        return " ".join(parts) if parts else "[unknown]"


class TemplateRenderer(Renderer):
    def render(self, state: MachineState, render_args: dict) -> str:
        keys = render_args.get("keys", [])
        mode = str(render_args.get("mode", "answer"))
        parts = []
        for key in keys:
            key_s = str(key)
            val = state.storage.slots.get(key_s)
            if val is None:
                val = state.working.last_read.get(key_s)
            if val is not None:
                parts.append(reference_text(str(key), str(val), mode=mode))
        return " ".join(parts) if parts else "[unknown]"


class ByteRendererModel(nn.Module):
    """Slot + value chars -> rendered text (per-char CE)."""

    def __init__(self, *, hidden: int = 256, max_len: int = MAX_RENDER_LEN, num_slots: int | None = None):
        super().__init__()
        self.max_len = max_len
        n_slot = num_slots or len(SLOT_TO_ID)
        self.slot_emb = nn.Embedding(n_slot, 32)
        self.char_emb = nn.Embedding(PRINTABLE_COUNT + 1, 16)
        flat = 32 + VALUE_CHARS * 16
        self.backbone = nn.Sequential(
            nn.Linear(flat, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, max_len * (PRINTABLE_COUNT + 1))

    def forward(self, slot_ids: torch.Tensor, value_ids: torch.Tensor) -> torch.Tensor:
        b = slot_ids.size(0)
        s = self.slot_emb(slot_ids)
        c = self.char_emb(value_ids.long()).reshape(b, -1)
        h = self.backbone(torch.cat([s, c], dim=-1))
        return self.head(h).view(b, self.max_len, PRINTABLE_COUNT + 1)

    @torch.no_grad()
    def generate(self, slot_id: int, value: str, device: torch.device) -> str:
        self.eval()
        v = torch.tensor([encode_value(value)], device=device)
        sid = torch.tensor([slot_id], device=device)
        logits = self.forward(sid, v)
        ids = logits.argmax(dim=-1)[0].tolist()
        return decode_text(ids)


@dataclass
class LearnedRenderer(Renderer):
    model: ByteRendererModel
    device: torch.device

    def render(self, state: MachineState, render_args: dict) -> str:
        keys = render_args.get("keys", [])
        parts = []
        for key in keys:
            key_s = str(key)
            val = state.storage.slots.get(key_s)
            if val is None:
                val = state.working.last_read.get(key_s)
            if val is None:
                continue
            slot_id = SLOT_TO_ID.get(key_s, SLOT_TO_ID["__none__"])
            if slot_id >= self.model.slot_emb.num_embeddings:
                slot_id = SLOT_TO_ID["__none__"]
            parts.append(self.model.generate(slot_id, str(val), self.device))
        return " ".join(parts) if parts else "[unknown]"
