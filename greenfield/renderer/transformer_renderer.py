"""E9b — causal transformer renderer (slot + value → answer text)."""

from __future__ import annotations

import torch
import torch.nn as nn

from greenfield.renderer.core import (
    MAX_RENDER_LEN,
    PRINTABLE_COUNT,
    VALUE_CHARS,
    decode_text,
    encode_value,
)
from greenfield.train.features import SLOT_TO_ID


class TransformerRendererModel(nn.Module):
    """Prefix (slot + value chars) + causal decode → rendered string."""

    PREFIX_LEN = 1 + VALUE_CHARS

    def __init__(
        self,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_len: int = MAX_RENDER_LEN,
        num_slots: int | None = None,
    ):
        super().__init__()
        self.max_len = max_len
        n_slot = num_slots or len(SLOT_TO_ID)
        self.slot_emb = nn.Embedding(n_slot, 32)
        self.slot_proj = nn.Linear(32, d_model)
        self.char_emb = nn.Embedding(PRINTABLE_COUNT + 1, d_model)
        self.pos = nn.Embedding(self.PREFIX_LEN + max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, PRINTABLE_COUNT + 1)

    def _prefix(self, slot_ids: torch.Tensor, value_ids: torch.Tensor) -> torch.Tensor:
        slot = self.slot_proj(self.slot_emb(slot_ids)).unsqueeze(1)
        chars = self.char_emb(value_ids.long())
        return torch.cat([slot, chars], dim=1)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((length, length), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        slot_ids: torch.Tensor,
        value_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits (B, max_len, vocab). Teacher-forced when target_ids given."""
        prefix = self._prefix(slot_ids, value_ids)
        b = slot_ids.size(0)
        if target_ids is None:
            target_ids = torch.zeros(b, self.max_len, dtype=torch.long, device=slot_ids.device)
        shifted = torch.cat(
            [
                torch.zeros(b, 1, dtype=torch.long, device=target_ids.device),
                target_ids[:, :-1],
            ],
            dim=1,
        )
        tgt_emb = self.char_emb(shifted)
        seq = torch.cat([prefix, tgt_emb], dim=1)
        positions = torch.arange(seq.size(1), device=seq.device)
        h = self.transformer(
            seq + self.pos(positions).unsqueeze(0),
            mask=self._causal_mask(seq.size(1), seq.device),
        )
        return self.head(h[:, self.PREFIX_LEN :, :])

    @torch.no_grad()
    def generate(self, slot_id: int, value: str, device: torch.device) -> str:
        self.eval()
        slot = torch.tensor([slot_id], device=device)
        vids = torch.tensor([encode_value(value)], device=device)
        ids: list[int] = []
        for step in range(self.max_len):
            if step == 0:
                target = torch.zeros(1, self.max_len, dtype=torch.long, device=device)
            else:
                target = torch.zeros(1, self.max_len, dtype=torch.long, device=device)
                for i, tid in enumerate(ids):
                    target[0, i] = tid
            logits = self.forward(slot, vids, target)
            next_id = int(logits[0, step].argmax().item())
            if next_id <= 0 and step > 0:
                break
            ids.append(next_id)
        return decode_text(ids)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
