"""E9a — causal transformer over fixed utterance char window (NL front)."""

from __future__ import annotations

import torch
import torch.nn as nn

from greenfield.train.value_codec import VALUE_CHARS, VALUE_VOCAB


class UtteranceTransformerEncoder(nn.Module):
    """Byte/char window → pooled hidden (replaces E7 MLP utterance slice)."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        seq_len: int = VALUE_CHARS,
        vocab: int = VALUE_VOCAB,
        out_dim: int = 96,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, utt: torch.Tensor) -> torch.Tensor:
        """utt: (B, VALUE_CHARS) normalized char ids in [0, 1]."""
        ids = (utt * 96.0).round().long().clamp(0, VALUE_VOCAB - 1)
        b, length = ids.shape
        pos = torch.arange(length, device=ids.device).unsqueeze(0).expand(b, -1)
        h = self.embed(ids) + self.pos(pos)
        h = self.encoder(h)
        return self.proj(h.mean(dim=1))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
