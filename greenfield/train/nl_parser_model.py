"""E10.1 — standalone NL parser (96-char window, decoupled from E6 opcode encoder)."""

from __future__ import annotations

import torch
import torch.nn as nn

from greenfield.train.features import E7_INTENTS, E7_SLOTS
from greenfield.train.nl_transformer import UtteranceTransformerEncoder, count_parameters
from greenfield.train.value_codec import UTTERANCE_LEN


class NlParserModel(nn.Module):
    """Utterance transformer → intent + slot (value via span templates)."""

    def __init__(
        self,
        *,
        utterance_len: int = UTTERANCE_LEN,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.utterance_len = utterance_len
        self.out_dim = out_dim
        self.encoder = UtteranceTransformerEncoder(
            seq_len=utterance_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            out_dim=out_dim,
        )
        self.intent_head = nn.Linear(out_dim, len(E7_INTENTS))
        self.slot_head = nn.Linear(out_dim, len(E7_SLOTS))

    def forward(self, utt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(utt)
        return self.intent_head(h), self.slot_head(h)

    @torch.no_grad()
    def predict_fields(self, utt: torch.Tensor) -> tuple[int, int]:
        self.eval()
        x = utt.unsqueeze(0) if utt.dim() == 1 else utt
        intent_logits, slot_logits = self.forward(x)
        return (
            int(intent_logits.argmax(dim=-1).item()),
            int(slot_logits.argmax(dim=-1).item()),
        )


def load_nl_parser(path, device: torch.device) -> NlParserModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    utterance_len = int(ckpt.get("utterance_len", UTTERANCE_LEN))
    cfg = dict(ckpt.get("model_config") or {})
    model = NlParserModel(utterance_len=utterance_len, **cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
