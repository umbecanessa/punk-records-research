"""Small MLP event → opcode (+ optional slot) classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from greenfield.train.features import INTENTS, MAX_STEP, OPCODES, SOURCES, SLOTS, STAGES
from greenfield.train.value_codec import VALUE_CHARS, VALUE_VOCAB


class EventEncoderModel(nn.Module):
    """
    Maps structured event context → opcode logits (+ optional slot logits).

    Categorical fields use embeddings; numeric fields concat at the end.
    """

    def __init__(
        self,
        *,
        hidden: int = 128,
        dropout: float = 0.1,
        predict_slot: bool = False,
        predict_value: bool = False,
        num_intents: int | None = None,
        num_sources: int | None = None,
        num_slots: int | None = None,
        num_stages: int | None = None,
        num_opcodes: int | None = None,
        max_step: int | None = None,
    ):
        super().__init__()
        self.predict_slot = predict_slot
        self.predict_value = predict_value
        n_intent = num_intents or len(INTENTS)
        n_source = num_sources or len(SOURCES)
        n_slot = num_slots or len(SLOTS)
        n_stage = num_stages or len(STAGES)
        n_op = num_opcodes or len(OPCODES)
        n_step = max_step or MAX_STEP
        none_op_id = n_op

        self.intent_emb = nn.Embedding(n_intent, 16)
        self.source_emb = nn.Embedding(n_source, 8)
        self.slot_emb = nn.Embedding(n_slot, 16)
        self.stage_emb = nn.Embedding(n_stage, 8)
        self.step_emb = nn.Embedding(n_step, 8)
        self.prev_op_emb = nn.Embedding(n_op + 1, 16)

        self.in_dim = 16 + 8 + 16 + 8 + 8 + 16 + 3 + VALUE_CHARS
        self.backbone = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.op_head = nn.Linear(hidden, n_op)
        self.slot_head = nn.Linear(hidden, n_slot) if predict_slot else None
        if predict_value:
            # E6: decode PUT/RUN bytes from OBS percept chars (+ backbone context)
            self.value_context = nn.Linear(hidden, 32)
            self.value_char_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1 + 32, 48),
                        nn.ReLU(),
                        nn.Linear(48, VALUE_VOCAB),
                    )
                    for _ in range(VALUE_CHARS)
                ]
            )
            self.value_head = None
        else:
            self.value_head = None
            self.value_char_mlps = None
            self.value_context = None
        self._none_op_id = none_op_id

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        intent = x[:, 0].long()
        source = x[:, 1].long()
        slot = x[:, 2].long()
        stage = x[:, 3].long()
        step = x[:, 4].long()
        prev = x[:, 5].long()
        nums = x[:, 6 : 9 + VALUE_CHARS]

        h = torch.cat(
            [
                self.intent_emb(intent),
                self.source_emb(source),
                self.slot_emb(slot),
                self.stage_emb(stage),
                self.step_emb(step),
                self.prev_op_emb(prev),
                nums,
            ],
            dim=-1,
        )
        return self.backbone(h)

    def _value_logits(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        percept = x[:, 9 : 9 + VALUE_CHARS]
        percept_ids = (percept * 96.0).round().long().clamp(0, VALUE_VOCAB - 1)
        # Strong percept-copy prior: E6 values come from OBS bytes in features, not chat.
        copy_logits = torch.full(
            (x.size(0), VALUE_CHARS, VALUE_VOCAB),
            -20.0,
            device=x.device,
            dtype=h.dtype,
        )
        copy_logits.scatter_(2, percept_ids.unsqueeze(-1), 20.0)

        ctx = self.value_context(h).unsqueeze(1).expand(-1, VALUE_CHARS, -1)
        combined = torch.cat([percept.unsqueeze(-1), ctx], dim=-1)
        mlp_logits = torch.stack(
            [self.value_char_mlps[i](combined[:, i]) for i in range(VALUE_CHARS)],
            dim=1,
        )
        return copy_logits + mlp_logits

    def forward(self, x: torch.Tensor, *, return_slot: bool = False, return_value: bool = False):
        h = self.encode(x)
        op_logits = self.op_head(h)
        slot_logits = self.slot_head(h) if return_slot and self.slot_head is not None else None
        value_logits = None
        if return_value and self.value_char_mlps is not None:
            value_logits = self._value_logits(h, x)
        if return_value:
            return op_logits, slot_logits, value_logits
        if return_slot and slot_logits is not None:
            return op_logits, slot_logits
        return op_logits

    @torch.no_grad()
    def predict_value_chars(self, x: torch.Tensor) -> list[int]:
        if self.value_char_mlps is None:
            raise RuntimeError("model has no value head")
        h = self.encode(x.unsqueeze(0) if x.dim() == 1 else x)
        xb = x.unsqueeze(0) if x.dim() == 1 else x
        logits = self._value_logits(h, xb)
        return logits.argmax(dim=-1)[0].tolist()

    @torch.no_grad()
    def predict_op_id(self, x: torch.Tensor) -> int:
        logits = self.forward(x.unsqueeze(0) if x.dim() == 1 else x)
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def predict_slot_id(self, x: torch.Tensor) -> int:
        if self.slot_head is None:
            raise RuntimeError("model has no slot head")
        _, slot_logits = self.forward(x.unsqueeze(0) if x.dim() == 1 else x, return_slot=True)
        return int(slot_logits.argmax(dim=-1).item())
