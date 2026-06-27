"""Small MLP event → opcode (+ optional slot) classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from greenfield.train.features import INTENTS, MAX_STEP, OPCODES, SOURCES, SLOTS, STAGES, UTTERANCE_OFFSET
from greenfield.train.value_codec import VALUE_CHARS, VALUE_VOCAB


def _build_nl_utterance_enc(
    *,
    nl_backbone: str,
    dropout: float,
) -> nn.Module:
    if nl_backbone == "transformer":
        from greenfield.train.nl_transformer import UtteranceTransformerEncoder

        return UtteranceTransformerEncoder(dropout=dropout)
    return nn.Sequential(
        nn.Linear(VALUE_CHARS, 96),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(96, 96),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


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
        predict_event: bool = False,
        nl_backbone: str = "mlp",
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
        self.predict_event = predict_event
        self.nl_backbone = nl_backbone
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

        self.in_dim = 16 + 8 + 16 + 8 + 8 + 16 + 3 + VALUE_CHARS + VALUE_CHARS
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
        if predict_event:
            from greenfield.train.features import E7_INTENTS, E7_SLOTS

            self.nl_utterance_enc = _build_nl_utterance_enc(nl_backbone=nl_backbone, dropout=dropout)
            self.event_intent_head = nn.Linear(96, len(E7_INTENTS))
            self.event_slot_head = nn.Linear(96, len(E7_SLOTS))
            self.utterance_context = None
            self.utterance_char_mlps = None
        else:
            self.nl_utterance_enc = None
            self.event_intent_head = None
            self.event_slot_head = None
            self.utterance_context = None
            self.utterance_char_mlps = None
        if predict_value:
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

    def _nl_hidden(self, x: torch.Tensor) -> torch.Tensor:
        assert self.nl_utterance_enc is not None
        utt = x[:, UTTERANCE_OFFSET : UTTERANCE_OFFSET + VALUE_CHARS]
        return self.nl_utterance_enc(utt)

    def _event_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_nl = self._nl_hidden(x)
        intent_logits = self.event_intent_head(h_nl)
        slot_logits = self.event_slot_head(h_nl) if self.event_slot_head is not None else None
        return intent_logits, slot_logits

    def _percept_copy_logits(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        offset: int,
        context: nn.Linear,
        mlps: nn.ModuleList,
    ) -> torch.Tensor:
        """E6: percept[i] == value[i] — strong same-index copy prior."""
        chars = x[:, offset : offset + VALUE_CHARS]
        char_ids = (chars * 96.0).round().long().clamp(0, VALUE_VOCAB - 1)
        copy_logits = torch.full(
            (x.size(0), VALUE_CHARS, VALUE_VOCAB),
            -20.0,
            device=x.device,
            dtype=h.dtype,
        )
        copy_logits.scatter_(2, char_ids.unsqueeze(-1), 20.0)
        ctx = context(h).unsqueeze(1).expand(-1, VALUE_CHARS, -1)
        combined = torch.cat([chars.unsqueeze(-1), ctx], dim=-1)
        mlp_logits = torch.stack([mlps[i](combined[:, i]) for i in range(VALUE_CHARS)], dim=1)
        return copy_logits + mlp_logits

    def _extract_value_logits(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        offset: int,
        context: nn.Linear,
        mlps: nn.ModuleList,
    ) -> torch.Tensor:
        """E7: value is a substring — each output char MLP sees the full utterance window."""
        chars = x[:, offset : offset + VALUE_CHARS]
        ctx = context(h)
        window = torch.cat([chars, ctx], dim=-1)
        window = window.unsqueeze(1).expand(-1, VALUE_CHARS, -1)
        return torch.stack([mlps[i](window[:, i]) for i in range(VALUE_CHARS)], dim=1)

    def _value_logits(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert self.value_context is not None and self.value_char_mlps is not None
        return self._percept_copy_logits(h, x, 9, self.value_context, self.value_char_mlps)

    def _utterance_logits(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("E7 uses percept-copy value head, not utterance_char_mlps")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        intent = x[:, 0].long()
        source = x[:, 1].long()
        slot = x[:, 2].long()
        stage = x[:, 3].long()
        step = x[:, 4].long()
        prev = x[:, 5].long()
        nums = x[:, 6 : 6 + 3 + VALUE_CHARS + VALUE_CHARS]

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

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_slot: bool = False,
        return_value: bool = False,
        return_event: bool = False,
    ):
        h = self.encode(x)
        op_logits = self.op_head(h)
        slot_logits = self.slot_head(h) if return_slot and self.slot_head is not None else None
        value_logits = None
        if return_value and self.value_char_mlps is not None:
            value_logits = self._value_logits(h, x)
        if return_event:
            event_intent_logits, event_slot_logits = self._event_logits(x)
            return event_intent_logits, event_slot_logits, None
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

    @torch.no_grad()
    def predict_event_fields(self, x: torch.Tensor) -> tuple[int, int, list[int]]:
        if self.event_intent_head is None:
            raise RuntimeError("model has no event parse heads")
        xb = x.unsqueeze(0) if x.dim() == 1 else x
        intent_logits, slot_logits = self._event_logits(xb)
        intent_id = int(intent_logits.argmax(dim=-1).item())
        slot_id = int(slot_logits.argmax(dim=-1).item()) if slot_logits is not None else 0
        value_ids: list[int] = []
        if self.value_char_mlps is not None:
            h = self.encode(xb)
            value_ids = self._value_logits(h, xb).argmax(dim=-1)[0].tolist()
        return intent_id, slot_id, value_ids
