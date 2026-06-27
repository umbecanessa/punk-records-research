"""Load E1/E2/E5b encoder checkpoints into current model layout."""

from __future__ import annotations

from pathlib import Path

import torch

from greenfield.train.features import INTENTS, MAX_STEP, OPCODES, SOURCES, SLOTS, STAGES
from greenfield.train.model import EventEncoderModel


def current_vocab() -> dict[str, int]:
    return {
        "num_intents": len(INTENTS),
        "num_sources": len(SOURCES),
        "num_slots": len(SLOTS),
        "num_stages": len(STAGES),
        "num_opcodes": len(OPCODES),
        "max_step": MAX_STEP,
    }


def migrate_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, val in state.items():
        if key == "mlp.6.weight":
            out["op_head.weight"] = val
        elif key == "mlp.6.bias":
            out["op_head.bias"] = val
        elif key.startswith("mlp."):
            out[key.replace("mlp.", "backbone.", 1)] = val
        else:
            out[key] = val
    return out


def _vocab_from_state(state: dict[str, torch.Tensor]) -> dict[str, int]:
    op_rows = state.get("op_head.weight")
    if op_rows is None:
        op_rows = state["backbone.6.weight"]
    return {
        "num_intents": state["intent_emb.weight"].shape[0],
        "num_sources": state["source_emb.weight"].shape[0],
        "num_slots": state["slot_emb.weight"].shape[0],
        "num_stages": state["stage_emb.weight"].shape[0],
        "num_opcodes": op_rows.shape[0],
        "max_step": state["step_emb.weight"].shape[0],
    }


def _load_compatible(model: EventEncoderModel, state: dict[str, torch.Tensor]) -> None:
    """Copy checkpoint weights where shapes match or prefix-fit (vocab expansion)."""
    model_state = dict(model.state_dict())
    for key, val in state.items():
        if key not in model_state:
            continue
        target = model_state[key]
        if target.shape == val.shape:
            model_state[key] = val
            continue
        if target.dim() != val.dim():
            continue
        if target.dim() == 2:
            rows = min(target.shape[0], val.shape[0])
            cols = min(target.shape[1], val.shape[1])
            merged = target.clone()
            merged[:rows, :cols] = val[:rows, :cols]
            model_state[key] = merged
            continue
        elif target.dim() == 1:
            n = min(target.shape[0], val.shape[0])
            merged = target.clone()
            merged[:n] = val[:n]
            model_state[key] = merged
    model.load_state_dict(model_state)


def load_encoder_model(
    path: str | Path,
    device: torch.device,
    *,
    predict_slot: bool = False,
    predict_value: bool = False,
    expand_vocab: bool = False,
) -> EventEncoderModel:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    hidden = int(ckpt.get("hidden", 128))
    predict_slot = predict_slot or bool(ckpt.get("predict_slot", False))
    predict_value = predict_value or bool(ckpt.get("predict_value", False))
    state = migrate_state_dict(ckpt["model"])
    if expand_vocab:
        vocab = current_vocab()
    else:
        vocab = ckpt.get("vocab") or _vocab_from_state(state)
    model = EventEncoderModel(
        hidden=hidden,
        predict_slot=predict_slot,
        predict_value=predict_value,
        **vocab,
    )
    _load_compatible(model, state)
    return model.to(device)
