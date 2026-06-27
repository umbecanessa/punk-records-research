"""Regression: opcode masks for operand training must be boolean."""

from __future__ import annotations

import torch

from greenfield.train.features import OP_TO_ID
from greenfield.train.train_encoder_e6 import VALUE_OP_IDS, _op_mask
from greenfield.types import OpCode


def test_op_mask_is_boolean_not_row_indices():
    y = torch.tensor([OP_TO_ID[OpCode.OBS], OP_TO_ID[OpCode.PUT], OP_TO_ID[OpCode.GET]])
    mask = _op_mask(y, VALUE_OP_IDS, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, False]
    # Integer 0/1 mask would index rows [0,1,0] — three rows, not one PUT row.
    bad = torch.tensor([0, 1, 0])
    assert mask.sum().item() == 1
    assert bad.sum().item() == 1  # same sum but wrong semantics for indexing
