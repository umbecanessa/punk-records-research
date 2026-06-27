"""E9b transformer renderer smoke + storage read-only tests."""

from __future__ import annotations

import torch

from greenfield.kernel import Kernel
from greenfield.renderer.core import LearnedTransformerRenderer
from greenfield.renderer.templates import reference_text
from greenfield.renderer.transformer_renderer import TransformerRendererModel, count_parameters
from greenfield.runner import load_policy
from greenfield.train.features import SLOT_TO_ID


def test_transformer_renderer_forward():
    model = TransformerRendererModel()
    slot = torch.tensor([SLOT_TO_ID["fact.name"]])
    vids = torch.zeros(1, 12, dtype=torch.long)
    target = torch.zeros(1, 48, dtype=torch.long)
    logits = model(slot, vids, target)
    assert logits.shape == (1, 48, 96)
    assert count_parameters(model) > 200_000


def test_renderer_reads_storage_only():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    state.storage.slots["fact.name"] = "Ada"
    before = dict(state.storage.slots)
    model = TransformerRendererModel()
    renderer = LearnedTransformerRenderer(model, device=torch.device("cpu"))
    out = renderer.render(state, {"keys": ["fact.name"], "mode": "answer"})
    assert state.storage.slots == before
    assert isinstance(out, str)


def test_reference_templates_unchanged():
    assert reference_text("fact.name", "Ada") == "My name is Ada."
