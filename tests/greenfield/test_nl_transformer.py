"""E9a utterance transformer smoke tests."""

from __future__ import annotations

import torch

from greenfield.train.features import FEATURE_DIM, featurize_utterance
from greenfield.train.model import EventEncoderModel
from greenfield.train.nl_transformer import UtteranceTransformerEncoder, count_parameters
from greenfield.kernel import Kernel
from greenfield.runner import load_policy


def test_transformer_forward_shape():
    enc = UtteranceTransformerEncoder()
    x = torch.randn(4, 12)
    out = enc(x)
    assert out.shape == (4, 96)
    assert count_parameters(enc) > 100_000


def test_event_model_transformer_predict_event():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    fv = featurize_utterance("my name is Ada", state)
    x = torch.tensor(fv.as_list(), dtype=torch.float32)

    model = EventEncoderModel(predict_event=True, nl_backbone="transformer")
    intent_id, slot_id, _ = model.predict_event_fields(x)
    assert intent_id in (0, 1, 2)
    assert slot_id in range(4)
