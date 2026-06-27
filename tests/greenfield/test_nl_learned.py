"""E7 feature vector + learned NL parser tests."""

from __future__ import annotations

import pytest

from greenfield.kernel import Kernel
from greenfield.nl_gateway import LearnedEventParser, parse_nl
from greenfield.runner import load_policy
from greenfield.train.features import FEATURE_DIM, encode_utterance, featurize, featurize_utterance
from greenfield.types import EpisodeEvent, Intent, OpCode


def test_feature_dim_includes_utterance_slice():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    fv = featurize_utterance("my name is Ada", state)
    assert len(fv.as_list()) == FEATURE_DIM == 33


def test_utterance_encoding_differs_from_value_target():
    """Document the E7 label gap: utterance slice != left-aligned value encoding."""
    from greenfield.train.value_codec import decode_value, encode_value

    text = "my name is Ada"
    utt_ids = [int(round(u * 96)) for u in encode_utterance(text)]
    val_ids = encode_value("Ada")
    assert decode_value(utt_ids) != decode_value(val_ids)


def test_oracle_featurize_zeros_utterance_without_payload():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    event = EpisodeEvent(
        t=0,
        source="user",
        intent=Intent.PLANT,
        payload={"slot": "fact.name", "value": "Ada"},
    )
    fv = featurize(event, state, stage="A", step_idx=0, prev_op=OpCode.OBS)
    assert fv.utterance_value == [0.0] * 12


@pytest.fixture
def e7_parser():
    import torch

    path = "greenfield/checkpoints/encoder_e7_best.pt"
    try:
        return LearnedEventParser.from_checkpoint(path, device=torch.device("cpu"))
    except FileNotFoundError:
        pytest.skip("encoder_e7_best.pt not trained yet")


def test_learned_name_plant(e7_parser: LearnedEventParser):
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    parsed = e7_parser.parse("my name is Ada", state)
    assert parsed is not None
    assert parsed.intent == Intent.PLANT
    assert parsed.payload["slot"] == "fact.name"
    assert parsed.payload["value"] == "Ada"


def test_learned_name_query(e7_parser: LearnedEventParser):
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    parsed = e7_parser.parse("what is my name?", state)
    assert parsed is not None
    assert parsed.intent == Intent.QUERY
    assert parsed.payload["slot"] == "fact.name"


def test_parse_nl_fallback_template():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    parsed = parse_nl("Remember my name is Ada", state, checkpoint="__missing__.pt")
    assert parsed is not None
    assert parsed.intent == Intent.PLANT
