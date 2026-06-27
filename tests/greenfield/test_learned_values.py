"""E6: value head reads percept encoding; materialize never calls event.slot_value()."""

from __future__ import annotations

import random

import pytest

from greenfield.episodes import CurriculumStage, generate_script
from greenfield.learned_encoder import LearnedEncoder
from greenfield.kernel import Kernel
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import sample_world
from greenfield.train.features import encode_percept_value, featurize
from greenfield.types import EpisodeEvent, Intent, OpCode, OpProposal


@pytest.fixture
def e6_encoder():
    import torch

    path = "greenfield/checkpoints/encoder_e6_best.pt"
    try:
        return LearnedEncoder.from_checkpoint(path, device=torch.device("cpu"))
    except FileNotFoundError:
        pytest.skip("encoder_e6_best.pt not trained yet")


def test_materialize_does_not_read_event_slot_value(e6_encoder: LearnedEncoder):
    event = EpisodeEvent(
        t=0,
        source="user",
        intent=Intent.PLANT,
        payload={"slot": "fact.name", "value": "SHOULD_NOT_BE_USED"},
        requires_seal=True,
    )
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    kernel.apply(
        state,
        OpProposal(
            op=OpCode.OBS,
            args={"source": "user", "payload": {"intent": "plant", "slot": "fact.name", "value": "Ada"}},
        ),
    )
    fv = featurize(event, state, stage="A", step_idx=1, prev_op=OpCode.OBS)
    x = e6_encoder._tensor_features(fv)
    val = e6_encoder._value_str(event, x)
    assert val != "SHOULD_NOT_BE_USED"
    assert val == "Ada"


def test_e6_stage_a_query_accuracy(e6_encoder: LearnedEncoder):
    policy = load_policy("greenfield/deploy/policy.v0.json")
    hits = 0
    for i in range(20):
        world = sample_world(random.Random(i), num_facts=1)
        script = generate_script(world, stage=CurriculumStage.A, rng=random.Random(i + 1))
        _, m = run_episode(
            world=world,
            script=script,
            policy=policy,
            encoder=e6_encoder,
            stage="A",
        )
        hits += m.query_accuracy
    assert hits / 20 >= 0.95


def test_percept_value_encoding():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    kernel.apply(
        state,
        OpProposal(
            op=OpCode.OBS,
            args={"source": "user", "payload": {"value": "1234"}},
        ),
    )
    event = EpisodeEvent(t=0, source="user", intent=Intent.PLANT, payload={"value": "ignored"})
    enc = encode_percept_value(state, event)
    assert enc[0] > 0
