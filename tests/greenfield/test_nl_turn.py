"""E7b OBS-first NL turn tests."""

from __future__ import annotations

import pytest
import torch

from greenfield.kernel import Kernel
from greenfield.learned_encoder import LearnedEncoder
from greenfield.nl_turn import obs_utterance_op, run_nl_episode_obs_first
from greenfield.nl_gateway import deploy_policy_path
from greenfield.renderer.core import TemplateRenderer
from greenfield.runner import load_policy
from greenfield.types import OpCode


@pytest.fixture
def stack_encoder():
    path = "greenfield/checkpoints/encoder_e6_best.pt"
    try:
        return LearnedEncoder.from_checkpoint(path, device=torch.device("cpu"))
    except FileNotFoundError:
        pytest.skip("encoder_e6_best.pt missing")


@pytest.fixture
def e7_ckpt():
    from pathlib import Path

    path = Path("greenfield/checkpoints/encoder_e7_best.pt")
    if not path.is_file():
        pytest.skip("encoder_e7_best.pt missing")
    return path


def test_obs_utterance_sets_percept():
    kernel = Kernel(load_policy(deploy_policy_path()))
    state = kernel.genesis()
    op = obs_utterance_op("Remember my name is Ada")
    state = kernel.apply(state, op)
    payload = state.working.percept["payload"]
    assert payload["utterance"] == "Remember my name is Ada"
    assert payload["value"] == "Ada"


def test_obs_first_episode_query_hit(stack_encoder: LearnedEncoder, e7_ckpt: str):
    policy = load_policy(deploy_policy_path())
    state, trace, metrics, err = run_nl_episode_obs_first(
        "Remember my name is Ada",
        "What is my name?",
        seed=0,
        policy=policy,
        encoder=stack_encoder,
        renderer=TemplateRenderer(),
        parser_checkpoint=e7_ckpt,
    )
    assert err == ""
    assert trace[0].op == OpCode.OBS.value
    assert metrics["query_hits"] == 1
    assert metrics["queries"] == 1
    assert "Ada" in metrics["answer"]
