"""Tests for greenfield kernel."""

from __future__ import annotations

import pytest

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.runner import run_episode
from greenfield.simulator import bind_tools, default_tool_executor, sample_world
from greenfield.types import Intent, KernelRevert, OpCode, OpProposal, Policy


@pytest.fixture
def policy() -> Policy:
    return Policy()


@pytest.fixture
def kernel(policy: Policy) -> Kernel:
    return Kernel(policy, tool_executor=default_tool_executor)


def test_write_once_reverts(kernel: Kernel):
    state = kernel.genesis()
    kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {"intent": "plant"}}))
    obs_hash = kernel.last_obs_hash(state)
    kernel.apply(
        state,
        OpProposal(op=OpCode.PUT, args={"key": "fact.name", "value": "Ada", "evidence_ref": obs_hash}),
    )
    with pytest.raises(KernelRevert):
        kernel.apply(
            state,
            OpProposal(op=OpCode.PUT, args={"key": "fact.name", "value": "Bob", "evidence_ref": obs_hash}),
        )


def test_seal_checkpoint_and_storage(kernel: Kernel):
    state = kernel.genesis()
    kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
    obs_hash = kernel.last_obs_hash(state)
    kernel.apply(
        state,
        OpProposal(op=OpCode.PUT, args={"key": "fact.code", "value": "1234", "evidence_ref": obs_hash}),
    )
    kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))
    assert state.storage.meta_seal_hash is not None
    assert state.storage.slots["fact.code"] == "1234"
    assert len(state.checkpoints) == 1


def test_tool_plant_via_run_and_seal(kernel: Kernel):
    world = sample_world(__import__("random").Random(0), num_facts=1)
    key, val = next(iter(world.facts.items()))
    state = kernel.genesis()
    bind_tools(state.storage, world)
    state.storage.plan.steps = ["run"]
    kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "system", "payload": {"intent": "tool"}}))
    kernel.apply(
        state,
        OpProposal(op=OpCode.RUN, args={"handle": "plant_fact", "args": {"key": key, "value": val}}),
    )
    kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))
    assert state.storage.slots[key] == val


def test_oracle_episode_stage_a(policy: Policy):
    world = sample_world(__import__("random").Random(1), num_facts=1)
    script = generate_script(world, stage=CurriculumStage.A)
    _, metrics = run_episode(
        world=world,
        script=script,
        policy=policy,
        encoder=OracleEncoder(),
        stage="A",
    )
    assert metrics.query_accuracy == 1.0
    assert metrics.revert_rate == 0.0


def test_stage_d_distractor_reverts(policy: Policy):
    world = sample_world(__import__("random").Random(2), num_facts=1)
    script = generate_script(world, stage=CurriculumStage.D, rng=__import__("random").Random(3))
    _, metrics = run_episode(
        world=world,
        script=script,
        policy=policy,
        encoder=OracleEncoder(),
        stage="D",
    )
    assert metrics.reverts >= 1
    assert metrics.query_accuracy == 1.0
