"""E5d stress tests: tight caps, promote, revert after overflow."""

from __future__ import annotations

import random

import pytest

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.overflow import fact_keys_in_hot
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import default_tool_executor, overflow_world
from greenfield.types import OpCode, OpProposal


@pytest.fixture
def promote_policy():
    return load_policy("greenfield/deploy/policy.promote.json")


def test_promote_cold_on_get(promote_policy):
    kernel = Kernel(promote_policy, tool_executor=default_tool_executor)
    state = kernel.genesis()

    def plant(key: str, val: str):
        kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
        obs = kernel.last_obs_hash(state)
        kernel.apply(
            state,
            OpProposal(op=OpCode.PUT, args={"key": key, "value": val, "evidence_ref": obs}),
        )
        kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))

    for i in range(4):
        plant(f"fact.item{i}", f"v{i}")

    assert "fact.item0" not in state.storage.slots
    kernel.apply(state, OpProposal(op=OpCode.GET, args={"key": "fact.item0"}))
    assert "fact.item0" in state.storage.slots
    assert len(fact_keys_in_hot(state)) <= promote_policy.max_hot_fact_slots


def test_large_overflow_world_stage_f(promote_policy):
    world = overflow_world(random.Random(99), num_facts=8)
    script = generate_script(world, stage=CurriculumStage.F, rng=random.Random(100))
    _, metrics = run_episode(
        world=world,
        script=script,
        policy=promote_policy,
        encoder=OracleEncoder(),
        stage="F",
    )
    assert metrics.query_accuracy == 1.0
    assert metrics.overflow_evictions >= 6


def test_revert_after_overflow_restores_hot(promote_policy):
    kernel = Kernel(promote_policy, tool_executor=default_tool_executor)
    state = kernel.genesis()

    def plant(key: str, val: str):
        kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
        obs = kernel.last_obs_hash(state)
        kernel.apply(
            state,
            OpProposal(op=OpCode.PUT, args={"key": key, "value": val, "evidence_ref": obs}),
        )
        kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))

    plant("fact.item0", "a")
    first_seal = state.storage.meta_seal_hash
    plant("fact.item1", "b")
    plant("fact.item2", "c")
    assert "fact.item0" not in state.storage.slots

    kernel.apply(state, OpProposal(op=OpCode.REVERT, args={"to": first_seal}))
    assert state.storage.slots.get("fact.item0") == "a"
    assert "fact.item2" not in state.storage.slots
