"""E4 overflow / cold-store tests."""

from __future__ import annotations

import random

import pytest

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.overflow import overflow_evict, read_slot_with_cold
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import default_tool_executor, overflow_world
from greenfield.types import OpCode, OpProposal, Policy


@pytest.fixture
def overflow_policy() -> Policy:
    return load_policy("greenfield/deploy/policy.overflow.json")


@pytest.fixture
def overflow_kernel(overflow_policy: Policy) -> Kernel:
    return Kernel(overflow_policy, tool_executor=default_tool_executor)


def _plant_and_seal(kernel: Kernel, state, key: str, value: str):
    kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {"intent": "plant"}}))
    obs_hash = kernel.last_obs_hash(state)
    kernel.apply(
        state,
        OpProposal(op=OpCode.PUT, args={"key": key, "value": value, "evidence_ref": obs_hash}),
    )
    kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))


def test_overflow_evicts_oldest_fact(overflow_kernel: Kernel, overflow_policy: Policy):
    state = overflow_kernel.genesis()
    _plant_and_seal(overflow_kernel, state, "fact.item0", "a")
    _plant_and_seal(overflow_kernel, state, "fact.item1", "b")
    _plant_and_seal(overflow_kernel, state, "fact.item2", "c")

    assert "fact.item0" not in state.storage.slots
    assert "fact.item1" in state.storage.slots
    assert "fact.item2" in state.storage.slots
    assert state.overflow_evictions >= 1
    assert "fact.item0" in state.cold_index

    value, source = read_slot_with_cold(state, "fact.item0")
    assert source == "cold"
    assert value == "a"


def test_get_cold_by_hash(overflow_kernel: Kernel):
    state = overflow_kernel.genesis()
    for i in range(4):
        _plant_and_seal(overflow_kernel, state, f"fact.item{i}", f"v{i}")

    cold_hash = state.cold_index["fact.item0"]
    value, source = read_slot_with_cold(state, "fact.item0", cold_hash=cold_hash)
    assert source == "cold"
    assert value == "v0"
    assert state.cold_hits >= 1


def test_stage_f_oracle_episode(overflow_policy: Policy):
    world = overflow_world(random.Random(42), num_facts=5)
    script = generate_script(world, stage=CurriculumStage.F, rng=random.Random(43))
    state, metrics = run_episode(
        world=world,
        script=script,
        policy=overflow_policy,
        encoder=OracleEncoder(),
        stage="F",
    )
    assert metrics.query_accuracy == 1.0
    assert metrics.cold_hits > 0
    assert metrics.overflow_evictions > 0
    assert len(state.cold_store.records) >= 3
