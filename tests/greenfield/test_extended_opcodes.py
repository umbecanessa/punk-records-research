"""E5c extended opcode stub tests."""

from __future__ import annotations

import pytest

from greenfield.kernel import Kernel
from greenfield.runner import load_policy
from greenfield.types import KernelRevert, OpCode, OpProposal


@pytest.fixture
def extended_kernel() -> Kernel:
    return Kernel(load_policy("greenfield/deploy/policy.extended.json"))


def test_fork_merge_roundtrip(extended_kernel: Kernel):
    state = extended_kernel.genesis()
    extended_kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
    obs_hash = extended_kernel.last_obs_hash(state)
    extended_kernel.apply(
        state,
        OpProposal(op=OpCode.PUT, args={"key": "fact.code", "value": "9999", "evidence_ref": obs_hash}),
    )
    extended_kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))
    extended_kernel.apply(state, OpProposal(op=OpCode.FORK, args={"branch_id": "alt"}))
    extended_kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {"noise": True}}))
    extended_kernel.apply(state, OpProposal(op=OpCode.SEAL, args={}))
    extended_kernel.apply(state, OpProposal(op=OpCode.MERGE, args={"branch_id": "alt"}))
    assert state.storage.slots["fact.code"] == "9999"


def test_delegate_logs_only(extended_kernel: Kernel):
    state = extended_kernel.genesis()
    extended_kernel.apply(
        state,
        OpProposal(op=OpCode.DELEGATE, args={"handle": "external_api", "payload": {"q": "ping"}}),
    )
    assert state.log[-1].op == OpCode.DELEGATE
    assert len(state.storage.slots) == 0


def test_fork_disabled_by_default():
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    with pytest.raises(KernelRevert):
        kernel.apply(state, OpProposal(op=OpCode.FORK, args={"branch_id": "x"}))
