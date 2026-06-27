"""E5c evidence merkle chain tests."""

from __future__ import annotations

import pytest

from greenfield.evidence import chain_root, prefix_root, verify_put_evidence
from greenfield.kernel import Kernel
from greenfield.runner import load_policy
from greenfield.types import KernelRevert, OpCode, OpProposal


@pytest.fixture
def merkle_kernel() -> Kernel:
    return Kernel(load_policy("greenfield/deploy/policy.merkle.json"))


def test_chain_root_deterministic(merkle_kernel: Kernel):
    state = merkle_kernel.genesis()
    merkle_kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
    merkle_kernel.apply(
        state,
        OpProposal(
            op=OpCode.PUT,
            args={
                "key": "fact.name",
                "value": "Ada",
                "evidence_ref": merkle_kernel.last_obs_hash(state),
            },
        ),
    )
    assert chain_root(state.log) == prefix_root(state.log, len(state.log) - 1)
    assert all("chain_root" in e.args for e in state.log)


def test_put_rejects_tampered_evidence(merkle_kernel: Kernel):
    state = merkle_kernel.genesis()
    merkle_kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
    obs = state.log[-1]
    obs.args["chain_root"] = "deadbeef"
    with pytest.raises(KernelRevert):
        merkle_kernel.apply(
            state,
            OpProposal(
                op=OpCode.PUT,
                args={"key": "fact.name", "value": "Ada", "evidence_ref": obs.entry_hash},
            ),
        )


def test_verify_put_evidence(merkle_kernel: Kernel):
    state = merkle_kernel.genesis()
    merkle_kernel.apply(state, OpProposal(op=OpCode.OBS, args={"source": "user", "payload": {}}))
    obs_hash = merkle_kernel.last_obs_hash(state)
    entry = verify_put_evidence(state.log, obs_hash, require_merkle=True)
    assert entry.op == OpCode.OBS
