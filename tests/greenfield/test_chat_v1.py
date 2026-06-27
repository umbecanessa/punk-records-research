"""E10 chat v1 session tests."""

from __future__ import annotations

import pytest

from greenfield.chat_v1 import default_chat_script, load_chat_v1_stack, run_nl_chat_session


@pytest.fixture
def chat_stack():
    try:
        return load_chat_v1_stack(device=__import__("torch").device("cpu"))
    except FileNotFoundError:
        pytest.skip("E10 checkpoints not built yet")


def test_default_script_has_mixed_intents():
    script = default_chat_script()
    assert len(script) >= 6
    joined = " ".join(script).lower()
    assert "name" in joined and ("hello" in joined or "good" in joined)


def test_chat_session_queries_hit(chat_stack):
    _, _, metrics, err = run_nl_chat_session(
        default_chat_script(),
        stack=chat_stack,
        token_curve=True,
    )
    assert err == ""
    assert metrics["query_hits"] == metrics["queries"]
    assert metrics["reverts"] == 0
    assert metrics["tokens"]["savings_ratio"] > 0.3


def test_chitchat_does_not_plant(chat_stack):
    state, _, metrics, err = run_nl_chat_session(
        ["hello there", "hmm ok"],
        stack=chat_stack,
    )
    assert err == ""
    assert not any(k.startswith("fact.") for k in state.storage.slots)
