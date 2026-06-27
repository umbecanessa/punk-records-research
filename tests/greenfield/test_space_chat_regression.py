"""Regression: Space live-chat confusion cases + E12 dynamic keys."""

from __future__ import annotations

import pytest

from greenfield.chat_v1 import init_chat_session, load_chat_v1_stack, run_chat_turn


@pytest.fixture(scope="module")
def session():
    stack = load_chat_v1_stack(device=__import__("torch").device("cpu"))
    return init_chat_session(stack)


def test_space_transcript(session):
    turns = [
        ("hi there", "OK.", "chitchat"),
        ("my name is Umberto", "Umberto", "plant"),
        ("I'm living in Amsterdam", "Amsterdam", "plant"),
        ("oggi piove brutta giornata", "OK.", "chitchat"),
        ("where do I live ?", "Amsterdam", "query"),
        ("whats my name ?", "Umberto", "query"),
    ]
    for user, expect_substr, intent in turns:
        session, reply, err = run_chat_turn(session, user)
        assert not err, err
        assert expect_substr.lower() in reply.lower(), (user, reply)
        assert session.metrics["turns"][-1]["intent"] == intent
