"""Regression: Space live-chat confusion cases."""

from __future__ import annotations

import pytest

from greenfield.chat_v1 import init_chat_session, load_chat_v1_stack, run_chat_turn


@pytest.fixture(scope="module")
def session():
    stack = load_chat_v1_stack(device=__import__("torch").device("cpu"))
    return init_chat_session(stack)


def test_space_transcript(session):
    turns = [
        ("hi there", "OK.", "CHITCHAT"),
        ("my name is Umberto", "Umberto", "PLANT"),
        ("I'm living in Amsterdam", "OK.", "CHITCHAT"),
        ("oggi piove brutta giornata", "OK.", "CHITCHAT"),
        ("where do I live ?", "only remember", "CHITCHAT"),
        ("whats my name ?", "Umberto", "QUERY"),
    ]
    for user, expect_substr, intent in turns:
        session, reply, err = run_chat_turn(session, user)
        assert not err, err
        assert expect_substr.lower() in reply.lower(), (user, reply)
        assert session.metrics["turns"][-1]["intent"].upper() == intent
