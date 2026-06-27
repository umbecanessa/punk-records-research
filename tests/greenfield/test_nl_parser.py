"""E7a NL template parser tests."""

from __future__ import annotations

from greenfield.parser.template_parser import parse_utterance
from greenfield.types import Intent


def test_name_plant():
    p = parse_utterance("Remember my name is Ada")
    assert p is not None
    assert p.intent == Intent.PLANT
    assert p.payload == {"slot": "fact.name", "value": "Ada"}


def test_name_query():
    p = parse_utterance("What's my name?")
    assert p is not None
    assert p.intent == Intent.QUERY
    assert p.payload == {"slot": "fact.name"}


def test_code_plant():
    p = parse_utterance("my code is 42xy")
    assert p is not None
    assert p.payload["slot"] == "fact.code"


def test_unknown():
    assert parse_utterance("hello there") is None
