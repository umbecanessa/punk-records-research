"""E12 dynamic plant/query templates."""

from greenfield.memory.dynamic_plant import parse_dynamic_utterance
from greenfield.types import Intent


def test_living_in_amsterdam():
    p = parse_dynamic_utterance("I'm living in Amsterdam")
    assert p is not None
    assert p.intent == Intent.PLANT
    assert p.payload["slot"] == "user.location"
    assert p.payload["value"] == "Amsterdam"


def test_where_do_i_live_query():
    p = parse_dynamic_utterance("where do I live ?")
    assert p is not None
    assert p.intent == Intent.QUERY
    assert p.payload["slot"] == "user.location"


def test_remember_generic_user_key():
    p = parse_dynamic_utterance("remember favorite color is blue")
    assert p is not None
    assert p.payload["slot"] == "user.favorite_color"
    assert p.payload["value"] == "blue"
