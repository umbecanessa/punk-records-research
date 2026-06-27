"""E7e value span extraction."""

from __future__ import annotations

from greenfield.parser.value_span import (
    extract_plant_value,
    normalize_for_match,
    parse_extended_utterance,
    parse_template_utterance,
)
from greenfield.types import Intent


def test_multi_word_item_span():
    val = extract_plant_value("my item is brass key", "fact.item0")
    assert val == "brass key"


def test_name_span_still_works():
    val = extract_plant_value("remember my name is Umberto", "fact.name")
    assert val == "Umberto"


def test_item_index_plant():
    p = parse_extended_utterance("remember item 2 is val2-999")
    assert p is not None
    assert p.intent == Intent.PLANT
    assert p.payload["slot"] == "fact.item2"
    assert p.payload["value"] == "val2-999"


def test_item_index_query():
    p = parse_extended_utterance("what is item 3?")
    assert p is not None
    assert p.intent == Intent.QUERY
    assert p.payload["slot"] == "fact.item3"


def test_messy_filler_plant():
    p = parse_template_utterance("well, the item is silver coin!")
    assert p is not None
    assert p.intent == Intent.PLANT
    assert p.payload["value"] == "silver coin"


def test_messy_code_plant():
    p = parse_template_utterance("remember code 7208!")
    assert p is not None
    assert p.payload["slot"] == "fact.code"
    assert p.payload["value"] == "7208"


def test_normalize_is_casing():
    assert normalize_for_match("What IS My code") == "What is My code"
