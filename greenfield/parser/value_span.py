"""E7e — template span extraction for multi-word plant values (PRI-aligned)."""

from __future__ import annotations

import re

from greenfield.parser.open_phrasing import (
    OPEN_CODE_PLANT_PREFIXES,
    OPEN_ITEM_PLANT_PREFIXES,
    OPEN_NAME_PLANT_PREFIXES,
    OPEN_QUERY_CODE,
    OPEN_QUERY_ITEM,
    OPEN_QUERY_NAME,
)
from greenfield.parser.template_parser import ParsedUtterance
from greenfield.types import Intent

_FILLER_RE = re.compile(
    r"^\s*(?:um,?\s*|well,?\s*|so\s+|hey\s*[—-]\s*|hmm\s+)",
    re.I,
)

_QUERY_NAME = frozenset(
    {
        "what is my name",
        "what's my name",
        "who am i",
        "tell me my name",
        "do you know my name",
    }
) | OPEN_QUERY_NAME
_QUERY_CODE = frozenset(
    {
        "what is my code",
        "what's my code",
        "tell me the code",
    }
) | OPEN_QUERY_CODE
_QUERY_ITEM = frozenset(
    {
        "what is my item",
        "what's my item",
        "what item do i have",
        "tell me my item",
    }
) | OPEN_QUERY_ITEM

# Prefixes must be lowercase; value = remainder of original string (case preserved).
_NAME_PREFIXES = (
    "please remember my name is ",
    "remember my name is ",
    "my name is ",
    "call me ",
    "i am ",
    "i'm ",
) + OPEN_NAME_PLANT_PREFIXES

_CODE_PREFIXES = (
    "remember code ",
    "my code is ",
    "the code is ",
) + OPEN_CODE_PLANT_PREFIXES

_ITEM_PREFIXES = (
    "please remember item ",
    "remember item ",
    "my item is ",
    "i have item ",
    "the item is ",
) + OPEN_ITEM_PLANT_PREFIXES

_ITEM_IDX_PLANT = re.compile(
    r"^\s*(?:remember|store)\s+item\s+(?P<idx>\d+)\s+is\s+(?P<val>.+?)\s*\.?\s*$",
    re.I,
)
_ITEM_IDX_QUERY = re.compile(
    r"^\s*what(?:'s|\s+is)\s+item\s+(?P<idx>\d+)\s*\??\s*$",
    re.I,
)


def strip_leading_fillers(text: str) -> str:
    """Remove E8 messy openers so template prefixes still align."""
    t = str(text).strip()
    while True:
        m = _FILLER_RE.match(t)
        if not m:
            break
        t = t[m.end() :].strip()
    return t


def normalize_for_match(text: str) -> str:
    """Surface cleanup for prefix/regex matching (labels unchanged)."""
    t = strip_leading_fillers(text)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\bIS\b", "is", t)
    return t.strip()


def _strip_end_punct(text: str) -> str:
    return str(text).strip().rstrip("?.! ").strip()


def _suffix_after_prefix(text: str, prefix: str) -> str:
    if not text.lower().startswith(prefix):
        return ""
    return text[len(prefix) :].strip().rstrip("?.!")


def extract_plant_value(text: str, slot: str | None = None) -> str:
    """Extract planted value span from utterance (supports multi-word suffixes)."""
    t = normalize_for_match(str(text))
    if not t:
        return ""

    m = _ITEM_IDX_PLANT.match(t)
    if m:
        return m.group("val").strip()

    lower = t.lower()
    groups: list[tuple[str, tuple[str, ...]]] = []
    if slot == "fact.name" or slot is None:
        groups.append(("fact.name", _NAME_PREFIXES))
    if slot == "fact.code" or slot is None:
        groups.append(("fact.code", _CODE_PREFIXES))
    if slot and slot.startswith("fact.item") or slot is None:
        groups.append(("fact.item0", _ITEM_PREFIXES))

    best = ""
    for _slot_key, prefixes in groups:
        for prefix in sorted(prefixes, key=len, reverse=True):
            if lower.startswith(prefix):
                cand = _suffix_after_prefix(t, prefix)
                if len(cand) > len(best):
                    best = cand
    if best:
        return best

    parts = t.rstrip("?.!").split()
    return parts[-1] if parts else ""


def obs_value_hint(text: str) -> str:
    """Best-effort value hint for OBS percept (longest template span wins)."""
    return extract_plant_value(text, slot=None)


def detect_plant_slot(text: str) -> str | None:
    """Infer target slot from plant template prefix."""
    t = normalize_for_match(text)
    lower = t.lower()
    for prefix in _NAME_PREFIXES:
        if lower.startswith(prefix):
            return "fact.name"
    for prefix in _CODE_PREFIXES:
        if lower.startswith(prefix):
            return "fact.code"
    for prefix in _ITEM_PREFIXES:
        if lower.startswith(prefix):
            return "fact.item0"
    m = _ITEM_IDX_PLANT.match(t)
    if m:
        return f"fact.item{m.group('idx')}"
    return None


def parse_template_utterance(text: str) -> ParsedUtterance | None:
    """Name/code/item plant+query via span templates (E8 messy-tolerant)."""
    t = normalize_for_match(text)
    if not t:
        return None

    lower = _strip_end_punct(t.lower())
    for slot, templates in (
        ("fact.name", _QUERY_NAME),
        ("fact.code", _QUERY_CODE),
        ("fact.item0", _QUERY_ITEM),
    ):
        if lower in templates:
            return ParsedUtterance(Intent.QUERY, {"slot": slot}, "template_query")

    slot = detect_plant_slot(t)
    if slot:
        val = extract_plant_value(t, slot)
        if val:
            return ParsedUtterance(Intent.PLANT, {"slot": slot, "value": val}, "template_plant")
    return None


def parse_extended_utterance(text: str) -> ParsedUtterance | None:
    """Indexed item plant/query — used for overflow NL (fact.item0..item4)."""
    t = normalize_for_match(text)
    if not t:
        return None

    m = _ITEM_IDX_PLANT.match(t)
    if m:
        idx = int(m.group("idx"))
        val = m.group("val").strip()
        return ParsedUtterance(
            Intent.PLANT,
            {"slot": f"fact.item{idx}", "value": val},
            "item_idx_plant",
        )

    m = _ITEM_IDX_QUERY.match(t)
    if m:
        idx = int(m.group("idx"))
        return ParsedUtterance(Intent.QUERY, {"slot": f"fact.item{idx}"}, "item_idx_query")

    return None
