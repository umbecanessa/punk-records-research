"""E12 — template proposals for dynamic user.* keys (retention policy v1)."""

from __future__ import annotations

import re

from greenfield.parser.template_parser import ParsedUtterance
from greenfield.parser.value_span import normalize_for_match, query_key
from greenfield.types import Intent

# Writable namespace — kernel schema allows user.* strings; overwrite OK (not in write-once list).
USER_LOCATION = "user.location"
USER_CITY = "user.city"
USER_HOME = "user.home"

_LOCATION_PLANT = (
    (re.compile(r"^i'?m living in (?P<val>.+?)\s*\.?\s*$", re.I), USER_LOCATION),
    (re.compile(r"^i live in (?P<val>.+?)\s*\.?\s*$", re.I), USER_LOCATION),
    (re.compile(r"^my home is (?P<val>.+?)\s*\.?\s*$", re.I), USER_HOME),
    (re.compile(r"^i'?m from (?P<val>.+?)\s*\.?\s*$", re.I), USER_CITY),
    (re.compile(r"^i moved to (?P<val>.+?)\s*\.?\s*$", re.I), USER_LOCATION),
)

_LOCATION_QUERY = (
    (re.compile(r"^where do i live\s*$", re.I), USER_LOCATION),
    (re.compile(r"^where am i living\s*$", re.I), USER_LOCATION),
    (re.compile(r"^what city am i in\s*$", re.I), USER_CITY),
    (re.compile(r"^where am i from\s*$", re.I), USER_CITY),
)

_GENERIC_REMEMBER = re.compile(
    r"^remember (?:that )?(?P<key>[a-z][a-z0-9_ ]{1,24}) is (?P<val>.+?)\s*\.?\s*$",
    re.I,
)


def _sanitize_user_key(raw: str) -> str:
    slug = re.sub(r"\s+", "_", raw.strip().lower())
    slug = re.sub(r"[^a-z0-9_]", "", slug)
    if not slug:
        return "user.note"
    if slug.startswith("user."):
        return slug[:48]
    return f"user.{slug[:44]}"


def parse_dynamic_utterance(text: str) -> ParsedUtterance | None:
    """Propose PLANT/QUERY with dynamic user.* keys before learned parser."""
    t = normalize_for_match(text)
    if not t:
        return None

    for pattern, slot in _LOCATION_PLANT:
        m = pattern.match(t)
        if m:
            val = m.group("val").strip().rstrip("?.!")
            if val:
                return ParsedUtterance(Intent.PLANT, {"slot": slot, "value": val}, "dynamic_plant")

    m = _GENERIC_REMEMBER.match(t)
    if m:
        key_raw = m.group("key").strip().lower()
        if key_raw not in ("my name", "my code", "my item", "the code", "the item"):
            key = _sanitize_user_key(m.group("key"))
            val = m.group("val").strip().rstrip("?.!")
            if val:
                return ParsedUtterance(Intent.PLANT, {"slot": key, "value": val}, "dynamic_plant")

    key = query_key(t)
    for pattern, slot in _LOCATION_QUERY:
        if pattern.match(key):
            return ParsedUtterance(Intent.QUERY, {"slot": slot}, "dynamic_query")

    return None
