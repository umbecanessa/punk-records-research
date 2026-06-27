"""E7a: template NL → structured EpisodeEvent fields (untrusted front-end)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from greenfield.types import Intent

_NAME_PLANT = re.compile(
    r"^\s*(?:remember\s+)?(?:my\s+name\s+is|call\s+me|i\s+am|i'm)\s+(?P<val>[A-Za-z]{2,12})\s*\.?\s*$",
    re.I,
)
_NAME_QUERY = re.compile(
    r"^\s*(?:what(?:'s|\s+is)\s+my\s+name|who\s+am\s+i)\s*\??\s*$",
    re.I,
)
_CODE_PLANT = re.compile(
    r"^\s*(?:remember\s+)?(?:my\s+code\s+is|code\s+is)\s+(?P<val>[A-Za-z0-9]{2,12})\s*\.?\s*$",
    re.I,
)
_CODE_QUERY = re.compile(
    r"^\s*what(?:'s|\s+is)\s+my\s+code\s*\??\s*$",
    re.I,
)


@dataclass(frozen=True)
class ParsedUtterance:
    intent: Intent
    payload: dict
    template: str

    def as_event_fields(self) -> dict:
        return {"intent": self.intent, "payload": dict(self.payload)}


def parse_utterance(text: str) -> ParsedUtterance | None:
    """Map short English utterances to kernel intents. Returns None if unknown."""
    t = text.strip()
    if not t:
        return None

    m = _NAME_PLANT.match(t)
    if m:
        val = m.group("val").strip()
        return ParsedUtterance(
            Intent.PLANT,
            {"slot": "fact.name", "value": val.capitalize() if val.isalpha() else val},
            "name_plant",
        )

    if _NAME_QUERY.match(t):
        return ParsedUtterance(Intent.QUERY, {"slot": "fact.name"}, "name_query")

    m = _CODE_PLANT.match(t)
    if m:
        return ParsedUtterance(
            Intent.PLANT,
            {"slot": "fact.code", "value": m.group("val").strip()},
            "code_plant",
        )

    if _CODE_QUERY.match(t):
        return ParsedUtterance(Intent.QUERY, {"slot": "fact.code"}, "code_query")

    return None
