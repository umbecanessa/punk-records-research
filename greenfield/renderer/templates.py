"""Reference text templates — structured, not chat transcripts."""

from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "fact.name": "My name is {value}.",
    "fact.code": "The secret code is {value}.",
    "fact.answer": "The answer is {value}.",
    "user.location": "You live in {value}.",
    "user.city": "You're from {value}.",
    "user.home": "Your home is {value}.",
    **{f"fact.item{i}": "Stored value {value}." for i in range(8)},
}


def reference_text(key: str, value: str, *, mode: str = "answer") -> str:
    del mode  # v0: single template per slot
    tpl = TEMPLATES.get(key, "{value}")
    return tpl.format(value=value)
