"""Fixed-width byte codec for PUT/RUN values (E6)."""

from __future__ import annotations

PRINTABLE_OFFSET = 32
PRINTABLE_COUNT = 95
PAD_ID = 0
VALUE_CHARS = 12
VALUE_VOCAB = PRINTABLE_COUNT + 1


def encode_value(value: str, max_len: int = VALUE_CHARS) -> list[int]:
    ids = [PAD_ID] * max_len
    for i, ch in enumerate(str(value)[:max_len]):
        code = ord(ch)
        if PRINTABLE_OFFSET <= code < PRINTABLE_OFFSET + PRINTABLE_COUNT:
            ids[i] = code - PRINTABLE_OFFSET + 1
    return ids


def decode_value(ids: list[int]) -> str:
    chars = []
    for tid in ids:
        if tid <= 0:
            continue
        chars.append(chr(tid - 1 + PRINTABLE_OFFSET))
    return "".join(chars)


def percept_ids_from_features(x) -> list[int]:
    """Recover byte ids from normalized percept slice (features[..., 9:9+VALUE_CHARS])."""
    if hasattr(x, "detach"):
        vals = x.detach().cpu().tolist()
    else:
        vals = list(x)
    start = 9
    end = start + VALUE_CHARS
    percept = vals[start:end] if len(vals) >= end else vals[-VALUE_CHARS:]
    return [max(0, min(VALUE_VOCAB - 1, int(round(float(v) * 96.0)))) for v in percept]


def decode_value_from_features(x) -> str:
    """Decode PUT/RUN value from OBS percept encoding in the feature vector."""
    return decode_value(percept_ids_from_features(x))
