"""Fixed-width byte codec for PUT/RUN values (E6)."""

from __future__ import annotations

PRINTABLE_OFFSET = 32
PRINTABLE_COUNT = 95
PAD_ID = 0
VALUE_CHARS = 12
UTTERANCE_LEN = 96
VALUE_VOCAB = PRINTABLE_COUNT + 1


def encode_value(value: str, max_len: int = VALUE_CHARS) -> list[int]:
    ids = [PAD_ID] * max_len
    for i, ch in enumerate(str(value)[:max_len]):
        code = ord(ch)
        if PRINTABLE_OFFSET <= code < PRINTABLE_OFFSET + PRINTABLE_COUNT:
            ids[i] = code - PRINTABLE_OFFSET + 1
    return ids


def encode_utterance_window(text: str, max_len: int = UTTERANCE_LEN) -> list[float]:
    """Right-aligned printable window for E10.1 NL parser (full sentences)."""
    s = str(text)
    if len(s) > max_len:
        s = s[-max_len:]
    ids = encode_value(s, max_len=max_len)
    return [i / 96.0 for i in ids]


def decode_value(ids: list[int]) -> str:
    chars = []
    for tid in ids:
        if tid <= 0:
            continue
        chars.append(chr(tid - 1 + PRINTABLE_OFFSET))
    return "".join(chars)


def ids_from_features_slice(x, offset: int) -> list[int]:
    """Recover byte ids from a normalized char slice in the feature vector."""
    if hasattr(x, "detach"):
        vals = x.detach().cpu().tolist()
    else:
        vals = list(x)
    end = offset + VALUE_CHARS
    if len(vals) >= end:
        chunk = vals[offset:end]
    elif len(vals) >= 9 + VALUE_CHARS:
        chunk = vals[-VALUE_CHARS:]
    else:
        chunk = [0.0] * VALUE_CHARS
    return [max(0, min(VALUE_VOCAB - 1, int(round(float(v) * 96.0)))) for v in chunk]


def percept_ids_from_features(x) -> list[int]:
    """Recover byte ids from normalized percept slice (features[..., 9:9+VALUE_CHARS])."""
    return ids_from_features_slice(x, 9)


def utterance_ids_from_features(x) -> list[int]:
    """Recover byte ids from normalized utterance slice."""
    from greenfield.train.features import UTTERANCE_OFFSET

    return ids_from_features_slice(x, UTTERANCE_OFFSET)


def decode_value_from_features(x) -> str:
    """Decode PUT/RUN value from OBS percept encoding in the feature vector."""
    return decode_value(percept_ids_from_features(x))


def decode_value_from_utterance_features(x) -> str:
    """Decode planted value from NL utterance encoding in the feature vector."""
    return decode_value(utterance_ids_from_features(x))
