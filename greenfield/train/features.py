"""Feature vocabulary for event encoder (E1)."""

from __future__ import annotations

from dataclasses import dataclass

from greenfield.parser.value_span import extract_plant_value, obs_value_hint
from greenfield.train.value_codec import VALUE_CHARS, encode_value
from greenfield.types import EpisodeEvent, Intent, MachineState, OpCode

INTENTS = list(Intent)
SOURCES = ["user", "system", "adversary", "tool"]
SLOTS = (
    ["fact.name", "fact.code", "fact.answer"]
    + [f"fact.item{i}" for i in range(8)]
    + ["__none__"]
)
STAGES = ["A", "B", "C", "D", "E", "F", "G"]
OPCODES = list(OpCode)
MAX_STEP = 8

INTENT_TO_ID = {x: i for i, x in enumerate(INTENTS)}
SOURCE_TO_ID = {x: i for i, x in enumerate(SOURCES)}
SLOT_TO_ID = {x: i for i, x in enumerate(SLOTS)}
STAGE_TO_ID = {x: i for i, x in enumerate(STAGES)}
OP_TO_ID = {x: i for i, x in enumerate(OPCODES)}
ID_TO_OP = {i: x for x, i in OP_TO_ID.items()}
ID_TO_SLOT = {i: name for i, name in enumerate(SLOTS)}
ID_TO_INTENT = {i: x for i, x in enumerate(INTENTS)}
NONE_OP_ID = len(OPCODES)
PERCEPT_OFFSET = 9
UTTERANCE_OFFSET = 9 + VALUE_CHARS
FEATURE_DIM = UTTERANCE_OFFSET + VALUE_CHARS
NL_DUMMY_INTENT = Intent.CHITCHAT

# E7 NL parser label space (subset of full kernel vocab).
E7_INTENTS = [Intent.PLANT, Intent.QUERY, Intent.CHITCHAT]
E7_INTENT_TO_ID = {x: i for i, x in enumerate(E7_INTENTS)}
E7_ID_TO_INTENT = {i: x for i, x in enumerate(E7_INTENTS)}
E7_SLOTS = ["fact.name", "fact.code", "fact.item0", "__none__"]
E7_SLOT_TO_ID = {x: i for i, x in enumerate(E7_SLOTS)}
E7_ID_TO_SLOT = {i: x for i, x in enumerate(E7_SLOTS)}


def e7_intent_id(intent: Intent) -> int:
    return E7_INTENT_TO_ID.get(intent, E7_INTENT_TO_ID[Intent.CHITCHAT])


def e7_slot_id(slot: str) -> int:
    return E7_SLOT_TO_ID.get(slot, E7_SLOT_TO_ID["__none__"])


def global_slot_id(e7_slot_idx: int) -> int:
    name = E7_ID_TO_SLOT.get(e7_slot_idx, "__none__")
    return SLOT_TO_ID[name]


@dataclass
class FeatureVector:
    intent_id: int
    source_id: int
    slot_id: int
    stage_id: int
    step_idx: int
    prev_op_id: int
    requires_seal: float
    storage_slots: float
    has_plan: float
    percept_value: list[float]
    utterance_value: list[float]

    def as_list(self) -> list[int | float]:
        return [
            self.intent_id,
            self.source_id,
            self.slot_id,
            self.stage_id,
            self.step_idx,
            self.prev_op_id,
            self.requires_seal,
            self.storage_slots,
            self.has_plan,
            *self.percept_value,
            *self.utterance_value,
        ]


def percept_value_str(state: MachineState, event: EpisodeEvent) -> str:
    payload = state.working.percept.get("payload", {})
    if "value" in payload:
        return str(payload["value"])
    val = event.slot_value()
    return str(val) if val is not None else ""


def encode_percept_value(state: MachineState, event: EpisodeEvent) -> list[float]:
    ids = encode_value(percept_value_str(state, event))
    return [i / 96.0 for i in ids]


def slot_key_id(event: EpisodeEvent) -> int:
    key = event.slot_key() or "__none__"
    if key in SLOT_TO_ID:
        return SLOT_TO_ID[key]
    return SLOT_TO_ID["__none__"]


def last_token(text: str) -> str:
    """Trailing whitespace token — matches {val} suffix in E7 paraphrase templates."""
    parts = str(text).strip().rstrip("?.!").split()
    return parts[-1] if parts else ""


def encode_utterance(text: str) -> list[float]:
    s = str(text)
    # Right-align so planted values at utterance suffix survive the fixed width.
    if len(s) > VALUE_CHARS:
        s = s[-VALUE_CHARS:]
    ids = encode_value(s)
    return [i / 96.0 for i in ids]


def utterance_from_event(event: EpisodeEvent) -> str:
    if "utterance" in event.payload:
        return str(event.payload["utterance"])
    return ""


def featurize(
    event: EpisodeEvent,
    state: MachineState,
    *,
    stage: str,
    step_idx: int,
    prev_op: OpCode | None,
    utterance: str | None = None,
) -> FeatureVector:
    source = event.source if event.source in SOURCE_TO_ID else "user"
    utext = utterance if utterance is not None else utterance_from_event(event)
    return FeatureVector(
        intent_id=INTENT_TO_ID[event.intent],
        source_id=SOURCE_TO_ID[source],
        slot_id=slot_key_id(event),
        stage_id=STAGE_TO_ID.get(stage, 0),
        step_idx=min(step_idx, MAX_STEP - 1),
        prev_op_id=OP_TO_ID[prev_op] if prev_op else NONE_OP_ID,
        requires_seal=1.0 if event.requires_seal else 0.0,
        storage_slots=min(len(state.storage.slots), 32) / 32.0,
        has_plan=1.0 if state.storage.plan.steps else 0.0,
        percept_value=encode_percept_value(state, event),
        utterance_value=encode_utterance(utext),
    )


def featurize_utterance(
    text: str,
    state: MachineState,
    *,
    stage: str = "B",
) -> FeatureVector:
    """NL parse features — intent/slot from utterance; percept bootstrapped with trailing token."""
    dummy = EpisodeEvent(
        t=0,
        source="user",
        intent=NL_DUMMY_INTENT,
        payload={"utterance": text},
        requires_seal=True,
    )
    fv = featurize(dummy, state, stage=stage, step_idx=0, prev_op=None, utterance=text)
    # Prefer OBS percept when utterance was already captured (E7b).
    percept_payload = state.working.percept.get("payload", {})
    if percept_payload:
        percept_value = encode_percept_value(state, dummy)
    else:
        guess = extract_plant_value(text) if text else ""
        percept_value = [i / 96.0 for i in encode_value(guess)] if guess else [0.0] * VALUE_CHARS
    return FeatureVector(
        intent_id=fv.intent_id,
        source_id=fv.source_id,
        slot_id=fv.slot_id,
        stage_id=fv.stage_id,
        step_idx=fv.step_idx,
        prev_op_id=fv.prev_op_id,
        requires_seal=fv.requires_seal,
        storage_slots=fv.storage_slots,
        has_plan=fv.has_plan,
        percept_value=percept_value,
        utterance_value=fv.utterance_value,
    )
