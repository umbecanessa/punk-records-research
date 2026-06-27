"""Feature vocabulary for event encoder (E1)."""

from __future__ import annotations

from dataclasses import dataclass

from greenfield.train.value_codec import VALUE_CHARS, encode_value
from greenfield.types import EpisodeEvent, Intent, MachineState, OpCode

INTENTS = list(Intent)
SOURCES = ["user", "system", "adversary", "tool"]
SLOTS = (
    ["fact.name", "fact.code", "fact.answer"]
    + [f"fact.item{i}" for i in range(8)]
    + ["__none__"]
)
STAGES = ["A", "B", "C", "D", "E", "F"]
OPCODES = list(OpCode)
MAX_STEP = 8

INTENT_TO_ID = {x: i for i, x in enumerate(INTENTS)}
SOURCE_TO_ID = {x: i for i, x in enumerate(SOURCES)}
SLOT_TO_ID = {x: i for i, x in enumerate(SLOTS)}
STAGE_TO_ID = {x: i for i, x in enumerate(STAGES)}
OP_TO_ID = {x: i for i, x in enumerate(OPCODES)}
ID_TO_OP = {i: x for x, i in OP_TO_ID.items()}
ID_TO_SLOT = {i: name for i, name in enumerate(SLOTS)}
NONE_OP_ID = len(OPCODES)


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


def featurize(
    event: EpisodeEvent,
    state: MachineState,
    *,
    stage: str,
    step_idx: int,
    prev_op: OpCode | None,
) -> FeatureVector:
    source = event.source if event.source in SOURCE_TO_ID else "user"
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
    )
