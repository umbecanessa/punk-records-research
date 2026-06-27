"""Core types for the greenfield state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OpCode(str, Enum):
    OBS = "OBS"
    PUT = "PUT"
    GET = "GET"
    FOCUS = "FOCUS"
    RUN = "RUN"
    STEP = "STEP"
    SEAL = "SEAL"
    REVERT = "REVERT"
    RENDER = "RENDER"
    FORK = "FORK"
    MERGE = "MERGE"
    DELEGATE = "DELEGATE"


class Intent(str, Enum):
    PLANT = "plant"
    QUERY = "query"
    CHITCHAT = "chitchat"
    TOOL_PLANT = "tool_plant"
    DISTRACTOR_PUT = "distractor_put"


@dataclass
class Policy:
    fact_write_once: list[str] = field(default_factory=lambda: ["fact.name", "fact.code"])
    plan_required_before_run: bool = True
    gas_per_episode: int = 10_000
    gas_cost: dict[str, int] = field(
        default_factory=lambda: {
            "OBS": 1,
            "PUT": 10,
            "GET": 2,
            "FOCUS": 1,
            "RUN": 50,
            "STEP": 2,
            "SEAL": 5,
            "REVERT": 20,
            "RENDER": 100,
        }
    )
    allow_missing_get: bool = False
    max_working_hot: int = 32
    max_hot_fact_slots: int = 32
    overflow_on_seal: bool = False
    promote_cold_on_get: bool = False
    evidence_merkle: bool = False
    enable_fork: bool = False
    enable_merge: bool = False
    enable_delegate: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        return cls(
            fact_write_once=list(data.get("fact_write_once", ["fact.name", "fact.code"])),
            plan_required_before_run=bool(data.get("plan_required_before_run", True)),
            gas_per_episode=int(data.get("gas_per_episode", 10_000)),
            gas_cost=dict(data.get("gas_cost", cls().gas_cost)),
            allow_missing_get=bool(data.get("allow_missing_get", False)),
            max_working_hot=int(data.get("max_working_hot", 32)),
            max_hot_fact_slots=int(data.get("max_hot_fact_slots", 32)),
            overflow_on_seal=bool(data.get("overflow_on_seal", False)),
            promote_cold_on_get=bool(data.get("promote_cold_on_get", False)),
            evidence_merkle=bool(data.get("evidence_merkle", False)),
            enable_fork=bool(data.get("enable_fork", False)),
            enable_merge=bool(data.get("enable_merge", False)),
            enable_delegate=bool(data.get("enable_delegate", False)),
        )


@dataclass
class PlanState:
    steps: list[str] = field(default_factory=list)
    ptr: int = 0


@dataclass
class Storage:
    slots: dict[str, Any] = field(default_factory=dict)
    plan: PlanState = field(default_factory=PlanState)
    meta_epoch: int = 0
    meta_seal_hash: str | None = None

    def copy(self) -> Storage:
        return Storage(
            slots=dict(self.slots),
            plan=PlanState(steps=list(self.plan.steps), ptr=self.plan.ptr),
            meta_epoch=self.meta_epoch,
            meta_seal_hash=self.meta_seal_hash,
        )


@dataclass
class Working:
    percept: dict[str, Any] = field(default_factory=dict)
    hot: list[str] = field(default_factory=list)
    pending: dict[str, Any] = field(default_factory=dict)
    last_read: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.percept.clear()
        self.hot.clear()
        self.pending.clear()
        self.last_read.clear()


@dataclass
class LogEntry:
    idx: int
    op: OpCode
    args: dict[str, Any]
    prev_hash: str
    entry_hash: str


@dataclass
class EpisodeEvent:
    t: int
    source: str
    intent: Intent
    payload: dict[str, Any] = field(default_factory=dict)
    requires_seal: bool = True

    def slot_key(self) -> str | None:
        key = self.payload.get("slot")
        return str(key) if key else None

    def slot_value(self) -> Any:
        return self.payload.get("value")


@dataclass
class World:
    """Hidden simulator ground truth."""

    facts: dict[str, Any] = field(default_factory=dict)
    tool_handles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def expected_value(self, key: str) -> Any:
        return self.facts.get(key)


@dataclass
class OpProposal:
    op: OpCode
    args: dict[str, Any]


@dataclass
class KernelRevert(Exception):
    reason: str
    op: OpCode | None = None

    def __str__(self) -> str:
        if self.op:
            return f"{self.op.value}: {self.reason}"
        return self.reason


@dataclass
class MachineState:
    storage: Storage
    working: Working
    log: list[LogEntry]
    checkpoints: dict[str, Storage]
    gas_used: int = 0
    cold_store: Any = None
    cold_index: dict[str, str] = field(default_factory=dict)
    cold_hits: int = 0
    overflow_evictions: int = 0
