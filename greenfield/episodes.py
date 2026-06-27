"""Synthetic event scripts — no chat transcripts."""

from __future__ import annotations

import random
from enum import Enum

from greenfield.types import EpisodeEvent, Intent, World


class CurriculumStage(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"  # E7c — multi-type quest world (name + code + item)


def _plant_event(t: int, key: str, value: str) -> EpisodeEvent:
    return EpisodeEvent(
        t=t,
        source="user",
        intent=Intent.PLANT,
        payload={"slot": key, "value": value},
        requires_seal=True,
    )


def _query_event(t: int, key: str) -> EpisodeEvent:
    return EpisodeEvent(
        t=t,
        source="user",
        intent=Intent.QUERY,
        payload={"slot": key},
        requires_seal=False,
    )


def _filler_event(t: int) -> EpisodeEvent:
    return EpisodeEvent(
        t=t,
        source="user",
        intent=Intent.CHITCHAT,
        payload={"noise": True},
        requires_seal=True,
    )


def _tool_plant_event(t: int, key: str, value: str) -> EpisodeEvent:
    return EpisodeEvent(
        t=t,
        source="system",
        intent=Intent.TOOL_PLANT,
        payload={"slot": key, "value": value, "handle": "plant_fact"},
        requires_seal=True,
    )


def _distractor_put_event(t: int, key: str, bad_value: str) -> EpisodeEvent:
    return EpisodeEvent(
        t=t,
        source="adversary",
        intent=Intent.DISTRACTOR_PUT,
        payload={"slot": key, "value": bad_value},
        requires_seal=True,
    )


def generate_script(
    world: World,
    *,
    stage: CurriculumStage = CurriculumStage.B,
    rng: random.Random | None = None,
    filler_ratio: float = 0.4,
) -> list[EpisodeEvent]:
    rng = rng or random.Random()
    facts = list(world.facts.items())
    if not facts:
        raise ValueError("world has no facts")

    events: list[EpisodeEvent] = []
    t = 0

    if stage == CurriculumStage.A:
        key, val = facts[0]
        events.append(_plant_event(t, key, val))
        t += 1
        events.append(_query_event(t, key))
        return events

    if stage == CurriculumStage.B:
        key, val = facts[0]
        events.append(_plant_event(t, key, val))
        t += 1
        num_filler = rng.randint(2, 6)
        for _ in range(num_filler):
            events.append(_filler_event(t))
            t += 1
        events.append(_query_event(t, key))
        return events

    if stage == CurriculumStage.C:
        key, val = facts[0]
        events.append(_tool_plant_event(t, key, val))
        t += 1
        events.append(_query_event(t, key))
        return events

    if stage == CurriculumStage.D:
        for i, (key, val) in enumerate(facts):
            events.append(_plant_event(t, key, val))
            t += 1
            if i == 0 or rng.random() < 0.5:
                events.append(_distractor_put_event(t, key, "WRONG"))
                t += 1
        for key, _ in facts:
            events.append(_query_event(t, key))
            t += 1
        return events

    if stage == CurriculumStage.E:
        key, val = facts[0]
        events.append(
            EpisodeEvent(
                t=t,
                source="system",
                intent=Intent.TOOL_PLANT,
                payload={"slot": key, "value": val, "handle": "plant_fact", "plan": ["bind", "run", "seal"]},
                requires_seal=True,
            )
        )
        t += 1
        for _ in range(rng.randint(1, 3)):
            events.append(_filler_event(t))
            t += 1
        events.append(_query_event(t, key))
        return events

    if stage == CurriculumStage.F:
        for key, val in facts:
            events.append(_plant_event(t, key, val))
            t += 1
        for key, _ in facts:
            events.append(_query_event(t, key))
            t += 1
        return events

    if stage == CurriculumStage.G:
        order = ["fact.name", "fact.code", "fact.item0"]
        for key in order:
            if key not in world.facts:
                continue
            events.append(_plant_event(t, key, str(world.facts[key])))
            t += 1
            if rng.random() < 0.35:
                events.append(_filler_event(t))
                t += 1
        for key in order:
            if key in world.facts:
                events.append(_query_event(t, key))
                t += 1
        return events

    # default B-like multi-fact
    for key, val in facts:
        events.append(_plant_event(t, key, val))
        t += 1
        if rng.random() < filler_ratio:
            events.append(_filler_event(t))
            t += 1
    for key, _ in facts:
        events.append(_query_event(t, key))
        t += 1
    return events
