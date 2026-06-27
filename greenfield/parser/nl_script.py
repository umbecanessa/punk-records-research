"""Build script events from parsed NL utterances."""

from __future__ import annotations

import random

from greenfield.episodes import CurriculumStage, generate_script
from greenfield.parser.template_parser import parse_utterance
from greenfield.simulator import sample_world
from greenfield.types import EpisodeEvent, Intent


def script_from_nl(
    plant_text: str,
    query_text: str,
    *,
    seed: int = 0,
) -> tuple[list[EpisodeEvent], str]:
    """Return (events, error). Uses parser for plant + query; filler from stage B."""
    plant = parse_utterance(plant_text)
    query = parse_utterance(query_text)
    if plant is None:
        return [], f"could not parse plant: {plant_text!r}"
    if query is None:
        return [], f"could not parse query: {query_text!r}"
    if plant.intent != Intent.PLANT:
        return [], "plant utterance must be a PLANT intent"
    if query.intent != Intent.QUERY:
        return [], "query utterance must be a QUERY intent"

    key = plant.payload.get("slot")
    val = plant.payload.get("value")
    if not key or val is None:
        return [], "plant missing slot/value"

    world = sample_world(random.Random(seed), num_facts=1)
    world.facts[str(key)] = str(val)

    base = generate_script(world, stage=CurriculumStage.B, rng=random.Random(seed + 1))
    # Keep filler from B but replace first plant and last query
    events: list[EpisodeEvent] = []
    planted = False
    for ev in base:
        if ev.intent == Intent.PLANT and not planted:
            events.append(
                EpisodeEvent(
                    t=ev.t,
                    source="user",
                    intent=Intent.PLANT,
                    payload=dict(plant.payload),
                    requires_seal=True,
                )
            )
            planted = True
        elif ev.intent == Intent.QUERY:
            continue
        else:
            events.append(ev)
    events.append(
        EpisodeEvent(
            t=max(e.t for e in events) + 1,
            source="user",
            intent=Intent.QUERY,
            payload=dict(query.payload),
            requires_seal=False,
        )
    )
    return events, ""
