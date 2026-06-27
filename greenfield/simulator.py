"""Simulator world + deterministic tool execution."""

from __future__ import annotations

import random
from typing import Any

from greenfield.types import Storage, World


def default_tool_executor(handle: str, args: dict, storage: Storage) -> dict[str, Any]:
    """Simulated external tools — deterministic, no LLM."""
    if handle == "lookup":
        key = str(args.get("key", ""))
        value = storage.slots.get(key)
        return {"slot_writes": {}, "observation": {"key": key, "value": value}}
    if handle == "plant_fact":
        key = str(args["key"])
        value = str(args["value"])
        return {"slot_writes": {key: value}, "observation": {"planted": key}}
    if handle == "echo":
        return {"slot_writes": {}, "observation": {"echo": args.get("text", "")}}
    raise ValueError(f"unknown sim tool: {handle}")


def overflow_world(rng: random.Random, *, num_facts: int = 5) -> World:
    """World with many dynamic fact.* slots for cold-store overflow tests."""
    facts = {f"fact.item{i}": f"val{i}-{rng.randint(100, 999)}" for i in range(num_facts)}
    handles = {
        "plant_fact": {"uri": "sim://plant_fact", "budget": 100},
        "lookup": {"uri": "sim://lookup", "budget": 10},
    }
    return World(facts=facts, tool_handles=handles)


def quest_world(rng: random.Random) -> World:
    """E7c: fixed trio of fact types for stage G."""
    return World(
        facts={
            "fact.name": rng.choice(["Ada", "Lin", "Sam", "Rin"]),
            "fact.code": str(rng.randint(1000, 9999)),
            "fact.item0": rng.choice(["brass key", "old map", "red gem"]),
        },
        tool_handles={
            "plant_fact": {"uri": "sim://plant_fact", "budget": 100},
            "lookup": {"uri": "sim://lookup", "budget": 10},
        },
    )


def sample_world(rng: random.Random, *, num_facts: int = 1) -> World:
    keys = ["fact.name", "fact.code", "fact.answer"]
    rng.shuffle(keys)
    chosen = keys[:num_facts]
    facts: dict[str, str] = {}
    for key in chosen:
        if key == "fact.name":
            facts[key] = rng.choice(["Ada", "Lin", "Sam", "Rin"])
        elif key == "fact.code":
            facts[key] = f"{rng.randint(1000, 9999)}"
        else:
            facts[key] = str(rng.randint(1, 99))
    handles = {
        "plant_fact": {"uri": "sim://plant_fact", "budget": 100},
        "lookup": {"uri": "sim://lookup", "budget": 10},
    }
    return World(facts=facts, tool_handles=handles)


def bind_tools(storage: Storage, world: World) -> None:
    for name, handle in world.tool_handles.items():
        storage.slots[f"tool.{name}"] = dict(handle)
