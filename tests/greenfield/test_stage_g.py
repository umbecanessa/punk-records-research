"""Stage G (E7c) quest world — oracle smoke test."""

from __future__ import annotations

import random

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import bind_tools, default_tool_executor, quest_world


def test_stage_g_oracle_query_accuracy():
    policy = load_policy("greenfield/deploy/policy.v0.json")
    encoder = OracleEncoder()
    hits = 0
    for i in range(10):
        world = quest_world(random.Random(i))
        script = generate_script(world, stage=CurriculumStage.G, rng=random.Random(i + 1))
        _, metrics = run_episode(
            world=world,
            script=script,
            policy=policy,
            encoder=encoder,
            stage="G",
        )
        hits += metrics.query_accuracy
    assert hits / 10 >= 0.99


def test_stage_g_script_covers_all_facts():
    world = quest_world(random.Random(0))
    script = generate_script(world, stage=CurriculumStage.G, rng=random.Random(1))
    plants = {e.payload["slot"] for e in script if e.intent.value == "plant"}
    queries = {e.payload["slot"] for e in script if e.intent.value == "query"}
    assert plants == {"fact.name", "fact.code", "fact.item0"}
    assert queries == {"fact.name", "fact.code", "fact.item0"}
