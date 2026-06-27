"""Shared eval helpers for greenfield research phases."""

from __future__ import annotations

import random

from greenfield.episodes import CurriculumStage, generate_script
from greenfield.learned_encoder import LearnedEncoder
from greenfield.renderer.core import Renderer
from greenfield.runner import run_episode, summarize
from greenfield.simulator import overflow_world, quest_world, sample_world
from greenfield.types import Policy, World


def world_for_stage(stage: CurriculumStage, ep_seed: int, *, num_facts: int = 5) -> World:
    if stage == CurriculumStage.F:
        return overflow_world(random.Random(ep_seed), num_facts=num_facts)
    if stage == CurriculumStage.G:
        return quest_world(random.Random(ep_seed))
    return sample_world(random.Random(ep_seed), num_facts=1 + (ep_seed % 2))


def run_stage_batch(
    *,
    stage: CurriculumStage,
    policy: Policy,
    encoder,
    episodes: int,
    seed: int,
    renderer: Renderer | None = None,
    reference_render=None,
    num_facts: int = 5,
) -> list:
    results = []
    for i in range(episodes):
        ep_seed = seed + i + ord(stage.value[0]) * 1000
        world = world_for_stage(stage, ep_seed, num_facts=num_facts)
        script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 1))
        _, metrics = run_episode(
            world=world,
            script=script,
            policy=policy,
            encoder=encoder,
            seed=ep_seed,
            stage=stage.value,
            renderer=renderer,
            reference_render=reference_render,
        )
        results.append(metrics)
    return results


def summarize_stages(
    metrics: list,
    stages: list[CurriculumStage],
) -> dict[str, dict]:
    by_stage: dict[str, list] = {}
    for m in metrics:
        by_stage.setdefault(m.stage, []).append(m)
    out = {}
    for stage in stages:
        items = by_stage.get(stage.value, [])
        if not items:
            continue
        base = summarize(items)[stage.value]
        out[stage.value] = {
            **base,
            "total_cold_hits": sum(x.cold_hits for x in items),
            "total_overflow_evictions": sum(x.overflow_evictions for x in items),
        }
    return out


def eval_encoder_curriculum(
    model,
    policy: Policy,
    stages: list[CurriculumStage],
    *,
    device,
    episodes: int,
    seed: int,
    use_learned_args: bool = False,
    use_learned_values: bool = False,
    num_facts: int = 5,
    overflow_policy: Policy | None = None,
    lambda_revert: float = 0.5,
) -> dict:
    per_stage = {}
    scores = []
    for stage in stages:
        stage_policy = overflow_policy if stage == CurriculumStage.F and overflow_policy else policy
        enc = LearnedEncoder(
            model,
            device=device,
            stage=stage.value,
            use_learned_args=use_learned_args,
            use_learned_values=use_learned_values,
        )
        metrics = run_stage_batch(
            stage=stage,
            policy=stage_policy,
            encoder=enc,
            episodes=episodes,
            seed=seed,
            num_facts=num_facts,
        )
        s = summarize(metrics)[stage.value]
        reward = s["mean_query_accuracy"] - lambda_revert * s["mean_revert_rate"]
        per_stage[stage.value] = {
            **s,
            "reward": round(reward, 4),
            "cold_hits": sum(m.cold_hits for m in metrics),
            "overflow_evictions": sum(m.overflow_evictions for m in metrics),
        }
        scores.append(reward)
    return {"mean_reward": sum(scores) / max(1, len(scores)), "stages": per_stage}
