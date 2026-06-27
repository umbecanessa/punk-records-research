"""Episode runner, metrics, and pluggable renderer."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from greenfield.encoder import NoisyEncoder, OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.renderer.core import Renderer, StubRenderer, TemplateRenderer
from greenfield.simulator import bind_tools, default_tool_executor, sample_world
from greenfield.types import Intent, KernelRevert, MachineState, OpCode, Policy, World


@dataclass
class EpisodeMetrics:
    stage: str
    seed: int
    events: int = 0
    ops_attempted: int = 0
    ops_applied: int = 0
    reverts: int = 0
    queries: int = 0
    query_hits: int = 0
    renders: int = 0
    render_hits: int = 0
    render_total: int = 0
    gas_used: int = 0
    storage_keys: list[str] = field(default_factory=list)
    cold_hits: int = 0
    overflow_evictions: int = 0

    @property
    def revert_rate(self) -> float:
        return self.reverts / max(1, self.ops_attempted)

    @property
    def query_accuracy(self) -> float:
        return self.query_hits / max(1, self.queries)

    @property
    def render_fidelity(self) -> float:
        return self.render_hits / max(1, self.render_total)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "seed": self.seed,
            "events": self.events,
            "ops_attempted": self.ops_attempted,
            "ops_applied": self.ops_applied,
            "reverts": self.reverts,
            "revert_rate": round(self.revert_rate, 4),
            "queries": self.queries,
            "query_hits": self.query_hits,
            "query_accuracy": round(self.query_accuracy, 4),
            "renders": self.renders,
            "render_hits": self.render_hits,
            "render_total": self.render_total,
            "render_fidelity": round(self.render_fidelity, 4),
            "gas_used": self.gas_used,
            "storage_keys": self.storage_keys,
            "cold_hits": self.cold_hits,
            "overflow_evictions": self.overflow_evictions,
        }


def stub_renderer(state: MachineState, render_args: dict) -> str:
    return StubRenderer().render(state, render_args)


def run_episode(
    *,
    world: World,
    script: list,
    policy: Policy,
    encoder,
    seed: int = 0,
    stage: str = "B",
    renderer: Renderer | None = None,
    reference_render=None,
) -> tuple[MachineState, EpisodeMetrics]:
    kernel = Kernel(policy, tool_executor=default_tool_executor)
    state = kernel.genesis()
    bind_tools(state.storage, world)
    oracle = encoder if isinstance(encoder, OracleEncoder) else getattr(encoder, "oracle", encoder)
    view = renderer or TemplateRenderer()
    ref_fn = reference_render

    metrics = EpisodeMetrics(stage=stage, seed=seed, events=len(script))

    if hasattr(encoder, "stage"):
        encoder.stage = stage

    for event in script:
        proposals = encoder.propose(event, state, kernel)
        for proposal in proposals:
            metrics.ops_attempted += 1
            resolved = oracle.resolve_evidence(state, kernel, proposal)
            try:
                if resolved.op == OpCode.RENDER:
                    text = view.render(state, resolved.args)
                    kernel.apply(state, resolved)
                    metrics.renders += 1
                    if ref_fn and event.intent == Intent.QUERY:
                        key = event.slot_key()
                        if key:
                            expected = world.expected_value(key)
                            if expected is not None:
                                ref = ref_fn(key, str(expected))
                                metrics.render_total += 1
                                if text.strip() == ref.strip():
                                    metrics.render_hits += 1
                else:
                    state = kernel.apply(state, resolved)
                metrics.ops_applied += 1
            except KernelRevert:
                metrics.reverts += 1

            if event.intent == Intent.QUERY and resolved.op == OpCode.GET:
                metrics.queries += 1
                key = event.slot_key()
                expected = world.expected_value(key) if key else None
                got = state.working.last_read.get(key)
                if got == expected:
                    metrics.query_hits += 1

    metrics.gas_used = state.gas_used
    metrics.storage_keys = sorted(state.storage.slots.keys())
    metrics.cold_hits = state.cold_hits
    metrics.overflow_evictions = state.overflow_evictions
    return state, metrics


def run_curriculum_batch(
    *,
    policy: Policy,
    stages: list[CurriculumStage],
    episodes_per_stage: int,
    seed: int = 0,
    noise_rate: float = 0.0,
    renderer: Renderer | None = None,
) -> list[EpisodeMetrics]:
    oracle = OracleEncoder()
    encoder: OracleEncoder | NoisyEncoder = oracle
    if noise_rate > 0:
        encoder = NoisyEncoder(oracle, noise_rate=noise_rate, rng=random.Random(seed + 99))

    results: list[EpisodeMetrics] = []
    for stage in stages:
        for i in range(episodes_per_stage):
            ep_seed = seed + i + ord(stage.value[0]) * 1000
            world = sample_world(random.Random(ep_seed), num_facts=1 + (i % 2))
            script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 1))
            _, metrics = run_episode(
                world=world,
                script=script,
                policy=policy,
                encoder=encoder,
                seed=ep_seed,
                stage=stage.value,
                renderer=renderer,
            )
            results.append(metrics)
    return results


def load_policy(path: str | Path) -> Policy:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Policy.from_dict(data)


def summarize(results: list[EpisodeMetrics]) -> dict:
    by_stage: dict[str, list[EpisodeMetrics]] = {}
    for m in results:
        by_stage.setdefault(m.stage, []).append(m)
    summary = {}
    for stage, items in sorted(by_stage.items()):
        render_f = [x.render_fidelity for x in items if x.render_total > 0]
        summary[stage] = {
            "episodes": len(items),
            "mean_query_accuracy": round(sum(x.query_accuracy for x in items) / len(items), 4),
            "mean_revert_rate": round(sum(x.revert_rate for x in items) / len(items), 4),
            "mean_render_fidelity": round(sum(render_f) / len(render_f), 4) if render_f else None,
            "mean_gas": round(sum(x.gas_used for x in items) / len(items), 1),
        }
    return summary
