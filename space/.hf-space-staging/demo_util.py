"""Shared helpers for the Gradio Space demo."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from greenfield.episodes import CurriculumStage, generate_script
from greenfield.eval_util import run_stage_batch, summarize_stages
from greenfield.kernel import Kernel
from greenfield.learned_encoder import LearnedEncoder
from greenfield.renderer.core import ByteRendererModel, LearnedRenderer, TemplateRenderer
from greenfield.renderer.templates import reference_text
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import bind_tools, default_tool_executor, overflow_world, sample_world
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.parser.nl_script import script_from_nl
from greenfield.types import EpisodeEvent, Intent

HF_MODEL_REPO = "wasnaga/punk-records-research-kernel-v0.1"
CACHE = Path(__file__).resolve().parent / ".cache"


@dataclass
class TraceStep:
    event_idx: int
    event_intent: str
    op: str
    args: dict
    applied: bool
    revert_reason: str | None
    storage_facts: dict[str, str]
    cold_keys: list[str]
    gas: int


def download_asset(filename: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / filename.replace("/", "_")
    if local.is_file():
        return local
    path = hf_hub_download(HF_MODEL_REPO, filename, local_dir=str(CACHE))
    return Path(path)


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "greenfield").is_dir():
        return here
    return here.parent


def policy_path(name: str) -> Path:
    return repo_root() / "greenfield" / "deploy" / name


class DemoStack:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc_path = download_asset("encoder_e6_best.pt")
        ren_path = download_asset("renderer_e3_best.pt")
        ckpt = torch.load(enc_path, map_location=self.device, weights_only=False)
        self.enc_model = load_encoder_model(
            enc_path,
            self.device,
            predict_slot=True,
            predict_value=bool(ckpt.get("predict_value", True)),
        )
        ren_ckpt = torch.load(ren_path, map_location=self.device, weights_only=False)
        ren_state = ren_ckpt["model"]
        num_slots = ren_state["slot_emb.weight"].shape[0]
        ren_model = ByteRendererModel(hidden=int(ren_ckpt.get("hidden", 256)), num_slots=num_slots)
        ren_model.load_state_dict(ren_state, strict=False)
        ren_model.to(self.device)
        ren_model.eval()
        self.learned_renderer = LearnedRenderer(ren_model, device=self.device)
        self.base_policy = load_policy(policy_path("policy.v0.json"))
        self.overflow_policy = load_policy(policy_path("policy.overflow.json"))

    def encoder_for(self, stage: str) -> LearnedEncoder:
        return LearnedEncoder(
            self.enc_model,
            device=self.device,
            stage=stage,
            use_learned_args=True,
            use_learned_values=True,
        )

    def renderer_for(self, stage: str):
        if stage == "F":
            return TemplateRenderer()
        return self.learned_renderer


def fact_slots(state: MachineState) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in state.storage.slots.items():
        if key.startswith("fact."):
            out[key] = str(val)
    return dict(sorted(out.items()))


def cold_keys(state: MachineState) -> list[str]:
    if state.cold_store is None:
        return []
    return sorted(state.cold_store.key_index.keys())


def format_log(state: MachineState, *, tail: int = 24) -> str:
    lines = []
    for entry in state.log[-tail:]:
        args = {k: v for k, v in entry.args.items() if k != "payload"}
        if entry.op == OpCode.OBS:
            payload = entry.args.get("payload", {})
            if "value" in payload:
                args["value"] = payload["value"]
        lines.append(f"{entry.idx:03d} {entry.op.value:6s}  {json.dumps(args, ensure_ascii=False)}")
    return "\n".join(lines) if lines else "(empty log)"


def run_traced_episode(
    *,
    world,
    script,
    policy,
    encoder,
    renderer,
    seed: int,
    stage: str,
) -> tuple[MachineState, list[TraceStep], dict]:
    kernel = Kernel(policy, tool_executor=default_tool_executor)
    state = kernel.genesis()
    bind_tools(state.storage, world)
    trace: list[TraceStep] = []
    metrics = {
        "queries": 0,
        "query_hits": 0,
        "reverts": 0,
        "ops": 0,
        "answer": "",
    }

    for event_idx, event in enumerate(script):
        if hasattr(encoder, "stage"):
            encoder.stage = stage
        proposals = encoder.propose(event, state, kernel)
        for proposal in proposals:
            metrics["ops"] += 1
            resolved = encoder.resolve_evidence(state, kernel, proposal)
            applied = True
            revert_reason = None
            try:
                if resolved.op == OpCode.RENDER:
                    text = renderer.render(state, resolved.args)
                    kernel.apply(state, resolved)
                    if event.intent == Intent.QUERY:
                        metrics["answer"] = text.strip()
                else:
                    state = kernel.apply(state, resolved)
            except KernelRevert as exc:
                applied = False
                revert_reason = str(exc)
                metrics["reverts"] += 1

            if event.intent == Intent.QUERY and resolved.op == OpCode.GET:
                metrics["queries"] += 1
                key = event.slot_key()
                expected = world.expected_value(key) if key else None
                got = state.working.last_read.get(key)
                if got == expected:
                    metrics["query_hits"] += 1

            trace.append(
                TraceStep(
                    event_idx=event_idx,
                    event_intent=event.intent.value,
                    op=proposal.op.value,
                    args=dict(proposal.args),
                    applied=applied,
                    revert_reason=revert_reason,
                    storage_facts=fact_slots(state),
                    cold_keys=cold_keys(state),
                    gas=state.gas_used,
                )
            )

    return state, trace, metrics


def run_stage_demo(stage: str, seed: int, episodes: int = 5) -> str:
    stack = DemoStack()
    st = CurriculumStage(stage)
    policy = stack.overflow_policy if st == CurriculumStage.F else stack.base_policy
    enc = stack.encoder_for(stage)
    view = stack.renderer_for(stage)
    metrics = run_stage_batch(
        stage=st,
        policy=policy,
        encoder=enc,
        episodes=episodes,
        seed=seed,
        renderer=view,
        reference_render=reference_text,
        num_facts=5 if st == CurriculumStage.F else 1,
    )
    summary = summarize_stages(metrics, [st])[stage]
    lines = [
        f"**Stage {stage}** — {episodes} episodes, seed {seed}",
        "",
        f"- query accuracy: **{summary['mean_query_accuracy']:.3f}**",
        f"- revert rate: **{summary['mean_revert_rate']:.3f}**",
        f"- reward: **{summary['mean_query_accuracy'] - 0.5 * summary['mean_revert_rate']:.3f}**",
    ]
    if stage == "F":
        lines.append(f"- cold hits: **{summary.get('total_cold_hits', 0)}**")
        lines.append(f"- overflow evictions: **{summary.get('total_overflow_evictions', 0)}**")
    else:
        rf = summary.get("mean_render_fidelity")
        if rf is not None:
            lines.append(f"- render fidelity: **{rf:.3f}**")
    return "\n".join(lines)


def run_interactive(stage: str, seed: int, name: str | None = None) -> tuple[str, str, str, str]:
    stack = DemoStack()
    st = CurriculumStage(stage)
    rng = random.Random(seed)
    if st == CurriculumStage.F:
        world = overflow_world(rng, num_facts=5)
        policy = stack.overflow_policy
    else:
        world = sample_world(rng, num_facts=1)
        if name and "fact.name" in world.facts:
            world.facts["fact.name"] = name
        policy = stack.base_policy

    script = generate_script(world, stage=st, rng=random.Random(seed + 1))
    enc = stack.encoder_for(stage)
    view = stack.renderer_for(stage)
    final, trace, metrics = run_traced_episode(
        world=world,
        script=script,
        policy=policy,
        encoder=enc,
        renderer=view,
        seed=seed,
        stage=stage,
    )

    world_lines = [f"- `{k}` → `{v}` (ground truth, hidden from encoder labels at inference)" for k, v in world.facts.items()]
    trace_lines = []
    for step in trace:
        status = "ok" if step.applied else f"REVERT: {step.revert_reason}"
        trace_lines.append(f"**{step.op}** ({step.event_intent}) — {status}")
        if step.op in ("PUT", "RUN") and "value" in step.args:
            trace_lines.append(f"  value: `{step.args.get('value')}`")
        elif step.op == "RUN":
            inner = step.args.get("args", {})
            if "value" in inner:
                trace_lines.append(f"  value: `{inner.get('value')}`")

    hot = fact_slots(final)
    cold = cold_keys(final)
    storage_block = "**Hot storage (fact.*)**\n" + (
        "\n".join(f"- `{k}` = `{v}`" for k, v in hot.items()) if hot else "_(empty)_"
    )
    storage_block += "\n\n**Cold keys**\n" + (
        "\n".join(f"- `{k}`" for k in cold) if cold else "_(none)_"
    )

    q_acc = metrics["query_hits"] / max(1, metrics["queries"])
    summary = (
        f"### Episode result\n"
        f"- query accuracy: **{q_acc:.0f}** ({metrics['query_hits']}/{metrics['queries']})\n"
        f"- reverts: **{metrics['reverts']}**\n"
        f"- gas used: **{final.gas_used}**\n"
        f"- cold hits: **{final.cold_hits}** · evictions: **{final.overflow_evictions}**\n"
        f"- renderer answer: **{metrics['answer'] or '(none)'}**\n\n"
        f"{storage_block}"
    )

    return (
        summary,
        "\n".join(world_lines),
        format_log(final),
        "\n".join(trace_lines),
    )


def run_nl_episode(plant_text: str, query_text: str, seed: int) -> tuple[str, str, str, str]:
    script, err = script_from_nl(plant_text, query_text, seed=int(seed))
    if err:
        return f"**Parse error:** {err}", "", "", ""

    stack = DemoStack()
    rng = random.Random(int(seed))
    world = sample_world(rng, num_facts=1)
    plant = script[0]
    if plant.payload.get("slot") and plant.payload.get("value") is not None:
        world.facts[str(plant.payload["slot"])] = str(plant.payload["value"])

    enc = stack.encoder_for("B")
    view = stack.learned_renderer
    final, trace, metrics = run_traced_episode(
        world=world,
        script=script,
        policy=stack.base_policy,
        encoder=enc,
        renderer=view,
        seed=int(seed),
        stage="B",
    )

    world_lines = [f"- `{k}` → `{v}`" for k, v in world.facts.items()]
    trace_lines = [f"**{s.op}** ({s.event_intent})" for s in trace]
    q_acc = metrics["query_hits"] / max(1, metrics["queries"])
    summary = (
        f"### NL episode (stage B filler)\n"
        f"- plant: `{plant_text}`\n"
        f"- query: `{query_text}`\n"
        f"- query accuracy: **{q_acc:.0f}** · answer: **{metrics['answer'] or '(none)'}**"
    )
    return summary, "\n".join(world_lines), format_log(final), "\n".join(trace_lines)
