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
from greenfield.chat_v1 import (
    ChatSessionState,
    default_chat_script,
    init_chat_session,
    load_chat_v1_stack,
    run_chat_turn,
    run_nl_chat_session,
)
from greenfield.renderer.templates import reference_text
from greenfield.runner import load_policy, run_episode
from greenfield.simulator import bind_tools, default_tool_executor, overflow_world, sample_world
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.nl_turn import (
    default_quest_turns,
    run_nl_episode_obs_first,
    run_nl_long_session,
    run_nl_overflow_episode,
    run_nl_quest_episode,
)
from greenfield.types import EpisodeEvent, Intent, KernelRevert, MachineState, OpCode

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


def resolve_checkpoint(name: str) -> Path:
    """Local dev checkpoint or Hub download (Space runtime)."""
    local = repo_root() / "greenfield" / "checkpoints" / name
    if local.is_file():
        return local
    parent = repo_root().parent / "greenfield" / "checkpoints" / name
    if parent.is_file():
        return parent
    return download_asset(name)


def load_demo_chat_stack():
    """E10 chat stack — prefers E10a parser + E9b renderer from Hub or local."""
    root = repo_root()
    parser_path = None
    for parser_name in ("encoder_e10a_best.pt", "encoder_e9a_best.pt", "encoder_e7_best.pt"):
        try:
            parser_path = resolve_checkpoint(parser_name)
            break
        except Exception:
            continue
    if parser_path is None:
        raise FileNotFoundError("no NL parser checkpoint on Hub or locally")
    return load_chat_v1_stack(
        root=root,
        device=torch.device("cpu"),
        e6=resolve_checkpoint("encoder_e6_best.pt"),
        e9b=resolve_checkpoint("renderer_e9b_best.pt"),
        parser_checkpoint=parser_path,
    )


class DemoStack:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc_path = download_asset("encoder_e6_best.pt")
        self.e7_path: Path | None = None
        try:
            self.e7_path = download_asset("encoder_e7_best.pt")
        except Exception:
            local = repo_root() / "greenfield" / "checkpoints" / "encoder_e7_best.pt"
            if local.is_file():
                self.e7_path = local
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


def format_tokens(tokens: dict) -> str:
    if not tokens:
        return "_(no token metrics)_"
    return (
        f"- baseline total: **{tokens.get('baseline_total_tokens', 0)}** tokens\n"
        f"- kernel total: **{tokens.get('kernel_total_tokens', 0)}** tokens\n"
        f"- **saved: {tokens.get('tokens_saved', 0)}** "
        f"({100 * tokens.get('savings_ratio', 0):.1f}% · input saved {tokens.get('input_tokens_saved', 0)})"
    )


def format_token_curve(curve: list[dict]) -> str:
    if not curve:
        return "_(no token curve)_"
    lines = [
        "| Query | Saved | Ratio | Baseline | Kernel |",
        "|------:|------:|------:|---------:|-------:|",
    ]
    for row in curve:
        lines.append(
            f"| {row['query_index']} "
            f"| {row['tokens_saved']} "
            f"| {100 * row['savings_ratio']:.1f}% "
            f"| {row['baseline_total']} "
            f"| {row['kernel_total']} |"
        )
    return "\n".join(lines)


def _nl_trace_md(trace) -> str:
    lines = []
    for s in trace:
        mark = "ok" if s.applied else f"REVERT: {s.revert_reason}"
        utt = f" `{s.utterance}` →" if s.utterance else ""
        lines.append(f"**{s.op}** ({s.event_intent}){utt} {mark}")
    return "\n".join(lines)


def run_nl_episode(plant_text: str, query_text: str, seed: int) -> tuple[str, str, str, str]:
    try:
        stack = DemoStack()
        final, trace, metrics, err = run_nl_episode_obs_first(
            plant_text,
            query_text,
            seed=int(seed),
            policy=stack.base_policy,
            encoder=stack.encoder_for("B"),
            renderer=stack.learned_renderer,
            parser_checkpoint=stack.e7_path,
            stage="B",
        )
        if err:
            return f"**Parse error:** {err}", "", "", ""

        world_lines = [f"- `{k}` → `{v}`" for k, v in final.storage.slots.items() if str(k).startswith("fact.")]
        q_acc = metrics["query_hits"] / max(1, metrics["queries"])
        parser_note = "E7 learned · OBS-first" if stack.e7_path else "template · OBS-first"
        tok = metrics.get("tokens", {})
        summary = (
            f"### NL episode (E7b OBS-first)\n"
            f"- parser: **{parser_note}**\n"
            f"- plant: `{plant_text}`\n"
            f"- query: `{query_text}`\n"
            f"- query accuracy: **{q_acc:.0f}** · answer: **{metrics['answer'] or '(none)'}**\n\n"
            f"### Token savings vs chat replay\n{format_tokens(tok)}"
        )
        return summary, "\n".join(world_lines), format_log(final), _nl_trace_md(trace)
    except Exception as exc:
        return f"**Error:** {exc}", "", str(exc), ""


def run_nl_quest_demo(seed: int) -> tuple[str, str, str, str]:
    try:
        stack = DemoStack()
        final, trace, metrics, err = run_nl_quest_episode(
            default_quest_turns(),
            seed=int(seed),
            policy=stack.base_policy,
            encoder=stack.encoder_for("G"),
            renderer=stack.learned_renderer,
            parser_checkpoint=stack.e7_path,
        )
        if err:
            return f"**Quest error:** {err}", "", "", ""

        answers = metrics.get("answers", {})
        ans_lines = "\n".join(f"- `{k}` → **{v}**" for k, v in answers.items())
        hot = fact_slots(final)
        summary = (
            "### Stage-G quest (3 facts via NL)\n"
            f"- answers: {len(answers)} / 3\n\n"
            f"{ans_lines}\n\n"
            f"### Token savings vs chat replay\n{format_tokens(metrics.get('tokens', {}))}\n\n"
            f"**Hot storage**\n"
            + ("\n".join(f"- `{k}` = `{v}`" for k, v in hot.items()) if hot else "_(empty)_")
        )
        return summary, ans_lines, format_log(final), _nl_trace_md(trace)
    except Exception as exc:
        return f"**Error:** {exc}", "", str(exc), ""


def run_nl_overflow_demo(seed: int) -> tuple[str, str, str, str]:
    try:
        stack = DemoStack()
        final, trace, metrics, err = run_nl_overflow_episode(
            seed=int(seed),
            policy=stack.overflow_policy,
            encoder=stack.encoder_for("F"),
            renderer=TemplateRenderer(),
            parser_checkpoint=stack.e7_path,
        )
        if err:
            return f"**Overflow error:** {err}", "", "", ""

        cold = cold_keys(final)
        hot = fact_slots(final)
        summary = (
            "### Stage-F overflow (NL plant/query × 5 items)\n"
            f"- overflow evictions: **{metrics.get('overflow_evictions', 0)}**\n"
            f"- cold hits: **{metrics.get('cold_hits', 0)}**\n"
            f"- hot slots: {', '.join(f'`{k}`' for k in hot) or '_(none)_'}\n"
            f"- cold keys: {', '.join(f'`{k}`' for k in cold) or '_(none)_'}\n\n"
            f"### Token savings vs chat replay\n{format_tokens(metrics.get('tokens', {}))}"
        )
        world = "\n".join(f"- `{k}` → `{v}`" for k, v in metrics.get("answers", {}).items())
        return summary, world, format_log(final), _nl_trace_md(trace)
    except Exception as exc:
        return f"**Error:** {exc}", "", str(exc), ""


def run_nl_long_session_demo(seed: int, num_pairs: int) -> tuple[str, str, str, str]:
    try:
        stack = DemoStack()
        final, trace, metrics, err = run_nl_long_session(
            num_pairs=int(num_pairs),
            seed=int(seed),
            policy=stack.base_policy,
            encoder=stack.encoder_for("G"),
            renderer=stack.learned_renderer,
            parser_checkpoint=stack.e7_path,
        )
        if err:
            return f"**Long session error:** {err}", "", "", ""

        curve = metrics.get("token_curve", [])
        q_acc = metrics["query_hits"] / max(1, metrics["queries"])
        summary = (
            f"### E8 long session ({int(num_pairs)} plant/query pairs)\n"
            f"- query accuracy: **{q_acc:.0%}** "
            f"({metrics['query_hits']}/{metrics['queries']})\n\n"
            f"### Final token savings vs chat replay\n"
            f"{format_tokens(metrics.get('tokens', {}))}\n\n"
            f"### Token curve (after each query)\n"
            f"{format_token_curve(curve)}"
        )
        hot = fact_slots(final)
        world = "\n".join(f"- `{k}` = `{v}`" for k, v in hot.items()) or "_(empty)_"
        return summary, world, format_log(final), _nl_trace_md(trace)
    except Exception as exc:
        return f"**Error:** {exc}", "", str(exc), ""


def format_chat_transcript(turns: list[dict]) -> str:
    lines = []
    for row in turns:
        lines.append(f"**You:** {row.get('user', '')}")
        intent = row.get("intent", "?")
        reply = row.get("reply", "")
        lines.append(f"**Bot** ({intent}): {reply or '_(no reply)_'}")
        lines.append("")
    return "\n".join(lines).strip()


def run_chat_v1_demo(chat_text: str) -> tuple[str, str, str, str]:
    try:
        root = repo_root()
        stack = load_demo_chat_stack()
        lines = [ln.strip() for ln in str(chat_text).splitlines() if ln.strip()]
        if not lines:
            lines = default_chat_script()
        final, trace, metrics, err = run_nl_chat_session(lines, stack=stack, token_curve=True)
        if err:
            return f"**Chat error:** {err}", "", "", ""

        transcript = format_chat_transcript(metrics.get("turns", []))
        hot = fact_slots(final)
        summary = (
            "### E10 chat v1 (E9a NL + E9b renderer + kernel STORAGE)\n"
            f"- turns: **{metrics.get('user_turns', 0)}** · "
            f"queries: **{metrics['query_hits']}/{metrics['queries']}** · "
            f"reverts: **{metrics.get('reverts', 0)}**\n\n"
            f"### Token savings vs chat replay\n{format_tokens(metrics.get('tokens', {}))}\n\n"
            f"### Token curve\n{format_token_curve(metrics.get('token_curve', []))}\n\n"
            f"### Transcript\n{transcript}\n\n"
            f"**Hot storage**\n"
            + ("\n".join(f"- `{k}` = `{v}`" for k, v in hot.items()) if hot else "_(empty)_")
        )
        storage = "\n".join(f"- `{k}` = `{v}`" for k, v in hot.items()) or "_(empty)_"
        return summary, storage, format_log(final), _nl_trace_md(trace)
    except FileNotFoundError as exc:
        return f"**Missing E10 checkpoints:** {exc}", "", "", ""
    except Exception as exc:
        return f"**Error:** {exc}", "", str(exc), ""


def _empty_chat_session() -> ChatSessionState | None:
    try:
        return init_chat_session(load_demo_chat_stack())
    except FileNotFoundError:
        return None


def run_live_chat(message: str, history: list, session: ChatSessionState | None):
    """Gradio Chatbot — one message at a time; kernel memory persists in session."""
    if not str(message).strip():
        return history, session or _empty_chat_session(), "", ""
    try:
        if session is None:
            session = init_chat_session(load_demo_chat_stack())
        session, reply, err = run_chat_turn(session, message)
        if err:
            history = history + [[message, f"*(error)* {err}"]]
            return history, session, "", format_log(session.state)
        history = history + [[message, reply]]
        tok = session.ledger.to_dict()
        stats = (
            f"queries **{session.metrics['query_hits']}/{session.metrics['queries']}** · "
            f"reverts **{session.metrics.get('reverts', 0)}** · "
            f"saved **{tok.get('tokens_saved', 0)}** tok "
            f"({100 * tok.get('savings_ratio', 0):.0f}%)"
        )
        return history, session, stats, format_log(session.state)
    except FileNotFoundError as exc:
        history = history + [[message, f"*(missing checkpoints)* {exc}"]]
        return history, session, "", ""
    except Exception as exc:
        history = history + [[message, f"*(error)* {exc}"]]
        return history, session, "", str(exc)


def reset_live_chat():
    session = _empty_chat_session()
    return [], session, "Session reset.", ""
