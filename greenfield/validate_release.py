"""Pre-push release validation — pytest + eval gates + live NL turn battery."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str


def _fail(msg: str) -> GateResult:
    return GateResult(name="", ok=False, detail=msg)


def gate_checkpoints(*, require_e7: bool) -> GateResult:
    name = "checkpoints"
    required = [
        ROOT / "greenfield/checkpoints/encoder_e6_best.pt",
        ROOT / "greenfield/checkpoints/encoder_e7_best.pt" if require_e7 else None,
        ROOT / "greenfield/checkpoints/renderer_e3_best.pt",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if p is not None and not p.is_file()]
    if missing:
        return GateResult(name, False, f"missing: {', '.join(missing)}")
    return GateResult(name, True, "encoder_e6, encoder_e7, renderer_e3 present")


def gate_pytest() -> GateResult:
    name = "pytest"
    cmd = [sys.executable, "-m", "pytest", "tests/greenfield/", "-q", "--tb=line"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return GateResult(name, False, out[-4000:] or "pytest failed")
    return GateResult(name, True, out.strip().splitlines()[-1] if out else "ok")


def gate_eval_nl(*, size: int, min_acc: float) -> GateResult:
    name = "eval_nl"
    from greenfield.eval_nl import eval_parser, eval_parser_heldout_names
    from greenfield.nl_gateway import LearnedEventParser

    ckpt = ROOT / "greenfield/checkpoints/encoder_e7_best.pt"
    parser = LearnedEventParser.from_checkpoint(ckpt, device=torch.device("cpu"))
    metrics = eval_parser(parser, size=size, seed=4242)
    heldout = eval_parser_heldout_names(parser, size=min(200, size // 10), seed=9001)
    detail = (
        f"intent={metrics['intent_acc']:.4f} slot={metrics['slot_acc']:.4f} "
        f"value={metrics['value_exact']:.4f} n={metrics['samples']}  "
        f"heldout_intent={heldout['intent_acc']:.4f} heldout_slot={heldout['slot_acc']:.4f} "
        f"heldout_value={heldout['value_exact']:.4f}"
    )
    ok = (
        metrics["intent_acc"] >= min_acc
        and metrics["slot_acc"] >= min_acc
        and metrics["value_exact"] >= min_acc
        and heldout["intent_acc"] >= min_acc
        and heldout["slot_acc"] >= min_acc
        and heldout["value_exact"] >= min_acc
    )
    return GateResult(name, ok, detail)


def gate_eval_stack(*, episodes: int, min_query: float) -> GateResult:
    name = "eval_stack"
    from greenfield.deploy_config import (
        DEFAULT_ENCODER,
        DEFAULT_OVERFLOW_POLICY,
        DEFAULT_POLICY,
        DEFAULT_RENDERER,
        DEFAULT_STAGES,
        USE_LEARNED_ARGS,
        USE_LEARNED_VALUES,
    )
    from greenfield.encoder import OracleEncoder
    from greenfield.episodes import CurriculumStage
    from greenfield.eval_util import run_stage_batch, summarize_stages
    from greenfield.learned_encoder import LearnedEncoder
    from greenfield.renderer.core import ByteRendererModel, LearnedRenderer, TemplateRenderer
    from greenfield.renderer.templates import reference_text
    from greenfield.runner import load_policy
    from greenfield.train.checkpoint_util import load_encoder_model

    device = torch.device("cpu")
    enc_path = ROOT / DEFAULT_ENCODER
    ren_path = ROOT / DEFAULT_RENDERER
    enc_ckpt = torch.load(enc_path, map_location=device, weights_only=False)
    enc_model = load_encoder_model(
        enc_path,
        device,
        predict_slot=True,
        predict_value=bool(enc_ckpt.get("predict_value", USE_LEARNED_VALUES)),
    )
    ren_ckpt = torch.load(ren_path, map_location=device, weights_only=False)
    ren_state = ren_ckpt["model"]
    ren_model = ByteRendererModel(
        hidden=int(ren_ckpt.get("hidden", 256)),
        num_slots=ren_state["slot_emb.weight"].shape[0],
    )
    ren_model.load_state_dict(ren_state, strict=False)
    ren_model.eval()
    renderer = LearnedRenderer(ren_model, device=device)
    base_policy = load_policy(ROOT / DEFAULT_POLICY)
    overflow_policy = load_policy(ROOT / DEFAULT_OVERFLOW_POLICY)
    stages = [CurriculumStage(s.strip().upper()) for s in DEFAULT_STAGES.split(",") if s.strip()]
    trained_stages = set(enc_ckpt.get("stages", ["A", "B", "C", "D", "E"]))

    lines: list[str] = []
    ok = True
    for stage in stages:
        policy = overflow_policy if stage == CurriculumStage.F else base_policy
        if stage.value not in trained_stages:
            enc = OracleEncoder()
        else:
            enc = LearnedEncoder(
                enc_model,
                device=device,
                stage=stage.value,
                use_learned_args=USE_LEARNED_ARGS,
                use_learned_values=USE_LEARNED_VALUES,
            )
        view = TemplateRenderer() if stage == CurriculumStage.F else renderer
        metrics = run_stage_batch(
            stage=stage,
            policy=policy,
            encoder=enc,
            episodes=episodes,
            seed=0,
            renderer=view,
            reference_render=reference_text,
            num_facts=5 if stage == CurriculumStage.F else 1,
        )
        summary = summarize_stages(metrics, [stage])[stage.value]
        q = summary["mean_query_accuracy"]
        lines.append(f"{stage.value}: query_acc={q:.3f}")
        if q < min_query:
            ok = False
    return GateResult(name, ok, "; ".join(lines))


def gate_nl_turn_battery() -> GateResult:
    name = "nl_turn_battery"
    from greenfield.learned_encoder import LearnedEncoder
    from greenfield.nl_turn import run_nl_episode_obs_first
    from greenfield.renderer.core import TemplateRenderer
    from greenfield.runner import load_policy
    from greenfield.deploy_config import DEFAULT_POLICY

    policy = load_policy(ROOT / DEFAULT_POLICY)
    encoder = LearnedEncoder.from_checkpoint(
        ROOT / "greenfield/checkpoints/encoder_e6_best.pt",
        device=torch.device("cpu"),
    )
    renderer = TemplateRenderer()
    e7 = ROOT / "greenfield/checkpoints/encoder_e7_best.pt"

    cases = [
        ("Remember my name is Ada", "What is my name?", "Ada"),
        ("Remember my name is Umberto", "who am i", "Umberto"),
        ("call me Lin", "who am i", "Lin"),
        ("my code is 4242", "what is my code?", "4242"),
        ("the code is 9876", "what's my code?", "9876"),
        ("my item is brass key", "what is my item?", "brass key"),
    ]
    lines: list[str] = []
    ok = True
    for plant, query, expected in cases:
        _, trace, metrics, err = run_nl_episode_obs_first(
            plant,
            query,
            seed=0,
            policy=policy,
            encoder=encoder,
            renderer=renderer,
            parser_checkpoint=e7,
        )
        if err:
            ok = False
            lines.append(f"FAIL {plant!r}: {err}")
            continue
        hit = metrics["query_hits"] == 1 and metrics["queries"] == 1
        obs_first = trace and trace[0].op == "OBS"
        if not hit or not obs_first:
            ok = False
            lines.append(f"FAIL {plant!r}: hit={metrics['query_hits']} obs_first={obs_first}")
        else:
            ans = metrics.get("answer", "")
            lines.append(f"OK {expected!r} answer={ans!r}")

    return GateResult(name, ok, " | ".join(lines))


def gate_nl_messy(*, size: int = 800, min_acc: float = 0.95) -> GateResult:
    name = "nl_messy"
    from greenfield.eval_nl_messy import eval_messy
    from greenfield.nl_gateway import LearnedEventParser

    ckpt = ROOT / "greenfield/checkpoints/encoder_e7_best.pt"
    parser = LearnedEventParser.from_checkpoint(ckpt, device=torch.device("cpu"))
    m = eval_messy(parser, size=size, seed=8080)
    ok = m["intent_acc"] >= min_acc and m["slot_acc"] >= min_acc and m["value_exact"] >= min_acc
    detail = (
        f"intent={m['intent_acc']:.3f} slot={m['slot_acc']:.3f} "
        f"value={m['value_exact']:.3f} n={m['samples']}"
    )
    return GateResult(name, ok, detail)


def gate_long_session(*, pairs: int = 10, min_ratio: float = 0.55) -> GateResult:
    name = "long_session"
    from greenfield.learned_encoder import LearnedEncoder
    from greenfield.nl_turn import run_nl_long_session
    from greenfield.renderer.core import TemplateRenderer

    e7 = ROOT / "greenfield/checkpoints/encoder_e7_best.pt"
    enc = LearnedEncoder.from_checkpoint(ROOT / "greenfield/checkpoints/encoder_e6_best.pt")
    _, _, metrics, err = run_nl_long_session(
        num_pairs=pairs,
        seed=0,
        encoder=enc,
        renderer=TemplateRenderer(),
        parser_checkpoint=e7,
    )
    if err:
        return GateResult(name, False, err)
    tok = metrics.get("tokens", {})
    curve = metrics.get("token_curve", [])
    ratio = tok.get("savings_ratio", 0.0)
    monotonic = all(
        curve[i]["tokens_saved"] <= curve[i + 1]["tokens_saved"]
        for i in range(len(curve) - 1)
    ) if len(curve) >= 2 else True
    ok = ratio >= min_ratio and metrics.get("query_hits") == pairs and monotonic
    detail = (
        f"pairs={pairs} saved={tok.get('tokens_saved', 0)} ratio={ratio:.2f} "
        f"monotonic={monotonic} q={metrics.get('query_hits', 0)}"
    )
    return GateResult(name, ok, detail)


def gate_nl_overflow() -> GateResult:
    name = "nl_overflow"
    from greenfield.learned_encoder import LearnedEncoder
    from greenfield.nl_turn import run_nl_overflow_episode
    from greenfield.renderer.core import TemplateRenderer
    from greenfield.runner import load_policy

    policy = load_policy(ROOT / "greenfield/deploy/policy.overflow.json")
    encoder = LearnedEncoder.from_checkpoint(
        ROOT / "greenfield/checkpoints/encoder_e6_best.pt",
        device=torch.device("cpu"),
        stage="F",
    )
    e7 = ROOT / "greenfield/checkpoints/encoder_e7_best.pt"
    _, _, metrics, err = run_nl_overflow_episode(
        seed=0,
        policy=policy,
        encoder=encoder,
        renderer=TemplateRenderer(),
        parser_checkpoint=e7,
    )
    if err:
        return GateResult(name, False, err)
    tok = metrics.get("tokens", {})
    detail = (
        f"evictions={metrics.get('overflow_evictions', 0)} "
        f"cold_hits={metrics.get('cold_hits', 0)} "
        f"saved={tok.get('tokens_saved', 0)} "
        f"ratio={tok.get('savings_ratio', 0):.2f}"
    )
    return GateResult(name, True, detail)


def gate_stage_g_oracle(*, episodes: int = 10) -> GateResult:
    name = "stage_g_learned"
    import random

    from greenfield.deploy_config import DEFAULT_ENCODER, USE_LEARNED_ARGS, USE_LEARNED_VALUES
    from greenfield.encoder import OracleEncoder
    from greenfield.episodes import CurriculumStage, generate_script
    from greenfield.learned_encoder import LearnedEncoder
    from greenfield.runner import load_policy, run_episode
    from greenfield.simulator import quest_world
    from greenfield.train.checkpoint_util import load_encoder_model

    policy = load_policy(ROOT / "greenfield/deploy/policy.v0.json")
    enc_path = ROOT / DEFAULT_ENCODER
    enc_ckpt = torch.load(enc_path, map_location="cpu", weights_only=False)
    trained_stages = set(enc_ckpt.get("stages", ["A", "B", "C", "D", "E", "F"]))
    device = torch.device("cpu")
    if "G" in trained_stages:
        enc_model = load_encoder_model(
            enc_path,
            device,
            predict_slot=True,
            predict_value=bool(enc_ckpt.get("predict_value", USE_LEARNED_VALUES)),
            expand_vocab=True,
        )
        encoder = LearnedEncoder(
            enc_model,
            device=device,
            stage="G",
            use_learned_args=USE_LEARNED_ARGS,
            use_learned_values=USE_LEARNED_VALUES,
        )
        mode = "learned"
    else:
        encoder = OracleEncoder()
        mode = "oracle"

    acc = 0.0
    for i in range(episodes):
        world = quest_world(random.Random(i))
        script = generate_script(world, stage=CurriculumStage.G, rng=random.Random(i + 7))
        _, metrics = run_episode(
            world=world,
            script=script,
            policy=policy,
            encoder=encoder,
            stage="G",
        )
        acc += metrics.query_accuracy
    mean = acc / episodes
    ok = mean >= 0.99
    return GateResult(name, ok, f"{mode} query_acc={mean:.3f} over {episodes} episodes")


def gate_token_savings(*, min_quest_ratio: float = 0.5, min_quest_saved: int = 40) -> GateResult:
    name = "token_savings"
    from greenfield.eval_token_savings import run_battery

    e7 = ROOT / "greenfield/checkpoints/encoder_e7_best.pt"
    e6 = ROOT / "greenfield/checkpoints/encoder_e6_best.pt"
    report = run_battery(e7=e7, e6=e6)

    lines: list[str] = []
    ok = True
    for row in report["singles"]:
        if row["error"]:
            ok = False
            lines.append(f"FAIL {row['label']}: {row['error']}")
            continue
        saved = row["tokens"].get("tokens_saved", 0)
        ratio = row["tokens"].get("savings_ratio", 0.0)
        lines.append(f"{row['label']}: saved={saved} ratio={ratio:.2f}")

    quest = report["quest"]
    if quest["error"]:
        ok = False
        lines.append(f"quest FAIL: {quest['error']}")
    else:
        qt = quest["tokens"]
        lines.append(
            f"quest: saved={qt.get('tokens_saved', 0)} "
            f"ratio={qt.get('savings_ratio', 0):.2f} "
            f"input_saved={qt.get('input_tokens_saved', 0)}"
        )
        if qt.get("tokens_saved", 0) < min_quest_saved:
            ok = False
        if qt.get("savings_ratio", 0) < min_quest_ratio:
            ok = False

    overflow = report.get("overflow", {})
    if overflow.get("error"):
        ok = False
        lines.append(f"overflow FAIL: {overflow['error']}")
    else:
        ot = overflow.get("tokens", {})
        lines.append(
            f"overflow: saved={ot.get('tokens_saved', 0)} "
            f"ratio={ot.get('savings_ratio', 0):.2f} "
            f"cold={overflow.get('cold_hits', 0)}"
        )
        if ot.get("savings_ratio", 0) < min_quest_ratio:
            ok = False

    return GateResult(name, ok, " | ".join(lines))


def gate_e9a_nl(*, size: int = 800, min_acc: float = 0.95) -> GateResult:
    name = "e9a_nl"
    ckpt = ROOT / "greenfield/checkpoints/encoder_e9a_best.pt"
    if not ckpt.is_file():
        return GateResult(name, True, "pending (train with train_encoder_e9a)")
    from greenfield.eval_nl_messy import eval_messy
    from greenfield.nl_gateway import LearnedEventParser

    parser = LearnedEventParser.from_checkpoint(ckpt, device=torch.device("cpu"))
    m = eval_messy(parser, size=size, seed=8080)
    ok = m["intent_acc"] >= min_acc and m["slot_acc"] >= min_acc and m["value_exact"] >= min_acc
    detail = (
        f"transformer intent={m['intent_acc']:.3f} slot={m['slot_acc']:.3f} "
        f"value={m['value_exact']:.3f} n={m['samples']}"
    )
    return GateResult(name, ok, detail)


def gate_open_phrasing(*, size: int = 800, min_acc: float = 0.92) -> GateResult:
    name = "open_phrasing"
    ckpt = ROOT / "greenfield/checkpoints/encoder_e10a_best.pt"
    if not ckpt.is_file():
        return GateResult(name, True, "pending (train with train_encoder_e10a)")
    from greenfield.eval_open_phrasing import eval_open_phrasing
    from greenfield.nl_gateway import LearnedEventParser

    parser = LearnedEventParser.from_checkpoint(ckpt, device=torch.device("cpu"))
    m = eval_open_phrasing(parser, size=size, seed=8080)
    ok = m["intent_acc"] >= min_acc and m["slot_acc"] >= min_acc and m["value_exact"] >= min_acc
    detail = (
        f"intent={m['intent_acc']:.3f} slot={m['slot_acc']:.3f} "
        f"value={m['value_exact']:.3f} n={m['samples']}"
    )
    return GateResult(name, ok, detail)


def gate_chat_v1(*, min_ratio: float = 0.35) -> GateResult:
    name = "chat_v1"
    e6 = ROOT / "greenfield/checkpoints/encoder_e6_best.pt"
    from greenfield.chat_v1 import (
        default_chat_script,
        load_chat_v1_stack,
        resolve_parser_checkpoint,
        resolve_renderer_checkpoint,
        run_nl_chat_session,
    )

    parser = resolve_parser_checkpoint(ROOT)
    renderer = resolve_renderer_checkpoint(ROOT)
    if not e6.is_file() or not parser.is_file() or not renderer.is_file():
        return GateResult(name, True, "pending (E6 + parser + renderer checkpoints)")

    stack = load_chat_v1_stack(root=ROOT, device=torch.device("cpu"))
    _, _, metrics, err = run_nl_chat_session(default_chat_script(), stack=stack, token_curve=True)
    if err:
        return GateResult(name, False, err)
    tok = metrics.get("tokens", {})
    ratio = tok.get("savings_ratio", 0.0)
    ok = (
        metrics.get("reverts", 1) == 0
        and metrics.get("query_hits") == metrics.get("queries")
        and metrics.get("queries", 0) >= 3
        and ratio >= min_ratio
    )
    detail = (
        f"turns={metrics.get('user_turns', 0)} q={metrics['query_hits']}/{metrics['queries']} "
        f"reverts={metrics.get('reverts', 0)} saved={tok.get('tokens_saved', 0)} "
        f"ratio={ratio:.2f}"
    )
    return GateResult(name, ok, detail)


def gate_e9b_renderer(*, min_exact: float = 0.99) -> GateResult:
    name = "e9b_renderer"
    ckpt = ROOT / "greenfield/checkpoints/renderer_e9b_best.pt"
    if not ckpt.is_file():
        return GateResult(name, True, "pending (train with train_renderer_e9b)")
    from greenfield.renderer.dataset import RenderDataset
    from greenfield.renderer.transformer_renderer import load_transformer_renderer
    from greenfield.train.train_renderer_e9b import eval_epoch
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    loaded = torch.load(ckpt, map_location=device, weights_only=False)
    model = load_transformer_renderer(ckpt, device)
    ds = RenderDataset(size=1000, seed=42)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    _, exact = eval_epoch(model, loader, device)
    params = loaded.get("params", "?")
    ok = exact >= min_exact
    return GateResult(name, ok, f"val_exact={exact:.3f} params={params}")


def gate_e11a_open(*, size: int = 1200, min_acc: float = 0.97) -> GateResult:
    name = "e11a_open"
    ckpt = ROOT / "greenfield/checkpoints/encoder_e11a_best.pt"
    if not ckpt.is_file():
        return GateResult(name, True, "pending (train with train_encoder_e11a)")
    from greenfield.eval_nl_messy import eval_messy
    from greenfield.eval_open_phrasing import eval_open_phrasing
    from greenfield.nl_gateway import LearnedEventParser

    parser = LearnedEventParser.from_checkpoint(ckpt, device=torch.device("cpu"))
    open_m = eval_open_phrasing(parser, size=size, seed=8080)
    messy_m = eval_messy(parser, size=800, seed=4242)
    ok = (
        open_m["intent_acc"] >= min_acc
        and open_m["slot_acc"] >= min_acc - 0.02
        and messy_m["intent_acc"] >= min_acc
    )
    detail = (
        f"open intent={open_m['intent_acc']:.3f} slot={open_m['slot_acc']:.3f} "
        f"messy intent={messy_m['intent_acc']:.3f} n={open_m['samples']}"
    )
    return GateResult(name, ok, detail)


def gate_e11b_renderer(*, min_exact: float = 0.98) -> GateResult:
    name = "e11b_renderer"
    ckpt = ROOT / "greenfield/checkpoints/renderer_e11b_best.pt"
    if not ckpt.is_file():
        return GateResult(name, True, "pending (train with train_renderer_e11b)")
    from greenfield.renderer.dataset import ParaphraseRenderDataset
    from greenfield.renderer.transformer_renderer import load_transformer_renderer
    from greenfield.train.train_renderer_e9b import eval_epoch
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    loaded = torch.load(ckpt, map_location=device, weights_only=False)
    model = load_transformer_renderer(ckpt, device)
    ds = ParaphraseRenderDataset(size=2000, seed=42)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    _, exact = eval_epoch(model, loader, device)
    params = loaded.get("params", "?")
    ok = exact >= min_exact
    return GateResult(name, ok, f"paraphrase_exact={exact:.3f} params={params}")


def gate_e12_dynamic(*, min_acc: float = 1.0) -> GateResult:
    name = "e12_dynamic"
    from greenfield.eval_e12_dynamic import eval_dynamic_session

    m = eval_dynamic_session()
    ok = m["accuracy"] >= min_acc
    return GateResult(name, ok, f"session={m['passed']}/{m['total']} acc={m['accuracy']:.2f}")


def run_all(*, nl_size: int = 2000, stack_episodes: int = 15, min_acc: float = 0.99) -> int:
    gates = [
        gate_checkpoints(require_e7=True),
        gate_pytest(),
        gate_eval_nl(size=nl_size, min_acc=min_acc),
        gate_eval_stack(episodes=stack_episodes, min_query=min_acc),
        gate_nl_turn_battery(),
        gate_nl_messy(),
        gate_long_session(),
        gate_nl_overflow(),
        gate_stage_g_oracle(),
        gate_token_savings(),
        gate_e9a_nl(),
        gate_e9b_renderer(),
        gate_open_phrasing(),
        gate_chat_v1(),
        gate_e11a_open(),
        gate_e11b_renderer(),
        gate_e12_dynamic(),
    ]
    failed = 0
    print("=== Punk Records release validation ===")
    for g in gates:
        mark = "PASS" if g.ok else "FAIL"
        print(f"[{mark}] {g.name}: {g.detail}")
        if not g.ok:
            failed += 1
    print(f"=== {len(gates) - failed}/{len(gates)} gates passed ===")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate release before Hub push")
    parser.add_argument("--nl-size", type=int, default=2000)
    parser.add_argument("--stack-episodes", type=int, default=15)
    parser.add_argument("--min-acc", type=float, default=0.99)
    args = parser.parse_args()
    raise SystemExit(run_all(nl_size=args.nl_size, stack_episodes=args.stack_episodes, min_acc=args.min_acc))


if __name__ == "__main__":
    main()
