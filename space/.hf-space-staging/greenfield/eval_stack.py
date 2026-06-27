"""Eval full stack: E2/E5b encoder + E3 renderer (stages A–F, optional overflow)."""

from __future__ import annotations

import argparse
import random

import torch

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
from greenfield.log_util import configure_unbuffered, log
from greenfield.renderer.core import ByteRendererModel, LearnedRenderer, TemplateRenderer
from greenfield.renderer.templates import reference_text
from greenfield.runner import load_policy
from greenfield.train.checkpoint_util import load_encoder_model


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Eval encoder + renderer stack")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--renderer", default=DEFAULT_RENDERER)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--overflow-policy", default=DEFAULT_OVERFLOW_POLICY)
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-facts", type=int, default=5)
    parser.add_argument(
        "--oracle-args",
        action="store_true",
        help="use oracle to materialize PUT/GET keys (E2 ablation; not release default)",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    stages = [CurriculumStage(s.strip().upper()) for s in args.stages.split(",") if s.strip()]
    base_policy = load_policy(args.policy)
    overflow_policy = load_policy(args.overflow_policy)

    enc_ckpt = torch.load(args.encoder, map_location=device, weights_only=False)
    trained_stages = set(enc_ckpt.get("stages", ["A", "B", "C", "D", "E"]))
    enc_model = load_encoder_model(
        args.encoder,
        device,
        predict_slot=True,
        predict_value=bool(enc_ckpt.get("predict_value", USE_LEARNED_VALUES)),
    )

    ckpt = torch.load(args.renderer, map_location=device, weights_only=False)
    ren_state = ckpt["model"]
    num_slots = ren_state["slot_emb.weight"].shape[0]
    ren_model = ByteRendererModel(hidden=int(ckpt.get("hidden", 256)), num_slots=num_slots)
    ren_model.load_state_dict(ren_state, strict=False)
    ren_model.to(device)
    ren_model.eval()
    renderer = LearnedRenderer(ren_model, device=device)

    log(f"encoder: {args.encoder}")
    log(f"renderer: {args.renderer}")
    log(f"device: {device}")
    use_learned_args = USE_LEARNED_ARGS and not args.oracle_args
    use_learned_values = USE_LEARNED_VALUES and not args.oracle_args
    log(f"stages: {[s.value for s in stages]}  learned_args: {use_learned_args}  learned_values: {use_learned_values}")

    all_metrics = []
    for stage in stages:
        policy = overflow_policy if stage == CurriculumStage.F else base_policy
        if stage.value not in trained_stages:
            enc = OracleEncoder()
            log(f"stage {stage.value}: oracle encoder (not in checkpoint stages)")
        else:
            enc = LearnedEncoder(
                enc_model,
                device=device,
                stage=stage.value,
                use_learned_args=use_learned_args,
                use_learned_values=use_learned_values,
            )
        if stage == CurriculumStage.F:
            view = TemplateRenderer()
        else:
            view = renderer
        metrics = run_stage_batch(
            stage=stage,
            policy=policy,
            encoder=enc,
            episodes=args.episodes,
            seed=args.seed,
            renderer=view,
            reference_render=reference_text,
            num_facts=args.num_facts,
        )
        all_metrics.extend(metrics)

    summary = summarize_stages(all_metrics, stages)
    for stage, stats in sorted(summary.items()):
        cold = stats.get("total_cold_hits", 0)
        evict = stats.get("total_overflow_evictions", 0)
        render_f = stats.get("mean_render_fidelity")
        render_s = f"{render_f:.3f}" if render_f is not None else "n/a"
        extra = ""
        if stage == "F":
            extra = f"  cold_hits={cold}  evictions={evict}"
        log(
            f"stage {stage}: query_acc={stats['mean_query_accuracy']:.3f}  "
            f"revert={stats['mean_revert_rate']:.3f}  render_fidelity={render_s}{extra}"
        )


if __name__ == "__main__":
    main()
