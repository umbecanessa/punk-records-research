"""Report token savings: kernel STORAGE path vs naive chat context replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from greenfield.learned_encoder import LearnedEncoder
from greenfield.nl_turn import (
    default_quest_turns,
    run_nl_episode_obs_first,
    run_nl_overflow_episode,
    run_nl_quest_episode,
)
from greenfield.renderer.core import TemplateRenderer
from greenfield.runner import load_policy


def run_battery(*, e7: Path, e6: Path) -> dict:
    policy = load_policy("greenfield/deploy/policy.v0.json")
    overflow_policy = load_policy("greenfield/deploy/policy.overflow.json")
    encoder = LearnedEncoder.from_checkpoint(e6)
    enc_f = LearnedEncoder.from_checkpoint(e6, stage="F")
    renderer = TemplateRenderer()

    singles: list[dict] = []
    for plant, query, _label in [
        ("Remember my name is Ada", "What is my name?", "ada"),
        ("Remember my name is Umberto", "who am i", "umberto"),
        ("my item is brass key", "what is my item?", "item"),
    ]:
        _, _, metrics, err = run_nl_episode_obs_first(
            plant,
            query,
            policy=policy,
            encoder=encoder,
            renderer=renderer,
            parser_checkpoint=e7,
        )
        singles.append(
            {
                "label": _label,
                "error": err,
                "tokens": metrics.get("tokens", {}),
                "query_hits": metrics.get("query_hits", 0),
            }
        )

    _, _, quest_metrics, quest_err = run_nl_quest_episode(
        default_quest_turns(),
        policy=policy,
        encoder=encoder,
        renderer=renderer,
        parser_checkpoint=e7,
    )
    quest_tok = quest_metrics.get("tokens", {})

    _, _, overflow_metrics, overflow_err = run_nl_overflow_episode(
        seed=0,
        policy=overflow_policy,
        encoder=enc_f,
        renderer=renderer,
        parser_checkpoint=e7,
    )
    overflow_tok = overflow_metrics.get("tokens", {})

    total_saved = sum(s["tokens"].get("tokens_saved", 0) for s in singles if not s["error"])
    total_saved += quest_tok.get("tokens_saved", 0) if not quest_err else 0
    total_saved += overflow_tok.get("tokens_saved", 0) if not overflow_err else 0

    return {
        "singles": singles,
        "quest": {"error": quest_err, "tokens": quest_tok, "answers": quest_metrics.get("answers", {})},
        "overflow": {
            "error": overflow_err,
            "tokens": overflow_tok,
            "cold_hits": overflow_metrics.get("cold_hits", 0),
            "evictions": overflow_metrics.get("overflow_evictions", 0),
        },
        "total_tokens_saved": total_saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Token savings vs chat-LLM baseline")
    parser.add_argument("--e7", default="greenfield/checkpoints/encoder_e7_best.pt")
    parser.add_argument("--e6", default="greenfield/checkpoints/encoder_e6_best.pt")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_battery(e7=Path(args.e7), e6=Path(args.e6))
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=== Token savings (kernel vs chat context replay) ===")
    for row in report["singles"]:
        if row["error"]:
            print(f"  {row['label']}: FAIL {row['error']}")
            continue
        t = row["tokens"]
        print(
            f"  {row['label']}: saved {t['tokens_saved']} / {t['baseline_total_tokens']} "
            f"({100 * t['savings_ratio']:.1f}%)"
        )
    q = report["quest"]
    if q["error"]:
        print(f"  quest: FAIL {q['error']}")
    else:
        t = q["tokens"]
        print(
            f"  stage-G quest (3 facts): saved {t['tokens_saved']} / {t['baseline_total_tokens']} "
            f"({100 * t['savings_ratio']:.1f}%)  input_saved={t['input_tokens_saved']}"
        )
    o = report["overflow"]
    if o["error"]:
        print(f"  overflow: FAIL {o['error']}")
    else:
        t = o["tokens"]
        print(
            f"  stage-F overflow NL: saved {t['tokens_saved']} / {t['baseline_total_tokens']} "
            f"({100 * t['savings_ratio']:.1f}%)  cold_hits={o['cold_hits']} evictions={o['evictions']}"
        )
    print(f"total_tokens_saved across battery: {report['total_tokens_saved']}")


if __name__ == "__main__":
    main()
