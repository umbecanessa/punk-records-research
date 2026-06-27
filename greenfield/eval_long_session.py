"""E8 — long-session token curve vs chat baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from greenfield.learned_encoder import LearnedEncoder
from greenfield.nl_turn import run_nl_long_session
from greenfield.renderer.core import TemplateRenderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-session token savings curve")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e7", default="greenfield/checkpoints/encoder_e7_best.pt")
    parser.add_argument("--e6", default="greenfield/checkpoints/encoder_e6_best.pt")
    parser.add_argument("--json-out", default="bench/greenfield/long_session_latest.json")
    args = parser.parse_args()

    enc = LearnedEncoder.from_checkpoint(args.e6)
    ren = TemplateRenderer()
    _, _, metrics, err = run_nl_long_session(
        num_pairs=args.pairs,
        seed=args.seed,
        encoder=enc,
        renderer=ren,
        parser_checkpoint=Path(args.e7),
    )
    if err:
        raise SystemExit(err)

    curve = metrics.get("token_curve", [])
    final = metrics.get("tokens", {})
    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "pairs": args.pairs,
        "seed": args.seed,
        "error": err,
        "token_curve": curve,
        "final_tokens": final,
        "queries": metrics.get("queries", 0),
        "query_hits": metrics.get("query_hits", 0),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"=== Long session ({args.pairs} pairs) ===")
    print(f"queries: {metrics.get('query_hits', 0)}/{metrics.get('queries', 0)}")
    print(f"final saved: {final.get('tokens_saved', 0)} / {final.get('baseline_total_tokens', 0)} "
          f"({100 * final.get('savings_ratio', 0):.1f}%)")
    print("\nCurve (after each query):")
    for row in curve:
        print(
            f"  q{row['query_index']:2d}: saved={row['tokens_saved']:4d}  "
            f"ratio={row['savings_ratio']:.3f}  "
            f"baseline={row['baseline_total']} kernel={row['kernel_total']}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
