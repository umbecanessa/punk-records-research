"""CLI for greenfield research runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from greenfield.episodes import CurriculumStage
from greenfield.runner import load_policy, run_curriculum_batch, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Greenfield kernel research runner")
    parser.add_argument(
        "--policy",
        default=str(Path(__file__).resolve().parent / "deploy" / "policy.v0.json"),
    )
    parser.add_argument("--stages", default="A,B,C,D,E", help="comma-separated curriculum stages")
    parser.add_argument("--episodes", type=int, default=20, help="episodes per stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0, help="invalid op injection rate")
    parser.add_argument("--out", default="", help="write JSON results to path")
    args = parser.parse_args()

    stages = [CurriculumStage(s.strip().upper()) for s in args.stages.split(",") if s.strip()]
    policy = load_policy(args.policy)
    results = run_curriculum_batch(
        policy=policy,
        stages=stages,
        episodes_per_stage=args.episodes,
        seed=args.seed,
        noise_rate=args.noise,
    )
    summary = summarize(results)

    print("=== Greenfield kernel research batch ===")
    print(f"policy: {args.policy}")
    print(f"stages: {[s.value for s in stages]}  episodes/stage: {args.episodes}  noise: {args.noise}")
    print()
    for stage, stats in summary.items():
        print(
            f"  stage {stage}: query_acc={stats['mean_query_accuracy']:.3f}  "
            f"revert={stats['mean_revert_rate']:.3f}  gas={stats['mean_gas']:.0f}"
        )

    payload = {
        "summary": summary,
        "episodes": [m.to_dict() for m in results],
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
