"""E5d regression bench — writes JSON artifacts under bench/greenfield/."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage
from greenfield.eval_util import run_stage_batch, summarize_stages
from greenfield.runner import load_policy, run_curriculum_batch, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description="Greenfield regression bench")
    parser.add_argument("--out-dir", default="bench/greenfield")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    base_policy = load_policy("greenfield/deploy/policy.v0.json")
    overflow_policy = load_policy("greenfield/deploy/policy.overflow.json")

    oracle_abce = run_curriculum_batch(
        policy=base_policy,
        stages=[CurriculumStage.A, CurriculumStage.B, CurriculumStage.C, CurriculumStage.D, CurriculumStage.E],
        episodes_per_stage=args.episodes,
        seed=args.seed,
    )
    overflow_f = run_stage_batch(
        stage=CurriculumStage.F,
        policy=overflow_policy,
        encoder=OracleEncoder(),
        episodes=args.episodes,
        seed=args.seed + 100,
    )

    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "oracle_abce": summarize([m for m in oracle_abce]),
        "oracle_overflow_f": summarize_stages(overflow_f, [CurriculumStage.F]).get("F", {}),
        "episodes": args.episodes,
    }
    out_path = out_dir / f"regression_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = out_dir / "regression_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {latest}")


if __name__ == "__main__":
    main()
