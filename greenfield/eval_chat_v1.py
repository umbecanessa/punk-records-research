"""E10 chat v1 eval — multi-turn NL + token savings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from greenfield.chat_v1 import default_chat_script, load_chat_v1_stack, run_nl_chat_session


def main() -> None:
    parser = argparse.ArgumentParser(description="E10 kernel-native chat eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="bench/greenfield/chat_v1_latest.json")
    args = parser.parse_args()

    stack = load_chat_v1_stack()
    script = default_chat_script()
    _, _, metrics, err = run_nl_chat_session(
        script,
        seed=args.seed,
        stack=stack,
        token_curve=True,
    )
    if err:
        raise SystemExit(err)

    tok = metrics.get("tokens", {})
    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "user_turns": metrics.get("user_turns", 0),
        "queries": metrics.get("queries", 0),
        "query_hits": metrics.get("query_hits", 0),
        "reverts": metrics.get("reverts", 0),
        "tokens": tok,
        "token_curve": metrics.get("token_curve", []),
        "turns": metrics.get("turns", []),
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== E10 chat v1 ===")
    print(f"turns: {report['user_turns']}  queries: {report['query_hits']}/{report['queries']}")
    print(f"reverts: {report['reverts']}")
    print(
        f"saved: {tok.get('tokens_saved', 0)} / {tok.get('baseline_total_tokens', 0)} "
        f"({100 * tok.get('savings_ratio', 0):.1f}%)"
    )
    for row in report["token_curve"]:
        print(
            f"  t{row['turn']:2d}: saved={row['tokens_saved']:4d}  "
            f"ratio={row['savings_ratio']:.3f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
