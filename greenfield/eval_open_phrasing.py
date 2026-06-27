"""E10.1 — open phrasing NL eval."""

from __future__ import annotations

import random

from greenfield.kernel import Kernel
from greenfield.nl_gateway import LearnedEventParser
from greenfield.parser.open_phrasing import generate_open_paraphrase_batch
from greenfield.runner import load_policy
from greenfield.types import Intent


def eval_open_phrasing(parser: LearnedEventParser, *, size: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    state = Kernel(load_policy("greenfield/deploy/policy.v0.json")).genesis()
    rows = generate_open_paraphrase_batch(rng, size, messy_fraction=0.35)

    intent_ok = slot_ok = value_ok = 0
    intent_n = slot_n = value_n = 0

    for text, intent, slot, value in rows:
        parsed = parser.parse(text, state)
        if parsed is None:
            continue
        intent_n += 1
        if parsed.intent == intent:
            intent_ok += 1
        if intent in (Intent.PLANT, Intent.QUERY):
            slot_n += 1
            if parsed.payload.get("slot") == slot:
                slot_ok += 1
        if intent == Intent.PLANT:
            value_n += 1
            if str(parsed.payload.get("value", "")) == str(value):
                value_ok += 1

    return {
        "intent_acc": intent_ok / max(1, intent_n),
        "slot_acc": slot_ok / max(1, slot_n),
        "value_exact": value_ok / max(1, value_n),
        "samples": intent_n,
    }


def main() -> None:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="greenfield/checkpoints/encoder_e10a_best.pt")
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--seed", type=int, default=8080)
    args = p.parse_args()
    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    parser = LearnedEventParser.from_checkpoint(path)
    print(eval_open_phrasing(parser, size=args.size, seed=args.seed))


if __name__ == "__main__":
    main()
