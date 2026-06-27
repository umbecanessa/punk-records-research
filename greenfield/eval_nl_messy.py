"""E8 — messy NL generalization eval."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from greenfield.kernel import Kernel
from greenfield.nl_gateway import LearnedEventParser
from greenfield.parser.paraphrase_messy import generate_messy_paraphrase_batch
from greenfield.runner import load_policy
from greenfield.types import Intent


def eval_messy(parser: LearnedEventParser, *, size: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    rows = generate_messy_paraphrase_batch(rng, size)

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
    parser = argparse.ArgumentParser(description="E8 messy NL eval")
    parser.add_argument("--checkpoint", default="greenfield/checkpoints/encoder_e7_best.pt")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"missing checkpoint: {path}")

    learned = LearnedEventParser.from_checkpoint(path)
    metrics = eval_messy(learned, size=args.size, seed=args.seed)
    print(metrics)


if __name__ == "__main__":
    main()
