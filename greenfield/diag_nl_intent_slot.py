"""Diagnose E7 intent/slot errors."""

from __future__ import annotations

import random
from collections import Counter

import torch

from greenfield.kernel import Kernel
from greenfield.nl_gateway import LearnedEventParser
from greenfield.parser.paraphrases import generate_paraphrase_batch
from greenfield.runner import load_policy


def main() -> None:
    parser = LearnedEventParser.from_checkpoint(
        "greenfield/checkpoints/encoder_e7_best.pt",
        device=torch.device("cpu"),
    )
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    rows = generate_paraphrase_batch(random.Random(42), 500)

    intent_err: Counter[str] = Counter()
    slot_err: Counter[str] = Counter()

    for text, intent, slot, value in rows:
        p = parser.parse(text, state)
        if p is None:
            intent_err[f"None <- {intent.value}"] += 1
            continue
        if p.intent != intent:
            intent_err[f"{p.intent.value} <- {intent.value} | {text!r}"] += 1
        if intent in (intent.PLANT, intent.QUERY) or True:
            from greenfield.types import Intent

            if intent in (Intent.PLANT, Intent.QUERY):
                got = p.payload.get("slot")
                if got != slot:
                    slot_err[f"{got} <- {slot} ({intent.value}) | {text!r}"] += 1

    print("intent errors:", sum(intent_err.values()))
    for k, v in intent_err.most_common(15):
        print(f"  {v}x {k}")
    print("slot errors:", sum(slot_err.values()))
    for k, v in slot_err.most_common(15):
        print(f"  {v}x {k}")


if __name__ == "__main__":
    main()
