"""Quick diagnostic for E7 value extraction errors."""

from __future__ import annotations

import random

import torch

from greenfield.kernel import Kernel
from greenfield.nl_gateway import LearnedEventParser
from greenfield.parser.paraphrases import generate_paraphrase_batch
from greenfield.runner import load_policy
from greenfield.types import Intent


def main() -> None:
    parser = LearnedEventParser.from_checkpoint(
        "greenfield/checkpoints/encoder_e7_best.pt",
        device=torch.device("cpu"),
    )
    kernel = Kernel(load_policy("greenfield/deploy/policy.v0.json"))
    state = kernel.genesis()
    rows = generate_paraphrase_batch(random.Random(123), 100)

    plant_total = plant_ok = 0
    for text, intent, slot, value in rows:
        if intent != Intent.PLANT:
            continue
        plant_total += 1
        p = parser.parse(text, state)
        got = p.payload.get("value") if p else None
        ok = got == value
        plant_ok += int(ok)
        mark = "OK" if ok else "XX"
        print(f"{mark} {text!r} want={value!r} got={got!r}")

    print(f"\nvalue_exact on plants: {plant_ok}/{plant_total} = {plant_ok/max(1,plant_total):.3f}")


if __name__ == "__main__":
    main()
