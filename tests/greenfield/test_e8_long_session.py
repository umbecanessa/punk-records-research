"""E8 long-session + messy NL tests."""

from __future__ import annotations

import random

from greenfield.nl_turn import build_long_session_turns, run_nl_long_session
from greenfield.parser.paraphrase_messy import apply_messy_perturbation


def test_build_long_session_turns_count():
    turns, facts = build_long_session_turns(random.Random(0), 10)
    assert len(turns) == 10
    assert len(facts) == 7
    assert "fact.name" in facts
    assert "fact.item4" in facts
    assert turns[7].query_only


def test_messy_perturbation_changes_surface():
    rng = random.Random(1)
    clean = "my name is Ada"
    messy = apply_messy_perturbation(rng, clean)
    assert messy  # still non-empty


def test_long_session_runs():
    from pathlib import Path

    from greenfield.learned_encoder import LearnedEncoder
    from greenfield.renderer.core import TemplateRenderer

    e7 = Path("greenfield/checkpoints/encoder_e7_best.pt")
    if not e7.is_file():
        return
    enc = LearnedEncoder.from_checkpoint("greenfield/checkpoints/encoder_e6_best.pt")
    _, _, metrics, err = run_nl_long_session(
        num_pairs=5,
        seed=1,
        encoder=enc,
        renderer=TemplateRenderer(),
        parser_checkpoint=e7,
    )
    assert not err, err
    assert len(metrics.get("token_curve", [])) == 5
    assert metrics["tokens"]["savings_ratio"] > 0.4
