"""Token accounting and savings vs chat-LLM baseline."""

from __future__ import annotations

from greenfield.token_accounting import TokenLedger, estimate_tokens, plant_ack_text


def test_estimate_tokens_nonempty():
    assert estimate_tokens("hello world") >= 2
    assert estimate_tokens("") == 0


def test_ledger_single_plant_query_saves_on_query():
    ledger = TokenLedger()
    plant = "Remember my name is Umberto"
    query = "what is my name?"
    answer = "My name is Umberto."

    ledger.record_kernel_turn(plant)
    ledger.record_baseline_user(plant)
    ledger.record_baseline_assistant(plant_ack_text("fact.name", "Umberto"))

    ledger.record_kernel_turn(query, render_output=answer)
    ledger.record_baseline_user(query)
    ledger.record_baseline_assistant(answer)

    assert ledger.kernel_input < ledger.baseline_input
    assert ledger.tokens_saved > 0
    assert ledger.savings_ratio > 0.25


def test_quest_growth_baseline_grows_kernel_does_not_replay():
    ledger = TokenLedger()
    plants = [
        ("Remember my name is Ada", "fact.name", "Ada"),
        ("my code is 4242", "fact.code", "4242"),
        ("my item is brasskey", "fact.item0", "brasskey"),
    ]
    for text, slot, val in plants:
        ledger.record_kernel_turn(text)
        ledger.record_baseline_user(text)
        ledger.record_baseline_assistant(plant_ack_text(slot, val))

    baseline_before_last_query = ledger.baseline_input
    ledger.record_kernel_turn("what is my item?")
    ledger.record_baseline_user("what is my item?")

    assert ledger.baseline_input > baseline_before_last_query + 20
    assert ledger.kernel_input < ledger.baseline_input
