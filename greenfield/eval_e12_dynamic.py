"""E12 — eval dynamic retention (user.* keys + location)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from greenfield.chat_v1 import init_chat_session, load_chat_v1_stack, run_chat_turn


def eval_dynamic_session() -> dict:
    stack = load_chat_v1_stack(device=torch.device("cpu"))
    session = init_chat_session(stack)
    script = [
        ("hi there", "CHITCHAT", None),
        ("my name is Umberto", "PLANT", "fact.name"),
        ("I'm living in Amsterdam", "PLANT", "user.location"),
        ("oggi piove brutta giornata", "CHITCHAT", None),
        ("where do I live ?", "QUERY", "user.location"),
        ("whats my name ?", "QUERY", "fact.name"),
    ]
    rows = []
    ok = 0
    for text, intent, slot in script:
        session, reply, err = run_chat_turn(session, text)
        got_intent = session.metrics["turns"][-1]["intent"]
        passed = not err and got_intent.lower() == intent.lower()
        if intent == "QUERY" and slot:
            val = session.state.storage.slots.get(slot) or session.state.working.last_read.get(slot)
            passed = passed and val and str(val).lower() in reply.lower()
        if intent == "PLANT" and slot:
            passed = passed and session.state.storage.slots.get(slot)
        if passed:
            ok += 1
        rows.append({"user": text, "intent": got_intent, "reply": reply, "pass": passed})

    return {
        "passed": ok,
        "total": len(script),
        "accuracy": ok / len(script),
        "rows": rows,
    }


def main() -> None:
    metrics = eval_dynamic_session()
    out = Path("bench/greenfield/e12_dynamic_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    raise SystemExit(0 if metrics["accuracy"] >= 1.0 else 1)


if __name__ == "__main__":
    main()
