# E7 — Natural language front + richer worlds

**Status:** planned (post v0.1 public release)  
**Builds on:** E6 release stack + Gradio Space demo

## Goal

Move from **structured synthetic events** to **player-facing input** without breaking kernel law:

```
free text  →  event parser  →  EpisodeEvent  →  E6 encoder  →  kernel
```

Storage truth remains in the kernel; the parser is untrusted like the renderer.

## Tracks (priority order)

### E7a — Text → event parser (minimal)

- Map short utterances to `Intent` + payload:
  - `"remember my name is Ada"` → `PLANT fact.name=Ada`
  - `"what is my name?"` → `QUERY fact.name`
- Start with **template + slot filling** (regex / tiny classifier), not full LLM.
- Metric: **parse accuracy** on synthetic NL corpus aligned with curriculum A–B.

### E7b — Gradio NL tab

- Add NL input to the Space demo (uses E7a parser → existing stack).
- Side-by-side: structured episode vs parsed episode on same world.

### E7c — Richer world schema

- Quest-sized worlds: inventory, NPC flags, multiple fact keys.
- Kernel unchanged; extend `fact.*` schema + curriculum stage G.

### E7d — Real extended opcodes

- FORK / MERGE / DELEGATE beyond stubs (`policy.extended.json`).
- Scenarios: branch exploration, merge cold segments, delegate tool budget.

## Non-goals (E7)

- End-to-end chat LLM pretrain
- Lane C chain-native merge (parallel track)
- Replacing E6 percept-copy value path before parser is stable

## Success criteria

| Milestone | Bar |
|-----------|-----|
| E7a parser | ≥95% intent+slot match on held-out NL templates |
| E7b Space | User can plant/query by typing English |
| E7c stage G | query_acc ≥0.95 with 3+ fact types |
| E7d fork/merge | Deterministic replay + query_acc 1.0 on branch script |

## Suggested first PR

1. `greenfield/parser/template_parser.py` + tests
2. `tests/greenfield/test_nl_parser.py`
3. Wire optional `--nl` path in demo / Space tab
