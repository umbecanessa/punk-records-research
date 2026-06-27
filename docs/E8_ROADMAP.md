# E8 — Roadmap to kernel-native chat (1–3B)

**Status:** E8 complete (10/10 gates) — E9a in progress  
**Track:** Greenfield kernel only (Lane C is separate)  
**North star:** A chatbot where **memory lives in STORAGE**, not in context replay.

---

## Architecture target

```
User chat ──► NL front (scale to 0.5–1B) ──► structured events
                      │
                      ▼
              Kernel + STORAGE (fixed, deterministic)
                      │
                      ▼
              Renderer (scale to 1–2B) ──► reply text
```

Language is a **renderer**. The kernel never learns from token CE on chat logs.

---

## Phase map

| Phase | Goal | Param scale | Exit criteria |
|-------|------|-------------|---------------|
| **E8a** | Generalization + long sessions | ~1M (current) | Messy NL eval ≥95%; 10-pair session savings ≥55% |
| **E8b** | Token curve + Space long demo | ~1M | Published curve JSON; Space long-session tab | **done** |
| **E9a** | Transformer NL front | 50M–350M | Same gates at scale; open phrasing corpus |
| **E9b** | Generative renderer | 350M–1B | Fluent answers; storage-only reads |
| **E10** | Chatbot v1 | 1–3B combined | Multi-turn NL demo; N-turn token savings chart |

---

## E8a — Generalization (now)

- Messy paraphrases: casing, fillers, punctuation (`eval_nl_messy`)
- **Chitchat in training mix** — expanded small-talk corpus + 85%+ messy augmentation on chitchat rows (`dataset_nl.augment_surface`, `train_encoder_e7 --messy-fraction 0.5`)
- Span extraction less template-bound (`value_span.py` → learned slot hints)
- Held-out names + items never in `TRAIN_*` pools
- **Gate:** `nl_messy` in `validate_release`

```bash
python -m greenfield.train.train_encoder_e7 --messy-fraction 0.5
python -m greenfield.eval_nl_messy
python -m greenfield.eval_long_session --pairs 10
```

---

## E8b — Long sessions (now)

- `run_nl_long_session(num_pairs=10..50)` — plant/query many facts
- **`token_curve`**: cumulative `tokens_saved` and `savings_ratio` after each query
- Prove savings **grow with session length** (kernel vs chat replay)
- **Gate:** `long_session` in `validate_release`

| Turns | Expected trend |
|-------|----------------|
| 3 facts (quest) | ~65% savings |
| 10 pairs | ≥55% savings |
| 20+ pairs | ratio increases (baseline grows O(n²), kernel O(n)) |

---

## E9 — Scale the right parts

### E9a — NL front (encoder)

Replace 128-d MLP utterance slice with small **Transformer** over byte/char window:

- Input: OBS utterance + working context features
- Output: intent, slot, value span boundaries
- Train on synthetic + messy corpus; kernel labels unchanged

Checkpoint ladder: **50M → 350M → 1B** (ablation at each rung).

### E9b — Renderer

Replace byte template renderer with **causal LM** that:

- Reads only `RENDER` keys from storage (never writes)
- Conditions on slot values + render_spec
- Target: natural replies at 350M–1B

---

## E10 — Chatbot v1 (1–3B)

- Free-form multi-turn NL in Space (not scripted plant/query boxes)
- Kernel plants/queries from parsed intents; chitchat → OBS only
- Success metrics:
  - `query_accuracy` on planted facts
  - `revert_rate` low
  - **`tokens_saved` vs turn count** (primary product metric)
  - User-facing: “it remembers me after 50 turns”

Not success metrics: perplexity on OpenWebText, MMLU, etc.

---

## Commands (E8)

```bash
python -m greenfield.validate_release          # 10 gates incl. E8
python -m greenfield.eval_long_session         # token curve report
python -m greenfield.eval_nl_messy             # generalization
python -m greenfield.eval_token_savings        # quest + overflow battery
```

---

## Relation to v0.1

| v0.1 (done) | E8+ |
|-------------|-----|
| Tiny MLP E6/E7 | Scale NL + renderer only |
| Template span | Learned span + messy text |
| 3-fact quest | 10–50 pair sessions |
| Token counter | Token **curve** vs turns |
| Space tabs | Long-session live chart |

Hub publish remains: **`validate_release` green → publish**.
