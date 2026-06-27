# E9 — Scale NL front + renderer

**Status:** E9 complete · E10 v1 green — see [`E10_ROADMAP.md`](E10_ROADMAP.md)  
**Prerequisite:** E8 green

---

## E9a — Transformer NL front ✓

621K-param causal transformer over the 12-char utterance window.

| Checkpoint | `greenfield/checkpoints/encoder_e9a_best.pt` |
| Train | `python -m greenfield.train.train_encoder_e9a --device cuda` |

Messy NL eval: **100%**

---

## E9b — Generative renderer ✓

830K-param causal transformer: **slot + value → answer**. Reads `RENDER` keys only — never writes storage.

| Checkpoint | `greenfield/checkpoints/renderer_e9b_best.pt` |
| Train | `python -m greenfield.train.train_renderer_e9b --device cuda` |

Template fidelity: **100%** · stack render fidelity: **1.0**

---

## E10 — Chatbot v1 (next)

Free-form multi-turn NL in Space; kernel plants/queries from parsed intents; chitchat → OBS only.

---

## Commands

```bash
python -m greenfield.validate_release
python -m greenfield.eval_renderer_e9b
python -m greenfield.eval_nl_messy --checkpoint greenfield/checkpoints/encoder_e9a_best.pt
```
