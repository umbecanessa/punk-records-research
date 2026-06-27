# E11 — v1.2 scale-up (~50M NL + ~50M renderer)

**Status:** training (`encoder_e11a_best.pt`, `renderer_e11b_best.pt`)  
**North star:** same kernel law, richer surface — open NL + paraphrase answers at 50M scale

---

## Rung ladder

| Rung | NL | Renderer | Params (each) | Capability |
|------|-----|----------|---------------|------------|
| v1.1 (E10.1) | E10a | E9b | ~2M / ~830K | Open phrasing, live chat |
| **v1.2 (E11)** | E11a | E11b | **~50M / ~49M** | Messy NL robustness, multi-template replies |
| v2 (next) | 350M–1B | 350M–1B | — | Arbitrary phrasing, broader slots |

---

## Architecture (unchanged law)

```
Natural chat → E11a parser (96-char, 50M) → PLANT | QUERY | CHITCHAT
                      │
         CHITCHAT → OBS only
         PLANT/QUERY → kernel STORAGE → E11b renderer → reply
```

Memory still lives in **STORAGE**, not context replay.

---

## Presets (`greenfield/train/model_presets.py`)

| Preset | d_model | layers | nhead | ~params |
|--------|---------|--------|-------|---------|
| `NL_E11A` | 640 | 10 | 8 | 49.4M |
| `RENDERER_E11B` | 640 | 10 | 8 | 49.4M |

Checkpoints store `model_config` for load-time reconstruction.

---

## Train

```bash
# NL front (~200k open+messy samples, RTX 4080 ~2–4h)
python -m greenfield.train.train_encoder_e11a --device cuda

# Paraphrase renderer (~80k multi-template answers)
python -m greenfield.train.train_renderer_e11b --device cuda

python -m greenfield.validate_release   # 16 gates (+ e11a_open, e11b_renderer)
python -m greenfield.eval_chat_v1
```

---

## New modules

| Piece | Path |
|-------|------|
| Presets | `greenfield/train/model_presets.py` |
| E11a train | `greenfield/train/train_encoder_e11a.py` |
| E11b train | `greenfield/train/train_renderer_e11b.py` |
| Answer variants | `greenfield/renderer/phrasing.py` |
| Paraphrase dataset | `greenfield/renderer/dataset.py` → `ParaphraseRenderDataset` |

---

## Exit criteria (E11)

- `e11a_open`: intent ≥97%, slot ≥95%, messy intent ≥97%
- `e11b_renderer`: paraphrase val exact ≥98%
- `chat_v1`: 3/3 queries, 0 reverts, token savings ≥35%
- Release gates: **16/16 PASS** with E11 checkpoints

---

## After E11 (v2 preview)

- Byte/stream NL front (longer context than 96-char window)
- New memory slots beyond name/code/item
- Hub stack v0.2 manifest + Space auto-pull E11 weights
