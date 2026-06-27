---
license: apache-2.0
language:
  - en
tags:
  - agent
  - research
  - pytorch
  - state-machine
  - memory
library_name: pytorch
pipeline_tag: other
datasets:
  - synthetic
---

# Punk Records Research — Agent Kernel v0.1

**Research artifact — not a chat LLM.**

[*Punk Records*](https://github.com/umbecanessa/punk-records-research) — lawful agent memory as kernel state transitions (One Piece punk records metaphor: immutable logs that outlive any speaker).

| Layer | Artifact | Role |
|-------|----------|------|
| Kernel law | `policies/*.json` + `KERNEL.md` | 8-opcode invariants, gas, overflow |
| Encoder **E6** | `encoder_e6_best.pt` | Learned opcodes + slot keys + values |
| NL parser **E7** | `encoder_e7_best.pt` | Learned intent/slot + percept value bootstrap |
| Renderer **E3** | `renderer_e3_best.pt` | Untrusted text from storage |

**Source code:** [github.com/umbecanessa/punk-records-research](https://github.com/umbecanessa/punk-records-research)

---

## Results (v0.1)

| Metric | Score |
|--------|-------|
| Mean reward (A–F) | **0.991** |
| Query accuracy | **1.0** all stages |
| Render fidelity | **1.0** full stack |
| Value exact match | **1.0** |
| E7 NL parse (intent / slot / value) | **1.0 / 1.0 / 1.0** |

Frozen reports in `eval/`. Reproduce with the GitHub repo + these weights.

---

## Download

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download wasnaga/punk-records-research-kernel-v0.1 --local-dir ./kernel-v0.1
```

| File | Description |
|------|-------------|
| `encoder_e6_best.pt` | Release encoder (~460 KB) |
| `encoder_e7_best.pt` | NL event parser (~510 KB) |
| `encoder_e10a_best.pt` | Open-phrasing NL parser v2, 96-char (~7 MB) |
| `encoder_e9a_best.pt` | Transformer NL front (~3 MB) |
| `renderer_e9b_best.pt` | Transformer renderer (~3 MB) |
| `renderer_e3_best.pt` | Byte renderer (~5 MB) |
| `stack.json` | Bundle manifest |
| `policies/` | Kernel policy JSON |
| `eval/` | Training eval reports |
| `KERNEL.md` | Opcode ABI summary |

---

## Architecture

```
structured event + machine state
        ↓
  encoder E6 (MLP)
        ↓
  opcode trace (OBS → PUT → SEAL → … → GET → RENDER)
        ↓
  deterministic kernel
        ↓
  storage (+ cold overflow)
        ↓
  renderer E3 → answer text
```

Values at inference come from **OBS percept bytes in features**, not chat payload strings.

---

## Metrics

- **query_accuracy** — storage read matches world truth
- **revert_rate** — invalid ops rejected by kernel
- **render_fidelity** — template match (renderer untrusted)
- **reward** — `query_accuracy − 0.5 × revert_rate`

---

## Citation

```bibtex
@software{punk_records_research_kernel_v01,
  title  = {Punk Records Research — Agent Kernel v0.1},
  year   = {2026},
  url    = {https://huggingface.co/wasnaga/punk-records-research-kernel-v0.1},
  note   = {8-opcode session state machine; structured-event training.}
}
```

## License

Apache-2.0
