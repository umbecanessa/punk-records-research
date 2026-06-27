# Punk Records Research

Public research track for **lawful agent memory** — session continuity as kernel state transitions, not chat tape continuation.

Named after the *One Piece* punk records: immutable logs that outlive any single speaker.  
The reference kernel implementation lives in the `greenfield/` Python package (legacy internal codename).

## Agent Kernel v0.1 (release stack)

| Layer | Role |
|-------|------|
| **Kernel** | Deterministic 8-opcode state machine + hash-linked log |
| **Encoder E6** | Learned opcodes, slot keys, values (~120k params) |
| **Renderer E3** | Untrusted text from storage (~1.3M params) |

**Weights:** [Hugging Face — `punk-records-research-kernel-v0.1`](https://huggingface.co/wasnaga/punk-records-research-kernel-v0.1)  
Training labels come from **world ground truth**, not chat transcripts.

### Results (v0.1)

| Metric | Score |
|--------|-------|
| Mean reward (curriculum A–F) | **0.991** |
| Query accuracy | **1.0** all stages |
| Render fidelity | **1.0** full stack |

See `hub/README.md` for the Hugging Face model card and `docs/KERNEL.md` for the opcode ABI.

---

## Install

```bash
git clone https://github.com/umbecanessa/punk-records-research.git
cd punk-records-research
pip install -e ".[dev]"
```

Download release weights:

```bash
huggingface-cli download wasnaga/punk-records-research-kernel-v0.1 \
  --local-dir greenfield/checkpoints/hf
```

Copy or symlink to expected paths:

```bash
cp greenfield/checkpoints/hf/encoder_e6_best.pt greenfield/checkpoints/
cp greenfield/checkpoints/hf/renderer_e3_best.pt greenfield/checkpoints/
```

---

## Reproduce

```bash
pytest tests/greenfield -q
python -u -m greenfield.eval_stack --device cuda
python -u -m greenfield.eval_learned --checkpoint greenfield/checkpoints/encoder_e6_best.pt
python -u -m greenfield.bench
```

Oracle baseline (no ML):

```bash
python -m greenfield.cli --stages A,B,C,D,E,F --episodes 20 --seed 0
```

---

## Train (optional)

Requires CUDA recommended. Warm-start chain:

```bash
python -u -m greenfield.train.train_encoder --device cuda
python -u -m greenfield.train.train_encoder_e2 --device cuda
python -u -m greenfield.train.train_encoder_e5b --device cuda
python -u -m greenfield.train.train_encoder_e6 --device cuda
python -u -m greenfield.train.train_renderer --device cuda
```

---

## Hugging Face

| Artifact | Link |
|----------|------|
| Weights | [wasnaga/punk-records-research-kernel-v0.1](https://huggingface.co/wasnaga/punk-records-research-kernel-v0.1) |
| Demo Space | [wasnaga/punk-records-research-demo](https://huggingface.co/spaces/wasnaga/punk-records-research-demo) |

Publish Space:

```powershell
cd punk-records-research
python -u scripts/publish_space_hf.py --dry-run
$env:HF_REPO_ID = "wasnaga/punk-records-research-demo"
python -u scripts/publish_space_hf.py --repo-id $env:HF_REPO_ID
```

## Next research (E7)

See [`docs/E7_RESEARCH.md`](docs/E7_RESEARCH.md) — NL parser, Space NL tab, richer worlds, real FORK/MERGE.

## Layout

```
greenfield/          Kernel runtime, train, eval (module name = legacy codename)
tests/greenfield/    Regression tests
docs/                Public specs
hub/                 Hugging Face model card + bundle manifest
scripts/             Publish + sync helpers
bench/greenfield/    Regression JSON artifacts
```

---

## Relation to private work

This repo is the **public face** of the agent-kernel research line.  
Chain-native LLM work (Lane C) and other private experiments are kept separate.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{punk_records_research_kernel_v01,
  title  = {Punk Records Research — Agent Kernel v0.1},
  year   = {2026},
  url    = {https://github.com/umbecanessa/punk-records-research},
  note   = {8-opcode session state machine; structured-event training; storage fidelity metrics.}
}
```
