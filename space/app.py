"""Punk Records Research — interactive kernel demo (Gradio Space)."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
if not (ROOT / "greenfield").is_dir() and (ROOT.parent / "greenfield").is_dir():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from demo_util import run_interactive, run_nl_episode, run_stage_demo
except ImportError:
    from space.demo_util import run_interactive, run_nl_episode, run_stage_demo

DESCRIPTION = """
# Punk Records Research — Agent Kernel Demo

**Memory as lawful state transitions**, not chat history.

This Space runs the release stack (E6 encoder + E3 renderer + deterministic kernel) on synthetic curriculum episodes.
Plant a fact → kernel log + storage → query → renderer answer. Stage **F** overflows hot storage into **cold** memory.

Weights: [wasnaga/punk-records-research-kernel-v0.1](https://huggingface.co/wasnaga/punk-records-research-kernel-v0.1)  
Code: [github.com/umbecanessa/punk-records-research](https://github.com/umbecanessa/punk-records-research)
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Punk Records Research — Kernel Demo") as demo:
        gr.Markdown(DESCRIPTION)

        gr.Markdown(
            "Space: [punk-records-research-demo](https://huggingface.co/spaces/wasnaga/punk-records-research-demo) · "
            "Weights: [kernel-v0.1](https://huggingface.co/wasnaga/punk-records-research-kernel-v0.1)"
        )

        with gr.Tab("Interactive episode"):
            gr.Markdown("Run one traced episode. Pick a stage and seed; optional name override for stage A/B.")
            with gr.Row():
                stage_i = gr.Dropdown(["A", "B", "C", "D", "E", "F"], value="A", label="Curriculum stage")
                seed_i = gr.Number(value=0, precision=0, label="Seed")
                name_i = gr.Dropdown(["Ada", "Lin", "Sam", "Rin"], value="Ada", label="fact.name (A/B)")
            run_i = gr.Button("Run episode", variant="primary")
            summary_i = gr.Markdown()
            with gr.Row():
                world_i = gr.Markdown(label="World (ground truth)")
                trace_i = gr.Markdown(label="Opcode trace")
            log_i = gr.Textbox(label="Kernel log (tail)", lines=14)

            run_i.click(
                fn=lambda stage, seed, name: run_interactive(str(stage), int(seed), str(name)),
                inputs=[stage_i, seed_i, name_i],
                outputs=[summary_i, world_i, log_i, trace_i],
            )

        with gr.Tab("Natural language (E7a)"):
            gr.Markdown("Type English utterances → template parser → same kernel stack.")
            plant_t = gr.Textbox(label="Plant", value="Remember my name is Ada")
            query_t = gr.Textbox(label="Query", value="What's my name?")
            seed_n = gr.Number(value=0, precision=0, label="Seed")
            run_n = gr.Button("Run NL episode", variant="primary")
            sum_n = gr.Markdown()
            world_n = gr.Markdown()
            log_n = gr.Textbox(label="Kernel log", lines=10)
            trace_n = gr.Markdown()
            run_n.click(
                fn=lambda p, q, s: run_nl_episode(str(p), str(q), int(s)),
                inputs=[plant_t, query_t, seed_n],
                outputs=[sum_n, world_n, log_n, trace_n],
            )

        with gr.Tab("Batch eval"):
            gr.Markdown("Run N episodes and aggregate query accuracy / revert rate (same metrics as research eval).")
            with gr.Row():
                stage_b = gr.Dropdown(["A", "B", "C", "D", "E", "F"], value="A", label="Stage")
                seed_b = gr.Number(value=0, precision=0, label="Seed")
                episodes_b = gr.Slider(1, 25, value=5, step=1, label="Episodes")
            run_b = gr.Button("Run batch", variant="primary")
            out_b = gr.Markdown()
            run_b.click(
                fn=lambda stage, seed, n: run_stage_demo(str(stage), int(seed), int(n)),
                inputs=[stage_b, seed_b, episodes_b],
                outputs=out_b,
            )

        with gr.Tab("How it works"):
            gr.Markdown(
                """
### Stack
| Layer | Role |
|-------|------|
| **Kernel** | 8 opcodes — OBS, PUT, GET, RUN, SEAL, REVERT, … |
| **Encoder E6** | Predicts opcode trace + slot keys + values from OBS percept |
| **Renderer E3** | Untrusted text from storage (never writes state) |

### Stage F (overflow)
Hot fact cap = 2. Planting many sealed facts evicts oldest slots to **cold store**. Queries still hit cold memory via GET.

### Metrics
- **query_accuracy** — storage read matches world truth
- **revert_rate** — kernel rejected invalid ops
- **reward** = query_accuracy − 0.5 × revert_rate
                """
            )

    return demo


if __name__ == "__main__":
    build_app().launch()
