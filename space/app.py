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
    from demo_util import (
        run_chat_v1_demo,
        run_interactive,
        run_live_chat,
        reset_live_chat,
        run_nl_episode,
        run_nl_long_session_demo,
        run_nl_overflow_demo,
        run_nl_quest_demo,
        run_stage_demo,
    )
except ImportError:
    from space.demo_util import (
        run_chat_v1_demo,
        run_interactive,
        run_live_chat,
        reset_live_chat,
        run_nl_episode,
        run_nl_long_session_demo,
        run_nl_overflow_demo,
        run_nl_quest_demo,
        run_stage_demo,
    )

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

        with gr.Tab("Natural language (E7)"):
            gr.Markdown(
                "Type English → **OBS capture** → E7 parse → E6 opcode trace → kernel. "
                "Each utterance is logged as OBS before PUT/GET."
            )
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

        with gr.Tab("Quest (Stage G + tokens)"):
            gr.Markdown(
                "Three NL plant/query pairs — name, code, multi-word item. "
                "Shows **token savings** vs re-feeding full chat history each turn."
            )
            seed_q = gr.Number(value=0, precision=0, label="Seed")
            run_q = gr.Button("Run quest", variant="primary")
            sum_q = gr.Markdown()
            ans_q = gr.Markdown()
            log_q = gr.Textbox(label="Kernel log", lines=10)
            trace_q = gr.Markdown()
            run_q.click(
                fn=lambda s: run_nl_quest_demo(int(s)),
                inputs=[seed_q],
                outputs=[sum_q, ans_q, log_q, trace_q],
            )

        with gr.Tab("Overflow (Stage F + cold + tokens)"):
            gr.Markdown(
                "Plant/query five indexed items via NL. Hot cap = **2** → evictions to **cold store**. "
                "Queries still hit cold memory; token counter shows savings vs chat replay."
            )
            seed_o = gr.Number(value=0, precision=0, label="Seed")
            run_o = gr.Button("Run overflow NL", variant="primary")
            sum_o = gr.Markdown()
            ans_o = gr.Markdown()
            log_o = gr.Textbox(label="Kernel log", lines=12)
            trace_o = gr.Markdown()
            run_o.click(
                fn=lambda s: run_nl_overflow_demo(int(s)),
                inputs=[seed_o],
                outputs=[sum_o, ans_o, log_o, trace_o],
            )

        with gr.Tab("Long session (E8 + token curve)"):
            gr.Markdown(
                "Run **10+ NL plant/query pairs** in one session. "
                "Baseline grows like chat replay (O(n²)); kernel stays O(n). "
                "Table shows cumulative savings after each query."
            )
            with gr.Row():
                seed_l = gr.Number(value=0, precision=0, label="Seed")
                pairs_l = gr.Slider(3, 20, value=10, step=1, label="Plant/query pairs")
            run_l = gr.Button("Run long session", variant="primary")
            sum_l = gr.Markdown()
            storage_l = gr.Markdown(label="Hot storage")
            log_l = gr.Textbox(label="Kernel log", lines=12)
            trace_l = gr.Markdown()
            run_l.click(
                fn=lambda s, p: run_nl_long_session_demo(int(s), int(p)),
                inputs=[seed_l, pairs_l],
                outputs=[sum_l, storage_l, log_l, trace_l],
            )

        with gr.Tab("Live chat (E10.1 GPT path)"):
            gr.Markdown(
                "Talk turn-by-turn — plants stick in **kernel STORAGE** across messages. "
                "Try open phrasing: *By the way my name is Ada* → later *what name did I give you?*"
            )
            chatbot = gr.Chatbot(label="Chat", type="messages")
            chat_msg = gr.Textbox(label="Message", placeholder="Type and press Enter...")
            chat_session = gr.State(None)
            with gr.Row():
                send_c = gr.Button("Send", variant="primary")
                reset_c = gr.Button("Reset session")
            chat_stats = gr.Markdown()
            chat_log = gr.Textbox(label="Kernel log", lines=8)
            send_c.click(
                fn=run_live_chat,
                inputs=[chat_msg, chatbot, chat_session],
                outputs=[chatbot, chat_session, chat_stats, chat_log],
            ).then(lambda: "", outputs=[chat_msg])
            chat_msg.submit(
                fn=run_live_chat,
                inputs=[chat_msg, chatbot, chat_session],
                outputs=[chatbot, chat_session, chat_stats, chat_log],
            ).then(lambda: "", outputs=[chat_msg])
            reset_c.click(
                fn=reset_live_chat,
                outputs=[chatbot, chat_session, chat_stats, chat_log],
            )

        with gr.Tab("Chat v1 (E10)"):
            gr.Markdown(
                "Free-form multi-turn chat — **one message per line**. "
                "Plants/queries hit kernel STORAGE; chitchat is OBS-only. "
                "E9a transformer NL + E9b transformer renderer."
            )
            default_chat = "\n".join([
                "Remember my name is Ada",
                "hello there",
                "my code is 4242",
                "what is my name?",
                "what is my code?",
            ])
            chat_in = gr.Textbox(
                label="Messages (one per line)",
                lines=8,
                value=default_chat,
            )
            run_c = gr.Button("Run chat session", variant="primary")
            sum_c = gr.Markdown()
            storage_c = gr.Markdown()
            log_c = gr.Textbox(label="Kernel log", lines=10)
            trace_c = gr.Markdown()
            run_c.click(
                fn=run_chat_v1_demo,
                inputs=[chat_in],
                outputs=[sum_c, storage_c, log_c, trace_c],
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
