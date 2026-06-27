"""Gradio version shims — Chatbot history format differs across 4.x / 5.x / Space images."""

from __future__ import annotations

import inspect

import gradio as gr

# Set once when the Chatbot component is constructed.
_CHAT_HISTORY_FORMAT: str = "tuples"  # "tuples" | "messages"


def _major_version() -> int:
    raw = getattr(gr, "__version__", "0").split(".")[0]
    try:
        return int(raw)
    except ValueError:
        return 0


def chatbot_supports_type_kwarg() -> bool:
    try:
        return "type" in inspect.signature(gr.Chatbot.__init__).parameters
    except (TypeError, ValueError):
        return False


def create_chatbot(**kwargs):
    """Build a Chatbot and record which history shape this runtime expects."""
    global _CHAT_HISTORY_FORMAT
    label = kwargs.pop("label", "Chat")
    if chatbot_supports_type_kwarg():
        _CHAT_HISTORY_FORMAT = "messages"
        return gr.Chatbot(label=label, type="messages", **kwargs)
    if _major_version() >= 5:
        _CHAT_HISTORY_FORMAT = "messages"
    else:
        _CHAT_HISTORY_FORMAT = "tuples"
    return gr.Chatbot(label=label, **kwargs)


def chat_history_format(history: list | None = None) -> str:
    """Infer tuple vs messages format from runtime flag or existing history."""
    if history:
        if isinstance(history[0], dict):
            return "messages"
        if isinstance(history[0], (list, tuple)):
            return "tuples"
    return _CHAT_HISTORY_FORMAT


def append_chat_turn(history: list, user: str, assistant: str) -> list:
    if chat_history_format(history) == "messages":
        return history + [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    return history + [[user, assistant]]
