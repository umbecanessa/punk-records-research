"""Token accounting — kernel path vs naive chat-LLM context replay."""

from __future__ import annotations

from dataclasses import dataclass, field


# Fixed per-turn cost for small encoder pass (33-d feature vector vs full LM layer).
ENCODER_OVERHEAD_TOKENS = 8


def estimate_tokens(text: str) -> int:
    """Rough token count (word-piece heuristic, no external tokenizer)."""
    s = str(text).strip()
    if not s:
        return 0
    words = len(s.split())
    if words == 0:
        return max(1, len(s) // 4)
    return max(1, int(words * 1.33))


@dataclass
class TokenLedger:
    """Compare cumulative tokens: chat-LLM re-feeds history; kernel reads STORAGE."""

    chat_messages: list[tuple[str, str]] = field(default_factory=list)
    kernel_input: int = 0
    kernel_output: int = 0
    baseline_input: int = 0
    baseline_output: int = 0

    def record_chat(self, role: str, text: str) -> None:
        self.chat_messages.append((role, text))

    def record_kernel_turn(self, utterance: str, *, render_output: str = "") -> None:
        self.kernel_input += estimate_tokens(utterance) + ENCODER_OVERHEAD_TOKENS
        if render_output:
            self.kernel_output += estimate_tokens(render_output)

    def record_baseline_user(self, user_text: str) -> int:
        """Input tokens this turn — full chat history re-read (standard LLM API)."""
        charged = self._history_tokens() + estimate_tokens(user_text)
        self.baseline_input += charged
        self.record_chat("user", user_text)
        return charged

    def record_baseline_assistant(self, text: str) -> int:
        out = estimate_tokens(text)
        self.baseline_output += out
        self.record_chat("assistant", text)
        return out

    def _history_tokens(self) -> int:
        return sum(estimate_tokens(text) for _, text in self.chat_messages)

    @property
    def kernel_total(self) -> int:
        return self.kernel_input + self.kernel_output

    @property
    def baseline_total(self) -> int:
        return self.baseline_input + self.baseline_output

    @property
    def tokens_saved(self) -> int:
        return max(0, self.baseline_total - self.kernel_total)

    @property
    def savings_ratio(self) -> float:
        if self.baseline_total <= 0:
            return 0.0
        return self.tokens_saved / self.baseline_total

    @property
    def input_tokens_saved(self) -> int:
        return max(0, self.baseline_input - self.kernel_input)

    def to_dict(self) -> dict:
        return {
            "baseline_input_tokens": self.baseline_input,
            "baseline_output_tokens": self.baseline_output,
            "baseline_total_tokens": self.baseline_total,
            "kernel_input_tokens": self.kernel_input,
            "kernel_output_tokens": self.kernel_output,
            "kernel_total_tokens": self.kernel_total,
            "tokens_saved": self.tokens_saved,
            "input_tokens_saved": self.input_tokens_saved,
            "savings_ratio": round(self.savings_ratio, 4),
            "turns": len([m for m in self.chat_messages if m[0] == "user"]),
        }


def plant_ack_text(slot: str, value: str) -> str:
    if slot == "fact.name":
        return f"Got it — I'll remember your name is {value}."
    if slot == "fact.code":
        return f"Got it — code {value} stored."
    if slot.startswith("fact.item"):
        return f"Got it — item {value} stored."
    return f"Got it — {slot} = {value}."
