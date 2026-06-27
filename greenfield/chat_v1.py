"""E10 — kernel-native chat v1 (E9a parser + E9b renderer + multi-turn NL)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import random
import torch

from greenfield.kernel import Kernel
from greenfield.learned_encoder import LearnedEncoder
from greenfield.nl_gateway import LearnedEventParser
from greenfield.nl_turn import TurnTrace, deploy_policy_path, run_nl_turn
from greenfield.renderer.core import LearnedTransformerRenderer, Renderer
from greenfield.renderer.templates import reference_text
from greenfield.runner import load_policy
from greenfield.simulator import bind_tools, default_tool_executor
from greenfield.token_accounting import TokenLedger, plant_ack_text
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.types import Intent, MachineState, Policy, World


@dataclass
class ChatTurn:
    user: str
    intent: str
    reply: str
    applied: bool


@dataclass
class ChatV1Stack:
    """Release stack for E10: E6 opcodes + E9a NL + E9b renderer."""

    encoder: LearnedEncoder
    parser: LearnedEventParser
    renderer: LearnedTransformerRenderer
    policy: Policy
    parser_checkpoint: Path
    renderer_checkpoint: Path
    device: torch.device


def default_chat_script() -> list[str]:
    """Open-phrasing multi-turn session (GPT-like surface forms)."""
    return [
        "By the way my name is Ada",
        "good to see you",
        "secret code for me is 4242",
        "what name did i give you",
        "what's the code i gave you",
        "don't forget my item is brass key",
        "what item did i mention",
    ]


@dataclass
class ChatSessionState:
    """Mutable session for live Space chatbot."""

    stack: ChatV1Stack
    kernel: Kernel
    state: MachineState
    world: World
    trace: list[TurnTrace]
    metrics: dict
    ledger: TokenLedger
    t: int


def init_chat_session(stack: ChatV1Stack) -> ChatSessionState:
    kernel = Kernel(stack.policy, tool_executor=default_tool_executor)
    state = kernel.genesis()
    world = World(
        facts={},
        tool_handles={
            "plant_fact": {"uri": "sim://plant_fact", "budget": 100},
            "lookup": {"uri": "sim://lookup", "budget": 10},
        },
    )
    bind_tools(state.storage, world)
    ledger = TokenLedger()
    return ChatSessionState(
        stack=stack,
        kernel=kernel,
        state=state,
        world=world,
        trace=[],
        metrics={
            "ops": 0,
            "queries": 0,
            "query_hits": 0,
            "reverts": 0,
            "turns": [],
            "_ledger": ledger,
            "_world_facts": world.facts,
        },
        ledger=ledger,
        t=0,
    )


def run_chat_turn(session: ChatSessionState, message: str) -> tuple[ChatSessionState, str, str]:
    """Single user message; returns (session, bot_reply, error)."""
    text = str(message).strip()
    if not text:
        return session, "", ""

    session.metrics["answer"] = ""
    parsed = session.stack.parser.parse(text, session.state, stage="G")
    if parsed and parsed.intent == Intent.QUERY:
        slot = parsed.payload.get("slot")
        if slot:
            session.metrics["_expected"] = session.world.expected_value(str(slot))

    session.state, event = run_nl_turn(
        text,
        t=session.t,
        state=session.state,
        kernel=session.kernel,
        encoder=session.stack.encoder,
        parser=session.stack.parser,
        renderer=session.stack.renderer,
        metrics=session.metrics,
        trace=session.trace,
        checkpoint=session.stack.parser_checkpoint,
        stage="G",
    )
    if event is None:
        return session, "", f"could not parse: {text!r}"

    reply = _reply_for_event(event, session.metrics, session.state)
    if event.intent == Intent.CHITCHAT:
        session.ledger.record_baseline_assistant(reply)

    session.metrics["turns"].append(
        {
            "user": text,
            "intent": event.intent.value,
            "reply": reply,
            "slot": event.slot_key(),
        }
    )
    session.t += 1
    return session, reply, ""


def _reply_for_event(event, metrics: dict, state: MachineState | None = None) -> str:
    if event.intent == Intent.QUERY:
        answer = str(metrics.get("answer", "") or "")
        key = event.slot_key()
        if key and state is not None:
            val = state.working.last_read.get(key)
            if val is None:
                val = state.storage.slots.get(key)
            if val is not None:
                val_s = str(val)
                if val_s.lower() not in answer.lower():
                    return reference_text(str(key), val_s)
        return answer
    if event.intent == Intent.PLANT:
        key = event.slot_key()
        val = event.slot_value()
        if key and val is not None:
            return plant_ack_text(str(key), str(val))
        return "Stored."
    if event.intent == Intent.CHITCHAT:
        if event.payload.get("reason") == "unsupported_query":
            return "I don't have that in memory yet — try name, code, item, or location."
        return "OK."
    return ""


def resolve_parser_checkpoint(root: Path) -> Path:
    for name in ("encoder_e11a_best.pt", "encoder_e10a_best.pt", "encoder_e9a_best.pt", "encoder_e7_best.pt"):
        path = root / "greenfield/checkpoints" / name
        if path.is_file():
            return path
    return root / "greenfield/checkpoints/encoder_e7_best.pt"


def resolve_renderer_checkpoint(root: Path) -> Path:
    for name in ("renderer_e11b_best.pt", "renderer_e9b_best.pt"):
        path = root / "greenfield/checkpoints" / name
        if path.is_file():
            return path
    return root / "greenfield/checkpoints/renderer_e3_best.pt"


def load_chat_v1_stack(
    *,
    root: Path | None = None,
    device: torch.device | None = None,
    e6: str | Path = "greenfield/checkpoints/encoder_e6_best.pt",
    renderer_checkpoint: str | Path | None = None,
    parser_checkpoint: Path | None = None,
    policy: str | Path | None = None,
) -> ChatV1Stack:
    root = root or Path(__file__).resolve().parents[1]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e6_path = Path(e6) if Path(e6).is_absolute() else root / e6
    parser_path = parser_checkpoint or resolve_parser_checkpoint(root)
    ren_path = Path(renderer_checkpoint) if renderer_checkpoint else resolve_renderer_checkpoint(root)
    if not ren_path.is_absolute():
        ren_path = root / ren_path
    if policy is not None and Path(policy).is_file():
        pol_path = Path(policy) if Path(policy).is_absolute() else root / policy
    else:
        dynamic = root / "greenfield/deploy/policy.dynamic.json"
        pol_path = dynamic if dynamic.is_file() else deploy_policy_path()
    if not pol_path.is_file():
        pol_path = root / "greenfield/deploy/policy.v0.json"

    for label, path in (("e6", e6_path), ("parser", parser_path), ("renderer", ren_path)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label} checkpoint: {path}")

    enc_model = load_encoder_model(
        e6_path,
        device,
        predict_slot=True,
        predict_value=True,
        predict_event=True,
        expand_vocab=True,
    )
    encoder = LearnedEncoder(
        enc_model,
        device=device,
        stage="G",
        use_learned_args=True,
        use_learned_values=True,
    )
    from greenfield.renderer.transformer_renderer import load_transformer_renderer

    parser = LearnedEventParser.from_checkpoint(parser_path, device=device, stage="G")
    ren_model = load_transformer_renderer(ren_path, device)
    renderer = LearnedTransformerRenderer(ren_model, device=device)

    return ChatV1Stack(
        encoder=encoder,
        parser=parser,
        renderer=renderer,
        policy=load_policy(pol_path),
        parser_checkpoint=parser_path,
        renderer_checkpoint=ren_path,
        device=device,
    )


def run_nl_chat_session(
    messages: list[str],
    *,
    seed: int = 0,
    stack: ChatV1Stack | None = None,
    policy: Policy | None = None,
    encoder: LearnedEncoder | None = None,
    renderer: Renderer | None = None,
    parser: LearnedEventParser | None = None,
    parser_checkpoint: Path | None = None,
    stage: str = "G",
    token_curve: bool = False,
) -> tuple[MachineState, list[TurnTrace], dict, str]:
    """Free-form user messages — one NL turn each; memory in STORAGE."""
    if stack is not None:
        policy = stack.policy
        encoder = stack.encoder
        renderer = stack.renderer
        parser = stack.parser
        parser_checkpoint = stack.parser_checkpoint

    policy = policy or load_policy(deploy_policy_path())
    kernel = Kernel(policy, tool_executor=default_tool_executor)
    state = kernel.genesis()
    world = World(facts={}, tool_handles={
        "plant_fact": {"uri": "sim://plant_fact", "budget": 100},
        "lookup": {"uri": "sim://lookup", "budget": 10},
    })
    bind_tools(state.storage, world)

    if encoder is None:
        return state, [], {}, "encoder required"

    trace: list[TurnTrace] = []
    ledger = TokenLedger()
    metrics: dict = {
        "ops": 0,
        "queries": 0,
        "query_hits": 0,
        "reverts": 0,
        "turns": [],
        "token_curve": [],
        "_ledger": ledger,
        "_world_facts": world.facts,
    }

    t = 0
    user_turn = 0
    for raw in messages:
        text = str(raw).strip()
        if not text:
            continue
        user_turn += 1
        metrics["answer"] = ""
        parsed = parser.parse(text, state, stage=stage) if parser else None
        if parsed and parsed.intent == Intent.QUERY:
            slot = parsed.payload.get("slot")
            if slot:
                metrics["_expected"] = world.expected_value(str(slot))

        state, event = run_nl_turn(
            text,
            t=t,
            state=state,
            kernel=kernel,
            encoder=encoder,
            parser=parser,
            renderer=renderer,
            metrics=metrics,
            trace=trace,
            checkpoint=parser_checkpoint,
            stage=stage,
        )
        if event is None:
            return state, trace, metrics, f"could not handle turn {user_turn}: {text!r}"

        reply = _reply_for_event(event, metrics, state)
        if event.intent == Intent.CHITCHAT:
            ledger.record_baseline_assistant(reply)

        metrics["turns"].append(
            {
                "user": text,
                "intent": event.intent.value,
                "reply": reply,
                "slot": event.slot_key(),
            }
        )
        if token_curve:
            snap = ledger.to_dict()
            metrics["token_curve"].append(
                {
                    "turn": user_turn,
                    "tokens_saved": snap["tokens_saved"],
                    "savings_ratio": snap["savings_ratio"],
                    "baseline_total": snap["baseline_total_tokens"],
                    "kernel_total": snap["kernel_total_tokens"],
                }
            )
        t += 1

    metrics.pop("_expected", None)
    metrics.pop("_world_facts", None)
    metrics.pop("_ledger", None)
    metrics["tokens"] = ledger.to_dict()
    metrics["user_turns"] = user_turn
    return state, trace, metrics, ""
