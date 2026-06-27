"""E7b — OBS-first NL turns: utterance → OBS → parse → encoder → kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import random

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.learned_encoder import LearnedEncoder
from greenfield.nl_gateway import LearnedEventParser, parse_nl
from greenfield.parser.template_parser import ParsedUtterance
from greenfield.renderer.core import Renderer
from greenfield.runner import load_policy
from greenfield.parser.value_span import obs_value_hint
from greenfield.simulator import bind_tools, default_tool_executor, overflow_world, quest_world, sample_world
from greenfield.token_accounting import TokenLedger, plant_ack_text
from greenfield.types import EpisodeEvent, Intent, KernelRevert, MachineState, OpCode, OpProposal, Policy, World


@dataclass
class TurnTrace:
    utterance: str | None
    event_intent: str
    op: str
    args: dict
    applied: bool
    revert_reason: str | None


def deploy_policy_path(name: str = "policy.v0.json") -> Path:
    return Path(__file__).resolve().parent / "deploy" / name


def obs_utterance_op(text: str, *, source: str = "user") -> OpProposal:
    """Capture raw user text in working.percept (+ trailing token as value hint)."""
    utext = text.strip()
    payload: dict = {"utterance": utext}
    tok = obs_value_hint(utext)
    if tok:
        payload["value"] = tok
    return OpProposal(op=OpCode.OBS, args={"source": source, "payload": payload})


def event_from_parsed(parsed: ParsedUtterance, text: str, *, t: int) -> EpisodeEvent:
    payload = {"utterance": text.strip(), **dict(parsed.payload)}
    requires_seal = parsed.intent in (
        Intent.PLANT,
        Intent.CHITCHAT,
        Intent.TOOL_PLANT,
        Intent.DISTRACTOR_PUT,
    )
    return EpisodeEvent(
        t=t,
        source="user",
        intent=parsed.intent,
        payload=payload,
        requires_seal=requires_seal,
    )


def _apply_proposals(
    *,
    state: MachineState,
    kernel: Kernel,
    encoder,
    proposals: list[OpProposal],
    event: EpisodeEvent,
    renderer: Renderer | None,
    metrics: dict,
    trace: list[TurnTrace],
    utterance: str | None,
) -> MachineState:
    oracle = encoder if isinstance(encoder, OracleEncoder) else getattr(encoder, "oracle", encoder)
    for proposal in proposals:
        metrics["ops"] += 1
        resolved = oracle.resolve_evidence(state, kernel, proposal)
        applied = True
        revert_reason = None
        try:
            if resolved.op == OpCode.RENDER:
                text = renderer.render(state, resolved.args) if renderer else ""
                kernel.apply(state, resolved)
                if event.intent == Intent.QUERY:
                    metrics["answer"] = text.strip()
            else:
                state = kernel.apply(state, resolved)
        except KernelRevert as exc:
            applied = False
            revert_reason = str(exc)
            metrics["reverts"] += 1

        if event.intent == Intent.QUERY and resolved.op == OpCode.GET:
            metrics["queries"] += 1
            key = event.slot_key()
            if key and metrics.get("_expected") is not None:
                got = state.working.last_read.get(key)
                if got == metrics["_expected"]:
                    metrics["query_hits"] += 1

        trace.append(
            TurnTrace(
                utterance=utterance,
                event_intent=event.intent.value,
                op=proposal.op.value,
                args=dict(proposal.args),
                applied=applied,
                revert_reason=revert_reason,
            )
        )
    return state


def _account_nl_turn(
    metrics: dict,
    *,
    utext: str,
    event: EpisodeEvent,
    slot: str | None,
    value: str | None,
) -> None:
    ledger: TokenLedger | None = metrics.get("_ledger")
    if ledger is None:
        return
    answer = metrics.get("answer", "") if event.intent == Intent.QUERY else ""
    ledger.record_kernel_turn(utext, render_output=answer if event.intent == Intent.QUERY else "")
    ledger.record_baseline_user(utext)
    if event.intent == Intent.PLANT and slot and value is not None:
        ledger.record_baseline_assistant(plant_ack_text(slot, value))
    elif event.intent == Intent.QUERY and answer:
        ledger.record_baseline_assistant(answer)


def _account_filler_chat(metrics: dict, utterance: str | None) -> None:
    ledger: TokenLedger | None = metrics.get("_ledger")
    if ledger is None or not utterance:
        return
    ledger.record_baseline_user(utterance)
    ledger.record_baseline_assistant("OK.")


def run_structured_turn(
    event: EpisodeEvent,
    state: MachineState,
    kernel: Kernel,
    encoder,
    renderer: Renderer | None,
    metrics: dict,
    trace: list[TurnTrace],
) -> MachineState:
    proposals = encoder.propose(event, state, kernel)
    state = _apply_proposals(
        state=state,
        kernel=kernel,
        encoder=encoder,
        proposals=proposals,
        event=event,
        renderer=renderer,
        metrics=metrics,
        trace=trace,
        utterance=event.payload.get("utterance"),
    )
    if metrics.get("_ledger") and event.intent == Intent.CHITCHAT:
        _account_filler_chat(metrics, event.payload.get("utterance"))
    return state


def run_nl_turn(
    text: str,
    *,
    t: int,
    state: MachineState,
    kernel: Kernel,
    encoder: LearnedEncoder,
    parser: LearnedEventParser | None,
    renderer: Renderer | None,
    metrics: dict,
    trace: list[TurnTrace],
    checkpoint: Path | None,
    stage: str = "B",
) -> tuple[MachineState, EpisodeEvent | None]:
    """OBS-first single utterance: capture → parse → opcode trace."""
    utext = text.strip()
    if not utext:
        return state, None

    obs = obs_utterance_op(utext)
    try:
        state = kernel.apply(state, obs)
    except KernelRevert as exc:
        trace.append(
            TurnTrace(
                utterance=utext,
                event_intent="?",
                op=OpCode.OBS.value,
                args=dict(obs.args),
                applied=False,
                revert_reason=str(exc),
            )
        )
        metrics["reverts"] += 1
        return state, None

    trace.append(
        TurnTrace(
            utterance=utext,
            event_intent="?",
            op=OpCode.OBS.value,
            args=dict(obs.args),
            applied=True,
            revert_reason=None,
        )
    )
    metrics["ops"] += 1

    parsed = (
        parser.parse(utext, state, stage=stage)
        if parser is not None
        else parse_nl(utext, state, checkpoint=checkpoint, stage=stage)
    )
    if parsed is None:
        return state, None

    event = event_from_parsed(parsed, utext, t=t)
    if event.intent == Intent.PLANT:
        key = event.slot_key()
        val = event.slot_value()
        if key and val is not None:
            metrics.setdefault("_world_facts", {})[str(key)] = str(val)

    proposals = encoder.propose_after_obs(event, state, kernel)
    state = _apply_proposals(
        state=state,
        kernel=kernel,
        encoder=encoder,
        proposals=proposals,
        event=event,
        renderer=renderer,
        metrics=metrics,
        trace=trace,
        utterance=utext,
    )
    _account_nl_turn(
        metrics,
        utext=utext,
        event=event,
        slot=event.slot_key(),
        value=str(event.slot_value()) if event.slot_value() is not None else None,
    )
    return state, event


def run_nl_episode_obs_first(
    plant_text: str,
    query_text: str,
    *,
    seed: int = 0,
    policy: Policy | None = None,
    encoder: LearnedEncoder | None = None,
    renderer: Renderer | None = None,
    parser_checkpoint: Path | None = None,
    stage: str = "B",
) -> tuple[MachineState, list[TurnTrace], dict, str]:
    """Plant + query via OBS-first NL turns with stage-B filler between."""
    policy = policy or load_policy(deploy_policy_path())
    kernel = Kernel(policy, tool_executor=default_tool_executor)
    state = kernel.genesis()

    parser: LearnedEventParser | None = None
    if parser_checkpoint is not None and parser_checkpoint.is_file():
        parser = LearnedEventParser.from_checkpoint(parser_checkpoint)

    genesis = kernel.genesis()
    plant_parsed = (
        parser.parse(plant_text.strip(), genesis, stage=stage)
        if parser
        else parse_nl(plant_text.strip(), genesis, checkpoint=parser_checkpoint, stage=stage)
    )
    if plant_parsed is None or plant_parsed.intent != Intent.PLANT:
        return state, [], {}, f"could not parse plant: {plant_text!r}"
    query_parsed = (
        parser.parse(query_text.strip(), genesis, stage=stage)
        if parser
        else parse_nl(query_text.strip(), genesis, checkpoint=parser_checkpoint, stage=stage)
    )
    if query_parsed is None or query_parsed.intent != Intent.QUERY:
        return state, [], {}, f"could not parse query: {query_text!r}"

    key = plant_parsed.payload.get("slot")
    val = plant_parsed.payload.get("value")
    if not key or val is None:
        return state, [], {}, "plant missing slot/value"

    world = sample_world(random.Random(seed), num_facts=1)
    world.facts[str(key)] = str(val)
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
        "answer": "",
        "_world_facts": dict(world.facts),
        "_expected": world.expected_value(query_parsed.payload.get("slot")),
        "_ledger": ledger,
    }

    base = generate_script(world, stage=CurriculumStage.B, rng=random.Random(seed + 1))
    t = 0
    planted = False
    for ev in base:
        if ev.intent == Intent.PLANT and not planted:
            state, _ = run_nl_turn(
                plant_text,
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
            planted = True
            t += 1
        elif ev.intent == Intent.QUERY:
            continue
        else:
            ev.t = t
            state = run_structured_turn(ev, state, kernel, encoder, renderer, metrics, trace)
            t += 1

    state, _ = run_nl_turn(
        query_text,
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

    metrics.pop("_world_facts", None)
    metrics.pop("_expected", None)
    metrics.pop("_ledger", None)
    metrics["tokens"] = ledger.to_dict()
    return state, trace, metrics, ""


@dataclass
class QuestTurn:
    plant: str
    query: str
    slot: str
    query_only: bool = False


def run_nl_quest_episode(
    turns: list[QuestTurn],
    *,
    seed: int = 0,
    policy: Policy | None = None,
    encoder: LearnedEncoder | None = None,
    renderer: Renderer | None = None,
    parser_checkpoint: Path | None = None,
    stage: str = "G",
) -> tuple[MachineState, list[TurnTrace], dict, str]:
    """Stage-G style episode: multiple NL plant/query pairs; facts live in STORAGE."""
    if not turns:
        policy = policy or load_policy(deploy_policy_path())
        return Kernel(policy).genesis(), [], {}, "no turns"

    world = quest_world(random.Random(seed))
    parser: LearnedEventParser | None = None
    if parser_checkpoint is not None and parser_checkpoint.is_file():
        parser = LearnedEventParser.from_checkpoint(parser_checkpoint, stage=stage)
    genesis = Kernel(load_policy(deploy_policy_path())).genesis()
    for qt in turns:
        p = (
            parser.parse(qt.plant.strip(), genesis, stage=stage)
            if parser
            else parse_nl(qt.plant.strip(), genesis, checkpoint=parser_checkpoint, stage=stage)
        )
        if p and p.payload.get("value"):
            world.facts[qt.slot] = str(p.payload["value"])

    return _run_nl_multi_turn(
        turns,
        world=world,
        seed=seed,
        policy=policy,
        encoder=encoder,
        renderer=renderer,
        parser_checkpoint=parser_checkpoint,
        stage=stage,
        token_curve=False,
    )


def default_quest_turns() -> list[QuestTurn]:
    return [
        QuestTurn("Remember my name is Umberto", "what is my name?", "fact.name"),
        QuestTurn("my code is 4242", "what is my code?", "fact.code"),
        QuestTurn("my item is brass key", "what is my item?", "fact.item0"),
    ]


def build_long_session_turns(rng: random.Random, num_pairs: int) -> tuple[list[QuestTurn], dict[str, str]]:
    """E8 — N plant/query pairs; core 3 + item1–4, then query-only replays."""
    from greenfield.parser.paraphrases import TRAIN_ITEMS, TRAIN_NAMES, sample_paraphrase

    facts: dict[str, str] = {}
    turns: list[QuestTurn] = []
    core_slots = ["fact.name", "fact.code", "fact.item0"]
    for i in range(num_pairs):
        if i < 3:
            slot = core_slots[i]
            if slot == "fact.name":
                val = rng.choice(TRAIN_NAMES)
            elif slot == "fact.code":
                val = str(rng.randint(1000, 9999))
            else:
                val = rng.choice(TRAIN_ITEMS)
            facts[slot] = val
            plant = sample_paraphrase(rng, intent=Intent.PLANT, slot=slot, value=val)
            query = sample_paraphrase(rng, intent=Intent.QUERY, slot=slot, value="")
            turns.append(QuestTurn(plant, query, slot))
        elif i < 7:
            idx = i - 2
            slot = f"fact.item{idx}"
            val = f"shard-{idx}-{rng.randint(100, 999)}"
            facts[slot] = val
            plant = f"remember item {idx} is {val}"
            query = f"what is item {idx}?"
            turns.append(QuestTurn(plant, query, slot))
        else:
            slot = core_slots[(i - 7) % 3]
            query = sample_paraphrase(rng, intent=Intent.QUERY, slot=slot, value="")
            turns.append(QuestTurn("", query, slot, query_only=True))
    return turns, facts


def _tool_handles() -> dict:
    return {
        "plant_fact": {"uri": "sim://plant_fact", "budget": 100},
        "lookup": {"uri": "sim://lookup", "budget": 10},
    }


def run_nl_long_session(
    *,
    num_pairs: int = 10,
    seed: int = 0,
    policy: Policy | None = None,
    encoder: LearnedEncoder | None = None,
    renderer: Renderer | None = None,
    parser_checkpoint: Path | None = None,
    stage: str = "G",
) -> tuple[MachineState, list[TurnTrace], dict, str]:
    """E8 — many NL plant/query pairs; records token_curve after each query."""
    turns, facts = build_long_session_turns(random.Random(seed + 1), num_pairs)
    world = World(facts=facts, tool_handles=_tool_handles())
    return _run_nl_multi_turn(
        turns,
        world=world,
        seed=seed,
        policy=policy,
        encoder=encoder,
        renderer=renderer,
        parser_checkpoint=parser_checkpoint,
        stage=stage,
        token_curve=True,
    )


def _run_nl_multi_turn(
    turns: list[QuestTurn],
    *,
    world: World,
    seed: int,
    policy: Policy | None,
    encoder: LearnedEncoder | None,
    renderer: Renderer | None,
    parser_checkpoint: Path | None,
    stage: str,
    token_curve: bool = False,
) -> tuple[MachineState, list[TurnTrace], dict, str]:
    policy = policy or load_policy(deploy_policy_path())
    kernel = Kernel(policy, tool_executor=default_tool_executor)
    state = kernel.genesis()
    bind_tools(state.storage, world)

    parser: LearnedEventParser | None = None
    if parser_checkpoint is not None and parser_checkpoint.is_file():
        parser = LearnedEventParser.from_checkpoint(parser_checkpoint, stage=stage)

    genesis = kernel.genesis()
    for qt in turns:
        for text, intent in ((qt.plant, Intent.PLANT), (qt.query, Intent.QUERY)):
            if intent == Intent.PLANT and (qt.query_only or not qt.plant.strip()):
                continue
            parsed = (
                parser.parse(text.strip(), genesis, stage=stage)
                if parser
                else parse_nl(text.strip(), genesis, checkpoint=parser_checkpoint, stage=stage)
            )
            if parsed is None or parsed.intent != intent:
                return state, [], {}, f"could not parse {intent.value}: {text!r}"
            if intent == Intent.PLANT:
                if parsed.payload.get("slot") != qt.slot:
                    return state, [], {}, f"plant slot mismatch for {text!r}"
                if not parsed.payload.get("value"):
                    return state, [], {}, f"plant missing value: {text!r}"

    if encoder is None:
        return state, [], {}, "encoder required"

    trace: list[TurnTrace] = []
    ledger = TokenLedger()
    metrics: dict = {
        "ops": 0,
        "queries": 0,
        "query_hits": 0,
        "reverts": 0,
        "answers": {},
        "_ledger": ledger,
        "token_curve": [],
    }

    t = 0
    query_index = 0
    for qt in turns:
        metrics["_expected"] = world.expected_value(qt.slot)
        metrics["answer"] = ""
        if not qt.query_only and qt.plant.strip():
            state, _ = run_nl_turn(
                qt.plant,
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
            t += 1
        metrics["answer"] = ""
        state, _ = run_nl_turn(
            qt.query,
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
        query_index += 1
        metrics["answers"][qt.slot] = metrics.get("answer", "")
        expected = world.expected_value(qt.slot)
        got = state.storage.slots.get(qt.slot) or state.working.last_read.get(qt.slot)
        if got != expected:
            return state, trace, metrics, f"query miss on {qt.slot}: got {got!r} expected {expected!r}"
        if token_curve:
            snap = ledger.to_dict()
            metrics["token_curve"].append(
                {
                    "query_index": query_index,
                    "tokens_saved": snap["tokens_saved"],
                    "savings_ratio": snap["savings_ratio"],
                    "baseline_total": snap["baseline_total_tokens"],
                    "kernel_total": snap["kernel_total_tokens"],
                }
            )
        t += 1

    metrics.pop("_expected", None)
    metrics.pop("_ledger", None)
    metrics["tokens"] = ledger.to_dict()
    metrics["num_pairs"] = len(turns)
    return state, trace, metrics, ""


def run_nl_overflow_episode(
    *,
    seed: int = 0,
    num_facts: int = 5,
    policy: Policy | None = None,
    encoder: LearnedEncoder | None = None,
    renderer: Renderer | None = None,
    parser_checkpoint: Path | None = None,
) -> tuple[MachineState, list[TurnTrace], dict, str]:
    """Stage F — NL plant/query many items; hot cap 2 → cold store + token ledger."""
    policy = policy or load_policy(deploy_policy_path("policy.overflow.json"))
    kernel = Kernel(policy, tool_executor=default_tool_executor)
    state = kernel.genesis()
    world = overflow_world(random.Random(seed), num_facts=num_facts)
    bind_tools(state.storage, world)

    parser: LearnedEventParser | None = None
    if parser_checkpoint is not None and parser_checkpoint.is_file():
        parser = LearnedEventParser.from_checkpoint(parser_checkpoint, stage="F")

    turns: list[QuestTurn] = []
    for i in range(num_facts):
        key = f"fact.item{i}"
        val = str(world.facts[key])
        turns.append(
            QuestTurn(
                plant=f"remember item {i} is {val}",
                query=f"what is item {i}?",
                slot=key,
            )
        )

    if encoder is None:
        return state, [], {}, "encoder required"

    trace: list[TurnTrace] = []
    ledger = TokenLedger()
    metrics: dict = {
        "ops": 0,
        "queries": 0,
        "query_hits": 0,
        "reverts": 0,
        "answers": {},
        "_ledger": ledger,
    }

    t = 0
    for qt in turns:
        metrics["_expected"] = world.expected_value(qt.slot)
        metrics["answer"] = ""
        state, _ = run_nl_turn(
            qt.plant,
            t=t,
            state=state,
            kernel=kernel,
            encoder=encoder,
            parser=parser,
            renderer=renderer,
            metrics=metrics,
            trace=trace,
            checkpoint=parser_checkpoint,
            stage="F",
        )
        t += 1

    for qt in turns:
        metrics["_expected"] = world.expected_value(qt.slot)
        metrics["answer"] = ""
        state, _ = run_nl_turn(
            qt.query,
            t=t,
            state=state,
            kernel=kernel,
            encoder=encoder,
            parser=parser,
            renderer=renderer,
            metrics=metrics,
            trace=trace,
            checkpoint=parser_checkpoint,
            stage="F",
        )
        metrics["answers"][qt.slot] = metrics.get("answer", "")
        expected = world.expected_value(qt.slot)
        got = state.working.last_read.get(qt.slot)
        if got != expected:
            return state, trace, metrics, f"query miss on {qt.slot}: got {got!r} expected {expected!r}"
        t += 1

    metrics.pop("_expected", None)
    metrics.pop("_ledger", None)
    metrics["tokens"] = ledger.to_dict()
    metrics["cold_hits"] = state.cold_hits
    metrics["overflow_evictions"] = state.overflow_evictions
    metrics["cold_keys"] = sorted(state.cold_index.keys()) if state.cold_index else []
    metrics["hot_keys"] = sorted(k for k in state.storage.slots if k.startswith("fact."))
    if state.overflow_evictions < 1:
        return state, trace, metrics, "expected overflow evictions >= 1"
    if state.cold_hits < 1:
        return state, trace, metrics, "expected cold_hits >= 1"
    return state, trace, metrics, ""
