"""E7 learned NL front — utterance → structured event fields."""

from __future__ import annotations

from pathlib import Path

import torch

from greenfield.kernel import Kernel
from greenfield.parser.template_parser import ParsedUtterance, parse_utterance
from greenfield.runner import load_policy
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.parser.value_span import (
    detect_plant_slot,
    extract_plant_value,
    normalize_for_match,
    parse_extended_utterance,
    parse_relaxed_query,
    parse_template_utterance,
    parse_unsupported_question,
)
from greenfield.train.features import FEATURE_DIM, E7_ID_TO_INTENT, E7_ID_TO_SLOT, featurize_utterance
from greenfield.train.model import EventEncoderModel
from greenfield.train.nl_parser_model import NlParserModel, load_nl_parser
from greenfield.train.value_codec import decode_value, encode_utterance_window
from greenfield.types import Intent, MachineState


class LearnedEventParser:
    """Map raw user text to kernel intents (legacy E7/E9a or E10.1 NlParserModel)."""

    def __init__(
        self,
        *,
        legacy_model: EventEncoderModel | None = None,
        nl_model: NlParserModel | None = None,
        device: torch.device | None = None,
        stage: str = "B",
    ):
        if legacy_model is None and nl_model is None:
            raise ValueError("need legacy_model or nl_model")
        self.legacy_model = legacy_model
        self.nl_model = nl_model
        self.device = device or torch.device("cpu")
        self.stage = stage
        if legacy_model is not None:
            self.legacy_model.to(self.device)
            self.legacy_model.eval()
        if nl_model is not None:
            self.nl_model.to(self.device)
            self.nl_model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | None = None,
        stage: str = "B",
    ) -> LearnedEventParser:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        if ckpt.get("parser") == "nl_v2" or ckpt.get("experiment") in ("e10a_open", "e11a_open_50m"):
            nl_model = load_nl_parser(path, device)
            return cls(nl_model=nl_model, device=device, stage=stage)
        legacy = load_encoder_model(
            path,
            device,
            predict_slot=True,
            predict_value=True,
            predict_event=True,
        )
        return cls(legacy_model=legacy, device=device, stage=stage)

    def _tensor(self, fv) -> torch.Tensor:
        lst = list(fv.as_list())
        while len(lst) < FEATURE_DIM:
            lst.append(0.0)
        return torch.tensor(lst, dtype=torch.float32, device=self.device)

    def _predict_legacy(self, text: str, state: MachineState, *, stage: str) -> tuple[int, int, list[int]]:
        fv = featurize_utterance(text, state, stage=stage)
        x = self._tensor(fv)
        return self.legacy_model.predict_event_fields(x)

    def _predict_v2(self, text: str) -> tuple[int, int]:
        utt = torch.tensor(encode_utterance_window(text), dtype=torch.float32, device=self.device)
        return self.nl_model.predict_fields(utt)

    def parse(self, text: str, state: MachineState, *, stage: str | None = None) -> ParsedUtterance | None:
        t = text.strip()
        if not t:
            return None

        norm = normalize_for_match(t)

        extended = parse_extended_utterance(norm)
        if extended is not None:
            return extended

        template = parse_template_utterance(norm)
        if template is not None:
            return template

        st = stage or self.stage
        value_ids: list[int] = []
        if self.nl_model is not None:
            intent_id, slot_id = self._predict_v2(t)
        else:
            intent_id, slot_id, value_ids = self._predict_legacy(t, state, stage=st)

        intent = E7_ID_TO_INTENT.get(intent_id, Intent.CHITCHAT)
        slot = E7_ID_TO_SLOT.get(slot_id, "__none__")

        if intent == Intent.CHITCHAT:
            return ParsedUtterance(Intent.CHITCHAT, {}, "chitchat")

        if intent not in (Intent.PLANT, Intent.QUERY):
            return None

        if intent == Intent.QUERY:
            relaxed = parse_relaxed_query(norm)
            if relaxed is not None:
                return relaxed
            unsupported = parse_unsupported_question(norm)
            if unsupported is not None:
                return unsupported
            slot_key = slot if slot != "__none__" else "fact.name"
            if slot_key not in ("fact.name", "fact.code", "fact.item0") and not str(slot_key).startswith(
                "fact.item"
            ):
                return ParsedUtterance(
                    Intent.CHITCHAT,
                    {"reason": "unsupported_query"},
                    "unsupported_query",
                )
            return ParsedUtterance(intent, {"slot": slot_key}, "learned_v2" if self.nl_model else "learned")

        # PLANT — require an explicit template prefix (never guess from last word).
        detected = detect_plant_slot(norm)
        if not detected:
            return ParsedUtterance(Intent.CHITCHAT, {}, "plant_guard")
        slot_key = detected
        span_val = extract_plant_value(norm, slot_key)
        if not span_val:
            return ParsedUtterance(Intent.CHITCHAT, {}, "plant_guard")
        payload: dict = {"slot": slot_key, "value": span_val}
        return ParsedUtterance(intent, payload, "learned_v2" if self.nl_model else "learned")


def deploy_policy_path(name: str = "policy.v0.json") -> Path:
    """Resolve policy JSON next to this package (works regardless of process cwd)."""
    return Path(__file__).resolve().parent / "deploy" / name


def default_parser_state() -> MachineState:
    kernel = Kernel(load_policy(deploy_policy_path()))
    return kernel.genesis()


def parse_nl(
    text: str,
    state: MachineState | None = None,
    *,
    checkpoint: str | Path = "greenfield/checkpoints/encoder_e7_best.pt",
    stage: str = "B",
) -> ParsedUtterance | None:
    """Prefer learned parser when checkpoint exists; else regex template parser."""
    path = Path(checkpoint) if checkpoint is not None else None
    if path is not None and path.is_file():
        parser = LearnedEventParser.from_checkpoint(path, stage=stage)
        st = state or default_parser_state()
        return parser.parse(text, st, stage=stage)
    return parse_utterance(text)
