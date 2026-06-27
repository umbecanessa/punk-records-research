"""Learned event encoder — opcodes + slot keys (E5b) + values (E6)."""

from __future__ import annotations

from pathlib import Path

import torch

from greenfield.encoder import OracleEncoder
from greenfield.kernel import Kernel
from greenfield.state_util import clone_machine_state
from greenfield.train.features import FEATURE_DIM, ID_TO_OP, ID_TO_SLOT, MAX_STEP, featurize
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.train.model import EventEncoderModel
from greenfield.train.value_codec import decode_value_from_features
from greenfield.types import EpisodeEvent, Intent, MachineState, OpCode, OpProposal


class LearnedEncoder:
    def __init__(
        self,
        model: EventEncoderModel,
        *,
        oracle: OracleEncoder | None = None,
        device: torch.device | None = None,
        stage: str = "B",
        use_learned_args: bool = False,
        use_learned_values: bool = False,
    ):
        self.model = model
        self.oracle = oracle or OracleEncoder()
        self.device = device or torch.device("cpu")
        self.stage = stage
        self.use_learned_args = use_learned_args and model.slot_head is not None
        self.use_learned_values = use_learned_values and model.value_char_mlps is not None
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | None = None,
        stage: str = "B",
        use_learned_args: bool = True,
        use_learned_values: bool | None = None,
    ) -> LearnedEncoder:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        model = load_encoder_model(
            path,
            device,
            predict_slot=True,
            predict_value=bool(ckpt.get("predict_value", False)),
        )
        stages = ckpt.get("stages") or []
        learned_args = bool(ckpt.get("use_learned_args", use_learned_args))
        learned_vals = (
            bool(ckpt.get("use_learned_values", True))
            if use_learned_values is None
            else use_learned_values
        )
        return cls(
            model,
            device=device,
            stage=stage,
            use_learned_args=learned_args,
            use_learned_values=learned_vals,
        )

    def _prepare_event(self, event: EpisodeEvent, state: MachineState) -> None:
        if event.intent == Intent.TOOL_PLANT:
            plan = event.payload.get("plan", ["run"])
            state.storage.plan.steps = list(plan)
            state.storage.plan.ptr = 0

    def _terminal(self, event: EpisodeEvent, op: OpCode) -> bool:
        if op == OpCode.RENDER:
            return True
        if op == OpCode.SEAL and event.requires_seal and event.intent in (
            Intent.PLANT,
            Intent.CHITCHAT,
            Intent.TOOL_PLANT,
            Intent.DISTRACTOR_PUT,
        ):
            return True
        return False

    def _slot_key(self, event: EpisodeEvent, slot_id: int | None) -> str:
        explicit = event.slot_key()
        if explicit and explicit.startswith("user."):
            return explicit
        if slot_id is not None and self.use_learned_args:
            key = ID_TO_SLOT.get(slot_id, "__none__")
            if key != "__none__":
                return key
        return explicit or "fact.name"

    def _value_str(self, event: EpisodeEvent, x: torch.Tensor | None) -> str:
        key = event.slot_key() or ""
        if key.startswith("user.") and event.slot_value() is not None:
            return str(event.slot_value())
        if self.use_learned_values and x is not None:
            # Values come from OBS percept encoded in features — not event.slot_value().
            return decode_value_from_features(x)
        return str(event.slot_value() or "")

    def _materialize(
        self,
        event: EpisodeEvent,
        op: OpCode,
        slot_id: int | None,
        x: torch.Tensor | None,
    ) -> OpProposal:
        if not self.use_learned_args or slot_id is None:
            if self.use_learned_values and op == OpCode.PUT and x is not None:
                key = event.slot_key() or "fact.name"
                return OpProposal(
                    op=OpCode.PUT,
                    args={
                        "key": key,
                        "value": self._value_str(event, x),
                        "evidence_ref": "__LAST_OBS__",
                    },
                )
            return self.oracle.materialize(event, op)

        key = self._slot_key(event, slot_id)
        if op == OpCode.PUT:
            return OpProposal(
                op=OpCode.PUT,
                args={
                    "key": key,
                    "value": self._value_str(event, x),
                    "evidence_ref": "__LAST_OBS__",
                },
            )
        if op == OpCode.GET:
            return OpProposal(op=OpCode.GET, args={"key": key})
        if op == OpCode.FOCUS:
            return OpProposal(op=OpCode.FOCUS, args={"keys": [key] if key else []})
        if op == OpCode.RENDER:
            return OpProposal(op=OpCode.RENDER, args={"mode": "answer", "keys": [key], "max_tokens": 32})
        if op == OpCode.RUN:
            handle = str(event.payload.get("handle", "plant_fact"))
            return OpProposal(
                op=OpCode.RUN,
                args={"handle": handle, "args": {"key": key, "value": self._value_str(event, x)}},
            )
        return self.oracle.materialize(event, op)

    def _tensor_features(self, fv) -> torch.Tensor:
        lst = list(fv.as_list())
        lst[0] = min(int(lst[0]), self.model.intent_emb.num_embeddings - 1)
        lst[1] = min(int(lst[1]), self.model.source_emb.num_embeddings - 1)
        lst[2] = min(int(lst[2]), self.model.slot_emb.num_embeddings - 1)
        lst[3] = min(int(lst[3]), self.model.stage_emb.num_embeddings - 1)
        lst[4] = min(int(lst[4]), self.model.step_emb.num_embeddings - 1)
        lst[5] = min(int(lst[5]), self.model.prev_op_emb.num_embeddings - 1)
        while len(lst) < FEATURE_DIM:
            lst.append(0.0)
        return torch.tensor(lst, dtype=torch.float32, device=self.device)

    def propose(self, event: EpisodeEvent, state: MachineState, kernel: Kernel) -> list[OpProposal]:
        return self._propose_loop(event, state, kernel, obs_already=False)

    def propose_after_obs(self, event: EpisodeEvent, state: MachineState, kernel: Kernel) -> list[OpProposal]:
        """E7b: OBS already applied — opcode trace starts at PUT/GET (prev_op=OBS)."""
        return self._propose_loop(event, state, kernel, obs_already=True)

    def _propose_loop(
        self,
        event: EpisodeEvent,
        state: MachineState,
        kernel: Kernel,
        *,
        obs_already: bool,
    ) -> list[OpProposal]:
        self._prepare_event(event, state)
        proposals: list[OpProposal] = []
        prev: OpCode | None = OpCode.OBS if obs_already else None
        sim = clone_machine_state(state)
        start_step = 1 if obs_already else 0

        for step_idx in range(start_step, MAX_STEP):
            fv = featurize(event, sim, stage=self.stage, step_idx=step_idx, prev_op=prev)
            x = self._tensor_features(fv)
            slot_id: int | None = None
            with torch.no_grad():
                if self.use_learned_args or self.use_learned_values:
                    op_logits, slot_logits, _ = self.model(x.unsqueeze(0), return_slot=True, return_value=True)
                    op_id = int(op_logits.argmax(dim=-1).item())
                    if slot_logits is not None:
                        slot_id = int(slot_logits.argmax(dim=-1).item())
                else:
                    op_id = self.model.predict_op_id(x)
            op = ID_TO_OP[op_id]
            proposal = self._materialize(event, op, slot_id, x)
            proposals.append(proposal)

            resolved = self.oracle.resolve_evidence(sim, kernel, proposal)
            try:
                if resolved.op != OpCode.RENDER:
                    sim = kernel.apply(sim, resolved)
            except Exception:
                break

            prev = op
            if op == OpCode.GET and event.intent == Intent.QUERY:
                render_slot = slot_id if self.use_learned_args else None
                proposals.append(self._materialize(event, OpCode.RENDER, render_slot, x))
                break
            if self._terminal(event, op):
                break

        return proposals

    def resolve_evidence(self, state: MachineState, kernel: Kernel, proposal: OpProposal) -> OpProposal:
        return self.oracle.resolve_evidence(state, kernel, proposal)
