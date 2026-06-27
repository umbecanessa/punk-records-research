"""Operand dataset: opcode + slot + value char targets from oracle trajectories."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.simulator import bind_tools, default_tool_executor, overflow_world, sample_world
from greenfield.train.features import OP_TO_ID, SLOTS, slot_key_id, featurize
from greenfield.train.value_codec import VALUE_CHARS, encode_value
from greenfield.types import OpCode, Policy

VALUE_OPS = {OpCode.PUT, OpCode.RUN}


class OperandDataset(Dataset):
    """(features, opcode_id, slot_id, value_char_ids) — labels from oracle, not chat."""

    def __init__(
        self,
        *,
        size: int,
        seed: int,
        stages: list[CurriculumStage],
        policy: Policy,
        overflow_policy: Policy | None = None,
    ):
        self.size = size
        self.seed = seed
        self.stages = stages
        self.policy = policy
        self.overflow_policy = overflow_policy or policy
        self.oracle = OracleEncoder()
        self._cache: list[tuple[list[float], int, int, list[int]]] | None = None

    def _kernel_for(self, stage: CurriculumStage) -> Kernel:
        pol = self.overflow_policy if stage == CurriculumStage.F else self.policy
        return Kernel(pol, tool_executor=default_tool_executor)

    @staticmethod
    def _value_from_proposal(proposal) -> str:
        if proposal.op == OpCode.PUT:
            return str(proposal.args.get("value", ""))
        if proposal.op == OpCode.RUN:
            run_args = proposal.args.get("args", {})
            return str(run_args.get("value", ""))
        return ""

    def _build(self) -> list[tuple[list[float], int, int, list[int]]]:
        rows: list[tuple[list[float], int, int, list[int]]] = []
        stage_cycle = list(self.stages)
        i = 0
        while len(rows) < self.size:
            stage = stage_cycle[i % len(stage_cycle)]
            ep_seed = self.seed + i * 19
            if stage == CurriculumStage.F:
                world = overflow_world(random.Random(ep_seed), num_facts=5)
            else:
                world = sample_world(random.Random(ep_seed), num_facts=1 + (i % 2))
            script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 1))
            kernel = self._kernel_for(stage)
            state = kernel.genesis()
            bind_tools(state.storage, world)

            for event in script:
                steps = self.oracle.propose(event, state, kernel)
                prev: OpCode | None = None
                for step_idx, proposal in enumerate(steps):
                    fv = featurize(
                        event,
                        state,
                        stage=stage.value,
                        step_idx=step_idx,
                        prev_op=prev,
                    )
                    slot_id = slot_key_id(event)
                    if proposal.op in (OpCode.PUT, OpCode.GET):
                        key = str(proposal.args.get("key", ""))
                        if key in {s for s in SLOTS if s != "__none__"}:
                            slot_id = SLOTS.index(key) if key in SLOTS else slot_id
                    value_ids = (
                        encode_value(self._value_from_proposal(proposal))
                        if proposal.op in VALUE_OPS
                        else [0] * VALUE_CHARS
                    )
                    rows.append((fv.as_list(), OP_TO_ID[proposal.op], slot_id, value_ids))
                    if len(rows) >= self.size:
                        break
                    resolved = self.oracle.resolve_evidence(state, kernel, proposal)
                    try:
                        if proposal.op != OpCode.RENDER:
                            state = kernel.apply(state, resolved)
                    except Exception:
                        break
                    prev = proposal.op
                i += 1
                if len(rows) >= self.size:
                    break
        return rows

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if self._cache is None:
            self._cache = self._build()
        feats, op_id, slot_id, value_ids = self._cache[idx]
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(op_id, dtype=torch.long),
            torch.tensor(slot_id, dtype=torch.long),
            torch.tensor(value_ids, dtype=torch.long),
        )
