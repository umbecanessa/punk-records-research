"""Synthetic opcode dataset from oracle trajectories."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.simulator import bind_tools, default_tool_executor, overflow_world, sample_world
from greenfield.train.features import OP_TO_ID, featurize
from greenfield.types import OpCode, Policy


class OpcodeDataset(Dataset):
    """Lazy-generated (features, opcode_id) pairs — no transcripts."""

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
        self._cache: list[tuple[list[float], int]] | None = None

    def _kernel_for(self, stage: CurriculumStage) -> Kernel:
        pol = self.overflow_policy if stage == CurriculumStage.F else self.policy
        return Kernel(pol, tool_executor=default_tool_executor)

    def _build(self) -> list[tuple[list[float], int]]:
        rng = random.Random(self.seed)
        rows: list[tuple[list[float], int]] = []
        stage_cycle = list(self.stages)
        i = 0
        while len(rows) < self.size:
            stage = stage_cycle[i % len(stage_cycle)]
            ep_seed = self.seed + i * 17
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
                    rows.append((fv.as_list(), OP_TO_ID[proposal.op]))
                    if len(rows) >= self.size:
                        break
                    resolved = self.oracle.resolve_evidence(state, kernel, proposal)
                    try:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is None:
            self._cache = self._build()
        feats, op_id = self._cache[idx]
        x = torch.tensor(feats, dtype=torch.float32)
        y = torch.tensor(op_id, dtype=torch.long)
        return x, y
