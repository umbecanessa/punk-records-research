"""Recovery dataset: train after simulated wrong-op reverts."""

from __future__ import annotations

import random

from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.kernel import Kernel
from greenfield.simulator import bind_tools, default_tool_executor, overflow_world, sample_world
from greenfield.state_util import clone_machine_state
from greenfield.train.features import OP_TO_ID, featurize
from greenfield.types import Intent, KernelRevert, OpCode, Policy

RECOVERY_OPS = [OpCode.PUT, OpCode.RUN, OpCode.STEP, OpCode.SEAL, OpCode.GET]


class RecoveryDataset:
    """(features, opcode_id) pairs emphasizing post-revert recovery."""

    def __init__(
        self,
        *,
        size: int,
        seed: int,
        stages: list[CurriculumStage],
        policy: Policy,
        overflow_policy: Policy | None = None,
        noise_rate: float = 0.3,
    ):
        self.size = size
        self.seed = seed
        self.stages = stages
        self.policy = policy
        self.overflow_policy = overflow_policy or policy
        self.noise_rate = noise_rate
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
            ep_seed = self.seed + i * 31
            if stage == CurriculumStage.F:
                world = overflow_world(random.Random(ep_seed), num_facts=5)
            else:
                world = sample_world(random.Random(ep_seed), num_facts=1 + (i % 2))
            script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 7))
            kernel = self._kernel_for(stage)
            state = kernel.genesis()
            bind_tools(state.storage, world)

            for event in script:
                if event.intent == Intent.TOOL_PLANT:
                    plan = event.payload.get("plan", ["run"])
                    state.storage.plan.steps = list(plan)
                    state.storage.plan.ptr = 0

                steps = self.oracle.propose(event, state, kernel)
                sim = clone_machine_state(state)
                prev: OpCode | None = None

                for step_idx, gold in enumerate(steps):
                    if rng.random() < self.noise_rate:
                        wrong_candidates = [o for o in RECOVERY_OPS if o != gold.op]
                        wrong_op = rng.choice(wrong_candidates)
                        wrong = self.oracle.materialize(event, wrong_op)
                        sim_branch = clone_machine_state(sim)
                        if wrong_op == OpCode.PUT and not kernel.last_obs_hash(sim_branch):
                            obs_only = self.oracle.materialize(event, OpCode.OBS)
                            sim_branch = kernel.apply(sim_branch, obs_only)
                        resolved_wrong = self.oracle.resolve_evidence(sim_branch, kernel, wrong)
                        try:
                            if resolved_wrong.op != OpCode.RENDER:
                                kernel.apply(sim_branch, resolved_wrong)
                        except KernelRevert:
                            fv = featurize(
                                event,
                                sim,
                                stage=stage.value,
                                step_idx=step_idx,
                                prev_op=wrong_op,
                            )
                            rows.append((fv.as_list(), OP_TO_ID[gold.op]))
                            if len(rows) >= self.size:
                                break

                    fv = featurize(event, sim, stage=stage.value, step_idx=step_idx, prev_op=prev)
                    rows.append((fv.as_list(), OP_TO_ID[gold.op]))
                    if len(rows) >= self.size:
                        break

                    resolved = self.oracle.resolve_evidence(sim, kernel, gold)
                    try:
                        if resolved.op != OpCode.RENDER:
                            sim = kernel.apply(sim, resolved)
                    except KernelRevert:
                        break
                    prev = gold.op

                i += 1
                if len(rows) >= self.size:
                    break
        return rows

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        import torch

        if self._cache is None:
            self._cache = self._build()
        feats, op_id = self._cache[idx]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(op_id, dtype=torch.long)
