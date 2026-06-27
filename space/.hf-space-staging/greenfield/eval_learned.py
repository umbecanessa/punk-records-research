"""Eval learned encoder — supports E1/E2 checkpoints and stress noise."""

from __future__ import annotations

import argparse
import random

import torch

from greenfield.deploy_config import DEFAULT_ENCODER, DEFAULT_STAGES, USE_LEARNED_ARGS, USE_LEARNED_VALUES
from greenfield.episodes import CurriculumStage
from greenfield.eval_util import eval_encoder_curriculum
from greenfield.learned_encoder import LearnedEncoder
from greenfield.runner import load_policy


class NoisyLearnedWrapper:
    """Insert random junk ops like NoisyEncoder — stress test learned policy."""

    def __init__(self, inner: LearnedEncoder, *, noise_rate: float, rng: random.Random | None = None):
        self.inner = inner
        self.noise_rate = noise_rate
        self.rng = rng or random.Random()
        self.oracle = inner.oracle

    def propose(self, event, state, kernel):
        steps = self.inner.propose(event, state, kernel)
        if not steps or self.rng.random() >= self.noise_rate:
            return steps
        from greenfield.types import OpCode, OpProposal

        junk = OpProposal(
            op=self.rng.choice([OpCode.PUT, OpCode.RUN, OpCode.STEP]),
            args={
                "key": "fact.name",
                "value": "HALLUCINATED",
                "evidence_ref": "deadbeef",
                "handle": "missing",
                "delta": 99,
            },
        )
        insert_at = self.rng.randint(1, max(1, len(steps)))
        out = list(steps)
        out.insert(insert_at, junk)
        return out

    def resolve_evidence(self, state, kernel, proposal):
        return self.inner.resolve_evidence(state, kernel, proposal)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval greenfield learned encoder")
    parser.add_argument("--checkpoint", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0, help="stress: inject junk op rate")
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument(
        "--oracle-args",
        action="store_true",
        help="oracle-materialized keys (E2 ablation)",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_learned = USE_LEARNED_ARGS and not args.oracle_args
    use_values = USE_LEARNED_VALUES and not args.oracle_args
    enc = LearnedEncoder.from_checkpoint(
        args.checkpoint,
        device=device,
        use_learned_args=use_learned,
        use_learned_values=use_values,
    )
    model = enc.model

    policy = load_policy("greenfield/deploy/policy.v0.json")
    overflow = load_policy("greenfield/deploy/policy.overflow.json")
    stages = [CurriculumStage(s.strip().upper()) for s in args.stages.split(",") if s.strip()]

    print(f"checkpoint: {args.checkpoint}  device: {device}  noise: {args.noise}  learned_args: {use_learned}  learned_values: {use_values}")

    if args.noise > 0:
        rng = random.Random(args.seed + 99)
        # noise path: per-episode wrapper (legacy)
        from greenfield.simulator import sample_world
        from greenfield.episodes import generate_script
        from greenfield.runner import run_episode, summarize

        total_reward = 0.0
        for stage in stages:
            metrics_list = []
            for i in range(args.episodes):
                ep_seed = args.seed + i + ord(stage.value[0]) * 1000
                world = sample_world(random.Random(ep_seed), num_facts=1 + (i % 2))
                script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 1))
                inner = LearnedEncoder(
                    model,
                    device=device,
                    stage=stage.value,
                    use_learned_args=use_learned,
                    use_learned_values=use_values,
                )
                enc_noisy: LearnedEncoder | NoisyLearnedWrapper = NoisyLearnedWrapper(
                    inner, noise_rate=args.noise, rng=rng
                )
                _, m = run_episode(
                    world=world,
                    script=script,
                    policy=policy,
                    encoder=enc_noisy,
                    stage=stage.value,
                )
                metrics_list.append(m)
            s = summarize(metrics_list)[stage.value]
            reward = s["mean_query_accuracy"] - 0.5 * s["mean_revert_rate"]
            total_reward += reward
            print(
                f"stage {stage.value}: query_acc={s['mean_query_accuracy']:.3f} "
                f"revert={s['mean_revert_rate']:.3f}  reward={reward:.3f}"
            )
        print(f"mean_reward: {total_reward / max(1, len(stages)):.3f}")
        return

    result = eval_encoder_curriculum(
        model,
        policy,
        stages,
        device=device,
        episodes=args.episodes,
        seed=args.seed,
        overflow_policy=overflow,
        use_learned_args=use_learned,
        use_learned_values=use_values,
    )
    for stage, s in result["stages"].items():
        print(
            f"stage {stage}: query_acc={s['mean_query_accuracy']:.3f} "
            f"revert={s['mean_revert_rate']:.3f}  reward={s['reward']:.3f}"
        )
    print(f"mean_reward: {result['mean_reward']:.3f}")


if __name__ == "__main__":
    main()
