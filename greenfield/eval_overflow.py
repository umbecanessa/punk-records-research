"""E4 eval: oracle + learned encoder on overflow curriculum (stage F)."""

from __future__ import annotations

import argparse
import random

import torch

from greenfield.deploy_config import DEFAULT_ENCODER
from greenfield.encoder import OracleEncoder
from greenfield.episodes import CurriculumStage, generate_script
from greenfield.learned_encoder import LearnedEncoder
from greenfield.log_util import configure_unbuffered, log
from greenfield.runner import load_policy, run_episode, summarize
from greenfield.simulator import overflow_world


def run_batch(
    *,
    policy_path: str,
    encoder,
    episodes: int,
    seed: int,
    num_facts: int,
) -> list:
    policy = load_policy(policy_path)
    results = []
    for i in range(episodes):
        ep_seed = seed + i
        world = overflow_world(random.Random(ep_seed), num_facts=num_facts)
        script = generate_script(world, stage=CurriculumStage.F, rng=random.Random(ep_seed + 1))
        _, metrics = run_episode(
            world=world,
            script=script,
            policy=policy,
            encoder=encoder,
            seed=ep_seed,
            stage="F",
        )
        results.append(metrics)
    return results


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Eval cold-store overflow (E4)")
    parser.add_argument("--policy", default="greenfield/deploy/policy.overflow.json")
    parser.add_argument("--encoder-checkpoint", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-facts", type=int, default=5)
    args = parser.parse_args()

    log(f"policy: {args.policy}")
    log(f"num_facts: {args.num_facts}  episodes: {args.episodes}")

    oracle_results = run_batch(
        policy_path=args.policy,
        encoder=OracleEncoder(),
        episodes=args.episodes,
        seed=args.seed,
        num_facts=args.num_facts,
    )
    s = summarize(oracle_results)["F"]
    cold_hits = sum(m.cold_hits for m in oracle_results)
    evictions = sum(m.overflow_evictions for m in oracle_results)
    log(
        f"oracle stage F: query_acc={s['mean_query_accuracy']:.3f}  "
        f"revert={s['mean_revert_rate']:.3f}  cold_hits={cold_hits}  evictions={evictions}"
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = LearnedEncoder.from_checkpoint(args.encoder_checkpoint, device=device)
    learned_results = run_batch(
        policy_path=args.policy,
        encoder=enc,
        episodes=args.episodes,
        seed=args.seed + 10_000,
        num_facts=args.num_facts,
    )
    ls = summarize(learned_results)["F"]
    lcold = sum(m.cold_hits for m in learned_results)
    lev = sum(m.overflow_evictions for m in learned_results)
    log(
        f"learned stage F: query_acc={ls['mean_query_accuracy']:.3f}  "
        f"revert={ls['mean_revert_rate']:.3f}  cold_hits={lcold}  evictions={lev}"
    )


if __name__ == "__main__":
    main()
