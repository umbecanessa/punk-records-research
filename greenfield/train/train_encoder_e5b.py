"""E5b training: fine-tune slot head; inference uses learned PUT/GET keys."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from greenfield.episodes import CurriculumStage
from greenfield.eval_util import eval_encoder_curriculum
from greenfield.log_util import configure_unbuffered, log
from greenfield.runner import load_policy
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.train.dataset import OpcodeDataset
from greenfield.train.dataset_recovery import RecoveryDataset
from greenfield.train.train_encoder import eval_epoch, resolve_device
from greenfield.train.train_encoder_e2 import train_epoch_multitask


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E5b — learned structured args")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default="greenfield/checkpoints/encoder_e2_best.pt")
    parser.add_argument("--train-size", type=int, default=30_000)
    parser.add_argument("--recovery-size", type=int, default=15_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", default="A,B,C,D,E,F")
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--policy", default="greenfield/deploy/policy.v0.json")
    parser.add_argument("--overflow-policy", default="greenfield/deploy/policy.overflow.json")
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    device = resolve_device(args.device)
    log(f"device: {device}")

    stages = [CurriculumStage(s.strip().upper()) for s in args.stages.split(",") if s.strip()]
    policy = load_policy(args.policy)
    overflow_policy = load_policy(args.overflow_policy)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = load_encoder_model(args.warm_start, device, predict_slot=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_path = ckpt_dir / "encoder_e5b_best.pt"
    best_reward = float("-inf")

    for epoch in range(1, args.epochs + 1):
        clean = OpcodeDataset(
            size=args.train_size,
            seed=args.seed + epoch,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
        )
        recovery = RecoveryDataset(
            size=args.recovery_size,
            seed=args.seed + epoch + 50,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
            noise_rate=0.15,
        )
        val = OpcodeDataset(
            size=args.val_size,
            seed=args.seed + 7777,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
        )
        train_loader = DataLoader(ConcatDataset([clean, recovery]), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)

        tr_loss, tr_acc = train_epoch_multitask(
            model, train_loader, optim, device, slot_weight=args.slot_weight
        )
        va_loss, va_acc = eval_epoch(model, val_loader, device)

        ep_eval = eval_encoder_curriculum(
            model,
            policy,
            stages,
            device=device,
            episodes=args.eval_episodes,
            seed=args.seed + epoch * 13,
            overflow_policy=overflow_policy,
            use_learned_args=True,
            lambda_revert=0.5,
        )
        reward = ep_eval["mean_reward"]
        mark = ""
        if reward > best_reward:
            best_reward = reward
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": 128,
                    "predict_slot": True,
                    "use_learned_args": True,
                    "vocab": {
                        "num_intents": model.intent_emb.num_embeddings,
                        "num_sources": model.source_emb.num_embeddings,
                        "num_slots": model.slot_emb.num_embeddings,
                        "num_stages": model.stage_emb.num_embeddings,
                        "num_opcodes": model.op_head.out_features,
                        "max_step": model.step_emb.num_embeddings,
                    },
                    "stages": [s.value for s in stages],
                    "mean_reward": reward,
                    "experiment": "e5b",
                    "epoch": epoch,
                },
                best_path,
            )
            mark = " *best*"

        log(
            f"epoch {epoch:02d}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  "
            f"learned_reward={reward:.3f}{mark}"
        )

    final = eval_encoder_curriculum(
        model,
        policy,
        stages,
        device=device,
        episodes=max(args.eval_episodes, 25),
        seed=args.seed + 9000,
        overflow_policy=overflow_policy,
        use_learned_args=True,
    )
    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e5b",
        "checkpoint": str(best_path),
        "best_mean_reward": best_reward,
        "final_eval": final,
    }
    report_path = ckpt_dir / "encoder_e5b_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
