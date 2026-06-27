"""E6 fine-tune on stage G — quest world opcode + values."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from greenfield.episodes import CurriculumStage
from greenfield.eval_util import eval_encoder_curriculum
from greenfield.log_util import configure_unbuffered, log
from greenfield.runner import load_policy
from greenfield.train.checkpoint_util import current_vocab, load_encoder_model
from greenfield.train.dataset_operands import OperandDataset
from greenfield.train.train_encoder import resolve_device
from greenfield.train.train_encoder_e6 import (
    set_requires_grad,
    train_epoch_operands,
    value_exact_match,
)

STAGES = [CurriculumStage.G]


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E6G — stage G fine-tune")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default="greenfield/checkpoints/encoder_e6_best.pt")
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--val-size", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-weight", type=float, default=0.3)
    parser.add_argument("--value-weight", type=float, default=3.0)
    parser.add_argument("--op-weight", type=float, default=0.2)
    parser.add_argument("--policy", default="greenfield/deploy/policy.v0.json")
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=15)
    args = parser.parse_args()

    device = resolve_device(args.device)
    log(f"device: {device}")

    policy = load_policy(args.policy)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = load_encoder_model(
        args.warm_start,
        device,
        predict_slot=True,
        predict_value=True,
        expand_vocab=True,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_path = ckpt_dir / "encoder_e6_best.pt"
    best_reward = float("-inf")

    warm_ckpt = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    base_stages = [CurriculumStage(s) for s in warm_ckpt.get("stages", ["A", "B", "C", "D", "E", "F"])]
    all_stages = list(dict.fromkeys([*base_stages, CurriculumStage.G]))

    for epoch in range(1, args.epochs + 1):
        train_ds = OperandDataset(
            size=args.train_size,
            seed=args.seed + epoch,
            stages=STAGES,
            policy=policy,
        )
        val_ds = OperandDataset(
            size=args.val_size,
            seed=args.seed + 7777,
            stages=STAGES,
            policy=policy,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        tr_loss, tr_acc = train_epoch_operands(
            model,
            train_loader,
            optim,
            device,
            slot_weight=args.slot_weight,
            value_weight=args.value_weight,
            op_weight=args.op_weight,
        )
        val_exact = value_exact_match(model, val_loader, device)
        ep_eval = eval_encoder_curriculum(
            model,
            policy,
            STAGES,
            device=device,
            episodes=args.eval_episodes,
            seed=args.seed + epoch * 13,
            use_learned_args=True,
            use_learned_values=True,
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
                    "predict_value": True,
                    "use_learned_args": True,
                    "use_learned_values": True,
                    "vocab": current_vocab(),
                    "stages": [s.value for s in all_stages],
                    "mean_reward": reward,
                    "experiment": "e6g_stage_g",
                    "epoch": epoch,
                },
                best_path,
            )
            mark = " *best*"

        log(
            f"epoch {epoch:02d}  train_acc={tr_acc:.3f}  val_value_exact={val_exact:.3f}  "
            f"G_reward={reward:.3f}{mark}"
        )

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e6g_stage_g",
        "checkpoint": str(best_path),
        "best_mean_reward": best_reward,
        "stages": [s.value for s in all_stages],
    }
    (ckpt_dir / "encoder_e6g_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"checkpoint: {best_path}  best G reward: {best_reward:.3f}")


if __name__ == "__main__":
    main()
