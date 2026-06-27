"""E2 training: noise curriculum + revert-aware checkpoint selection."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from greenfield.episodes import CurriculumStage
from greenfield.eval_util import eval_encoder_curriculum
from greenfield.runner import load_policy
from greenfield.train.checkpoint_util import current_vocab, load_encoder_model
from greenfield.train.dataset import OpcodeDataset
from greenfield.train.dataset_recovery import RecoveryDataset
from greenfield.train.features import OP_TO_ID
from greenfield.train.model import EventEncoderModel
from greenfield.train.train_encoder import eval_epoch, resolve_device
from greenfield.log_util import configure_unbuffered, log

from greenfield.types import OpCode, Policy

PUT_GET_OPS = {OP_TO_ID[OpCode.PUT], OP_TO_ID[OpCode.GET]}


def train_epoch_multitask(
    model,
    loader,
    optim,
    device,
    *,
    slot_weight: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if model.slot_head is not None:
            op_logits, slot_logits = model(x, return_slot=True)
            loss = F.cross_entropy(op_logits, y)
            slot_targets = x[:, 2].long()
            mask = torch.tensor([t.item() in PUT_GET_OPS for t in y], device=device, dtype=torch.bool)
            if mask.any():
                slot_loss = F.cross_entropy(slot_logits[mask], slot_targets[mask])
                loss = loss + slot_weight * slot_loss
        else:
            op_logits = model(x)
            loss = F.cross_entropy(op_logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        correct += (op_logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E2 — revert-aware encoder training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default="greenfield/checkpoints/encoder_e1_best.pt")
    parser.add_argument("--train-size", type=int, default=40_000)
    parser.add_argument("--recovery-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", default="A,B,C,D,E,F")
    parser.add_argument("--overflow-policy", default="greenfield/deploy/policy.overflow.json")
    parser.add_argument("--max-noise", type=float, default=0.35)
    parser.add_argument("--noise-ramp-epochs", type=int, default=15)
    parser.add_argument("--lambda-revert", type=float, default=0.5)
    parser.add_argument("--slot-weight", type=float, default=0.3)
    parser.add_argument("--recovery-weight", type=float, default=0.5)
    parser.add_argument("--policy", default=str(Path(__file__).resolve().parents[1] / "deploy" / "policy.v0.json"))
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=15)
    parser.add_argument("--eval-every", type=int, default=5, help="episode reward eval frequency (epochs)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        log(f"device: {device} ({name}, {vram:.1f} GiB)")
    else:
        log(f"device: {device}")

    stages = [CurriculumStage(s.strip().upper()) for s in args.stages.split(",") if s.strip()]
    policy = load_policy(args.policy)
    overflow_policy = load_policy(args.overflow_policy)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    warm = Path(args.warm_start)
    if warm.is_file():
        model = load_encoder_model(warm, device, predict_slot=True, expand_vocab=True)
        log(f"warm start: {warm} (expanded to current vocab incl. stage F)")
    else:
        model = EventEncoderModel(hidden=args.hidden, predict_slot=True, **current_vocab()).to(device)
        log("no warm start — training from scratch")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())

    best_reward = float("-inf")
    best_path = ckpt_dir / "encoder_e2_best.pt"

    log(
        f"params: {params:,}  clean: {args.train_size}  recovery: {args.recovery_size}  "
        f"lambda_revert: {args.lambda_revert}"
    )

    for epoch in range(1, args.epochs + 1):
        noise = min(args.max_noise, args.max_noise * epoch / max(1, args.noise_ramp_epochs))
        clean = OpcodeDataset(
            size=args.train_size,
            seed=args.seed + epoch,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
        )
        recovery = RecoveryDataset(
            size=args.recovery_size,
            seed=args.seed + epoch + 100,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
            noise_rate=noise,
        )
        combined = ConcatDataset([clean, recovery])
        val_clean = OpcodeDataset(
            size=args.val_size,
            seed=args.seed + 9999,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
        )
        train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_clean, batch_size=args.batch_size, shuffle=False, num_workers=0)

        tr_loss, tr_acc = train_epoch_multitask(
            model, train_loader, optim, device, slot_weight=args.slot_weight
        )
        va_loss, va_acc = eval_epoch(model, val_loader, device)

        reward = best_reward if best_reward > float("-inf") else 0.0
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ep_eval = eval_encoder_curriculum(
                model,
                policy,
                stages,
                device=device,
                episodes=args.eval_episodes,
                seed=args.seed + epoch * 17,
                overflow_policy=overflow_policy,
                lambda_revert=args.lambda_revert,
            )
            reward = ep_eval["mean_reward"]
        else:
            mark = ""
            log(
                f"epoch {epoch:02d}  noise={noise:.2f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  "
                f"(skipped episode eval)"
            )
            continue

        mark = ""
        if reward > best_reward:
            best_reward = reward
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": args.hidden,
                    "predict_slot": True,
                    "vocab": current_vocab(),
                    "stages": [s.value for s in stages],
                    "stage_default": "B",
                    "mean_reward": reward,
                    "lambda_revert": args.lambda_revert,
                    "epoch": epoch,
                    "experiment": "e2",
                },
                best_path,
            )
            mark = " *best*"

        log(
            f"epoch {epoch:02d}  noise={noise:.2f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  "
            f"reward={reward:.3f}{mark}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "predict_slot": True,
            "mean_reward": reward,
            "experiment": "e2",
            "epoch": args.epochs,
        },
        ckpt_dir / "encoder_e2_latest.pt",
    )

    log("\nFinal curriculum eval (best checkpoint):")
    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    final = eval_encoder_curriculum(
        model,
        policy,
        stages,
        device=device,
        episodes=max(args.eval_episodes, 25),
        seed=args.seed + 4242,
        overflow_policy=overflow_policy,
        lambda_revert=args.lambda_revert,
    )
    for st, stats in final["stages"].items():
        log(
            f"  stage {st}: query_acc={stats['mean_query_accuracy']:.3f}  "
            f"revert={stats['mean_revert_rate']:.3f}  reward={stats['reward']:.3f}"
        )

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "experiment": "e2",
        "params": params,
        "lambda_revert": args.lambda_revert,
        "best_mean_reward": best_reward,
        "checkpoint": str(best_path),
        "final_eval": final,
    }
    report_path = ckpt_dir / "encoder_e2_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nbest reward: {best_reward:.3f}  checkpoint: {best_path}")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
