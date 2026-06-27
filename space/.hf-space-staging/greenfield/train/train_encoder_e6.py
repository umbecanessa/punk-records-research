"""E6 training: percept-copy value head + phased fine-tune."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from greenfield.episodes import CurriculumStage
from greenfield.eval_util import eval_encoder_curriculum
from greenfield.log_util import configure_unbuffered, log
from greenfield.runner import load_policy
from greenfield.train.checkpoint_util import current_vocab, load_encoder_model
from greenfield.train.dataset_operands import OperandDataset, VALUE_OPS
from greenfield.train.features import OP_TO_ID
from greenfield.train.train_encoder import resolve_device
from greenfield.train.value_codec import decode_value
from greenfield.types import OpCode

VALUE_OP_IDS = {OP_TO_ID[op] for op in VALUE_OPS}


def _op_mask(y: torch.Tensor, op_ids: set[int], device) -> torch.Tensor:
    """Boolean mask — never use int(0/1) tensors as indices (PyTorch treats them as row ids)."""
    return torch.tensor([t.item() in op_ids for t in y], device=device, dtype=torch.bool)


def set_requires_grad(model, *, value_only: bool) -> None:
    for name, param in model.named_parameters():
        if value_only:
            param.requires_grad = name.startswith("value_")
        else:
            param.requires_grad = True


def value_loss(value_logits, value_targets) -> torch.Tensor:
    """CE over char positions; up-weight non-PAD target chars."""
    b, chars, vocab = value_logits.shape
    logits = value_logits.reshape(b * chars, vocab)
    targets = value_targets.reshape(b * chars)
    weights = torch.where(targets > 0, 2.0, 0.5)
    return (F.cross_entropy(logits, targets, reduction="none") * weights).mean()


@torch.no_grad()
def value_exact_match(model, loader, device) -> float:
    model.eval()
    exact = total = 0
    for x, y, _slot_y, value_y in loader:
        x = x.to(device)
        y = y.to(device)
        value_y = value_y.to(device)
        _, _, vlog = model(x, return_slot=True, return_value=True)
        mask = _op_mask(y, VALUE_OP_IDS, device)
        if not mask.any():
            continue
        pred = vlog[mask].argmax(dim=-1)
        targets = value_y[mask]
        for i in range(pred.size(0)):
            if decode_value(pred[i].tolist()) == decode_value(targets[i].tolist()):
                exact += 1
            total += 1
    return exact / max(1, total)


def train_epoch_operands(
    model,
    loader,
    optim,
    device,
    *,
    slot_weight: float,
    value_weight: float,
    op_weight: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y, slot_y, value_y in loader:
        x = x.to(device)
        y = y.to(device)
        slot_y = slot_y.to(device)
        value_y = value_y.to(device)

        op_logits, slot_logits, value_logits = model(x, return_slot=True, return_value=True)
        loss = op_weight * F.cross_entropy(op_logits, y)

        put_get = {OP_TO_ID[OpCode.PUT], OP_TO_ID[OpCode.GET]}
        slot_mask = _op_mask(y, put_get, device)
        if slot_logits is not None and slot_mask.any() and slot_weight > 0:
            loss = loss + slot_weight * F.cross_entropy(slot_logits[slot_mask], slot_y[slot_mask])

        value_mask = _op_mask(y, VALUE_OP_IDS, device)
        if value_logits is not None and value_mask.any() and value_weight > 0:
            loss = loss + value_weight * value_loss(value_logits[value_mask], value_y[value_mask])

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        correct += (op_logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E6 — learned values (v2)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default="greenfield/checkpoints/encoder_e5b_best.pt")
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--val-size", type=int, default=6_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--value-only-epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--joint-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", default="A,B,C,D,E,F")
    parser.add_argument("--slot-weight", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=4.0)
    parser.add_argument("--op-weight", type=float, default=0.1)
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

    model = load_encoder_model(
        args.warm_start,
        device,
        predict_slot=True,
        predict_value=True,
    )
    set_requires_grad(model, value_only=True)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    best_path = ckpt_dir / "encoder_e6_best.pt"
    best_reward = float("-inf")

    log("phase 1: train value head only (backbone/op/slot frozen)")

    for epoch in range(1, args.epochs + 1):
        if epoch == args.value_only_epochs + 1:
            set_requires_grad(model, value_only=False)
            optim = torch.optim.AdamW(model.parameters(), lr=args.joint_lr)
            log("phase 2: joint fine-tune (all params)")

        value_only = epoch <= args.value_only_epochs
        set_requires_grad(model, value_only=value_only)
        op_w = 0.0 if value_only else args.op_weight
        slot_w = 0.0 if value_only else args.slot_weight

        train_ds = OperandDataset(
            size=args.train_size,
            seed=args.seed + epoch,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
        )
        val_ds = OperandDataset(
            size=args.val_size,
            seed=args.seed + 8888,
            stages=stages,
            policy=policy,
            overflow_policy=overflow_policy,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        tr_loss, tr_acc = train_epoch_operands(
            model,
            train_loader,
            optim,
            device,
            slot_weight=slot_w,
            value_weight=args.value_weight,
            op_weight=op_w,
        )
        val_exact = value_exact_match(model, val_loader, device)

        ep_eval = eval_encoder_curriculum(
            model,
            policy,
            stages,
            device=device,
            episodes=args.eval_episodes,
            seed=args.seed + epoch * 11,
            overflow_policy=overflow_policy,
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
                    "stages": [s.value for s in stages],
                    "mean_reward": reward,
                    "experiment": "e6_v2",
                    "epoch": epoch,
                },
                best_path,
            )
            mark = " *best*"

        phase = "V" if value_only else "J"
        log(
            f"epoch {epoch:02d} [{phase}]  train_acc={tr_acc:.3f}  val_value_exact={val_exact:.3f}  "
            f"reward={reward:.3f}{mark}"
        )

    final = eval_encoder_curriculum(
        model,
        policy,
        stages,
        device=device,
        episodes=max(args.eval_episodes, 30),
        seed=args.seed + 7000,
        overflow_policy=overflow_policy,
        use_learned_args=True,
        use_learned_values=True,
    )
    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e6_v2",
        "checkpoint": str(best_path),
        "best_mean_reward": best_reward,
        "final_eval": final,
    }
    (ckpt_dir / "encoder_e6_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"checkpoint: {best_path}  best_reward: {best_reward:.3f}")


if __name__ == "__main__":
    main()
