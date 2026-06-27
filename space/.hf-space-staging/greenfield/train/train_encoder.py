"""E1 training: structured event → opcode classifier (CUDA-friendly)."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from greenfield.episodes import CurriculumStage, generate_script
from greenfield.learned_encoder import LearnedEncoder
from greenfield.runner import load_policy, run_episode, summarize
from greenfield.simulator import sample_world
from greenfield.log_util import configure_unbuffered, log
from greenfield.train.dataset import OpcodeDataset
from greenfield.train.model import EventEncoderModel
from greenfield.types import Policy


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optim, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def eval_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E1 — train event encoder")
    parser.add_argument("--device", default=None, help="cuda (default if available), cpu, cuda:0")
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stages",
        default="A,B,C,D,E",
        help="curriculum stages for synthetic data",
    )
    parser.add_argument(
        "--policy",
        default=str(Path(__file__).resolve().parents[1] / "deploy" / "policy.v0.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="greenfield/checkpoints",
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"device: {device} ({name}, {vram:.1f} GiB)")
    else:
        print(f"device: {device}")

    stages = [CurriculumStage(s.strip().upper()) for s in args.stages.split(",") if s.strip()]
    policy: Policy = load_policy(args.policy)

    torch.manual_seed(args.seed)
    full = OpcodeDataset(size=args.train_size + args.val_size, seed=args.seed, stages=stages, policy=policy)
    val_len = args.val_size
    train_len = len(full) - val_len
    train_ds, val_ds = random_split(
        full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EventEncoderModel(hidden=args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_path = ckpt_dir / "encoder_e1_best.pt"

    print(f"params: {params:,}  train: {train_len}  val: {val_len}  batch: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optim, device)
        va_loss, va_acc = eval_epoch(model, val_loader, device)
        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": args.hidden,
                    "stages": [s.value for s in stages],
                    "stage_default": "B",
                    "val_acc": va_acc,
                    "epoch": epoch,
                },
                best_path,
            )
            mark = " *best*"
        print(
            f"epoch {epoch:02d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.3f}{mark}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "hidden": args.hidden,
            "stages": [s.value for s in stages],
            "stage_default": "B",
            "val_acc": va_acc,
            "epoch": args.epochs,
        },
        ckpt_dir / "encoder_e1_latest.pt",
    )

    print(f"\nEvaluating learned encoder on curriculum ({args.eval_episodes} eps/stage)...")
    policy_obj = load_policy(args.policy)
    eval_summary = {}
    for stage in stages:
        metrics_list = []
        for i in range(args.eval_episodes):
            ep_seed = args.seed + 1000 + i + ord(stage.value[0]) * 1000
            world = sample_world(random.Random(ep_seed), num_facts=1 + (i % 2))
            script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 1))
            enc = LearnedEncoder(model, device=device, stage=stage.value)
            _, m = run_episode(
                world=world,
                script=script,
                policy=policy_obj,
                encoder=enc,
                seed=ep_seed,
                stage=stage.value,
            )
            metrics_list.append(m)
        s = summarize(metrics_list)
        eval_summary[stage.value] = s[stage.value]

    for st, stats in eval_summary.items():
        print(
            f"  stage {st}: query_acc={stats['mean_query_accuracy']:.3f}  "
            f"revert={stats['mean_revert_rate']:.3f}"
        )

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "params": params,
        "best_val_acc": best_acc,
        "checkpoint": str(best_path),
        "curriculum_eval": eval_summary,
    }
    report_path = ckpt_dir / "encoder_e1_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nbest val acc: {best_acc:.3f}  checkpoint: {best_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
