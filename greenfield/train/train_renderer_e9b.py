"""E9b training — causal transformer renderer (storage read-only at inference)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from greenfield.log_util import configure_unbuffered, log
from greenfield.renderer.core import LearnedTransformerRenderer
from greenfield.renderer.dataset import RenderDataset
from greenfield.renderer.transformer_renderer import TransformerRendererModel, count_parameters
from greenfield.train.train_encoder import resolve_device
from greenfield.train.train_renderer import eval_render_fidelity


def train_epoch(model, loader, optim, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    exact = 0
    total = 0
    for slot, vids, target in loader:
        slot = slot.to(device)
        vids = vids.to(device)
        target = target.to(device)
        logits = model(slot, vids, target)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item() * slot.size(0)
        pred = logits.argmax(dim=-1)
        exact += (pred == target).all(dim=1).sum().item()
        total += slot.size(0)
    return total_loss / max(1, total), exact / max(1, total)


@torch.no_grad()
def eval_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    exact = 0
    total = 0
    for slot, vids, target in loader:
        slot = slot.to(device)
        vids = vids.to(device)
        target = target.to(device)
        logits = model(slot, vids, target)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss += loss.item() * slot.size(0)
        pred = logits.argmax(dim=-1)
        exact += (pred == target).all(dim=1).sum().item()
        total += slot.size(0)
    return total_loss / max(1, total), exact / max(1, total)


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E9b — transformer renderer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-size", type=int, default=24_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--encoder-checkpoint", default="greenfield/checkpoints/encoder_e6_best.pt")
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    device = resolve_device(args.device)
    log(f"device: {device}")

    torch.manual_seed(args.seed)
    full = RenderDataset(size=args.train_size + args.val_size, seed=args.seed)
    train_ds, val_ds = random_split(
        full,
        [args.train_size, args.val_size],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TransformerRendererModel().to(device)
    params = count_parameters(model)
    log(f"transformer renderer params: {params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "renderer_e9b_best.pt"
    best_exact = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_exact = train_epoch(model, train_loader, optim, device)
        va_loss, va_exact = eval_epoch(model, val_loader, device)
        mark = ""
        if va_exact > best_exact:
            best_exact = va_exact
            torch.save(
                {
                    "model": model.state_dict(),
                    "experiment": "e9b_transformer_renderer",
                    "val_exact": va_exact,
                    "params": params,
                    "epoch": epoch,
                },
                best_path,
            )
            mark = " *best*"
        log(
            f"epoch {epoch:02d}  train_loss={tr_loss:.4f} exact={tr_exact:.3f}  "
            f"val_loss={va_loss:.4f} exact={va_exact:.3f}{mark}"
        )
        if va_exact >= 1.0:
            log("E9b target val exact reached — early stop")
            break

    from greenfield.runner import load_policy

    policy = load_policy("greenfield/deploy/policy.v0.json")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    renderer = LearnedTransformerRenderer(model, device=device)
    end_to_end = {}
    enc_path = Path(args.encoder_checkpoint)
    if enc_path.is_file():
        end_to_end = eval_render_fidelity(
            renderer,  # type: ignore[arg-type]
            policy,
            enc_path,
            device=device,
            episodes=args.eval_episodes,
            seed=args.seed + 777,
        )
        log(f"mean_render_fidelity={end_to_end.get('mean_render_fidelity', 0):.3f}")

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e9b_transformer_renderer",
        "params": params,
        "best_val_exact": best_exact,
        "checkpoint": str(best_path),
        "end_to_end": end_to_end,
    }
    (ckpt_dir / "renderer_e9b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"checkpoint: {best_path}  best_val_exact: {best_exact:.3f}")


if __name__ == "__main__":
    main()
