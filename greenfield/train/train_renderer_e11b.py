"""E11b — train ~50M paraphrase renderer (v1.2 scale-up)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from greenfield.log_util import configure_unbuffered, log
from greenfield.renderer.dataset import ParaphraseRenderDataset
from greenfield.renderer.transformer_renderer import TransformerRendererModel, count_parameters
from greenfield.train.model_presets import RENDERER_E11B
from greenfield.train.train_encoder import resolve_device
from greenfield.train.train_renderer_e9b import eval_epoch, train_epoch


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="E11b — 50M paraphrase renderer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-size", type=int, default=80_000)
    parser.add_argument("--val-size", type=int, default=6_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    args = parser.parse_args()

    preset = RENDERER_E11B
    device = resolve_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "renderer_e11b_best.pt"
    best_exact = 0.0

    model = TransformerRendererModel(**preset.as_dict()).to(device)
    params = count_parameters(model)
    log(f"device: {device}  preset: {preset.name}  params: {params:,}")

    torch.manual_seed(args.seed)
    full = ParaphraseRenderDataset(size=args.train_size + args.val_size, seed=args.seed)
    train_ds, val_ds = random_split(
        full,
        [args.train_size, args.val_size],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_exact = train_epoch(model, train_loader, optim, device)
        va_loss, va_exact = eval_epoch(model, val_loader, device)
        mark = ""
        if va_exact > best_exact:
            best_exact = va_exact
            torch.save(
                {
                    "model": model.state_dict(),
                    "experiment": "e11b_paraphrase_50m",
                    "preset": preset.name,
                    "model_config": preset.as_dict(),
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
        if va_exact >= 0.995:
            log("E11b target val exact reached — early stop")
            break

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e11b_paraphrase_50m",
        "preset": preset.name,
        "params": params,
        "best_val_exact": best_exact,
        "checkpoint": str(best_path),
    }
    (ckpt_dir / "renderer_e11b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"checkpoint: {best_path}  best_val_exact: {best_exact:.3f}")


if __name__ == "__main__":
    main()
