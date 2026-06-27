"""E11a — train ~50M open-phrasing NL parser (v1.2 scale-up)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from greenfield.eval_nl_messy import eval_messy
from greenfield.eval_open_phrasing import eval_open_phrasing
from greenfield.log_util import configure_unbuffered, log
from greenfield.nl_gateway import LearnedEventParser
from greenfield.train.dataset_open_nl import OpenNlDataset
from greenfield.train.model_presets import NL_E11A
from greenfield.train.nl_parser_model import NlParserModel
from greenfield.train.nl_transformer import count_parameters
from greenfield.train.train_encoder import resolve_device
from greenfield.train.train_encoder_e10a import eval_loader, train_epoch


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="E11a — 50M open NL parser")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--val-size", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    args = parser.parse_args()

    preset = NL_E11A
    device = resolve_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "encoder_e11a_best.pt"
    best_score = float("-inf")

    model = NlParserModel(**preset.as_dict()).to(device)
    params = count_parameters(model)
    log(f"device: {device}  preset: {preset.name}  params: {params:,}  utterance_len=96")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    val_ds = OpenNlDataset(size=args.val_size, seed=args.seed + 9999, open_fraction=0.7, messy_fraction=0.5)

    for epoch in range(1, args.epochs + 1):
        train_ds = OpenNlDataset(
            size=args.train_size,
            seed=args.seed + epoch,
            open_fraction=0.7,
            messy_fraction=0.5,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        tr_loss, tr_i, tr_s = train_epoch(model, train_loader, optim, device)
        va_i, va_s = eval_loader(model, val_loader, device)

        tmp = {
            "model": model.state_dict(),
            "parser": "nl_v2",
            "experiment": "e11a_open_50m",
            "preset": preset.name,
            "model_config": preset.as_dict(),
            "utterance_len": 96,
            "params": params,
            "epoch": epoch,
        }
        torch.save(tmp, ckpt_dir / "_e11a_eval_tmp.pt")
        learned = LearnedEventParser.from_checkpoint(ckpt_dir / "_e11a_eval_tmp.pt", device=device)
        open_m = eval_open_phrasing(learned, size=1200, seed=8080)
        messy_m = eval_messy(learned, size=800, seed=4242)
        score = open_m["intent_acc"] + open_m["slot_acc"] + messy_m["intent_acc"] + messy_m["slot_acc"]
        mark = ""
        if score > best_score:
            best_score = score
            tmp.update({"open_intent": open_m["intent_acc"], "messy_intent": messy_m["intent_acc"]})
            torch.save(tmp, best_path)
            mark = " *best*"
        log(
            f"epoch {epoch:02d} loss={tr_loss:.4f} train_i={tr_i:.3f} train_s={tr_s:.3f} "
            f"val_i={va_i:.3f} val_s={va_s:.3f} open_i={open_m['intent_acc']:.3f} "
            f"open_s={open_m['slot_acc']:.3f} messy_i={messy_m['intent_acc']:.3f}{mark}"
        )
        if open_m["intent_acc"] >= 0.99 and open_m["slot_acc"] >= 0.97 and messy_m["intent_acc"] >= 0.99:
            log("E11a targets reached — early stop")
            break

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e11a_open_50m",
        "preset": preset.name,
        "checkpoint": str(best_path),
        "params": params,
        "best_score": best_score,
    }
    (ckpt_dir / "encoder_e11a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ckpt_dir / "_e11a_eval_tmp.pt").unlink(missing_ok=True)
    log(f"checkpoint: {best_path}  best_score: {best_score:.3f}")


if __name__ == "__main__":
    main()
