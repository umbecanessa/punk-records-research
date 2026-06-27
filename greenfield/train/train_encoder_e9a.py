"""E9a training: transformer NL front (warm-start heads from E7)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from greenfield.log_util import configure_unbuffered, log
from greenfield.nl_gateway import LearnedEventParser
from greenfield.eval_nl import eval_parser, eval_parser_heldout_names
from greenfield.eval_nl_messy import eval_messy
from greenfield.train.checkpoint_util import current_vocab, load_encoder_model
from greenfield.train.dataset_nl import NlParseDataset
from greenfield.train.nl_transformer import count_parameters
from greenfield.train.train_encoder import resolve_device
from greenfield.train.train_encoder_e7 import eval_loader, set_nl_requires_grad, train_epoch


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E9a — transformer NL front")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default="greenfield/checkpoints/encoder_e7_best.pt")
    parser.add_argument("--train-size", type=int, default=80_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--messy-fraction", type=float, default=0.5)
    parser.add_argument("--messy-eval-size", type=int, default=800)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "encoder_e9a_best.pt"
    best_score = float("-inf")

    model = load_encoder_model(
        args.warm_start,
        device,
        predict_slot=True,
        predict_value=True,
        predict_event=True,
        expand_vocab=True,
    )
    from greenfield.train.model import _build_nl_utterance_enc

    model.nl_backbone = "transformer"
    model.nl_utterance_enc = _build_nl_utterance_enc(nl_backbone="transformer", dropout=0.1).to(device)
    nl_params = count_parameters(model.nl_utterance_enc)
    log(f"device: {device}  nl_backbone: transformer  nl_params: {nl_params:,}")

    set_nl_requires_grad(model, train_nl=True)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    val_ds = NlParseDataset(
        size=args.val_size,
        seed=args.seed + 9999,
        messy_fraction=args.messy_fraction,
    )

    for epoch in range(1, args.epochs + 1):
        train_ds = NlParseDataset(
            size=args.train_size,
            seed=args.seed + epoch,
            messy_fraction=args.messy_fraction,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

        tr_loss, tr_intent, tr_slot = train_epoch(
            model,
            train_loader,
            optim,
            device,
            intent_w=1.0,
            slot_w=2.0,
        )
        val_intent, val_slot = eval_loader(model, val_loader, device)

        tmp_ckpt = {
            "model": model.state_dict(),
            "hidden": 128,
            "predict_slot": True,
            "predict_value": True,
            "predict_event": True,
            "nl_backbone": "transformer",
            "vocab": current_vocab(),
            "experiment": "e9a_nl_transformer",
        }
        torch.save(tmp_ckpt, ckpt_dir / "_e9a_eval_tmp.pt")
        learned = LearnedEventParser.from_checkpoint(ckpt_dir / "_e9a_eval_tmp.pt", device=device)
        end_to_end = eval_parser(learned, size=args.eval_size, seed=args.seed + 4242)
        heldout = eval_parser_heldout_names(
            learned,
            size=min(100, args.eval_size // 5),
            seed=args.seed + 9001,
        )
        messy = eval_messy(learned, size=args.messy_eval_size, seed=8080)
        score = sum(
            [
                end_to_end["intent_acc"],
                end_to_end["slot_acc"],
                end_to_end["value_exact"],
                heldout["intent_acc"],
                heldout["slot_acc"],
                heldout["value_exact"],
                messy["intent_acc"],
                messy["slot_acc"],
                messy["value_exact"],
            ]
        )
        mark = ""
        if score > best_score:
            best_score = score
            tmp_ckpt.update(
                {
                    "val_messy_intent": messy["intent_acc"],
                    "val_messy_slot": messy["slot_acc"],
                    "val_messy_value": messy["value_exact"],
                    "epoch": epoch,
                    "nl_params": nl_params,
                }
            )
            torch.save(tmp_ckpt, best_path)
            mark = " *best*"

        log(
            f"epoch {epoch:02d}  loss={tr_loss:.4f}  "
            f"train intent={tr_intent:.3f} slot={tr_slot:.3f}  "
            f"loader val intent={val_intent:.3f} slot={val_slot:.3f}  "
            f"parse intent={end_to_end['intent_acc']:.3f} slot={end_to_end['slot_acc']:.3f} "
            f"value={end_to_end['value_exact']:.3f}  "
            f"messy intent={messy['intent_acc']:.3f} slot={messy['slot_acc']:.3f} "
            f"value={messy['value_exact']:.3f}{mark}"
        )

        if score >= 9.0 and messy["intent_acc"] >= 1.0:
            log("E9a target metrics reached — early stop")
            break

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e9a_nl_transformer",
        "checkpoint": str(best_path),
        "nl_params": nl_params,
        "best_score": best_score,
    }
    (ckpt_dir / "encoder_e9a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ckpt_dir / "_e9a_eval_tmp.pt").unlink(missing_ok=True)
    log(f"checkpoint: {best_path}  best_score: {best_score:.3f}")


if __name__ == "__main__":
    main()
