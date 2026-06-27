"""E7 training: utterance → intent / slot (value via percept bootstrap + E6 head)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from greenfield.log_util import configure_unbuffered, log
from greenfield.nl_gateway import LearnedEventParser
from greenfield.eval_nl import eval_parser, eval_parser_heldout_names
from greenfield.eval_nl_messy import eval_messy
from greenfield.train.checkpoint_util import current_vocab, load_encoder_model
from greenfield.train.dataset_nl import NlParseDataset
from greenfield.train.features import E7_INTENT_TO_ID
from greenfield.train.train_encoder import resolve_device
from greenfield.types import Intent


def set_nl_requires_grad(model, *, train_nl: bool, train_all: bool = False) -> None:
    for name, param in model.named_parameters():
        if train_all:
            param.requires_grad = True
        elif train_nl and name.startswith(("nl_utterance_enc.", "event_intent_head.", "event_slot_head.")):
            param.requires_grad = True
        else:
            param.requires_grad = False


def train_epoch(
    model,
    loader,
    optim,
    device,
    *,
    intent_w: float,
    slot_w: float,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    intent_ok = slot_ok = 0
    intent_n = slot_n = 0
    plant_id = E7_INTENT_TO_ID[Intent.PLANT]
    query_id = E7_INTENT_TO_ID[Intent.QUERY]

    for x, intent_y, slot_y, _value_y in loader:
        x = x.to(device)
        intent_y = intent_y.to(device)
        slot_y = slot_y.to(device)

        intent_logits, slot_logits, _ = model(x, return_event=True)
        loss = intent_w * F.cross_entropy(intent_logits, intent_y)

        slot_mask = (intent_y == plant_id) | (intent_y == query_id)
        if slot_logits is not None and slot_mask.any() and slot_w > 0:
            loss = loss + slot_w * F.cross_entropy(slot_logits[slot_mask], slot_y[slot_mask])

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total_loss += loss.item() * x.size(0)
        intent_ok += (intent_logits.argmax(dim=-1) == intent_y).sum().item()
        intent_n += x.size(0)
        if slot_logits is not None and slot_mask.any():
            pred = slot_logits[slot_mask].argmax(dim=-1)
            slot_ok += (pred == slot_y[slot_mask]).sum().item()
            slot_n += int(slot_mask.sum().item())

    return (
        total_loss / max(1, intent_n),
        intent_ok / max(1, intent_n),
        slot_ok / max(1, slot_n),
    )


@torch.no_grad()
def eval_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    intent_ok = slot_ok = 0
    intent_n = slot_n = 0
    plant_id = E7_INTENT_TO_ID[Intent.PLANT]
    query_id = E7_INTENT_TO_ID[Intent.QUERY]

    for x, intent_y, slot_y, _value_y in loader:
        x = x.to(device)
        intent_y = intent_y.to(device)
        slot_y = slot_y.to(device)
        intent_logits, slot_logits, _ = model(x, return_event=True)

        intent_ok += (intent_logits.argmax(dim=-1) == intent_y).sum().item()
        intent_n += x.size(0)

        slot_mask = (intent_y == plant_id) | (intent_y == query_id)
        if slot_logits is not None and slot_mask.any():
            pred = slot_logits[slot_mask].argmax(dim=-1)
            slot_ok += (pred == slot_y[slot_mask]).sum().item()
            slot_n += int(slot_mask.sum().item())

    return intent_ok / max(1, intent_n), slot_ok / max(1, slot_n)


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E7 — NL intent/slot parser")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default="greenfield/checkpoints/encoder_e6_best.pt")
    parser.add_argument("--train-size", type=int, default=80_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--intent-weight", type=float, default=1.0)
    parser.add_argument("--slot-weight", type=float, default=2.0)
    parser.add_argument("--messy-fraction", type=float, default=0.5)
    parser.add_argument("--messy-eval-size", type=int, default=800)
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    parser.add_argument("--eval-size", type=int, default=500)
    args = parser.parse_args()

    device = resolve_device(args.device)
    log(f"device: {device}  messy_fraction: {args.messy_fraction}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "encoder_e7_best.pt"
    best_score = float("-inf")

    model = load_encoder_model(
        args.warm_start,
        device,
        predict_slot=True,
        predict_value=True,
        predict_event=True,
        expand_vocab=True,
    )
    set_nl_requires_grad(model, train_nl=True)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log("training nl_utterance_enc + event intent/slot heads (backbone frozen)")

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
            intent_w=args.intent_weight,
            slot_w=args.slot_weight,
        )
        val_intent, val_slot = eval_loader(model, val_loader, device)

        # End-to-end parse eval (includes value via percept bootstrap).
        torch.save({"model": model.state_dict()}, ckpt_dir / "_e7_eval_tmp.pt")
        end_to_end = eval_parser(
            LearnedEventParser.from_checkpoint(ckpt_dir / "_e7_eval_tmp.pt", device=device),
            size=args.eval_size,
            seed=args.seed + 4242,
        )
        heldout = eval_parser_heldout_names(
            LearnedEventParser.from_checkpoint(ckpt_dir / "_e7_eval_tmp.pt", device=device),
            size=min(100, args.eval_size // 5),
            seed=args.seed + 9001,
        )
        messy = eval_messy(
            LearnedEventParser.from_checkpoint(ckpt_dir / "_e7_eval_tmp.pt", device=device),
            size=args.messy_eval_size,
            seed=8080,
        )
        score = (
            end_to_end["intent_acc"]
            + end_to_end["slot_acc"]
            + end_to_end["value_exact"]
            + heldout["intent_acc"]
            + heldout["slot_acc"]
            + heldout["value_exact"]
            + messy["intent_acc"]
            + messy["slot_acc"]
            + messy["value_exact"]
        )
        mark = ""
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": 128,
                    "predict_slot": True,
                    "predict_value": True,
                    "predict_event": True,
                    "vocab": current_vocab(),
                    "experiment": "e7_nl_v5_messy",
                    "messy_fraction": args.messy_fraction,
                    "val_intent_acc": end_to_end["intent_acc"],
                    "val_slot_acc": end_to_end["slot_acc"],
                    "val_value_exact": end_to_end["value_exact"],
                    "val_heldout_intent": heldout["intent_acc"],
                    "val_heldout_value": heldout["value_exact"],
                    "val_messy_intent": messy["intent_acc"],
                    "val_messy_slot": messy["slot_acc"],
                    "val_messy_value": messy["value_exact"],
                    "epoch": epoch,
                },
                best_path,
            )
            mark = " *best*"

        log(
            f"epoch {epoch:02d}  loss={tr_loss:.4f}  "
            f"train intent={tr_intent:.3f} slot={tr_slot:.3f}  "
            f"loader val intent={val_intent:.3f} slot={val_slot:.3f}  "
            f"parse intent={end_to_end['intent_acc']:.3f} slot={end_to_end['slot_acc']:.3f} "
            f"value={end_to_end['value_exact']:.3f}  "
            f"heldout intent={heldout['intent_acc']:.3f} slot={heldout['slot_acc']:.3f} "
            f"value={heldout['value_exact']:.3f}  "
            f"messy intent={messy['intent_acc']:.3f} slot={messy['slot_acc']:.3f} "
            f"value={messy['value_exact']:.3f}{mark}"
        )

        if (
            end_to_end["intent_acc"] >= 1.0
            and end_to_end["slot_acc"] >= 1.0
            and end_to_end["value_exact"] >= 1.0
            and heldout["intent_acc"] >= 1.0
            and heldout["slot_acc"] >= 1.0
            and heldout["value_exact"] >= 1.0
            and messy["intent_acc"] >= 1.0
            and messy["slot_acc"] >= 1.0
            and messy["value_exact"] >= 1.0
        ):
            log("target metrics reached — early stop")
            break

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e7_nl_v5_messy",
        "checkpoint": str(best_path),
        "best_score": best_score,
    }
    (ckpt_dir / "encoder_e7_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ckpt_dir / "_e7_eval_tmp.pt").unlink(missing_ok=True)
    log(f"checkpoint: {best_path}  best_score: {best_score:.3f}")


if __name__ == "__main__":
    main()
