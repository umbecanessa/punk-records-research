"""E10.1 — train 96-char open-phrasing NL parser."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from greenfield.eval_nl_messy import eval_messy
from greenfield.eval_open_phrasing import eval_open_phrasing
from greenfield.log_util import configure_unbuffered, log
from greenfield.nl_gateway import LearnedEventParser
from greenfield.train.dataset_open_nl import OpenNlDataset
from greenfield.train.model_presets import NL_E10A
from greenfield.train.nl_parser_model import NlParserModel
from greenfield.train.nl_transformer import count_parameters
from greenfield.train.train_encoder import resolve_device
from greenfield.types import Intent


def train_epoch(model, loader, optim, device) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    intent_ok = slot_ok = 0
    intent_n = slot_n = 0
    plant_id = 0
    query_id = 1
    for utt, intent_y, slot_y in loader:
        utt = utt.to(device)
        intent_y = intent_y.to(device)
        slot_y = slot_y.to(device)
        intent_logits, slot_logits = model(utt)
        loss = F.cross_entropy(intent_logits, intent_y)
        mask = (intent_y == plant_id) | (intent_y == query_id)
        if mask.any():
            loss = loss + 2.0 * F.cross_entropy(slot_logits[mask], slot_y[mask])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total_loss += loss.item() * utt.size(0)
        intent_ok += (intent_logits.argmax(-1) == intent_y).sum().item()
        intent_n += utt.size(0)
        if mask.any():
            slot_ok += (slot_logits[mask].argmax(-1) == slot_y[mask]).sum().item()
            slot_n += int(mask.sum().item())
    return total_loss / max(1, intent_n), intent_ok / max(1, intent_n), slot_ok / max(1, slot_n)


@torch.no_grad()
def eval_loader(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    intent_ok = slot_ok = 0
    intent_n = slot_n = 0
    plant_id = 0
    query_id = 1
    for utt, intent_y, slot_y in loader:
        utt = utt.to(device)
        intent_y = intent_y.to(device)
        slot_y = slot_y.to(device)
        intent_logits, slot_logits = model(utt)
        intent_ok += (intent_logits.argmax(-1) == intent_y).sum().item()
        intent_n += utt.size(0)
        mask = (intent_y == plant_id) | (intent_y == query_id)
        if mask.any():
            slot_ok += (slot_logits[mask].argmax(-1) == slot_y[mask]).sum().item()
            slot_n += int(mask.sum().item())
    return intent_ok / max(1, intent_n), slot_ok / max(1, slot_n)


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="E10.1 open NL parser")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-size", type=int, default=60_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "encoder_e10a_best.pt"
    best_score = float("-inf")

    model = NlParserModel().to(device)
    params = count_parameters(model)
    log(f"device: {device}  NlParserModel params: {params:,}  utterance_len=96")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    val_ds = OpenNlDataset(size=args.val_size, seed=args.seed + 9999)

    for epoch in range(1, args.epochs + 1):
        train_ds = OpenNlDataset(size=args.train_size, seed=args.seed + epoch)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        tr_loss, tr_i, tr_s = train_epoch(model, train_loader, optim, device)
        va_i, va_s = eval_loader(model, val_loader, device)

        tmp = {
            "model": model.state_dict(),
            "parser": "nl_v2",
            "experiment": "e10a_open",
            "preset": NL_E10A.name,
            "model_config": NL_E10A.as_dict(),
            "utterance_len": 96,
            "params": params,
            "epoch": epoch,
        }
        torch.save(tmp, ckpt_dir / "_e10a_eval_tmp.pt")
        learned = LearnedEventParser.from_checkpoint(ckpt_dir / "_e10a_eval_tmp.pt", device=device)
        open_m = eval_open_phrasing(learned, size=800, seed=8080)
        messy_m = eval_messy(learned, size=400, seed=4242)
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
        if open_m["intent_acc"] >= 0.98 and open_m["slot_acc"] >= 0.98 and messy_m["intent_acc"] >= 0.98:
            log("E10.1 targets reached — early stop")
            break

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": "e10a_open",
        "checkpoint": str(best_path),
        "params": params,
        "best_score": best_score,
    }
    (ckpt_dir / "encoder_e10a_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ckpt_dir / "_e10a_eval_tmp.pt").unlink(missing_ok=True)
    log(f"checkpoint: {best_path}  best_score: {best_score:.3f}")


if __name__ == "__main__":
    main()
