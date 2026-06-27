"""E3 — train byte renderer (structured templates, never writes storage)."""

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
from greenfield.log_util import configure_unbuffered, log
from greenfield.renderer.core import ByteRendererModel, LearnedRenderer, TemplateRenderer
from greenfield.renderer.dataset import RenderDataset
from greenfield.renderer.templates import reference_text
from greenfield.runner import load_policy, run_episode, summarize
from greenfield.simulator import sample_world
from greenfield.train.checkpoint_util import load_encoder_model
from greenfield.train.train_encoder import resolve_device
from greenfield.types import OpCode, Policy


def train_epoch(model, loader, optim, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    exact = 0
    total = 0
    for slot, vids, target in loader:
        slot = slot.to(device)
        vids = vids.to(device)
        target = target.to(device)
        logits = model(slot, vids)
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
        logits = model(slot, vids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss += loss.item() * slot.size(0)
        pred = logits.argmax(dim=-1)
        exact += (pred == target).all(dim=1).sum().item()
        total += slot.size(0)
    return total_loss / max(1, total), exact / max(1, total)


@torch.no_grad()
def eval_render_fidelity(
    renderer: LearnedRenderer,
    policy: Policy,
    encoder_ckpt: Path,
    *,
    device: torch.device,
    episodes: int,
    seed: int,
) -> dict:
    enc_model = load_encoder_model(encoder_ckpt, device)
    stages = list(CurriculumStage)
    per_stage = {}
    hits = 0
    total = 0
    for stage in stages:
        stage_hits = 0
        stage_total = 0
        for i in range(episodes):
            ep_seed = seed + i + ord(stage.value[0]) * 1000
            world = sample_world(random.Random(ep_seed), num_facts=1)
            script = generate_script(world, stage=stage, rng=random.Random(ep_seed + 1))
            enc = LearnedEncoder(enc_model, device=device, stage=stage.value)
            _, m = run_episode(
                world=world,
                script=script,
                policy=policy,
                encoder=enc,
                renderer=renderer,
                reference_render=reference_text,
                stage=stage.value,
            )
            if m.render_total > 0:
                stage_total += m.render_total
                stage_hits += m.render_hits
                total += m.render_total
                hits += m.render_hits
        rate = stage_hits / max(1, stage_total)
        per_stage[stage.value] = {"render_fidelity": round(rate, 4), "n": stage_total}
    return {"mean_render_fidelity": hits / max(1, total), "stages": per_stage}


def main() -> None:
    configure_unbuffered()
    parser = argparse.ArgumentParser(description="Greenfield E3 — train renderer")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-checkpoint", default="greenfield/checkpoints/encoder_e5b_best.pt")
    parser.add_argument("--checkpoint-dir", default="greenfield/checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=15)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        log(f"device: {device} ({name}, {vram:.1f} GiB)")
    else:
        log(f"device: {device}")

    torch.manual_seed(args.seed)
    full = RenderDataset(size=args.train_size + args.val_size, seed=args.seed)
    train_len = args.train_size
    val_len = args.val_size
    train_ds, val_ds = random_split(
        full,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ByteRendererModel(hidden=args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "renderer_e3_best.pt"
    best_exact = 0.0

    log(f"params: {params:,}  train: {train_len}  val: {val_len}  batch: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_exact = train_epoch(model, train_loader, optim, device)
        va_loss, va_exact = eval_epoch(model, val_loader, device)
        mark = ""
        if va_exact > best_exact:
            best_exact = va_exact
            torch.save(
                {
                    "model": model.state_dict(),
                    "hidden": args.hidden,
                    "val_exact": va_exact,
                    "epoch": epoch,
                    "experiment": "e3",
                },
                best_path,
            )
            mark = " *best*"
        log(
            f"epoch {epoch:02d}  train_loss={tr_loss:.4f} exact={tr_exact:.3f}  "
            f"val_loss={va_loss:.4f} exact={va_exact:.3f}{mark}"
        )

    policy = load_policy("greenfield/deploy/policy.v0.json")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    renderer = LearnedRenderer(model, device=device)
    enc_path = Path(args.encoder_checkpoint)
    end_to_end = {}
    if enc_path.is_file():
        log("\nEnd-to-end render fidelity (E2 encoder + E3 renderer)...")
        end_to_end = eval_render_fidelity(
            renderer,
            policy,
            enc_path,
            device=device,
            episodes=args.eval_episodes,
            seed=args.seed + 777,
        )
        for st, stats in end_to_end["stages"].items():
            if stats["n"] > 0:
                log(f"  stage {st}: render_fidelity={stats['render_fidelity']:.3f}  n={stats['n']}")
        log(f"  mean_render_fidelity={end_to_end['mean_render_fidelity']:.3f}")

    report = {
        "time": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "experiment": "e3",
        "params": params,
        "best_val_exact": best_exact,
        "checkpoint": str(best_path),
        "end_to_end": end_to_end,
    }
    report_path = ckpt_dir / "renderer_e3_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"\nbest val exact: {best_exact:.3f}  checkpoint: {best_path}")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
