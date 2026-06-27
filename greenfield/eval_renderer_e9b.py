"""Evaluate E9b transformer renderer on template dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from greenfield.renderer.dataset import RenderDataset
from greenfield.renderer.transformer_renderer import TransformerRendererModel
from greenfield.train.train_renderer_e9b import eval_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="E9b renderer eval")
    parser.add_argument("--checkpoint", default="greenfield/checkpoints/renderer_e9b_best.pt")
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.is_file():
        raise SystemExit(f"missing checkpoint: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TransformerRendererModel().to(device)
    model.load_state_dict(ckpt["model"])

    ds = RenderDataset(size=args.size, seed=args.seed)
    _, val = random_split(ds, [max(1, args.size - 200), min(200, args.size - 1)])
    loader = DataLoader(val, batch_size=64, shuffle=False)
    loss, exact = eval_epoch(model, loader, device)
    print({"val_loss": loss, "val_exact": exact, "params": ckpt.get("params")})


if __name__ == "__main__":
    main()
