#!/usr/bin/env python3
"""Stage and upload Punk Records Research kernel v0.1 to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HUB_DIR = REPO_ROOT / "hub"
STAGING = HUB_DIR / ".staging"
DEPLOY = REPO_ROOT / "greenfield" / "deploy"

POLICY_FILES = (
    "policy.v0.json",
    "policy.overflow.json",
    "policy.merkle.json",
    "policy.promote.json",
    "policy.extended.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def checkpoint_dir() -> Path:
    """Prefer local checkpoints; fall back to parent monorepo when nested."""
    candidates = [
        REPO_ROOT / "greenfield" / "checkpoints",
        REPO_ROOT.parent / "greenfield" / "checkpoints",
    ]
    for path in candidates:
        enc = path / "encoder_e6_best.pt"
        ren = path / "renderer_e3_best.pt"
        if enc.is_file() and ren.is_file():
            return path
    fail(
        "missing encoder_e6_best.pt / renderer_e3_best.pt\n"
        "  train locally, download from Hub into greenfield/checkpoints/, or run from monorepo workspace"
    )


def build_staging(*, clean: bool = True) -> Path:
    ckpt = checkpoint_dir()
    required = (ckpt / "encoder_e6_best.pt", ckpt / "renderer_e3_best.pt")

    if clean and STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    shutil.copy2(HUB_DIR / "README.md", STAGING / "README.md")
    shutil.copy2(REPO_ROOT / "docs" / "KERNEL.md", STAGING / "KERNEL.md")
    shutil.copy2(HUB_DIR / "stack.json", STAGING / "stack.json")
    shutil.copy2(required[0], STAGING / "encoder_e6_best.pt")
    shutil.copy2(required[1], STAGING / "renderer_e3_best.pt")

    policies = STAGING / "policies"
    policies.mkdir(exist_ok=True)
    for name in POLICY_FILES:
        src = DEPLOY / name
        if src.is_file():
            shutil.copy2(src, policies / name)

    eval_dir = STAGING / "eval"
    eval_dir.mkdir(exist_ok=True)
    for name in ("encoder_e6_report.json", "renderer_e3_report.json"):
        src = ckpt / name
        if src.is_file():
            shutil.copy2(src, eval_dir / name)

    manifest = {
        "bundle": "punk-records-research-kernel-v0.1",
        "stack": json.loads((HUB_DIR / "stack.json").read_text(encoding="utf-8")),
        "files": sorted(
            str(p.relative_to(STAGING)).replace("\\", "/") for p in STAGING.rglob("*") if p.is_file()
        ),
    }
    (STAGING / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return STAGING


def upload(repo_id: str, *, private: bool, commit_message: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        fail('install hub client: pip install -e ".[hub]"')

    api = HfApi()
    log(f"creating/updating repo: {repo_id} (private={private})")
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    log(f"uploading {STAGING} -> {repo_id}")
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    log(f"done: https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish kernel v0.1 to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="",
        help="Hub model repo, e.g. umbecanessa/punk-records-research-kernel-v0.1",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--message",
        default="Punk Records Research kernel v0.1 — E6 + E3 + policies",
    )
    args = parser.parse_args()

    staging = build_staging()
    total_bytes = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    n_files = len([p for p in staging.rglob("*") if p.is_file()])
    log(f"staged {n_files} files, {total_bytes / 1024 / 1024:.2f} MiB -> {staging}")

    if args.dry_run:
        log("dry-run: skipping upload")
        return

    repo_id = args.repo_id.strip()
    if not repo_id:
        fail("pass --repo-id umbecanessa/punk-records-research-kernel-v0.1")

    upload(repo_id, private=args.private, commit_message=args.message)


if __name__ == "__main__":
    main()
