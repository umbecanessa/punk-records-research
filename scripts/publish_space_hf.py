#!/usr/bin/env python3
"""Publish Gradio Space to Hugging Face (standalone space repo)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGING = REPO_ROOT / "space" / ".hf-space-staging"


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def build_staging() -> Path:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # Space frontmatter README
    readme = (REPO_ROOT / "space" / "README.md").read_text(encoding="utf-8")
    (STAGING / "README.md").write_text(readme, encoding="utf-8")

    shutil.copy2(REPO_ROOT / "space" / "app.py", STAGING / "app.py")
    shutil.copy2(REPO_ROOT / "space" / "demo_util.py", STAGING / "demo_util.py")
    shutil.copy2(REPO_ROOT / "space" / "requirements.txt", STAGING / "requirements.txt")

    dest_gf = STAGING / "greenfield"
    shutil.copytree(
        REPO_ROOT / "greenfield",
        dest_gf,
        ignore=shutil.ignore_patterns("__pycache__", "*.pt", "checkpoints"),
    )
    (dest_gf / "checkpoints").mkdir(exist_ok=True)
    return STAGING


def upload(repo_id: str, *, private: bool, message: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        fail('pip install -e ".[hub]"')

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=repo_id,
        repo_type="space",
        commit_message=message,
    )
    log(f"space live: https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Gradio Space")
    parser.add_argument("--repo-id", default="", help="e.g. wasnaga/punk-records-research-demo")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--message", default="Punk Records Research kernel demo")
    args = parser.parse_args()

    staging = build_staging()
    n = len([p for p in staging.rglob("*") if p.is_file()])
    log(f"staged {n} files -> {staging}")

    if args.dry_run:
        return

    repo_id = args.repo_id.strip()
    if not repo_id:
        fail("pass --repo-id wasnaga/punk-records-research-demo")
    upload(repo_id, private=args.private, message=args.message)


if __name__ == "__main__":
    main()
