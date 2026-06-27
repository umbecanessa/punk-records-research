"""Run release validation before Hub upload (monorepo or synced public tree)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def validation_root(start: Path) -> Path | None:
    """Find a workspace root that contains greenfield.validate_release."""
    for candidate in (start, start.parent):
        mod = candidate / "greenfield" / "validate_release.py"
        if mod.is_file():
            return candidate
    return None


def run_release_validation(*, cwd: Path | None = None) -> None:
    root = validation_root(cwd or Path(__file__).resolve().parents[1])
    if root is None:
        print(
            "error: cannot find greenfield/validate_release.py — sync from monorepo first",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"=== release validation ({root}) ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "greenfield.validate_release"],
        cwd=root,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
