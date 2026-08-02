#!/usr/bin/env python3
"""Entry point for the autonomous Stage 01 Segment Search Fleet."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def discover_repo_root(script_path: Path | None = None) -> Path:
    """Find a full repository or a deployable partial Stage 01 checkout."""

    configured = os.environ.get("HELENA_REPO_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / "framework/stages/01-segmentation/fleet").is_dir():
            raise RuntimeError(
                "HELENA_REPO_ROOT does not contain "
                "framework/stages/01-segmentation/fleet"
            )
        return candidate

    resolved = (script_path or Path(__file__)).resolve()
    for parent in resolved.parents:
        if (parent / ".git").exists():
            return parent

    # Stateless GPU workers may receive only framework/stages/01-segmentation,
    # without Git metadata or the rest of Helena Framework.  The fixed stage layout
    # remains sufficient to run the fleet client in that deployment.
    if len(resolved.parents) > 4:
        candidate = resolved.parents[4]
        if (candidate / "framework/stages/01-segmentation/fleet").is_dir():
            return candidate

    raise RuntimeError(
        "cannot discover Helena Framework root; set HELENA_REPO_ROOT to the "
        "checkout containing framework/stages/01-segmentation/fleet"
    )


STAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = discover_repo_root()
for path in (STAGE_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fleet.cli import main


if __name__ == "__main__":
    raise SystemExit(main(repo_root=REPO_ROOT))
