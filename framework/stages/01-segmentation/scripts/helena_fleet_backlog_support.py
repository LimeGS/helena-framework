#!/usr/bin/env python3
"""Shared, dependency-free helpers for the Stage 01 fleet backlog reports.

Three separate closeouts read the same immutable evidence (planner packets,
growth receipts, the geometry surface catalogue and the post-fit relation
guard) and must agree on how the repository root is discovered, how JSON is
hashed, and how an artefact is written atomically.  Keeping that in one module
prevents three subtly different definitions of the same number.

Importing this module also puts the Stage 01 directory on ``sys.path`` so that
``fleet.planner`` resolves from a checkout or from a partial stage deployment,
matching ``helena_segment_search_fleet.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repository_root(script_path: Path | None = None) -> Path:
    """Find the Helena Framework checkout root, or a partial Stage 01 deployment."""

    configured = os.environ.get("HELENA_REPO_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / "framework/stages/01-segmentation").is_dir():
            raise RuntimeError(
                "HELENA_REPO_ROOT does not contain framework/stages/01-segmentation"
            )
        return candidate
    resolved = (script_path or Path(__file__)).resolve()
    for parent in resolved.parents:
        if (parent / ".git").exists():
            return parent
    if len(resolved.parents) > 4:
        candidate = resolved.parents[4]
        if (candidate / "framework/stages/01-segmentation").is_dir():
            return candidate
    raise RuntimeError(
        "cannot discover the Helena Framework root; set HELENA_REPO_ROOT"
    )


STAGE_ROOT = Path(__file__).resolve().parents[1]
for _path in (STAGE_ROOT, repository_root(Path(__file__))):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def framework_version(root: Path) -> str | None:
    version_file = Path(root) / "VERSION"
    if not version_file.is_file():
        return None
    return version_file.read_text(encoding="utf-8").strip()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_stdev(values: list[float]) -> float | None:
    """Population standard deviation; ``None`` below two observations.

    A single observation has no spread, and reporting ``0.0`` for it would let
    a one-sample recipe look perfectly reproducible.
    """

    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5
