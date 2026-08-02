#!/usr/bin/env python3
"""Resolve a stage-owned Helena Framework script by its stable filename.

The framework keeps implementations beside their owning stages instead of in
one flat ``scripts/`` directory.  Cross-stage orchestrators use this resolver
so moving a script to its semantic stage does not silently break a pipeline.
Legacy flat paths are accepted when restoring an old immutable runtime, but a
new checkout resolves from ``framework/stages/*/scripts``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _validate_name(name: str) -> None:
    candidate = Path(name)
    if not name or candidate.name != name or name in {".", ".."}:
        raise ValueError(f"script name must be a basename: {name!r}")


def stage_script_candidates(root: Path, name: str) -> list[Path]:
    """Return all existing canonical/legacy candidates in stable order."""

    _validate_name(name)
    root = root.resolve()
    candidates = [
        root / "scripts" / "harness" / name,
        root / "scripts" / name,
        *sorted((root / "framework" / "stages").glob(f"*/scripts/{name}")),
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(path)
    return unique


def resolve_stage_script(root: Path, name: str) -> Path:
    """Resolve one script or fail closed on missing/ambiguous ownership."""

    candidates = stage_script_candidates(root, name)
    if not candidates:
        raise FileNotFoundError(f"stage-owned script is missing: {name}")
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"ambiguous stage ownership for {name}: {rendered}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(resolve_stage_script(args.root, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
