#!/usr/bin/env python3
"""Write SHA-256 provenance for the real maps used by calibration.

The calibration registry deliberately stores coordinates rather than copying
large source rasters into this repository. This command binds every registry
path to exact byte size and SHA-256 so an integration replay can reject a
silently changed local data tree.

Run from the repository root after rebuilding ``calibration_set.json``:

    python calibration/record_source_hashes.py --project-root ../..
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=HERE.parents[2])
    parser.add_argument("--set", dest="set_path", default=HERE / "calibration_set.json")
    parser.add_argument("--out", default=HERE / "source_manifest.json")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve()
    set_path = Path(args.set_path).resolve()
    registry_bytes = set_path.read_bytes()
    registry = json.loads(registry_bytes)
    records = []
    for rel in sorted({row["map"] for row in registry["windows"]}):
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"calibration source is missing: {path}")
        records.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    output = {
        "schema_version": "v1",
        "calibration_set_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "sources": records,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out}: {len(records)} source maps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
