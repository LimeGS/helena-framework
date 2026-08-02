#!/usr/bin/env python3
"""Validate the separately-built Villa flatten/TIFXYZ runtime input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


EXPECTED_COMMIT = "05dcf0349356bc833670d61e5eca00be58376e35"
REQUIRED = ("flatboi", "vc_obj2tifxyz_legacy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, object]:
    manifest_path = root / "VILLA_RUNTIME_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "campaignx.villa_runtime_bundle.v1":
        raise ValueError("unsupported Villa runtime manifest schema")
    if manifest.get("source_commit") != EXPECTED_COMMIT:
        raise ValueError("Villa source commit is not frozen")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Villa manifest artifacts must be an object")
    verified: dict[str, str] = {}
    for name in REQUIRED:
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"missing Villa runtime artifact: {name}")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe Villa artifact path: {name}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid Villa artifact SHA-256: {name}")
        path = root / relative
        if not path.is_file() or not (path.stat().st_mode & 0o111):
            raise ValueError(f"Villa artifact is missing or not executable: {name}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Villa artifact hash mismatch: {name}")
        verified[name] = actual
    return {
        "schema": "campaignx.villa_runtime_verification.v1",
        "status": "VERIFIED",
        "source_commit": EXPECTED_COMMIT,
        "manifest_sha256": sha256(manifest_path),
        "artifacts": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = verify(args.root.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Villa runtime verification failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
