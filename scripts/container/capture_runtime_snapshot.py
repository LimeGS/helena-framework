#!/usr/bin/env python3
"""Capture a sanitized, hash-addressed runtime inventory for a container base.

The snapshot contains only tool/model identities required to reproduce an
existing runtime. It intentionally never serializes arbitrary environment
variables, CT paths, cloud URLs, tokens, or file contents.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SAFE_ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_named_path(raw: str) -> tuple[str, Path]:
    name, separator, value = raw.partition("=")
    if not separator or not name or not value:
        raise argparse.ArgumentTypeError("expected NAME=/absolute/or/relative/path")
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a regular file: {path}")
    return name, path


def inventory(items: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for name, path in items:
        if name in seen:
            raise ValueError(f"duplicate artifact name: {name}")
        seen.add(name)
        rows.append({"id": name, "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    return rows


def gpu_inventory() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": type(error).__name__}
    if result.returncode:
        return {"available": False, "reason": f"nvidia-smi exit {result.returncode}"}
    return {"available": True, "query": "name,driver_version", "rows": [line for line in result.stdout.splitlines() if line]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("vc3d", "ink", "qc", "orchestrator"), required=True)
    parser.add_argument("--binary", action="append", default=[], type=parse_named_path, metavar="NAME=PATH")
    parser.add_argument("--model", action="append", default=[], type=parse_named_path, metavar="NAME=PATH")
    parser.add_argument("--package", action="append", default=[], metavar="DISTRIBUTION")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing overwrite: {args.output}")
    try:
        packages = []
        for distribution in sorted(set(args.package)):
            packages.append({"distribution": distribution, "version": importlib.metadata.version(distribution)})
        payload = {
            "kind": "campaign_x_container_runtime_snapshot_v1",
            "generated_at_utc": utc_now(),
            "stage": args.stage,
            "python": {"executable": sys.executable, "version": sys.version, "implementation": platform.python_implementation()},
            "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "gpu": gpu_inventory(),
            "safe_environment": {key: os.environ[key] for key in SAFE_ENVIRONMENT if key in os.environ},
            "packages": packages,
            "binaries": inventory(args.binary),
            "models": inventory(args.model),
            "policy": ["no secrets or arbitrary environment variables", "no CT paths, URLs, data, or model contents", "output is immutable and never overwritten"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({"status": "CAPTURED", "output": str(args.output), "sha256": sha256(args.output)}, sort_keys=True))
        return 0
    except (ValueError, importlib.metadata.PackageNotFoundError) as error:
        print(f"SNAPSHOT_FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
