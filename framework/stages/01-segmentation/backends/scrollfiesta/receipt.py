"""Content-addressed file and JSON helpers for immutable adapter outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def file_artifact(path: Path, *, role: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    artifact: dict[str, Any] = {
        "uri": path.as_uri(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if role is not None:
        artifact = {"role": role, **artifact}
    return artifact


def build_tifxyz_manifest(tifxyz_dir: Path, destination: Path) -> dict[str, Any]:
    tifxyz_dir = Path(tifxyz_dir).resolve(strict=True)
    rows = []
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        path = tifxyz_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty TIFXYZ artifact: {path}")
        rows.append(
            {
                "name": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema": "campaignx.tifxyz_artifact_manifest.v1",
        "coordinate_order": "XYZ",
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    write_new_json(destination, manifest)
    return manifest
