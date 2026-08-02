#!/usr/bin/env python3
"""Verify archived TIFXYZ and prepare an immutable fleet-QC backfill manifest.

The geometry catalogue's historical ``tifxyz_sha256`` hashes only the four file
digests.  The distributed QC adapter consumes the newer artifact-set digest,
which also binds file sizes and a schema-bearing ``ARTIFACT_SET.json``.  This
tool bridges those contracts without changing any TIFXYZ coordinate file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
STAGE01 = ROOT / "framework/stages/01-segmentation"
for candidate in (ROOT, STAGE01):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fleet.common import (
    artifact_manifest,
    content_sha256,
    file_sha256,
    read_json,
    utc_now,
    write_json_atomic,
)


REQUIRED = ("x.tif", "y.tif", "z.tif", "meta.json")


def artifact_uri(root: str, relative: str) -> str:
    parsed = urlparse(root)
    if parsed.scheme == "s3":
        return f"{root.rstrip('/')}/{relative.strip('/')}"
    return str((Path(root).expanduser().resolve() / relative).resolve())


def verify_catalog_files(
    directory: Path, catalog_files: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    actual = artifact_manifest(directory, REQUIRED)
    for name in REQUIRED:
        expected = catalog_files.get(name)
        if not isinstance(expected, dict):
            raise RuntimeError(f"catalogue omitted {name}: {directory}")
        if actual[name] != {
            "size_bytes": int(expected.get("size_bytes", -1)),
            "sha256": expected.get("sha256"),
        }:
            raise RuntimeError(f"catalogue/file mismatch for {directory / name}")
    return actual


def prepare(
    *,
    catalog_path: Path,
    surface_root: Path,
    artifact_uri_root: str,
    output: Path,
    write_artifact_manifests: bool,
) -> dict[str, Any]:
    catalog = read_json(catalog_path)
    if not isinstance(catalog, dict):
        raise RuntimeError("geometry catalogue is not a JSON object")
    catalog_rows = catalog.get("campaign_surfaces")
    if not isinstance(catalog_rows, list):
        raise RuntimeError("geometry catalogue has no campaign_surfaces")
    surfaces: list[dict[str, Any]] = []
    total_area = 0.0
    for row in sorted(catalog_rows, key=lambda item: str(item["surface_id"])):
        relative = str(row["archive_relative"])
        directory = (surface_root / relative).resolve()
        if not directory.is_dir() or surface_root.resolve() not in directory.parents:
            raise RuntimeError(f"surface path escapes or is missing: {relative}")
        files = verify_catalog_files(directory, row.get("files", {}))
        artifact_sha256 = content_sha256(files)
        manifest = {
            "schema": "campaignx.segmentation_artifact_set.v1",
            "surface_id": str(row["surface_id"]),
            "sample_id": str(row["sample_id"]),
            "files": files,
            "artifact_sha256": artifact_sha256,
            "legacy_tifxyz_sha256": str(row["tifxyz_sha256"]),
            "area_cm2": float(row["area_cm2"]),
            "bbox_xyz": row["bbox_l0_xyz"],
            "source_catalog_sha256": file_sha256(catalog_path),
            "ink_used": False,
            "no_automatic_acceptance": True,
        }
        manifest_path = directory / "ARTIFACT_SET.json"
        artifact_set = manifest
        if manifest_path.exists():
            existing_manifest = read_json(manifest_path)
            if not isinstance(existing_manifest, dict) or (
                existing_manifest.get("artifact_sha256") != artifact_sha256
                or existing_manifest.get("files") != files
            ):
                raise RuntimeError(f"conflicting ARTIFACT_SET.json: {manifest_path}")
            # Preserve richer historical receipts (attempt, points, shape, etc.)
            # byte-for-byte. The immutable files and artifact digest are the
            # compatibility boundary required by the downstream adapter.
            artifact_set = existing_manifest
        elif write_artifact_manifests:
            write_json_atomic(manifest_path, manifest)
        area = float(row["area_cm2"])
        total_area += area
        surfaces.append(
            {
                "surface_id": str(row["surface_id"]),
                "sample_id": str(row["sample_id"]),
                "owner": str(row.get("owner", "campaign-x")),
                "archive_relative": relative,
                "artifact_uri": artifact_uri(artifact_uri_root, relative),
                "artifact_sha256": artifact_sha256,
                "artifact_manifest_sha256": content_sha256(artifact_set),
                "legacy_tifxyz_sha256": str(row["tifxyz_sha256"]),
                "area_cm2": area,
                "bbox_xyz": row["bbox_l0_xyz"],
                "files": files,
            }
        )
    core = {
        "schema": "campaignx.surface_qc_backfill_manifest.v1",
        "generated_at_utc": utc_now(),
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "surface_root": str(surface_root.resolve()),
        "artifact_uri_root": artifact_uri_root,
        "surface_count": len(surfaces),
        "gross_area_cm2": total_area,
        "surfaces": surfaces,
        "ink_used": False,
        "no_automatic_acceptance": True,
        "semantics": [
            "every row binds the four immutable TIFXYZ files by hash and size",
            "gross area is not deduplicated physical sheet coverage",
            "enqueuing a surface does not validate geometry, ink, text or letters",
        ],
    }
    core["manifest_sha256"] = content_sha256(core)
    write_json_atomic(output, core)
    return core


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--catalog", type=Path, required=True)
    value.add_argument("--surface-root", type=Path, required=True)
    value.add_argument("--artifact-uri-root", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--write-artifact-manifests", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    result = prepare(
        catalog_path=args.catalog.resolve(),
        surface_root=args.surface_root.resolve(),
        artifact_uri_root=args.artifact_uri_root,
        output=args.output.resolve(),
        write_artifact_manifests=args.write_artifact_manifests,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "surface_count": result["surface_count"],
                "gross_area_cm2": result["gross_area_cm2"],
                "manifest_sha256": result["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
