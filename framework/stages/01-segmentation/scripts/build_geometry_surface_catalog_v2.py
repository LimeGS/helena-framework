#!/usr/bin/env python3
"""Build the current archive-first Helena Framework TIFXYZ catalogue.

V1 reconstructed coverage from a historical subset of execution summaries.
That was useful for recovery work, but it could omit a surface that had been
preserved by a newer run.  V2 treats the durable archive as authoritative:
every complete ``<sample>/<surface>/{x,y,z}.tif + meta.json`` directory is one
and only one catalogue row.  Receipts are attached as provenance when found;
they never determine whether an archived surface exists.

The catalogue records candidate geometry, not validated physical sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REQUIRED = ("x.tif", "y.tif", "z.tif", "meta.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_bbox(value: Any, *, source: Path) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"missing two-corner bbox: {source}")
    if any(not isinstance(point, list) or len(point) != 3 for point in value):
        raise RuntimeError(f"bbox must be 2x3 XYZ: {source}")
    try:
        low, high = [[float(coordinate) for coordinate in point] for point in value]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"bbox contains a non-number: {source}") from error
    if any(lo > hi for lo, hi in zip(low, high, strict=True)):
        raise RuntimeError(f"bbox lower corner exceeds upper corner: {source}")
    return [low, high]


def overlaps(a: list[list[float]], b: list[list[float]]) -> bool:
    return all(a[0][axis] <= b[1][axis] and b[0][axis] <= a[1][axis] for axis in range(3))


def receipt_identity(receipt: dict[str, Any]) -> tuple[str, str] | None:
    sample_id, seed_id = receipt.get("sample_id"), receipt.get("seed_id")
    if not isinstance(sample_id, str) or not isinstance(seed_id, str):
        return None
    return sample_id, seed_id


def discover_receipts(root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Attach individual and embedded receipts without double counting bytes."""
    found: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    def add(receipt: dict[str, Any], source: Path, position: int | None) -> None:
        kind = str(receipt.get("kind", ""))
        structural_growth_fields = ("profile", "area_cm2", "seed_coordinate_xyz_l0", "files")
        if "growth_receipt" not in kind and not all(field in receipt for field in structural_growth_fields):
            # Screen and validation summaries also carry sample_id/seed_id,
            # but they are not provenance for how the TIFXYZ was grown.
            return
        identity = receipt_identity(receipt)
        if identity is None:
            return
        digest = canonical_hash(receipt)
        relative = str(source.relative_to(root)) if source.is_relative_to(root) else str(source)
        record = found[identity].setdefault(
            digest,
            {
                "sha256": digest,
                "status": receipt.get("status"),
                "area_cm2": receipt.get("area_cm2"),
                "seed_coordinate_xyz_l0": receipt.get("seed_coordinate_xyz_l0"),
                "profile": receipt.get("profile"),
                "sources": [],
            },
        )
        source_record = {"path": relative, "position": position}
        if source_record not in record["sources"]:
            record["sources"].append(source_record)

    for path in sorted(root.rglob("GROWTH_RECEIPT.json"), key=str):
        try:
            add(load(path), path, None)
        except (OSError, json.JSONDecodeError, RuntimeError):
            continue

    patterns = ("GEOMETRY_RECOVERY*_EXECUTION*.json", "GEOMETRY_RECOVERY_V1_EXECUTION*.json")
    paths = {path for pattern in patterns for path in root.rglob(pattern)}
    for path in sorted(paths, key=str):
        try:
            summary = load(path)
        except (OSError, json.JSONDecodeError, RuntimeError):
            continue
        receipts = summary.get("receipts")
        if not isinstance(receipts, list):
            continue
        for position, receipt in enumerate(receipts):
            if isinstance(receipt, dict):
                add(receipt, path, position)

    output: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for identity, by_hash in found.items():
        records = list(by_hash.values())
        for record in records:
            record["sources"].sort(key=lambda row: (row["path"], -1 if row["position"] is None else row["position"]))
        output[identity] = sorted(records, key=lambda row: row["sha256"])
    return output


def scan_archive(
    archive_root: Path, receipts: dict[tuple[str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not archive_root.is_dir():
        raise RuntimeError(f"TIFXYZ archive is unavailable: {archive_root}")
    for directory in sorted((path for path in archive_root.glob("*/*") if path.is_dir()), key=str):
        missing = [name for name in REQUIRED if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(f"incomplete archived TIFXYZ {directory}: missing {missing}")
        sample_id, seed_id = directory.parts[-2:]
        meta = load(directory / "meta.json")
        bbox = normalized_bbox(meta.get("bbox"), source=directory / "meta.json")
        try:
            area = float(meta["area_cm2"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"missing numeric area_cm2: {directory / 'meta.json'}") from error
        if area <= 0:
            raise RuntimeError(f"non-positive area_cm2: {directory / 'meta.json'}")
        files = {
            name: {"sha256": sha256(directory / name), "size_bytes": (directory / name).stat().st_size}
            for name in REQUIRED
        }
        tifxyz_digest = canonical_hash({name: files[name]["sha256"] for name in REQUIRED})
        attached = receipts.get((sample_id, seed_id), [])
        rows.append(
            {
                "surface_id": f"campaign-x:{sample_id}:{seed_id}",
                "owner": "campaign-x",
                "sample_id": sample_id,
                "seed_id": seed_id,
                "archive_relative": str(directory.relative_to(archive_root)),
                "tifxyz_sha256": tifxyz_digest,
                "files": files,
                "area_cm2": area,
                "bbox_l0_xyz": bbox,
                "seed_xyz_l0": meta.get("seed"),
                "target_volume": meta.get("target_volume"),
                "profile": meta.get("vc_gsfs_params"),
                "metadata_uuid": meta.get("uuid"),
                "growth_receipts": attached,
                "growth_receipt_count": len(attached),
                "state": "PRESERVED_TIFXYZ_CANDIDATE",
                "physical_qc_state": "UNVALIDATED",
                "non_claim": "Archived TIFXYZ is candidate geometry, not a validated sheet, ink, text, or First Letters evidence.",
            }
        )
    return rows


def public_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = load(path)
    rows = payload.get("surfaces")
    if not isinstance(rows, list) or payload.get("downloaded_measured_count") != len(rows):
        raise RuntimeError("public inventory must be complete and measured")
    for row in rows:
        if row.get("archive_state") != "DOWNLOADED_AND_MEASURED" or not row.get("bbox_l0_xyz"):
            raise RuntimeError("public inventory contains an unmeasured surface")
    return rows


def overlap_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    pairs: list[tuple[str, str]] = []
    for candidates in grouped.values():
        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                if overlaps(first["bbox_l0_xyz"], second["bbox_l0_xyz"]):
                    pairs.append((first["surface_id"], second["surface_id"]))
    return pairs


def campaign_public_pairs(
    campaign: list[dict[str, Any]], public: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    return [
        (candidate["surface_id"], official["public_surface_id"])
        for candidate in campaign
        for official in public
        if candidate["sample_id"] == official["campaign_sample_id"]
        and overlaps(candidate["bbox_l0_xyz"], official["bbox_l0_xyz"])
    ]


def write_database(
    path: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    public: list[dict[str, Any]],
    internal_pairs: Iterable[tuple[str, str]],
    public_pairs: Iterable[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE campaign_surfaces (
              surface_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, seed_id TEXT NOT NULL,
              archive_relative TEXT NOT NULL, tifxyz_sha256 TEXT NOT NULL,
              area_cm2 REAL NOT NULL, bbox_l0_xyz_json TEXT NOT NULL,
              target_volume TEXT, profile_json TEXT, receipt_count INTEGER NOT NULL,
              physical_qc_state TEXT NOT NULL, row_json TEXT NOT NULL
            );
            CREATE INDEX campaign_surfaces_by_scroll ON campaign_surfaces(sample_id, seed_id);
            CREATE TABLE growth_receipts (
              surface_id TEXT NOT NULL, receipt_sha256 TEXT NOT NULL, status TEXT,
              receipt_json TEXT NOT NULL,
              PRIMARY KEY(surface_id, receipt_sha256),
              FOREIGN KEY(surface_id) REFERENCES campaign_surfaces(surface_id)
            );
            CREATE TABLE campaign_aabb_overlap_warnings (
              first_surface_id TEXT NOT NULL, second_surface_id TEXT NOT NULL,
              PRIMARY KEY(first_surface_id, second_surface_id)
            );
            CREATE TABLE public_surfaces (
              public_surface_id TEXT PRIMARY KEY, campaign_sample_id TEXT NOT NULL,
              segment_id TEXT NOT NULL, area_cm2 REAL NOT NULL, bbox_l0_xyz_json TEXT NOT NULL,
              row_json TEXT NOT NULL
            );
            CREATE INDEX public_surfaces_by_scroll ON public_surfaces(campaign_sample_id);
            CREATE TABLE campaign_public_aabb_overlap_warnings (
              campaign_surface_id TEXT NOT NULL, public_surface_id TEXT NOT NULL,
              PRIMARY KEY(campaign_surface_id, public_surface_id)
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            ((key, json.dumps(value, sort_keys=True)) for key, value in metadata.items()),
        )
        connection.executemany(
            "INSERT INTO campaign_surfaces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["surface_id"], row["sample_id"], row["seed_id"], row["archive_relative"],
                    row["tifxyz_sha256"], row["area_cm2"], json.dumps(row["bbox_l0_xyz"]),
                    row["target_volume"], json.dumps(row["profile"], sort_keys=True),
                    row["growth_receipt_count"], row["physical_qc_state"],
                    json.dumps(row, sort_keys=True),
                )
                for row in rows
            ],
        )
        connection.executemany(
            "INSERT INTO growth_receipts VALUES (?, ?, ?, ?)",
            [
                (row["surface_id"], receipt["sha256"], receipt["status"], json.dumps(receipt, sort_keys=True))
                for row in rows
                for receipt in row["growth_receipts"]
            ],
        )
        connection.executemany("INSERT INTO campaign_aabb_overlap_warnings VALUES (?, ?)", internal_pairs)
        connection.executemany(
            "INSERT INTO public_surfaces VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row["public_surface_id"], row["campaign_sample_id"], row["segment_id"],
                    float(row["measured_area_cm2"]), json.dumps(row["bbox_l0_xyz"]),
                    json.dumps(row, sort_keys=True),
                )
                for row in public
            ],
        )
        connection.executemany("INSERT INTO campaign_public_aabb_overlap_warnings VALUES (?, ?)", public_pairs)
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)


def build(
    root: Path,
    archive_root: Path,
    public_inventory: Path | None,
    database: Path,
    summary_path: Path,
) -> dict[str, Any]:
    root, archive_root = root.resolve(), archive_root.resolve()
    receipts = discover_receipts(root)
    rows = scan_archive(archive_root, receipts)
    public = public_rows(public_inventory.resolve() if public_inventory else None)
    internal = overlap_pairs(rows)
    against_public = campaign_public_pairs(rows, public)
    archive_snapshot = canonical_hash(
        [{"sample_id": row["sample_id"], "seed_id": row["seed_id"], "sha256": row["tifxyz_sha256"]} for row in rows]
    )
    per_scroll = []
    for sample_id in sorted({row["sample_id"] for row in rows} | {row["campaign_sample_id"] for row in public}):
        own = [row for row in rows if row["sample_id"] == sample_id]
        official = [row for row in public if row["campaign_sample_id"] == sample_id]
        per_scroll.append(
            {
                "sample_id": sample_id,
                "campaign_surface_count": len(own),
                "campaign_gross_area_cm2": sum(float(row["area_cm2"]) for row in own),
                "campaign_with_receipt_count": sum(bool(row["growth_receipts"]) for row in own),
                "public_surface_count": len(official),
                "public_gross_area_cm2": sum(float(row["measured_area_cm2"]) for row in official),
            }
        )
    metadata = {
        "kind": "campaign_x_geometry_surface_catalog_v2",
        "generated_at_utc": utc_now(),
        "archive_root": str(archive_root),
        "archive_snapshot_sha256": archive_snapshot,
        "semantics": "one row per complete TIFXYZ directory physically present in the durable archive",
        "policy": [
            "the archive, not a historical receipt glob, defines Helena Framework surface existence",
            "receipt association supplies provenance but never creates a missing TIFXYZ row",
            "gross area is not deduplicated physical sheet coverage",
            "AABB overlap is a duplicate-work warning, not proof of sheet identity",
            "no row is physical-QC validated, ink, text, or First Letters evidence",
        ],
    }
    summary = {
        **metadata,
        "database": str(database),
        "campaign_surface_count": len(rows),
        "campaign_gross_area_cm2": sum(float(row["area_cm2"]) for row in rows),
        "campaign_with_receipt_count": sum(bool(row["growth_receipts"]) for row in rows),
        "campaign_without_receipt_count": sum(not row["growth_receipts"] for row in rows),
        "campaign_aabb_overlap_warning_count": len(internal),
        "public_surface_count": len(public),
        "public_gross_area_cm2": sum(float(row["measured_area_cm2"]) for row in public),
        "campaign_public_aabb_overlap_warning_count": len(against_public),
        "per_scroll": per_scroll,
        "campaign_surfaces": rows,
        "campaign_aabb_overlap_warnings": [
            {"first_surface_id": first, "second_surface_id": second} for first, second in internal
        ],
        "campaign_public_aabb_overlap_warnings": [
            {"campaign_surface_id": first, "public_surface_id": second} for first, second in against_public
        ],
    }
    write_database(database, metadata, rows, public, internal, against_public)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--public-inventory", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.archive_root, args.public_inventory, args.database, args.summary)
    print(
        json.dumps(
            {key: result[key] for key in (
                "campaign_surface_count", "campaign_with_receipt_count", "campaign_without_receipt_count",
                "campaign_aabb_overlap_warning_count", "public_surface_count",
                "campaign_public_aabb_overlap_warning_count",
            )},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
