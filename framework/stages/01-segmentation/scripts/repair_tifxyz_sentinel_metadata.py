#!/usr/bin/env python3
"""Repair fleet catalogue metadata that treated negative TIFXYZ as geometry.

The TIFXYZ artifacts themselves are immutable and are never rewritten.  This
utility recomputes their derived area, bounds and samples with the canonical
FINITE_AND_NONNEGATIVE_CT_L0 policy, records every before/after value, and
updates only the local SQLite catalogue rows when ``--apply`` is present.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


STAGE = Path(__file__).resolve().parents[1]
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256, file_sha256, utc_now, write_json_atomic
from fleet.dedup import find_duplicate_in_surfaces
from fleet.finalizer import inspect_tifxyz


POLICY = "FINITE_AND_NONNEGATIVE_CT_L0"
EVENT = "SURFACE_METADATA_REPAIRED_TIFXYZ_SENTINEL"


def _json(value: str | None) -> object:
    return json.loads(value) if value is not None else None


def build_repair(database: Path, run_root: Path) -> dict:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT surface_id,sample_id,artifact_sha256,area_cm2,bbox_xyz_json,
                  sample_points_json,payload_json,created_at
           FROM surfaces
           WHERE json_extract(payload_json,'$.task_id') IS NOT NULL
           ORDER BY created_at,surface_id"""
    ).fetchall()
    repaired: list[dict] = []
    corrected_known: list[dict] = []
    for row in rows:
        payload = dict(_json(row["payload_json"]) or {})
        attempt_dir = run_root / "attempts" / payload["task_id"] / payload["attempt_id"]
        surface_dir = attempt_dir / "surface"
        artifact_set_path = attempt_dir / "ARTIFACT_SET.json"
        if not artifact_set_path.is_file():
            raise RuntimeError(f"missing immutable artifact manifest: {artifact_set_path}")
        artifact_set = json.loads(artifact_set_path.read_text(encoding="utf-8"))
        for name, expected in artifact_set["files"].items():
            path = surface_dir / name
            if not path.is_file() or file_sha256(path) != expected["sha256"]:
                raise RuntimeError(f"immutable TIFXYZ hash mismatch: {path}")
        source = connection.execute(
            "SELECT payload_json FROM source_snapshots WHERE source_snapshot_id=?",
            (payload["source_snapshot_id"],),
        ).fetchone()
        if source is None:
            raise RuntimeError(f"missing source snapshot for {row['surface_id']}")
        source_payload = dict(_json(source["payload_json"]) or {})
        inspection = inspect_tifxyz(surface_dir, float(source_payload["voxel_size_um"]))
        duplicate_of, duplicate_diagnostics = find_duplicate_in_surfaces(
            corrected_known,
            str(row["artifact_sha256"]),
            inspection["sample_points"],
        )
        corrected_known.append({
            "surface_id": str(row["surface_id"]),
            "artifact_sha256": str(row["artifact_sha256"]),
            "sample_points": inspection["sample_points"],
        })
        repaired.append({
            "surface_id": str(row["surface_id"]),
            "sample_id": str(row["sample_id"]),
            "task_id": payload["task_id"],
            "attempt_id": payload["attempt_id"],
            "artifact_sha256": str(row["artifact_sha256"]),
            "before": {
                "area_cm2": float(row["area_cm2"]),
                "bbox_xyz": _json(row["bbox_xyz_json"]),
                "sample_points": _json(row["sample_points_json"]),
            },
            "after": inspection,
            "duplicate_reaudit": {
                "duplicate_of_prior_repaired_surface": duplicate_of,
                "diagnostics": duplicate_diagnostics,
            },
        })
    connection.close()
    core = {
        "schema": "campaignx.tifxyz_sentinel_metadata_repair.v1",
        "policy": POLICY,
        "database": str(database),
        "run_root": str(run_root),
        "surface_count": len(repaired),
        "surfaces": repaired,
        "artifact_mutation": False,
        "non_claim": "Corrected metadata and non-duplication do not validate physical geometry, ink, text or First Letters.",
    }
    return {**core, "repair_sha256": content_sha256(core)}


def apply_repair(database: Path, receipt: dict, receipt_path: Path) -> None:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("BEGIN IMMEDIATE")
    try:
        for row in receipt["surfaces"]:
            current = connection.execute(
                "SELECT payload_json FROM surfaces WHERE surface_id=?", (row["surface_id"],)
            ).fetchone()
            if current is None:
                raise RuntimeError(f"surface disappeared before repair: {row['surface_id']}")
            payload = dict(_json(current["payload_json"]) or {})
            after = row["after"]
            payload.update({
                "area_cm2": after["area_cm2"],
                "bbox_xyz": after["bbox_xyz"],
                "sample_points": after["sample_points"],
                "geometry_inspection_policy": POLICY,
                "metadata_repair": {
                    "schema": receipt["schema"],
                    "repair_sha256": receipt["repair_sha256"],
                    "receipt": str(receipt_path),
                },
            })
            connection.execute(
                """UPDATE surfaces
                   SET area_cm2=?,bbox_xyz_json=?,sample_points_json=?,payload_json=?
                   WHERE surface_id=?""",
                (
                    after["area_cm2"],
                    json.dumps(after["bbox_xyz"], sort_keys=True, separators=(",", ":")),
                    json.dumps(after["sample_points"], sort_keys=True, separators=(",", ":")),
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    row["surface_id"],
                ),
            )
            event_payload = {
                "surface_id": row["surface_id"],
                "policy": POLICY,
                "repair_sha256": receipt["repair_sha256"],
                "receipt": str(receipt_path),
                "before_area_cm2": row["before"]["area_cm2"],
                "after_area_cm2": after["area_cm2"],
            }
            connection.execute(
                """INSERT INTO events(task_id,attempt_id,event_type,payload_json,created_at)
                   SELECT ?,?,?,?,?
                   WHERE NOT EXISTS (
                     SELECT 1 FROM events
                     WHERE task_id=? AND attempt_id=? AND event_type=?
                       AND json_extract(payload_json,'$.repair_sha256')=?
                   )""",
                (
                    row["task_id"], row["attempt_id"], EVENT,
                    json.dumps(event_payload, sort_keys=True, separators=(",", ":")), utc_now(),
                    row["task_id"], row["attempt_id"], EVENT, receipt["repair_sha256"],
                ),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        receipt = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "campaignx.tifxyz_sentinel_metadata_repair.v1"
            or receipt.get("policy") != POLICY
            or Path(receipt.get("database", "")) != args.db.resolve()
            or Path(receipt.get("run_root", "")) != args.run_root.resolve()
        ):
            raise RuntimeError("repair output does not match this database and run root")
        if args.apply and receipt.get("status") != "APPLIED":
            apply_repair(args.db.resolve(), receipt, args.output.resolve())
            receipt["status"] = "APPLIED"
            receipt["applied_at_utc"] = utc_now()
            write_json_atomic(args.output, receipt)
        print(json.dumps({"status": "ALREADY_RECORDED", "surface_count": receipt["surface_count"]}, sort_keys=True))
        return 0
    receipt = build_repair(args.db.resolve(), args.run_root.resolve())
    receipt = {
        **receipt,
        "status": "PREPARED" if args.apply else "DRY_RUN",
        "generated_at_utc": utc_now(),
    }
    write_json_atomic(args.output, receipt)
    if args.apply:
        apply_repair(args.db.resolve(), receipt, args.output.resolve())
        receipt["status"] = "APPLIED"
        receipt["applied_at_utc"] = utc_now()
        write_json_atomic(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "surface_count": receipt["surface_count"], "repair_sha256": receipt["repair_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
