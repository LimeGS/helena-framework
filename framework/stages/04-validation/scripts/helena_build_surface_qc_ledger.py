#!/usr/bin/env python3
"""Build an auditable, per-surface ledger for the complete QC pipeline.

The fleet database is authoritative for queue state and scientific outcomes.
The optional backfill manifest proves which historical artifacts were verified
before import.  Regenerable run directories are inspected only to expose the
current operational stage; they never override a committed database outcome.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "framework/stages/01-segmentation").is_dir()
)
STAGE01 = ROOT / "framework/stages/01-segmentation"
for candidate in (ROOT, STAGE01):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fleet.common import content_sha256, file_sha256, read_json, utc_now, write_json_atomic
from framework.contracts.slice_order import ordered_tiff_files


def read_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None


def read_stage_json(path: Path | None) -> dict[str, Any] | None:
    try:
        return read_optional_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def manifest_binds_local_file(
    root: Path | None,
    path: Path | None,
    manifest: dict[str, Any] | None,
) -> bool:
    if root is None or path is None or manifest is None:
        return False
    rows = manifest.get("files")
    if not isinstance(rows, list):
        return False
    try:
        relative_path = str(path.relative_to(root))
        local_size = path.stat().st_size
        local_sha256 = file_sha256(path)
    except (OSError, ValueError):
        return False
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("path") == relative_path
    ]
    if len(matches) != 1:
        return False
    row = matches[0]
    return (
        type(row.get("size_bytes")) is int
        and row["size_bytes"] == local_size
        and row.get("sha256") == local_sha256
    )


def latest_attempt(qc_run_root: Path | None, qc_job_id: str | None) -> Path | None:
    if qc_run_root is None or not qc_job_id:
        return None
    root = qc_run_root / qc_job_id
    if not root.is_dir():
        return None
    attempts = sorted(path for path in root.iterdir() if path.is_dir())
    return attempts[-1] if attempts else None


def first_match(root: Path | None, name: str) -> Path | None:
    if root is None:
        return None
    return next(iter(sorted(root.rglob(name))), None)


def stage_evidence(
    attempt: Path | None,
    *,
    expected_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    scientific = attempt / "scientific-output" if attempt else None
    tiff_root = scientific / "tiffs" if scientific else None
    if tiff_root and tiff_root.is_dir():
        rendered, slice_ordering = ordered_tiff_files(tiff_root, allow_empty=True)
    else:
        rendered, slice_ordering = [], None
    inference = first_match(scientific, "INK_SCREENING_RECEIPT.json")
    stability = first_match(scientific, "INK_STABILITY_ANALYSIS.json")
    gate = first_match(scientific, "CT_FIBER_GATE_EVALUATION.json")
    evidence = first_match(scientific, "EVIDENCE_MANIFEST.json")
    screening_payload = read_stage_json(inference)
    liveness = (
        screening_payload.get("liveness")
        if isinstance(screening_payload, dict)
        else None
    )
    liveness_verdict = liveness.get("verdict") if isinstance(liveness, dict) else None
    liveness_reason = liveness.get("reason") if isinstance(liveness, dict) else None
    evidence_payload = read_stage_json(evidence)
    retained = None
    if gate:
        gate_value = read_optional_json(gate)
        retained = gate_value.get("retained_count") if gate_value else None
    evidence_sha256 = file_sha256(evidence) if evidence else None
    return {
        "attempt_path": str(attempt) if attempt else None,
        "rendered_slice_count": len(rendered),
        "slice_ordering": slice_ordering,
        "render_complete": len(rendered) == 65,
        "inference_complete": inference is not None,
        "stability_complete": stability is not None,
        "ct_gate_complete": gate is not None,
        "ct_gate_retained_count": retained,
        "evidence_manifest_complete": evidence is not None,
        "evidence_manifest_sha256": evidence_sha256,
        "evidence_manifest_matches_database": (
            evidence_sha256 == expected_evidence_sha256
            if evidence_sha256 and expected_evidence_sha256
            else None
        ),
        "screening_liveness_verdict": liveness_verdict,
        "screening_liveness_reason": liveness_reason,
        "screening_receipt_manifest_bound": manifest_binds_local_file(
            scientific,
            inference,
            evidence_payload,
        ),
    }


GEOMETRY_REJECTED_PREFIX = "GEOMETRY_REJECTED_"
GEOMETRY_UNMEASURED = "GEOMETRY_UNMEASURED"


def next_required_step(
    *,
    surface_present: bool,
    qc_state: str | None,
    outcome: str | None,
    stages: dict[str, Any],
    geometry_qc_state: str | None = None,
) -> str:
    """Name the one action a surface is actually waiting on.

    The geometry axis is evaluated first and independently of the ink/CT
    outcome.  Before this existed, a surface whose geometry had never been
    measured reported ``NONE_AUTOMATED_SCREEN_COMPLETE`` -- "nothing left to
    do" -- which is exactly the claim the campaign could not support: the ink
    screen ran, but no gate had ever looked at the geometry it ran on.
    """

    if not surface_present:
        return "IMPORT_AND_ENQUEUE"
    geometry = str(geometry_qc_state or GEOMETRY_UNMEASURED)
    if geometry.startswith(GEOMETRY_REJECTED_PREFIX):
        return "RESEGMENT_GEOMETRY_REJECTED"
    if geometry == GEOMETRY_UNMEASURED:
        return "CERTIFY_GEOMETRY"
    if qc_state is None:
        return "ENQUEUE_QC"
    if qc_state == "FAILED":
        return "RESOLVE_FAILED_QC_JOB"
    if qc_state == "COMPLETED":
        if outcome == "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY":
            return "SELECT_DIFFERENT_SURFACE_SCREEN_INSUFFICIENT"
        if outcome == "CT_SUPPORTED_RETAINED_FOR_REVIEW":
            return "HUMAN_VISUAL_INTERPRETATION"
        if outcome == "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS":
            return "RESOLVE_CT_INSUFFICIENT"
        return "NONE_AUTOMATED_SCREEN_COMPLETE"
    if not stages["render_complete"]:
        return "RENDER_CT"
    if not stages["inference_complete"]:
        return "RUN_SIX_REPLICA_INFERENCE"
    if not stages["stability_complete"]:
        return "ANALYZE_REPLICA_STABILITY"
    if not stages["ct_gate_complete"]:
        return "RUN_CT_FIBER_GATE"
    return "FINALIZE_QC_RECEIPT"


def database_rows(database: Path, profile_id: str) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT s.*,j.qc_job_id,j.profile_id,j.state AS qc_state,
                      j.result_json,j.retry_after,j.updated_at AS qc_updated_at
               FROM surfaces s
               LEFT JOIN qc_jobs j
                 ON j.surface_id=s.surface_id AND j.profile_id=?
               ORDER BY s.sample_id,s.surface_id""",
            (profile_id,),
        ).fetchall()
        return {str(row["surface_id"]): dict(row) for row in rows}
    finally:
        connection.close()


def build(
    *,
    database: Path,
    backfill_manifest: Path | None,
    qc_run_root: Path | None,
    profile_id: str,
    output: Path,
) -> dict[str, Any]:
    db_rows = database_rows(database, profile_id)
    backfill = read_optional_json(backfill_manifest)
    expected = {
        str(row["surface_id"]): row
        for row in (backfill or {}).get("surfaces", [])
        if isinstance(row, dict) and row.get("surface_id")
    }
    surfaces: list[dict[str, Any]] = []
    for surface_id in sorted(set(db_rows) | set(expected)):
        database_row = db_rows.get(surface_id)
        manifest_row = expected.get(surface_id)
        result = (
            json.loads(database_row["result_json"])
            if database_row and database_row.get("result_json")
            else None
        )
        executor = result.get("executor_receipt", {}) if isinstance(result, dict) else {}
        outcome = result.get("outcome") if isinstance(result, dict) else None
        qc_job_id = database_row.get("qc_job_id") if database_row else None
        attempt = latest_attempt(qc_run_root, qc_job_id)
        expected_evidence_sha256 = (
            result.get("evidence_manifest_sha256")
            if isinstance(result, dict)
            else None
        )
        stages = stage_evidence(
            attempt,
            expected_evidence_sha256=expected_evidence_sha256,
        )
        qc_state = database_row.get("qc_state") if database_row else None
        item = {
            "surface_id": surface_id,
            "sample_id": (
                database_row.get("sample_id") if database_row else manifest_row.get("sample_id")
            ),
            "origin": "HISTORICAL_BACKFILL" if manifest_row else "FLEET_NATIVE",
            "artifact_contract_state": (
                "VERIFIED_BACKFILL_MANIFEST"
                if manifest_row
                else "VERIFIED_FLEET_FINALIZER"
            ),
            "artifact_sha256": (
                database_row.get("artifact_sha256")
                if database_row
                else manifest_row.get("artifact_sha256")
            ),
            "artifact_uri": (
                database_row.get("artifact_uri")
                if database_row
                else manifest_row.get("artifact_uri")
            ),
            "area_cm2": (
                database_row.get("area_cm2") if database_row else manifest_row.get("area_cm2")
            ),
            "surface_present_in_control_plane": database_row is not None,
            "qc_job_id": qc_job_id,
            "qc_profile_id": profile_id if qc_job_id else None,
            "qc_state": qc_state,
            "physical_qc_state": (
                database_row.get("physical_qc_state") if database_row else "UNVALIDATED"
            ),
            # Orthogonal to physical_qc_state; databases predating the geometry
            # gate have no column, and "no column" means unmeasured.
            "geometry_qc_state": (
                (database_row.get("geometry_qc_state") or GEOMETRY_UNMEASURED)
                if database_row
                else GEOMETRY_UNMEASURED
            ),
            "outcome": outcome,
            "executor_code_commit": executor.get("code_commit"),
            "evidence_uri": result.get("evidence_uri") if isinstance(result, dict) else None,
            "evidence_manifest_sha256": expected_evidence_sha256,
            "retained_for_visual_review_count": executor.get(
                "retained_for_visual_review_count"
            ),
            "retry_after": database_row.get("retry_after") if database_row else None,
            "qc_updated_at": (
                database_row.get("qc_updated_at") if database_row else None
            ),
            "stages": stages,
        }
        item["next_required_step"] = next_required_step(
            surface_present=database_row is not None,
            qc_state=qc_state,
            outcome=outcome,
            stages=stages,
            geometry_qc_state=item["geometry_qc_state"],
        )
        surfaces.append(item)

    by_next = Counter(row["next_required_step"] for row in surfaces)
    by_geometry = Counter(row["geometry_qc_state"] for row in surfaces)
    by_qc_state = Counter(row["qc_state"] or "NO_JOB" for row in surfaces)
    by_outcome = Counter(row["outcome"] or "NO_OUTCOME" for row in surfaces)
    by_sample = Counter(str(row["sample_id"]) for row in surfaces)
    expected_backfill = [row for row in surfaces if row["origin"] == "HISTORICAL_BACKFILL"]
    fleet_native = [row for row in surfaces if row["origin"] == "FLEET_NATIVE"]
    completed = [row for row in surfaces if row["qc_state"] == "COMPLETED"]
    integrity_mismatches = [
        row
        for row in completed
        if row["stages"]["evidence_manifest_matches_database"] is False
    ]
    database_files = {}
    for suffix in ("", "-wal", "-shm"):
        path = Path(f"{database}{suffix}")
        if path.is_file():
            database_files[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    core = {
        "schema": "campaignx.surface_qc_ledger.v1",
        "generated_at_utc": utc_now(),
        "database": str(database),
        "database_files_at_ledger_write": database_files,
        "profile_id": profile_id,
        "backfill_manifest": str(backfill_manifest) if backfill_manifest else None,
        "backfill_manifest_sha256": (
            file_sha256(backfill_manifest) if backfill_manifest and backfill_manifest.is_file() else None
        ),
        "qc_run_root": str(qc_run_root) if qc_run_root else None,
        "counts": {
            "surfaces": len(surfaces),
            "expected_backfill_surfaces": len(expected_backfill),
            "imported_backfill_surfaces": sum(
                row["surface_present_in_control_plane"] for row in expected_backfill
            ),
            "completed_backfill_surfaces": sum(
                row["qc_state"] == "COMPLETED" for row in expected_backfill
            ),
            "fleet_native_surfaces": len(fleet_native),
            "completed_surfaces": len(completed),
            "evidence_integrity_mismatches": len(integrity_mismatches),
            "by_next_required_step": dict(sorted(by_next.items())),
            "by_geometry_qc_state": dict(sorted(by_geometry.items())),
            "by_qc_state": dict(sorted(by_qc_state.items())),
            "by_outcome": dict(sorted(by_outcome.items())),
            "by_sample": dict(sorted(by_sample.items())),
        },
        "surfaces": surfaces,
        "semantics": [
            "a completed automated screen is not proof of no ink or no letters",
            "a retained signal requires visual interpretation and is not accepted ink",
            "run-directory stage detection is operational evidence; database outcomes are authoritative",
            "gross surface area is not deduplicated physical sheet coverage",
            "geometry_qc_state is orthogonal to physical_qc_state; an unmeasured geometry is not a certified one",
        ],
        "no_automatic_acceptance": True,
    }
    core["ledger_sha256"] = content_sha256(core)
    write_json_atomic(output, core)
    return core


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--db", type=Path, required=True)
    value.add_argument("--backfill-manifest", type=Path)
    value.add_argument("--qc-run-root", type=Path)
    value.add_argument("--profile-id", default="geometry-screen-v1")
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    result = build(
        database=args.db.resolve(),
        backfill_manifest=args.backfill_manifest.resolve() if args.backfill_manifest else None,
        qc_run_root=args.qc_run_root.resolve() if args.qc_run_root else None,
        profile_id=args.profile_id,
        output=args.output.resolve(),
    )
    print(json.dumps({"output": str(args.output), **result["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
