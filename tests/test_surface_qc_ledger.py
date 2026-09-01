from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/helena_build_surface_qc_ledger.py"


def load_module():
    spec = importlib.util.spec_from_file_location("helena_build_surface_qc_ledger", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_next_step_reports_each_pipeline_boundary() -> None:
    module = load_module()
    blank = {
        "render_complete": False,
        "inference_complete": False,
        "stability_complete": False,
        "ct_gate_complete": False,
    }
    # The ink/CT boundaries below are only reachable once the geometry axis has
    # a verdict; certified is the neutral value for those cases.
    certified = "GEOMETRY_CERTIFIED"
    assert module.next_required_step(
        surface_present=False, qc_state=None, outcome=None, stages=blank
    ) == "IMPORT_AND_ENQUEUE"
    assert module.next_required_step(
        surface_present=True,
        qc_state=None,
        outcome=None,
        stages=blank,
        geometry_qc_state=certified,
    ) == "ENQUEUE_QC"
    assert module.next_required_step(
        surface_present=True,
        qc_state="PENDING",
        outcome=None,
        stages=blank,
        geometry_qc_state=certified,
    ) == "RENDER_CT"
    rendered = {**blank, "render_complete": True}
    assert module.next_required_step(
        surface_present=True,
        qc_state="CLAIMED",
        outcome=None,
        stages=rendered,
        geometry_qc_state=certified,
    ) == "RUN_SIX_REPLICA_INFERENCE"
    inferred = {**rendered, "inference_complete": True}
    assert module.next_required_step(
        surface_present=True,
        qc_state="CLAIMED",
        outcome=None,
        stages=inferred,
        geometry_qc_state=certified,
    ) == "ANALYZE_REPLICA_STABILITY"
    analyzed = {**inferred, "stability_complete": True}
    assert module.next_required_step(
        surface_present=True,
        qc_state="CLAIMED",
        outcome=None,
        stages=analyzed,
        geometry_qc_state=certified,
    ) == "RUN_CT_FIBER_GATE"
    gated = {**analyzed, "ct_gate_complete": True}
    assert module.next_required_step(
        surface_present=True,
        qc_state="CLAIMED",
        outcome=None,
        stages=gated,
        geometry_qc_state=certified,
    ) == "FINALIZE_QC_RECEIPT"
    assert module.next_required_step(
        surface_present=True,
        qc_state="COMPLETED",
        outcome="CT_SUPPORTED_RETAINED_FOR_REVIEW",
        stages=gated,
        geometry_qc_state=certified,
    ) == "HUMAN_VISUAL_INTERPRETATION"
    assert module.next_required_step(
        surface_present=True,
        qc_state="COMPLETED",
        outcome="CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
        stages=gated,
        geometry_qc_state=certified,
    ) == "RESOLVE_CT_INSUFFICIENT"
    assert module.next_required_step(
        surface_present=True,
        qc_state="COMPLETED",
        outcome="CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
        stages=gated,
        geometry_qc_state=certified,
    ) == "NONE_AUTOMATED_SCREEN_COMPLETE"
    assert module.next_required_step(
        surface_present=True,
        qc_state="COMPLETED",
        outcome="INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY",
        stages=inferred,
        geometry_qc_state=certified,
    ) == "SELECT_DIFFERENT_SURFACE_SCREEN_INSUFFICIENT"


def test_next_step_puts_the_geometry_axis_before_the_ink_screen() -> None:
    """A completed ink screen on unmeasured geometry is not "nothing left to do"."""

    module = load_module()
    gated = {
        "render_complete": True,
        "inference_complete": True,
        "stability_complete": True,
        "ct_gate_complete": True,
    }
    assert module.next_required_step(
        surface_present=True,
        qc_state="COMPLETED",
        outcome="CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
        stages=gated,
        geometry_qc_state=None,
    ) == "CERTIFY_GEOMETRY"
    assert module.next_required_step(
        surface_present=True,
        qc_state="COMPLETED",
        outcome="CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
        stages=gated,
        geometry_qc_state="GEOMETRY_UNMEASURED",
    ) == "CERTIFY_GEOMETRY"
    for rejected in (
        "GEOMETRY_REJECTED_BRIDGE",
        "GEOMETRY_REJECTED_LAMINA_SWITCH",
        "GEOMETRY_REJECTED_DISTORTION",
        "GEOMETRY_REJECTED_COVERAGE",
    ):
        assert module.next_required_step(
            surface_present=True,
            qc_state="COMPLETED",
            outcome="CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
            stages=gated,
            geometry_qc_state=rejected,
        ) == "RESEGMENT_GEOMETRY_REJECTED"
    assert module.next_required_step(
        surface_present=True,
        qc_state="FAILED",
        outcome=None,
        stages=gated,
        geometry_qc_state="GEOMETRY_CERTIFIED",
    ) == "RESOLVE_FAILED_QC_JOB"


def test_stage_evidence_parses_and_binds_the_screening_receipt(
    tmp_path: Path,
) -> None:
    module = load_module()
    attempt = tmp_path / "attempt"
    scientific = attempt / "scientific-output"
    receipt = scientific / "robust" / "INK_SCREENING_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "liveness": {
                    "verdict": "DEGENERATE",
                    "reason": "std 0.0001 < 0.02",
                }
            }
        ),
        encoding="utf-8",
    )
    (scientific / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "robust/INK_SCREENING_RECEIPT.json",
                        "size_bytes": receipt.stat().st_size,
                        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    stages = module.stage_evidence(attempt)

    assert stages["screening_liveness_verdict"] == "DEGENERATE"
    assert stages["screening_liveness_reason"] == "std 0.0001 < 0.02"
    assert stages["screening_receipt_manifest_bound"] is True


@pytest.mark.parametrize("mismatch", ["omitted", "path", "size", "sha256"])
def test_stage_evidence_rejects_an_unbound_screening_receipt(
    tmp_path: Path,
    mismatch: str,
) -> None:
    module = load_module()
    attempt = tmp_path / "attempt"
    scientific = attempt / "scientific-output"
    receipt = scientific / "robust" / "INK_SCREENING_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "liveness": {
                    "verdict": "EMPTY",
                    "reason": "aggregate has no valid pixels",
                }
            }
        ),
        encoding="utf-8",
    )
    row = {
        "path": "robust/INK_SCREENING_RECEIPT.json",
        "size_bytes": receipt.stat().st_size,
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
    }
    if mismatch == "path":
        row["path"] = "other/INK_SCREENING_RECEIPT.json"
    elif mismatch == "size":
        row["size_bytes"] += 1
    elif mismatch == "sha256":
        row["sha256"] = "0" * 64
    files = [] if mismatch == "omitted" else [row]
    (scientific / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps({"files": files}),
        encoding="utf-8",
    )

    stages = module.stage_evidence(attempt)

    assert stages["screening_receipt_manifest_bound"] is False


def test_stage_evidence_does_not_invent_liveness_for_a_malformed_receipt(
    tmp_path: Path,
) -> None:
    module = load_module()
    attempt = tmp_path / "attempt"
    scientific = attempt / "scientific-output"
    receipt = scientific / "robust" / "INK_SCREENING_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{malformed", encoding="utf-8")
    (scientific / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "robust/INK_SCREENING_RECEIPT.json",
                        "size_bytes": receipt.stat().st_size,
                        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    stages = module.stage_evidence(attempt)

    assert stages["screening_liveness_verdict"] is None
    assert stages["screening_liveness_reason"] is None


def test_stage_evidence_marks_a_malformed_manifest_unbound(
    tmp_path: Path,
) -> None:
    module = load_module()
    attempt = tmp_path / "attempt"
    scientific = attempt / "scientific-output"
    receipt = scientific / "robust" / "INK_SCREENING_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "liveness": {
                    "verdict": "EMPTY",
                    "reason": "aggregate has no valid pixels",
                }
            }
        ),
        encoding="utf-8",
    )
    (scientific / "EVIDENCE_MANIFEST.json").write_text(
        "{malformed",
        encoding="utf-8",
    )

    stages = module.stage_evidence(attempt)

    assert stages["screening_liveness_verdict"] == "EMPTY"
    assert stages["screening_liveness_reason"] == "aggregate has no valid pixels"
    assert stages["screening_receipt_manifest_bound"] is False


def test_build_reports_backfill_completion_and_evidence_integrity(tmp_path: Path) -> None:
    module = load_module()
    database = tmp_path / "fleet.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE surfaces (
          surface_id TEXT PRIMARY KEY,
          source_snapshot_id TEXT,
          sample_id TEXT,
          owner TEXT,
          artifact_sha256 TEXT,
          artifact_uri TEXT,
          bbox_xyz_json TEXT,
          sample_points_json TEXT,
          area_cm2 REAL,
          state TEXT,
          physical_qc_state TEXT,
          geometry_qc_state TEXT,
          payload_json TEXT,
          created_at TEXT
        );
        CREATE TABLE qc_jobs (
          qc_job_id TEXT PRIMARY KEY,
          surface_id TEXT,
          profile_id TEXT,
          state TEXT,
          payload_json TEXT,
          created_at TEXT,
          worker_id TEXT,
          lease_token TEXT,
          lease_expires_at TEXT,
          retry_after TEXT,
          result_json TEXT,
          updated_at TEXT
        );
        """
    )
    surface_id = "campaign-x:PHercTEST:surface-01"
    job_id = "qc-job-01"
    run_root = tmp_path / "qc-runtime"
    output_root = run_root / job_id / "20260722T000000Z-test" / "scientific-output"
    tiffs = output_root / "tiffs"
    tiffs.mkdir(parents=True)
    for index in range(65):
        (tiffs / f"{index:02d}.tif").write_bytes(b"fixture")
    inference = output_root / "robust" / "INK_SCREENING_RECEIPT.json"
    stability = output_root / "robust" / "analysis" / "INK_STABILITY_ANALYSIS.json"
    gate = output_root / "high-recall" / "gate" / "CT_FIBER_GATE_EVALUATION.json"
    for path, value in (
        (inference, {}),
        (stability, {}),
        (gate, {"retained_count": 0}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    evidence = output_root / "EVIDENCE_MANIFEST.json"
    evidence.write_text('{"schema":"fixture"}\n', encoding="utf-8")
    evidence_sha256 = module.file_sha256(evidence)
    result = {
        "schema": "campaignx.segment_qc_result.v1",
        "surface_id": surface_id,
        "outcome": "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
        "evidence_manifest_sha256": evidence_sha256,
        "evidence_uri": "s3://fixture/evidence.json",
        "ink_used": True,
        "executor_receipt": {
            "code_commit": "a" * 40,
            "retained_for_visual_review_count": 0,
        },
    }
    connection.execute(
        "INSERT INTO surfaces VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            surface_id,
            "source-01",
            "PHercTEST",
            "campaign-x",
            "b" * 64,
            "/fixture/surface",
            "[[0,0,0],[1,1,1]]",
            None,
            1.5,
            "QC_COMPLETE",
            "CT_SUPPORTED",
            "GEOMETRY_CERTIFIED",
            "{}",
            "2026-07-22T00:00:00Z",
        ),
    )
    connection.execute(
        "INSERT INTO qc_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            job_id,
            surface_id,
            "geometry-screen-v1",
            "COMPLETED",
            "{}",
            "2026-07-22T00:00:00Z",
            None,
            None,
            None,
            None,
            json.dumps(result),
            "2026-07-22T00:10:00Z",
        ),
    )
    connection.commit()
    connection.close()

    backfill = tmp_path / "backfill.json"
    backfill.write_text(
        json.dumps(
            {
                "schema": "campaignx.surface_qc_backfill_manifest.v1",
                "surfaces": [
                    {
                        "surface_id": surface_id,
                        "sample_id": "PHercTEST",
                        "artifact_sha256": "b" * 64,
                        "artifact_uri": "/fixture/surface",
                        "area_cm2": 1.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.json"
    ledger = module.build(
        database=database,
        backfill_manifest=backfill,
        qc_run_root=run_root,
        profile_id="geometry-screen-v1",
        output=ledger_path,
    )

    assert ledger["counts"]["surfaces"] == 1
    assert ledger["counts"]["expected_backfill_surfaces"] == 1
    assert ledger["counts"]["imported_backfill_surfaces"] == 1
    assert ledger["counts"]["completed_backfill_surfaces"] == 1
    assert ledger["counts"]["evidence_integrity_mismatches"] == 0
    assert ledger["surfaces"][0]["stages"] == {
        "attempt_path": str(output_root.parent),
        "rendered_slice_count": 65,
        "slice_ordering": "NUMERIC_STEM_ASCENDING",
        "render_complete": True,
        "inference_complete": True,
        "stability_complete": True,
        "ct_gate_complete": True,
        "ct_gate_retained_count": 0,
        "evidence_manifest_complete": True,
        "evidence_manifest_sha256": evidence_sha256,
        "evidence_manifest_matches_database": True,
        "screening_liveness_verdict": None,
        "screening_liveness_reason": None,
        "screening_receipt_manifest_bound": False,
    }
    assert ledger["surfaces"][0]["next_required_step"] == (
        "NONE_AUTOMATED_SCREEN_COMPLETE"
    )
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["ledger_sha256"] == (
        ledger["ledger_sha256"]
    )
