from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/helena_verify_surface_qc_ledger.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "helena_verify_surface_qc_ledger", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_surface() -> dict:
    digest = "a" * 64
    return {
        "surface_id": "campaign-x:PHercTEST:surface-01",
        "origin": "HISTORICAL_BACKFILL",
        "qc_state": "COMPLETED",
        "outcome": "CT_SUPPORTED_RETAINED_FOR_REVIEW",
        "evidence_manifest_sha256": digest,
        "evidence_uri": "s3://fixture/evidence.json",
        "stages": {
            "render_complete": True,
            "rendered_slice_count": 65,
            "inference_complete": True,
            "stability_complete": True,
            "ct_gate_complete": True,
            "evidence_manifest_complete": True,
            "evidence_manifest_matches_database": True,
        },
    }


def seal(module, ledger: dict) -> dict:
    ledger["ledger_sha256"] = module.content_sha256(ledger)
    return ledger


def test_verify_accepts_complete_hash_bound_backfill() -> None:
    module = load_module()
    ledger = seal(
        module,
        {
            "schema": "campaignx.surface_qc_ledger.v1",
            "surfaces": [completed_surface()],
            "no_automatic_acceptance": True,
        },
    )
    receipt = module.verify(
        ledger,
        expected_backfill_surfaces=1,
        require_all_ledger_surfaces_complete=True,
        verify_remote=True,
        head_reader=lambda _: {
            "size_bytes": 123,
            "sha256": "a" * 64,
            "etag": "fixture",
        },
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["counts"] == {
        "ledger_surfaces": 1,
        "backfill_surfaces": 1,
        "required_surfaces": 1,
        "completed_required_surfaces": 1,
        "remote_evidence_verified": 1,
        "failures": 0,
    }
    assert receipt["no_automatic_acceptance"] is True


def test_verify_fails_closed_for_incomplete_surface() -> None:
    module = load_module()
    surface = completed_surface()
    surface["qc_state"] = "PENDING"
    ledger = seal(
        module,
        {
            "schema": "campaignx.surface_qc_ledger.v1",
            "surfaces": [surface],
            "no_automatic_acceptance": True,
        },
    )
    receipt = module.verify(
        ledger,
        expected_backfill_surfaces=1,
        require_all_ledger_surfaces_complete=True,
        verify_remote=True,
        head_reader=lambda _: {
            "size_bytes": 123,
            "sha256": "b" * 64,
            "etag": "fixture",
        },
    )

    assert receipt["status"] == "FAILED"
    assert receipt["counts"]["completed_required_surfaces"] == 0
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "QC_NOT_COMPLETED",
            "state": "PENDING",
        }
    ]


def test_verify_fails_closed_for_mismatched_remote_evidence() -> None:
    module = load_module()
    ledger = seal(
        module,
        {
            "schema": "campaignx.surface_qc_ledger.v1",
            "surfaces": [completed_surface()],
            "no_automatic_acceptance": True,
        },
    )
    receipt = module.verify(
        ledger,
        expected_backfill_surfaces=1,
        require_all_ledger_surfaces_complete=False,
        verify_remote=True,
        head_reader=lambda _: {
            "size_bytes": 123,
            "sha256": "b" * 64,
            "etag": "fixture",
        },
    )

    assert receipt["status"] == "FAILED"
    assert receipt["failures"][0]["reason"] == "REMOTE_EVIDENCE_HASH_MISMATCH"
