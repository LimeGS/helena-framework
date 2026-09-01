from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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


def completed_insufficient_surface(
    *,
    verdict: str = "DEGENERATE",
    reason: str = "std 0.0001 < 0.02",
    physical_qc_state: str = "INK_SCREEN_INSUFFICIENT",
) -> dict:
    surface = completed_surface()
    surface["outcome"] = "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY"
    surface["physical_qc_state"] = physical_qc_state
    surface["stages"]["stability_complete"] = False
    surface["stages"]["ct_gate_complete"] = False
    surface["stages"]["screening_liveness_verdict"] = verdict
    surface["stages"]["screening_liveness_reason"] = reason
    surface["stages"]["screening_receipt_manifest_bound"] = True
    return surface


def seal(module, ledger: dict) -> dict:
    ledger["ledger_sha256"] = module.content_sha256(ledger)
    return ledger


def verify_surface(
    module,
    surface: dict,
    *,
    verify_remote: bool = False,
) -> dict:
    ledger = seal(
        module,
        {
            "schema": "campaignx.surface_qc_ledger.v1",
            "surfaces": [surface],
            "no_automatic_acceptance": True,
        },
    )
    return module.verify(
        ledger,
        expected_backfill_surfaces=1,
        require_all_ledger_surfaces_complete=True,
        verify_remote=verify_remote,
        head_reader=lambda _: {
            "size_bytes": 123,
            "sha256": "a" * 64,
            "etag": "fixture",
        },
    )


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


@pytest.mark.parametrize("verdict", ["DEGENERATE", "EMPTY"])
def test_verify_accepts_terminal_ink_screen_insufficiency(verdict: str) -> None:
    module = load_module()
    receipt = verify_surface(
        module,
        completed_insufficient_surface(verdict=verdict),
        verify_remote=True,
    )

    assert receipt["status"] == "VERIFIED"
    assert receipt["failures"] == []


@pytest.mark.parametrize(
    "physical_qc_state",
    [None, "CT_SUPPORTED", "CT_SUPPORTED_REVIEW"],
)
def test_verify_terminal_ink_screen_insufficiency_requires_its_exact_physical_state(
    physical_qc_state: str | None,
) -> None:
    module = load_module()
    surface = completed_insufficient_surface()
    if physical_qc_state is None:
        del surface["physical_qc_state"]
    else:
        surface["physical_qc_state"] = physical_qc_state

    receipt = verify_surface(module, surface)

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "PHYSICAL_QC_STATE_INVALID_FOR_INK_SCREEN_INSUFFICIENCY",
            "actual": physical_qc_state,
        }
    ]


@pytest.mark.parametrize("verdict", ["ALIVE", "UNKNOWN"])
def test_verify_rejects_invalid_terminal_insufficiency_verdicts(
    verdict: str,
) -> None:
    module = load_module()

    receipt = verify_surface(
        module,
        completed_insufficient_surface(verdict=verdict),
    )

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "SCREENING_LIVENESS_VERDICT_INVALID",
            "actual": verdict,
        }
    ]


def test_verify_rejects_missing_terminal_insufficiency_liveness() -> None:
    module = load_module()
    surface = completed_insufficient_surface()
    del surface["stages"]["screening_liveness_verdict"]
    del surface["stages"]["screening_liveness_reason"]

    receipt = verify_surface(module, surface)

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "SCREENING_LIVENESS_VERDICT_INVALID",
            "actual": None,
        },
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "SCREENING_LIVENESS_REASON_MISSING",
        },
    ]


def test_verify_rejects_malformed_terminal_insufficiency_liveness() -> None:
    module = load_module()
    surface = completed_insufficient_surface()
    surface["stages"]["screening_liveness_verdict"] = {
        "verdict": "DEGENERATE"
    }

    receipt = verify_surface(module, surface)

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "SCREENING_LIVENESS_VERDICT_INVALID",
            "actual": {"verdict": "DEGENERATE"},
        }
    ]


@pytest.mark.parametrize("reason", ["", " \t\n"])
def test_verify_rejects_blank_terminal_insufficiency_reason(reason: str) -> None:
    module = load_module()

    receipt = verify_surface(
        module,
        completed_insufficient_surface(reason=reason),
    )

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "SCREENING_LIVENESS_REASON_MISSING",
        }
    ]


@pytest.mark.parametrize("binding", [None, False])
def test_verify_rejects_an_unbound_terminal_insufficiency_receipt(
    binding: bool | None,
) -> None:
    module = load_module()
    surface = completed_insufficient_surface()
    if binding is None:
        del surface["stages"]["screening_receipt_manifest_bound"]
    else:
        surface["stages"]["screening_receipt_manifest_bound"] = binding

    receipt = verify_surface(module, surface)

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": "SCREENING_RECEIPT_MANIFEST_BINDING_FALSE",
        }
    ]


@pytest.mark.parametrize(
    ("stage", "reason"),
    [
        ("stability_complete", "STABILITY_COMPLETE_UNEXPECTED"),
        ("ct_gate_complete", "CT_GATE_COMPLETE_UNEXPECTED"),
    ],
)
def test_verify_rejects_later_stages_for_terminal_ink_screen_insufficiency(
    stage: str,
    reason: str,
) -> None:
    module = load_module()
    surface = completed_insufficient_surface()
    surface["stages"][stage] = True
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
        verify_remote=False,
    )

    assert receipt["status"] == "FAILED"
    assert receipt["failures"] == [
        {
            "scope": "campaign-x:PHercTEST:surface-01",
            "reason": reason,
        }
    ]


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
