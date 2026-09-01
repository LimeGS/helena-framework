#!/usr/bin/env python3
"""Fail-closed verification of a completed Helena Framework surface-QC ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse


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


HeadReader = Callable[[str], dict[str, Any]]


def default_head_reader(uri: str) -> dict[str, Any]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        import boto3

        response = boto3.client("s3").head_object(
            Bucket=parsed.netloc,
            Key=unquote(parsed.path.lstrip("/")),
        )
        return {
            "size_bytes": int(response["ContentLength"]),
            "sha256": response.get("Metadata", {}).get("sha256"),
            "etag": str(response.get("ETag", "")).strip('"') or None,
        }
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme == "":
        path = Path(uri)
    else:
        raise RuntimeError(f"unsupported evidence URI scheme: {parsed.scheme}")
    if not path.is_file():
        raise RuntimeError(f"evidence manifest is missing: {path}")
    return {
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "etag": None,
    }


def verify(
    ledger: dict[str, Any],
    *,
    expected_backfill_surfaces: int,
    require_all_ledger_surfaces_complete: bool,
    verify_remote: bool,
    head_reader: HeadReader = default_head_reader,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    ledger_core = dict(ledger)
    embedded_ledger_sha256 = ledger_core.pop("ledger_sha256", None)
    calculated_ledger_sha256 = content_sha256(ledger_core)
    if embedded_ledger_sha256 != calculated_ledger_sha256:
        failures.append(
            {
                "scope": "ledger",
                "reason": "LEDGER_HASH_MISMATCH",
                "expected": embedded_ledger_sha256,
                "actual": calculated_ledger_sha256,
            }
        )
    rows = ledger.get("surfaces")
    if ledger.get("schema") != "campaignx.surface_qc_ledger.v1":
        failures.append({"scope": "ledger", "reason": "UNSUPPORTED_SCHEMA"})
    if ledger.get("no_automatic_acceptance") is not True:
        failures.append({"scope": "ledger", "reason": "AUTOMATIC_ACCEPTANCE_NOT_FALSE"})
    if not isinstance(rows, list):
        failures.append({"scope": "ledger", "reason": "SURFACE_ROWS_MISSING"})
        rows = []

    identifiers = [str(row.get("surface_id", "")) for row in rows if isinstance(row, dict)]
    if len(identifiers) != len(set(identifiers)):
        failures.append({"scope": "ledger", "reason": "DUPLICATE_SURFACE_ID"})
    backfill = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("origin") == "HISTORICAL_BACKFILL"
    ]
    if len(backfill) != expected_backfill_surfaces:
        failures.append(
            {
                "scope": "ledger",
                "reason": "BACKFILL_COUNT_MISMATCH",
                "expected": expected_backfill_surfaces,
                "actual": len(backfill),
            }
        )

    required = rows if require_all_ledger_surfaces_complete else backfill
    remote: list[dict[str, Any]] = []
    allowed_outcomes = {
        "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
        "CT_SUPPORTED_RETAINED_FOR_REVIEW",
        "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
        "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY",
    }
    for row in required:
        surface_id = str(row.get("surface_id", ""))
        if row.get("qc_state") != "COMPLETED":
            failures.append(
                {"scope": surface_id, "reason": "QC_NOT_COMPLETED", "state": row.get("qc_state")}
            )
            continue
        outcome = row.get("outcome")
        if outcome not in allowed_outcomes:
            failures.append(
                {"scope": surface_id, "reason": "INVALID_OR_MISSING_OUTCOME", "outcome": outcome}
            )
        stages = row.get("stages") if isinstance(row.get("stages"), dict) else {}
        if outcome == "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY":
            physical_qc_state = row.get("physical_qc_state")
            if physical_qc_state != "INK_SCREEN_INSUFFICIENT":
                failures.append(
                    {
                        "scope": surface_id,
                        "reason": "PHYSICAL_QC_STATE_INVALID_FOR_INK_SCREEN_INSUFFICIENCY",
                        "actual": physical_qc_state,
                    }
                )
            required_complete_fields = (
                "render_complete",
                "inference_complete",
                "evidence_manifest_complete",
            )
            for field in ("stability_complete", "ct_gate_complete"):
                if stages.get(field) is not False:
                    failures.append(
                        {"scope": surface_id, "reason": f"{field.upper()}_UNEXPECTED"}
                    )
            if stages.get("screening_receipt_manifest_bound") is not True:
                failures.append(
                    {
                        "scope": surface_id,
                        "reason": "SCREENING_RECEIPT_MANIFEST_BINDING_FALSE",
                    }
                )
            verdict = stages.get("screening_liveness_verdict")
            if not isinstance(verdict, str) or verdict not in {
                "DEGENERATE",
                "EMPTY",
            }:
                failures.append(
                    {
                        "scope": surface_id,
                        "reason": "SCREENING_LIVENESS_VERDICT_INVALID",
                        "actual": verdict,
                    }
                )
            reason = stages.get("screening_liveness_reason")
            if not isinstance(reason, str) or not reason.strip():
                failures.append(
                    {
                        "scope": surface_id,
                        "reason": "SCREENING_LIVENESS_REASON_MISSING",
                    }
                )
        else:
            required_complete_fields = (
                "render_complete",
                "inference_complete",
                "stability_complete",
                "ct_gate_complete",
                "evidence_manifest_complete",
            )
        for field in required_complete_fields:
            if stages.get(field) is not True:
                failures.append({"scope": surface_id, "reason": f"{field.upper()}_FALSE"})
        if stages.get("rendered_slice_count") != 65:
            failures.append(
                {
                    "scope": surface_id,
                    "reason": "RENDERED_SLICE_COUNT_MISMATCH",
                    "actual": stages.get("rendered_slice_count"),
                }
            )
        if stages.get("evidence_manifest_matches_database") is not True:
            failures.append(
                {"scope": surface_id, "reason": "LOCAL_EVIDENCE_HASH_MISMATCH"}
            )
        expected_sha256 = row.get("evidence_manifest_sha256")
        uri = row.get("evidence_uri")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            failures.append({"scope": surface_id, "reason": "EVIDENCE_SHA256_MISSING"})
            continue
        if not isinstance(uri, str) or not uri:
            failures.append({"scope": surface_id, "reason": "EVIDENCE_URI_MISSING"})
            continue
        if verify_remote:
            try:
                head = head_reader(uri)
                remote.append({"surface_id": surface_id, "uri": uri, **head})
                if head.get("size_bytes", 0) <= 0:
                    failures.append({"scope": surface_id, "reason": "REMOTE_EVIDENCE_EMPTY"})
                if head.get("sha256") != expected_sha256:
                    failures.append(
                        {
                            "scope": surface_id,
                            "reason": "REMOTE_EVIDENCE_HASH_MISMATCH",
                            "expected": expected_sha256,
                            "actual": head.get("sha256"),
                        }
                    )
            except Exception as exc:
                failures.append(
                    {"scope": surface_id, "reason": "REMOTE_EVIDENCE_HEAD_FAILED", "detail": str(exc)}
                )

    receipt = {
        "schema": "campaignx.surface_qc_ledger_verification.v1",
        "generated_at_utc": utc_now(),
        "ledger_sha256": calculated_ledger_sha256,
        "expected_backfill_surfaces": expected_backfill_surfaces,
        "require_all_ledger_surfaces_complete": require_all_ledger_surfaces_complete,
        "verify_remote": verify_remote,
        "counts": {
            "ledger_surfaces": len(rows),
            "backfill_surfaces": len(backfill),
            "required_surfaces": len(required),
            "completed_required_surfaces": sum(
                row.get("qc_state") == "COMPLETED" for row in required
            ),
            "remote_evidence_verified": len(remote),
            "failures": len(failures),
        },
        "failures": failures,
        "remote_evidence": remote,
        "status": "VERIFIED" if not failures else "FAILED",
        "no_automatic_acceptance": True,
        "semantics": [
            "VERIFIED proves execution and artifact integrity, not presence or absence of letters",
            "CT_SUPPORTED_RETAINED_FOR_REVIEW remains a review queue item, not accepted ink",
        ],
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--ledger", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--expected-backfill-surfaces", type=int, required=True)
    value.add_argument("--require-all-ledger-surfaces-complete", action="store_true")
    value.add_argument("--verify-remote", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    ledger = read_json(args.ledger.resolve())
    receipt = verify(
        ledger,
        expected_backfill_surfaces=args.expected_backfill_surfaces,
        require_all_ledger_surfaces_complete=args.require_all_ledger_surfaces_complete,
        verify_remote=args.verify_remote,
    )
    write_json_atomic(args.output.resolve(), receipt)
    print(json.dumps({"output": str(args.output), **receipt["counts"], "status": receipt["status"]}, indent=2))
    return 0 if receipt["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
