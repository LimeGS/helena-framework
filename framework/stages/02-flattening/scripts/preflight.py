#!/usr/bin/env python3
"""Produce a Phase 3 readiness receipt without starting Phase 3 work.

Phase 3 is deliberately unavailable unless Phase 2 has reached its explicit
completion state.  This command does not read H0/H1 contents, select targets,
or create constraints; it only reads the Phase 2 terminal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
INDEPENDENT_COMPLETION = "PHASE2_COMPLETE"
LOCAL_COMPLETION = "COMPLETED_LOCAL_HOLDOUT_V1_ONLY"
R6_LOCAL_COMPLETION = "COMPLETED_LOCAL_FUNCTIONAL_ONLY"
LOCAL_SCOPE = "LOCAL_PIPELINE_CONTINUATION_ONLY"
LOCAL_ARTIFACT_SCOPE = "LOCAL_HOLDOUT_V1_ONLY"
R6_LOCAL_ARTIFACT_SCOPE = "R6_LOCAL_FUNCTIONAL_ONLY"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_receipt_if_changed(path: Path, receipt: dict[str, Any]) -> None:
    stable = {key: value for key, value in receipt.items() if key != "generated_at_utc"}
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_stable = {key: value for key, value in previous.items() if key != "generated_at_utc"}
            if previous_stable == stable:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _local_completion_is_valid(phase2: dict[str, Any]) -> bool:
    """Recognize only the exact, explicitly limited Amendment-012 transition."""

    local = phase2.get("relation_v2_local_holdout_v1")
    if not isinstance(local, dict):
        return False
    transition = local.get("transition")
    limitations = local.get("limitations")
    if not isinstance(transition, dict) or not isinstance(limitations, dict):
        return False
    expected_transition = {
        "complete": True,
        "overall": LOCAL_COMPLETION,
        "completion_kind": LOCAL_COMPLETION,
        "eligible": True,
        "validation_scope": LOCAL_SCOPE,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
    }
    return (
        phase2.get("overall") == LOCAL_COMPLETION
        and phase2.get("complete") is True
        and phase2.get("completion_kind") == LOCAL_COMPLETION
        and phase2.get("eligible") is True
        and phase2.get("validation_scope") == LOCAL_SCOPE
        and phase2.get("independent_h1_validated") is False
        and phase2.get("external_generalization_claim") is False
        and local.get("status") == "PASSED_LOCAL_HOLDOUT_V1"
        and local.get("validation_protocol") == "LOCAL_HOLDOUT_V1"
        and local.get("validation_authority") == "LOCAL_SELF_VALIDATION"
        and local.get("independent_h1_validated") is False
        and local.get("external_generalization_claim") is False
        and transition == expected_transition
        and limitations.get("artifact_scope") == LOCAL_ARTIFACT_SCOPE
        and limitations.get("limitation_banner_required") is True
        and limitations.get("public_or_external_deployment_claim_allowed") is False
        and limitations.get("independent_validation_claim_allowed") is False
        and limitations.get("external_generalization_claim_allowed") is False
    )


def _completion_mode(phase2: dict[str, Any]) -> str | None:
    if (
        phase2.get("overall") == INDEPENDENT_COMPLETION
        and phase2.get("eligible") is True
    ):
        return "INDEPENDENT_PHASE2_COMPLETION"
    if _local_completion_is_valid(phase2):
        return "LOCAL_HOLDOUT_V1_ONLY"
    r6 = phase2.get("relation_v2_r6")
    if (
        phase2.get("overall") == R6_LOCAL_COMPLETION
        and phase2.get("complete") is True
        and phase2.get("completion_kind") == R6_LOCAL_COMPLETION
        and phase2.get("eligible") is True
        and phase2.get("validation_scope") == LOCAL_SCOPE
        and phase2.get("independent_h1_validated") is False
        and phase2.get("external_generalization_claim") is False
        and isinstance(r6, dict)
        and r6.get("status") == "PASSED_R6_LOCAL_FUNCTIONAL"
        and r6.get("scope") == LOCAL_SCOPE
        and r6.get("h0_reused") is False
        and r6.get("h1_opened") is False
        and r6.get("independent_h1_validated") is False
        and r6.get("external_generalization_claim") is False
        and r6.get("first_letters_eligible") is True
    ):
        return "R6_LOCAL_FUNCTIONAL_ONLY"
    return None


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-results", type=Path, default=ROOT / "phase2/PHASE2_RESULTS.json")
    parser.add_argument("--output", type=Path, default=ROOT / "phase3/PHASE3_READINESS.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    try:
        phase2 = json.loads(args.results.read_text(encoding="utf-8"))
        if not isinstance(phase2, dict):
            raise ValueError("Phase 2 results must be a JSON object")
        completion_mode = _completion_mode(phase2)
        eligible = completion_mode is not None
        local_only = completion_mode in {
            "LOCAL_HOLDOUT_V1_ONLY",
            "R6_LOCAL_FUNCTIONAL_ONLY",
        }
        receipt = {
            "kind": "campaign_x_phase3_preflight_v1",
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "results_path": _display_path(args.results),
            "results_sha256": sha256_file(args.results),
            "overall": phase2.get("overall"),
            "complete": phase2.get("complete"),
            "completion_kind": phase2.get("completion_kind"),
            "phase3_eligible": phase2.get("eligible"),
            "validation_scope": phase2.get("validation_scope"),
            "independent_h1_validated": phase2.get("independent_h1_validated"),
            "external_generalization_claim": phase2.get("external_generalization_claim"),
            "completion_mode": completion_mode,
            "artifact_scope": (
                R6_LOCAL_ARTIFACT_SCOPE
                if completion_mode == "R6_LOCAL_FUNCTIONAL_ONLY"
                else (
                    LOCAL_ARTIFACT_SCOPE
                    if completion_mode == "LOCAL_HOLDOUT_V1_ONLY"
                    else (
                        "INDEPENDENT_PHASE2"
                        if completion_mode == "INDEPENDENT_PHASE2_COMPLETION"
                        else None
                    )
                )
            ),
            "limitation_banner_required": local_only,
            "status": "READY_FOR_EXPLICIT_PHASE3_AUTHORIZATION" if eligible else "BLOCKED_BY_PHASE2_TERMINAL_STATE",
            "execution_started": False,
            "reason": (
                "Phase 2 reached COMPLETED_LOCAL_FUNCTIONAL_ONLY through R6; Phase 3 is limited to LOCAL_PIPELINE_CONTINUATION_ONLY, every artifact must carry R6_LOCAL_FUNCTIONAL_ONLY, and no independent-H1 or external-generalization claim is allowed."
                if completion_mode == "R6_LOCAL_FUNCTIONAL_ONLY"
                else (
                    "Phase 2 reached COMPLETED_LOCAL_HOLDOUT_V1_ONLY; Phase 3 is limited to LOCAL_PIPELINE_CONTINUATION_ONLY, every artifact must carry LOCAL_HOLDOUT_V1_ONLY, and execution still requires separate explicit authorization."
                    if completion_mode == "LOCAL_HOLDOUT_V1_ONLY"
                    else (
                        "Phase 2 reached PHASE2_COMPLETE; Phase 3 still requires a separate explicit authorization."
                        if eligible
                        else "Phase 2 has no recognized, internally coherent completion state; no Phase 3 data or claims may be created."
                    )
                )
            ),
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        receipt = {"status": "FAILED_PHASE3_PREFLIGHT", "reason": f"{type(error).__name__}: {error}", "execution_started": False}
    if not args.no_write:
        write_receipt_if_changed(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "READY_FOR_EXPLICIT_PHASE3_AUTHORIZATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
