#!/usr/bin/env python3
"""Finalize the frozen Large TimeSformer control A/B fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PASS_OUTCOME = "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def finalize(
    *,
    root: Path,
    manifest_path: Path,
    amendment_path: Path,
    runtime: Path,
    ct_spec_path: Path,
    ct_gate_path: Path | None,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    ct_spec = read_json(ct_spec_path)
    decisions: list[dict[str, Any]] = []
    ct_gate_sha256 = None
    if ct_gate_path is not None and ct_gate_path.is_file():
        decisions = read_json(ct_gate_path)
        ct_gate_sha256 = sha256_file(ct_gate_path)
    decisions_by_group: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        decisions_by_group.setdefault(str(decision["group_id"]), []).append(
            decision
        )

    rows: list[dict[str, Any]] = []
    incomplete = False
    for control in manifest["controls"]:
        control_id = str(control["id"])
        analysis_path = (
            runtime / control_id / "analysis" / "INK_STABILITY_ANALYSIS.json"
        )
        if not analysis_path.is_file():
            incomplete = True
            rows.append(
                {
                    "control_id": control_id,
                    "role": control["expected_role"],
                    "status": "MISSING_ANALYSIS",
                }
            )
            continue
        analysis = read_json(analysis_path)
        screening = analysis["text_like_screening"]
        candidate_count = int(screening["glyph_like_candidate_count"])
        row: dict[str, Any] = {
            "control_id": control_id,
            "role": control["expected_role"],
            "analysis_sha256": sha256_file(analysis_path),
            "screening_outcome": screening["screening_outcome"],
            "glyph_like_candidate_count": candidate_count,
            "rows_with_at_least_four_candidates": int(
                screening["rows_with_at_least_four_candidates"]
            ),
        }
        if str(control["expected_role"]).startswith("POSITIVE_"):
            row["positive_morphology_pass"] = (
                screening["screening_outcome"] == PASS_OUTCOME
            )
        else:
            group_decisions = decisions_by_group.get(control_id, [])
            if candidate_count == 0:
                row["hard_negative_specificity_pass"] = True
                row["ct_retained_count"] = 0
            elif len(group_decisions) != candidate_count:
                incomplete = True
                row["hard_negative_specificity_pass"] = False
                row["status"] = "CT_DECISION_COUNT_MISMATCH"
                row["ct_decision_count"] = len(group_decisions)
            else:
                retained = sum(
                    bool(item["retained"]) for item in group_decisions
                )
                row["ct_retained_count"] = retained
                row["hard_negative_specificity_pass"] = retained == 0
        rows.append(row)

    by_id = {str(row["control_id"]): row for row in rows}
    if incomplete:
        conclusion = "INCOMPLETE_FAIL_CLOSED"
    else:
        ph0139_pass = bool(
            by_id["positive-pherc0139"]["positive_morphology_pass"]
        )
        ph172_pass = bool(
            by_id["positive-pherc172"]["positive_morphology_pass"]
        )
        negatives_pass = all(
            bool(row["hard_negative_specificity_pass"])
            for row in rows
            if row["role"] == "HARD_NEGATIVE_CT_DEPTH_DIFFUSE"
        )
        if not ph0139_pass or not negatives_pass:
            conclusion = "REJECT_LARGE_MODEL"
        elif ph172_pass:
            conclusion = "ADOPT_AS_ENSEMBLE_ARM"
        else:
            conclusion = "ADOPT_AS_SCROLL1_SPECIALIST_ONLY"
    return {
        "kind": "campaign_x_phase4_timesformer_large_control_ab_result_v1",
        "status": conclusion,
        "generated_at_utc": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "amendment_sha256": sha256_file(amendment_path),
        "ct_spec_sha256": sha256_file(ct_spec_path),
        "ct_gate_decisions_sha256": ct_gate_sha256,
        "controls": rows,
        "interpretation": (
            "The conclusion follows the prospectively frozen adoption policy. "
            "A passing positive is only a morphology-control response, and a "
            "CT-retained negative is only a specificity failure. No result "
            "accepts ink, identifies letters, or establishes First Letters."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--ct-spec", type=Path, required=True)
    parser.add_argument("--ct-gate-decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize(
        root=args.root.resolve(),
        manifest_path=args.manifest.resolve(),
        amendment_path=args.amendment.resolve(),
        runtime=args.runtime.resolve(),
        ct_spec_path=args.ct_spec.resolve(),
        ct_gate_path=(
            args.ct_gate_decisions.resolve()
            if args.ct_gate_decisions
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
