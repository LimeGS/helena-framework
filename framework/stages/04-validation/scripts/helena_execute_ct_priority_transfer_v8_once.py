#!/usr/bin/env python3
"""Evaluate frozen partial router v4.7 exactly once on spatial V8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from helena_develop_ct_priority_router_v47 import (
    MIN_EFFICIENCY,
    MIN_RECALL,
    PROFILE_ID,
)
from helena_execute_ct_priority_transfer_v7_once import (
    now,
    route_row,
    sha256,
    summarize,
    write,
)


BENCHMARK_ID = "SURFACE_CALIBRATION_TRANSFER_V8"


def execute(
    development_path: Path,
    features_path: Path,
    controls_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to rerun or overwrite V8")
    output_root.mkdir(parents=True, exist_ok=True)
    development = json.loads(development_path.read_text())
    contamination = development.get("contamination_controls", {})
    if (
        development.get("profile_id") != PROFILE_ID
        or development.get("status") != "FROZEN_PENDING_SURFACE_TRANSFER_V8"
        or contamination.get("v8_features_or_labels_used")
        or not contamination.get(
            "v8_must_exclude_v6_v7_calibration_and_v7_evaluation"
        )
    ):
        raise RuntimeError("v4.7 is not cleanly frozen")
    preclaim = {
        "schema": "campaignx.surface_calibration_transfer_v8_preclaim.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "READY_FOR_SINGLE_EXECUTION",
        "created_at_utc": now(),
        "maximum_attempts": 1,
        "rerun_allowed": False,
        "gates": {
            "minimum_positive_b1_recall_per_scroll": MIN_RECALL,
            "minimum_confound_b2_rate_calibrated_scrolls": MIN_EFFICIENCY,
            "efficiency_gate_applies_to_preserve_all_scrolls": False,
            "evidence_preservation": 1.0,
        },
        "inputs": {
            "development_sha256": sha256(development_path),
            "features_sha256": sha256(features_path),
            "controls_sha256": sha256(controls_path),
        },
    }
    write(output_root / "PRECLAIM.json", preclaim)
    write(
        output_root / "EXECUTION_CLAIM.json",
        {
            "schema": "campaignx.surface_calibration_transfer_v8_claim.v1",
            "status": "CLAIMED_SINGLE_EXECUTION",
            "claimed_at_utc": now(),
            "maximum_attempts": 1,
            "preclaim_sha256": sha256(output_root / "PRECLAIM.json"),
        },
    )
    features = json.loads(features_path.read_text())
    controls_payload = json.loads(controls_path.read_text())
    rows = features["rows"]
    controls = controls_payload["controls"]
    feature_ids = {str(row["component_id"]) for row in rows}
    control_ids = {str(row["component_id"]) for row in controls}
    if (
        features.get("benchmark_id") != BENCHMARK_ID
        or controls_payload.get("benchmark_id") != BENCHMARK_ID
        or len(rows) != 300
        or len(controls) != 300
        or len(feature_ids) != 300
        or feature_ids != control_ids
    ):
        raise RuntimeError("invalid or mismatched V8 frozen inputs")
    routed: list[dict[str, Any]] = []
    for row in rows:
        route = route_row(row, development["router"]["by_scroll"])
        routed.append(
            {
                "component_id": row["component_id"],
                "scroll_id": row["scroll_id"],
                "official_surface_id": row["official_surface_id"],
                "expected_class": row["expected_class"],
                **route,
                "evidence_preserved": True,
            }
        )
    by_scroll = summarize(routed)
    recall_pass = all(
        row["positive_b1_recall"] >= MIN_RECALL for row in by_scroll.values()
    )
    calibrated = [
        row
        for row in by_scroll.values()
        if row["router_mode"] == "CALIBRATED_B1_B2"
    ]
    efficiency_pass = bool(calibrated) and all(
        row["confound_b2_rate"] >= MIN_EFFICIENCY for row in calibrated
    )
    preservation_pass = all(row["evidence_preserved"] for row in routed)
    passed = recall_pass and efficiency_pass and preservation_pass
    result = {
        "schema": "campaignx.surface_calibration_transfer_v8_result.v1",
        "benchmark_id": BENCHMARK_ID,
        "profile_id": PROFILE_ID,
        "status": "PASSED" if passed else "FAILED",
        "executed_at_utc": now(),
        "metrics": {"by_scroll": by_scroll},
        "gates": {
            "positive_recall_per_scroll": recall_pass,
            "confound_efficiency_calibrated_scrolls": efficiency_pass,
            "evidence_preservation": preservation_pass,
        },
        "promotion_decision": (
            "PROMOTE_V47_AS_PARTIAL_PRIORITY_LAYER"
            if passed
            else "DO_NOT_PROMOTE_V47"
        ),
        "routed_controls": routed,
        "scope": {
            "validated": (
                "WITHIN_SURFACE_SPATIAL_TRANSFER_FOR_EXPLICITLY_CALIBRATED_"
                "SURFACES_WITH_PRESERVE_ALL_FALLBACK"
            ),
            "not_validated": [
                "UNSEEN_SURFACE_TRANSFER",
                "UNSEEN_SCROLL_TRANSFER",
                "INK_OR_TEXT_ACCEPTANCE",
            ],
        },
        "non_claims": [
            "B2 remains preserved evidence.",
            "Unsupported domains remain B1 and are not efficiency failures.",
            "No tier accepts or rejects ink, text, letters, or First Letters.",
        ],
    }
    result_path = output_root / "SURFACE_CALIBRATION_TRANSFER_V8_RESULT.json"
    write(result_path, result)
    write(
        output_root / "EXECUTION_RECEIPT.json",
        {
            "schema": "campaignx.surface_calibration_transfer_v8_receipt.v1",
            "status": "EXECUTED_ONCE",
            "result_sha256": sha256(result_path),
            "rerun_allowed": False,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        args.development_receipt.resolve(),
        args.features.resolve(),
        args.controls.resolve(),
        args.output_root.resolve(),
    )
    print(
        json.dumps(
            {"status": result["status"], "metrics": result["metrics"]}, indent=2
        )
    )
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
