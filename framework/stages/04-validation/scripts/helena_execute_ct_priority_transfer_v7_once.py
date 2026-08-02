#!/usr/bin/env python3
"""Evaluate frozen surface-calibrated router v4.6 exactly once on V7."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helena_develop_ct_priority_router_v46 import PROFILE_ID


BENCHMARK_ID = "SURFACE_CALIBRATION_TRANSFER_V7"
MIN_POSITIVE_B1_RECALL = 0.95
MIN_CONFOUND_B2_RATE = 0.15


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def route_row(
    row: dict[str, Any], routers: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    scroll_id = str(row["scroll_id"])
    router = routers.get(scroll_id)
    if router is None or router["mode"] == "B1_PRESERVE_ALL":
        return {
            "router_mode": "B1_PRESERVE_ALL",
            "feature_name": None,
            "threshold": None,
            "score": None,
            "tier": "B1",
        }
    if router["mode"] != "CALIBRATED_B1_B2":
        raise RuntimeError(f"unsupported v4.6 router mode for {scroll_id}")
    feature_name = str(router["feature_name"])
    threshold = float(router["threshold"])
    score = float(row["rank_features"][feature_name])
    return {
        "router_mode": "CALIBRATED_B1_B2",
        "feature_name": feature_name,
        "threshold": threshold,
        "score": score,
        "tier": "B1" if score > threshold else "B2",
    }


def summarize(routed: list[dict[str, Any]]) -> dict[str, Any]:
    by_scroll: dict[str, Any] = {}
    for scroll in sorted({str(row["scroll_id"]) for row in routed}):
        selected = [row for row in routed if row["scroll_id"] == scroll]
        positives = [row for row in selected if row["expected_class"] == "POSITIVE"]
        confounds = [row for row in selected if row["expected_class"] == "CONFOUND"]
        positive_b1 = sum(row["tier"] == "B1" for row in positives)
        confound_b2 = sum(row["tier"] == "B2" for row in confounds)
        modes = sorted({str(row["router_mode"]) for row in selected})
        if len(modes) != 1:
            raise RuntimeError(f"mixed router modes within {scroll}")
        by_scroll[scroll] = {
            "router_mode": modes[0],
            "positive_total": len(positives),
            "positive_b1": positive_b1,
            "positive_b1_recall": positive_b1 / len(positives),
            "confound_total": len(confounds),
            "confound_b2": confound_b2,
            "confound_b2_rate": confound_b2 / len(confounds),
        }
    return by_scroll


def execute(
    development_path: Path,
    features_path: Path,
    controls_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to rerun or overwrite V7")
    output_root.mkdir(parents=True, exist_ok=True)

    development = json.loads(development_path.read_text())
    contamination = development.get("contamination_controls", {})
    if (
        development.get("profile_id") != PROFILE_ID
        or development.get("status")
        != "DEVELOPMENT_FROZEN_PENDING_SURFACE_TRANSFER_V7"
        or contamination.get("v7_evaluation_r02_features_or_labels_used")
        or not contamination.get(
            "evaluation_controls_exclude_v6_and_calibration_with_radius_176"
        )
    ):
        raise RuntimeError("v4.6 is not cleanly frozen")

    preclaim = {
        "schema": "campaignx.surface_calibration_transfer_v7_preclaim.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "READY_FOR_SINGLE_EXECUTION",
        "created_at_utc": now(),
        "maximum_attempts": 1,
        "rerun_allowed": False,
        "gates": {
            "minimum_positive_b1_recall_per_scroll": MIN_POSITIVE_B1_RECALL,
            "minimum_confound_b2_rate_calibrated_scrolls": MIN_CONFOUND_B2_RATE,
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
            "schema": "campaignx.surface_calibration_transfer_v7_claim.v1",
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
    if (
        features.get("benchmark_id") != BENCHMARK_ID
        or controls_payload.get("benchmark_id") != BENCHMARK_ID
        or len(rows) != 300
        or len(controls) != 300
    ):
        raise RuntimeError("invalid V7 frozen inputs")
    feature_ids = {str(row["component_id"]) for row in rows}
    control_ids = {str(row["component_id"]) for row in controls}
    if len(feature_ids) != 300 or feature_ids != control_ids:
        raise RuntimeError("V7 controls/features do not match exactly")

    routers = development["router"]["by_scroll"]
    routed: list[dict[str, Any]] = []
    for row in rows:
        route = route_row(row, routers)
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
        row["positive_b1_recall"] >= MIN_POSITIVE_B1_RECALL
        for row in by_scroll.values()
    )
    calibrated = [
        row
        for row in by_scroll.values()
        if row["router_mode"] == "CALIBRATED_B1_B2"
    ]
    efficiency_pass = bool(calibrated) and all(
        row["confound_b2_rate"] >= MIN_CONFOUND_B2_RATE for row in calibrated
    )
    preservation_pass = all(row["evidence_preserved"] for row in routed)
    passed = recall_pass and efficiency_pass and preservation_pass
    result = {
        "schema": "campaignx.surface_calibration_transfer_v7_result.v1",
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
            "PROMOTE_V46_AS_PARTIAL_PRIORITY_LAYER"
            if passed
            else "DO_NOT_PROMOTE_V46"
        ),
        "routed_controls": routed,
        "scope": {
            "validated": "WITHIN_SURFACE_SPATIAL_TRANSFER_FOR_LISTED_SURFACES",
            "not_validated": [
                "UNSEEN_SURFACE_TRANSFER",
                "UNSEEN_SCROLL_TRANSFER",
                "INK_OR_TEXT_ACCEPTANCE",
            ],
        },
        "non_claims": [
            "B2 remains preserved evidence.",
            "Preserve-all is the mandatory fallback for unsupported surfaces.",
            "No tier accepts or rejects ink, text, letters, or First Letters.",
        ],
    }
    result_path = output_root / "SURFACE_CALIBRATION_TRANSFER_V7_RESULT.json"
    write(result_path, result)
    write(
        output_root / "EXECUTION_RECEIPT.json",
        {
            "schema": "campaignx.surface_calibration_transfer_v7_receipt.v1",
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
