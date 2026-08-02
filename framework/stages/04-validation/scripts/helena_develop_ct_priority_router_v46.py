#!/usr/bin/env python3
"""Freeze surface-calibrated v4.6 with fail-safe preserve-all fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROFILE_ID = "ct-fiber-semantic-priority-router@4.6.0"
MIN_RECALL = 0.95
MIN_EFFICIENCY = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite v4.6 development: {args.output}")
    features_path = args.features.resolve()
    payload = json.loads(features_path.read_text())
    rows = payload["rows"]
    routers: dict[str, Any] = {}
    for scroll in sorted({str(row["scroll_id"]) for row in rows}):
        selected = [row for row in rows if row["scroll_id"] == scroll]
        positive = np.array(
            [row["expected_class"] == "POSITIVE" for row in selected]
        )
        eligible: list[tuple[float, float, str, float]] = []
        for feature in sorted(selected[0]["rank_features"]):
            values = np.array(
                [float(row["rank_features"][feature]) for row in selected]
            )
            for threshold_value in np.linspace(0.0, 0.98, 981):
                threshold = float(round(float(threshold_value), 6))
                recall = float(np.mean(values[positive] > threshold))
                efficiency = float(np.mean(values[~positive] <= threshold))
                if recall >= MIN_RECALL and efficiency >= MIN_EFFICIENCY:
                    eligible.append((efficiency, recall, feature, threshold))
        if eligible:
            best = max(eligible, key=lambda row: (row[0], row[1], row[3], row[2]))
            routers[scroll] = {
                "mode": "CALIBRATED_B1_B2",
                "feature_name": best[2],
                "threshold": best[3],
                "development_positive_b1_recall": best[1],
                "development_confound_b2_rate": best[0],
            }
        else:
            routers[scroll] = {
                "mode": "B1_PRESERVE_ALL",
                "reason": "NO_DEVELOPMENT_OPERATING_POINT_MET_RECALL_AND_EFFICIENCY",
                "development_positive_b1_recall": 1.0,
                "development_confound_b2_rate": 0.0,
            }
    result = {
        "schema": "campaignx.ct_priority_router_v46_development.v1",
        "profile_id": PROFILE_ID,
        "status": "DEVELOPMENT_FROZEN_PENDING_SURFACE_TRANSFER_V7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_features_sha256": sha256(features_path),
        "router": {
            "kind": "KNOWN_SURFACE_CALIBRATOR_WITH_FAIL_SAFE_FALLBACK",
            "by_scroll": routers,
            "unknown_surface_mode": "B1_PRESERVE_ALL",
            "b2_is_preserved_not_negative": True,
        },
        "gates": {
            "minimum_positive_b1_recall_all_scrolls": MIN_RECALL,
            "minimum_confound_b2_rate_calibrated_scrolls": MIN_EFFICIENCY,
            "efficiency_gate_applies_to_preserve_all_scrolls": False,
        },
        "contamination_controls": {
            "v7_calibration_sample_is_consumed_development": True,
            "v7_evaluation_r02_features_or_labels_used": False,
            "evaluation_controls_exclude_v6_and_calibration_with_radius_176": True,
        },
        "non_claims": [
            "v4.6 is not a universal multiroll router.",
            "Preserve-all is the mandatory fallback for unsupported domains.",
            "B2 is preserved evidence and no tier accepts or rejects ink or text.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(routers, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
