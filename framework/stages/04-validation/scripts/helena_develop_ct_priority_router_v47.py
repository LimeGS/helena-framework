#!/usr/bin/env python3
"""Freeze v4.7 from the prospective V7 result.

Only surface routers that passed both prospective V7 gates remain calibrated.
Every failed, unknown, or unsupported domain falls back to B1 preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROFILE_ID = "ct-fiber-semantic-priority-router@4.7.0"
MIN_RECALL = 0.95
MIN_EFFICIENCY = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    v46_development: dict[str, Any],
    v7_result: dict[str, Any],
    *,
    development_sha256: str,
    v7_result_sha256: str,
) -> dict[str, Any]:
    if (
        v46_development.get("profile_id")
        != "ct-fiber-semantic-priority-router@4.6.0"
        or v7_result.get("benchmark_id") != "SURFACE_CALIBRATION_TRANSFER_V7"
        or v7_result.get("status") != "FAILED"
    ):
        raise RuntimeError("v4.7 requires the frozen failed V7 result")
    prior = v46_development["router"]["by_scroll"]
    metrics = v7_result["metrics"]["by_scroll"]
    routers: dict[str, Any] = {}
    for scroll, router in sorted(prior.items()):
        observed = metrics[scroll]
        eligible = (
            router["mode"] == "CALIBRATED_B1_B2"
            and observed["positive_b1_recall"] >= MIN_RECALL
            and observed["confound_b2_rate"] >= MIN_EFFICIENCY
        )
        if eligible:
            routers[scroll] = {
                **router,
                "v7_positive_b1_recall": observed["positive_b1_recall"],
                "v7_confound_b2_rate": observed["confound_b2_rate"],
                "provenance": "RETAINED_AFTER_PROSPECTIVE_V7_PASS",
            }
        else:
            routers[scroll] = {
                "mode": "B1_PRESERVE_ALL",
                "reason": (
                    "V7_PROSPECTIVE_GATE_FAILED"
                    if router["mode"] == "CALIBRATED_B1_B2"
                    else "NO_PROSPECTIVELY_VALIDATED_CALIBRATED_ROUTER"
                ),
                "development_positive_b1_recall": 1.0,
                "development_confound_b2_rate": 0.0,
            }
    return {
        "schema": "campaignx.ct_priority_router_v47_development.v1",
        "profile_id": PROFILE_ID,
        "status": "FROZEN_PENDING_SURFACE_TRANSFER_V8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v46_development_sha256": development_sha256,
            "v7_result_sha256": v7_result_sha256,
        },
        "router": {
            "kind": "PROSPECTIVELY_RETAINED_SURFACE_ROUTER_WITH_FAIL_SAFE_FALLBACK",
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
            "v7_consumed_for_v47_design": True,
            "v8_features_or_labels_used": False,
            "v8_must_exclude_v6_v7_calibration_and_v7_evaluation": True,
        },
        "non_claims": [
            "v4.7 is a partial within-surface priority layer, not a universal router.",
            "Preserve-all is mandatory for unsupported or failed domains.",
            "No tier accepts or rejects ink, text, letters, or First Letters.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v46-development", type=Path, required=True)
    parser.add_argument("--v7-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite v4.7: {args.output}")
    development_path = args.v46_development.resolve()
    result_path = args.v7_result.resolve()
    result = build(
        json.loads(development_path.read_text()),
        json.loads(result_path.read_text()),
        development_sha256=sha256(development_path),
        v7_result_sha256=sha256(result_path),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["router"]["by_scroll"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
