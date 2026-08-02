#!/usr/bin/env python3
"""Route every v3 CT-gate decision into a non-destructive v4 shadow tier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_violation(decision: dict[str, Any]) -> float:
    total = 0.0
    for check in decision["checks"]:
        if check["passed"]:
            continue
        value = float(check["value"])
        threshold = float(check["threshold"])
        scale = max(abs(threshold), 1e-9)
        if str(check["operator"]).startswith(">"):
            total += max(0.0, threshold - value) / scale
        elif str(check["operator"]).startswith("<"):
            total += max(0.0, value - threshold) / scale
        else:
            raise ValueError(f"unsupported operator: {check['operator']}")
    return total


def route_decision(
    decision: dict[str, Any],
    *,
    coverage_features: set[str],
    physical_features: dict[str, str] | None,
    minimum_window_coverage_fraction: float,
) -> dict[str, Any]:
    failed = set(map(str, decision.get("failed_features", [])))
    if decision.get("retained") is True:
        tier = "TIER_A_V3_RETAINED_REVIEW"
        action = "REVIEW_ORTHOGONAL_CT"
        reason = "all frozen v3 checks passed"
    elif failed & coverage_features:
        tier = "TIER_C_EXTEND_OR_RESEGMENT"
        action = "EXTEND_OR_RESEGMENT_SURFACE"
        reason = "candidate intersects unsupported surface or CT coverage"
    else:
        tier = "TIER_B_SHADOW_REVIEW"
        action = "PRESERVE_AND_REVIEW_IN_SHADOW_QUEUE"
        reason = "depth-localization failure is not treated as absence"

    physical_status = "NOT_PROVIDED"
    if physical_features is not None:
        coverage = float(physical_features["physical_window_coverage_fraction"])
        physical_status = (
            "PHYSICAL_WINDOW_COMPLETE"
            if coverage >= minimum_window_coverage_fraction
            else "PHYSICAL_WINDOW_INCOMPLETE"
        )
        if physical_status == "PHYSICAL_WINDOW_INCOMPLETE":
            tier = "TIER_C_EXTEND_OR_RESEGMENT"
            action = "EXTEND_CT_DEPTH_OR_RESEGMENT_SURFACE"
            reason = "fixed physical z window is incompletely supported"

    return {
        "group_id": decision["group_id"],
        "candidate_id": decision["candidate_id"],
        "class": decision.get("class"),
        "v3_retained": decision["retained"],
        "v3_decision_unchanged": decision["decision"],
        "failed_features": sorted(failed),
        "failed_check_count": len(failed),
        "normalized_gate_violation": normalized_violation(decision),
        "shadow_tier": tier,
        "required_action": action,
        "routing_reason": reason,
        "physical_feature_status": physical_status,
        "physical_features": physical_features,
        "not_discarded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-evaluation", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--physical-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate_path = args.gate_evaluation.resolve()
    profile_path = args.profile.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to overwrite v4 shadow routing evidence")
    output.mkdir(parents=True, exist_ok=True)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    decisions_path = gate_path.parent / gate["artifacts"]["decisions"]
    if sha256(decisions_path) != gate["artifacts"]["decisions_sha256"]:
        raise RuntimeError("v3 decision artifact hash mismatch")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))

    physical_by_key: dict[tuple[str, str], dict[str, str]] = {}
    if args.physical_features:
        with args.physical_features.resolve().open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (str(row["group_id"]), str(row["candidate_id"]))
                if key in physical_by_key:
                    raise RuntimeError(f"duplicate physical feature row: {key}")
                physical_by_key[key] = row

    coverage_features = set(profile["routing"]["coverage_features"])
    minimum_coverage = float(
        profile["physical_depth_sampling"]["minimum_window_coverage_fraction"]
    )
    routes = [
        route_decision(
            decision,
            coverage_features=coverage_features,
            physical_features=physical_by_key.get(
                (str(decision["group_id"]), str(decision["candidate_id"]))
            ),
            minimum_window_coverage_fraction=minimum_coverage,
        )
        for decision in decisions
    ]
    routes.sort(key=lambda row: (str(row["group_id"]), str(row["candidate_id"])))
    route_path = output / "CT_GATE_V4_SHADOW_ROUTES.json"
    route_path.write_text(
        json.dumps(routes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    extension_requests = [
        {
            "schema": "campaignx.surface_extension_request.v1",
            "group_id": row["group_id"],
            "candidate_id": row["candidate_id"],
            "required_action": row["required_action"],
            "failed_features": row["failed_features"],
            "reason": row["routing_reason"],
            "automatic_execution_authorized": False,
        }
        for row in routes
        if row["shadow_tier"] == "TIER_C_EXTEND_OR_RESEGMENT"
    ]
    extension_path = output / "SURFACE_EXTENSION_REQUESTS.json"
    extension_path.write_text(
        json.dumps(extension_requests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = Counter(str(row["shadow_tier"]) for row in routes)
    receipt = {
        "schema": "campaignx.ct_gate_shadow_routing.v4",
        "status": "SHADOW_ROUTING_COMPLETE_V3_UNCHANGED",
        "generated_at_utc": utc_now(),
        "gate": {"path": str(gate_path), "sha256": sha256(gate_path)},
        "profile": {"path": str(profile_path), "sha256": sha256(profile_path)},
        "physical_features": (
            {
                "path": str(args.physical_features.resolve()),
                "sha256": sha256(args.physical_features.resolve()),
            }
            if args.physical_features
            else None
        ),
        "candidate_count": len(routes),
        "tier_counts": dict(sorted(counts.items())),
        "not_discarded_count": sum(bool(row["not_discarded"]) for row in routes),
        "extension_request_count": len(extension_requests),
        "artifacts": {
            "routes": route_path.name,
            "routes_sha256": sha256(route_path),
            "extension_requests": extension_path.name,
            "extension_requests_sha256": sha256(extension_path),
        },
        "interpretation": (
            "The frozen v3 gate remains the operational decision. Every v3 "
            "downrank is preserved either for shadow review or as a machine-readable "
            "surface-extension request. No candidate is converted into an absence claim."
        ),
    }
    (output / "CT_GATE_V4_SHADOW_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
