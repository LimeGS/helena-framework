#!/usr/bin/env python3
"""Instrument every U0 decision without changing any frozen threshold.

The report joins the global strict-screen decision, each per-component v3
CT/fiber decision, and the non-destructive v4.1 route.  It is diagnostic
evidence only: it never promotes a component to ink, text, or letters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRICT_MINIMUM_CANDIDATES = 10
STRICT_MINIMUM_QUALIFYING_ROWS = 2


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def inspect_u0(qc_output: Path) -> dict[str, Any]:
    analysis_path = unique_match(
        qc_output, "robust_*/analysis/INK_STABILITY_ANALYSIS.json"
    )
    gate_path = unique_match(
        qc_output,
        "high-recall/ct_application/gate/CT_FIBER_GATE_EVALUATION.json",
    )
    shadow_receipt_path = unique_match(
        qc_output,
        "high-recall/ct_application/shadow_router_v4*/"
        "CT_GATE_V4_SHADOW_RECEIPT.json",
    )

    analysis = load(analysis_path)
    gate = load(gate_path)
    decisions_path = gate_path.parent / gate["artifacts"]["decisions"]
    if sha256(decisions_path) != gate["artifacts"]["decisions_sha256"]:
        raise RuntimeError("v3 decision hash differs from its evaluation receipt")
    decisions = load(decisions_path)
    shadow_receipt = load(shadow_receipt_path)
    routes_path = shadow_receipt_path.parent / shadow_receipt["artifacts"]["routes"]
    if sha256(routes_path) != shadow_receipt["artifacts"]["routes_sha256"]:
        raise RuntimeError("v4.1 routes hash differs from its receipt")
    routes = load(routes_path)

    text_like = analysis["text_like_screening"]
    observed_candidates = int(text_like["glyph_like_candidate_count"])
    observed_rows = int(text_like["rows_with_at_least_four_candidates"])
    strict_checks = [
        {
            "decision_id": "strict.glyph_like_candidate_count",
            "value": observed_candidates,
            "operator": ">=",
            "threshold": STRICT_MINIMUM_CANDIDATES,
            "passed": observed_candidates >= STRICT_MINIMUM_CANDIDATES,
        },
        {
            "decision_id": "strict.rows_with_at_least_four_candidates",
            "value": observed_rows,
            "operator": ">=",
            "threshold": STRICT_MINIMUM_QUALIFYING_ROWS,
            "passed": observed_rows >= STRICT_MINIMUM_QUALIFYING_ROWS,
        },
    ]
    strict_retained = all(check["passed"] for check in strict_checks)

    route_by_key = {
        (str(route["group_id"]), str(route["candidate_id"])): route
        for route in routes
    }
    if len(route_by_key) != len(routes):
        raise RuntimeError("duplicate v4.1 component route")

    components: list[dict[str, Any]] = []
    failed_feature_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for decision in decisions:
        key = (str(decision["group_id"]), str(decision["candidate_id"]))
        route = route_by_key.pop(key, None)
        if route is None:
            raise RuntimeError(f"missing v4.1 route for {key}")
        gate_checks = [
            {
                "decision_id": f"v3.{check['feature']}",
                "value": check["value"],
                "operator": check["operator"],
                "threshold": check["threshold"],
                "passed": bool(check["passed"]),
            }
            for check in decision["checks"]
        ]
        failed_feature_counts.update(map(str, decision.get("failed_features", [])))
        tier = str(route["shadow_tier"])
        tier_counts[tier] += 1
        components.append(
            {
                "group_id": key[0],
                "candidate_id": key[1],
                "decision_points": {
                    "strict_screen_global": strict_checks,
                    "ct_fiber_gate_v3": gate_checks,
                    "supported_window_router_v4_1": {
                        "tier": tier,
                        "required_action": route["required_action"],
                        "routing_reason": route["routing_reason"],
                        "not_discarded": bool(route["not_discarded"]),
                    },
                },
                "retention": {
                    "strict_screen_retained": strict_retained,
                    "v3_retained_if_component_reached": bool(decision["retained"]),
                    "strict_destructive_chain_retained": (
                        strict_retained and bool(decision["retained"])
                    ),
                    "v4_1_non_destructive_chain_preserved": bool(
                        route["not_discarded"]
                    ),
                },
            }
        )
    if route_by_key:
        raise RuntimeError("v4.1 contains routes absent from the v3 decisions")

    components.sort(key=lambda row: (row["group_id"], row["candidate_id"]))
    return {
        "schema": "campaignx.u0_threshold_chain_instrumentation.v1",
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "generated_at_utc": utc_now(),
        "inputs": {
            "analysis": {"path": str(analysis_path), "sha256": sha256(analysis_path)},
            "gate": {"path": str(gate_path), "sha256": sha256(gate_path)},
            "shadow_router": {
                "path": str(shadow_receipt_path),
                "sha256": sha256(shadow_receipt_path),
            },
        },
        "strict_screen": {
            "checks": strict_checks,
            "retained": strict_retained,
            "reported_outcome": text_like["screening_outcome"],
        },
        "component_count": len(components),
        "components": components,
        "aggregate": {
            "v3_retained_count": sum(
                component["retention"]["v3_retained_if_component_reached"]
                for component in components
            ),
            "strict_destructive_chain_retained_count": sum(
                component["retention"]["strict_destructive_chain_retained"]
                for component in components
            ),
            "v4_1_preserved_count": sum(
                component["retention"]["v4_1_non_destructive_chain_preserved"]
                for component in components
            ),
            "v3_failed_feature_counts": dict(sorted(failed_feature_counts.items())),
            "v4_1_tier_counts": dict(sorted(tier_counts.items())),
        },
        "policy": [
            "all thresholds are observed, never changed",
            "v4.1 preservation is a review route, not a positive classification",
            "no result establishes ink, text, letters, or First Letters",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qc_output = args.qc_output.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = inspect_u0(qc_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
