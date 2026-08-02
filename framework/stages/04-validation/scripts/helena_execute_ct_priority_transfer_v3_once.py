#!/usr/bin/env python3
"""Evaluate the frozen v4.2 priority router once on MULTISCROLL_TRANSFER_V3.

The executor consumes the already frozen v4.1 one-shot output, then routes the
same controls through the frozen v4.2 model.  It is deliberately fail-closed:
all input hashes are checked, an execution claim is written before scoring,
and a non-empty output directory prevents a rerun.

Tier B2 is preserved evidence, not a negative.  The benchmark evaluates
review-ordering efficiency while requiring every positive to remain reachable.
It never accepts or rejects ink, text, letters, or First Letters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from helena_apply_ct_priority_router_v42 import apply_router


BENCHMARK_ID = "MULTISCROLL_TRANSFER_V3"
MINIMUM_POSITIVE_HIGH_PRIORITY_RECALL_PER_SCROLL = 0.95
MINIMUM_CONFOUND_DOWNRANK_RATE_PER_SCROLL = 0.30
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20260724


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


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_tier(route: dict[str, Any]) -> str:
    shadow_tier = str(route["shadow_tier"])
    if shadow_tier == "TIER_A_V3_RETAINED_REVIEW":
        return shadow_tier
    if shadow_tier == "TIER_C_EXTEND_OR_RESEGMENT":
        return shadow_tier
    if shadow_tier == "TIER_B_SHADOW_REVIEW":
        return shadow_tier
    raise RuntimeError(f"unsupported frozen v4.1 route {shadow_tier}")


def _group_bootstrap_interval(
    rows: list[dict[str, Any]],
    success_key: str,
    *,
    seed: int,
) -> list[float]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["surface_group_id"])].append(row)
    groups = sorted(by_group)
    if not groups:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        selected = rng.choice(groups, size=len(groups), replace=True)
        replicate = [row for group in selected for row in by_group[str(group)]]
        samples[iteration] = np.mean(
            [bool(row[success_key]) for row in replicate]
        )
    return [
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    ]


def _metric(
    rows: list[dict[str, Any]],
    success_key: str,
    *,
    seed: int,
) -> dict[str, Any]:
    successes = sum(bool(row[success_key]) for row in rows)
    return {
        "components": len(rows),
        "successes": successes,
        "rate": successes / len(rows) if rows else 0.0,
        "surface_groups": len({row["surface_group_id"] for row in rows}),
        "surface_group_bootstrap_95": _group_bootstrap_interval(
            rows,
            success_key,
            seed=seed,
        ),
    }


def execute(
    *,
    materialized_root: Path,
    v41_execution_root: Path,
    v42_profile_path: Path,
    v42_development_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to rerun or overwrite V3 priority evaluation")
    output_root.mkdir(parents=True, exist_ok=True)

    materialization_path = materialized_root / "MATERIALIZATION_RECEIPT.json"
    controls_path = materialized_root / "FROZEN_CONTROLS.json"
    tensor_path = materialized_root / "CONTROL_CT_PATCHES.npy"
    v41_receipt_path = v41_execution_root / "EXECUTION_RECEIPT.json"
    v41_routes_path = v41_execution_root / "V4_ROUTES.json"
    v41_result_path = (
        v41_execution_root / "MULTISCROLL_TRANSFER_V3_RESULT.json"
    )
    required = [
        materialization_path,
        controls_path,
        tensor_path,
        v41_receipt_path,
        v41_routes_path,
        v41_result_path,
        v42_profile_path,
        v42_development_receipt_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen V3 inputs: {missing}")

    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if materialization.get("benchmark_id") != BENCHMARK_ID:
        raise RuntimeError("materialization is not MULTISCROLL_TRANSFER_V3")
    if materialization.get("status") != "OFFICIAL_CONTROLS_FROZEN_BEFORE_V3_V4":
        raise RuntimeError("materialization was not frozen before gate execution")
    artifacts = materialization["artifacts"]
    if sha256(controls_path) != artifacts["frozen_controls_sha256"]:
        raise RuntimeError("frozen V3 controls hash mismatch")
    if sha256(tensor_path) != artifacts["patch_tensor_sha256"]:
        raise RuntimeError("frozen V3 tensor hash mismatch")

    v41_receipt = json.loads(v41_receipt_path.read_text(encoding="utf-8"))
    v41_result = json.loads(v41_result_path.read_text(encoding="utf-8"))
    if (
        v41_receipt.get("benchmark_id") != BENCHMARK_ID
        or v41_receipt.get("status") != "ONE_SHOT_EXECUTION_COMPLETE"
    ):
        raise RuntimeError("v4.1 V3 execution is not a complete frozen one-shot")
    if v41_result.get("status") != "MULTISCROLL_TRANSFER_V3_PASSED":
        raise RuntimeError("v4.1 did not pass V3 preservation gates")
    expected_routes_hash = v41_receipt["artifacts"].get("V4_ROUTES.json")
    if expected_routes_hash != sha256(v41_routes_path):
        raise RuntimeError("frozen v4.1 routes hash mismatch")

    profile = json.loads(v42_profile_path.read_text(encoding="utf-8"))
    if profile.get("profile_id") != "ct-fiber-texture-priority-router@4.2.0":
        raise RuntimeError("wrong v4.2 profile")
    development_receipt = json.loads(
        v42_development_receipt_path.read_text(encoding="utf-8")
    )
    if development_receipt.get("profile_id") != profile["profile_id"]:
        raise RuntimeError("v4.2 profile/development receipt mismatch")

    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    routes = json.loads(v41_routes_path.read_text(encoding="utf-8"))
    if not isinstance(controls, list) or len(controls) != 300:
        raise RuntimeError("V3 must contain exactly 300 frozen controls")
    if not isinstance(routes, list) or len(routes) != len(controls):
        raise RuntimeError("v4.1 route/control count mismatch")
    by_component = {str(row["candidate_id"]): row for row in routes}
    if len(by_component) != len(routes):
        raise RuntimeError("duplicate component in frozen v4.1 routes")

    items: list[dict[str, Any]] = []
    for control in controls:
        component_id = str(control["component_id"])
        route = by_component.get(component_id)
        if route is None:
            raise RuntimeError(f"missing v4.1 route for {component_id}")
        items.append(
            {
                "component_id": component_id,
                "patch_tensor_index": int(control["patch_tensor_index"]),
                "analysis_bbox_xyxy": list(
                    map(int, control["analysis_bbox_xyxy"])
                ),
                "base_v4_tier": _base_tier(route),
            }
        )
    items_path = output_root / "FROZEN_V42_INPUT_ITEMS.json"
    write_json(items_path, items)

    preclaim = {
        "schema": "campaignx.multiscroll_transfer_v3_priority_preclaim.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "READY_FOR_SINGLE_EXECUTION",
        "created_at_utc": utc_now(),
        "attempt": 1,
        "maximum_attempts": 1,
        "rerun_allowed": False,
        "policy": {
            "minimum_positive_high_priority_recall_per_scroll": (
                MINIMUM_POSITIVE_HIGH_PRIORITY_RECALL_PER_SCROLL
            ),
            "minimum_confound_downrank_rate_per_scroll": (
                MINIMUM_CONFOUND_DOWNRANK_RATE_PER_SCROLL
            ),
            "positive_preservation_required": 1.0,
            "tier_b2_is_preserved_not_negative": True,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "threshold_changes_after_claim_prohibited": True,
        },
        "inputs": {
            "materialization_receipt_sha256": sha256(materialization_path),
            "controls_sha256": sha256(controls_path),
            "tensor_sha256": sha256(tensor_path),
            "v41_execution_receipt_sha256": sha256(v41_receipt_path),
            "v41_routes_sha256": sha256(v41_routes_path),
            "v42_profile_sha256": sha256(v42_profile_path),
            "v42_development_receipt_sha256": sha256(
                v42_development_receipt_path
            ),
            "v42_items_sha256": sha256(items_path),
        },
    }
    preclaim["content_sha256"] = canonical_sha256(
        {key: value for key, value in preclaim.items() if key != "content_sha256"}
    )
    preclaim_path = output_root / "PRECLAIM.json"
    write_json(preclaim_path, preclaim)
    claim = {
        "schema": "campaignx.multiscroll_transfer_v3_priority_execution_claim.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "CLAIMED_SINGLE_EXECUTION",
        "claimed_at_utc": utc_now(),
        "preclaim_sha256": sha256(preclaim_path),
        "attempt": 1,
        "maximum_attempts": 1,
    }
    claim["content_sha256"] = canonical_sha256(
        {key: value for key, value in claim.items() if key != "content_sha256"}
    )
    claim_path = output_root / "EXECUTION_CLAIM.json"
    write_json(claim_path, claim)

    application_path = output_root / "V42_APPLICATION_RECEIPT.json"
    application = apply_router(
        v42_development_receipt_path,
        tensor_path,
        items_path,
        application_path,
    )
    routes_by_component = {
        str(row["component_id"]): row for row in application["routes"]
    }
    evaluated: list[dict[str, Any]] = []
    for control in controls:
        component_id = str(control["component_id"])
        route = routes_by_component[component_id]
        priority_route = str(route["priority_route"])
        evaluated.append(
            {
                "component_id": component_id,
                "scroll_id": str(control["scroll_id"]),
                "surface_group_id": str(control["surface_group_id"]),
                "official_surface_id": str(control["official_surface_id"]),
                "expected_class": str(control["expected_class"]),
                "base_v4_tier": str(route["base_v4_tier"]),
                "priority_route": priority_route,
                "raw_decision_score": route["raw_decision_score"],
                "ct_priority_score": route["ct_priority_score"],
                "positive_high_priority": priority_route
                in {
                    "TIER_A_V3_RETAINED_REVIEW",
                    "TIER_B1_HIGH_PRIORITY_REVIEW",
                },
                "confound_downranked": priority_route
                == "TIER_B2_PRESERVED_LOW_PRIORITY",
                "not_discarded": bool(route["not_discarded"]),
                "automatic_ink_claim": False,
            }
        )
    evaluated_path = output_root / "FROZEN_V3_EVALUATED_CONTROLS.json"
    write_json(evaluated_path, evaluated)

    metrics_by_scroll: dict[str, Any] = {}
    failures: list[str] = []
    for scroll_index, scroll_id in enumerate(
        sorted({row["scroll_id"] for row in evaluated})
    ):
        scroll_rows = [row for row in evaluated if row["scroll_id"] == scroll_id]
        positives = [
            row for row in scroll_rows if row["expected_class"] == "POSITIVE"
        ]
        confounds = [
            row for row in scroll_rows if row["expected_class"] == "CONFOUND"
        ]
        positive_metric = _metric(
            positives,
            "positive_high_priority",
            seed=BOOTSTRAP_SEED + scroll_index,
        )
        confound_metric = _metric(
            confounds,
            "confound_downranked",
            seed=BOOTSTRAP_SEED + 100 + scroll_index,
        )
        preservation_metric = _metric(
            scroll_rows,
            "not_discarded",
            seed=BOOTSTRAP_SEED + 200 + scroll_index,
        )
        metrics_by_scroll[scroll_id] = {
            "positive_high_priority_recall": positive_metric,
            "confound_b2_downrank_rate": confound_metric,
            "evidence_preservation_recall": preservation_metric,
            "route_counts": dict(
                sorted(Counter(row["priority_route"] for row in scroll_rows).items())
            ),
        }
        if (
            positive_metric["rate"]
            < MINIMUM_POSITIVE_HIGH_PRIORITY_RECALL_PER_SCROLL
        ):
            failures.append(f"{scroll_id}:POSITIVE_B1_RECALL_BELOW_95_PERCENT")
        if (
            confound_metric["rate"]
            < MINIMUM_CONFOUND_DOWNRANK_RATE_PER_SCROLL
        ):
            failures.append(f"{scroll_id}:CONFOUND_B2_RATE_BELOW_30_PERCENT")
        if preservation_metric["rate"] != 1.0:
            failures.append(f"{scroll_id}:EVIDENCE_WAS_SILENTLY_DISCARDED")

    result = {
        "schema": "campaignx.multiscroll_transfer_v3_priority_result.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": (
            "MULTISCROLL_TRANSFER_V3_PASSED"
            if not failures
            else "MULTISCROLL_TRANSFER_V3_FAILED"
        ),
        "completed_at_utc": utc_now(),
        "evaluated_component_count": len(evaluated),
        "metrics_by_scroll": metrics_by_scroll,
        "aggregate_route_counts": dict(
            sorted(Counter(row["priority_route"] for row in evaluated).items())
        ),
        "blocking_or_failure_reasons": failures,
        "promotion_decision": (
            "PROMOTE_V42_AS_DEFAULT_PRIORITY_ROUTER"
            if not failures
            else "DO_NOT_PROMOTE_V42"
        ),
        "interpretation": {
            "tier_b1": "review first; not accepted ink",
            "tier_b2": "preserved lower-priority evidence; not a negative",
            "tier_c": "extend or resegment; not a negative",
            "scope": "review-order transfer, not First Letters validation",
        },
    }
    result_path = output_root / "MULTISCROLL_TRANSFER_V3_RESULT.json"
    write_json(result_path, result)
    receipt = {
        "schema": "campaignx.multiscroll_transfer_v3_priority_execution.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "ONE_SHOT_EXECUTION_COMPLETE",
        "completed_at_utc": utc_now(),
        "benchmark_status": result["status"],
        "rerun_performed": False,
        "profile_thresholds_changed": False,
        "control_count": len(evaluated),
        "route_counts": result["aggregate_route_counts"],
        "artifacts": {
            path.name: sha256(path)
            for path in [
                items_path,
                preclaim_path,
                claim_path,
                application_path,
                evaluated_path,
                result_path,
            ]
        },
        "non_claims": [
            "No control outcome accepts ink, text, letters, or First Letters.",
            "B2 is preserved evidence and never a negative classification.",
            "The frozen score is not a calibrated ink probability.",
        ],
    }
    receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    write_json(output_root / "EXECUTION_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--v41-execution-root", type=Path, required=True)
    parser.add_argument("--v42-profile", type=Path, required=True)
    parser.add_argument("--v42-development-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = execute(
        materialized_root=args.materialized_root.resolve(),
        v41_execution_root=args.v41_execution_root.resolve(),
        v42_profile_path=args.v42_profile.resolve(),
        v42_development_receipt_path=args.v42_development_receipt.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
