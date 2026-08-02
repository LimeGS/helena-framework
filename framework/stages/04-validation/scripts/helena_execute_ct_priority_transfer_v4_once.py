#!/usr/bin/env python3
"""Evaluate frozen CT priority router v4.3 once on disjoint V4 controls.

The command is fail-closed.  It verifies every frozen input and the model
bundle, writes an irreversible execution claim, then extracts features and
routes controls within each complete official surface.  B2 is preserved
evidence, never a negative or a discard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import beta

from helena_develop_ct_priority_router_v43 import (
    FEATURE_SCHEMA_ID,
    PROFILE_ID,
    extract_compact_physical_features,
    surface_relative_ranks,
)


BENCHMARK_ID = "MULTISCROLL_TRANSFER_V4"
MINIMUM_POSITIVE_B1_RECALL_PER_SCROLL = 0.95
MINIMUM_CONFOUND_B2_RATE_PER_SCROLL = 0.15


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


def _surface_identity(control: dict[str, Any]) -> str:
    surface_id = str(control.get("official_surface_id", ""))
    if surface_id:
        return f"{control['scroll_id']}:{surface_id}"
    return str(control["surface_group_id"]).rsplit(":region-", 1)[0]


def _exact_interval(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta.ppf(0.975, successes + 1, total - successes))
    )
    return [lower, upper]


def execute(
    *,
    materialized_root: Path,
    development_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to rerun or overwrite V4 evaluation")
    output_root.mkdir(parents=True, exist_ok=True)

    materialization_path = materialized_root / "MATERIALIZATION_RECEIPT.json"
    controls_path = materialized_root / "FROZEN_CONTROLS.json"
    tensor_path = materialized_root / "CONTROL_CT_PATCHES.npy"
    required = [
        materialization_path,
        controls_path,
        tensor_path,
        development_receipt_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen V4 input: {missing}")

    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if (
        materialization.get("benchmark_id") != BENCHMARK_ID
        or materialization.get("status")
        != "OFFICIAL_CONTROLS_FROZEN_BEFORE_V3_V4"
    ):
        raise RuntimeError("materialization is not a frozen V4 benchmark")
    artifacts = materialization["artifacts"]
    if sha256(controls_path) != artifacts["frozen_controls_sha256"]:
        raise RuntimeError("V4 control hash mismatch")
    if sha256(tensor_path) != artifacts["patch_tensor_sha256"]:
        raise RuntimeError("V4 tensor hash mismatch")

    development = json.loads(
        development_receipt_path.read_text(encoding="utf-8")
    )
    if (
        development.get("profile_id") != PROFILE_ID
        or development.get("feature_schema_id") != FEATURE_SCHEMA_ID
        or development.get("status")
        != "DEVELOPMENT_FROZEN_PENDING_MULTISCROLL_TRANSFER_V4"
        or development.get("development_gates_passed") is not True
    ):
        raise RuntimeError("v4.3 development is not frozen and V4-ready")
    contamination = development["contamination_controls"]
    if any(
        contamination[key]
        for key in (
            "multiscroll_transfer_v3_used_for_training",
            "multiscroll_transfer_v3_used_for_threshold_selection",
            "multiscroll_transfer_v4_used",
        )
    ):
        raise RuntimeError("v4.3 development receipt reports contamination")
    model_record = development["model"]
    model_path = development_receipt_path.parent / str(model_record["artifact"])
    if sha256(model_path) != model_record["sha256"]:
        raise RuntimeError("v4.3 model hash mismatch")
    bundle = joblib.load(model_path)
    if (
        bundle.get("profile_id") != PROFILE_ID
        or bundle.get("feature_schema_id") != FEATURE_SCHEMA_ID
    ):
        raise RuntimeError("v4.3 model bundle identity mismatch")
    downrank_fraction = float(bundle["surface_relative_downrank_fraction"])
    minimum_cohort_size = int(bundle["minimum_routing_cohort_size"])
    model = bundle["model"]

    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    tensor = np.load(tensor_path, mmap_mode="r")
    if not isinstance(controls, list) or len(controls) != 300:
        raise RuntimeError("V4 must contain exactly 300 frozen controls")
    if int(tensor.shape[0]) != len(controls):
        raise RuntimeError("V4 tensor/control count mismatch")

    preclaim = {
        "schema": "campaignx.multiscroll_transfer_v4_priority_preclaim.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "READY_FOR_SINGLE_EXECUTION",
        "created_at_utc": utc_now(),
        "attempt": 1,
        "maximum_attempts": 1,
        "rerun_allowed": False,
        "policy": {
            "minimum_positive_b1_recall_per_scroll": (
                MINIMUM_POSITIVE_B1_RECALL_PER_SCROLL
            ),
            "minimum_confound_b2_rate_per_scroll": (
                MINIMUM_CONFOUND_B2_RATE_PER_SCROLL
            ),
            "evidence_preservation_required": 1.0,
            "tier_b2_is_preserved_not_negative": True,
            "threshold_changes_after_claim_prohibited": True,
        },
        "inputs": {
            "materialization_receipt_sha256": sha256(materialization_path),
            "controls_sha256": sha256(controls_path),
            "tensor_sha256": sha256(tensor_path),
            "development_receipt_sha256": sha256(development_receipt_path),
            "model_sha256": sha256(model_path),
        },
    }
    preclaim["content_sha256"] = canonical_sha256(
        {key: value for key, value in preclaim.items() if key != "content_sha256"}
    )
    preclaim_path = output_root / "PRECLAIM.json"
    write_json(preclaim_path, preclaim)
    claim = {
        "schema": "campaignx.multiscroll_transfer_v4_priority_claim.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "CLAIMED_SINGLE_EXECUTION",
        "claimed_at_utc": utc_now(),
        "attempt": 1,
        "maximum_attempts": 1,
        "preclaim_sha256": sha256(preclaim_path),
    }
    claim["content_sha256"] = canonical_sha256(
        {key: value for key, value in claim.items() if key != "content_sha256"}
    )
    claim_path = output_root / "EXECUTION_CLAIM.json"
    write_json(claim_path, claim)

    matrix = np.stack(
        [
            extract_compact_physical_features(
                tensor[int(control["patch_tensor_index"])],
                list(map(int, control["analysis_bbox_xyxy"])),
                voxel_size_um=list(map(float, control["voxel_size_um"])),
                patch_xy_spacing_um=(
                    float(control["patch_xy_spacing_um"])
                    if control.get("patch_xy_spacing_um") is not None
                    else None
                ),
            )
            for control in controls
        ]
    )
    scores = model.predict_proba(matrix)[:, 1]
    groups = np.asarray([_surface_identity(control) for control in controls])
    component_ids = np.asarray(
        [str(control["component_id"]) for control in controls]
    )
    ranks = surface_relative_ranks(
        scores,
        groups,
        component_ids,
        minimum_cohort_size=minimum_cohort_size,
    )
    b1 = ranks > downrank_fraction
    evaluated: list[dict[str, Any]] = []
    for control, score, rank, high in zip(
        controls,
        scores,
        ranks,
        b1,
        strict=True,
    ):
        evaluated.append(
            {
                "component_id": str(control["component_id"]),
                "scroll_id": str(control["scroll_id"]),
                "official_surface_id": str(control["official_surface_id"]),
                "surface_group_id": str(control["surface_group_id"]),
                "complete_surface_group": _surface_identity(control),
                "expected_class": str(control["expected_class"]),
                "raw_model_score": float(score),
                "surface_relative_rank": float(rank),
                "priority_route": (
                    "TIER_B1_HIGH_PRIORITY_REVIEW"
                    if high
                    else "TIER_B2_PRESERVED_LOW_PRIORITY"
                ),
                "not_discarded": True,
                "automatic_ink_claim": False,
            }
        )
    evaluated_path = output_root / "FROZEN_V4_EVALUATED_CONTROLS.json"
    write_json(evaluated_path, evaluated)

    failures: list[str] = []
    metrics_by_scroll: dict[str, Any] = {}
    for scroll in sorted({row["scroll_id"] for row in evaluated}):
        rows = [row for row in evaluated if row["scroll_id"] == scroll]
        positives = [row for row in rows if row["expected_class"] == "POSITIVE"]
        confounds = [row for row in rows if row["expected_class"] == "CONFOUND"]
        positive_recall = float(
            np.mean(
                [
                    row["priority_route"] == "TIER_B1_HIGH_PRIORITY_REVIEW"
                    for row in positives
                ]
            )
        )
        confound_b2 = float(
            np.mean(
                [
                    row["priority_route"]
                    == "TIER_B2_PRESERVED_LOW_PRIORITY"
                    for row in confounds
                ]
            )
        )
        preservation = float(np.mean([row["not_discarded"] for row in rows]))
        positive_successes = sum(
            row["priority_route"] == "TIER_B1_HIGH_PRIORITY_REVIEW"
            for row in positives
        )
        confound_successes = sum(
            row["priority_route"] == "TIER_B2_PRESERVED_LOW_PRIORITY"
            for row in confounds
        )
        metrics_by_scroll[scroll] = {
            "positive_count": len(positives),
            "positive_b1_recall": positive_recall,
            "positive_b1_exact_95": _exact_interval(
                positive_successes,
                len(positives),
            ),
            "confound_count": len(confounds),
            "confound_b2_rate": confound_b2,
            "confound_b2_exact_95": _exact_interval(
                confound_successes,
                len(confounds),
            ),
            "evidence_preservation_recall": preservation,
            "route_counts": dict(
                sorted(Counter(row["priority_route"] for row in rows).items())
            ),
        }
        if positive_recall < MINIMUM_POSITIVE_B1_RECALL_PER_SCROLL:
            failures.append(f"{scroll}:POSITIVE_B1_RECALL_BELOW_95_PERCENT")
        if confound_b2 < MINIMUM_CONFOUND_B2_RATE_PER_SCROLL:
            failures.append(f"{scroll}:CONFOUND_B2_RATE_BELOW_15_PERCENT")
        if preservation != 1.0:
            failures.append(f"{scroll}:EVIDENCE_WAS_SILENTLY_DISCARDED")

    result = {
        "schema": "campaignx.multiscroll_transfer_v4_priority_result.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": (
            "MULTISCROLL_TRANSFER_V4_PASSED"
            if not failures
            else "MULTISCROLL_TRANSFER_V4_FAILED"
        ),
        "completed_at_utc": utc_now(),
        "evaluated_component_count": len(evaluated),
        "metrics_by_scroll": metrics_by_scroll,
        "aggregate_route_counts": dict(
            sorted(Counter(row["priority_route"] for row in evaluated).items())
        ),
        "blocking_or_failure_reasons": failures,
        "promotion_decision": (
            "PROMOTE_V43_AS_DEFAULT_PRIORITY_ROUTER"
            if not failures
            else "DO_NOT_PROMOTE_V43"
        ),
        "interpretation": {
            "tier_b1": "review first; not accepted ink",
            "tier_b2": "preserved lower-priority evidence; not a negative",
            "scope": "multiroll review-order transfer, not First Letters validation",
        },
    }
    result_path = output_root / "MULTISCROLL_TRANSFER_V4_RESULT.json"
    write_json(result_path, result)
    receipt = {
        "schema": "campaignx.multiscroll_transfer_v4_priority_execution.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "ONE_SHOT_EXECUTION_COMPLETE",
        "benchmark_status": result["status"],
        "completed_at_utc": utc_now(),
        "rerun_performed": False,
        "profile_or_routing_changes_after_claim": False,
        "control_count": len(evaluated),
        "route_counts": result["aggregate_route_counts"],
        "artifacts": {
            path.name: sha256(path)
            for path in [preclaim_path, claim_path, evaluated_path, result_path]
        },
        "non_claims": [
            "No outcome accepts ink, text, letters, or First Letters.",
            "B2 is preserved evidence and never a negative classification.",
            "The model score is not a calibrated ink probability.",
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
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = execute(
        materialized_root=args.materialized_root.resolve(),
        development_receipt_path=args.development_receipt.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
