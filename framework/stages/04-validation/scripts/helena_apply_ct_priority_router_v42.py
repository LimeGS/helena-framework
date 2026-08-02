#!/usr/bin/env python3
"""Apply the frozen v4.2 development router without discarding evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from helena_develop_ct_priority_router_v42 import (
    FEATURE_SCHEMA_ID,
    PROFILE_ID,
    _sigmoid,
    extract_ct_texture_features,
)

DECISION_SCORE_TOLERANCE = 1e-5


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


def apply_router(
    development_receipt_path: Path,
    tensor_path: Path,
    items_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise RuntimeError("refusing to overwrite an existing v4.2 route receipt")
    development_receipt = json.loads(
        development_receipt_path.read_text(encoding="utf-8")
    )
    if development_receipt.get("profile_id") != PROFILE_ID:
        raise RuntimeError("development receipt is not v4.2")
    model_record = development_receipt["model"]
    model_path = development_receipt_path.parent / str(model_record["artifact"])
    if sha256(model_path) != model_record["sha256"]:
        raise RuntimeError("v4.2 model hash mismatch")
    threshold = float(
        development_receipt["routing"]["fitted_development_threshold"]
    )
    model = joblib.load(model_path)
    tensor = np.load(tensor_path, mmap_mode="r")
    items = json.loads(items_path.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("items", items.get("controls", []))
    if not isinstance(items, list) or not items:
        raise RuntimeError("v4.2 input items must be a non-empty JSON list")

    routes = []
    for item in items:
        component_id = str(
            item.get("component_id", item.get("candidate_id", "UNKNOWN"))
        )
        base_tier = str(item.get("base_v4_tier", "TIER_B_SHADOW_REVIEW"))
        if base_tier == "TIER_A_V3_RETAINED_REVIEW":
            route = "TIER_A_V3_RETAINED_REVIEW"
            raw_score = None
        elif base_tier == "TIER_C_EXTEND_OR_RESEGMENT":
            route = "TIER_C_EXTEND_OR_RESEGMENT"
            raw_score = None
        else:
            index = int(item["patch_tensor_index"])
            features = extract_ct_texture_features(
                tensor[index],
                list(map(int, item["analysis_bbox_xyxy"])),
            )
            raw_score = float(model.decision_function(features[None, :])[0])
            route = (
                "TIER_B1_HIGH_PRIORITY_REVIEW"
                if raw_score + DECISION_SCORE_TOLERANCE >= threshold
                else "TIER_B2_PRESERVED_LOW_PRIORITY"
            )
        audit_sample = (
            route == "TIER_B2_PRESERVED_LOW_PRIORITY"
            and int(hashlib.sha256(component_id.encode()).hexdigest()[:8], 16) % 10
            == 0
        )
        routes.append(
            {
                "component_id": component_id,
                "base_v4_tier": base_tier,
                "priority_route": route,
                "deterministic_b2_audit_sample": audit_sample,
                "raw_decision_score": raw_score,
                "ct_priority_score": (
                    _sigmoid(raw_score) if raw_score is not None else None
                ),
                "not_discarded": True,
                "automatic_ink_claim": False,
            }
        )

    payload = {
        "schema": "campaignx.ct_priority_router_application.v1",
        "profile_id": PROFILE_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "status": "ROUTED_NONDESTRUCTIVELY",
        "generated_at_utc": utc_now(),
        "development_receipt": {
            "path": str(development_receipt_path),
            "sha256": sha256(development_receipt_path),
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "decision_score_tolerance": DECISION_SCORE_TOLERANCE,
        },
        "inputs": {
            "tensor": {
                "path": str(tensor_path),
                "sha256": sha256(tensor_path),
            },
            "items": {
                "path": str(items_path),
                "sha256": sha256(items_path),
            },
        },
        "routes": routes,
        "counts": {
            route: sum(row["priority_route"] == route for row in routes)
            for route in sorted({row["priority_route"] for row in routes})
        },
        "non_claims": [
            "B2 is preserved evidence, not a negative classification",
            "the score is not a calibrated ink probability",
            "no ink, text, letters, or First Letters are accepted automatically",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-receipt", type=Path, required=True)
    parser.add_argument("--patch-tensor", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = apply_router(
        args.development_receipt.resolve(),
        args.patch_tensor.resolve(),
        args.items.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
