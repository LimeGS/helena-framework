#!/usr/bin/env python3
"""Evaluate the prospectively frozen PHerc1667 iteration-0 control A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


CONTROL_IDS = (
    "positive-pherc172",
    "negative-pherc1447-rank01",
    "negative-pherc800-rank02",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_ensemble(
    runtime: Path,
    control_id: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    directional: list[np.ndarray] = []
    coverage: list[np.ndarray] = []
    receipts: dict[str, str] = {}
    for direction in ("forward", "reverse"):
        directory = runtime / control_id / direction
        receipt_path = directory / "INFERENCE_RECEIPT.json"
        receipt = read_json(receipt_path)
        if receipt.get("status") != "COMPLETED_SCREENING_ONLY":
            raise RuntimeError(f"incomplete inference receipt: {receipt_path}")
        if receipt.get("model", {}).get("transform") != "clip-divide-255":
            raise RuntimeError(f"wrong input transform: {receipt_path}")
        directional.append(np.load(directory / "probability.npy"))
        coverage.append(np.load(directory / "prediction_count.npy") > 0)
        receipts[direction] = sha256_file(receipt_path)
    if directional[0].shape != directional[1].shape:
        raise RuntimeError(f"directional map shape mismatch: {control_id}")
    return (
        np.maximum(directional[0], directional[1]),
        coverage[0] & coverage[1],
        receipts,
    )


def evaluate_positive(
    prediction: np.ndarray,
    coverage: np.ndarray,
    label: np.ndarray,
) -> dict[str, float]:
    if prediction.shape != coverage.shape or prediction.shape != label.shape:
        raise RuntimeError("positive prediction, coverage, and label shapes differ")
    values = prediction[coverage]
    binary = label[coverage]
    if not binary.any() or binary.all():
        raise RuntimeError("positive label is single-class over covered pixels")
    return {
        "covered_fraction": float(coverage.mean()),
        "label_fraction_covered": float(binary.mean()),
        "roc_auc": float(roc_auc_score(binary, values)),
        "average_precision": float(average_precision_score(binary, values)),
        "labeled_median_probability": float(
            np.median(prediction[coverage & label])
        ),
        "unlabeled_median_probability": float(
            np.median(prediction[coverage & ~label])
        ),
        "p99_probability": float(np.quantile(values, 0.99)),
    }


def evaluate_negative(
    prediction: np.ndarray,
    coverage: np.ndarray,
) -> dict[str, float]:
    values = prediction[coverage]
    if not values.size:
        raise RuntimeError("negative control has no covered pixels")
    return {
        "covered_fraction": float(coverage.mean()),
        "median_probability": float(np.median(values)),
        "p99_probability": float(np.quantile(values, 0.99)),
        "maximum_probability": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    amendment = read_json(args.amendment)
    manifest_sha = sha256_file(args.manifest)
    if amendment.get("parent_manifest", {}).get("sha256") != manifest_sha:
        raise RuntimeError("amendment does not bind the supplied manifest")

    positive_prediction, positive_coverage, positive_receipts = load_ensemble(
        args.runtime,
        "positive-pherc172",
    )
    x, y, width, height = manifest["controls"][0]["label"][
        "crop_x_y_width_height"
    ]
    with Image.open(args.label) as opened:
        label = np.asarray(
            opened.convert("L").crop((x, y, x + width, y + height))
        ) > 0
    positive = evaluate_positive(
        positive_prediction,
        positive_coverage,
        label,
    )

    negatives: dict[str, Any] = {}
    negative_receipts: dict[str, Any] = {}
    for control_id in CONTROL_IDS[1:]:
        prediction, coverage, receipts = load_ensemble(
            args.runtime,
            control_id,
        )
        negatives[control_id] = evaluate_negative(prediction, coverage)
        negative_receipts[control_id] = receipts

    gates = {
        "positive_roc_auc_ge_0_70": positive["roc_auc"] >= 0.70,
        "positive_average_precision_ge_0_30": (
            positive["average_precision"] >= 0.30
        ),
        "positive_labeled_median_gt_pherc1447_p99": (
            positive["labeled_median_probability"]
            > negatives["negative-pherc1447-rank01"]["p99_probability"]
        ),
        "positive_labeled_median_gt_pherc800_p99": (
            positive["labeled_median_probability"]
            > negatives["negative-pherc800-rank02"]["p99_probability"]
        ),
    }
    passed = all(gates.values())
    result = {
        "kind": "campaign_x_phase4_pherc1667_iteration0_control_ab_result_v1",
        "generated_at_utc": utc_now(),
        "status": "ADOPT_FOR_TARGET_SCREENING" if passed else "REJECT_MODEL",
        "manifest": {
            "path": str(args.manifest),
            "sha256": manifest_sha,
        },
        "amendment": {
            "path": str(args.amendment),
            "sha256": sha256_file(args.amendment),
        },
        "label": {
            "path": str(args.label),
            "sha256": sha256_file(args.label),
            "crop_x_y_width_height": [x, y, width, height],
        },
        "direction_ensemble": "PIXELWISE_MAXIMUM",
        "positive": positive,
        "negatives": negatives,
        "gates": gates,
        "inference_receipt_sha256": {
            "positive-pherc172": positive_receipts,
            **negative_receipts,
        },
        "decision": (
            "The cross-segment detector is licensed for screening only."
            if passed
            else "The detector is not applied to any of the thirteen targets."
        ),
        "explicit_non_claims": [
            "the A/B does not accept ink",
            "the A/B does not identify letters",
            "rejection is detector-specific and does not prove absence of ink",
        ],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
