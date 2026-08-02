#!/usr/bin/env python3
"""Develop the non-destructive CT texture priority router v4.2.

This command is deliberately a *development* command.  It may consume labeled
MULTISCROLL_TRANSFER_V1/V2 controls, but it must never describe its own
cross-validation or resubstitution metrics as external transfer validation.

The router does not reject evidence.  It divides supported Tier-B evidence
into B1 (review first) and B2 (review later).  Missing physical support remains
Tier C and the historical v3 Tier A remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from skimage.feature import hog
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROFILE_ID = "ct-fiber-texture-priority-router@4.2.0"
FEATURE_SCHEMA_ID = "ct-centered-texture-morphology@1.0.0"
RANDOM_SEED = 20260724
SUPPORTED_BENCHMARKS = {
    "MULTISCROLL_TRANSFER_V1",
    "MULTISCROLL_TRANSFER_V2",
}


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


def _sigmoid(value: float) -> float:
    # A display score only.  It is not a calibrated probability.
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _validate_bbox(bbox: list[int], shape_y_x: tuple[int, int]) -> tuple[slice, slice]:
    if len(bbox) != 4:
        raise RuntimeError("analysis_bbox_xyxy must contain four integers")
    x0, y0, x1, y1 = map(int, bbox)
    if not (0 <= x0 < x1 <= shape_y_x[1] and 0 <= y0 < y1 <= shape_y_x[0]):
        raise RuntimeError(f"invalid analysis bbox {bbox} for {shape_y_x}")
    if (x1 - x0, y1 - y0) != (35, 35):
        raise RuntimeError("v4.2 requires the frozen 35x35 physical audit ROI")
    return slice(y0, y1), slice(x0, x1)


def extract_ct_texture_features(
    patch_z_y_x: np.ndarray,
    analysis_bbox_xyxy: list[int],
) -> np.ndarray:
    """Extract centered 3-D texture descriptors without using a label."""

    if patch_z_y_x.ndim != 3 or patch_z_y_x.shape[0] != 65:
        raise RuntimeError("v4.2 expects a 65-plane surface-normal CT stack")
    y_slice, x_slice = _validate_bbox(
        analysis_bbox_xyxy,
        (int(patch_z_y_x.shape[1]), int(patch_z_y_x.shape[2])),
    )
    patch = np.asarray(patch_z_y_x[:, y_slice, x_slice], dtype=np.float32)
    low, high = np.percentile(patch, [1.0, 99.0])
    patch = np.clip((patch - low) / (high - low + 1e-6), 0.0, 1.0)

    y_grid, x_grid = np.mgrid[:35, :35]
    radius = np.sqrt((y_grid - 17.0) ** 2 + (x_grid - 17.0) ** 2)
    masks = [
        radius <= 2.0,
        (radius > 2.0) & (radius <= 5.0),
        (radius > 5.0) & (radius <= 10.0),
        (radius > 10.0) & (radius <= 16.0),
    ]
    features: list[float] = []
    for mask in masks:
        values = patch[:, mask]
        features.extend(values.mean(axis=1).tolist())
        features.extend(values.std(axis=1).tolist())
        features.extend(np.percentile(values, 10.0, axis=1).tolist())
        features.extend(np.percentile(values, 90.0, axis=1).tolist())

    median_projection = np.median(patch, axis=0)
    maximum_projection = patch.max(axis=0)
    projection_maps = [
        patch[32],
        maximum_projection,
        patch.min(axis=0),
        patch.std(axis=0),
        maximum_projection - median_projection,
        np.argmax(patch, axis=0).astype(np.float32) / 64.0,
    ]
    for projection in projection_maps:
        features.extend(
            resize(
                projection,
                (12, 12),
                anti_aliasing=True,
                preserve_range=True,
            ).ravel().tolist()
        )
        features.extend(
            hog(
                projection,
                orientations=8,
                pixels_per_cell=(7, 7),
                cells_per_block=(2, 2),
                feature_vector=True,
            ).tolist()
        )

    features.extend(patch.mean(axis=(1, 2)).tolist())
    features.extend(patch.std(axis=(1, 2)).tolist())
    features.extend(np.percentile(patch, 90.0, axis=(1, 2)).tolist())
    features.extend(np.percentile(patch, 10.0, axis=(1, 2)).tolist())
    output = np.asarray(features, dtype=np.float32)
    if output.shape != (5236,) or not np.isfinite(output).all():
        raise RuntimeError(f"unexpected or non-finite v4.2 feature vector {output.shape}")
    return output


def build_model() -> Any:
    """Return the completely specified development model."""

    return make_pipeline(
        StandardScaler(),
        PCA(
            n_components=60,
            whiten=True,
            svd_solver="auto",
            random_state=RANDOM_SEED,
        ),
        SVC(
            C=1.0,
            gamma="scale",
            probability=False,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )


def _load_development_dataset(
    benchmark_id: str,
    root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    if benchmark_id not in SUPPORTED_BENCHMARKS:
        raise RuntimeError(f"unsupported development benchmark {benchmark_id}")
    receipt_path = root / "MATERIALIZATION_RECEIPT.json"
    controls_path = root / "FROZEN_CONTROLS.json"
    tensor_path = root / "CONTROL_CT_PATCHES.npy"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("benchmark_id") != benchmark_id:
        raise RuntimeError(f"benchmark mismatch in {receipt_path}")
    artifacts = receipt["artifacts"]
    if sha256(controls_path) != artifacts["frozen_controls_sha256"]:
        raise RuntimeError(f"frozen controls hash mismatch in {root}")
    if sha256(tensor_path) != artifacts["patch_tensor_sha256"]:
        raise RuntimeError(f"CT tensor hash mismatch in {root}")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    tensor = np.load(tensor_path, mmap_mode="r")
    if len(controls) != int(tensor.shape[0]):
        raise RuntimeError(f"control/tensor length mismatch in {root}")
    return controls, tensor, {
        "benchmark_id": benchmark_id,
        "materialization_receipt_sha256": sha256(receipt_path),
        "controls_sha256": sha256(controls_path),
        "tensor_sha256": sha256(tensor_path),
        "tensor_shape": list(map(int, tensor.shape)),
    }


def _threshold_for_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    recall_strata: np.ndarray,
    minimum_recall: float,
) -> float:
    """Choose one threshold satisfying every benchmark-by-scroll stratum."""

    thresholds: list[float] = []
    for stratum in sorted(set(map(str, recall_strata))):
        positive_scores = scores[(recall_strata == stratum) & (labels == 1)]
        if not len(positive_scores):
            continue
        # At most floor((1-recall)*N) positives may sit below the threshold.
        allowed_below = int(math.floor((1.0 - minimum_recall) * len(positive_scores)))
        ordered = np.sort(positive_scores)
        thresholds.append(float(ordered[min(allowed_below, len(ordered) - 1)]))
    if not thresholds:
        raise RuntimeError("no positive development scores")
    return min(thresholds)


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    scrolls: np.ndarray,
) -> dict[str, Any]:
    high = scores >= threshold
    payload: dict[str, Any] = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_high_priority_recall": float(high[labels == 1].mean()),
        "confound_high_priority_rate": float(high[labels == 0].mean()),
        "confound_downrank_rate": float((~high[labels == 0]).mean()),
        "total_high_priority_rate": float(high.mean()),
        "by_scroll": {},
    }
    for scroll_id in sorted(set(map(str, scrolls))):
        selected = scrolls == scroll_id
        positive = selected & (labels == 1)
        confound = selected & (labels == 0)
        payload["by_scroll"][scroll_id] = {
            "positive_count": int(positive.sum()),
            "positive_high_priority_recall": float(high[positive].mean()),
            "confound_count": int(confound.sum()),
            "confound_high_priority_rate": float(high[confound].mean()),
            "confound_downrank_rate": float((~high[confound]).mean()),
        }
    return payload


def develop(
    datasets: list[tuple[str, Path]],
    output_root: Path,
    *,
    minimum_recall: float,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty v4.2 development output")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for benchmark_id, root in datasets:
        controls, tensor, source = _load_development_dataset(benchmark_id, root)
        sources.append(source)
        for control in controls:
            index = int(control["patch_tensor_index"])
            vectors.append(
                extract_ct_texture_features(
                    tensor[index],
                    list(map(int, control["analysis_bbox_xyxy"])),
                )
            )
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "component_id": str(control["component_id"]),
                    "surface_group_id": str(control["surface_group_id"]),
                    "scroll_id": str(control["scroll_id"]),
                    "expected_class": str(control["expected_class"]),
                }
            )

    matrix = np.stack(vectors)
    labels = np.asarray(
        [row["expected_class"] == "POSITIVE" for row in rows],
        dtype=np.int8,
    )
    groups = np.asarray([row["surface_group_id"] for row in rows])
    scrolls = np.asarray([row["scroll_id"] for row in rows])
    recall_strata = np.asarray(
        [f"{row['benchmark_id']}:{row['scroll_id']}" for row in rows]
    )
    cross_validation = StratifiedGroupKFold(n_splits=5, shuffle=False)
    out_of_fold_scores = cross_val_predict(
        build_model(),
        matrix,
        labels,
        groups=groups,
        cv=cross_validation,
        method="decision_function",
        n_jobs=1,
    )
    out_of_fold_threshold = _threshold_for_recall(
        out_of_fold_scores,
        labels,
        recall_strata,
        minimum_recall,
    )

    model = build_model()
    model.fit(matrix, labels)
    fitted_scores = model.decision_function(matrix)
    fitted_threshold = _threshold_for_recall(
        fitted_scores,
        labels,
        recall_strata,
        minimum_recall,
    )
    model_path = output_root / "CT_PRIORITY_ROUTER_V42.joblib"
    joblib.dump(model, model_path, compress=3)

    decisions: list[dict[str, Any]] = []
    for row, score in zip(rows, fitted_scores, strict=True):
        decisions.append(
            {
                **row,
                "ct_priority_score": _sigmoid(float(score)),
                "raw_decision_score": float(score),
                "priority_route": (
                    "TIER_B1_HIGH_PRIORITY_REVIEW"
                    if score >= fitted_threshold
                    else "TIER_B2_PRESERVED_LOW_PRIORITY"
                ),
                "not_discarded": True,
            }
        )
    decisions_path = output_root / "DEVELOPMENT_ROUTES.json"
    decisions_path.write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    compact_features_path = output_root / "DEVELOPMENT_SCORE_TABLE.csv"
    with compact_features_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(decisions[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(decisions)

    receipt: dict[str, Any] = {
        "schema": "campaignx.ct_priority_router_development.v1",
        "profile_id": PROFILE_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "status": "DEVELOPMENT_OPTIMIZED_NOT_EXTERNALLY_VALIDATED",
        "generated_at_utc": utc_now(),
        "development_sets": sources,
        "development_component_count": len(rows),
        "development_surface_group_count": len(set(groups)),
        "development_scrolls": sorted(set(map(str, scrolls))),
        "model": {
            "kind": "STANDARD_SCALE_PCA60_RBF_SVC",
            "random_seed": RANDOM_SEED,
            "feature_count": int(matrix.shape[1]),
            "artifact": model_path.name,
            "sha256": sha256(model_path),
            "dependencies": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
            },
        },
        "routing": {
            "historical_tier_a_unchanged": True,
            "tier_b1": "review first",
            "tier_b2": "preserved; review later or sample for audit",
            "tier_c": "extend or resegment; not a negative",
            "fitted_development_threshold": float(fitted_threshold),
            "score_is_calibrated_probability": False,
            "automatic_discard": False,
        },
        "grouped_out_of_fold_diagnostic": _metrics(
            labels,
            out_of_fold_scores,
            out_of_fold_threshold,
            scrolls,
        ),
        "fitted_development_diagnostic": _metrics(
            labels,
            fitted_scores,
            fitted_threshold,
            scrolls,
        ),
        "artifacts": {
            "routes": {
                "path": decisions_path.name,
                "sha256": sha256(decisions_path),
            },
            "score_table": {
                "path": compact_features_path.name,
                "sha256": sha256(compact_features_path),
            },
        },
        "limitations": [
            "V1 and V2 are development data for v4.2.",
            "Grouped cross-validation is an internal diagnostic, not transfer proof.",
            "The fitted threshold is intentionally optimized on known labels.",
            "Only a spatially disjoint MULTISCROLL_TRANSFER_V3 may estimate transfer.",
            "B2 evidence is downranked but never rejected or deleted.",
        ],
        "non_claims": [
            "not externally validated",
            "not a calibrated ink probability",
            "not accepted ink, text, letters, or First Letters",
            "not evidence of absence",
        ],
    }
    receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    receipt_path = output_root / "DEVELOPMENT_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="BENCHMARK_ID=PATH",
        help="Repeat for each labeled development materialization.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-development-recall", type=float, default=0.95)
    args = parser.parse_args()
    if not 0.0 < args.minimum_development_recall <= 1.0:
        raise RuntimeError("minimum development recall must be in (0, 1]")
    datasets: list[tuple[str, Path]] = []
    for item in args.dataset:
        benchmark_id, separator, raw_path = item.partition("=")
        if not separator:
            raise RuntimeError(f"invalid --dataset {item!r}")
        datasets.append((benchmark_id, Path(raw_path).resolve()))
    if {benchmark_id for benchmark_id, _ in datasets} != SUPPORTED_BENCHMARKS:
        raise RuntimeError("v4.2 development requires V1 and V2 exactly once")
    receipt = develop(
        datasets,
        args.output_root.resolve(),
        minimum_recall=float(args.minimum_development_recall),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
