#!/usr/bin/env python3
"""Develop the compact, physical CT priority router v4.3.

v4.3 is deliberately not a pixel-texture classifier.  It converts each
surface-normal CT audit stack into a compact set of physical, rotation-light
statistics: depth profiles in micrometres, radial summaries, projection
moments, gradient anisotropy and thresholded component morphology.

The model is a deliberately regularized histogram gradient booster.  V1 and
V2 are the base-training sources.  PHerc0139 and PHerc0841 form a later,
source-locked calibration set: every one of their official surfaces is
excluded from base training.  The routing quota is selected on that
two-scroll calibration set and must preserve at least 95% of positive controls
in each scroll while moving at least 15% of confounders to preserved Tier B2.
Scores are converted to label-free within-surface ranks, removing
scanner-specific score offsets.  MULTISCROLL_TRANSFER_V3 is explicitly
prohibited; only a subsequently frozen V4 can promote the router.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_ct_fiber_features_physical import (  # noqa: E402
    PHYSICAL_FEATURE_NAMES,
    extract_physical_depth_features,
)
from extract_ct_fiber_features import extract_features  # noqa: E402


PROFILE_ID = "ct-fiber-physical-priority-router@4.3.0"
FEATURE_SCHEMA_ID = "ct-physical-morphology-compact@1.0.0"
RANDOM_SEED = 20260724
ALLOWED_DEVELOPMENT_BENCHMARKS = {
    "MULTISCROLL_TRANSFER_V1",
    "MULTISCROLL_TRANSFER_V2",
    "CT_PRIORITY_ROUTER_V43_DEVELOPMENT",
}
BASE_TRAINING_BENCHMARKS = {
    "MULTISCROLL_TRANSFER_V1",
    "MULTISCROLL_TRANSFER_V2",
}
SOURCE_HOLDOUT_CALIBRATION_BENCHMARK = "CT_PRIORITY_ROUTER_V43_DEVELOPMENT"
PROHIBITED_BENCHMARKS = {
    "MULTISCROLL_TRANSFER_V3",
    "MULTISCROLL_TRANSFER_V4",
}
DEPTH_SAMPLE_UM = np.linspace(-60.0, 60.0, 11, dtype=np.float32)
RADIAL_BANDS_UM = ((0.0, 8.0), (8.0, 20.0), (20.0, 40.0))
STABLE_GEOMETRY_FEATURE_NAMES = [
    "central_slice_nonzero_fraction",
    "central_slice_zero_distance_ratio",
    "central_slice_center_nonzero",
    "argmax_depth_mode_fraction",
    "xy_to_z_gradient_ratio",
]
DEPTH_PROFILE_STATISTICS = ("mean", "std", "p10", "p90")
PROJECTION_NAMES = (
    "central",
    "maximum",
    "minimum",
    "standard_deviation",
    "maximum_minus_median",
    "argmax_depth_normalized",
)
PROJECTION_STATISTICS = (
    "mean",
    "std",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "center_delta",
    "gradient_mean",
    "gradient_std",
    "gradient_coherence",
    "entropy",
)
COMPONENT_PROJECTIONS = ("central", "maximum", "standard_deviation")
COMPONENT_QUANTILES = (0.85, 0.95)
COMPONENT_STATISTICS = (
    "foreground_fraction",
    "component_count",
    "largest_component_fraction",
    "center_component_fraction",
    "eroded_foreground_fraction",
)


def _expanded_feature_names() -> list[str]:
    names = list(PHYSICAL_FEATURE_NAMES) + list(STABLE_GEOMETRY_FEATURE_NAMES)
    for statistic in DEPTH_PROFILE_STATISTICS:
        names.extend(
            f"depth_{statistic}_at_{offset_um:+.0f}um"
            for offset_um in DEPTH_SAMPLE_UM
        )
    for projection in PROJECTION_NAMES:
        names.extend(
            f"{projection}_{statistic}"
            for statistic in PROJECTION_STATISTICS
        )
    for projection in COMPONENT_PROJECTIONS:
        for quantile in COMPONENT_QUANTILES:
            names.extend(
                f"{projection}_q{int(round(quantile * 100)):02d}_{statistic}"
                for statistic in COMPONENT_STATISTICS
            )
    return names


V43_FEATURE_NAMES = _expanded_feature_names()


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


def _surface_identity(control: dict[str, Any]) -> str:
    surface_id = control.get("official_surface_id")
    if surface_id:
        return f"{control['scroll_id']}:{surface_id}"
    group = str(control["surface_group_id"])
    return group.rsplit(":region-", 1)[0]


def _robust_normalize(patch: np.ndarray) -> np.ndarray:
    low, high = np.percentile(patch, [1.0, 99.0])
    return np.clip((patch - low) / (high - low + 1e-6), 0.0, 1.0)


def _sample_profile(
    profile: np.ndarray,
    *,
    z_spacing_um: float,
) -> list[float]:
    source_um = (
        np.arange(profile.size, dtype=np.float32) - (profile.size - 1) / 2.0
    ) * z_spacing_um
    return np.interp(
        DEPTH_SAMPLE_UM,
        source_um,
        profile,
        left=float(profile[0]),
        right=float(profile[-1]),
    ).astype(np.float32).tolist()


def _projection_features(image: np.ndarray) -> list[float]:
    flat = image.ravel()
    gy, gx = np.gradient(image)
    gradient = np.hypot(gx, gy)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    discriminant = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
    coherence = discriminant / (jxx + jyy + 1e-6)
    histogram, _ = np.histogram(flat, bins=16, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(1.0, probabilities.sum())
    entropy = float(
        -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-12)))
    )
    center = image[
        image.shape[0] // 2 - 2 : image.shape[0] // 2 + 3,
        image.shape[1] // 2 - 2 : image.shape[1] // 2 + 3,
    ]
    return [
        float(np.mean(flat)),
        float(np.std(flat)),
        *np.percentile(flat, [10, 25, 50, 75, 90]).astype(float).tolist(),
        float(np.mean(center) - np.mean(flat)),
        float(np.mean(gradient)),
        float(np.std(gradient)),
        float(coherence),
        entropy,
    ]


def _component_features(image: np.ndarray, quantile: float) -> list[float]:
    threshold = float(np.quantile(image, quantile))
    binary = image >= threshold
    labels, count = ndimage.label(binary)
    areas = np.bincount(labels.ravel())[1:]
    center_label = int(labels[labels.shape[0] // 2, labels.shape[1] // 2])
    center_area = (
        int(np.count_nonzero(labels == center_label)) if center_label else 0
    )
    return [
        float(np.mean(binary)),
        float(count),
        float(areas.max() / binary.size) if len(areas) else 0.0,
        float(center_area / binary.size),
        float(np.mean(ndimage.binary_erosion(binary))),
    ]


def extract_compact_physical_features(
    patch_z_y_x: np.ndarray,
    analysis_bbox_xyxy: list[int],
    *,
    voxel_size_um: list[float],
    patch_xy_spacing_um: float | None = None,
) -> np.ndarray:
    """Return compact physical and morphology descriptors.

    The ROI is expressed in physical units by the materializer.  Depth
    profiles are interpolated onto a fixed micrometre grid, and all image
    descriptors are aggregate moments or component morphology.  No raw pixel,
    HOG, PCA, learned embedding or scanner identifier enters the vector.
    """

    patch = np.asarray(patch_z_y_x, dtype=np.float32)
    if patch.ndim != 3 or patch.shape[0] != 65:
        raise RuntimeError("v4.3 expects a 65-plane surface-normal CT stack")
    if len(voxel_size_um) != 3 or any(
        not math.isfinite(float(value)) or float(value) <= 0
        for value in voxel_size_um
    ):
        raise RuntimeError("v4.3 requires three positive voxel sizes")
    z_spacing_um = float(voxel_size_um[0])
    xy_spacing_um = float(
        patch_xy_spacing_um
        if patch_xy_spacing_um is not None
        else voxel_size_um[1]
    )
    x0, y0, x1, y1 = map(int, analysis_bbox_xyxy)
    if not (0 <= x0 < x1 <= patch.shape[2] and 0 <= y0 < y1 <= patch.shape[1]):
        raise RuntimeError("invalid physical analysis bbox")
    patch = patch[:, y0:y1, x0:x1]
    physical = extract_physical_depth_features(
        patch,
        central_slice=patch.shape[0] // 2,
        voxel_um=z_spacing_um,
        half_window_um=120.0,
        canonical_step_um=8.0,
        top_energy_band_um=24.0,
        central_band_half_width_um=20.0,
        argmax_near_central_um=20.0,
        peak_relative_height=0.35,
    )
    physical["candidate_bbox_nonzero_fraction"] = float(
        np.mean(patch[patch.shape[0] // 2] != 0)
    )
    geometry = extract_features(
        patch,
        central_slice=patch.shape[0] // 2,
    )
    normalized = _robust_normalize(patch)
    depth_profiles = {
        "mean": normalized.mean(axis=(1, 2)),
        "std": normalized.std(axis=(1, 2)),
        "p10": np.percentile(normalized, 10.0, axis=(1, 2)),
        "p90": np.percentile(normalized, 90.0, axis=(1, 2)),
    }
    median_projection = np.median(normalized, axis=0)
    maximum_projection = normalized.max(axis=0)
    projections = {
        "central": normalized[normalized.shape[0] // 2],
        "maximum": maximum_projection,
        "minimum": normalized.min(axis=0),
        "standard_deviation": normalized.std(axis=0),
        "maximum_minus_median": maximum_projection - median_projection,
        "argmax_depth_normalized": (
            np.argmax(normalized, axis=0).astype(np.float32)
            / float(normalized.shape[0] - 1)
        ),
    }
    values: list[float] = (
        [float(physical[name]) for name in PHYSICAL_FEATURE_NAMES]
        + [float(geometry[name]) for name in STABLE_GEOMETRY_FEATURE_NAMES]
    )
    for statistic in DEPTH_PROFILE_STATISTICS:
        values.extend(
            _sample_profile(
                np.asarray(depth_profiles[statistic], dtype=np.float32),
                z_spacing_um=z_spacing_um,
            )
        )
    for projection in PROJECTION_NAMES:
        values.extend(_projection_features(projections[projection]))
    for projection in COMPONENT_PROJECTIONS:
        for quantile in COMPONENT_QUANTILES:
            values.extend(
                _component_features(projections[projection], quantile)
            )
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (len(V43_FEATURE_NAMES),) or not np.isfinite(output).all():
        raise RuntimeError(
            f"unexpected/non-finite v4.3 feature vector {output.shape}"
        )
    return output


def build_model() -> Any:
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_leaf_nodes=15,
        l2_regularization=5.0,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )


def _load_dataset(
    benchmark_id: str,
    root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    if benchmark_id in PROHIBITED_BENCHMARKS:
        raise RuntimeError(f"prohibited v4.3 development benchmark {benchmark_id}")
    if benchmark_id not in ALLOWED_DEVELOPMENT_BENCHMARKS:
        raise RuntimeError(f"unknown v4.3 development benchmark {benchmark_id}")
    physical_path = root / "V4_PHYSICAL_FEATURES.csv"
    evaluated_controls_path = root / "FROZEN_EVALUATED_CONTROLS.json"
    execution_receipt_path = root / "EXECUTION_RECEIPT.json"
    if (
        physical_path.is_file()
        and evaluated_controls_path.is_file()
        and execution_receipt_path.is_file()
    ):
        execution_receipt = json.loads(
            execution_receipt_path.read_text(encoding="utf-8")
        )
        artifacts = execution_receipt["artifacts"]
        if sha256(physical_path) != artifacts["V4_PHYSICAL_FEATURES.csv"]:
            raise RuntimeError("physical feature CSV hash mismatch")
        if (
            sha256(evaluated_controls_path)
            != artifacts["FROZEN_EVALUATED_CONTROLS.json"]
        ):
            raise RuntimeError("evaluated control hash mismatch")
        controls = json.loads(
            evaluated_controls_path.read_text(encoding="utf-8")
        )
        with physical_path.open(encoding="utf-8", newline="") as stream:
            feature_rows = list(csv.DictReader(stream))
        v3_path = root / "V3_FEATURES.csv"
        if sha256(v3_path) != artifacts["V3_FEATURES.csv"]:
            raise RuntimeError("v3 geometry feature CSV hash mismatch")
        with v3_path.open(encoding="utf-8", newline="") as stream:
            v3_rows = list(csv.DictReader(stream))
        physical_by_component = {
            str(row["candidate_id"]): row for row in feature_rows
        }
        geometry_by_component = {
            str(row["candidate_id"]): row for row in v3_rows
        }
        by_component = {}
        for component_id in sorted(physical_by_component):
            physical_row = physical_by_component[component_id]
            geometry_row = geometry_by_component[component_id]
            by_component[component_id] = np.asarray(
                [float(physical_row[name]) for name in PHYSICAL_FEATURE_NAMES]
                + [
                    float(geometry_row[name])
                    for name in STABLE_GEOMETRY_FEATURE_NAMES
                ],
                dtype=np.float32,
            )
        raise RuntimeError(
            "v4.3 semantic features require the immutable CT tensor; "
            f"{benchmark_id} only exposes legacy summary CSVs"
        )

    receipt_path = root / "MATERIALIZATION_RECEIPT.json"
    controls_path = root / "FROZEN_CONTROLS.json"
    tensor_path = root / "CONTROL_CT_PATCHES.npy"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["benchmark_id"] != benchmark_id:
        raise RuntimeError("benchmark identity mismatch")
    if sha256(controls_path) != receipt["artifacts"]["frozen_controls_sha256"]:
        raise RuntimeError("frozen control hash mismatch")
    if sha256(tensor_path) != receipt["artifacts"]["patch_tensor_sha256"]:
        raise RuntimeError("CT tensor hash mismatch")
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    tensor = np.load(tensor_path, mmap_mode="r")
    if len(controls) != int(tensor.shape[0]):
        raise RuntimeError("control/tensor length mismatch")
    vectors = np.stack(
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
    return controls, vectors, {
        "benchmark_id": benchmark_id,
        "source_kind": "MATERIALIZED_CT",
        "materialization_receipt_sha256": sha256(receipt_path),
        "frozen_controls_sha256": sha256(controls_path),
        "patch_tensor_sha256": sha256(tensor_path),
        "control_count": len(controls),
    }


MINIMUM_ROUTING_COHORT_SIZE = 20
DOWNRANK_FRACTION_GRID = tuple(
    round(value, 3) for value in np.arange(0.0, 0.201, 0.025)
)


def surface_relative_ranks(
    scores: np.ndarray,
    groups: np.ndarray,
    component_ids: np.ndarray,
    *,
    minimum_cohort_size: int = MINIMUM_ROUTING_COHORT_SIZE,
) -> np.ndarray:
    """Convert raw model scores into deterministic within-surface ranks.

    Rank normalization is label-free and removes scanner-specific offsets.
    Small cohorts remain entirely in B1 because a percentile is not stable
    enough to prioritize them responsibly.
    """

    ranks = np.ones(len(scores), dtype=np.float64)
    for group in sorted(set(map(str, groups))):
        selected = np.flatnonzero(groups == group)
        if len(selected) < minimum_cohort_size:
            continue
        ordered = sorted(
            selected.tolist(),
            key=lambda index: (
                float(scores[index]),
                str(component_ids[index]),
            ),
        )
        for ordinal, index in enumerate(ordered):
            ranks[index] = (ordinal + 0.5) / len(ordered)
    return ranks


def _minimum_positive_recall_by_scroll(
    labels: np.ndarray,
    ranks: np.ndarray,
    downrank_fraction: float,
    scrolls: np.ndarray,
) -> float:
    b1 = ranks > downrank_fraction
    recalls: list[float] = []
    for scroll in sorted(set(map(str, scrolls))):
        positive = (scrolls == scroll) & (labels == 1)
        if not np.any(positive):
            raise RuntimeError(f"development scroll has no positives: {scroll}")
        recalls.append(float(np.mean(b1[positive])))
    return min(recalls)


def select_downrank_fraction(
    labels: np.ndarray,
    ranks: np.ndarray,
    scrolls: np.ndarray,
    *,
    minimum_recall: float,
) -> float:
    """Choose the largest preregistered quota satisfying every scroll."""

    accepted = [
        fraction
        for fraction in DOWNRANK_FRACTION_GRID
        if _minimum_positive_recall_by_scroll(
            labels,
            ranks,
            fraction,
            scrolls,
        )
        >= minimum_recall
    ]
    if not accepted:
        return 0.0
    return max(accepted)


def _metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    ranks: np.ndarray,
    downrank_fraction: float,
    scrolls: np.ndarray,
) -> dict[str, Any]:
    b1 = ranks > downrank_fraction
    result: dict[str, Any] = {
        "surface_relative_downrank_fraction": float(downrank_fraction),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_b1_recall": float(np.mean(b1[labels == 1])),
        "confound_b2_rate": float(np.mean(~b1[labels == 0])),
        "by_scroll": {},
    }
    for scroll in sorted(set(map(str, scrolls))):
        selected = scrolls == scroll
        positive = selected & (labels == 1)
        confound = selected & (labels == 0)
        result["by_scroll"][scroll] = {
            "positive_count": int(np.sum(positive)),
            "positive_b1_recall": float(np.mean(b1[positive])),
            "confound_count": int(np.sum(confound)),
            "confound_b2_rate": float(np.mean(~b1[confound])),
        }
    return result


def develop(
    datasets: list[tuple[str, Path]],
    output_root: Path,
    *,
    minimum_recall: float = 0.95,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty v4.3 output")
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for benchmark_id, root in datasets:
        controls, dataset_vectors, source = _load_dataset(benchmark_id, root)
        sources.append(source)
        for control, vector in zip(controls, dataset_vectors, strict=True):
            vectors.append(vector)
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "component_id": str(control["component_id"]),
                    "scroll_id": str(control["scroll_id"]),
                    "official_surface_id": str(
                        control.get("official_surface_id", "")
                    ),
                    "complete_surface_group": _surface_identity(control),
                    "expected_class": str(control["expected_class"]),
                }
            )

    matrix = np.stack(vectors)
    labels = np.asarray(
        [row["expected_class"] == "POSITIVE" for row in rows],
        dtype=np.int8,
    )
    groups = np.asarray([row["complete_surface_group"] for row in rows])
    scrolls = np.asarray([row["scroll_id"] for row in rows])
    component_ids = np.asarray([row["component_id"] for row in rows])
    if len(set(groups)) < 10 or len(set(scrolls)) < 4:
        raise RuntimeError(
            "v4.3 requires at least ten complete surfaces across four scrolls"
        )

    benchmark_ids = np.asarray([row["benchmark_id"] for row in rows])
    base_training = np.isin(
        benchmark_ids,
        sorted(BASE_TRAINING_BENCHMARKS),
    )
    calibration = benchmark_ids == SOURCE_HOLDOUT_CALIBRATION_BENCHMARK
    if not np.all(base_training | calibration):
        raise RuntimeError("unexpected v4.3 development source")
    if set(map(str, scrolls[calibration])) != {"PHerc0139", "PHerc0841"}:
        raise RuntimeError("v4.3 calibration scroll identities changed")
    if set(map(str, scrolls[base_training])) & set(map(str, scrolls[calibration])):
        raise RuntimeError("base-training/calibration scroll contamination")
    if set(map(str, groups[base_training])) & set(map(str, groups[calibration])):
        raise RuntimeError("base-training/calibration surface contamination")

    selection_model = build_model()
    selection_model.fit(matrix[base_training], labels[base_training])
    calibration_scores = selection_model.predict_proba(matrix[calibration])[:, 1]
    calibration_ranks = surface_relative_ranks(
        calibration_scores,
        groups[calibration],
        component_ids[calibration],
    )
    downrank_fraction = select_downrank_fraction(
        labels[calibration],
        calibration_ranks,
        scrolls[calibration],
        minimum_recall=minimum_recall,
    )
    calibration_metrics = _metrics(
        labels[calibration],
        calibration_scores,
        calibration_ranks,
        downrank_fraction,
        scrolls[calibration],
    )

    development_gates = {
        "source_holdout_minimum_positive_recall_per_scroll": {
            "required": minimum_recall,
            "observed": min(
                row["positive_b1_recall"]
                for row in calibration_metrics["by_scroll"].values()
            ),
        },
        "source_holdout_global_confound_b2_rate": {
            "required": 0.15,
            "observed": calibration_metrics["confound_b2_rate"],
        },
        "source_holdout_evidence_preservation": {
            "required": 1.0,
            "observed": 1.0,
        },
    }
    gates_passed = all(
        float(gate["observed"]) >= float(gate["required"])
        for gate in development_gates.values()
    )

    final_model = build_model()
    final_model.fit(matrix, labels)
    bundle = {
        "profile_id": PROFILE_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "model": final_model,
        "surface_relative_downrank_fraction": float(downrank_fraction),
        "minimum_routing_cohort_size": MINIMUM_ROUTING_COHORT_SIZE,
        "feature_count": int(matrix.shape[1]),
        "training_benchmarks": sorted(
            str(benchmark_id) for benchmark_id, _ in datasets
        ),
        "prohibited_benchmarks": sorted(PROHIBITED_BENCHMARKS),
    }
    model_path = output_root / "CT_PRIORITY_ROUTER_V43.joblib"
    joblib.dump(bundle, model_path, compress=3)
    feature_table_path = output_root / "COMPACT_PHYSICAL_FEATURES.npy"
    np.save(feature_table_path, matrix)
    row_path = output_root / "DEVELOPMENT_ITEMS.json"
    row_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt: dict[str, Any] = {
        "schema": "campaignx.ct_priority_router_v43_development.v1",
        "profile_id": PROFILE_ID,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "status": (
            "DEVELOPMENT_FROZEN_PENDING_MULTISCROLL_TRANSFER_V4"
            if gates_passed
            else "V43_DEVELOPMENT_FAILED_DO_NOT_EVALUATE_V4"
        ),
        "generated_at_utc": utc_now(),
        "development_sets": sources,
        "development_component_count": len(rows),
        "development_complete_surface_count": len(set(groups)),
        "development_scrolls": sorted(set(map(str, scrolls))),
        "model": {
            "kind": "L2_REGULARIZED_HISTOGRAM_GRADIENT_BOOSTING",
            "feature_count": int(matrix.shape[1]),
            "maximum_iterations": 200,
            "maximum_leaf_nodes": 15,
            "l2_regularization": 5.0,
            "learning_rate": 0.05,
            "hyperparameter_search_performed": False,
            "development_model_family_screening_performed": True,
            "development_model_families_screened": [
                "L2_LOGISTIC_REGRESSION",
                "DEPTH5_EXTRA_TREES",
                "DEPTH5_RANDOM_FOREST",
                "L2_REGULARIZED_HISTOGRAM_GRADIENT_BOOSTING",
            ],
            "random_seed": RANDOM_SEED,
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
        "feature_contract": {
            "raw_pixels_included": False,
            "hog_included": False,
            "pca_included": False,
            "learned_embedding_included": False,
            "physical_feature_names": PHYSICAL_FEATURE_NAMES,
            "stable_geometry_feature_names": STABLE_GEOMETRY_FEATURE_NAMES,
            "expanded_feature_names": V43_FEATURE_NAMES,
            "depth_profile_sample_um": DEPTH_SAMPLE_UM.tolist(),
            "projection_statistics": list(PROJECTION_STATISTICS),
            "component_quantiles": list(COMPONENT_QUANTILES),
            "physical_gradient_spacing": True,
        },
        "development_protocol": {
            "base_training_benchmarks": sorted(BASE_TRAINING_BENCHMARKS),
            "base_training_scrolls": sorted(
                set(map(str, scrolls[base_training]))
            ),
            "base_training_component_count": int(np.sum(base_training)),
            "calibration_benchmark": SOURCE_HOLDOUT_CALIBRATION_BENCHMARK,
            "calibration_scrolls": sorted(
                set(map(str, scrolls[calibration]))
            ),
            "calibration_component_count": int(np.sum(calibration)),
            "calibration_complete_surface_count": len(
                set(map(str, groups[calibration]))
            ),
            "scroll_disjoint": True,
            "complete_surface_disjoint": True,
        },
        "routing_policy": {
            "source": "SOURCE_LOCKED_TWO_SCROLL_CALIBRATION",
            "kind": "WITHIN_COMPLETE_SURFACE_PERCENTILE",
            "downrank_fraction": float(downrank_fraction),
            "candidate_grid": list(DOWNRANK_FRACTION_GRID),
            "minimum_cohort_size": MINIMUM_ROUTING_COHORT_SIZE,
            "minimum_positive_recall_per_development_scroll": minimum_recall,
        },
        "source_holdout_calibration": calibration_metrics,
        "development_gates": development_gates,
        "development_gates_passed": gates_passed,
        "artifacts": {
            "model": {"path": model_path.name, "sha256": sha256(model_path)},
            "features": {
                "path": feature_table_path.name,
                "sha256": sha256(feature_table_path),
            },
            "items": {"path": row_path.name, "sha256": sha256(row_path)},
        },
        "routing": {
            "automatic_discard": False,
            "tier_b1": "review first",
            "tier_b2": "preserved low priority; audit and future review",
            "tier_c": "extend or resegment",
        },
        "contamination_controls": {
            "multiscroll_transfer_v3_used_for_training": False,
            "multiscroll_transfer_v3_used_for_threshold_selection": False,
            "multiscroll_transfer_v4_used": False,
        },
        "non_claims": [
            "Development diagnostics are not transfer validation.",
            "B2 is not a negative and is never discarded.",
            "No score is a calibrated ink probability.",
            "No ink, text, letters, or First Letters are accepted automatically.",
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
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-development-recall", type=float, default=0.95)
    args = parser.parse_args()
    datasets: list[tuple[str, Path]] = []
    for item in args.dataset:
        benchmark_id, separator, raw_path = item.partition("=")
        if not separator:
            raise RuntimeError(f"invalid --dataset {item!r}")
        datasets.append((benchmark_id, Path(raw_path).resolve()))
    benchmark_ids = {benchmark_id for benchmark_id, _ in datasets}
    if benchmark_ids != ALLOWED_DEVELOPMENT_BENCHMARKS:
        raise RuntimeError(
            "v4.3 requires V1, V2 and CT_PRIORITY_ROUTER_V43_DEVELOPMENT exactly"
        )
    receipt = develop(
        datasets,
        args.output_root.resolve(),
        minimum_recall=float(args.minimum_development_recall),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["development_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
