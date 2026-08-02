#!/usr/bin/env python3
"""Recall-oriented, explainable ranking of Phase 4 coarse ink windows.

This is an additive successor to ``rank_coarse_ink_windows.py``.  It
reads the same already-computed coarse probability maps but writes a separate
v2 receipt.  It does not alter v1, run inference, accept ink, or accept letters.

The v2 score preserves the exact v1 score as an audit feature, adds bounded
penalties for broad saturation and fiber-like responses, and reserves a
legacy-score rescue lane so that those heuristic penalties cannot silently
remove every high-response window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


SCHEMA_VERSION = "campaign_x_phase4_coarse_ink_window_ranking_v2"
ALGORITHM_VERSION = "recall_fiber_saturation_v2.0.0"

# These weights are deliberately explicit and bounded.  They are ranking
# heuristics, not learned probabilities and not scientific acceptance gates.
SCORE_WEIGHTS = {
    "signal": 0.70,
    "text_like_support": 0.20,
    "coverage": 0.10,
    "fiber_penalty": 0.20,
    "saturation_penalty": 0.15,
}
LEGACY_RESCUE_FRACTION = 0.25


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def physical_side_pixels(size_mm: float, pixel_um: float) -> int:
    if size_mm <= 0 or pixel_um <= 0:
        raise ValueError("physical values must be positive")
    return math.floor(size_mm * 1000.0 / pixel_um)


def fitted_source_side_pixels(
    source_shape_y_x: tuple[int, int],
    requested_side: int,
    *,
    allow_smaller_fit: bool,
) -> int:
    if requested_side < 1 or min(source_shape_y_x) < 1:
        raise ValueError("pixel dimensions must be positive")
    available = min(source_shape_y_x)
    if available < requested_side and not allow_smaller_fit:
        raise ValueError("source cannot contain the requested physical square")
    return min(requested_side, available)


def starts(length: int, side: int, step: int) -> list[int]:
    if length < side or side < 1 or step < 1:
        raise ValueError("invalid window geometry")
    values = list(range(0, length - side + 1, step))
    if values[-1] != length - side:
        values.append(length - side)
    return values


def legacy_window_score(values: np.ndarray) -> dict[str, float]:
    """Reproduce the v1 formula byte-for-byte at the numeric level."""

    if values.size == 0:
        raise ValueError("window has no valid values")
    p99, p999 = np.percentile(values, [99.0, 99.9])
    top = values[values >= p99]
    active_fraction = float((values >= 0.5).mean())
    score = (
        float(p99)
        + float(p999)
        + float(top.mean())
        + 5.0 * min(active_fraction, 0.05)
    )
    return {
        "score": score,
        "p99": float(p99),
        "p99_9": float(p999),
        "top_1pct_mean": float(top.mean()),
        "active_fraction_ge_0_5": active_fraction,
    }


def normalized_weighted_entropy(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 12,
) -> float:
    if angles.size < 2 or float(weights.sum()) <= 0:
        return 0.0
    histogram, _ = np.histogram(
        np.mod(angles, math.pi),
        bins=bins,
        range=(0.0, math.pi),
        weights=weights,
    )
    probabilities = histogram / max(float(histogram.sum()), 1e-12)
    nonzero = probabilities[probabilities > 0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return clip01(entropy / math.log(bins))


def component_features(
    high: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    """Describe high-response topology without treating it as a letter gate."""

    high = np.asarray(high, dtype=bool) & np.asarray(valid, dtype=bool)
    high_count = int(high.sum())
    valid_count = int(valid.sum())
    if high_count == 0 or valid_count == 0:
        return {
            "component_count": 0,
            "retained_component_count": 0,
            "largest_component_fraction_of_high": 0.0,
            "largest_component_fraction_of_valid": 0.0,
            "elongated_mass_fraction": 0.0,
            "component_orientation_entropy": 0.0,
            "occupied_cell_fraction_4x4": 0.0,
        }

    labels, count = ndimage.label(high, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=int)
    minimum_size = max(3, round(valid_count * 0.00005))
    retained_ids = [
        int(index + 1)
        for index in np.argsort(sizes)[::-1]
        if int(sizes[index]) >= minimum_size
    ][:512]

    angles: list[float] = []
    angle_weights: list[float] = []
    elongated_pixels = 0
    for component_id in retained_ids:
        coordinates = np.argwhere(labels == component_id)
        size = int(coordinates.shape[0])
        if size < 3:
            continue
        covariance = np.cov(coordinates.astype(np.float64), rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = max(float(eigenvalues[-1]), 1e-6)
        minor = max(float(eigenvalues[0]), 1e-6)
        aspect = math.sqrt(major / minor)
        vector = eigenvectors[:, -1]
        angles.append(float(math.atan2(vector[0], vector[1]) % math.pi))
        angle_weights.append(float(size))
        if aspect >= 4.0:
            elongated_pixels += size

    entropy = normalized_weighted_entropy(
        np.asarray(angles, dtype=np.float64),
        np.asarray(angle_weights, dtype=np.float64),
    )
    occupied = 0
    for row_index in range(4):
        y0 = round(row_index * high.shape[0] / 4)
        y1 = round((row_index + 1) * high.shape[0] / 4)
        for column_index in range(4):
            x0 = round(column_index * high.shape[1] / 4)
            x1 = round((column_index + 1) * high.shape[1] / 4)
            if np.any(high[y0:y1, x0:x1]):
                occupied += 1

    largest = int(sizes.max()) if sizes.size else 0
    return {
        "component_count": int(count),
        "retained_component_count": len(retained_ids),
        "largest_component_fraction_of_high": float(largest / high_count),
        "largest_component_fraction_of_valid": float(largest / valid_count),
        "elongated_mass_fraction": float(elongated_pixels / high_count),
        "component_orientation_entropy": entropy,
        "occupied_cell_fraction_4x4": float(occupied / 16.0),
    }


def safe_correlation(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    a = np.asarray(first[mask], dtype=np.float64)
    b = np.asarray(second[mask], dtype=np.float64)
    if a.size < 32 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ct_alignment_features(
    probability: np.ndarray,
    ct: np.ndarray | None,
    high: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | bool]:
    """Measure whether activation mostly follows bright/edge CT structures."""

    if ct is None:
        return {
            "ct_available": False,
            "ct_edge_overlap": 0.0,
            "ct_edge_alignment_excess": 0.0,
            "ct_bright_overlap": 0.0,
            "ct_bright_alignment_excess": 0.0,
            "probability_ct_edge_correlation": 0.0,
            "ct_coupling_score": 0.0,
        }
    ct = np.asarray(ct, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(ct)
    values = ct[mask]
    high_count = int((high & mask).sum())
    if values.size < 32 or high_count == 0:
        return {
            "ct_available": True,
            "ct_edge_overlap": 0.0,
            "ct_edge_alignment_excess": 0.0,
            "ct_bright_overlap": 0.0,
            "ct_bright_alignment_excess": 0.0,
            "probability_ct_edge_correlation": 0.0,
            "ct_coupling_score": 0.0,
        }

    low, high_ct = np.percentile(values, [2.0, 98.0])
    if high_ct <= low + 1e-6:
        return {
            "ct_available": True,
            "ct_edge_overlap": 0.0,
            "ct_edge_alignment_excess": 0.0,
            "ct_bright_overlap": 0.0,
            "ct_bright_alignment_excess": 0.0,
            "probability_ct_edge_correlation": 0.0,
            "ct_coupling_score": 0.0,
        }
    normalized = np.clip((ct - low) / max(float(high_ct - low), 1e-6), 0.0, 1.0)
    gradient = np.hypot(
        ndimage.sobel(normalized, axis=0),
        ndimage.sobel(normalized, axis=1),
    )
    edge_threshold = float(np.percentile(gradient[mask], 75.0))
    bright_threshold = float(np.percentile(normalized[mask], 85.0))
    edge = mask & (gradient >= edge_threshold) & (gradient > 1e-6)
    bright = mask & (normalized >= bright_threshold)
    high_mask = high & mask
    edge_overlap = float((high_mask & edge).sum() / high_count)
    bright_overlap = float((high_mask & bright).sum() / high_count)
    edge_baseline = float(edge[mask].mean())
    bright_baseline = float(bright[mask].mean())
    edge_excess = clip01(
        (edge_overlap - edge_baseline) / max(1.0 - edge_baseline, 1e-6)
    )
    bright_excess = clip01(
        (bright_overlap - bright_baseline)
        / max(1.0 - bright_baseline, 1e-6)
    )
    correlation = max(0.0, safe_correlation(probability, gradient, mask))
    coupling = clip01(max(edge_excess, bright_excess, correlation))
    return {
        "ct_available": True,
        "ct_edge_overlap": edge_overlap,
        "ct_edge_alignment_excess": edge_excess,
        "ct_bright_overlap": bright_overlap,
        "ct_bright_alignment_excess": bright_excess,
        "probability_ct_edge_correlation": correlation,
        "ct_coupling_score": coupling,
    }


def largest_component_fraction(mask: np.ndarray, valid_count: int) -> float:
    if valid_count < 1 or not np.any(mask):
        return 0.0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())[1:] if count else np.asarray([])
    return float(sizes.max() / valid_count) if sizes.size else 0.0


def extract_window_features(
    probability: np.ndarray,
    valid: np.ndarray,
    ct: np.ndarray | None,
    *,
    minimum_valid_ratio: float,
) -> dict[str, Any]:
    """Return transparent raw features, subscores, penalties, and v2 score."""

    probability = np.asarray(probability, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(probability)
    valid_ratio = float(valid.mean())
    values = probability[valid]
    if values.size == 0:
        raise ValueError("window has no valid probability values")
    q50, q90, q95, q99, q999, qmax = np.percentile(
        values,
        [50.0, 90.0, 95.0, 99.0, 99.9, 100.0],
    )
    top_1pct = values[values >= q99]
    fixed_fractions = {
        "fraction_ge_0_50": float((values >= 0.50).mean()),
        "fraction_ge_0_75": float((values >= 0.75).mean()),
        "fraction_ge_0_90": float((values >= 0.90).mean()),
    }
    adaptive_threshold = float(max(0.55, q95))
    high = valid & (probability >= adaptive_threshold)
    topology = component_features(high, valid)
    ct_features = ct_alignment_features(probability, ct, high, valid)

    absolute_tail = clip01((float(top_1pct.mean()) - 0.45) / 0.45)
    tail_lift = clip01((float(q99) - float(q50)) / 0.45)
    extreme_lift = clip01((float(q999) - float(q90)) / 0.25)
    sparse_high_support = clip01(fixed_fractions["fraction_ge_0_75"] / 0.03)
    signal_score = (
        0.40 * absolute_tail
        + 0.35 * tail_lift
        + 0.15 * extreme_lift
        + 0.10 * sparse_high_support
    )

    component_support = clip01(
        math.log1p(int(topology["retained_component_count"])) / math.log(17.0)
    )
    text_like_support = (
        0.45 * component_support
        + 0.25 * float(topology["component_orientation_entropy"])
        + 0.30 * float(topology["occupied_cell_fraction_4x4"])
    )
    coverage_score = clip01(
        (valid_ratio - minimum_valid_ratio)
        / max(1.0 - minimum_valid_ratio, 1e-6)
    )

    elongated_mass = float(topology["elongated_mass_fraction"])
    orientation_concentration = 1.0 - float(
        topology["component_orientation_entropy"]
    )
    fiber_geometry = clip01(elongated_mass * orientation_concentration)
    # Geometry alone contributes only 35% of the fiber penalty.  Strong
    # downranking requires agreement with CT edges/brightness.
    fiber_penalty = clip01(
        fiber_geometry * (0.35 + 0.65 * float(ct_features["ct_coupling_score"]))
    )

    broad_high = clip01(
        (fixed_fractions["fraction_ge_0_75"] - 0.12) / 0.28
    )
    elevated_median = clip01((float(q50) - 0.45) / 0.35)
    fixed_high = valid & (probability >= 0.75)
    largest_fixed_high = largest_component_fraction(fixed_high, int(valid.sum()))
    giant_high = clip01((largest_fixed_high - 0.12) / 0.35)
    saturation_penalty = max(broad_high, elevated_median, giant_high)

    positive_contribution = (
        SCORE_WEIGHTS["signal"] * signal_score
        + SCORE_WEIGHTS["text_like_support"] * text_like_support
        + SCORE_WEIGHTS["coverage"] * coverage_score
    )
    penalty_contribution = (
        SCORE_WEIGHTS["fiber_penalty"] * fiber_penalty
        + SCORE_WEIGHTS["saturation_penalty"] * saturation_penalty
    )
    priority_score = clip01(positive_contribution - penalty_contribution)
    legacy = legacy_window_score(values)
    return {
        "valid_ratio": valid_ratio,
        "probability": {
            "q50": float(q50),
            "q90": float(q90),
            "q95": float(q95),
            "q99": float(q99),
            "q99_9": float(q999),
            "maximum": float(qmax),
            "top_1pct_mean": float(top_1pct.mean()),
            "adaptive_component_threshold": adaptive_threshold,
            **fixed_fractions,
        },
        "topology": topology,
        "ct_alignment": ct_features,
        "subscores": {
            "absolute_tail": absolute_tail,
            "tail_lift": tail_lift,
            "extreme_lift": extreme_lift,
            "sparse_high_support": sparse_high_support,
            "signal": signal_score,
            "component_support": component_support,
            "text_like_support": text_like_support,
            "coverage": coverage_score,
        },
        "penalties": {
            "fiber_geometry": fiber_geometry,
            "fiber": fiber_penalty,
            "broad_high": broad_high,
            "elevated_median": elevated_median,
            "giant_high": giant_high,
            "largest_fixed_high_fraction_of_valid": largest_fixed_high,
            "saturation": saturation_penalty,
        },
        "score_contributions": {
            "positive": positive_contribution,
            "fiber_subtraction": SCORE_WEIGHTS["fiber_penalty"] * fiber_penalty,
            "saturation_subtraction": (
                SCORE_WEIGHTS["saturation_penalty"] * saturation_penalty
            ),
        },
        "priority_score_v2": priority_score,
        "legacy_v1_audit": legacy,
    }


def rank_percentile(rank: int, count: int) -> float:
    if count <= 1:
        return 1.0
    return float(1.0 - (rank - 1) / (count - 1))


def annotate_candidate_ranks(candidates: list[dict[str, Any]]) -> None:
    v2_order = sorted(
        candidates,
        key=lambda row: (
            -float(row["features"]["priority_score_v2"]),
            row["surface_id"],
            row["source_crop_xyxy"],
        ),
    )
    legacy_order = sorted(
        candidates,
        key=lambda row: (
            -float(row["features"]["legacy_v1_audit"]["score"]),
            row["surface_id"],
            row["source_crop_xyxy"],
        ),
    )
    count = len(candidates)
    for rank, row in enumerate(v2_order, start=1):
        row["candidate_rank_v2"] = rank
        row["candidate_percentile_v2"] = rank_percentile(rank, count)
    for rank, row in enumerate(legacy_order, start=1):
        row["candidate_rank_legacy_v1"] = rank
        row["candidate_percentile_legacy_v1"] = rank_percentile(rank, count)
    for row in candidates:
        row["recall_fused_score_v2"] = (
            0.80 * float(row["candidate_percentile_v2"])
            + 0.20 * float(row["candidate_percentile_legacy_v1"])
        )


def separated(
    candidate: dict[str, Any],
    selected: Iterable[dict[str, Any]],
    *,
    minimum_center_distance: float,
) -> bool:
    for other in selected:
        if other["surface_id"] != candidate["surface_id"]:
            continue
        dy = float(candidate["center_y_x"][0]) - float(other["center_y_x"][0])
        dx = float(candidate["center_y_x"][1]) - float(other["center_y_x"][1])
        if math.hypot(dy, dx) < minimum_center_distance:
            return False
    return True


def eligible_for_selection(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    per_surface: dict[str, int],
    *,
    max_per_surface: int,
) -> bool:
    surface_id = str(candidate["surface_id"])
    if per_surface.get(surface_id, 0) >= max_per_surface:
        return False
    return separated(
        candidate,
        selected,
        minimum_center_distance=float(candidate["normalized_side_pixels"]) * 0.5,
    )


def select_with_recall_rescue(
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    max_per_surface: int,
    rescue_fraction: float = LEGACY_RESCUE_FRACTION,
) -> list[dict[str, Any]]:
    """Select mostly by v2 while reserving a bounded exact-v1 rescue lane."""

    if top_n < 1 or max_per_surface < 1:
        raise ValueError("selection limits must be positive")
    if not 0.0 <= rescue_fraction <= 0.5:
        raise ValueError("rescue_fraction must be between 0 and 0.5")
    if candidates and "recall_fused_score_v2" not in candidates[0]:
        annotate_candidate_ranks(candidates)

    rescue_target = min(top_n, math.ceil(top_n * rescue_fraction))
    primary_target = max(0, top_n - rescue_target)
    selected: list[dict[str, Any]] = []
    per_surface: dict[str, int] = {}
    selected_ids: set[int] = set()

    fused_order = sorted(
        candidates,
        key=lambda row: (
            -float(row["recall_fused_score_v2"]),
            -float(row["features"]["priority_score_v2"]),
            row["surface_id"],
            row["source_crop_xyxy"],
        ),
    )
    legacy_order = sorted(
        candidates,
        key=lambda row: (
            -float(row["features"]["legacy_v1_audit"]["score"]),
            row["surface_id"],
            row["source_crop_xyxy"],
        ),
    )

    def add(row: dict[str, Any], lane: str) -> bool:
        if id(row) in selected_ids or not eligible_for_selection(
            row,
            selected,
            per_surface,
            max_per_surface=max_per_surface,
        ):
            return False
        row["selection_lane_v2"] = lane
        selected.append(row)
        selected_ids.add(id(row))
        surface_id = str(row["surface_id"])
        per_surface[surface_id] = per_surface.get(surface_id, 0) + 1
        return True

    for row in fused_order:
        if len(selected) >= primary_target:
            break
        add(row, "V2_PRIMARY")
    rescue_added = 0
    for row in legacy_order:
        if rescue_added >= rescue_target or len(selected) >= top_n:
            break
        if add(row, "LEGACY_V1_RECALL_RESCUE"):
            rescue_added += 1
    for row in fused_order:
        if len(selected) >= top_n:
            break
        add(row, "V2_PRIMARY_FILL")
    return selected


def load_ct_at_probability_scale(
    tiff_path: Path,
    probability_shape: tuple[int, int],
    *,
    downsample: int,
) -> np.ndarray:
    with Image.open(tiff_path) as image:
        resized = image.convert("F").resize(
            (probability_shape[1], probability_shape[0]),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32)
    return array[::downsample, ::downsample]


def draw_preview(
    tiff_path: Path,
    windows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.open(tiff_path).convert("RGB")
    scale = min(1.0, 1400.0 / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.BILINEAR,
        )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row in windows:
        x0, y0, x1, y1 = row["source_crop_xyxy"]
        box = tuple(round(value * scale) for value in (x0, y0, x1, y1))
        color = (
            "#66ddff"
            if str(row["selection_lane_v2"]).startswith("V2_PRIMARY")
            else "#ffcc33"
        )
        draw.rectangle(box, outline=color, width=3)
        label = (
            f"#{row['rank']} v2={row['features']['priority_score_v2']:.3f} "
            f"F={row['features']['penalties']['fiber']:.2f} "
            f"S={row['features']['penalties']['saturation']:.2f}"
        )
        label_width = max(180, 6 * len(label))
        draw.rectangle(
            (box[0], box[1], box[0] + label_width, box[1] + 17),
            fill="#07101d",
        )
        draw.text((box[0] + 3, box[1] + 2), label, fill=color, font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def validate_namespace(screening_name: str, output: Path, root: Path) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", screening_name):
        raise ValueError(f"unsafe screening name: {screening_name!r}")
    if output == root:
        raise ValueError("v2 output must be a dedicated child namespace")
    # It must never share a screening directory or the frozen v1 namespace,
    # even when a caller supplies an output outside the sample root.
    if output.name in {screening_name, "coarse_ranking_v1"}:
        raise ValueError("v2 output must not overwrite screening or v1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screening-name", default="coarse_screen_v1")
    parser.add_argument("--source-pixel-um", type=float, required=True)
    parser.add_argument("--normalized-pixel-um", type=float, required=True)
    parser.add_argument("--size-mm", type=float, default=20.0)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--step-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.70)
    parser.add_argument("--top-n", type=int, default=16)
    parser.add_argument("--max-per-surface", type=int, default=4)
    parser.add_argument(
        "--legacy-rescue-fraction",
        type=float,
        default=LEGACY_RESCUE_FRACTION,
    )
    parser.add_argument("--allow-smaller-fit", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    validate_namespace(args.screening_name, output, root)
    if args.downsample < 1 or not 0 < args.step_fraction <= 1:
        raise ValueError("invalid downsample or step fraction")
    if not 0 < args.minimum_valid_ratio <= 1:
        raise ValueError("minimum valid ratio must be in (0, 1]")
    output.mkdir(parents=True, exist_ok=True)

    normalized_side = physical_side_pixels(
        args.size_mm,
        args.normalized_pixel_um,
    )
    source_side = physical_side_pixels(args.size_mm, args.source_pixel_um)
    candidates: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for surface_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        surface_id = surface_dir.name
        screening_dir = surface_dir / args.screening_name
        map_path = screening_dir / "mean_probability.npy"
        receipt_path = screening_dir / "INK_SCREENING_RECEIPT.json"
        tiff_files, slice_ordering = ordered_tiff_files(
            surface_dir / "tiffs",
            allow_empty=True,
        )
        if not map_path.is_file() or not receipt_path.is_file() or not tiff_files:
            continue
        screen_receipt = read_json(receipt_path)
        if screen_receipt.get("status") != "COMPLETED_DIAGNOSTIC_ONLY":
            raise RuntimeError(f"coarse screening is not complete: {receipt_path}")

        # ``len // 2`` is the central slice of the *ordered* stack.  Before the
        # shared contract this list was lexicographic, so on an unpadded
        # 0.tif..64.tif render the "central" slice was 38.tif, not 32.tif.
        central_tiff = tiff_files[len(tiff_files) // 2]
        with Image.open(central_tiff) as image:
            source_shape_y_x = (image.height, image.width)
        try:
            surface_source_side = fitted_source_side_pixels(
                source_shape_y_x,
                source_side,
                allow_smaller_fit=args.allow_smaller_fit,
            )
        except ValueError:
            source_records.append(
                {
                    "surface_id": surface_id,
                    "coarse_receipt": str(receipt_path),
                    "coarse_receipt_sha256": sha256_file(receipt_path),
                    "map": str(map_path),
                    "map_sha256": sha256_file(map_path),
                    "source_tiff_count": len(tiff_files),
                    "slice_ordering": slice_ordering,
                    "source_shape_y_x": list(source_shape_y_x),
                    "eligible_window_count": 0,
                    "skip_reason": "source cannot contain requested physical square",
                }
            )
            continue

        probability_full = np.load(map_path).astype(np.float32)
        if probability_full.ndim != 2:
            raise RuntimeError(f"probability map must be two dimensional: {map_path}")
        reduced = probability_full[:: args.downsample, :: args.downsample]
        ct_reduced = load_ct_at_probability_scale(
            central_tiff,
            probability_full.shape,
            downsample=args.downsample,
        )
        if ct_reduced.shape != reduced.shape:
            raise RuntimeError("CT/probability shape mismatch after normalization")
        surface_normalized_side = max(
            1,
            math.floor(
                surface_source_side
                * args.source_pixel_um
                / args.normalized_pixel_um
            ),
        )
        window_side = min(
            max(1, surface_normalized_side // args.downsample),
            reduced.shape[0],
            reduced.shape[1],
        )
        valid_full = probability_full > 0
        valid_reduced = valid_full[:: args.downsample, :: args.downsample]
        normalized_scale = args.normalized_pixel_um / args.source_pixel_um
        step = max(1, round(window_side * args.step_fraction))
        surface_candidates = 0

        for y0 in starts(reduced.shape[0], window_side, step):
            for x0 in starts(reduced.shape[1], window_side, step):
                local_valid = valid_reduced[
                    y0 : y0 + window_side,
                    x0 : x0 + window_side,
                ]
                valid_ratio = float(local_valid.mean())
                if valid_ratio < args.minimum_valid_ratio:
                    continue
                local_probability = reduced[
                    y0 : y0 + window_side,
                    x0 : x0 + window_side,
                ]
                local_ct = ct_reduced[
                    y0 : y0 + window_side,
                    x0 : x0 + window_side,
                ]
                features = extract_window_features(
                    local_probability,
                    local_valid,
                    local_ct,
                    minimum_valid_ratio=args.minimum_valid_ratio,
                )

                center_y = (y0 + window_side / 2) * args.downsample
                center_x = (x0 + window_side / 2) * args.downsample
                source_center_y = round(center_y * normalized_scale)
                source_center_x = round(center_x * normalized_scale)
                source_x0 = min(
                    max(0, source_center_x - surface_source_side // 2),
                    source_shape_y_x[1] - surface_source_side,
                )
                source_y0 = min(
                    max(0, source_center_y - surface_source_side // 2),
                    source_shape_y_x[0] - surface_source_side,
                )
                candidates.append(
                    {
                        "surface_id": surface_id,
                        "center_y_x": [center_y, center_x],
                        "normalized_window_yxyx": [
                            y0 * args.downsample,
                            x0 * args.downsample,
                            (y0 + window_side) * args.downsample,
                            (x0 + window_side) * args.downsample,
                        ],
                        "source_crop_xyxy": [
                            source_x0,
                            source_y0,
                            source_x0 + surface_source_side,
                            source_y0 + surface_source_side,
                        ],
                        "source_side_pixels": surface_source_side,
                        "normalized_side_pixels": window_side * args.downsample,
                        "achieved_area_cm2": (
                            surface_source_side
                            * args.source_pixel_um
                            / 10000.0
                        )
                        ** 2,
                        "features": features,
                    }
                )
                surface_candidates += 1

        source_records.append(
            {
                "surface_id": surface_id,
                "coarse_receipt": str(receipt_path),
                "coarse_receipt_sha256": sha256_file(receipt_path),
                "map": str(map_path),
                "map_sha256": sha256_file(map_path),
                "central_tiff": str(central_tiff),
                "central_tiff_sha256": sha256_file(central_tiff),
                "source_tiff_count": len(tiff_files),
                "slice_ordering": slice_ordering,
                "source_shape_y_x": list(source_shape_y_x),
                "source_side_pixels": surface_source_side,
                "achieved_area_cm2": (
                    surface_source_side * args.source_pixel_um / 10000.0
                )
                ** 2,
                "eligible_window_count": surface_candidates,
            }
        )

    if candidates:
        annotate_candidate_ranks(candidates)
    candidate_ledger = sorted(
        candidates,
        key=lambda row: (
            int(row["candidate_rank_v2"]),
            row["surface_id"],
            row["source_crop_xyxy"],
        ),
    )
    feature_ledger_path = output / "COARSE_INK_WINDOW_FEATURES_V2.jsonl"
    write_jsonl(feature_ledger_path, candidate_ledger)
    selected = select_with_recall_rescue(
        candidates,
        top_n=args.top_n,
        max_per_surface=args.max_per_surface,
        rescue_fraction=args.legacy_rescue_fraction,
    )
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        row["within_sample_priority_percentile_v2"] = rank_percentile(
            int(row["candidate_rank_v2"]),
            len(candidates),
        )

    previews: list[dict[str, str]] = []
    for source in source_records:
        surface_id = str(source["surface_id"])
        rows = [row for row in selected if row["surface_id"] == surface_id]
        if not rows:
            continue
        central_tiff = Path(str(source["central_tiff"]))
        preview_path = output / f"{surface_id}-ranked-windows-v2.jpg"
        draw_preview(central_tiff, rows, preview_path)
        previews.append(
            {
                "surface_id": surface_id,
                "path": str(preview_path),
                "sha256": sha256_file(preview_path),
            }
        )

    receipt = {
        "kind": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_PRIORITIZATION_ONLY",
        "sample_id": args.sample_id,
        "physical_window": {
            "requested_size_mm": args.size_mm,
            "source_pixel_um": args.source_pixel_um,
            "source_side_pixels": source_side,
            "source_area_cm2": (
                source_side * args.source_pixel_um / 10000.0
            )
            ** 2,
            "normalized_pixel_um": args.normalized_pixel_um,
            "normalized_side_pixels": normalized_side,
        },
        "search": {
            "screening_name": args.screening_name,
            "downsample": args.downsample,
            "step_fraction": args.step_fraction,
            "minimum_valid_ratio": args.minimum_valid_ratio,
            "top_n": args.top_n,
            "max_per_surface": args.max_per_surface,
            "legacy_rescue_fraction": args.legacy_rescue_fraction,
            "eligible_window_count": len(candidates),
        },
        "algorithm": {
            "score_weights": SCORE_WEIGHTS,
            "fused_rank_weights": {"v2": 0.80, "legacy_v1": 0.20},
            "fiber_penalty_requires": (
                "elongated aligned response; CT coupling strengthens but is "
                "not required"
            ),
            "saturation_penalty_uses": [
                "fraction probability >= 0.75",
                "elevated probability median",
                "largest connected region probability >= 0.75",
            ],
            "selection": (
                "75% v2/fused primary lane plus 25% exact-v1 recall rescue"
            ),
        },
        "sources": source_records,
        "candidate_feature_ledger": {
            "path": str(feature_ledger_path),
            "sha256": sha256_file(feature_ledger_path),
            "row_count": len(candidate_ledger),
        },
        "ranked_windows": selected,
        "previews": previews,
        "policy": [
            "v2 consumes existing coarse maps and never runs model inference",
            "the exact v1 formula is retained as legacy_v1_audit",
            "fiber and saturation penalties are bounded ranking heuristics",
            "no feature or penalty hard-rejects a window",
            "a v1 rescue lane protects recall against heuristic misspecification",
            "each selected source crop is at or below 4 cm2",
            "selected windows still require the frozen six-replica and CT stages",
        ],
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not a First Letters submission claim",
            "not calibrated probability of ink",
            "not validated on the 13 target scrolls",
        ],
    }
    receipt_path = output / "COARSE_INK_WINDOW_RANKING_V2.json"
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "eligible_windows": len(candidates),
                "selected_windows": len(selected),
                "legacy_rescues": sum(
                    row["selection_lane_v2"] == "LEGACY_V1_RECALL_RESCUE"
                    for row in selected
                ),
                "best": selected[0] if selected else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
