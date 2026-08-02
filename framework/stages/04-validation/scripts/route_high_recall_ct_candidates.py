#!/usr/bin/env python3
"""Route high-recall Phase 4 activations to CT review without accepting ink.

This is an additive companion to ``analyze_ink_stability.py``.  It does
not read, overwrite, or reinterpret the v1 routing decision.  For each window
it consumes the existing six probability maps (three sampled depths by two
tiling offsets), keeps support present in at least two depths at *each* offset,
and ranks bounded components for downstream CT localization.

The router intentionally emits small per-window and per-scroll quotas even
when the strict two-row v1 gate did not pass.  A routed component is only a
model activation selected for CT review.  It is not accepted ink, a letter, a
reading, or evidence that a scroll does or does not contain writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

_CONFIGURED_ROOT = os.environ.get("HELENA_REPO_ROOT", "").strip()
_ROOT_CANDIDATES = (
    [Path(_CONFIGURED_ROOT).expanduser().resolve()] if _CONFIGURED_ROOT else []
) + list(Path(__file__).resolve().parents)
_STAGE_ROOT = next(
    candidate / "framework/stages"
    for candidate in _ROOT_CANDIDATES
    if (candidate / "framework/stages").is_dir()
)
for _stage_scripts in _STAGE_ROOT.glob("*/scripts"):
    _stage_scripts_text = str(_stage_scripts)
    if _stage_scripts_text not in sys.path:
        sys.path.insert(0, _stage_scripts_text)
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

from replica_evidence import (
    ReplicaMapArtifact,
    discover_replica_maps,
)

MAP_PATTERN = re.compile(r"center-(\d+)_offset-(\d+)\.npy")
MANIFEST_KIND = "campaign_x_phase4_high_recall_router_manifest_v1"
MANIFEST_STATUS = "READY_FOR_HIGH_RECALL_CT_ROUTING"
ROTATION_KEYS = {
    "ROTATE_090": 1,
    "ROTATE_180": 2,
    "ROTATE_270": 3,
}
DEPTH_TTA_KEY = "DEPTH_INVERTED"
ALLOWED_TTA_KEYS = set(ROTATION_KEYS) | {DEPTH_TTA_KEY}
ORIENTATION_ANGLES_DEG = (0, 30, 60, 90, 120, 150)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def parse_map_coordinates(path: Path | ReplicaMapArtifact) -> tuple[int, int]:
    match = MAP_PATTERN.fullmatch(path.name)
    if match is None:
        raise RuntimeError(f"unexpected screening map name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_replica_grid(
    screening_dir: Path,
) -> tuple[
    np.ndarray,
    list[ReplicaMapArtifact],
    list[tuple[int, int]],
]:
    """Load one verified 3x2 grid from raw NPYs or compact evidence."""

    paths = discover_replica_maps(screening_dir)
    coordinates = [parse_map_coordinates(path) for path in paths]
    arrays = [path.load().astype(np.float32) for path in paths]
    shapes = {value.shape for value in arrays}
    if len(shapes) != 1:
        raise RuntimeError(f"{screening_dir} replica maps have different shapes")
    if arrays[0].ndim != 2:
        raise RuntimeError(f"{screening_dir} replica maps must be two-dimensional")
    return np.stack(arrays), paths, coordinates


def validate_replica_manifest_binding(
    window: dict[str, Any],
    artifacts: list[ReplicaMapArtifact],
    maps: np.ndarray,
) -> None:
    """Bind builder-generated provenance to raw or archived replica bytes.

    Legacy hand-authored manifests may omit ``provenance``. Once provenance
    exists, it is fail-closed: all six names, hashes, sizes, and shapes must
    equal the bytes that will be routed. The raw path is not compared because
    verified compaction replaces it with an immutable ZIP member.
    """

    provenance = window.get("provenance")
    if provenance is None:
        return
    if not isinstance(provenance, dict):
        raise RuntimeError("manifest provenance must be an object")
    grid = provenance.get("replica_grid")
    if not isinstance(grid, dict):
        raise RuntimeError("manifest provenance lacks replica_grid")
    rows = grid.get("maps")
    if not isinstance(rows, list) or len(rows) != 6:
        raise RuntimeError("manifest replica_grid must bind exactly six maps")
    expected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("manifest replica map row is not an object")
        name = str(row.get("file", ""))
        if MAP_PATTERN.fullmatch(name) is None or name in expected:
            raise RuntimeError(f"invalid or duplicate manifest map name: {name!r}")
        expected[name] = row
    actual = {artifact.name: artifact for artifact in artifacts}
    if set(actual) != set(expected):
        raise RuntimeError("manifest and loaded replica-map names differ")
    if maps.shape[0] != len(artifacts):
        raise RuntimeError("loaded replica count differs from artifact inventory")
    for index, artifact in enumerate(artifacts):
        row = expected[artifact.name]
        if str(row.get("sha256", "")) != artifact.sha256:
            raise RuntimeError(f"manifest replica hash mismatch: {artifact.name}")
        if int(row.get("size_bytes", -1)) != artifact.size_bytes:
            raise RuntimeError(f"manifest replica size mismatch: {artifact.name}")
        if row.get("shape_y_x") != list(maps[index].shape):
            raise RuntimeError(f"manifest replica shape mismatch: {artifact.name}")


def two_of_three_consensus(
    maps: np.ndarray,
    coordinates: list[tuple[int, int]],
) -> dict[str, Any]:
    """Build per-offset order statistics and require agreement between offsets.

    Invalid pixels are non-finite or non-positive, matching the existing Phase
    4 map convention.  The support at one offset is the second-highest valid
    probability across its three depths.  The canonical consensus is the
    minimum of those two per-offset supports.
    """

    if maps.shape[0] != len(coordinates):
        raise ValueError("map and coordinate counts differ")
    depths = sorted({depth for depth, _ in coordinates})
    offsets = sorted({offset for _, offset in coordinates})
    if len(depths) != 3 or len(offsets) != 2:
        raise ValueError("two_of_three_consensus requires 3 depths x 2 offsets")
    lookup = {coordinate: maps[index] for index, coordinate in enumerate(coordinates)}
    offset_support: dict[int, np.ndarray] = {}
    offset_valid: dict[int, np.ndarray] = {}
    offset_depth_pass_count: dict[int, np.ndarray] = {}
    for offset in offsets:
        stack = np.stack([lookup[(depth, offset)] for depth in depths])
        valid_stack = np.isfinite(stack) & (stack > 0)
        valid_count = valid_stack.sum(axis=0)
        sortable = np.where(valid_stack, stack, -np.inf)
        second_highest = np.partition(sortable, -2, axis=0)[-2]
        is_valid = valid_count >= 2
        offset_support[offset] = np.where(is_valid, second_highest, np.nan)
        offset_valid[offset] = is_valid
        offset_depth_pass_count[offset] = valid_count.astype(np.uint8)

    first = offset_support[offsets[0]]
    second = offset_support[offsets[1]]
    common_valid = offset_valid[offsets[0]] & offset_valid[offsets[1]]
    consensus = np.where(common_valid, np.minimum(first, second), np.nan)
    offset_gap = np.where(common_valid, np.abs(first - second), np.nan)
    # Ranking remains dominated by the conservative two-offset value.  The
    # small gap penalty merely breaks ties in favor of better offset agreement.
    ranking_score = np.where(common_valid, consensus - 0.10 * offset_gap, np.nan)
    return {
        "depths": depths,
        "offsets": offsets,
        "offset_support": offset_support,
        "offset_valid": offset_valid,
        "offset_valid_depth_count": offset_depth_pass_count,
        "common_valid": common_valid,
        "consensus": consensus.astype(np.float32),
        "offset_gap": offset_gap.astype(np.float32),
        "ranking_score": ranking_score.astype(np.float32),
    }


def support_mask(
    consensus: np.ndarray,
    valid: np.ndarray,
    *,
    fixed_threshold: float,
    relative_percentile: float,
    minimum_relative_threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the union of a fixed and within-window relative recall channel."""

    values = consensus[valid]
    if values.size == 0:
        raise RuntimeError("window has no pixels valid in both offsets")
    relative_value = float(np.percentile(values, relative_percentile))
    relative_threshold = max(
        float(minimum_relative_threshold),
        min(float(fixed_threshold), relative_value),
    )
    fixed = valid & (consensus >= fixed_threshold)
    relative = valid & (consensus >= relative_threshold)
    mask = fixed | relative
    below_floor_fallback = False
    if not mask.any():
        # Preserve a traceable review candidate even in a uniformly weak
        # window.  This is a quota fallback, not evidence that the maximum is
        # meaningful.  The exact observed maximum is recorded below.
        relative_threshold = float(np.max(values))
        relative = valid & (consensus >= relative_threshold)
        mask = relative
        below_floor_fallback = True
    return mask, {
        "fixed_threshold": float(fixed_threshold),
        "relative_percentile": float(relative_percentile),
        "raw_relative_percentile_value": relative_value,
        "effective_relative_threshold": relative_threshold,
        "fixed_channel_pixels": int(fixed.sum()),
        "relative_channel_pixels": int(relative.sum()),
        "union_pixels": int(mask.sum()),
        "below_minimum_threshold_fallback": below_floor_fallback,
    }


def _projection_bands(mask: np.ndarray) -> list[dict[str, Any]]:
    """Detect zero or more horizontal bands; a single band is retained."""

    if not mask.any():
        return []
    projection = ndimage.gaussian_filter1d(mask.sum(axis=1).astype(float), 2.0)
    maximum = float(projection.max())
    if maximum <= 0:
        return []
    positive = projection[projection > 0]
    baseline = float(np.percentile(positive, 55)) if positive.size else 0.0
    cutoff = max(1.5, baseline, maximum * 0.32)
    active = projection >= cutoff
    labels, count = ndimage.label(active)
    bands: list[dict[str, Any]] = []
    for label in range(1, count + 1):
        rows = np.flatnonzero(labels == label)
        if rows.size == 0:
            continue
        start = int(rows[0])
        end = int(rows[-1] + 1)
        local_peak = start + int(np.argmax(projection[start:end]))
        if end - start > max(6, round(mask.shape[0] * 0.22)):
            continue
        band_mask = mask[start:end]
        component_count = int(
            ndimage.label(
                band_mask,
                structure=np.ones((3, 3), dtype=np.uint8),
            )[1]
        )
        bands.append(
            {
                "start": start,
                "end": end,
                "peak": local_peak,
                "peak_projection": float(projection[local_peak]),
                "active_pixels": int(band_mask.sum()),
                "component_count": component_count,
            }
        )
    return bands


def detect_oriented_bands(
    mask: np.ndarray,
    *,
    angles_deg: Iterable[int] = ORIENTATION_ANGLES_DEG,
) -> list[dict[str, Any]]:
    """Audit line-like support at several orientations without using it as a gate."""

    audits: list[dict[str, Any]] = []
    for angle in angles_deg:
        normalized = int(angle) % 180
        if normalized == 0:
            rotated = mask
        elif normalized == 90:
            rotated = np.rot90(mask)
        else:
            rotated = ndimage.rotate(
                mask.astype(np.uint8),
                normalized,
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            ).astype(bool)
        bands = _projection_bands(rotated)
        audits.append(
            {
                "rotation_applied_deg": normalized,
                "source_band_axis_deg": int((-normalized) % 180),
                "band_count": len(bands),
                "single_line_allowed": True,
                "bands": bands,
                "total_band_pixels": int(sum(item["active_pixels"] for item in bands)),
            }
        )
    audits.sort(
        key=lambda item: (
            -int(item["total_band_pixels"]),
            -int(item["band_count"]),
            int(item["rotation_applied_deg"]),
        )
    )
    return audits


def _component_shape_features(
    coordinates_yx: np.ndarray,
) -> tuple[float, float, float]:
    if coordinates_yx.shape[0] < 2:
        return 1.0, 0.0, 0.0
    centered = coordinates_yx.astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = max(float(eigenvalues[-1]), 1e-9)
    minor = max(float(eigenvalues[0]), 0.0)
    elongation = math.sqrt(major / max(minor, 1e-9))
    vector_yx = eigenvectors[:, -1]
    angle_xy = math.degrees(math.atan2(vector_yx[0], vector_yx[1])) % 180
    linearity = max(0.0, min(1.0, 1.0 - minor / major))
    return float(elongation), float(linearity), float(angle_xy)


def extract_components(
    *,
    support: np.ndarray,
    consensus: np.ndarray,
    ranking_score: np.ndarray,
    offset_gap: np.ndarray,
    offset_support: dict[int, np.ndarray],
    minimum_pixels: int,
) -> list[dict[str, Any]]:
    """Describe bounded 8-connected activations and rank them deterministically."""

    labels, count = ndimage.label(
        support,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    objects = ndimage.find_objects(labels)
    sizes = np.bincount(labels.ravel())[1:]
    effective_minimum_pixels = minimum_pixels
    if sizes.size and int(sizes.max()) < minimum_pixels:
        # When every component is tiny, retain the largest size class so the
        # per-window quota still has auditable coordinates.  It remains
        # explicitly flagged below and is never called ink.
        effective_minimum_pixels = int(sizes.max())
    candidates: list[dict[str, Any]] = []
    offsets = sorted(offset_support)
    for label_index in range(1, count + 1):
        component_mask = labels == label_index
        pixels = int(component_mask.sum())
        if pixels < effective_minimum_pixels:
            continue
        bounds = objects[label_index - 1]
        if bounds is None:
            continue
        y0, y1 = int(bounds[0].start), int(bounds[0].stop)
        x0, x1 = int(bounds[1].start), int(bounds[1].stop)
        coordinates = np.argwhere(component_mask)
        center_y, center_x = coordinates.mean(axis=0)
        component_consensus = consensus[component_mask]
        component_rank = ranking_score[component_mask]
        component_gap = offset_gap[component_mask]
        elongation, linearity, angle = _component_shape_features(coordinates)
        bbox_area = max(1, (y1 - y0) * (x1 - x0))
        fill_fraction = pixels / bbox_area
        mean_probability = float(np.nanmean(component_consensus))
        peak_probability = float(np.nanmax(component_consensus))
        mean_gap = float(np.nanmean(component_gap))
        # No shape is rejected.  A modest line/compactness bonus only orders
        # equally supported components for limited CT-review capacity.
        shape_bonus = 0.04 * linearity + 0.02 * min(fill_fraction, 0.5) / 0.5
        score = (
            0.65 * float(np.nanmean(component_rank))
            + 0.35 * peak_probability
            + shape_bonus
        )
        candidates.append(
            {
                "bbox_y0_x0_y1_x1": [y0, x0, y1, x1],
                "center_y_x": [float(center_y), float(center_x)],
                "pixels": pixels,
                "below_configured_minimum_pixels": pixels < minimum_pixels,
                "bbox_fill_fraction": float(fill_fraction),
                "mean_consensus_probability": mean_probability,
                "peak_consensus_probability": peak_probability,
                "mean_offset_gap": mean_gap,
                "mean_support_by_offset": {
                    str(offset): float(
                        np.nanmean(offset_support[offset][component_mask])
                    )
                    for offset in offsets
                },
                "elongation": elongation,
                "linearity": linearity,
                "principal_axis_deg": angle,
                "routing_score": float(score),
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item["routing_score"]),
            -float(item["peak_consensus_probability"]),
            -int(item["pixels"]),
            int(item["bbox_y0_x0_y1_x1"][0]),
            int(item["bbox_y0_x0_y1_x1"][1]),
        )
    )
    return candidates


def derived_transform_audit(
    consensus: np.ndarray,
    support: np.ndarray,
    *,
    depths: list[int],
    maps: np.ndarray,
    coordinates: list[tuple[int, int]],
) -> dict[str, Any]:
    """Audit coordinate equivariance and the order-statistic depth permutation."""

    rotations: list[dict[str, Any]] = []
    for degrees, quarter_turns in ((0, 0), (90, 1), (180, 2), (270, 3)):
        transformed = np.rot90(consensus, quarter_turns)
        transformed_mask = np.rot90(support, quarter_turns)
        restored = np.rot90(transformed, -quarter_turns)
        restored_mask = np.rot90(transformed_mask, -quarter_turns)
        rotations.append(
            {
                "rotation_deg": degrees,
                "transformed_shape_y_x": list(transformed.shape),
                "score_sha256": sha256_array(transformed),
                "support_sha256": sha256_array(transformed_mask),
                "coordinate_roundtrip_score_equal": bool(
                    np.array_equal(restored, consensus, equal_nan=True)
                ),
                "coordinate_roundtrip_support_equal": bool(
                    np.array_equal(restored_mask, support)
                ),
                "component_count": int(
                    ndimage.label(
                        transformed_mask,
                        structure=np.ones((3, 3), dtype=np.uint8),
                    )[1]
                ),
            }
        )

    inverted_coordinates: list[tuple[int, int]] = []
    inverted_maps: list[np.ndarray] = []
    depth_reverse = dict(zip(depths, reversed(depths)))
    lookup = {coordinate: maps[index] for index, coordinate in enumerate(coordinates)}
    offsets = sorted({offset for _, offset in coordinates})
    for depth in depths:
        for offset in offsets:
            inverted_coordinates.append((depth, offset))
            inverted_maps.append(lookup[(depth_reverse[depth], offset)])
    inverted = two_of_three_consensus(
        np.stack(inverted_maps),
        inverted_coordinates,
    )["consensus"]
    difference = np.abs(inverted - consensus)
    finite = np.isfinite(difference)
    return {
        "kind": "POST_INFERENCE_TRANSFORM_DIAGNOSTIC_NOT_MODEL_REINFERENCE",
        "rotations": rotations,
        "depth_order_inversion": {
            "original_depth_order": depths,
            "inverted_source_depth_order": list(reversed(depths)),
            "consensus_sha256": sha256_array(inverted),
            "max_abs_difference": (
                float(difference[finite].max()) if finite.any() else None
            ),
            "order_statistic_invariant": bool(
                np.array_equal(inverted, consensus, equal_nan=True)
            ),
            "interpretation": (
                "This only verifies that the 2-of-3 order statistic is invariant "
                "to permuting already-generated depth maps. It does not measure "
                "the model response to reversing CT frames."
            ),
        },
    }


def compare_actual_tta(
    *,
    canonical_consensus: np.ndarray,
    tta_key: str,
    tta_consensus: np.ndarray,
    fixed_threshold: float,
) -> dict[str, Any]:
    if tta_key in ROTATION_KEYS:
        aligned = np.rot90(tta_consensus, -ROTATION_KEYS[tta_key])
    elif tta_key == DEPTH_TTA_KEY:
        aligned = tta_consensus
    else:
        raise ValueError(f"unsupported TTA key: {tta_key}")
    if aligned.shape != canonical_consensus.shape:
        raise RuntimeError(
            f"{tta_key} aligned shape {aligned.shape} differs from canonical "
            f"{canonical_consensus.shape}"
        )
    common = np.isfinite(canonical_consensus) & np.isfinite(aligned)
    if not common.any():
        raise RuntimeError(f"{tta_key} has no common valid pixels")
    canonical_values = canonical_consensus[common]
    aligned_values = aligned[common]
    if np.std(canonical_values) > 0 and np.std(aligned_values) > 0:
        correlation = float(np.corrcoef(canonical_values, aligned_values)[0, 1])
    else:
        correlation = None
    transformed_support = common & (aligned >= fixed_threshold)
    canonical_fixed_support = common & (canonical_consensus >= fixed_threshold)
    intersection = int((canonical_fixed_support & transformed_support).sum())
    union = int((canonical_fixed_support | transformed_support).sum())
    return {
        "transform": tta_key,
        "source_kind": "MODEL_REINFERENCE_PROVIDED_BY_MANIFEST",
        "aligned_shape_y_x": list(aligned.shape),
        "common_valid_pixels": int(common.sum()),
        "mean_absolute_probability_difference": float(
            np.mean(np.abs(canonical_values - aligned_values))
        ),
        "pearson_correlation": correlation,
        "support_iou_at_fixed_threshold": float(intersection / union) if union else 1.0,
        "support_comparison_threshold": float(fixed_threshold),
        "aligned_consensus_sha256": sha256_array(aligned),
        "used_for_routing": False,
    }


def strict_gate_reference(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "passed": None,
            "outcome": "NOT_PROVIDED",
        }
    receipt = read_json(path)
    outcome = str(
        receipt.get("text_like_screening", {}).get(
            "screening_outcome",
            "MISSING_SCREENING_OUTCOME",
        )
    )
    return {
        "available": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "kind": receipt.get("kind"),
        "passed": outcome == "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW",
        "outcome": outcome,
        "used_for_routing": False,
    }


def _safe_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "unnamed"


def analyze_window(
    *,
    window: dict[str, Any],
    manifest_path: Path,
    fixed_threshold: float,
    relative_percentile: float,
    minimum_relative_threshold: float,
    minimum_component_pixels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scroll_id = str(window.get("scroll_id", "")).strip()
    window_id = str(window.get("window_id", "")).strip()
    if not scroll_id or not window_id:
        raise RuntimeError("every window requires non-empty scroll_id and window_id")
    screening_dir = resolve_manifest_path(
        manifest_path,
        str(window["screening_dir"]),
    )
    maps, paths, coordinates = load_replica_grid(screening_dir)
    validate_replica_manifest_binding(window, paths, maps)
    aggregate = two_of_three_consensus(maps, coordinates)
    mask, threshold_audit = support_mask(
        aggregate["consensus"],
        aggregate["common_valid"],
        fixed_threshold=fixed_threshold,
        relative_percentile=relative_percentile,
        minimum_relative_threshold=minimum_relative_threshold,
    )
    bands = detect_oriented_bands(mask)
    components = extract_components(
        support=mask,
        consensus=aggregate["consensus"],
        ranking_score=aggregate["ranking_score"],
        offset_gap=aggregate["offset_gap"],
        offset_support=aggregate["offset_support"],
        minimum_pixels=minimum_component_pixels,
    )
    prefix = f"HR-{_safe_identifier(scroll_id)}-{_safe_identifier(window_id)}"
    for rank, component in enumerate(components, start=1):
        component["candidate_id"] = f"{prefix}-C{rank:03d}"
        component["component_rank_in_window"] = rank
        component["scroll_id"] = scroll_id
        component["window_id"] = window_id
        component["screening_dir"] = str(screening_dir)
        provenance = window.get("provenance")
        if isinstance(provenance, dict):
            component["physical_window_provenance"] = {
                key: provenance[key]
                for key in (
                    "global_rank",
                    "sample_id",
                    "surface_id",
                    "source_crop_xyxy",
                    "model_family",
                    "checkpoint_sha256",
                )
                if key in provenance
            }

    strict_path = None
    if window.get("strict_gate_analysis"):
        strict_path = resolve_manifest_path(
            manifest_path,
            str(window["strict_gate_analysis"]),
        )
    actual_tta: list[dict[str, Any]] = []
    tta_directories = window.get("model_tta_screening_dirs", {})
    if not isinstance(tta_directories, dict):
        raise RuntimeError("model_tta_screening_dirs must be an object")
    unknown_tta = set(tta_directories) - ALLOWED_TTA_KEYS
    if unknown_tta:
        raise RuntimeError(f"unsupported TTA transforms: {sorted(unknown_tta)}")
    tta_inputs: list[dict[str, Any]] = []
    for key in sorted(tta_directories):
        tta_dir = resolve_manifest_path(manifest_path, str(tta_directories[key]))
        tta_maps, tta_paths, tta_coordinates = load_replica_grid(tta_dir)
        tta_aggregate = two_of_three_consensus(tta_maps, tta_coordinates)
        actual_tta.append(
            compare_actual_tta(
                canonical_consensus=aggregate["consensus"],
                tta_key=key,
                tta_consensus=tta_aggregate["consensus"],
                fixed_threshold=fixed_threshold,
            )
        )
        tta_inputs.append(
            {
                "transform": key,
                "screening_dir": str(tta_dir),
                "maps": [
                    {"path": path.source, "sha256": path.sha256} for path in tta_paths
                ],
            }
        )

    strict = strict_gate_reference(strict_path)
    receipt = {
        "scroll_id": scroll_id,
        "window_id": window_id,
        "status": "COMPLETED_DIAGNOSTIC_ROUTING_ONLY",
        "screening_dir": str(screening_dir),
        "input_maps": [
            {
                "path": path.source,
                "depth_center": depth,
                "tiling_offset": offset,
                "sha256": path.sha256,
            }
            for path, (depth, offset) in zip(paths, coordinates)
        ],
        "shape_y_x": list(maps.shape[1:]),
        "depth_centers": aggregate["depths"],
        "tiling_offsets": aggregate["offsets"],
        "common_valid_pixels": int(aggregate["common_valid"].sum()),
        "aggregation": {
            "per_offset": "second-highest valid probability across 3 depths",
            "between_offsets": "minimum of the two per-offset values",
            "ranking": "consensus minus 0.10 times absolute offset gap",
            "required_depth_support": "at least 2 of 3 per offset",
            "required_offset_support": "both offsets",
        },
        "threshold_audit": threshold_audit,
        "component_count": len(components),
        "oriented_band_audit": {
            "angles_tested_deg": list(ORIENTATION_ANGLES_DEG),
            "single_line_is_retained": True,
            "best_first": bands,
        },
        "strict_v1_gate_reference": strict,
        "derived_transform_audit": derived_transform_audit(
            aggregate["consensus"],
            mask,
            depths=aggregate["depths"],
            maps=maps,
            coordinates=coordinates,
        ),
        "model_reinference_tta": {
            "provided_transforms": sorted(tta_directories),
            "required_for_complete_audit": sorted(ALLOWED_TTA_KEYS),
            "complete": set(tta_directories) == ALLOWED_TTA_KEYS,
            "used_for_routing": False,
            "comparisons": actual_tta,
            "inputs": tta_inputs,
        },
        "components": components,
    }
    return receipt, components


def apply_quotas(
    window_receipts: list[dict[str, Any]],
    *,
    top_k_per_window: int,
    top_k_per_scroll: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return per-scroll rankings and the union of both quota channels."""

    by_scroll: dict[str, list[dict[str, Any]]] = defaultdict(list)
    routed: dict[str, dict[str, Any]] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for receipt in window_receipts:
        components = list(receipt["components"])
        scroll_id = str(receipt["scroll_id"])
        by_scroll[scroll_id].extend(components)
        for item in components[:top_k_per_window]:
            candidate_id = str(item["candidate_id"])
            routed[candidate_id] = dict(item)
            reasons[candidate_id].add("TOP_K_PER_WINDOW")

    per_scroll: dict[str, list[dict[str, Any]]] = {}
    for scroll_id, components in sorted(by_scroll.items()):
        ranked = sorted(
            components,
            key=lambda item: (
                -float(item["routing_score"]),
                str(item["window_id"]),
                int(item["component_rank_in_window"]),
            ),
        )
        per_scroll[scroll_id] = ranked
        for scroll_rank, item in enumerate(ranked[:top_k_per_scroll], start=1):
            candidate_id = str(item["candidate_id"])
            routed[candidate_id] = dict(item)
            reasons[candidate_id].add("TOP_K_PER_SCROLL")
            routed[candidate_id]["component_rank_in_scroll"] = scroll_rank

    queue: list[dict[str, Any]] = []
    for candidate_id, item in routed.items():
        item["quota_reasons"] = sorted(reasons[candidate_id])
        item["routing_outcome"] = "QUEUE_FOR_RAW_CT_LOCALIZATION_REVIEW"
        item["interpretation"] = (
            "High-recall model activation only; not accepted ink or a letter."
        )
        queue.append(item)
    queue.sort(
        key=lambda item: (
            str(item["scroll_id"]),
            -float(item["routing_score"]),
            str(item["window_id"]),
            int(item["component_rank_in_window"]),
        )
    )
    return per_scroll, queue


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("kind") != MANIFEST_KIND:
        raise RuntimeError(
            f"unexpected high-recall manifest kind: {manifest.get('kind')!r}"
        )
    if manifest.get("status") != MANIFEST_STATUS:
        raise RuntimeError("high-recall manifest is not ready")
    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise RuntimeError("manifest must contain a non-empty windows list")
    if manifest.get("window_count") != len(windows):
        raise RuntimeError("manifest window_count differs from windows")
    keys: set[tuple[str, str]] = set()
    for window in windows:
        if not isinstance(window, dict):
            raise RuntimeError("each manifest window must be an object")
        for key in ("scroll_id", "window_id", "screening_dir"):
            if not str(window.get(key, "")).strip():
                raise RuntimeError(f"manifest window is missing {key}")
        identity = (str(window["scroll_id"]), str(window["window_id"]))
        if identity in keys:
            raise RuntimeError(f"duplicate scroll/window identity: {identity}")
        keys.add(identity)
        provenance = window.get("provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(
                "builder provenance is required for every manifest window"
            )
        grid = provenance.get("replica_grid")
        if not isinstance(grid, dict) or grid.get("map_count") != 6:
            raise RuntimeError("builder provenance must bind one six-map replica_grid")
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-threshold", type=float, default=0.45)
    parser.add_argument("--relative-percentile", type=float, default=92.0)
    parser.add_argument("--minimum-relative-threshold", type=float, default=0.30)
    parser.add_argument("--minimum-component-pixels", type=int, default=6)
    parser.add_argument("--top-k-per-window", type=int, default=3)
    parser.add_argument("--top-k-per-scroll", type=int, default=12)
    args = parser.parse_args()

    if not 0 < args.fixed_threshold < 1:
        raise RuntimeError("--fixed-threshold must be between zero and one")
    if not 0 < args.minimum_relative_threshold <= args.fixed_threshold:
        raise RuntimeError(
            "--minimum-relative-threshold must be positive and no greater "
            "than --fixed-threshold"
        )
    if not 0 < args.relative_percentile < 100:
        raise RuntimeError("--relative-percentile must be between zero and 100")
    if (
        min(
            args.minimum_component_pixels,
            args.top_k_per_window,
            args.top_k_per_scroll,
        )
        < 1
    ):
        raise RuntimeError("component and quota arguments must be positive")

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    windows = validate_manifest(manifest)
    output = args.output.resolve()
    receipt_path = output / "HIGH_RECALL_CT_ROUTER_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite high-recall receipt: {receipt_path}")
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"high-recall output is not a directory: {output}")
    receipts: list[dict[str, Any]] = []
    for window in windows:
        receipt, _ = analyze_window(
            window=window,
            manifest_path=manifest_path,
            fixed_threshold=args.fixed_threshold,
            relative_percentile=args.relative_percentile,
            minimum_relative_threshold=args.minimum_relative_threshold,
            minimum_component_pixels=args.minimum_component_pixels,
        )
        receipts.append(receipt)

    per_scroll, queue = apply_quotas(
        receipts,
        top_k_per_window=args.top_k_per_window,
        top_k_per_scroll=args.top_k_per_scroll,
    )
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "kind": "campaign_x_phase4_high_recall_ct_router_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_DIAGNOSTIC_ROUTING_ONLY",
        "scope": "ADDITIVE_HIGH_RECALL_CHANNEL_SEPARATE_FROM_V1",
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "parameters": {
            "fixed_threshold": args.fixed_threshold,
            "relative_percentile": args.relative_percentile,
            "minimum_relative_threshold": args.minimum_relative_threshold,
            "minimum_component_pixels": args.minimum_component_pixels,
            "top_k_per_window": args.top_k_per_window,
            "top_k_per_scroll": args.top_k_per_scroll,
            "orientation_angles_deg": list(ORIENTATION_ANGLES_DEG),
        },
        "policy": {
            "v1_outputs_modified": False,
            "v1_decision_used_as_gate": False,
            "single_line_allowed": True,
            "depth_support": "at least 2 of 3 independently at each offset",
            "offset_support": "required at both tiling offsets",
            "quota_union": "top-K per window UNION top-K per scroll",
            "model_tta_is_diagnostic_only": True,
        },
        "window_count": len(receipts),
        "scroll_count": len(per_scroll),
        "windows": receipts,
        "per_scroll_rankings": per_scroll,
        "ct_review_queue_count": len(queue),
        "ct_review_queue": queue,
        "interpretation_policy": [
            "every queued item is only a model activation for raw-CT review",
            "failure of the strict v1 row gate does not prevent quota routing",
            "one oriented band is sufficient for a line-like diagnostic",
            "neither a component nor a band is automatically accepted as ink",
            "actual rotated/depth-inverted model reinference is never used to "
            "change the canonical routing decision",
        ],
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not a papyrological reading",
            "not a First Letters submission claim",
            "not evidence that a non-routed region or scroll lacks writing",
        ],
    }
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "window_count": receipt["window_count"],
                "scroll_count": receipt["scroll_count"],
                "ct_review_queue_count": receipt["ct_review_queue_count"],
                "receipt": str(receipt_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
