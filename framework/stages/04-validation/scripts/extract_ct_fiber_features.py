#!/usr/bin/env python3
"""Extract physically normalized depth-localization features for ink screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
ROOT = _STAGE_ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

import numpy as np
from PIL import Image

from framework.contracts.slice_order import (
    NUMERIC_STEM_INDEX,
    ordered_tiff_files,
    ordered_tiff_stack_position,
)
from build_orthogonal_candidate_review import (
    map_analysis_point_to_source,
)

BASE_FIELDS = [
    "group_id",
    "class",
    "candidate_id",
    "analysis_bbox_x0",
    "analysis_bbox_y0",
    "analysis_bbox_x1",
    "analysis_bbox_y1",
    "analysis_center_y",
    "analysis_center_x",
    "source_bbox_x0",
    "source_bbox_y0",
    "source_bbox_x1",
    "source_bbox_y1",
    "source_center_y",
    "source_center_x",
    "patch_radius_pixels",
    "patch_radius_um",
]

FEATURE_NAMES = [
    "candidate_bbox_nonzero_fraction",
    "central_slice_nonzero_fraction",
    "central_slice_zero_distance_ratio",
    "central_slice_center_nonzero",
    "depth_profile_top1_fraction",
    "depth_profile_top3_fraction",
    "depth_profile_entropy",
    "depth_profile_peak_count",
    "central_depth_energy_fraction",
    "argmax_depth_near_central_fraction",
    "argmax_depth_mode_fraction",
    "argmax_depth_p90_p10_span",
    "xy_to_z_gradient_ratio",
    "surface_localization_score",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stack(directory: Path) -> tuple[np.ndarray, list[Path], str]:
    files, ordering = ordered_tiff_files(directory)
    arrays = [np.asarray(Image.open(path), dtype=np.float32) for path in files]
    if len({array.shape for array in arrays}) != 1:
        raise RuntimeError(f"TIFF shape mismatch in {directory}")
    return np.stack(arrays), files, ordering


def crop_stack_with_padding(
    stack: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError("stack must be depth,y,x")
    size = radius * 2 + 1
    output = np.zeros((stack.shape[0], size, size), dtype=stack.dtype)
    y0 = max(0, center_y - radius)
    y1 = min(stack.shape[1], center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(stack.shape[2], center_x + radius + 1)
    oy0 = y0 - (center_y - radius)
    ox0 = x0 - (center_x - radius)
    output[:, oy0 : oy0 + y1 - y0, ox0 : ox0 + x1 - x0] = stack[
        :, y0:y1, x0:x1
    ]
    return output


def map_analysis_bbox_to_source(
    *,
    bbox_xyxy: tuple[int, int, int, int],
    analysis_shape_y_x: tuple[int, int],
    source_shape_y_x: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map one half-open analysis bbox into a half-open source CT bbox.

    Starts use ``floor`` and ends use ``ceil`` so every source pixel touched by
    the analysis-space candidate remains inside the audited region.  The
    result is clamped to the source extent and an empty mapped region fails
    closed instead of silently reporting full support.
    """
    x0, y0, x1, y1 = bbox_xyxy
    analysis_height, analysis_width = analysis_shape_y_x
    source_height, source_width = source_shape_y_x
    if min(analysis_height, analysis_width, source_height, source_width) < 1:
        raise ValueError("analysis and source shapes must be positive")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("candidate bbox must be non-empty")

    source_x0 = math.floor(x0 * source_width / analysis_width)
    source_y0 = math.floor(y0 * source_height / analysis_height)
    source_x1 = math.ceil(x1 * source_width / analysis_width)
    source_y1 = math.ceil(y1 * source_height / analysis_height)
    source_x0 = min(max(source_x0, 0), source_width)
    source_y0 = min(max(source_y0, 0), source_height)
    source_x1 = min(max(source_x1, 0), source_width)
    source_y1 = min(max(source_y1, 0), source_height)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        raise ValueError("candidate bbox does not intersect source CT")
    return source_x0, source_y0, source_x1, source_y1


def candidate_bbox_nonzero_fraction(
    stack: np.ndarray,
    *,
    central_slice: int,
    source_bbox_xyxy: tuple[int, int, int, int],
) -> float:
    """Return supported-pixel fraction across the full candidate footprint."""
    if stack.ndim != 3:
        raise ValueError("stack must be depth,y,x")
    if not 0 <= central_slice < stack.shape[0]:
        raise ValueError("central slice outside stack")
    x0, y0, x1, y1 = source_bbox_xyxy
    if not (0 <= x0 < x1 <= stack.shape[2] and 0 <= y0 < y1 <= stack.shape[1]):
        raise ValueError("source bbox must be non-empty and inside source CT")
    region = stack[central_slice, y0:y1, x0:x1]
    if region.size == 0:  # Defensive: the bounds check above should guarantee this.
        raise ValueError("source bbox mapped to an empty CT region")
    return float(np.count_nonzero(region) / region.size)


def normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0 or len(values) <= 1:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(
        -(probabilities * np.log(probabilities)).sum() / math.log(len(values))
    )


def local_peak_count(profile: np.ndarray) -> int:
    if len(profile) < 3 or float(profile.max()) <= 0:
        return 0
    threshold = float(profile.max()) * 0.35
    return sum(
        1
        for index in range(1, len(profile) - 1)
        if profile[index] >= threshold
        and profile[index] >= profile[index - 1]
        and profile[index] > profile[index + 1]
    )


def extract_features(
    patch: np.ndarray,
    *,
    central_slice: int,
) -> dict[str, float | int]:
    if not 0 <= central_slice < patch.shape[0]:
        raise ValueError("central slice outside stack")
    clipped = np.clip(patch, 0, 200)
    central_valid = clipped[central_slice] > 0
    center_y = central_valid.shape[0] // 2
    center_x = central_valid.shape[1] // 2
    radius = max(1, min(center_y, center_x))
    zero_y_x = np.argwhere(~central_valid)
    if len(zero_y_x):
        distances = np.sqrt(
            (zero_y_x[:, 0] - center_y) ** 2
            + (zero_y_x[:, 1] - center_x) ** 2
        )
        zero_distance = float(distances.min())
    else:
        # The exact distance beyond the audited patch is irrelevant: the v2
        # validity contract only requires the whole patch to be supported.
        zero_distance = float(radius + 1)
    depth_gradient = np.abs(np.diff(clipped, axis=0))
    profile = depth_gradient.mean(axis=(1, 2))
    total = float(profile.sum())
    order = np.sort(profile)[::-1]
    top1 = float(order[:1].sum() / total) if total else 0.0
    top3 = float(order[:3].sum() / total) if total else 0.0
    central_transition = min(max(central_slice - 1, 0), len(profile) - 1)
    c0 = max(0, central_transition - 2)
    c1 = min(len(profile), central_transition + 3)
    central_fraction = float(profile[c0:c1].sum() / total) if total else 0.0

    argmax_depth = np.argmax(depth_gradient, axis=0)
    near_central = float(
        np.mean(np.abs(argmax_depth - central_transition) <= 2)
    )
    counts = np.bincount(argmax_depth.ravel(), minlength=len(profile))
    mode = int(np.argmax(counts))
    m0 = max(0, mode - 1)
    m1 = min(len(counts), mode + 2)
    mode_fraction = float(counts[m0:m1].sum() / argmax_depth.size)
    q10, q90 = np.quantile(argmax_depth, [0.10, 0.90])

    central = clipped[central_slice]
    xy_gradient = (
        float(np.abs(np.diff(central, axis=0)).mean())
        + float(np.abs(np.diff(central, axis=1)).mean())
    ) / 2
    z_gradient = float(depth_gradient.mean())
    localization_score = (
        0.35 * top3 + 0.35 * central_fraction + 0.30 * near_central
    )
    return {
        "central_slice_nonzero_fraction": float(central_valid.mean()),
        "central_slice_zero_distance_ratio": float(zero_distance / radius),
        "central_slice_center_nonzero": int(central_valid[center_y, center_x]),
        "depth_profile_top1_fraction": top1,
        "depth_profile_top3_fraction": top3,
        "depth_profile_entropy": normalized_entropy(profile),
        "depth_profile_peak_count": local_peak_count(profile),
        "central_depth_energy_fraction": central_fraction,
        "argmax_depth_near_central_fraction": near_central,
        "argmax_depth_mode_fraction": mode_fraction,
        "argmax_depth_p90_p10_span": float(q90 - q10),
        "xy_to_z_gradient_ratio": float(xy_gradient / max(z_gradient, 1e-6)),
        "surface_localization_score": float(localization_score),
    }


def summarize(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    classes = sorted({str(row["class"]) for row in rows})
    for label in classes:
        subset = [row for row in rows if row["class"] == label]
        result[label] = {
            "count": len(subset),
            "groups": sorted({str(row["group_id"]) for row in subset}),
            "features": {},
        }
        for name in feature_names:
            values = np.asarray([float(row[name]) for row in subset])
            result[label]["features"][name] = {
                "min": float(values.min()),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(values.max()),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    spec_path = args.spec.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = read_json(spec_path)
    radius_um = float(spec["patch_radius_um"])
    rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []

    for group in spec["groups"]:
        tiff_dir = (root / group["tiff_directory"]).resolve()
        analysis_path = (root / group["analysis"]).resolve()
        stack, files, slice_ordering = load_stack(tiff_dir)
        analysis = read_json(analysis_path)
        analysis_shape = tuple(map(int, analysis["input"]["shape_y_x"]))
        source_shape = (int(stack.shape[1]), int(stack.shape[2]))
        # ``central_slice`` is a physical slice number.  Resolve it through the
        # shared contract so zero padding or a non-contiguous render cannot
        # silently shift the sampled depth by list position.
        central_slice = ordered_tiff_stack_position(
            files,
            int(group["central_slice"]),
        )
        if not 0 <= central_slice < stack.shape[0]:
            raise ValueError(
                f"central slice outside stack for {group['group_id']}: "
                f"{central_slice} not in [0, {stack.shape[0]})"
            )
        radius = max(2, round(radius_um / float(group["voxel_um"])))
        candidates = analysis["text_like_screening"]["candidates"]
        for candidate in candidates:
            x0, y0, x1, y1 = map(int, candidate["bbox_xyxy"])
            source_x0, source_y0, source_x1, source_y1 = (
                map_analysis_bbox_to_source(
                    bbox_xyxy=(x0, y0, x1, y1),
                    analysis_shape_y_x=analysis_shape,
                    source_shape_y_x=source_shape,
                )
            )
            analysis_y = (y0 + y1) // 2
            analysis_x = (x0 + x1) // 2
            source_y, source_x = map_analysis_point_to_source(
                analysis_y=analysis_y,
                analysis_x=analysis_x,
                analysis_shape_y_x=analysis_shape,
                source_shape_y_x=source_shape,
            )
            patch = crop_stack_with_padding(
                stack,
                center_y=source_y,
                center_x=source_x,
                radius=radius,
            )
            row: dict[str, Any] = {
                "group_id": group["group_id"],
                "class": group["class"],
                "candidate_id": candidate["candidate_id"],
                "analysis_bbox_x0": x0,
                "analysis_bbox_y0": y0,
                "analysis_bbox_x1": x1,
                "analysis_bbox_y1": y1,
                "analysis_center_y": analysis_y,
                "analysis_center_x": analysis_x,
                "source_bbox_x0": source_x0,
                "source_bbox_y0": source_y0,
                "source_bbox_x1": source_x1,
                "source_bbox_y1": source_y1,
                "source_center_y": source_y,
                "source_center_x": source_x,
                "patch_radius_pixels": radius,
                "patch_radius_um": radius * float(group["voxel_um"]),
                "candidate_bbox_nonzero_fraction": (
                    candidate_bbox_nonzero_fraction(
                        stack,
                        central_slice=central_slice,
                        source_bbox_xyxy=(
                            source_x0,
                            source_y0,
                            source_x1,
                            source_y1,
                        ),
                    )
                ),
            }
            row.update(
                extract_features(
                    patch,
                    central_slice=central_slice,
                )
            )
            rows.append(row)
        source_receipts.append(
            {
                "group_id": group["group_id"],
                "class": group["class"],
                "candidate_count": len(candidates),
                "shape_depth_y_x": list(stack.shape),
                "slice_ordering": slice_ordering,
                "central_slice_resolution": NUMERIC_STEM_INDEX,
                "central_slice_name": files[central_slice].name,
                "central_slice_stack_position": central_slice,
                "ordered_slice_names": {
                    "first": files[0].name,
                    "middle": files[len(files) // 2].name,
                    "last": files[-1].name,
                },
                "analysis_sha256": sha256_file(analysis_path),
                "first_tiff_sha256": sha256_file(files[0]),
                "central_tiff_sha256": sha256_file(
                    files[central_slice]
                ),
                "last_tiff_sha256": sha256_file(files[-1]),
            }
        )

    csv_path = output / "CT_FIBER_FEATURES.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=BASE_FIELDS + FEATURE_NAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "kind": "campaign_x_phase4_ct_fiber_feature_benchmark_v1",
        "status": "FEATURES_EXTRACTED_THRESHOLDS_NOT_FROZEN",
        "generated_at_utc": utc_now(),
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "row_count": len(rows),
        "feature_names": FEATURE_NAMES,
        "sources": source_receipts,
        "summary": summarize(rows, FEATURE_NAMES),
        "artifacts": {
            "csv": csv_path.name,
            "csv_sha256": sha256_file(csv_path),
            "csv_size_bytes": csv_path.stat().st_size,
        },
        "non_claims": [
            "not an independent ink benchmark",
            "not target acceptance or rejection",
            "not a First Letters qualification",
        ],
    }
    (output / "CT_FIBER_FEATURE_BENCHMARK.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
