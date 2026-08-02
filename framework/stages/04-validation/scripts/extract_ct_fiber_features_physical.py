#!/usr/bin/env python3
"""Extract CT depth-localization features on a fixed physical z grid.

This is the v4 shadow feature extractor.  It does not replace or mutate the
frozen v3 gate.  Its purpose is to make features comparable across CT volumes
whose voxel sizes differ, while retaining explicit physical-window coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
    if str(_stage_scripts) not in sys.path:
        sys.path.insert(0, str(_stage_scripts))

from build_orthogonal_candidate_review import map_analysis_point_to_source
from extract_ct_fiber_features import (
    candidate_bbox_nonzero_fraction,
    crop_stack_with_padding,
    load_stack,
    map_analysis_bbox_to_source,
    normalized_entropy,
    read_json,
    sha256_file,
)


IDENTITY_FIELDS = [
    "group_id",
    "class",
    "candidate_id",
    "analysis_bbox_x0",
    "analysis_bbox_y0",
    "analysis_bbox_x1",
    "analysis_bbox_y1",
    "source_center_y",
    "source_center_x",
    "voxel_um",
    "patch_radius_um",
]

PHYSICAL_FEATURE_NAMES = [
    "candidate_bbox_nonzero_fraction",
    "physical_window_coverage_fraction",
    "physical_window_half_width_um",
    "physical_canonical_step_um",
    "depth_profile_top_energy_band_um",
    "depth_profile_top_energy_band_fraction",
    "depth_profile_entropy_physical",
    "depth_profile_peak_count_physical",
    "depth_profile_peak_density_per_100um",
    "central_depth_energy_fraction_physical",
    "argmax_depth_near_central_fraction_physical",
    "argmax_depth_p90_p10_span_um",
]


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


def _peak_count(profile: np.ndarray, relative_height: float) -> int:
    if profile.size < 3 or float(profile.max(initial=0.0)) <= 0:
        return 0
    threshold = float(profile.max()) * relative_height
    return sum(
        1
        for index in range(1, len(profile) - 1)
        if profile[index] >= threshold
        and profile[index] >= profile[index - 1]
        and profile[index] > profile[index + 1]
    )


ADAPTIVE_WINDOW_POLICY = "ADAPT_TO_AVAILABLE_SYMMETRIC_SUPPORT"


def resolve_half_window_um(
    physical_config: dict[str, Any],
    *,
    depth_slices: int,
    central_slice: int,
    voxel_um: float,
) -> float:
    """Resolve the operative half-window for one patch under the frozen policy.

    Single source of truth for `ct-fiber-supported-window-router@4.1.0`'s
    `physical_depth_sampling` block.  It is the arithmetic the benchmark
    executor has always used; it now also serves the production extractor,
    which previously read `half_window_um` directly and honoured neither
    `window_policy` nor `minimum_supported_half_window_um`.

    That divergence meant `MULTISCROLL_TRANSFER_V2`/`V3` measured
    `physical_window_half_width_um = 72.0` on all 300 controls while the
    production run measured 120.0 — the benchmark that validated v4.1 did not
    exercise the code path production used.  v4.1 exists precisely to
    implement this policy: its predecessor's fixed 120 um window produced
    coverage 0.6 on all 300 controls and failed transfer V1.

    No declared value is altered here.  When a stack genuinely carries the full
    declared support the result is the declared half-window unchanged; the
    policy only binds when the available symmetric support is smaller.
    """

    declared_half_window_um = float(physical_config["half_window_um"])
    if physical_config.get("window_policy") != ADAPTIVE_WINDOW_POLICY:
        return declared_half_window_um

    # Preserved byte-for-byte from the validated benchmark path, asymmetric
    # -0.5/-1.5 margins included.  Changing it would re-open the transfer
    # result that accepted v4.1.
    raw_symmetric_half_span_um = (
        min(central_slice - 0.5, depth_slices - central_slice - 1.5) * voxel_um
    )
    step_um = float(physical_config["canonical_step_um"])
    return min(
        declared_half_window_um,
        max(
            float(physical_config["minimum_supported_half_window_um"]),
            np.floor(raw_symmetric_half_span_um / step_um) * step_um,
        ),
    )


def extract_physical_depth_features(
    patch: np.ndarray,
    *,
    central_slice: int,
    voxel_um: float,
    half_window_um: float = 120.0,
    canonical_step_um: float = 8.0,
    top_energy_band_um: float = 24.0,
    central_band_half_width_um: float = 20.0,
    argmax_near_central_um: float = 20.0,
    peak_relative_height: float = 0.35,
) -> dict[str, float | int]:
    """Return resolution-normalized depth features for one candidate patch."""

    if patch.ndim != 3 or patch.shape[0] < 2:
        raise ValueError("patch must be depth,y,x with at least two slices")
    if not 0 <= central_slice < patch.shape[0]:
        raise ValueError("central slice outside stack")
    for name, value in {
        "voxel_um": voxel_um,
        "half_window_um": half_window_um,
        "canonical_step_um": canonical_step_um,
        "top_energy_band_um": top_energy_band_um,
        "central_band_half_width_um": central_band_half_width_um,
        "argmax_near_central_um": argmax_near_central_um,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 < peak_relative_height <= 1:
        raise ValueError("peak_relative_height must be in (0, 1]")

    clipped = np.clip(patch.astype(np.float64, copy=False), 0, 200)
    depth_gradient = np.abs(np.diff(clipped, axis=0))
    raw_profile = depth_gradient.mean(axis=(1, 2))
    raw_positions_um = (
        np.arange(len(raw_profile), dtype=np.float64) + 0.5 - central_slice
    ) * voxel_um
    target_positions_um = np.arange(
        -half_window_um + canonical_step_um / 2,
        half_window_um,
        canonical_step_um,
        dtype=np.float64,
    )
    supported = (
        (target_positions_um >= raw_positions_um[0])
        & (target_positions_um <= raw_positions_um[-1])
    )
    if not np.any(supported):
        raise ValueError("CT stack does not overlap the requested physical window")
    profile = np.interp(
        target_positions_um[supported],
        raw_positions_um,
        raw_profile,
    )
    positions_um = target_positions_um[supported]
    total = float(profile.sum())
    top_bin_count = max(1, int(math.ceil(top_energy_band_um / canonical_step_um)))
    top_fraction = (
        float(np.sort(profile)[::-1][:top_bin_count].sum() / total)
        if total
        else 0.0
    )
    central_fraction = (
        float(profile[np.abs(positions_um) <= central_band_half_width_um].sum() / total)
        if total
        else 0.0
    )
    peaks = _peak_count(profile, peak_relative_height)
    supported_span_um = max(canonical_step_um, len(profile) * canonical_step_um)

    in_window = (
        (raw_positions_um >= -half_window_um)
        & (raw_positions_um <= half_window_um)
    )
    window_gradient = depth_gradient[in_window]
    window_positions_um = raw_positions_um[in_window]
    if window_gradient.shape[0] == 0:
        raise ValueError("no raw transitions in requested physical window")
    argmax_index = np.argmax(window_gradient, axis=0)
    argmax_um = window_positions_um[argmax_index]
    q10_um, q90_um = np.quantile(argmax_um, [0.10, 0.90])

    return {
        "physical_window_coverage_fraction": float(supported.mean()),
        "physical_window_half_width_um": float(half_window_um),
        "physical_canonical_step_um": float(canonical_step_um),
        "depth_profile_top_energy_band_um": float(top_energy_band_um),
        "depth_profile_top_energy_band_fraction": top_fraction,
        "depth_profile_entropy_physical": normalized_entropy(profile),
        "depth_profile_peak_count_physical": peaks,
        "depth_profile_peak_density_per_100um": float(
            peaks * 100.0 / supported_span_um
        ),
        "central_depth_energy_fraction_physical": central_fraction,
        "argmax_depth_near_central_fraction_physical": float(
            np.mean(np.abs(argmax_um) <= argmax_near_central_um)
        ),
        "argmax_depth_p90_p10_span_um": float(q90_um - q10_um),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    spec_path = args.spec.resolve()
    profile_path = args.profile.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to overwrite physical feature evidence")
    output.mkdir(parents=True, exist_ok=True)
    spec = read_json(spec_path)
    profile = read_json(profile_path)
    config = dict(profile["physical_depth_sampling"])
    radius_um = float(spec["patch_radius_um"])
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for group in spec["groups"]:
        voxel_um = float(group["voxel_um"])
        tiff_dir = (root / group["tiff_directory"]).resolve()
        analysis_path = (root / group["analysis"]).resolve()
        stack, files, ordering = load_stack(tiff_dir)
        analysis = read_json(analysis_path)
        analysis_shape = tuple(map(int, analysis["input"]["shape_y_x"]))
        source_shape = (int(stack.shape[1]), int(stack.shape[2]))
        central_slice = int(group["central_slice"])
        radius = max(2, round(radius_um / voxel_um))
        for candidate in analysis["text_like_screening"]["candidates"]:
            x0, y0, x1, y1 = map(int, candidate["bbox_xyxy"])
            source_bbox = map_analysis_bbox_to_source(
                bbox_xyxy=(x0, y0, x1, y1),
                analysis_shape_y_x=analysis_shape,
                source_shape_y_x=source_shape,
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
                "source_center_y": source_y,
                "source_center_x": source_x,
                "voxel_um": voxel_um,
                "patch_radius_um": radius * voxel_um,
                "candidate_bbox_nonzero_fraction": candidate_bbox_nonzero_fraction(
                    stack,
                    central_slice=central_slice,
                    source_bbox_xyxy=source_bbox,
                ),
            }
            row.update(
                extract_physical_depth_features(
                    patch,
                    central_slice=central_slice,
                    voxel_um=voxel_um,
                    half_window_um=resolve_half_window_um(
                        config,
                        depth_slices=patch.shape[0],
                        central_slice=central_slice,
                        voxel_um=voxel_um,
                    ),
                    canonical_step_um=float(config["canonical_step_um"]),
                    top_energy_band_um=float(config["top_energy_band_um"]),
                    central_band_half_width_um=float(
                        config["central_band_half_width_um"]
                    ),
                    argmax_near_central_um=float(
                        config["argmax_near_central_um"]
                    ),
                    peak_relative_height=float(config["peak_relative_height"]),
                )
            )
            rows.append(row)
        sources.append(
            {
                "group_id": group["group_id"],
                "voxel_um": voxel_um,
                "slice_count": len(files),
                "slice_ordering": ordering,
                "analysis_sha256": sha256_file(analysis_path),
                "first_tiff_sha256": sha256_file(files[0]),
                "last_tiff_sha256": sha256_file(files[-1]),
            }
        )

    csv_path = output / "CT_FIBER_PHYSICAL_FEATURES.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=IDENTITY_FIELDS + PHYSICAL_FEATURE_NAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "schema": "campaignx.ct_fiber_physical_features.v1",
        "status": "SHADOW_FEATURES_EXTRACTED_NOT_VALIDATED",
        "generated_at_utc": utc_now(),
        "spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
        "profile": {"path": str(profile_path), "sha256": sha256(profile_path)},
        "row_count": len(rows),
        "feature_names": PHYSICAL_FEATURE_NAMES,
        "sources": sources,
        "artifacts": {
            "csv": csv_path.name,
            "csv_sha256": sha256(csv_path),
        },
        "non_claims": [
            "not an independently validated classifier",
            "does not modify the frozen v3 decision",
            "not accepted ink, text, letters, or First Letters",
        ],
    }
    (output / "CT_FIBER_PHYSICAL_FEATURE_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
