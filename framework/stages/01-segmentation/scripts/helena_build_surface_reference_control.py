#!/usr/bin/env python3
"""Freeze a deterministic ScrollFiesta control ROI from a known TIFXYZ surface.

The control is selected exclusively from reference geometry.  No candidate
backend output, CT render, or ink prediction is inspected while choosing the
anchor.  The pixel farthest from the invalid TIFXYZ boundary becomes the
anchor, which makes the test reproducible and avoids a hand-picked easy point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt


SCHEMA = "campaignx.surface_reference_control.v1"


class ControlSelectionError(RuntimeError):
    """Raised when a reference surface cannot define a valid control."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tifxyz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for axis in "xyz":
        source = path / f"{axis}.tif"
        if not source.is_file():
            raise ControlSelectionError(f"missing reference coordinate: {source}")
        arrays.append(np.asarray(tifffile.imread(source), dtype=np.float64))
    x, y, z = arrays
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape:
        raise ControlSelectionError("x/y/z must be equal-sized two-dimensional arrays")
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x >= 0.0)
        & (y >= 0.0)
        & (z >= 0.0)
    )
    if int(valid.sum()) < 9:
        raise ControlSelectionError("reference contains fewer than nine valid coordinates")
    return x, y, z, valid


def _aligned_lower(value: float, *, cube_edge: int, cubes_per_axis: int) -> int:
    if not math.isfinite(value):
        raise ControlSelectionError("anchor coordinate is not finite")
    base = math.floor(value / cube_edge) * cube_edge
    candidates = [base - index * cube_edge for index in range(cubes_per_axis)]
    span = cube_edge * cubes_per_axis
    containing = [lower for lower in candidates if lower <= value < lower + span]
    if not containing:
        raise ControlSelectionError("could not align ROI around anchor")
    return max(
        containing,
        key=lambda lower: (min(value - lower, lower + span - value), -lower),
    )


def select_control(
    reference: Path,
    *,
    sample_id: str,
    segment_id: str,
    cube_edge: int,
    cubes_per_axis: int,
    voxel_size_um: float,
) -> dict:
    if cube_edge <= 0 or cubes_per_axis <= 0:
        raise ControlSelectionError("cube dimensions must be positive")
    if not math.isfinite(voxel_size_um) or voxel_size_um <= 0:
        raise ControlSelectionError("voxel size must be positive and finite")

    reference = reference.resolve()
    x, y, z, valid = _load_tifxyz(reference)
    distance = distance_transform_edt(valid)
    row, column = np.unravel_index(int(np.argmax(distance)), valid.shape)
    if row == 0 or column == 0 or row + 1 == x.shape[0] or column + 1 == x.shape[1]:
        raise ControlSelectionError("selected anchor does not have a full local stencil")

    points = np.stack((x, y, z), axis=-1)
    anchor_xyz = points[row, column]
    tangent_row = points[row + 1, column] - points[row - 1, column]
    tangent_column = points[row, column + 1] - points[row, column - 1]
    normal = np.cross(tangent_row, tangent_column)
    norm = float(np.linalg.norm(normal))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ControlSelectionError("selected anchor has a degenerate reference normal")
    normal /= norm

    lower_xyz = np.asarray(
        [
            _aligned_lower(float(value), cube_edge=cube_edge, cubes_per_axis=cubes_per_axis)
            for value in anchor_xyz
        ],
        dtype=np.int64,
    )
    upper_xyz = lower_xyz + cube_edge * cubes_per_axis
    in_roi = valid.copy()
    for axis_array, lower, upper in zip((x, y, z), lower_xyz, upper_xyz, strict=True):
        in_roi &= (axis_array >= lower) & (axis_array < upper)
    roi_reference_points = int(in_roi.sum())
    if roi_reference_points < 9:
        raise ControlSelectionError("aligned ROI contains fewer than nine reference points")

    meta = reference / "meta.json"
    files = {}
    for name in ("meta.json", "x.tif", "y.tif", "z.tif"):
        source = reference / name
        if not source.is_file():
            raise ControlSelectionError(f"missing reference file: {source}")
        files[name] = {
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        }

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selection_policy": "MAXIMUM_DISTANCE_FROM_INVALID_TIFXYZ_BOUNDARY",
        "selection_uses_candidate_output": False,
        "selection_uses_ink": False,
        "sample_id": sample_id,
        "segment_id": segment_id,
        "coordinate_frame": "LEVEL0_XYZ_VOXELS",
        "voxel_size_um_xyz": [voxel_size_um] * 3,
        "reference": {
            "path": str(reference),
            "grid_shape_yx": list(valid.shape),
            "valid_point_count": int(valid.sum()),
            "files": files,
        },
        "anchor": {
            "pixel_yx": [int(row), int(column)],
            "xyz": [float(value) for value in anchor_xyz],
            "native_zyx": [float(value) for value in anchor_xyz[::-1]],
            "normal_xyz": [float(value) for value in normal],
            "distance_from_invalid_boundary_pixels": float(distance[row, column]),
        },
        "roi": {
            "cube_edge_voxels": cube_edge,
            "cubes_per_axis": cubes_per_axis,
            "level0_xyz": [
                *[int(value) for value in lower_xyz],
                *[int(value) for value in upper_xyz],
            ],
            "level0_zyx": [
                *[int(value) for value in lower_xyz[::-1]],
                *[int(value) for value in upper_xyz[::-1]],
            ],
            "scrollfiesta_bbox_cli_zyx_pairs": [
                int(lower_xyz[2]),
                int(upper_xyz[2]),
                int(lower_xyz[1]),
                int(upper_xyz[1]),
                int(lower_xyz[0]),
                int(upper_xyz[0]),
            ],
            "reference_point_count": roi_reference_points,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-tifxyz", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--cube-edge", type=int, default=128)
    parser.add_argument("--cubes-per-axis", type=int, default=2)
    parser.add_argument("--voxel-size-um", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ControlSelectionError(f"refusing to overwrite: {args.output}")
    result = select_control(
        args.reference_tifxyz,
        sample_id=args.sample_id,
        segment_id=args.segment_id,
        cube_edge=args.cube_edge,
        cubes_per_axis=args.cubes_per_axis,
        voxel_size_um=args.voxel_size_um,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
