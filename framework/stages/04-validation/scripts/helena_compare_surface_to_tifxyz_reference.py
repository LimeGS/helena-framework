#!/usr/bin/env python3
"""Compare a candidate triangle mesh with a frozen public TIFXYZ surface.

TIFXYZ reference grids are commonly much sparser than ScrollFiesta meshes.
Candidate fidelity therefore uses distance to the local reference tangent
plane, not raw nearest-point distance.  Reference recovery uses the reverse
nearest-vertex distance because the candidate mesh is dense.  Neither metric
can repair or override a failed topology/flattening gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from framework.stages.segmentation_import_shim import load_scrollfiesta_triangle_obj  # noqa: E402


SCHEMA = "campaignx.surface_reference_agreement.v1"


class ReferenceComparisonError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(tifffile.imread(path / f"{axis}.tif"), dtype=np.float64)
        for axis in "xyz"
    ]
    if any(array.ndim != 2 or array.shape != arrays[0].shape for array in arrays):
        raise ReferenceComparisonError("reference x/y/z arrays must be equal 2D grids")
    xyz = np.stack(arrays, axis=-1)
    valid = np.all(np.isfinite(xyz), axis=-1) & np.all(xyz >= 0.0, axis=-1)

    normals = np.full_like(xyz, np.nan)
    stencil = valid.copy()
    stencil[0, :] = False
    stencil[-1, :] = False
    stencil[:, 0] = False
    stencil[:, -1] = False
    stencil[1:-1, 1:-1] &= (
        valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )
    rows, columns = np.nonzero(stencil)
    row_tangent = xyz[rows + 1, columns] - xyz[rows - 1, columns]
    column_tangent = xyz[rows, columns + 1] - xyz[rows, columns - 1]
    values = np.cross(row_tangent, column_tangent)
    lengths = np.linalg.norm(values, axis=1)
    usable = np.isfinite(lengths) & (lengths > 1e-9)
    normals[rows[usable], columns[usable]] = values[usable] / lengths[usable, None]
    normal_valid = np.all(np.isfinite(normals), axis=-1)
    if int(normal_valid.sum()) < 3:
        raise ReferenceComparisonError("reference has fewer than three normal-bearing points")
    return xyz[normal_valid], normals[normal_valid], valid


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    values = np.zeros_like(vertices)
    for column in range(3):
        np.add.at(values, faces[:, column], face_normals)
    lengths = np.linalg.norm(values, axis=1)
    valid = np.isfinite(lengths) & (lengths > 1e-9)
    values[valid] /= lengths[valid, None]
    values[~valid] = np.nan
    return values


def _percentiles(values: np.ndarray, rows: tuple[int, ...]) -> dict[str, float]:
    return {
        f"p{row:02d}": float(np.percentile(values, row, method="linear"))
        for row in rows
    }


def compare(
    candidate_obj: Path,
    reference_tifxyz: Path,
    *,
    roi_level0_zyx: list[int],
    threshold_voxels: float,
    minimum_candidate_fidelity_fraction: float,
    minimum_reference_recovery_fraction: float,
    minimum_normal_dot: float,
    maximum_reference_support_radius_voxels: float,
    maximum_candidate_samples: int,
) -> dict:
    if len(roi_level0_zyx) != 6:
        raise ReferenceComparisonError("ROI must contain z0,y0,x0,z1,y1,x1")
    if maximum_candidate_samples <= 0:
        raise ReferenceComparisonError("maximum candidate samples must be positive")
    candidate_obj = candidate_obj.resolve()
    reference_tifxyz = reference_tifxyz.resolve()
    mesh = load_scrollfiesta_triangle_obj(candidate_obj)
    candidate_normals = _vertex_normals(mesh.vertices, mesh.faces)
    reference_points, reference_normals, _ = _load_reference(reference_tifxyz)

    z0, y0, x0, z1, y1, x1 = map(float, roi_level0_zyx)
    lower = np.asarray([x0, y0, z0])
    upper = np.asarray([x1, y1, z1])
    candidate_mask = np.all((mesh.vertices >= lower) & (mesh.vertices < upper), axis=1)
    candidate_mask &= np.all(np.isfinite(candidate_normals), axis=1)
    reference_mask = np.all((reference_points >= lower) & (reference_points < upper), axis=1)
    candidate_indices = np.flatnonzero(candidate_mask)
    if not len(candidate_indices):
        raise ReferenceComparisonError("candidate has no evaluable vertices inside ROI")
    if int(reference_mask.sum()) < 3:
        raise ReferenceComparisonError("reference has fewer than three normal-bearing points inside ROI")
    if len(candidate_indices) > maximum_candidate_samples:
        positions = np.linspace(0, len(candidate_indices) - 1, maximum_candidate_samples, dtype=np.int64)
        candidate_indices = candidate_indices[positions]

    candidate_points = mesh.vertices[candidate_indices]
    candidate_sample_normals = candidate_normals[candidate_indices]
    reference_roi = reference_points[reference_mask]
    reference_roi_normals = reference_normals[reference_mask]

    reference_tree = cKDTree(reference_roi)
    candidate_euclidean, nearest_reference = reference_tree.query(candidate_points, k=1)
    supported = candidate_euclidean <= maximum_reference_support_radius_voxels
    if int(supported.sum()) < 3:
        raise ReferenceComparisonError("fewer than three candidate samples have reference support")
    deltas = candidate_points[supported] - reference_roi[nearest_reference[supported]]
    matched_reference_normals = reference_roi_normals[nearest_reference[supported]]
    point_to_plane = np.abs(np.einsum("ij,ij->i", deltas, matched_reference_normals))
    normal_dot = np.abs(
        np.einsum("ij,ij->i", candidate_sample_normals[supported], matched_reference_normals)
    )

    candidate_tree = cKDTree(candidate_points)
    reverse_euclidean, _ = candidate_tree.query(reference_roi, k=1)
    candidate_fidelity = float(np.mean(point_to_plane <= threshold_voxels))
    reference_recovery = float(np.mean(reverse_euclidean <= threshold_voxels))
    normal_p05 = float(np.percentile(normal_dot, 5, method="linear"))
    plane_p95 = float(np.percentile(point_to_plane, 95, method="linear"))

    requirements = {
        "candidate_point_to_plane_p95": {
            "value": plane_p95,
            "operator": "<=",
            "threshold": threshold_voxels,
            "status": "PASS" if plane_p95 <= threshold_voxels else "FAIL",
        },
        "candidate_within_tolerance_fraction": {
            "value": candidate_fidelity,
            "operator": ">=",
            "threshold": minimum_candidate_fidelity_fraction,
            "status": "PASS" if candidate_fidelity >= minimum_candidate_fidelity_fraction else "FAIL",
        },
        "reference_recovery_fraction": {
            "value": reference_recovery,
            "operator": ">=",
            "threshold": minimum_reference_recovery_fraction,
            "status": "PASS" if reference_recovery >= minimum_reference_recovery_fraction else "FAIL",
        },
        "normal_abs_dot_p05": {
            "value": normal_p05,
            "operator": ">=",
            "threshold": minimum_normal_dot,
            "status": "PASS" if normal_p05 >= minimum_normal_dot else "FAIL",
        },
    }
    status = "PASS" if all(row["status"] == "PASS" for row in requirements.values()) else "FAIL"
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "inputs": {
            "candidate_obj": {"path": str(candidate_obj), "sha256": sha256(candidate_obj)},
            "reference_tifxyz": {
                "path": str(reference_tifxyz),
                "files": {
                    name: sha256(reference_tifxyz / name)
                    for name in ("meta.json", "x.tif", "y.tif", "z.tif")
                },
            },
            "roi_level0_zyx": [int(value) for value in roi_level0_zyx],
        },
        "policy": {
            "candidate_metric": "NEAREST_REFERENCE_LOCAL_TANGENT_PLANE",
            "reference_metric": "NEAREST_CANDIDATE_VERTEX",
            "normal_sign_policy": "ABSOLUTE_DOT",
            "deterministic_candidate_sampling": "EVENLY_SPACED_VERTEX_INDICES",
            "maximum_candidate_samples": maximum_candidate_samples,
            "maximum_reference_support_radius_voxels": maximum_reference_support_radius_voxels,
        },
        "counts": {
            "candidate_vertices_total": int(len(mesh.vertices)),
            "candidate_vertices_in_roi": int(candidate_mask.sum()),
            "candidate_samples": int(len(candidate_points)),
            "candidate_samples_with_reference_support": int(supported.sum()),
            "reference_normal_points_in_roi": int(len(reference_roi)),
        },
        "metrics": {
            "candidate_reference_euclidean_voxels": _percentiles(candidate_euclidean[supported], (50, 95, 99)),
            "candidate_reference_point_to_plane_voxels": _percentiles(point_to_plane, (50, 95, 99)),
            "candidate_reference_normal_abs_dot": _percentiles(normal_dot, (5, 50, 95)),
            "reference_candidate_euclidean_voxels": _percentiles(reverse_euclidean, (50, 95, 99)),
            "candidate_within_tolerance_fraction": candidate_fidelity,
            "reference_recovery_fraction": reference_recovery,
        },
        "requirements": requirements,
        "topology_or_flattening_override_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-obj", type=Path, required=True)
    parser.add_argument("--reference-tifxyz", type=Path, required=True)
    parser.add_argument("--roi-level0-zyx", nargs=6, type=int, required=True)
    parser.add_argument("--threshold-voxels", type=float, default=2.0)
    parser.add_argument("--minimum-candidate-fidelity-fraction", type=float, default=0.98)
    parser.add_argument("--minimum-reference-recovery-fraction", type=float, default=0.80)
    parser.add_argument("--minimum-normal-dot", type=float, default=math.cos(math.radians(30.0)))
    parser.add_argument("--maximum-reference-support-radius-voxels", type=float, default=40.0)
    parser.add_argument("--maximum-candidate-samples", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ReferenceComparisonError(f"refusing to overwrite: {args.output}")
    result = compare(
        args.candidate_obj,
        args.reference_tifxyz,
        roi_level0_zyx=args.roi_level0_zyx,
        threshold_voxels=args.threshold_voxels,
        minimum_candidate_fidelity_fraction=args.minimum_candidate_fidelity_fraction,
        minimum_reference_recovery_fraction=args.minimum_reference_recovery_fraction,
        minimum_normal_dot=args.minimum_normal_dot,
        maximum_reference_support_radius_voxels=args.maximum_reference_support_radius_voxels,
        maximum_candidate_samples=args.maximum_candidate_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
