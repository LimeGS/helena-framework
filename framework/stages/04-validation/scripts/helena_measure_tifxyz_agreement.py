#!/usr/bin/env python3
"""Measure like-for-like geometry inside a frozen level-0 ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_surface(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    arrays = [np.asarray(tifffile.imread(directory / f"{axis}.tif"), dtype=np.float64) for axis in "xyz"]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError(f"TIFXYZ shapes disagree in {directory}")
    xyz = np.stack(arrays, axis=-1)
    valid = np.all(np.isfinite(xyz), axis=-1) & np.all(xyz >= 0, axis=-1)
    if not np.any(valid):
        raise ValueError(f"TIFXYZ has no valid coordinates: {directory}")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    return xyz, valid, meta


def roi_mask(xyz: np.ndarray, valid: np.ndarray, roi: list[int]) -> np.ndarray:
    z0, y0, x0, z1, y1, x1 = roi
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return valid & (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= z0) & (z < z1)


def triangles(xyz: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = xyz.shape[:2]
    centroids, normals, areas = [], [], []
    for offsets in (((0, 0), (1, 0), (0, 1)), ((1, 1), (0, 1), (1, 0))):
        if rows < 2 or columns < 2:
            continue
        points = [xyz[dy : rows - 1 + dy, dx : columns - 1 + dx] for dy, dx in offsets]
        keep = np.logical_and.reduce([mask[dy : rows - 1 + dy, dx : columns - 1 + dx] for dy, dx in offsets])
        a, b, c = points
        cross = np.cross(b - a, c - a)
        double_area = np.linalg.norm(cross, axis=-1)
        keep &= np.isfinite(double_area) & (double_area > 1e-12)
        if not np.any(keep):
            continue
        centroids.append(((a + b + c) / 3.0)[keep])
        normals.append((cross[keep] / double_area[keep, None]))
        areas.append(double_area[keep] / 2.0)
    if not centroids:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))
    return np.concatenate(centroids), np.concatenate(normals), np.concatenate(areas)


def summarize_surface(directory: Path, roi: list[int], voxel_um: float):
    xyz, valid, meta = load_surface(directory)
    mask = roi_mask(xyz, valid, roi)
    centers, normals, areas = triangles(xyz, mask)
    components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(components.ravel())[1:] if count else np.empty(0, dtype=int)
    area_cm2 = float(areas.sum() * voxel_um * voxel_um / 1e8)
    return {
        "directory": directory,
        "xyz": xyz,
        "mask": mask,
        "centers": centers,
        "normals": normals,
        "areas": areas,
        "summary": {
            "raster_shape": list(mask.shape),
            "valid_pixels_total": int(valid.sum()),
            "valid_pixels_in_roi": int(mask.sum()),
            "triangle_count_in_roi": int(areas.size),
            "usable_area_cm2_in_roi": area_cm2,
            "component_count_in_roi": int(count),
            "largest_component_pixels": int(sizes.max()) if sizes.size else 0,
            "source_meta_area_cm2": meta.get("area_cm2"),
        },
        "artifacts": {
            name: {"sha256": sha256(directory / name), "bytes": (directory / name).stat().st_size}
            for name in ("x.tif", "y.tif", "z.tif", "meta.json")
        },
    }


def directed(first: dict, second: dict, maximum_points: int) -> dict:
    if first["centers"].size == 0 or second["centers"].size == 0:
        return {"status": "UNMEASURED", "reason": "one surface has no ROI triangles"}
    index = np.linspace(0, len(first["centers"]) - 1, min(maximum_points, len(first["centers"])), dtype=np.int64)
    query = first["centers"][index]
    query_normals = first["normals"][index]
    distances, nearest = cKDTree(second["centers"]).query(query, k=1, workers=1)
    normal_dot = np.abs(np.einsum("ij,ij->i", query_normals, second["normals"][nearest]))
    return {
        "status": "MEASURED",
        "sample_count": int(len(index)),
        "distance_voxels": {key: float(np.quantile(distances, q)) for key, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))},
        "normal_dot_absolute": {key: float(np.quantile(normal_dot, q)) for key, q in (("p05", 0.05), ("p50", 0.5))},
        "within_2_voxels_fraction": float(np.mean(distances <= 2.0)),
        "normal_dot_at_least_0_866_fraction": float(np.mean(normal_dot >= 0.866)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vc3d", type=Path, required=True)
    parser.add_argument("--scrollfiesta", type=Path, required=True)
    parser.add_argument("--roi", type=int, nargs=6, required=True, metavar=("Z0", "Y0", "X0", "Z1", "Y1", "X1"))
    parser.add_argument("--voxel-size-um", type=float, required=True)
    parser.add_argument("--maximum-points", type=int, default=200000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    if args.voxel_size_um <= 0 or args.maximum_points < 1:
        raise ValueError("voxel size and maximum points must be positive")
    if any(args.roi[i] >= args.roi[i + 3] for i in range(3)):
        raise ValueError("ROI extents must be positive")

    vc3d = summarize_surface(args.vc3d, args.roi, args.voxel_size_um)
    scrollfiesta = summarize_surface(args.scrollfiesta, args.roi, args.voxel_size_um)
    result = {
        "schema": "campaignx.tifxyz_geometry_agreement.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "roi_level0_zyx": args.roi,
        "voxel_size_um": args.voxel_size_um,
        "surfaces": {
            "vc3d": {"summary": vc3d["summary"], "artifacts": vc3d["artifacts"]},
            "scrollfiesta": {"summary": scrollfiesta["summary"], "artifacts": scrollfiesta["artifacts"]},
        },
        "directed": {
            "vc3d_to_scrollfiesta": directed(vc3d, scrollfiesta, args.maximum_points),
            "scrollfiesta_to_vc3d": directed(scrollfiesta, vc3d, args.maximum_points),
        },
        "non_claim": "Geometric agreement is not raw-CT validation, ink, or text evidence."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": result["schema"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
