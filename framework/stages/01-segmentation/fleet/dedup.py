from __future__ import annotations

from typing import Any, Iterable


def bidirectional_overlap(
    first: list[list[float]], second: list[list[float]], tolerance: float
) -> dict[str, float]:
    import numpy as np

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if not len(a) or not len(b):
        return {
            "first_fraction": 0.0,
            "second_fraction": 0.0,
            "median_voxels": float("inf"),
            "p90_voxels": float("inf"),
        }
    nearest_a: list[float] = []
    for start in range(0, len(a), 64):
        distance = np.linalg.norm(a[start : start + 64, None, :] - b[None, :, :], axis=2)
        nearest_a.extend(distance.min(axis=1).tolist())
    nearest_b: list[float] = []
    for start in range(0, len(b), 64):
        distance = np.linalg.norm(b[start : start + 64, None, :] - a[None, :, :], axis=2)
        nearest_b.extend(distance.min(axis=1).tolist())
    combined = np.asarray(nearest_a + nearest_b)
    return {
        "first_fraction": float(np.mean(np.asarray(nearest_a) <= tolerance)),
        "second_fraction": float(np.mean(np.asarray(nearest_b) <= tolerance)),
        "median_voxels": float(np.median(combined)),
        "p90_voxels": float(np.percentile(combined, 90)),
    }


def find_duplicate_in_surfaces(
    surfaces: Iterable[dict[str, Any]],
    artifact_sha256: str,
    sample_points: list[list[float]],
    *,
    tolerance_voxels: float = 2.5,
    overlap_fraction: float = 0.92,
    maximum_median_voxels: float = 1.0,
) -> tuple[str | None, dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for surface in surfaces:
        if surface.get("artifact_sha256") == artifact_sha256:
            return str(surface["surface_id"]), {
                "rule": "EXACT_ARTIFACT_SHA256",
                "comparisons": diagnostics,
            }
        known_points = surface.get("sample_points")
        if not known_points:
            continue
        metrics = bidirectional_overlap(sample_points, known_points, tolerance_voxels)
        diagnostics.append({"surface_id": surface["surface_id"], **metrics})
        if (
            metrics["first_fraction"] >= overlap_fraction
            and metrics["second_fraction"] >= overlap_fraction
            and metrics["median_voxels"] <= maximum_median_voxels
            and metrics["p90_voxels"] <= tolerance_voxels
        ):
            return str(surface["surface_id"]), {
                "rule": "BIDIRECTIONAL_POINT_OVERLAP",
                "tolerance_voxels": tolerance_voxels,
                "minimum_fraction": overlap_fraction,
                "maximum_median_voxels": maximum_median_voxels,
                "metrics": metrics,
                "comparisons": diagnostics,
            }
    return None, {"rule": "NO_DUPLICATE", "comparisons": diagnostics}
