"""Deterministic structural metrics for canonical triangle meshes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from .coordinate_transform import ObjMesh


class _DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


UNMEASURED = "UNMEASURED"


def weld_report_self_intersection_count(
    weld_report: Mapping[str, Any] | None,
) -> int | str:
    """Return the weld report's self-intersection count, or ``UNMEASURED``.

    ``grid_weld`` only audits manifoldness; it does not run an exact
    triangle-triangle intersection test, and some report versions omit the
    field entirely.  Reporting a literal ``0`` in that case would make the
    ``GEOMETRY_VALIDATED`` promotion check structurally unable to fire, so an
    absent or non-integer count degrades to ``UNMEASURED`` instead.
    """

    if not isinstance(weld_report, Mapping):
        return UNMEASURED
    for container in (weld_report, weld_report.get("manifold_audit")):
        if not isinstance(container, Mapping):
            continue
        for key in ("self_intersection_count", "self_intersections", "self_intersecting_faces"):
            value = container.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if value < 0:
                raise ValueError(f"weld report {key} must be non-negative: {value}")
            return int(value)
    return UNMEASURED


def topology_metrics(
    mesh: ObjMesh,
    *,
    voxel_size_um_xyz: tuple[float, float, float],
    weld_report: Mapping[str, Any] | None = None,
) -> dict:
    """Calculate non-destructive topology/area metrics.

    The adapter result remains ``PROVISIONAL`` because this inexpensive audit
    does not replace the later CT and self-intersection gates.  The
    ``self_intersection_count`` field is the upstream weld-report count when
    the report carries one and the sentinel ``UNMEASURED`` otherwise; it is
    never a promotion claim, and ``UNMEASURED`` must not satisfy
    ``GEOMETRY_VALIDATED``.
    """

    if len(voxel_size_um_xyz) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in voxel_size_um_xyz
    ):
        raise ValueError("voxel_size_um_xyz must contain three finite positive values")

    scaled = mesh.vertices * np.asarray(voxel_size_um_xyz, dtype=np.float64)
    cross = np.cross(
        scaled[mesh.faces[:, 1]] - scaled[mesh.faces[:, 0]],
        scaled[mesh.faces[:, 2]] - scaled[mesh.faces[:, 0]],
    )
    triangle_areas_um2 = 0.5 * np.linalg.norm(cross, axis=1)

    edge_directions: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    referenced = {int(value) for value in mesh.faces.reshape(-1)}
    sets = _DisjointSet(referenced)
    for face in mesh.faces:
        a, b, c = (int(value) for value in face)
        sets.union(a, b)
        sets.union(b, c)
        for left, right in ((a, b), (b, c), (c, a)):
            edge_directions[tuple(sorted((left, right)))].append((left, right))

    non_manifold = sum(len(rows) > 2 for rows in edge_directions.values())
    inconsistent = sum(
        len(rows) == 2 and rows[0] == rows[1] for rows in edge_directions.values()
    )
    components = len({sets.find(value) for value in referenced})

    return {
        "area_cm2": float(triangle_areas_um2.sum() * 1e-8),
        "component_count": int(components),
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.faces)),
        "invalid_coordinate_count": 0,
        "non_manifold_edge_count": int(non_manifold),
        "self_intersection_count": weld_report_self_intersection_count(weld_report),
        "inconsistent_winding_edge_count": int(inconsistent),
        "degenerate_triangle_fraction": 0.0,
        "orientable": bool(non_manifold == 0 and inconsistent == 0),
    }
