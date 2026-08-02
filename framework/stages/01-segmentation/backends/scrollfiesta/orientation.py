"""Deterministic orientation repair for manifold triangle meshes.

ScrollFiesta shards can reach ``grid_weld`` with locally inconsistent face
winding even when the welded surface is orientable.  This module solves the
binary parity constraints induced by every shared edge.  It never changes
vertices, connectivity, or triangle membership: the only permitted mutation
is swapping the final two indices of a triangle.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .coordinate_transform import ObjMesh


class OrientationError(ValueError):
    """Raised when a mesh cannot be oriented without changing topology."""

    def __init__(self, message: str, *, report: dict | None = None) -> None:
        super().__init__(message)
        self.report = report


def orient_triangle_faces(mesh: ObjMesh) -> tuple[ObjMesh, dict]:
    """Return a consistently wound copy and a deterministic repair receipt.

    Each manifold shared edge contributes one XOR constraint between its two
    incident faces.  A contradiction proves that this face complex cannot be
    oriented by face flips alone, so the function fails closed.
    """

    edge_rows: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    for face_index, face in enumerate(mesh.faces):
        a, b, c = (int(value) for value in face)
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (min(left, right), max(left, right))
            edge_rows[edge].append((face_index, left < right))

    non_manifold = sorted(edge for edge, rows in edge_rows.items() if len(rows) > 2)
    if non_manifold:
        raise OrientationError(
            "face orientation repair requires a manifold edge complex; "
            f"found {len(non_manifold)} non-manifold edges"
        )

    adjacency: dict[int, list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
    shared_edge_count = 0
    inconsistent_before = 0
    for edge in sorted(edge_rows):
        rows = edge_rows[edge]
        if len(rows) != 2:
            continue
        shared_edge_count += 1
        (left_face, left_forward), (right_face, right_forward) = rows
        # Equal directions require exactly one face flip; opposite directions
        # require either both or neither face to be flipped.
        required_xor = int(left_forward == right_forward)
        inconsistent_before += required_xor
        adjacency[left_face].append((right_face, required_xor, edge))
        adjacency[right_face].append((left_face, required_xor, edge))

    assignments = np.full(len(mesh.faces), -1, dtype=np.int8)
    face_component_count = 0
    parity_conflicts: dict[tuple[int, int], dict] = {}
    for start in range(len(mesh.faces)):
        if assignments[start] >= 0:
            continue
        face_component_count += 1
        assignments[start] = 0
        queue = deque([start])
        while queue:
            face_index = queue.popleft()
            for neighbor, required_xor, edge in sorted(adjacency.get(face_index, [])):
                expected = int(assignments[face_index]) ^ required_xor
                if assignments[neighbor] < 0:
                    assignments[neighbor] = expected
                    queue.append(neighbor)
                elif int(assignments[neighbor]) != expected:
                    parity_conflicts[edge] = {
                        "edge_vertex_indices": [int(edge[0]), int(edge[1])],
                        "face_indices": sorted([int(face_index), int(neighbor)]),
                        "required_xor": int(required_xor),
                        "observed_xor": int(assignments[face_index])
                        ^ int(assignments[neighbor]),
                    }

    if parity_conflicts:
        report = {
            "schema": "campaignx.scrollfiesta_orientation_repair.v1",
            "method": "MANIFOLD_SHARED_EDGE_XOR_PROPAGATION_V1",
            "triangle_count": int(len(mesh.faces)),
            "shared_edge_count": int(shared_edge_count),
            "face_component_count": int(face_component_count),
            "inconsistent_winding_edge_count_before": int(inconsistent_before),
            "parity_conflict_count": int(len(parity_conflicts)),
            "parity_conflicts": [parity_conflicts[key] for key in sorted(parity_conflicts)],
            "orientable_by_face_flips": False,
            "vertices_changed": False,
            "connectivity_changed": False,
            "triangle_membership_changed": False,
            "physical_mesh_fusion_performed": False,
        }
        raise OrientationError(
            "face-orientation parity conflict proves the selected mesh is "
            "non-orientable by triangle flips",
            report=report,
        )

    repaired_faces = mesh.faces.copy()
    flipped_rows = np.flatnonzero(assignments == 1)
    if len(flipped_rows):
        repaired_faces[flipped_rows, 1], repaired_faces[flipped_rows, 2] = (
            repaired_faces[flipped_rows, 2].copy(),
            repaired_faces[flipped_rows, 1].copy(),
        )
    repaired = ObjMesh(
        vertices=mesh.vertices.copy(),
        faces=repaired_faces,
        vertex_trailing_fields=mesh.vertex_trailing_fields,
        passthrough_lines=mesh.passthrough_lines,
        source_triangle_count=mesh.source_triangle_count,
        dropped_degenerate_triangle_count=mesh.dropped_degenerate_triangle_count,
    )

    repaired_edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face in repaired.faces:
        a, b, c = (int(value) for value in face)
        for left, right in ((a, b), (b, c), (c, a)):
            repaired_edges[(min(left, right), max(left, right))].append((left, right))
    inconsistent_after = sum(
        len(rows) == 2 and rows[0] == rows[1] for rows in repaired_edges.values()
    )
    if inconsistent_after:
        raise OrientationError(
            "internal error: deterministic orientation repair left "
            f"{inconsistent_after} inconsistent shared edges"
        )

    report = {
        "schema": "campaignx.scrollfiesta_orientation_repair.v1",
        "method": "MANIFOLD_SHARED_EDGE_XOR_PROPAGATION_V1",
        "triangle_count": int(len(mesh.faces)),
        "shared_edge_count": int(shared_edge_count),
        "face_component_count": int(face_component_count),
        "inconsistent_winding_edge_count_before": int(inconsistent_before),
        "inconsistent_winding_edge_count_after": int(inconsistent_after),
        "parity_conflict_count": 0,
        "parity_conflicts": [],
        "flipped_face_count": int(len(flipped_rows)),
        "vertices_changed": False,
        "connectivity_changed": False,
        "triangle_membership_changed": False,
        "orientable_by_face_flips": True,
        "physical_mesh_fusion_performed": False,
    }
    return repaired, report


def _subset_and_compact(mesh: ObjMesh, retained_original_faces: np.ndarray) -> ObjMesh:
    faces = mesh.faces[retained_original_faces]
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return ObjMesh(
        vertices=mesh.vertices[used],
        faces=remap[faces],
        vertex_trailing_fields=tuple(
            mesh.vertex_trailing_fields[int(index)] for index in used
        ),
        passthrough_lines=mesh.passthrough_lines,
        source_triangle_count=mesh.source_triangle_count,
        dropped_degenerate_triangle_count=mesh.dropped_degenerate_triangle_count,
    )


def orient_with_conflict_quarantine(
    mesh: ObjMesh, *, maximum_quarantined_triangle_fraction: float
) -> tuple[ObjMesh, dict]:
    """Orient a mesh, quarantining only a bounded contradictory micro-patch.

    A parity contradiction can be introduced by a bad local weld.  Rather than
    accepting the non-orientable result or rewriting its geometry, this policy
    deterministically removes the smallest greedy set of implicated triangles
    while the frozen fraction cap permits it.  All retained vertices preserve
    their exact coordinates; downstream CT and self-intersection gates remain
    mandatory.
    """

    if (
        not np.isfinite(maximum_quarantined_triangle_fraction)
        or maximum_quarantined_triangle_fraction < 0
        or maximum_quarantined_triangle_fraction > 1
    ):
        raise ValueError("maximum quarantined triangle fraction must be in [0, 1]")
    initial_count = len(mesh.faces)
    if initial_count == 0:
        raise OrientationError("cannot orient an empty mesh")
    maximum_count = int(np.floor(initial_count * maximum_quarantined_triangle_fraction))
    retained_original = np.arange(initial_count, dtype=np.int64)
    quarantined: list[int] = []
    iterations: list[dict] = []

    while True:
        candidate = _subset_and_compact(mesh, retained_original)
        try:
            oriented, orientation = orient_triangle_faces(candidate)
            break
        except OrientationError as exc:
            if exc.report is None:
                raise
            if len(quarantined) >= maximum_count:
                raise OrientationError(
                    "orientation conflict quarantine would exceed the frozen "
                    f"{maximum_quarantined_triangle_fraction:.8f} triangle cap",
                    report={
                        **exc.report,
                        "quarantined_original_face_indices": quarantined,
                        "maximum_quarantined_triangle_count": maximum_count,
                    },
                ) from exc
            implicated_current = sorted(
                {
                    int(face)
                    for conflict in exc.report["parity_conflicts"]
                    for face in conflict["face_indices"]
                }
            )
            if not implicated_current:
                raise OrientationError(
                    "orientation solver reported a contradiction without "
                    "implicated triangles",
                    report=exc.report,
                ) from exc
            if len(implicated_current) > 64:
                raise OrientationError(
                    "orientation conflict is not a bounded micro-patch: "
                    f"{len(implicated_current)} implicated triangles",
                    report=exc.report,
                ) from exc

            scored: list[tuple[int, int, int]] = []
            for current_face in implicated_current:
                original_face = int(retained_original[current_face])
                trial_original = retained_original[retained_original != original_face]
                trial_mesh = _subset_and_compact(mesh, trial_original)
                try:
                    orient_triangle_faces(trial_mesh)
                    remaining_conflicts = 0
                except OrientationError as trial_exc:
                    if trial_exc.report is None:
                        raise
                    remaining_conflicts = int(
                        trial_exc.report.get("parity_conflict_count", initial_count)
                    )
                scored.append((remaining_conflicts, original_face, current_face))
            remaining_conflicts, selected_original, selected_current = min(scored)
            quarantined.append(selected_original)
            retained_original = retained_original[
                retained_original != selected_original
            ]
            iterations.append(
                {
                    "iteration": len(iterations) + 1,
                    "selected_original_face_index": selected_original,
                    "selected_current_face_index": selected_current,
                    "candidate_count": len(scored),
                    "remaining_parity_conflict_count": remaining_conflicts,
                }
            )

    report = {
        "schema": "campaignx.scrollfiesta_orientation_quarantine.v1",
        "method": "GREEDY_MIN_CONFLICT_FACE_QUARANTINE_THEN_XOR_ORIENTATION_V1",
        "source_triangle_count": int(initial_count),
        "retained_triangle_count": int(len(oriented.faces)),
        "quarantined_triangle_count": int(len(quarantined)),
        "quarantined_triangle_fraction": float(len(quarantined) / initial_count),
        "maximum_quarantined_triangle_fraction": float(
            maximum_quarantined_triangle_fraction
        ),
        "quarantined_original_face_indices": quarantined,
        "iterations": iterations,
        "orientation": orientation,
        "vertices_moved": False,
        "retained_triangle_connectivity_changed": False,
        "mesh_topology_changed": bool(quarantined),
        "triangle_membership_changed": bool(quarantined),
        "physical_mesh_fusion_performed": False,
    }
    return oriented, report
