"""Deterministic single-lamina selection for welded ScrollFiesta meshes."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .coordinate_transform import ObjMesh


def _face_components(mesh: ObjMesh) -> list[np.ndarray]:
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    for face_index, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_index)
    unseen = set(range(len(mesh.faces)))
    components: list[np.ndarray] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        rows: list[int] = []
        while queue:
            face_index = queue.popleft()
            rows.append(face_index)
            for vertex in mesh.faces[face_index]:
                for neighbor in vertex_faces[int(vertex)]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        components.append(np.asarray(sorted(rows), dtype=np.int64))
    return sorted(components, key=lambda rows: (-len(rows), int(rows[0])))


def _reindex_component(mesh: ObjMesh, face_rows: np.ndarray) -> ObjMesh:
    faces = mesh.faces[face_rows]
    used = np.unique(faces.reshape(-1))
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return ObjMesh(
        vertices=mesh.vertices[used],
        faces=remap[faces],
        vertex_trailing_fields=tuple(mesh.vertex_trailing_fields[int(i)] for i in used),
        passthrough_lines=mesh.passthrough_lines,
        source_triangle_count=len(face_rows),
        dropped_degenerate_triangle_count=0,
    )


def select_component_nearest_point(
    mesh: ObjMesh, point: tuple[float, float, float]
) -> tuple[ObjMesh, dict]:
    """Select one component by its nearest vertex to a frozen seed."""

    target = np.asarray(point, dtype=np.float64)
    if target.shape != (3,) or np.any(~np.isfinite(target)):
        raise ValueError("component selection point must be three finite coordinates")
    components = _face_components(mesh)
    rows = []
    for rank, face_rows in enumerate(components, start=1):
        vertices = np.unique(mesh.faces[face_rows].reshape(-1))
        distances = np.linalg.norm(mesh.vertices[vertices] - target, axis=1)
        rows.append(
            {
                "component_rank_by_faces": rank,
                "first_source_face_index": int(face_rows[0]),
                "face_count": int(len(face_rows)),
                "vertex_count": int(len(vertices)),
                "minimum_seed_distance_voxels": float(distances.min()),
                "face_rows": face_rows,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["minimum_seed_distance_voxels"],
            -row["face_count"],
            row["first_source_face_index"],
        ),
    )
    report = {
        "schema": "campaignx.scrollfiesta_component_selection.v1",
        "policy": "NEAREST_VERTEX_TO_FROZEN_SEED",
        "seed_native_zyx": [float(value) for value in target],
        "component_count": len(rows),
        "selected_component_rank_by_faces": selected["component_rank_by_faces"],
        "selected_minimum_seed_distance_voxels": selected[
            "minimum_seed_distance_voxels"
        ],
        "components": [
            {key: value for key, value in row.items() if key != "face_rows"}
            for row in rows
        ],
        "physical_mesh_fusion_performed": False,
    }
    return _reindex_component(mesh, selected["face_rows"]), report


def write_triangle_obj(path: Path, mesh: ObjMesh, *, header: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"# {header}\n")
        for line in mesh.passthrough_lines:
            stream.write(f"{line}\n")
        for vertex, rest in zip(
            mesh.vertices, mesh.vertex_trailing_fields, strict=True
        ):
            suffix = f" {' '.join(rest)}" if rest else ""
            stream.write(
                f"v {vertex[0]:.10f} {vertex[1]:.10f} {vertex[2]:.10f}{suffix}\n"
            )
        for a, b, c in mesh.faces + 1:
            stream.write(f"f {a} {b} {c}\n")
