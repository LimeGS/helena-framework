"""Deterministic PCA UV initialization for numerically difficult CT meshes.

This module also owns the post-optimization UV distortion measurement.  The
initialization counters below (``initial_flipped_triangle_count``) describe the
*input* handed to SLIM; they say nothing about the solver's output.  The
``uv_distortion_metrics`` producer at the bottom of this file measures the
flattened OBJ that flatboi actually wrote, which is the artifact the profile's
``uv_flipped_triangles`` and stretch gates are declared against.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .coordinate_transform import ObjMesh


def _fix_vector_sign(vector: np.ndarray) -> np.ndarray:
    index = int(np.argmax(np.abs(vector)))
    return -vector if vector[index] < 0 else vector


def write_pca_uv_obj(mesh: ObjMesh, output: Path) -> dict:
    """Write the same 3D mesh with deterministic per-vertex PCA UVs.

    PCA is only an initialization for SLIM.  The 3D vertices and triangle
    indices are preserved exactly; downstream G4 still rejects flips or high
    distortion in the optimized result.
    """

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    centered = mesh.vertices - mesh.vertices.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:2]].copy()
    basis[:, 0] = _fix_vector_sign(basis[:, 0])
    basis[:, 1] = _fix_vector_sign(basis[:, 1])
    uv = centered @ basis

    triangles = uv[mesh.faces]
    signed_twice_area = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    if int(np.count_nonzero(signed_twice_area < 0)) > int(
        np.count_nonzero(signed_twice_area > 0)
    ):
        uv[:, 1] *= -1.0
        signed_twice_area *= -1.0
        basis[:, 1] *= -1.0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# campaignx deterministic PCA UV initialization v1\n")
        for vertex in mesh.vertices:
            # Flatboi/libigl requires an Nx3 position matrix.  ScrollFiesta
            # appends RGB values to native OBJ vertices; carrying those fields
            # into this solver-only OBJ makes igl::readOBJ construct an Nx6
            # matrix and overflows libigl's fixed-size 3-vector in grad_tri.
            # The coloured source OBJ remains preserved separately.
            stream.write(
                f"v {vertex[0]:.10f} {vertex[1]:.10f} {vertex[2]:.10f}\n"
            )
        for u, v in uv:
            stream.write(f"vt {u:.10f} {v:.10f}\n")
        for a, b, c in mesh.faces + 1:
            stream.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    epsilon = 1e-12
    return {
        "schema": "campaignx.scrollfiesta_uv_initialization.v1",
        "method": "PCA_TOP_TWO_EIGENVECTORS_WITH_DETERMINISTIC_SIGNS",
        "eigenvalues_descending": [float(eigenvalues[i]) for i in order],
        "basis_xyz_to_uv": basis.tolist(),
        "triangle_count": int(len(mesh.faces)),
        "initial_flipped_triangle_count": int(
            np.count_nonzero(signed_twice_area < -epsilon)
        ),
        "initial_degenerate_triangle_count": int(
            np.count_nonzero(np.abs(signed_twice_area) <= epsilon)
        ),
        "coordinates_3d_changed": False,
        "solver_obj_vertex_trailing_fields_stripped": True,
        "physical_mesh_fusion_performed": False,
    }


class UvDistortionError(ValueError):
    """Raised when a flattened OBJ cannot be measured unambiguously."""


def load_uv_mapped_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a flattened OBJ into 3D vertices, faces, and per-corner UVs.

    Per-corner UVs are kept instead of a per-vertex array because a flattening
    solver is free to cut seams, in which case one 3D vertex carries several
    distinct texture coordinates.  Collapsing them would silently average
    across the cut and hide exactly the distortion this measurement exists to
    detect.
    """

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise UvDistortionError(f"cannot read flattened OBJ {path}: {exc}") from exc

    vertices: list[tuple[float, float, float]] = []
    texture: list[tuple[float, float]] = []
    face_vertices: list[tuple[int, int, int]] = []
    face_texture: list[tuple[int, int, int]] = []
    for number, line in enumerate(lines, start=1):
        if line.startswith("v "):
            fields = line.split()[1:]
            if len(fields) < 3:
                raise UvDistortionError(f"{path}:{number}: vertex needs three coordinates")
            vertices.append(tuple(float(value) for value in fields[:3]))
        elif line.startswith("vt "):
            fields = line.split()[1:]
            if len(fields) < 2:
                raise UvDistortionError(f"{path}:{number}: vt needs two coordinates")
            texture.append(tuple(float(value) for value in fields[:2]))
        elif line.startswith("f "):
            tokens = line.split()[1:]
            if len(tokens) != 3:
                raise UvDistortionError(
                    f"{path}:{number}: only triangles can be measured, found "
                    f"{len(tokens)} corners"
                )
            corner_vertices: list[int] = []
            corner_texture: list[int] = []
            for token in tokens:
                parts = token.split("/")
                if len(parts) < 2 or not parts[1]:
                    raise UvDistortionError(
                        f"{path}:{number}: face corner {token!r} carries no UV index"
                    )
                try:
                    vertex_index = int(parts[0])
                    texture_index = int(parts[1])
                except ValueError as exc:
                    raise UvDistortionError(
                        f"{path}:{number}: invalid face corner {token!r}"
                    ) from exc
                if vertex_index <= 0 or texture_index <= 0:
                    raise UvDistortionError(
                        f"{path}:{number}: relative, zero, and negative OBJ indices "
                        "are forbidden"
                    )
                corner_vertices.append(vertex_index - 1)
                corner_texture.append(texture_index - 1)
            face_vertices.append(tuple(corner_vertices))
            face_texture.append(tuple(corner_texture))

    if not vertices or not texture or not face_vertices:
        raise UvDistortionError(
            f"{path}: a UV-mapped OBJ needs v, vt, and f records; found "
            f"{len(vertices)}/{len(texture)}/{len(face_vertices)}"
        )
    vertex_array = np.asarray(vertices, dtype=np.float64)
    texture_array = np.asarray(texture, dtype=np.float64)
    if not np.all(np.isfinite(vertex_array)) or not np.all(np.isfinite(texture_array)):
        raise UvDistortionError(f"{path}: NaN or infinite coordinates are forbidden")
    face_vertex_array = np.asarray(face_vertices, dtype=np.int64)
    face_texture_array = np.asarray(face_texture, dtype=np.int64)
    if int(face_vertex_array.max()) >= len(vertex_array):
        raise UvDistortionError(f"{path}: a face references an unparsed vertex")
    if int(face_texture_array.max()) >= len(texture_array):
        raise UvDistortionError(f"{path}: a face references an unparsed vt record")
    return vertex_array, face_vertex_array, texture_array[face_texture_array]


def uv_distortion_metrics(
    vertices_xyz: np.ndarray,
    faces: np.ndarray,
    face_uv: np.ndarray,
) -> dict:
    """Measure per-triangle UV→3D Jacobian singular values and flips.

    For triangle ``(P0, P1, P2)`` with corner UVs ``(q0, q1, q2)`` the affine
    map from UV to 3D is ``J = [P1-P0, P2-P0] @ inv([q1-q0, q2-q0])``, a 3x2
    matrix whose two singular values ``s1 >= s2`` are the principal stretch
    factors.  Because a flattening is only defined up to a global scale, the
    singular values are normalized by ``sqrt(total 3D area / total UV area)``
    before the per-triangle quasi-isometric distortion ``max(s1, 1/s2) >= 1``
    is formed.  A perfectly isometric flattening therefore reports 1.0.

    ``uv_flipped_triangle_count`` counts triangles whose UV signed area has the
    sign opposite to the mesh majority: the 3D mesh is consistently oriented
    before flattening, so an orientation-preserving flattening must not produce
    two signs at all.
    """

    vertices_xyz = np.asarray(vertices_xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_uv = np.asarray(face_uv, dtype=np.float64)
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise UvDistortionError("faces must be a non-empty (M, 3) triangle array")
    if face_uv.shape != (len(faces), 3, 2):
        raise UvDistortionError("face_uv must have shape (M, 3, 2)")

    corners = vertices_xyz[faces]
    edge_1 = corners[:, 1, :] - corners[:, 0, :]
    edge_2 = corners[:, 2, :] - corners[:, 0, :]
    delta_1 = face_uv[:, 1, :] - face_uv[:, 0, :]
    delta_2 = face_uv[:, 2, :] - face_uv[:, 0, :]

    determinant = delta_1[:, 0] * delta_2[:, 1] - delta_1[:, 1] * delta_2[:, 0]
    scale = float(np.max(np.abs(determinant)))
    epsilon = 1e-12 * scale if scale > 0 else 1e-12
    positive = int(np.count_nonzero(determinant > epsilon))
    negative = int(np.count_nonzero(determinant < -epsilon))
    degenerate = int(len(determinant) - positive - negative)
    flipped = min(positive, negative)

    uv_area = 0.5 * np.abs(determinant)
    triangle_area_3d = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
    total_uv_area = float(uv_area.sum())
    total_area_3d = float(triangle_area_3d.sum())
    if not np.isfinite(total_uv_area) or total_uv_area <= 0:
        raise UvDistortionError("flattened UV layout has no positive total area")
    if not np.isfinite(total_area_3d) or total_area_3d <= 0:
        raise UvDistortionError("3D mesh has no positive total area")
    normalization = float(np.sqrt(total_area_3d / total_uv_area))

    measurable = np.abs(determinant) > epsilon
    stretch = np.full(len(faces), np.inf, dtype=np.float64)
    if int(np.count_nonzero(measurable)):
        inverse = np.empty((int(np.count_nonzero(measurable)), 2, 2), dtype=np.float64)
        det_measurable = determinant[measurable]
        inverse[:, 0, 0] = delta_2[measurable, 1] / det_measurable
        inverse[:, 0, 1] = -delta_2[measurable, 0] / det_measurable
        inverse[:, 1, 0] = -delta_1[measurable, 1] / det_measurable
        inverse[:, 1, 1] = delta_1[measurable, 0] / det_measurable
        basis = np.stack((edge_1[measurable], edge_2[measurable]), axis=2)
        jacobian = basis @ inverse
        gram = np.transpose(jacobian, (0, 2, 1)) @ jacobian
        trace = gram[:, 0, 0] + gram[:, 1, 1]
        gram_determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
        discriminant = np.maximum(0.25 * trace * trace - gram_determinant, 0.0)
        root = np.sqrt(discriminant)
        sigma_1 = np.sqrt(np.maximum(0.5 * trace + root, 0.0)) / normalization
        sigma_2 = np.sqrt(np.maximum(0.5 * trace - root, 0.0)) / normalization
        with np.errstate(divide="ignore"):
            local = np.maximum(sigma_1, np.where(sigma_2 > 0, 1.0 / sigma_2, np.inf))
        stretch[measurable] = local

    finite = stretch[np.isfinite(stretch)]
    if finite.size == 0:
        raise UvDistortionError("no triangle produced a finite stretch measurement")
    return {
        "schema": "campaignx.scrollfiesta_uv_distortion.v1",
        "method": "UV_TO_XYZ_JACOBIAN_SINGULAR_VALUES_AREA_NORMALIZED_V1",
        "triangle_count": int(len(faces)),
        "uv_flipped_triangle_count": flipped,
        "uv_degenerate_triangle_count": degenerate,
        "uv_positive_orientation_triangle_count": positive,
        "uv_negative_orientation_triangle_count": negative,
        "measured_triangle_count": int(finite.size),
        "area_normalization_uv_to_um": normalization,
        "total_area_3d_um2": total_area_3d,
        "total_area_uv": total_uv_area,
        "stretch_p50": float(np.percentile(finite, 50)),
        "stretch_p95": float(np.percentile(finite, 95)),
        "stretch_max": float(finite.max()),
        "coordinates_3d_changed": False,
        "physical_mesh_fusion_performed": False,
    }


def _ordered_boundary_loop(mesh: ObjMesh) -> list[int]:
    counts: Counter[tuple[int, int]] = Counter()
    for face in mesh.faces:
        a, b, c = (int(value) for value in face)
        for left, right in ((a, b), (b, c), (c, a)):
            counts[tuple(sorted((left, right)))] += 1
    boundary_edges = [edge for edge, count in counts.items() if count == 1]
    neighbors: dict[int, list[int]] = defaultdict(list)
    for left, right in boundary_edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    if not neighbors or any(len(rows) != 2 for rows in neighbors.values()):
        raise ValueError("Tutte initialization requires exactly one simple boundary loop")
    start = min(neighbors)
    previous = -1
    current = start
    loop: list[int] = []
    while True:
        loop.append(current)
        options = sorted(value for value in neighbors[current] if value != previous)
        if not options:
            raise ValueError("boundary traversal terminated before closing")
        following = options[0]
        if following == start:
            break
        if following in loop:
            raise ValueError("multiple boundary loops or a boundary pinch were detected")
        previous, current = current, following
    if len(loop) != len(neighbors):
        raise ValueError("multiple boundary loops are not supported")
    return loop


def write_tutte_uv_obj(mesh: ObjMesh, output: Path) -> dict:
    """Write a flip-free uniform Tutte embedding for a topological disk."""

    loop = _ordered_boundary_loop(mesh)
    boundary = np.asarray(loop, dtype=np.int64)
    boundary_xyz = mesh.vertices[boundary]
    lengths = np.linalg.norm(
        np.roll(boundary_xyz, -1, axis=0) - boundary_xyz, axis=1
    )
    perimeter = float(lengths.sum())
    if not np.isfinite(perimeter) or perimeter <= 0:
        raise ValueError("boundary perimeter must be finite and positive")
    arc = np.concatenate(([0.0], np.cumsum(lengths[:-1]))) / perimeter
    boundary_uv = np.column_stack(
        (np.cos(2.0 * np.pi * arc), np.sin(2.0 * np.pi * arc))
    )

    adjacency: dict[int, set[int]] = defaultdict(set)
    for face in mesh.faces:
        a, b, c = (int(value) for value in face)
        for left, right in ((a, b), (b, c), (c, a)):
            adjacency[left].add(right)
            adjacency[right].add(left)
    boundary_index = {int(vertex): index for index, vertex in enumerate(boundary)}
    interior = [index for index in range(len(mesh.vertices)) if index not in boundary_index]
    interior_index = {vertex: index for index, vertex in enumerate(interior)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros((len(interior), 2), dtype=np.float64)
    for vertex in interior:
        row = interior_index[vertex]
        neighbors = sorted(adjacency[vertex])
        rows.append(row)
        cols.append(row)
        data.append(float(len(neighbors)))
        for neighbor in neighbors:
            if neighbor in interior_index:
                rows.append(row)
                cols.append(interior_index[neighbor])
                data.append(-1.0)
            else:
                rhs[row] += boundary_uv[boundary_index[neighbor]]
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(interior), len(interior))
    )
    solved = (
        np.column_stack(
            [sparse_linalg.spsolve(matrix, rhs[:, column]) for column in range(2)]
        )
        if interior
        else np.empty((0, 2), dtype=np.float64)
    )
    uv = np.zeros((len(mesh.vertices), 2), dtype=np.float64)
    uv[boundary] = boundary_uv
    uv[np.asarray(interior, dtype=np.int64)] = solved
    triangles = uv[mesh.faces]
    signed_twice_area = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    if int(np.count_nonzero(signed_twice_area < 0)) > int(
        np.count_nonzero(signed_twice_area > 0)
    ):
        uv[:, 1] *= -1.0
        signed_twice_area *= -1.0

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("# campaignx uniform Tutte UV initialization v1\n")
        for vertex in mesh.vertices:
            # See write_pca_uv_obj: Flatboi's libigl path is strictly Nx3.
            stream.write(
                f"v {vertex[0]:.10f} {vertex[1]:.10f} {vertex[2]:.10f}\n"
            )
        for u, v in uv:
            stream.write(f"vt {u:.10f} {v:.10f}\n")
        for a, b, c in mesh.faces + 1:
            stream.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    residual = matrix @ solved - rhs
    maximum_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    epsilon = 1e-12
    return {
        "schema": "campaignx.scrollfiesta_uv_initialization.v1",
        "method": "UNIFORM_TUTTE_BOUNDARY_CIRCLE_ARCLENGTH_V1",
        "boundary_vertex_count": int(len(boundary)),
        "interior_vertex_count": int(len(interior)),
        "linear_system_max_abs_residual": maximum_residual,
        "triangle_count": int(len(mesh.faces)),
        "initial_flipped_triangle_count": int(
            np.count_nonzero(signed_twice_area < -epsilon)
        ),
        "initial_degenerate_triangle_count": int(
            np.count_nonzero(np.abs(signed_twice_area) <= epsilon)
        ),
        "coordinates_3d_changed": False,
        "solver_obj_vertex_trailing_fields_stripped": True,
        "physical_mesh_fusion_performed": False,
    }
