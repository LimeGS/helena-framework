"""Fail-closed conversion of ScrollFiesta OBJ coordinates to Helena Framework.

ScrollFiesta's welded OBJ stores the first three vertex fields as ``Z Y X``.
Helena Framework and ``vc_obj2tifxyz_legacy`` consume ``X Y Z``.  Reversing three
axes is a reflection, so the triangle winding must be flipped exactly once.
The conversion marker emitted here prevents accidentally applying the same
operation to an already-canonical OBJ.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRANSFORM_MARKER = "# helena-coordinate-transform=ZYX_TO_XYZ_V1"
_PASSTHROUGH_PREFIXES = ("o ", "g ", "usemtl ", "mtllib ", "s ")


class CoordinateTransformError(ValueError):
    """Raised when OBJ geometry cannot cross the coordinate boundary safely."""


@dataclass(frozen=True)
class ObjMesh:
    """Strict triangle OBJ geometry plus optional per-vertex trailing fields."""

    vertices: np.ndarray
    faces: np.ndarray
    vertex_trailing_fields: tuple[tuple[str, ...], ...]
    passthrough_lines: tuple[str, ...]
    source_triangle_count: int
    dropped_degenerate_triangle_count: int


def _parse_face_index(token: str, *, vertex_count: int, path: Path) -> int:
    head = token.split("/", 1)[0]
    try:
        value = int(head)
    except ValueError as exc:
        raise CoordinateTransformError(f"{path}: invalid face index {token!r}") from exc
    if value <= 0:
        raise CoordinateTransformError(
            f"{path}: relative, zero, and negative OBJ indices are forbidden"
        )
    index = value - 1
    if index >= vertex_count:
        raise CoordinateTransformError(
            f"{path}: face index {value} exceeds {vertex_count} parsed vertices"
        )
    return index


def load_triangle_obj(
    path: Path,
    *,
    reject_transform_marker: bool = False,
    drop_degenerate_triangles: bool = False,
) -> ObjMesh:
    """Load a strict finite triangle OBJ and reject ambiguous geometry.

    ScrollFiesta emits triangles.  Fan-triangulating arbitrary polygons here
    would silently change topology, so non-triangular faces are rejected.
    """

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CoordinateTransformError(f"cannot read OBJ {path}: {exc}") from exc
    if reject_transform_marker and any(line.startswith(TRANSFORM_MARKER) for line in lines):
        raise CoordinateTransformError(
            f"{path}: coordinate transform marker already present; refusing double transform"
        )

    vertices: list[tuple[float, float, float]] = []
    trailing: list[tuple[str, ...]] = []
    face_tokens: list[tuple[str, str, str]] = []
    passthrough: list[str] = []
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            fields = line.split()
            if len(fields) < 4:
                raise CoordinateTransformError(
                    f"{path}:{line_number}: vertex needs three coordinates"
                )
            try:
                xyz = tuple(float(value) for value in fields[1:4])
            except ValueError as exc:
                raise CoordinateTransformError(
                    f"{path}:{line_number}: invalid vertex coordinates"
                ) from exc
            if not all(np.isfinite(value) for value in xyz):
                raise CoordinateTransformError(
                    f"{path}:{line_number}: NaN or infinite coordinate is forbidden"
                )
            vertices.append(xyz)
            trailing.append(tuple(fields[4:]))
        elif line.startswith("f "):
            fields = line.split()[1:]
            if len(fields) != 3:
                raise CoordinateTransformError(
                    f"{path}:{line_number}: expected a triangle, received {len(fields)} vertices"
                )
            face_tokens.append((fields[0], fields[1], fields[2]))
        elif line.startswith(_PASSTHROUGH_PREFIXES):
            passthrough.append(raw.rstrip("\n"))

    if len(vertices) < 3 or not face_tokens:
        raise CoordinateTransformError(f"{path}: OBJ must contain vertices and triangles")

    faces = np.asarray(
        [
            tuple(
                _parse_face_index(token, vertex_count=len(vertices), path=path)
                for token in tokens
            )
            for tokens in face_tokens
        ],
        dtype=np.int64,
    )
    vertex_array = np.asarray(vertices, dtype=np.float64)
    a = vertex_array[faces[:, 1]] - vertex_array[faces[:, 0]]
    b = vertex_array[faces[:, 2]] - vertex_array[faces[:, 0]]
    doubled_areas = np.linalg.norm(np.cross(a, b), axis=1)
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    degenerate = repeated | ~np.isfinite(doubled_areas) | (doubled_areas <= 1e-12)
    dropped = int(np.count_nonzero(degenerate))
    if dropped and not drop_degenerate_triangles:
        raise CoordinateTransformError(f"{path}: degenerate triangle is forbidden")
    if dropped:
        faces = faces[~degenerate]
        if not len(faces):
            raise CoordinateTransformError(f"{path}: all triangles are degenerate")

    return ObjMesh(
        vertices=vertex_array,
        faces=faces,
        vertex_trailing_fields=tuple(trailing),
        passthrough_lines=tuple(passthrough),
        source_triangle_count=len(face_tokens),
        dropped_degenerate_triangle_count=dropped,
    )


def coordinate_matrix(*, level: int) -> list[list[float]]:
    """Return the homogeneous native-ZYX to level-0-XYZ matrix."""

    if level < 0:
        raise CoordinateTransformError("level must be non-negative")
    scale = float(2**level)
    return [
        [0.0, 0.0, scale, 0.0],
        [0.0, scale, 0.0, 0.0],
        [scale, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_native_zyx_to_canonical_xyz(
    source_obj: Path,
    destination_obj: Path,
    *,
    level: int,
    drop_degenerate_triangles: bool = False,
) -> ObjMesh:
    """Apply the sole allowed ZYX→XYZ transform and one winding flip."""

    source_obj = Path(source_obj)
    destination_obj = Path(destination_obj)
    if destination_obj.exists():
        raise CoordinateTransformError(f"output already exists: {destination_obj}")
    if level < 0:
        raise CoordinateTransformError("level must be non-negative")

    native = load_triangle_obj(
        source_obj,
        reject_transform_marker=True,
        drop_degenerate_triangles=drop_degenerate_triangles,
    )
    scale = float(2**level)
    canonical_vertices = native.vertices[:, ::-1] * scale
    canonical_faces = native.faces[:, [0, 2, 1]]

    destination_obj.parent.mkdir(parents=True, exist_ok=True)
    with destination_obj.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f"{TRANSFORM_MARKER} source_order=ZYX canonical_order=XYZ "
            f"winding_flip=1 level={level} scale={scale:g}\n"
        )
        for line in native.passthrough_lines:
            stream.write(f"{line}\n")
        for vertex, rest in zip(
            canonical_vertices, native.vertex_trailing_fields, strict=True
        ):
            suffix = f" {' '.join(rest)}" if rest else ""
            stream.write(
                f"v {vertex[0]:.10f} {vertex[1]:.10f} {vertex[2]:.10f}{suffix}\n"
            )
        for a, b, c in canonical_faces + 1:
            stream.write(f"f {a} {b} {c}\n")

    canonical = load_triangle_obj(destination_obj)
    if not np.array_equal(canonical.faces, canonical_faces):
        raise CoordinateTransformError("canonical OBJ winding did not round-trip")
    maximum_error = float(np.max(np.abs(canonical.vertices - canonical_vertices)))
    if maximum_error > 1e-4:
        raise CoordinateTransformError(
            f"canonical OBJ coordinate error {maximum_error} exceeds 1e-4 voxel"
        )
    return canonical
