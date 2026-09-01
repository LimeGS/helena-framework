"""A grid point no triangle uses is not a broken mesh.

The control reached P4 and the orientation proof answered

    status UNPROVEN, reason_code GEOMETRY_INVALID, error_type ValueError

from `_grown_vertex_normals`: "grown mesh contains a vertex with no finite
area-weighted normal". Measured on the control's own mesh:

    vertices 5184, faces 8978, all coordinates finite
    zero-normal vertices                 560
      .. of those with NO incident face  560
      .. of those with incident faces      0
    zero-area faces                        0 of 8978

Every one of the 560 is isolated. A TIFXYZ grid has holes -- a quad whose
corners are not all valid is not triangulated -- and the grid points around
those holes remain in the vertex array while belonging to no triangle. They
carry no winding, contribute to no face, and cannot be compared to anything.
The proof was rejecting the whole geometry over points that are not part of
the surface it is proving.

This is deliberately NOT a relaxation, and the tests below are written to hold
that line:

* a vertex that DOES have incident faces and still sums to a zero normal is a
  real fold, and stays fatal;
* a non-finite or zero-area triangle stays fatal;
* isolated vertices are skipped in the correspondence walk too, not merely
  allowed through the first gate. Leaving them with a zero normal would have
  moved the same rejection downstream to NONFINITE_OR_ZERO_NORMAL_DOT, which is
  the trap this pass is avoiding, not the fix.

What the proof still requires is unchanged: every vertex that participates in
the surface must have a finite normal, and parity must be decided by absolute
evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/02-flattening"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from orientation_parity import _grown_vertex_normals  # noqa: E402


def _square_with_spare_point():
    """Two triangles, plus a grid point nothing references -- the shape a
    TIFXYZ hole leaves behind."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [9.0, 9.0, 9.0],   # isolated: no face uses it
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def test_an_isolated_vertex_does_not_condemn_the_mesh() -> None:
    vertices, faces = _square_with_spare_point()

    normals, usable = _grown_vertex_normals(vertices, faces)

    assert usable.sum() == 4, "the four cornered vertices are the surface"
    assert not usable[4], "the isolated point is not part of the surface"
    for index in range(4):
        assert np.isclose(np.linalg.norm(normals[index]), 1.0)


def test_a_real_fold_is_still_fatal() -> None:
    """A vertex whose incident faces cancel exactly is a degeneracy, not a hole.
    This is the case the original check existed for and it must keep failing."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    # Two triangles around vertex 0 with opposite winding: the summed normal is
    # exactly zero while every vertex has incident faces.
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=np.int64)

    with pytest.raises(ValueError, match="no finite area-weighted normal"):
        _grown_vertex_normals(vertices, faces)


def test_a_zero_area_triangle_is_still_fatal() -> None:
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="non-finite or zero-area triangle"):
        _grown_vertex_normals(vertices, faces)


def test_a_mesh_with_no_faces_is_still_fatal() -> None:
    with pytest.raises(ValueError, match="no triangle faces"):
        _grown_vertex_normals(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int64))


def test_an_out_of_bounds_face_is_still_fatal() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 7]], dtype=np.int64)

    with pytest.raises(ValueError, match="out of bounds"):
        _grown_vertex_normals(vertices, faces)


def test_the_proof_skips_isolated_vertices_rather_than_comparing_them() -> None:
    """The other half. If an isolated vertex reached the correspondence walk it
    would carry a zero normal, the dot product would be 0.0, and the proof would
    fail with NONFINITE_OR_ZERO_NORMAL_DOT -- the same refusal wearing a
    different reason code."""
    from orientation_parity import prove_orientation

    # A reference grid the grown square sits exactly on.
    grid = np.zeros((2, 2, 3), dtype=np.float64)
    grid[0, 0] = [0.0, 0.0, 0.0]
    grid[0, 1] = [1.0, 0.0, 0.0]
    grid[1, 0] = [0.0, 1.0, 0.0]
    grid[1, 1] = [1.0, 1.0, 0.0]

    vertices, faces = _square_with_spare_point()
    # The frozen policy from the parity suite, with only the two spatial
    # numbers narrowed to this two-triangle fixture. Copying the whole dict by
    # hand is how a fixture drifts from the policy it claims to exercise.
    policy = {
        "profile_id": "first-letters-orientation-parity@1.0.0",
        "maximum_distance_ct_l0_voxels": 2.0,
        "minimum_correspondences": 1,
        "maximum_sampled_correspondences": 1024,
        "minimum_sign_consensus": 0.95,
        "minimum_median_absolute_dot": 0.90,
        "spatial_cell_edge_voxels": 4.0,
        "distance_tie_epsilon_squared_voxels": 1e-12,
        "maximum_reference_triangles": 250000,
        "maximum_grown_vertices": 2000000,
        "maximum_spatial_index_insertions": 4000000,
        "maximum_candidates_per_vertex": 4096,
        "maximum_elapsed_seconds": 60.0,
    }

    receipt = prove_orientation(grid, vertices, faces, {"note": "unit"}, policy)

    assert receipt["reason_code"] != "NONFINITE_OR_ZERO_NORMAL_DOT", (
        "the isolated vertex reached the comparison and failed it there")
    assert receipt["reason_code"] != "GEOMETRY_INVALID"
