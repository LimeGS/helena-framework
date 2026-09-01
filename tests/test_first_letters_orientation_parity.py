from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/02-flattening"))

from fleet.finalizer import triangulate_tifxyz_grid  # noqa: E402
import orientation_parity as orientation  # noqa: E402
from orientation_parity import prove_orientation  # noqa: E402


def grid(size: int = 9):
    y, x = np.mgrid[:size, :size].astype(np.float64)
    return np.stack((x, y, np.zeros_like(x)), axis=-1)


def absolute_receipt(verified: bool) -> dict:
    if not verified:
        return {
            "verified": False,
            "evidence_receipt_sha256": None,
            "same_winding_flip_normals": None,
        }
    evidence = {
        "schema": "campaignx.first_letters_absolute_orientation_evidence.v1",
        "reference_read_set": {
            "uri": "locked://w025",
            "objects": [],
            "canonical_manifest_sha256": "1" * 64,
        },
        "lineage": {
            "control_profile_id": "first-letters-control@1.0.0",
            "orientation_profile_id": "first-letters-orientation-parity@1.0.0",
        },
        "side_decision": {"same_winding_flip_normals": True},
    }
    evidence["receipt_sha256"] = hashlib.sha256(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return {
        "verified": True,
        "evidence_receipt_sha256": "6" * 64,
        "same_winding_flip_normals": True,
        "evidence": evidence,
    }


def lineage(verified: bool) -> dict:
    return {
        "reference": {"uri": "locked://w025", "read_set_sha256": "1" * 64},
        "grown_mesh_artifact": {"artifact_id": "grown", "sha256": "2" * 64,
                                "faces_sha256": "3" * 64},
        "flattened_artifact": {"artifact_id": "flat", "sha256": "4" * 64},
        "p3": {"job_id": "p3", "profile_id": "flatten@1", "receipt_sha256": "5" * 64},
        "absolute_orientation": absolute_receipt(verified),
    }


def policy(**changes) -> dict:
    value = {
        "profile_id": "first-letters-orientation-parity@1.0.0",
        "maximum_distance_ct_l0_voxels": 2.0,
        "minimum_correspondences": 64,
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
    value.update(changes)
    return value


def test_shared_tifxyz_triangulation_uses_the_frozen_antidiagonal_per_triangle():
    xyz = grid(2)
    mesh = triangulate_tifxyz_grid(xyz)
    assert mesh["faces"].tolist() == [[0, 2, 1], [2, 3, 1]]
    xyz[0, 0] = -1
    mesh = triangulate_tifxyz_grid(xyz)
    assert mesh["faces"].tolist() == [[2, 3, 1]]


def test_parity_is_deterministic_but_absolute_side_fails_closed_without_evidence():
    reference = grid()
    mesh = triangulate_tifxyz_grid(reference)
    first = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                              lineage(False), policy())
    second = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                               lineage(False), policy())
    assert first == second
    assert first["parity_state"] == "PROVEN_SAME_WINDING"
    assert first["status"] == "UNPROVEN"
    assert first["reason_code"] == "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING"
    assert first["selected_flip_normals"] is None
    assert first["receipt_sha256"] == second["receipt_sha256"]


def test_verified_synthetic_absolute_receipt_selects_boolean_from_parity():
    reference = grid()
    mesh = triangulate_tifxyz_grid(reference)
    same = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                             lineage(True), policy())
    opposite = prove_orientation(reference, mesh["vertices"], mesh["faces"][:, ::-1],
                                 lineage(True), policy())
    assert same["status"] == "PROVEN" and same["selected_flip_normals"] is True
    assert opposite["parity_state"] == "PROVEN_OPPOSITE_WINDING"
    assert opposite["status"] == "PROVEN" and opposite["selected_flip_normals"] is False


def test_absolute_orientation_boolean_and_digest_without_verified_receipt_fail_closed():
    reference = grid()
    mesh = triangulate_tifxyz_grid(reference)
    forged = lineage(False)
    forged["absolute_orientation"] = {
        "verified": True,
        "evidence_receipt_sha256": "a" * 64,
        "same_winding_flip_normals": True,
    }
    result = prove_orientation(
        reference, mesh["vertices"], mesh["faces"], forged, policy())
    assert result["status"] == "UNPROVEN"
    assert result["reason_code"] == "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING"


def test_orientation_caps_fail_closed_instead_of_starting_unbounded_search():
    reference = grid()
    mesh = triangulate_tifxyz_grid(reference)
    result = prove_orientation(
        reference, mesh["vertices"], mesh["faces"], lineage(True),
        policy(maximum_reference_triangles=1))
    assert result["status"] == "UNPROVEN"
    assert result["reason_code"] == "REFERENCE_TRIANGLE_CAP_EXCEEDED"
    assert result["selected_flip_normals"] is None

    result = prove_orientation(
        reference, mesh["vertices"], mesh["faces"], lineage(True),
        policy(maximum_grown_vertices=1))
    assert result["reason_code"] == "GROWN_VERTEX_CAP_EXCEEDED"

    result = prove_orientation(
        reference, mesh["vertices"], mesh["faces"], lineage(True),
        policy(maximum_spatial_index_insertions=1))
    assert result["reason_code"] == "SPATIAL_INDEX_INSERTION_CAP_EXCEEDED"

    result = prove_orientation(
        reference, mesh["vertices"], mesh["faces"], lineage(True),
        policy(maximum_candidates_per_vertex=0))
    assert result["reason_code"] == "CANDIDATES_PER_VERTEX_CAP_EXCEEDED"


def test_orientation_rejects_63_correspondences_and_invalid_grown_normals():
    reference = grid()
    mesh = triangulate_tifxyz_grid(reference)
    mesh_63 = triangulate_tifxyz_grid(reference[:7])
    result = prove_orientation(reference, mesh_63["vertices"], mesh_63["faces"],
                               lineage(True), policy())
    assert result["status"] == "UNPROVEN"
    assert result["reason_code"] == "INSUFFICIENT_CORRESPONDENCES"
    assert result["retained_correspondence_count"] == 63

    zero_area_faces = mesh["faces"].copy()
    zero_area_faces[0] = zero_area_faces[0, 0]
    result = prove_orientation(reference, mesh["vertices"], zero_area_faces,
                               lineage(True), policy())
    assert result["status"] == "UNPROVEN"
    assert result["reason_code"] == "GEOMETRY_INVALID"

    nonfinite_vertices = mesh["vertices"].copy()
    nonfinite_vertices[0, 0] = np.nan
    result = prove_orientation(reference, nonfinite_vertices, mesh["faces"],
                               lineage(True), policy())
    assert result["status"] == "UNPROVEN"
    assert result["reason_code"] == "GROWN_VERTICES_INVALID"



def _all_used(normals):
    """A double for a mesh where every vertex takes part in a face.

    `_grown_vertex_normals` returns the normals and a mask of the vertices that
    any triangle actually uses; these fixtures are dense meshes with no holes,
    so the mask is all True. Spelled out rather than assumed, because a double
    that quietly disagrees with the contract is how the last three defects hid.
    """
    import numpy as np

    return normals, np.ones(len(normals), dtype=bool)

def test_orientation_threshold_boundaries_are_exact(monkeypatch):
    reference = grid(10)
    mesh = triangulate_tifxyz_grid(reference)
    reference_direction = np.asarray([0.0, 0.0, -1.0])

    def normals_with(opposite: int, absolute_dot: float = 1.0):
        normals = np.tile(reference_direction, (len(mesh["vertices"]), 1))
        tangent = float(np.sqrt(max(0.0, 1.0 - absolute_dot ** 2)))
        normals[:] = [tangent, 0.0, -absolute_dot]
        normals[:opposite] *= -1
        return normals

    monkeypatch.setattr(orientation, "_grown_vertex_normals",
                        lambda _vertices, _faces: _all_used(normals_with(5)))
    exact = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                              lineage(True), policy())
    assert exact["status"] == "PROVEN" and exact["sign_consensus"] == 0.95

    monkeypatch.setattr(orientation, "_grown_vertex_normals",
                        lambda _vertices, _faces: _all_used(normals_with(6)))
    below = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                              lineage(True), policy())
    assert below["reason_code"] == "SIGN_CONSENSUS_BELOW_THRESHOLD"

    monkeypatch.setattr(orientation, "_grown_vertex_normals",
                        lambda _vertices, _faces: _all_used(normals_with(0, 0.89)))
    weak = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                             lineage(True), policy())
    assert weak["reason_code"] == "MEDIAN_ABSOLUTE_DOT_BELOW_THRESHOLD"


def test_orientation_sampling_and_shared_edge_ties_are_deterministic():
    reference = grid(33)
    mesh = triangulate_tifxyz_grid(reference)
    result = prove_orientation(reference, mesh["vertices"], mesh["faces"],
                               lineage(True), policy())
    assert result["status"] == "PROVEN"
    assert len(result["sample_indices"]) == 1024
    assert len(set(result["sample_indices"])) == 1024
    assert result["sample_indices"][0] == 0
    assert result["sample_indices"][-1] == result["retained_correspondence_count"] - 1
    # Grid vertices lie on multiple reference triangles. The lower frozen
    # row-major ordinal wins every equal-distance shared-edge/vertex tie.
    origin = next(row for row in result["samples"] if row["grown_vertex_index"] == 0)
    assert origin["reference_triangle_ordinal"] == 0
