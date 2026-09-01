"""Bounded, geometry-only winding parity for the First Letters control."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from fleet.finalizer import triangulate_tifxyz_grid


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _finish(receipt: dict[str, Any]) -> dict[str, Any]:
    return {**receipt, "receipt_sha256": canonical_sha256(receipt)}


def _unproven(base: dict[str, Any], reason: str, **fields: Any) -> dict[str, Any]:
    return _finish({
        **base, **fields, "status": "UNPROVEN", "reason_code": reason,
        "parity_state": fields.get("parity_state", "UNPROVEN"),
        "selected_flip_normals": None,
    })


def _closest_point_triangle(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Exact float64 closest point and barycentrics (Ericson region tests)."""
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (1.0, 0.0, 0.0)
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, (1.0 - v, v, 0.0)
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, (1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), (0.0, 1.0 - w, w)
    denominator = 1.0 / (va + vb + vc)
    v, w = vb * denominator, vc * denominator
    return a + ab * v + ac * w, (1.0 - v - w, v, w)


def _grown_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("grown mesh has no triangle faces")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("grown face index is out of bounds")
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    if not np.isfinite(crosses).all() or np.any(lengths <= 0.0):
        raise ValueError("grown mesh contains a non-finite or zero-area triangle")
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], crosses)
    normal_lengths = np.linalg.norm(normals, axis=1)

    # A TIFXYZ grid has holes: a quad whose corners are not all valid is never
    # triangulated, and the grid points around it stay in the vertex array while
    # belonging to no face. They carry no winding and cannot be compared to
    # anything, so requiring a normal of them rejected sound geometry -- 560 of
    # them on the control's own mesh, against zero real degeneracies.
    #
    # Not a relaxation. A vertex that DOES take part in faces and still sums to
    # nothing is a fold, and stays fatal below.
    used = np.zeros(len(vertices), dtype=bool)
    used[np.unique(faces)] = True

    if not np.isfinite(normals[used]).all() or np.any(normal_lengths[used] <= 0.0):
        raise ValueError("grown mesh contains a vertex with no finite area-weighted normal")

    unit = np.zeros_like(normals)
    unit[used] = normals[used] / normal_lengths[used][:, None]
    return unit, used


def prove_orientation(
    reference_xyz: Any,
    grown_vertices: Any,
    grown_faces: Any,
    lineage: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove winding parity and select a side only with absolute evidence."""
    base = {
        "schema": "campaignx.first_letters_orientation_parity.v1",
        "profile_id": policy.get("profile_id"),
        "profile_sha256": canonical_sha256(policy),
        "lineage": dict(lineage),
        "policy": dict(policy),
    }
    started = time.monotonic()
    try:
        reference = triangulate_tifxyz_grid(reference_xyz)
        ref_vertices = np.asarray(reference["vertices"], dtype=np.float64)
        ref_faces = np.asarray(reference["faces"], dtype=np.int64)
        ref_ordinals = np.asarray(reference["triangle_ordinals"], dtype=np.int64)
        vertices = np.asarray(grown_vertices, dtype=np.float64)
        faces = np.asarray(grown_faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            return _unproven(base, "GROWN_VERTICES_INVALID")
        if len(ref_faces) > int(policy["maximum_reference_triangles"]):
            return _unproven(base, "REFERENCE_TRIANGLE_CAP_EXCEEDED",
                             reference_triangle_count=int(len(ref_faces)))
        if len(vertices) > int(policy["maximum_grown_vertices"]):
            return _unproven(base, "GROWN_VERTEX_CAP_EXCEEDED",
                             grown_vertex_count=int(len(vertices)))
        ref_triangles = ref_vertices[ref_faces]
        ref_crosses = np.cross(ref_triangles[:, 1] - ref_triangles[:, 0],
                               ref_triangles[:, 2] - ref_triangles[:, 0])
        ref_lengths = np.linalg.norm(ref_crosses, axis=1)
        if not np.isfinite(ref_crosses).all() or np.any(ref_lengths <= 0.0):
            return _unproven(base, "REFERENCE_TRIANGLE_INVALID")
        ref_normals = ref_crosses / ref_lengths[:, None]
        grown_normals, grown_used = _grown_vertex_normals(vertices, faces)
    except (KeyError, TypeError, ValueError) as exc:
        return _unproven(base, "GEOMETRY_INVALID", error_type=type(exc).__name__)

    tolerance = float(policy["maximum_distance_ct_l0_voxels"])
    cell_edge = float(policy["spatial_cell_edge_voxels"])
    tie_epsilon = float(policy["distance_tie_epsilon_squared_voxels"])
    index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    insertions = 0
    for triangle_index, triangle in enumerate(ref_triangles):
        low = np.floor((triangle.min(axis=0) - tolerance) / cell_edge).astype(np.int64)
        high = np.floor((triangle.max(axis=0) + tolerance) / cell_edge).astype(np.int64)
        for z in range(int(low[2]), int(high[2]) + 1):
            for y in range(int(low[1]), int(high[1]) + 1):
                for x in range(int(low[0]), int(high[0]) + 1):
                    insertions += 1
                    if insertions > int(policy["maximum_spatial_index_insertions"]):
                        return _unproven(base, "SPATIAL_INDEX_INSERTION_CAP_EXCEEDED",
                                         spatial_index_insertions=insertions)
                    index[(x, y, z)].append(triangle_index)
        if time.monotonic() - started > float(policy["maximum_elapsed_seconds"]):
            return _unproven(base, "ORIENTATION_TIME_CAP_EXCEEDED")

    correspondences: list[dict[str, Any]] = []
    tolerance_squared = tolerance * tolerance
    candidate_cap = int(policy["maximum_candidates_per_vertex"])
    for vertex_index, point in enumerate(vertices):
        # A point no triangle uses is not part of the surface being proved.
        # Skipping it here as well as above is the whole fix: left in, it would
        # arrive with a zero normal, the dot product would be 0.0, and the proof
        # would refuse with NONFINITE_OR_ZERO_NORMAL_DOT -- the same rejection
        # wearing a different reason code.
        if not grown_used[vertex_index]:
            continue
        key = tuple(int(value) for value in np.floor(point / cell_edge))
        candidates = sorted(index.get(key, ()), key=lambda i: int(ref_ordinals[i]))
        if len(candidates) > candidate_cap:
            return _unproven(base, "CANDIDATES_PER_VERTEX_CAP_EXCEEDED",
                             grown_vertex_index=vertex_index,
                             candidate_triangle_count=len(candidates))
        best: tuple[float, int, tuple[float, float, float], int, np.ndarray] | None = None
        for triangle_index in candidates:
            triangle = ref_triangles[triangle_index]
            closest, barycentric = _closest_point_triangle(
                point, triangle[0], triangle[1], triangle[2])
            distance_squared = float(np.dot(point - closest, point - closest))
            ordinal = int(ref_ordinals[triangle_index])
            barycentric_quantized = tuple(round(value, 12) for value in barycentric)
            candidate = (distance_squared, ordinal, barycentric_quantized,
                         triangle_index, closest)
            if best is None or distance_squared < best[0] - tie_epsilon:
                best = candidate
            elif abs(distance_squared - best[0]) <= tie_epsilon \
                    and (ordinal, barycentric_quantized) < (best[1], best[2]):
                best = candidate
        if best is None or best[0] > tolerance_squared:
            continue
        distance_squared, ordinal, barycentric, triangle_index, _closest = best
        dot = float(np.dot(grown_normals[vertex_index], ref_normals[triangle_index]))
        if not math.isfinite(dot) or dot == 0.0:
            return _unproven(base, "NONFINITE_OR_ZERO_NORMAL_DOT")
        correspondences.append({
            "reference_triangle_ordinal": ordinal,
            "barycentric_weights": list(barycentric),
            "distance_ct_l0_voxels": math.sqrt(distance_squared),
            "grown_vertex_index": vertex_index,
            "dot": dot,
            "sign": 1 if dot > 0 else -1,
        })
        if time.monotonic() - started > float(policy["maximum_elapsed_seconds"]):
            return _unproven(base, "ORIENTATION_TIME_CAP_EXCEEDED")

    correspondences.sort(key=lambda row: (
        row["reference_triangle_ordinal"], tuple(row["barycentric_weights"]),
        row["grown_vertex_index"]))
    count = len(correspondences)
    minimum = int(policy["minimum_correspondences"])
    if count < minimum:
        return _unproven(base, "INSUFFICIENT_CORRESPONDENCES",
                         retained_correspondence_count=count)
    maximum = int(policy["maximum_sampled_correspondences"])
    if count > maximum:
        sample_indices = [math.floor(i * (count - 1) / (maximum - 1))
                          for i in range(maximum)]
        if len(set(sample_indices)) != len(sample_indices):
            return _unproven(base, "SAMPLING_INDEX_COLLISION")
    else:
        sample_indices = list(range(count))
    samples = [correspondences[index_value] for index_value in sample_indices]
    dots = np.asarray([row["dot"] for row in samples], dtype=np.float64)
    positive, negative = int(np.sum(dots > 0)), int(np.sum(dots < 0))
    consensus = max(positive, negative) / len(samples)
    median_absolute_dot = float(np.median(np.abs(dots)))
    evidence = {
        "reference_triangle_count": int(len(ref_faces)),
        "grown_vertex_count": int(len(vertices)),
        "spatial_index_insertions": insertions,
        "retained_correspondence_count": count,
        "sample_indices": sample_indices,
        "samples": samples,
        "positive_sign_count": positive,
        "negative_sign_count": negative,
        "sign_consensus": consensus,
        "median_absolute_dot": median_absolute_dot,
    }
    if consensus < float(policy["minimum_sign_consensus"]):
        return _unproven(base, "SIGN_CONSENSUS_BELOW_THRESHOLD", **evidence)
    if median_absolute_dot < float(policy["minimum_median_absolute_dot"]):
        return _unproven(base, "MEDIAN_ABSOLUTE_DOT_BELOW_THRESHOLD", **evidence)
    parity_state = ("PROVEN_SAME_WINDING" if positive > negative
                    else "PROVEN_OPPOSITE_WINDING")
    absolute = lineage.get("absolute_orientation") or {}
    absolute_evidence = absolute.get("evidence")
    if (absolute.get("verified") is not True
            or not isinstance(absolute.get("evidence_receipt_sha256"), str)
            or not isinstance(absolute_evidence, dict)
            or absolute_evidence.get("schema") !=
                "campaignx.first_letters_absolute_orientation_evidence.v1"):
        return _unproven(base, "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING",
                         parity_state=parity_state, **evidence)
    evidence_receipt_sha256 = absolute_evidence.get("receipt_sha256")
    evidence_body = {
        key: value for key, value in absolute_evidence.items()
        if key != "receipt_sha256"
    }
    same_flip = (absolute_evidence.get("side_decision") or {}).get(
        "same_winding_flip_normals")
    if (not isinstance(evidence_receipt_sha256, str)
            or evidence_receipt_sha256 != canonical_sha256(evidence_body)
            or same_flip != absolute.get("same_winding_flip_normals")):
        return _unproven(base, "ABSOLUTE_ORIENTATION_EVIDENCE_MISSING",
                         parity_state=parity_state, **evidence)
    if not isinstance(same_flip, bool):
        return _unproven(base, "ABSOLUTE_ORIENTATION_RULE_INVALID",
                         parity_state=parity_state, **evidence)
    selected = same_flip if parity_state == "PROVEN_SAME_WINDING" else not same_flip
    return _finish({
        **base, **evidence, "status": "PROVEN", "reason_code": "ORIENTATION_PROVEN",
        "parity_state": parity_state, "selected_flip_normals": selected,
    })
