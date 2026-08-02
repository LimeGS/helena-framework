"""Find seed candidates in a region of an m7 surface prediction.

This replaces a binary whose source is gone. `vc_mcp_server` on the GPU host
carries `/workspace/campaign-x-vc3d-mcp-linux-build` in its build path -- an
ephemeral machine -- and nothing in any repository has its C++ back. What
survived were three patches against it, which is the fix without the thing
fixed.

Upstream is not the answer either: villa gained a `tools/vc3d-mcp` in July, but
it drives the desktop application -- "preview rays", "toggle annotation mode" --
and this is a headless service that reads a prediction volume.

So it is rebuilt from the contract rather than recovered, and in Python, because
what mattered about the original was never that it was C++. The whole job is:

    given a region of the m7 prediction, return points that look like sheet,
    far enough apart to be different places.

The screens that follow -- CT material support, interior clearance, novelty
against what is already segmented -- belong to the worker and stay there. This
answers only the first question, which is the one that returned nothing 116
times.

The chunk cap the lost patch introduced is kept: a query walks prediction chunks
one at a time, and 27 (3x3x3) is what a radius-128 probe needs. Above that the
service is being asked to scan unbounded regions.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

MAX_CANDIDATE_CHUNKS = 27
CHUNK = 128
MAX_CANDIDATE_VOXELS = MAX_CANDIDATE_CHUNKS * CHUNK ** 3
MAX_CANDIDATE_READ_BYTES = 256 * 1024 * 1024

EVIDENCE_POLICY_ID = "m7-local-structure-v1"
DEFAULT_EVIDENCE_WINDOW_RADIUS_VOXELS = 16
DEFAULT_PEAK_NEIGHBORHOOD_RADIUS_VOXELS = 3
MIN_EVIDENCE_WINDOW_RADIUS_VOXELS = 4
MAX_EVIDENCE_WINDOW_RADIUS_VOXELS = 32
MAX_PEAK_NEIGHBORHOOD_RADIUS_VOXELS = 8

# The m7 predictions are published at threshold 0.2 and stored as uint8, so the
# byte that means "sheet here" is 0.2 * 255. Reading the threshold from the
# array rather than assuming it would be better; the published volumes do not
# carry one.
DEFAULT_THRESHOLD = 0.2


class SeedSearchError(RuntimeError):
    """The region could not be searched. Distinct from finding nothing in it."""


def region_bounds(region: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """The box a region names, however it names it.

    The fleet sends centre and radius:

        {"center": {"x":.., "y":.., "z":..}, "radius": {"x":.., "y":.., "z":..}}

    This module was written expecting x_min/x_max and every task that reached it
    died on `KeyError: 'x_min'` -- reported as BLOCKED_SOURCE_UNAVAILABLE, which
    reads as the bucket being down. It went unnoticed because nothing had ever
    been queued through this path: the encoding was reconstructed from a prose
    contract when the original service was lost, and prose does not say whether
    a box is corners or a centre.

    Both are accepted. Min/max is what the tests and a hand-built call use;
    centre/radius is what the queue actually writes.

    Clamped at zero. A centre closer to the edge than its radius yields a
    negative low bound, and a negative index into a zarr array does not fail --
    it wraps to the far side of the volume and returns geometry from somewhere
    else entirely.
    """
    if "center" in region and "radius" in region:
        centre, radius = region["center"], region["radius"]
        return {axis: (max(0, int(centre[axis]) - int(radius[axis])),
                       int(centre[axis]) + int(radius[axis]))
                for axis in "xyz"}
    try:
        return {axis: (max(0, int(region[f"{axis}_min"])), int(region[f"{axis}_max"]))
                for axis in "xyz"}
    except KeyError as missing:
        raise SeedSearchError(
            f"region names neither center/radius nor {missing} bounds: "
            f"{sorted(region)}") from None


def _bounded_environment_limit(name: str, hard_maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return hard_maximum
    try:
        value = int(raw)
    except ValueError:
        raise SeedSearchError(
            f"{name} must be an integer from 1 through {hard_maximum}") from None
    if not 1 <= value <= hard_maximum:
        raise SeedSearchError(
            f"{name} must be 1 through {hard_maximum}")
    return value


def _volume_chunk_shape_zyx(volume: Any | None) -> tuple[int, int, int]:
    """Return the physical zarr chunk shape without reading the array.

    Numpy fixtures have no chunk metadata; the historical 128-cube remains the
    fallback for those and for legacy slice-like test doubles. Production zarr
    arrays expose ``chunks`` in storage order, z, y, x.
    """
    chunks = getattr(volume, "chunks", None) if volume is not None else None
    if chunks is None:
        return (CHUNK, CHUNK, CHUNK)
    try:
        values = tuple(int(value) for value in chunks)
    except (TypeError, ValueError):
        raise SeedSearchError(
            f"volume.chunks must be three positive integers in z,y,x order; got {chunks!r}"
        ) from None
    if len(values) != 3 or any(value <= 0 for value in values):
        raise SeedSearchError(
            f"volume.chunks must be three positive integers in z,y,x order; got {chunks!r}")
    return values


def chunk_span(region: dict[str, Any], volume: Any | None = None) -> int:
    """How many prediction chunks a region touches.

    Refused above the cap rather than clamped: a caller that asked for more than
    it may have should learn that, not receive a silently smaller answer.
    """
    counts = []
    bounds = region_bounds(region)
    chunk_by_axis = dict(zip("zyx", _volume_chunk_shape_zyx(volume), strict=True))
    for axis in "xyz":
        low, high = bounds[axis]
        if high <= low:
            raise SeedSearchError(f"region has no extent on {axis}: {low}..{high}")
        chunk = chunk_by_axis[axis]
        # The high bound is exclusive. 0..128 touches one 128-wide chunk, not
        # two; using high//chunk over-counted every boundary-aligned request.
        counts.append((high - 1) // chunk - low // chunk + 1)
    total = counts[0] * counts[1] * counts[2]
    cap = _bounded_environment_limit(
        "VC_MCP_MAX_SEED_CANDIDATE_CHUNKS", MAX_CANDIDATE_CHUNKS)
    if total > cap:
        raise SeedSearchError(
            f"region touches {total} chunks, more than the {cap} this service will scan")
    return total


def _read_preflight(
    volume: Any,
    region: dict[str, Any],
) -> tuple[dict[str, tuple[int, int]], Any, dict[str, Any]]:
    """Validate all storage and memory bounds before the first array read."""
    import numpy as np

    bounds = region_bounds(region)
    touched_chunks = chunk_span(region, volume)
    extents = []
    for axis in "xyz":
        low, high = bounds[axis]
        if high <= low:
            raise SeedSearchError(f"region has no extent on {axis}: {low}..{high}")
        extents.append(high - low)
    voxel_count = math.prod(extents)
    voxel_cap = _bounded_environment_limit(
        "VC_MCP_MAX_SEED_CANDIDATE_VOXELS", MAX_CANDIDATE_VOXELS)
    if voxel_count > voxel_cap:
        raise SeedSearchError(
            f"region requests {voxel_count} voxels, more than the {voxel_cap} "
            "this service will read")

    raw_dtype = getattr(volume, "dtype", None)
    if raw_dtype is None:
        raise SeedSearchError(
            "prediction volume must expose a concrete dtype before it can be read")
    try:
        dtype = np.dtype(raw_dtype)
    except TypeError:
        raise SeedSearchError(
            "prediction volume must expose a concrete dtype before it can be read") from None
    if dtype != np.dtype(np.uint8) and dtype.kind != "f":
        raise SeedSearchError(
            f"prediction dtype {dtype} is unsupported; expected uint8 or a "
            "floating dtype containing normalized [0,1] values")
    if dtype.hasobject:
        raise SeedSearchError("object prediction arrays are not safe to read")

    byte_count = voxel_count * dtype.itemsize
    byte_cap = _bounded_environment_limit(
        "VC_MCP_MAX_SEED_CANDIDATE_READ_BYTES", MAX_CANDIDATE_READ_BYTES)
    if byte_count > byte_cap:
        raise SeedSearchError(
            f"region requests {byte_count} bytes, more than the {byte_cap} "
            "this service will read")

    return bounds, dtype, {
        "chunk_count": touched_chunks,
        "chunk_shape_zyx": list(_volume_chunk_shape_zyx(volume)),
        "requested_voxel_count": voxel_count,
        "requested_byte_count": byte_count,
    }


def candidate_id(prediction_uri: str, x: int, y: int, z: int) -> str:
    """Stable across calls, so the same point in the same volume is the same id.

    The planner records which candidate it chose and the validator checks the
    choice copies one that was offered; both break if an id is a fresh uuid
    every probe.
    """
    seed = f"{prediction_uri}\0{x}\0{y}\0{z}".encode("utf-8")
    return "cand-" + hashlib.sha256(seed).hexdigest()[:16]


def far_enough(chosen: list[tuple[int, int, int]], point: tuple[int, int, int],
               separation: int) -> bool:
    """Euclidean, not per-axis.

    Per-axis would accept two points a step apart diagonally, which are the same
    place as far as growing a surface is concerned.
    """
    for other in chosen:
        distance = math.dist(point, other)
        if distance < separation:
            return False
    return True


def _rounded_measurement(value: float) -> float:
    rounded = round(float(value), 8)
    return 0.0 if rounded == 0 else rounded


def _validate_evidence_policy(window_radius: int, peak_radius: int) -> None:
    if not MIN_EVIDENCE_WINDOW_RADIUS_VOXELS <= window_radius <= MAX_EVIDENCE_WINDOW_RADIUS_VOXELS:
        raise SeedSearchError(
            "evidence_window_radius_voxels must be "
            f"{MIN_EVIDENCE_WINDOW_RADIUS_VOXELS} through "
            f"{MAX_EVIDENCE_WINDOW_RADIUS_VOXELS}")
    maximum_peak_radius = min(MAX_PEAK_NEIGHBORHOOD_RADIUS_VOXELS, window_radius)
    if not 1 <= peak_radius <= maximum_peak_radius:
        raise SeedSearchError(
            "peak_neighborhood_radius_voxels must be 1 through "
            f"{maximum_peak_radius} for this evidence window")


def _pca_descriptor(component_zyx: Any) -> dict[str, Any]:
    """Population-covariance PCA over every observed seed-component voxel."""
    import numpy as np

    points_zyx = np.argwhere(component_zyx)
    # The evidence contract is XYZ. Reversing the columns before covariance
    # avoids silently publishing ZYX descriptors under an XYZ label.
    points_xyz = points_zyx[:, ::-1].astype(np.float64, copy=False)
    point_count = int(points_xyz.shape[0])
    if point_count == 0:  # Defensive: the selected seed must be in its mask.
        return {
            "method": "population_covariance_pca_xyz",
            "point_count": 0,
            "eigenvalues_descending_voxels_squared": [0.0, 0.0, 0.0],
            "linearity": None,
            "planarity": None,
            "scattering": None,
            "state": "DEGENERATE",
        }

    centred = points_xyz - points_xyz.mean(axis=0)
    covariance = centred.T @ centred / point_count
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    # Numerical eigensolvers can return a tiny negative value for a
    # positive-semidefinite covariance matrix.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    l1, l2, l3 = (float(value) for value in eigenvalues)
    if l1 <= np.finfo(np.float64).eps:
        linearity = planarity = scattering = None
        state = "DEGENERATE"
    else:
        linearity = _rounded_measurement((l1 - l2) / l1)
        planarity = _rounded_measurement((l2 - l3) / l1)
        scattering = _rounded_measurement(l3 / l1)
        state = "MEASURED"

    return {
        "method": "population_covariance_pca_xyz",
        "point_count": point_count,
        "eigenvalues_descending_voxels_squared": [
            _rounded_measurement(value) for value in eigenvalues
        ],
        "linearity": linearity,
        "planarity": planarity,
        "scattering": scattering,
        "state": state,
    }


def _canonical_peak_representatives(
    peak_labels: Any,
    peak_count: int,
    *,
    window_origin_zyx: tuple[int, int, int],
    query_origin_zyx: tuple[int, int, int],
) -> tuple[Any, Any]:
    """One deterministic XYZ representative per connected peak plateau."""
    import numpy as np

    if peak_count == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0, 3), dtype=np.int64)

    local_zyx = np.argwhere(peak_labels > 0)
    labels = peak_labels[local_zyx[:, 0], local_zyx[:, 1], local_zyx[:, 2]]
    global_zyx = (
        local_zyx
        + np.asarray(window_origin_zyx, dtype=np.int64)
        + np.asarray(query_origin_zyx, dtype=np.int64)
    )
    global_xyz = global_zyx[:, ::-1]
    # Label is primary, then canonical global x, y, z. ``lexsort`` consumes
    # keys from least to most significant.
    order = np.lexsort((
        global_xyz[:, 2],
        global_xyz[:, 1],
        global_xyz[:, 0],
        labels,
    ))
    sorted_labels = labels[order]
    first = np.empty(sorted_labels.shape, dtype=bool)
    first[0] = True
    first[1:] = sorted_labels[1:] != sorted_labels[:-1]
    selected = order[first]
    return labels[selected], global_xyz[selected]


def _candidate_evidence(
    block: Any,
    *,
    candidate_local_zyx: tuple[int, int, int],
    query_origin_zyx: tuple[int, int, int],
    threshold: float,
    dtype: Any,
    window_radius: int,
    peak_radius: int,
) -> dict[str, Any]:
    """Measure local prediction structure from ``block`` without another read."""
    import numpy as np
    from scipy import ndimage

    shape = tuple(int(value) for value in block.shape)
    desired_low = tuple(value - window_radius for value in candidate_local_zyx)
    desired_high = tuple(value + window_radius + 1 for value in candidate_local_zyx)
    low = tuple(max(0, value) for value in desired_low)
    high = tuple(min(shape[index], desired_high[index]) for index in range(3))
    window = block[tuple(slice(low[index], high[index]) for index in range(3))]
    seed_in_window = tuple(
        candidate_local_zyx[index] - low[index] for index in range(3))

    if dtype == np.dtype(np.uint8):
        values = window.astype(np.float64) / 255.0
        stored_value: int | float = int(
            block[candidate_local_zyx[0], candidate_local_zyx[1], candidate_local_zyx[2]])
    else:
        values = window.astype(np.float64, copy=False)
        stored_value = float(
            block[candidate_local_zyx[0], candidate_local_zyx[1], candidate_local_zyx[2]])
    mask = values > threshold

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, component_count = ndimage.label(mask, structure=structure)
    seed_label = int(labels[seed_in_window])
    if seed_label == 0:
        raise SeedSearchError(
            "selected candidate is not above the evidence threshold; "
            "candidate extraction and evidence measurement disagree")
    component = labels == seed_label
    component_points = np.argwhere(component)
    positive_voxel_count = int(np.count_nonzero(mask))
    component_voxel_count = int(component_points.shape[0])

    extent_zyx = component_points.max(axis=0) - component_points.min(axis=0) + 1
    deltas = component_points.astype(np.float64) - np.asarray(seed_in_window)
    support_radius = float(np.sqrt(np.max(np.sum(deltas * deltas, axis=1))))

    face_checks = (
        ("x_min", component[:, :, 0]),
        ("x_max", component[:, :, -1]),
        ("y_min", component[:, 0, :]),
        ("y_max", component[:, -1, :]),
        ("z_min", component[0, :, :]),
        ("z_max", component[-1, :, :]),
    )
    touched_faces = [name for name, face in face_checks if bool(np.any(face))]

    disconnected = mask & ~component
    nearest_disconnected = None
    if bool(np.any(disconnected)):
        distance_to_component = ndimage.distance_transform_edt(~component)
        nearest_disconnected = _rounded_measurement(
            float(np.min(distance_to_component[disconnected])))

    maximum = ndimage.maximum_filter(
        values,
        size=2 * peak_radius + 1,
        mode="constant",
        cval=-np.inf,
    )
    peak_mask = mask & (values == maximum)
    peak_labels, peak_count = ndimage.label(peak_mask, structure=structure)
    own_peak_label = int(peak_labels[seed_in_window])
    representative_labels, representatives_xyz = _canonical_peak_representatives(
        peak_labels,
        int(peak_count),
        window_origin_zyx=low,
        query_origin_zyx=query_origin_zyx,
    )

    query_origin_xyz = tuple(reversed(query_origin_zyx))
    candidate_xyz = np.asarray(tuple(reversed(candidate_local_zyx)), dtype=np.int64)
    candidate_xyz += np.asarray(query_origin_xyz, dtype=np.int64)
    own_representative = None
    if own_peak_label:
        own_rows = np.flatnonzero(representative_labels == own_peak_label)
        if own_rows.size:
            own_representative = representatives_xyz[int(own_rows[0])]
    canonical_point = bool(
        own_representative is not None and np.array_equal(own_representative, candidate_xyz))

    competitor_rows = (
        representative_labels != own_peak_label
        if own_peak_label
        else np.ones(representative_labels.shape, dtype=bool)
    )
    competitor_xyz = representatives_xyz[competitor_rows]
    nearest_competitor = strongest_competitor = margin = None
    if competitor_xyz.shape[0]:
        competitor_distances = np.sqrt(np.sum(
            (competitor_xyz.astype(np.float64) - candidate_xyz) ** 2, axis=1))
        nearest_competitor = _rounded_measurement(float(np.min(competitor_distances)))
        competitor_zyx = competitor_xyz[:, ::-1] - np.asarray(query_origin_zyx)
        competitor_values = block[
            competitor_zyx[:, 0],
            competitor_zyx[:, 1],
            competitor_zyx[:, 2],
        ].astype(np.float64)
        if dtype == np.dtype(np.uint8):
            competitor_values /= 255.0
        strongest_competitor = _rounded_measurement(float(np.max(competitor_values)))
        point_value = (
            stored_value / 255.0
            if dtype == np.dtype(np.uint8)
            else float(stored_value)
        )
        margin = _rounded_measurement(point_value - strongest_competitor)

    truncated_faces: list[str] = []
    for index, axis in enumerate("zyx"):
        if desired_low[index] < 0:
            truncated_faces.append(f"{axis}_min")
        if desired_high[index] > shape[index]:
            truncated_faces.append(f"{axis}_max")
    # Contracts use XYZ face order even though slices are held as ZYX.
    face_order = {f"{axis}_{side}": order
                  for order, (axis, side) in enumerate(
                      (("x", "min"), ("x", "max"), ("y", "min"),
                       ("y", "max"), ("z", "min"), ("z", "max")))}
    truncated_faces.sort(key=face_order.__getitem__)

    window_low_global_zyx = tuple(
        query_origin_zyx[index] + low[index] for index in range(3))
    window_high_global_zyx = tuple(
        query_origin_zyx[index] + high[index] for index in range(3))
    window_bounds = {
        axis: {
            "min": window_low_global_zyx["zyx".index(axis)],
            "max_exclusive": window_high_global_zyx["zyx".index(axis)],
        }
        for axis in "xyz"
    }

    return {
        "schema": "campaignx.m7_seed_candidate_evidence.v1",
        "policy_id": EVIDENCE_POLICY_ID,
        "ink_used": False,
        "score_semantics": "normalized_m7_intensity_not_probability",
        "non_claim": (
            "local_m7_structure_only; does_not_establish_ct_material_support,"
            "lamina_identity,or_growth_success"
        ),
        "window": {
            "radius_voxels": window_radius,
            "bounds_ct_l0_xyz": window_bounds,
            "shape_zyx": [int(value) for value in window.shape],
            "coverage": (
                "QUERY_BOUNDARY_TRUNCATED" if truncated_faces else "COMPLETE"),
            "query_faces_truncated": truncated_faces,
        },
        "point": {
            "ct_l0_coordinate": {
                "x": int(candidate_xyz[0]),
                "y": int(candidate_xyz[1]),
                "z": int(candidate_xyz[2]),
            },
            "stored_value": stored_value,
            "normalized_m7_intensity": _rounded_measurement(
                stored_value / 255.0
                if dtype == np.dtype(np.uint8)
                else float(stored_value)),
            "threshold_normalized": _rounded_measurement(threshold),
            "threshold_comparison": "strictly_greater",
        },
        "threshold_component": {
            "connectivity": 26,
            "positive_component_count": int(component_count),
            "positive_voxel_count": positive_voxel_count,
            "seed_component_voxel_count": component_voxel_count,
            "seed_component_share_of_positive_voxels": _rounded_measurement(
                component_voxel_count / positive_voxel_count),
            "bounding_box_extent_xyz_voxels": {
                "x": int(extent_zyx[2]),
                "y": int(extent_zyx[1]),
                "z": int(extent_zyx[0]),
            },
            "support_radius_voxels": _rounded_measurement(support_radius),
            "touched_window_faces": touched_faces,
            "observation_state": (
                "WINDOW_CENSORED" if touched_faces else "COMPLETE_WITHIN_WINDOW"),
            "nearest_disconnected_positive_distance_voxels": nearest_disconnected,
        },
        "shape_descriptor": _pca_descriptor(component),
        "spatial_peak_competition": {
            "neighborhood_radius_voxels": peak_radius,
            "plateau_connectivity": 26,
            "point_is_local_maximum": bool(own_peak_label),
            "point_is_canonical_plateau_representative": canonical_point,
            "distinct_peak_count": int(peak_count),
            "competing_peak_count": int(np.count_nonzero(competitor_rows)),
            "nearest_competitor_distance_voxels": nearest_competitor,
            "strongest_competitor_normalized_m7_intensity": strongest_competitor,
            "point_minus_strongest_competitor_normalized_m7_intensity": margin,
        },
    }


def _strongest_eligible_point(
    block: Any,
    eligible: Any,
) -> tuple[tuple[int, int, int], Any] | None:
    """Highest value, then lexicographically smallest global x/y/z.

    The caller adds the global origin, which is a constant and therefore does
    not alter the local lexicographic ordering. This avoids constructing and
    sorting an unbounded ``argwhere`` result for a dense prediction block.
    """
    import numpy as np

    if not bool(np.any(eligible)):
        return None
    strength = np.max(block, where=eligible, initial=0)
    # X is the first tie-break, then Y, then Z. Numpy storage order is ZYX, so
    # scan x planes explicitly instead of inheriting incidental memory order.
    for x in range(block.shape[2]):
        plane_eligible = eligible[:, :, x]
        if not bool(np.any(plane_eligible)):
            continue
        matches = plane_eligible & (block[:, :, x] == strength)
        if not bool(np.any(matches)):
            continue
        ys = np.flatnonzero(np.any(matches, axis=0))
        y = int(ys[0])
        zs = np.flatnonzero(matches[:, y])
        return (int(zs[0]), y, x), strength
    raise SeedSearchError("eligible prediction voxels exist but no maximum could be located")


def _exclude_nearby(
    eligible: Any,
    point_zyx: tuple[int, int, int],
    separation: int,
) -> None:
    """Apply the same Euclidean ``distance < separation`` rule in-place."""
    import numpy as np

    z, y, x = point_zyx
    radius = max(1, separation)
    z0, z1 = max(0, z - radius + 1), min(eligible.shape[0], z + radius)
    y0, y1 = max(0, y - radius + 1), min(eligible.shape[1], y + radius)
    x0, x1 = max(0, x - radius + 1), min(eligible.shape[2], x + radius)
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    too_close = (
        (zz - z) ** 2 + (yy - y) ** 2 + (xx - x) ** 2
        < separation ** 2
    )
    eligible[z0:z1, y0:y1, x0:x1][too_close] = False
    # A separation of zero used to be accepted by ``far_enough``. Refusing it
    # at the public boundary is clearer, but keep this defensive assignment so
    # the selector can never return the same point twice.
    eligible[z, y, x] = False


def find_candidates(
    volume: Any,
    region: dict[str, Any],
    *,
    prediction_uri: str,
    max_candidates: int = 8,
    minimum_separation_voxels: int = 16,
    threshold: float = DEFAULT_THRESHOLD,
    evidence_window_radius_voxels: int = DEFAULT_EVIDENCE_WINDOW_RADIUS_VOXELS,
    peak_neighborhood_radius_voxels: int = DEFAULT_PEAK_NEIGHBORHOOD_RADIUS_VOXELS,
) -> list[dict[str, Any]]:
    """Points above threshold in the region, spread out, strongest first.

    `volume` is anything that slices like a zarr array in z, y, x order. Passed
    in rather than opened here so this is testable without a bucket.
    """
    import numpy as np

    if max_candidates < 1:
        raise SeedSearchError("max_candidates must be at least 1")
    if minimum_separation_voxels < 1:
        raise SeedSearchError("minimum_separation_voxels must be at least 1")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SeedSearchError("threshold must be a finite normalized value from 0 through 1")
    _validate_evidence_policy(
        evidence_window_radius_voxels, peak_neighborhood_radius_voxels)

    bounds, dtype, read_accounting = _read_preflight(volume, region)
    box = tuple(slice(*bounds[axis]) for axis in "zyx")
    block = np.asarray(volume[box])
    if block.size == 0:
        return []
    if block.ndim != 3:
        raise SeedSearchError(
            f"prediction slice must be three-dimensional in z,y,x order; got {block.shape}")
    if block.dtype != dtype:
        raise SeedSearchError(
            f"prediction slice dtype changed from {dtype} to {block.dtype} while reading")
    if block.size > read_accounting["requested_voxel_count"]:
        raise SeedSearchError(
            "prediction slice returned more voxels than its preflight bounds allowed")
    if dtype.kind == "f":
        if not bool(np.all(np.isfinite(block))):
            raise SeedSearchError(
                "floating prediction values must all be finite normalized [0,1] values")
        if bool(np.any(block < 0.0)) or bool(np.any(block > 1.0)):
            raise SeedSearchError(
                "floating prediction values must all be normalized to [0,1]")

    scale = 255 if dtype == np.dtype(np.uint8) else 1.0
    cutoff = threshold * scale
    eligible = block > cutoff
    if not bool(np.any(eligible)):
        return []

    origin_zyx = tuple(bounds[axis][0] for axis in "zyx")
    out: list[dict[str, Any]] = []
    while len(out) < max_candidates:
        selected = _strongest_eligible_point(block, eligible)
        if selected is None:
            break
        local_zyx, strength = selected
        z = local_zyx[0] + origin_zyx[0]
        y = local_zyx[1] + origin_zyx[1]
        x = local_zyx[2] + origin_zyx[2]
        score = float(strength) / scale
        evidence = _candidate_evidence(
            block,
            candidate_local_zyx=local_zyx,
            query_origin_zyx=origin_zyx,
            threshold=threshold,
            dtype=dtype,
            window_radius=evidence_window_radius_voxels,
            peak_radius=peak_neighborhood_radius_voxels,
        )
        out.append({
            "candidate_id": candidate_id(prediction_uri, x, y, z),
            "ct_l0_coordinate": {"x": x, "y": y, "z": z},
            "x": x, "y": y, "z": z,
            "surface_score": score,
            "score": score,
            # Kept for backwards compatibility with consumers written before
            # the score semantics were made explicit. It is not a calibrated
            # confidence or probability.
            "confidence": score,
            "score_semantics": "normalized_m7_intensity_not_probability",
            "prediction_uri": prediction_uri,
            "candidate_evidence": evidence,
            # Clearance is the worker's to measure -- it knows what is already
            # segmented and this does not. Reported as null rather than zero so
            # nothing reads an unmeasured distance as a measured one.
            "clearance_voxels": None,
            "cell_interior_clearance_voxels": None,
            "volume_interior_clearance_voxels": None,
        })
        _exclude_nearby(eligible, local_zyx, minimum_separation_voxels)
    return out
