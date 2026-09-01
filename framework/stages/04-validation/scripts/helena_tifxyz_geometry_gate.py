#!/usr/bin/env python3
"""Port the frozen ScrollFiesta seam / self-intersection gate to TIFXYZ.

`helena_audit_mesh_integrity.py` already encodes the gate Helena Framework needs
(near-coincident overlap, interpenetration, fold-back self-intersection, plus
an exact self-intersection verdict, ``PASS`` only when ``hard_defects == 0``).
It consumes ``.obj`` meshes produced by the ScrollFiesta route.  Production
VC3D segmentation emits ``x/y/z.tif + meta.json`` instead, so no production
surface has ever been measured by it.

This module derives the mesh directly from the TIFXYZ grid -- every grid quad
becomes two triangles, using exactly the triangulation and the
finite-and-non-negative validity policy already frozen in
``fleet/finalizer.py:inspect_tifxyz`` -- computes the same metric names, and
reuses ``helena_audit_mesh_integrity.hard_defect_count`` so a single
implementation decides what "hard defect" means on both routes.

The verdict is an axis orthogonal to ``physical_qc_state``: a surface may be
``CT_SUPPORTED`` and ``GEOMETRY_REJECTED_BRIDGE`` at the same time.

Non-claims
----------
* A ``GEOMETRY_CERTIFIED`` verdict is not physical-sheet acceptance, not ink,
  not text, and not a statement that the segmentation followed the correct
  lamina.  It states only that the frozen hard-defect detectors found nothing
  at the sampling density of the artifact that was measured.
* The detectors cannot resolve a defect finer than the TIFXYZ grid step.  Every
  receipt therefore publishes ``median_edge_voxels`` and
  ``resolution_limited``; a grid whose step is at or above the inter-lamina
  spacing cannot certify against a single-lamina switch.
* ``resolution_limited`` is measured against this scroll's own inter-lamina
  spacing only when a caller supplies one.  Without it the frozen 150 um floor
  stands in, which is the bottom of a published range rather than a fact about
  the scroll, and ``resolution_limit_measured_here`` says which of the two the
  verdict was reached against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence


SCHEMA = "campaignx.tifxyz_geometry_certification.v1"
BATCH_SCHEMA = "campaignx.tifxyz_geometry_certification_batch.v1"

GEOMETRY_CERTIFIED = "GEOMETRY_CERTIFIED"
GEOMETRY_REJECTED_BRIDGE = "GEOMETRY_REJECTED_BRIDGE"
GEOMETRY_REJECTED_LAMINA_SWITCH = "GEOMETRY_REJECTED_LAMINA_SWITCH"
GEOMETRY_REJECTED_DISTORTION = "GEOMETRY_REJECTED_DISTORTION"
GEOMETRY_REJECTED_COVERAGE = "GEOMETRY_REJECTED_COVERAGE"
GEOMETRY_UNMEASURED = "GEOMETRY_UNMEASURED"

REQUIRED_FILES = ("x.tif", "y.tif", "z.tif")
# One byte per grid cell, 1 where the sheet folds. Named beside the receipt so a
# renderer can exclude those cells rather than re-deriving them.
FOLD_BACK_MASK = "FOLD_BACK_MASK.tif"

# Frozen defaults.  ``band_cells`` and ``parallel_angle_deg`` mirror the
# ScrollFiesta invocation in helena_audit_mesh_integrity.py (--band 4,
# --angle 20).
#
# The distance thresholds used to be expressed relative to the mesh's global
# median step, on the grounds that TIFXYZ grids are arc-length uniform by
# construction.  That is true of a seeded grow -- whose longest edge is 7.1x its
# median, against a factor of 8.0 -- and false of a global fit, whose longest is
# 13x to 26x and whose local step ranges over threefold across one strip.  The
# step rule is measured locally now, and the counts below are read as the
# lengths they stand for when a caller supplies the scroll's scale.
DEFAULT_POLICY: dict[str, Any] = {
    "band_cells": 4,
    # A scroll sheet legitimately passes within one inter-lamina spacing of
    # itself every turn, so the doubled-surface test must be far tighter than
    # that spacing or every multi-winding surface is a false positive.  Three
    # voxels is about 28 um at these scan scales -- below the thickness of a
    # single papyrus lamina, so only a sheet doubled onto itself can trip it.
    "gap_voxels": 3.0,
    "parallel_angle_deg": 20.0,
    "fold_angle_deg": 150.0,
    "step_discontinuity_factor": 8.0,
    "minimum_valid_fraction": 0.25,
    "minimum_largest_component_fraction": 0.90,
    "minimum_valid_triangles": 64,
    "maximum_candidate_pairs": 20_000_000,
    # Inter-lamina spacing in a Herculaneum scroll is of order 150-250 um; at
    # the ~9.4 um/voxel scale of these volumes that is roughly 16-27 voxels.  A
    # TIFXYZ grid whose step reaches that range cannot resolve a switch of a
    # single lamina, so its certification is explicitly resolution limited.
    "resolution_limit_voxels": 16.0,
    # The same two numbers as lengths, which is what they were always standing
    # for. `band_cells: 4` on the corpus this gate was calibrated against is
    # 4 x 12.94 voxels x 9.362 um; `resolution_limit_voxels: 16` is the 150 um
    # floor of the inter-lamina range the comment above cites. When a caller
    # supplies the scroll's scale these are what the gate derives from, and the
    # counts above are the fallback for when it does not.
    # Fold-back stopped being a hard defect and became an area measure, and
    # this is the ceiling on it. See `_fold_back_pairs` for why.
    #
    # Not calibrated, and it must not be read as if it were. It is set to catch
    # a mesh that is folded *everywhere* -- a fit that failed globally -- and it
    # cannot discriminate a wrinkle from an error, because nothing here can:
    # measured on the only corpus where this has been looked at, three fitted
    # windings of PHerc0826 carry 2.34%, 2.72% and 2.52% of their cells in a
    # fold, and the three grown VC3D segments carry none.
    "maximum_fold_back_fraction": 0.05,
    "band_um": 480.0,
    "resolution_limit_um": 150.0,
    # 0 means "measure the step against the mesh's own median", which is what
    # the gate did before it could derive a window from the scale.
    "step_window_cells": 0,
}


class GeometryGateError(RuntimeError):
    """Raised when the gate cannot produce a measurement it can stand behind."""


def _mesh_integrity_module() -> ModuleType:
    """Load the frozen ScrollFiesta gate so its arithmetic is single-sourced."""

    path = Path(__file__).resolve().with_name("helena_audit_mesh_integrity.py")
    spec = spec_from_file_location("helena_audit_mesh_integrity", path)
    if spec is None or spec.loader is None:
        raise GeometryGateError(f"cannot load the frozen mesh gate: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tifxyz(surface_dir: Path):
    """Return ``(points, valid)`` for one TIFXYZ surface directory."""

    import numpy as np
    import tifffile

    surface_dir = Path(surface_dir)
    missing = [name for name in REQUIRED_FILES if not (surface_dir / name).is_file()]
    if missing:
        raise GeometryGateError(f"TIFXYZ is missing {missing} in {surface_dir}")
    arrays = [
        np.asarray(tifffile.imread(surface_dir / name), dtype=np.float64)
        for name in REQUIRED_FILES
    ]
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise GeometryGateError(f"TIFXYZ shapes differ: {[a.shape for a in arrays]}")
    if arrays[0].ndim != 2 or min(arrays[0].shape) < 2:
        raise GeometryGateError(f"TIFXYZ must be a 2-D grid, got {arrays[0].shape}")
    # Identical policy to fleet/finalizer.py:inspect_tifxyz -- VC3D writes -1 as
    # the invalid-coordinate sentinel and CT-L0 coordinates cannot be negative.
    valid = np.logical_and.reduce([np.isfinite(a) & (a >= 0.0) for a in arrays])
    points = np.stack(arrays, axis=-1)
    return points, valid


def build_mesh(points, valid) -> dict[str, Any]:
    """Derive the triangle mesh from the grid; no .obj is required.

    Each quad becomes two triangles with the same split *and the same winding*
    as `fleet/finalizer.py:triangulate_tifxyz_grid`, so mesh area and gate
    geometry cannot disagree.

    The winding is not a detail. The second triangle used to be
    `(i11, i10, i01)`, which walks the shared edge i10->i01 in the same
    direction the first one does -- so the two triangles of every quad came out
    with opposite normals. On a flat unit quad their dot product is exactly
    -1.0; on a fitted spiral winding 97% of quads had their two normals more
    than 90 degrees apart, which reads as a mesh that is folded everywhere and
    is nothing of the kind.

    No detector here consumed that sign -- fold-back reads only the first
    triangle of each quad, and the parallel test takes the absolute value -- so
    no verdict this gate has issued was affected, and the regression test beside
    this file pins that. It was still wrong to publish: `mesh["normal"]` is half
    a normal field, this module's own docstring claimed the finalizer's
    triangulation, and the orientation path elsewhere in the platform is exactly
    the kind of consumer that would have read it and been silently wrong.
    """

    import numpy as np

    height, width = valid.shape
    index = np.arange(height * width).reshape(height, width)
    rows, cols = np.divmod(np.arange(height * width), width)
    i00, i10, i01, i11 = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    m00, m10, m01, m11 = valid[:-1, :-1], valid[1:, :-1], valid[:-1, 1:], valid[1:, 1:]
    # (v00,v10,v01) and (v10,v11,v01): the finalizer's own pair, vertex for
    # vertex. The masks are unchanged -- each triangle is still retained
    # independently so a mask boundary cannot discard its valid half.
    first = np.stack([i00, i10, i01], axis=-1)[m00 & m10 & m01]
    second = np.stack([i10, i11, i01], axis=-1)[m10 & m11 & m01]
    triangles = np.concatenate([first, second], axis=0).astype(np.int64)
    vertices = points.reshape(-1, 3)
    if len(triangles) == 0:
        raise GeometryGateError("TIFXYZ grid yields no valid triangle")
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    raw_normal = np.cross(b - a, c - a)
    length = np.linalg.norm(raw_normal, axis=1)
    unit = np.zeros_like(raw_normal)
    finite = length > 1e-12
    unit[finite] = raw_normal[finite] / length[finite, None]
    centroid = (a + b + c) / 3.0
    circumradius = np.maximum.reduce([
        np.linalg.norm(a - centroid, axis=1),
        np.linalg.norm(b - centroid, axis=1),
        np.linalg.norm(c - centroid, axis=1),
    ])
    return {
        "vertices": vertices,
        "triangles": triangles,
        "vertex_row": rows,
        "vertex_col": cols,
        "corner_a": a,
        "corner_b": b,
        "corner_c": c,
        "normal": unit,
        "degenerate": ~finite,
        "centroid": centroid,
        "circumradius": circumradius,
        "triangle_row": rows[triangles].mean(axis=1),
        "triangle_col": cols[triangles].mean(axis=1),
    }


def _grid_step(points, valid) -> dict[str, float]:
    import numpy as np

    row_mask = valid[1:, :] & valid[:-1, :]
    col_mask = valid[:, 1:] & valid[:, :-1]
    lengths = np.concatenate([
        np.linalg.norm(points[1:, :] - points[:-1, :], axis=-1)[row_mask],
        np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)[col_mask],
    ])
    if lengths.size == 0:
        raise GeometryGateError("TIFXYZ grid has no connected valid edge")
    return {
        "adjacent_edge_count": int(lengths.size),
        "median_edge_voxels": float(np.median(lengths)),
        "maximum_edge_voxels": float(lengths.max()),
    }


def _coverage(valid) -> dict[str, Any]:
    from scipy import ndimage

    labelled, count = ndimage.label(valid)
    if count == 0:
        return {
            "valid_fraction": 0.0,
            "connected_component_count": 0,
            "largest_component_fraction": 0.0,
        }
    sizes = ndimage.sum(valid, labelled, range(1, count + 1))
    return {
        "valid_fraction": float(valid.mean()),
        "connected_component_count": int(count),
        "largest_component_fraction": float(sizes.max() / float(valid.sum())),
    }


def _fold_back_pairs(mesh, valid, fold_angle_deg: float):
    """Edge-adjacent triangles whose normals have flipped: the sheet folds back.

    Returns the count and the cells it touches, because on this corpus the
    count alone was the wrong shape of answer.

    It was a hard defect, and any of them rejected the surface outright. What
    that rejected was wrinkles. Four independent lines say so, measured on
    PHerc0826: the folds land in the same physical places in both seeds
    (17x enrichment over the base rate, p < 0.0005), the cells carrying them
    are not slivers (aspect 2.2 against 1.9 on clean ones, so the normals are
    well conditioned), they are no likelier than clean cells to sit where
    another lamina passes within 375 um (41.4% against 39.7%), and 36 of 43
    connected components larger than five cells are elongated along the
    scroll's axis. A flattened scroll wrinkles along its axis, and the fit was
    following the wrinkles.

    The distinguishing evidence is spatial. Cropping to 4 cm2 drops the
    lamina-switch detectors by 26x, 111x and 34x, and at 1 cm2 all four reach
    zero somewhere on the surface -- they are hunting a genuinely non-local
    signature and they find clean subregions. Fold-back does not move: about
    120 per 4 cm2 everywhere, touching 322 of 636 rows. A defect distributed
    evenly across a whole surface is either everywhere-real or everywhere-wrong,
    and it is not localised.

    No angle separates the two. Raising the threshold from 150 to 170 degrees
    still leaves 83 flips on w010, because the angles are genuinely sharp.

    So the count and the mask are reported, the fraction of affected area is
    what the verdict reads, and the mask is what P3 and P4 have to exclude. The
    reason that matters is specific: a fold-back is an inverted normal in that
    cell, and this platform has already paid for the global version of that bug
    -- a renderer whose default came out inverted against the published stack,
    correlating 0.09 with it. The fix was to force one sign for the whole mesh
    against the umbilicus axis, and a per-mesh sign does nothing about a cell
    that disagrees with its neighbours.
    """

    import numpy as np

    height, width = valid.shape
    normal = np.zeros((height - 1, width - 1, 3))
    live = np.zeros((height - 1, width - 1), dtype=bool)
    first_count = int(
        (valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]).sum()
    )
    cell_mask = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]
    normal[cell_mask] = mesh["normal"][:first_count]
    live[cell_mask] = ~mesh["degenerate"][:first_count]
    limit = float(np.cos(np.radians(fold_angle_deg)))
    total = 0
    touched = np.zeros((height - 1, width - 1), dtype=bool)
    for axis, (shifted, base, mask) in enumerate((
        (normal[1:, :], normal[:-1, :], live[1:, :] & live[:-1, :]),
        (normal[:, 1:], normal[:, :-1], live[:, 1:] & live[:, :-1]),
    )):
        if not mask.any():
            continue
        cosine = np.einsum("ijk,ijk->ij", shifted, base)
        hit = (cosine < limit) & mask
        total += int(np.count_nonzero(hit))
        # Both cells of a flipped pair are marked: which of the two is the
        # wrong one is not knowable from the pair, and a consumer excluding
        # only one of them keeps an inverted normal.
        if axis == 0:
            touched[1:, :] |= hit
            touched[:-1, :] |= hit
        else:
            touched[:, 1:] |= hit
            touched[:, :-1] |= hit
    return total, touched


def _write_mask(folded, path: Path) -> dict[str, Any]:
    """Publish the folded cells as a grid, and name it by its digest."""
    import numpy as np
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(folded, dtype=np.uint8))
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "shape": [int(value) for value in np.asarray(folded).shape],
        "note": ("one byte per grid cell, 1 where the sheet folds back. Both "
                 "cells of a flipped pair are marked: which of the two carries "
                 "the wrong normal is not knowable from the pair, and excluding "
                 "one of them keeps an inverted normal."),
    }


def _fold_back_extent(folded, valid) -> dict[str, Any]:
    """How much of the surface a fold touches, and in what shape.

    The shape is reported because it is what told this corpus apart: folds that
    are elongated along the scroll's axis are what a flattened scroll does, and
    folds scattered as isolated cells are what a noisy normal field does.
    """
    import numpy as np
    from scipy import ndimage

    cells = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:]
    live = int(cells.sum())
    touched = int(np.count_nonzero(folded))
    labelled, count = ndimage.label(folded)
    sizes = (ndimage.sum(folded, labelled, range(1, count + 1))
             if count else np.zeros(0))
    return {
        "cells": touched,
        "live_cells": live,
        "fraction": (touched / live) if live else 0.0,
        "components": int(count),
        "components_over_five_cells": int((sizes > 5).sum()) if count else 0,
        "rows_touched": int(np.unique(np.argwhere(folded)[:, 0]).size)
                        if touched else 0,
        "rows": int(folded.shape[0]),
    }


def _step_discontinuities(points, valid, median_edge: float, factor: float,
                         window_cells: int = 0) -> int:
    """Edges far longer than the sampling around them.

    Measured against a *local* median, not the mesh's. The global rule rested
    on a premise this module used to state outright -- that TIFXYZ grids are
    arc-length uniform by construction -- which is true of a seeded grow and
    false of a global fit: a grown surface's longest edge is 7.1x its median,
    a fitted spiral winding's is 13x to 26x, and its local step ranges from 21
    to 65 voxels across one strip. Against one number for the whole mesh that
    non-uniformity reads as hundreds of lamina steps.

    The rule is unchanged where the premise held. On the three grown segments
    the count is zero either way; on the spiral windings it falls by an order of
    magnitude and does not reach zero, which is the part the rule was for.

    `window_cells` of 0 or 1 restores the global median exactly, so a caller
    that has no scale to derive a window from gets the published behaviour.
    """
    import numpy as np
    from scipy import ndimage

    row_mask = valid[1:, :] & valid[:-1, :]
    col_mask = valid[:, 1:] & valid[:, :-1]
    total = 0
    for lengths, mask in (
        (np.linalg.norm(points[1:, :] - points[:-1, :], axis=-1), row_mask),
        (np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1), col_mask),
    ):
        if window_cells and window_cells > 1:
            # Edges with no valid pair are filled with the mesh median so a hole
            # neither drags the local estimate down nor invents a step beside
            # itself; the comparison below still only counts real edges.
            filled = np.where(mask, lengths, median_edge)
            reference = ndimage.median_filter(
                filled, size=int(window_cells), mode="nearest")
        else:
            reference = np.full(lengths.shape, median_edge, dtype=float)
        total += int(np.count_nonzero(mask & (lengths > factor * reference)))
    return total


def scale_derived_policy(
    policy: dict[str, Any], step: dict[str, float], *, voxel_um: float | None,
    inter_lamina_um: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn the policy's cell and voxel counts into what they stand for.

    Three of the frozen numbers are lengths written in the units of one corpus:

      * `band_cells` -- how far apart on the grid two triangles must be before
        their proximity in space is evidence rather than adjacency. Four cells
        on the grown corpus is 4 x 12.94 voxels x 9.362 um, about 480 um of
        sheet; four cells on a fitted winding is 731 um, so the same number asks
        a different question of each.
      * `resolution_limit_voxels` -- 16 voxels, which the source justifies as
        150 um at ~9.4 um per voxel, the bottom of the 150-250 um inter-lamina
        range. It never reads the scroll it is measuring, and PHerc0826's own
        spacing is about 371 um.
      * the window the local step median is taken over, which has no frozen
        value because it did not exist.

    Derived here when the scale is known, left frozen when it is not, and the
    receipt says which. Deriving them is not a loosening and should not be sold
    as one: on a fitted winding the derived band is three cells rather than
    four, which is stricter. What it buys is that both corpora are asked the
    same physical question.
    """
    import math

    derived = dict(policy)
    notes: dict[str, Any] = {"voxel_um": voxel_um,
                             "inter_lamina_um": inter_lamina_um,
                             "derived": []}
    if not voxel_um or voxel_um <= 0:
        notes["reason"] = ("no voxel size was supplied, so the frozen cell and "
                           "voxel counts apply and this verdict does not claim "
                           "to have been measured against this scroll's scale")
        return derived, notes

    step_um = float(step["median_edge_voxels"]) * float(voxel_um)
    band = max(1, int(math.ceil(float(policy["band_um"]) / step_um)))
    derived["band_cells"] = band
    notes["derived"].append({"name": "band_cells", "was": policy["band_cells"],
                             "now": band, "from_um": policy["band_um"]})
    # One window spanning about an inter-lamina spacing: wide enough to estimate
    # the local sampling, narrow enough not to average across a real change in
    # it. Odd, because a median filter's window is centred on its own cell.
    spacing_um = float(inter_lamina_um or policy["resolution_limit_um"])
    window = max(3, int(math.ceil(spacing_um / step_um)) | 1)
    derived["step_window_cells"] = window
    notes["derived"].append({"name": "step_window_cells", "was": None,
                             "now": window, "from_um": spacing_um})
    derived["resolution_limit_voxels"] = spacing_um / float(voxel_um)
    notes["derived"].append({
        "name": "resolution_limit_voxels",
        "was": policy["resolution_limit_voxels"],
        "now": derived["resolution_limit_voxels"],
        "from_um": spacing_um,
        "measured": inter_lamina_um is not None})
    if inter_lamina_um is None:
        notes["reason"] = ("this scroll's inter-lamina spacing was not supplied, "
                           "so the resolution limit falls back to the frozen "
                           f"{policy['resolution_limit_um']} um floor rather "
                           "than a spacing measured here")
    return derived, notes


def _far_candidate_pairs(mesh, band_cells: int, gap: float, maximum_pairs: int,
                        size_bins: int = 8):
    """Triangle pairs that are close in space but far apart on the grid.

    Two triangles can only come within `gap` if their centroids are within
    `gap + r_a + r_b`, so the search radius depends on *both* of them. The
    partner's radius was bounded globally -- the 99.9th-percentile circumradius
    for most triangles, the maximum for the largest few -- and that bound is
    what a stretched mesh breaks. One 601-voxel edge widened the search for
    every triangle in the mesh, and two of three fitted spiral windings
    exhausted the pair budget without a single detector having run.

    Bounding the partner per size bin instead keeps the wide radius where the
    tree is small. Nothing can be missed: a triangle is only ever searched
    against a bin whose own largest member bounds every triangle in it.

    That also closes a gap in the published search rather than only speeding it
    up. A triangle whose partner sat in the top 0.1% searched with a radius
    below the one that pair needed, and the pair came back only when index
    order happened to recover it from the other side -- one such pair on the
    grown corpus, two large triangles at exactly their reach. A gate that fails
    closed must not drop candidates quietly.
    """

    import numpy as np
    from scipy.spatial import cKDTree

    centroid = mesh["centroid"]
    circumradius = mesh["circumradius"]
    rows, cols = mesh["triangle_row"], mesh["triangle_col"]
    # Quantile edges rather than even ones: the sizes are heavy-tailed, and a
    # linear split would put almost everything in the first bin and leave the
    # bound as global as it was.
    edges = np.unique(np.quantile(circumradius, np.linspace(0.0, 1.0, size_bins + 1)))
    membership = np.clip(
        np.searchsorted(edges, circumradius, side="right") - 1, 0, len(edges) - 2
    )
    left: list = []
    right: list = []
    kept = 0
    chunk = 4096
    for index in range(len(edges) - 1):
        members = np.flatnonzero(membership == index)
        if members.size == 0:
            continue
        tree = cKDTree(centroid[members])
        radii = gap + circumradius + float(circumradius[members].max())
        for start in range(0, len(centroid), chunk):
            stop = min(start + chunk, len(centroid))
            neighbours = tree.query_ball_point(
                centroid[start:stop], r=radii[start:stop])
            for offset, candidates in enumerate(neighbours):
                if not candidates:
                    continue
                here = start + offset
                other = members[np.asarray(candidates, dtype=np.int64)]
                other = other[other > here]
                if other.size == 0:
                    continue
                separation = np.maximum(
                    np.abs(rows[other] - rows[here]), np.abs(cols[other] - cols[here])
                )
                other = other[separation > band_cells]
                if other.size == 0:
                    continue
                kept += int(other.size)
                if kept > maximum_pairs:
                    raise GeometryGateError(
                        "TIFXYZ geometry gate exceeded its candidate-pair budget"
                    )
                left.append(np.full(other.size, here, dtype=np.int64))
                right.append(other)
    if not left:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(left), np.concatenate(right)


def _point_triangle_distance(point, a, b, c):
    """Vectorized exact point-to-triangle distance (Ericson, RTCD 5.1.5)."""

    import numpy as np

    ab, ac = b - a, c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denominator = va + vb + vc

    def ratio(numerator, divisor):
        safe = np.where(np.abs(divisor) < 1e-30, 1.0, divisor)
        return np.where(np.abs(divisor) < 1e-30, 0.0, numerator / safe)

    v = ratio(vb, denominator)
    w = ratio(vc, denominator)
    closest = a + v[:, None] * ab + w[:, None] * ac
    edge_ab = a + np.clip(ratio(d1, d1 - d3), 0.0, 1.0)[:, None] * ab
    edge_ac = a + np.clip(ratio(d2, d2 - d6), 0.0, 1.0)[:, None] * ac
    edge_bc = b + np.clip(ratio(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)[:, None] * (c - b)
    closest = np.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[:, None], edge_bc, closest)
    closest = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], edge_ac, closest)
    closest = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], edge_ab, closest)
    closest = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, closest)
    closest = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, closest)
    closest = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, closest)
    return np.linalg.norm(point - closest, axis=1)


def _pair_separation(mesh, left, right):
    """Minimum vertex-to-triangle distance for each candidate triangle pair.

    Centroid distance minus circumradii is only a lower bound, and on a mesh
    with stretched triangles that bound is far below the real clearance: it
    reports two windings of one scroll as a doubled surface.  Measuring the six
    vertex-to-other-triangle distances gives the actual separation wherever the
    closest feature involves a vertex, and a true crossing is caught exactly by
    the segment/triangle test regardless.
    """

    import numpy as np

    corners = ("corner_a", "corner_b", "corner_c")
    best = np.full(len(left), np.inf)
    chunk = 200_000
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        sub_left, sub_right = left[start:stop], right[start:stop]
        la, lb, lc = (mesh[name][sub_left] for name in corners)
        ra, rb, rc = (mesh[name][sub_right] for name in corners)
        window = np.full(stop - start, np.inf)
        for point in (la, lb, lc):
            window = np.minimum(window, _point_triangle_distance(point, ra, rb, rc))
        for point in (ra, rb, rc):
            window = np.minimum(window, _point_triangle_distance(point, la, lb, lc))
        best[start:stop] = window
    return best


def _segment_hits_triangle(origin, direction, a, b, c):
    """Vectorized Moller-Trumbore segment/triangle intersection."""

    import numpy as np

    edge1, edge2 = b - a, c - a
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    parallel = np.abs(det) < 1e-12
    safe = np.where(parallel, 1.0, det)
    tvec = origin - a
    u = np.einsum("ij,ij->i", tvec, pvec) / safe
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", direction, qvec) / safe
    t = np.einsum("ij,ij->i", edge2, qvec) / safe
    return (
        ~parallel
        & (u >= 0.0)
        & (u <= 1.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
        & (t >= 0.0)
        & (t <= 1.0)
    )


def _exact_self_intersections(mesh, left, right) -> int:
    """Count candidate pairs whose triangles actually cross in 3-D.

    Coplanar overlap is deliberately not counted here: it is reported by the
    near-coincident-overlap detector, which is the doubled-surface signal.
    """

    import numpy as np

    if len(left) == 0:
        return 0
    # Two triangles can only cross if their centroids are within the sum of
    # their circumradii.  That bound is far too loose to measure separation with
    # -- it is what made stretched meshes look doubled -- but as a necessary
    # condition it discards most candidate pairs before the exact test.
    reachable = np.linalg.norm(
        mesh["centroid"][left] - mesh["centroid"][right], axis=1
    ) <= (mesh["circumradius"][left] + mesh["circumradius"][right])
    left, right = left[reachable], right[reachable]
    if len(left) == 0:
        return 0
    corners = ("corner_a", "corner_b", "corner_c")
    total = 0
    chunk = 200_000
    for start in range(0, len(left), chunk):
        stop = min(start + chunk, len(left))
        a1, b1, c1 = (mesh[name][left[start:stop]] for name in corners)
        a2, b2, c2 = (mesh[name][right[start:stop]] for name in corners)
        hit = np.zeros(stop - start, dtype=bool)
        for origin, target, ta, tb, tc in (
            (a1, b1, a2, b2, c2),
            (b1, c1, a2, b2, c2),
            (c1, a1, a2, b2, c2),
            (a2, b2, a1, b1, c1),
            (b2, c2, a1, b1, c1),
            (c2, a2, a1, b1, c1),
        ):
            hit |= _segment_hits_triangle(origin, target - origin, ta, tb, tc)
        total += int(np.count_nonzero(hit))
    return total


def measure(surface_dir: Path, policy: dict[str, Any] | None = None, *,
            voxel_um: float | None = None,
            inter_lamina_um: float | None = None) -> dict[str, Any]:
    """Compute every geometry metric for one TIFXYZ surface.

    `voxel_um` and `inter_lamina_um` are the scroll's own scale. With them the
    lengths this gate measures against are derived rather than assumed; without
    them the frozen counts apply and the receipt records that the verdict was
    made without knowing the scale.
    """

    import numpy as np

    settings = {**DEFAULT_POLICY, **(policy or {})}
    points, valid = load_tifxyz(Path(surface_dir))
    coverage = _coverage(valid)
    if int(valid.sum()) < 4:
        raise GeometryGateError("TIFXYZ has fewer than four valid coordinates")
    step = _grid_step(points, valid)
    settings, scale = scale_derived_policy(
        settings, step, voxel_um=voxel_um, inter_lamina_um=inter_lamina_um)
    mesh = build_mesh(points, valid)
    median_edge = step["median_edge_voxels"]
    gap = float(settings["gap_voxels"])
    # The local detectors are cheap and decisive.  Running them first means a
    # surface whose mesh is too large for the pair search still gets a verdict
    # rather than silently becoming unmeasured.
    fold_back, folded_cells = _fold_back_pairs(
        mesh, valid, float(settings["fold_angle_deg"]))
    fold = _fold_back_extent(folded_cells, valid)
    discontinuities = _step_discontinuities(
        points, valid, median_edge, float(settings["step_discontinuity_factor"]),
        window_cells=int(settings.get("step_window_cells") or 0),
    )
    near_parallel: int | None = 0
    non_parallel: int | None = 0
    offending_count: int | None = 0
    self_intersections: int | None = 0
    minimum_far_separation = None
    candidate_count: int | None = 0
    pair_stage = "COMPLETE"
    try:
        left, right = _far_candidate_pairs(
            mesh,
            int(settings["band_cells"]),
            gap,
            int(settings["maximum_candidate_pairs"]),
        )
    except GeometryGateError:
        pair_stage = "BUDGET_EXCEEDED"
        near_parallel = non_parallel = offending_count = None
        self_intersections = candidate_count = None
    else:
        candidate_count = int(len(left))
        offending: set[int] = set()
        near_parallel = 0
        non_parallel = 0
        if len(left):
            clearance = _pair_separation(mesh, left, right)
            minimum_far_separation = float(clearance.min())
            close = clearance < gap
            if close.any():
                cosine = np.abs(
                    np.einsum(
                        "ij,ij->i",
                        mesh["normal"][left[close]],
                        mesh["normal"][right[close]],
                    )
                )
                parallel = cosine >= float(
                    np.cos(np.radians(settings["parallel_angle_deg"]))
                )
                near_parallel = int(np.count_nonzero(parallel))
                non_parallel = int(np.count_nonzero(~parallel))
                offending.update(left[close].tolist())
                offending.update(right[close].tolist())
        offending_count = len(offending)
        self_intersections = _exact_self_intersections(mesh, left, right)
    # The three that stay hard, and they are the lamina-switch detectors: on a
    # surface cropped to a clean subregion all three reach zero, which is what a
    # non-local defect looks like when you stop looking at the place it is.
    seam = {
        "near_coincident_overlap_pairs": near_parallel,
        "interpenetration_pairs": non_parallel,
        "offending_triangles": offending_count,
        "lamina_step_discontinuity_edges": discontinuities,
    }
    return {
        "seam": seam,
        # Beside the seam rather than in it: an area measure, not a hard defect.
        "fold_back": {**fold, "pairs": fold_back,
                      "angle_deg": float(settings["fold_angle_deg"]),
                      "ceiling": float(settings["maximum_fold_back_fraction"])},
        "_folded_cells": folded_cells,
        "exact_self_intersection": {
            "schema": "campaignx.mesh_self_intersection.v1",
            "vertices": int(valid.sum()),
            "triangles": int(len(mesh["triangles"])),
            "self_intersections_present": (
                None if self_intersections is None else self_intersections > 0
            ),
            "intersecting_pair_count": self_intersections,
        },
        "grid": {
            "shape": [int(value) for value in valid.shape],
            "valid_coordinate_count": int(valid.sum()),
            "valid_triangle_count": int(len(mesh["triangles"])),
            "degenerate_triangle_count": int(mesh["degenerate"].sum()),
            "grid_far_candidate_pair_count": candidate_count,
            "minimum_far_separation_voxels": minimum_far_separation,
            "non_local_stage": pair_stage,
            **step,
            **coverage,
        },
        "policy": settings,
        "scale": scale,
    }


def classify(metrics: dict[str, Any], integrity: ModuleType | None = None) -> dict[str, Any]:
    """Map measured metrics onto the geometry axis.

    The precedence is frozen and reported: a non-local defect outranks a local
    one, and a hard defect outranks insufficient coverage.
    """

    module = integrity or _mesh_integrity_module()
    seam = metrics["seam"]
    grid = metrics["grid"]
    policy = metrics["policy"]
    fold = metrics.get("fold_back") or {}
    present = metrics["exact_self_intersection"]["self_intersections_present"]
    measured = {name: value for name, value in seam.items() if value is not None}
    complete = len(measured) == len(seam) and present is not None
    hard_defects = module.hard_defect_count(measured, bool(present))
    coverage_failures = []
    if grid["valid_fraction"] < float(policy["minimum_valid_fraction"]):
        coverage_failures.append("VALID_FRACTION_BELOW_FLOOR")
    if grid["largest_component_fraction"] < float(policy["minimum_largest_component_fraction"]):
        coverage_failures.append("FRAGMENTED_COVERAGE")
    if grid["valid_triangle_count"] < int(policy["minimum_valid_triangles"]):
        coverage_failures.append("TOO_FEW_VALID_TRIANGLES")
    # Fold-back reads as an area, not as a count. Any of them used to reject the
    # surface, and what that rejected was wrinkles: the maximum area this gate
    # would certify on a fitted winding was half a square centimetre, against
    # the four a campaign asks for, and no angle separates a wrinkle from an
    # error. What the ceiling catches is a mesh folded everywhere.
    folded = float(fold.get("fraction") or 0.0)
    over_ceiling = folded > float(policy["maximum_fold_back_fraction"])

    def positive(name: str) -> bool:
        return bool(seam.get(name))

    if positive("interpenetration_pairs") or positive("lamina_step_discontinuity_edges"):
        state, reason = GEOMETRY_REJECTED_LAMINA_SWITCH, "NON_PARALLEL_STAB_OR_STEP_DISCONTINUITY"
    elif positive("near_coincident_overlap_pairs"):
        state, reason = GEOMETRY_REJECTED_BRIDGE, "NEAR_COINCIDENT_DOUBLED_SURFACE"
    elif bool(present):
        state, reason = GEOMETRY_REJECTED_DISTORTION, "SELF_INTERSECTION"
    elif over_ceiling:
        state, reason = (GEOMETRY_REJECTED_DISTORTION,
                         "FOLD_BACK_AREA_ABOVE_CEILING")
    elif coverage_failures:
        state, reason = GEOMETRY_REJECTED_COVERAGE, ",".join(coverage_failures)
    elif not complete:
        # Fail closed.  No detector fired, but a detector did not run, so the
        # surface has not been certified against the defects it would have found.
        state, reason = GEOMETRY_UNMEASURED, f"INCOMPLETE_{grid.get('non_local_stage', 'STAGE')}"
    else:
        state, reason = GEOMETRY_CERTIFIED, "NO_HARD_DEFECT_AT_ARTIFACT_SAMPLING"
    return {
        "geometry_qc_state": state,
        "reason": reason,
        "hard_defects_observed": hard_defects if complete else None,
        "coverage_failures": coverage_failures,
        "measurement_complete": complete,
        # Reported whatever the verdict: a certified surface can carry folded
        # cells, and the number is how a reader knows how many.
        "fold_back_fraction": folded,
        "fold_back_within_ceiling": not over_ceiling,
        "status": "PASS" if state == GEOMETRY_CERTIFIED else "FAIL",
    }


def certify(surface_dir: Path, policy: dict[str, Any] | None = None, *,
            voxel_um: float | None = None,
            inter_lamina_um: float | None = None,
            mask_dir: Path | None = None) -> dict[str, Any]:
    """Return a complete, hash-linked certification receipt for one surface.

    `mask_dir` is where the fold-back mask is written when there is one. Left
    unset the mask is measured and not published, which is the right default
    for a read-only survey and the wrong one for a certification a renderer
    will consume.
    """

    surface_dir = Path(surface_dir).resolve()
    inputs: dict[str, Any] = {}
    for name in (*REQUIRED_FILES, "meta.json"):
        path = surface_dir / name
        if path.is_file():
            inputs[name] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    try:
        metrics = measure(surface_dir, policy, voxel_um=voxel_um,
                          inter_lamina_um=inter_lamina_um)
        verdict = classify(metrics)
        # The mask is the deliverable, not the count. A fold-back is an inverted
        # normal in that cell, and the platform's orientation fix forces one
        # sign for a whole mesh -- which does nothing for a cell that disagrees
        # with its neighbours. Written as a grid the same shape as the TIFXYZ so
        # a consumer can index it directly.
        folded = metrics.pop("_folded_cells", None)
        if folded is not None and mask_dir is not None:
            metrics["fold_back"]["mask"] = _write_mask(
                folded, Path(mask_dir) / FOLD_BACK_MASK)
        metrics.pop("_folded_cells", None)
        error = None
    except Exception as failure:  # noqa: BLE001 - unmeasurable is a real verdict
        metrics = {"seam": None, "exact_self_intersection": None, "grid": None,
                   "fold_back": None,
                   "scale": {"voxel_um": voxel_um,
                             "inter_lamina_um": inter_lamina_um, "derived": []},
                   "policy": {**DEFAULT_POLICY, **(policy or {})}}
        verdict = {
            "geometry_qc_state": GEOMETRY_UNMEASURED,
            "reason": "MEASUREMENT_FAILED",
            "hard_defects_observed": None,
            "coverage_failures": [],
            "measurement_complete": False,
            "status": "FAIL",
        }
        error = f"{type(failure).__name__}: {failure}"
    grid = metrics.get("grid") or {}
    median_edge = grid.get("median_edge_voxels")
    receipt = {
        "schema": SCHEMA,
        "surface_dir": str(surface_dir),
        "surface_name": surface_dir.name,
        "inputs": inputs,
        **verdict,
        **{key: value for key, value in metrics.items() if key != "policy"},
        "policy": metrics["policy"],
        "error": error,
        "resolution_limited": (
            None
            if median_edge is None
            else bool(median_edge >= float(metrics["policy"]["resolution_limit_voxels"]))
        ),
        # Against a spacing measured on this scroll, or against the frozen floor
        # standing in for one. A reader deciding what a resolution-limited
        # certification is worth needs to know which of the two it was.
        "resolution_limit_measured_here": bool(
            (metrics.get("scale") or {}).get("inter_lamina_um") is not None),
        "non_claims": [
            "GEOMETRY_CERTIFIED is not physical-sheet acceptance, ink, text, or First Letters",
            "the gate cannot resolve a defect finer than the TIFXYZ grid step",
            "an unmeasured surface is not a certified surface",
            "a certified surface may still carry folded cells: fold-back is an "
            "area measure here and not a hard defect, because on the corpus "
            "where it was looked at the folds were wrinkles in a flattened "
            "scroll and rejecting them capped the certifiable area at half a "
            "square centimetre",
            "the fold-back mask is not advisory. Each marked cell has a normal "
            "inverted against its neighbours, and the platform's orientation "
            "fix sets one sign per mesh -- so a renderer that samples those "
            "cells samples the far side of the sheet, which is the failure that "
            "produced an ink map correlating 0.09 with the published one",
        ],
    }
    return receipt


def _iter_surface_dirs(root: Path) -> Iterable[Path]:
    # workspace/surfaces publishes symlinked trees, so the walk must follow
    # links; a set of resolved paths keeps a linked duplicate from being
    # certified twice and inflating the published distribution.
    seen: set[Path] = set()
    for directory, _subdirectories, files in os.walk(root, followlinks=True):
        if not all(name in files for name in REQUIRED_FILES):
            continue
        resolved = Path(directory).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def run_batch(roots: Sequence[Path], policy: dict[str, Any] | None = None,
              require_meta: bool = False, *, voxel_um: float | None = None,
              inter_lamina_um: float | None = None,
              write_masks: bool = False) -> dict[str, Any]:
    from collections import Counter

    receipts = []
    for root in roots:
        root = Path(root).resolve()
        candidates = [root] if (root / "x.tif").is_file() else list(_iter_surface_dirs(root))
        for surface_dir in candidates:
            if require_meta and not (surface_dir / "meta.json").is_file():
                continue
            receipts.append(certify(
                surface_dir, policy, voxel_um=voxel_um,
                inter_lamina_um=inter_lamina_um,
                # Beside the surface it describes, so a consumer that has the
                # TIFXYZ has the mask.
                mask_dir=surface_dir if write_masks else None))
    receipts.sort(key=lambda row: row["surface_dir"])
    distribution = Counter(row["geometry_qc_state"] for row in receipts)
    reasons = Counter(row["reason"] for row in receipts)
    return {
        "schema": BATCH_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "roots": [str(Path(root).resolve()) for root in roots],
        "require_meta_json": require_meta,
        "policy": {**DEFAULT_POLICY, **(policy or {})},
        "scale": {"voxel_um": voxel_um, "inter_lamina_um": inter_lamina_um},
        "counts": {
            "surfaces": len(receipts),
            "by_geometry_qc_state": dict(sorted(distribution.items())),
            "by_reason": dict(sorted(reasons.items())),
            "certified": distribution[GEOMETRY_CERTIFIED],
            "rejected": sum(
                value for key, value in distribution.items()
                if key.startswith("GEOMETRY_REJECTED_")
            ),
            "unmeasured": distribution[GEOMETRY_UNMEASURED],
        },
        "surfaces": receipts,
        "non_claims": [
            "this distribution measures the artifacts present on this host, not the full fleet",
            "GEOMETRY_CERTIFIED is not physical-sheet acceptance, ink, text, or First Letters",
        ],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--surface-dir", type=Path, action="append", default=[])
    value.add_argument("--surface-root", type=Path, action="append", default=[])
    value.add_argument("--require-meta-json", action="store_true")
    value.add_argument("--output", type=Path)
    # Facts about the scroll, not policy: the gate derives its lengths from
    # these when they are given, and says in the receipt when they were not.
    value.add_argument("--write-masks", action="store_true",
                       help="publish the fold-back mask beside each surface; a "
                            "renderer needs it, a survey does not")
    value.add_argument("--voxel-um", type=float, default=None,
                       help="microns per voxel of the volume these surfaces "
                            "are addressed in")
    value.add_argument("--inter-lamina-um", type=float, default=None,
                       help="this scroll's measured inter-lamina spacing; "
                            "without it the frozen 150 um floor stands in")
    for name, default in DEFAULT_POLICY.items():
        value.add_argument(
            f"--{name.replace('_', '-')}", type=type(default), default=None
        )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    roots = [*args.surface_dir, *args.surface_root]
    if not roots:
        raise SystemExit("at least one --surface-dir or --surface-root is required")
    policy = {
        name: getattr(args, name)
        for name in DEFAULT_POLICY
        if getattr(args, name, None) is not None
    }
    report = run_batch(roots, policy, require_meta=args.require_meta_json,
                       voxel_um=args.voxel_um,
                       inter_lamina_um=args.inter_lamina_um,
                       write_masks=args.write_masks)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(json.dumps({"output": str(args.output), **report["counts"]}, indent=2, sort_keys=True))
    else:
        print(payload)
    return 0 if report["counts"]["rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
