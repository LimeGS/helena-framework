"""Build a strip-v0 reference strip from a tifxyz segment window.

Given a segment's x.tif/y.tif/z.tif grids (the tifxyz format used
throughout Vesuvius Challenge tooling, see
release/neural-tracing-audit/seed_segment/ for a real example) and a
rectangular (row, col) window of that grid, this script:

1. Estimates the scroll's local rotation axis (or takes one explicitly via
   --axis) and computes each grid point's angular position (theta) and
   radius around that axis.
2. Detects which grid axis (row or column) is the one that actually winds
   through revolutions, by comparing the total unwrapped phase sweep along
   each axis at its most densely valid line. (Verified empirically on the
   one real multi-wrap band available to this repo -- see README.md
   "Verified geometry note" -- the column axis is the winding axis there;
   this script does not hardcode that and measures it per-input.)
3. Phase-unwraps theta independently along each line of the winding axis
   (no cross-line phase propagation -- see "Limitations" below) and bins
   the result into 2*pi-wide slices to assign every valid grid point a
   sequential, zero-based wrap id.
4. Groups points by wrap id into the strip's per-wrap point sets, estimates
   local pitch as the pooled nearest-neighbor 3D distance between
   consecutive wraps' point sets (the SIMPLE choice: plain point-set
   nearest-neighbor, not a normal-projected distance -- see README for why),
   and assigns a tier from the measured pitch.

Limitations (v0, deliberately kept simple per the project brief):

- Per-line phase unwrapping is independent line-to-line; there is no
  explicit cross-line branch-matching like
  release/neural-tracing-audit/benchmark_core.py's `load_reference` does.
  This is fine as long as theta varies slowly across the non-winding axis
  (true for both the synthetic test scroll and the one real geometry this
  was checked against), but is not guaranteed in general -- a segment with
  large or noisy theta variation across the non-winding axis could get
  inconsistent wrap-id branches between lines. Not fixed in v0; documented.
- Normals are only computed when the requested window has zero masked/
  invalid cells (np.gradient-based finite differences do not account for a
  mask). A window with any invalid cells silently skips normals for the
  whole strip.
- The rotation axis defaults to a pure Z direction through the window's own
  centroid, matching the one real geometry checked (see README). A tilted
  axis needs an explicit --axis with two points.
- Tested end-to-end on a synthetic analytic spiral only (see tests/). It
  was also run against the real seed_segment/ directory bundled in
  neural-tracing-audit as a smoke test of the failure path: that directory
  is a real Vesuvius Challenge tifxyz segment, but it is a narrow single-
  revolution-fraction crop (measured ~0.002 revolutions across its own
  200x200 window -- see README "make_strip.py on seed_segment" for the
  exact measurement) and correctly raises InsufficientRevolutionsError
  rather than producing a bogus strip. It has not been validated end-to-end
  on any real window that actually spans >= 2 wraps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from scoring_core import nearest_neighbor_distances
from strip_format import (
    MIN_POINTS_PER_WRAP,
    SCHEMA_VERSION,
    edge_flags_from_phase,
    save_strip,
)

# PROVISIONAL tier thresholds, in micrometers of local pitch, exactly as
# specified in the project brief. Not independently calibrated against a
# corpus of segments -- they encode "ultra-compressed / medium / easy"
# as three round-number bins, nothing more.
TIER_ULTRA_MAX_UM = 120.0
TIER_MEDIUM_MAX_UM = 250.0

# PROVISIONAL: sanity-check threshold for the auto-estimated (--axis
# omitted) rotation axis. A genuine, distant scroll rotation axis gives a
# radius that varies only mildly within one local window (it grows slowly
# with revolution count, at most a few local gaps' worth). A window's own
# centroid used as a stand-in axis instead measures "distance to the
# window's own middle", which typically ranges from ~0 (near the middle) to
# ~half the window's own diagonal (at its corners) -- a span ratio far
# larger than true scroll geometry produces, and one that quietly manufactures
# fake "winding" out of the patch's own shape rather than real revolutions.
# Threshold not calibrated against a corpus of segments; chosen to clearly
# separate the one real failure case measured (seed_segment/, ratio ~334x,
# see README.md) from the synthetic test scroll (ratio ~1.4x).
MAX_AUTO_AXIS_RADIUS_SPAN_RATIO = 10.0

# PROVISIONAL: how close (in revolutions) a point may sit to its wrap's
# angular-coverage boundary before being edge-flagged (see
# strip_format.Strip.edges for what the flag does downstream). 0.01 rev
# is ~3 sample steps on the synthetic test scroll and ~84 columns on the
# UC-01 source band -- generous, but exclusion is the safe direction for a
# reference (the source audit excluded rather than guessed at its band
# edges too). Not calibrated beyond "flags the boundary artifacts observed
# while validating the null-baseline check" (see qualify_strip.py (c)).
EDGE_MARGIN_REVOLUTIONS = 0.01


class MakeStripError(ValueError):
    """Base class for make_strip failures that should abort strip
    construction with a clear diagnostic rather than emit a bogus strip."""


class InsufficientRevolutionsError(MakeStripError):
    """Raised when a requested window does not span enough revolutions to
    produce >= 2 wraps with >= MIN_POINTS_PER_WRAP points each."""


class UnreliableAxisEstimateError(MakeStripError):
    """Raised when --axis was omitted and the window's own centroid produces
    a radius span too large to plausibly be a genuine distant scroll axis
    (see MAX_AUTO_AXIS_RADIUS_SPAN_RATIO). This is what actually happens on
    the real seed_segment/ directory: it is a small, locally near-planar
    patch, and computing angle-around-its-own-centroid manufactures a fake
    ~0.5 revolution "winding" out of the patch's own shape. Pass --axis
    explicitly (the segment's own fitted umbilicus) to avoid this, or set
    allow_unreliable_axis=True / --allow-unreliable-axis to override."""


@dataclass
class BuiltStrip:
    wraps: Dict[int, np.ndarray]
    normals: Dict[int, np.ndarray]
    edges: Dict[int, np.ndarray]
    pitch_um: Dict[str, float]
    tier: str
    diagnostics: Dict = field(default_factory=dict)


def assign_tier(pitch_um_median: float) -> str:
    """ultra < 120 <= medium < 250 <= easy, per the project brief.
    PROVISIONAL thresholds -- see module docstring."""
    if pitch_um_median < TIER_ULTRA_MAX_UM:
        return "ultra"
    elif pitch_um_median < TIER_MEDIUM_MAX_UM:
        return "medium"
    else:
        return "easy"


def parse_axis_arg(s: Optional[str]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse '--axis z0,y0,x0[,z1,y1,x1]' into (center_xyz, direction_xyz).

    Input order is z,y,x (matching the CLI spec in the project brief);
    everything downstream of this function works in xyz.
    """
    if s is None:
        return None
    vals = [float(v) for v in s.split(",")]
    if len(vals) == 3:
        z0, y0, x0 = vals
        center = np.array([x0, y0, z0], dtype=np.float64)
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif len(vals) == 6:
        z0, y0, x0, z1, y1, x1 = vals
        p0 = np.array([x0, y0, z0], dtype=np.float64)
        p1 = np.array([x1, y1, z1], dtype=np.float64)
        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            raise ValueError("--axis: the two axis points must not coincide")
        direction = direction / norm
        center = p0
    else:
        raise ValueError(
            "--axis expects 3 or 6 comma-separated floats: z0,y0,x0[,z1,y1,x1]"
        )
    return center, direction


def project_to_axis_frame(points_xyz: np.ndarray, center: np.ndarray, direction: np.ndarray):
    """Angular position (theta), radius, and along-axis coordinate of each
    point relative to a rotation axis (center + direction).

    Reduces to the familiar atan2(y - cy, x - cx) / sqrt((x-cx)^2+(y-cy)^2)
    used throughout the source neural-tracing-audit scripts when
    direction == (0, 0, 1).
    """
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, direction)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    e1 = helper - np.dot(helper, direction) * direction
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(direction, e1)

    rel = points_xyz - center
    along = rel @ direction
    u = rel @ e1
    v = rel @ e2
    theta = np.arctan2(v, u)
    radius = np.sqrt(u ** 2 + v ** 2)
    return theta, radius, along


def _measure_sweep(theta: np.ndarray, valid: np.ndarray, vary_axis: int) -> float:
    """Total unwrapped phase sweep along `vary_axis` (0=rows, 1=cols), taken
    along whichever line (fixed index on the other axis) has the most valid
    points. Returns 0.0 if no line has >= 2 valid points."""
    if vary_axis == 1:
        counts = valid.sum(axis=1)
        if counts.size == 0 or counts.max() < 2:
            return 0.0
        r = int(np.argmax(counts))
        cols = np.where(valid[r])[0]
        th = np.unwrap(theta[r, cols])
        return float(th[-1] - th[0])
    else:
        counts = valid.sum(axis=0)
        if counts.size == 0 or counts.max() < 2:
            return 0.0
        c = int(np.argmax(counts))
        rows = np.where(valid[:, c])[0]
        th = np.unwrap(theta[rows, c])
        return float(th[-1] - th[0])


def detect_winding_axis(theta: np.ndarray, valid: np.ndarray) -> Tuple[str, float]:
    """Which grid axis winds through revolutions: whichever of (rows vary,
    cols fixed) / (cols vary, rows fixed) shows the larger total unwrapped
    phase sweep along its most-valid line."""
    sweep_row = _measure_sweep(theta, valid, vary_axis=0)
    sweep_col = _measure_sweep(theta, valid, vary_axis=1)
    if abs(sweep_col) >= abs(sweep_row):
        return "col", sweep_col
    return "row", sweep_row


def unwrap_full(theta: np.ndarray, valid: np.ndarray, winding_axis: str) -> np.ndarray:
    """Independent per-line 1D phase unwrap along the winding axis. See
    module docstring "Limitations" -- no cross-line propagation."""
    U = np.full(theta.shape, np.nan)
    if winding_axis == "col":
        for r in range(theta.shape[0]):
            cols = np.where(valid[r])[0]
            if cols.size == 0:
                continue
            if cols.size == 1:
                U[r, cols[0]] = theta[r, cols[0]]
                continue
            U[r, cols] = np.unwrap(theta[r, cols])
    else:
        for c in range(theta.shape[1]):
            rows = np.where(valid[:, c])[0]
            if rows.size == 0:
                continue
            if rows.size == 1:
                U[rows[0], c] = theta[rows[0], c]
                continue
            U[rows, c] = np.unwrap(theta[rows, c])
    return U


def assign_wrap_ids(U: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Bin unwrapped phase into 2*pi-wide slices -> sequential 0-based wrap
    ids (0 = the slice containing the window's smallest phase value)."""
    finite = valid & np.isfinite(U)
    wrap_id = np.full(U.shape, -1, dtype=np.int64)
    if not finite.any():
        return wrap_id
    u_min = np.min(U[finite])
    wrap_id[finite] = np.floor((U[finite] - u_min) / (2 * np.pi)).astype(np.int64)
    return wrap_id


def compute_grid_normals(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Per-point unit normal from local finite differences (np.gradient
    handles the array edges via one-sided differences automatically).
    Orientation (which side the normal points to) is NOT validated or
    guaranteed consistent -- it falls out of row/col index order. Returns
    (unit_normal (R,C,3), degenerate (R,C) bool where the local tangents
    were collinear/zero and no normal could be formed)."""
    dx_dr, dx_dc = np.gradient(x)
    dy_dr, dy_dc = np.gradient(y)
    dz_dr, dz_dc = np.gradient(z)
    tangent_row = np.stack([dx_dr, dy_dr, dz_dr], axis=-1)
    tangent_col = np.stack([dx_dc, dy_dc, dz_dc], axis=-1)
    normal = np.cross(tangent_col, tangent_row)
    norm = np.linalg.norm(normal, axis=-1)
    degenerate = norm <= 1e-9
    norm_safe = np.where(degenerate, 1.0, norm)
    unit_normal = normal / norm_safe[..., None]
    return unit_normal, degenerate


def compute_pitch(wraps: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Pooled nearest-neighbor 3D distance between every pair of
    consecutive (by sorted wrap index) wrap point sets.

    This is the SIMPLE option documented in the project brief: plain
    point-set nearest-neighbor distance, not a distance projected along
    local surface normals. It slightly overestimates the true perpendicular
    sheet spacing when the local surface is tilted relative to the
    wrap-to-wrap offset direction (the nearest neighbor is not exactly
    "straight across"), but needs no per-point normal and reuses the same
    KD-tree primitive as every other distance computation in this repo.
    """
    ids = sorted(wraps.keys())
    all_d = []
    for a, b in zip(ids[:-1], ids[1:]):
        # Query from whichever of the two point sets is SMALLER into the
        # larger one. A window's first/last wrap is usually a partial
        # revolution (the window rarely starts/ends exactly on a 2*pi
        # boundary); querying from the fuller set into a partial one
        # inflates the distance for every query point whose angular phase
        # falls outside the partial wrap's limited coverage (verified: this
        # skewed p90 to ~10x the true pitch on the synthetic test scroll).
        # Querying the other way avoids it, since every point in the
        # smaller/partial wrap has a genuinely nearby neighbor in the fuller
        # one.
        pa, pb = wraps[a], wraps[b]
        if pa.shape[0] <= pb.shape[0]:
            d, _ = nearest_neighbor_distances(pa, pb)
        else:
            d, _ = nearest_neighbor_distances(pb, pa)
        all_d.append(d)
    if not all_d:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    pooled = np.concatenate(all_d)
    return {
        "median": float(np.median(pooled)),
        "p10": float(np.percentile(pooled, 10)),
        "p90": float(np.percentile(pooled, 90)),
    }


def build_strip_from_grids(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    window: Tuple[int, int, int, int],
    axis: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    voxel_size_um: float = 1.0,
    expected_revolutions: Optional[float] = None,
    compute_normals_flag: bool = True,
    min_points_per_wrap: int = MIN_POINTS_PER_WRAP,
    allow_unreliable_axis: bool = False,
) -> BuiltStrip:
    """Core windowing + revolution-splitting logic, decoupled from any file
    I/O so tests can drive it directly with synthetic in-memory grids.

    window = (row_start, row_end, col_start, col_end), end-exclusive.
    axis = (center_xyz, direction_xyz) or None to estimate (see module
    docstring: default is a pure-Z axis through the window's centroid).
    """
    r0, r1, c0, c1 = window
    xw = x[r0:r1, c0:c1]
    yw = y[r0:r1, c0:c1]
    zw = z[r0:r1, c0:c1]
    vw = valid[r0:r1, c0:c1]

    if int(vw.sum()) < 2 * min_points_per_wrap:
        raise InsufficientRevolutionsError(
            f"window [{r0}:{r1}, {c0}:{c1}] has only {int(vw.sum())} valid "
            f"point(s); need enough for at least 2 wraps of >= "
            f"{min_points_per_wrap} points each"
        )

    axis_was_given = axis is not None
    if axis is None:
        center = np.array(
            [float(np.mean(xw[vw])), float(np.mean(yw[vw])), float(np.mean(zw[vw]))]
        )
        direction = np.array([0.0, 0.0, 1.0])
    else:
        center, direction = axis

    pts = np.stack([xw, yw, zw], axis=-1)
    theta, radius, _along = project_to_axis_frame(pts, center, direction)
    theta = np.where(vw, theta, np.nan)

    if not axis_was_given and not allow_unreliable_axis:
        r_valid = radius[vw]
        r_lo = float(np.percentile(r_valid, 1))
        r_hi = float(np.percentile(r_valid, 99))
        span_ratio = r_hi / max(r_lo, 1e-9)
        if span_ratio > MAX_AUTO_AXIS_RADIUS_SPAN_RATIO:
            raise UnreliableAxisEstimateError(
                f"auto-estimated axis (window's own centroid) gives radius "
                f"spanning {r_lo:.3g} to {r_hi:.3g} (ratio {span_ratio:.1f}x, "
                f"limit {MAX_AUTO_AXIS_RADIUS_SPAN_RATIO}x) across this "
                f"window. A genuine distant scroll axis should give a much "
                f"steadier radius; this usually means the window is a small "
                f"local patch and its own centroid is being measured "
                f"instead of the true rotation axis (see seed_segment/ in "
                f"README.md for a real example of exactly this failure). "
                f"Pass --axis explicitly (the segment's own fitted "
                f"umbilicus), or allow_unreliable_axis=True / "
                f"--allow-unreliable-axis if you are sure."
            )

    winding_axis, sweep_rad = detect_winding_axis(theta, vw)
    measured_revolutions = abs(sweep_rad) / (2 * np.pi)

    U = unwrap_full(theta, vw, winding_axis)
    wrap_id = assign_wrap_ids(U, vw)

    ids_present, counts = np.unique(wrap_id[wrap_id >= 0], return_counts=True)
    kept_ids = sorted(
        int(i) for i, c in zip(ids_present, counts) if c >= min_points_per_wrap
    )
    wrap_point_counts_all = {int(i): int(c) for i, c in zip(ids_present, counts)}

    if len(kept_ids) < 2:
        raise InsufficientRevolutionsError(
            f"only {len(kept_ids)} wrap(s) with >= {min_points_per_wrap} "
            f"points survived binning (measured {measured_revolutions:.4f} "
            f"revolutions along the {winding_axis} axis of this window, "
            f"raw per-id point counts {wrap_point_counts_all}); need >= 2. "
            "This window does not span enough revolutions to build a "
            "multi-wrap strip -- pick a wider window or a more compressed "
            "region."
        )

    rows_idx, cols_idx = np.where(vw)
    wid_flat = wrap_id[rows_idx, cols_idx]
    xyz_flat = pts[rows_idx, cols_idx]

    normals_grid = None
    degenerate_grid = None
    if compute_normals_flag:
        if vw.all():
            normals_grid, degenerate_grid = compute_grid_normals(xw, yw, zw)
        else:
            print(
                "make_strip: window contains masked/invalid cells; skipping "
                "normal computation for this strip (v0 limitation, see "
                "README.md)",
                file=sys.stderr,
            )

    U_flat = U[rows_idx, cols_idx]
    margin_rad = EDGE_MARGIN_REVOLUTIONS * 2 * np.pi

    wraps: Dict[int, np.ndarray] = {}
    normals: Dict[int, np.ndarray] = {}
    edges: Dict[int, np.ndarray] = {}
    for wid in kept_ids:
        sel = wid_flat == wid
        wraps[wid] = xyz_flat[sel].astype(np.float32)
        edges[wid] = edge_flags_from_phase(U_flat[sel], margin_rad)
        if normals_grid is not None:
            nrm = normals_grid[rows_idx[sel], cols_idx[sel]]
            deg = degenerate_grid[rows_idx[sel], cols_idx[sel]]
            if not deg.any():
                normals[wid] = nrm.astype(np.float32)

    pitch_native = compute_pitch(wraps)
    pitch_um = {
        k: (v * voxel_size_um if v == v else v)  # v == v is a NaN-safe check
        for k, v in pitch_native.items()
    }
    tier = assign_tier(pitch_um["median"]) if pitch_um["median"] == pitch_um["median"] else "unknown"

    revolutions_note = None
    if expected_revolutions is not None:
        diff = abs(measured_revolutions - expected_revolutions)
        if diff > 0.25:  # PROVISIONAL sanity-check tolerance
            revolutions_note = (
                f"measured revolutions ({measured_revolutions:.3f}) differs "
                f"from --revolutions ({expected_revolutions:.3f}) by more "
                f"than 0.25 -- double-check the window or the axis"
            )
            print(f"make_strip: WARNING: {revolutions_note}", file=sys.stderr)

    diagnostics = {
        "winding_axis": winding_axis,
        "measured_revolutions": measured_revolutions,
        "expected_revolutions": expected_revolutions,
        "revolutions_warning": revolutions_note,
        "n_wraps": len(wraps),
        "wrap_point_counts": {str(k): int(v.shape[0]) for k, v in wraps.items()},
        "pitch_native_units": pitch_native,
        "n_wraps_before_min_points_filter": int(len(ids_present)),
        "edge_flagged_counts": {str(k): int(v.sum()) for k, v in edges.items()},
    }
    return BuiltStrip(wraps=wraps, normals=normals, edges=edges,
                      pitch_um=pitch_um, tier=tier, diagnostics=diagnostics)


def load_tifxyz_grids(dirpath) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read x.tif/y.tif/z.tif (+ optional mask.tif) from a tifxyz directory.

    Optional dependency: needs `tifffile` (not required by the core
    strip-v0 format/scoring tools, only by this real-segment loader).
    """
    try:
        import tifffile
    except ImportError as exc:
        raise SystemExit(
            "Reading a tifxyz segment directory needs the optional "
            "'tifffile' package: pip install tifffile. The core strip "
            "format, qualify_strip.py and score_strip.py do not need it."
        ) from exc

    dirpath = Path(dirpath)
    x = np.asarray(tifffile.imread(dirpath / "x.tif"), dtype=np.float64)
    y = np.asarray(tifffile.imread(dirpath / "y.tif"), dtype=np.float64)
    z = np.asarray(tifffile.imread(dirpath / "z.tif"), dtype=np.float64)
    mask_path = dirpath / "mask.tif"
    if mask_path.exists():
        valid = np.asarray(tifffile.imread(mask_path)) != 0
    else:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        zero = (x == 0) & (y == 0) & (z == 0)
        valid = finite & ~zero
        print(
            "make_strip: no mask.tif found next to x.tif; inferring "
            "validity from finite-and-not-all-zero coordinates (v0 "
            "fallback)",
            file=sys.stderr,
        )
    return x, y, z, valid


def _sha256_of_files(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segment_dir", help="tifxyz directory (x.tif, y.tif, z.tif, optional mask.tif)")
    parser.add_argument("--window", required=True, help="row_start,row_end,col_start,col_end (end-exclusive)")
    parser.add_argument("--axis", default=None, help="z0,y0,x0[,z1,y1,x1]; default: pure-Z axis through the window centroid")
    parser.add_argument("--voxel-size-um", type=float, default=1.0, help="micrometers per coordinate unit (default 1.0; override for real CT-voxel data)")
    parser.add_argument("--revolutions", type=float, default=None, help="expected revolutions the window spans (sanity-check only)")
    parser.add_argument("--scroll", default="unknown")
    parser.add_argument("--segment-id", default="unknown")
    parser.add_argument("--out", required=True, help="output strip .npz path")
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument(
        "--allow-unreliable-axis", action="store_true",
        help="override the auto-estimated-axis sanity check (see "
             "UnreliableAxisEstimateError); only useful without --axis",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, z, valid = load_tifxyz_grids(args.segment_dir)

    window_vals = [int(v) for v in args.window.split(",")]
    if len(window_vals) != 4:
        raise SystemExit("--window expects row_start,row_end,col_start,col_end")
    window = tuple(window_vals)

    axis = parse_axis_arg(args.axis)

    dirpath = Path(args.segment_dir)
    source_checksum = _sha256_of_files(dirpath / "x.tif", dirpath / "y.tif", dirpath / "z.tif")

    try:
        built = build_strip_from_grids(
            x, y, z, valid, window,
            axis=axis,
            voxel_size_um=args.voxel_size_um,
            expected_revolutions=args.revolutions,
            compute_normals_flag=not args.no_normals,
            allow_unreliable_axis=args.allow_unreliable_axis,
        )
    except MakeStripError as exc:
        print(f"make_strip: FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        sys.exit(1)

    meta = {
        "scroll": args.scroll,
        "segment_id": args.segment_id,
        "window": {
            "row_start": window[0], "row_end": window[1],
            "col_start": window[2], "col_end": window[3],
            "winding_axis": built.diagnostics["winding_axis"],
            "measured_revolutions": built.diagnostics["measured_revolutions"],
            "expected_revolutions": built.diagnostics["expected_revolutions"],
        },
        "voxel_size_um": args.voxel_size_um,
        "tier": built.tier,
        "schema_version": SCHEMA_VERSION,
        "source_checksum": source_checksum,
        "source_kind": "tifxyz_directory",
        "built_by": "make_strip.py",
    }

    save_strip(args.out, built.wraps, built.normals, built.pitch_um, meta,
               edges=built.edges)

    print(f"make_strip: wrote {args.out}")
    print(f"  wraps: {sorted(built.wraps.keys())}")
    print(f"  winding axis: {built.diagnostics['winding_axis']}  "
          f"measured revolutions: {built.diagnostics['measured_revolutions']:.3f}")
    print(f"  pitch_um: median={built.pitch_um['median']:.1f}  "
          f"p10={built.pitch_um['p10']:.1f}  p90={built.pitch_um['p90']:.1f}")
    print(f"  tier: {built.tier}")
    print(f"  normals computed for wraps: {sorted(built.normals.keys())}")


if __name__ == "__main__":
    main()
