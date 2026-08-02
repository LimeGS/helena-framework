"""strip-v0 container format: a qualified multi-wrap reference strip.

A "strip" is a small local patch of a scroll segment's own geometry that
happens to record two or more consecutive papyrus wraps in the same local
neighborhood (the wraps a tracer or mesher would need to connect). It is
stored as a single .npz file with one flat point set per wrap, plus a small
amount of provenance metadata. It is NOT a full segmentation and it is NOT
manual ground truth -- see README.md for what a strip does and does not
claim.

Design choices (v0, documented so they can be revisited):

- Coordinate order is **xyz** everywhere (matching the tifxyz convention and
  the band_r1145_200_xyz.npz layout in the source neural-tracing-audit
  repo), not zyx. Every array of shape (N, 3) in this format is x, y, z.
- Each wrap is stored as a flat (N, 3) point set, not a 2D grid. Any row/col
  connectivity the source segment had is not preserved in strip-v0 -- only
  the point geometry. This keeps the format mesher/tracer agnostic (both
  consume point clouds), at the cost of not being able to re-derive normals
  or connectivity after the fact unless they were computed at build time
  and stored alongside.
- Wraps are numbered sequentially from 0 (wrap_00, wrap_01, ...), in the
  order they were encountered while winding through the source geometry.
  This is deliberately simpler than the signed winding-class numbering
  (..., -1, 0, +1, ...) used internally by the source audit's scoring
  scripts, which numbered wraps relative to a seed. A strip converted from
  that representation must record the original class-to-wrap-index mapping
  in `meta` (see convert_ntaudit_band.py) so provenance is not lost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCHEMA_VERSION = "strip-v0"

# PROVISIONAL: a floor on how many points a wrap needs before we trust it to
# support a KD-tree nearest-surface query at all. Not calibrated against any
# particular tracer/mesher; just large enough to reject obviously-degenerate
# wraps (a handful of stray points) while being far below what any real
# strip should contain (thousands of points per wrap).
MIN_POINTS_PER_WRAP = 8

# Required keys in the `meta` JSON blob. `meta` is intentionally a loose
# JSON dict (not a fixed struct) so new fields can be added without breaking
# the .npz schema, but qualify_strip.py / score_strip.py rely on these being
# present for provenance and unit handling.
REQUIRED_META_KEYS = (
    "scroll",
    "segment_id",
    "window",
    "voxel_size_um",
    "tier",
    "schema_version",
    "source_checksum",
)


@dataclass
class Strip:
    """In-memory representation of a strip-v0 file.

    wraps: {wrap_index: (N_k, 3) float32 xyz points}, wrap_index >= 0.
    normals: {wrap_index: (N_k, 3) float32}, may be a strict subset of
        wraps.keys() (normals are optional per wrap) or empty.
    edges: {wrap_index: (N_k,) bool}, optional per wrap. True marks a
        point near that wrap's angular-coverage boundary (the 2*pi cut
        line where one wrap's bin ends and the next begins, or the strip's
        overall winding ends). Scoring uses this the way the source audit
        (release/neural-tracing-audit/gate_3090/score_native.py) used its
        band row-0/row-199 rule: a query whose nearest reference point is
        edge-flagged is EXCLUDED, because near the coverage boundary the
        strip cannot tell a real inter-sheet distance from an
        along-the-spiral one. Strips without edge flags still score, with
        leakier behavior near the cut lines (documented in README.md).
    pitch_um: {"median": float, "p10": float, "p90": float} -- local pitch
        profile in micrometers, or NaNs if unknown.
    meta: parsed JSON metadata dict (see REQUIRED_META_KEYS).
    """

    wraps: Dict[int, np.ndarray]
    normals: Dict[int, np.ndarray] = field(default_factory=dict)
    edges: Dict[int, np.ndarray] = field(default_factory=dict)
    pitch_um: Dict[str, float] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)

    @property
    def wrap_indices(self) -> List[int]:
        return sorted(self.wraps.keys())

    @property
    def n_wraps(self) -> int:
        return len(self.wraps)

    def total_points(self) -> int:
        return int(sum(v.shape[0] for v in self.wraps.values()))


def _wrap_key(idx: int) -> str:
    if idx < 0 or idx > 99:
        raise ValueError(
            f"wrap index {idx} out of strip-v0 v0 range [0, 99] "
            "(2-digit zero-padded keys); widen _wrap_key if you legitimately "
            "need more wraps in one strip"
        )
    return f"wrap_{idx:02d}"


def _normals_key(idx: int) -> str:
    return f"normals_{idx:02d}"


def _edges_key(idx: int) -> str:
    return f"edges_{idx:02d}"


def save_strip(
    path,
    wraps: Dict[int, np.ndarray],
    normals: Optional[Dict[int, np.ndarray]] = None,
    pitch_um: Optional[Dict[str, float]] = None,
    meta: Optional[Dict] = None,
    edges: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """Write a strip-v0 .npz file.

    `wraps` keys do not need to be contiguous or start at 0 on input; they
    are re-indexed to a sorted, contiguous 0-based sequence on write (with
    the original-to-new mapping recorded in meta["wrap_reindex"] if it
    changed) so downstream code can always assume wrap_00.. wrap_{K-1}.
    """
    if len(wraps) == 0:
        raise ValueError("cannot save a strip with zero wraps")

    normals = dict(normals or {})
    edges = dict(edges or {})
    meta = dict(meta or {})
    meta["schema_version"] = SCHEMA_VERSION

    original_indices = sorted(wraps.keys())
    reindex = {old: new for new, old in enumerate(original_indices)}
    if any(old != new for old, new in reindex.items()):
        meta["wrap_reindex"] = {str(old): new for old, new in reindex.items()}

    arrays = {}
    for old_idx, new_idx in reindex.items():
        pts = np.asarray(wraps[old_idx], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"wrap {old_idx}: expected shape (N, 3), got {pts.shape}"
            )
        arrays[_wrap_key(new_idx)] = pts
        if old_idx in normals:
            nrm = np.asarray(normals[old_idx], dtype=np.float32)
            if nrm.shape != pts.shape:
                raise ValueError(
                    f"normals for wrap {old_idx}: shape {nrm.shape} != "
                    f"wrap shape {pts.shape}"
                )
            arrays[_normals_key(new_idx)] = nrm
        if old_idx in edges:
            edg = np.asarray(edges[old_idx], dtype=bool)
            if edg.shape != (pts.shape[0],):
                raise ValueError(
                    f"edges for wrap {old_idx}: shape {edg.shape} != "
                    f"({pts.shape[0]},)"
                )
            arrays[_edges_key(new_idx)] = edg

    arrays["wrap_indices"] = np.asarray(sorted(reindex.values()), dtype=np.int64)

    pitch_um = pitch_um or {}
    arrays["pitch_um"] = np.asarray(
        [
            float(pitch_um.get("median", np.nan)),
            float(pitch_um.get("p10", np.nan)),
            float(pitch_um.get("p90", np.nan)),
        ],
        dtype=np.float64,
    )

    arrays["meta"] = np.asarray(json.dumps(meta))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_strip(path) -> Strip:
    """Read a strip-v0 .npz file into a Strip object."""
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    if "wrap_indices" in keys:
        wrap_indices = [int(i) for i in data["wrap_indices"]]
    else:
        wrap_indices = sorted(
            int(k.split("_")[1]) for k in keys if k.startswith("wrap_")
        )

    wraps: Dict[int, np.ndarray] = {}
    normals: Dict[int, np.ndarray] = {}
    edges: Dict[int, np.ndarray] = {}
    for idx in wrap_indices:
        wkey = _wrap_key(idx)
        if wkey not in keys:
            raise ValueError(f"strip file missing {wkey} listed in wrap_indices")
        wraps[idx] = data[wkey]
        nkey = _normals_key(idx)
        if nkey in keys:
            normals[idx] = data[nkey]
        ekey = _edges_key(idx)
        if ekey in keys:
            edges[idx] = data[ekey].astype(bool)

    if "pitch_um" in keys:
        p = data["pitch_um"]
        pitch_um = {"median": float(p[0]), "p10": float(p[1]), "p90": float(p[2])}
    else:
        pitch_um = {"median": float("nan"), "p10": float("nan"), "p90": float("nan")}

    if "meta" in keys:
        meta = json.loads(str(data["meta"]))
    else:
        meta = {}

    return Strip(wraps=wraps, normals=normals, edges=edges,
                 pitch_um=pitch_um, meta=meta)


def validate_strip(strip: Strip) -> List[str]:
    """Structural + basic-sanity validation. Returns a list of problem
    strings; an empty list means the strip passed validation.

    This checks *shape*: enough wraps, enough points, finite values,
    required metadata present. It does NOT check whether the wraps are
    geometrically trustworthy (a wrong-side wrap could still pass this) --
    that is qualify_strip.py's job and is a separate, stronger notion of
    "qualified" than merely "well-formed".
    """
    problems: List[str] = []

    if len(strip.wraps) < 2:
        problems.append(
            f"strip has {len(strip.wraps)} wrap(s); need >= 2 for a "
            "multi-wrap reference"
        )

    for idx in sorted(strip.wraps.keys()):
        pts = strip.wraps[idx]
        label = f"wrap_{idx:02d}"
        if pts.ndim != 2 or pts.shape[1] != 3:
            problems.append(f"{label}: expected shape (N, 3), got {pts.shape}")
            continue
        if pts.shape[0] < MIN_POINTS_PER_WRAP:
            problems.append(
                f"{label}: only {pts.shape[0]} point(s), need >= "
                f"{MIN_POINTS_PER_WRAP}"
            )
        if pts.shape[0] > 0 and not np.all(np.isfinite(pts)):
            n_bad = int((~np.isfinite(pts)).any(axis=1).sum())
            problems.append(f"{label}: {n_bad} point(s) contain NaN/Inf")

        if idx in strip.normals:
            nrm = strip.normals[idx]
            nlabel = f"normals_{idx:02d}"
            if nrm.shape != pts.shape:
                problems.append(
                    f"{nlabel}: shape {nrm.shape} does not match {label} "
                    f"shape {pts.shape}"
                )
            elif nrm.shape[0] > 0 and not np.all(np.isfinite(nrm)):
                problems.append(f"{nlabel}: contains non-finite values")

        if idx in strip.edges:
            edg = strip.edges[idx]
            elabel = f"edges_{idx:02d}"
            if edg.shape != (pts.shape[0],):
                problems.append(
                    f"{elabel}: shape {edg.shape} does not match "
                    f"({pts.shape[0]},)"
                )
            elif edg.dtype != np.bool_:
                problems.append(f"{elabel}: dtype {edg.dtype} is not bool")
            elif edg.all():
                problems.append(
                    f"{elabel}: every point is edge-flagged; the wrap has "
                    "no scoreable interior"
                )

    missing_meta = [k for k in REQUIRED_META_KEYS if k not in strip.meta]
    if missing_meta:
        problems.append(f"meta missing required keys: {missing_meta}")

    if "pitch_um" in strip.__dict__:
        pm = strip.pitch_um.get("median")
        if pm is not None and not (pm != pm):  # not NaN
            if pm <= 0:
                problems.append(f"pitch_um.median = {pm} is not positive")

    return problems


def edge_flags_from_phase(phase: np.ndarray, margin: float) -> np.ndarray:
    """Boolean edge flags for one wrap's points from their per-point
    angular phase (any consistent monotonically-increasing-along-the-wind
    parameterization, e.g. unwrapped theta): True for points within
    `margin` (same units as `phase`) of the wrap's own observed phase
    extremes -- its coverage boundaries. Used by make_strip.py,
    convert_ntaudit_band.py and the test fixtures so all edge flags mean
    the same thing.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if phase.size == 0:
        return np.zeros(0, dtype=bool)
    lo, hi = float(np.min(phase)), float(np.max(phase))
    return (phase <= lo + margin) | (phase >= hi - margin)


def is_qualified(strip_path) -> bool:
    """True iff a sibling qualification.json exists next to `strip_path` and
    records an overall pass. This is a cheap filesystem check used by
    score_strip.py to decide whether to print the UNQUALIFIED warning; it
    does not re-run the qualification suite itself.
    """
    strip_path = Path(strip_path)
    qual_path = strip_path.with_name(strip_path.stem + ".qualification.json")
    if not qual_path.exists():
        # also accept a plain "qualification.json" next to the strip file
        qual_path = strip_path.parent / "qualification.json"
    if not qual_path.exists():
        return False
    try:
        report = json.loads(qual_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(report.get("overall_pass", False))
