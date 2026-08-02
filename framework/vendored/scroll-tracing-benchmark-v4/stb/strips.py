"""Qualified strip packs: a self-contained, re-qualifiable export of one
window, independent of the original band npz.

export_strip(ref, start, cfg, gates, path) writes a .npz {xyz window +
classes -1/0/+1 point sets local to the window's column neighborhood,
pitch, gates dict, provenance}; load_strip(path) reads it back into a
plain dict; qualify_strip(strip) rebuilds a minimal stb.core.Reference
from ONLY that dict and re-runs gates a/b (stb.gates.coverage_and_gates_ab)
independently of whatever `gates` export_strip stored for provenance.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from . import core
from . import gates as gates_mod

STRIP_CLASSES = (-1, 0, 1)


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def matched_column_neighborhood(ref, n, seed_pts, margin):
    """The column range of class `n`'s points actually needed to
    reproduce gaps_for/score_prediction's nearest-neighbor results for
    `seed_pts`, found by querying the real, full-band tree once (plus
    `margin`) -- NOT assumed to be near the window's own column range.

    This band's winding classes +/-1 sit a full spiral revolution away
    from class 0 (physically adjacent -- one pitch over in 3D -- but
    displaced by however many columns one full 2*pi of unwrap takes,
    which can be thousands of columns for a low-curvature band), so
    "near the window" and "near the window's column index" are different
    things; the matched columns of the actual nearest neighbors are the
    only reliable guide to which chunk of class n's points a strip needs.
    """
    _, idx = ref.trees[n].query(seed_pts, workers=-1)
    matched_cols = ref.cols_of[n][idx]
    return int(matched_cols.min()) - margin, int(matched_cols.max()) + margin


def export_strip(ref, start, cfg, gates, path, pitch=None, meta=None,
                  neighborhood_margin=300):
    """Export a self-contained "strip" for the window built as `ref`
    (stb.reference.reference_at(..., start, cfg)): its own xyz seed grid,
    STEP-sampled seed classification, and the classes {-1,0,+1} point
    sets, each restricted to a `neighborhood_margin`-vox column
    neighborhood around where THAT class's points actually match the
    window's seed points (see matched_column_neighborhood -- restricted
    for file size, not correctness; see qualify_strip's docstring for why
    this is safe for re-qualification), plus enough of cfg (classes
    restricted to -1/0/1, threshold_kind/value) and the original band's
    row/col shape (for the same band-edge check
    core.gaps_for/score_prediction do) to re-run gates a/b from the saved
    data alone, no original band npz required.

    `gates` is the caller's already-computed
    stb.gates.coverage_and_gates_ab(ref, cfg) result, stored verbatim as
    provenance (qualify_strip does NOT read it back -- it recomputes
    independently, so a re-qualification failure would show up even if
    this stored snapshot is stale or was hand-edited). `pitch` is the
    window's CT-measured pitch estimate (e.g. windows_v2.json's `p2_ct`)
    if known, else None. `meta` is an arbitrary caller-supplied provenance
    dict (stratum, kappa, coverage, ...), stored verbatim.

    Returns the metadata dict that was written (path included);
    load_strip(path) reads the file back.
    """
    band_shape = tuple(int(s) for s in ref.xyz.shape[:2])
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]

    arrays = {
        "seed_xyz": np.asarray(ref.seed, dtype=np.float64),
        "rr": np.asarray(ref.rr, dtype=np.int64),
        "cc": np.asarray(ref.cc, dtype=np.int64),
        "seed_cls": np.asarray(ref.seed_cls, dtype=np.int64),
    }
    populated = []
    for n in STRIP_CLASSES:
        if n in ref.trees:
            lo, hi = matched_column_neighborhood(ref, n, seed_pts, neighborhood_margin)
            rows_n, cols_n, pts_n = ref.rows_of[n], ref.cols_of[n], ref.pts_of[n]
            keep = (cols_n >= lo) & (cols_n < hi)
            rows_n, cols_n, pts_n = rows_n[keep], cols_n[keep], pts_n[keep]
            populated.append(int(n))
        else:
            rows_n = np.empty(0, dtype=np.int64)
            cols_n = np.empty(0, dtype=np.int64)
            pts_n = np.empty((0, 3), dtype=np.float64)
        arrays[f"rows_class_{n}"] = rows_n.astype(np.int64)
        arrays[f"cols_class_{n}"] = cols_n.astype(np.int64)
        arrays[f"pts_class_{n}"] = pts_n.astype(np.float64)

    meta_blob = {
        "start": int(start),
        "band_shape": list(band_shape),
        "classes": list(STRIP_CLASSES),
        "populated_classes": populated,
        "threshold_kind": cfg.threshold_kind,
        "threshold_value": float(cfg.threshold_value),
        "step": int(cfg.step),
        "neighborhood_margin": int(neighborhood_margin),
        "pitch": None if pitch is None else float(pitch),
        "gates": gates,
        "meta": meta or {},
    }
    arrays["_meta_json"] = np.array(json.dumps(meta_blob, default=_json_default))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {"path": str(path), **json.loads(json.dumps(meta_blob, default=_json_default))}


def load_strip(path):
    """Load a strip written by export_strip into a plain dict: numpy
    arrays (seed_xyz, rr, cc, seed_cls, {rows,cols,pts}_class_{-1,0,1})
    merged with the JSON metadata (start, band_shape, classes,
    threshold_kind/value, step, pitch, gates, meta)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        strip = {k: data[k] for k in data.files if k != "_meta_json"}
        meta_blob = json.loads(data["_meta_json"].item())
    strip.update(meta_blob)
    strip["path"] = str(path)
    return strip


def _rebuild_reference(strip):
    """Minimal stb.core.Reference sufficient for
    stb.gates.coverage_and_gates_ab, built ONLY from a loaded strip's own
    arrays: xyz is a zero-cost same-shape stand-in (core.gaps_for/
    score_prediction only ever read xyz.shape, never a value, for the
    row/col band-edge check), cls is unused downstream so left None,
    valid is a placeholder for the same reason."""
    band_shape = tuple(int(s) for s in strip["band_shape"])
    xyz_stand_in = np.broadcast_to(np.float64(0.0), band_shape + (3,))
    trees, rows_of, cols_of, pts_of = {}, {}, {}, {}
    for n in STRIP_CLASSES:
        pts = strip[f"pts_class_{n}"]
        if pts.shape[0] == 0:
            continue
        rows_of[n] = strip[f"rows_class_{n}"]
        cols_of[n] = strip[f"cols_class_{n}"]
        pts_of[n] = pts
        trees[n] = cKDTree(pts)
    return core.Reference(
        xyz=xyz_stand_in, valid=np.zeros(1, dtype=bool), row0=0,
        seed=strip["seed_xyz"], cls=None, trees=trees,
        rows_of=rows_of, cols_of=cols_of, pts_of=pts_of,
        rr=strip["rr"], cc=strip["cc"], seed_cls=strip["seed_cls"],
    )


def qualify_strip(strip):
    """Re-run gates a/b (stb.gates.coverage_and_gates_ab) using ONLY a
    loaded strip's own data -- an independent re-verification, not a
    replay of the `gates` field export_strip stored for provenance.

    This is only as faithful as export_strip's column-neighborhood
    restriction of the classes {-1,0,+1} point sets
    (matched_column_neighborhood): it is safe as long as the margin is
    generous relative to how far a seed cell's true nearest
    same/adjacent-class match can drift in column across the window's own
    100 seed rows/cols, which holds empirically for PHerc0332's geometry
    (a winding sheet's nearest neighbor in an adjacent class is one pitch
    away in 3D -- possibly thousands of columns away in this band's raw
    indexing, since a full 2*pi of unwrap spans many columns at low
    curvature -- but that displacement is essentially the SAME for every
    seed cell in one 200-column window, so a few-hundred-column margin
    around the matched range covers the whole window). test_strips.py's
    deliberately label-shuffled-strip regression test exercises the
    failure mode this would otherwise mask: shuffling breaks the classes'
    spatial separation and gate b correctly fails.
    """
    cfg_stub = SimpleNamespace(
        classes=tuple(strip["classes"]),
        threshold_kind=str(strip["threshold_kind"]),
        threshold_value=float(strip["threshold_value"]),
    )
    ref = _rebuild_reference(strip)
    return gates_mod.coverage_and_gates_ab(ref, cfg_stub)
