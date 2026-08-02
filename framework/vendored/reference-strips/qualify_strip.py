"""Automated qualification suite for a strip-v0 reference strip.

A strip is only trustworthy as a benchmark reference if its per-wrap point
sets actually are what they claim to be: distinct, spatially separated
sheets of the same spiral, correctly labeled, at the recorded pitch. This
script runs the checks and writes `<strip>.qualification.json` (machine)
plus `<strip>.qualification.md` (human). score_strip.py warns when used
with a strip that has no passing qualification report next to it.

The four checks, and what each one actually proves
--------------------------------------------------

(a) SELF-TEST (pipeline plumbing). Every wrap's own points are pushed
    through the full KD-tree scoring pipeline; each point must come back
    assigned to its own wrap at 0.0 distance. By construction this cannot
    fail on label geometry (a point is always at distance 0 from the tree
    it is a member of) -- it is deliberately a *plumbing* check, catching
    indexing bugs, coordinate-order mixups (xyz vs zyx), dtype truncation
    and tree-construction errors. The source audit used the same check the
    same way (seed vs its own wrap = 0.000 vox) and three earlier metric
    versions were discarded for failing it. It is NOT the check that
    catches mislabeled wraps; that is (b)'s job.

(b) WRONG-SIDE SEPARATION (label geometry -- the mislabeling detector).
    For each ordered pair of adjacent wraps (k -> j), every wrap-k point's
    distance to wrap j's surface (d_cross) is compared against that
    point's own within-wrap sampling distance (d_self, its distance to the
    nearest *other* point of its own wrap). A point is "wrongly close" if
    d_cross < separation_factor * d_self. In a clean strip d_cross is on
    the order of the inter-sheet pitch while d_self is the sampling step,
    so almost no point is wrongly close; a point whose *label* was swapped
    with the neighboring wrap sits at sampling distance from its true
    wrap's tree and is flagged. The check passes when >= min_fail_pct of
    points correctly "fail" to land near the wrong side.

    Why not simply "d_cross must exceed half the local gap"? Because the
    local gap to the wrong side IS d_cross -- any threshold derived from
    the quantity under test is a tautology that passes every strip,
    corrupted or not. Comparing against the own-wrap sampling distance is
    what makes this check able to fail. It also (intentionally) fails
    strips whose sampling is too coarse relative to their pitch: such a
    reference genuinely cannot discriminate wraps at half-gap resolution,
    and deserves to be UNQUALIFIED.

    Known benign edge effect, absorbed by the tolerance: the wraps of a
    strip are consecutive turns of one continuous spiral, so the handful
    of points immediately adjacent to the 2*pi cut line where wrap k ends
    and wrap k+1 begins are legitimately at ~sampling distance from the
    neighboring wrap's point set. On the geometries checked this is <<1%
    of points; the default 5% tolerance absorbs it.

(c) NULL BASELINE (scorer sensitivity). A synthetic no-skill predictor --
    each wrap-k point pushed NULL_OFFSET_GAP_MULTIPLIER (default 2.0) x
    its own local gap along the direction toward the target wrap, i.e. a
    constant-rule offset that overshoots to roughly one wrap PAST the
    target -- must score ~100% wrong-hop. If the scorer cannot flag an
    intentional overshoot on this strip's geometry, the strip (or scorer)
    cannot be trusted to flag a tracer's overshoot either.
    PROVENANCE note: the source audit's Baseline A
    (release/neural-tracing-audit/winding_audit_v4.py) offset seed points
    by the *network's own median displacement magnitude* along the PPM
    surface normals and measured 100% wrong at ~2x the network's distance.
    strip-v0 files may not carry normals and have no network output at
    qualification time, so the offset here is ADAPTED to: 2x the local
    gap, along the exact toward-target direction (which needs no sign
    disambiguation). A 1x-gap offset would land ON the target wrap and
    pass, which is why the multiplier is 2.
    Two seed exclusions apply before building the null, both instances of
    "the reference cannot support this test vector here": (i) spiral-cut
    edge seeds (gap at sampling scale -- the strip-v0 equivalent of the
    source audit's band row-edge exclusion) and (ii) coverage-hole seeds
    (gap a strong outlier vs the pair median -- the target sheet is
    missing/unsegmented locally, so a gap-scaled offset is meaningless;
    observed on the real UC-01 band, where such nulls flew across the
    hole and landed on the target sheet elsewhere).

(d) CT INTENSITY CROSS-CHECK (--ct-check, OPTIONAL, NETWORK-REQUIRED).
    Samples raw CT intensity along the strip's stored normals and measures
    the inter-sheet peak spacing, comparing it against the strip's
    recorded pitch. This is the only check whose truth signal comes from
    outside the segmentation geometry (image intensity vs geometry), which
    is exactly why it is valuable -- and why it needs the CT volume and
    therefore the network (or a local zarr). It is NEVER run by the test
    suite and is skipped unless explicitly requested. Requires the
    optional `zarr` dependency and per-wrap normals in the strip.

Edge flags: both strip builders in this repo (make_strip.py and
strips/UC-01/convert_ntaudit_band.py) store per-wrap coverage-boundary
flags, and the scorer excludes queries whose nearest reference point is
flagged (the port of the source audit's band row-edge rule). A hand-built
strip without edge flags can still be scored, but will typically miss the
null-baseline bar by a fraction of a percent because of un-excludable
spiral-cut boundary artifacts (measured ~99.0% vs the 99% bar on the
synthetic test geometry): qualification effectively expects edge flags.

Exit code: 0 if overall_pass, 2 if not (so shell pipelines can gate on
qualification).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from scoring_core import (
    NULL_OFFSET_GAP_MULTIPLIER,
    build_trees,
    score_points,
    summarize,
)
from strip_format import Strip, load_strip, validate_strip

# ----------------------------------------------------------------------
# PROVISIONAL qualification thresholds. None of these are calibrated
# against a corpus of strips; they encode the pass criteria observed in
# the one source audit (see docs pointers in README.md) plus round-number
# tolerances. Change them in one place here; they are recorded verbatim
# into every qualification.json for auditability.
# ----------------------------------------------------------------------
CONFIG = {
    # (a) self-test: max allowed distance from a wrap's own point to its
    # own wrap's surface (should be exactly 0.0; tolerance covers float
    # round-trip), and required fraction assigned to their own wrap.
    "self_test_max_dist": 1e-6,                 # PROVISIONAL
    "self_test_min_own_wrap_pct": 100.0,        # PROVISIONAL
    # (b) wrong-side separation.
    "wrong_side_min_fail_pct": 95.0,            # PROVISIONAL (per project brief)
    "wrong_side_separation_factor": 3.0,        # PROVISIONAL
    # (c) null baseline.
    "null_min_wrong_pct": 99.0,                 # PROVISIONAL ("~100% wrong")
    "null_offset_gap_multiplier": NULL_OFFSET_GAP_MULTIPLIER,  # 2.0, see scoring_core
    # Seeds whose gap to the target wrap exceeds this multiple of the
    # pair's median gap are excluded from the null: such a point faces a
    # coverage hole in the target wrap (damaged / unsegmented sheet), its
    # "gap" is not a local inter-sheet distance, and an offset scaled by it
    # is meaningless as a test vector (measured on the real UC-01 band:
    # seeds with ~4.7x-median gaps produced nulls that flew ~135 vox and
    # landed on the target sheet elsewhere, faking "hop-correct").
    "null_gap_outlier_multiplier": 3.0,         # PROVISIONAL
    # ... and seeds whose gap VECTOR is strongly non-perpendicular to the
    # local sheet are excluded for the same underlying reason seen from a
    # second angle: a genuine inter-sheet hop reference points across the
    # sheets (along the local normal), while a nearest-target vector that
    # runs along the sheet means the target wrap is locally unsegmented
    # and the "nearest point" is the far rim of a hole. Local sheet normal
    # is estimated per seed by PCA over its own-wrap neighborhood (no
    # stored normals needed). Measured on the real UC-01 band: offenders
    # had |cos(gap_vec, normal)| ~0.03-0.2 (nearest target point displaced
    # ~97% along the scroll axis, i.e. along-sheet) vs ~1.0 for genuine
    # trans-sheet references, so 0.5 (60 degrees) splits them with wide
    # margin on both sides.
    "null_min_abs_cos_gap_normal": 0.5,         # PROVISIONAL
    "null_normal_neighbors": 10,                # PROVISIONAL (PCA k-NN)
    # (d) CT cross-check: acceptable ratio of CT-measured spacing to the
    # strip's recorded pitch. The source audit measured CT 13.0 vox vs
    # KD-tree gaps 8.8-11.2 vox (ratio 1.16-1.48), so a [0.5, 2.0] band
    # accepts that while still catching order-of-magnitude disagreement.
    "ct_ratio_min": 0.5,                        # PROVISIONAL
    "ct_ratio_max": 2.0,                        # PROVISIONAL
    # Deterministic query-side subsampling cap for large strips (trees are
    # never subsampled -- only the points being tested). rng seed is fixed.
    "max_points_per_check": 250_000,            # PROVISIONAL
    "subsample_seed": 0,
}


def _subsample(points: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Deterministic subsample of query points (never of reference trees)."""
    if points.shape[0] <= cap:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=cap, replace=False)
    return points[np.sort(idx)]


def _local_normals_pca(points: np.ndarray, tree, k: int) -> np.ndarray:
    """Per-point local sheet normal estimated as the smallest principal
    component of the point's k-nearest-neighbor neighborhood on its own
    wrap. Sign is arbitrary (callers must use |cos|). Vectorized eigh over
    (n, 3, 3) covariance stacks."""
    k_eff = min(k + 1, tree.n)
    _, nn_idx = tree.query(points, k=k_eff, workers=-1)
    if k_eff == 1:
        return np.tile(np.array([np.nan, np.nan, np.nan]), (points.shape[0], 1))
    neigh = tree.data[nn_idx]                       # (n, k_eff, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k_eff
    _, eigvecs = np.linalg.eigh(cov)                # ascending eigenvalues
    return eigvecs[:, :, 0]                          # smallest -> normal


# ----------------------------------------------------------------------
# Check (a): self-test
# ----------------------------------------------------------------------

def check_self_test(strip: Strip, trees, config=CONFIG) -> Dict:
    per_wrap = {}
    worst_dist = 0.0
    worst_own_pct = 100.0
    for k in strip.wrap_indices:
        pts = _subsample(
            np.asarray(strip.wraps[k], dtype=np.float64),
            config["max_points_per_check"],
            config["subsample_seed"],
        )
        # full pipeline: nearest wrap over ALL trees + distance to own tree
        d_own, _ = trees[k].query(pts, workers=-1)
        dists = np.full((pts.shape[0], len(trees)), np.inf)
        wrap_ids = sorted(trees.keys())
        for j, wid in enumerate(wrap_ids):
            dists[:, j], _ = trees[wid].query(pts, workers=-1)
        nearest = np.asarray(wrap_ids)[np.argmin(dists, axis=1)]
        own_pct = 100.0 * float((nearest == k).mean())
        max_d = float(d_own.max()) if d_own.size else float("nan")
        per_wrap[str(k)] = {
            "n_tested": int(pts.shape[0]),
            "max_dist_to_own_wrap": max_d,
            "pct_nearest_own_wrap": own_pct,
        }
        worst_dist = max(worst_dist, max_d)
        worst_own_pct = min(worst_own_pct, own_pct)

    passed = (
        worst_dist <= config["self_test_max_dist"]
        and worst_own_pct >= config["self_test_min_own_wrap_pct"]
    )
    return {
        "pass": bool(passed),
        "max_dist_any_wrap": worst_dist,
        "min_pct_nearest_own_wrap": worst_own_pct,
        "per_wrap": per_wrap,
        "what_this_proves": (
            "pipeline plumbing only (indexing, coordinate order, dtype, "
            "tree construction); label geometry is check (b)'s job"
        ),
    }


# ----------------------------------------------------------------------
# Check (b): wrong-side separation
# ----------------------------------------------------------------------

def check_wrong_side(strip: Strip, trees, config=CONFIG) -> Dict:
    ids = strip.wrap_indices
    pairs = {}
    worst_fail_pct = 100.0
    for k in ids:
        pts_full = np.asarray(strip.wraps[k], dtype=np.float64)
        pts = _subsample(pts_full, config["max_points_per_check"],
                         config["subsample_seed"])
        # own-wrap sampling distance: nearest DISTINCT point in own wrap
        # (k=2: first hit is the point itself at distance 0)
        d_self2, _ = trees[k].query(pts, k=2, workers=-1)
        d_self = d_self2[:, 1]
        for j in (k - 1, k + 1):
            if j not in trees:
                continue
            d_cross, _ = trees[j].query(pts, workers=-1)
            wrongly_close = d_cross < config["wrong_side_separation_factor"] * d_self
            fail_pct = 100.0 * float((~wrongly_close).mean())
            pairs[f"{k}->{j}"] = {
                "n_tested": int(pts.shape[0]),
                "fail_pct": fail_pct,  # correctly rejected as wrong side
                "wrongly_close_pct": 100.0 - fail_pct,
                "median_d_cross": float(np.median(d_cross)),
                "median_d_self": float(np.median(d_self)),
            }
            worst_fail_pct = min(worst_fail_pct, fail_pct)

    passed = worst_fail_pct >= config["wrong_side_min_fail_pct"]
    return {
        "pass": bool(passed),
        "min_fail_pct_over_pairs": worst_fail_pct,
        "required_min_fail_pct": config["wrong_side_min_fail_pct"],
        "separation_factor": config["wrong_side_separation_factor"],
        "pairs": pairs,
        "what_this_proves": (
            "adjacent wraps are spatially distinct at this strip's own "
            "sampling resolution; catches wrap-label shuffles and "
            "undersampled references"
        ),
    }


# ----------------------------------------------------------------------
# Check (c): null baseline
# ----------------------------------------------------------------------

def check_null_baseline(strip: Strip, trees, config=CONFIG) -> Dict:
    ids = strip.wrap_indices
    pairs = {}
    worst_wrong_pct = 100.0
    mult = config["null_offset_gap_multiplier"]
    sep = config["wrong_side_separation_factor"]
    for k in ids:
        target = k + 1
        if target not in trees:
            continue
        pts = _subsample(
            np.asarray(strip.wraps[k], dtype=np.float64),
            config["max_points_per_check"],
            config["subsample_seed"],
        )
        gap, idx = trees[target].query(pts, workers=-1)

        # Exclude spiral-cut edge points BEFORE building the null: a seed
        # point whose "gap" to the target wrap is at its own sampling scale
        # (gap < separation_factor * own-wrap sampling distance) sits right
        # at the 2*pi cut where consecutive turns of the continuous spiral
        # meet -- its measured gap is not a real inter-sheet gap, so a
        # gap-scaled offset from it is meaningless. This mirrors the source
        # audit's exclusion of cells whose nearest reference point sat on
        # the band's first/last row (reference-coverage edge): excluded,
        # not scored. Same points, different geometry of the boundary.
        d_self2, _ = trees[k].query(pts, k=2, workers=-1)
        d_self = d_self2[:, 1]
        interior = gap >= sep * d_self
        n_edge_excluded = int((~interior).sum())

        # Coverage-hole exclusions (see the two config entries): a seed
        # faces a hole in the target wrap's coverage when (i) its gap is a
        # strong outlier vs the pair's median, or (ii) its gap VECTOR runs
        # along the sheet instead of across it (|cos| vs the local
        # PCA-estimated sheet normal below threshold). Excluded, not
        # tested.
        med_gap = float(np.median(gap[interior])) if interior.any() else 0.0
        hole = interior & (gap > config["null_gap_outlier_multiplier"] * med_gap)

        normals = _local_normals_pca(pts, trees[k],
                                     config["null_normal_neighbors"])
        gap_vec = np.asarray(strip.wraps[target], dtype=np.float64)[idx] - pts
        gap_vec_unit = gap_vec / np.maximum(
            np.linalg.norm(gap_vec, axis=1, keepdims=True), 1e-12
        )
        cos_gap_normal = np.abs(np.sum(gap_vec_unit * normals, axis=1))
        along_sheet = interior & (
            cos_gap_normal < config["null_min_abs_cos_gap_normal"]
        )

        n_hole_excluded = int((hole | along_sheet).sum())
        keep = interior & ~hole & ~along_sheet

        pts, gap, idx = pts[keep], gap[keep], idx[keep]
        if pts.shape[0] == 0:
            pairs[f"{k}->front({target})"] = {
                "n_included": 0,
                "n_excluded": 0,
                "n_edge_excluded": n_edge_excluded,
                "n_coverage_hole_excluded": n_hole_excluded,
                "wrong_hop_pct": float("nan"),
                "note": "no interior points to test",
            }
            continue

        nearest_target_pts = np.asarray(strip.wraps[target], dtype=np.float64)[idx]
        toward = nearest_target_pts - pts
        norm = np.linalg.norm(toward, axis=1, keepdims=True)
        toward = toward / np.maximum(norm, 1e-12)
        null_pred = pts + mult * gap[:, None] * toward

        result = score_points(strip, trees, null_pred, from_wrap=k,
                              direction="front")
        s = summarize(result)
        pairs[f"{k}->front({target})"] = {
            "n_included": s["included"],
            "n_excluded": s["excluded"],
            "n_edge_excluded": n_edge_excluded,
            "n_coverage_hole_excluded": n_hole_excluded,
            "wrong_hop_pct": s["wrong_hop_pct"],
            "wrong_wrap_pct": s["wrong_wrap_pct"],
        }
        if s["included"] > 0:
            worst_wrong_pct = min(worst_wrong_pct, s["wrong_hop_pct"])

    passed = worst_wrong_pct >= config["null_min_wrong_pct"]
    return {
        "pass": bool(passed),
        "min_wrong_hop_pct_over_pairs": worst_wrong_pct,
        "required_min_wrong_pct": config["null_min_wrong_pct"],
        "offset_gap_multiplier": mult,
        "pairs": pairs,
        "what_this_proves": (
            "the half-gap scorer flags a deliberate constant-rule "
            "overshoot on this strip's geometry (scorer sensitivity)"
        ),
    }


# ----------------------------------------------------------------------
# Check (d): OPTIONAL CT intensity cross-check (network / local zarr)
# ----------------------------------------------------------------------

def check_ct(strip: Strip, ct_url: str, n_cells: int = 9,
             prominence: float = 15.0, config=CONFIG) -> Dict:
    """PROVENANCE: profile sampling + peak-spacing logic ported from
    release/neural-tracing-audit/gap_verify.py -- specifically: sample CT
    intensity at 1-vox steps along the surface normal over t in [-90, 90],
    boxcar-smooth with a width-5 kernel, scipy.signal.find_peaks with
    (prominence, distance=5), spacing = diff of sorted peak positions,
    summarized by the pooled median. That logic is UNCHANGED. ADAPTED
    here: (1) normals come from the strip's stored per-wrap normals
    instead of the source PPM's normal channels; (2) sample cells are an
    evenly spaced deterministic selection over the middle wrap's points
    instead of the original's fixed 3x3 (row, col) grid; (3) the volume
    is opened via the `zarr` package exactly like the original, but the
    URL/path comes from --ct-url or strip meta instead of a hardcoded
    constant; (4) the pass criterion (ratio of CT spacing to strip pitch
    within [ct_ratio_min, ct_ratio_max]) is new -- the original printed
    numbers for a human to compare.

    NETWORK-REQUIRED unless ct_url is a local zarr path. Never run in the
    offline test suite.
    """
    try:
        import zarr  # optional dependency, deliberately not in core requirements
    except ImportError:
        return {
            "run": False,
            "pass": None,
            "reason": "optional dependency `zarr` not installed "
                      "(pip install zarr fsspec aiohttp for remote URLs)",
        }
    from scipy.signal import find_peaks  # scipy is a core dependency

    mid_wrap = strip.wrap_indices[len(strip.wrap_indices) // 2]
    if mid_wrap not in strip.normals:
        return {
            "run": False,
            "pass": None,
            "reason": f"strip has no stored normals for wrap {mid_wrap}; "
                      "the CT check samples along normals and cannot run",
        }

    voxel_um = float(strip.meta.get("voxel_size_um", float("nan")))
    pitch_um = float(strip.pitch_um.get("median", float("nan")))
    if not np.isfinite(voxel_um) or not np.isfinite(pitch_um) or voxel_um <= 0:
        return {
            "run": False,
            "pass": None,
            "reason": "strip meta lacks a finite voxel_size_um / pitch_um; "
                      "cannot convert CT voxel spacings for comparison",
        }

    root = zarr.open(ct_url, mode="r")
    try:
        vol = root["0"]
    except Exception:
        vol = root

    pts = np.asarray(strip.wraps[mid_wrap], dtype=np.float64)
    nrm = np.asarray(strip.normals[mid_wrap], dtype=np.float64)
    sel = np.linspace(0, pts.shape[0] - 1, num=min(n_cells, pts.shape[0]),
                      dtype=int)

    T = np.arange(-90.0, 90.5, 1.0)
    spacings = []
    for i in sel:
        p0, n = pts[i], nrm[i]
        line = p0[None, :] + T[:, None] * n[None, :]
        zi = np.clip(np.round(line[:, 2]).astype(int), 0, vol.shape[0] - 1)
        yi = np.clip(np.round(line[:, 1]).astype(int), 0, vol.shape[1] - 1)
        xi = np.clip(np.round(line[:, 0]).astype(int), 0, vol.shape[2] - 1)
        z0, z1 = zi.min(), zi.max() + 1
        y0, y1 = yi.min(), yi.max() + 1
        x0, x1 = xi.min(), xi.max() + 1
        blk = np.asarray(vol[z0:z1, y0:y1, x0:x1])
        prof = blk[zi - z0, yi - y0, xi - x0].astype(float)
        prof_s = np.convolve(prof, np.ones(5) / 5, mode="same")
        pk, _ = find_peaks(prof_s, prominence=prominence, distance=5)
        spacings.extend(np.diff(np.sort(T[pk])).tolist())

    if not spacings:
        return {
            "run": True,
            "pass": False,
            "reason": "no intensity peaks found along any sampled normal "
                      "(wrong volume? wrong prominence for this intensity "
                      "scale?)",
            "n_cells": int(len(sel)),
        }

    sp = np.asarray(spacings)
    ct_median_vox = float(np.median(sp))
    ct_median_um = ct_median_vox * voxel_um
    ratio = ct_median_um / pitch_um
    passed = config["ct_ratio_min"] <= ratio <= config["ct_ratio_max"]
    return {
        "run": True,
        "pass": bool(passed),
        "ct_median_spacing_vox": ct_median_vox,
        "ct_median_spacing_um": ct_median_um,
        "ct_p10_vox": float(np.percentile(sp, 10)),
        "ct_p90_vox": float(np.percentile(sp, 90)),
        "n_spacings": int(sp.size),
        "n_cells": int(len(sel)),
        "strip_pitch_um": pitch_um,
        "ratio_ct_over_pitch": ratio,
        "accepted_ratio_band": [config["ct_ratio_min"], config["ct_ratio_max"]],
    }


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def qualify(strip: Strip, strip_path: Optional[str] = None,
            ct_url: Optional[str] = None, config=CONFIG) -> Dict:
    """Run the suite on an in-memory Strip. Pure function of its inputs
    (given fixed config); does not touch the filesystem."""
    problems = validate_strip(strip)
    report = {
        "schema_version": strip.meta.get("schema_version", "unknown"),
        "strip_path": str(strip_path) if strip_path else None,
        "generated_by": "qualify_strip.py (reference-strips v0)",
        "config": dict(config),
        "validation": {"pass": not problems, "problems": problems},
        "checks": {},
        "overall_pass": False,
    }
    if problems:
        report["checks"]["skipped_reason"] = (
            "structural validation failed; geometric checks not run"
        )
        return report

    trees = build_trees(strip)
    report["checks"]["self_test"] = check_self_test(strip, trees, config)
    report["checks"]["wrong_side"] = check_wrong_side(strip, trees, config)
    report["checks"]["null_baseline"] = check_null_baseline(strip, trees, config)

    if ct_url:
        report["checks"]["ct_check"] = check_ct(strip, ct_url, config=config)
    else:
        report["checks"]["ct_check"] = {
            "run": False,
            "pass": None,
            "reason": "not requested (pass --ct-check; NETWORK-REQUIRED "
                      "unless the URL is a local zarr path)",
        }

    required = ["self_test", "wrong_side", "null_baseline"]
    overall = all(report["checks"][name]["pass"] for name in required)
    ct = report["checks"]["ct_check"]
    if ct.get("run"):
        # if the optional CT check was actually run, its verdict counts
        overall = overall and bool(ct.get("pass"))
    report["overall_pass"] = bool(overall)
    return report


def render_markdown(report: Dict) -> str:
    lines = []
    verdict = "QUALIFIED" if report["overall_pass"] else "UNQUALIFIED"
    lines.append(f"# Strip qualification report: **{verdict}**")
    lines.append("")
    if report.get("strip_path"):
        lines.append(f"Strip: `{report['strip_path']}`")
    lines.append(f"Schema: {report.get('schema_version')}")
    lines.append("")
    v = report["validation"]
    lines.append(f"## Structural validation: {'pass' if v['pass'] else 'FAIL'}")
    for p in v["problems"]:
        lines.append(f"- {p}")
    lines.append("")
    checks = report["checks"]
    if "self_test" in checks:
        c = checks["self_test"]
        lines.append(f"## (a) Self-test (plumbing): {'pass' if c['pass'] else 'FAIL'}")
        lines.append(f"- max distance to own wrap: {c['max_dist_any_wrap']:.6f} "
                     f"(limit {report['config']['self_test_max_dist']})")
        lines.append(f"- min % nearest own wrap: {c['min_pct_nearest_own_wrap']:.2f}")
        lines.append("")
    if "wrong_side" in checks:
        c = checks["wrong_side"]
        lines.append(f"## (b) Wrong-side separation: {'pass' if c['pass'] else 'FAIL'}")
        lines.append(f"- min fail% over adjacent pairs: "
                     f"{c['min_fail_pct_over_pairs']:.2f} "
                     f"(required >= {c['required_min_fail_pct']})")
        for pair, d in c["pairs"].items():
            lines.append(f"  - {pair}: fail {d['fail_pct']:.2f}% "
                         f"(median cross-dist {d['median_d_cross']:.3f}, "
                         f"median sampling {d['median_d_self']:.3f})")
        lines.append("")
    if "null_baseline" in checks:
        c = checks["null_baseline"]
        lines.append(f"## (c) Null baseline: {'pass' if c['pass'] else 'FAIL'}")
        lines.append(f"- min wrong-hop% over pairs: "
                     f"{c['min_wrong_hop_pct_over_pairs']:.2f} "
                     f"(required >= {c['required_min_wrong_pct']}) at "
                     f"{c['offset_gap_multiplier']}x local gap offset")
        lines.append("")
    ct = checks.get("ct_check", {})
    if ct.get("run"):
        lines.append(f"## (d) CT intensity cross-check: "
                     f"{'pass' if ct['pass'] else 'FAIL'}")
        lines.append(f"- CT median spacing: {ct.get('ct_median_spacing_um', float('nan')):.1f} um "
                     f"vs strip pitch {ct.get('strip_pitch_um', float('nan')):.1f} um "
                     f"(ratio {ct.get('ratio_ct_over_pitch', float('nan')):.2f}, "
                     f"accepted {ct.get('accepted_ratio_band')})")
    else:
        lines.append("## (d) CT intensity cross-check: not run")
        lines.append(f"- {ct.get('reason', 'not requested')}")
    lines.append("")
    lines.append("All thresholds in the `config` block of the JSON report "
                 "are PROVISIONAL (uncalibrated); see README.md.")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("strip", help="strip-v0 .npz file")
    parser.add_argument(
        "--ct-check", action="store_true",
        help="ALSO run the optional CT intensity cross-check "
             "(NETWORK-REQUIRED unless --ct-url points at a local zarr; "
             "needs the optional `zarr` package; never run in tests)",
    )
    parser.add_argument(
        "--ct-url", default=None,
        help="zarr URL or local path for --ct-check; defaults to the "
             "strip meta's `ct_volume_url` if present",
    )
    parser.add_argument("--out-json", default=None,
                        help="default: <strip>.qualification.json")
    parser.add_argument("--out-md", default=None,
                        help="default: <strip>.qualification.md")
    return parser.parse_args()


def main():
    args = parse_args()
    strip_path = Path(args.strip)
    strip = load_strip(strip_path)

    ct_url = None
    if args.ct_check:
        ct_url = args.ct_url or strip.meta.get("ct_volume_url")
        if not ct_url:
            print(
                "qualify_strip: --ct-check requested but no --ct-url given "
                "and strip meta has no ct_volume_url; skipping CT check",
                file=sys.stderr,
            )

    report = qualify(strip, strip_path=str(strip_path), ct_url=ct_url)

    out_json = Path(args.out_json) if args.out_json else strip_path.with_name(
        strip_path.stem + ".qualification.json"
    )
    out_md = Path(args.out_md) if args.out_md else strip_path.with_name(
        strip_path.stem + ".qualification.md"
    )
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    md = render_markdown(report)
    out_md.write_text(md)

    print(md)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    sys.exit(0 if report["overall_pass"] else 2)


if __name__ == "__main__":
    main()
