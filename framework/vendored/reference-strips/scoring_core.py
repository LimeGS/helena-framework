"""Shared KD-tree scoring primitives for strip-v0 reference strips.

PROVENANCE: the hop-correct scoring rule below -- nearest-surface distance
to a wrap via a KD-tree, threshold = 0.5x the local gap to the target wrap,
decomposed into wrong-wrap-identity vs correct-wrap-but-too-far -- is ported
from:

  release/neural-tracing-audit/gate_3090/score_native.py (the `score`
  function: `wrongwrap`, `toofar`, `correct`, `excluded`, threshold
  `0.5 * gap`)
  release/neural-tracing-audit/benchmark_core.py (`score_prediction`,
  `summarize_score` -- the same rule refactored into reusable functions)

The core decision logic (excluded / wrong_wrap / distance_miss / correct,
threshold = 0.5 * local gap) is UNCHANGED from that source.

What is ADAPTED for strip-v0's simpler, grid-agnostic, sequentially
numbered wraps (this repo's own design choice, not a change carried over
from the source):

- The original scored a fixed *grid* of seed cells against per-winding-class
  KD-trees, using each seed cell's own (row, col) grid position to look up
  its own local gap directly. strip-v0 wraps are point sets with no
  preserved row/col correspondence to predictions (a tracer or mesher may
  not even share the seed's grid), so "the seed point behind this
  prediction" is redefined as *the nearest point on the from-wrap* to that
  prediction, and its local gap is that from-wrap point's own
  nearest-surface distance to the target wrap. This is a point-cloud-native
  generalization of the same rule, not a different rule: for a prediction
  that starts exactly at a from-wrap grid point (the common case), it
  reduces to the original definition exactly.
- The original's "excluded" rule was "the nearest reference point used for
  this cell lies on the band's first/last row" -- a grid-edge proxy for "the
  true nearest wrap surface might lie just outside the bundled reference".
  strip-v0 point sets have no row/col edges, so exclusion is redefined
  directly in terms of what that rule was a proxy for: a prediction farther
  than a coverage radius from EVERY wrap in the strip has no reliable
  nearby reference and is excluded rather than scored.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

# PROVISIONAL: half-gap fraction for the hop-correct threshold. Matches the
# "headline threshold = 0.5 * local_gap" convention documented in
# release/neural-tracing-audit/docs/BENCHMARK.md section 5.
DEFAULT_GAP_FRACTION = 0.5

# PROVISIONAL: default coverage-radius multiplier (x the median local gap
# observed in this call). A prediction farther than this from every wrap in
# the strip is excluded rather than scored -- not calibrated, chosen to be
# clearly larger than one gap so genuine near-misses are still scored while
# predictions that landed nowhere near any reference are not.
DEFAULT_COVERAGE_RADIUS_GAP_MULTIPLIER = 3.0

# PROVISIONAL: null-baseline offset for qualify_strip.py's check (c),
# expressed as a multiple of the local FULL gap (not the half-gap threshold
# value). A naive 1x-gap offset aimed at the target wrap would land ~on top
# of it in a strip with regular pitch (distance ~0, i.e. it would PASS,
# defeating the point of a null control). 2x the full gap overshoots to
# roughly "two wraps away" regardless of offset-direction precision, which
# is what makes it a robust null even for a strip with near-uniform pitch.
NULL_OFFSET_GAP_MULTIPLIER = 2.0


def build_trees(strip) -> Dict[int, cKDTree]:
    """One cKDTree per wrap, keyed by wrap index."""
    return {
        idx: cKDTree(np.asarray(pts, dtype=np.float64))
        for idx, pts in strip.wraps.items()
    }


def nearest_neighbor_distances(points_a, points_b):
    """Distance + index into `points_b` of its nearest point, for each point
    in `points_a`. The one KD-tree primitive everything else (pitch
    estimation in make_strip.py, every check in qualify_strip.py and
    score_strip.py) is built from."""
    points_a = np.asarray(points_a, dtype=np.float64)
    tree = cKDTree(np.asarray(points_b, dtype=np.float64))
    dist, idx = tree.query(points_a, workers=-1)
    return dist, idx


def nearest_wrap(points, trees: Dict[int, cKDTree]):
    """For each point, the wrap index it is closest to and that distance,
    scanning every wrap's tree. This is the "which wrap does this
    prediction actually land nearest" test used for wrong-wrap detection."""
    points = np.asarray(points, dtype=np.float64)
    wrap_ids = sorted(trees.keys())
    if not wrap_ids:
        raise ValueError("no wraps to compare against (empty trees dict)")
    dists = np.full((points.shape[0], len(wrap_ids)), np.inf)
    for j, wid in enumerate(wrap_ids):
        d, _ = trees[wid].query(points, workers=-1)
        dists[:, j] = d
    best = np.argmin(dists, axis=1)
    nearest_wrap_id = np.asarray(wrap_ids)[best]
    nearest_dist = dists[np.arange(points.shape[0]), best]
    return nearest_wrap_id, nearest_dist


def score_points(
    strip,
    trees: Dict[int, cKDTree],
    pred_points,
    from_wrap: int,
    direction: str,
    gap_fraction: float = DEFAULT_GAP_FRACTION,
    coverage_radius: Optional[float] = None,
    coverage_radius_gap_multiplier: float = DEFAULT_COVERAGE_RADIUS_GAP_MULTIPLIER,
):
    """Score predicted points against the strip's target wrap.

    direction: "front" -> target_wrap = from_wrap + 1
               "back"  -> target_wrap = from_wrap - 1

    Returns a dict of per-point arrays (status, ok, excluded, wrong_wrap,
    distance_miss, correct, gap, d_target, nearest_wrap_id,
    nearest_overall_dist, threshold) plus the scalars used
    (from_wrap, target_wrap, coverage_radius).

    status codes: 0 = hop-correct, 1 = wrong adjacent-wrap identity,
    2 = correct wrap but past the distance threshold, 3 = excluded.
    """
    if direction not in ("front", "back"):
        raise ValueError(f"direction must be 'front' or 'back', got {direction!r}")
    target_wrap = from_wrap + 1 if direction == "front" else from_wrap - 1

    if from_wrap not in trees:
        raise ValueError(f"from_wrap {from_wrap} is not a wrap in this strip")
    if target_wrap not in trees:
        raise ValueError(
            f"direction={direction!r} from wrap {from_wrap} needs wrap "
            f"{target_wrap}, which is not in this strip (available: "
            f"{sorted(trees.keys())})"
        )

    pred_points = np.asarray(pred_points, dtype=np.float64)
    n = pred_points.shape[0]

    # Proxy seed point on the from-wrap for each prediction, and that
    # point's own local gap to the target wrap (see module docstring).
    _, idx_from = trees[from_wrap].query(pred_points, workers=-1)
    seed_proxy = strip.wraps[from_wrap][idx_from]
    gap, gap_idx = trees[target_wrap].query(seed_proxy, workers=-1)

    d_target, tgt_idx = trees[target_wrap].query(pred_points, workers=-1)
    nearest_wrap_id, nearest_overall_dist = nearest_wrap(pred_points, trees)

    finite_gap = gap[np.isfinite(gap)]
    if coverage_radius is None:
        base = float(np.median(finite_gap)) if finite_gap.size else 0.0
        coverage_radius = coverage_radius_gap_multiplier * base

    # Coverage-boundary exclusion, ported from the source audit's rule
    # "excluded if the nearest expected-wrap point used for the prediction
    # distance OR for the local gap sits on the band's first/last row"
    # (gate_3090/score_native.py: `exp_edge | gap_edge`). strip-v0 stores
    # the equivalent information as per-wrap boolean edge flags (points
    # near a wrap's angular-coverage boundary); strips without edge flags
    # simply skip this component of the exclusion.
    target_edges = getattr(strip, "edges", {}).get(target_wrap)
    if target_edges is not None:
        gap_edge = target_edges[gap_idx]
        exp_edge = target_edges[tgt_idx]
    else:
        gap_edge = np.zeros(n, dtype=bool)
        exp_edge = np.zeros(n, dtype=bool)

    excluded = (nearest_overall_dist > coverage_radius) | gap_edge | exp_edge
    ok = ~excluded

    threshold = gap_fraction * gap
    wrong_wrap = ok & (nearest_wrap_id != target_wrap)
    distance_miss = ok & (nearest_wrap_id == target_wrap) & (d_target >= threshold)
    correct = ok & (nearest_wrap_id == target_wrap) & (d_target < threshold)

    status = np.full(n, 3, dtype=np.int8)
    status[wrong_wrap] = 1
    status[distance_miss] = 2
    status[correct] = 0

    return {
        "status": status,
        "ok": ok,
        "excluded": excluded,
        "excluded_no_coverage": nearest_overall_dist > coverage_radius,
        "excluded_edge": gap_edge | exp_edge,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct": correct,
        "gap": gap,
        "d_target": d_target,
        "nearest_wrap_id": nearest_wrap_id,
        "nearest_overall_dist": nearest_overall_dist,
        "threshold": threshold,
        "coverage_radius": coverage_radius,
        "from_wrap": from_wrap,
        "target_wrap": target_wrap,
        "pred_points": pred_points,
    }


def summarize(result) -> Dict:
    """Counts + percentages, matching the numerator/denominator convention
    from the source audit: percentages are of the *included* (non-excluded)
    count, not of the total."""
    included = int(result["ok"].sum())
    excluded = int(result["excluded"].sum())
    correct = int(result["correct"].sum())
    wrong_wrap = int(result["wrong_wrap"].sum())
    distance_miss = int(result["distance_miss"].sum())
    wrong_hop = wrong_wrap + distance_miss
    denom = max(included, 1)
    return {
        "total": int(result["status"].shape[0]),
        "included": included,
        "excluded": excluded,
        "excluded_no_coverage": int(result.get(
            "excluded_no_coverage", np.zeros(0, dtype=bool)).sum()),
        "excluded_edge": int(result.get(
            "excluded_edge", np.zeros(0, dtype=bool)).sum()),
        "correct": correct,
        "wrong_hop": wrong_hop,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct_pct": 100.0 * correct / denom,
        "wrong_hop_pct": 100.0 * wrong_hop / denom,
        "wrong_wrap_pct": 100.0 * wrong_wrap / denom,
        "distance_miss_pct": 100.0 * distance_miss / denom,
    }
