"""The pipeline-agnostic candidate contract (PLAN_V3.md, stb/contract.py).

A WindowTask packages exactly what a tracing method/adapter needs to
produce a Prediction for one window+direction, without exposing any of
stb.core.Reference's KD-tree/scoring internals to the adapter:

    scroll_id        cfg.scroll_id, informational only
    col_start        the window's column start (full-band numbering)
    seed_points      (n,3) float64, ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    normals          (n,3) float64, the band normal at each seed point
                     (a row is NaN where the band has no valid normal there)
    direction        +1 (front) or -1 (back): which winding neighbor to hop
                     toward (same convention as stb.arms/reference_src:
                     front hops by -normal, back by +normal)
    gap_hint_median  scalar vox: a representative KD-tree gap for this
                     direction, for adapters that need a plausible search
                     radius (e.g. mesh ray-casting bounds) without
                     importing stb.core/gates themselves

A Prediction is a bare (n,3) float ndarray, row-aligned with
WindowTask.seed_points/normals; a row with any non-finite coordinate is an
ABSTAIN for that cell -- the adapter declining to predict there.

score_candidate(ref, task, prediction, cfg) is the one piece of scoring
logic that doesn't already exist in stb.core, because
core.score_prediction requires an all-finite grid (abstention is a
contract-layer concept, not a core-scoring one): it excludes abstained
cells from BOTH the numerator and the scorable denominator, while still
reporting how many there were, and otherwise defers entirely to
core.score_prediction/summarize_score for the correct/wrong-wrap/
distance-miss classification.
"""
import dataclasses

import numpy as np

from . import core


@dataclasses.dataclass
class WindowTask:
    scroll_id: str
    col_start: int
    seed_points: np.ndarray
    normals: np.ndarray
    direction: int
    gap_hint_median: float
    grid_shape: tuple | None = None
    sample_rows: np.ndarray | None = None
    sample_cols: np.ndarray | None = None

    def __post_init__(self):
        if self.direction not in (+1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction!r}")
        seed = np.asarray(self.seed_points)
        normals = np.asarray(self.normals)
        if seed.ndim != 2 or seed.shape[1] != 3:
            raise ValueError(f"seed_points must be (n, 3), got {seed.shape}")
        if normals.shape != seed.shape:
            raise ValueError(
                f"normals shape {normals.shape} must match seed_points {seed.shape}"
            )
        if not np.isfinite(self.gap_hint_median) or self.gap_hint_median <= 0:
            raise ValueError("gap_hint_median must be finite and positive")
        finite_normals = np.isfinite(normals).all(axis=1)
        if finite_normals.any() and np.any(np.linalg.norm(normals[finite_normals], axis=1) <= 1e-9):
            raise ValueError("finite normals must have non-zero length")
        if self.grid_shape is not None:
            if len(self.grid_shape) != 2 or any(int(v) <= 0 for v in self.grid_shape):
                raise ValueError("grid_shape must be a positive (rows, cols) pair")
        if (self.sample_rows is None) != (self.sample_cols is None):
            raise ValueError("sample_rows and sample_cols must be supplied together")
        if self.sample_rows is not None:
            if len(self.sample_rows) != len(seed) or len(self.sample_cols) != len(seed):
                raise ValueError("sample indices must align with seed_points")


def task_for_window(ref, normals_band, col_start, direction, cfg, scroll_id=None):
    """Build the WindowTask for `ref` (from stb.reference.reference_at) in
    one direction. gap_hint_median is that direction's median KD-tree gap
    among ref's prediction-independent scorable cells -- the same
    quantity stb.gates.coverage_and_gates_ab reports as
    gap_median_front/back, recomputed directly via core.gaps_for so
    callers don't need to have already run the gates.
    """
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    normals = normals_band[ref.rr.ravel(), ref.cc.ravel() + col_start]
    gap, _idx, gap_edge = core.gaps_for(ref, direction)
    scorable = ~(gap_edge | (ref.seed_cls != 0))
    gap_hint = float(np.median(gap[scorable])) if scorable.any() else float("nan")
    return WindowTask(
        scroll_id=scroll_id if scroll_id is not None else getattr(cfg, "scroll_id", ""),
        col_start=int(col_start), seed_points=seed_pts, normals=normals,
        direction=int(direction), gap_hint_median=gap_hint,
        grid_shape=tuple(int(v) for v in ref.seed.shape[:2]),
        sample_rows=np.asarray(ref.rr.ravel(), dtype=np.int64),
        sample_cols=np.asarray(ref.cc.ravel(), dtype=np.int64),
    )


def score_candidate(ref, task, prediction, cfg):
    """Score a pipeline-agnostic Prediction against `ref` for
    task.direction.

    A row of `prediction` with any non-finite coordinate is an ABSTAIN:
    excluded from both the correct/wrong-hop numerators and the scorable
    denominator, counted separately. Every non-abstained row is scored
    exactly as core.score_prediction/summarize_score do (same
    correct/wrong_wrap/distance_miss classification against ref's
    expected winding class == task.direction), with the scorable ("ok")
    set additionally intersected with "not abstained".
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.shape != task.seed_points.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} != task.seed_points "
            f"{task.seed_points.shape}"
        )
    abstain = ~np.isfinite(prediction).all(axis=-1)
    n = len(abstain)
    n_abstain = int(abstain.sum())

    # core.score_prediction requires an all-finite grid; fill abstained
    # rows with the seed point itself (a finite placeholder -- its actual
    # classification is discarded below by ANDing every partition array
    # with `keep`).
    keep = ~abstain
    filled = np.where(abstain[:, None], task.seed_points, prediction)
    grid = ref.seed.copy()
    grid[ref.rr.ravel(), ref.cc.ravel()] = filled

    result = dict(core.score_prediction(ref, grid, task.direction, cfg))
    # correct/wrong_wrap/distance_miss partition "ok" exactly (score_prediction's
    # invariant); AND every one of the four by the same `keep` mask so that
    # invariant survives abstention -- only masking "ok" would leave the
    # other three still counting abstained cells, breaking summarize_score's
    # correct_pct + wrong_hop_pct == 100% identity.
    for key in ("ok", "correct", "wrong_wrap", "distance_miss"):
        result[key] = result[key] & keep
    result["excluded"] = ~result["ok"]  # keep included + excluded == n
    summary = core.summarize_score(result)
    summary["abstained"] = n_abstain
    summary["abstained_pct"] = 100.0 * n_abstain / n if n else float("nan")
    # V4 fixed-denominator metrics. `reference_eligible` is determined only
    # by the reference geometry, before model abstention, so selective models
    # cannot improve these yields by declining hard cells.
    reference_eligible_mask = ~core.score_prediction(
        ref, np.asarray(ref.seed, dtype=np.float64), task.direction, cfg
    )["excluded"]
    reference_eligible = int(reference_eligible_mask.sum())
    answered_eligible = int((reference_eligible_mask & keep).sum())
    pct_reference = lambda value: (
        100.0 * value / reference_eligible if reference_eligible else float("nan")
    )
    summary.update({
        "reference_eligible": reference_eligible,
        "answered_eligible": answered_eligible,
        "coverage_pct": pct_reference(answered_eligible),
        "correct_yield_pct": pct_reference(summary["correct"]),
        "wrong_wrap_yield_pct": pct_reference(summary["wrong_wrap"]),
        "distance_miss_yield_pct": pct_reference(summary["distance_miss"]),
        "total_error_yield_pct": pct_reference(
            summary["wrong_wrap"] + summary["distance_miss"]
        ),
        "valid_result": bool(reference_eligible and answered_eligible),
    })
    return summary


def coverage_gate(summary, minimum_coverage_pct=80.0):
    """Return an explicit V4 promotion decision for a candidate summary."""
    coverage = float(summary.get("coverage_pct", float("nan")))
    passed = bool(summary.get("valid_result", False) and np.isfinite(coverage)
                  and coverage >= minimum_coverage_pct)
    return {
        "pass": passed,
        "minimum_coverage_pct": float(minimum_coverage_pct),
        "coverage_pct": coverage,
        "reason": "pass" if passed else "insufficient fixed-denominator coverage",
    }
