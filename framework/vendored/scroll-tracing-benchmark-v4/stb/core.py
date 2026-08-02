"""Local-reference scoring core (port of reference_src/benchmark_core.py):
the Reference container, per-cell nearest-wrap classification and summary
percentages.

VOX_UM plays no role in this arithmetic (it is a unit label carried on
ScrollConfig for consumers outside this module, e.g. arm-A's vox_um-ratio
scaling); CLASSES and the scoring threshold DO drive the computation below
and are read off the ScrollConfig passed to score_prediction instead of
benchmark_core's module constants, so the same code scores any scroll.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Reference:
    xyz: np.ndarray
    valid: np.ndarray
    row0: int
    seed: np.ndarray
    cls: np.ndarray
    trees: dict
    rows_of: dict
    cols_of: dict
    pts_of: dict
    rr: np.ndarray
    cc: np.ndarray
    seed_cls: np.ndarray


def gaps_for(ref, expected_class):
    """Distance from every seed cell to its own expected-wrap tree, plus
    which of those nearest points sit on the band's row/col edge."""
    if expected_class not in ref.trees:
        raise ValueError(f"reference has no populated winding class {expected_class}")
    seed_points = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    dist, idx = ref.trees[expected_class].query(seed_points, workers=-1)
    edge = np.isin(
        ref.rows_of[expected_class][idx], (0, ref.xyz.shape[0] - 1)
    ) | np.isin(ref.cols_of[expected_class][idx], (0, ref.xyz.shape[1] - 1))
    return dist, idx, edge


def score_prediction(ref, pred_grid, expected_class, cfg):
    """Classify every seed cell of pred_grid as correct / wrong-wrap /
    distance-miss / excluded against ref.

    Port of reference_src/benchmark_core.score_prediction; CLASSES and the
    threshold_kind/threshold_value pair come from cfg (a ScrollConfig)
    instead of module globals.
    """
    pred_grid = np.asarray(pred_grid, dtype=np.float64)
    if pred_grid.shape != ref.seed.shape:
        raise ValueError(
            f"prediction shape {pred_grid.shape} does not match seed {ref.seed.shape}"
        )
    if not np.isfinite(pred_grid).all():
        raise ValueError("prediction grid contains non-finite coordinates")
    if float(cfg.threshold_value) <= 0:
        raise ValueError("threshold value must be positive")
    pred_points = pred_grid[ref.rr.ravel(), ref.cc.ravel()]
    gap, gap_idx, gap_edge = gaps_for(ref, expected_class)
    d_exp, idx_exp = ref.trees[expected_class].query(pred_points, workers=-1)
    exp_edge = np.isin(
        ref.rows_of[expected_class][idx_exp], (0, ref.xyz.shape[0] - 1)
    ) | np.isin(
        ref.cols_of[expected_class][idx_exp], (0, ref.xyz.shape[1] - 1)
    )

    dists = np.full((pred_points.shape[0], len(cfg.classes)), np.inf)
    for j, n in enumerate(cfg.classes):
        if n in ref.trees:
            dists[:, j], _ = ref.trees[n].query(pred_points, workers=-1)
    nearest_cls = np.asarray(cfg.classes)[np.argmin(dists, axis=1)]

    if cfg.threshold_kind == "gap_fraction":
        threshold = cfg.threshold_value * gap
    elif cfg.threshold_kind == "fixed_vox":
        threshold = np.full_like(gap, float(cfg.threshold_value))
    else:
        raise ValueError(f"unknown threshold kind: {cfg.threshold_kind}")

    excluded = exp_edge | gap_edge | (ref.seed_cls != 0)
    ok = ~excluded
    wrong_wrap = ok & (nearest_cls != expected_class)
    distance_miss = ok & (nearest_cls == expected_class) & (d_exp >= threshold)
    correct = ok & (nearest_cls == expected_class) & (d_exp < threshold)

    status = np.full(pred_points.shape[0], 3, dtype=np.int8)
    status[wrong_wrap] = 1
    status[distance_miss] = 2
    status[correct] = 0

    return {
        "status": status,
        "ok": ok,
        "excluded": excluded,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct": correct,
        "gap": gap,
        "gap_idx": gap_idx,
        "d_exp": d_exp,
        "idx_exp": idx_exp,
        "nearest_cls": nearest_cls,
        "wrap_index_error": nearest_cls - expected_class,
        "threshold": threshold,
        "ratio": d_exp / np.maximum(gap, 1e-9),
        "pred_points": pred_points,
        "expected_class": expected_class,
    }


def summarize_score(result):
    """Included/excluded counts and correct / wrong-hop percentages."""
    included = int(result["ok"].sum())
    excluded = int(result["excluded"].sum())
    correct = int(result["correct"].sum())
    wrong_wrap = int(result["wrong_wrap"].sum())
    distance_miss = int(result["distance_miss"].sum())
    wrong_hop = wrong_wrap + distance_miss

    def pct(value):
        return 100.0 * value / included if included else float("nan")

    return {
        "included": included,
        "excluded": excluded,
        "correct": correct,
        "wrong_hop": wrong_hop,
        "wrong_wrap": wrong_wrap,
        "distance_miss": distance_miss,
        "correct_pct": pct(correct),
        "wrong_hop_pct": pct(wrong_hop),
        "wrong_wrap_pct": pct(wrong_wrap),
        "distance_miss_pct": pct(distance_miss),
    }
