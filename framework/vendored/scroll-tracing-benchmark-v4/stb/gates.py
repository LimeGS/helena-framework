"""Prediction-free window gates.

coverage_and_gates_ab is a straight port of
reference_src/v2_pipeline.coverage_and_gates_ab (cfg supplies CLASSES/
threshold to the score_prediction call inside gate b's wrong-side check).

gate_c is the CT-estimator-vs-KD-tree pitch-agreement check referenced by
PLAN_V3.md and pinned by fixtures/windows_v2.json's `windows` entries
(p2_ct, kd_gap_median, gate_c_ratio, gate_c_pass): ratio = ct_spacing /
kd_gap, must fall in [0.60, 1.60]. It is not exercised by score_prediction
or coverage_and_gates_ab, since it needs a CT pitch estimate (estimator.py)
that selection.py's replacement loop supplies from injected inputs.
"""
import numpy as np

from . import core


def coverage_and_gates_ab(ref, cfg):
    """Prediction-free coverage + gates (a) self-test and (b) wrong-side.

    coverage: fraction of sampled cells scorable for BOTH directions under
    the exclusion rules, using the prediction-independent parts (gap_edge
    and seed_cls != 0; exp_edge is prediction-dependent and, for on-target
    predictions, coincides with gap_edge).
    gate a: distance from each scorable seed point to its own class-0 tree
    (median must be 0.000, p90 <= 0.5).
    gate b: the oracle target grid (each cell moved to its nearest
    expected-wrap point) scored against the OPPOSITE class must be
    <= 5% correct, in both directions.
    """
    out = {}
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    scorable = {}
    for e in (+1, -1):
        if e not in ref.trees:
            return {"coverage": 0.0, "error": f"class {e} unpopulated"}
        gap, idx, gap_edge = core.gaps_for(ref, e)
        scorable[e] = ~(gap_edge | (ref.seed_cls != 0))
        out[f"gap_median_{'front' if e == 1 else 'back'}"] = (
            float(np.median(gap[scorable[e]])) if scorable[e].any() else float("nan"))
    both = scorable[+1] & scorable[-1]
    out["coverage"] = float(both.mean())
    if not both.any():
        out["gate_a_pass"] = out["gate_b_pass"] = False
        return out

    d0, _ = ref.trees[0].query(seed_pts[both], workers=-1)
    out["selftest_median"] = float(np.median(d0))
    out["selftest_p90"] = float(np.percentile(d0, 90))
    out["gate_a_pass"] = bool(out["selftest_median"] == 0.0 and out["selftest_p90"] <= 0.5)

    wrong = {}
    for e in (+1, -1):
        gap, idx, _ = core.gaps_for(ref, e)
        oracle = seed_pts.copy()
        oracle[:] = ref.pts_of[e][idx]
        grid = ref.seed.copy()
        grid[ref.rr.ravel(), ref.cc.ravel()] = oracle
        res = core.score_prediction(ref, grid, -e, cfg)
        s = core.summarize_score(res)
        wrong[e] = s["correct_pct"]
    out["wrongside_front_correct_pct"] = float(wrong[+1])
    out["wrongside_back_correct_pct"] = float(wrong[-1])
    out["gate_b_pass"] = bool(wrong[+1] <= 5.0 and wrong[-1] <= 5.0)
    return out


GATE_C_LO, GATE_C_HI = 0.60, 1.60


def gate_c(ct_spacing, kd_gap):
    """Pitch-agreement gate: the CT-measured spacing (estimator P2) must
    fall within [GATE_C_LO, GATE_C_HI] x the KD-tree geometric gap
    (ratio = ct_spacing / kd_gap). Returns the same keys as the v2 run
    recorded in fixtures/windows_v2.json."""
    ratio = float(ct_spacing) / float(kd_gap)
    return {
        "p2_ct": float(ct_spacing),
        "kd_gap_median": float(kd_gap),
        "gate_c_ratio": ratio,
        "gate_c_pass": bool(GATE_C_LO <= ratio <= GATE_C_HI),
    }
