"""Pinned regression numbers for stb.arms (PLAN_V3.md, "Pinned regression
numbers" (b) and (d)): reproduce fixtures/v2_scores_20260711.json's
per-window/direction/arm correct_pct -- injecting the *recorded* p2 (never
calling the CT estimator/volume) -- and the E1/G1 saturation summary.
Offline: only reads configs/pherc0332.json and
fixtures/{band_r1145_200_xyz.npz, out_v2/..., v2_scores_20260711.json}.
"""
import json
from pathlib import Path

import pytest

from stb import arms

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"

TOL_PP = 0.01
ARMS_CHECKED = ("A", "B_p2", "C_p2", "D_9vox", "null_perm")

# KNOWN, documented gap -- see BLOCKERS.md, "Agent B: stb.arms's frozen
# denominator can't be reproduced bit-exactly offline for every unit".
# fixtures/v2_scores_20260711.json's frozen_included was computed
# upstream using all SIX arms (A, B_p2, C_p2, B_p1, C_p1, D_9vox); offline,
# B_p1/C_p1 can't be reconstructed (their per-cell values need real CT
# profile sampling that isn't in any fixture -- only the reduced
# p2/p2_cell_valid_frac scalars are recorded), so stb.arms.score_window's
# frozen set here is the intersection of the other four. That's a strict
# superset (or equal) of the true frozen set; empirically it is
# byte-for-byte identical in 6 of 8 window/direction units, differs by 1
# cell of 5306 in a 7th (still within 0.01pp on every arm there), and by 2
# cells of 4841 in the 8th -- invisible on every arm EXCEPT this one exact
# (window, direction, arm) combination. See test_known_gap_* below.
KNOWN_GAP = {(13400, "front", "null_perm")}


@pytest.fixture(scope="module")
def fixture_scores():
    with open(FIXTURES / "v2_scores_20260711.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scored_windows(resolved_cfg_332, band_332, normals_332, fixture_scores):
    """score_window for every window in the fixture, injecting that
    window's recorded p2/p2_cell_valid_frac (no volume, no network)."""
    xyz, valid, _row0 = band_332
    normals_band, _n_ok = normals_332
    out_root = FIXTURES / "out_v2"
    per_window = []
    for w in fixture_scores["windows"]:
        p2 = w["p2"] if w["p2"] is not None else float("nan")
        p2_estimate = {"p2": p2, "cell_valid_frac": w["p2_cell_valid_frac"]}
        result = arms.score_window(xyz, valid, normals_band, None, w["start"],
                                    out_root, resolved_cfg_332, rng_seed=0,
                                    p2_estimate=p2_estimate)
        per_window.append(result)
    return per_window


def test_factor_arm_a_matches_published(resolved_cfg_332, fixture_scores):
    # v2_score.py: FACTOR = 4.8 / 7.91 (train_um / vox_um).
    assert arms.factor_arm_a(resolved_cfg_332) == pytest.approx(
        fixture_scores["factor_armA"], abs=1e-12
    )


def test_arms_reproduce_v2_scores_per_window_direction_arm(scored_windows, fixture_scores):
    mismatches = []
    for got_w, want_w in zip(scored_windows, fixture_scores["windows"]):
        assert got_w["start"] == want_w["start"]
        for name in ("front", "back"):
            got_d = got_w["directions"][name]
            want_d = want_w["directions"][name]
            for arm in ARMS_CHECKED:
                if arm not in want_d:
                    continue
                if (got_w["start"], name, arm) in KNOWN_GAP:
                    continue
                got_pct = got_d[arm]["correct_pct"]
                want_pct = want_d[arm]["correct_pct"]
                diff = abs(got_pct - want_pct)
                if diff > TOL_PP:
                    mismatches.append(
                        f"start={got_w['start']} {name:5s} {arm:10s} "
                        f"got={got_pct:.6f} want={want_pct:.6f} diff={diff:.6f}"
                    )
    assert not mismatches, "mismatches vs fixtures/v2_scores_20260711.json:\n" + "\n".join(
        mismatches
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN, documented gap (BLOCKERS.md, Agent B): frozen_included at "
        "s13400 front was computed upstream using arms B_p1/C_p1 too, "
        "whose exact per-cell values need real CT profile data this "
        "repo's offline fixtures don't carry. Our 4-arm frozen set is a "
        "strict superset by 2 of ~4841 cells there, shifting null_perm's "
        "correct_pct by 0.0108pp -- just over the 0.01pp bar. All other "
        "39 of 40 (window, direction, arm) combinations checked by "
        "test_arms_reproduce_v2_scores_per_window_direction_arm match "
        "exactly or within tolerance. strict=True: if this ever starts "
        "passing (e.g. once real per-cell CT data becomes available "
        "offline), that's a signal to update this test and BLOCKERS.md,"
        " not a silent fix."
    ),
)
def test_known_gap_s13400_front_null_perm_frozen_denominator(scored_windows, fixture_scores):
    got_w = next(w for w in scored_windows if w["start"] == 13400)
    want_w = next(w for w in fixture_scores["windows"] if w["start"] == 13400)
    got = got_w["directions"]["front"]["null_perm"]["correct_pct"]
    want = want_w["directions"]["front"]["null_perm"]["correct_pct"]
    assert abs(got - want) <= TOL_PP


def test_e1_summary_matches_published(scored_windows, fixture_scores):
    summary = arms.summarize_run(scored_windows)
    want = fixture_scores["summary"]

    assert abs(summary["E1_mean_B_minus_C_pp"] - (-55.2038)) < 0.01
    assert abs(summary["E1_mean_B_minus_C_pp"] - want["E1_mean_B_minus_C_pp"]) < 0.01
    # same discriminating-unit classification as the published run
    assert summary["discriminating_units"] == want["discriminating_units"] == 8
    assert summary["saturated_units"] == want["saturated_units"] == []
    assert summary["B_gt_C_count"] == want["B_gt_C_count"] == 1
    assert summary["G1_directional_skill"] == want["G1_directional_skill"] is False
