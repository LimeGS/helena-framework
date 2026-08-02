"""Offline unit tests for stb/estimator.py: spacing_from_profile's peak
detection and estimator_p1's per-cell fallback logic. No volume/network
access; all inputs are synthetic arrays."""
import numpy as np
import pytest

from stb import estimator


def test_spacing_from_profile_detects_10vox_pitch():
    # A clean 10-vox-period oscillation over PROFILE_T (step 1.0 vox, so a
    # 10-sample period == a 10-vox peak spacing); peaks land exactly on
    # sample indices, so the recovered spacing should be dead-on 10.0.
    raw = np.cos(2 * np.pi * estimator.PROFILE_T / 10.0)
    spacing = estimator.spacing_from_profile(raw, estimator.TUNED_SIGMA, estimator.TUNED_PROM)
    assert abs(spacing - 10.0) < 0.2


def test_spacing_from_profile_abstains_below_two_peaks():
    # A single bump has at most one peak -> abstain (NaN), regardless of
    # smoothing/prominence.
    raw = np.exp(-(estimator.PROFILE_T ** 2) / (2 * 5.0 ** 2))
    spacing = estimator.spacing_from_profile(raw, estimator.TUNED_SIGMA, estimator.TUNED_PROM)
    assert np.isnan(spacing)

    # Flat profile: no peaks at all.
    flat = np.zeros_like(estimator.PROFILE_T)
    assert np.isnan(estimator.spacing_from_profile(flat, estimator.TUNED_SIGMA, estimator.TUNED_PROM))


def test_profile_t_matches_frozen_module_constant_at_40():
    # docs/PHERC1667_INSTANCE.md (c): profile_t(halfwidth_vox) generalizes
    # the old bare module constant PROFILE_T (+/-40 vox, PHerc0332's
    # frozen value); profile_t(40) must be bit-identical to it so every
    # existing call site that doesn't pass profile_halfwidth_vox keeps
    # PHerc0332's exact behavior.
    assert np.array_equal(estimator.profile_t(40), estimator.PROFILE_T)


def test_halfwidth_vox_from_physical_matches_lead_decisions():
    # docs/PHERC1667_INSTANCE.md (c): physical half-range fixed at 316um.
    # PHerc0332 (vox_um=7.91) must recover its own historical 40-vox
    # default exactly; PHerc1667 (vox_um=2.399) is this instance's own
    # pinned number (132 vox), so it can't silently drift.
    assert estimator.halfwidth_vox_from_physical(7.91) == 40
    assert estimator.halfwidth_vox_from_physical(2.399) == 132


def test_estimator_p1_fallback_fraction_on_crafted_grid():
    # 9x9 grid of spacings: columns 0-1 hold a valid constant spacing,
    # columns 2-8 are NaN. estimator_p1's neighborhood is a 5x5 block
    # (S[i-2:i+3, j-2:j+3]), so whether a cell falls back to P2 depends on
    # how many of columns 0-1 its block still overlaps:
    #   j=0,1,2  -> block always includes both of columns 0,1
    #               (>=3 rows x 2 cols = >=6 valid, never <5) -> no fallback
    #   j=3      -> block includes only column 1; valid count = rows_in_block
    #               (3,4,5,5,5,5,5,4,3 for i=0..8) -> falls back exactly
    #               where rows_in_block < 5, i.e. i in {0,1,7,8} (4 cells)
    #   j=4..8   -> block excludes columns 0,1 entirely -> always falls back
    #               (5 columns x 9 rows = 45 cells)
    # Expected fallback count = 4 + 45 = 49 of 81 cells.
    n = 9
    S = np.full((n, n), np.nan)
    S[:, 0:2] = 7.0
    spacings = S.ravel()
    rr = np.arange(n * n)
    cc = np.arange(n * n)
    p2 = 42.0

    p1, fallback_frac = estimator.estimator_p1(spacings, rr, cc, p2)

    assert fallback_frac == pytest.approx(49 / 81)
    p1_grid = p1.reshape(n, n)
    # Non-fallback cells (all of columns 0,1,2) recover the constant 7.0.
    assert np.all(p1_grid[:, 0:3] == 7.0)
    # Fallback cells take the injected P2 value verbatim.
    fallback_mask = np.ones((n, n), dtype=bool)
    fallback_mask[:, 0:3] = False
    fallback_mask[2:7, 3] = False  # the 5 non-fallback cells in column 3
    assert np.all(p1_grid[fallback_mask] == p2)
