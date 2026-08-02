"""Pinned regression numbers for PHerc0332 (PLAN_V3.md, "Pinned regression
numbers"), parts (a) selection and (c) center fit. Offline: only reads
configs/pherc0332.json and fixtures/{band_r1145_200_xyz.npz,windows_v2.json}.

Part (a) drives stb.selection over the *entire* band (217 candidate
windows, each building up to 7 KD-trees over a ~5.1M-point band) and is
therefore slow (~10-15 minutes); it is marked accordingly but must still be
run to completion at least once per change to the port.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from stb import band as stb_band
from stb import config as stb_config
from stb import normals as stb_normals
from stb import selection as stb_selection

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"
CONFIGS = REPO_ROOT / "configs"

HARDCODED_CENTER = (1809.26609333, 1732.01937758)


def _nan_aware_equal(got, want, abs_tol=None):
    got_nan = got is None or (isinstance(got, float) and np.isnan(got))
    want_nan = want is None or (isinstance(want, float) and np.isnan(want))
    if got_nan or want_nan:
        return got_nan and want_nan
    if abs_tol is None:
        return got == want
    return abs(got - want) <= abs_tol


@pytest.mark.slow
def test_all_candidates_match_windows_v2():
    cfg = stb_config.load_config(CONFIGS / "pherc0332.json")
    xyz, valid, _row0 = stb_band.load_band(cfg.band_path)
    cfg = stb_config.resolve(cfg, xyz, valid)

    normals, n_ok = stb_normals.band_normals(xyz, valid)
    kappa = stb_normals.kappa_per_column(normals, n_ok)
    rows = stb_selection.scan_candidates(xyz, valid, cfg, kappa)
    got = stb_selection.all_candidates_rows(rows)

    with open(FIXTURES / "windows_v2.json") as f:
        want = json.load(f)["all_candidates"]

    assert len(got) == len(want), f"candidate count differs: {len(got)} vs {len(want)}"

    mismatches = []
    for i, (g, w) in enumerate(zip(got, want)):
        if g["start"] != w["start"]:
            mismatches.append(f"[{i}] start {g['start']} != {w['start']}")
            continue
        if g["gate_a"] != w["gate_a"]:
            mismatches.append(f"[start={g['start']}] gate_a {g['gate_a']!r} != {w['gate_a']!r}")
        if g["gate_b"] != w["gate_b"]:
            mismatches.append(f"[start={g['start']}] gate_b {g['gate_b']!r} != {w['gate_b']!r}")
        if not _nan_aware_equal(g["coverage"], w["coverage"], abs_tol=1e-6):
            mismatches.append(f"[start={g['start']}] coverage {g['coverage']!r} != {w['coverage']!r}")
        if not _nan_aware_equal(g["kappa"], w["kappa"], abs_tol=1e-6):
            mismatches.append(f"[start={g['start']}] kappa {g['kappa']!r} != {w['kappa']!r}")

    assert not mismatches, "mismatches vs fixtures/windows_v2.json all_candidates:\n" + "\n".join(
        mismatches[:20]
    ) + (f"\n... and {len(mismatches) - 20} more" if len(mismatches) > 20 else "")


def test_fit_center_is_functionally_adequate_for_classing():
    """The lead replaced the original 2-vox geometric pin (see BLOCKERS.md:
    a band row is a ~3-revolution spiral, so no circle fit can recover the
    historical center constant). What actually matters is FUNCTIONAL: a
    fitted center must still produce a working winding reference. Pin: with
    fit_center's center instead of the literal one, window 11000 must still
    pass gates a/b and its coverage must be within 0.05 of the literal-center
    value."""
    import dataclasses
    from stb.band import load_band, fit_center
    from stb.config import load_config
    from stb.reference import reference_at
    from stb.gates import coverage_and_gates_ab

    cfg = load_config("configs/pherc0332.json")
    xyz, valid, _ = load_band(cfg.band_path)
    ref_lit = reference_at(xyz, valid, 11000, cfg)
    g_lit = coverage_and_gates_ab(ref_lit, cfg)

    cx, cy = fit_center(xyz, valid, row=100)
    cfg_fit = dataclasses.replace(cfg, center=(float(cx), float(cy)))
    ref_fit = reference_at(xyz, valid, 11000, cfg_fit)
    g_fit = coverage_and_gates_ab(ref_fit, cfg_fit)

    assert g_fit["gate_a_pass"] and g_fit["gate_b_pass"], g_fit
    assert abs(g_fit["coverage"] - g_lit["coverage"]) < 0.05, (
        g_fit["coverage"], g_lit["coverage"])
