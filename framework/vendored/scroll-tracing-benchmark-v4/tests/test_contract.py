"""stb.contract + adapters/* (PLAN_V3.md, "Pinned regression numbers",
test_contract.py):

- the tifxyz adapter re-scores s11000 front IDENTICALLY to scoring the
  same raw prediction grid directly with stb.core (0 abstains, so the
  contract layer must be a lossless pass-through of core's own
  correct/wrong-hop classification);
- the mesh adapter, ray-cast against a synthetic oracle mesh built from
  the band's own true class +1 points near s11000, scores >= 95% correct
  front;
- abstain accounting: clipping that oracle mesh to half its column range
  roughly doubles the abstain rate and shrinks the scored denominator
  accordingly.

Offline: only reads configs/pherc0332.json, fixtures/band_r1145_200_xyz.npz
and fixtures/out_v2/seed_v2_s11000_front.
"""
from pathlib import Path

import pytest

from stb import contract, core, reference as reference_mod, strips as strips_mod
from adapters import mesh_adapter, tifxyz_tracer

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "fixtures"

START = 11000
MESH_MARGIN = 300


@pytest.fixture(scope="module")
def ref_s11000(resolved_cfg_332, band_332):
    xyz, valid, _row0 = band_332
    return reference_mod.reference_at(xyz, valid, START, resolved_cfg_332)


def test_tifxyz_adapter_matches_direct_score(ref_s11000, normals_332, resolved_cfg_332):
    ref, cfg = ref_s11000, resolved_cfg_332
    normals_band, _n_ok = normals_332
    dirpath = tifxyz_tracer.prediction_dir(FIXTURES / "out_v2", START, "front")

    # direct path: raw tifxyz grid scored straight through stb.core
    raw_grid = tifxyz_tracer.load_tifxyz(dirpath)
    direct = core.summarize_score(core.score_prediction(ref, raw_grid, +1, cfg))

    # contract path: build the pipeline-agnostic task, run the tifxyz
    # adapter to get a Prediction, score it through stb.contract
    task = contract.task_for_window(ref, normals_band, START, +1, cfg)
    prediction = tifxyz_tracer.predict(task, dirpath, cfg)
    via_contract = contract.score_candidate(ref, task, prediction, cfg)

    assert via_contract["abstained"] == 0
    for key in ("included", "excluded", "correct", "wrong_hop", "wrong_wrap",
                "distance_miss", "correct_pct", "wrong_hop_pct", "wrong_wrap_pct"):
        assert via_contract[key] == direct[key], key


@pytest.fixture(scope="module")
def class1_neighborhood(ref_s11000):
    """The true class +1 point set (rows, cols, pts) in the column
    neighborhood that actually matters for s11000's seed cells -- class
    +1 sits a full spiral revolution away from the window in this band's
    raw column indexing (see stb.strips.matched_column_neighborhood), not
    physically near start=11000's own column range."""
    ref = ref_s11000
    seed_pts = ref.seed[ref.rr.ravel(), ref.cc.ravel()]
    lo, hi = strips_mod.matched_column_neighborhood(ref, 1, seed_pts, MESH_MARGIN)
    rows1, cols1, pts1 = ref.rows_of[1], ref.cols_of[1], ref.pts_of[1]
    keep = (cols1 >= lo) & (cols1 < hi)
    return rows1[keep], cols1[keep], pts1[keep], lo, hi


def test_mesh_adapter_oracle_scores_high_correct_front(
    ref_s11000, normals_332, resolved_cfg_332, class1_neighborhood
):
    ref, cfg = ref_s11000, resolved_cfg_332
    normals_band, _n_ok = normals_332
    rows1, cols1, pts1, _lo, _hi = class1_neighborhood

    mesh = mesh_adapter.build_grid_mesh(rows1, cols1, pts1)
    task = contract.task_for_window(ref, normals_band, START, +1, cfg)
    prediction = mesh_adapter.predict(task, mesh)
    summary = contract.score_candidate(ref, task, prediction, cfg)

    assert summary["correct_pct"] >= 95.0
    assert summary["included"] > 0


def test_mesh_adapter_abstain_accounting_half_window(
    ref_s11000, normals_332, resolved_cfg_332, class1_neighborhood
):
    ref, cfg = ref_s11000, resolved_cfg_332
    normals_band, _n_ok = normals_332
    rows1, cols1, pts1, lo, hi = class1_neighborhood
    task = contract.task_for_window(ref, normals_band, START, +1, cfg)

    mesh_full = mesh_adapter.build_grid_mesh(rows1, cols1, pts1)
    full = contract.score_candidate(ref, task, mesh_adapter.predict(task, mesh_full), cfg)

    mid = (cols1.min() + cols1.max()) / 2.0
    keep_half = cols1 < mid
    mesh_half = mesh_adapter.build_grid_mesh(rows1[keep_half], cols1[keep_half], pts1[keep_half])
    half = contract.score_candidate(ref, task, mesh_adapter.predict(task, mesh_half), cfg)

    # clipping the mesh to (roughly) half its column range should push
    # abstention up substantially and shrink the scored denominator by
    # (roughly) half -- generous bounds since this is real, noisy grid
    # data, not a synthetic exact-half split.
    assert half["abstained_pct"] > full["abstained_pct"] + 10.0
    ratio = half["included"] / full["included"]
    assert 0.3 <= ratio <= 0.7, f"expected ~half the denominator, got ratio={ratio:.3f}"
