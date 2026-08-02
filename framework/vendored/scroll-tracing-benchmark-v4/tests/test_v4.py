from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from adapters import tifxyz_tracer
from stb import cli, contract, estimator, features, metrics, scale, sequential, splits
from stb.config import ScrollConfig


@pytest.fixture(scope="module")
def ref_s11000(resolved_cfg_332, band_332):
    from stb import reference as reference_mod
    xyz, valid, _ = band_332
    return reference_mod.reference_at(xyz, valid, 11000, resolved_cfg_332)


def test_all_abstain_is_not_a_valid_result(ref_s11000, normals_332, resolved_cfg_332):
    ref, cfg = ref_s11000, resolved_cfg_332
    normals_band, _ = normals_332
    task = contract.task_for_window(ref, normals_band, 11000, +1, cfg)
    prediction = np.full_like(task.seed_points, np.nan)
    summary = contract.score_candidate(ref, task, prediction, cfg)
    assert summary["coverage_pct"] == 0.0
    assert summary["correct_yield_pct"] == 0.0
    assert summary["valid_result"] is False
    assert contract.coverage_gate(summary, 80.0)["pass"] is False


def _trace_task():
    anchors = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    candidates = np.array([
        [[1, 0, 0], [0, 1, 0]],
        [[2, 0, 0], [1, 1, 0]],
        [[3, 0, 0], [2, 1, 0]],
    ], dtype=float)
    task = sequential.SequentialTraceTask(
        trace_id="chain",
        anchors=anchors,
        candidates=candidates,
        candidate_valid=np.ones((3, 2), dtype=bool),
        anchor_ids=np.array([10, 11, 12]),
        candidate_ids=np.array([[11, 91], [12, 92], [13, 93]]),
    )
    ref = sequential.SequentialTraceReference(
        relation_labels=np.array([[0, 1], [0, 2], [0, 1]])
    )
    return task, ref


def test_sequential_trace_requires_real_state_chain():
    task, ref = _trace_task()
    good = sequential.score_trace(task, ref, [0, 0, 0])
    assert good["valid_completed"] is True
    bad_relation = sequential.score_trace(task, ref, [0, 1, 0])
    assert bad_relation["failure_kind"] == "unrelated"
    assert bad_relation["valid_completed"] is False

    broken = sequential.SequentialTraceTask(
        trace_id="broken", anchors=task.anchors, candidates=task.candidates,
        candidate_valid=task.candidate_valid, anchor_ids=task.anchor_ids,
        candidate_ids=np.array([[77, 91], [12, 92], [13, 93]]),
    )
    continuity = sequential.score_trace(broken, ref, [0, 0, 0])
    assert continuity["failure_kind"] == "continuity_violation"


def test_any_non_same_relation_counts_as_trace_failure():
    task, ref = _trace_task()
    adjacent = sequential.score_trace(task, ref, [1, 0, 0])
    unrelated = sequential.score_trace(task, ref, [0, 1, 0])
    assert adjacent["first_failure_step"] == 0
    assert unrelated["first_failure_step"] == 1
    assert adjacent["total_relation_errors"] == unrelated["total_relation_errors"] == 1


def test_split_validator_detects_neighbor_and_cross_overlap():
    train = [splits.SampleProvenance("a", "s", "seg", 100, 200, 1000, 1100)]
    test = [splits.SampleProvenance("b", "s", "seg", 400, 500, 1050, 1150)]
    with pytest.raises(ValueError, match="leakage"):
        splits.validate_split(train, test)
    independent = [splits.SampleProvenance("c", "other", "seg2", 0, 100, 200, 300)]
    assert splits.validate_split(train, independent, require_cross_scroll=True)["pass"]


def test_feature_manifest_blocks_reference_derived_test_features():
    clean = [{"name": "ct_intensity", "source": "native_ct"}]
    assert features.validate_feature_manifest(clean, "test")["pass"]
    leaked = clean + [{"name": "gap", "source": "reference_kdtree_gap"}]
    with pytest.raises(ValueError, match="leakage"):
        features.validate_feature_manifest(leaked, "test")
    audit = features.copied_feature_audit(
        anchor_features=np.array([[1.0, 2.0], [3.0, 4.0]]),
        candidate_features=np.array([[[1.0, 9.0]], [[3.0, 8.0]]]),
        relation_labels=np.array([[1], [1]]), feature_names=["copied", "independent"],
    )
    assert audit["exact_copy_rate_by_relation"][1]["copied"] == 1.0
    assert audit["exact_copy_rate_by_relation"][1]["independent"] == 0.0


def test_candidate_recall_and_risk_coverage_use_fixed_groups():
    recall = metrics.candidate_set_recall(
        np.array([[10, 11], [20, 21], [30, 31]]), np.array([11, 99, 30])
    )
    assert recall["hits"] == 2
    assert recall["oracle_recall_pct"] == pytest.approx(200 / 3)
    curve = metrics.risk_coverage_curve(
        correct=[True, False, True], catastrophic=[False, True, False],
        confidence=[0.9, 0.8, 0.1],
    )
    assert curve["eligible"] == 3
    assert len(curve["coverage"]) == 3
    ci = metrics.cluster_bootstrap_mean(
        [0.0, 1.0, 0.5, 0.5], ["window-a", "window-a", "window-b", "window-b"],
        iterations=100, seed=13,
    )
    assert ci["clusters"] == 2
    assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]


def test_physical_column_sampling_scales_between_scrolls():
    p332 = scale.derive_column_sampling(8484)
    p1667 = scale.derive_column_sampling(711)
    assert p332["window"] == pytest.approx(200, abs=5)
    assert p1667["window"] < p332["window"]


def test_cli_version(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "4.0.0"


def test_grid_cells_generalizes_but_preserves_legacy():
    rr, cc = estimator.grid_cells(0, 2)
    assert np.array_equal(np.unique(rr), np.linspace(10, 190, 9).astype(int) // 2 * 2)
    rr2, cc2 = estimator.grid_cells(0, 1, height=80, window=120)
    assert rr2.min() >= 0 and rr2.max() < 80
    assert cc2.min() >= 0 and cc2.max() < 120


def test_strict_window_preflight_is_additive(resolved_cfg_332, band_332):
    from stb import reference as reference_mod
    xyz, valid, _ = band_332
    with pytest.raises(ValueError, match="window center"):
        reference_mod.validate_window_center(xyz, valid, 0, resolved_cfg_332)


def test_ct_profiles_abstain_when_profile_leaves_volume():
    volume = np.arange(10 ** 3, dtype=np.float32).reshape(10, 10, 10)
    seeds = np.array([[0.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    raw, complete = estimator.sample_profiles(
        volume, seeds, normals, profile_halfwidth_vox=2, return_valid=True
    )
    assert complete.tolist() == [False, True]
    assert np.isnan(raw[0]).all()
    assert np.isfinite(raw[1]).all()


def test_tifxyz_adapter_rejects_same_count_wrong_shape(tmp_path):
    for axis in "xyz":
        tifffile.imwrite(tmp_path / f"{axis}.tif", np.ones((2, 3), dtype=np.float32))
    task = contract.WindowTask(
        scroll_id="s", col_start=0,
        seed_points=np.zeros((6, 3)),
        normals=np.tile([1.0, 0.0, 0.0], (6, 1)),
        direction=1, gap_hint_median=2.0, grid_shape=(3, 2),
    )
    with pytest.raises(ValueError, match="exact grid shape"):
        tifxyz_tracer.predict(task, tmp_path, SimpleNamespace(step=1))


@pytest.mark.parametrize("kwargs", [
    {"vox_um": 0.0}, {"step": 0}, {"classes": (0, 0, 1)},
    {"threshold_kind": "mystery"}, {"center": (1.0,)},
])
def test_config_rejects_invalid_values(kwargs):
    base = dict(scroll_id="s", volume_url="u", vox_um=1.0, center=(0.0, 0.0),
                band_path=tmp_path_placeholder())
    base.update(kwargs)
    with pytest.raises(ValueError):
        ScrollConfig(**base)


def tmp_path_placeholder():
    from pathlib import Path
    return Path("unused.npz")
