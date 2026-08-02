from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework/stages/04-validation/scripts"


def load(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_patch(voxel_um: float) -> tuple[np.ndarray, int]:
    depth = int(round(480.0 / voxel_um)) + 1
    central = depth // 2
    positions = (np.arange(depth) - central) * voxel_um
    signal = (
        35.0
        + 80.0 / (1.0 + np.exp(-(positions + 28.0) / 6.0))
        - 55.0 / (1.0 + np.exp(-(positions - 36.0) / 8.0))
    )
    patch = np.broadcast_to(signal[:, None, None], (depth, 9, 9)).copy()
    return patch.astype(np.float32), central


def test_physical_features_are_stable_across_voxel_sizes():
    module = load("extract_ct_fiber_features_physical.py")
    patch_791, center_791 = synthetic_patch(7.91)
    patch_9362, center_9362 = synthetic_patch(9.362)
    a = module.extract_physical_depth_features(
        patch_791, central_slice=center_791, voxel_um=7.91
    )
    b = module.extract_physical_depth_features(
        patch_9362, central_slice=center_9362, voxel_um=9.362
    )

    assert a["physical_window_coverage_fraction"] == 1.0
    assert b["physical_window_coverage_fraction"] == 1.0
    assert abs(
        a["depth_profile_top_energy_band_fraction"]
        - b["depth_profile_top_energy_band_fraction"]
    ) < 0.08
    assert abs(
        a["argmax_depth_p90_p10_span_um"]
        - b["argmax_depth_p90_p10_span_um"]
    ) < 12.0


def decision(*, retained: bool, failed: list[str]):
    checks = []
    for feature in failed:
        checks.append(
            {
                "feature": feature,
                "value": 0.5,
                "threshold": 0.9,
                "operator": ">=",
                "passed": False,
            }
        )
    return {
        "group_id": "surface-1",
        "class": "target",
        "candidate_id": "G01",
        "retained": retained,
        "decision": "retained" if retained else "downranked",
        "failed_features": failed,
        "checks": checks,
    }


def test_shadow_router_never_discards_and_separates_boundary_failures():
    module = load("helena_route_ct_gate_shadow_v4.py")
    coverage = {
        "candidate_bbox_nonzero_fraction",
        "central_slice_nonzero_fraction",
    }
    boundary = module.route_decision(
        decision(retained=False, failed=["candidate_bbox_nonzero_fraction"]),
        coverage_features=coverage,
        physical_features=None,
        minimum_window_coverage_fraction=0.95,
    )
    diffuse = module.route_decision(
        decision(retained=False, failed=["depth_profile_entropy"]),
        coverage_features=coverage,
        physical_features=None,
        minimum_window_coverage_fraction=0.95,
    )
    retained = module.route_decision(
        decision(retained=True, failed=[]),
        coverage_features=coverage,
        physical_features=None,
        minimum_window_coverage_fraction=0.95,
    )

    assert boundary["shadow_tier"] == "TIER_C_EXTEND_OR_RESEGMENT"
    assert diffuse["shadow_tier"] == "TIER_B_SHADOW_REVIEW"
    assert retained["shadow_tier"] == "TIER_A_V3_RETAINED_REVIEW"
    assert all(row["not_discarded"] for row in [boundary, diffuse, retained])


def test_multiscroll_validator_blocks_single_positive_scroll():
    module = load("helena_validate_ct_gate_multiscroll.py")
    manifest = {
        "policy": {
            "minimum_independent_positive_scrolls": 3,
            "minimum_independent_confound_scrolls": 2,
            "minimum_positive_recall_per_scroll": 0.95,
        },
        "threshold_development_groups": ["dev-1"],
        "controls": [
            {
                "group_id": "p139-1",
                "scroll_id": "PHerc0139",
                "expected_class": "POSITIVE",
                "retained": True,
            },
            {
                "group_id": "n172-1",
                "scroll_id": "PHerc0172",
                "expected_class": "CONFOUND",
                "retained": False,
            },
        ],
    }
    result = module.evaluate_manifest(manifest)

    assert result["status"].startswith("BLOCKED_")
    assert "INSUFFICIENT_INDEPENDENT_POSITIVE_SCROLLS" in result["blocking_reasons"]


def test_multiscroll_validator_excludes_development_scroll_by_scroll_id():
    module = load("helena_validate_ct_gate_multiscroll.py")
    result = module.evaluate_manifest(
        {
            "policy": {
                "minimum_independent_positive_scrolls": 1,
                "minimum_independent_confound_scrolls": 0,
                "minimum_positive_recall_per_scroll": 0.95,
            },
            "threshold_development_groups": [],
            "threshold_development_scrolls": ["PHerc0139"],
            "controls": [
                {
                    "group_id": "renamed-group-that-looks-independent",
                    "scroll_id": "PHerc0139",
                    "expected_class": "POSITIVE",
                    "retained": True,
                }
            ],
        }
    )
    assert result["independent_positive_scroll_count"] == 0
    assert result["excluded_threshold_development_scrolls"] == ["PHerc0139"]
    assert "INSUFFICIENT_INDEPENDENT_POSITIVE_SCROLLS" in result["blocking_reasons"]


def test_multiscroll_validator_passes_disjoint_controls():
    module = load("helena_validate_ct_gate_multiscroll.py")
    controls = []
    for scroll in ["P1", "P2", "P3"]:
        controls.extend(
            {
                "group_id": f"{scroll}-{index}",
                "scroll_id": scroll,
                "expected_class": "POSITIVE",
                "retained": True,
            }
            for index in range(20)
        )
    for scroll in ["N1", "N2"]:
        controls.append(
            {
                "group_id": f"{scroll}-1",
                "scroll_id": scroll,
                "expected_class": "CONFOUND",
                "retained": False,
            }
        )
    result = module.evaluate_manifest(
        {
            "policy": {
                "minimum_independent_positive_scrolls": 3,
                "minimum_independent_confound_scrolls": 2,
                "minimum_positive_recall_per_scroll": 0.95,
            },
            "threshold_development_groups": ["dev-1"],
            "controls": controls,
        }
    )

    assert result["status"] == "INDEPENDENT_MULTISCROLL_BENCHMARK_PASSED"


def test_stage_registry_resolves_v4_scripts():
    from scripts.harness.stage_script_registry import resolve_stage_script

    for name in [
        "extract_ct_fiber_features_physical.py",
        "helena_route_ct_gate_shadow_v4.py",
        "helena_validate_ct_gate_multiscroll.py",
    ]:
        assert resolve_stage_script(ROOT, name).name == name
