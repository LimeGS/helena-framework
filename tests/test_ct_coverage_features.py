from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/extract_ct_fiber_features.py"


def load_module():
    spec = importlib.util.spec_from_file_location("helena_ct_features", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_central_patch_is_coverage_safe():
    module = load_module()
    patch = np.full((9, 11, 11), 100, dtype=np.float32)

    features = module.extract_features(patch, central_slice=4)

    assert features["central_slice_center_nonzero"] == 1
    assert features["central_slice_nonzero_fraction"] == 1.0
    assert features["central_slice_zero_distance_ratio"] > 1.0


def test_zero_boundary_is_measured_relative_to_physical_patch_radius():
    module = load_module()
    patch = np.full((9, 11, 11), 100, dtype=np.float32)
    patch[4, :, :4] = 0

    features = module.extract_features(patch, central_slice=4)

    assert features["central_slice_center_nonzero"] == 1
    assert round(features["central_slice_nonzero_fraction"], 6) == round(77 / 121, 6)
    assert features["central_slice_zero_distance_ratio"] == 0.4


def test_zero_candidate_center_is_explicitly_invalid():
    module = load_module()
    patch = np.full((9, 11, 11), 100, dtype=np.float32)
    patch[4, 5, 5] = 0

    features = module.extract_features(patch, central_slice=4)

    assert features["central_slice_center_nonzero"] == 0
    assert features["central_slice_zero_distance_ratio"] == 0.0


def test_analysis_bbox_mapping_covers_every_touched_source_pixel():
    module = load_module()

    mapped = module.map_analysis_bbox_to_source(
        bbox_xyxy=(1, 2, 4, 6),
        analysis_shape_y_x=(10, 10),
        source_shape_y_x=(23, 17),
    )

    # x: floor(1*1.7)..ceil(4*1.7), y: floor(2*2.3)..ceil(6*2.3)
    assert mapped == (1, 4, 7, 14)


def test_analysis_bbox_mapping_clamps_to_source_extent():
    module = load_module()

    assert module.map_analysis_bbox_to_source(
        bbox_xyxy=(-4, -3, 14, 12),
        analysis_shape_y_x=(10, 10),
        source_shape_y_x=(20, 30),
    ) == (0, 0, 30, 20)


def test_full_candidate_bbox_detects_partial_coverage_missed_by_center_patch():
    module = load_module()
    stack = np.full((5, 20, 20), 100, dtype=np.float32)
    # The candidate center remains supported, but its elongated footprint
    # crosses the renderer's zero sentinel at the right edge.
    stack[2, 2:18, 17:20] = 0

    fraction = module.candidate_bbox_nonzero_fraction(
        stack,
        central_slice=2,
        source_bbox_xyxy=(10, 2, 20, 18),
    )

    assert fraction == 0.7


def test_full_candidate_bbox_reports_complete_support():
    module = load_module()
    stack = np.full((5, 20, 20), 100, dtype=np.float32)

    assert module.candidate_bbox_nonzero_fraction(
        stack,
        central_slice=2,
        source_bbox_xyxy=(3, 4, 17, 19),
    ) == 1.0


def test_v2_profile_distinguishes_source_locked_and_coverage_only_controls():
    profile = json.loads(
        (
            ROOT
            / "framework/profiles/validation/ct-fiber-localization-gate-v2-coverage-safe.json"
        ).read_text()
    )
    evidence = profile["calibration_evidence"]

    assert evidence["source_locked_full_positive_controls"] == [
        "PHerc0139-public-positive"
    ]
    assert evidence["source_locked_full_positive_control_candidate_count"] == 15
    assert evidence["coverage_diagnostic_candidate_count"] == 33
    assert evidence["coverage_diagnostic_requirements_passed_count"] == 33
    assert evidence[
        "additive_v1_v2_retention_mismatch_count_on_identical_current_inputs"
    ] == 0
    assert evidence["pherc172_full_positive_control_status"].startswith("EXCLUDED")


def test_v3_profile_adds_only_full_candidate_coverage_requirement():
    v2_path = (
        ROOT
        / "framework/profiles/validation/ct-fiber-localization-gate-v2-coverage-safe.json"
    )
    v3 = json.loads(
        (
            ROOT
            / "framework/profiles/validation/ct-fiber-localization-gate-v3-candidate-coverage.json"
        ).read_text()
    )
    assert v3["inherits"]["sha256"] == hashlib.sha256(v2_path.read_bytes()).hexdigest()
    requirements = {row["feature"]: row for row in v3["requirements"]}
    assert requirements["candidate_bbox_nonzero_fraction"]["operator"] == ">="
    assert requirements["candidate_bbox_nonzero_fraction"]["threshold"] == 0.95
    assert len(v3["requirements"]) == 8
    assert v3["calibration_evidence"][
        "candidate_bbox_nonzero_fraction_control_min"
    ] == 1.0
    assert v3["calibration_evidence"]["pherc268_posthoc_boundary_candidate"][
        "candidate_bbox_nonzero_fraction"
    ] < 0.95
