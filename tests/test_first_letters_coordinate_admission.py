from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import ct_support, executor, planner, seed_probe, worker


EXPECTED_COORDINATE_SHA = "732f1d8424f0bf0aef663e3a25929af51a14738cdaa53893cd50e6418ed7d0a6"
EXPECTED_POLICY_SHA = "c3ad5e10010502c9c8a42e388f5c2116194308e2b69a63f4ad497e1afee51df0"


def _api(module, name: str):
    assert hasattr(module, name), f"Task 6 API {module.__name__}.{name} is missing"
    return getattr(module, name)


def _response_bytes(x=12, *, candidates=None) -> bytes:
    identity = {
        "request_id": "req", "cell_id": "cell-a",
        "source_snapshot_id": "source-a", "source_snapshot_sha256": "1" * 64,
        "prediction_root_sha256": "2" * 64, "resolution": 4, "level": 1,
        "model_id": "m7-v1", "model_sha256": "3" * 64,
        "provider_id": "provider-v1", "provider_sha256": "4" * 64,
        "cell_region_sha256": "5" * 64, "grid_spec_sha256": "6" * 64,
        "dependency_manifest_sha256": "7" * 64, "maximum_candidates": 2,
    }
    if candidates is None:
        candidates = [{
            "candidate_id": "c1", "cell_id": "cell-a",
            "ct_l0_coordinate": {"x": x, "y": 34, "z": 56},
            "score": 0.9,
        }]
    return json.dumps({
        "prediction_identity": identity, "candidates": candidates,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_coordinate_policy_freezes_reject_nonintegral_for_promotion_v1_and_hash_binding():
    path = STAGE / "fleet/profiles/first-letters-coordinate-admission-v1.json"
    policy = json.loads(path.read_text())
    assert policy["rule_id"] == "REJECT_NONINTEGRAL_FOR_PROMOTION_V1"
    assert policy["serialized_shape"] == "JSON_ARRAY_LENGTH_THREE_X_Y_Z"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_POLICY_SHA


def test_provider_native_ct_l0_coordinate_xyz_mapping_projects_once_to_canonical_array_while_response_bytes_remain_unchanged():
    project = _api(seed_probe, "project_provider_response_v1")
    raw = _response_bytes()
    result = project(raw)
    assert result["response_bytes"] == raw
    assert result["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["candidates"][0]["raw_coordinate_ct_l0_xyz"] == [12, 34, 56]
    assert "ct_l0_coordinate" not in result["candidates"][0]


@pytest.mark.parametrize("value", [
    {"x": 12, "y": 34, "z": 56},
    (12, 34, 56),
    [12, 34],
    [12, 34, 56, 78],
    [56, 34, 12],
])
def test_task6_coordinate_contract_rejects_xyz_object_serialized_tuple_wrong_length_and_axis_order(value):
    with pytest.raises(ValueError):
        seed_probe.validate_task6_coordinate(
            value,
            expected_coordinate=[12, 34, 56],
        )


def test_coordinate_sha_uses_only_literal_schema_frame_and_xyz_array_projection():
    assert seed_probe.coordinate_sha256_v1([12, 34, 56]) == EXPECTED_COORDINATE_SHA


def test_integral_coordinate_is_identical_in_provider_artifact_ct_planner_promotion_plan_and_executor_argv():
    candidate = seed_probe.project_provider_response_v1(_response_bytes())["candidates"][0]
    ct = _api(ct_support, "task6_ct_coordinate_terminal")(
        candidate, sampler=lambda coordinate: {"nonzero_voxel_count": 1}
    )
    packet = _api(planner, "task6_planner_candidate")(candidate, ct_terminal=ct)
    argv = _api(executor, "task6_seed_argv")({
        "selected_seed_ct_l0_xyz": packet["promotion_coordinate_ct_l0_xyz"],
        "selected_seed_sha256": packet["promotion_coordinate_sha256"],
    })
    assert candidate["raw_coordinate_ct_l0_xyz"] == [12, 34, 56]
    assert ct["coordinate_ct_l0_xyz"] == [12, 34, 56]
    assert packet["promotion_coordinate_ct_l0_xyz"] == [12, 34, 56]
    assert candidate["promotion_coordinate_sha256"] == ct["coordinate_sha256"] == packet["promotion_coordinate_sha256"] == EXPECTED_COORDINATE_SHA
    assert argv == ["--seed", "12", "34", "56"]


@pytest.mark.parametrize("x", [12.0, 12.75, -0.25])
def test_nonintegral_raw_coordinate_is_preserved_but_rejected_before_ct_planner_promotion_and_executor(x):
    candidate = seed_probe.project_provider_response_v1(_response_bytes(x))["candidates"][0]
    assert candidate["raw_coordinate_ct_l0_xyz"] == [x, 34, 56]
    assert candidate["promotion_coordinate_ct_l0_xyz"] is None
    ct = ct_support.task6_ct_coordinate_terminal(candidate, sampler=lambda _: pytest.fail("sampler called"))
    assert ct["status"] == "CT_NOT_RUN_NONINTEGRAL_COORDINATE"
    with pytest.raises(ValueError):
        planner.task6_planner_candidate(candidate, ct_terminal=ct)
    with pytest.raises(ValueError):
        executor.task6_seed_argv({"selected_seed_ct_l0_xyz": [x, 34, 56], "selected_seed_sha256": candidate["raw_coordinate_sha256"]})


def test_json_float_12_0_is_not_silently_promoted_as_integer_12():
    candidate = seed_probe.project_provider_response_v1(_response_bytes(12.0))["candidates"][0]
    assert candidate["raw_coordinate_ct_l0_xyz"][0] == 12.0
    assert candidate["coordinate_admission_state"] == "REJECTED_NONINTEGRAL_COORDINATE_V1"


def test_nonintegral_rejection_emits_ct_not_run_and_clearance_not_run_without_sampler_call():
    candidate = seed_probe.project_provider_response_v1(_response_bytes(12.75))["candidates"][0]
    called = False
    def sampler(_):
        nonlocal called
        called = True
    terminal = ct_support.task6_ct_coordinate_terminal(candidate, sampler=sampler)
    assert called is False
    assert terminal["status"] == "CT_NOT_RUN_NONINTEGRAL_COORDINATE"
    assert terminal["clearance_status"] == "CLEARANCE_NOT_RUN_DUE_TO_CT"


def test_coordinate_policy_or_coordinate_sha_drift_fails_at_ct_planner_promotion_and_executor():
    candidate = seed_probe.project_provider_response_v1(_response_bytes())["candidates"][0]
    candidate["promotion_coordinate_sha256"] = "0" * 64
    for operation in (
        lambda: ct_support.task6_ct_coordinate_terminal(candidate, sampler=lambda _: {}),
        lambda: planner.task6_planner_candidate(candidate, ct_terminal={"status": "CT_RETAINED"}),
        lambda: executor.task6_seed_argv({"selected_seed_ct_l0_xyz": [12, 34, 56], "selected_seed_sha256": "0" * 64}),
    ):
        with pytest.raises(ValueError):
            operation()


@pytest.mark.parametrize("x", [12.9, -0.9])
def test_planner_ct_support_and_executor_never_round_floor_or_truncate_task6_coordinates(x):
    candidate = seed_probe.project_provider_response_v1(_response_bytes(x))["candidates"][0]
    assert candidate["raw_coordinate_ct_l0_xyz"][0] == x
    assert ct_support.task6_ct_coordinate_terminal(candidate, sampler=lambda _: pytest.fail("called"))["status"] == "CT_NOT_RUN_NONINTEGRAL_COORDINATE"
    with pytest.raises(ValueError):
        planner.task6_planner_candidate(candidate, ct_terminal={"status": "CT_RETAINED"})


def test_worker_recenter_and_chunk_safe_paths_do_not_truncate_or_rank_nonintegral_candidates():
    candidates = seed_probe.project_provider_response_v1(
        _response_bytes(candidates=[
            {
                "candidate_id": "bad", "cell_id": "cell-a",
                "ct_l0_coordinate": {"x": 12.75, "y": 34, "z": 56},
                "score": 1.0,
            },
            {
                "candidate_id": "good", "cell_id": "cell-a",
                "ct_l0_coordinate": {"x": 20, "y": 40, "z": 60},
                "score": 0.5,
            },
        ])
    )["candidates"]
    result = _api(worker, "task6_recenter_candidates")(candidates)
    assert result["eligible_candidate_ids"] == ["good"]
    assert result["median_coordinate_ct_l0_xyz"] == [20, 40, 60]
    assert result["rejected_candidate_ids"] == ["bad"]


def test_python_tuple_is_transient_only_and_never_appears_in_json_or_hash_input():
    with pytest.raises(ValueError):
        seed_probe.coordinate_sha256_v1((12, 34, 56))
    assert json.loads(json.dumps(seed_probe.validate_task6_coordinate([12, 34, 56]))) == [12, 34, 56]
