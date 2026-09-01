from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import seed_probe


PROFILE_PATH = (
    STAGE / "fleet/profiles/first-letters-discovery-v1.json"
)
COORDINATE_POLICY_PATH = (
    STAGE / "fleet/profiles/first-letters-coordinate-admission-v1.json"
)


def _api(name: str):
    assert hasattr(seed_probe, name), f"Task 6 API {name} is missing"
    return getattr(seed_probe, name)


def _sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _coordinate_policy_sha() -> str:
    return hashlib.sha256(COORDINATE_POLICY_PATH.read_bytes()).hexdigest()


_CELL_ID = "cell-a"
_CELL_REGION = {"minimum": [0, 0, 0], "maximum": [64, 64, 64]}
_CELL_REGION_SHA256 = _sha(_CELL_REGION)
_GRID_SPEC_SHA256 = "4" * 64


def _dependencies() -> list[dict]:
    rows = []
    for index, role in enumerate((
        "CT_VOLUME",
        "M7_PREDICTION_VOLUME",
        "CANONICAL_GRID_SPEC",
        "CT_MATERIAL_SUPPORT_POLICY",
        "CELL_AND_VOLUME_CLEARANCE_POLICY",
    ), start=2):
        rows.append({
            "role": role,
            "artifact_sha256": str(index) * 64,
            "read_set_manifest_sha256": chr(96 + index) * 64,
            "cell_id": _CELL_ID,
            "cell_region_sha256": _CELL_REGION_SHA256,
            "grid_spec_sha256": _GRID_SPEC_SHA256,
        })
    return rows


def _provider_prediction_identity() -> dict:
    return {
        "request_id": "request-a",
        "cell_id": _CELL_ID,
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "prediction_root_sha256": "3" * 64,
        "resolution": 4,
        "level": 1,
        "model_id": "m7-v1",
        "model_sha256": "7" * 64,
        "provider_id": "fixture-provider-v1",
        "provider_sha256": "8" * 64,
        "cell_region_sha256": _CELL_REGION_SHA256,
        "grid_spec_sha256": _GRID_SPEC_SHA256,
        "dependency_manifest_sha256": _sha(_dependencies()),
        "maximum_candidates": 2,
    }


def _provider_native_response() -> dict:
    return {
        "prediction_identity": _provider_prediction_identity(),
        "candidates": [{
            "candidate_id": "candidate-a",
            "cell_id": _CELL_ID,
            "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56},
            "score": 0.9,
        }],
    }


def _provider_response_bytes(native: dict | None = None) -> bytes:
    return json.dumps(
        native or _provider_native_response(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _exact_inputs_sha(value: dict) -> str:
    """Independent test implementation of the frozen bytes-aware input hash."""

    core = copy.deepcopy({
        key: row for key, row in value.items()
        if key != "discovery_inputs_sha256"
    })
    response_bytes = core["provider_response"].pop("response_bytes")
    canonical_core = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical_core + b"\0" + response_bytes).hexdigest()


def _legacy_inputs() -> dict:
    value = {
        "schema": "campaignx.first_letters_discovery_inputs.v1",
        "source_snapshot": {
            "role": "SOURCE_SNAPSHOT_METADATA",
            "source_snapshot_id": "source-a",
            "source_snapshot_sha256": "1" * 64,
            "source_content_lock_sha256": "d" * 64,
            "cell_id": _CELL_ID,
            "cell_region_sha256": _CELL_REGION_SHA256,
            "grid_spec_sha256": _GRID_SPEC_SHA256,
        },
        "dependencies": _dependencies(),
        "provider_request": {
            "request_id": "request-a",
            "cell_id": _CELL_ID,
            "source_snapshot_id": "source-a",
            "source_snapshot_sha256": "1" * 64,
            "prediction_root_sha256": "3" * 64,
            "resolution": 4,
            "level": 1,
            "threshold": 0.5,
            "ct_l0_region": copy.deepcopy(_CELL_REGION),
            "cell_region_sha256": _CELL_REGION_SHA256,
            "grid_spec_sha256": _GRID_SPEC_SHA256,
            "coordinate_frame": "ct_l0_xyz",
            "maximum_candidates": 2,
            "minimum_separation": 12,
            "model_id": "m7-v1",
            "model_sha256": "7" * 64,
            "provider_id": "fixture-provider-v1",
            "provider_sha256": "8" * 64,
            "coordinate_admission_rule_id": "REJECT_NONINTEGRAL_FOR_PROMOTION_V1",
            "coordinate_admission_rule_sha256": _coordinate_policy_sha(),
            "dependency_manifest_sha256": _sha(_dependencies()),
        },
        "provider_response": {
            "response_sha256": "a" * 64,
            "response_bytes": 100,
            "prediction_identity": "m7-v1@4/1",
            "candidates": [
                {
                    "candidate_id": "candidate-a",
                    "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56},
                    "score": 0.9,
                }
            ],
        },
        "allow_unvalidated": False,
    }
    value["discovery_inputs_sha256"] = _sha(value)
    return value


def _inputs() -> dict:
    value = _legacy_inputs()
    response_bytes = _provider_response_bytes()
    value["provider_response"] = {
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "response_bytes": response_bytes,
        "prediction_identity": _provider_prediction_identity(),
        "candidates": _provider_native_response()["candidates"],
    }
    value["discovery_inputs_sha256"] = _exact_inputs_sha(value)
    return value


def _profile() -> dict:
    profile = json.loads(PROFILE_PATH.read_text())
    profile["discovery_inputs_sha256"] = _inputs()["discovery_inputs_sha256"]
    core = {key: row for key, row in profile.items() if key != "scientific_core_sha256"}
    profile["scientific_core_sha256"] = _sha(core)
    return profile


def _arm(arm_id: str = "arm-a") -> dict:
    value = {
        "schema": "campaignx.first_letters_experimental_arm_admission.v1",
        "arm_id": arm_id,
        "mission_id": "mission-a",
        "accepted_p0_id": "p0-a",
        "accepted_p0_sha256": "0" * 64,
        "source_snapshot_id": f"source-{arm_id}",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "2" * 64,
        "ct_metadata_sha256": "3" * 64,
        "ct_read_set_manifest_sha256": "4" * 64,
        "m7_metadata_sha256": "5" * 64,
        "m7_read_set_manifest_sha256": "6" * 64,
        "m7_model_id": f"m7-{arm_id}",
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_transform_sha256": "7" * 64,
        "m7_threshold": 0.5,
        "discovery_policy_id": f"first-letters-{arm_id}@1.0.0",
        "discovery_profile_sha256": "8" * 64,
        "deployed_revision": "1" * 40,
        "preflight_private_sha256": "9" * 64,
        "preflight_sanitized_sha256": "a" * 64,
        "ordered_cell_ids": ["cell-a", "cell-b"],
        "ordered_cell_set_sha256": _sha(["cell-a", "cell-b"]),
        "mission_compute_cap_authority_id": "cap-a",
        "mission_compute_cap_authority_sha256": "b" * 64,
        "requested_units": 48,
        "active_policy_chain_sha256": "c" * 64,
        "may_update_accepted_p0": False,
        "statistical_budget_delta": 0,
        "allow_unvalidated": False,
    }
    value["admission_sha256"] = _sha(value)
    return value


def test_discovery_profile_requires_noncanonical_namespace_shadow_2x12_and_false_override():
    validate = _api("validate_first_letters_discovery_profile")
    assert validate(_profile())["namespace"] == "NONCANONICAL_DISCOVERY"
    for field, bad in (
        ("namespace", "CANONICAL"),
        ("canonical_admission", "ALLOWED"),
        ("top_k", 3),
        ("probe_generations", 11),
        ("allow_unvalidated", True),
    ):
        malformed = _profile()
        malformed[field] = bad
        malformed["scientific_core_sha256"] = _sha({
            key: row for key, row in malformed.items()
            if key != "scientific_core_sha256"
        })
        with pytest.raises(ValueError):
            validate(malformed)


def test_profile_loader_binds_literal_exact_file_hash_distinct_from_scientific_core():
    load = _api("load_first_letters_discovery_profile")
    result = load(PROFILE_PATH)
    assert result["profile_file_sha256"] == (
        "3ccb36604930b1eef2c644f549ec899458508736b1e8580631949120ff93b80c"
    )
    assert result["scientific_core_sha256"] == (
        "5a5f96a5a78c87502b7e92905e9678a3db57e98a3eab5ed255ca46131c82b3cf"
    )
    assert result["profile_file_sha256"] != result["scientific_core_sha256"]


def test_probe_execution_profile_cannot_substitute_for_campaign_discovery_profile():
    validate = _api("validate_first_letters_discovery_profile")
    probe = json.loads((STAGE / "fleet/profiles/vc3d-m7-probe-v1.json").read_text())
    with pytest.raises(ValueError):
        validate(probe)


@pytest.mark.parametrize("retained", [100, {"not": "bytes"}])
def test_provider_response_requires_exact_retained_bytes_not_count_or_object(retained):
    value = _legacy_inputs()
    value["provider_response"]["response_bytes"] = retained
    value["discovery_inputs_sha256"] = _sha({
        key: row for key, row in value.items()
        if key != "discovery_inputs_sha256"
    })
    with pytest.raises(ValueError, match="exact bytes"):
        _api("validate_first_letters_discovery_inputs")(value)


def test_provider_response_sha256_recomputes_from_exact_raw_bytes():
    value = _inputs()
    value["provider_response"]["response_sha256"] = "a" * 64
    value["discovery_inputs_sha256"] = _exact_inputs_sha(value)
    with pytest.raises(ValueError, match="SHA-256"):
        _api("validate_first_letters_discovery_inputs")(value)


def test_provider_response_exact_bytes_parse_to_the_sole_candidate_projection():
    result = _api("validate_first_letters_discovery_inputs")(_inputs())
    assert result["provider_response"]["response_bytes"] == _provider_response_bytes()
    assert result["provider_response"]["prediction_identity"] == (
        _provider_prediction_identity()
    )
    assert [row["candidate_id"] for row in result["projected_candidates"]] == [
        "candidate-a"
    ]


def test_provider_response_rejects_divergent_secondary_candidate_list():
    value = _inputs()
    value["provider_response"]["candidates"][0]["candidate_id"] = "forged"
    value["discovery_inputs_sha256"] = _exact_inputs_sha(value)
    with pytest.raises(ValueError, match="parsed bytes"):
        _api("validate_first_letters_discovery_inputs")(value)


@pytest.mark.parametrize("location", [
    "response_extra", "prediction_ocr", "candidate_content", "coordinate_human",
])
def test_provider_response_schema_is_recursively_closed_and_ink_blind(location):
    native = _provider_native_response()
    if location == "response_extra":
        native["extra"] = False
    elif location == "prediction_ocr":
        native["prediction_identity"]["ocr_text"] = "A"
    elif location == "candidate_content":
        native["candidates"][0]["content"] = "A"
    else:
        native["candidates"][0]["ct_l0_coordinate"]["human_review"] = False
    value = _inputs()
    response_bytes = _provider_response_bytes(native)
    value["provider_response"]["response_bytes"] = response_bytes
    value["provider_response"]["response_sha256"] = hashlib.sha256(
        response_bytes
    ).hexdigest()
    value["discovery_inputs_sha256"] = _exact_inputs_sha(value)
    with pytest.raises(ValueError):
        _api("validate_first_letters_discovery_inputs")(value)


@pytest.mark.parametrize("credential_value", [
    "https://provider.invalid/prediction?token=secret",
    "Authorization: Bearer secret-value",
])
def test_prediction_identity_rejects_credential_bearing_values(credential_value):
    native = _provider_native_response()
    native["prediction_identity"]["provider_id"] = credential_value
    value = _inputs()
    response_bytes = _provider_response_bytes(native)
    value["provider_response"]["response_bytes"] = response_bytes
    value["provider_response"]["response_sha256"] = hashlib.sha256(
        response_bytes
    ).hexdigest()
    value["provider_response"]["prediction_identity"] = copy.deepcopy(
        native["prediction_identity"]
    )
    value["discovery_inputs_sha256"] = _exact_inputs_sha(value)
    with pytest.raises(ValueError, match="credential"):
        _api("validate_first_letters_discovery_inputs")(value)


@pytest.mark.parametrize(
    "field",
    ["p5_score", "p7_routing", "ink_probability", "ocr_text", "human_review", "phrase_location"],
)
def test_discovery_inputs_reject_every_content_informed_role_and_extra_nested_field(field):
    validate = _api("validate_first_letters_discovery_inputs")
    malformed = _inputs()
    malformed["provider_request"][field] = "forbidden"
    malformed["discovery_inputs_sha256"] = _exact_inputs_sha(malformed)
    with pytest.raises(ValueError):
        validate(malformed)


def test_content_signal_changes_cannot_change_candidates_order_adaptive_or_promotion_core():
    projection = _api("discovery_scientific_projection")
    baseline = _inputs()
    candidate = projection(baseline)
    for field, value in (
        ("p5", {"score": 0.99}),
        ("p7", {"glyph": "A"}),
        ("human", {"reading": "alpha"}),
        ("ocr", "alpha"),
        ("lexical", 1.0),
        ("phrase_location", [1, 2, 3]),
    ):
        outside = {"discovery_inputs": copy.deepcopy(baseline), field: value}
        assert projection(outside["discovery_inputs"]) == candidate


def test_nonintegral_candidate_remains_exact_discovery_evidence_but_is_not_promotable():
    project = _api("project_provider_candidate_v1")
    row = project({
        "candidate_id": "c",
        "cell_id": _CELL_ID,
        "ct_l0_coordinate": {"x": 12.75, "y": 34, "z": 56},
        "score": 0.5,
    }, provider_response_sha256="a" * 64)
    assert row["raw_coordinate_ct_l0_xyz"] == [12.75, 34, 56]
    assert row["promotion_coordinate_ct_l0_xyz"] is None
    assert row["coordinate_admission_state"] == "REJECTED_NONINTEGRAL_COORDINATE_V1"


def _stored_artifact_and_receipt(
    tmp_path: Path, *, candidates=None, measurements=None,
):
    """Exercise artifact/receipt validation only through the concrete store."""

    from test_first_letters_discovery_evidence_store import (
        _claim_job, _complete, _store,
    )

    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    completed = _complete(
        store, handle, candidates=candidates, measurements=measurements,
    )
    artifact, receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    return store, completed, artifact, receipt, completed["retained_files"]


def _two_provider_candidates() -> list[dict]:
    return [
        {
            "candidate_id": "candidate-a", "cell_id": "cell-a",
            "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9,
        },
        {
            "candidate_id": "candidate-b", "cell_id": "cell-a",
            "ct_l0_coordinate": {"x": 40, "y": 10, "z": 20}, "score": 0.8,
        },
    ]


def test_artifact_builder_has_store_owned_evidence_set_boundary_only():
    """Promotion evidence cannot be supplied or rehashed by the caller."""

    import inspect
    from fleet.store import FleetStore

    parameters = inspect.signature(
        FleetStore.build_first_letters_discovery_artifact_and_receipt
    ).parameters
    assert set(parameters) == {"self", "evidence_set_id"}
    assert not hasattr(
        seed_probe, "build_first_letters_discovery_artifact_and_receipt"
    )


def test_artifact_builder_derives_complete_candidate_set_only_from_retained_bytes(tmp_path):
    store, completed, artifact, receipt, retained = _stored_artifact_and_receipt(
        tmp_path
    )
    assert [row["candidate_id"] for row in artifact["candidates"]] == [
        "candidate-a"
    ]
    assert _sha({
        key: row for key, row in artifact.items() if key != "artifact_sha256"
    }) == artifact["artifact_sha256"]
    assert _sha({
        key: row for key, row in receipt.items() if key != "receipt_sha256"
    }) == receipt["receipt_sha256"]
    assert _api("validate_first_letters_discovery_artifact")(
        artifact, retained_files=retained
    ) == artifact

    caller_copy = copy.deepcopy(completed)
    caller_copy["candidate_outcomes"] = []
    caller_copy["selection"]["selected_candidate_id"] = "forged"
    assert store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    ) == (artifact, receipt)


def test_artifact_builder_has_no_independent_candidate_injection_parameter():
    import inspect
    from fleet.store import FleetStore

    parameters = inspect.signature(
        FleetStore.build_first_letters_discovery_artifact_and_receipt
    ).parameters
    assert set(parameters) == {"self", "evidence_set_id"}
    assert "candidates" not in parameters
    assert "file_manifest" not in parameters


def test_multiple_candidates_may_retain_distinct_probe_files_under_one_closed_role(
    tmp_path,
):
    _, _, artifact, _, _ = _stored_artifact_and_receipt(
        tmp_path, candidates=_two_provider_candidates(),
    )
    assert [row["role"] for row in artifact["file_manifest"]].count(
        "NONCANONICAL_PROBE_GEOMETRY"
    ) == 2
    assert [row["candidate_id"] for row in artifact["candidates"]] == [
        "candidate-a", "candidate-b",
    ]


def test_zero_projected_candidates_retain_exact_exchange_without_dummy_probe_file(
    tmp_path,
):
    _, _, artifact, receipt, _ = _stored_artifact_and_receipt(
        tmp_path, candidates=[], measurements=(),
    )
    assert artifact["candidates"] == []
    assert artifact["funnel_counts"] == {
        "raw_candidates": 0, "ct_supported_candidates": 0,
        "clearance_supported_candidates": 0, "probe_measurable_candidates": 0,
    }
    assert "NONCANONICAL_PROBE_GEOMETRY" not in {
        row["role"] for row in artifact["file_manifest"]
    }
    assert receipt["used_units"] == 0
    assert receipt["selected_candidate_id"] is None
    assert receipt["selection_outcome"] == "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER"


def test_ct_rejected_candidate_needs_no_probe_geometry_file(tmp_path):
    from test_first_letters_discovery_evidence_store import _measurement

    _, _, artifact, receipt, _ = _stored_artifact_and_receipt(
        tmp_path,
        measurements=(_measurement(
            "candidate-a", ct_nonzero_voxels=0,
        ),),
    )
    assert artifact["funnel_counts"] == {
        "raw_candidates": 1, "ct_supported_candidates": 0,
        "clearance_supported_candidates": 0, "probe_measurable_candidates": 0,
    }
    assert "NONCANONICAL_PROBE_GEOMETRY" not in {
        row["role"] for row in artifact["file_manifest"]
    }
    assert receipt["used_units"] == 0
    assert receipt["selection_outcome"] == "DISCOVERY_REJECTED_CANDIDATES"


@pytest.mark.parametrize("case", [
    "nonintegral", "ct_rejected", "clearance_rejected", "unmeasurable",
])
def test_winner_selection_requires_complete_measurable_promotion_candidate(
    tmp_path, case,
):
    from test_first_letters_discovery_evidence_store import _measurement

    candidate = {
        "candidate_id": "candidate-a", "cell_id": "cell-a",
        "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9,
    }
    changes = {
        "nonintegral": {"ct_nonzero_voxels": None},
        "ct_rejected": {"ct_nonzero_voxels": 0},
        "clearance_rejected": {
            "probe_measurable": None, "coordinate": (0, 34, 56),
        },
        "unmeasurable": {"probe_measurable": False},
    }[case]
    if case == "nonintegral":
        candidate["ct_l0_coordinate"]["x"] = 12.75
    elif case == "clearance_rejected":
        candidate["ct_l0_coordinate"]["x"] = 0
    _, _, artifact, receipt, _ = _stored_artifact_and_receipt(
        tmp_path, candidates=[candidate],
        measurements=(_measurement("candidate-a", **changes),),
    )
    assert artifact["selected_candidate_id"] is None
    assert receipt["selected_candidate_id"] is None
    assert receipt["selection_outcome"] == "DISCOVERY_REJECTED_CANDIDATES"


@pytest.mark.parametrize("mutation", [
    "absolute", "query", "traversal", "wrong_size", "wrong_hash",
    "duplicate_path", "duplicate_role", "credential_value", "missing_role",
])
def test_artifact_manifest_recomputes_actual_safe_closed_inventory(tmp_path, mutation):
    _, _, artifact, _, retained = _stored_artifact_and_receipt(tmp_path)
    bad = copy.deepcopy(artifact)
    if mutation == "absolute":
        bad["file_manifest"][0]["relative_path"] = "/private/request.json"
    elif mutation == "query":
        bad["file_manifest"][0]["relative_path"] += "?token=secret"
    elif mutation == "traversal":
        bad["file_manifest"][0]["relative_path"] = "probes/../secret"
    elif mutation == "wrong_size":
        bad["file_manifest"][0]["byte_count"] += 1
    elif mutation == "wrong_hash":
        bad["file_manifest"][0]["sha256"] = "f" * 64
    elif mutation == "duplicate_path":
        bad["file_manifest"][1]["relative_path"] = (
            bad["file_manifest"][0]["relative_path"]
        )
    elif mutation == "duplicate_role":
        bad["file_manifest"][1]["role"] = bad["file_manifest"][0]["role"]
    elif mutation == "credential_value":
        retained[0]["bytes"] = b"Authorization: Bearer private"
    else:
        bad["file_manifest"].pop()
    bad["file_manifest_sha256"] = _sha(bad["file_manifest"])
    bad["artifact_sha256"] = _sha({
        key: row for key, row in bad.items() if key != "artifact_sha256"
    })
    with pytest.raises(ValueError):
        _api("validate_first_letters_discovery_artifact")(
            bad, retained_files=retained
        )


@pytest.mark.parametrize("mutation", [
    "candidate_extra", "coordinate_object", "ct_extra", "clearance_hash",
    "probe_bool_units", "source_row_ocr", "provider_hash", "canonical_key",
])
def test_artifact_nested_candidate_evidence_is_closed_cross_bound_and_ink_blind(
    tmp_path, mutation,
):
    _, _, artifact, _, retained = _stored_artifact_and_receipt(tmp_path)
    bad = copy.deepcopy(artifact)
    candidate = bad["candidates"][0]
    if mutation == "candidate_extra":
        candidate["extra"] = False
    elif mutation == "coordinate_object":
        candidate["raw_coordinate_ct_l0_xyz"] = {"x": 12, "y": 34, "z": 56}
    elif mutation == "ct_extra":
        candidate["ct_terminal"]["extra"] = False
    elif mutation == "clearance_hash":
        candidate["clearance_terminal"]["ct_terminal_sha256"] = "f" * 64
    elif mutation == "probe_bool_units":
        candidate["probe_evidence"]["used_units"] = True
    elif mutation == "source_row_ocr":
        candidate["contributing_source_rows"][0]["ocr_text"] = "A"
    elif mutation == "provider_hash":
        candidate["provider_response_sha256"] = "f" * 64
    else:
        candidate["canonical_surface"] = {"surface_id": "bad"}
    bad["artifact_sha256"] = _sha({
        key: row for key, row in bad.items() if key != "artifact_sha256"
    })
    with pytest.raises(ValueError):
        _api("validate_first_letters_discovery_artifact")(
            bad, retained_files=retained
        )


def test_complete_receipt_has_exact_closed_promotion_evidence_and_validates(tmp_path):
    _, _, artifact, receipt, retained = _stored_artifact_and_receipt(tmp_path)
    validated = _api("validate_first_letters_discovery_receipt")(
        receipt, artifact=artifact, retained_files=retained
    )
    assert validated == receipt
    assert {
        "evidence_set_id", "execution_authority_sha256", "run_id",
        "run_authority_sha256", "cell_id", "cell_region_sha256",
        "grid_spec_sha256", "dependency_manifest_sha256",
        "reservation_request_id", "reservation_work_kind",
        "reservation_work_authority_id", "reservation_work_authority_sha256",
        "reservation_source", "reservation_ordered_item_ids",
        "selection_policy_receipt", "selection_policy_receipt_sha256",
    } < set(validated)
    assert validated["artifact_sha256"] == artifact["artifact_sha256"]
    assert validated["used_units"] == 12


@pytest.mark.parametrize("mutation", [
    "parent_attempt", "sample", "artifact", "response", "reservation",
    "used_bool", "selection", "extra", "human_nested",
])
def test_receipt_validator_rejects_self_hashed_cross_binding_and_privacy_drift(
    tmp_path, mutation,
):
    _, _, artifact, receipt, retained = _stored_artifact_and_receipt(tmp_path)
    bad = copy.deepcopy(receipt)
    if mutation == "parent_attempt":
        bad["parent_attempt_id"] = "forged"
    elif mutation == "sample":
        bad["sample_id"] = "forged"
    elif mutation == "artifact":
        bad["artifact_sha256"] = "f" * 64
    elif mutation == "response":
        bad["provider_response_sha256"] = "f" * 64
    elif mutation == "reservation":
        bad["reservation_sha256"] = "f" * 64
    elif mutation == "used_bool":
        bad["used_units"] = True
    elif mutation == "selection":
        bad["selected_candidate_id"] = "forged"
    elif mutation == "extra":
        bad["extra"] = False
    else:
        bad["prediction_identity"]["human_review"] = False
    bad["receipt_sha256"] = _sha({
        key: row for key, row in bad.items() if key != "receipt_sha256"
    })
    with pytest.raises(ValueError):
        _api("validate_first_letters_discovery_receipt")(
            bad, artifact=artifact, retained_files=retained
        )


def test_authentic_artifact_receipt_resolves_exact_promotion_evidence(tmp_path):
    store, completed, artifact, receipt, _ = _stored_artifact_and_receipt(tmp_path)
    evidence = store.resolve_discovery_promotion_evidence(
        completed["evidence_set_id"]
    )
    assert evidence["selected_candidate"] == artifact["candidates"][0]
    assert evidence["receipt"] == receipt


def test_promotion_resolver_accepts_only_a_registered_evidence_set_id():
    from fleet.store import FleetStore

    parameters = inspect.signature(
        FleetStore.resolve_discovery_promotion_evidence
    ).parameters
    assert set(parameters) == {"self", "evidence_set_id"}
    assert not hasattr(seed_probe, "resolve_discovery_promotion_evidence")



def test_every_resolution_level_transform_threshold_or_source_is_a_separate_arm():
    validate = _api("validate_experimental_arm_admission")
    first = validate(_arm("a"))
    second = _arm("b")
    second["m7_resolution"] = 8
    second["admission_sha256"] = _sha({key: row for key, row in second.items() if key != "admission_sha256"})
    assert validate(second)["arm_id"] != first["arm_id"]


def test_v1_rejects_multiscale_same_source_arm():
    validate = _api("validate_first_letters_discovery_profile")
    bad = _profile()
    bad["arm_kind"] = "MULTISCALE_SAME_SOURCE_ARM"
    bad["scientific_core_sha256"] = _sha({key: row for key, row in bad.items() if key != "scientific_core_sha256"})
    with pytest.raises(ValueError):
        validate(bad)


def test_experimental_arm_has_own_source_preflight_cap_policy_and_cannot_update_p0():
    validate = _api("validate_experimental_arm_admission")
    assert validate(_arm())["may_update_accepted_p0"] is False
    for field in ("source_snapshot_sha256", "preflight_private_sha256", "mission_compute_cap_authority_sha256", "discovery_policy_id"):
        bad = _arm()
        del bad[field]
        with pytest.raises(ValueError):
            validate(bad)


def test_arm_comparison_requires_complete_identical_ordered_cell_membership():
    compare = _api("compare_experimental_arms")
    arms = [
        {"arm_id": "a", "cells": [{"cell_id": "one", "status": "PRESENT", "candidates": []}]},
        {"arm_id": "b", "cells": [{"cell_id": "two", "status": "PRESENT", "candidates": []}]},
    ]
    with pytest.raises(ValueError):
        compare(arms, ordered_cell_ids=["one"])


def test_arm_comparison_reports_literal_sets_and_survival_without_score_merge():
    compare = _api("compare_experimental_arms")
    report = compare([
        {"arm_id": "a", "cells": [{"cell_id": "one", "status": "PRESENT", "candidates": [
            {"candidate_id": "shared", "ct_retained": True, "clearance_retained": True, "probe_measurable": False, "score": 0.9},
            {"candidate_id": "only-a", "ct_retained": False, "clearance_retained": False, "probe_measurable": False, "score": 0.8},
        ]}]},
        {"arm_id": "b", "cells": [{"cell_id": "one", "status": "PRESENT", "candidates": [
            {"candidate_id": "shared", "ct_retained": True, "clearance_retained": True, "probe_measurable": True, "score": 0.1},
            {"candidate_id": "only-b", "ct_retained": True, "clearance_retained": False, "probe_measurable": False, "score": 1.0},
        ]}]},
    ], ordered_cell_ids=["one"])
    assert report["union"] == ["only-a", "only-b", "shared"]
    assert report["intersection"] == ["shared"]
    assert report["unique"] == {"a": ["only-a"], "b": ["only-b"]}
    assert "score" not in json.dumps(report).lower()


def _benchmark_manifest_v2() -> dict:
    row = {
        "sample_id": "PHercA",
        "cell_id": "cell-a",
        "cell_order": 0,
        "coordinate_frame": "ct_l0_xyz",
        "grid_spec_sha256": "0" * 64,
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "2" * 64,
        "ct_metadata_sha256": "3" * 64,
        "ct_read_set_manifest_sha256": "4" * 64,
        "m7_metadata_sha256": "5" * 64,
        "m7_read_set_manifest_sha256": "6" * 64,
        "model_id": "m7-a",
        "provider_id": "provider-a",
        "resolution": 4,
        "level": 1,
        "transform_sha256": "7" * 64,
        "threshold": 0.5,
        "discovery_profile_file_sha256": "8" * 64,
        "discovery_scientific_core_sha256": "9" * 64,
        "policy_version": "first-letters-discovery@1.0.0",
        "deployed_revision": "1" * 40,
        "baseline_result_sha256": "a" * 64,
        "select_result_sha256": "b" * 64,
        "baseline_status": "PRESENT",
        "select_status": "PRESENT",
    }
    lock = {
        key: row[key] for key in (
            "sample_id", "cell_id", "cell_order", "coordinate_frame",
            "grid_spec_sha256", "source_snapshot_id", "source_snapshot_sha256",
            "source_content_lock_sha256", "ct_metadata_sha256",
            "ct_read_set_manifest_sha256", "m7_metadata_sha256",
            "m7_read_set_manifest_sha256", "model_id", "provider_id", "resolution",
            "level", "transform_sha256", "threshold",
            "discovery_profile_file_sha256", "discovery_scientific_core_sha256",
            "policy_version", "deployed_revision",
        )
    }
    row["baseline_lock_sha256"] = _sha({"arm": "baseline", **lock})
    row["select_lock_sha256"] = _sha({"arm": "select", **lock})
    value = {
        "schema": "campaignx.seed_probe_benchmark_execution_manifest.v2",
        "benchmark_id": "benchmark-v2",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "deployed_revision": "1" * 40,
        "ordered_cohort": [row],
        "ordered_cohort_sha256": _sha([row]),
        "arms": {
            "baseline": {"mode": "off", "profile_sha256": "c" * 64},
            "select": {"mode": "select", "profile_sha256": "d" * 64},
        },
    }
    value["manifest_sha256"] = _sha(value)
    return value


def _benchmark_decision_v2(manifest: dict | None = None) -> dict:
    manifest = manifest or _benchmark_manifest_v2()
    value = {
        "schema": "campaignx.seed_probe_benchmark_decision.v2",
        "benchmark_id": manifest["benchmark_id"],
        "status": "APPROVED_SELECT",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "execution_manifest_sha256": manifest["manifest_sha256"],
        "spec_sha256": "e" * 64,
        "results_sha256": "f" * 64,
        "deployed_revision": manifest["deployed_revision"],
        "ordered_cohort_sha256": manifest["ordered_cohort_sha256"],
        "checks": [
            {"check_id": name, "status": "PASS"}
            for name in ("COHORT_COMPLETE", "SOURCE_LOCKS_MATCH", "PAIRED_RESULTS_COMPLETE")
        ],
    }
    value["decision_sha256"] = _sha(value)
    return value


def test_benchmark_v2_binds_complete_cohort_cells_sources_read_sets_profiles_and_revision():
    validate_manifest = _api("validate_seed_probe_benchmark_execution_manifest_v2")
    validate_decision = _api("validate_seed_probe_benchmark_receipt_v2")
    manifest = validate_manifest(_benchmark_manifest_v2())
    authorization = validate_decision(_benchmark_decision_v2(manifest), execution_manifest=manifest)
    assert authorization["schema"] == "campaignx.seed_probe_benchmark_authorization.v2"
    assert authorization["ordered_cohort"] == manifest["ordered_cohort"]


@pytest.mark.parametrize("field", [
    "sample_id", "cell_id", "source_snapshot_sha256", "ct_read_set_manifest_sha256",
    "m7_read_set_manifest_sha256", "model_id", "resolution", "level",
    "transform_sha256", "threshold", "grid_spec_sha256",
    "discovery_profile_file_sha256", "policy_version", "deployed_revision",
])
def test_benchmark_v2_rejects_each_cross_arm_lock_or_membership_drift(field):
    validate = _api("validate_seed_probe_benchmark_execution_manifest_v2")
    manifest = _benchmark_manifest_v2()
    manifest["ordered_cohort"][0][field] = "drift" if field not in {"resolution", "level", "threshold"} else 99
    manifest["ordered_cohort_sha256"] = _sha(manifest["ordered_cohort"])
    manifest["manifest_sha256"] = _sha({key: row for key, row in manifest.items() if key != "manifest_sha256"})
    with pytest.raises(ValueError):
        validate(manifest)


def test_v1_compact_authorization_cannot_authorize_task6_select():
    validate = _api("validate_benchmark_authorization_for_promotion")
    with pytest.raises(ValueError):
        validate({"schema": "campaignx.seed_probe_benchmark_authorization.v1"}, task={}, profile={}, source={}, deployed_revision="1" * 40)


def test_promotion_revalidates_retained_benchmark_v2_bytes_not_caller_hashes():
    manifest = _benchmark_manifest_v2()
    authorization = seed_probe.validate_seed_probe_benchmark_receipt_v2(
        _benchmark_decision_v2(manifest), execution_manifest=manifest
    )
    task = {"sample_id": "PHercA", "cell_id": "cell-a"}
    profile = {"profile_file_sha256": "8" * 64, "discovery_policy_id": "first-letters-discovery@1.0.0"}
    source = {"source_snapshot_id": "source-a", "source_snapshot_sha256": "1" * 64}
    assert seed_probe.validate_benchmark_authorization_for_promotion(
        authorization, task=task, profile=profile, source=source,
        deployed_revision="1" * 40,
    )["authorization_sha256"] == authorization["authorization_sha256"]
    forged = copy.deepcopy(authorization)
    forged["ordered_cohort"][0]["source_snapshot_id"] = "forged"
    with pytest.raises(ValueError):
        seed_probe.validate_benchmark_authorization_for_promotion(
            forged, task=task, profile=profile, source=source,
            deployed_revision="1" * 40,
        )
