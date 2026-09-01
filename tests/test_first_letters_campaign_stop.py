"""Candidate-starvation decisions use only frozen scientific-terminal evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from fleet import campaign_decision  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402
from test_first_letters_campaign_decision import (  # noqa: E402
    budget_admission,
    budget_task,
    store_source,
)


POLICY = json.loads((
    ROOT / "framework/profiles/01-segmentation/"
    "first-letters-campaign-decision-policy-1.2.0.json"
).read_text(encoding="utf-8"))


def attempt(
    index: int, *, state: str = "NO_SEED", raw_m7: object = 0,
    sample_id: str = "PHerc358", policy_version: str = "search-v1",
    failure_class: str | None = None,
) -> dict:
    task_id = f"task-{index:02d}"
    attempt_id = f"attempt-{index:02d}"
    raw_is_zero = (
        isinstance(raw_m7, int) and not isinstance(raw_m7, bool) and raw_m7 == 0)
    raw_is_positive = (
        isinstance(raw_m7, int) and not isinstance(raw_m7, bool) and raw_m7 > 0)
    raw_count = raw_m7 if raw_is_positive else 0
    cause_counts = {
        "NO_M7_CANDIDATES": 1 if raw_is_zero else 0,
        "CT_MATERIAL_SUPPORT_REJECTED": 0,
        "MALFORMED_COORDINATE_OR_SCORE": 0,
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": raw_count,
        "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": 0,
    }
    diagnosis = {
        "schema": "campaignx.no_seed_causal_diagnosis.v1",
        "status": state,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "m7_raw_candidate_count": raw_m7,
        "ct_support_input_candidate_count": raw_count,
        "ct_support_retained_candidate_count": raw_count,
        "ct_support_rejected_candidate_count": 0,
        "post_ct_candidate_count": raw_count,
        "eligible_after_clearance_count": 0,
        "cause_counts": cause_counts,
        "primary_causes": sorted(
            key for key, value in cause_counts.items() if value > 0),
        "ink_used": False,
    }
    diagnosis["diagnosis_sha256"] = content_sha256(diagnosis)
    result = {
        "status": state,
        "raw_candidate_count": raw_m7,
        "post_ct_candidate_count": raw_count,
        "usable_candidate_count": 0,
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis": diagnosis,
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
        "ink_used": False,
    }
    if failure_class is not None:
        result["failure_class"] = failure_class
    return {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": 1,
        "mission_id": "first-letters",
        "sample_id": sample_id,
        "policy_version": policy_version,
        "state": state,
        "result": result,
        "terminal_at_utc": f"2026-08-03T12:{index:02d}:00Z",
    }


def admission(sample_id: str, index: int, count: int = 2) -> dict:
    prefix = [f"{sample_id}-cell-{rank}" for rank in range(count)]
    value = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": "first-letters",
        "sample_id": sample_id,
        "receipt_sha256": f"{index + 1:064x}",
        "preflight_receipt_sha256": f"{index + 101:064x}",
        "preflight_sanitized_receipt_sha256": f"{index + 201:064x}",
        "approved_task_count": count,
        "order_seed_sha256": "a" * 64,
        "population_order_sha256": "b" * 64,
        "prefix_sha256": content_sha256(prefix),
        "prefix_cell_ids": prefix,
        "execution_bindings": {"policy_version": "search-v1"},
        "registered_at_utc": f"2026-08-03T11:{index:02d}:00Z",
    }
    value["admission_sha256"] = content_sha256({
        key: row for key, row in value.items()
        if key != "registered_at_utc"
    })
    return value


def admitted_attempt(index: int, authority: dict, rank: int, *, raw_m7: int = 0) -> dict:
    row = attempt(
        index, raw_m7=raw_m7, sample_id=authority["sample_id"])
    row["cell_id"] = authority["prefix_cell_ids"][rank]
    row["campaign_budget"] = {
        key: value for key, value in authority.items()
        if key != "registered_at_utc"
    }
    row["campaign_budget"]["selection_rank"] = rank
    return row


def resized_admission(authority: dict, count: int) -> dict:
    value = copy.deepcopy(authority)
    prefix = value["prefix_cell_ids"][:count]
    value.update({
        "approved_task_count": count,
        "prefix_cell_ids": prefix,
        "prefix_sha256": content_sha256(prefix),
    })
    value["admission_sha256"] = content_sha256({
        key: row for key, row in value.items()
        if key != "admission_sha256"
    })
    return value


def derive(attempts: list[dict], admissions: list[dict] | None = None) -> list[dict]:
    implementation = getattr(
        campaign_decision, "derive_campaign_decision_receipts", None)
    assert callable(implementation), (
        "derive_campaign_decision_receipts is not implemented")
    bound_attempts = copy.deepcopy(attempts)
    authorities = admissions
    if admissions is None:
        task_ranks: dict[str, int] = {}
        for position, row in enumerate(bound_attempts):
            key = str(row.get("task_id") or f"malformed-{position}")
            task_ranks.setdefault(key, len(task_ranks))
        authority = admission("PHerc358", 90, count=len(task_ranks))
        for position, row in enumerate(bound_attempts):
            key = str(row.get("task_id") or f"malformed-{position}")
            rank = task_ranks[key]
            row["cell_id"] = authority["prefix_cell_ids"][rank]
            row["campaign_budget"] = {
                key: copy.deepcopy(value) for key, value in authority.items()
                if key != "registered_at_utc"
            }
            row["campaign_budget"]["selection_rank"] = rank
        authorities = [authority]
    return implementation(
        bound_attempts,
        authorities or [],
        POLICY,
        mission_id="first-letters",
        policy_version="search-v1",
    )


def no_seed_result(claim: dict, raw_m7: int) -> dict:
    cause_counts = {
        "NO_M7_CANDIDATES": 1 if raw_m7 == 0 else 0,
        "CT_MATERIAL_SUPPORT_REJECTED": 0,
        "MALFORMED_COORDINATE_OR_SCORE": 0,
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": raw_m7,
        "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": 0,
    }
    diagnosis = {
        "schema": "campaignx.no_seed_causal_diagnosis.v1",
        "status": "NO_SEED",
        "task_id": claim["task_id"],
        "attempt_id": claim["attempt_id"],
        "m7_raw_candidate_count": raw_m7,
        "ct_support_input_candidate_count": raw_m7,
        "ct_support_retained_candidate_count": raw_m7,
        "ct_support_rejected_candidate_count": 0,
        "post_ct_candidate_count": raw_m7,
        "eligible_after_clearance_count": 0,
        "cause_counts": cause_counts,
        "primary_causes": sorted(
            key for key, value in cause_counts.items() if value > 0),
        "ink_used": False,
    }
    diagnosis["diagnosis_sha256"] = content_sha256(diagnosis)
    return {
        "status": "NO_SEED",
        "raw_candidate_count": raw_m7,
        "post_ct_candidate_count": raw_m7,
        "usable_candidate_count": 0,
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis": diagnosis,
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
        "ink_used": False,
    }


def additional_sample_admission(
    store: FleetStore, authority: dict, *, sample_id: str = "PHerc999",
) -> tuple[dict, list[dict]]:
    source_id = store.register_snapshot({
        "sample_id": sample_id, "ct_uri": "fixture://ct",
        "ct_sha256": "2" * 64, "m7_uri": "fixture://m7",
        "m7_sha256": "3" * 64, "shape_xyz": [32768, 32768, 32768],
        "voxel_size_um": 9.362, "coordinate_frame": "ct_l0_xyz",
        "m7_threshold": 0.2,
        "source_content_lock": {"schema": "fixture-source-lock"},
    })
    additional = copy.deepcopy(authority)
    additional.update({
        "sample_id": sample_id,
        "receipt_sha256": "e" * 64,
    })
    additional["execution_bindings"].update({
        "sample_id": sample_id,
        "source_snapshot_id": source_id,
    })
    additional["admission_sha256"] = content_sha256({
        key: value for key, value in additional.items()
        if key != "admission_sha256"
    })
    tasks = [
        budget_task(source_id, cell_id, additional)
        for cell_id in additional["prefix_cell_ids"]
    ]
    for task in tasks:
        task["sample_id"] = sample_id
    return additional, campaign_decision.bind_campaign_budget_to_tasks(
        tasks, additional)


def resumed_sample_admission(
    store: FleetStore, authority: dict,
) -> tuple[dict, list[dict]]:
    m7_uri = "fixture://m7-v2"
    source_id = store.register_snapshot({
        "sample_id": authority["sample_id"], "ct_uri": "fixture://ct",
        "ct_sha256": "2" * 64, "m7_uri": m7_uri,
        "m7_sha256": "4" * 64, "shape_xyz": [32768, 32768, 32768],
        "voxel_size_um": 9.362, "coordinate_frame": "ct_l0_xyz",
        "m7_threshold": 0.2,
        "source_content_lock": {"schema": "fixture-source-lock"},
    })
    resumed = copy.deepcopy(authority)
    resumed["receipt_sha256"] = "f" * 64
    resumed["execution_bindings"].update({
        "source_snapshot_id": source_id,
        "policy_version": "search-v2",
        "m7_sha256": "4" * 64,
        "m7_uri_sha256": hashlib.sha256(m7_uri.encode("utf-8")).hexdigest(),
    })
    resumed["admission_sha256"] = content_sha256({
        key: value for key, value in resumed.items()
        if key != "admission_sha256"
    })
    tasks = [
        budget_task(source_id, cell_id, resumed)
        for cell_id in resumed["prefix_cell_ids"]
    ]
    for task in tasks:
        task["candidate_discovery"]["prediction_uri"] = m7_uri
    return resumed, campaign_decision.bind_campaign_budget_to_tasks(
        tasks, resumed)


def resume_authorization(
    prior: dict, resumed: dict, decision: dict, *, field: str = "m7_source",
) -> dict:
    def causal_value(authority: dict, name: str) -> object:
        execution = authority["execution_bindings"]
        if name == "m7_source":
            return {
                "m7_sha256": execution.get("m7_sha256"),
                "m7_uri_sha256": execution.get("m7_uri_sha256"),
            }
        if name == "planner":
            return (execution.get("queue_execution") or {}).get("planner")
        if name == "calibrated_m7_threshold":
            return execution.get("m7_threshold")
        if name == "grid_version":
            return execution.get("grid_version")
        if name == "discovery_provider":
            return execution.get("provider")
        if name == "authorized_seed_probe_mode":
            return (execution.get("queue_execution") or {}).get(
                "seed_probe_mode")
        if name == "evidence_backed_clearance_policy":
            gates = execution.get("gates") or {}
            return {key: copy.deepcopy(gates.get(key)) for key in (
                "cell_clearance", "volume_clearance",
                "candidate_interior_clearance", "ct_material_support_gate",
            )}
        raise AssertionError(name)

    material_change = {
        "field": field,
        "prior_value_sha256": content_sha256({
            "field": field, "value": causal_value(prior, field)}),
        "new_value_sha256": content_sha256({
            "field": field, "value": causal_value(resumed, field)}),
        "evidence_sha256": (
            campaign_decision.campaign_resume_material_evidence_sha256(
                prior, resumed, field)
            if field in {"m7_source", "grid_version", "discovery_provider"}
            else "9" * 64
        ),
    }
    value = {
        "schema": "campaignx.first_letters_campaign_resume_authorization.v1",
        "mission_id": prior["mission_id"],
        "prior_sample_id": prior["sample_id"],
        "new_sample_id": resumed["sample_id"],
        "prior_policy_version": prior["execution_bindings"]["policy_version"],
        "new_policy_version": resumed["execution_bindings"]["policy_version"],
        "prior_admission_sha256": prior["admission_sha256"],
        "new_admission_sha256": resumed["admission_sha256"],
        "prior_decision_receipt_sha256": decision["receipt_sha256"],
        "material_changes": [material_change],
        "authorized_by": "campaign-owner",
        "authentication_context": {
            "mechanism": "HELENA_AUTHENTICATED_PANEL_SESSION",
            "principal": "campaign-owner",
            "session_fingerprint_sha256": "6" * 64,
            "request_method": "POST",
            "request_path": "/api/segmentation/runs",
        },
    }
    value["authorization_evidence_sha256"] = content_sha256({
        "schema": "campaignx.first_letters_resume_authorization_evidence.v1",
        **{key: value[key] for key in (
            "mission_id", "prior_sample_id", "new_sample_id",
            "prior_policy_version", "new_policy_version",
            "prior_admission_sha256", "new_admission_sha256",
            "prior_decision_receipt_sha256", "material_changes", "authorized_by",
            "authentication_context",
        )},
    })
    value["authorization_sha256"] = content_sha256(value)
    return value


def rehash_resume_authorization(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("authorization_sha256", None)
    value["authorization_evidence_sha256"] = content_sha256({
        "schema": "campaignx.first_letters_resume_authorization_evidence.v1",
        **{key: value[key] for key in (
            "mission_id", "prior_sample_id", "new_sample_id",
            "prior_policy_version", "new_policy_version",
            "prior_admission_sha256", "new_admission_sha256",
            "prior_decision_receipt_sha256", "material_changes", "authorized_by",
            "authentication_context",
        )},
    })
    value["authorization_sha256"] = content_sha256(value)
    return value


def test_resume_identity_is_rebound_to_the_authenticated_panel_principal():
    proposal = {
        "schema": "campaignx.first_letters_campaign_resume_authorization.v1",
        "mission_id": "first-letters",
        "prior_sample_id": "PHerc358",
        "new_sample_id": "PHerc358",
        "prior_policy_version": "search-v1",
        "new_policy_version": "search-v2",
        "prior_admission_sha256": "1" * 64,
        "new_admission_sha256": "2" * 64,
        "prior_decision_receipt_sha256": "3" * 64,
        "material_changes": [],
        "authorized_by": "caller-forged-admin",
        "authentication_context": {
            "mechanism": "HELENA_AUTHENTICATED_PANEL_SESSION",
            "principal": "caller-forged-admin",
            "session_fingerprint_sha256": "7" * 64,
            "request_method": "POST",
            "request_path": "/api/segmentation/runs",
        },
    }
    proposal = rehash_resume_authorization(proposal)
    bound = campaign_decision.bind_campaign_resume_authorization_principal(
        proposal,
        authorized_by="tester",
        session_fingerprint_sha256="8" * 64,
        request_method="POST",
        request_path="/api/segmentation/runs",
    )
    assert bound["authorized_by"] == "tester"
    assert bound["authentication_context"] == {
        "mechanism": "HELENA_AUTHENTICATED_PANEL_SESSION",
        "principal": "tester",
        "session_fingerprint_sha256": "8" * 64,
        "request_method": "POST",
        "request_path": "/api/segmentation/runs",
    }
    assert bound["authorization_sha256"] == content_sha256({
        key: value for key, value in bound.items()
        if key != "authorization_sha256"
    })


def test_seven_of_eight_raw_m7_empty_attempts_pause_the_exact_policy():
    rows = [attempt(index, raw_m7=(1 if index == 7 else 0))
            for index in range(8)]
    receipts = derive(rows)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema"] == "campaignx.first_letters_campaign_decision.v1"
    assert receipt["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert receipt["mission_id"] == "first-letters"
    assert receipt["policy_version"] == "search-v1"
    assert receipt["evaluation_kind"] == "SCIENTIFIC_TERMINAL_BLOCK"
    assert receipt["evaluation_index"] == 1
    assert receipt["no_m7_numerator"] == 7
    assert receipt["scientific_terminal_denominator"] == 8
    assert receipt["trigger_attempt_ids"] == [
        f"attempt-{index:02d}" for index in range(7)]
    assert receipt["allowed_next_actions"] == [
        "CREATE_MATERIALLY_CHANGED_VERSIONED_STRATEGY",
        "CLOSE_CAMPAIGN",
    ]
    assert receipt["receipt_sha256"] == content_sha256({
        key: value for key, value in receipt.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })


def test_pause_receipt_binds_each_attempt_to_its_governing_admission():
    authority = admission("PHerc358", 0, count=8)
    rows = [
        admitted_attempt(index, authority, index, raw_m7=(2 if index == 7 else 0))
        for index in range(8)
    ]
    receipt = derive(rows, [authority])[0]
    assert receipt["governing_admission_sha256s"] == [
        authority["admission_sha256"]]
    assert receipt["trigger_governing_admission_sha256s"] == [
        authority["admission_sha256"]]
    assert receipt["scientific_terminal_attempts"] == [
        {
            "attempt_id": f"attempt-{index:02d}",
            "task_id": f"task-{index:02d}",
            "sample_id": "PHerc358",
            "admission_sha256": authority["admission_sha256"],
            "budget_receipt_sha256": authority["receipt_sha256"],
        }
        for index in range(8)
    ]


def test_pause_resume_authority_comes_only_from_the_seven_trigger_attempts():
    trigger_authority = admission("PHerc358", 0, count=7)
    denominator_only_authority = admission("PHerc358", 1, count=1)
    rows = [
        admitted_attempt(index, trigger_authority, index)
        for index in range(7)
    ]
    rows.append(admitted_attempt(
        7, denominator_only_authority, 0, raw_m7=2))

    pause = derive(rows, [trigger_authority, denominator_only_authority])[0]
    assert pause["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert pause["governing_admission_sha256s"] == [
        trigger_authority["admission_sha256"]]
    assert pause["trigger_governing_admission_sha256s"] == [
        trigger_authority["admission_sha256"]]
    assert pause["scientific_terminal_attempts"][-1][
        "admission_sha256"] == denominator_only_authority["admission_sha256"]

    successor = copy.deepcopy(denominator_only_authority)
    successor["receipt_sha256"] = "f" * 64
    successor["execution_bindings"]["policy_version"] = "search-v2"
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key not in {"admission_sha256", "registered_at_utc"}
    })
    forged = resume_authorization(
        denominator_only_authority, successor, pause, field="planner")
    with pytest.raises(ValueError, match="did not govern the active pause"):
        campaign_decision.validate_campaign_resume_authorization(
            forged,
            prior_admission=denominator_only_authority,
            new_admission=successor,
            prior_decision=pause,
            policy=POLICY,
            authoritative_attempts=rows,
            registered_admissions=[
                trigger_authority, denominator_only_authority],
            trusted_authorization_sha256s={forged["authorization_sha256"]},
        )


def test_trigger_attempts_split_across_admissions_fail_closed():
    first = admission("PHerc358", 0, count=4)
    second = admission("PHerc358", 1, count=4)
    rows = [
        admitted_attempt(index, first, index)
        for index in range(4)
    ]
    rows.extend([
        admitted_attempt(index, second, index - 4, raw_m7=(2 if index == 7 else 0))
        for index in range(4, 8)
    ])

    receipt = derive(rows, [first, second])[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["evidence_status"] == "INCOMPLETE"
    assert receipt["governing_admission_sha256s"] == []
    assert receipt["trigger_governing_admission_sha256s"] == sorted([
        first["admission_sha256"], second["admission_sha256"]])
    assert receipt["trigger_attempt_ids"] == []
    assert receipt["control_incomplete_reasons"] == [
        "AMBIGUOUS_TRIGGER_GOVERNING_ADMISSIONS"]
    assert receipt["allowed_next_actions"] == [
        "REPAIR_OR_REPLAY_CAUSAL_EVIDENCE",
        "CLOSE_CAMPAIGN",
    ]


def test_rehashed_pause_cannot_rebind_authority_to_a_denominator_attempt():
    trigger_authority = admission("PHerc358", 0, count=7)
    denominator_only_authority = admission("PHerc358", 1, count=1)
    rows = [
        admitted_attempt(index, trigger_authority, index)
        for index in range(7)
    ]
    rows.append(admitted_attempt(
        7, denominator_only_authority, 0, raw_m7=2))
    pause = derive(rows, [trigger_authority, denominator_only_authority])[0]

    forged_pause = copy.deepcopy(pause)
    for bound_attempt in forged_pause["scientific_terminal_attempts"][:7]:
        bound_attempt.update({
            "sample_id": denominator_only_authority["sample_id"],
            "admission_sha256": denominator_only_authority[
                "admission_sha256"],
            "budget_receipt_sha256": denominator_only_authority[
                "receipt_sha256"],
        })
    forged_pause["trigger_governing_admission_sha256s"] = [
        denominator_only_authority["admission_sha256"]]
    forged_pause["governing_admission_sha256s"] = [
        denominator_only_authority["admission_sha256"]]
    forged_pause["receipt_sha256"] = content_sha256({
        key: value for key, value in forged_pause.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })

    successor = copy.deepcopy(denominator_only_authority)
    successor["receipt_sha256"] = "f" * 64
    successor["execution_bindings"].update({
        "policy_version": "search-v2",
        "provider": "vc3d-mcp-v2",
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key not in {"admission_sha256", "registered_at_utc"}
    })
    authorization = resume_authorization(
        denominator_only_authority, successor, forged_pause,
        field="discovery_provider",
    )
    with pytest.raises(ValueError, match="authoritative persisted pause"):
        campaign_decision.validate_campaign_resume_authorization(
            authorization,
            prior_admission=denominator_only_authority,
            new_admission=successor,
            prior_decision=forged_pause,
            policy=POLICY,
            authoritative_attempts=rows,
            registered_admissions=[
                trigger_authority, denominator_only_authority],
            trusted_authorization_sha256s={
                authorization["authorization_sha256"]},
        )


def test_rehashed_cross_scroll_pause_cannot_reverse_authoritative_order():
    first = admission("PHerc268", 0)
    second = admission("PHerc211", 1)
    rows = [
        admitted_attempt(0, first, 0),
        admitted_attempt(1, first, 1),
        admitted_attempt(2, second, 0),
        admitted_attempt(3, second, 1),
    ]
    pause = derive(rows, [first, second])[0]
    forged_pause = copy.deepcopy(pause)
    forged_pause["completed_zero_raw_m7_scrolls"] = list(reversed(
        forged_pause["completed_zero_raw_m7_scrolls"]))
    forged_pause["governing_admission_sha256s"] = [
        first["admission_sha256"]]
    forged_pause["receipt_sha256"] = content_sha256({
        key: value for key, value in forged_pause.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })

    successor = copy.deepcopy(first)
    successor["receipt_sha256"] = "f" * 64
    successor["execution_bindings"].update({
        "policy_version": "search-v2",
        "provider": "vc3d-mcp-v2",
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key not in {"admission_sha256", "registered_at_utc"}
    })
    authorization = resume_authorization(
        first, successor, forged_pause, field="discovery_provider")
    with pytest.raises(ValueError, match="authoritative persisted pause"):
        campaign_decision.validate_campaign_resume_authorization(
            authorization,
            prior_admission=first,
            new_admission=successor,
            prior_decision=forged_pause,
            policy=POLICY,
            authoritative_attempts=rows,
            registered_admissions=[first, second],
            trusted_authorization_sha256s={
                authorization["authorization_sha256"]},
        )


def test_forged_task_admission_binding_cannot_govern_a_pause():
    authority = admission("PHerc358", 0, count=9)
    rows = [admitted_attempt(index, authority, index) for index in range(7)]
    forged = admitted_attempt(7, authority, 7, raw_m7=2)
    forged["campaign_budget"]["admission_sha256"] = "f" * 64
    rows.extend([forged, admitted_attempt(8, authority, 8, raw_m7=3)])
    receipt = derive(rows, [authority])[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": "CONFIGURATION_BLOCK:UNBOUND_CAMPAIGN_BUDGET_ADMISSION",
    }]


def test_platform_failures_are_visible_but_never_fill_the_scientific_denominator():
    rows = [attempt(index) for index in range(7)]
    rows.append(attempt(
        7, state="GROW_FAILED", raw_m7=None,
        failure_class="WORKER_FAILURE"))
    rows.append(attempt(8, raw_m7=3))
    receipt = derive(rows)[0]
    assert receipt["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert receipt["no_m7_numerator"] == 7
    assert receipt["scientific_terminal_denominator"] == 8
    assert receipt["excluded_attempt_count"] == 1
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": "WORKER_FAILURE",
    }]
    assert "attempt-07" not in receipt["scientific_terminal_attempt_ids"]


def test_scientific_terminal_with_platform_failure_class_fails_closed():
    rows = [attempt(index) for index in range(7)]
    rows.append(attempt(
        7, state="ARCHIVED", raw_m7=None,
        failure_class="SOURCE_FAILURE"))
    rows.append(attempt(8, raw_m7=3))
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": (
            "CONFIGURATION_BLOCK:INCONSISTENT_TERMINAL_FAILURE_CLASS:"
            "ARCHIVED:SOURCE_FAILURE"),
    }]


@pytest.mark.parametrize(("state", "canonical"), [
    ("CANCELLED", "CANCELLED"),
    ("POLICY_REJECTED", "CONFIGURATION_BLOCK"),
    ("LEASE_EXPIRED", "LEASE_EXHAUSTION"),
    ("LEASE_EXHAUSTED", "LEASE_EXHAUSTION"),
    ("FINALIZATION_FAILED", "PUBLICATION_FAILURE"),
    ("BLOCKED_SOURCE_UNAVAILABLE", "SOURCE_FAILURE"),
    ("GROW_FAILED", "WORKER_FAILURE"),
    ("FIXTURE_ONLY", "FIXTURE_ONLY"),
])
def test_excluded_terminal_requires_its_canonical_failure_class(
    state, canonical,
):
    mismatch = "SOURCE_FAILURE" if canonical != "SOURCE_FAILURE" else "WORKER_FAILURE"
    rows = [attempt(index) for index in range(7)]
    rows.append(attempt(
        7, state=state, raw_m7=None,
        failure_class=mismatch))
    rows.append(attempt(8, raw_m7=3))
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"][0]["reason"] == (
        "CONFIGURATION_BLOCK:INCONSISTENT_TERMINAL_FAILURE_CLASS:"
        f"{state}:{mismatch}")

    rows[7] = attempt(7, state=state, raw_m7=None)
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"][0]["reason"] == (
        "CONFIGURATION_BLOCK:INCONSISTENT_TERMINAL_FAILURE_CLASS:"
        f"{state}:None")

    rows[7] = attempt(
        7, state=state, raw_m7=None, failure_class=canonical)
    receipt = derive(rows)[0]
    assert receipt["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert receipt["excluded_attempts"][0]["reason"] == canonical


def test_two_consecutive_completed_zero_raw_m7_scroll_budgets_pause_before_eight():
    first = admission("PHerc268", 0)
    second = admission("PHerc211", 1)
    rows = [
        admitted_attempt(0, first, 0),
        admitted_attempt(1, first, 1),
        admitted_attempt(2, second, 0),
        admitted_attempt(3, second, 1),
    ]
    receipt = derive(rows, [first, second])[0]
    assert receipt["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert receipt["evaluation_kind"] == (
        "CONSECUTIVE_ZERO_RAW_M7_SCROLL_BUDGETS")
    assert receipt["scientific_terminal_denominator"] == 4
    assert receipt["no_m7_numerator"] == 4
    assert receipt["completed_zero_raw_m7_scrolls"] == [
        {"sample_id": "PHerc268", "budget_receipt_sha256": first["receipt_sha256"]},
        {"sample_id": "PHerc211", "budget_receipt_sha256": second["receipt_sha256"]},
    ]
    assert receipt["trigger_attempt_ids"] == [
        "attempt-00", "attempt-01", "attempt-02", "attempt-03"]
    assert receipt["trigger_governing_admission_sha256s"] == sorted([
        first["admission_sha256"], second["admission_sha256"]])
    assert receipt["governing_admission_sha256s"] == [
        second["admission_sha256"]]

    successor = copy.deepcopy(second)
    successor["receipt_sha256"] = "f" * 64
    successor["execution_bindings"].update({
        "policy_version": "search-v2",
        "provider": "vc3d-mcp-v2",
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key not in {"admission_sha256", "registered_at_utc"}
    })
    authorization = resume_authorization(
        second, successor, receipt, field="discovery_provider")
    assert campaign_decision.validate_campaign_resume_authorization(
        authorization,
        prior_admission=second,
        new_admission=successor,
        prior_decision=receipt,
        policy=POLICY,
        authoritative_attempts=rows,
        registered_admissions=[first, second],
        trusted_authorization_sha256s={authorization["authorization_sha256"]},
    ) == authorization


def test_cross_scroll_pause_wins_before_a_later_eight_attempt_checkpoint():
    first = admission("PHerc268", 0)
    second = admission("PHerc211", 1)
    rows = [
        admitted_attempt(0, first, 0), admitted_attempt(1, first, 1),
        admitted_attempt(2, second, 0), admitted_attempt(3, second, 1),
        attempt(4, raw_m7=3), attempt(5, raw_m7=3),
        attempt(6, raw_m7=3), attempt(7, raw_m7=3),
    ]
    receipts = derive(rows, [first, second])
    assert len(receipts) == 1
    assert receipts[0]["evaluation_kind"] == (
        "CONSECUTIVE_ZERO_RAW_M7_SCROLL_BUDGETS")


def test_malformed_no_seed_evidence_is_excluded_and_makes_the_decision_incomplete():
    rows = [attempt(index) for index in range(7)]
    rows.append(attempt(7, raw_m7=None))
    rows.append(attempt(8, raw_m7=2))
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["evidence_status"] == "INCOMPLETE"
    assert receipt["no_m7_numerator"] == 7
    assert receipt["scientific_terminal_denominator"] == 8
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS",
    }]
    assert receipt["trigger_attempt_ids"] == []
    assert receipt["allowed_next_actions"] == [
        "REPAIR_OR_REPLAY_CAUSAL_EVIDENCE",
        "CLOSE_CAMPAIGN",
    ]


@pytest.mark.parametrize("failure_class", [
    "SOURCE_FAILURE", "WORKER_FAILURE", "PUBLICATION_FAILURE",
])
def test_no_seed_with_any_platform_failure_class_fails_closed(failure_class):
    rows = [attempt(index) for index in range(7)]
    tampered = attempt(7)
    tampered["result"]["failure_class"] = failure_class
    rows.extend([tampered, attempt(8, raw_m7=2)])
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["no_m7_numerator"] == 7
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": (
            "CONFIGURATION_BLOCK:INCONSISTENT_NO_SEED_FAILURE_CLASS:"
            f"{failure_class}"),
    }]


def test_no_seed_impossible_stage_counts_fail_closed_even_when_rehashed():
    rows = [attempt(index) for index in range(7)]
    tampered = attempt(7)
    diagnosis = tampered["result"]["no_seed_causal_diagnosis"]
    diagnosis.update({
        "ct_support_input_candidate_count": 1,
        "ct_support_retained_candidate_count": 1,
        "post_ct_candidate_count": 1,
        "cause_counts": {
            "NO_M7_CANDIDATES": 1,
            "CT_MATERIAL_SUPPORT_REJECTED": 0,
            "MALFORMED_COORDINATE_OR_SCORE": 0,
            "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": 1,
            "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": 0,
        },
        "primary_causes": [
            "INSUFFICIENT_CELL_INTERIOR_CLEARANCE",
            "NO_M7_CANDIDATES",
        ],
    })
    diagnosis["diagnosis_sha256"] = content_sha256({
        key: value for key, value in diagnosis.items()
        if key != "diagnosis_sha256"
    })
    tampered["result"].update({
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
    })
    rows.extend([tampered, attempt(8, raw_m7=2)])
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["no_m7_numerator"] == 7
    assert receipt["excluded_attempts"][0]["reason"] == (
        "CONFIGURATION_BLOCK:IMPOSSIBLE_NO_SEED_CAUSAL_COUNTS")


@pytest.mark.parametrize("unknown_label", [
    "NEW_CAUSAL_LABEL",
    "insufficient_cell_interior_clearance",
    "INSUFFICIENT_CELL_INTERIOR_CLEARANCE ",
])
def test_unknown_no_seed_causal_labels_fail_closed_even_when_rehashed(
    unknown_label,
):
    rows = [attempt(index) for index in range(7)]
    tampered = attempt(7, raw_m7=2)
    diagnosis = tampered["result"]["no_seed_causal_diagnosis"]
    diagnosis["cause_counts"] = {
        "NO_M7_CANDIDATES": 0,
        "CT_MATERIAL_SUPPORT_REJECTED": 0,
        "MALFORMED_COORDINATE_OR_SCORE": 0,
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": 0,
        "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": 0,
        unknown_label: 2,
    }
    diagnosis["primary_causes"] = [unknown_label]
    diagnosis.pop("diagnosis_sha256")
    diagnosis["diagnosis_sha256"] = content_sha256(diagnosis)
    tampered["result"].update({
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
    })
    rows.extend([tampered, attempt(8, raw_m7=3)])

    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["no_m7_numerator"] == 7
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": "CONFIGURATION_BLOCK:UNKNOWN_NO_SEED_CAUSAL_LABEL",
    }]


def test_mixed_known_and_unknown_no_seed_causal_labels_fail_closed():
    rows = [attempt(index) for index in range(7)]
    tampered = attempt(7, raw_m7=2)
    diagnosis = tampered["result"]["no_seed_causal_diagnosis"]
    diagnosis["cause_counts"].update({
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": 1,
        "UNKNOWN_CLEARANCE_REASON": 1,
    })
    diagnosis["primary_causes"] = sorted([
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE",
        "UNKNOWN_CLEARANCE_REASON",
    ])
    diagnosis.pop("diagnosis_sha256")
    diagnosis["diagnosis_sha256"] = content_sha256(diagnosis)
    tampered["result"].update({
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
    })
    rows.extend([tampered, attempt(8, raw_m7=3)])

    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"][0]["reason"] == (
        "CONFIGURATION_BLOCK:UNKNOWN_NO_SEED_CAUSAL_LABEL")


@pytest.mark.parametrize("missing_label", sorted(
    campaign_decision.NO_SEED_SCIENTIFIC_CAUSAL_LABELS
))
def test_missing_canonical_no_seed_causal_labels_fail_closed(
    missing_label,
):
    rows = [attempt(index) for index in range(7)]
    tampered = attempt(7, raw_m7=2)
    diagnosis = tampered["result"]["no_seed_causal_diagnosis"]
    diagnosis["cause_counts"].pop(missing_label)
    diagnosis["primary_causes"] = sorted(
        key for key, value in diagnosis["cause_counts"].items()
        if value > 0)
    diagnosis.pop("diagnosis_sha256")
    diagnosis["diagnosis_sha256"] = content_sha256(diagnosis)
    tampered["result"].update({
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
    })
    rows.extend([tampered, attempt(8, raw_m7=3)])

    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"][0]["reason"] == (
        "CONFIGURATION_BLOCK:MISSING_NO_SEED_CAUSAL_LABEL")


def test_only_the_latest_terminal_attempt_per_task_enters_a_block():
    rows = [attempt(index) for index in range(6)]
    retry = attempt(6)
    retry.update({
        "task_id": "task-00",
        "attempt_id": "attempt-00-retry",
        "attempt_number": 2,
        "terminal_at_utc": "2026-08-03T12:06:30Z",
    })
    retry_diagnosis = retry["result"]["no_seed_causal_diagnosis"]
    retry_diagnosis.update({
        "task_id": retry["task_id"], "attempt_id": retry["attempt_id"]})
    retry_diagnosis["diagnosis_sha256"] = content_sha256({
        key: value for key, value in retry_diagnosis.items()
        if key != "diagnosis_sha256"
    })
    retry["result"]["no_seed_causal_diagnosis_sha256"] = (
        retry_diagnosis["diagnosis_sha256"])
    rows.extend([retry, attempt(7, raw_m7=2), attempt(8, raw_m7=4)])
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTINUE"
    assert receipt["no_m7_numerator"] == 6
    assert receipt["scientific_terminal_denominator"] == 8
    assert receipt["superseded_attempt_ids"] == ["attempt-00"]
    assert "attempt-00" not in receipt["scientific_terminal_attempt_ids"]
    assert "attempt-00-retry" in receipt["scientific_terminal_attempt_ids"]


def test_a_rehashed_or_task_mismatched_no_seed_diagnosis_cannot_count_as_no_m7():
    rows = [attempt(index) for index in range(7)]
    rows[0]["result"]["no_seed_causal_diagnosis"][
        "m7_raw_candidate_count"] = 4
    rows.extend([attempt(7, raw_m7=2), attempt(8, raw_m7=3)])
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["no_m7_numerator"] == 6
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-00",
        "task_id": "task-00",
        "reason": "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS",
    }]


def test_no_seed_with_outer_platform_failure_label_is_control_incomplete():
    rows = [attempt(index) for index in range(7)]
    rows[0]["result"]["failure_class"] = "SOURCE_FAILURE"
    rows.extend([attempt(7, raw_m7=2), attempt(8, raw_m7=3)])
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["no_m7_numerator"] == 6
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-00",
        "task_id": "task-00",
        "reason": "CONFIGURATION_BLOCK:INCONSISTENT_NO_SEED_FAILURE_CLASS:SOURCE_FAILURE",
    }]


def test_malformed_terminal_identity_is_visible_and_fails_closed():
    rows = [attempt(index) for index in range(7)]
    rows[0]["task_id"] = None
    rows.extend([attempt(7, raw_m7=2), attempt(8, raw_m7=3)])
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["excluded_attempts"][0]["reason"] == (
        "CONFIGURATION_BLOCK:MALFORMED_ATTEMPT_IDENTITY")


@pytest.mark.parametrize(("state", "reason"), [
    ("CANCELLED", "CANCELLED"),
    ("POLICY_REJECTED", "CONFIGURATION_BLOCK"),
    ("LEASE_EXPIRED", "LEASE_EXHAUSTION"),
    ("FINALIZATION_FAILED", "PUBLICATION_FAILURE"),
    ("BLOCKED_SOURCE_UNAVAILABLE", "SOURCE_FAILURE"),
    ("GROW_FAILED", "WORKER_FAILURE"),
    ("FIXTURE_ONLY", "FIXTURE_ONLY"),
])
def test_only_the_explicit_platform_outcome_allowlist_is_excluded_cleanly(
    state, reason,
):
    rows = [attempt(index) for index in range(7)]
    rows.append(attempt(
        7, state=state, raw_m7=None, failure_class=reason))
    rows.append(attempt(8, raw_m7=2))
    receipt = derive(rows)[0]
    assert receipt["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert receipt["evidence_status"] == "COMPLETE"
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07", "task_id": "task-07", "reason": reason}]


def test_an_unknown_terminal_state_is_excluded_and_fails_closed():
    rows = [attempt(index) for index in range(7)]
    rows.append(attempt(7, state="MYSTERY_TERMINAL", raw_m7=None))
    rows.append(attempt(8, raw_m7=2))
    receipt = derive(rows)[0]
    assert receipt["decision"] == "CONTROL_INCOMPLETE"
    assert receipt["evidence_status"] == "INCOMPLETE"
    assert receipt["excluded_attempts"] == [{
        "attempt_id": "attempt-07",
        "task_id": "task-07",
        "reason": "CONFIGURATION_BLOCK:UNKNOWN_TERMINAL_STATE:MYSTERY_TERMINAL",
    }]


def test_sqlite_eighth_terminal_persists_an_immutable_pause_visible_when_empty(
    tmp_path,
):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    authority = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(authority)
    tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, authority)
         for cell_id in authority["prefix_cell_ids"]],
        authority,
    )
    assert store.create_tasks(tasks) == (8, 8)
    next_authority, next_tasks = additional_sample_admission(store, authority)
    for task in next_tasks:
        task["priority"] = -1.0
    store.register_campaign_budget_admission(next_authority)
    assert store.create_tasks(next_tasks[:1]) == (1, 1)
    for index in range(8):
        claim = store.claim(f"worker-{index}", 60)
        assert claim is not None
        assert claim["sample_id"] == authority["sample_id"]
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 2 if index == 7 else 0),
        )
    read_decisions = getattr(store, "campaign_decisions", None)
    assert callable(read_decisions), "FleetStore.campaign_decisions is not implemented"
    first = read_decisions(
        mission_id="first-letters",
        policy_version=authority["execution_bindings"]["policy_version"],
    )
    second = read_decisions(
        mission_id="first-letters",
        policy_version=authority["execution_bindings"]["policy_version"],
    )
    assert first == second
    assert len(first) == 1
    assert first[0]["decision"] == "PAUSE_CANDIDATE_STARVATION"
    assert first[0]["no_m7_numerator"] == 7
    assert store.status()["tasks"] == {"NO_SEED": 8, "PENDING": 1}
    with pytest.raises(ValueError, match="candidate starvation|campaign decision"):
        store.create_tasks(next_tasks[1:2])
    assert store.status()["tasks"] == {"NO_SEED": 8, "PENDING": 1}


def test_sqlite_controlled_mission_accepts_a_second_exact_sample_authority(
    tmp_path,
):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    first = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(first)
    first_tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, first)
         for cell_id in first["prefix_cell_ids"]],
        first,
    )
    assert store.create_tasks(first_tasks) == (8, 8)

    second, second_tasks = additional_sample_admission(store, first)
    assert store.register_campaign_budget_admission(second) == second
    assert store.create_tasks(second_tasks[:1]) == (1, 1)

    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM campaign_budget_admissions"
        ).fetchone()[0] == 2


def test_sqlite_eighth_terminal_and_creation_have_a_serializable_boundary(
    tmp_path,
):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    authority = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(authority)
    tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, authority)
         for cell_id in authority["prefix_cell_ids"]],
        authority,
    )
    assert store.create_tasks(tasks) == (8, 8)
    next_authority, next_tasks = additional_sample_admission(store, authority)
    store.register_campaign_budget_admission(next_authority)
    for index in range(7):
        claim = store.claim(f"worker-{index}", 60)
        assert claim is not None
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 0),
        )
    eighth = store.claim("worker-eighth", 60)
    assert eighth is not None
    boundary = Barrier(2)

    def terminalize() -> str:
        boundary.wait()
        store.mark_terminal(
            eighth["task_id"], eighth["attempt_id"],
            eighth["lease_token"], "NO_SEED", no_seed_result(eighth, 2),
        )
        return "terminal"

    def create() -> str:
        boundary.wait()
        try:
            assert store.create_tasks(next_tasks[:1]) == (1, 1)
        except ValueError as error:
            assert "campaign decision blocks" in str(error)
            return "blocked"
        return "inserted-before-terminal"

    with ThreadPoolExecutor(max_workers=2) as executor:
        terminal_future = executor.submit(terminalize)
        create_future = executor.submit(create)
        assert terminal_future.result() == "terminal"
        creation_result = create_future.result()

    decisions = store.campaign_decisions(
        mission_id="first-letters",
        policy_version=authority["execution_bindings"]["policy_version"],
    )
    assert decisions[-1]["decision"] == "PAUSE_CANDIDATE_STARVATION"
    pending = store.status()["tasks"].get("PENDING", 0)
    assert (creation_result, pending) in {
        ("blocked", 0),
        ("inserted-before-terminal", 1),
    }


def test_postgres_schema_and_store_expose_immutable_campaign_decisions():
    migration = (
        ROOT / "framework/stages/01-segmentation/fleet/migrations/"
        "001_postgresql.sql"
    ).read_text(encoding="utf-8")
    normalized = " ".join(migration.split())
    assert "CREATE TABLE IF NOT EXISTS segment_campaign_decisions" in normalized
    assert (
        "CREATE TABLE IF NOT EXISTS segment_campaign_resume_authorizations"
        in normalized
    )
    assert (
        "CREATE TABLE IF NOT EXISTS segment_campaign_resume_principal_attestations"
        in normalized
    )
    assert (
        "UNIQUE(mission_id, policy_version, evaluation_kind, evaluation_index)"
        in normalized
    )
    assert (
        "VALUES (15, 'authenticated campaign resume principal attestations')"
        in normalized
    )
    assert "segment_campaign_resume_by_predecessor" in normalized
    assert callable(getattr(PostgresFleetStore, "campaign_decisions", None))
    assert callable(getattr(PostgresFleetStore, "campaign_active_decision", None))
    assert callable(getattr(PostgresFleetStore, "_refresh_campaign_decisions", None))
    assert callable(getattr(
        PostgresFleetStore,
        "register_campaign_resume_principal_attestation",
        None,
    ))


def test_sqlite_resume_requires_new_policy_and_hash_bound_material_change(
    tmp_path,
):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    authority = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(authority)
    tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, authority)
         for cell_id in authority["prefix_cell_ids"]],
        authority,
    )
    assert store.create_tasks(tasks) == (8, 8)
    for index in range(8):
        claim = store.claim(f"worker-{index}", 60)
        assert claim is not None
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 2 if index == 7 else 0),
        )
    decision = store.campaign_decisions(
        mission_id="first-letters",
        policy_version=authority["execution_bindings"]["policy_version"],
    )[-1]
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        authoritative_attempts, registered_admissions = (
            store._campaign_decision_inputs(
                connection,
                mission_id="first-letters",
                policy_version=authority["execution_bindings"][
                    "policy_version"],
            )
        )
        connection.rollback()
    resumed, resumed_tasks = resumed_sample_admission(store, authority)

    with pytest.raises(ValueError, match="resume authorization"):
        store.register_campaign_budget_admission(resumed)
    planner_only = resume_authorization(
        authority, resumed, decision, field="planner")
    store.register_campaign_resume_principal_attestation(
        planner_only, authenticated_principal="campaign-owner")
    with pytest.raises(ValueError, match="material|planner|causal"):
        store.register_campaign_budget_admission(
            resumed, resume_authorization=planner_only)

    authorization = resume_authorization(authority, resumed, decision)
    with pytest.raises(ValueError, match="trusted.*attestation"):
        campaign_decision.validate_campaign_resume_authorization(
            authorization,
            prior_admission=authority,
            new_admission=resumed,
            prior_decision=decision,
            policy=POLICY,
        )
    with pytest.raises(ValueError, match="trusted.*attestation|attested"):
        store.register_campaign_budget_admission(
            resumed, resume_authorization=authorization)
    register_attestation = getattr(
        store, "register_campaign_resume_principal_attestation", None)
    assert callable(register_attestation), (
        "FleetStore.register_campaign_resume_principal_attestation is not implemented")
    with pytest.raises(TypeError):
        register_attestation(authorization)
    with pytest.raises(ValueError, match="panel-authenticated"):
        register_attestation(
            authorization, authenticated_principal="caller-forged-admin")
    register_attestation(
        authorization, authenticated_principal="campaign-owner")
    with pytest.raises(ValueError, match="authoritative persisted pause"):
        campaign_decision.validate_campaign_resume_authorization(
            authorization,
            prior_admission=authority,
            new_admission=resumed,
            prior_decision=decision,
            policy=POLICY,
            trusted_authorization_sha256s={
                authorization["authorization_sha256"]},
        )
    with pytest.raises(ValueError, match="authoritative persisted pause"):
        campaign_decision.validate_campaign_resume_authorization(
            authorization,
            prior_admission=authority,
            new_admission=resumed,
            prior_decision=decision,
            policy=POLICY,
            authoritative_attempts=authoritative_attempts,
            registered_admissions=[authority, authority],
            trusted_authorization_sha256s={
                authorization["authorization_sha256"]},
        )
    arbitrary_evidence = copy.deepcopy(authorization)
    arbitrary_evidence["material_changes"][0]["evidence_sha256"] = "7" * 64
    arbitrary_evidence = rehash_resume_authorization(arbitrary_evidence)
    with pytest.raises(ValueError, match="evidence|preflight|budget"):
        campaign_decision.validate_campaign_resume_authorization(
            arbitrary_evidence, prior_admission=authority,
            new_admission=resumed, prior_decision=decision, policy=POLICY,
            authoritative_attempts=authoritative_attempts,
            registered_admissions=registered_admissions,
            trusted_authorization_sha256s={
                arbitrary_evidence["authorization_sha256"]})

    threshold_only = copy.deepcopy(resumed)
    threshold_only["execution_bindings"]["m7_threshold"] = 0.3
    threshold_only["admission_sha256"] = content_sha256({
        key: value for key, value in threshold_only.items()
        if key != "admission_sha256"
    })
    threshold_authorization = resume_authorization(
        authority, threshold_only, decision,
        field="calibrated_m7_threshold")
    with pytest.raises(ValueError, match="validator|verified evidence"):
        campaign_decision.validate_campaign_resume_authorization(
            threshold_authorization, prior_admission=authority,
            new_admission=threshold_only, prior_decision=decision, policy=POLICY,
            authoritative_attempts=authoritative_attempts,
            registered_admissions=registered_admissions,
            trusted_authorization_sha256s={
                threshold_authorization["authorization_sha256"]})

    assert store.register_campaign_budget_admission(
        resumed, resume_authorization=authorization) == resumed
    # A bootstrap retry must be idempotent after the new authority and its
    # authorization were committed together. Refusing the exact replay would
    # turn an ambiguous client timeout into an unrecoverable operator error.
    assert store.register_campaign_budget_admission(
        resumed, resume_authorization=authorization) == resumed
    assert store.create_tasks(resumed_tasks[:1]) == (1, 1)
    with store.connect() as connection:
        persisted = connection.execute(
            "SELECT authorization_json FROM campaign_resume_authorizations"
        ).fetchall()
    assert len(persisted) == 1
    assert json.loads(persisted[0]["authorization_json"])[
        "authorization_sha256"] == authorization["authorization_sha256"]


def test_pause_cannot_be_resumed_from_an_unrelated_registered_admission(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    governing = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(governing)
    tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, governing)
         for cell_id in governing["prefix_cell_ids"]],
        governing,
    )
    assert store.create_tasks(tasks) == (8, 8)
    unrelated, _ = additional_sample_admission(
        store, governing, sample_id="PHerc999")
    store.register_campaign_budget_admission(unrelated)
    for index in range(8):
        claim = store.claim(f"worker-{index}", 60)
        assert claim is not None
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 2 if index == 7 else 0),
        )
    pause = store.campaign_decisions(
        mission_id="first-letters",
        policy_version=governing["execution_bindings"]["policy_version"],
    )[-1]
    successor, _ = additional_sample_admission(
        store, governing, sample_id="PHerc777")
    successor["execution_bindings"].update({
        "policy_version": "search-v2", "provider": "vc3d-mcp-v2"})
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key != "admission_sha256"
    })
    forged_authorization = resume_authorization(
        unrelated, successor, pause, field="discovery_provider")
    store.register_campaign_resume_principal_attestation(
        forged_authorization, authenticated_principal="campaign-owner")
    with pytest.raises(ValueError, match="govern|pause"):
        store.register_campaign_budget_admission(
            successor, resume_authorization=forged_authorization)


def test_sqlite_resume_rederives_pause_from_rows_inside_the_mission_lock(
    tmp_path, monkeypatch,
):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    first_source = store_source(store)
    trigger_authority = resized_admission(
        budget_admission(tmp_path / "mission", first_source), 7)
    denominator_authority, _unused = additional_sample_admission(
        store, trigger_authority, sample_id="PHerc999")
    denominator_authority = resized_admission(denominator_authority, 1)
    store.register_campaign_budget_admission(trigger_authority)
    store.register_campaign_budget_admission(denominator_authority)

    trigger_tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(first_source, cell_id, trigger_authority)
         for cell_id in trigger_authority["prefix_cell_ids"]],
        trigger_authority,
    )
    denominator_source = denominator_authority["execution_bindings"][
        "source_snapshot_id"]
    denominator_task = budget_task(
        denominator_source,
        denominator_authority["prefix_cell_ids"][0],
        denominator_authority,
    )
    denominator_task["sample_id"] = denominator_authority["sample_id"]
    denominator_tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [denominator_task], denominator_authority)
    assert store.create_tasks(trigger_tasks) == (7, 7)
    assert store.create_tasks(denominator_tasks) == (1, 1)

    for index in range(8):
        claim = store.claim(f"worker-{index}", 60)
        assert claim is not None
        raw_m7 = 2 if claim["sample_id"] == "PHerc999" else 0
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, raw_m7),
        )
    pause = store.campaign_decisions(
        mission_id="first-letters",
        policy_version=trigger_authority["execution_bindings"][
            "policy_version"],
    )[-1]
    assert pause["governing_admission_sha256s"] == [
        trigger_authority["admission_sha256"]]

    forged_pause = copy.deepcopy(pause)
    for bound_attempt in forged_pause["scientific_terminal_attempts"]:
        if bound_attempt["attempt_id"] in forged_pause["trigger_attempt_ids"]:
            bound_attempt.update({
                "sample_id": denominator_authority["sample_id"],
                "admission_sha256": denominator_authority[
                    "admission_sha256"],
                "budget_receipt_sha256": denominator_authority[
                    "receipt_sha256"],
            })
    forged_pause["trigger_governing_admission_sha256s"] = [
        denominator_authority["admission_sha256"]]
    forged_pause["governing_admission_sha256s"] = [
        denominator_authority["admission_sha256"]]
    forged_pause["receipt_sha256"] = content_sha256({
        key: value for key, value in forged_pause.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """UPDATE campaign_decisions
                  SET receipt_sha256=?,receipt_json=?
                WHERE receipt_sha256=?""",
            (
                forged_pause["receipt_sha256"],
                json.dumps(forged_pause, sort_keys=True, separators=(",", ":")),
                pause["receipt_sha256"],
            ),
        )
        connection.commit()

    successor = copy.deepcopy(denominator_authority)
    successor["receipt_sha256"] = "f" * 64
    successor["execution_bindings"].update({
        "policy_version": "search-v2",
        "provider": "vc3d-mcp-v2",
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key != "admission_sha256"
    })
    authorization = resume_authorization(
        denominator_authority, successor, forged_pause,
        field="discovery_provider",
    )
    store.register_campaign_resume_principal_attestation(
        authorization, authenticated_principal="campaign-owner")
    original_inputs = store._campaign_decision_inputs
    transaction_observations: list[bool] = []

    def observed_inputs(connection, **scope):
        transaction_observations.append(connection.in_transaction)
        return original_inputs(connection, **scope)

    monkeypatch.setattr(store, "_campaign_decision_inputs", observed_inputs)
    with pytest.raises(ValueError, match="authoritative persisted pause"):
        store.register_campaign_budget_admission(
            successor, resume_authorization=authorization)
    assert transaction_observations == [True]
    with store.connect() as connection:
        registered = connection.execute(
            """SELECT 1 FROM campaign_budget_admissions
                WHERE admission_sha256=?""",
            (successor["admission_sha256"],),
        ).fetchone()
    assert registered is None


def test_scroll_local_m7_source_cannot_authorize_a_cross_sample_resume(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    prior = budget_admission(tmp_path / "mission", source_id)
    successor, _ = additional_sample_admission(
        store, prior, sample_id="PHerc999")
    successor["execution_bindings"].update({
        "policy_version": "search-v2",
        "m7_sha256": "8" * 64,
        "m7_uri_sha256": "7" * 64,
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key != "admission_sha256"
    })
    with pytest.raises(ValueError, match="same sample|scroll-local"):
        campaign_decision.campaign_resume_material_evidence_sha256(
            prior, successor, "m7_source")


def test_mission_pause_cannot_be_bypassed_by_a_new_sample_or_policy(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    authority = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(authority)
    tasks = campaign_decision.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, authority)
         for cell_id in authority["prefix_cell_ids"]],
        authority,
    )
    assert store.create_tasks(tasks) == (8, 8)
    for index in range(8):
        claim = store.claim(f"worker-{index}", 60)
        assert claim is not None
        store.mark_terminal(
            claim["task_id"], claim["attempt_id"], claim["lease_token"],
            "NO_SEED", no_seed_result(claim, 2 if index == 7 else 0),
        )
    pause = store.campaign_decisions(
        mission_id="first-letters",
        policy_version=authority["execution_bindings"]["policy_version"],
    )[-1]

    successor, successor_tasks = additional_sample_admission(
        store, authority, sample_id="PHerc999")
    successor["execution_bindings"].update({
        "policy_version": "search-v2",
        "provider": "vc3d-mcp-v2",
    })
    successor["admission_sha256"] = content_sha256({
        key: value for key, value in successor.items()
        if key != "admission_sha256"
    })
    successor_tasks = [
        budget_task(
            successor["execution_bindings"]["source_snapshot_id"], cell_id,
            successor,
        )
        for cell_id in successor["prefix_cell_ids"]
    ]
    for task in successor_tasks:
        task["sample_id"] = successor["sample_id"]
        task["candidate_discovery"]["prediction_uri"] = "fixture://m7"
    successor_tasks = campaign_decision.bind_campaign_budget_to_tasks(
        successor_tasks, successor)

    with pytest.raises(ValueError, match="pause|resume authorization|active policy"):
        store.register_campaign_budget_admission(successor)

    authorization = resume_authorization(
        authority, successor, pause, field="discovery_provider")
    store.register_campaign_resume_principal_attestation(
        authorization, authenticated_principal="campaign-owner")
    assert store.register_campaign_budget_admission(
        successor, resume_authorization=authorization) == successor
    active = store.campaign_active_decision(mission_id="first-letters")
    assert active is not None
    assert active["policy_chain"] == [
        authority["execution_bindings"]["policy_version"], "search-v2"]
    assert active["policy_version"] == "search-v2"
    assert active["decision"] == "CONTINUE"
    assert active["no_m7_numerator"] == 0
    assert active["scientific_terminal_attempt_count"] == 0
    assert active["scientific_terminal_denominator"] == 8
    assert store.campaign_decisions(mission_id="first-letters") == [pause]
    assert store.create_tasks(successor_tasks[:1]) == (1, 1)

    competing, _tasks = additional_sample_admission(
        store, authority, sample_id="PHerc777")
    competing["execution_bindings"].update({
        "policy_version": "search-v3", "provider": "vc3d-mcp-v3"})
    competing["admission_sha256"] = content_sha256({
        key: value for key, value in competing.items()
        if key != "admission_sha256"
    })
    competing_authorization = resume_authorization(
        authority, competing, pause, field="discovery_provider")
    store.register_campaign_resume_principal_attestation(
        competing_authorization, authenticated_principal="campaign-owner")
    with pytest.raises(ValueError, match="active policy|successor|chain"):
        store.register_campaign_budget_admission(
            competing, resume_authorization=competing_authorization)


def _authorized_seed_probe_evidence() -> dict:
    value = {
        "schema": "campaignx.first_letters_authorized_seed_probe_mode_evidence.v1",
        "prior_decision_receipt_sha256": "0" * 64,
        "prior_admission_sha256": "1" * 64,
        "new_admission_sha256": "2" * 64,
        "predecessor_policy_file_sha256": "3" * 64,
        "successor_policy_file_sha256": "4" * 64,
        "old_seed_probe_mode": "off",
        "new_seed_probe_mode": "select",
        "unchanged_source_grid_sha256": "5" * 64,
        "benchmark_authorization_v2": {
            "schema": "campaignx.seed_probe_benchmark_authorization.v2",
            "authorization_sha256": "6" * 64,
            "deployed_revision": "7" * 40,
        },
        "review_owner": "campaign-owner",
        "allow_unvalidated": False,
    }
    value["evidence_sha256"] = content_sha256(value)
    return value


def test_authorized_seed_probe_resume_binds_pause_predecessor_successor_and_benchmark_v2():
    assert hasattr(campaign_decision, "validate_authorized_seed_probe_mode_evidence")
    prior_admission = {"admission_sha256": "1" * 64, "seed_probe_mode": "off", "source_grid_sha256": "5" * 64}
    new_admission = {"admission_sha256": "2" * 64, "seed_probe_mode": "select", "source_grid_sha256": "5" * 64}
    decision = {"receipt_sha256": "0" * 64}
    policy = {"predecessor_profile_file_sha256": "3" * 64, "profile_file_sha256": "4" * 64}
    result = campaign_decision.validate_authorized_seed_probe_mode_evidence(
        _authorized_seed_probe_evidence(), prior_admission=prior_admission,
        new_admission=new_admission, prior_decision=decision, policy=policy,
        benchmark_authorization_v2={"schema": "campaignx.seed_probe_benchmark_authorization.v2", "authorization_sha256": "6" * 64, "deployed_revision": "7" * 40},
    )
    assert result["old_seed_probe_mode"] == "off"
    assert result["new_seed_probe_mode"] == "select"


def test_authorized_seed_probe_resume_rejects_planner_only_or_unrelated_material_change():
    assert hasattr(campaign_decision, "validate_authorized_seed_probe_mode_evidence")
    evidence = _authorized_seed_probe_evidence()
    evidence["new_seed_probe_mode"] = "off"
    evidence["evidence_sha256"] = content_sha256({key: row for key, row in evidence.items() if key != "evidence_sha256"})
    with pytest.raises(ValueError):
        campaign_decision.validate_authorized_seed_probe_mode_evidence(
            evidence,
            prior_admission={"admission_sha256": "1" * 64, "seed_probe_mode": "off", "source_grid_sha256": "5" * 64},
            new_admission={"admission_sha256": "2" * 64, "seed_probe_mode": "off", "source_grid_sha256": "5" * 64},
            prior_decision={"receipt_sha256": "0" * 64},
            policy={"predecessor_profile_file_sha256": "3" * 64, "profile_file_sha256": "4" * 64},
            benchmark_authorization_v2=evidence["benchmark_authorization_v2"],
        )


def test_old_resume_hash_fallback_still_rejects_authorized_seed_probe_mode():
    with pytest.raises(ValueError, match="verified external evidence"):
        campaign_decision.campaign_resume_material_evidence_sha256(
            {}, {}, "authorized_seed_probe_mode"
        )
