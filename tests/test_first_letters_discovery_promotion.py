from __future__ import annotations

import inspect
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import campaign_decision, seed_probe
from fleet.common import content_sha256, stable_id
from fleet.store import FleetStore
from test_first_letters_discovery_contracts import (
    _benchmark_decision_v2,
    _benchmark_manifest_v2,
)


def _gate() -> dict:
    value = {
        "schema": "campaignx.first_letters_task9_discovery_gate.v1",
        "mission_id": "mission-a", "readiness_sha256": "1" * 64,
        "control_binding_sha256": "2" * 64, "wave_receipt_sha256": "3" * 64,
        "policy_chain_sha256": "4" * 64, "deployed_revision": "1" * 40,
        "allow_unvalidated": False,
    }
    value["gate_sha256"] = content_sha256(value)
    return value


def _benchmark():
    manifest = _benchmark_manifest_v2()
    return seed_probe.validate_seed_probe_benchmark_receipt_v2(
        _benchmark_decision_v2(manifest), execution_manifest=manifest
    )


def _candidate():
    candidate = seed_probe.project_provider_candidate_v1(
        {
            "candidate_id": "candidate-a", "cell_id": "cell-a",
            "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56},
            "score": 0.9,
        },
        provider_response_sha256="a" * 64,
    )
    candidate.update({"ct_terminal_sha256": "b" * 64, "clearance_terminal_sha256": "c" * 64})
    return candidate


def _budget_admission():
    value = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": "mission-a", "sample_id": "PHercA", "receipt_sha256": "d" * 64,
        "preflight_receipt_sha256": "e" * 64,
        "preflight_sanitized_receipt_sha256": "f" * 64,
        "approved_task_count": 1, "prefix_cell_ids": ["cell-a"],
        "prefix_sha256": content_sha256(["cell-a"]), "order_seed_sha256": "0" * 64,
        "population_order_sha256": "1" * 64,
        "execution_bindings": {
            "source_snapshot_id": "source-a", "grid_version": "grid-v1",
            "policy_version": "first-letters-search@1.0.0",
            "p0_artifact_id": "p0-a", "p0_artifact_sha256": "2" * 64,
            "catalog_snapshot_sha256": "3" * 64,
        },
    }
    value["admission_sha256"] = content_sha256(value)
    return value


def _parent():
    admission = _budget_admission()
    return {
        "task_id": "parent-a", "attempt_id": stable_id(
            "attempt", {"task_id": "parent-a", "attempt_number": 1}
        ),
        "mission_id": "mission-a", "sample_id": "PHercA",
        "source_snapshot_id": "source-a", "grid_version": "grid-v1",
        "cell_id": "cell-a", "policy_version": "first-letters-search@1.0.0",
        "selection_rank": 0, "campaign_budget_admission_sha256": admission["admission_sha256"],
        "p0_artifact_id": "p0-a", "p0_artifact_sha256": "2" * 64,
        "catalog_snapshot_sha256": "3" * 64,
    }


def _receipt():
    value = {
        "schema": "campaignx.first_letters_discovery_receipt.v1",
        "receipt_id": "receipt-a", "mission_id": "mission-a",
        "parent_task_id": "parent-a", "parent_attempt_id": _parent()["attempt_id"],
        "artifact_sha256": "4" * 64, "namespace": "NONCANONICAL_DISCOVERY",
        "allow_unvalidated": False,
    }
    value["receipt_sha256"] = content_sha256(value)
    return value


def _normal_lock():
    assert hasattr(seed_probe, "load_first_letters_normal_growth_lock")
    return seed_probe.load_first_letters_normal_growth_lock(
        source_snapshot_id="source-a", coordinate=_candidate()["promotion_coordinate_ct_l0_xyz"],
        coordinate_sha256=_candidate()["promotion_coordinate_sha256"],
        deployed_revision="1" * 40, retry_budget=2,
    )


def _admission():
    return campaign_decision.authorize_promotion_child(
        parent_task=_parent(), registered_budget_admission=_budget_admission(),
        active_policy_chain={"active_policy_version": "first-letters-search@1.1.0", "policy_chain_sha256": "4" * 64, "paused": False},
        benchmark_authorization_v2=_benchmark(), discovery_receipt=_receipt(),
        selected_candidate=_candidate(), normal_growth_lock=_normal_lock(), task9_gate=_gate(),
    )


def test_select_is_dormant_without_current_task9_control_readiness_and_wave_gate():
    assert hasattr(campaign_decision, "authorize_promotion_child")
    with pytest.raises(ValueError, match="TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED"):
        campaign_decision.authorize_promotion_child(
            parent_task=_parent(), registered_budget_admission=_budget_admission(),
            active_policy_chain={}, benchmark_authorization_v2=_benchmark(),
            discovery_receipt=_receipt(), selected_candidate=_candidate(),
            normal_growth_lock=_normal_lock(), task9_gate=None,
        )


def test_promotion_child_reuses_one_budgeted_opportunity_without_new_rank():
    admission = _admission()
    assert admission["selection_rank"] == 0
    assert admission["statistical_rank_delta"] == 0
    assert admission["scientific_denominator_limit"] == 1


@pytest.mark.parametrize("field", ["child_task_id", "selection_rank", "source_snapshot_id", "grid_version", "cell_id", "successor_policy_version"])
def test_promotion_child_rejects_second_child_rank_source_grid_cell_or_policy_drift(field):
    admission = _admission()
    child = campaign_decision.build_promotion_child_task(admission)
    child[field] = "drift" if field != "selection_rank" else 1
    with pytest.raises(ValueError):
        campaign_decision.validate_promotion_child_task(
            child, admission=admission,
            registered_budget_admission=_budget_admission(),
            authoritative_source_snapshot={"source_snapshot_id": "source-a"},
        )


def _store(tmp_path):
    from test_first_letters_discovery_evidence_store import (
        _claim_job,
        _complete as _complete_evidence,
        _store as _evidence_store,
    )

    budget = _budget_admission()
    budget["execution_bindings"]["grid_version"] = "first-letters-grid-v1"
    budget["admission_sha256"] = content_sha256({
        key: value for key, value in budget.items()
        if key != "admission_sha256"
    })
    opportunity_id = stable_id("first-letters-opportunity", {
        "admission_sha256": budget["admission_sha256"],
        "selection_rank": 0,
    })
    parent_authority = {
        "selection_rank": 0,
        "campaign_budget_admission_sha256": budget["admission_sha256"],
        "p0_artifact_id": budget["execution_bindings"]["p0_artifact_id"],
        "p0_artifact_sha256":
            budget["execution_bindings"]["p0_artifact_sha256"],
    }
    promotion_authority = {
        "active_policy_chain": {
            "active_policy_version": "first-letters-search@1.1.0",
            "policy_chain_sha256": "4" * 64,
            "paused": False,
        },
        "benchmark_authorization_v2": _benchmark(),
    }
    store, profile_bytes, reservation = _evidence_store(
        tmp_path,
        scientific_opportunity_id=opportunity_id,
        p0_artifact_id=budget["execution_bindings"]["p0_artifact_id"],
        p0_artifact_sha256=budget["execution_bindings"]["p0_artifact_sha256"],
        promotion_authority=promotion_authority,
        parent_authority=parent_authority,
        registered_budget_admission=budget,
        claim_parent=True,
    )
    store._task9_discovery_gate_resolver = lambda _mission_id: _gate()
    handle = _claim_job(store, reservation)
    completed = _complete_evidence(store, handle)
    return store, completed["evidence_set_id"]


def test_promotion_atomically_commits_authority_child_and_parent_terminal(tmp_path):
    store, evidence_set_id = _store(tmp_path)
    result = store.begin_discovery_promotion(request_id="promotion-a", evidence_set_id=evidence_set_id, task9_gate=_gate())
    assert result["authority"]["terminal_state"] == "CHILD_CREATED_PARENT_TERMINAL"
    assert result["child"]["state"] == "PENDING"
    assert result["parent"]["state"] == "DISCOVERY_PROMOTED"


def test_promotion_mutation_accepts_evidence_id_not_caller_admission():
    parameters = inspect.signature(
        FleetStore.begin_discovery_promotion
    ).parameters
    assert "evidence_set_id" in parameters
    assert "admission" not in parameters


def test_promotion_mutation_rejects_caller_admission_and_reads_registry(tmp_path):
    store, evidence_set_id = _store(tmp_path)
    with pytest.raises(TypeError):
        store.begin_discovery_promotion(
            request_id="forged", admission=_admission(),
            task9_gate=_gate(),
        )
    forged_response = json.dumps({
        "schema": "campaignx.first_letters_candidate_provider_response.v1",
        "prediction_identity": {}, "candidates": [],
    }, sort_keys=True, separators=(",", ":")).encode()
    with store.connect() as connection:
        connection.execute(
            """UPDATE first_letters_discovery_evidence_files
                  SET payload=?,byte_count=?,sha256=?
                WHERE evidence_set_id=?
                  AND role='CANDIDATE_PROVIDER_RESPONSE'""",
            (
                forged_response, len(forged_response),
                content_sha256(json.loads(forged_response)), evidence_set_id,
            ),
        )
    with pytest.raises(ValueError):
        store.begin_discovery_promotion(
            request_id="promotion-a", evidence_set_id=evidence_set_id,
            task9_gate=_gate(),
        )
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_promotions"
        ).fetchone()[0] == 0


def _raise_at(target: str):
    def failpoint(name: str) -> None:
        if name == target:
            if name == "promotion.commit_outcome_unknown":
                raise RuntimeError(
                    "CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK"
                )
            raise RuntimeError(name)

    return failpoint


@pytest.mark.parametrize("failpoint", [
    "promotion.before_authority_insert",
    "promotion.after_authority_insert_before_child_insert",
    "promotion.after_child_insert_before_parent_terminal",
    "promotion.after_parent_terminal_before_commit",
    "promotion.before_commit",
])
def test_each_precommit_promotion_failpoint_leaves_literal_zero_zero_unchanged_state(
    tmp_path, failpoint,
):
    store, evidence_set_id = _store(tmp_path)
    with store.connect() as connection:
        parent_state_before = connection.execute(
            "SELECT state FROM tasks WHERE task_id=?", ("task-a",)
        ).fetchone()[0]
        attempt_state_before = connection.execute(
            "SELECT state FROM attempts WHERE attempt_id=?",
            (stable_id("attempt", {
                "task_id": "task-a", "attempt_number": 1,
            }),),
        ).fetchone()[0]
    with pytest.raises(RuntimeError, match=failpoint):
        store.begin_discovery_promotion(
            request_id="promotion-a", evidence_set_id=evidence_set_id,
            task9_gate=_gate(),
            promotion_failpoint=_raise_at(failpoint),
        )
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_promotions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_id<>?", ("task-a",)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT state FROM tasks WHERE task_id=?", ("task-a",)
        ).fetchone()[0] == parent_state_before
        assert connection.execute(
            "SELECT state FROM attempts WHERE attempt_id=?",
            (stable_id("attempt", {
                "task_id": "task-a", "attempt_number": 1,
            }),),
        ).fetchone()[0] == attempt_state_before


def test_commit_unknown_stops_until_exact_three_fact_readback(tmp_path):
    store, evidence_set_id = _store(tmp_path)
    with pytest.raises(
        RuntimeError, match="CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK",
    ):
        store.begin_discovery_promotion(
            request_id="promotion-a", evidence_set_id=evidence_set_id,
            task9_gate=_gate(),
            promotion_failpoint=_raise_at(
                "promotion.commit_outcome_unknown"
            ),
        )
    readback = store.read_discovery_promotion("mission-a", "promotion-a")
    assert readback["child"]["state"] == "PENDING"
    assert readback["parent"]["state"] == "DISCOVERY_PROMOTED"


def test_after_commit_response_loss_recovers_exact_authority_child_and_parent_bytes(
    tmp_path,
):
    store, evidence_set_id = _store(tmp_path)
    with pytest.raises(
        RuntimeError, match="promotion.after_commit_before_response",
    ):
        store.begin_discovery_promotion(
            request_id="promotion-a", evidence_set_id=evidence_set_id,
            task9_gate=_gate(),
            promotion_failpoint=_raise_at(
                "promotion.after_commit_before_response"
            ),
        )
    first = store.read_discovery_promotion("mission-a", "promotion-a")
    assert store.begin_discovery_promotion(
        request_id="promotion-a", evidence_set_id=evidence_set_id, task9_gate=_gate(),
    ) == first


def test_sqlite_concurrent_promotion_yields_one_authority_one_child_one_parent_terminal(
    tmp_path,
):
    store, evidence_set_id = _store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _index: store.begin_discovery_promotion(
                request_id="promotion-a", evidence_set_id=evidence_set_id,
                task9_gate=_gate(),
            ),
            (0, 1),
        ))
    assert results[0] == results[1]
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_promotions"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_id<>?", ("task-a",)
        ).fetchone()[0] == 1


def test_immutable_authority_has_no_attempt_id_and_claim_appends_binding(tmp_path):
    store, evidence_set_id = _store(tmp_path)
    result = store.begin_discovery_promotion(request_id="promotion-a", evidence_set_id=evidence_set_id, task9_gate=_gate())
    assert "attempt_id" not in result["authority"]
    binding = store.append_discovery_promotion_attempt_binding(
        promotion_id=result["authority"]["promotion_id"], attempt_number=1,
        attempt_id="child-attempt-1", claim_event_sha256="5" * 64,
        predecessor_attempt_id=None, retry_reason=None,
    )
    assert binding["attempt_id"] == "child-attempt-1"
    assert result == store.read_discovery_promotion("mission-a", "promotion-a")


def test_platform_retry_appends_binding_within_locked_budget_without_new_denominator_entry(tmp_path):
    store, evidence_set_id = _store(tmp_path)
    result = store.begin_discovery_promotion(request_id="promotion-a", evidence_set_id=evidence_set_id, task9_gate=_gate())
    pid = result["authority"]["promotion_id"]
    store.append_discovery_promotion_attempt_binding(promotion_id=pid, attempt_number=1, attempt_id="a1", claim_event_sha256="5" * 64, predecessor_attempt_id=None, retry_reason=None)
    retry = store.append_discovery_promotion_attempt_binding(promotion_id=pid, attempt_number=2, attempt_id="a2", claim_event_sha256="6" * 64, predecessor_attempt_id="a1", retry_reason="WORKER_FAILURE")
    assert retry["attempt_number"] == 2
    assert retry["scientific_denominator_delta"] == 0


def test_second_live_attempt_or_retry_after_scientific_terminal_is_rejected(tmp_path):
    store, evidence_set_id = _store(tmp_path)
    result = store.begin_discovery_promotion(request_id="promotion-a", evidence_set_id=evidence_set_id, task9_gate=_gate())
    pid = result["authority"]["promotion_id"]
    store.append_discovery_promotion_attempt_binding(promotion_id=pid, attempt_number=1, attempt_id="a1", claim_event_sha256="5" * 64, predecessor_attempt_id=None, retry_reason=None)
    with pytest.raises(ValueError):
        store.append_discovery_promotion_attempt_binding(promotion_id=pid, attempt_number=2, attempt_id="a2", claim_event_sha256="6" * 64, predecessor_attempt_id="a1", retry_reason=None)


def test_one_opportunity_contributes_exactly_one_scientific_terminal_across_parent_child_retries():
    rows = [
        {"role": "parent", "state": "DISCOVERY_PROMOTED", "scientific_terminal": False},
        {"role": "child", "state": "GROW_FAILED", "platform_excluded": True, "scientific_terminal": False},
        {"role": "child", "state": "NO_SEED", "platform_excluded": False, "scientific_terminal": True},
    ]
    assert campaign_decision.classify_discovery_scientific_opportunity(rows)["denominator_contribution"] == 1


@pytest.mark.parametrize("state,count", [
    ("PROBE_REVIEW_PENDING", 0), ("DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER", 1),
    ("DISCOVERY_REJECTED_CANDIDATES", 1), ("DISCOVERY_PROMOTED", 0),
])
def test_task5_classifier_handles_every_literal_discovery_parent_child_state(state, count):
    assert campaign_decision.classify_discovery_scientific_opportunity([
        {"role": "parent", "state": state, "scientific_terminal": count == 1}
    ])["denominator_contribution"] == count


def test_promoted_child_is_distinct_fresh_normal_grow_with_exact_locked_profiles():
    admission = _admission(); child = campaign_decision.build_promotion_child_task(admission)
    assert child["task_id"] != _parent()["task_id"]
    assert child["normal_growth_lock"] == admission["normal_growth_lock"]
    assert child["normal_growth_lock"]["fresh_start"] is True


def test_promoted_child_contains_no_resume_probe_uri_checkpoint_or_probe_generation_fields():
    child = campaign_decision.build_promotion_child_task(_admission())
    for forbidden in ("resume_from", "resume_artifact", "probe_uri", "probe_checkpoint", "probe_generations"):
        assert forbidden not in child
    lock = child["normal_growth_lock"]
    assert lock["resume_from"] is None
    assert lock["resume_artifact"] is None
    assert lock["probe_checkpoint"] is None


def test_parent_discovery_artifact_never_finalizes_or_registers_as_canonical():
    admission = _admission()
    assert admission["discovery_namespace"] == "NONCANONICAL_DISCOVERY"
    assert admission["normal_growth_lock"]["fresh_start"] is True
    assert admission["normal_growth_lock"]["resume_artifact"] is None


@pytest.mark.parametrize("boundary", ["authority", "task", "claim", "finalize"])
def test_allow_unvalidated_true_or_missing_fails_at_authority_task_claim_and_finalize(boundary):
    admission = _admission()
    if boundary == "authority":
        admission["allow_unvalidated"] = True
        with pytest.raises(ValueError): campaign_decision.validate_promotion_child_admission(admission)
    else:
        child = campaign_decision.build_promotion_child_task(admission)
        child["allow_unvalidated"] = True if boundary != "claim" else None
        with pytest.raises(ValueError): campaign_decision.validate_promotion_child_task(child, admission=admission, registered_budget_admission=_budget_admission(), authoritative_source_snapshot={"source_snapshot_id": "source-a"})
