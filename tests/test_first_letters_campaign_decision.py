"""Task budgets come from measured candidate survival, never a fixed task cap."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256  # noqa: E402
from fleet.generator import DEFAULT_ENVELOPE, generate_tasks_for_snapshot  # noqa: E402
from fleet.store import FleetStore  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.cli import build_parser, _cli_p1_campaign_admission  # noqa: E402
from framework.contracts import mission as mission_contract  # noqa: E402


POLICY_PATH = (
    ROOT / "framework/profiles/01-segmentation/"
    "first-letters-campaign-decision-policy-1.2.0.json"
)


def decision_module():
    try:
        return importlib.import_module("fleet.campaign_decision")
    except ModuleNotFoundError:
        pytest.fail("fleet.campaign_decision is not implemented")


def policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_new_controlled_admissions_resolve_policy_1_2_without_rewriting_history():
    assert decision_module().load_campaign_policy_profile()["profile_id"] == (
        "first-letters-campaign-decision-policy@1.2.0"
    )
    assert mission_contract.FIRST_LETTERS_CAMPAIGN_POLICY_ID == (
        "first-letters-campaign-decision-policy@1.2.0"
    )


@pytest.mark.parametrize("policy_id", [
    "first-letters-campaign-decision-policy@1.0.0",
    "first-letters-campaign-decision-policy@1.1.0",
])
def test_historical_controlled_bindings_remain_readable(policy_id, tmp_path):
    """A successor is for new admissions, not a rewrite of mission evidence."""
    manifest = mission_contract.create(
        tmp_path, mission_id=f"historic-{policy_id[-5]}", name="historic",
        scrolls=["PHerc0358"], created_by="alice",
        campaign_kind="FIRST_LETTERS_DISCOVERY", campaign_policy_id=policy_id,
        campaign_policy_sha256="a" * 64, deployed_revision="b" * 40,
    )
    assert mission_contract.load(tmp_path / manifest["mission_id"]) == manifest
    assert mission_contract.is_first_letters_discovery_manifest(manifest) is True


def compute_cap(maximum_tasks: int = 100) -> dict:
    return {
        "schema": "campaignx.first_letters_compute_cap.v1",
        "cap_id": "first-letters-local-cap-1",
        "mission_id": "first-letters",
        "sample_id": "PHerc358",
        "maximum_tasks": maximum_tasks,
    }


def preflight(
    *, population: int, usable_cells: int, sampled_cells: int | None = None,
    measurement_kind: str = "CENSUS", evidence_status: str = "CURRENT",
    source_errors: int = 0,
) -> dict:
    sampled = population if sampled_cells is None else sampled_cells
    receipt = {
        "schema": "campaignx.segment_candidate_coverage_preflight.sanitized.v1",
        "status": "COMPLETE_WITH_SOURCE_ERRORS" if source_errors else "COMPLETE",
        "measurement_kind": measurement_kind,
        "evidence_status": evidence_status,
        "evidence_status_reason": "fixture",
        "private_receipt_sha256": "a" * 64,
        "sample_id": "PHerc358",
        "source_snapshot_id": "source-1",
        "bindings": {
            "sample_id": "PHerc358", "mission_id": "first-letters",
            "source_snapshot_id": "source-1",
            "p0_artifact_id": "p0-1", "p0_artifact_sha256": "5" * 64,
            "p0_selection_version": "selection-1",
            "p0_selection_sha256": "6" * 64,
            "catalog_snapshot_sha256": "7" * 64,
            "source_content_lock_sha256": content_sha256({
                "schema": "fixture-source-lock"}),
            "ct_sha256": "2" * 64, "m7_sha256": "3" * 64,
            "m7_uri_sha256": hashlib.sha256(b"fixture://m7").hexdigest(),
            "coordinate_frame": "ct_l0_xyz", "voxel_size_um": 9.362,
            "shape_xyz": [32768, 32768, 32768],
            "grid_version": "first-letters-grid@1.0.0",
            "policy_version": "first-letters-preflight@1.0.0",
            "grid_step": 2048, "query_radius": 512, "parallelism": 2,
            "maximum_cells": 100,
            "provider": "vc3d-mcp", "m7_threshold": 0.2,
            "selection_strategy": "stratified-clearance-v1",
            "candidate_selection_policy": "score-cell-volume-clearance-v1",
            "seed_region_policy": "fixed-v1", "code_revision": "4" * 40,
        },
        "gates": {
            "cell_clearance": 0.0, "volume_clearance": 512,
            "candidate_interior_clearance": 2, "packet_candidate_limit": 2,
            "ct_material_support_gate": {
                "policy": "ome-zarr-nearby-material-v1", "level": 1,
                "radius_l0_voxels": 4, "minimum_nonzero_voxels": 1,
            },
        },
        "sampling_design": {
            "name": (
                "complete-eligible-grid-census-v1"
                if measurement_kind.removeprefix("INCOMPLETE_") == "CENSUS"
                else "deterministic-golden-coprime-rank1-grid-sample-v1"
            ),
            "measurement_kind": measurement_kind.removeprefix("INCOMPLETE_"),
            "population_grid_cells": population,
            "sampled_grid_cells": sampled,
        },
        "funnel": {
            "geometrically_eligible_cells": (
                population if measurement_kind.endswith("CENSUS") else None),
            "geometrically_eligible_cells_estimate": population,
            "geometrically_eligible_sampled_cells": sampled,
            "cells_surveyed_successfully": sampled - source_errors,
            "cells_failed_source": source_errors,
            "source_errors": source_errors,
            "packet_retained_candidates": usable_cells,
        },
        "spatial_bins": [{
            "bin_xyz": [0, 0, 0], "total_cells": population,
            "sampled_eligible_cells": sampled,
            "surveyed_cells": sampled - source_errors,
            "candidate_bearing_cells": usable_cells,
            "usable_candidate_cells": usable_cells,
        }],
        "non_claim": "Candidate scarcity is not evidence of surface or ink absence.",
    }
    hashed = {key: value for key, value in receipt.items()
              if key not in {"generated_at_utc", "receipt_sha256",
                             "evidence_status", "evidence_status_reason"}}
    receipt["receipt_sha256"] = content_sha256(hashed)
    return receipt


def eligible_cell_ids(population: int) -> list[str]:
    return [
        "r%05dc%05da%05d" % (
            index % 16, (index // 16) % 16, index // (16 * 16))
        for index in range(population)
    ]


def derive(
    value: dict, *, cap: int = 100, manual: int | None = None,
    manual_reason: str | None = None,
) -> dict:
    return decision_module().derive_task_budget(
        value, policy(), compute_cap(cap), manual_task_count=manual,
        manual_lower_reason=manual_reason,
        eligible_cell_ids=eligible_cell_ids(
            value["funnel"]["geometrically_eligible_cells"]
            if value["measurement_kind"] == "CENSUS"
            else value["funnel"]["geometrically_eligible_cells_estimate"]),
    )


def test_census_uses_the_smallest_without_replacement_budget_reaching_95_percent():
    receipt = derive(preflight(population=10, usable_cells=2))
    assert receipt["decision"] == "CONTINUE"
    assert receipt["requested_task_count"] == 8
    assert receipt["planned_task_count"] == 8
    assert receipt["achieved_detection_probability"] == pytest.approx(44 / 45)
    assert receipt["planned_sampling_percentage"] == 80.0
    assert receipt["probability_model"] == "EXACT_HYPERGEOMETRIC_WITHOUT_REPLACEMENT"


def test_one_census_success_in_one_hundred_requires_ninety_five_tasks():
    receipt = derive(preflight(population=100, usable_cells=1))
    assert receipt["requested_task_count"] == 95
    assert receipt["achieved_detection_probability"] == pytest.approx(0.95)


def test_all_census_cells_usable_needs_one_task():
    receipt = derive(preflight(population=10, usable_cells=10))
    assert receipt["requested_task_count"] == 1
    assert receipt["achieved_detection_probability"] == 1.0


def test_census_zero_refuses_current_source_without_claiming_content_absence():
    receipt = derive(preflight(population=100, usable_cells=0))
    assert receipt["decision"] == "DO_NOT_QUEUE_CURRENT_SOURCE"
    assert receipt["requested_task_count"] is None
    assert receipt["planned_task_count"] == 0
    assert receipt["achieved_detection_probability"] == 0.0
    assert "not evidence" in receipt["non_claim"]


def test_compute_cap_reports_the_unclipped_request_and_clipped_probability():
    receipt = derive(preflight(population=10, usable_cells=2), cap=3)
    assert receipt["requested_task_count"] == 8
    assert receipt["compute_cap_tasks"] == 3
    assert receipt["planned_task_count"] == 3
    assert receipt["achieved_detection_probability"] == pytest.approx(8 / 15)
    assert receipt["target_detection_probability_met"] is False
    assert receipt["planned_sampling_percentage"] == 30.0


def test_zero_compute_cap_is_explicitly_nonqueueable():
    receipt = derive(preflight(population=10, usable_cells=2), cap=0)
    assert receipt["decision"] == "NO_COMPUTE_AUTHORIZED"
    assert receipt["planned_task_count"] == 0
    assert receipt["achieved_detection_probability"] == 0.0
    assert receipt["allowed_next_actions"] == ["INCREASE_FROZEN_COMPUTE_CAP"]


def test_manual_lower_budget_is_bound_to_its_lower_achieved_probability():
    receipt = derive(
        preflight(population=10, usable_cells=2), manual=2,
        manual_reason="bounded local smoke",
    )
    assert receipt["requested_task_count"] == 8
    assert receipt["planned_task_count"] == 2
    assert receipt["manual_lower_budget"] is True
    assert receipt["manual_lower_reason"] == "bounded local smoke"
    assert receipt["achieved_detection_probability"] == pytest.approx(17 / 45)
    assert receipt["target_detection_probability_met"] is False


def test_sampled_positive_uses_preregistered_one_sided_lower_bound():
    receipt = derive(preflight(
        population=1000, sampled_cells=100, usable_cells=10,
        measurement_kind="ESTIMATE"))
    assert receipt["probability_model"] == (
        "CONSERVATIVE_CLOPPER_PEARSON_TO_FINITE_POPULATION")
    assert receipt["sampling_inference_assumption"] == (
        "MODEL_BASED_BINOMIAL_EXCHANGEABILITY_NOT_DESIGN_BASED")
    assert receipt["observed_usable_cells"] == 10
    assert receipt["successful_preflight_trials"] == 100
    assert receipt["prevalence_lower_confidence_bound"] == pytest.approx(
        0.055263237682870045)
    assert receipt["conservative_population_usable_cells"] == 55
    assert receipt["requested_task_count"] == 52
    assert len(receipt["queue_selection"]["prefix_cell_ids"]) == 52
    assert receipt["achieved_detection_probability"] == pytest.approx(
        0.9512803512178811)


def test_sampled_zero_reports_upper_bound_and_never_invents_a_budget():
    receipt = derive(preflight(
        population=1000, sampled_cells=100, usable_cells=0,
        measurement_kind="ESTIMATE"))
    assert receipt["decision"] == "MORE_PREFLIGHT_OR_NEW_SOURCE_REQUIRED"
    assert receipt["requested_task_count"] is None
    assert receipt["planned_task_count"] == 0
    assert receipt["prevalence_upper_confidence_bound"] == pytest.approx(
        0.029513049607039925)
    assert receipt["allowed_next_actions"] == [
        "RUN_LARGER_PREFLIGHT", "CHANGE_CANDIDATE_SOURCE"]


def test_sampled_preflight_never_claims_design_based_probability():
    receipt = derive(preflight(
        population=1000, sampled_cells=100, usable_cells=10,
        measurement_kind="ESTIMATE"))
    assert receipt["sampling_inference_assumption"] == (
        "MODEL_BASED_BINOMIAL_EXCHANGEABILITY_NOT_DESIGN_BASED")
    assert "not a design-based" in receipt["non_claim"]


def test_sampled_budget_uses_exact_current_population_not_estimated_n_for_queue():
    value = preflight(
        population=1000, sampled_cells=100, usable_cells=10,
        measurement_kind="ESTIMATE")
    receipt = decision_module().derive_task_budget(
        value, policy(), compute_cap(),
        eligible_cell_ids=eligible_cell_ids(500),
    )
    assert receipt["preflight_eligible_population_cells"] == 1000
    assert receipt["eligible_population_cells"] == 500
    assert receipt["conservative_population_usable_cells"] == 27
    assert receipt["requested_task_count"] == 52
    assert len(receipt["queue_selection"]["prefix_cell_ids"]) == 52
    assert receipt["sampling_inference_assumption"] == (
        "MODEL_BASED_BINOMIAL_EXCHANGEABILITY_NOT_DESIGN_BASED")


def test_population_accumulator_hashes_every_cell_but_retains_only_cap_prefix():
    module = decision_module()
    ids = eligible_cell_ids(1000)
    seed = "9" * 64
    accumulator = module.EligiblePopulationAccumulator(
        order_seed_sha256=seed, prefix_limit=7)
    for cell_id in ids:
        accumulator.observe(cell_id)
    proof = accumulator.finish()
    assert proof["eligible_population_cells"] == 1000
    assert proof["population_order_sha256"] == content_sha256(ids)
    assert proof["ranked_prefix_capacity"] == 7
    assert len(proof["ranked_prefix_cell_ids"]) == 7
    assert proof == module.summarize_eligible_population(
        ids, order_seed_sha256=seed, prefix_limit=7)


def test_population_scan_policy_is_explicit_and_fail_closed():
    changed = policy()
    changed["task_budget"].pop("population_scan")
    with pytest.raises(ValueError, match="statistics"):
        decision_module().derive_task_budget(
            preflight(population=10, usable_cells=2), changed, compute_cap(),
            eligible_cell_ids=eligible_cell_ids(10),
        )


def test_sampled_assumption_mismatch_refuses_a_budget():
    changed = policy()
    changed["task_budget"]["sampled_preflight"]["inference_assumption"] = (
        "DESIGN_BASED"
    )
    with pytest.raises(ValueError, match="inference assumption"):
        decision_module().derive_task_budget(
            preflight(population=1000, sampled_cells=100, usable_cells=10,
                      measurement_kind="ESTIMATE"),
            changed, compute_cap(), eligible_cell_ids=None,
        )


def test_census_freezes_an_outcome_independent_without_replacement_prefix():
    first = derive(preflight(population=10, usable_cells=2))
    second = derive(preflight(population=10, usable_cells=4))
    selection = first["queue_selection"]
    assert selection["sampling"] == "WITHOUT_REPLACEMENT"
    assert selection["ordering"] == "SERVER_DERIVED_SHA256_PERMUTATION_V1"
    assert selection["selection_strategy"] == "probability-prefix-v1"
    assert len(selection["prefix_cell_ids"]) == first["planned_task_count"]
    assert len(selection["prefix_cell_ids"]) == len(set(selection["prefix_cell_ids"]))
    assert selection["population_order_sha256"] == (
        second["queue_selection"]["population_order_sha256"]
    )
    assert selection["order_seed_sha256"] == (
        second["queue_selection"]["order_seed_sha256"]
    )
    assert first["achieved_detection_probability"] == pytest.approx(44 / 45)


def test_compute_cap_only_truncates_one_server_frozen_nested_order():
    value = preflight(population=100, usable_cells=1)
    small = derive(value, cap=3)
    large = derive(value, cap=20)
    renamed_cap = decision_module().derive_task_budget(
        value, policy(), {**compute_cap(20), "cap_id": "another-client-label"},
        eligible_cell_ids=eligible_cell_ids(100),
    )
    assert small["queue_selection"]["order_seed_sha256"] == (
        large["queue_selection"]["order_seed_sha256"])
    assert renamed_cap["queue_selection"]["order_seed_sha256"] == (
        large["queue_selection"]["order_seed_sha256"])
    assert small["queue_selection"]["prefix_cell_ids"] == (
        large["queue_selection"]["prefix_cell_ids"][:3])


def test_minimum_draws_scales_to_the_policy_population_limit():
    started = time.monotonic()
    assert decision_module()._minimum_draws(262_144, 1, 0.95) == 249_037
    assert time.monotonic() - started < 1.0


def test_census_rejects_missing_duplicate_or_drifted_eligible_cell_population():
    value = preflight(population=10, usable_cells=2)
    module = decision_module()
    for cells in (None, eligible_cell_ids(9), ["same"] * 10):
        with pytest.raises(ValueError, match="eligible cell"):
            module.derive_task_budget(
                value, policy(), compute_cap(), eligible_cell_ids=cells)


def test_manual_lower_budget_requires_an_explicit_reason():
    with pytest.raises(ValueError, match="reason"):
        derive(preflight(population=10, usable_cells=2), manual=2)


@pytest.mark.parametrize("change", ["STALE", "INVALID"])
def test_stale_or_invalid_preflight_cannot_create_a_budget(change):
    with pytest.raises(ValueError, match="CURRENT"):
        derive(preflight(population=10, usable_cells=2, evidence_status=change))


def test_source_errors_cannot_shrink_the_scientific_denominator():
    with pytest.raises(ValueError, match="source errors"):
        derive(preflight(population=10, usable_cells=2, source_errors=1))


def test_preflight_hash_mismatch_is_rejected():
    value = preflight(population=10, usable_cells=2)
    value["funnel"]["geometrically_eligible_cells"] = 999
    with pytest.raises(ValueError, match="hash"):
        derive(value)


def test_compute_cap_is_required_content_bound_and_scope_exact():
    value = preflight(population=10, usable_cells=2)
    with pytest.raises(ValueError, match="compute cap"):
        decision_module().derive_task_budget(value, policy(), {}, manual_task_count=None)
    wrong = compute_cap()
    wrong["sample_id"] = "PHerc9999"
    with pytest.raises(ValueError, match="sample"):
        decision_module().derive_task_budget(value, policy(), wrong, manual_task_count=None)


def test_budget_receipt_hash_scope_profile_and_preflight_are_fail_closed():
    module = decision_module()
    receipt = derive(preflight(population=10, usable_cells=2))
    accepted = module.validate_task_budget_for_queue(
        receipt, mission_id="first-letters", sample_id="PHerc358",
        preflight_receipt_sha256="a" * 64,
        policy_sha256=content_sha256(policy()), requested_tasks=8,
        execution_bindings=receipt["execution_bindings"])
    assert accepted["planned_task_count"] == 8
    for field, value in (
        ("mission_id", "other"), ("sample_id", "PHerc9999"),
        ("preflight_receipt_sha256", "b" * 64),
        ("policy_sha256", "c" * 64), ("requested_tasks", 9),
        ("requested_tasks", 7),
    ):
        args = {
            "mission_id": "first-letters", "sample_id": "PHerc358",
            "preflight_receipt_sha256": "a" * 64,
            "policy_sha256": content_sha256(policy()), "requested_tasks": 8,
            "execution_bindings": receipt["execution_bindings"],
        }
        args[field] = value
        with pytest.raises(ValueError):
            module.validate_task_budget_for_queue(receipt, **args)


@pytest.mark.parametrize("field,value", [
    ("grid_version", "changed-grid@2.0.0"),
    ("m7_threshold", 0.3),
    ("gates", {"cell_clearance": 999}),
])
def test_changed_grid_m7_or_clearance_profile_invalidates_queue_admission(field, value):
    module = decision_module()
    receipt = derive(preflight(population=10, usable_cells=2))
    execution = copy.deepcopy(receipt["execution_bindings"])
    execution[field] = value
    with pytest.raises(ValueError, match="execution"):
        module.validate_task_budget_for_queue(
            receipt, mission_id="first-letters", sample_id="PHerc358",
            preflight_receipt_sha256="a" * 64,
            policy_sha256=content_sha256(policy()), requested_tasks=8,
            execution_bindings=execution)


def test_receipt_is_content_hashed_and_carries_the_frozen_cap_and_policy():
    receipt = derive(preflight(population=10, usable_cells=2))
    hashed = {key: value for key, value in receipt.items()
              if key not in {"generated_at_utc", "receipt_sha256"}}
    assert receipt["receipt_sha256"] == content_sha256(hashed)
    assert receipt["preflight_receipt_sha256"] == "a" * 64
    assert receipt["campaign_policy_profile_id"] == (
        "first-letters-campaign-decision-policy@1.2.0")
    assert receipt["campaign_policy_sha256"] == content_sha256(policy())
    assert receipt["compute_cap_sha256"] == content_sha256(compute_cap())
    assert math.isfinite(receipt["achieved_detection_probability"])


def test_budget_sanitized_projection_hides_cell_prefix_and_binds_private_hash():
    module = decision_module()
    private = derive(preflight(population=10, usable_cells=2))
    public = module.sanitize_task_budget_receipt(private)
    assert public["schema"] == "campaignx.first_letters_task_budget.sanitized.v1"
    assert public["private_receipt_sha256"] == private["receipt_sha256"]
    assert "prefix_cell_ids" not in public["queue_selection"]
    assert public["queue_selection"]["prefix_count"] == 8
    assert module.validate_task_budget_receipt_pair(private, public) == (
        private, public)


def test_budget_pair_is_create_once_and_rejects_a_changed_second_publication(tmp_path):
    module = decision_module()
    private = derive(preflight(population=10, usable_cells=2))
    public = module.sanitize_task_budget_receipt(private)
    private_path = tmp_path / "budget.private.json"
    public_path = tmp_path / "budget.sanitized.json"
    first = module.persist_task_budget_receipt_pair(
        private_path, public_path, private, public)
    second = module.persist_task_budget_receipt_pair(
        private_path, public_path, copy.deepcopy(private), copy.deepcopy(public))
    assert first == second
    assert private_path.stat().st_mode & 0o777 == 0o600
    assert public_path.stat().st_mode & 0o777 == 0o644
    changed = copy.deepcopy(private)
    changed["generated_at_utc"] = "2099-01-01T00:00:00Z"
    replay = module.persist_task_budget_receipt_pair(
        private_path, public_path, changed,
        module.sanitize_task_budget_receipt(changed))
    assert replay == first

    public_path.unlink()
    repaired = module.persist_task_budget_receipt_pair(
        private_path, public_path, changed,
        module.sanitize_task_budget_receipt(changed))
    assert repaired == first
    assert json.loads(public_path.read_text(encoding="utf-8")) == first[
        "sanitized_receipt"]


def test_checked_in_policy_declares_the_executable_budget_admission_contract():
    task_budget = policy()["task_budget"]
    assert task_budget["receipt_validation"] == "SIGNED_CURRENT_PREFLIGHT_ONLY"
    assert task_budget["queue_admission"] == (
        "EXACT_HASH_SCOPE_EXECUTION_BINDINGS_AND_PLANNED_TASK_COUNT")


def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir(exist_ok=True)
    import panel.app as module
    module.RUNS = tmp_path / "runs"
    return module


def queue_options() -> dict:
    return {
        "grid_step": 2048, "query_radius": 512, "volume_edge_margin": 512,
        "clearance": 0.0, "candidate_interior_clearance": 2,
        "selection_strategy": "stratified-clearance-v1",
        "candidate_selection_policy": "score-cell-volume-clearance-v1",
        "seed_region_policy": "fixed-v1", "ct_material_support_gate": "on",
        "ct_support_level": 1, "ct_support_radius_l0": 4,
        "ct_support_minimum_nonzero_voxels": 1,
        "grid_version": "first-letters-grid@1.0.0",
        "policy_version": "first-letters-preflight@1.0.0",
    }


def test_panel_budget_api_uses_current_exact_preflight_and_persists_hash_named_receipt(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    evidence = preflight(population=10, usable_cells=2)
    manifest = controlled_manifest(tmp_path / "campaign")
    monkeypatch.setattr(app, "require_write_sample", lambda *_args: "PHerc358")
    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app, "_mission_campaign_manifest", lambda _mission: manifest)
    monkeypatch.setattr(app, "_latest_candidate_preflight_evidence",
                        lambda *_args: evidence)
    monkeypatch.setattr(app, "_candidate_preflight_eligible_cell_ids",
                        lambda *_args, **_kwargs: decision_module().summarize_eligible_population(
                            eligible_cell_ids(10),
                            order_seed_sha256=_kwargs["order_seed_sha256"],
                            prefix_limit=_kwargs["prefix_limit"],
                        ))
    body = app.SegmentationTaskBudgetRequest(
        sample_id="PHerc0358", mission_id="first-letters",
        preflight_receipt_sha256="a" * 64,
        compute_cap_id="first-letters-local-cap-1", compute_cap_tasks=100,
    )
    response = app._run_task_budget_api(body)
    result = json.loads(response.body)
    assert result["planned_task_count"] == 8
    assert result["preflight_receipt_sha256"] == "a" * 64
    root = tmp_path / "evidence/task-budgets/PHerc0358"
    public_path = root / f"{result['private_receipt_sha256']}.sanitized.json"
    private_path = root / f"{result['private_receipt_sha256']}.private.json"
    assert json.loads(public_path.read_text(encoding="utf-8")) == result
    assert private_path.exists()


def test_panel_budget_api_rejects_a_noncurrent_or_different_preflight(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    evidence = preflight(population=10, usable_cells=2, evidence_status="STALE")
    monkeypatch.setattr(app, "require_write_sample", lambda *_args: "PHerc358")
    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app, "_latest_candidate_preflight_evidence",
                        lambda *_args: evidence)
    monkeypatch.setattr(app, "_candidate_preflight_eligible_cell_ids",
                        lambda *_args, **_kwargs: decision_module().summarize_eligible_population(
                            eligible_cell_ids(10),
                            order_seed_sha256=_kwargs["order_seed_sha256"],
                            prefix_limit=_kwargs["prefix_limit"],
                        ))
    body = app.SegmentationTaskBudgetRequest(
        sample_id="PHerc0358", mission_id="first-letters",
        preflight_receipt_sha256="a" * 64,
        compute_cap_id="cap", compute_cap_tasks=10,
    )
    with pytest.raises(Exception, match="CURRENT"):
        app._run_task_budget_api(body)
    evidence["evidence_status"] = "CURRENT"
    body.preflight_receipt_sha256 = "b" * 64
    with pytest.raises(Exception, match="preflight"):
        app._run_task_budget_api(body)


def test_budget_get_readback_is_exactly_scoped_by_mission_sample_and_sha(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    private = derive(preflight(population=10, usable_cells=2))
    public = decision_module().sanitize_task_budget_receipt(private)
    root = tmp_path / "evidence/task-budgets/PHerc0358"
    root.mkdir(parents=True)
    (root / f"{private['receipt_sha256']}.private.json").write_text(
        json.dumps(private), encoding="utf-8")
    (root / f"{private['receipt_sha256']}.sanitized.json").write_text(
        json.dumps(public), encoding="utf-8")
    monkeypatch.setattr(app, "require_write_sample", lambda *_args: "PHerc358")
    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    response = app._read_task_budget_api(
        "first-letters", "PHerc0358", private["receipt_sha256"])
    assert json.loads(response.body) == public
    with pytest.raises(Exception, match="scope"):
        app._read_task_budget_api(
            "first-letters", "PHerc9999", private["receipt_sha256"])


def test_current_preflight_makes_budget_receipt_mandatory_and_removes_fixed_four(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    evidence = preflight(population=10, usable_cells=2)
    manifest = controlled_manifest(tmp_path / "campaign")
    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app, "_mission_campaign_manifest", lambda _mission: manifest)
    monkeypatch.setattr(app, "_latest_candidate_preflight_evidence",
                        lambda *_args: evidence)
    request = app.SegmentationRunRequest(
        sample_id="PHerc0358", mission_id="first-letters",
        options=queue_options(), max_tasks=None)
    with pytest.raises(Exception, match="task-budget receipt"):
        app._resolve_task_budget_admission(request, "PHerc358", grid_step=2048)


def test_budgeted_queue_uses_exact_planned_count_and_execution_bindings(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    evidence = preflight(population=10, usable_cells=2)
    manifest = controlled_manifest(tmp_path / "campaign")
    budget = derive(evidence)
    public_budget = decision_module().sanitize_task_budget_receipt(budget)
    root = tmp_path / "evidence/task-budgets/PHerc0358"
    root.mkdir(parents=True)
    (root / f"{budget['receipt_sha256']}.private.json").write_text(
        json.dumps(budget), encoding="utf-8")
    (root / f"{budget['receipt_sha256']}.sanitized.json").write_text(
        json.dumps(public_budget), encoding="utf-8")
    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app, "_mission_campaign_manifest", lambda _mission: manifest)
    monkeypatch.setattr(app, "_latest_candidate_preflight_evidence",
                        lambda *_args: evidence)
    request = app.SegmentationRunRequest(
        sample_id="PHerc0358", mission_id="first-letters",
        options=queue_options(), max_tasks=None,
        task_budget_receipt_sha256=budget["receipt_sha256"])
    task_count, admitted = app._resolve_task_budget_admission(
        request, "PHerc358", grid_step=2048)
    assert task_count == 8
    assert admitted["receipt_sha256"] == budget["receipt_sha256"]

    request.max_tasks = 7
    with pytest.raises(Exception, match="task count"):
        app._resolve_task_budget_admission(request, "PHerc358", grid_step=2048)
    request.max_tasks = None
    request.options["grid_version"] = "changed-grid@2.0.0"
    with pytest.raises(Exception, match="execution"):
        app._resolve_task_budget_admission(request, "PHerc358", grid_step=2048)


def test_controlled_budget_binds_all_queue_generator_knobs():
    receipt = derive(preflight(population=10, usable_cells=2))
    queue = receipt["execution_bindings"]["queue_execution"]
    assert queue == {
        "parameter_envelope": {
            **DEFAULT_ENVELOPE,
            "maximum_candidate_count": 2,
        },
        "planner": "cost-aware-v2",
        "planner_model": None,
        "prediction_space": "ct_l0_xyz",
        "minimum_separation_voxels": 16,
        "recenter_probe_max_candidates": 100,
        "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
        "seed_probe_mode": "off",
        "seed_probe_top_k": 2,
        "seed_probe_generations": 12,
        "candidate_rank": 1,
        "reconsider_covered": False,
        "verify_sources": True,
    }


def test_controlled_queue_rejects_recenter_planner_or_candidate_authority_drift(tmp_path, monkeypatch):
    app = app_module(tmp_path, monkeypatch)
    evidence = preflight(population=10, usable_cells=2)
    manifest = controlled_manifest(tmp_path / "campaign")
    budget = derive(evidence)
    public_budget = decision_module().sanitize_task_budget_receipt(budget)
    root = tmp_path / "evidence/task-budgets/PHerc0358"
    root.mkdir(parents=True)
    (root / f"{budget['receipt_sha256']}.private.json").write_text(
        json.dumps(budget), encoding="utf-8")
    (root / f"{budget['receipt_sha256']}.sanitized.json").write_text(
        json.dumps(public_budget), encoding="utf-8")
    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    monkeypatch.setattr(app, "_mission_campaign_manifest", lambda _mission: manifest)
    monkeypatch.setattr(
        app, "_latest_candidate_preflight_evidence", lambda *_args: evidence)
    for options, planner, seed_config in (
        ({**queue_options(), "recenter_radius_x": 65}, "cost-aware-v2", {}),
        (queue_options(), "deterministic-v2", {}),
        (queue_options(), "cost-aware-v2", {"candidate_rank": "2"}),
        (queue_options(), "cost-aware-v2", {"reconsider_covered": "true"}),
    ):
        request = app.SegmentationRunRequest(
            sample_id="PHerc0358", mission_id="first-letters",
            options=options, planner=planner, seed_config=seed_config,
            task_budget_receipt_sha256=budget["receipt_sha256"])
        with pytest.raises(Exception, match="execution"):
            app._resolve_task_budget_admission(
                request, "PHerc358", grid_step=2048)


@pytest.mark.parametrize("rank", ["0", "-1"])
def test_panel_rejects_nonpositive_candidate_rank_before_budget_binding(
    tmp_path, monkeypatch, rank,
):
    app = app_module(tmp_path, monkeypatch)
    request = app.SegmentationRunRequest(
        sample_id="PHerc0358", options=queue_options(),
        seed_config={"candidate_rank": rank},
    )
    with pytest.raises(Exception, match="candidate_rank must be a positive"):
        app._budget_execution_bindings(
            request, preflight(population=10, usable_cells=2), grid_step=2048,
        )


def test_legacy_mission_without_preflight_keeps_explicit_or_legacy_default_budget(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    monkeypatch.setattr(app, "_latest_candidate_preflight_evidence",
                        lambda *_args: None)
    request = app.SegmentationRunRequest(sample_id="PHerc0826", max_tasks=None)
    assert app._resolve_task_budget_admission(
        request, "PHerc826", grid_step=2048) == (4, None)
    request.max_tasks = 6
    assert app._resolve_task_budget_admission(
        request, "PHerc826", grid_step=2048) == (6, None)


def test_generic_mission_with_preflight_keeps_legacy_budget_behavior(
    tmp_path, monkeypatch,
):
    app = app_module(tmp_path, monkeypatch)
    generic = mission_contract.create(
        tmp_path / "missions", mission_id="generic-one", name="generic",
        scrolls=["PHerc0358"], created_by="alice")
    monkeypatch.setattr(app, "_mission_campaign_manifest", lambda _mission: generic)
    monkeypatch.setattr(
        app, "_latest_candidate_preflight_evidence",
        lambda *_args: preflight(population=10, usable_cells=2))
    request = app.SegmentationRunRequest(
        sample_id="PHerc0358", mission_id="generic-one", max_tasks=None)
    assert app._resolve_task_budget_admission(
        request, "PHerc358", grid_step=2048) == (4, None)


def controlled_campaign_binding() -> dict:
    return {
        "campaign_kind": "FIRST_LETTERS_DISCOVERY",
        "campaign_policy_id": "first-letters-campaign-decision-policy@1.2.0",
        "campaign_policy_sha256": content_sha256(policy()),
        "deployed_revision": "4" * 40,
    }


def test_generic_mission_creation_remains_backward_compatible(tmp_path):
    manifest = mission_contract.create(
        tmp_path, mission_id="generic-one", name="generic",
        scrolls=["PHerc0826"], created_by="alice")
    assert manifest["created_by"] == "alice"
    assert "campaign_kind" not in manifest
    assert mission_contract.load(tmp_path / "generic-one") == manifest


def test_controlled_mission_freezes_exact_policy_revision_and_creator(tmp_path):
    manifest = mission_contract.create(
        tmp_path, mission_id="letters-one", name="letters",
        scrolls=["PHerc0358"], created_by="alice",
        **controlled_campaign_binding())
    assert {key: manifest[key] for key in controlled_campaign_binding()} == (
        controlled_campaign_binding())
    assert manifest["created_by"] == "alice"
    assert mission_contract.is_first_letters_discovery_manifest(manifest) is True


@pytest.mark.parametrize("field", [
    "campaign_policy_id", "campaign_policy_sha256", "deployed_revision",
])
def test_controlled_mission_refuses_an_incomplete_creation_binding(tmp_path, field):
    binding = controlled_campaign_binding()
    binding.pop(field)
    mission_id = f"missing-{field[:8]}"
    with pytest.raises(mission_contract.MissionError, match="campaign"):
        mission_contract.create(
            tmp_path, mission_id=mission_id, name="letters",
            scrolls=["PHerc0358"], created_by="alice", **binding)
    assert not (tmp_path / mission_id).exists()


@pytest.mark.parametrize("field,value", [
    ("campaign_policy_sha256", "0" * 64),
    ("deployed_revision", "0" * 40),
    ("created_by", "mallory"),
])
def test_controlled_mission_load_rejects_immutable_binding_tampering(
    tmp_path, field, value,
):
    manifest = mission_contract.create(
        tmp_path, mission_id="letters-two", name="letters",
        scrolls=["PHerc0358"], created_by="alice",
        **controlled_campaign_binding())
    manifest[field] = value
    mission_contract.write(tmp_path / "letters-two", manifest)
    with pytest.raises(mission_contract.MissionError, match="campaign"):
        mission_contract.load(tmp_path / "letters-two")


def test_controlled_mission_binding_and_root_include_exact_mission_id(tmp_path):
    mission_contract.create(
        tmp_path, mission_id="letters-three", name="letters",
        scrolls=["PHerc0358"], created_by="alice",
        **controlled_campaign_binding())
    path = tmp_path / "letters-three"
    manifest = json.loads((path / "MISSION.json").read_text(encoding="utf-8"))
    manifest["mission_id"] = "letters-four"
    mission_contract.write(path, manifest)
    with pytest.raises(mission_contract.MissionError, match="campaign|directory"):
        mission_contract.load(path)


def test_mission_name_or_id_prefix_never_activates_controlled_admission(tmp_path):
    manifest = mission_contract.create(
        tmp_path, mission_id="first-letters-lookalike", name="First Letters",
        scrolls=["PHerc0358"], created_by="alice")
    assert mission_contract.is_first_letters_discovery_manifest(manifest) is False


def controlled_manifest(tmp_path) -> dict:
    return mission_contract.create(
        tmp_path, mission_id="first-letters", name="letters",
        scrolls=["PHerc0358"], created_by="alice",
        **controlled_campaign_binding())


@pytest.mark.parametrize("creation_path", [
    "bootstrap", "manual-seeds", "replan", "resume-correction",
    "seed-recovery", "adaptive-retry",
])
def test_generic_mission_keeps_every_existing_p1_creation_path(
    tmp_path, creation_path,
):
    manifest = mission_contract.create(
        tmp_path, mission_id=f"generic-{creation_path}", name="generic",
        scrolls=["PHerc0358"], created_by="alice")
    assert decision_module().admit_p1_creation(
        manifest, creation_path=creation_path) is None


@pytest.mark.parametrize("creation_path", [
    "manual-seeds", "replan", "resume-correction", "seed-recovery",
    "adaptive-retry",
])
def test_controlled_campaign_blocks_every_nonprefix_p1_creation_path(
    tmp_path, creation_path,
):
    with pytest.raises(ValueError, match="not authorized"):
        decision_module().admit_p1_creation(
            controlled_manifest(tmp_path), creation_path=creation_path)


def test_controlled_bootstrap_cannot_omit_budget_pair(tmp_path):
    with pytest.raises(ValueError, match="budget receipt pair"):
        decision_module().admit_p1_creation(
            controlled_manifest(tmp_path), creation_path="bootstrap")


def test_controlled_bootstrap_returns_exact_store_envelope(tmp_path):
    module = decision_module()
    evidence = preflight(population=10, usable_cells=2)
    private = derive(evidence)
    public = module.sanitize_task_budget_receipt(private)
    envelope = module.admit_p1_creation(
        controlled_manifest(tmp_path), creation_path="bootstrap",
        budget_private=private, budget_public=public,
        mission_id="first-letters", sample_id="PHerc358",
        preflight_receipt_sha256="a" * 64,
        policy_sha256=content_sha256(policy()), requested_tasks=8,
        execution_bindings=private["execution_bindings"],
    )
    expected = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": "first-letters", "sample_id": "PHerc358",
        "receipt_sha256": private["receipt_sha256"],
        "preflight_receipt_sha256": private["preflight_receipt_sha256"],
        "preflight_sanitized_receipt_sha256": private[
            "preflight_sanitized_receipt_sha256"],
        "approved_task_count": 8,
        "order_seed_sha256": private["queue_selection"]["order_seed_sha256"],
        "population_order_sha256": private["queue_selection"][
            "population_order_sha256"],
        "prefix_sha256": private["queue_selection"]["prefix_sha256"],
        "prefix_cell_ids": private["queue_selection"]["prefix_cell_ids"],
        "execution_bindings": private["execution_bindings"],
    }
    expected["admission_sha256"] = content_sha256(expected)
    assert envelope == expected


def store_source(store: FleetStore) -> str:
    return store.register_snapshot({
        "sample_id": "PHerc358", "ct_uri": "fixture://ct",
        "ct_sha256": "2" * 64, "m7_uri": "fixture://m7",
        "m7_sha256": "3" * 64, "shape_xyz": [32768, 32768, 32768],
        "voxel_size_um": 9.362, "coordinate_frame": "ct_l0_xyz",
        "m7_threshold": 0.2,
        "source_content_lock": {"schema": "fixture-source-lock"},
    })


def budget_task(source_id: str, cell_id: str, admission: dict) -> dict:
    execution = admission["execution_bindings"]
    queue = execution["queue_execution"]
    gates = execution["gates"]
    center, bounds = decision_module()._expected_cell_geometry(
        cell_id, execution, gates)
    return {
        "source_snapshot_id": source_id, "sample_id": "PHerc358",
        "mission_id": "first-letters", "cell_id": cell_id,
        "grid_version": execution["grid_version"],
        "policy_version": execution["policy_version"],
        "bounds_xyz": bounds,
        "center_xyz": center,
        "priority": 1.0,
        "parameter_envelope": queue["parameter_envelope"],
        "catalog_snapshot_sha256": execution["catalog_snapshot_sha256"],
        "candidate_selection_policy": execution["candidate_selection_policy"],
        "planner": queue["planner"],
        "candidate_discovery": {
            "provider": execution["provider"],
            "prediction_uri": "fixture://m7",
            "prediction_space": queue["prediction_space"],
            "m7_threshold": execution["m7_threshold"],
            "region": {
                "center": center,
                "radius": {axis: execution["query_radius"] for axis in "xyz"},
            },
            "max_candidates": queue["parameter_envelope"][
                "maximum_candidate_count"],
            "minimum_separation_voxels": queue[
                "minimum_separation_voxels"],
            "minimum_cell_interior_clearance_voxels": gates[
                "candidate_interior_clearance"],
            "minimum_volume_interior_clearance_voxels": gates[
                "volume_clearance"],
            "seed_region_policy": execution["seed_region_policy"],
            "recenter_probe_max_candidates": queue[
                "recenter_probe_max_candidates"],
            "recenter_radius_xyz": queue["recenter_radius_xyz"],
            "ct_material_support_gate": gates["ct_material_support_gate"],
        },
        "p0_selection_version": execution["p0_selection_version"],
        "p0_selection_sha256": execution["p0_selection_sha256"],
        "p0_artifact_id": execution["p0_artifact_id"],
        "p0_artifact_sha256": execution["p0_artifact_sha256"],
        "ink_used": False,
    }


def budget_admission(tmp_path, source_id: str = "source-1") -> dict:
    module = decision_module()
    evidence = preflight(population=10, usable_cells=2)
    evidence["source_snapshot_id"] = source_id
    evidence["bindings"]["source_snapshot_id"] = source_id
    evidence["receipt_sha256"] = content_sha256({
        key: value for key, value in evidence.items()
        if key not in {"generated_at_utc", "receipt_sha256",
                       "evidence_status", "evidence_status_reason"}
    })
    private = derive(evidence)
    return module.admit_p1_creation(
        controlled_manifest(tmp_path), creation_path="bootstrap",
        budget_private=private,
        budget_public=module.sanitize_task_budget_receipt(private),
        mission_id="first-letters", sample_id="PHerc358",
        preflight_receipt_sha256="a" * 64,
        policy_sha256=content_sha256(policy()), requested_tasks=8,
        execution_bindings=private["execution_bindings"],
    )


def test_budget_binding_adds_exact_prefix_rank_to_each_task(tmp_path):
    module = decision_module()
    admission = budget_admission(tmp_path)
    tasks = [budget_task("source-1", cell_id, admission)
             for cell_id in admission["prefix_cell_ids"]]
    bound = module.bind_campaign_budget_to_tasks(tasks, admission)
    assert [task["campaign_budget"]["selection_rank"] for task in bound] == (
        list(range(8)))
    assert all(task["campaign_budget"]["receipt_sha256"] ==
               admission["receipt_sha256"] for task in bound)


def test_sqlite_store_enforces_cumulative_budget_under_its_write_lock(tmp_path):
    module = decision_module()
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(admission)
    tasks = module.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, admission)
         for cell_id in admission["prefix_cell_ids"]], admission)
    assert store.create_tasks(tasks[:4]) == (4, 4)
    assert store.register_campaign_budget_admission(admission) == admission
    assert store.create_tasks(tasks[4:]) == (4, 4)
    assert store.create_tasks(tasks) == (0, 8)

    replacement_budget = copy.deepcopy(tasks[0])
    replacement_budget["campaign_budget"]["receipt_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="already bound|not registered"):
        store.create_tasks([replacement_budget])

    forged = copy.deepcopy(tasks[0])
    forged["cell_id"] = "outside-prefix"
    forged["policy_version"] = "forged-new-identity"
    forged["campaign_budget"]["selection_rank"] = 8
    with pytest.raises(ValueError, match="budget|prefix|rank"):
        store.create_tasks([forged])
    assert store.status()["tasks"]["PENDING"] == 8


def test_controlled_first_writer_requires_registered_signed_admission(tmp_path):
    store = FleetStore(tmp_path / "unregistered.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / "mission-unregistered", source_id)
    task = decision_module().bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, admission)
         for cell_id in admission["prefix_cell_ids"]], admission)[0]
    with pytest.raises(ValueError, match="not registered"):
        store.create_tasks([task])


@pytest.mark.parametrize("sample_id", ["PHerc358", "PHerc999"])
def test_registered_admission_blocks_unbudgeted_first_writer_for_entire_mission(
    tmp_path, sample_id,
):
    store = FleetStore(tmp_path / f"first-writer-{sample_id}.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / f"mission-{sample_id}", source_id)
    store.register_campaign_budget_admission(admission)
    bypass = budget_task(source_id, admission["prefix_cell_ids"][0], admission)
    bypass["sample_id"] = sample_id
    with pytest.raises(ValueError, match="cannot be omitted"):
        store.create_tasks([bypass])
    assert store.status()["tasks"].get("PENDING", 0) == 0


def test_postgres_registered_admission_blocks_unbudgeted_cross_sample_first_writer(
    tmp_path, monkeypatch,
):
    statements: list[str] = []

    class Cursor:
        rowcount = 1
        last = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, _parameters=()):
            self.last = " ".join(str(statement).split())
            statements.append(self.last)
            self.rowcount = 1

        def fetchall(self):
            return []

        def fetchone(self):
            if "segment_campaign_budget_admissions" in self.last:
                return {"admission_exists": True}
            return None

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

    admission = budget_admission(tmp_path / "mission-postgres-gap")
    bypass = budget_task("source-1", admission["prefix_cell_ids"][0], admission)
    bypass["sample_id"] = "PHerc999"
    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(store, "connect", lambda: Connection())
    with pytest.raises(ValueError, match="cannot be omitted"):
        store.create_tasks([bypass])
    assert any("segment_campaign_budget_admissions" in statement
               for statement in statements)
    assert not any("INSERT INTO segment_tasks" in statement
                   for statement in statements)


def test_registered_authority_rejects_forged_task_and_envelope_together(tmp_path):
    store = FleetStore(tmp_path / "forged-both.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / "mission-forged-both", source_id)
    store.register_campaign_budget_admission(admission)
    task = decision_module().bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, admission)
         for cell_id in admission["prefix_cell_ids"]], admission)[0]
    task["grid_version"] = "attacker-grid"
    forged = task["campaign_budget"]
    forged["execution_bindings"]["grid_version"] = "attacker-grid"
    forged["admission_sha256"] = content_sha256({
        key: value for key, value in forged.items()
        if key not in {"selection_rank", "admission_sha256"}
    })
    with pytest.raises(ValueError, match="registered|signed authority"):
        store.create_tasks([task])


def test_sqlite_admission_registration_is_concurrent_and_idempotent(tmp_path):
    store = FleetStore(tmp_path / "registry-race.sqlite")
    store.initialize()
    admission = budget_admission(tmp_path / "mission-registry-race")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda _index: store.register_campaign_budget_admission(admission),
            range(16),
        ))
    assert all(result == admission for result in results)
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM campaign_budget_admissions").fetchone()[0] == 1


@pytest.mark.parametrize("sample_id", ["PHerc358", "PHerc999"])
def test_sqlite_task_first_blocks_later_controlled_admission_for_entire_mission(
    tmp_path, sample_id,
):
    store = FleetStore(tmp_path / f"task-first-{sample_id}.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / f"mission-task-first-{sample_id}", source_id)
    generic = budget_task(source_id, admission["prefix_cell_ids"][0], admission)
    generic["sample_id"] = sample_id
    assert store.create_tasks([generic]) == (1, 1)
    with pytest.raises(ValueError, match="pre-existing|controlled admission"):
        store.register_campaign_budget_admission(admission)
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM campaign_budget_admissions").fetchone()[0] == 0


def test_postgres_task_first_blocks_registration_under_the_mission_lock(
    tmp_path, monkeypatch,
):
    admission = budget_admission(tmp_path / "mission-postgres-task-first")
    generic = budget_task(
        "source-1", admission["prefix_cell_ids"][0], admission)
    statements: list[tuple[str, tuple]] = []

    class Cursor:
        last = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, parameters=()):
            self.last = " ".join(str(statement).split())
            statements.append((self.last, tuple(parameters)))

        def fetchone(self):
            if "SELECT admission FROM segment_campaign_budget_admissions" in self.last:
                inserted = any(
                    "INSERT INTO segment_campaign_budget_admissions" in statement
                    for statement, _parameters in statements
                )
                return {"admission": admission} if inserted else None
            return None

        def fetchall(self):
            if "FROM segment_tasks" in self.last:
                return [{"payload": generic}]
            return []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(store, "connect", lambda: Connection())
    with pytest.raises(ValueError, match="pre-existing|controlled admission"):
        store.register_campaign_budget_admission(admission)
    advisory_parameters = [parameters for statement, parameters in statements
                           if "pg_advisory_xact_lock" in statement]
    assert advisory_parameters[0] == (
        "campaign-budget-mission:first-letters",
    )
    assert not any("INSERT INTO segment_campaign_budget_admissions" in statement
                   for statement, _parameters in statements)


def test_sanitized_budget_and_browser_binding_never_expose_credentialed_m7_uri():
    secret_uri = "https://user:password@example.invalid/private-m7.zarr"
    evidence = preflight(population=10, usable_cells=2)
    evidence["bindings"]["m7_uri_sha256"] = hashlib.sha256(
        secret_uri.encode("utf-8")).hexdigest()
    evidence["receipt_sha256"] = content_sha256({
        key: value for key, value in evidence.items()
        if key not in {"generated_at_utc", "receipt_sha256",
                       "evidence_status", "evidence_status_reason"}
    })
    private = derive(evidence)
    public = decision_module().sanitize_task_budget_receipt(private)
    assert secret_uri not in json.dumps(evidence)
    assert secret_uri not in json.dumps(private)
    assert secret_uri not in json.dumps(public)
    assert private["execution_bindings"]["m7_uri_sha256"] == hashlib.sha256(
        secret_uri.encode("utf-8")).hexdigest()


def test_fresh_store_rejects_forged_controlled_geometry_or_source_query(tmp_path):
    for name in (
        "center", "bounds", "region-center", "prediction-uri",
        "prediction-space", "minimum-separation",
    ):
        store = FleetStore(tmp_path / f"{name}.sqlite")
        store.initialize()
        source_id = store_source(store)
        admission = budget_admission(tmp_path / f"mission-{name}", source_id)
        store.register_campaign_budget_admission(admission)
        task = decision_module().bind_campaign_budget_to_tasks([
            budget_task(source_id, admission["prefix_cell_ids"][0], admission)
        ], {**admission, "approved_task_count": 1,
            "prefix_cell_ids": admission["prefix_cell_ids"][:1],
            "prefix_sha256": content_sha256(admission["prefix_cell_ids"][:1])})[0]
        if name == "center":
            task["center_xyz"]["x"] += 7000
        elif name == "bounds":
            task["bounds_xyz"][0][0] += 7000
        elif name == "region-center":
            task["candidate_discovery"]["region"]["center"]["x"] += 7000
        elif name == "prediction-uri":
            task["candidate_discovery"]["prediction_uri"] = "https://attacker.invalid/m7"
        elif name == "prediction-space":
            task["candidate_discovery"]["prediction_space"] = "other-space"
        else:
            task["candidate_discovery"]["minimum_separation_voxels"] = 999
        with pytest.raises(ValueError, match="campaign budget|frozen|execution"):
            store.create_tasks([task])
        assert store.status()["tasks"].get("PENDING", 0) == 0


def test_store_rejects_duplicate_rank_or_mixed_budget_envelopes_atomically(tmp_path):
    module = decision_module()
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(admission)
    tasks = module.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, admission)
         for cell_id in admission["prefix_cell_ids"]], admission)
    duplicate = copy.deepcopy(tasks[1])
    duplicate["campaign_budget"]["selection_rank"] = 0
    with pytest.raises(ValueError, match="rank"):
        store.create_tasks([tasks[0], duplicate])
    mixed = copy.deepcopy(tasks[1])
    mixed.pop("campaign_budget")
    with pytest.raises(ValueError, match="campaign budget"):
        store.create_tasks([tasks[0], mixed])
    assert store.status()["tasks"].get("PENDING", 0) == 0


def test_store_rejects_budget_omission_after_controlled_mission_starts(tmp_path):
    module = decision_module()
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store_source(store)
    admission = budget_admission(tmp_path / "mission", source_id)
    store.register_campaign_budget_admission(admission)
    tasks = module.bind_campaign_budget_to_tasks(
        [budget_task(source_id, cell_id, admission)
         for cell_id in admission["prefix_cell_ids"]], admission)
    assert store.create_tasks([tasks[0]]) == (1, 1)
    bypass = copy.deepcopy(tasks[1])
    bypass.pop("campaign_budget")
    bypass["cell_id"] = "unbudgeted-cell"
    bypass["policy_version"] = "bypass"
    with pytest.raises(ValueError, match="cannot be omitted"):
        store.create_tasks([bypass])
    assert store.status()["tasks"]["PENDING"] == 1


def test_postgres_store_declares_transaction_scoped_budget_lock():
    source = (ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py"
              ).read_text(encoding="utf-8")
    assert "pg_advisory_xact_lock" in source
    assert "validate_campaign_budget_task_batch" in source


def test_controlled_generator_uses_probability_prefix_not_max_clearance():
    class SurfaceView:
        @staticmethod
        def surfaces_for_snapshot(_source_id):
            return []

    prefix = ["r00003c00000a00000", "r00000c00003a00000",
              "r00001c00001a00001"]
    admission = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": "first-letters", "sample_id": "PHerc358",
        "receipt_sha256": "8" * 64, "approved_task_count": 3,
        "preflight_receipt_sha256": "b" * 64,
        "preflight_sanitized_receipt_sha256": "c" * 64,
        "order_seed_sha256": "9" * 64,
        "population_order_sha256": "a" * 64,
        "prefix_sha256": content_sha256(prefix), "prefix_cell_ids": prefix,
        "execution_bindings": {"queue_execution": {
            "parameter_envelope": {
                **DEFAULT_ENVELOPE, "maximum_candidate_count": 2,
            },
        }},
    }
    admission["admission_sha256"] = content_sha256(admission)
    tasks = generate_tasks_for_snapshot(
        SurfaceView(),
        {"source_snapshot_id": "source", "sample_id": "PHerc358",
         "m7_uri": "fixture://m7", "shape_xyz": [65, 65, 65]},
        catalog_snapshot_sha256="7" * 64, grid_step=16, query_radius=8,
        clearance=0, volume_edge_margin=8, candidate_interior_clearance=0,
        selection_strategy="max-clearance-v1", max_tasks=3,
        grid_version="grid", policy_version="policy",
        mission_id="first-letters", campaign_budget_admission=admission,
    )
    assert [task["cell_id"] for task in tasks] == prefix
    assert [task["campaign_budget"]["selection_rank"] for task in tasks] == [0, 1, 2]
    assert all(task["parameter_envelope"]["maximum_candidate_count"] == 2
               for task in tasks)


def test_panel_exposes_every_seed_region_policy_supported_by_preflight_and_worker():
    app = importlib.import_module("panel.app")
    option = next(row for row in app.SEGMENTATION_OPTIONS
                  if row["field"] == "seed_region_policy")
    assert set(option["choices"]) == {
        "fixed-v1", "m7-recenter-z-v1", "m7-recenter-xyz-v1",
        "m7-recenter-z-chunk-safe-v1", "m7-chunk-safe-merge-interior-v2",
    }


def test_generator_streams_every_eligible_cell_independent_of_task_limit():
    class SurfaceView:
        @staticmethod
        def surfaces_for_snapshot(_source_id):
            return []

    population: dict[str, int] = {}
    observed: list[str] = []
    tasks = generate_tasks_for_snapshot(
        SurfaceView(),
        {"source_snapshot_id": "source", "sample_id": "PHerc358",
         "m7_uri": "fixture://m7", "shape_xyz": [65, 65, 65]},
        catalog_snapshot_sha256="7" * 64, grid_step=16, query_radius=8,
        clearance=0, volume_edge_margin=8, candidate_interior_clearance=0,
        selection_strategy="max-clearance-v1", max_tasks=1,
        grid_version="grid", policy_version="policy",
        population_count_out=population,
        population_cell_observer=observed.append,
    )
    assert len(tasks) == 1
    assert population == {
        "total_grid_cells": 64, "geometrically_eligible_cells": 64,
    }
    assert len(observed) == len(set(observed)) == 64


def controlled_bootstrap_args(tmp_path, monkeypatch, *, include_root=True):
    evidence = preflight(population=10, usable_cells=2)
    private = derive(evidence)
    public = decision_module().sanitize_task_budget_receipt(private)
    mission_root = tmp_path / "first-letters"
    mission_contract.create(
        tmp_path, mission_id="first-letters", name="letters",
        scrolls=["PHerc0358"], created_by="alice",
        **controlled_campaign_binding())
    private_path = tmp_path / "budget.private.json"
    public_path = tmp_path / "budget.sanitized.json"
    private_path.write_text(json.dumps(private), encoding="utf-8")
    public_path.write_text(json.dumps(public), encoding="utf-8")
    monkeypatch.setattr("fleet.cli.file_sha256", lambda _path: "7" * 64)
    monkeypatch.setattr(
        "fleet.cli._validate_cli_current_preflight", lambda *_args: None)
    argv = [
        "bootstrap", "--db", "fixture.sqlite", "--eligible", "eligible.json",
        "--catalog", "catalog.sqlite", "--sample", "PHerc358",
        "--mission-id", "first-letters", "--max-tasks-per-sample", "8",
        "--grid-step", "2048", "--query-radius", "512", "--clearance", "0",
        "--volume-edge-margin", "512", "--candidate-interior-clearance", "2",
        "--selection-strategy", "stratified-clearance-v1",
        "--grid-version", "first-letters-grid@1.0.0",
        "--policy-version", "first-letters-preflight@1.0.0",
        "--candidate-selection-policy", "score-cell-volume-clearance-v1",
        "--seed-region-policy", "fixed-v1", "--ct-support-level", "1",
        "--ct-support-radius-l0", "4",
        "--ct-support-minimum-nonzero-voxels", "1",
        "--p0-selection-version", "selection-1", "--p0-selection-sha256", "6" * 64,
        "--p0-artifact-id", "p0-1", "--p0-artifact-sha256", "5" * 64,
        "--task-budget-private", str(private_path),
        "--task-budget-sanitized", str(public_path),
    ]
    if include_root:
        argv.extend(["--mission-root", str(mission_root)])
    return build_parser(ROOT).parse_args(argv)


def test_cli_controlled_bootstrap_requires_mission_root_and_exact_receipts(
    tmp_path, monkeypatch,
):
    admitted = _cli_p1_campaign_admission(
        controlled_bootstrap_args(tmp_path, monkeypatch),
        creation_path="bootstrap")
    assert admitted["approved_task_count"] == 8
    without_root = tmp_path / "without-root"
    without_root.mkdir()
    with pytest.raises(RuntimeError, match="mission-root"):
        _cli_p1_campaign_admission(
            controlled_bootstrap_args(
                without_root, monkeypatch, include_root=False),
            creation_path="bootstrap")


def test_cli_controlled_bootstrap_rejects_queue_knob_drift(tmp_path, monkeypatch):
    args = controlled_bootstrap_args(tmp_path, monkeypatch)
    args.selection_strategy = "max-clearance-v1"
    with pytest.raises(RuntimeError, match="execution"):
        _cli_p1_campaign_admission(args, creation_path="bootstrap")


def test_seed_recovery_cannot_bypass_controlled_authority_by_omitting_mission(
    tmp_path,
):
    source_db = tmp_path / "controlled-source.sqlite"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE tasks(payload_json TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO tasks(payload_json) VALUES(?)",
            (json.dumps({"campaign_budget": {
                "schema": "campaignx.first_letters_task_budget_admission.v1",
            }}),),
        )
    output_db = tmp_path / "recovery.sqlite"
    args = build_parser(ROOT).parse_args([
        "bootstrap-seed-recovery",
        "--source-db", str(source_db),
        "--source-attempt-root", str(tmp_path / "attempts"),
        "--db", str(output_db),
        "--receipt", str(tmp_path / "receipt.json"),
        "--grid-version", "recovery-grid-v1",
        "--policy-version", "recovery-policy-v1",
    ])
    with pytest.raises(RuntimeError, match="not authorized"):
        args.handler(args)
    assert not output_db.exists()


def test_resume_cannot_be_unfiled_by_omitting_a_controlled_parent_mission(
    tmp_path, monkeypatch,
):
    class ParentStore:
        def surface_artifact(self, _surface_id):
            return {
                "surface_id": "surface-1", "source_snapshot_id": "source-1",
                "sample_id": "PHerc358", "artifact_uri": "fixture://surface",
                "artifact_sha256": "f" * 64,
                "payload": {"task_id": "controlled-parent"},
            }

        def task_packet(self, _task_id):
            return {
                "task_id": "controlled-parent", "mission_id": "first-letters",
                "source_snapshot_id": "source-1",
                "campaign_budget": {
                    "schema": "campaignx.first_letters_task_budget_admission.v1",
                },
            }

        def initialize(self):
            raise AssertionError("resume mutated the store before mission admission")

    monkeypatch.setattr("fleet.cli.open_fleet_store", lambda _db: ParentStore())
    args = build_parser(ROOT).parse_args([
        "bootstrap-resume", "--db", "fixture.sqlite",
        "--surface", "surface-1", "--corrections", str(tmp_path / "points.json"),
    ])
    with pytest.raises(RuntimeError, match="exact --mission-id"):
        args.handler(args)


@pytest.mark.parametrize("creation_path", [
    "manual-seeds", "replan", "resume-correction", "seed-recovery",
    "adaptive-retry",
])
def test_panel_guard_invokes_central_authority_for_every_nonprefix_route(
    tmp_path, monkeypatch, creation_path,
):
    mission_contract.create(
        tmp_path, mission_id="first-letters", name="letters",
        scrolls=["PHerc0358"], created_by="alice",
        **controlled_campaign_binding())
    app = app_module(tmp_path, monkeypatch)
    app.RUNS = tmp_path
    with pytest.raises(Exception, match="not authorized"):
        app._guard_controlled_p1_creation_path(
            "first-letters", creation_path=creation_path)
