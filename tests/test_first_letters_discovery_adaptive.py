from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import generator, seed_probe
from fleet.common import content_sha256
from fleet.store import FleetStore


def _candidate(candidate_id: str, ct: str, clearance: str, probe: str = "PROBE_NOT_RUN_DUE_TO_UPSTREAM") -> dict:
    return {
        "candidate_id": candidate_id,
        "ct_terminal": {"candidate_id": candidate_id, "status": ct, "row_sha256": (candidate_id * 64)[:64]},
        "clearance_terminal": {"candidate_id": candidate_id, "status": clearance, "row_sha256": "e" * 64},
        "probe_terminal": {"candidate_id": candidate_id, "status": probe, "row_sha256": "f" * 64},
    }


def _parent(rows, reason="RAW_CANDIDATES_FAILED_CT_OR_CLEARANCE") -> dict:
    value = {
        "schema": "campaignx.first_letters_discovery_causal_receipt.v1",
        "parent_task_id": "parent-a", "parent_attempt_id": "attempt-a",
        "scientific_opportunity_id": "opportunity-a", "reason": reason,
        "raw_candidate_count": len(rows), "candidates": rows,
        "namespace": "NONCANONICAL_DISCOVERY", "allow_unvalidated": False,
    }
    value["receipt_sha256"] = content_sha256(value)
    return value


def _reconcile(parent):
    assert hasattr(seed_probe, "reconcile_adaptive_causal_receipt")
    return seed_probe.reconcile_adaptive_causal_receipt(parent)


def _grid():
    return {
        "schema": "campaignx.canonical_grid_spec.v1",
        "grid_version": "grid-v1",
        "topology_id": "AXIS_ALIGNED_FACE_NEIGHBORS_V1",
        "shape_indices_xyz": [3, 3, 3],
    }


def _profile():
    return {"top_k": 2, "probe_generations": 12, "maximum_attempts_per_candidate": 1,
            "probe_profile_sha256": "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33",
            "allow_unvalidated": False}


def test_every_raw_candidate_has_one_ct_terminal_and_one_clearance_terminal():
    duplicate = _candidate("a", "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_NOT_RUN_DUE_TO_CT")
    parent = _parent([duplicate, copy.deepcopy(duplicate)])
    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE"):
        _reconcile(parent)


def test_ct_rejection_requires_literal_clearance_not_run_due_to_ct():
    parent = _parent([_candidate("a", "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_PASSED")])
    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE"):
        _reconcile(parent)


@pytest.mark.parametrize("rows,vector", [
    ([_candidate("a", "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_NOT_RUN_DUE_TO_CT")], "ALL_CT_REJECTED"),
    ([_candidate("a", "CT_RETAINED", "CLEARANCE_REJECTED_CELL_INTERIOR")], "ALL_CLEARANCE_REJECTED"),
    ([_candidate("a", "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_NOT_RUN_DUE_TO_CT"), _candidate("b", "CT_RETAINED", "CLEARANCE_REJECTED_VOLUME_INTERIOR")], "MIXED_CT_AND_CLEARANCE_REJECTED"),
])
def test_only_all_ct_rejected_all_clearance_rejected_and_reconciled_mixed_vectors_are_eligible(rows, vector):
    assert _reconcile(_parent(rows))["cause_vector_id"] == vector


def test_measurable_probe_geometry_requires_exact_complete_ct_clearance_probe_vectors():
    row = _candidate("a", "CT_RETAINED", "CLEARANCE_PASSED", "PROBE_MEASURABLE_NONCANONICAL_GEOMETRY")
    row["probe_terminal"].update({"namespace": "NONCANONICAL_DISCOVERY", "measurement_manifest_sha256": "1" * 64})
    result = _reconcile(_parent([row], "MEASURABLE_NONCANONICAL_PROBE_GEOMETRY"))
    assert result["eligible"] is True


@pytest.mark.parametrize("row", [
    _candidate("a", "CT_NOT_RUN_NONINTEGRAL_COORDINATE", "CLEARANCE_NOT_RUN_DUE_TO_CT"),
    _candidate("a", "CT_NOT_RUN_MALFORMED_COORDINATE", "CLEARANCE_NOT_RUN_DUE_TO_CT"),
    _candidate("a", "CT_INCOMPLETE_PLATFORM_OR_SOURCE", "CLEARANCE_INCOMPLETE"),
    _candidate("a", "CT_RETAINED", "CLEARANCE_PASSED"),
])
def test_nonintegral_malformed_incomplete_passed_or_unlisted_mixed_rows_are_ineligible(row):
    assert _reconcile(_parent([row]))["eligible"] is False


def test_adaptive_uses_versioned_canonical_grid_topology_and_literal_boundary_neighbors():
    assert hasattr(generator, "canonical_grid_neighbors")
    assert generator.canonical_grid_neighbors(_grid(), "r00001c00001a00001") == [
        "r00000c00001a00001", "r00002c00001a00001",
        "r00001c00000a00001", "r00001c00002a00001",
        "r00001c00001a00000", "r00001c00001a00002",
    ]
    assert generator.canonical_grid_neighbors(_grid(), "r00000c00000a00000") == [
        "r00001c00000a00000", "r00000c00001a00000", "r00000c00000a00001"
    ]


def test_adaptive_without_declared_topology_is_unavailable_not_chebyshev_inferred():
    grid = _grid(); del grid["topology_id"]
    with pytest.raises(ValueError, match="ADAPTIVE_GRID_TOPOLOGY_UNAVAILABLE"):
        generator.canonical_grid_neighbors(grid, "r00001c00001a00001")


def test_adaptive_v1_requires_exact_probe_profile_sha_top2_generations12_attempts1():
    assert hasattr(seed_probe, "validate_adaptive_profile_v1")
    assert seed_probe.validate_adaptive_profile_v1(_profile())["top_k"] == 2
    for field, bad in (("top_k", 3), ("probe_generations", 11), ("maximum_attempts_per_candidate", 2), ("probe_profile_sha256", "0" * 64)):
        profile = _profile(); profile[field] = bad
        with pytest.raises(ValueError, match="ADAPTIVE_PROFILE_UNSUPPORTED_V1"):
            seed_probe.validate_adaptive_profile_v1(profile)


def test_adaptive_budget_uses_literal_24_integer_units_floor_and_mission_cap():
    assert seed_probe.adaptive_item_prefix_count(cap_units=240, committed_units=49, available_neighbor_count=8) == 7
    assert seed_probe.adaptive_item_prefix_count(cap_units=48, committed_units=0, available_neighbor_count=8) == 2


@pytest.mark.parametrize("value", [True, 1.5, -1, 2**63])
def test_adaptive_rejects_fraction_bool_negative_overflow_or_mixed_units(value):
    with pytest.raises(ValueError):
        seed_probe.adaptive_item_prefix_count(cap_units=value, committed_units=0, available_neighbor_count=1)


def test_adaptive_is_one_generation_eight_max_and_zero_statistical_rank_delta():
    proposal = seed_probe.build_adaptive_proposal_v1(
        parent_reconciliation=_reconcile(_parent([_candidate("a", "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_NOT_RUN_DUE_TO_CT")])),
        grid_spec=_grid(), parent_cell_id="r00001c00001a00001", profile=_profile(),
        cap_units=1000, committed_units=0,
    )
    assert proposal["generation"] == 1
    assert len(proposal["selected_neighbor_ids"]) <= 8
    assert proposal["statistical_budget_delta"] == 0


def test_adaptive_selection_is_invariant_to_outcomes_and_content_signals():
    reconciliation = _reconcile(_parent([_candidate("a", "CT_REJECTED_NO_NEARBY_MATERIAL", "CLEARANCE_NOT_RUN_DUE_TO_CT")]))
    first = seed_probe.build_adaptive_proposal_v1(parent_reconciliation=reconciliation, grid_spec=_grid(), parent_cell_id="r00001c00001a00001", profile=_profile(), cap_units=240, committed_units=0)
    outside = {"parent_reconciliation": copy.deepcopy(reconciliation), "p5_score": 0.99}
    second = seed_probe.build_adaptive_proposal_v1(parent_reconciliation=outside["parent_reconciliation"], grid_spec=_grid(), parent_cell_id="r00001c00001a00001", profile=_profile(), cap_units=240, committed_units=0)
    assert first["selected_neighbor_ids"] == second["selected_neighbor_ids"]


def test_adaptive_enqueue_requires_same_active_task9_wave_and_never_adds_rank(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    assert hasattr(store, "reserve_discovery_compute")
    with pytest.raises(ValueError, match="TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED"):
        store.reserve_discovery_compute(mission_id="m", request_id="r", work_kind="ADAPTIVE_CHILD", work_authority={}, work_authority_id="a", work_authority_sha256="0" * 64, ordered_item_ids=["cell"], cap_authority_id="c", cap_authority_sha256="1" * 64, reservation_mode="PREFIX_TO_CAP")


def test_adaptive_prefix_is_derived_from_all_work_kinds_in_common_ledger_under_mission_lock():
    assert seed_probe.adaptive_item_prefix_count(cap_units=96, committed_units=72, available_neighbor_count=8) == 1


def test_adaptive_zero_unit_prefix_creates_no_reservation_authority_or_child():
    assert seed_probe.adaptive_item_prefix_count(cap_units=48, committed_units=48, available_neighbor_count=8) == 0
