from __future__ import annotations

import copy
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256
from fleet.store import FleetStore


def _cap(cap_units: int = 48) -> dict:
    value = {
        "schema": "campaignx.first_letters_discovery_compute_cap.v1",
        "mission_id": "mission-a",
        "cap_authority_id": "cap-a",
        "compute_unit": "probe_generation_units",
        "mission_compute_cap_units": cap_units,
        "top_k": 2,
        "probe_generations": 12,
        "maximum_attempts_per_candidate": 1,
        "probe_profile_id": "vc3d-m7-probe-v1",
        "probe_profile_file_sha256": "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33",
        "deployed_revision": "1" * 40,
        "policy_chain_id": "policy-chain-a",
        "policy_chain_sha256": "2" * 64,
        "allow_unvalidated": False,
    }
    value["authority_sha256"] = content_sha256(value)
    return value


def _work(kind: str, request_id: str, items=("cell-a", "cell-b")) -> dict:
    schema = {
        "BASELINE_ARM": "campaignx.first_letters_discovery_baseline_work_admission.v1",
        "ALTERNATIVE_SOURCE_ARM": "campaignx.first_letters_experimental_arm_admission.v1",
        "ADAPTIVE_CHILD": "campaignx.first_letters_discovery_adaptive.v1",
    }[kind]
    region = {"minimum": [0, 0, 0], "maximum": [64, 64, 64]}
    bindings = [{
        "schema": "campaignx.first_letters_discovery_work_item_binding.v1",
        "item_id": item_id, "sample_id": "PHercA",
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "5" * 64,
        "cell_region": region,
        "cell_region_sha256": content_sha256(region),
        "grid_version": "first-letters-grid-v1",
        "grid_spec_sha256": content_sha256({
            "grid_version": "first-letters-grid-v1", "cell_id": item_id,
            "ct_l0_region": region,
        }),
        "scientific_opportunity_id": f"opportunity-{item_id}",
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": "a" * 64,
        "parent_task_id": None, "parent_attempt_id": None,
        "allow_unvalidated": False,
    } for item_id in items]
    value = {
        "schema": schema,
        "work_authority_id": f"authority-{request_id}",
        "mission_id": "mission-a",
        "work_kind": kind,
        "ordered_item_ids": list(items),
        "ordered_item_ids_sha256": content_sha256(list(items)),
        "ordered_item_bindings": bindings,
        "ordered_item_bindings_sha256": content_sha256(bindings),
        "cap_authority_id": "cap-a",
        "cap_authority_sha256": _cap()["authority_sha256"],
        "profile_sha256": "3" * 64,
        "policy_sha256": "4" * 64,
        "source_sha256": "5" * 64,
        "deployed_revision": "1" * 40,
        "requested_item_count": len(items),
        "requested_units": len(items) * 24,
        "allow_unvalidated": False,
    }
    value["work_authority_sha256"] = content_sha256(value)
    return value


def _store(tmp_path: Path, cap_units: int = 48) -> FleetStore:
    cap = _cap(cap_units)
    profile = json.loads((
        STAGE / "fleet/profiles/first-letters-discovery-v1.json"
    ).read_text(encoding="utf-8"))
    profile.update({
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "5" * 64,
        "source_content_lock_sha256": "d" * 64,
        "ct_metadata_sha256": "6" * 64,
        "ct_read_set_manifest_sha256": "7" * 64,
        "m7_model_id": "m7-v1", "m7_resolution": 4, "m7_level": 1,
        "m7_threshold": 0.5, "m7_transform_sha256": "8" * 64,
        "m7_read_set_manifest_sha256": "9" * 64,
        "canonical_ordered_cell_set_sha256": content_sha256(
            ["cell-a", "cell-b"]
        ),
        "mission_compute_cap_authority_id": "cap-a",
        "mission_compute_cap_authority_sha256": cap["authority_sha256"],
        "mission_compute_cap_units": cap_units,
        "deployed_revision": "1" * 40,
    })
    profile["scientific_core_sha256"] = content_sha256({
        key: value for key, value in profile.items()
        if key != "scientific_core_sha256"
    })
    profile_bytes = json.dumps(
        profile, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    store = FleetStore(
        tmp_path / "fleet.sqlite",
        task9_discovery_gate_resolver=lambda mission_id: {
            "schema": "campaignx.first_letters_task9_discovery_gate.v1",
            "mission_id": mission_id,
            "gate_sha256": "9" * 64,
            "allow_unvalidated": False,
        },
        first_letters_discovery_profile_resolver=(
            lambda mission_id, source_snapshot_id: store._fixture_profile_bytes
        ),
    )
    store.initialize()
    store.register_snapshot({
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "5" * 64,
        "source_content_lock_sha256": "d" * 64,
        "sample_id": "PHercA", "ct_uri": "fixture://ct",
        "ct_sha256": "6" * 64, "ct_metadata_sha256": "6" * 64,
        "ct_read_set_manifest_sha256": "7" * 64,
        "m7_uri": "fixture://m7", "m7_sha256": "a" * 64,
        "m7_metadata_sha256": "a" * 64,
        "m7_read_set_manifest_sha256": "9" * 64,
        "m7_model_id": "m7-v1", "m7_resolution": 4, "m7_level": 1,
        "m7_threshold": 0.5, "m7_transform_sha256": "8" * 64,
        "shape_xyz": [64, 64, 64], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a", "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": "a" * 64,
            "scientific_opportunities": {},
        },
    })
    assert hasattr(store, "register_discovery_compute_cap")
    store.register_discovery_compute_cap(cap)
    store._fixture_profile_template = copy.deepcopy(profile)
    store._fixture_profile_bytes = profile_bytes
    store._fixture_native_lock = threading.Lock()
    store._fixture_admissions = {}
    return store


def _native_authority(store: FleetStore, items: tuple[str, ...]):
    key = tuple(items)
    with store._fixture_native_lock:
        profile = copy.deepcopy(store._fixture_profile_template)
        profile["canonical_ordered_cell_set_sha256"] = content_sha256(
            list(items)
        )
        profile["scientific_core_sha256"] = content_sha256({
            field: value for field, value in profile.items()
            if field != "scientific_core_sha256"
        })
        store._fixture_profile_bytes = json.dumps(
            profile, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if key in store._fixture_admissions:
            return store._fixture_admissions[key]
        core = {
            "schema": "campaignx.first_letters_task_budget_admission.v1",
            "mission_id": "mission-a", "sample_id": "PHercA",
            "receipt_sha256": content_sha256({"items": list(items)}),
            "preflight_receipt_sha256": "b" * 64,
            "preflight_sanitized_receipt_sha256": "c" * 64,
            "approved_task_count": len(items), "order_seed_sha256": "d" * 64,
            "population_order_sha256": "e" * 64,
            "prefix_sha256": content_sha256(list(items)),
            "prefix_cell_ids": list(items),
            "execution_bindings": {
                "source_snapshot_id": "source-a",
                "grid_version": "first-letters-grid-v1",
                "policy_version": "first-letters-search@1.0.0",
                "p0_artifact_id": "p0-a", "p0_artifact_sha256": "a" * 64,
                "catalog_snapshot_sha256": "f" * 64,
            },
        }
        admission = {**core, "admission_sha256": content_sha256(core)}
        with store.connect() as connection:
            connection.execute(
                """INSERT INTO campaign_budget_admissions
                   (mission_id,sample_id,receipt_sha256,admission_json,
                    admission_sha256,created_at) VALUES(?,?,?,?,?,?)""",
                (
                    "mission-a", "PHercA", admission["receipt_sha256"],
                    json.dumps(admission, sort_keys=True, separators=(",", ":")),
                    admission["admission_sha256"], "2026-01-01T00:00:00Z",
                ),
            )
            source = json.loads(connection.execute(
                "SELECT payload_json FROM source_snapshots "
                "WHERE source_snapshot_id='source-a'"
            ).fetchone()[0])
            scope = source["first_letters_discovery_authority"]
            scope["scientific_opportunities"].update({
                item: f"opportunity-{item}" for item in items
            })
            connection.execute(
                "UPDATE source_snapshots SET payload_json=? "
                "WHERE source_snapshot_id='source-a'",
                (json.dumps(source, sort_keys=True, separators=(",", ":")),),
            )
            for rank, item in enumerate(items):
                if connection.execute(
                    "SELECT 1 FROM tasks WHERE mission_id=? AND cell_id=?",
                    ("mission-a", item),
                ).fetchone() is not None:
                    continue
                payload = {
                    "sample_id": "PHercA", "selection_rank": rank,
                    "campaign_budget_admission_sha256":
                        admission["admission_sha256"],
                    "scientific_opportunity_id": f"opportunity-{item}",
                    "p0_artifact_id": "p0-a",
                    "p0_artifact_sha256": "a" * 64,
                    "candidate_discovery": {"region": {
                        "minimum": [0, 0, 0], "maximum": [64, 64, 64],
                    }},
                }
                connection.execute(
                    """INSERT INTO tasks
                       (task_id,mission_id,source_snapshot_id,cell_id,
                        grid_version,policy_version,bounds_xyz_json,
                        center_xyz_json,priority,parameter_envelope_json,
                        catalog_snapshot_sha256,payload_json,state,gpu_required,
                        minimum_vram_gb,seed_probe_required,created_at,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        f"task-{item}", "mission-a", "source-a", item,
                        "first-letters-grid-v1", "first-letters-search@1.0.0",
                        "[[0,0,0],[64,64,64]]", '{"x":32,"y":32,"z":32}',
                        1.0, "{}", "f" * 64,
                        json.dumps(payload, sort_keys=True,
                                   separators=(",", ":")),
                        "PENDING", 0, 0.0, 0,
                        "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
                    ),
                )
        store._fixture_admissions[key] = admission
        return admission


def _reserve(store: FleetStore, kind="BASELINE_ARM", request_id="request-a", items=("cell-a", "cell-b"), **extra):
    if kind in {"BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM"}:
        admission = _native_authority(store, tuple(items))
        failpoint = extra.pop("failpoint", None)
        if extra:
            unexpected = next(iter(extra))
            raise TypeError(f"unexpected keyword argument: {unexpected}")
        if kind == "BASELINE_ARM":
            return store.reserve_first_letters_baseline_shadow(
                request_id=request_id,
                budget_admission_sha256=admission["admission_sha256"],
                failpoint=failpoint,
            )
        with store.connect() as connection:
            source = store._snapshot(connection.execute(
                "SELECT * FROM source_snapshots "
                "WHERE source_snapshot_id='source-a'"
            ).fetchone())
        profile_sha = hashlib.sha256(store._fixture_profile_bytes).hexdigest()
        arm_core = {
            "schema": "campaignx.first_letters_experimental_arm_admission.v1",
            "arm_id": "arm-a", "mission_id": "mission-a",
            "accepted_p0_id": "p0-a", "accepted_p0_sha256": "a" * 64,
            **{field: source[field] for field in (
                "source_snapshot_id", "source_snapshot_sha256",
                "source_content_lock_sha256", "ct_metadata_sha256",
                "ct_read_set_manifest_sha256", "m7_metadata_sha256",
                "m7_read_set_manifest_sha256", "m7_model_id", "m7_resolution",
                "m7_level", "m7_transform_sha256", "m7_threshold",
            )},
            "discovery_policy_id": "first-letters-alt@1.0.0",
            "discovery_profile_sha256": profile_sha,
            "deployed_revision": "1" * 40,
            "preflight_private_sha256": "0" * 64,
            "preflight_sanitized_sha256": "1" * 64,
            "ordered_cell_ids": list(items),
            "ordered_cell_set_sha256": content_sha256(list(items)),
            "mission_compute_cap_authority_id": "cap-a",
            "mission_compute_cap_authority_sha256":
                store.discovery_compute_cap("mission-a")["authority_sha256"],
            "requested_units": len(items) * 24,
            "active_policy_chain_sha256": "2" * 64,
            "may_update_accepted_p0": False,
            "statistical_budget_delta": 0, "allow_unvalidated": False,
        }
        arm = {**arm_core, "admission_sha256": content_sha256(arm_core)}
        store._first_letters_experimental_arm_resolver = (
            lambda arm_id: copy.deepcopy(arm) if arm_id == "arm-a" else None
        )
        return store.reserve_first_letters_alternative_shadow(
            request_id=request_id,
            budget_admission_sha256=admission["admission_sha256"],
            arm_id="arm-a", failpoint=failpoint,
        )
    work = _work(kind, request_id, items)
    return store.reserve_discovery_compute(
        mission_id="mission-a", request_id=request_id, work_kind=kind,
        work_authority=work,
        work_authority_id=work["work_authority_id"],
        work_authority_sha256=work["work_authority_sha256"],
        ordered_item_ids=list(items), cap_authority_id="cap-a",
        cap_authority_sha256=_cap()["authority_sha256"],
        reservation_mode="PREFIX_TO_CAP" if kind == "ADAPTIVE_CHILD" else "EXACT",
        task9_gate=(
            {"schema": "campaignx.first_letters_task9_discovery_gate.v1", "mission_id": "mission-a", "gate_sha256": "9" * 64, "allow_unvalidated": False}
            if kind == "ADAPTIVE_CHILD" else None
        ),
        **extra,
    )


def test_compute_cap_authority_freezes_unit_2x12x1_cost_cap_policy_profile_and_false_override(tmp_path):
    store = _store(tmp_path)
    assert store.discovery_compute_cap("mission-a")["mission_compute_cap_units"] == 48
    for field, bad in (("top_k", 3), ("probe_generations", 11), ("maximum_attempts_per_candidate", 2), ("allow_unvalidated", True)):
        authority = _cap()
        authority[field] = bad
        authority["authority_sha256"] = content_sha256({key: row for key, row in authority.items() if key != "authority_sha256"})
        with pytest.raises(ValueError):
            FleetStore(tmp_path / f"{field}.sqlite").register_discovery_compute_cap(authority)


def test_baseline_alternative_and_adaptive_use_one_reservation_schema_table_and_api(tmp_path):
    for index, kind in enumerate(("BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM", "ADAPTIVE_CHILD")):
        store = _store(tmp_path / str(index), 48)
        result = _reserve(store, kind, f"request-{index}", (f"cell-{index}",))
        assert result["reservation"]["schema"] == "campaignx.first_letters_discovery_compute_reservation.v1"
        assert store.discovery_compute_total("mission-a") == 24


def test_every_v1_item_costs_literal_24_units_and_rejects_caller_cost_or_unit_coercion(tmp_path):
    store = _store(tmp_path)
    result = _reserve(store)
    assert result["reservation"]["units_per_item"] == 24
    assert result["reservation"]["reserved_units"] == 48
    with pytest.raises(TypeError):
        _reserve(store, request_id="request-b", caller_units=1)


@pytest.mark.parametrize("kind", ["BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM", "ADAPTIVE_CHILD"])
def test_baseline_reservation_and_durable_dispatch_commit_or_rollback_together(tmp_path, kind):
    store = _store(tmp_path, 48)
    result = _reserve(store, kind, "request-a")
    assert result["work"]["work_kind"] == kind
    assert store.discovery_compute_rows("mission-a") == [{
        "reservation": result["reservation"], "work": result["work"],
    }]


def test_alternative_reservation_and_arm_dispatch_commit_or_rollback_together(tmp_path):
    store = _store(tmp_path)
    assert _reserve(store, "ALTERNATIVE_SOURCE_ARM")["work"]["dispatch_kind"] == "ALTERNATIVE_SOURCE_DISPATCH"


def test_adaptive_reservation_authority_and_children_commit_or_rollback_together(tmp_path):
    store = _store(tmp_path)
    result = _reserve(store, "ADAPTIVE_CHILD")
    assert result["work"]["dispatch_kind"] == "ADAPTIVE_CHILDREN"
    assert result["work"]["ordered_item_ids"] == ["cell-a", "cell-b"]


def test_claim_and_executor_reject_missing_wrong_kind_wrong_authority_or_wrong_item_reservation(tmp_path):
    store = _store(tmp_path)
    result = _reserve(store)
    reservation = result["reservation"]
    assert store.validate_discovery_compute_reservation(
        reservation["reservation_id"], reservation["reservation_sha256"],
        mission_id="mission-a", work_kind="BASELINE_ARM",
        work_authority_sha256=reservation["work_authority_sha256"],
        ordered_item_ids=["cell-a", "cell-b"],
    ) == reservation
    for drift in ({"work_kind": "ADAPTIVE_CHILD"}, {"work_authority_sha256": "0" * 64}, {"ordered_item_ids": ["cell-a"]}):
        args = {"mission_id": "mission-a", "work_kind": "BASELINE_ARM", "work_authority_sha256": reservation["work_authority_sha256"], "ordered_item_ids": ["cell-a", "cell-b"]}
        args.update(drift)
        with pytest.raises(ValueError):
            store.validate_discovery_compute_reservation(reservation["reservation_id"], reservation["reservation_sha256"], **args)


def test_exact_request_replay_returns_identical_rows_and_changed_payload_conflicts(tmp_path):
    store = _store(tmp_path)
    first = _reserve(store)
    assert _reserve(store) == first
    with pytest.raises(ValueError, match="CONFLICT|budget authority"):
        _reserve(store, items=("cell-a",), request_id="request-a")
    assert store.discovery_compute_total("mission-a") == 48


def _historical(kind="BASELINE_ARM", items=("cell-a",)):
    return {
        "schema": "campaignx.first_letters_discovery_historical_execution_manifest.v1",
        "work_kind": kind, "ordered_item_ids": list(items),
        "top_k": 2, "probe_generations": 12,
        "maximum_attempts_per_candidate": 1,
        "probe_profile_file_sha256": "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33",
        "receipt_sha256s": ["8" * 64 for _ in items],
        "allow_unvalidated": False,
    }


def test_historical_caller_attestation_cannot_enter_native_reservation(tmp_path):
    store = _store(tmp_path, 48)
    with pytest.raises(TypeError):
        _reserve(
            store, request_id="historical", items=("old",),
            source="IMPORTED_HISTORICAL_EXACT",
            historical_execution_manifest=_historical(items=("old",)),
        )
    assert store.discovery_compute_total("mission-a") == 0


def test_incomplete_reconstructed_history_blocks_all_new_work(tmp_path):
    from test_first_letters_discovery_shadow_bridge import (
        _insert_incomplete_legacy_probe,
    )

    store = _store(tmp_path)
    complete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    assert complete["state"] == "COMPLETE"
    _insert_incomplete_legacy_probe(store)
    incomplete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"):
        _reserve(store, request_id="new")


def test_cancelled_failed_or_abstained_work_does_not_release_or_reduce_reservation(tmp_path):
    store = _store(tmp_path)
    _reserve(store)
    for outcome in ("CANCELLED", "FAILED", "ABSTAINED"):
        store.record_discovery_compute_outcome("mission-a", "request-a", outcome)
        assert store.discovery_compute_total("mission-a") == 48


@pytest.mark.parametrize("pair", [
    ("BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM"),
    ("BASELINE_ARM", "ADAPTIVE_CHILD"),
    ("ALTERNATIVE_SOURCE_ARM", "ADAPTIVE_CHILD"),
])
def test_sqlite_mixed_baseline_alternative_adaptive_concurrency_never_exceeds_48_unit_cap(tmp_path, pair):
    store = _store(tmp_path)
    def reserve(index):
        try:
            return _reserve(store, pair[index], f"request-{index}")
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, (0, 1)))
    assert sum(result is not None and result["reservation"]["reserved_units"] or 0 for result in results) == 48
    assert store.discovery_compute_total("mission-a") == 48


def test_same_request_concurrency_commits_one_byte_identical_reservation_and_work_set(tmp_path):
    store = _store(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _reserve(store), (0, 1)))
    assert results[0] == results[1]
    assert len(store.discovery_compute_rows("mission-a")) == 1


@pytest.mark.parametrize("failpoint", [
    "bridge.before_reservation",
    "bridge.after_reservation_before_adapter",
    "bridge.after_adapter_before_dispatch",
    "bridge.after_dispatch_before_jobs",
    "bridge.after_each_job",
    "bridge.after_jobs_before_commit",
    "bridge.before_commit",
])
def test_each_compute_precommit_failpoint_leaves_zero_reservation_and_work_rows(tmp_path, failpoint):
    store = _store(tmp_path)
    with pytest.raises(RuntimeError):
        _reserve(store, failpoint=failpoint)
    assert store.discovery_compute_rows("mission-a") == []


def test_compute_commit_unknown_requires_complete_exact_readback_before_replay(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(RuntimeError, match="CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK"):
        _reserve(store, failpoint="bridge.commit_outcome_unknown")
    readback = store.read_first_letters_discovery_request(
        "mission-a", "request-a"
    )
    assert readback["reservation"]["reserved_units"] == 48
    assert _reserve(store) == readback


def test_compute_after_commit_response_loss_recovers_exact_reservation_and_work_rows(tmp_path):
    store = _store(tmp_path)
    result = _reserve(store, failpoint="bridge.after_commit_before_response")
    assert result == store.read_first_letters_discovery_request(
        "mission-a", "request-a"
    )


def test_no_baseline_alternative_benchmark_shadow_select_or_adaptive_probe_path_bypasses_ledger(tmp_path):
    store = _store(tmp_path)
    assert not hasattr(store, "enqueue_discovery_work")
    with pytest.raises(ValueError):
        store.validate_discovery_compute_reservation("missing", "0" * 64, mission_id="mission-a", work_kind="BASELINE_ARM", work_authority_sha256="1" * 64, ordered_item_ids=["cell-a"])
