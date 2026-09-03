from __future__ import annotations

import copy
import concurrent.futures
import importlib
import inspect
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import seed_probe
from fleet.common import content_sha256
from fleet.store import FleetStore


_DISCOVERY_SHADOW_TABLE_ALLOWLIST = frozenset({
    "first_letters_discovery_compute_blocks",
    "first_letters_discovery_compute_caps",
    "first_letters_discovery_compute_outcomes",
    "first_letters_discovery_compute_reservations",
    "first_letters_discovery_dispatches_v19",
    "first_letters_discovery_evidence_files",
    "first_letters_discovery_evidence_runs",
    "first_letters_discovery_evidence_sets",
    "first_letters_discovery_executor_claims",
    "first_letters_discovery_executor_registry",
    "first_letters_discovery_historical_imports_v19",
    "first_letters_discovery_history_reconciliations_v19",
    "first_letters_discovery_jobs_v19",
    "first_letters_discovery_native_adapters_v19",
    "first_letters_discovery_promotion_attempt_bindings",
    "first_letters_discovery_promotions",
    "first_letters_discovery_work_bindings",
})


def _sqlite_table_projection(
    path: Path, *, excluded: frozenset[str] = frozenset(),
) -> dict[str, dict[str, object]]:
    """Return every persisted user-table byte/value outside an explicit cut."""

    with sqlite3.connect(path) as connection:
        tables = [str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )]
        observed_discovery = {
            table for table in tables
            if table.startswith("first_letters_discovery_")
        }
        assert observed_discovery == _DISCOVERY_SHADOW_TABLE_ALLOWLIST
        projection = {}
        for table in tables:
            if table in excluded:
                continue
            columns = tuple(str(row[1]) for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            ))
            rows = [tuple(row) for row in connection.execute(
                f'SELECT * FROM "{table}"'
            )]
            projection[table] = {
                "columns": columns,
                "rows": tuple(sorted(rows, key=repr)),
            }
    return projection


def _clone_live_bridge_store(template: FleetStore, path: Path) -> FleetStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    with template.connect() as source, sqlite3.connect(path) as target:
        source.backup(target)
    return FleetStore(
        path,
        first_letters_discovery_executor=(
            template._first_letters_discovery_executor
        ),
        first_letters_discovery_worker_id=(
            template._first_letters_discovery_worker_id
        ),
        first_letters_discovery_executor_id=(
            template._first_letters_discovery_executor_id
        ),
        first_letters_discovery_profile_resolver=(
            template._first_letters_discovery_profile_resolver
        ),
        first_letters_experimental_arm_resolver=(
            template._first_letters_experimental_arm_resolver
        ),
    )


def _bridge():
    return importlib.import_module("fleet.discovery_bridge")


def _binding(item_id: str = "cell-a", rank: int = 0) -> dict:
    region = {"minimum": [0, 0, 0], "maximum": [64, 64, 64]}
    return {
        "schema": "campaignx.first_letters_discovery_work_item_binding.v1",
        "item_id": item_id,
        "selection_rank": rank,
        "sample_id": "PHercA",
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "cell_region": region,
        "cell_region_sha256": content_sha256(region),
        "grid_version": "first-letters-grid-v1",
        "grid_spec_sha256": content_sha256({
            "grid_version": "first-letters-grid-v1",
            "cell_id": item_id,
            "ct_l0_region": region,
        }),
        "scientific_opportunity_id": f"opportunity-{item_id}",
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": "a" * 64,
        "parent_task_id": f"task-{item_id}",
        "parent_attempt_id": None,
        "allow_unvalidated": False,
    }


def _reconciliation(items: tuple[str, ...] = ("cell-a",)) -> dict:
    bindings = [_binding(item, rank) for rank, item in enumerate(items)]
    core = {
        "schema":
            "campaignx.first_letters_discovery_baseline_reconciliation.v1",
        "mission_id": "mission-a",
        "request_id": "request-a",
        "sample_id": "PHercA",
        "budget_admission_sha256": "b" * 64,
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "2" * 64,
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": "a" * 64,
        "grid_version": "first-letters-grid-v1",
        "ordered_item_ids": list(items),
        "ordered_item_ids_sha256": content_sha256(list(items)),
        "ordered_item_bindings": bindings,
        "ordered_item_bindings_sha256": content_sha256(bindings),
        "cap_authority_id": "cap-a",
        "cap_authority_sha256": "c" * 64,
        "profile_file_sha256": "d" * 64,
        "profile_scientific_core_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
        "deployed_revision": "1" * 40,
        "history_manifest_sha256": "9" * 64,
        "mode": "shadow",
        "namespace": "NONCANONICAL_DISCOVERY",
        "canonical_admission": "PROHIBITED",
        "top_k": 2,
        "probe_generations": 12,
        "maximum_attempts_per_candidate": 1,
        "units_per_item": 24,
        "allow_unvalidated": False,
    }
    return {**core, "reconciliation_sha256": content_sha256(core)}


def _arm(items: tuple[str, ...] = ("cell-a",)) -> dict:
    value = {
        "schema": "campaignx.first_letters_experimental_arm_admission.v1",
        "arm_id": "arm-a",
        "mission_id": "mission-a",
        "accepted_p0_id": "p0-a",
        "accepted_p0_sha256": "a" * 64,
        "source_snapshot_id": "source-alt",
        "source_snapshot_sha256": "3" * 64,
        "source_content_lock_sha256": "4" * 64,
        "ct_metadata_sha256": "5" * 64,
        "ct_read_set_manifest_sha256": "6" * 64,
        "m7_metadata_sha256": "7" * 64,
        "m7_read_set_manifest_sha256": "8" * 64,
        "m7_model_id": "m7-alt",
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_transform_sha256": "9" * 64,
        "m7_threshold": 0.5,
        "discovery_policy_id": "first-letters-alt@1.0.0",
        "discovery_profile_sha256": "d" * 64,
        "deployed_revision": "1" * 40,
        "preflight_private_sha256": "0" * 64,
        "preflight_sanitized_sha256": "1" * 64,
        "ordered_cell_ids": list(items),
        "ordered_cell_set_sha256": content_sha256(list(items)),
        "mission_compute_cap_authority_id": "cap-a",
        "mission_compute_cap_authority_sha256": "c" * 64,
        "requested_units": len(items) * 24,
        "active_policy_chain_sha256": "f" * 64,
        "may_update_accepted_p0": False,
        "statistical_budget_delta": 0,
        "allow_unvalidated": False,
    }
    value["admission_sha256"] = content_sha256(value)
    return seed_probe.validate_experimental_arm_admission(value)


def _alternative_source() -> dict:
    return {
        "source_snapshot_id": "source-alt",
        "source_snapshot_sha256": "3" * 64,
        "source_content_lock_sha256": "4" * 64,
        "ct_metadata_sha256": "5" * 64,
        "ct_read_set_manifest_sha256": "6" * 64,
        "m7_metadata_sha256": "7" * 64,
        "m7_read_set_manifest_sha256": "8" * 64,
        "m7_model_id": "m7-alt",
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_transform_sha256": "9" * 64,
        "m7_threshold": 0.5,
    }


def test_real_baseline_reconciliation_derives_generic_projection_without_caller_fields():
    bridge = _bridge()
    adapter = bridge.adapt_first_letters_baseline_shadow(_reconciliation())

    assert adapter["producer_kind"] == "BASELINE_RECONCILIATION"
    assert adapter["work_kind"] == "BASELINE_ARM"
    assert adapter["reservation_mode"] == "EXACT"
    assert adapter["generic_work_authority"]["requested_units"] == 24
    assert adapter["generic_work_authority"]["ordered_item_ids"] == ["cell-a"]
    assert adapter["generic_work_authority"]["work_authority_sha256"] == (
        adapter["generic_work_authority_sha256"]
    )
    assert adapter["adapter_sha256"] == content_sha256({
        key: value for key, value in adapter.items()
        if key != "adapter_sha256"
    })


def test_real_experimental_arm_and_reconciliation_derive_alternative_projection():
    bridge = _bridge()
    adapter = bridge.adapt_first_letters_alternative_shadow(
        _reconciliation(), _arm(), _alternative_source(),
    )

    assert adapter["producer_kind"] == "EXPERIMENTAL_ARM_ADMISSION"
    assert adapter["work_kind"] == "ALTERNATIVE_SOURCE_ARM"
    assert adapter["native_authority"]["arm_admission"]["arm_id"] == "arm-a"
    assert adapter["generic_work_authority"]["source_sha256"] == "3" * 64
    assert all(
        binding["parent_task_id"] is None
        for binding in adapter["generic_work_authority"][
            "ordered_item_bindings"]
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("ordered_item_ids", ["cell-b"]),
        ("units_per_item", 12),
        ("mode", "select"),
        ("allow_unvalidated", True),
    ],
)
def test_baseline_reconciliation_rejects_closed_contract_drift(
    field, replacement,
):
    bridge = _bridge()
    bad = _reconciliation()
    bad[field] = replacement
    bad["reconciliation_sha256"] = content_sha256({
        key: value for key, value in bad.items()
        if key != "reconciliation_sha256"
    })
    with pytest.raises(ValueError):
        bridge.adapt_first_letters_baseline_shadow(bad)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("mission_id", "mission-b"),
        ("accepted_p0_id", "p0-b"),
        ("ordered_cell_ids", ["cell-b"]),
        ("requested_units", 48),
        ("may_update_accepted_p0", True),
    ],
)
def test_alternative_adapter_rejects_cohort_p0_units_or_safety_drift(
    field, replacement,
):
    bridge = _bridge()
    arm = copy.deepcopy(_arm())
    arm[field] = replacement
    arm["admission_sha256"] = content_sha256({
        key: value for key, value in arm.items()
        if key != "admission_sha256"
    })
    with pytest.raises(ValueError):
        bridge.adapt_first_letters_alternative_shadow(
            _reconciliation(), arm, _alternative_source(),
        )


def test_controller_surface_never_accepts_authority_profile_or_item_bytes():
    from fleet.discovery_controller import FirstLettersDiscoveryController

    for method in (
        "reserve_baseline_shadow", "reserve_alternative_shadow", "run_job",
    ):
        assert hasattr(FirstLettersDiscoveryController, method)
        parameters = inspect.signature(
            getattr(FirstLettersDiscoveryController, method)
        ).parameters
        assert not {
            "work_authority", "profile_bytes", "item_id", "arm_admission",
        } & set(parameters)


@pytest.mark.parametrize("retained_v16", [False, True])
def test_all_direct_item_surfaces_reject_before_provider_prepare(
    tmp_path, retained_v16,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from fleet.discovery_worker import FirstLettersDiscoveryWorker

    store, admission, profile_bytes = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    if retained_v16:
        with store.connect() as connection:
            connection.execute(
                "DELETE FROM first_letters_discovery_jobs_v19 "
                "WHERE reservation_id=?",
                (branch["reservation"]["reservation_id"],),
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_dispatches_v19 "
                "WHERE reservation_id=?",
                (branch["reservation"]["reservation_id"],),
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_native_adapters_v19 "
                "WHERE reservation_id=?",
                (branch["reservation"]["reservation_id"],),
            )
    provider = _BridgeProvider()
    arguments = {
        "lease_seconds": 60,
        "reservation_id": branch["reservation"]["reservation_id"],
        "item_id": "cell-a", "profile_bytes": profile_bytes,
    }

    with pytest.raises(ValueError, match="DISCOVERY_JOB_ID_REQUIRED"):
        store.begin_first_letters_discovery_evidence_run(**arguments)
    with pytest.raises(ValueError, match="DISCOVERY_JOB_ID_REQUIRED"):
        FirstLettersDiscoveryWorker(
            store=store, provider=provider,
        ).run_item(**arguments)
    with pytest.raises(ValueError, match="DISCOVERY_JOB_ID_REQUIRED"):
        FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=provider,
        ).run_item(**arguments)

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_executor_claims"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(("relation", "column", "replacement"), [
    ("first_letters_discovery_native_adapters_v19", "reservation_id", "other"),
    ("first_letters_discovery_native_adapters_v19", "mission_id", "other"),
    ("first_letters_discovery_native_adapters_v19", "request_id", "other"),
    ("first_letters_discovery_native_adapters_v19", "work_kind",
     "ALTERNATIVE_SOURCE_ARM"),
    ("first_letters_discovery_native_adapters_v19", "producer_kind",
     "EXPERIMENTAL_ARM_ADMISSION"),
    ("first_letters_discovery_native_adapters_v19", "native_schema", "other"),
    ("first_letters_discovery_native_adapters_v19",
     "native_authority_sha256", "b" * 64),
    ("first_letters_discovery_native_adapters_v19",
     "generic_work_authority_sha256", "b" * 64),
    ("first_letters_discovery_dispatches_v19", "reservation_id", "other"),
    ("first_letters_discovery_dispatches_v19", "mission_id", "other"),
    ("first_letters_discovery_dispatches_v19", "request_id", "other"),
    ("first_letters_discovery_dispatches_v19", "work_kind",
     "ALTERNATIVE_SOURCE_ARM"),
    ("first_letters_discovery_dispatches_v19", "adapter_sha256", "b" * 64),
    ("first_letters_discovery_dispatches_v19", "profile_file_sha256", "b" * 64),
    ("first_letters_discovery_dispatches_v19", "source_snapshot_sha256", "b" * 64),
    ("first_letters_discovery_dispatches_v19", "ordered_item_ids_sha256", "b" * 64),
    ("first_letters_discovery_dispatches_v19", "item_count", 2),
    ("first_letters_discovery_jobs_v19", "work_item_binding_sha256", "b" * 64),
    ("first_letters_discovery_jobs_v19", "profile_file_sha256", "b" * 64),
    ("first_letters_discovery_jobs_v19", "source_snapshot_sha256", "b" * 64),
])
def test_readback_and_job_claim_reject_every_adapter_dispatch_job_scalar_drift(
    tmp_path, relation, column, replacement,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    connection = store.connect()
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            f"UPDATE {relation} SET {column}=?", (replacement,),
        )
        connection.commit()
    finally:
        connection.close()
    provider = _BridgeProvider()

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_DISCOVERY_DISPATCH"):
        store.read_first_letters_discovery_request("mission-a", "request-a")
    with pytest.raises(ValueError):
        FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=provider,
        ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)

    assert provider.prepare_calls == provider.execute_calls == 0


@pytest.mark.parametrize("authority", [
    "profile_bytes", "profile_resolver_error", "cap", "source_science",
    "task_opportunity", "task_region", "task_p0",
])
def test_readback_and_job_claim_reresolve_current_authority_and_block(
    tmp_path, authority,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from test_first_letters_discovery_evidence_store import _cap

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    if authority == "profile_bytes":
        store._first_letters_discovery_profile_resolver = (
            lambda _mission_id, _source_id: b"{}"
        )
    elif authority == "profile_resolver_error":
        def resolver_error(_mission_id, _source_id):
            raise RuntimeError("resolver unavailable")
        store._first_letters_discovery_profile_resolver = resolver_error
    elif authority == "cap":
        cap = _cap()
        cap["policy_chain_sha256"] = "f" * 64
        cap["authority_sha256"] = content_sha256({
            key: value for key, value in cap.items()
            if key != "authority_sha256"
        })
        with store.connect() as connection:
            connection.execute(
                "UPDATE first_letters_discovery_compute_caps "
                "SET authority_sha256=?,authority_json=? WHERE mission_id=?",
                (
                    cap["authority_sha256"],
                    json.dumps(cap, sort_keys=True, separators=(",", ":")),
                    "mission-a",
                ),
            )
    elif authority == "source_science":
        with store.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM source_snapshots "
                "WHERE source_snapshot_id='source-a'"
            ).fetchone()
            source = json.loads(row["payload_json"])
            source["m7_model_sha256"] = "9" * 64
            connection.execute(
                "UPDATE source_snapshots SET payload_json=? "
                "WHERE source_snapshot_id='source-a'",
                (json.dumps(source, sort_keys=True, separators=(",", ":")),),
            )
    else:
        with store.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM tasks WHERE task_id='task-cell-a'"
            ).fetchone()
            payload = json.loads(row["payload_json"])
            if authority == "task_opportunity":
                payload["scientific_opportunity_id"] = "opportunity-other"
            elif authority == "task_region":
                payload["candidate_discovery"]["region"] = {
                    "minimum": [1, 1, 1], "maximum": [63, 63, 63],
                }
            else:
                payload["p0_artifact_sha256"] = "9" * 64
            connection.execute(
                "UPDATE tasks SET payload_json=? WHERE task_id='task-cell-a'",
                (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
            )
    provider = _BridgeProvider()

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_DISCOVERY_DISPATCH"):
        store.read_first_letters_discovery_request("mission-a", "request-a")
    with pytest.raises(ValueError):
        FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=provider,
        ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        block = connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()
        assert block["reason"] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


@pytest.mark.parametrize("arm_authority", ["missing", "drift", "exception"])
def test_alternative_readback_reresolves_current_arm_and_durably_blocks(
    tmp_path, arm_authority,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_alternative_shadow(
        request_id="request-alt",
        budget_admission_sha256=admission["admission_sha256"],
        arm_id="arm-a",
    )
    persisted_arm = copy.deepcopy(
        branch["adapter"]["native_authority"]["arm_admission"]
    )
    if arm_authority == "missing":
        store._first_letters_experimental_arm_resolver = lambda _arm_id: None
    elif arm_authority == "drift":
        persisted_arm["m7_threshold"] = 0.6
        persisted_arm["admission_sha256"] = content_sha256({
            key: value for key, value in persisted_arm.items()
            if key != "admission_sha256"
        })
        store._first_letters_experimental_arm_resolver = (
            lambda _arm_id: copy.deepcopy(persisted_arm)
        )
    else:
        def resolver_error(_arm_id):
            raise RuntimeError("arm resolver unavailable")
        store._first_letters_experimental_arm_resolver = resolver_error
    provider = _BridgeProvider()

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_DISCOVERY_DISPATCH"):
        store.read_first_letters_discovery_request("mission-a", "request-alt")
    with pytest.raises(ValueError):
        FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=provider,
        ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()["reason"] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


def _history_store(tmp_path: Path) -> FleetStore:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    return store


def _insert_incomplete_legacy_probe(store: FleetStore) -> None:
    from test_first_letters_discovery_evidence_store import _cap, _profile_bytes

    store.register_snapshot({
        "source_snapshot_id": "legacy-source",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "2" * 64,
        "ct_metadata_sha256": "3" * 64,
        "ct_read_set_manifest_sha256": "4" * 64,
        "m7_metadata_sha256": "5" * 64,
        "m7_read_set_manifest_sha256": "6" * 64,
        "m7_model_id": "legacy-m7", "m7_model_sha256": "7" * 64,
        "candidate_provider_id": "legacy-provider",
        "candidate_provider_sha256": "8" * 64,
        "m7_resolution": 4, "m7_level": 1, "m7_threshold": 0.5,
        "m7_transform_sha256": "9" * 64,
        "sample_id": "PHercA",
        "ct_uri": "fixture://ct",
        "m7_uri": "fixture://m7",
        "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a",
            "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": "a" * 64,
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {
                "cell-a": "opportunity-cell-a",
            },
        },
    })
    now = "2026-08-03T00:00:00Z"
    task_payload = {
        "sample_id": "PHercA",
        "scientific_opportunity_id": "opportunity-cell-a",
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": "a" * 64,
        "candidate_discovery": {
            "region": {"minimum": [0, 0, 0], "maximum": [64, 64, 64]},
        },
    }
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO tasks
               (task_id,mission_id,source_snapshot_id,cell_id,grid_version,
                policy_version,bounds_xyz_json,center_xyz_json,priority,
                parameter_envelope_json,catalog_snapshot_sha256,payload_json,
                state,gpu_required,minimum_vram_gb,seed_probe_required,
                created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "legacy-task", "mission-a", "legacy-source", "cell-a",
                "grid-v1", "policy-v1", "[[0,0,0],[64,64,64]]",
                '{"x":32,"y":32,"z":32}', 1.0, "{}", "0" * 64,
                json.dumps(task_payload, sort_keys=True, separators=(",", ":")),
                "PENDING", 0, 0.0, 1, now, now,
            ),
        )
        connection.execute(
            """INSERT INTO attempts
               (attempt_id,task_id,attempt_number,worker_id,lease_token,state,
                created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            ("legacy-attempt", "legacy-task", 1, "worker", "token",
             "COMPLETED", now, now),
        )
        profile = json.loads(_profile_bytes())
        cap = _cap()
        profile.update({
            "source_snapshot_id": "legacy-source",
            "source_snapshot_sha256": "1" * 64,
            "source_content_lock_sha256": "2" * 64,
            "ct_metadata_sha256": "3" * 64,
            "ct_read_set_manifest_sha256": "4" * 64,
            "m7_read_set_manifest_sha256": "6" * 64,
            "m7_model_id": "legacy-m7", "m7_resolution": 4,
            "m7_level": 1, "m7_threshold": 0.5,
            "m7_transform_sha256": "9" * 64,
            "canonical_ordered_cell_set_sha256": content_sha256(["cell-a"]),
            "mission_compute_cap_authority_id": cap["cap_authority_id"],
            "mission_compute_cap_authority_sha256": cap["authority_sha256"],
            "mission_compute_cap_units": cap["mission_compute_cap_units"],
        })
        profile["scientific_core_sha256"] = content_sha256({
            key: value for key, value in profile.items()
            if key != "scientific_core_sha256"
        })
        profile_bytes = json.dumps(
            profile, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        store._legacy_profile_sha256 = __import__("hashlib").sha256(
            profile_bytes
        ).hexdigest()
        policy = {
            "mode": "shadow", "top_k": 2, "probe_generations": 12,
            "maximum_attempts_per_candidate": 1,
            "arm_kind": "BASELINE",
            "discovery_profile": profile,
            "discovery_profile_file_sha256": store._legacy_profile_sha256,
        }
        connection.execute(
            """INSERT INTO probe_runs
               (probe_run_id,task_id,created_by_attempt_id,source_snapshot_id,
                candidate_set_json,candidate_set_sha256,policy_id,policy_json,
                policy_sha256,executor_fingerprint_json,
                executor_fingerprint_sha256,state,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "legacy-run", "legacy-task", "legacy-attempt",
                "legacy-source", "[]", content_sha256([]), "probe-policy",
                json.dumps(policy, sort_keys=True, separators=(",", ":")),
                content_sha256(policy), "{}", content_sha256({}),
                "PROBING", now, now,
            ),
        )


def test_public_generic_reserve_has_no_historical_caller_attestation():
    parameters = inspect.signature(FleetStore.reserve_discovery_compute).parameters
    assert "historical_execution_manifest" not in parameters


def test_complete_empty_history_is_persisted_from_retained_rows(tmp_path):
    store = _history_store(tmp_path)
    first = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    second = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert first == second
    assert first["state"] == "COMPLETE"
    assert first["fixed_units"] == 0
    assert first["manifest"]["legacy_probe_run_ids"] == []
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_history_reconciliations_v19"
        ).fetchone()[0] == 1


def test_existing_sqlite_v19_import_table_drops_reservation_unique_constraint(
    tmp_path,
):
    store = _history_store(tmp_path)
    with store.connect() as connection:
        connection.execute("DROP TABLE first_letters_discovery_historical_imports_v19")
        connection.execute(
            """CREATE TABLE first_letters_discovery_historical_imports_v19 (
              import_id TEXT PRIMARY KEY,
              reservation_id TEXT NOT NULL UNIQUE,
              mission_id TEXT NOT NULL,
              logical_execution_id TEXT NOT NULL,
              producer_kind TEXT NOT NULL,
              source_snapshot_sha256 TEXT NOT NULL,
              profile_file_sha256 TEXT NOT NULL,
              item_id TEXT NOT NULL,
              fixed_units INTEGER NOT NULL CHECK(fixed_units = 24),
              retained_row_ids_json TEXT NOT NULL,
              retained_projection_sha256 TEXT NOT NULL,
              history_manifest_sha256 TEXT NOT NULL,
              import_json TEXT NOT NULL,
              import_sha256 TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL,
              UNIQUE(mission_id,logical_execution_id)
            )"""
        )

    store.initialize()

    with store.connect() as connection:
        unique_column_sets = []
        for index in connection.execute(
            "PRAGMA index_list(first_letters_discovery_historical_imports_v19)"
        ):
            if index["unique"]:
                unique_column_sets.append([
                    row["name"] for row in connection.execute(
                        f"PRAGMA index_info({index['name']})"
                    )
                ])
        assert ["reservation_id"] not in unique_column_sets


def test_new_incomplete_legacy_row_after_watermark_persists_block(tmp_path):
    store = _history_store(tmp_path)
    complete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    _insert_incomplete_legacy_probe(store)

    incomplete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    assert incomplete["reason"] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
    assert incomplete["manifest_sha256"] != complete["manifest_sha256"]
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


def test_history_reconciliation_accepts_no_receipt_sha_or_count(tmp_path):
    store = _history_store(tmp_path)
    with pytest.raises(TypeError):
        store.reconcile_first_letters_discovery_history(
            mission_id="mission-a", receipt_sha256s=["0" * 64], item_count=1,
        )


def _budget_admission() -> dict:
    value = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": "mission-a",
        "sample_id": "PHercA",
        "receipt_sha256": "6" * 64,
        "preflight_receipt_sha256": "7" * 64,
        "preflight_sanitized_receipt_sha256": "8" * 64,
        "approved_task_count": 1,
        "order_seed_sha256": "9" * 64,
        "population_order_sha256": "0" * 64,
        "prefix_sha256": content_sha256(["cell-a"]),
        "prefix_cell_ids": ["cell-a"],
        "execution_bindings": {
            "source_snapshot_id": "source-a",
            "grid_version": "first-letters-grid-v1",
            "policy_version": "first-letters-search@1.0.0",
            "p0_artifact_id": "p0-a",
            "p0_artifact_sha256": "a" * 64,
            "catalog_snapshot_sha256": "b" * 64,
        },
    }
    value["admission_sha256"] = content_sha256(value)
    return value


def _arm_for_profile(profile_bytes: bytes, cap: dict) -> dict:
    arm = _arm()
    arm["discovery_profile_sha256"] = __import__("hashlib").sha256(
        profile_bytes
    ).hexdigest()
    arm["active_policy_chain_sha256"] = "c" * 64
    arm["mission_compute_cap_authority_id"] = cap["cap_authority_id"]
    arm["mission_compute_cap_authority_sha256"] = cap["authority_sha256"]
    arm["admission_sha256"] = content_sha256({
        key: value for key, value in arm.items()
        if key != "admission_sha256"
    })
    return arm


def _live_bridge_store(tmp_path: Path) -> tuple[FleetStore, dict, bytes]:
    from test_first_letters_discovery_evidence_store import (
        _FixtureDiscoveryExecutor,
        _cap,
        _fixture_executor_registration,
        _profile_bytes,
    )

    profile_bytes = _profile_bytes()
    cap = _cap()
    alternative_profile = json.loads(profile_bytes)
    alternative_profile.update({
        "arm_kind": "ALTERNATIVE_SOURCE_ARM",
        "experimental_arm_admission_id": "arm-a",
        "experimental_arm_admission_sha256": "6" * 64,
        "source_snapshot_id": "source-alt",
        "source_snapshot_sha256": "3" * 64,
        "source_content_lock_sha256": "4" * 64,
        "ct_metadata_sha256": "5" * 64,
        "ct_read_set_manifest_sha256": "6" * 64,
        "m7_read_set_manifest_sha256": "8" * 64,
        "m7_model_id": "m7-alt", "m7_transform_sha256": "9" * 64,
    })
    alternative_profile["scientific_core_sha256"] = content_sha256({
        key: value for key, value in alternative_profile.items()
        if key != "scientific_core_sha256"
    })
    alternative_profile_bytes = json.dumps(
        alternative_profile, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    arm = _arm_for_profile(alternative_profile_bytes, cap)
    executor = _FixtureDiscoveryExecutor()
    store = FleetStore(
        tmp_path / "fleet.sqlite",
        first_letters_discovery_executor=executor,
        first_letters_discovery_worker_id=(
            "registered-fixture-discovery-worker"
        ),
        first_letters_discovery_executor_id="fixture-discovery-executor-v1",
        first_letters_discovery_profile_resolver=(
            lambda mission_id, source_snapshot_id: (
                alternative_profile_bytes
                if source_snapshot_id == "source-alt" else profile_bytes
            )
        ),
        first_letters_experimental_arm_resolver=(
            lambda arm_id: copy.deepcopy(arm) if arm_id == "arm-a" else None
        ),
    )
    store.initialize()
    store.register_first_letters_discovery_executor(
        _fixture_executor_registration()
    )
    store.register_snapshot({
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "d" * 64,
        "ct_metadata_sha256": "2" * 64,
        "ct_read_set_manifest_sha256": "e" * 64,
        "m7_read_set_manifest_sha256": "0" * 64,
        "m7_model_id": "m7-v1",
        "m7_model_sha256": "7" * 64,
        "candidate_provider_id": "fixture-provider-v1",
        "candidate_provider_sha256": "8" * 64,
        "discovery_minimum_separation": 12,
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_threshold": 0.5,
        "m7_transform_sha256": "f" * 64,
        "sample_id": "PHercA",
        "ct_uri": "fixture://ct",
        "ct_sha256": "3" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "4" * 64,
        "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a",
            "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": "a" * 64,
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {"cell-a": "opportunity-cell-a"},
        },
    })
    store.register_snapshot({
        "source_snapshot_id": "source-alt",
        "source_snapshot_sha256": "3" * 64,
        "source_content_lock_sha256": "4" * 64,
        "ct_metadata_sha256": "5" * 64,
        "ct_read_set_manifest_sha256": "6" * 64,
        "m7_metadata_sha256": "7" * 64,
        "m7_read_set_manifest_sha256": "8" * 64,
        "m7_model_id": "m7-alt",
        "m7_model_sha256": "7" * 64,
        "candidate_provider_id": "fixture-provider-v1",
        "candidate_provider_sha256": "8" * 64,
        "discovery_minimum_separation": 12,
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_transform_sha256": "9" * 64,
        "m7_threshold": 0.5,
        "sample_id": "PHercA",
        "ct_uri": "fixture://ct-alt",
        "ct_sha256": "5" * 64,
        "m7_uri": "fixture://m7-alt",
        "m7_sha256": "7" * 64,
        "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a",
            "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": "a" * 64,
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {"cell-a": "opportunity-cell-a"},
        },
    })
    store.register_discovery_compute_cap(cap)
    admission = _budget_admission()
    now = "2026-08-03T00:00:00Z"
    payload = {
        "sample_id": "PHercA",
        "selection_rank": 0,
        "campaign_budget_admission_sha256": admission["admission_sha256"],
        "campaign_budget": {**copy.deepcopy(admission), "selection_rank": 0},
        "p0_artifact_id": "p0-a",
        "p0_artifact_sha256": "a" * 64,
        "scientific_opportunity_id": "opportunity-cell-a",
        "candidate_discovery": {
            "region": {"minimum": [0, 0, 0], "maximum": [64, 64, 64]},
        },
    }
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO campaign_budget_admissions
               (mission_id,sample_id,receipt_sha256,admission_json,
                admission_sha256,created_at) VALUES(?,?,?,?,?,?)""",
            (
                "mission-a", "PHercA", admission["receipt_sha256"],
                json.dumps(admission, sort_keys=True, separators=(",", ":")),
                admission["admission_sha256"], now,
            ),
        )
        connection.execute(
            """INSERT INTO tasks
               (task_id,mission_id,source_snapshot_id,cell_id,grid_version,
                policy_version,bounds_xyz_json,center_xyz_json,priority,
                parameter_envelope_json,catalog_snapshot_sha256,payload_json,
                state,gpu_required,minimum_vram_gb,seed_probe_required,
                created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "task-cell-a", "mission-a", "source-a", "cell-a",
                "first-letters-grid-v1", "first-letters-search@1.0.0",
                "[[0,0,0],[64,64,64]]", '{"x":32,"y":32,"z":32}',
                1.0, "{}", "b" * 64,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                "PENDING", 0, 0.0, 0, now, now,
            ),
        )
    return store, admission, profile_bytes


def test_reservation_adapter_dispatch_and_one_job_per_item_commit_atomically(
    tmp_path,
):
    store, admission, _ = _live_bridge_store(tmp_path)
    result = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )

    assert result["reservation"]["work_kind"] == "BASELINE_ARM"
    assert result["adapter"]["producer_kind"] == "BASELINE_RECONCILIATION"
    assert result["dispatch"]["reservation_id"] == result["reservation"][
        "reservation_id"]
    assert [job["item_id"] for job in result["jobs"]] == ["cell-a"]
    with store.connect() as connection:
        counts = [connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0] for table in (
            "first_letters_discovery_compute_reservations",
            "first_letters_discovery_native_adapters_v19",
            "first_letters_discovery_dispatches_v19",
            "first_letters_discovery_jobs_v19",
        )]
    assert counts == [1, 1, 1, 1]


def test_real_experimental_arm_reserves_alternative_without_canonical_rows(
    tmp_path,
):
    store, admission, _ = _live_bridge_store(tmp_path)
    with store.connect() as connection:
        before = [tuple(row) for row in connection.execute(
            "SELECT * FROM tasks ORDER BY task_id"
        )]
    result = store.reserve_first_letters_alternative_shadow(
        request_id="request-alt",
        budget_admission_sha256=admission["admission_sha256"],
        arm_id="arm-a",
    )
    assert result["adapter"]["producer_kind"] == (
        "EXPERIMENTAL_ARM_ADMISSION"
    )
    assert result["reservation"]["work_kind"] == "ALTERNATIVE_SOURCE_ARM"
    with store.connect() as connection:
        after = [tuple(row) for row in connection.execute(
            "SELECT * FROM tasks ORDER BY task_id"
        )]
    assert after == before


def test_synthetic_generic_baseline_authority_cannot_reserve_or_dispatch(
    tmp_path,
):
    from test_first_letters_discovery_compute import _work

    store, _, _ = _live_bridge_store(tmp_path)
    work = _work("BASELINE_ARM", "forged", ("cell-a",))
    with pytest.raises(ValueError, match="DISCOVERY_NATIVE_PRODUCER_REQUIRED"):
        store.reserve_discovery_compute(
            mission_id="mission-a", request_id="forged",
            work_kind="BASELINE_ARM", work_authority=work,
            work_authority_id=work["work_authority_id"],
            work_authority_sha256=work["work_authority_sha256"],
            ordered_item_ids=["cell-a"], cap_authority_id="cap-a",
            cap_authority_sha256=work["cap_authority_sha256"],
        )


def test_missing_job_makes_exact_request_readback_control_incomplete(tmp_path):
    store, admission, _ = _live_bridge_store(tmp_path)
    result = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    with store.connect() as connection:
        connection.execute(
            "DELETE FROM first_letters_discovery_jobs_v19 WHERE job_id=?",
            (result["jobs"][0]["job_id"],),
        )
    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_DISCOVERY_DISPATCH"):
        store.read_first_letters_discovery_request("mission-a", "request-a")


@pytest.mark.parametrize("failpoint", [
    "bridge.before_reservation",
    "bridge.after_reservation_before_adapter",
    "bridge.after_adapter_before_dispatch",
    "bridge.after_dispatch_before_jobs",
    "bridge.after_each_job",
    "bridge.after_jobs_before_commit",
    "bridge.before_commit",
])
def test_each_bridge_precommit_failpoint_leaves_zero_live_branch_rows(
    tmp_path, failpoint,
):
    store, admission, _ = _live_bridge_store(tmp_path)
    with pytest.raises(RuntimeError, match=failpoint):
        store.reserve_first_letters_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=admission["admission_sha256"],
            failpoint=failpoint,
        )
    with store.connect() as connection:
        counts = [connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0] for table in (
            "first_letters_discovery_compute_reservations",
            "first_letters_discovery_work_bindings",
            "first_letters_discovery_native_adapters_v19",
            "first_letters_discovery_dispatches_v19",
            "first_letters_discovery_jobs_v19",
        )]
    assert counts == [0, 0, 0, 0, 0]


def test_bridge_commit_unknown_and_response_loss_require_exact_readback(
    tmp_path,
):
    first, admission, _ = _live_bridge_store(tmp_path / "unknown")
    with pytest.raises(
        RuntimeError, match="CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK"
    ):
        first.reserve_first_letters_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=admission["admission_sha256"],
            failpoint="bridge.commit_outcome_unknown",
        )
    recovered = first.read_first_letters_discovery_request(
        "mission-a", "request-a"
    )
    assert first.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    ) == recovered

    second, admission, _ = _live_bridge_store(tmp_path / "loss")
    response = second.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
        failpoint="bridge.after_commit_before_response",
    )
    assert response == second.read_first_letters_discovery_request(
        "mission-a", "request-a"
    )


class _PreparedBridgeProvider:
    def __init__(self, owner):
        self.owner = owner

    def execute(self, provider_request):
        self.owner.execute_calls += 1
        from test_first_letters_discovery_controller import _provider_response
        return _provider_response(provider_request)


class _BridgeProvider:
    def __init__(self):
        self.prepare_calls = 0
        self.execute_calls = 0

    def prepare(self):
        self.prepare_calls += 1
        return _PreparedBridgeProvider(self)


def test_controller_claims_by_job_id_and_never_accepts_item_or_profile_bytes(
    tmp_path,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, profile_bytes = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    provider = _BridgeProvider()
    controller = FirstLettersDiscoveryController(
        mode="shadow", store=store, provider=provider,
    )
    signature = inspect.signature(controller.run_job).parameters
    assert set(signature) == {"job_id", "lease_seconds"}
    assert "profile_bytes" not in signature and "item_id" not in signature

    result = controller.run_job(
        job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
    )

    assert result["state"] == "COMPLETED"
    assert provider.prepare_calls == provider.execute_calls == 1
    assert profile_bytes not in repr(result).encode()


def test_alternative_source_job_uses_its_server_owned_profile_before_provider(
    tmp_path,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_alternative_shadow(
        request_id="request-alt",
        budget_admission_sha256=admission["admission_sha256"], arm_id="arm-a",
    )
    provider = _BridgeProvider()
    result = FirstLettersDiscoveryController(
        mode="shadow", store=store, provider=provider,
    ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)

    assert result["state"] == "COMPLETED"
    assert provider.prepare_calls == provider.execute_calls == 1


def test_worker_revalidates_claim_graph_before_provider_prepare(tmp_path):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    provider = _BridgeProvider()

    class CorruptBeforeRevalidation:
        def __getattr__(self, name):
            if name != "revalidate_first_letters_discovery_job_claim":
                return getattr(store, name)

            def corrupt(*args, **kwargs):
                with store.connect() as connection:
                    connection.execute(
                        "UPDATE first_letters_discovery_jobs_v19 "
                        "SET item_id='cell-forged' WHERE job_id=?",
                        (branch["jobs"][0]["job_id"],),
                    )
                return getattr(store, name)(*args, **kwargs)

            return corrupt

    controller = FirstLettersDiscoveryController(
        mode="shadow", store=CorruptBeforeRevalidation(), provider=provider,
    )
    with pytest.raises(ValueError, match="claim|dispatch|graph"):
        controller.run_job(
            job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
        )
    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        states = [row[0] for row in connection.execute(
            "SELECT state FROM first_letters_discovery_evidence_runs"
        )]
        assert states == ["CLAIMED"]
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0] == 0


def test_controlled_v19_reservation_rejects_legacy_item_profile_claim(tmp_path):
    store, admission, profile_bytes = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    with pytest.raises(ValueError, match="DISCOVERY_JOB_ID_REQUIRED"):
        store.begin_first_letters_discovery_evidence_run(
            lease_seconds=60,
            reservation_id=branch["reservation"]["reservation_id"],
            item_id="cell-a", profile_bytes=profile_bytes,
        )


def test_off_run_job_returns_without_store_or_provider_access():
    from fleet.discovery_controller import FirstLettersDiscoveryController

    controller = FirstLettersDiscoveryController(
        mode="off", store=None, provider=None,
    )
    assert controller.run_job(job_id="never-read", lease_seconds=60)[
        "state"
    ] == "OFF_UNCHANGED"


def _retained_v16_execution_store(
    tmp_path: Path, *, alternative: bool = False,
):
    """Leave the exact v16 reservation/evidence graph after removing v19 control."""

    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _profile_bytes_value = _live_bridge_store(tmp_path)
    if alternative:
        branch = store.reserve_first_letters_alternative_shadow(
            request_id="request-alt",
            budget_admission_sha256=admission["admission_sha256"],
            arm_id="arm-a",
        )
    else:
        branch = store.reserve_first_letters_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=admission["admission_sha256"],
        )
    provider = _BridgeProvider()
    # Ten minutes, not one: the claim revalidates its own lease right after
    # writing it, and on a loaded CI runner one sqlite commit under saturated
    # I/O took the fixture past sixty seconds -- "must hold a live CLAIMED
    # lease" for a lease made moments earlier. No test reads this value; the
    # ones about expiry set the date to 2000 or the clock to 2999.
    result = FirstLettersDiscoveryController(
        mode="shadow", store=store, provider=provider,
    ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=600)
    assert result["state"] == "COMPLETED"
    reservation_id = branch["reservation"]["reservation_id"]
    with store.connect() as connection:
        connection.execute(
            "DELETE FROM first_letters_discovery_jobs_v19 WHERE reservation_id=?",
            (reservation_id,),
        )
        connection.execute(
            "DELETE FROM first_letters_discovery_dispatches_v19 "
            "WHERE reservation_id=?",
            (reservation_id,),
        )
        connection.execute(
            "DELETE FROM first_letters_discovery_native_adapters_v19 "
            "WHERE reservation_id=?",
            (reservation_id,),
        )
        connection.execute(
            "DELETE FROM first_letters_discovery_history_reconciliations_v19 "
            "WHERE mission_id='mission-a'"
        )
    return store, branch


def test_complete_retained_v16_execution_is_explicitly_imported_and_charged(
    tmp_path,
):
    store, branch = _retained_v16_execution_store(tmp_path)

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "COMPLETE"
    assert reconciliation["fixed_units"] == 24
    with store.connect() as connection:
        imported = connection.execute(
            "SELECT * FROM first_letters_discovery_historical_imports_v19"
        ).fetchall()
        assert len(imported) == 1
        assert imported[0]["reservation_id"] == branch["reservation"][
            "reservation_id"]
        assert imported[0]["item_id"] == "cell-a"
        assert imported[0]["fixed_units"] == 24
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_compute_reservations "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == 1
        first_reservations = [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_compute_reservations "
            "ORDER BY reservation_id"
        )]
        first_imports = [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_historical_imports_v19 "
            "ORDER BY import_id"
        )]
    assert store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    ) == reconciliation
    with store.connect() as connection:
        assert [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_compute_reservations "
            "ORDER BY reservation_id"
        )] == first_reservations
        assert [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_historical_imports_v19 "
            "ORDER BY import_id"
        )] == first_imports
    cap = store.discovery_compute_cap("mission-a")
    assert cap["mission_compute_cap_units"] - store.discovery_compute_total(
        "mission-a"
    ) == 0


@pytest.mark.parametrize("damage", ["delete", "mutate"])
def test_retained_v16_import_marker_damage_fails_closed(tmp_path, damage):
    store, _branch = _retained_v16_execution_store(tmp_path)
    complete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    assert complete["state"] == "COMPLETE"
    with store.connect() as connection:
        if damage == "delete":
            connection.execute(
                "DELETE FROM first_letters_discovery_historical_imports_v19"
            )
        else:
            connection.execute(
                "UPDATE first_letters_discovery_historical_imports_v19 "
                "SET profile_file_sha256=?", ("b" * 64,),
            )

    incomplete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


def test_retained_history_watermark_closes_every_v16_membership_table(tmp_path):
    store, branch = _retained_v16_execution_store(tmp_path)

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    graph = reconciliation["manifest"]["retained_execution_graphs"][0]
    expected = {
        "reservation_ids": [branch["reservation"]["reservation_id"]],
        "run_ids": [graph["run"]["run_id"]],
        "claim_ids": graph["retained_row_ids"]["claim_ids"],
        "evidence_set_ids": graph["retained_row_ids"]["evidence_set_ids"],
        "evidence_files": [{
            "evidence_set_id": row["evidence_set_id"],
            "file_order": row["file_order"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
        } for row in graph["evidence_files"]],
    }
    membership = reconciliation["watermark"]["retained_membership"]
    for key, value in expected.items():
        assert membership[key] == value
    assert set(membership) >= {
        "reservation_ids", "work_binding_ids", "run_ids", "claim_ids",
        "evidence_set_ids", "evidence_files", "source_snapshot_ids",
        "parent_task_ids", "parent_attempt_ids", "profile_file_sha256s",
        "producer_authorities", "retained_projection_sha256s",
    }
    assert reconciliation["watermark_sha256"] == content_sha256(
        reconciliation["watermark"]
    )


def test_retained_history_watermark_changes_when_retained_bytes_change(tmp_path):
    store, _branch = _retained_v16_execution_store(tmp_path)
    complete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE first_letters_discovery_evidence_files "
            "SET payload=? WHERE file_order=0", (b"changed",)
        )

    incomplete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    assert incomplete["watermark_sha256"] != complete["watermark_sha256"]


@pytest.mark.parametrize(("field", "replacement"), [
    ("source_snapshot_sha256", "b" * 64),
    ("source_content_lock_sha256", "b" * 64),
    ("ct_metadata_sha256", "b" * 64),
    ("ct_read_set_manifest_sha256", "b" * 64),
    ("m7_sha256", "b" * 64),
    ("m7_read_set_manifest_sha256", "b" * 64),
    ("m7_model_id", "m7-other"),
    ("m7_model_sha256", "b" * 64),
    ("candidate_provider_id", "provider-other"),
    ("candidate_provider_sha256", "b" * 64),
    ("discovery_minimum_separation", 13),
    ("m7_resolution", 5),
    ("m7_level", 2),
    ("m7_transform_sha256", "b" * 64),
    ("m7_threshold", 0.6),
])
def test_retained_v16_rederives_every_current_scientific_dependency(
    tmp_path, field, replacement,
):
    store, _branch = _retained_v16_execution_store(tmp_path)
    before_total = store.discovery_compute_total("mission-a")
    with store.connect() as connection:
        source = json.loads(connection.execute(
            "SELECT payload_json FROM source_snapshots "
            "WHERE source_snapshot_id='source-a'"
        ).fetchone()[0])
        source[field] = replacement
        connection.execute(
            "UPDATE source_snapshots SET payload_json=? "
            "WHERE source_snapshot_id='source-a'",
            (json.dumps(source, sort_keys=True, separators=(",", ":")),),
        )

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    assert store.discovery_compute_total("mission-a") == before_total
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM "
            "first_letters_discovery_historical_imports_v19"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("damage", [
    "missing_opportunity", "changed_opportunity",
    "missing_region", "changed_region",
])
def test_retained_v16_requires_exact_current_opportunity_and_region(
    tmp_path, damage,
):
    store, _branch = _retained_v16_execution_store(tmp_path)
    before_total = store.discovery_compute_total("mission-a")
    with store.connect() as connection:
        row = connection.execute(
            "SELECT payload_json FROM tasks WHERE task_id='task-cell-a'"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        if damage == "missing_opportunity":
            payload.pop("scientific_opportunity_id")
        elif damage == "changed_opportunity":
            payload["scientific_opportunity_id"] = "opportunity-other"
        elif damage == "missing_region":
            payload["candidate_discovery"].pop("region")
        else:
            payload["candidate_discovery"]["region"] = {
                "minimum": [1, 1, 1], "maximum": [63, 63, 63],
            }
        connection.execute(
            "UPDATE tasks SET payload_json=? WHERE task_id='task-cell-a'",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    assert store.discovery_compute_total("mission-a") == before_total


@pytest.mark.parametrize("resolver_result", ["malformed", "exception"])
def test_retained_v16_profile_resolver_failure_is_durably_fail_closed(
    tmp_path, resolver_result,
):
    store, _branch = _retained_v16_execution_store(tmp_path)
    if resolver_result == "malformed":
        store._first_letters_discovery_profile_resolver = (
            lambda _mission_id, _source_id: None
        )
    else:
        def resolver(_mission_id, _source_id):
            raise RuntimeError("profile resolver unavailable")
        store._first_letters_discovery_profile_resolver = resolver

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()["reason"] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


@pytest.mark.parametrize("damage", [
    "missing_reservation", "extra_reservation", "missing_work", "extra_work",
    "missing_run", "extra_run",
    "changed_run_scalar", "changed_run_json", "changed_provider_request_json",
    "missing_claim", "extra_claim", "changed_claim_scalar",
    "changed_claim_json", "missing_evidence_set", "extra_evidence_set",
    "changed_evidence_scalar", "changed_evidence_json",
    "missing_file", "extra_file", "changed_file_bytes",
    "changed_file_role", "changed_file_path", "changed_file_order",
    "changed_file_byte_count", "changed_file_hash",
    "changed_reservation_scalar", "changed_reservation_json",
    "changed_work_scalar", "changed_work_json",
    "changed_profile_bytes", "changed_profile_hash",
    "changed_source_snapshot", "missing_source_snapshot", "missing_parent_task",
    "changed_item",
])
def test_retained_v16_descendant_drift_persists_block_without_new_reservation(
    tmp_path, damage,
):
    store, _branch = _retained_v16_execution_store(tmp_path)
    with store.connect() as connection:
        evidence_set_id = connection.execute(
            "SELECT evidence_set_id FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0]
        run = connection.execute(
            "SELECT * FROM first_letters_discovery_evidence_runs"
        ).fetchone()
        claim = connection.execute(
            "SELECT * FROM first_letters_discovery_executor_claims"
        ).fetchone()
        evidence = connection.execute(
            "SELECT * FROM first_letters_discovery_evidence_sets"
        ).fetchone()

        def insert_extra_run():
            columns = [key for key in run.keys()]
            values = [run[key] for key in columns]
            replacements = {
                "run_id": "unexpected-run",
                "cell_id": "unexpected-cell",
                "run_token_sha256": "1" * 64,
                "run_authority_sha256": "2" * 64,
            }
            values = [replacements.get(key, value)
                      for key, value in zip(columns, values, strict=True)]
            connection.execute(
                f"INSERT INTO first_letters_discovery_evidence_runs "
                f"({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                values,
            )

        if damage == "missing_reservation":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                "DELETE FROM first_letters_discovery_compute_reservations"
            )
        elif damage == "extra_reservation":
            row = connection.execute(
                "SELECT * FROM first_letters_discovery_compute_reservations"
            ).fetchone()
            columns = [key for key in row.keys()]
            replacements = {
                "reservation_id": "unexpected-reservation",
                "request_id": "unexpected-request",
                "work_authority_id": "unexpected-work",
                "work_authority_sha256": "c" * 64,
                "request_sha256": "d" * 64,
                "reservation_sha256": "e" * 64,
            }
            connection.execute(
                f"INSERT INTO first_letters_discovery_compute_reservations "
                f"({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                [replacements.get(key, row[key]) for key in columns],
            )
        elif damage == "missing_work":
            connection.execute(
                "DELETE FROM first_letters_discovery_work_bindings"
            )
        elif damage == "extra_work":
            connection.execute("PRAGMA foreign_keys=OFF")
            row = connection.execute(
                "SELECT * FROM first_letters_discovery_work_bindings"
            ).fetchone()
            columns = [key for key in row.keys()]
            replacements = {
                "reservation_id": "unexpected-reservation",
                "request_id": "unexpected-request",
                "work_sha256": "d" * 64,
            }
            connection.execute(
                f"INSERT INTO first_letters_discovery_work_bindings "
                f"({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                [replacements.get(key, row[key]) for key in columns],
            )
        elif damage == "missing_run":
            connection.execute(
                "DELETE FROM first_letters_discovery_evidence_files"
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_evidence_sets"
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_executor_claims"
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_evidence_runs"
            )
        elif damage == "extra_run":
            insert_extra_run()
        elif damage == "changed_run_scalar":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_runs "
                "SET request_id='request-forged'"
            )
        elif damage == "changed_run_json":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_runs "
                "SET run_authority_json='{}'"
            )
        elif damage == "changed_provider_request_json":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_runs "
                "SET provider_request_json='{}'"
            )
        elif damage == "missing_claim":
            connection.execute(
                "DELETE FROM first_letters_discovery_executor_claims"
            )
        elif damage == "extra_claim":
            insert_extra_run()
            columns = [key for key in claim.keys()]
            replacements = {
                "claim_id": "unexpected-claim", "run_id": "unexpected-run",
                "execution_lease_token_sha256": "3" * 64,
                "claim_sha256": "4" * 64,
            }
            values = [replacements.get(key, claim[key]) for key in columns]
            connection.execute(
                f"INSERT INTO first_letters_discovery_executor_claims "
                f"({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                values,
            )
        elif damage == "changed_claim_scalar":
            connection.execute(
                "UPDATE first_letters_discovery_executor_claims "
                "SET capability='UNEXPECTED'"
            )
        elif damage == "changed_claim_json":
            connection.execute(
                "UPDATE first_letters_discovery_executor_claims SET claim_json='{}'"
            )
        elif damage == "missing_evidence_set":
            connection.execute(
                "DELETE FROM first_letters_discovery_evidence_files"
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_evidence_sets"
            )
        elif damage == "extra_evidence_set":
            insert_extra_run()
            columns = [key for key in evidence.keys()]
            replacements = {
                "evidence_set_id": "unexpected-evidence",
                "run_id": "unexpected-run", "evidence_set_sha256": "5" * 64,
            }
            values = [replacements.get(key, evidence[key]) for key in columns]
            connection.execute(
                f"INSERT INTO first_letters_discovery_evidence_sets "
                f"({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                values,
            )
        elif damage == "changed_evidence_scalar":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_sets "
                "SET evidence_set_sha256=?", ("f" * 64,)
            )
        elif damage == "changed_evidence_json":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_sets "
                "SET evidence_json='{}'"
            )
        elif damage == "missing_file":
            connection.execute(
                "DELETE FROM first_letters_discovery_evidence_files "
                "WHERE evidence_set_id=? AND file_order=0",
                (evidence_set_id,),
            )
        elif damage == "extra_file":
            connection.execute(
                """INSERT INTO first_letters_discovery_evidence_files
                   (evidence_set_id,file_order,relative_path,role,payload,
                    byte_count,sha256) VALUES(?,?,?,?,?,?,?)""",
                (
                    evidence_set_id, 99, "unexpected.bin", "UNEXPECTED",
                    b"unexpected", 10,
                    __import__("hashlib").sha256(b"unexpected").hexdigest(),
                ),
            )
        elif damage == "changed_file_bytes":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_files "
                "SET payload=? WHERE evidence_set_id=? AND file_order=0",
                (b"changed", evidence_set_id),
            )
        elif damage == "changed_file_role":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_files "
                "SET role='UNEXPECTED' WHERE file_order=0"
            )
        elif damage == "changed_file_path":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_files "
                "SET relative_path='unexpected.bin' WHERE file_order=0"
            )
        elif damage == "changed_file_order":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_files "
                "SET file_order=99 WHERE file_order=0"
            )
        elif damage == "changed_file_byte_count":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_files "
                "SET byte_count=byte_count+1 WHERE file_order=0"
            )
        elif damage == "changed_file_hash":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_files "
                "SET sha256=? WHERE file_order=0", ("f" * 64,)
            )
        elif damage == "changed_reservation_scalar":
            connection.execute(
                "UPDATE first_letters_discovery_compute_reservations "
                "SET request_id='request-forged'"
            )
        elif damage == "changed_reservation_json":
            connection.execute(
                "UPDATE first_letters_discovery_compute_reservations "
                "SET reservation_json='{}'"
            )
        elif damage == "changed_work_scalar":
            connection.execute(
                "UPDATE first_letters_discovery_work_bindings "
                "SET request_id='request-forged'"
            )
        elif damage == "changed_work_json":
            connection.execute(
                "UPDATE first_letters_discovery_work_bindings SET work_json='{}'"
            )
        elif damage == "changed_profile_bytes":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_runs "
                "SET profile_bytes=?", (b"{}",)
            )
        elif damage == "changed_profile_hash":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_runs "
                "SET profile_file_sha256=?", ("f" * 64,)
            )
        elif damage == "changed_source_snapshot":
            source = json.loads(connection.execute(
                "SELECT payload_json FROM source_snapshots "
                "WHERE source_snapshot_id='source-a'"
            ).fetchone()[0])
            source["source_snapshot_sha256"] = "f" * 64
            connection.execute(
                "UPDATE source_snapshots SET payload_json=? "
                "WHERE source_snapshot_id='source-a'",
                (json.dumps(source, sort_keys=True, separators=(",", ":")),),
            )
        elif damage == "missing_source_snapshot":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                "DELETE FROM source_snapshots WHERE source_snapshot_id='source-a'"
            )
        elif damage == "missing_parent_task":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DELETE FROM tasks WHERE task_id='task-cell-a'")
        elif damage == "changed_item":
            connection.execute(
                "UPDATE first_letters_discovery_evidence_runs "
                "SET cell_id='unexpected-cell'"
            )
        retained_reservation_ids = [row[0] for row in connection.execute(
            "SELECT reservation_id FROM "
            "first_letters_discovery_compute_reservations ORDER BY reservation_id"
        )]

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
        assert [row[0] for row in connection.execute(
            "SELECT reservation_id FROM "
            "first_letters_discovery_compute_reservations ORDER BY reservation_id"
        )] == retained_reservation_ids
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_historical_imports_v19"
        ).fetchone()[0] == 0


def test_job_claim_reenumerates_current_history_before_provider_prepare(tmp_path):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    _insert_incomplete_legacy_probe(store)
    provider = _BridgeProvider()
    with store.connect() as connection:
        before = {
            relation: connection.execute(
                f"SELECT COUNT(*) FROM {relation}"
            ).fetchone()[0]
            for relation in (
                "first_letters_discovery_evidence_runs",
                "first_letters_discovery_executor_claims",
                "first_letters_discovery_evidence_sets",
                "first_letters_discovery_evidence_files",
            )
        }

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"):
        FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=provider,
        ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        assert {
            relation: connection.execute(
                f"SELECT COUNT(*) FROM {relation}"
            ).fetchone()[0]
            for relation in before
        } == before == {
            "first_letters_discovery_evidence_runs": 0,
            "first_letters_discovery_executor_claims": 0,
            "first_letters_discovery_evidence_sets": 0,
            "first_letters_discovery_evidence_files": 0,
        }
        states = [row[0] for row in connection.execute(
            "SELECT state FROM "
            "first_letters_discovery_history_reconciliations_v19 "
            # rowid, not the digest: utc_now() truncates to whole seconds, so
            # two reconciliations in one second tie and reconciliation_id breaks
            # the tie by content hash, which reverses insertion order at random.
            # This test asks which state was written last, so it has to order by
            # something monotonic in writing. The stores were fixed the same way.
            "WHERE mission_id='mission-a' ORDER BY created_at,rowid"
        )]
        assert states[-1] == "CONTROL_INCOMPLETE"
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


def test_job_claim_honors_new_compute_block_before_provider_prepare(tmp_path):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    store._block_discovery_compute_ledger(
        "mission-a", {"reason": "retained graph changed"}
    )
    provider = _BridgeProvider()

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"):
        FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=provider,
        ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_executor_claims"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_files"
        ).fetchone()[0] == 0


def test_job_claim_serializes_concurrent_legacy_history_insert(tmp_path):
    from fleet.discovery_controller import FirstLettersDiscoveryController

    store, admission, _ = _live_bridge_store(tmp_path)
    branch = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    _insert_incomplete_legacy_probe(store)
    with store.connect() as connection:
        legacy_run = dict(connection.execute(
            "SELECT * FROM probe_runs WHERE probe_run_id='legacy-run'"
        ).fetchone())
        connection.execute("DELETE FROM probe_runs WHERE probe_run_id='legacy-run'")
        connection.execute(
            "UPDATE tasks SET seed_probe_required=0 "
            "WHERE task_id='legacy-task'"
        )
    provider = _BridgeProvider()
    writer = store.connect()
    writer.execute("BEGIN IMMEDIATE")
    writer.execute(
        "UPDATE tasks SET seed_probe_required=1 WHERE task_id='legacy-task'"
    )
    columns = list(legacy_run)
    writer.execute(
        f"INSERT INTO probe_runs ({','.join(columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        tuple(legacy_run[column] for column in columns),
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                FirstLettersDiscoveryController(
                    mode="shadow", store=store, provider=provider,
                ).run_job,
                job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
            )
            time.sleep(0.05)
            assert not future.done()
            writer.commit()
            with pytest.raises(
                ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            ):
                future.result(timeout=5)
    finally:
        if writer.in_transaction:
            writer.rollback()
        writer.close()

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_executor_claims"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
    with store.connect() as connection:
        assert all(row[0] == "CLAIMED" for row in connection.execute(
            "SELECT state FROM first_letters_discovery_evidence_runs"
        ))
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0] == 0


def _complete_legacy_probe_graph(store: FleetStore) -> None:
    now = "2026-08-03T00:00:00Z"
    candidates = [{
        "candidate_id": f"candidate-{rank}",
        "ct_l0_coordinate": [rank + 1] * 3,
    } for rank in (0, 1)]
    evaluation_hashes = []
    with store.connect() as connection:
        connection.execute(
            "UPDATE probe_runs SET state='DECIDED',candidate_set_json=?,"
            "candidate_set_sha256=? WHERE probe_run_id='legacy-run'",
            (
                json.dumps(candidates, sort_keys=True, separators=(",", ":")),
                content_sha256(candidates),
            ),
        )
        for rank in (0, 1):
            trial_id = f"legacy-trial-{rank}"
            attempt_id = f"legacy-probe-attempt-{rank}"
            artifact_id = f"legacy-artifact-{rank}"
            candidate = candidates[rank]
            locked_plan = {
                "schema": "campaignx.seed_probe_locked_plan.v1",
                "probe_run_id": "legacy-run", "probe_trial_id": trial_id,
                "candidate_id": candidate["candidate_id"],
                "profile_file_sha256": store._legacy_profile_sha256,
                "allow_unvalidated": False,
            }
            trial_result = {
                "probe_trial_id": trial_id, "candidate_id": candidate[
                    "candidate_id"], "state": "COMPLETED",
            }
            connection.execute(
                """INSERT INTO probe_trials
                   (probe_trial_id,probe_run_id,candidate_id,candidate_rank,
                    candidate_json,locked_plan_json,locked_plan_sha256,state,
                    result_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,'COMPLETED',?,?,?)""",
                (
                    trial_id, "legacy-run", f"candidate-{rank}", rank,
                    json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                    json.dumps(locked_plan, sort_keys=True, separators=(",", ":")),
                    content_sha256(locked_plan),
                    json.dumps(trial_result, sort_keys=True, separators=(",", ":")),
                    now, now,
                ),
            )
            growth = {
                "probe_run_id": "legacy-run", "probe_trial_id": trial_id,
                "probe_attempt_id": attempt_id,
                "locked_plan_sha256": content_sha256(locked_plan),
            }
            attempt_result = {
                "probe_attempt_id": attempt_id, "outcome": "COMPLETED",
            }
            connection.execute(
                """INSERT INTO probe_attempts
                   (probe_attempt_id,probe_trial_id,attempt_number,worker_id,
                    state,growth_receipt_json,result_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    attempt_id, trial_id, 1, "legacy-worker", "COMPLETED",
                    json.dumps(growth, sort_keys=True, separators=(",", ":")),
                    json.dumps(
                        attempt_result, sort_keys=True, separators=(",", ":")
                    ), now, now,
                ),
            )
            files = {
                "meta.json": {
                    "sha256": content_sha256({"trial": trial_id}),
                    "size_bytes": len(trial_id),
                }
            }
            manifest = {
                "schema": "campaignx.seed_probe_artifact_set.v1",
                "probe_run_id": "legacy-run", "probe_trial_id": trial_id,
                "locked_plan_sha256": content_sha256(locked_plan),
                "files": files, "artifact_sha256": content_sha256(files),
                "noncanonical": True, "ink_used": False,
            }
            connection.execute(
                """INSERT INTO probe_artifact_sets
                   (probe_artifact_set_id,probe_trial_id,probe_attempt_id,
                    manifest_json,manifest_sha256,artifact_uri,state,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    artifact_id, trial_id, attempt_id,
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                    content_sha256(manifest), f"fixture://{artifact_id}",
                    "RETAINED", now,
                ),
            )
            evaluation = {
                "evaluation_id": f"legacy-evaluation-{rank}",
                "probe_trial_id": trial_id,
                "probe_artifact_set_id": artifact_id,
                "artifact_sha256": manifest["artifact_sha256"],
                "verdict": "ELIGIBLE", "ink_used": False,
                "profile_sha256": store._legacy_profile_sha256,
            }
            evaluation_sha = content_sha256(evaluation)
            evaluation_hashes.append(evaluation_sha)
            connection.execute(
                """INSERT INTO probe_evaluations
                   (evaluation_id,probe_trial_id,probe_artifact_set_id,
                    profile_id,profile_sha256,verdict,result_json,
                    result_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    f"legacy-evaluation-{rank}", trial_id, artifact_id,
                    "legacy-profile", store._legacy_profile_sha256, "ELIGIBLE",
                    json.dumps(evaluation, sort_keys=True, separators=(",", ":")),
                    evaluation_sha, now,
                ),
            )
        run = connection.execute(
            "SELECT policy_id,policy_sha256 FROM probe_runs "
            "WHERE probe_run_id='legacy-run'"
        ).fetchone()
        evidence_set_sha = content_sha256({
            "probe_run_id": "legacy-run",
            "ordered_evaluation_sha256s": evaluation_hashes,
        })
        receipt = {
            "probe_run_id": "legacy-run", "action": "ABSTAIN",
            "winner_trial_id": None, "policy_id": run["policy_id"],
            "policy_sha256": run["policy_sha256"],
            "evidence_set_sha256": evidence_set_sha,
        }
        connection.execute(
            """INSERT INTO probe_decisions
               (decision_id,probe_run_id,policy_id,policy_sha256,
                evidence_set_sha256,action,winner_trial_id,receipt_json,
                receipt_sha256,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                "legacy-decision", "legacy-run", run["policy_id"],
                run["policy_sha256"], evidence_set_sha, "ABSTAIN", None,
                json.dumps(receipt, sort_keys=True, separators=(",", ":")),
                content_sha256(receipt), now,
            ),
        )


def test_complete_legacy_probe_creates_historical_reservation_import_and_cap_debit(
    tmp_path,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    assert store.discovery_compute_total("mission-a") == 0

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "COMPLETE"
    assert reconciliation["fixed_units"] == 24
    assert store.discovery_compute_total("mission-a") == 24
    with store.connect() as connection:
        reservation = connection.execute(
            "SELECT reservation_id,request_id,source FROM "
            "first_letters_discovery_compute_reservations"
        ).fetchone()
        assert reservation["source"] == "IMPORTED_HISTORICAL_EXACT"
        imported = json.loads(connection.execute(
            "SELECT import_json FROM "
            "first_letters_discovery_historical_imports_v19"
        ).fetchone()[0])
        assert imported["reservation_id"] == reservation["reservation_id"]
        assert imported["logical_execution_id"] == "legacy-run"
        assert imported["producer_kind"] == "LEGACY_PROBE_RUN"
        assert imported["source_snapshot_sha256"] == "1" * 64
        assert imported["profile_file_sha256"] == store._legacy_profile_sha256
        assert imported["item_id"] == "cell-a"
        assert imported["retained_row_ids"] == {
            "task_id": "legacy-task",
            "attempt_id": "legacy-attempt",
            "probe_run_id": "legacy-run",
            "probe_trial_ids": ["legacy-trial-0", "legacy-trial-1"],
            "probe_attempt_ids": [
                "legacy-probe-attempt-0", "legacy-probe-attempt-1",
            ],
            "probe_artifact_set_ids": [
                "legacy-artifact-0", "legacy-artifact-1",
            ],
            "evaluation_ids": [
                "legacy-evaluation-0", "legacy-evaluation-1",
            ],
            "decision_ids": ["legacy-decision"],
        }
    readback = store.read_discovery_compute_request(
        "mission-a", reservation["request_id"]
    )
    assert readback["reservation"]["source"] == "IMPORTED_HISTORICAL_EXACT"
    assert readback["work"]["dispatch_kind"] == "HISTORICAL_IMPORT_BINDING"
    with store.connect() as connection:
        legacy_policy = json.loads(connection.execute(
            "SELECT policy_json FROM probe_runs "
            "WHERE probe_run_id='legacy-run'"
        ).fetchone()[0])
    profile_bytes = json.dumps(
        legacy_policy["discovery_profile"], sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with pytest.raises(ValueError, match="DISCOVERY_JOB_ID_REQUIRED"):
        store.begin_first_letters_discovery_evidence_run(
            lease_seconds=60,
            reservation_id=reservation["reservation_id"],
            item_id="cell-a", profile_bytes=profile_bytes,
        )
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("table", [
    "task", "attempt", "run", "trial", "probe_attempt", "artifact",
    "evaluation", "decision",
])
def test_legacy_probe_missing_row_before_first_reconcile_fails_closed(
    tmp_path, table,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    targets = {
        "task": ("tasks", "task_id", "legacy-task"),
        "attempt": ("attempts", "attempt_id", "legacy-attempt"),
        "run": ("probe_runs", "probe_run_id", "legacy-run"),
        "trial": ("probe_trials", "probe_trial_id", "legacy-trial-0"),
        "probe_attempt": (
            "probe_attempts", "probe_attempt_id", "legacy-probe-attempt-0",
        ),
        "artifact": (
            "probe_artifact_sets", "probe_artifact_set_id",
            "legacy-artifact-0",
        ),
        "evaluation": (
            "probe_evaluations", "evaluation_id", "legacy-evaluation-0",
        ),
        "decision": (
            "probe_decisions", "decision_id", "legacy-decision",
        ),
    }
    relation, key, value = targets[table]
    connection = store.connect()
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(f"DELETE FROM {relation} WHERE {key}=?", (value,))
        connection.commit()
    finally:
        connection.close()

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_compute_reservations"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


@pytest.mark.parametrize("table", [
    "task", "attempt", "run", "trial", "probe_attempt", "artifact",
    "evaluation", "decision",
])
def test_legacy_probe_invalid_linkage_before_first_reconcile_fails_closed(
    tmp_path, table,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    empty = json.dumps({}, sort_keys=True, separators=(",", ":"))
    statements = {
        "task": (
            "UPDATE tasks SET payload_json=? WHERE task_id='legacy-task'",
            (empty,),
        ),
        "attempt": (
            "UPDATE attempts SET state='FAILED' "
            "WHERE attempt_id='legacy-attempt'", (),
        ),
        "run": (
            "UPDATE probe_runs SET source_snapshot_id='wrong-source' "
            "WHERE probe_run_id='legacy-run'", (),
        ),
        "trial": (
            "UPDATE probe_trials SET candidate_json=? "
            "WHERE probe_trial_id='legacy-trial-0'", (empty,),
        ),
        "probe_attempt": (
            "UPDATE probe_attempts SET growth_receipt_json=? "
            "WHERE probe_attempt_id='legacy-probe-attempt-0'", (empty,),
        ),
        "artifact": (
            "UPDATE probe_artifact_sets SET manifest_json=?,manifest_sha256=? "
            "WHERE probe_artifact_set_id='legacy-artifact-0'",
            (empty, content_sha256({})),
        ),
        "evaluation": (
            "UPDATE probe_evaluations SET profile_sha256=? "
            "WHERE evaluation_id='legacy-evaluation-0'", ("e" * 64,),
        ),
        "decision": (
            "UPDATE probe_decisions SET policy_sha256=? "
            "WHERE decision_id='legacy-decision'", ("d" * 64,),
        ),
    }
    connection = store.connect()
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(*statements[table])
        connection.commit()
    finally:
        connection.close()

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_historical_imports_v19"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("table", [
    "task", "attempt", "run", "trial", "probe_attempt", "artifact",
    "evaluation", "decision",
])
def test_legacy_probe_extra_row_before_first_reconcile_fails_closed(
    tmp_path, table,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    specifications = {
        "task": ("tasks", "task_id", "legacy-task", {
            "task_id": "legacy-extra-task", "cell_id": "cell-extra",
        }),
        "attempt": ("attempts", "attempt_id", "legacy-attempt", {
            "attempt_id": "legacy-extra-attempt", "attempt_number": 2,
        }),
        "run": ("probe_runs", "probe_run_id", "legacy-run", {
            "probe_run_id": "legacy-extra-run",
            "task_id": "legacy-missing-extra-task",
        }),
        "trial": ("probe_trials", "probe_trial_id", "legacy-trial-0", {
            "probe_trial_id": "legacy-extra-trial",
            "candidate_id": "candidate-extra", "candidate_rank": 2,
        }),
        "probe_attempt": (
            "probe_attempts", "probe_attempt_id", "legacy-probe-attempt-0", {
                "probe_attempt_id": "legacy-extra-probe-attempt",
                "attempt_number": 2,
            },
        ),
        "artifact": (
            "probe_artifact_sets", "probe_artifact_set_id",
            "legacy-artifact-0", {
                "probe_artifact_set_id": "legacy-extra-artifact",
                "probe_trial_id": "legacy-orphan-trial",
                "probe_attempt_id": "legacy-orphan-attempt",
            },
        ),
        "evaluation": (
            "probe_evaluations", "evaluation_id", "legacy-evaluation-0", {
                "evaluation_id": "legacy-extra-evaluation",
                "probe_trial_id": "legacy-orphan-trial",
                "probe_artifact_set_id": "legacy-orphan-artifact",
            },
        ),
        "decision": (
            "probe_decisions", "decision_id", "legacy-decision", {
                "decision_id": "legacy-extra-decision",
                "probe_run_id": "legacy-orphan-run",
            },
        ),
    }
    relation, key, value, replacements = specifications[table]
    connection = store.connect()
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        original = dict(connection.execute(
            f"SELECT * FROM {relation} WHERE {key}=?", (value,),
        ).fetchone())
        original.update(replacements)
        columns = list(original)
        connection.execute(
            f"INSERT INTO {relation} ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)})",
            tuple(original[column] for column in columns),
        )
        connection.commit()
    finally:
        connection.close()

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_compute_reservations"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("authority", [
    "source", "missing_source", "cap", "missing_cap", "profile",
    "alternative_arm",
])
def test_legacy_probe_current_authority_drift_before_import_fails_closed(
    tmp_path, authority,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    connection = store.connect()
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        if authority == "source":
            source = json.loads(connection.execute(
                "SELECT payload_json FROM source_snapshots "
                "WHERE source_snapshot_id='legacy-source'"
            ).fetchone()[0])
            source["source_snapshot_sha256"] = "b" * 64
            connection.execute(
                "UPDATE source_snapshots SET payload_json=? "
                "WHERE source_snapshot_id='legacy-source'",
                (json.dumps(source, sort_keys=True, separators=(",", ":")),),
            )
        elif authority == "missing_source":
            connection.execute(
                "DELETE FROM source_snapshots "
                "WHERE source_snapshot_id='legacy-source'"
            )
        elif authority == "cap":
            connection.execute(
                "UPDATE first_letters_discovery_compute_caps "
                "SET authority_sha256=? WHERE mission_id='mission-a'",
                ("b" * 64,),
            )
        elif authority == "missing_cap":
            connection.execute(
                "DELETE FROM first_letters_discovery_compute_caps "
                "WHERE mission_id='mission-a'"
            )
        else:
            run = connection.execute(
                "SELECT policy_json FROM probe_runs "
                "WHERE probe_run_id='legacy-run'"
            ).fetchone()
            policy = json.loads(run[0])
            if authority == "profile":
                policy["discovery_profile"]["deployed_revision"] = "2" * 40
                policy["discovery_profile"]["scientific_core_sha256"] = (
                    content_sha256({
                        key: value
                        for key, value in policy["discovery_profile"].items()
                        if key != "scientific_core_sha256"
                    })
                )
                profile_bytes = json.dumps(
                    policy["discovery_profile"], sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                policy["discovery_profile_file_sha256"] = __import__(
                    "hashlib"
                ).sha256(profile_bytes).hexdigest()
            else:
                policy["arm_kind"] = "ALTERNATIVE_SOURCE_ARM"
            connection.execute(
                "UPDATE probe_runs SET policy_json=?,policy_sha256=? "
                "WHERE probe_run_id='legacy-run'",
                (
                    json.dumps(policy, sort_keys=True, separators=(",", ":")),
                    content_sha256(policy),
                ),
            )
        connection.commit()
    finally:
        connection.close()

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_historical_imports_v19"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("seal", [
    "policy", "candidate_set", "locked_plan", "evaluation_result",
    "decision_receipt", "decision_evidence",
])
def test_legacy_probe_json_hash_and_cross_row_seals_fail_closed(
    tmp_path, seal,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    empty = json.dumps({}, sort_keys=True, separators=(",", ":"))
    statements = {
        "policy": (
            "UPDATE probe_runs SET policy_json=?,policy_sha256=? "
            "WHERE probe_run_id='legacy-run'",
            (empty, content_sha256({})),
        ),
        "candidate_set": (
            "UPDATE probe_runs SET candidate_set_json=?,candidate_set_sha256=? "
            "WHERE probe_run_id='legacy-run'",
            (empty, content_sha256({})),
        ),
        "locked_plan": (
            "UPDATE probe_trials SET locked_plan_json=?,locked_plan_sha256=? "
            "WHERE probe_trial_id='legacy-trial-0'",
            (empty, content_sha256({})),
        ),
        "evaluation_result": (
            "UPDATE probe_evaluations SET result_json=?,result_sha256=? "
            "WHERE evaluation_id='legacy-evaluation-0'",
            (empty, content_sha256({})),
        ),
        "decision_receipt": (
            "UPDATE probe_decisions SET receipt_json=?,receipt_sha256=? "
            "WHERE decision_id='legacy-decision'",
            (empty, content_sha256({})),
        ),
        "decision_evidence": (
            "UPDATE probe_decisions SET evidence_set_sha256=? "
            "WHERE decision_id='legacy-decision'",
            ("b" * 64,),
        ),
    }
    with store.connect() as connection:
        connection.execute(*statements[seal])

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0


def test_complete_legacy_probe_reconcile_is_byte_identical_and_idempotent(
    tmp_path,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)

    first = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    with store.connect() as connection:
        first_reservations = [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_compute_reservations "
            "ORDER BY reservation_id"
        )]
        first_work = [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_work_bindings "
            "ORDER BY reservation_id"
        )]
        first_imports = [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_historical_imports_v19 "
            "ORDER BY import_id"
        )]
    second = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert second == first
    assert store.discovery_compute_total("mission-a") == 24
    with store.connect() as connection:
        assert [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_compute_reservations "
            "ORDER BY reservation_id"
        )] == first_reservations
        assert [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_work_bindings "
            "ORDER BY reservation_id"
        )] == first_work
        assert [tuple(row) for row in connection.execute(
            "SELECT * FROM first_letters_discovery_historical_imports_v19 "
            "ORDER BY import_id"
        )] == first_imports


@pytest.mark.parametrize("damage", [
    "delete_import", "delete_work", "delete_reservation",
    "delete_materialized_graph", "mutate_import", "mutate_work",
    "mutate_reservation",
])
def test_legacy_materialized_import_damage_blocks_and_never_releases_cap(
    tmp_path, damage,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    complete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    assert complete["state"] == "COMPLETE"
    with store.connect() as connection:
        reservation_id = connection.execute(
            "SELECT reservation_id FROM "
            "first_letters_discovery_compute_reservations "
            "WHERE source='IMPORTED_HISTORICAL_EXACT'"
        ).fetchone()[0]
        if damage == "delete_import":
            connection.execute(
                "DELETE FROM first_letters_discovery_historical_imports_v19"
            )
        elif damage == "delete_work":
            connection.execute(
                "DELETE FROM first_letters_discovery_work_bindings "
                "WHERE reservation_id=?", (reservation_id,),
            )
        elif damage == "delete_reservation":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute(
                "DELETE FROM first_letters_discovery_compute_reservations "
                "WHERE reservation_id=?", (reservation_id,),
            )
        elif damage == "delete_materialized_graph":
            connection.execute(
                "DELETE FROM first_letters_discovery_historical_imports_v19"
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_work_bindings "
                "WHERE reservation_id=?", (reservation_id,),
            )
            connection.execute(
                "DELETE FROM first_letters_discovery_compute_reservations "
                "WHERE reservation_id=?", (reservation_id,),
            )
        elif damage == "mutate_import":
            connection.execute(
                "UPDATE first_letters_discovery_historical_imports_v19 "
                "SET source_snapshot_sha256=?", ("b" * 64,),
            )
        elif damage == "mutate_work":
            connection.execute(
                "UPDATE first_letters_discovery_work_bindings "
                "SET dispatch_kind='BASELINE_DISPATCH' "
                "WHERE reservation_id=?", (reservation_id,),
            )
        else:
            connection.execute(
                "UPDATE first_letters_discovery_compute_reservations "
                "SET reserved_units=48 WHERE reservation_id=?",
                (reservation_id,),
            )

    incomplete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    assert incomplete["fixed_units"] == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
        assert connection.execute(
            "SELECT COUNT(*) FROM "
            "first_letters_discovery_compute_reservations "
            "WHERE source='RESERVED_BEFORE_EXECUTION'"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("table", [
    "task", "attempt", "run", "trial", "probe_attempt", "artifact",
    "evaluation", "decision",
])
def test_legacy_probe_mutated_retained_row_changes_watermark_and_blocks(
    tmp_path, table,
):
    from test_first_letters_discovery_evidence_store import _cap

    store = _history_store(tmp_path)
    store.register_discovery_compute_cap(_cap())
    _insert_incomplete_legacy_probe(store)
    _complete_legacy_probe_graph(store)
    complete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    assert complete["state"] == "COMPLETE"
    statements = {
        "task": ("UPDATE tasks SET payload_json='{}' WHERE task_id='legacy-task'", ()),
        "attempt": (
            "UPDATE attempts SET result_json='{}' WHERE attempt_id='legacy-attempt'",
            (),
        ),
        "run": (
            "UPDATE probe_runs SET executor_fingerprint_json=? "
            "WHERE probe_run_id='legacy-run'",
            (json.dumps({"changed": True}, sort_keys=True, separators=(",", ":")),),
        ),
        "trial": (
            "UPDATE probe_trials SET candidate_json='{}' "
            "WHERE probe_trial_id='legacy-trial-0'", (),
        ),
        "probe_attempt": (
            "UPDATE probe_attempts SET result_json=? "
            "WHERE probe_attempt_id='legacy-probe-attempt-0'",
            (json.dumps({"changed": True}, sort_keys=True, separators=(",", ":")),),
        ),
        "artifact": (
            "UPDATE probe_artifact_sets SET manifest_json='{}' "
            "WHERE probe_artifact_set_id='legacy-artifact-0'", (),
        ),
        "evaluation": (
            "UPDATE probe_evaluations SET result_json='{}' "
            "WHERE evaluation_id='legacy-evaluation-0'", (),
        ),
        "decision": (
            "UPDATE probe_decisions SET receipt_json='{}' "
            "WHERE decision_id='legacy-decision'", (),
        ),
    }
    with store.connect() as connection:
        connection.execute(*statements[table])

    incomplete = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert incomplete["state"] == "CONTROL_INCOMPLETE"
    assert incomplete["watermark_sha256"] != complete["watermark_sha256"]
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


def test_retained_alternative_requires_exact_arm_authority(tmp_path):
    store, _branch = _retained_v16_execution_store(
        tmp_path, alternative=True,
    )
    store._first_letters_experimental_arm_resolver = lambda _arm_id: None

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT reason FROM first_letters_discovery_compute_blocks "
            "WHERE mission_id='mission-a'"
        ).fetchone()[0] == "CONTROL_INCOMPLETE_COMPUTE_LEDGER"


def test_two_item_retained_v16_reservation_imports_each_execution_once(tmp_path):
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from test_first_letters_discovery_compute import _reserve, _store
    from test_first_letters_discovery_evidence_store import (
        _FixtureDiscoveryExecutor,
        _fixture_executor_registration,
    )

    store = _store(tmp_path, cap_units=48)
    profile = json.loads(store._fixture_profile_bytes)
    profile["canonical_ordered_cell_set_sha256"] = content_sha256(
        ["cell-a", "cell-b"]
    )
    profile["scientific_core_sha256"] = content_sha256({
        key: value for key, value in profile.items()
        if key != "scientific_core_sha256"
    })
    profile_bytes = json.dumps(
        profile, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    store._first_letters_discovery_profile_resolver = (
        lambda _mission_id, _source_id: profile_bytes
    )
    executor = _FixtureDiscoveryExecutor()
    store._first_letters_discovery_executor = executor
    store._first_letters_discovery_worker_id = (
        "registered-fixture-discovery-worker"
    )
    store._first_letters_discovery_executor_id = "fixture-discovery-executor-v1"
    store.register_first_letters_discovery_executor(
        _fixture_executor_registration()
    )
    with store.connect() as connection:
        source = json.loads(connection.execute(
            "SELECT payload_json FROM source_snapshots "
            "WHERE source_snapshot_id='source-a'"
        ).fetchone()[0])
        source["first_letters_discovery_authority"].update({
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
        })
        source.update({
            "m7_model_sha256": "7" * 64,
            "candidate_provider_id": "fixture-provider-v1",
            "candidate_provider_sha256": "8" * 64,
            "discovery_minimum_separation": 12,
        })
        connection.execute(
            "UPDATE source_snapshots SET payload_json=? "
            "WHERE source_snapshot_id='source-a'",
            (json.dumps(source, sort_keys=True, separators=(",", ":")),),
        )
    branch = _reserve(
        store, "BASELINE_ARM", items=("cell-a", "cell-b")
    )
    for job in branch["jobs"]:
        result = FirstLettersDiscoveryController(
            mode="shadow", store=store, provider=_BridgeProvider(),
        ).run_job(job_id=job["job_id"], lease_seconds=60)
        assert result["state"] == "COMPLETED"
    reservation_id = branch["reservation"]["reservation_id"]
    with store.connect() as connection:
        connection.execute(
            "DELETE FROM first_letters_discovery_jobs_v19 WHERE reservation_id=?",
            (reservation_id,),
        )
        connection.execute(
            "DELETE FROM first_letters_discovery_dispatches_v19 "
            "WHERE reservation_id=?", (reservation_id,),
        )
        connection.execute(
            "DELETE FROM first_letters_discovery_native_adapters_v19 "
            "WHERE reservation_id=?", (reservation_id,),
        )
        connection.execute(
            "DELETE FROM first_letters_discovery_history_reconciliations_v19 "
            "WHERE mission_id='mission-a'"
        )

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "COMPLETE"
    assert reconciliation["fixed_units"] == 48
    with store.connect() as connection:
        imports = connection.execute(
            "SELECT reservation_id,item_id,fixed_units FROM "
            "first_letters_discovery_historical_imports_v19 ORDER BY item_id"
        ).fetchall()
    assert [row["item_id"] for row in imports] == ["cell-a", "cell-b"]
    assert {row["reservation_id"] for row in imports} == {reservation_id}
    assert [row["fixed_units"] for row in imports] == [24, 24]
    assert store.discovery_compute_total("mission-a") == 48
