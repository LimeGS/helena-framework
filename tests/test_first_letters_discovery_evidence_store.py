from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256
from fleet.discovery_executor import (
    PRODUCTION_DISCOVERY_EXECUTOR_ID,
    ProductionFirstLettersDiscoveryExecutor,
    production_discovery_executor_registration,
)
from fleet import seed_probe
from fleet.store import FleetStore
from fleet.store_factory import open_fleet_store


PROFILE_PATH = STAGE / "fleet/profiles/first-letters-discovery-v1.json"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _profile_bytes(
    kind: str = "BASELINE_ARM", *, ordered_items: tuple[str, ...] = ("cell-a",),
) -> bytes:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profile.update({
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "d" * 64,
        "ct_metadata_sha256": "2" * 64,
        "ct_read_set_manifest_sha256": "e" * 64,
        "m7_model_id": "m7-v1",
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_threshold": 0.5,
        "m7_transform_sha256": "f" * 64,
        "m7_read_set_manifest_sha256": "0" * 64,
        "canonical_ordered_cell_set_sha256": content_sha256(
            list(ordered_items)
        ),
        "mission_compute_cap_authority_id": "cap-a",
        "mission_compute_cap_authority_sha256": _cap()["authority_sha256"],
        "mission_compute_cap_units": 24,
        "deployed_revision": "1" * 40,
    })
    if kind == "ALTERNATIVE_SOURCE_ARM":
        profile.update({
            "arm_kind": "ALTERNATIVE_SOURCE_ARM",
            "experimental_arm_admission_id": "arm-a",
            "experimental_arm_admission_sha256": "6" * 64,
        })
    core = {
        key: value for key, value in profile.items()
        if key != "scientific_core_sha256"
    }
    profile["scientific_core_sha256"] = content_sha256(core)
    return _json_bytes(profile)


def _cap() -> dict:
    core = {
        "schema": "campaignx.first_letters_discovery_compute_cap.v1",
        "mission_id": "mission-a",
        "cap_authority_id": "cap-a",
        "compute_unit": "probe_generation_units",
        "mission_compute_cap_units": 24,
        "top_k": 2,
        "probe_generations": 12,
        "maximum_attempts_per_candidate": 1,
        "probe_profile_id": "vc3d-m7-probe-v1",
        "probe_profile_file_sha256": (
            "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33"
        ),
        "deployed_revision": "1" * 40,
        "policy_chain_id": "policy-chain-a",
        "policy_chain_sha256": "c" * 64,
        "allow_unvalidated": False,
    }
    return {**core, "authority_sha256": content_sha256(core)}


def _work(
    profile_bytes: bytes, items: tuple[str, ...], *,
    kind: str = "BASELINE_ARM", parent_task_id: str | None = "task-a",
) -> dict:
    region = {"minimum": [0, 0, 0], "maximum": [64, 64, 64]}
    item_bindings = [{
        "schema": "campaignx.first_letters_discovery_work_item_binding.v1",
        "item_id": item_id, "sample_id": "PHercA",
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "cell_region": copy.deepcopy(region),
        "cell_region_sha256": content_sha256(region),
        "grid_version": "first-letters-grid-v1",
        "grid_spec_sha256": content_sha256({
            "grid_version": "first-letters-grid-v1", "cell_id": item_id,
            "ct_l0_region": region,
        }),
        "scientific_opportunity_id": (
            "opportunity-a" if item_id == "cell-a"
            else f"opportunity-{item_id}"
        ),
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": "a" * 64,
        "parent_task_id": parent_task_id, "parent_attempt_id": None,
        "allow_unvalidated": False,
    } for item_id in items]
    core = {
        "schema": {
            "BASELINE_ARM":
                "campaignx.first_letters_discovery_baseline_work_admission.v1",
            "ALTERNATIVE_SOURCE_ARM":
                "campaignx.first_letters_experimental_arm_admission.v1",
        }[kind],
        "work_authority_id": "work-a",
        "mission_id": "mission-a",
        "work_kind": kind,
        "ordered_item_ids": list(items),
        "ordered_item_ids_sha256": content_sha256(list(items)),
        "ordered_item_bindings": item_bindings,
        "ordered_item_bindings_sha256": content_sha256(item_bindings),
        "cap_authority_id": "cap-a",
        "cap_authority_sha256": _cap()["authority_sha256"],
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "policy_sha256": "4" * 64,
        "source_sha256": "1" * 64,
        "deployed_revision": "1" * 40,
        "requested_item_count": len(items),
        "requested_units": len(items) * 24,
        "allow_unvalidated": False,
    }
    return {**core, "work_authority_sha256": content_sha256(core)}


class _FixtureDiscoveryExecutor:
    """Trusted server dependency used to exercise the executor-owned seam."""

    def __init__(self):
        self.measurements = None
        self._claim_tokens = {}

    def accept_first_letters_discovery_claim(self, *, run_id, claim_token):
        self._claim_tokens[run_id] = claim_token

    def first_letters_discovery_claim_token(self, *, run_id):
        return self._claim_tokens.get(run_id)

    def measure_first_letters_discovery_run(
        self, *, executor_claim, run_authority, provider_request,
        provider_response_bytes, source_snapshot,
    ):
        assert executor_claim == run_authority["executor_claim"]
        assert source_snapshot["source_snapshot_id"] == run_authority[
            "source_snapshot_id"
        ]
        assert provider_request["request_id"] == run_authority[
            "reservation_request_id"
        ]
        if self.measurements is not None:
            return tuple(self.measurements)
        candidates = json.loads(provider_response_bytes)["candidates"]
        return tuple(
            _measurement(
                row["candidate_id"],
                coordinate=tuple(
                    row["ct_l0_coordinate"][axis] for axis in "xyz"
                ),
            )
            for row in candidates
        )


def _fixture_executor_registration(
    *, worker_id="registered-fixture-discovery-worker",
    executor_sha256=None, capabilities=None,
):
    core = {
        "schema": "campaignx.first_letters_discovery_executor_registration.v1",
        "worker_id": worker_id,
        "executor_id": "fixture-discovery-executor-v1",
        "executor_sha256": executor_sha256 or hashlib.sha256(
            inspect.getsource(_FixtureDiscoveryExecutor).encode("utf-8")
        ).hexdigest(),
        "capabilities": capabilities if capabilities is not None else [
            "FIRST_LETTERS_DISCOVERY_CT_PROBE_V1"
        ],
        "enabled": True,
        "allow_unvalidated": False,
    }
    return {**core, "registration_sha256": content_sha256(core)}


def _store(
    tmp_path: Path, *, reservation_items=("cell-a",),
    kind: str = "BASELINE_ARM", create_parent: bool = True,
    provider_id: str = "fixture-provider-v1",
    work_binding_changes: dict | None = None,
    scientific_opportunity_id: str = "opportunity-a",
    p0_artifact_id: str = "p0-a", p0_artifact_sha256: str = "a" * 64,
    promotion_authority: dict | None = None,
    parent_authority: dict | None = None,
    registered_budget_admission: dict | None = None,
    claim_parent: bool = False,
):
    executor = _FixtureDiscoveryExecutor()
    profile_bytes = _profile_bytes(kind, ordered_items=reservation_items)
    cap = _cap()
    store = FleetStore(
        tmp_path / "fleet.sqlite",
        first_letters_discovery_executor=executor,
        first_letters_discovery_worker_id=(
            "registered-fixture-discovery-worker"
        ),
        first_letters_discovery_executor_id=(
            "fixture-discovery-executor-v1"
        ),
        first_letters_discovery_profile_resolver=(
            lambda mission_id, source_snapshot_id: profile_bytes
        ),
    )
    store._fixture_discovery_executor = executor
    store.initialize()
    store.register_first_letters_discovery_executor(
        _fixture_executor_registration()
    )
    store.register_snapshot({
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "d" * 64,
        "sample_id": "PHercA",
        "ct_uri": "https://provider.invalid/ct",
        "ct_sha256": "2" * 64,
        "ct_metadata_sha256": "2" * 64,
        "ct_read_set_manifest_sha256": "e" * 64,
        "m7_uri": "https://provider.invalid/m7?mode=shadow",
        "m7_sha256": "3" * 64,
        "m7_metadata_sha256": "3" * 64,
        "m7_read_set_manifest_sha256": "0" * 64,
        "m7_model_id": "m7-v1",
        "m7_model_sha256": "7" * 64,
        "candidate_provider_id": provider_id,
        "candidate_provider_sha256": "8" * 64,
        "discovery_minimum_separation": 12,
        "m7_resolution": 4,
        "m7_level": 1,
        "m7_threshold": 0.5,
        "m7_transform_sha256": "f" * 64,
        "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a",
            "accepted_p0_artifact_id": p0_artifact_id,
            "accepted_p0_artifact_sha256": p0_artifact_sha256,
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {
                item_id: (
                    scientific_opportunity_id if item_id == "cell-a"
                    else f"opportunity-{item_id}"
                )
                for item_id in reservation_items
            },
            **({"promotion_authority": copy.deepcopy(promotion_authority)}
               if promotion_authority is not None else {}),
        },
    })
    budget_core = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": "mission-a", "sample_id": "PHercA",
        "receipt_sha256": "6" * 64,
        "preflight_receipt_sha256": "7" * 64,
        "preflight_sanitized_receipt_sha256": "8" * 64,
        "approved_task_count": len(reservation_items),
        "order_seed_sha256": "9" * 64,
        "population_order_sha256": "0" * 64,
        "prefix_sha256": content_sha256(list(reservation_items)),
        "prefix_cell_ids": list(reservation_items),
        "execution_bindings": {
            "source_snapshot_id": "source-a",
            "grid_version": "first-letters-grid-v1",
            "policy_version": "first-letters-search@1.0.0",
            "p0_artifact_id": p0_artifact_id,
            "p0_artifact_sha256": p0_artifact_sha256,
            "catalog_snapshot_sha256": "3" * 64,
        },
    }
    budget = (
        copy.deepcopy(registered_budget_admission)
        if registered_budget_admission is not None
        else {**budget_core, "admission_sha256": content_sha256(budget_core)}
    )
    store.create_tasks([{
        "task_id": "task-a" if item_id == "cell-a" else f"task-{item_id}",
        "mission_id": "mission-a", "sample_id": "PHercA",
        "source_snapshot_id": "source-a", "cell_id": item_id,
        "grid_version": "first-letters-grid-v1",
        "policy_version": "first-letters-search@1.0.0",
        "bounds_xyz": [[0, 0, 0], [64, 64, 64]],
        "center_xyz": {"x": 32, "y": 32, "z": 32},
        "priority": 1.0,
        "parameter_envelope": {
            "generations": {"minimum": 20, "maximum": 45, "default": 35},
            "step_size": {"minimum": 12, "maximum": 24, "default": 20},
            "min_area_cm": {"minimum": 0.0, "maximum": 0.0, "default": 0.0},
            "use_cuda": {"allowed": [False], "default": False},
        },
        "catalog_snapshot_sha256": "3" * 64,
        "selection_rank": rank,
        "campaign_budget_admission_sha256": budget["admission_sha256"],
        "scientific_opportunity_id": (
            scientific_opportunity_id if item_id == "cell-a"
            else f"opportunity-{item_id}"
        ),
        "accepted_p0_artifact_id": p0_artifact_id,
        "accepted_p0_artifact_sha256": p0_artifact_sha256,
        "candidate_discovery": {
            "region": {"minimum": [0, 0, 0], "maximum": [64, 64, 64]},
            "minimum_separation": 12,
            "provider_id": "fixture-provider-v1",
            "provider_sha256": "8" * 64,
            "model_sha256": "7" * 64,
        },
        **copy.deepcopy(parent_authority or {}),
    } for rank, item_id in enumerate(reservation_items)])
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO campaign_budget_admissions
               (mission_id,sample_id,receipt_sha256,admission_json,
                admission_sha256,created_at) VALUES(?,?,?,?,?,?)""",
            (
                "mission-a", "PHercA", budget["receipt_sha256"],
                json.dumps(budget, sort_keys=True, separators=(",", ":")),
                budget["admission_sha256"], "2026-01-01T00:00:00Z",
            ),
        )
    parent_attempt_id = None
    if claim_parent:
        claim = store.claim("registered-parent-worker", 60, task_id="task-a")
        assert claim is not None
        parent_attempt_id = claim["attempt_id"]
    store.register_discovery_compute_cap(cap)
    if kind == "ALTERNATIVE_SOURCE_ARM":
        arm_core = {
            "schema": "campaignx.first_letters_experimental_arm_admission.v1",
            "arm_id": "arm-a", "mission_id": "mission-a",
            "accepted_p0_id": p0_artifact_id,
            "accepted_p0_sha256": p0_artifact_sha256,
            "source_snapshot_id": "source-a",
            "source_snapshot_sha256": "1" * 64,
            "source_content_lock_sha256": "d" * 64,
            "ct_metadata_sha256": "2" * 64,
            "ct_read_set_manifest_sha256": "e" * 64,
            "m7_metadata_sha256": "3" * 64,
            "m7_read_set_manifest_sha256": "0" * 64,
            "m7_model_id": "m7-v1", "m7_resolution": 4,
            "m7_level": 1, "m7_transform_sha256": "f" * 64,
            "m7_threshold": 0.5,
            "discovery_policy_id": "first-letters-alt@1.0.0",
            "discovery_profile_sha256": hashlib.sha256(
                profile_bytes
            ).hexdigest(),
            "deployed_revision": "1" * 40,
            "preflight_private_sha256": "0" * 64,
            "preflight_sanitized_sha256": "1" * 64,
            "ordered_cell_ids": list(reservation_items),
            "ordered_cell_set_sha256": content_sha256(
                list(reservation_items)
            ),
            "mission_compute_cap_authority_id": "cap-a",
            "mission_compute_cap_authority_sha256": cap["authority_sha256"],
            "requested_units": len(reservation_items) * 24,
            "active_policy_chain_sha256": "c" * 64,
            "may_update_accepted_p0": False,
            "statistical_budget_delta": 0, "allow_unvalidated": False,
        }
        arm = {**arm_core, "admission_sha256": content_sha256(arm_core)}
        store._first_letters_experimental_arm_resolver = (
            lambda arm_id: copy.deepcopy(arm) if arm_id == "arm-a" else None
        )
        reservation = store.reserve_first_letters_alternative_shadow(
            request_id="request-a",
            budget_admission_sha256=budget["admission_sha256"], arm_id="arm-a",
        )
    else:
        reservation = store.reserve_first_letters_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=budget["admission_sha256"],
        )
    assert reservation is not None
    return store, profile_bytes, reservation


def _claim_job(
    store, reservation: dict, *, item_id: str = "cell-a",
    lease_seconds: int = 60,
):
    matches = [
        job for job in reservation["jobs"] if job["item_id"] == item_id
    ]
    if len(matches) != 1:
        raise ValueError("fixture discovery job is missing or ambiguous")
    return store.claim_first_letters_discovery_job(
        job_id=matches[0]["job_id"], lease_seconds=lease_seconds,
    )._run_handle


@pytest.mark.parametrize("create_parent", [True, False])
def test_begin_rejects_coherently_rehashed_opportunity_and_p0_before_provider_io(
    tmp_path, create_parent,
):
    store, _profile_bytes_value, reservation = _store(
        tmp_path, create_parent=create_parent,
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE first_letters_discovery_jobs_v19 "
            "SET item_id='forged-opportunity-p0-cell'"
        )
    with pytest.raises(ValueError, match="(?i)job|dispatch|graph"):
        _claim_job(store, reservation)


def test_discovery_worker_identity_is_store_derived_not_a_claim_argument():
    parameters = inspect.signature(
        FleetStore.claim_first_letters_discovery_job
    ).parameters
    assert "worker_id" not in parameters


def test_begin_evidence_run_is_sealed_to_dedicated_worker_claim_item_and_profile(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    assert handle.worker_id and handle.worker_id != "worker-a"
    assert handle.cell_id == "cell-a"
    assert handle.provider_request["request_id"] == "request-a"

    for changed in ({"job_id": "job-forged"}, {"lease_seconds": 1}):
        arguments = {
            "lease_seconds": 60,
            "job_id": reservation["jobs"][0]["job_id"],
        }
        arguments.update(changed)
        with pytest.raises(ValueError):
            store.claim_first_letters_discovery_job(**arguments)

    wrong_store, _wrong_profile, wrong_reservation = _store(
        tmp_path / "wrong-item", reservation_items=("cell-b",),
    )
    with pytest.raises(ValueError, match="item|cell|job"):
        _claim_job(wrong_store, wrong_reservation, item_id="cell-a")


def test_sqlite_evidence_file_order_is_explicit_and_matches_postgres(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    with store.connect() as connection:
        columns = {
            row[1]: row for row in connection.execute(
                "PRAGMA table_info(first_letters_discovery_evidence_files)"
            ).fetchall()
        }
    assert columns["file_order"][3] == 1


def test_alternative_evidence_claim_needs_no_canonical_task_and_does_not_mutate_it(
    tmp_path,
):
    store, profile_bytes, reservation = _store(
        tmp_path, kind="ALTERNATIVE_SOURCE_ARM", create_parent=False,
    )
    with store.connect() as connection:
        before = (
            connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
        )
    handle = _claim_job(store, reservation)
    assert handle.worker_id and handle.worker_id != "worker-alt"
    assert handle.cell_id == "cell-a"
    with store.connect() as connection:
        after = (
            connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
        )
    assert after == before == (1, 0)


def test_shadow_evidence_run_leaves_canonical_task_and_attempt_rows_byte_identical(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    with store.connect() as connection:
        before = {
            "tasks": [tuple(row) for row in connection.execute(
                "SELECT * FROM tasks ORDER BY task_id"
            ).fetchall()],
            "attempts": [tuple(row) for row in connection.execute(
                "SELECT * FROM attempts ORDER BY attempt_id"
            ).fetchall()],
        }
    handle = _claim_job(store, reservation)
    completed = _complete(store, handle)
    store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    with store.connect() as connection:
        after = {
            "tasks": [tuple(row) for row in connection.execute(
                "SELECT * FROM tasks ORDER BY task_id"
            ).fetchall()],
            "attempts": [tuple(row) for row in connection.execute(
                "SELECT * FROM attempts ORDER BY attempt_id"
            ).fetchall()],
        }
    assert after == before


def _provider_response_bytes(handle, candidates=None) -> bytes:
    request = handle.provider_request
    identity_fields = (
        "request_id", "cell_id", "source_snapshot_id",
        "source_snapshot_sha256", "prediction_root_sha256", "resolution",
        "level", "model_id", "model_sha256", "provider_id",
        "provider_sha256", "cell_region_sha256", "grid_spec_sha256",
        "dependency_manifest_sha256", "maximum_candidates",
    )
    return _json_bytes({
        "prediction_identity": {
            field: copy.deepcopy(request[field]) for field in identity_fields
        },
        "candidates": candidates if candidates is not None else [{
            "candidate_id": "candidate-a",
            "cell_id": "cell-a",
            "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56},
            "score": 0.9,
        }],
    })


def _measurement(
    candidate_id: str, *, ct_nonzero_voxels: int | None = 1,
    probe_measurable: bool | None = True, coordinate=(12, 34, 56),
    run_handle=None,
):
    coordinate_sha = seed_probe.coordinate_sha256_v1(list(coordinate))
    if run_handle is not None:
        response = json.loads(
            _provider_response_bytes(run_handle).decode("utf-8")
        )
        candidate = next(
            row for row in response["candidates"]
            if row["candidate_id"] == candidate_id
        )
        coordinate = candidate["ct_l0_coordinate"]
        coordinate_sha = seed_probe.coordinate_sha256_v1([
            coordinate["x"], coordinate["y"], coordinate["z"],
        ])
    ct_read = {
        "schema": "campaignx.first_letters_ct_material_read_evidence.v1",
        "candidate_id": candidate_id,
        "source_snapshot_id": "source-a",
        "raw_coordinate_sha256": coordinate_sha,
        "ct_metadata_sha256": "2" * 64,
        "ct_read_set_manifest_sha256": "e" * 64,
        "sampled_voxel_count": 27,
        "nonzero_voxel_count": ct_nonzero_voxels,
        "allow_unvalidated": False,
    }
    probe_read = {
        "schema": "campaignx.first_letters_probe_geometry_read_evidence.v1",
        "candidate_id": candidate_id,
        "raw_coordinate_sha256": coordinate_sha,
        "probe_execution_profile_sha256": seed_probe.PROBE_PROFILE_SHA256,
        "measurement_complete": probe_measurable,
        "geometry_qc_state": (
            "GEOMETRY_CERTIFIED"
            if probe_measurable else "GEOMETRY_UNMEASURED"
        ),
        "allow_unvalidated": False,
    }
    return seed_probe.FirstLettersDiscoveryCandidateMeasurement(
        candidate_id=candidate_id,
        ct_read_evidence_bytes=(
            _json_bytes(ct_read) if ct_nonzero_voxels is not None else None
        ),
        probe_evidence_bytes=(
            _json_bytes(probe_read)
            if ct_nonzero_voxels and probe_measurable is not None else None
        ),
    )


def _complete(store, handle, *, candidates=None, measurements=None):
    response = _provider_response_bytes(handle, candidates)
    if measurements is None:
        candidate_rows = (
            json.loads(response.decode("utf-8"))["candidates"]
        )
        measurements = tuple(
            _measurement(
                row["candidate_id"],
                coordinate=tuple(
                    row["ct_l0_coordinate"][axis] for axis in "xyz"
                ),
            )
            for row in candidate_rows
        )
    store._fixture_discovery_executor.measurements = tuple(measurements)
    status = store.read_first_letters_discovery_evidence_run_status(handle.run_id)
    if status["state"] == "CLAIMED":
        store.start_first_letters_discovery_evidence_run(run_handle=handle)
    return store.complete_first_letters_discovery_evidence_run(
        run_handle=handle, provider_response_bytes=response,
    )


def test_complete_run_derives_authority_terminals_selection_and_builds_only_readback(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store._fixture_discovery_executor.measurements = (
        _measurement("candidate-a", run_handle=handle),
    )
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    completed = store.complete_first_letters_discovery_evidence_run(
        run_handle=handle,
        provider_response_bytes=_provider_response_bytes(handle),
    )
    artifact, receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    assert artifact["execution_authority_sha256"] == (
        completed["execution_authority"]["execution_authority_sha256"]
    )
    assert artifact["reservation_request_id"] == "request-a"
    assert artifact["reservation_work_authority_id"] == (
        reservation["reservation"]["work_authority_id"]
    )
    assert artifact["reservation_source"] == "RESERVED_BEFORE_EXECUTION"
    assert artifact["selected_candidate_id"] == "candidate-a"
    assert receipt["selection_outcome"] == "DISCOVERY_WINNER_RETAINED"
    ct_files = [
        row for row in completed["retained_files"]
        if row["role"] == "CT_MATERIAL_READ_EVIDENCE"
    ]
    assert len(ct_files) == 1
    assert hashlib.sha256(ct_files[0]["bytes"]).hexdigest() == (
        artifact["candidates"][0]["ct_terminal"]["ct_read_evidence_sha256"]
    )

    forged = copy.deepcopy(completed)
    forged["execution_authority"]["mission_id"] = "forged-mission"
    forged["execution_authority"]["execution_authority_sha256"] = content_sha256({
        key: value for key, value in forged["execution_authority"].items()
        if key != "execution_authority_sha256"
    })
    forged["reservation"]["mission_id"] = "forged-mission"
    forged["reservation"]["reservation_sha256"] = content_sha256({
        key: value for key, value in forged["reservation"].items()
        if key not in {"reservation_sha256", "created_at"}
    })
    forged["candidate_outcomes"][0]["ct_terminal"]["state"] = "CT_REJECTED"
    forged["candidate_outcomes"][0]["ct_terminal"][
        "ct_terminal_sha256"
    ] = content_sha256({
        key: value
        for key, value in forged["candidate_outcomes"][0]["ct_terminal"].items()
        if key != "ct_terminal_sha256"
    })
    forged["selection"]["selected_candidate_id"] = "forged-candidate"
    forged["selection"]["selection_sha256"] = content_sha256({
        key: value for key, value in forged["selection"].items()
        if key != "selection_sha256"
    })
    forged["retained_files"][0]["bytes"] = b'{"forged":true}'
    assert store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    ) == (artifact, receipt)


def test_completion_rejects_wrong_worker_item_token_and_stale_discovery_claim(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    for forged in (
        dataclasses.replace(handle, worker_id="worker-b"),
        dataclasses.replace(handle, cell_id="cell-b"),
        dataclasses.replace(handle, run_token="forged-token"),
    ):
        with pytest.raises(ValueError, match="stale|wrong|incomplete"):
            store.complete_first_letters_discovery_evidence_run(
                run_handle=forged,
                provider_response_bytes=_provider_response_bytes(handle),
            )
    with store.connect() as connection:
        connection.execute(
            "UPDATE first_letters_discovery_evidence_runs SET lease_expires_at=?",
            ("2000-01-01T00:00:00Z",),
        )
    with pytest.raises(ValueError, match="stale|wrong|incomplete"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
        )


@pytest.mark.parametrize("tamper", ["evidence_json", "retained_file"])
def test_store_readback_rejects_registry_tampering_before_artifact_build(
    tmp_path, tamper,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    completed = _complete(store, handle)
    with store.connect() as connection:
        if tamper == "evidence_json":
            evidence = json.loads(connection.execute(
                "SELECT evidence_json FROM first_letters_discovery_evidence_sets"
            ).fetchone()[0])
            evidence["selection"]["selected_candidate_id"] = "forged"
            connection.execute(
                "UPDATE first_letters_discovery_evidence_sets SET evidence_json=?",
                (_json_bytes(evidence).decode("utf-8"),),
            )
        else:
            connection.execute(
                """UPDATE first_letters_discovery_evidence_files
                      SET payload=? WHERE role='CANDIDATE_PROVIDER_RESPONSE'""",
                (b'{"forged":true}',),
            )
    with pytest.raises(ValueError, match="integrity drift|file drift"):
        store.build_first_letters_discovery_artifact_and_receipt(
            completed["evidence_set_id"]
        )


def test_selection_policy_chooses_unique_highest_score_and_retains_exact_inputs(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    candidates = [
        {"candidate_id": "candidate-a", "cell_id": "cell-a",
         "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9},
        {"candidate_id": "candidate-b", "cell_id": "cell-a",
         "ct_l0_coordinate": {"x": 40, "y": 10, "z": 20}, "score": 0.8},
    ]
    completed = _complete(store, handle, candidates=candidates)
    artifact, receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    assert receipt["selected_candidate_id"] == "candidate-a"
    policy = artifact["selection_policy_receipt"]
    assert [row["provider_order"] for row in policy["ordered_candidate_inputs"]] == [0, 1]
    assert [row["provider_score"] for row in policy["ordered_candidate_inputs"]] == [0.9, 0.8]


@pytest.mark.parametrize(
    ("candidates", "measurements", "expected"),
    [
        ([], (), "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER"),
        ([
            {"candidate_id": "candidate-a", "cell_id": "cell-a",
             "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9},
            {"candidate_id": "candidate-b", "cell_id": "cell-a",
             "ct_l0_coordinate": {"x": 40, "y": 10, "z": 20}, "score": 0.9},
        ], None, "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER"),
        ([
            {"candidate_id": "candidate-a", "cell_id": "cell-a",
             "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9},
        ], (_measurement(
            "candidate-a", ct_nonzero_voxels=0,
        ),), "DISCOVERY_REJECTED_CANDIDATES"),
    ],
)
def test_selection_policy_closes_zero_tie_and_all_rejected_outcomes(
    tmp_path, candidates, measurements, expected,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    completed = _complete(
        store, handle, candidates=candidates, measurements=measurements,
    )
    _artifact, receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    assert receipt["selection_outcome"] == expected
    assert receipt["selected_candidate_id"] is None


def test_content_informed_provider_or_selection_input_cannot_enter_registry(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    candidate = {
        "candidate_id": "candidate-a", "cell_id": "cell-a",
        "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9,
        "ocr_score": 1.0,
    }
    with pytest.raises(ValueError, match="content-informed"):
        _complete(store, handle, candidates=[candidate])


def _rehash_inputs(inputs: dict) -> None:
    core = copy.deepcopy({
        key: value for key, value in inputs.items()
        if key != "discovery_inputs_sha256"
    })
    response_bytes = core["provider_response"].pop("response_bytes")
    inputs["discovery_inputs_sha256"] = hashlib.sha256(
        _json_bytes(core) + b"\0" + response_bytes
    ).hexdigest()


@pytest.mark.parametrize("mutation", [
    lambda value: value["provider_request"].__setitem__("coordinate_frame", "zyx"),
    lambda value: value["provider_request"].__setitem__("threshold", True),
    lambda value: value["provider_request"].__setitem__("maximum_candidates", True),
    lambda value: value["provider_request"].__setitem__("minimum_separation", True),
    lambda value: value["provider_request"]["ct_l0_region"].__setitem__(
        "extra", [1, 2, 3]
    ),
    lambda value: value["dependencies"].pop(),
    lambda value: value["dependencies"].reverse(),
    lambda value: value["dependencies"][0].__setitem__("cell_id", "cell-b"),
    lambda value: value["source_snapshot"].__setitem__("grid_spec_sha256", "f" * 64),
    lambda value: value["provider_request"].__setitem__(
        "dependency_manifest_sha256", "f" * 64
    ),
])
def test_input_contract_closes_request_region_dependency_cell_and_limits(
    tmp_path, mutation,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    completed = _complete(store, handle)
    inputs = copy.deepcopy(completed["inputs"])
    mutation(inputs)
    _rehash_inputs(inputs)
    with pytest.raises(ValueError):
        seed_probe.validate_first_letters_discovery_inputs(inputs)


@pytest.mark.parametrize("candidates", [
    [{"candidate_id": "candidate-a", "cell_id": "cell-b",
      "ct_l0_coordinate": {"x": 12, "y": 34, "z": 56}, "score": 0.9}],
    [{"candidate_id": "candidate-a", "cell_id": "cell-a",
      "ct_l0_coordinate": {"x": 64, "y": 34, "z": 56}, "score": 0.9}],
    [{"candidate_id": "candidate-a", "cell_id": "cell-a",
      "ct_l0_coordinate": {"x": True, "y": 34, "z": 56}, "score": 0.9}],
    [
        {"candidate_id": "candidate-a", "cell_id": "cell-a",
         "ct_l0_coordinate": {"x": 12, "y": 12, "z": 12}, "score": 0.9},
        {"candidate_id": "candidate-b", "cell_id": "cell-a",
         "ct_l0_coordinate": {"x": 13, "y": 13, "z": 13}, "score": 0.8},
    ],
    [
        {"candidate_id": f"candidate-{index}", "cell_id": "cell-a",
         "ct_l0_coordinate": {"x": 2 + 20 * index, "y": 10, "z": 10},
         "score": 0.9 - index / 10}
        for index in range(3)
    ],
])
def test_provider_candidates_are_cell_region_separation_and_count_bounded(
    tmp_path, candidates,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    with pytest.raises(ValueError):
        _complete(store, handle, candidates=candidates)


def test_exact_json_rejects_literal_duplicate_keys_in_profile_response_and_request(
    tmp_path,
):
    profile_bytes = _profile_bytes()
    duplicate_profile = profile_bytes.replace(
        b'"mode":"shadow"', b'"mode":"shadow","mode":"shadow"', 1
    )
    with pytest.raises(ValueError, match="duplicate object key"):
        seed_probe.load_first_letters_discovery_profile_bytes(duplicate_profile)

    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    duplicate_response = _provider_response_bytes(handle).replace(
        b'"candidates":', b'"candidates":[],"candidates":', 1
    )
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with pytest.raises(ValueError, match="duplicate object key"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle, provider_response_bytes=duplicate_response,
        )

    completed = _complete(store, handle)
    artifact, _receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    retained = copy.deepcopy(completed["retained_files"])
    request_file = next(
        row for row in retained if row["role"] == "CANDIDATE_PROVIDER_REQUEST"
    )
    request_file["bytes"] = request_file["bytes"].replace(
        b'"request_id":"request-a"',
        b'"request_id":"request-a","request_id":"request-a"', 1,
    )
    with pytest.raises(ValueError, match="duplicate object key"):
        seed_probe.validate_first_letters_discovery_artifact(
            artifact, retained_files=retained
        )


def test_credential_policy_allows_clean_uri_but_rejects_userinfo_and_sensitive_query(
    tmp_path,
):
    allowed, profile_bytes, reservation = _store(
        tmp_path / "allowed",
        provider_id="https://provider.invalid/predict?mode=shadow",
    )
    handle = _claim_job(allowed, reservation)
    completed = _complete(allowed, handle)
    allowed.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )

    for index, provider_id in enumerate((
        "https://user:password@provider.invalid/predict",
        "https://provider.invalid/predict?access_token=abc123",
        "https://provider.invalid/predict?x-api-key=abc123",
    )):
        store, profile_bytes, reservation = _store(
            tmp_path / f"rejected-{index}", provider_id=provider_id,
        )
        handle = _claim_job(store, reservation)
        with pytest.raises(ValueError, match="credential"):
            _complete(store, handle)


def test_profile_object_validator_never_claims_raw_file_identity():
    profile = json.loads(_profile_bytes().decode("utf-8"))
    validated = seed_probe.validate_first_letters_discovery_profile(profile)
    assert "profile_file_sha256" not in validated


def test_completed_run_is_recovered_by_run_id_after_committed_response_loss(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with pytest.raises(RuntimeError, match="response lost|READBACK"):
        store._fixture_discovery_executor.measurements = (
            _measurement("candidate-a"),
        )
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
            failpoint="evidence.after_commit_before_response",
        )
    recovered = store.read_first_letters_discovery_evidence_run(handle.run_id)
    assert recovered["evidence_set_id"]
    assert recovered == store.read_first_letters_discovery_evidence_run(
        handle.run_id
    )


def test_candidate_terminals_are_not_direct_caller_boolean_fields():
    fields = {
        field.name
        for field in dataclasses.fields(
            seed_probe.FirstLettersDiscoveryCandidateMeasurement
        )
    }
    assert not fields.intersection({
        "ct_material_supported", "clearance_supported", "probe_measurable",
    })
    assert fields == {
        "candidate_id", "ct_read_evidence_bytes", "probe_evidence_bytes",
    }


def test_completion_boundary_accepts_no_caller_measurements():
    parameters = inspect.signature(
        FleetStore.complete_first_letters_discovery_evidence_run
    ).parameters
    assert set(parameters) == {
        "self", "run_handle", "provider_response_bytes", "failpoint",
    }


def test_unconfigured_executor_cannot_claim_or_persist_a_discovery_run(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    store._first_letters_discovery_executor = None
    with pytest.raises(ValueError, match="REGISTERED_DISCOVERY_EXECUTOR"):
        _claim_job(store, reservation)
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_runs"
        ).fetchone()[0] == 0


def test_executor_claim_not_deterministic_label_is_persisted_in_run_authority(
    tmp_path,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    with store.connect() as connection:
        row = connection.execute(
            "SELECT run_authority_json FROM "
            "first_letters_discovery_evidence_runs WHERE run_id=?",
            (handle.run_id,),
        ).fetchone()
        claim_row = connection.execute(
            "SELECT * FROM first_letters_discovery_executor_claims "
            "WHERE run_id=?",
            (handle.run_id,),
        ).fetchone()
    authority = json.loads(row["run_authority_json"])
    assert handle.worker_id == "registered-fixture-discovery-worker"
    assert authority["executor_claim"]["worker_id"] == handle.worker_id
    assert authority["executor_claim"]["claim_sha256"] == content_sha256({
        key: value for key, value in authority["executor_claim"].items()
        if key != "claim_sha256"
    })
    assert claim_row["state"] == "CLAIMED"
    assert claim_row["claim_attempt_number"] == 1
    assert json.loads(claim_row["claim_json"]) == authority["executor_claim"]


def test_caller_forged_measurement_json_cannot_enter_completion_boundary(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    forged = (_measurement("candidate-a", ct_nonzero_voxels=27),)
    with pytest.raises(TypeError):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
            measurements=forged,
        )
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0] == 0


def test_executor_instance_swapped_after_claim_cannot_complete_run(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    replacement = _FixtureDiscoveryExecutor()
    replacement.measurements = (_measurement("candidate-a"),)
    store._first_letters_discovery_executor = replacement

    with pytest.raises(ValueError, match="EXECUTOR_CLAIM_OWNERSHIP"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
        )

    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0] == 0


def test_unregistered_executor_worker_cannot_claim_discovery_job(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    store._first_letters_discovery_worker_id = "unregistered-worker"

    with pytest.raises(ValueError, match="REGISTERED_DISCOVERY_EXECUTOR_REQUIRED"):
        _claim_job(store, reservation)


@pytest.mark.parametrize(("registration", "message"), [
    (
        _fixture_executor_registration(
            worker_id="wrong-hash-worker", executor_sha256="0" * 64,
        ),
        "DISCOVERY_EXECUTOR_CODE_HASH_MISMATCH",
    ),
    (
        _fixture_executor_registration(
            worker_id="wrong-capability-worker", capabilities=[],
        ),
        "DISCOVERY_EXECUTOR_CAPABILITY_REQUIRED",
    ),
])
def test_wrong_registered_executor_hash_or_capability_cannot_claim(
    tmp_path, registration, message,
):
    store, profile_bytes, reservation = _store(tmp_path)
    store.register_first_letters_discovery_executor(registration)
    store._first_letters_discovery_worker_id = registration["worker_id"]

    with pytest.raises(ValueError, match=message):
        _claim_job(store, reservation)


@pytest.mark.parametrize(("change", "message"), [
    ({"executor_sha256": "0" * 64}, "DISCOVERY_EXECUTOR_CODE_HASH_MISMATCH"),
    ({"capabilities": []}, "DISCOVERY_EXECUTOR_CAPABILITY_REQUIRED"),
])
def test_completion_revalidates_executor_registry_hash_and_capability(
    tmp_path, change, message,
):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    registration = _fixture_executor_registration()
    registration.update(change)
    registration["registration_sha256"] = content_sha256({
        key: value for key, value in registration.items()
        if key != "registration_sha256"
    })
    with store.connect() as connection:
        connection.execute(
            """UPDATE first_letters_discovery_executor_registry
                  SET executor_sha256=?,capabilities_json=?,registration_json=?,
                      registration_sha256=? WHERE worker_id=?""",
            (
                registration["executor_sha256"],
                json.dumps(registration["capabilities"]),
                json.dumps(
                    registration, sort_keys=True, separators=(",", ":")
                ),
                registration["registration_sha256"],
                registration["worker_id"],
            ),
        )
    with pytest.raises(ValueError, match=message):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
        )
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM first_letters_discovery_evidence_sets"
        ).fetchone()[0] == 0


def test_stale_executor_claim_lease_cannot_complete_discovery_job(tmp_path):
    store, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with store.connect() as connection:
        connection.execute(
            "UPDATE first_letters_discovery_executor_claims "
            "SET lease_expires_at='2000-01-01T00:00:00Z' WHERE run_id=?",
            (handle.run_id,),
        )

    with pytest.raises(ValueError, match="DISCOVERY_EXECUTOR_CLAIM_STALE"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
        )


def test_same_code_nonclaiming_executor_instance_cannot_complete_job(tmp_path):
    claiming, profile_bytes, reservation = _store(tmp_path)
    handle = _claim_job(claiming, reservation)
    claiming.start_first_letters_discovery_evidence_run(run_handle=handle)
    nonclaiming = FleetStore(
        claiming.path,
        first_letters_discovery_executor=_FixtureDiscoveryExecutor(),
    )
    nonclaiming._first_letters_discovery_worker_id = (
        "registered-fixture-discovery-worker"
    )
    nonclaiming._first_letters_discovery_executor_id = (
        "fixture-discovery-executor-v1"
    )

    with pytest.raises(ValueError, match="EXECUTOR_CLAIM_OWNERSHIP"):
        nonclaiming.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_provider_response_bytes(handle),
        )


def test_production_store_factory_claims_and_completes_no_winner_path(
    tmp_path, monkeypatch,
):
    fixture, profile_bytes, reservation = _store(tmp_path)
    profile_path = tmp_path / "first-letters-discovery-profile.json"
    profile_path.write_bytes(profile_bytes)
    monkeypatch.setenv(
        "HELENA_FIRST_LETTERS_DISCOVERY_PROFILE_PATH", str(profile_path)
    )
    production = open_fleet_store(fixture.path)
    production.initialize()
    handle = _claim_job(production, reservation)
    production.start_first_letters_discovery_evidence_run(run_handle=handle)
    completed = production.complete_first_letters_discovery_evidence_run(
        run_handle=handle,
        provider_response_bytes=_provider_response_bytes(handle, candidates=[]),
    )
    artifact, _receipt = (
        production.build_first_letters_discovery_artifact_and_receipt(
            completed["evidence_set_id"]
        )
    )
    assert artifact["selection_outcome"] == (
        "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER"
    )
    assert artifact["selected_candidate_id"] is None


def test_production_executor_samples_ct_but_cannot_self_certify_probe(tmp_path):
    class CtSampler:
        def sample(self, ct_uri, coordinate_xyz, *, level, radius_l0_voxels):
            assert ct_uri == "https://provider.invalid/ct"
            assert coordinate_xyz == {"x": 12, "y": 34, "z": 56}
            assert (level, radius_l0_voxels) == (5, 1)
            return {"voxel_count": 27, "nonzero_voxel_count": 1}

    store, profile_bytes, reservation = _store(tmp_path)
    executor = ProductionFirstLettersDiscoveryExecutor(ct_sampler=CtSampler())
    worker_id = "production-test-worker"
    store.register_first_letters_discovery_executor(
        production_discovery_executor_registration(
            executor, worker_id=worker_id,
        )
    )
    store._first_letters_discovery_executor = executor
    store._first_letters_discovery_worker_id = worker_id
    store._first_letters_discovery_executor_id = (
        PRODUCTION_DISCOVERY_EXECUTOR_ID
    )
    handle = _claim_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    completed = store.complete_first_letters_discovery_evidence_run(
        run_handle=handle,
        provider_response_bytes=_provider_response_bytes(handle),
    )
    artifact, _receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    assert artifact["candidates"][0]["ct_terminal"]["state"] == "CT_SUPPORTED"
    assert artifact["candidates"][0]["probe_evidence"]["state"] == "UNMEASURABLE"
    assert artifact["selected_candidate_id"] is None
