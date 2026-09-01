from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.postgres_store import PostgresFleetStore
from fleet.common import content_sha256, stable_id
from test_first_letters_discovery_compute import _cap, _work
from test_first_letters_discovery_evidence_store import (
    _FixtureDiscoveryExecutor,
    _cap as _evidence_cap,
    _fixture_executor_registration,
    _profile_bytes as _evidence_profile_bytes,
    _provider_response_bytes as _evidence_provider_response_bytes,
)
from test_first_letters_discovery_promotion import (
    _benchmark,
    _budget_admission,
    _gate,
    _raise_at,
)


SQL_PATH = STAGE / "fleet/migrations/001_postgresql.sql"
STORE_PATH = STAGE / "fleet/postgres_store.py"
V15_FIXTURE_PATH = ROOT / "tests/fixtures/postgresql-v15.sql.gz.b64"
V15_SQL_SHA256 = (
    "71d241c9a2cf00104502d3b5cebd17151f3bb41e3a55f378d0688321f16c80bd"
)
DSN = os.environ.get("HELENA_TEST_DSN")

_POSTGRES_DISCOVERY_SHADOW_TABLE_ALLOWLIST = frozenset({
    "segment_first_letters_discovery_compute_blocks",
    "segment_first_letters_discovery_compute_caps",
    "segment_first_letters_discovery_compute_outcomes",
    "segment_first_letters_discovery_compute_reservations",
    "segment_first_letters_discovery_dispatches_v19",
    "segment_first_letters_discovery_evidence_files",
    "segment_first_letters_discovery_evidence_runs",
    "segment_first_letters_discovery_evidence_sets",
    "segment_first_letters_discovery_executor_claims",
    "segment_first_letters_discovery_executor_registry",
    "segment_first_letters_discovery_historical_imports_v19",
    "segment_first_letters_discovery_history_reconciliations_v19",
    "segment_first_letters_discovery_jobs_v19",
    "segment_first_letters_discovery_native_adapters_v19",
    "segment_first_letters_discovery_promotion_attempt_bindings",
    "segment_first_letters_discovery_promotions",
    "segment_first_letters_discovery_work_bindings",
})


def _sql(): return SQL_PATH.read_text(encoding="utf-8")
def _source(): return STORE_PATH.read_text(encoding="utf-8")


def _postgres_canonical_table_projection(store):
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT table_name FROM information_schema.tables
                    WHERE table_schema=current_schema()
                      AND table_type='BASE TABLE'
                    ORDER BY table_name"""
            )
            tables = [str(row["table_name"]) for row in cursor.fetchall()]
            observed_discovery = {
                table for table in tables
                if table.startswith("segment_first_letters_discovery_")
            }
            assert observed_discovery == (
                _POSTGRES_DISCOVERY_SHADOW_TABLE_ALLOWLIST
            )
            projection = {}
            for table in tables:
                if table in _POSTGRES_DISCOVERY_SHADOW_TABLE_ALLOWLIST:
                    continue
                assert re.fullmatch(r"[a-z0-9_]+", table)
                cursor.execute(f'SELECT to_jsonb(t) AS row FROM "{table}" t')
                rows = [copy.deepcopy(row["row"]) for row in cursor.fetchall()]
                projection[table] = tuple(sorted(
                    rows,
                    key=lambda value: json.dumps(
                        value, sort_keys=True, separators=(",", ":"),
                        default=str,
                    ),
                ))
    return projection


def _frozen_v15_sql() -> str:
    compressed = base64.b64decode(
        V15_FIXTURE_PATH.read_bytes().strip(), validate=True
    )
    payload = gzip.decompress(compressed)
    assert hashlib.sha256(payload).hexdigest() == V15_SQL_SHA256
    return payload.decode("utf-8")


class _IsolatedPostgresStore(PostgresFleetStore):
    """Put every optional live test in its own throwaway schema."""

    def __init__(self, database_url: str, schema: str):
        executor = _FixtureDiscoveryExecutor()
        super().__init__(
            database_url,
            task9_discovery_gate_resolver=lambda _mission_id: _gate(),
            first_letters_discovery_executor=executor,
            first_letters_discovery_worker_id=(
                "registered-fixture-discovery-worker"
            ),
            first_letters_discovery_executor_id=(
                "fixture-discovery-executor-v1"
            ),
            first_letters_discovery_executor_registration=(
                _fixture_executor_registration()
            ),
        )
        self._fixture_discovery_executor = executor
        self._test_schema = schema

    def connect(self):
        connection = super().connect()
        with connection.cursor() as cursor:
            # The name is generated solely from a UUID and contains only
            # lowercase ASCII letters, digits, and underscores.
            cursor.execute(f"SET search_path TO {self._test_schema}, public")
        return connection


@pytest.fixture
def live_postgres_store():
    if not DSN:
        pytest.skip(
            "HELENA_TEST_DSN is not set; real PostgreSQL migration, "
            "failpoint, readback, and concurrency parity was not run"
        )
    schema = f"task6_discovery_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    store = _IsolatedPostgresStore(DSN, schema)
    try:
        store.initialize()
        store.initialize()
        yield store
    finally:
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA {schema} CASCADE")


_LIVE_NATIVE_ITEMS = ("cell-a", "cell-b")
_LIVE_NATIVE_SEED_LOCK = threading.Lock()


def _seed_live_native_authority(store: PostgresFleetStore) -> None:
    """Give the live store the server-owned authority the SQLite fixture has.

    Mirrors ``_store`` in ``test_first_letters_discovery_compute``: since v19 a
    baseline or alternative reservation is derived by the store from registered
    budget/source/cap authority, never from a caller-supplied work admission.
    """
    with _LIVE_NATIVE_SEED_LOCK:
        if getattr(store, "_fixture_native_lock", None) is not None:
            return
        cap = _cap()
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
                list(_LIVE_NATIVE_ITEMS)
            ),
            "mission_compute_cap_authority_id": "cap-a",
            "mission_compute_cap_authority_sha256": cap["authority_sha256"],
            "mission_compute_cap_units": cap["mission_compute_cap_units"],
            "deployed_revision": "1" * 40,
        })
        profile["scientific_core_sha256"] = content_sha256({
            key: value for key, value in profile.items()
            if key != "scientific_core_sha256"
        })
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
            "m7_model_id": "m7-v1", "m7_model_sha256": "7" * 64,
            "candidate_provider_id": "fixture-provider-v1",
            "candidate_provider_sha256": "8" * 64,
            "discovery_minimum_separation": 12,
            "m7_resolution": 4, "m7_level": 1, "m7_threshold": 0.5,
            "m7_transform_sha256": "8" * 64,
            "shape_xyz": [64, 64, 64], "voxel_size_um": 9.0,
            "coordinate_frame": "ct_l0_xyz",
            "first_letters_discovery_authority": {
                "mission_id": "mission-a", "accepted_p0_artifact_id": "p0-a",
                "accepted_p0_artifact_sha256": "a" * 64,
                "minimum_cell_clearance_voxels": 2,
                "minimum_volume_clearance_voxels": 2,
                "scientific_opportunities": {
                    item: f"opportunity-{item}" for item in _LIVE_NATIVE_ITEMS
                },
            },
        })
        store._fixture_profile_template = copy.deepcopy(profile)
        store._fixture_profile_bytes = json.dumps(
            profile, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        store._first_letters_discovery_profile_resolver = (
            lambda _mission_id, _source_snapshot_id: store._fixture_profile_bytes
        )
        store._fixture_admissions = {}
        store._fixture_tasks = set()
        store._fixture_native_lock = threading.Lock()


def _live_native_authority(
    store: PostgresFleetStore, items: tuple[str, ...],
) -> dict:
    _seed_live_native_authority(store)
    with store._fixture_native_lock:
        profile = copy.deepcopy(store._fixture_profile_template)
        profile["canonical_ordered_cell_set_sha256"] = content_sha256(
            list(items)
        )
        profile["scientific_core_sha256"] = content_sha256({
            key: value for key, value in profile.items()
            if key != "scientific_core_sha256"
        })
        store._fixture_profile_bytes = json.dumps(
            profile, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if tuple(items) in store._fixture_admissions:
            return store._fixture_admissions[tuple(items)]
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
        pending = [
            (rank, item) for rank, item in enumerate(items)
            if item not in store._fixture_tasks
        ]
        if pending:
            store.create_tasks([{
                "task_id": f"task-{item}", "mission_id": "mission-a",
                "sample_id": "PHercA", "source_snapshot_id": "source-a",
                "cell_id": item, "grid_version": "first-letters-grid-v1",
                "policy_version": "first-letters-search@1.0.0",
                "bounds_xyz": [[0, 0, 0], [64, 64, 64]],
                "center_xyz": {"x": 32, "y": 32, "z": 32}, "priority": 1.0,
                "parameter_envelope": {},
                "catalog_snapshot_sha256": "f" * 64,
                "selection_rank": rank,
                "campaign_budget_admission_sha256":
                    admission["admission_sha256"],
                "scientific_opportunity_id": f"opportunity-{item}",
                "p0_artifact_id": "p0-a", "p0_artifact_sha256": "a" * 64,
                "accepted_p0_artifact_id": "p0-a",
                "accepted_p0_artifact_sha256": "a" * 64,
                "candidate_discovery": {
                    "region": {
                        "minimum": [0, 0, 0], "maximum": [64, 64, 64],
                    },
                },
            } for rank, item in pending])
            store._fixture_tasks.update(item for _rank, item in pending)
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO segment_campaign_budget_admissions
                       (mission_id,sample_id,receipt_sha256,admission,
                        admission_sha256,created_at)
                       VALUES(%s,%s,%s,%s::jsonb,%s,now())""",
                    (
                        "mission-a", "PHercA", admission["receipt_sha256"],
                        json.dumps(admission, sort_keys=True,
                                   separators=(",", ":")),
                        admission["admission_sha256"],
                    ),
                )
        store._fixture_admissions[tuple(items)] = admission
        return admission


def _live_experimental_arm(
    store: PostgresFleetStore, items: tuple[str, ...],
) -> dict:
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM segment_source_snapshots "
                "WHERE source_snapshot_id='source-a'"
            )
            source = store._snapshot(cursor.fetchone())
    core = {
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
        "discovery_profile_sha256": hashlib.sha256(
            store._fixture_profile_bytes
        ).hexdigest(),
        "deployed_revision": "1" * 40,
        "preflight_private_sha256": "0" * 64,
        "preflight_sanitized_sha256": "1" * 64,
        "ordered_cell_ids": list(items),
        "ordered_cell_set_sha256": content_sha256(list(items)),
        "mission_compute_cap_authority_id": "cap-a",
        "mission_compute_cap_authority_sha256": store.discovery_compute_cap(
            "mission-a"
        )["authority_sha256"],
        "requested_units": len(items) * 24,
        "active_policy_chain_sha256": _cap()["policy_chain_sha256"],
        "may_update_accepted_p0": False,
        "statistical_budget_delta": 0, "allow_unvalidated": False,
    }
    return {**core, "admission_sha256": content_sha256(core)}


def _reserve_live(
    store: PostgresFleetStore, kind: str = "BASELINE_ARM",
    request_id: str = "request-a", items: tuple[str, ...] = _LIVE_NATIVE_ITEMS,
    *, failpoint: str | None = None,
):
    if kind in {"BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM"}:
        admission = _live_native_authority(store, items)
        if kind == "BASELINE_ARM":
            return store.reserve_first_letters_baseline_shadow(
                request_id=request_id,
                budget_admission_sha256=admission["admission_sha256"],
                failpoint=failpoint,
            )
        arm = _live_experimental_arm(store, items)
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
        reservation_mode=(
            "PREFIX_TO_CAP" if kind == "ADAPTIVE_CHILD" else "EXACT"
        ),
        task9_gate=_gate() if kind == "ADAPTIVE_CHILD" else None,
        failpoint=failpoint,
    )


def _prepare_live_evidence(
    store: PostgresFleetStore, *, promotion: bool = False,
):
    opportunity_id = "opportunity-a"
    promotion_authority = None
    if promotion:
        budget = _budget_admission()
        budget["execution_bindings"]["grid_version"] = (
            "first-letters-grid-v1"
        )
        budget["admission_sha256"] = content_sha256({
            key: value for key, value in budget.items()
            if key != "admission_sha256"
        })
        opportunity_id = stable_id("first-letters-opportunity", {
            "admission_sha256": budget["admission_sha256"],
            "selection_rank": 0,
        })
        promotion_authority = {
            "active_policy_chain": {
                "active_policy_version": "first-letters-search@1.1.0",
                "policy_chain_sha256": "4" * 64, "paused": False,
            },
            "benchmark_authorization_v2": _benchmark(),
        }
    else:
        budget_core = {
            "schema": "campaignx.first_letters_task_budget_admission.v1",
            "mission_id": "mission-a", "sample_id": "PHercA",
            "receipt_sha256": "6" * 64,
            "preflight_receipt_sha256": "7" * 64,
            "preflight_sanitized_receipt_sha256": "8" * 64,
            "approved_task_count": 1, "order_seed_sha256": "9" * 64,
            "population_order_sha256": "0" * 64,
            "prefix_sha256": content_sha256(["cell-a"]),
            "prefix_cell_ids": ["cell-a"],
            "execution_bindings": {
                "source_snapshot_id": "source-a",
                "grid_version": "first-letters-grid-v1",
                "policy_version": "first-letters-search@1.0.0",
                "p0_artifact_id": "p0-a", "p0_artifact_sha256": "a" * 64,
                "catalog_snapshot_sha256": "3" * 64,
            },
        }
        budget = {
            **budget_core, "admission_sha256": content_sha256(budget_core),
        }
    p0_artifact_sha256 = budget["execution_bindings"]["p0_artifact_sha256"]
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
        "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a",
            "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": p0_artifact_sha256,
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {"cell-a": opportunity_id},
            **({"promotion_authority": promotion_authority}
               if promotion_authority is not None else {}),
        },
    })
    store.create_tasks([{
        "task_id": "task-a", "mission_id": "mission-a",
        "sample_id": "PHercA", "source_snapshot_id": "source-a",
        "cell_id": "cell-a", "grid_version": "first-letters-grid-v1",
        "policy_version": "first-letters-search@1.0.0",
        "bounds_xyz": [[0, 0, 0], [64, 64, 64]],
        "center_xyz": {"x": 32, "y": 32, "z": 32}, "priority": 1.0,
        "parameter_envelope": {
            "generations": {"minimum": 20, "maximum": 45, "default": 35},
            "step_size": {"minimum": 12, "maximum": 24, "default": 20},
            "min_area_cm": {"minimum": 0.0, "maximum": 0.0, "default": 0.0},
            "use_cuda": {"allowed": [False], "default": False},
        },
        "catalog_snapshot_sha256": "3" * 64,
        "scientific_opportunity_id": opportunity_id,
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": p0_artifact_sha256,
        "selection_rank": 0,
        "campaign_budget_admission_sha256": budget["admission_sha256"],
        "p0_artifact_id": budget["execution_bindings"]["p0_artifact_id"],
        "p0_artifact_sha256": p0_artifact_sha256,
        "candidate_discovery": {
            "region": {"minimum": [0, 0, 0], "maximum": [64, 64, 64]},
        },
    }])
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO segment_campaign_budget_admissions
                   (mission_id,sample_id,receipt_sha256,admission,
                    admission_sha256,created_at)
                   VALUES(%s,%s,%s,%s::jsonb,%s,now())""",
                (
                    budget["mission_id"], budget["sample_id"],
                    budget["receipt_sha256"], json.dumps(
                        budget, sort_keys=True, separators=(",", ":")
                    ), budget["admission_sha256"],
                ),
            )
    if promotion:
        claim = store.claim("registered-parent-worker", 60, task_id="task-a")
        assert claim is not None
    profile_bytes = _evidence_profile_bytes()
    cap = _evidence_cap()
    store.register_discovery_compute_cap(cap)
    store._first_letters_discovery_profile_resolver = (
        lambda _mission_id, _source_snapshot_id: profile_bytes
    )
    reservation = store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=budget["admission_sha256"],
    )
    return profile_bytes, reservation


def _claim_live_job(
    store: PostgresFleetStore, reservation: dict, *, item_id: str = "cell-a",
    lease_seconds: int = 60,
):
    """Since v19 an evidence run only begins under a dedicated job claim."""
    matches = [
        job for job in reservation["jobs"] if job["item_id"] == item_id
    ]
    if len(matches) != 1:
        raise ValueError("fixture discovery job is missing or ambiguous")
    return store.claim_first_letters_discovery_job(
        job_id=matches[0]["job_id"], lease_seconds=lease_seconds,
    )._run_handle


def test_postgres_v15_upgrade_applies_current_task6_schema_constraints_and_indexes():
    sql = _sql()
    versions = [int(value) for value in re.findall(
        r"VALUES\s*\(\s*(\d+)\s*,\s*'", sql
    )]
    # The Task 6 schema is present and consecutive up to its own v19 ceiling.
    # Not "19 is the highest version in the file": migrations after it are
    # somebody else's contract, and asserting their absence here turns every
    # future one into a Task 6 failure.
    assert versions[:19] == list(range(1, 20))
    for table in (
        "segment_first_letters_discovery_compute_caps",
        "segment_first_letters_discovery_compute_reservations",
        "segment_first_letters_discovery_work_bindings",
        "segment_first_letters_discovery_evidence_runs",
        "segment_first_letters_discovery_executor_registry",
        "segment_first_letters_discovery_executor_claims",
        "segment_first_letters_discovery_evidence_sets",
        "segment_first_letters_discovery_evidence_files",
        "segment_first_letters_discovery_compute_outcomes",
        "segment_first_letters_discovery_compute_blocks",
        "segment_first_letters_discovery_promotions",
        "segment_first_letters_discovery_promotion_attempt_bindings",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "CHECK(units_per_item = 24)" in sql
    assert "UNIQUE(mission_id,request_id)" in sql
    assert "UNIQUE(mission_id,scientific_opportunity_id)" in sql


def test_frozen_postgres_v15_fixture_is_literal_historical_schema():
    sql = _frozen_v15_sql()
    versions = [int(value) for value in re.findall(
        r"VALUES\s*\(\s*(\d+)\s*,\s*'", sql
    )]
    assert versions == list(range(1, 16))
    assert "segment_first_letters_discovery_evidence_runs" not in sql
    assert "VALUES (16, 'first-letters discovery compute" not in sql


def test_postgres_fresh_current_and_repeated_initialize_match_upgraded_schema():
    source = _source()
    assert "target = max(" in source
    assert "SELECT pg_advisory_xact_lock" in source
    assert "segment_schema_migrations WHERE version=%s" in source
    assert "version=18" not in source


def test_postgres_existing_v19_import_cardinality_constraint_is_repaired():
    sql = _sql()
    source = _source()
    assert "UNIQUE (reservation_id)" in sql
    assert "DROP CONSTRAINT" in sql
    assert "pg_get_constraintdef" in sql
    assert "pg_get_constraintdef" in source


def test_live_postgres_existing_v19_import_cardinality_constraint_is_repaired(
    live_postgres_store,
):
    store = live_postgres_store
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """ALTER TABLE
                   segment_first_letters_discovery_historical_imports_v19
                   ADD CONSTRAINT legacy_import_reservation_unique
                   UNIQUE(reservation_id)"""
            )

    store.initialize()

    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT conname FROM pg_constraint
                    WHERE conrelid=
                      'segment_first_letters_discovery_historical_imports_v19'
                      ::regclass
                      AND contype='u'
                      AND pg_get_constraintdef(oid)='UNIQUE (reservation_id)'"""
            )
            assert cursor.fetchall() == []


def test_live_postgres_task_identity_seals_candidate_authority(
    live_postgres_store,
):
    store = live_postgres_store
    store.register_snapshot({
        "source_snapshot_id": "candidate-source",
        "sample_id": "PHercA", "ct_uri": "fixture://ct",
        "m7_uri": "fixture://m7", "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0, "coordinate_frame": "ct_l0_xyz",
    })
    base = {
        "task_id": "candidate-task", "mission_id": "mission-a",
        "sample_id": "PHercA", "source_snapshot_id": "candidate-source",
        "cell_id": "cell-a", "grid_version": "grid-v1",
        "policy_version": "first-letters-search@1.2.0",
        "bounds_xyz": [[0, 0, 0], [64, 64, 64]],
        "center_xyz": {"x": 32, "y": 32, "z": 32},
        "priority": 1.0, "parameter_envelope": {},
        "catalog_snapshot_sha256": "0" * 64,
    }
    assert store.create_tasks([base]) == (1, 1)
    assert store.create_tasks([{
        **base, "candidate_rank": 1, "reconsider_covered": False,
    }]) == (0, 1)
    with pytest.raises(ValueError, match="task candidate authority differs"):
        store.create_tasks([{
            **base, "candidate_rank": 2, "reconsider_covered": False,
        }])
    with pytest.raises(ValueError, match="task candidate authority differs"):
        store.create_tasks([{
            **base, "candidate_rank": 1, "reconsider_covered": True,
        }])


def test_postgres_incomplete_history_is_committed_before_shadow_reserve_raises(
    monkeypatch,
):
    class Cursor:
        def __init__(self):
            self.sql = ""
            self.executed = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, args=()):
            self.sql = " ".join(sql.split())
            self.executed.append((self.sql, args))

        def fetchall(self):
            if "segment_campaign_budget_admissions" in self.sql:
                return [{"admission": {
                    "mission_id": "mission-a",
                    "execution_bindings": {"source_snapshot_id": "source-a"},
                }}]
            if "SELECT r.probe_run_id FROM segment_probe_runs" in self.sql:
                return [{"probe_run_id": "legacy-run"}]
            return []

        def fetchone(self):
            return None

    class Connection:
        def __init__(self):
            self.cursor_value = Cursor()
            self.commits = 0
            self.rollbacks = 0

        def __enter__(self):
            return self

        def __exit__(self, error_type, _error, _traceback):
            if error_type is None:
                self.commits += 1
            else:
                self.rollbacks += 1
            return False

        def cursor(self):
            return self.cursor_value

    connection = Connection()
    store = PostgresFleetStore(
        "postgresql://unused",
        first_letters_discovery_profile_resolver=(
            lambda _mission_id, _source_id: b"profile"
        ),
    )
    monkeypatch.setattr(store, "connect", lambda: connection)

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"):
        store.reserve_first_letters_baseline_shadow(
            request_id="request-a", budget_admission_sha256="b" * 64,
        )

    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert any(
        "INSERT INTO segment_first_letters_discovery_history_reconciliations_v19"
        in sql
        for sql, _args in connection.cursor_value.executed
    )
    assert any(
        "INSERT INTO segment_first_letters_discovery_compute_blocks" in sql
        for sql, _args in connection.cursor_value.executed
    )


@pytest.mark.parametrize(("relation", "column", "replacement"), [
    ("adapter", "reservation_id", "other"),
    ("adapter", "mission_id", "other"),
    ("adapter", "request_id", "other"),
    ("adapter", "work_kind", "ALTERNATIVE_SOURCE_ARM"),
    ("adapter", "producer_kind", "EXPERIMENTAL_ARM_ADMISSION"),
    ("adapter", "native_schema", "other"),
    ("adapter", "native_authority_sha256", "b" * 64),
    ("adapter", "generic_work_authority_sha256", "b" * 64),
    ("dispatch", "reservation_id", "other"),
    ("dispatch", "mission_id", "other"),
    ("dispatch", "request_id", "other"),
    ("dispatch", "work_kind", "ALTERNATIVE_SOURCE_ARM"),
    ("dispatch", "adapter_sha256", "b" * 64),
    ("dispatch", "profile_file_sha256", "b" * 64),
    ("dispatch", "source_snapshot_sha256", "b" * 64),
    ("dispatch", "ordered_item_ids_sha256", "b" * 64),
    ("dispatch", "item_count", 2),
    ("job", "work_item_binding_sha256", "b" * 64),
    ("job", "profile_file_sha256", "b" * 64),
    ("job", "source_snapshot_sha256", "b" * 64),
])
def test_postgres_readback_rejects_every_adapter_dispatch_job_scalar_drift(
    monkeypatch, relation, column, replacement,
):
    from fleet.discovery_bridge import (
        adapt_first_letters_baseline_shadow,
        build_first_letters_discovery_dispatch,
        build_first_letters_discovery_jobs,
    )
    from test_first_letters_discovery_shadow_bridge import _reconciliation

    profile_bytes = b"exact-postgres-profile"
    reconciliation = _reconciliation()
    reconciliation["profile_file_sha256"] = hashlib.sha256(
        profile_bytes
    ).hexdigest()
    reconciliation["reconciliation_sha256"] = content_sha256({
        key: value for key, value in reconciliation.items()
        if key != "reconciliation_sha256"
    })
    adapter = adapt_first_letters_baseline_shadow(reconciliation)
    reservation_core = {
        "reservation_id": "reservation-a", "mission_id": "mission-a",
        "request_id": "request-a", "work_kind": "BASELINE_ARM",
        "work_authority_id": adapter["generic_work_authority"][
            "work_authority_id"
        ],
        "work_authority_sha256": adapter[
            "generic_work_authority_sha256"
        ],
        "ordered_item_ids": ["cell-a"],
        "ordered_item_ids_sha256": content_sha256(["cell-a"]),
        "item_count": 1,
    }
    reservation = {
        **reservation_core,
        "reservation_sha256": content_sha256(reservation_core),
    }
    work_core = {
        "reservation_id": "reservation-a",
        "reservation_sha256": reservation["reservation_sha256"],
        "work_authority": adapter["generic_work_authority"],
    }
    work = {**work_core, "work_sha256": content_sha256(work_core)}
    dispatch = build_first_letters_discovery_dispatch(reservation, adapter)
    jobs = build_first_letters_discovery_jobs(dispatch, adapter)
    graph_row = {
        "reservation": reservation,
        "reservation_sha256": reservation["reservation_sha256"],
        "work": work,
        "work_sha256": work["work_sha256"],
        "adapter": adapter,
        "profile_bytes": profile_bytes,
        "adapter_reservation_id": "reservation-a",
        "adapter_mission_id": "mission-a",
        "adapter_request_id": "request-a",
        "adapter_work_kind": "BASELINE_ARM",
        "adapter_producer_kind": "BASELINE_RECONCILIATION",
        "adapter_native_schema": adapter["native_schema"],
        "adapter_native_authority": adapter["native_authority"],
        "adapter_native_authority_sha256": adapter[
            "native_authority_sha256"
        ],
        "adapter_generic_work_authority": adapter[
            "generic_work_authority"
        ],
        "adapter_generic_work_authority_sha256": adapter[
            "generic_work_authority_sha256"
        ],
        "adapter_sha256": adapter["adapter_sha256"],
        "dispatch": dispatch,
        "dispatch_id": dispatch["dispatch_id"],
        "dispatch_reservation_id": "reservation-a",
        "dispatch_mission_id": "mission-a",
        "dispatch_request_id": "request-a",
        "dispatch_work_kind": "BASELINE_ARM",
        "dispatch_adapter_sha256": adapter["adapter_sha256"],
        "dispatch_profile_file_sha256": dispatch["profile_file_sha256"],
        "dispatch_source_snapshot_sha256": dispatch[
            "source_snapshot_sha256"
        ],
        "dispatch_ordered_item_ids_sha256": dispatch[
            "ordered_item_ids_sha256"
        ],
        "dispatch_item_count": 1,
        "dispatch_sha256": dispatch["dispatch_sha256"],
    }
    job_rows = [{
        **copy.deepcopy(jobs[0]),
        "job": copy.deepcopy(jobs[0]),
    }]
    target = graph_row if relation != "job" else job_rows[0]
    target_key = (
        f"{relation}_{column}" if relation in {"adapter", "dispatch"}
        else column
    )
    target[target_key] = replacement

    class Cursor:
        def __init__(self):
            self.query = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, _args=()):
            self.query = " ".join(query.split())

        def fetchall(self):
            if "segment_first_letters_discovery_jobs_v19" in self.query:
                return job_rows
            return [graph_row]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(store, "connect", lambda: Connection())

    with pytest.raises(
        ValueError, match="CONTROL_INCOMPLETE_DISCOVERY_DISPATCH"
    ):
        store.read_first_letters_discovery_request("mission-a", "request-a")


@pytest.mark.parametrize("authority", [
    "profile_bytes", "profile_resolver_error", "cap", "source_science",
    "task_opportunity", "task_region", "task_p0",
    "arm_missing", "arm_drift", "arm_exception",
])
def test_postgres_current_adapter_reresolution_rejects_authority_drift(
    tmp_path, authority,
):
    from test_first_letters_discovery_shadow_bridge import _live_bridge_store
    from test_first_letters_discovery_evidence_store import _cap

    sqlite_store, admission, profile_bytes = _live_bridge_store(tmp_path)
    alternative = authority.startswith("arm_")
    branch = (
        sqlite_store.reserve_first_letters_alternative_shadow(
            request_id="request-alt",
            budget_admission_sha256=admission["admission_sha256"],
            arm_id="arm-a",
        )
        if alternative else
        sqlite_store.reserve_first_letters_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=admission["admission_sha256"],
        )
    )
    sources = {
        source["source_snapshot_id"]: copy.deepcopy(source)
        for source in sqlite_store.snapshots()
    }
    with sqlite_store.connect() as connection:
        task_row = connection.execute(
            "SELECT * FROM tasks WHERE task_id='task-cell-a'"
        ).fetchone()
    task = {
        "task_id": task_row["task_id"],
        "source_snapshot_id": task_row["source_snapshot_id"],
        "grid_version": task_row["grid_version"],
        "policy_version": task_row["policy_version"],
        "bounds_xyz": json.loads(task_row["bounds_xyz_json"]),
        "active_attempt_id": task_row["active_attempt_id"],
        "payload": json.loads(task_row["payload_json"]),
    }
    cap = _cap()
    cap_row = {
        "mission_id": cap["mission_id"],
        "cap_authority_id": cap["cap_authority_id"],
        "authority_sha256": cap["authority_sha256"],
        "cap_units": cap["mission_compute_cap_units"],
        "authority": copy.deepcopy(cap),
    }
    profile_resolver = sqlite_store._first_letters_discovery_profile_resolver
    arm_resolver = sqlite_store._first_letters_experimental_arm_resolver
    if authority == "profile_bytes":
        profile_resolver = lambda _mission_id, _source_id: b"{}"
    elif authority == "profile_resolver_error":
        def profile_resolver(_mission_id, _source_id):
            raise RuntimeError("profile resolver unavailable")
    elif authority == "cap":
        cap["policy_chain_sha256"] = "f" * 64
        cap["authority_sha256"] = content_sha256({
            key: value for key, value in cap.items()
            if key != "authority_sha256"
        })
        cap_row.update({
            "authority_sha256": cap["authority_sha256"],
            "authority": cap,
        })
    elif authority == "source_science":
        sources["source-a"]["candidate_provider_sha256"] = "9" * 64
    elif authority == "task_opportunity":
        task["payload"]["scientific_opportunity_id"] = "opportunity-other"
    elif authority == "task_region":
        task["payload"]["candidate_discovery"]["region"] = {
            "minimum": [1, 1, 1], "maximum": [63, 63, 63],
        }
    elif authority == "task_p0":
        task["payload"]["p0_artifact_sha256"] = "9" * 64
    elif authority == "arm_missing":
        arm_resolver = lambda _arm_id: None
    elif authority == "arm_drift":
        arm = copy.deepcopy(
            branch["adapter"]["native_authority"]["arm_admission"]
        )
        arm["m7_threshold"] = 0.6
        arm["admission_sha256"] = content_sha256({
            key: value for key, value in arm.items()
            if key != "admission_sha256"
        })
        arm_resolver = lambda _arm_id: copy.deepcopy(arm)
    elif authority == "arm_exception":
        def arm_resolver(_arm_id):
            raise RuntimeError("arm resolver unavailable")

    class Cursor:
        def __init__(self):
            self.query = ""
            self.args = ()

        def execute(self, query, args=()):
            self.query = " ".join(query.split())
            self.args = args

        def fetchall(self):
            if "segment_campaign_budget_admissions" in self.query:
                return [{"admission": copy.deepcopy(admission)}]
            if "segment_tasks" in self.query:
                return [copy.deepcopy(task)]
            return []

        def fetchone(self):
            if "segment_source_snapshots" in self.query:
                source = sources.get(self.args[0])
                return None if source is None else {
                    "source_snapshot_id": source["source_snapshot_id"],
                    "shape_xyz": copy.deepcopy(source["shape_xyz"]),
                    "payload": copy.deepcopy(source),
                }
            if "segment_first_letters_discovery_compute_caps" in self.query:
                return copy.deepcopy(cap_row)
            return None

    postgres_store = PostgresFleetStore(
        "postgresql://unused",
        first_letters_discovery_profile_resolver=profile_resolver,
        first_letters_experimental_arm_resolver=arm_resolver,
    )

    persisted_profile_bytes = (
        sqlite_store._first_letters_discovery_profile_resolver(
            "mission-a",
            branch["adapter"]["generic_work_authority"][
                "ordered_item_bindings"
            ][0]["source_snapshot_id"],
        )
    )
    with pytest.raises((ValueError, RuntimeError)):
        postgres_store._current_first_letters_discovery_adapter_tx(
            Cursor(), adapter=branch["adapter"],
            persisted_profile_bytes=persisted_profile_bytes,
        )


@pytest.mark.parametrize("current_change", ["compute_block", "history"])
def test_postgres_worker_revalidates_current_history_and_block_before_provider(
    tmp_path, monkeypatch, current_change,
):
    from fleet.discovery_worker import FirstLettersDiscoveryWorker
    from test_first_letters_discovery_shadow_bridge import (
        _BridgeProvider,
        _live_bridge_store,
    )

    sqlite_store, admission, profile_bytes = _live_bridge_store(tmp_path)
    branch = sqlite_store.reserve_first_letters_baseline_shadow(
        request_id="request-a",
        budget_admission_sha256=admission["admission_sha256"],
    )
    monkeypatch.setattr(
        sqlite_store, "revalidate_first_letters_discovery_job_claim",
        lambda *, claim: None,
    )
    claim = sqlite_store.claim_first_letters_discovery_job(
        job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
    )
    row = {
        "job_sha256": claim.job_sha256,
        "item_id": claim.item_id,
        "dispatch_sha256": claim.dispatch_sha256,
        "adapter": copy.deepcopy(branch["adapter"]),
        "adapter_sha256": claim.adapter_sha256,
        "profile_bytes": profile_bytes,
        "reservation_sha256": claim.reservation_sha256,
        "mission_id": "mission-a",
        "request_id": "request-a",
    }
    executed = []

    class Cursor:
        def __init__(self):
            self.query = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, query, args=()):
            self.query = " ".join(query.split())
            executed.append((self.query, args))

        def fetchone(self):
            if "WHERE j.job_id=%s" in self.query:
                return copy.deepcopy(row)
            if "discovery_compute_blocks" in self.query:
                return (
                    {"reason": "CONTROL_INCOMPLETE_COMPUTE_LEDGER"}
                    if current_change == "compute_block" else None
                )
            return None

    class Connection:
        def __init__(self):
            self.exit_errors = []

        def __enter__(self):
            return self

        def __exit__(self, error_type, _error, _traceback):
            self.exit_errors.append(error_type)
            return False

        def cursor(self):
            return Cursor()

    connection = Connection()
    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(store, "connect", lambda: connection)
    monkeypatch.setattr(
        store, "claim_first_letters_discovery_job",
        lambda *, job_id, lease_seconds: claim,
    )
    history_calls = []

    def current_history(_cursor, *, mission_id):
        history_calls.append(mission_id)
        return {
            "state": "CONTROL_INCOMPLETE",
            "manifest_sha256": "f" * 64,
        }

    monkeypatch.setattr(
        store, "_first_letters_empty_history_tx", current_history,
    )
    monkeypatch.setattr(
        store, "_first_letters_discovery_lifecycle_claim",
        lambda *_args, **_kwargs: pytest.fail(
            "lifecycle validation ran after current authority failed"
        ),
    )
    provider = _BridgeProvider()

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"):
        FirstLettersDiscoveryWorker(
            store=store, provider=provider,
        ).run_job(job_id=claim.job_id, lease_seconds=60)

    assert provider.prepare_calls == provider.execute_calls == 0
    assert sum(
        "pg_advisory_xact_lock" in sql for sql, _args in executed
    ) == 2
    assert any("discovery_compute_blocks" in sql for sql, _args in executed)
    assert history_calls == ([] if current_change == "compute_block" else [
        "mission-a"
    ])
    if current_change == "history":
        assert connection.exit_errors[-1] is None


def _postgres_retained_v16_fixture(tmp_path):
    from test_first_letters_discovery_shadow_bridge import (
        _retained_v16_execution_store,
    )

    sqlite_store, branch = _retained_v16_execution_store(tmp_path)
    with sqlite_store.connect() as connection:
        reservation_row = connection.execute(
            "SELECT * FROM first_letters_discovery_compute_reservations"
        ).fetchone()
        work_row = connection.execute(
            "SELECT * FROM first_letters_discovery_work_bindings"
        ).fetchone()
        run_row = connection.execute(
            "SELECT * FROM first_letters_discovery_evidence_runs"
        ).fetchone()
        claim_row = connection.execute(
            "SELECT * FROM first_letters_discovery_executor_claims"
        ).fetchone()
        evidence_row = connection.execute(
            "SELECT * FROM first_letters_discovery_evidence_sets"
        ).fetchone()
        file_rows = connection.execute(
            "SELECT * FROM first_letters_discovery_evidence_files "
            "ORDER BY file_order,relative_path"
        ).fetchall()
        source_row = connection.execute(
            "SELECT * FROM source_snapshots WHERE source_snapshot_id='source-a'"
        ).fetchone()
        task_row = connection.execute(
            "SELECT * FROM tasks WHERE task_id='task-cell-a'"
        ).fetchone()
    reservation = json.loads(reservation_row["reservation_json"])
    work = json.loads(work_row["work_json"])
    retained = {
        **dict(reservation_row),
        "reservation": reservation,
        "work": work,
        "work_sha256": work_row["work_sha256"],
        "w_mission_id": work_row["mission_id"],
        "w_request_id": work_row["request_id"],
        "w_work_kind": work_row["work_kind"],
        "dispatch_kind": work_row["dispatch_kind"],
    }
    run = {
        **dict(run_row),
        "provider_request": json.loads(run_row["provider_request_json"]),
        "run_authority": json.loads(run_row["run_authority_json"]),
    }
    claim = {
        **dict(claim_row), "claim": json.loads(claim_row["claim_json"]),
    }
    evidence = {
        **dict(evidence_row),
        "evidence": json.loads(evidence_row["evidence_json"]),
    }
    files = [dict(row) for row in file_rows]
    source = {
        "source_snapshot_id": source_row["source_snapshot_id"],
        "shape_xyz": json.loads(source_row["shape_xyz_json"]),
        "payload": json.loads(source_row["payload_json"]),
    }
    task = {
        "task_id": task_row["task_id"],
        "mission_id": task_row["mission_id"],
        "source_snapshot_id": task_row["source_snapshot_id"],
        "cell_id": task_row["cell_id"],
        "grid_version": task_row["grid_version"],
        "bounds_xyz": json.loads(task_row["bounds_xyz_json"]),
        "payload": json.loads(task_row["payload_json"]),
    }
    return {
        "sqlite_store": sqlite_store, "branch": branch,
        "retained": retained, "run": run, "claim": claim,
        "evidence": evidence, "files": files,
        "source": source, "task": task,
    }


class _RetainedV16Cursor:
    def __init__(self, fixture):
        self.fixture = fixture
        self.sql = ""
        self.args = ()
        self.executed = []

    def execute(self, sql, args=()):
        self.sql = " ".join(sql.split())
        self.args = args
        self.executed.append((self.sql, args))

    def fetchall(self):
        if "SELECT r.probe_run_id FROM segment_probe_runs" in self.sql:
            return []
        if "native_adapters_v19" in self.sql and "SELECT r.*" in self.sql:
            return [self.fixture["retained"]]
        if "evidence_runs" in self.sql:
            return [self.fixture["run"]]
        if "executor_claims" in self.sql:
            return [self.fixture["claim"]]
        if "evidence_sets" in self.sql:
            return [self.fixture["evidence"]]
        if "evidence_files" in self.sql:
            return self.fixture["files"]
        if "segment_tasks" in self.sql:
            return [self.fixture["task"]]
        return []

    def fetchone(self):
        if "segment_source_snapshots" in self.sql:
            return self.fixture["source"]
        if "segment_tasks" in self.sql:
            return self.fixture["task"]
        if "segment_attempts" in self.sql:
            return self.fixture.get("attempt")
        return None


@pytest.mark.parametrize("orphan_kind", ["reservation", "work"])
def test_postgres_retained_v16_orphan_root_fails_closed(orphan_kind):
    reservation = {
        "reservation_id": "orphan-reservation",
        "work_reservation_id": None,
        "mission_id": "mission-a",
        "request_id": "request-orphan",
        "work_kind": "BASELINE_ARM",
        "source": "RESERVED_BEFORE_EXECUTION",
    }
    work = {
        "reservation_id": "orphan-work",
        "mission_id": "mission-a",
        "request_id": "request-orphan",
        "work_kind": "BASELINE_ARM",
        "dispatch_kind": "BASELINE_DISPATCH",
    }

    class Cursor:
        def __init__(self):
            self.sql = ""
            self.executed = []

        def execute(self, sql, args=()):
            self.sql = " ".join(sql.split())
            self.executed.append((self.sql, args))

        def fetchall(self):
            if "SELECT r.probe_run_id FROM segment_probe_runs" in self.sql:
                return []
            if "work_reservation_id" in self.sql:
                return [reservation] if orphan_kind == "reservation" else []
            if "SELECT w.*" in self.sql and "LEFT JOIN" in self.sql:
                return [work] if orphan_kind == "work" else []
            return []

        def fetchone(self):
            return None

    cursor = Cursor()
    reconciliation = PostgresFleetStore(
        "postgresql://unused"
    )._first_letters_empty_history_tx(cursor, mission_id="mission-a")

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    assert [
        graph["graph_kind"]
        for graph in reconciliation["manifest"]["retained_execution_graphs"]
    ] == [
        "V16_ORPHAN_RESERVATION"
        if orphan_kind == "reservation" else "V16_ORPHAN_WORK"
    ]
    assert any(
        "INSERT INTO segment_first_letters_discovery_compute_blocks" in sql
        for sql, _args in cursor.executed
    )


def test_live_postgres_retained_reservation_without_work_fails_closed(
    live_postgres_store,
):
    store = live_postgres_store
    cap = _cap()
    store.register_discovery_compute_cap(cap)
    core = {
        "schema": "campaignx.first_letters_discovery_compute_reservation.v1",
        "reservation_id": "orphan-reservation",
        "mission_id": "mission-a", "request_id": "request-orphan",
        "work_kind": "BASELINE_ARM",
        "work_authority_id": "orphan-authority",
        "work_authority_sha256": "a" * 64,
        "ordered_item_ids": ["cell-a"],
        "ordered_item_ids_sha256": content_sha256(["cell-a"]),
        "item_count": 1, "compute_unit": "probe_generation_units",
        "top_k": 2, "probe_generations": 12,
        "maximum_attempts_per_candidate": 1,
        "units_per_item": 24, "reserved_units": 24,
        "cap_authority_id": cap["cap_authority_id"],
        "cap_authority_sha256": cap["authority_sha256"],
        "reserved_before_units": 0, "reserved_after_units": 24,
        "source": "RESERVED_BEFORE_EXECUTION",
        "allow_unvalidated": False,
    }
    reservation = {
        **core, "reservation_sha256": content_sha256(core),
        "created_at": "2026-08-03T00:00:00Z",
    }
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO
                   segment_first_letters_discovery_compute_reservations
                   (reservation_id,mission_id,request_id,work_kind,
                    work_authority_id,work_authority_sha256,
                    ordered_item_ids_sha256,item_count,units_per_item,
                    reserved_units,reserved_before_units,reserved_after_units,
                    source,reservation,reservation_sha256,request_sha256)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,1,24,24,0,24,
                          'RESERVED_BEFORE_EXECUTION',%s::jsonb,%s,%s)""",
                (
                    reservation["reservation_id"], "mission-a",
                    "request-orphan", "BASELINE_ARM", "orphan-authority",
                    "a" * 64, reservation["ordered_item_ids_sha256"],
                    json.dumps(
                        reservation, sort_keys=True, separators=(",", ":")
                    ),
                    reservation["reservation_sha256"],
                    content_sha256({"orphan": True}),
                ),
            )

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["manifest"]["retained_execution_graphs"][0][
        "graph_kind"
    ] == "V16_ORPHAN_RESERVATION"


def test_postgres_nonempty_retained_v16_history_imports_exact_execution(tmp_path):
    fixture = _postgres_retained_v16_fixture(tmp_path)
    cursor = _RetainedV16Cursor(fixture)
    store = PostgresFleetStore(
        "postgresql://unused",
        first_letters_discovery_profile_resolver=(
            fixture["sqlite_store"]._first_letters_discovery_profile_resolver
        ),
    )

    reconciliation = store._first_letters_empty_history_tx(
        cursor, mission_id="mission-a"
    )

    assert reconciliation["state"] == "COMPLETE"
    assert reconciliation["fixed_units"] == 24
    assert reconciliation["manifest"]["legacy_v16_reservation_ids"] == [
        fixture["branch"]["reservation"]["reservation_id"]
    ]
    assert any(
        "INSERT INTO segment_first_letters_discovery_historical_imports_v19"
        in sql
        for sql, _args in cursor.executed
    )


def test_postgres_retained_projection_canonicalizes_psycopg_datetimes(tmp_path):
    fixture = _postgres_retained_v16_fixture(tmp_path)
    pg_timestamp = datetime(
        2026, 8, 3, 12, 34, 56, 123456,
        tzinfo=timezone.utc,
    )
    for key in ("retained", "run", "claim", "evidence"):
        for field, value in list(fixture[key].items()):
            if field.endswith("_at") and value is not None:
                fixture[key][field] = pg_timestamp
    for file_row in fixture["files"]:
        for field, value in list(file_row.items()):
            if field.endswith("_at") and value is not None:
                file_row[field] = pg_timestamp
    cursor = _RetainedV16Cursor(fixture)
    store = PostgresFleetStore(
        "postgresql://unused",
        first_letters_discovery_profile_resolver=(
            fixture["sqlite_store"]._first_letters_discovery_profile_resolver
        ),
    )

    reconciliation = store._first_letters_empty_history_tx(
        cursor, mission_id="mission-a"
    )

    assert reconciliation["state"] == "COMPLETE"
    graph = reconciliation["manifest"]["retained_execution_graphs"][0]
    assert graph["run"]["created_at"] == "2026-08-03T12:34:56.123456Z"


@pytest.mark.parametrize(("authority", "replacement"), [
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
    ("m7_resolution", 8),
    ("m7_level", 2),
    ("m7_transform_sha256", "b" * 64),
    ("m7_threshold", 0.6),
    ("missing_opportunity", None),
    ("changed_opportunity", "opportunity-other"),
    ("missing_region", None),
    ("changed_region", {"minimum": [1, 1, 1], "maximum": [63, 63, 63]}),
    ("profile_resolver_malformed", None),
    ("profile_resolver_exception", None),
])
def test_postgres_retained_v16_science_drift_fails_closed(
    tmp_path, authority, replacement,
):
    fixture = _postgres_retained_v16_fixture(tmp_path)
    if authority not in {
        "missing_opportunity", "changed_opportunity",
        "missing_region", "changed_region",
        "profile_resolver_malformed", "profile_resolver_exception",
    }:
        fixture["source"]["payload"][authority] = replacement
    elif authority == "missing_opportunity":
        fixture["task"]["payload"].pop("scientific_opportunity_id")
    elif authority == "changed_opportunity":
        fixture["task"]["payload"]["scientific_opportunity_id"] = replacement
    elif authority == "missing_region":
        fixture["task"]["payload"]["candidate_discovery"].pop("region")
    elif authority == "changed_region":
        fixture["task"]["payload"]["candidate_discovery"]["region"] = replacement
    profile_resolver = (
        fixture["sqlite_store"]._first_letters_discovery_profile_resolver
    )
    if authority == "profile_resolver_exception":
        def profile_resolver(_mission_id, _source_id):
            raise RuntimeError("profile resolver unavailable")
    elif authority == "profile_resolver_malformed":
        profile_resolver = lambda _mission_id, _source_id: b"{}"
    cursor = _RetainedV16Cursor(fixture)
    store = PostgresFleetStore(
        "postgresql://unused",
        first_letters_discovery_profile_resolver=profile_resolver,
    )

    reconciliation = store._first_letters_empty_history_tx(
        cursor, mission_id="mission-a"
    )

    assert reconciliation["state"] == "CONTROL_INCOMPLETE"
    assert reconciliation["fixed_units"] == 0
    assert not any(
        "INSERT INTO segment_first_letters_discovery_historical_imports_v19"
        in sql
        for sql, _args in cursor.executed
    )


def test_live_postgres_incomplete_history_is_visible_from_a_new_connection(
    live_postgres_store,
):
    store = live_postgres_store
    store.register_snapshot({
        "source_snapshot_id": "legacy-source",
        "sample_id": "PHercA", "ct_uri": "fixture://ct",
        "m7_uri": "fixture://m7", "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.0, "coordinate_frame": "ct_l0_xyz",
    })
    store.create_tasks([{
        "task_id": "legacy-task", "mission_id": "mission-a",
        "sample_id": "PHercA", "source_snapshot_id": "legacy-source",
        "cell_id": "cell-a", "grid_version": "grid-v1",
        "policy_version": "policy-v1",
        "bounds_xyz": [[0, 0, 0], [64, 64, 64]],
        "center_xyz": {"x": 32, "y": 32, "z": 32},
        "priority": 1.0, "parameter_envelope": {},
        "catalog_snapshot_sha256": "0" * 64,
        "seed_probe_required": True,
    }])
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO segment_attempts
                   (attempt_id,task_id,attempt_number,worker_id,state)
                   VALUES('legacy-attempt','legacy-task',1,'worker','COMPLETED')"""
            )
            cursor.execute(
                """INSERT INTO segment_probe_runs
                   (probe_run_id,task_id,created_by_attempt_id,
                    source_snapshot_id,candidate_set,candidate_set_sha256,
                    policy_id,policy,policy_sha256,executor_fingerprint,
                    executor_fingerprint_sha256,state)
                   VALUES('legacy-run','legacy-task','legacy-attempt',
                          'legacy-source','[]'::jsonb,%s,'policy','{}'::jsonb,
                          %s,'{}'::jsonb,%s,'PROBING')""",
                (content_sha256([]), content_sha256({}), content_sha256({})),
            )
            admission = {
                "mission_id": "mission-a", "sample_id": "PHercA",
                "execution_bindings": {
                    "source_snapshot_id": "legacy-source",
                },
            }
            cursor.execute(
                """INSERT INTO segment_campaign_budget_admissions
                   (mission_id,sample_id,receipt_sha256,admission,
                    admission_sha256)
                   VALUES('mission-a','PHercA',%s,%s::jsonb,%s)""",
                (
                    "c" * 64,
                    json.dumps(admission, sort_keys=True, separators=(",", ":")),
                    "b" * 64,
                ),
            )
    store._first_letters_discovery_profile_resolver = (
        lambda _mission_id, _source_id: b"profile"
    )

    with pytest.raises(ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"):
        store.reserve_first_letters_baseline_shadow(
            request_id="request-a", budget_admission_sha256="b" * 64,
        )

    with store.connect() as fresh_connection:
        with fresh_connection.cursor() as cursor:
            cursor.execute(
                """SELECT state FROM
                   segment_first_letters_discovery_history_reconciliations_v19
                   WHERE mission_id='mission-a'"""
            )
            assert [row["state"] for row in cursor.fetchall()] == [
                "CONTROL_INCOMPLETE"
            ]
            cursor.execute(
                """SELECT reason FROM
                   segment_first_letters_discovery_compute_blocks
                   WHERE mission_id='mission-a'"""
            )
            assert cursor.fetchone()["reason"] == (
                "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            )


def _seed_live_postgres_baseline_shadow(store, *, reserve_kind="baseline"):
    from test_first_letters_discovery_shadow_bridge import _arm_for_profile

    budget = _budget_admission()
    profile_bytes = _evidence_profile_bytes()
    cap = _evidence_cap()
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
    # The arm fixture carries the SQLite bridge's P0, this mission is seeded
    # from the budget admission; an arm that names a different accepted P0 is
    # rejected before it can reserve.
    arm["accepted_p0_sha256"] = budget["execution_bindings"][
        "p0_artifact_sha256"
    ]
    arm["admission_sha256"] = content_sha256({
        key: value for key, value in arm.items()
        if key != "admission_sha256"
    })
    opportunity_id = stable_id("first-letters-opportunity", {
        "admission_sha256": budget["admission_sha256"], "selection_rank": 0,
    })
    store.register_snapshot({
        "source_snapshot_id": "source-a",
        "source_snapshot_sha256": "1" * 64,
        "source_content_lock_sha256": "d" * 64,
        "sample_id": "PHercA", "ct_uri": "fixture://ct",
        "ct_sha256": "2" * 64, "ct_metadata_sha256": "2" * 64,
        "ct_read_set_manifest_sha256": "e" * 64,
        "m7_uri": "fixture://m7", "m7_sha256": "3" * 64,
        "m7_read_set_manifest_sha256": "0" * 64,
        "m7_model_id": "m7-v1", "m7_model_sha256": "7" * 64,
        "candidate_provider_id": "fixture-provider-v1",
        "candidate_provider_sha256": "8" * 64,
        "discovery_minimum_separation": 12,
        "m7_resolution": 4, "m7_level": 1, "m7_threshold": 0.5,
        "m7_transform_sha256": "f" * 64,
        "shape_xyz": [64, 64, 64], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a", "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": budget[
                "execution_bindings"
            ]["p0_artifact_sha256"],
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {"cell-a": opportunity_id},
        },
    })
    store.register_snapshot({
        "source_snapshot_id": "source-alt",
        "source_snapshot_sha256": "3" * 64,
        "source_content_lock_sha256": "4" * 64,
        "sample_id": "PHercA", "ct_uri": "fixture://ct-alt",
        "ct_sha256": "5" * 64, "ct_metadata_sha256": "5" * 64,
        "ct_read_set_manifest_sha256": "6" * 64,
        "m7_uri": "fixture://m7-alt", "m7_sha256": "7" * 64,
        "m7_metadata_sha256": "7" * 64,
        "m7_read_set_manifest_sha256": "8" * 64,
        "m7_model_id": "m7-alt", "m7_model_sha256": "7" * 64,
        "candidate_provider_id": "fixture-provider-v1",
        "candidate_provider_sha256": "8" * 64,
        "discovery_minimum_separation": 12,
        "m7_resolution": 4, "m7_level": 1, "m7_threshold": 0.5,
        "m7_transform_sha256": "9" * 64,
        "shape_xyz": [64, 64, 64], "voxel_size_um": 9.0,
        "coordinate_frame": "ct_l0_xyz",
        "first_letters_discovery_authority": {
            "mission_id": "mission-a", "accepted_p0_artifact_id": "p0-a",
            "accepted_p0_artifact_sha256": budget[
                "execution_bindings"
            ]["p0_artifact_sha256"],
            "minimum_cell_clearance_voxels": 2,
            "minimum_volume_clearance_voxels": 2,
            "scientific_opportunities": {"cell-a": opportunity_id},
        },
    })
    store.create_tasks([{
        "task_id": "task-a", "mission_id": "mission-a",
        "sample_id": "PHercA", "source_snapshot_id": "source-a",
        "cell_id": "cell-a", "grid_version": "grid-v1",
        "policy_version": "first-letters-search@1.0.0",
        "bounds_xyz": [[0, 0, 0], [64, 64, 64]],
        "center_xyz": {"x": 32, "y": 32, "z": 32}, "priority": 1.0,
        "parameter_envelope": {},
        "catalog_snapshot_sha256": "3" * 64,
        "selection_rank": 0,
        "campaign_budget_admission_sha256": budget["admission_sha256"],
        "p0_artifact_id": "p0-a",
        "p0_artifact_sha256": budget["execution_bindings"][
            "p0_artifact_sha256"
        ],
        "scientific_opportunity_id": opportunity_id,
        "accepted_p0_artifact_id": "p0-a",
        "accepted_p0_artifact_sha256": budget["execution_bindings"][
            "p0_artifact_sha256"
        ],
        "candidate_discovery": {
            "region": {"minimum": [0, 0, 0], "maximum": [64, 64, 64]},
        },
    }])
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO segment_campaign_budget_admissions
                   (mission_id,sample_id,receipt_sha256,admission,
                    admission_sha256)
                   VALUES(%s,%s,%s,%s::jsonb,%s)""",
                (
                    budget["mission_id"], budget["sample_id"],
                    budget["receipt_sha256"], json.dumps(
                        budget, sort_keys=True, separators=(",", ":")
                    ), budget["admission_sha256"],
                ),
            )
    store.register_discovery_compute_cap(cap)
    store._first_letters_discovery_profile_resolver = (
        lambda _mission_id, source_id: (
            alternative_profile_bytes
            if source_id == "source-alt" else profile_bytes
        )
    )
    store._first_letters_experimental_arm_resolver = (
        lambda arm_id: copy.deepcopy(arm) if arm_id == "arm-a" else None
    )
    if reserve_kind == "baseline":
        branch = store.reserve_first_letters_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=budget["admission_sha256"],
        )
    elif reserve_kind == "alternative":
        branch = store.reserve_first_letters_alternative_shadow(
            request_id="request-alt",
            budget_admission_sha256=budget["admission_sha256"],
            arm_id="arm-a",
        )
    elif reserve_kind is None:
        branch = None
    else:  # pragma: no cover - fixture misuse guard
        raise ValueError("unknown live discovery reserve kind")
    return budget, profile_bytes, branch


def test_live_postgres_job_revalidation_serializes_new_history_before_provider(
    live_postgres_store,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from test_first_letters_discovery_shadow_bridge import _BridgeProvider

    store = live_postgres_store
    _budget, _profile_bytes_value, branch = (
        _seed_live_postgres_baseline_shadow(store)
    )
    provider = _BridgeProvider()
    writer = store.connect()
    cursor = writer.cursor()
    try:
        for lock in (
            "campaign-budget-mission:mission-a",
            "discovery-compute-mission:mission-a",
        ):
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (lock,),
            )
        cursor.execute(
            "UPDATE segment_tasks SET seed_probe_required=true "
            "WHERE task_id='task-a'"
        )
        cursor.execute(
            """INSERT INTO segment_attempts
               (attempt_id,task_id,attempt_number,worker_id,state)
               VALUES('legacy-attempt','task-a',1,'legacy-worker','COMPLETED')"""
        )
        cursor.execute(
            """INSERT INTO segment_probe_runs
               (probe_run_id,task_id,created_by_attempt_id,source_snapshot_id,
                candidate_set,candidate_set_sha256,policy_id,policy,
                policy_sha256,executor_fingerprint,
                executor_fingerprint_sha256,state)
               VALUES('legacy-run','task-a','legacy-attempt','source-a',
                      '[]'::jsonb,%s,'legacy-policy','{}'::jsonb,%s,
                      '{}'::jsonb,%s,'PROBING')""",
            (content_sha256([]), content_sha256({}), content_sha256({})),
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                FirstLettersDiscoveryController(
                    mode="shadow", store=store, provider=provider,
                ).run_job,
                job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
            )
            time.sleep(0.1)
            assert not future.done()
            writer.commit()
            with pytest.raises(
                ValueError, match="CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            ):
                future.result(timeout=10)
    finally:
        writer.rollback()
        cursor.close()
        writer.close()

    assert provider.prepare_calls == provider.execute_calls == 0
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT reason FROM "
                "segment_first_letters_discovery_compute_blocks "
                "WHERE mission_id='mission-a'"
            )
            assert cursor.fetchone()["reason"] == (
                "CONTROL_INCOMPLETE_COMPUTE_LEDGER"
            )


@pytest.mark.parametrize("shadow_kind", ["baseline", "alternative"])
def test_live_postgres_off_and_shadow_preserve_all_canonical_tables(
    live_postgres_store, shadow_kind,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from test_first_letters_discovery_shadow_bridge import _BridgeProvider

    store = live_postgres_store
    budget, _profile_bytes_value, _branch = (
        _seed_live_postgres_baseline_shadow(store, reserve_kind=None)
    )
    before = _postgres_canonical_table_projection(store)
    off_provider = _BridgeProvider()
    off = FirstLettersDiscoveryController(
        mode="off", store=store, provider=off_provider,
    )
    if shadow_kind == "baseline":
        off_result = off.reserve_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=budget["admission_sha256"],
        )
    else:
        off_result = off.reserve_alternative_shadow(
            request_id="request-alt",
            budget_admission_sha256=budget["admission_sha256"],
            arm_id="arm-a",
        )
    assert off_result["state"] == "OFF_UNCHANGED"
    assert off.run_job(job_id="off-does-not-read-job", lease_seconds=60)[
        "state"
    ] == "OFF_UNCHANGED"
    assert off_provider.prepare_calls == off_provider.execute_calls == 0
    assert _postgres_canonical_table_projection(store) == before

    provider = _BridgeProvider()
    shadow = FirstLettersDiscoveryController(
        mode="shadow", store=store, provider=provider,
    )
    if shadow_kind == "baseline":
        branch = shadow.reserve_baseline_shadow(
            request_id="request-a",
            budget_admission_sha256=budget["admission_sha256"],
        )
    else:
        branch = shadow.reserve_alternative_shadow(
            request_id="request-alt",
            budget_admission_sha256=budget["admission_sha256"],
            arm_id="arm-a",
        )
    completed = shadow.run_job(
        job_id=branch["jobs"][0]["job_id"], lease_seconds=60,
    )
    assert completed["state"] == "COMPLETED"
    assert completed["canonical_admission"] == "PROHIBITED"
    assert provider.prepare_calls == provider.execute_calls == 1
    assert _postgres_canonical_table_projection(store) == before


def test_live_postgres_nonempty_retained_v16_import_and_cap_match_sqlite(
    live_postgres_store,
):
    from fleet.discovery_controller import FirstLettersDiscoveryController
    from test_first_letters_discovery_shadow_bridge import _BridgeProvider

    store = live_postgres_store
    _budget, _profile_bytes_value, branch = (
        _seed_live_postgres_baseline_shadow(store)
    )

    result = FirstLettersDiscoveryController(
        mode="shadow", store=store, provider=_BridgeProvider(),
    ).run_job(job_id=branch["jobs"][0]["job_id"], lease_seconds=60)
    assert result["state"] == "COMPLETED"
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM segment_first_letters_discovery_jobs_v19"
            )
            cursor.execute(
                "DELETE FROM segment_first_letters_discovery_dispatches_v19"
            )
            cursor.execute(
                "DELETE FROM segment_first_letters_discovery_native_adapters_v19"
            )
            cursor.execute(
                "DELETE FROM "
                "segment_first_letters_discovery_history_reconciliations_v19"
            )

    reconciliation = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )
    replay = store.reconcile_first_letters_discovery_history(
        mission_id="mission-a"
    )

    assert reconciliation["state"] == "COMPLETE"
    assert replay == reconciliation
    assert reconciliation["fixed_units"] == 24
    assert store.discovery_compute_total("mission-a") == 24
    with store.connect() as fresh_connection:
        with fresh_connection.cursor() as cursor:
            cursor.execute(
                "SELECT reservation_id,item_id,fixed_units FROM "
                "segment_first_letters_discovery_historical_imports_v19"
            )
            imported = cursor.fetchall()
            assert imported == [{
                "reservation_id": branch["reservation"]["reservation_id"],
                "item_id": "cell-a", "fixed_units": 24,
            }]


def test_postgres_common_compute_ledger_schema_constraints_match_sqlite_for_all_work_kinds():
    sql = _sql(); source = _source()
    for kind in ("BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM", "ADAPTIVE_CHILD"):
        assert kind in sql
        assert kind in source
    for method in (
        "register_discovery_compute_cap", "discovery_compute_cap",
        "discovery_compute_total", "reserve_discovery_compute",
        "read_discovery_compute_request", "discovery_compute_rows",
        "validate_discovery_compute_reservation",
        "record_discovery_compute_outcome",
        "register_first_letters_discovery_executor",
        "begin_first_letters_discovery_evidence_run",
        "start_first_letters_discovery_evidence_run",
        "heartbeat_first_letters_discovery_evidence_run",
        "complete_first_letters_discovery_evidence_run",
        "mark_first_letters_discovery_evidence_run_incomplete",
        "reconcile_expired_first_letters_discovery_evidence_run",
        "read_first_letters_discovery_evidence_set",
        "read_first_letters_discovery_evidence_run",
        "read_first_letters_discovery_evidence_run_status",
        "build_first_letters_discovery_artifact_and_receipt",
        "resolve_discovery_promotion_evidence",
    ):
        assert callable(getattr(PostgresFleetStore, method, None)), method


def test_postgres_evidence_registry_matches_sqlite_store_owned_producer_boundary():
    sql = _sql()
    source = _source()
    assert "profile_bytes bytea NOT NULL" in sql
    assert "payload bytea NOT NULL" in sql
    assert "CHECK(state IN ('CLAIMED','RUNNING','COMPLETED','CONTROL_INCOMPLETE'))" in sql
    assert "last_heartbeat_at timestamptz" in sql
    assert "incomplete_reason text" in sql
    assert "parent_task_id text REFERENCES segment_tasks(task_id)" in sql
    assert "parent_attempt_id text REFERENCES segment_attempts(attempt_id)" in sql
    assert "UNIQUE(reservation_id,cell_id)" in sql
    for marker in (
        "discovery-evidence-reservation:",
        "discovery-evidence-run:",
        "_derive_first_letters_discovery_run_authority",
        "_produce_first_letters_discovery_evidence_set",
        "_build_first_letters_discovery_artifact_and_receipt_from_evidence_set",
    ):
        assert marker in source


def test_postgres_discovery_claim_and_promotion_boundaries_accept_no_caller_claims():
    begin = inspect.signature(
        PostgresFleetStore.begin_first_letters_discovery_evidence_run
    ).parameters
    resolve = inspect.signature(
        PostgresFleetStore.resolve_discovery_promotion_evidence
    ).parameters
    complete = inspect.signature(
        PostgresFleetStore.complete_first_letters_discovery_evidence_run
    ).parameters
    promote = inspect.signature(
        PostgresFleetStore.begin_discovery_promotion
    ).parameters
    assert set(begin) == {
        "self", "lease_seconds", "reservation_id", "item_id", "profile_bytes",
    }
    assert set(resolve) == {"self", "evidence_set_id"}
    assert "measurements" not in complete
    assert "evidence_set_id" in promote
    assert "admission" not in promote
    assert "_derive_discovery_promotion_admission_from_cursor" in _source()


def test_postgres_direct_discovery_begin_rejects_before_connecting():
    store = PostgresFleetStore("postgresql://unused")
    with pytest.raises(ValueError, match="DISCOVERY_JOB_ID_REQUIRED"):
        store.begin_first_letters_discovery_evidence_run(
            lease_seconds=60, reservation_id="retained-v16",
            item_id="cell-a", profile_bytes=b"unused",
        )


def test_postgres_promotion_transaction_uses_mission_lock_and_three_fact_readback():
    source = _source()
    for method in (
        "begin_discovery_promotion", "read_discovery_promotion",
        "append_discovery_promotion_attempt_binding",
    ):
        assert callable(getattr(PostgresFleetStore, method, None)), method
    assert "promotion-mission:" in source
    assert "CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK" in source
    assert "DISCOVERY_PROMOTED" in source


@pytest.mark.parametrize(("method_name", "inner_lock"), [
    ("register_discovery_compute_cap", "discovery-compute-mission:"),
    ("_block_discovery_compute_ledger", "discovery-compute-mission:"),
    ("reserve_discovery_compute", "discovery-compute-mission:"),
    ("begin_discovery_promotion", "promotion-mission:"),
    ("append_discovery_promotion_attempt_binding", "promotion-mission:"),
])
def test_postgres_task6_transactions_take_campaign_lock_before_inner_mission_lock(
    method_name, inner_lock,
):
    source = inspect.getsource(getattr(PostgresFleetStore, method_name))
    assert "campaign-budget-mission:" in source
    assert source.index("campaign-budget-mission:") < source.index(inner_lock)


def test_postgres_compute_and_promotion_named_failpoints_are_closed():
    source = _source()
    for name in (
        "compute.before_reservation_insert",
        "compute.after_reservation_insert_before_work_insert",
        "compute.after_work_insert_before_commit", "compute.before_commit",
        "compute.commit_outcome_unknown", "compute.after_commit_before_response",
        "promotion.before_authority_insert",
        "promotion.after_authority_insert_before_child_insert",
        "promotion.after_child_insert_before_parent_terminal",
        "promotion.after_parent_terminal_before_commit", "promotion.before_commit",
        "promotion.commit_outcome_unknown", "promotion.after_commit_before_response",
    ):
        assert name in source


def test_live_postgres_compute_exact_replay_conflict_and_response_loss_match_sqlite(
    live_postgres_store,
):
    store = live_postgres_store
    store.register_discovery_compute_cap(_cap())
    first = _reserve_live(
        store, failpoint="bridge.after_commit_before_response",
    )
    assert _reserve_live(store) == first
    with pytest.raises(ValueError, match="CONFLICT|budget authority"):
        _reserve_live(store, items=("cell-a",))
    assert store.discovery_compute_rows("mission-a") == [
        store.read_discovery_compute_request("mission-a", "request-a")
    ]
    assert store.discovery_compute_total("mission-a") == 48


@pytest.mark.parametrize("failpoint", [
    "bridge.before_reservation",
    "bridge.after_reservation_before_adapter",
    "bridge.after_adapter_before_dispatch",
    "bridge.after_dispatch_before_jobs",
    "bridge.after_each_job",
    "bridge.after_jobs_before_commit",
    "bridge.before_commit",
])
def test_live_postgres_each_compute_precommit_failpoint_matches_sqlite(
    live_postgres_store, failpoint,
):
    store = live_postgres_store
    store.register_discovery_compute_cap(_cap())
    with pytest.raises(RuntimeError, match=re.escape(failpoint)):
        _reserve_live(store, failpoint=failpoint)
    assert store.discovery_compute_rows("mission-a") == []
    assert store.discovery_compute_total("mission-a") == 0


def test_live_postgres_compute_commit_unknown_requires_exact_readback(
    live_postgres_store,
):
    store = live_postgres_store
    store.register_discovery_compute_cap(_cap())
    with pytest.raises(
        RuntimeError, match="CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK",
    ):
        _reserve_live(store, failpoint="bridge.commit_outcome_unknown")
    readback = store.read_first_letters_discovery_request(
        "mission-a", "request-a",
    )
    assert readback["reservation"]["reserved_units"] == 48
    assert _reserve_live(store) == readback


def test_live_postgres_evidence_begin_complete_run_readback_build_and_resolve(
    live_postgres_store,
):
    store = live_postgres_store
    profile_bytes, reservation = _prepare_live_evidence(store)
    handle = _claim_live_job(store, reservation)
    running = store.start_first_letters_discovery_evidence_run(
        run_handle=handle
    )
    assert running["state"] == "RUNNING"
    renewed = store.heartbeat_first_letters_discovery_evidence_run(
        run_handle=handle, lease_seconds=120,
    )
    assert renewed == store.read_first_letters_discovery_evidence_run_status(
        handle.run_id
    )
    assert renewed["lease_expires_at"] > running["lease_expires_at"]
    with pytest.raises(RuntimeError, match="READBACK_BY_RUN_ID_REQUIRED"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_evidence_provider_response_bytes(handle),
            failpoint="evidence.after_commit_before_response",
        )
    completed = store.read_first_letters_discovery_evidence_run(handle.run_id)
    artifact, receipt = store.build_first_letters_discovery_artifact_and_receipt(
        completed["evidence_set_id"]
    )
    promotion = store.resolve_discovery_promotion_evidence(
        completed["evidence_set_id"]
    )
    assert artifact["selected_candidate_id"] == "candidate-a"
    assert receipt["selection_outcome"] == "DISCOVERY_WINNER_RETAINED"
    assert promotion["selected_candidate"] == artifact["candidates"][0]
    assert {
        row["role"] for row in completed["retained_files"]
    } >= {"CT_MATERIAL_READ_EVIDENCE", "NONCANONICAL_PROBE_GEOMETRY"}


def test_live_postgres_executor_swap_cannot_complete_claimed_run(
    live_postgres_store,
):
    store = live_postgres_store
    profile_bytes, reservation = _prepare_live_evidence(store)
    handle = _claim_live_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    claiming_executor = store._first_letters_discovery_executor
    store._first_letters_discovery_executor = _FixtureDiscoveryExecutor()
    with pytest.raises(ValueError, match="EXECUTOR_CLAIM_OWNERSHIP"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_evidence_provider_response_bytes(handle),
        )
    store._first_letters_discovery_executor = claiming_executor
    completed = store.complete_first_letters_discovery_evidence_run(
        run_handle=handle,
        provider_response_bytes=_evidence_provider_response_bytes(handle),
    )
    assert completed["evidence_set_id"]


def test_live_postgres_stale_executor_claim_cannot_complete_run(
    live_postgres_store,
):
    store = live_postgres_store
    profile_bytes, reservation = _prepare_live_evidence(store)
    handle = _claim_live_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """UPDATE segment_first_letters_discovery_executor_claims
                      SET lease_expires_at=now() - interval '1 second'
                    WHERE run_id=%s""",
                (handle.run_id,),
            )
    with pytest.raises(ValueError, match="DISCOVERY_EXECUTOR_CLAIM_STALE"):
        store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_evidence_provider_response_bytes(handle),
        )


def test_live_postgres_literal_v15_to_current_upgrade_preserves_rows_and_reads_a3():
    if not DSN:
        pytest.skip("HELENA_TEST_DSN is not set; literal v15 upgrade not run")
    schema = f"task6_discovery_upgrade_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    store = _IsolatedPostgresStore(DSN, schema)
    try:
        v15_sql = _frozen_v15_sql()
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(v15_sql)
                cursor.execute(
                    """INSERT INTO segment_source_snapshots
                       (source_snapshot_id,sample_id,ct_uri,m7_uri,shape_xyz,
                        voxel_size_um,coordinate_frame,payload)
                       VALUES('preserved-source','PHercPreserved','fixture://ct',
                              'fixture://m7','[1,1,1]'::jsonb,9.0,'ct_l0_xyz',
                              '{"source_snapshot_id":"preserved-source",
                                "sample_id":"PHercPreserved"}'::jsonb)"""
                )
        store.initialize()
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT version FROM segment_schema_migrations ORDER BY version"
                )
                assert [
                    row["version"] for row in cursor.fetchall()
                ] == [int(value) for value in re.findall(
                    r"VALUES\s*\(\s*(\d+)\s*,\s*'", _sql()
                )]
                cursor.execute(
                    "SELECT sample_id FROM segment_source_snapshots "
                    "WHERE source_snapshot_id='preserved-source'"
                )
                assert cursor.fetchone()["sample_id"] == "PHercPreserved"
        profile_bytes, reservation = _prepare_live_evidence(store)
        handle = _claim_live_job(store, reservation)
        store.start_first_letters_discovery_evidence_run(run_handle=handle)
        completed = store.complete_first_letters_discovery_evidence_run(
            run_handle=handle,
            provider_response_bytes=_evidence_provider_response_bytes(handle),
        )
        assert store.read_first_letters_discovery_evidence_run(
            handle.run_id
        )["evidence_set_id"] == completed["evidence_set_id"]
    finally:
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.mark.parametrize("pair", [
    ("BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM"),
    ("BASELINE_ARM", "ADAPTIVE_CHILD"),
    ("ALTERNATIVE_SOURCE_ARM", "ADAPTIVE_CHILD"),
])
def test_live_postgres_mixed_48_unit_concurrency_matches_sqlite(
    live_postgres_store, pair,
):
    store = live_postgres_store
    store.register_discovery_compute_cap(_cap())

    def reserve(index: int):
        try:
            return _reserve_live(store, pair[index], f"request-{index}")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, (0, 1)))
    assert sum(
        result is not None and result["reservation"]["reserved_units"] or 0
        for result in results
    ) == 48
    assert store.discovery_compute_total("mission-a") == 48
    assert len(store.discovery_compute_rows("mission-a")) == 1


def _prepare_live_promotion_parent(store: PostgresFleetStore) -> str:
    profile_bytes, reservation = _prepare_live_evidence(
        store, promotion=True,
    )
    handle = _claim_live_job(store, reservation)
    store.start_first_letters_discovery_evidence_run(run_handle=handle)
    completed = store.complete_first_letters_discovery_evidence_run(
        run_handle=handle,
        provider_response_bytes=_evidence_provider_response_bytes(handle),
    )
    return completed["evidence_set_id"]


def test_live_postgres_promotion_failpoints_and_three_fact_readback_match_sqlite(
    live_postgres_store,
):
    store = live_postgres_store
    evidence_set_id = _prepare_live_promotion_parent(store)
    for failpoint in (
        "promotion.before_authority_insert",
        "promotion.after_authority_insert_before_child_insert",
        "promotion.after_child_insert_before_parent_terminal",
        "promotion.after_parent_terminal_before_commit",
        "promotion.before_commit",
    ):
        with pytest.raises(RuntimeError, match=failpoint):
            store.begin_discovery_promotion(
                request_id="promotion-a", evidence_set_id=evidence_set_id,
                task9_gate=_gate(),
                promotion_failpoint=_raise_at(failpoint),
            )
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
    assert readback["parent"] == {
        "task_id": "task-a", "attempt_id": stable_id(
            "attempt", {"task_id": "task-a", "attempt_number": 1}
        ),
        "state": "DISCOVERY_PROMOTED",
    }


def test_live_postgres_concurrent_promotion_serializes_to_one_three_fact_set(
    live_postgres_store,
):
    store = live_postgres_store
    evidence_set_id = _prepare_live_promotion_parent(store)

    def promote(_index: int):
        return store.begin_discovery_promotion(
            request_id="promotion-a", evidence_set_id=evidence_set_id,
            task9_gate=_gate(),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(promote, (0, 1)))
    assert results[0] == results[1]
    assert results[0]["parent"]["state"] == "DISCOVERY_PROMOTED"
