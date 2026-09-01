from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import surface_routing
from fleet.store import FleetStore
from fleet.common import content_sha256
from fleet.postgres_store import PostgresFleetStore
from fleet.store_factory import open_fleet_store


ENTRYPOINT = (
    ROOT
    / "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py"
)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("helena_fleet_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_store_factory_preserves_sqlite_paths(tmp_path: Path) -> None:
    store = open_fleet_store(tmp_path / "fleet.sqlite")
    assert isinstance(store, FleetStore)
    store.initialize()
    assert store.status()["database"] == str(tmp_path / "fleet.sqlite")


def test_postgres_geometry_certification_persists_causal_qc_promotion(monkeypatch):
    statements = []
    # The verdict releases a waiting job only for a surface the size gate
    # admits, so the fake catalogue has to answer the routing question too --
    # with a real receipt over a real area, not a stub, or this would be
    # asserting the promotion against a decision that never happened.
    surface = {
        "surface_id": "surface-control", "physical_qc_state": "UNVALIDATED",
        "artifact_sha256": "3" * 64, "area_cm2": 0.5,
        "source_snapshot_id": "source-control", "sample_id": "PHercTEST",
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]], "sample_points": [[0.0, 0.0, 0.0]],
        "payload": {},
    }
    routing = surface_routing.receipt_for_surface(
        {**surface, "geometry_qc_state": "GEOMETRY_CERTIFIED"})

    class Cursor:
        last = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, parameters=()):
            self.last = statement
            statements.append((statement, parameters))
            self.rowcount = 1

        def fetchone(self):
            if "segment_surface_routing_receipts" in self.last:
                return {"receipt": routing}
            return surface

        def fetchall(self):
            if "SELECT qc_job_id,payload" in self.last:
                return [{"qc_job_id": "qc-current", "payload": {}}]
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
    result = store.record_geometry_certification(
        "surface-control", "GEOMETRY_CERTIFIED", {"schema": "geometry.v1"},
        requested_by_job_id="p2-current", profile_id="geometry@1",
        profile_sha256="6" * 64)
    assert result["geometry_certified_by_job_id"] == "p2-current"
    assert result["qc_promotion_links"][0]["qc_job_id"] == "qc-current"
    encoded = json.dumps(statements, default=str)
    assert "geometry_certification_lineage" in encoded
    assert "unblocked_by_job_id" in encoded
    assert "promotion_event_id" in encoded


def test_postgres_flattening_returns_persisted_job_lineage_not_newer_claim(monkeypatch):
    old = {
        "surface_id": "surface-control", "profile_id": "flatten-abf-v1@1.0.0",
        "state": "FLATTENED", "requested_by_job_id": "p3-old",
        "source_artifact_sha256": "3" * 64, "profile_file_sha256": "5" * 64,
        "artifact_sha256": "7" * 64, "receipt_sha256": "8" * 64,
    }

    # P3 now reads the surface's routing receipt before it writes anything, so
    # the scripted cursor has to answer that read: this surface is on the
    # standard path, which is the precondition this test was always assuming.
    from fleet import surface_routing

    routed = surface_routing.build_receipt(
        surface_id="surface-control", area_cm2=0.5,
        policy=surface_routing.load_policy(),
        measurement={"bbox_xyz": [[0, 0, 0], [64, 64, 8]],
                     "sample_point_count": 4096},
        read_set={"source_snapshot_id": "snapshot-control",
                  "sample_id": "PHercCONTROL", "artifact_sha256": "7" * 64,
                  "geometry_qc_state": "GEOMETRY_CERTIFIED"})

    class Cursor:
        last = ""

        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, statement, parameters=()):
            self.last = statement
            self.rowcount = 0 if "INSERT INTO surface_flattenings" in statement else 1
        def fetchone(self):
            if "segment_surface_routing_receipts" in self.last:
                return {"receipt": routed}
            return {"payload": old}

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()

    current = {
        "surface_id": "surface-control", "profile_id": "flatten-abf-v1@1.0.0",
        "state": "FLATTENED", "requested_by_job_id": "p3-current",
        "source_artifact_sha256": "3" * 64, "profile_file_sha256": "5" * 64,
    }
    current["receipt_sha256"] = content_sha256(current)
    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(store, "connect", lambda: Connection())
    result = store.record_flattening(current)
    assert result["inserted"] is False
    assert result["requested_by_job_id"] == "p3-old"


def test_postgres_env_spec_keeps_secret_out_of_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "postgresql://fleet:do-not-log@example.invalid/fleet"
    monkeypatch.setenv("SEGMENT_FLEET_DATABASE_URL", secret)
    store = open_fleet_store("postgres-env://SEGMENT_FLEET_DATABASE_URL")
    assert store.database_url == secret
    assert store.identity == "postgres-env://SEGMENT_FLEET_DATABASE_URL"
    assert secret not in store.identity


@pytest.mark.parametrize(
    "spec",
    (
        "postgres-env://",
        "postgres-env://BAD-NAME",
        "postgres-env://NAME/EXTRA",
    ),
)
def test_postgres_env_spec_rejects_unsafe_variable_names(spec: str) -> None:
    with pytest.raises(ValueError):
        open_fleet_store(spec)


def test_postgres_env_spec_fails_closed_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEGMENT_FLEET_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        open_fleet_store("postgres-env://SEGMENT_FLEET_DATABASE_URL")


def test_entrypoint_discovers_explicit_partial_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial_root = tmp_path / "campaign-x-worker"
    (partial_root / "framework/stages/01-segmentation/fleet").mkdir(parents=True)
    monkeypatch.setenv("HELENA_REPO_ROOT", str(partial_root))
    module = _load_entrypoint()
    assert module.discover_repo_root() == partial_root.resolve()


def test_entrypoint_rejects_invalid_explicit_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HELENA_REPO_ROOT", str(tmp_path))
    with pytest.raises(RuntimeError, match="does not contain"):
        _load_entrypoint()


def test_postgres_migration_and_adapter_expose_transactional_qc_leases() -> None:
    migration = (
        ROOT
        / "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql"
    ).read_text(encoding="utf-8")
    for field in (
        "lease_token_hash",
        "lease_expires_at",
        "retry_after",
        "result jsonb",
        "segment_qc_jobs_ready",
    ):
        assert field in migration
    for method in ("claim_qc", "heartbeat_qc", "finalize_qc", "requeue_qc_unavailable"):
        assert callable(getattr(PostgresFleetStore, method))


def test_postgres_migration_and_adapter_expose_resource_aware_scheduling() -> None:
    migration = (
        ROOT
        / "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql"
    ).read_text(encoding="utf-8")
    for field in (
        "gpu_required",
        "minimum_vram_gb",
        "segment_worker_capabilities",
        "resource-aware GPU admission",
    ):
        assert field in migration
    assert callable(getattr(PostgresFleetStore, "requeue_for_larger_gpu"))


def test_postgres_probe_ledger_binds_one_budget_to_one_parent_task() -> None:
    migration = (
        ROOT
        / "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql"
    ).read_text(encoding="utf-8")
    normalized = " ".join(migration.split())
    assert (
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "segment_probe_runs_one_per_task ON segment_probe_runs(task_id)"
        in normalized
    )
    assert "VALUES (7, 'one immutable seed probe budget per parent task')" in (
        normalized
    )
    assert (
        "VALUES (8, 'bind probe promotion to the exact winner "
        "continuation plan')" in normalized
    )
    assert (
        "VALUES (9, 'enforce non-null probe promotion bindings and reject "
        "legacy rows')" in normalized
    )
    for column in (
        "continuation_attempt_id",
        "continuation_contract_sha256",
        "continuation_locked_plan_sha256",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in normalized
        assert f"ALTER COLUMN {column} SET NOT NULL" in normalized
    assert "seed-probe v8 cannot auto-bind legacy promotion rows" in normalized

    adapter_path = (
        ROOT
        / "framework/stages/01-segmentation/fleet/postgres_store.py"
    )
    adapter_source = adapter_path.read_text(encoding="utf-8")
    tree = ast.parse(adapter_source)
    store_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PostgresFleetStore"
    )
    methods = {
        node.name: ast.get_source_segment(adapter_source, node) or ""
        for node in store_class.body
        if isinstance(node, ast.FunctionDef)
    }
    trial_guard = methods["_assert_probe_trial_owner"]
    assert "JOIN segment_probe_runs" in trial_guard
    assert "JOIN segment_probe_attempts" in trial_guard
    assert "r.task_id=%s" in trial_guard
    run_guard = methods["_assert_probe_run_task"]
    assert "probe_run_id=%s AND task_id=%s" in run_guard

    for method_name in (
        "claim_probe_trial",
        "record_probe_decision",
        "begin_probe_promotion",
    ):
        assert "_assert_probe_run_task" in methods[method_name]
    for method_name in (
        "transition_probe_trial",
        "reserve_probe_artifact",
        "complete_probe_trial",
        "fail_probe_trial",
    ):
        assert "_assert_probe_trial_owner" in methods[method_name]


def test_postgres_finalization_uses_the_same_authoritative_select_gate() -> None:
    adapter_path = (
        ROOT
        / "framework/stages/01-segmentation/fleet/postgres_store.py"
    )
    adapter_source = adapter_path.read_text(encoding="utf-8")
    tree = ast.parse(adapter_source)
    store_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "PostgresFleetStore"
    )
    finalize = next(
        ast.get_source_segment(adapter_source, node) or ""
        for node in store_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "finalize"
    )
    for binding in (
        "t.seed_probe_required",
        "FROM segment_probe_runs",
        "LEFT JOIN segment_probe_decisions",
        "validate_probe_finalization_authority",
        "task_payload=dict(context[\"payload\"] or {})",
    ):
        assert binding in finalize

    recovery = next(
        ast.get_source_segment(adapter_source, node) or ""
        for node in store_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "recover_terminal_finalizer_dependency"
    )
    for recovery_binding in (
        "PROBE_PROMOTION_RECOVERED",
        "SET state='CONTINUING'",
        "SET state='WINNER_RETAINED'",
        "final_status",
    ):
        assert recovery_binding in recovery

    mark_terminal = next(
        ast.get_source_segment(adapter_source, node) or ""
        for node in store_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "mark_terminal"
    )
    for terminal_binding in (
        "CONTINUATION_QUEUED",
        "CONTINUATION_FAILED",
        "PROBE_CONTINUATION_FAILED",
        "REVIEW_RETAINED",
    ):
        assert terminal_binding in mark_terminal

    requeue = next(
        ast.get_source_segment(adapter_source, node) or ""
        for node in store_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_requeue_unavailable"
    )
    assert "maximum_requeues" in requeue
    assert "SELECT COUNT(*) AS count" in requeue


def test_postgres_probe_gc_rows_are_json_receipt_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retain_until = datetime(
        2026, 7, 30, 8, 9, 10, 123456,
        tzinfo=timezone(timedelta(hours=-3)),
    )
    row = {
        "probe_artifact_set_id": "artifact-1",
        "artifact_uri": "s3://bucket/probes/sample/run/trial/" + "a" * 64,
        "manifest": {"artifact_sha256": "a" * 64, "files": {}},
        "state": "LOSER_RETAINED",
        "retain_until": retain_until,
        "probe_run_id": "run-1",
        "task_id": "task-1",
    }

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, *_args):
            return None

        def fetchall(self):
            return [row]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

    store = PostgresFleetStore("postgresql://unused")
    monkeypatch.setattr(store, "connect", lambda: Connection())
    candidates = store.probe_artifacts_due_for_gc()

    assert candidates[0]["retain_until"] == "2026-07-30T11:09:10.123456Z"
    assert len(content_sha256(candidates)) == 64
    json.dumps(candidates)
