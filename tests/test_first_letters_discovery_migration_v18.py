from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.postgres_store import PostgresFleetStore
from fleet.store import FleetStore


SQLITE_V17_FIXTURE = (
    ROOT / "tests/fixtures/sqlite-discovery-v17.sql.gz.b64"
)
POSTGRES_V17_FIXTURE = (
    ROOT / "tests/fixtures/postgresql-discovery-v17.sql.gz.b64"
)
SQLITE_V17_SHA256 = (
    "21112318a959f2ea3b991761dbfeaa5b656d373b3c6fff620a375850c77d629e"
)
POSTGRES_V17_SHA256 = (
    "cb1a0554d1fe4de20f349177d56d2defeaaaeb07cfe555469fa84105ca4c8b5d"
)
DSN = os.environ.get("HELENA_TEST_DSN")
LIFECYCLE_COLUMNS = {
    "started_at", "last_heartbeat_at", "incomplete_at",
    "incomplete_reason",
}


def _frozen_sql(path: Path, expected_sha256: str) -> str:
    encoded = b"".join(path.read_bytes().split())
    payload = gzip.decompress(base64.b64decode(encoded, validate=True))
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    return payload.decode("utf-8")


def _frozen_sqlite_v17_sql() -> str:
    return _frozen_sql(SQLITE_V17_FIXTURE, SQLITE_V17_SHA256)


def _frozen_postgres_v17_sql() -> str:
    return _frozen_sql(POSTGRES_V17_FIXTURE, POSTGRES_V17_SHA256)


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table})")
    }


def _sqlite_v17_data_snapshot(
    connection: sqlite3.Connection,
) -> dict[str, list[tuple]]:
    snapshot = {}
    for table in (
        "first_letters_discovery_evidence_runs",
        "first_letters_discovery_executor_claims",
        "first_letters_discovery_evidence_sets",
        "first_letters_discovery_evidence_files",
    ):
        columns = [
            str(row[1]) for row in connection.execute(
                f"PRAGMA table_info({table})"
            )
            if str(row[1]) not in LIFECYCLE_COLUMNS
        ]
        projection = ",".join(columns)
        snapshot[table] = [
            tuple(row) for row in connection.execute(
                f"SELECT {projection} FROM {table} ORDER BY 1,2"
            ).fetchall()
        ]
    return snapshot


def _seed_sqlite_v17_rows(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(_frozen_sqlite_v17_sql())
        connection.execute(
            """INSERT INTO source_snapshots
               (source_snapshot_id,sample_id,ct_uri,m7_uri,shape_xyz_json,
                voxel_size_um,coordinate_frame,payload_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                "source-v17", "PHercV17", "fixture://ct", "fixture://m7",
                "[1,1,1]", 9.0, "ct_l0_xyz", "{}",
                "2026-01-01T00:00:00Z",
            ),
        )
        connection.execute(
            """INSERT INTO first_letters_discovery_compute_caps
               (mission_id,cap_authority_id,authority_sha256,cap_units,
                authority_json,created_at) VALUES(?,?,?,?,?,?)""",
            (
                "mission-v17", "cap-v17", "a" * 64, 48, "{}",
                "2026-01-01T00:00:00Z",
            ),
        )
        for suffix, request_id, work_sha, reservation_sha in (
            ("claimed", "request-claimed", "b" * 64, "c" * 64),
            ("completed", "request-completed", "d" * 64, "e" * 64),
        ):
            connection.execute(
                """INSERT INTO first_letters_discovery_compute_reservations
                   (reservation_id,mission_id,request_id,work_kind,
                    work_authority_id,work_authority_sha256,
                    ordered_item_ids_sha256,item_count,units_per_item,
                    reserved_units,reserved_before_units,reserved_after_units,
                    source,reservation_json,reservation_sha256,request_sha256,
                    created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"reservation-{suffix}", "mission-v17", request_id,
                    "BASELINE_ARM", f"work-{suffix}", work_sha,
                    ("1" if suffix == "claimed" else "2") * 64,
                    1, 24, 24, 0 if suffix == "claimed" else 24,
                    24 if suffix == "claimed" else 48,
                    "RESERVED_BEFORE_EXECUTION", "{}", reservation_sha,
                    ("3" if suffix == "claimed" else "4") * 64,
                    "2026-01-01T00:00:00Z",
                ),
            )
        connection.execute(
            """INSERT INTO first_letters_discovery_executor_registry
               (worker_id,executor_id,executor_sha256,capabilities_json,
                registration_json,registration_sha256,enabled,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                "worker-v17", "executor-v17", "5" * 64,
                '["FIRST_LETTERS_DISCOVERY_CT_PROBE_V1"]', "{}", "6" * 64,
                1, "2026-01-01T00:00:00Z",
            ),
        )
        for suffix, state, completed_at, token_sha, authority_sha in (
            ("claimed", "CLAIMED", None, "7" * 64, "8" * 64),
            (
                "completed", "COMPLETED", "2026-01-01T00:05:00Z",
                "9" * 64, "0" * 64,
            ),
        ):
            connection.execute(
                """INSERT INTO first_letters_discovery_evidence_runs
                   (run_id,reservation_id,mission_id,request_id,worker_id,
                    cell_id,source_snapshot_id,run_token_sha256,
                    lease_expires_at,profile_bytes,profile_file_sha256,
                    provider_request_json,run_authority_json,
                    run_authority_sha256,state,created_at,completed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"run-{suffix}", f"reservation-{suffix}",
                    "mission-v17", f"request-{suffix}", "worker-v17",
                    f"cell-{suffix}", "source-v17", token_sha,
                    "2026-01-01T01:00:00Z", b"legacy-profile",
                    ("a" if suffix == "claimed" else "b") * 64,
                    "{}", "{}", authority_sha, state,
                    "2026-01-01T00:00:00Z", completed_at,
                ),
            )
            connection.execute(
                """INSERT INTO first_letters_discovery_executor_claims
                   (claim_id,run_id,worker_id,executor_id,executor_sha256,
                    capability,claim_attempt_number,
                    execution_lease_token_sha256,lease_expires_at,claim_json,
                    claim_sha256,state,created_at,completed_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    f"claim-{suffix}", f"run-{suffix}", "worker-v17",
                    "executor-v17", "5" * 64,
                    "FIRST_LETTERS_DISCOVERY_CT_PROBE_V1", 1,
                    ("c" if suffix == "claimed" else "d") * 64,
                    "2026-01-01T01:00:00Z", "{}",
                    ("e" if suffix == "claimed" else "f") * 64,
                    state, "2026-01-01T00:00:00Z", completed_at,
                ),
            )
        connection.execute(
            """INSERT INTO first_letters_discovery_evidence_sets
               (evidence_set_id,run_id,evidence_json,evidence_set_sha256,
                created_at) VALUES(?,?,?,?,?)""",
            (
                "evidence-completed", "run-completed", '{"legacy":true}',
                "1" * 64, "2026-01-01T00:05:00Z",
            ),
        )
        connection.execute(
            """INSERT INTO first_letters_discovery_evidence_files
               (evidence_set_id,file_order,relative_path,role,payload,
                byte_count,sha256) VALUES(?,?,?,?,?,?,?)""",
            (
                "evidence-completed", 0, "legacy.bin",
                "CANDIDATE_PROVIDER_RESPONSE", b"v17-evidence", 12,
                "2" * 64,
            ),
        )


def test_frozen_sqlite_fixture_is_complete_literal_v17_schema(tmp_path):
    path = tmp_path / "frozen-v17.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(_frozen_sqlite_v17_sql())
        run_columns = _sqlite_columns(
            connection, "first_letters_discovery_evidence_runs"
        )
        claim_columns = _sqlite_columns(
            connection, "first_letters_discovery_executor_claims"
        )
        run_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            ("first_letters_discovery_evidence_runs",),
        ).fetchone()[0]
    assert not LIFECYCLE_COLUMNS & run_columns
    assert not LIFECYCLE_COLUMNS & claim_columns
    assert "CHECK(state IN ('CLAIMED','COMPLETED'))" in run_sql


def test_frozen_postgres_fixture_is_complete_literal_v17_migration():
    sql = _frozen_postgres_v17_sql()
    versions = [int(value) for value in re.findall(
        r"VALUES\s*\(\s*(\d+)\s*,\s*'", sql
    )]
    assert versions == list(range(1, 18))
    assert "CHECK(state IN ('CLAIMED','COMPLETED'))" in sql
    assert "running discovery claims, heartbeats" not in sql


def test_sqlite_literal_v17_upgrade_preserves_rows_and_executes_v18_states(
    tmp_path,
):
    path = tmp_path / "upgrade.sqlite"
    _seed_sqlite_v17_rows(path)
    with sqlite3.connect(path) as connection:
        before_rows = _sqlite_v17_data_snapshot(connection)
    store = FleetStore(path)
    store.initialize()

    claimed = store.read_first_letters_discovery_evidence_run_status(
        "run-claimed"
    )
    completed = store.read_first_letters_discovery_evidence_run_status(
        "run-completed"
    )
    assert claimed["state"] == "CLAIMED"
    assert claimed["started_at"] is None
    assert completed["state"] == "COMPLETED"
    assert completed["evidence_set_id"] == "evidence-completed"
    assert completed["started_at"] is None

    with store.connect() as connection:
        assert LIFECYCLE_COLUMNS <= _sqlite_columns(
            connection, "first_letters_discovery_evidence_runs"
        )
        assert LIFECYCLE_COLUMNS <= _sqlite_columns(
            connection, "first_letters_discovery_executor_claims"
        )
        assert connection.execute(
            "SELECT payload FROM first_letters_discovery_evidence_files "
            "WHERE evidence_set_id='evidence-completed'"
        ).fetchone()[0] == b"v17-evidence"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert _sqlite_v17_data_snapshot(connection) == before_rows
        for table in (
            "first_letters_discovery_evidence_runs",
            "first_letters_discovery_executor_claims",
        ):
            connection.execute(
                f"""UPDATE {table}
                       SET state='RUNNING',started_at=?,last_heartbeat_at=?
                     WHERE run_id='run-claimed'""",
                ("2026-01-01T00:10:00Z", "2026-01-01T00:10:00Z"),
            )
    assert store.read_first_letters_discovery_evidence_run_status(
        "run-claimed"
    )["state"] == "RUNNING"

    with store.connect() as connection:
        for table in (
            "first_letters_discovery_evidence_runs",
            "first_letters_discovery_executor_claims",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    f"UPDATE {table} SET state='INVALID' "
                    "WHERE run_id='run-claimed'"
                )
        for table in (
            "first_letters_discovery_evidence_runs",
            "first_letters_discovery_executor_claims",
        ):
            connection.execute(
                f"""UPDATE {table}
                       SET state='CONTROL_INCOMPLETE',incomplete_at=?,
                           incomplete_reason='WORKER_LOST_AFTER_RUNNING'
                     WHERE run_id='run-claimed'""",
                ("2026-01-01T00:20:00Z",),
            )
    assert store.read_first_letters_discovery_evidence_run_status(
        "run-claimed"
    )["state"] == "CONTROL_INCOMPLETE"

    store.initialize()
    assert store.read_first_letters_discovery_evidence_run_status(
        "run-completed"
    )["evidence_set_id"] == "evidence-completed"
    assert store.read_first_letters_discovery_evidence_run_status(
        "run-claimed"
    )["incomplete_reason"] == "WORKER_LOST_AFTER_RUNNING"


def test_sqlite_fresh_v18_initialize_and_repeat_are_idempotent(tmp_path):
    path = tmp_path / "fresh.sqlite"
    store = FleetStore(path)
    store.initialize()
    with store.connect() as connection:
        before = {
            table: connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()[0]
            for table in (
                "first_letters_discovery_evidence_runs",
                "first_letters_discovery_executor_claims",
            )
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert all(
            LIFECYCLE_COLUMNS <= _sqlite_columns(connection, table)
            for table in before
        )
    store.initialize()
    with store.connect() as connection:
        after = {
            table: connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()[0]
            for table in before
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert after == before


def test_sqlite_v18_upgrade_fails_closed_on_orphaned_v17_claim(tmp_path):
    path = tmp_path / "orphan.sqlite"
    _seed_sqlite_v17_rows(path)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute(
            "DELETE FROM first_letters_discovery_evidence_runs "
            "WHERE run_id='run-claimed'"
        )
    with pytest.raises(
        RuntimeError, match="SQLite discovery lifecycle v18 foreign-key drift",
    ):
        FleetStore(path).initialize()
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall()
        assert not LIFECYCLE_COLUMNS & _sqlite_columns(
            connection, "first_letters_discovery_evidence_runs"
        )
        assert not LIFECYCLE_COLUMNS & _sqlite_columns(
            connection, "first_letters_discovery_executor_claims"
        )
    with pytest.raises(
        RuntimeError, match="SQLite discovery lifecycle v18 foreign-key drift",
    ):
        FleetStore(path).initialize()


class _SchemaPostgresStore(PostgresFleetStore):
    def __init__(self, database_url: str, schema: str):
        super().__init__(database_url)
        self._test_schema = schema

    def connect(self):
        connection = super().connect()
        with connection.cursor() as cursor:
            cursor.execute(f"SET search_path TO {self._test_schema}, public")
        return connection


@contextmanager
def _isolated_postgres_store(prefix: str):
    if not DSN:
        pytest.skip(
            "HELENA_TEST_DSN is not set; real PostgreSQL v17-to-v18 "
            "migration was not run"
        )
    schema = f"{prefix}_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    try:
        yield _SchemaPostgresStore(DSN, schema)
    finally:
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA {schema} CASCADE")


def _seed_postgres_v17_rows(store: PostgresFleetStore) -> None:
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(_frozen_postgres_v17_sql())
            cursor.execute(
                """INSERT INTO segment_source_snapshots
                   (source_snapshot_id,sample_id,ct_uri,m7_uri,shape_xyz,
                    voxel_size_um,coordinate_frame,payload)
                   VALUES('source-v17','PHercV17','fixture://ct','fixture://m7',
                          '[1,1,1]'::jsonb,9.0,'ct_l0_xyz','{}'::jsonb)"""
            )
            cursor.execute(
                """INSERT INTO segment_first_letters_discovery_compute_caps
                   (mission_id,cap_authority_id,authority_sha256,cap_units,
                    authority) VALUES('mission-v17','cap-v17',%s,48,
                                      '{}'::jsonb)""",
                ("a" * 64,),
            )
            for suffix, request_id, work_sha, reservation_sha in (
                ("claimed", "request-claimed", "b" * 64, "c" * 64),
                ("completed", "request-completed", "d" * 64, "e" * 64),
            ):
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_compute_reservations
                       (reservation_id,mission_id,request_id,work_kind,
                        work_authority_id,work_authority_sha256,
                        ordered_item_ids_sha256,item_count,units_per_item,
                        reserved_units,reserved_before_units,
                        reserved_after_units,source,reservation,
                        reservation_sha256,request_sha256)
                       VALUES(%s,'mission-v17',%s,'BASELINE_ARM',%s,%s,%s,
                              1,24,24,%s,%s,'RESERVED_BEFORE_EXECUTION',
                              '{}'::jsonb,%s,%s)""",
                    (
                        f"reservation-{suffix}", request_id, f"work-{suffix}",
                        work_sha,
                        ("1" if suffix == "claimed" else "2") * 64,
                        0 if suffix == "claimed" else 24,
                        24 if suffix == "claimed" else 48,
                        reservation_sha,
                        ("3" if suffix == "claimed" else "4") * 64,
                    ),
                )
            cursor.execute(
                """INSERT INTO
                   segment_first_letters_discovery_executor_registry
                   (worker_id,executor_id,executor_sha256,capabilities,
                    registration,registration_sha256,enabled)
                   VALUES('worker-v17','executor-v17',%s,%s::jsonb,
                          '{}'::jsonb,%s,true)""",
                (
                    "5" * 64,
                    json.dumps(["FIRST_LETTERS_DISCOVERY_CT_PROBE_V1"]),
                    "6" * 64,
                ),
            )
            for suffix, state, completed_at, token_sha, authority_sha in (
                ("claimed", "CLAIMED", None, "7" * 64, "8" * 64),
                (
                    "completed", "COMPLETED", "2026-01-01T00:05:00Z",
                    "9" * 64, "0" * 64,
                ),
            ):
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_evidence_runs
                       (run_id,reservation_id,mission_id,request_id,worker_id,
                        cell_id,source_snapshot_id,run_token_sha256,
                        lease_expires_at,profile_bytes,profile_file_sha256,
                        provider_request,run_authority,run_authority_sha256,
                        state,completed_at)
                       VALUES(%s,%s,'mission-v17',%s,'worker-v17',%s,
                              'source-v17',%s,'2026-01-01T01:00:00Z',%s,%s,
                              '{}'::jsonb,'{}'::jsonb,%s,%s,%s)""",
                    (
                        f"run-{suffix}", f"reservation-{suffix}",
                        f"request-{suffix}", f"cell-{suffix}", token_sha,
                        b"legacy-profile",
                        ("a" if suffix == "claimed" else "b") * 64,
                        authority_sha, state, completed_at,
                    ),
                )
                cursor.execute(
                    """INSERT INTO
                       segment_first_letters_discovery_executor_claims
                       (claim_id,run_id,worker_id,executor_id,executor_sha256,
                        capability,claim_attempt_number,
                        execution_lease_token_sha256,lease_expires_at,claim,
                        claim_sha256,state,completed_at)
                       VALUES(%s,%s,'worker-v17','executor-v17',%s,
                              'FIRST_LETTERS_DISCOVERY_CT_PROBE_V1',1,%s,
                              '2026-01-01T01:00:00Z','{}'::jsonb,%s,%s,%s)""",
                    (
                        f"claim-{suffix}", f"run-{suffix}", "5" * 64,
                        ("c" if suffix == "claimed" else "d") * 64,
                        ("e" if suffix == "claimed" else "f") * 64,
                        state, completed_at,
                    ),
                )
            cursor.execute(
                """INSERT INTO segment_first_letters_discovery_evidence_sets
                   (evidence_set_id,run_id,evidence,evidence_set_sha256)
                   VALUES('evidence-completed','run-completed',
                          '{"legacy":true}'::jsonb,%s)""",
                ("1" * 64,),
            )
            cursor.execute(
                """INSERT INTO segment_first_letters_discovery_evidence_files
                   (evidence_set_id,file_order,relative_path,role,payload,
                    byte_count,sha256)
                   VALUES('evidence-completed',0,'legacy.bin',
                          'CANDIDATE_PROVIDER_RESPONSE',%s,12,%s)""",
                (b"v17-evidence", "2" * 64),
            )


def _postgres_lifecycle_shape(store: PostgresFleetStore) -> dict:
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT table_name,column_name
                     FROM information_schema.columns
                    WHERE table_schema=current_schema()
                      AND table_name IN (
                        'segment_first_letters_discovery_evidence_runs',
                        'segment_first_letters_discovery_executor_claims')
                      AND column_name IN (
                        'started_at','last_heartbeat_at','incomplete_at',
                        'incomplete_reason')
                    ORDER BY table_name,column_name"""
            )
            columns = [tuple(row.values()) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT c.relname,pg_get_constraintdef(k.oid) AS definition
                     FROM pg_constraint k
                     JOIN pg_class c ON c.oid=k.conrelid
                     JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname=current_schema()
                      AND c.relname IN (
                        'segment_first_letters_discovery_evidence_runs',
                        'segment_first_letters_discovery_executor_claims')
                      AND k.contype='c' AND pg_get_constraintdef(k.oid)
                          LIKE '%CONTROL_INCOMPLETE%'
                    ORDER BY c.relname"""
            )
            constraints = [tuple(row.values()) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT version FROM segment_schema_migrations ORDER BY version"
            )
            versions = [row["version"] for row in cursor.fetchall()]
    return {
        "columns": columns, "constraints": constraints,
        "versions": versions,
    }


def _assert_applied_contiguously_through_v18(versions: list[int]) -> None:
    """v18 owns its own head only; later migrations append past it."""
    assert versions == list(range(1, len(versions) + 1))
    assert versions[-1] >= 18


def _postgres_v17_data_snapshot(store: PostgresFleetStore) -> dict:
    snapshot = {}
    tables = {
        "segment_first_letters_discovery_evidence_runs": "run_id",
        "segment_first_letters_discovery_executor_claims": "claim_id",
        "segment_first_letters_discovery_evidence_sets": "evidence_set_id",
        "segment_first_letters_discovery_evidence_files": "file_order",
    }
    with store.connect() as connection:
        with connection.cursor() as cursor:
            for table, order_column in tables.items():
                cursor.execute(
                    f"""SELECT to_jsonb(row_value) - %s::text[] AS value
                          FROM {table} AS row_value
                         ORDER BY {order_column}""",
                    (list(LIFECYCLE_COLUMNS),),
                )
                snapshot[table] = [
                    row["value"] for row in cursor.fetchall()
                ]
    return snapshot


def test_live_postgres_literal_v17_to_v18_preserves_rows_and_repeats():
    with _isolated_postgres_store("task6_v18_upgrade") as store:
        import psycopg2

        _seed_postgres_v17_rows(store)
        before_rows = _postgres_v17_data_snapshot(store)
        store.initialize()
        first_shape = _postgres_lifecycle_shape(store)
        _assert_applied_contiguously_through_v18(first_shape["versions"])
        assert len(first_shape["columns"]) == 8
        assert len(first_shape["constraints"]) == 2
        assert all(
            all(state in definition for state in (
                "CLAIMED", "RUNNING", "COMPLETED", "CONTROL_INCOMPLETE",
            ))
            for _, definition in first_shape["constraints"]
        )
        assert _postgres_v17_data_snapshot(store) == before_rows
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT r.run_id,r.state,c.state AS claim_state,
                              e.evidence_set_id,f.payload
                         FROM segment_first_letters_discovery_evidence_runs r
                         JOIN segment_first_letters_discovery_executor_claims c
                           ON c.run_id=r.run_id
                         LEFT JOIN segment_first_letters_discovery_evidence_sets e
                           ON e.run_id=r.run_id
                         LEFT JOIN segment_first_letters_discovery_evidence_files f
                           ON f.evidence_set_id=e.evidence_set_id
                        ORDER BY r.run_id"""
                )
                rows = cursor.fetchall()
                assert [row["state"] for row in rows] == [
                    "CLAIMED", "COMPLETED",
                ]
                assert [row["claim_state"] for row in rows] == [
                    "CLAIMED", "COMPLETED",
                ]
                assert rows[1]["evidence_set_id"] == "evidence-completed"
                assert bytes(rows[1]["payload"]) == b"v17-evidence"
        with store.connect() as connection:
            with connection.cursor() as cursor:
                with pytest.raises(psycopg2.errors.ForeignKeyViolation):
                    cursor.execute(
                        "DELETE FROM "
                        "segment_first_letters_discovery_evidence_runs "
                        "WHERE run_id='run-claimed'"
                    )
                connection.rollback()
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_evidence_runs
                          SET state='RUNNING',started_at=now(),
                              last_heartbeat_at=now()
                        WHERE run_id='run-claimed'"""
                )
                cursor.execute(
                    """UPDATE segment_first_letters_discovery_executor_claims
                          SET state='RUNNING',started_at=now(),
                          last_heartbeat_at=now()
                        WHERE run_id='run-claimed'"""
                )
        for table in (
            "segment_first_letters_discovery_evidence_runs",
            "segment_first_letters_discovery_executor_claims",
        ):
            with store.connect() as connection:
                with connection.cursor() as cursor:
                    with pytest.raises(psycopg2.errors.CheckViolation):
                        cursor.execute(
                            f"UPDATE {table} SET state='INVALID' "
                            "WHERE run_id='run-claimed'"
                        )
                    connection.rollback()
        with store.connect() as connection:
            with connection.cursor() as cursor:
                for table in (
                    "segment_first_letters_discovery_evidence_runs",
                    "segment_first_letters_discovery_executor_claims",
                ):
                    cursor.execute(
                        f"""UPDATE {table}
                               SET state='CONTROL_INCOMPLETE',
                                   incomplete_at=now(),
                                   incomplete_reason='WORKER_LOST_AFTER_RUNNING'
                             WHERE run_id='run-claimed'"""
                    )
        store.initialize()
        assert _postgres_lifecycle_shape(store) == first_shape
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT state FROM segment_first_letters_discovery_evidence_runs "
                    "WHERE run_id='run-claimed'"
                )
                assert cursor.fetchone()["state"] == "CONTROL_INCOMPLETE"


def test_live_postgres_fresh_v18_initialize_and_repeat_are_idempotent():
    with _isolated_postgres_store("task6_v18_fresh") as store:
        store.initialize()
        first_shape = _postgres_lifecycle_shape(store)
        _assert_applied_contiguously_through_v18(first_shape["versions"])
        assert len(first_shape["columns"]) == 8
        assert len(first_shape["constraints"]) == 2
        assert all(
            all(state in definition for state in (
                "CLAIMED", "RUNNING", "COMPLETED", "CONTROL_INCOMPLETE",
            ))
            for _, definition in first_shape["constraints"]
        )
        store.initialize()
        assert _postgres_lifecycle_shape(store) == first_shape
