from __future__ import annotations

import os
import re
import inspect
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
FLEET = STAGE / "fleet"
sys.path.insert(0, str(STAGE))

from fleet.store import FleetStore
from fleet.postgres_store import PostgresFleetStore
from fleet.store_factory import open_fleet_store


BRIDGE_TABLES = (
    "first_letters_discovery_native_adapters_v19",
    "first_letters_discovery_dispatches_v19",
    "first_letters_discovery_jobs_v19",
    "first_letters_discovery_history_reconciliations_v19",
    "first_letters_discovery_historical_imports_v19",
)
BRIDGE_INDEXES = (
    "segment_first_letters_discovery_history_by_mission_v19",
    "segment_first_letters_discovery_jobs_ready_v19",
)
V19_MARKER = "-- V19 adds the immutable, executable shadow graph."
DSN = os.environ.get("HELENA_TEST_DSN")


def _postgres_sql() -> str:
    return (FLEET / "migrations/001_postgresql.sql").read_text(
        encoding="utf-8"
    )


def _declared_versions(sql: str) -> list[int]:
    return [int(value) for value in re.findall(
        r"INSERT\s+INTO\s+segment_schema_migrations\s*\([^)]*\)\s*"
        r"VALUES\s*\(\s*(\d+)", sql, re.IGNORECASE,
    )]


def _postgres_sql_through_v18() -> str:
    """Exactly what a v18 PostgreSQL deployment ran, from the same file.

    Cutting the shipped migration at the v19 marker is the only honest way to
    build a v18 database: a hand-written copy would prove that the upgrade
    works against the copy, not against what v18 deployments actually have.
    """
    head, marker, _tail = _postgres_sql().partition(V19_MARKER)
    assert marker, "the v19 section marker moved; this fixture cuts nothing"
    assert _declared_versions(head) == list(range(1, 19))
    return head + "\nCOMMIT;\n"


class _SchemaPostgresStore(PostgresFleetStore):
    def __init__(self, database_url: str, schema: str):
        super().__init__(database_url)
        self._test_schema = schema

    def connect(self):
        connection = super().connect()
        with connection.cursor() as cursor:
            # The name is a literal prefix plus a UUID hex.
            cursor.execute(f"SET search_path TO {self._test_schema}, public")
        return connection


@contextmanager
def _isolated_postgres_store(prefix: str):
    if not DSN:
        pytest.skip(
            "HELENA_TEST_DSN is not set; the real v18-to-v19 upgrade was "
            "not run"
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


def _postgres_bridge_shape(store: PostgresFleetStore) -> dict:
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT table_name FROM information_schema.tables
                    WHERE table_schema=current_schema()
                      AND table_type='BASE TABLE'
                      AND table_name LIKE %s
                    ORDER BY table_name""",
                ("segment\\_first\\_letters\\_discovery\\_%\\_v19",),
            )
            tables = [str(row["table_name"]) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT indexname FROM pg_indexes
                    WHERE schemaname=current_schema()
                      AND indexname LIKE %s
                    ORDER BY indexname""",
                ("segment\\_first\\_letters\\_discovery\\_%\\_v19",),
            )
            indexes = [str(row["indexname"]) for row in cursor.fetchall()]
            cursor.execute(
                """SELECT c.relname,pg_get_constraintdef(k.oid) AS definition
                     FROM pg_constraint k
                     JOIN pg_class c ON c.oid=k.conrelid
                     JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname=current_schema()
                      AND c.relname LIKE 'segment\\_%\\_v19'
                      AND k.contype IN ('u','f','c','p')
                    ORDER BY c.relname,definition"""
            )
            constraints = [tuple(row.values()) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT version FROM segment_schema_migrations ORDER BY version"
            )
            versions = [int(row["version"]) for row in cursor.fetchall()]
    return {
        "tables": tables, "indexes": indexes, "constraints": constraints,
        "versions": versions,
    }


def test_live_postgres_v18_database_upgrades_to_the_v19_bridge_and_repeats():
    with _isolated_postgres_store("task6_v19_upgrade") as store:
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(_postgres_sql_through_v18())
                cursor.execute(
                    """INSERT INTO segment_source_snapshots
                       (source_snapshot_id,sample_id,ct_uri,m7_uri,shape_xyz,
                        voxel_size_um,coordinate_frame,payload)
                       VALUES('preserved-v18','PHercV18','fixture://ct',
                              'fixture://m7','[1,1,1]'::jsonb,9.0,'ct_l0_xyz',
                              '{"source_snapshot_id":"preserved-v18"}'::jsonb)"""
                )
        before = _postgres_bridge_shape(store)
        assert before["versions"] == list(range(1, 19))
        assert before["tables"] == [] and before["indexes"] == []

        store.initialize()

        upgraded = _postgres_bridge_shape(store)
        assert upgraded["versions"] == _declared_versions(_postgres_sql())
        # The bridge arrived, and everything before it did too. Not "19 is the
        # last version there is": the line above already binds the database to
        # whatever the file declares, so pinning the ceiling here only makes
        # every later migration look like a v19 regression.
        assert 19 in upgraded["versions"]
        assert upgraded["versions"][:19] == list(range(1, 20))
        assert upgraded["tables"] == sorted(
            f"segment_{table}" for table in BRIDGE_TABLES
        )
        assert set(BRIDGE_INDEXES) <= set(upgraded["indexes"])
        with store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT sample_id FROM segment_source_snapshots "
                    "WHERE source_snapshot_id='preserved-v18'"
                )
                assert cursor.fetchone()["sample_id"] == "PHercV18"
                cursor.execute(
                    "SELECT COUNT(*) AS rows FROM "
                    "segment_first_letters_discovery_jobs_v19"
                )
                assert cursor.fetchone()["rows"] == 0

        store.initialize()
        assert _postgres_bridge_shape(store) == upgraded


def test_live_postgres_fresh_v19_install_matches_the_upgraded_bridge():
    """Fresh creation and v18 upgrade have to land on the same schema."""
    with _isolated_postgres_store("task6_v19_fresh") as fresh:
        fresh.initialize()
        fresh_shape = _postgres_bridge_shape(fresh)
        with _isolated_postgres_store("task6_v19_upgraded") as upgraded:
            with upgraded.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(_postgres_sql_through_v18())
            upgraded.initialize()
            assert _postgres_bridge_shape(upgraded) == fresh_shape
    assert fresh_shape["tables"] == sorted(
        f"segment_{table}" for table in BRIDGE_TABLES
    )


def test_live_postgres_and_sqlite_agree_on_the_v19_bridge_tables(tmp_path):
    sqlite_store = FleetStore(tmp_path / "fleet.sqlite")
    sqlite_store.initialize()
    with sqlite3.connect(sqlite_store.path) as connection:
        sqlite_bridge = sorted(
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE '%\\_v19' ESCAPE '\\'"
            )
        )
    with _isolated_postgres_store("task6_v19_parity") as store:
        store.initialize()
        postgres_bridge = _postgres_bridge_shape(store)["tables"]
    assert sqlite_bridge == sorted(BRIDGE_TABLES)
    assert postgres_bridge == [f"segment_{table}" for table in sqlite_bridge]


def test_fresh_and_repeated_sqlite_v19_initialize_is_idempotent(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    store.initialize()
    with sqlite3.connect(store.path) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        counts = {
            table: connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            for table in BRIDGE_TABLES
        }
    assert set(BRIDGE_TABLES) <= tables
    assert counts == {table: 0 for table in BRIDGE_TABLES}


def test_loaded_postgres_migration_appends_literal_v19_bridge_contract():
    sql = (FLEET / "migrations/001_postgresql.sql").read_text(
        encoding="utf-8"
    )
    versions = [int(value) for value in re.findall(
        r"INSERT\s+INTO\s+segment_schema_migrations\s*\([^)]*\)\s*"
        r"VALUES\s*\(\s*(\d+)", sql, re.IGNORECASE,
    )]
    # The first nineteen, in order: this is about v19 arriving literally and
    # after everything before it, not about v19 being the last migration there
    # will ever be. The ceiling is pinned by test_migration_sentinel, and
    # consecutiveness by test_every_declared_version_is_consecutive_from_one.
    assert versions[:19] == list(range(1, 20))
    for table in BRIDGE_TABLES:
        assert f"segment_{table}" in sql
    assert "VALUES (19, 'first-letters native shadow execution bridge')" in sql


def test_sqlite_v19_bridge_foreign_keys_and_uniqueness_are_literal(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    with store.connect() as connection:
        dispatch_indexes = connection.execute(
            "PRAGMA index_list(first_letters_discovery_dispatches_v19)"
        ).fetchall()
        job_indexes = connection.execute(
            "PRAGMA index_list(first_letters_discovery_jobs_v19)"
        ).fetchall()
        job_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(first_letters_discovery_jobs_v19)"
        ).fetchall()
    assert any(row[2] for row in dispatch_indexes)
    assert sum(bool(row[2]) for row in job_indexes) >= 3
    assert {row[2] for row in job_foreign_keys} >= {
        "first_letters_discovery_dispatches_v19",
        "first_letters_discovery_compute_reservations",
    }


def test_postgres_v19_public_surface_is_job_rooted_and_has_no_history_input():
    reserve = inspect.signature(
        PostgresFleetStore.reserve_discovery_compute
    ).parameters
    claim = inspect.signature(
        PostgresFleetStore.claim_first_letters_discovery_job
    ).parameters
    assert "historical_execution_manifest" not in reserve
    assert set(claim) == {"self", "job_id", "lease_seconds"}
    for name in (
        "reserve_first_letters_baseline_shadow",
        "reserve_first_letters_alternative_shadow",
        "read_first_letters_discovery_request",
        "revalidate_first_letters_discovery_job_claim",
    ):
        assert callable(getattr(PostgresFleetStore, name))


def test_postgres_direct_generic_native_work_fails_before_database_access():
    store = PostgresFleetStore("postgresql://never-connect")
    for kind in ("BASELINE_ARM", "ALTERNATIVE_SOURCE_ARM"):
        try:
            store.reserve_discovery_compute(
                mission_id="mission-a", request_id="request-a",
                work_kind=kind, work_authority={}, work_authority_id="work-a",
                work_authority_sha256="0" * 64, ordered_item_ids=["cell-a"],
                cap_authority_id="cap-a", cap_authority_sha256="1" * 64,
            )
        except ValueError as error:
            assert str(error) == "DISCOVERY_NATIVE_PRODUCER_REQUIRED"
        else:
            raise AssertionError("synthetic native authority reached PostgreSQL")


def test_store_factory_wires_only_configured_server_owned_authority(
    tmp_path, monkeypatch,
):
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(b'{"server":"owned"}')
    arm_path = tmp_path / "arms.json"
    arm_path.write_text(
        '{"arm-a":{"arm_id":"arm-a","server":"owned"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "HELENA_FIRST_LETTERS_DISCOVERY_PROFILE_PATH", str(profile_path)
    )
    monkeypatch.setenv(
        "HELENA_FIRST_LETTERS_EXPERIMENTAL_ARMS_PATH", str(arm_path)
    )

    configured = open_fleet_store(tmp_path / "configured.sqlite")
    assert configured._first_letters_discovery_profile_resolver(
        "mission-a", "source-a"
    ) == b'{"server":"owned"}'
    assert configured._first_letters_experimental_arm_resolver("arm-a") == {
        "arm_id": "arm-a", "server": "owned",
    }

    monkeypatch.delenv("HELENA_FIRST_LETTERS_DISCOVERY_PROFILE_PATH")
    monkeypatch.delenv("HELENA_FIRST_LETTERS_EXPERIMENTAL_ARMS_PATH")
    unconfigured = open_fleet_store(tmp_path / "unconfigured.sqlite")
    assert unconfigured._first_letters_discovery_profile_resolver is None
    assert unconfigured._first_letters_experimental_arm_resolver is None
