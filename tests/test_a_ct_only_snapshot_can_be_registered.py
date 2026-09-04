"""register_snapshot() required an m7 prediction for every source, and PHerc1667
has none to give it.

PHerc1667 has a community mesh, a published surface volume and a published ink
map, all at 2.399 um -- none of which need P1's spiral grow, and none of which
publish an m7 surface prediction. growable_scrolls() correctly excludes it
forever: P1 has nothing to screen against. But register_snapshot() was only
ever reachable from the P0 freeze flow, for a catalogued scroll or the one
pinned control cohort, and its own INSERT demanded `payload["m7_uri"]`,
`payload["shape_xyz"]` and `payload["voxel_size_um"]` -- a KeyError before the
database was ever asked, and NOT NULL columns behind that if it had not been.
PHerc1667's CT could not be registered at all, so scroll_has_a_source could
never have said yes to it no matter how that predicate was written.

This is the layer under panel/app.py's scroll_has_a_source: sample_id and
ct_uri are the only two fields a source snapshot has ever needed to exist, and
these tests ask both SQLite and PostgreSQL for that directly, plus the upgrade
that has to reach a database that already has the old, stricter table.
"""

from __future__ import annotations

import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
if str(STAGE) not in sys.path:
    sys.path.insert(0, str(STAGE))

from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

DSN = os.environ.get("HELENA_TEST_DSN")
needs_dsn = pytest.mark.skipif(
    not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")


@contextmanager
def _isolated_postgres(prefix: str = "ctonly"):
    if not DSN:  # pragma: no cover -- the marker skips first
        pytest.skip("HELENA_TEST_DSN is not set; no PostgreSQL behaviour was run")
    schema = f"{prefix}_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")

    class _ScopedStore(PostgresFleetStore):
        def connect(self):
            connection = super().connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
            return connection

    try:
        store = _ScopedStore(DSN)
        store.initialize()
        yield store
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


# ---------------------------------------------------------------------------
# Behaviours asked of both backends
# ---------------------------------------------------------------------------

def parity_a_ct_only_snapshot_registers(store) -> None:
    source_snapshot_id = store.register_snapshot({
        "sample_id": "PHerc1667",
        "ct_uri": "s3://vesuvius-challenge/PHerc1667/surface-volumes/"
                  "2.399um-0.22m-78keV-volume-20251217075048.zarr/",
    })
    assert source_snapshot_id
    rows = store.snapshots({"PHerc1667"})
    assert len(rows) == 1
    assert rows[0]["ct_uri"].startswith("s3://vesuvius-challenge/PHerc1667/")
    assert not rows[0].get("m7_uri")
    assert not rows[0].get("shape_xyz")
    assert rows[0].get("voxel_size_um") is None
    # Registering the identical payload again is the same source, not a second
    # row: register_snapshot is create-once, and that did not change here.
    assert store.register_snapshot({
        "sample_id": "PHerc1667",
        "ct_uri": "s3://vesuvius-challenge/PHerc1667/surface-volumes/"
                  "2.399um-0.22m-78keV-volume-20251217075048.zarr/",
    }) == source_snapshot_id
    assert len(store.snapshots({"PHerc1667"})) == 1


def parity_a_full_snapshot_still_registers(store) -> None:
    """Regression: a source with both fields must keep working exactly as
    bootstrap_sources and the control cohort require."""
    store.register_snapshot({
        "sample_id": "PHerc0268",
        "ct_uri": "s3://bucket/PHerc0268/ct.zarr",
        "m7_uri": "s3://bucket/PHerc0268/m7.zarr",
        "shape_xyz": [1024, 1024, 2048],
        "voxel_size_um": 7.91,
        "coordinate_frame": "ct_l0_xyz",
    })
    rows = store.snapshots({"PHerc0268"})
    assert rows[0]["ct_uri"] == "s3://bucket/PHerc0268/ct.zarr"
    assert rows[0]["m7_uri"] == "s3://bucket/PHerc0268/m7.zarr"
    assert rows[0]["shape_xyz"] == [1024, 1024, 2048]
    assert rows[0]["voxel_size_um"] == 7.91


def parity_ct_uri_is_still_required(store) -> None:
    with pytest.raises(ValueError):
        store.register_snapshot({"sample_id": "PHercNoCT"})
    with pytest.raises(ValueError):
        store.register_snapshot({"ct_uri": "s3://bucket/ct.zarr"})


def test_sqlite_accepts_a_ct_only_snapshot(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    parity_a_ct_only_snapshot_registers(store)


def test_sqlite_still_accepts_a_full_snapshot(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    parity_a_full_snapshot_still_registers(store)


def test_sqlite_still_requires_ct_uri(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    parity_ct_uri_is_still_required(store)


@needs_dsn
def test_postgres_accepts_a_ct_only_snapshot():
    with _isolated_postgres() as store:
        parity_a_ct_only_snapshot_registers(store)


@needs_dsn
def test_postgres_still_accepts_a_full_snapshot():
    with _isolated_postgres() as store:
        parity_a_full_snapshot_still_registers(store)


@needs_dsn
def test_postgres_still_requires_ct_uri():
    with _isolated_postgres() as store:
        parity_ct_uri_is_still_required(store)


# ---------------------------------------------------------------------------
# The upgrade itself: a database that predates this change
# ---------------------------------------------------------------------------

def test_sqlite_migrates_a_database_that_still_requires_m7(tmp_path):
    """A SQLite file initialized before this change has m7_uri, shape_xyz_json
    and voxel_size_um NOT NULL. `CREATE TABLE IF NOT EXISTS` will not drop that
    from a table that already exists, and SQLite cannot drop a NOT NULL in
    place -- the same reason `_migrate_preflight_retry` exists -- so this has
    to be reached the same way: rebuild the table, keep the rows.
    """
    path = tmp_path / "fleet.sqlite"
    store = FleetStore(path)
    store.initialize()
    with store.connect() as connection:
        connection.executescript(
            """BEGIN IMMEDIATE;
CREATE TABLE source_snapshots_old (
  source_snapshot_id TEXT PRIMARY KEY,
  sample_id TEXT NOT NULL,
  ct_uri TEXT NOT NULL,
  ct_sha256 TEXT,
  m7_uri TEXT NOT NULL,
  m7_sha256 TEXT,
  shape_xyz_json TEXT NOT NULL,
  voxel_size_um REAL NOT NULL,
  coordinate_frame TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
INSERT INTO source_snapshots_old SELECT * FROM source_snapshots;
DROP TABLE source_snapshots;
ALTER TABLE source_snapshots_old RENAME TO source_snapshots;
COMMIT;""")
        columns = {
            str(row["name"]): row
            for row in connection.execute("PRAGMA table_info(source_snapshots)")
        }
        assert columns["m7_uri"]["notnull"], "the rewind did not reach the old schema"

    # A fresh handle, the way a restarted process would open it.
    reopened = FleetStore(path)
    reopened.initialize()
    with reopened.connect() as connection:
        columns = {
            str(row["name"]): row
            for row in connection.execute("PRAGMA table_info(source_snapshots)")
        }
        assert not columns["m7_uri"]["notnull"], (
            "initialize() ran against a database with the old constraint and "
            "left it in place"
        )
    parity_a_ct_only_snapshot_registers(reopened)


DEPLOYED_VERSION = 26


@needs_dsn
def test_an_existing_postgres_database_gains_the_relaxation():
    """The upgrade on a server: a database recorded at version 26 -- m7_uri,
    shape_xyz and voxel_size_um still NOT NULL, the way gpu-1 was before this
    -- must end up accepting a CT-only source on the next initialize()."""
    with _isolated_postgres() as store:
        with store.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE segment_source_snapshots "
                "ALTER COLUMN m7_uri SET NOT NULL")
            cursor.execute(
                "DELETE FROM segment_schema_migrations WHERE version>%s",
                (DEPLOYED_VERSION,))
            cursor.execute(
                "SELECT max(version) AS v FROM segment_schema_migrations")
            assert cursor.fetchone()["v"] == DEPLOYED_VERSION, (
                "the rewind did not reach the state being simulated")

        store.initialize()
        parity_a_ct_only_snapshot_registers(store)
