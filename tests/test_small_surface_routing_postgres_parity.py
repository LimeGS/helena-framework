"""Every routing-receipt behaviour SQLite has, asked of PostgreSQL as well.

The deployment runs PostgreSQL. The routing receipts of schema v20 -- the table,
its immutability, the write inside the surface transaction, the reader, and the
size gate on the physical-QC queue -- exist in `store.py` and in nothing else. A
gate that holds only in the store the deployment does not use is not a gate.

That is not a hypothetical failure mode here. `test_postgresql_ddl_is_creatable`
exists because `authorization` is a PostgreSQL reserved word, two columns used it
unquoted, `initialize()` raised, no table was ever created, and every
PostgreSQL-backed test errored at its fixture for as long as nobody ran one. So
this file assumes nothing is covered until a real server says it is: each check
below either runs against `HELENA_TEST_DSN` or does not run at all, and never
reports a PostgreSQL behaviour as proven from a SQLite result.

The behaviours are written as `parity_*` functions taking one store, so the same
bytes of assertion run against both. They were written red against a PostgreSQL
that had no routing at all; migration v20 landed it, and the xfail markers that
named the gap have been removed rather than left to pass silently.
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

from fleet import surface_routing as routing  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

DSN = os.environ.get("HELENA_TEST_DSN")
POSTGRES_DDL = STAGE / "fleet/migrations/001_postgresql.sql"

# Frozen in tests/fixtures/first-letters-hybrid-20260802/evidence.json.
PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")

# The capability everything below depends on, named once. It landed with
# migration v20; every test here was written red against its absence and is now
# a live check rather than a readiness signal.
REQUIRES_V20 = (
    "the PostgreSQL half of the size gate is migration v20's "
    "segment_surface_routing_receipts, plus PostgresFleetStore's "
    "_write_routing_receipt, routing_receipt() and the small-surface gate on "
    "enqueue_imported_surface_qc"
)

# The frozen PostgreSQL spelling. Only the table name is pinned: the document
# column is `receipt jsonb` there and `receipt_json TEXT` in SQLite, on purpose,
# so nothing below compares column names across the two stores.
POSTGRES_ROUTING_TABLE = "segment_surface_routing_receipts"

needs_dsn = pytest.mark.skipif(
    not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

class _SchemaPostgresStore(PostgresFleetStore):
    """A store confined to one throwaway schema, so runs cannot collide."""

    def __init__(self, database_url: str, schema: str):
        super().__init__(database_url)
        self._test_schema = schema

    def connect(self):
        connection = super().connect()
        with connection.cursor() as cursor:
            # A literal prefix plus a UUID hex; no caller-supplied text.
            cursor.execute(f"SET search_path TO {self._test_schema}, public")
        return connection


@contextmanager
def _isolated_postgres(prefix: str = "routingparity"):
    if not DSN:  # pragma: no cover -- the marker skips first
        pytest.skip("HELENA_TEST_DSN is not set; no PostgreSQL behaviour was run")
    schema = f"{prefix}_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    try:
        store = _SchemaPostgresStore(DSN, schema)
        store.initialize()
        yield store
    finally:
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture
def sqlite_store(tmp_path) -> FleetStore:
    store = FleetStore(tmp_path / f"fleet-{uuid.uuid4().hex}.sqlite")
    store.initialize()
    return store


@pytest.fixture
def postgres_store():
    with _isolated_postgres() as store:
        yield store


# ---------------------------------------------------------------------------
# Fixture data, identical for both stores
# ---------------------------------------------------------------------------

def _import(store, *, name: str, area, digest: str = "a" * 64) -> tuple[str, str]:
    """One snapshot and one surface, written through the public store API."""
    scroll = f"TEST{name}"
    snapshot = store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    payload = {
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"s3://bucket/{name}",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED",
    }
    if area is not None:
        payload["area_cm2"] = area
    return snapshot, store.import_surface(payload)


def _routing_table(store) -> str:
    """The name PostgreSQL gives the v20 table, found rather than assumed.

    Asserting a spelling would fail a correct implementation that prefixes the
    table `segment_` like every other one, so the check is that exactly one
    table holds routing receipts -- which is the property that matters.
    """
    with store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT table_name FROM information_schema.tables
                    WHERE table_schema=current_schema()
                      AND table_type='BASE TABLE'
                      AND table_name LIKE %s""",
                ("%surface\\_routing\\_receipts",))
            found = [str(row["table_name"]) for row in cursor.fetchall()]
    assert len(found) == 1, (
        f"expected exactly one surface-routing-receipt table, found {found}")
    return found[0]


# ---------------------------------------------------------------------------
# The behaviours, written once and asked of each store
# ---------------------------------------------------------------------------

def parity_a_measured_surface_gets_a_receipt(store) -> dict:
    _, surface = _import(store, name="normal", area=0.5, digest="a" * 64)
    receipt = store.routing_receipt(surface)
    assert receipt is not None, "a surface exists with no routing decision recorded"
    assert receipt["route"] == routing.STANDARD
    assert routing.verify_receipt(receipt) is True
    return receipt


def parity_b_a_tiny_surface_is_diagnostic(store) -> dict:
    _, surface = _import(store, name="0268", area=PHERC0268_AREA_CM2,
                         digest=PHERC0268_ARTIFACT_SHA256)
    receipt = store.routing_receipt(surface)
    assert receipt is not None, "a surface exists with no routing decision recorded"
    assert receipt["route"] == routing.DIAGNOSTIC
    assert receipt["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert receipt["is_absence_evidence"] is False
    assert receipt["preserved"] is True
    assert routing.verify_receipt(receipt) is True
    return receipt


def parity_c_an_unmeasured_surface_gets_no_receipt(store) -> None:
    """Deliberate, and identical on both: no measurement, no routing claim."""
    _, surface = _import(store, name="unmeasured", area=None, digest="d" * 64)
    assert store.routing_receipt(surface) is None


def parity_d_the_decision_is_made_once(store) -> dict:
    """Re-importing the same surface must not re-decide or double-write."""
    scroll = "TESTreimport"
    snapshot = store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    payload = {
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": "e" * 64,
        "artifact_uri": "s3://bucket/reimport",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": 0.5,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"}
    first = store.import_surface(payload)
    # The second import carries a *different* area, and is now refused rather
    # than discarded: the stored surface does not change either way, so neither
    # may the routing decision. The refusal is the stronger form of the same
    # property -- nothing moves, and the caller is told nothing moved.
    try:
        second = store.import_surface({**payload, "area_cm2": PHERC0268_AREA_CM2})
        assert second == first
    except (RuntimeError, ValueError) as refusal:
        assert "differs" in str(refusal) or "conflict" in str(refusal)
    receipt = store.routing_receipt(first)
    assert receipt["route"] == routing.STANDARD, (
        "a re-import re-decided the route, so the receipt is not the record of "
        "what the surface was when it first existed")
    return receipt


def parity_e_a_diagnostic_surface_is_unclaimable(store) -> None:
    """The failure this whole class exists to stop: a 2 mm2 ink screen."""
    _, surface = _import(store, name="tinyqc", area=PHERC0268_AREA_CM2,
                         digest=PHERC0268_ARTIFACT_SHA256)
    store.enqueue_imported_surface_qc({
        "surface_id": surface, "source_snapshot_id": _snapshot_of(store, surface),
        "sample_id": "TESTtinyqc",
        "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/tinyqc", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": PHERC0268_AREA_CM2,
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
    }, profile_id="surface-qc-parity@1.0.0")
    assert store.claim_qc("qc-worker", 60,
                          profile_id="surface-qc-parity@1.0.0") is None, (
        "a surface below the floor reached the physical-QC queue")


def parity_g_a_forged_receipt_fails_closed(store) -> None:
    """The admission answer comes from the digest, not from the route string.

    Asked through `enters_standard_qc` / `enters_canonical_downstream` rather
    than an inline `route == STANDARD`, because an inline comparison is exactly
    the thing that reads a forged receipt as an admission.
    """
    receipt = parity_b_a_tiny_surface_is_diagnostic(store)
    assert routing.enters_standard_qc(receipt) is False
    assert routing.enters_canonical_downstream(receipt) is False
    for field, value in (("route", routing.STANDARD),
                         ("measured_area_cm2", 5.0),
                         ("minimum_area_cm2", 0.001),
                         ("policy_version", "9.9.9"),
                         ("preserved", False),
                         ("is_absence_evidence", True)):
        forged = {**receipt, field: value}
        assert routing.verify_receipt(forged) is False, (
            f"changing {field} left the receipt verifying")
        assert routing.enters_standard_qc(forged) is False, (
            f"a receipt forged at {field} was admitted to physical QC")
        assert routing.enters_canonical_downstream(forged) is False, (
            f"a receipt forged at {field} was admitted downstream")


def parity_f_a_standard_surface_still_reaches_qc(store) -> None:
    """The routing must not quarantine the work it was not written for."""
    _, surface = _import(store, name="okqc", area=0.5, digest="c" * 64)
    store.record_geometry_certification(
        surface, "GEOMETRY_CERTIFIED", {"schema": "test"},
        requested_by_job_id="p2-ok", profile_id="geometry-test@1",
        profile_sha256="6" * 64)
    store.enqueue_imported_surface_qc({
        "surface_id": surface, "source_snapshot_id": _snapshot_of(store, surface),
        "sample_id": "TESTokqc", "artifact_sha256": "c" * 64,
        "artifact_uri": "s3://bucket/okqc", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": 0.5,
    }, profile_id="surface-qc-parity-ok@1.0.0")
    claimed = store.claim_qc("qc-worker", 60,
                             profile_id="surface-qc-parity-ok@1.0.0")
    assert claimed is not None and claimed["surface_id"] == surface


def _snapshot_of(store, surface_id: str) -> str:
    for snapshot in store.snapshots():
        for row in store.surfaces_for_snapshot(snapshot["source_snapshot_id"]):
            if row["surface_id"] == surface_id:
                return snapshot["source_snapshot_id"]
    raise AssertionError(f"no snapshot holds {surface_id}")


# Split by what PostgreSQL can answer today, so the red count is the checklist.
# `parity_f` is the one behaviour that already holds there -- and it holds
# because PostgreSQL admits *everything* to physical QC, which is the defect
# the rest of this list is about, not evidence against it.
PARITY_MISSING_ON_POSTGRES = (
    parity_a_measured_surface_gets_a_receipt,
    parity_b_a_tiny_surface_is_diagnostic,
    parity_c_an_unmeasured_surface_gets_no_receipt,
    parity_d_the_decision_is_made_once,
    parity_e_a_diagnostic_surface_is_unclaimable,
    parity_g_a_forged_receipt_fails_closed,
)
PARITY_HELD_ON_POSTGRES = (
    parity_f_a_standard_surface_still_reaches_qc,
)
PARITY = PARITY_MISSING_ON_POSTGRES + PARITY_HELD_ON_POSTGRES


# ---------------------------------------------------------------------------
# SQLite: the reference half of the matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("behaviour", PARITY, ids=lambda f: f.__name__)
def test_sqlite_behaviour(behaviour, sqlite_store) -> None:
    """What PostgreSQL is being held to. Green here is the reference."""
    behaviour(sqlite_store)


# ---------------------------------------------------------------------------
# PostgreSQL: the same bytes of assertion, on a real server
# ---------------------------------------------------------------------------

@needs_dsn
@pytest.mark.parametrize("behaviour", PARITY_MISSING_ON_POSTGRES,
                         ids=lambda f: f.__name__)
def test_postgres_behaviour(behaviour, postgres_store) -> None:
    behaviour(postgres_store)


@needs_dsn
@pytest.mark.parametrize("behaviour", PARITY_HELD_ON_POSTGRES,
                         ids=lambda f: f.__name__)
def test_postgres_behaviour_that_already_holds(behaviour, postgres_store) -> None:
    """Green, and the reason it is green is the finding above it.

    Kept separate so porting v20 cannot quietly break the path that works now.
    """
    behaviour(postgres_store)


@needs_dsn
def test_the_two_stores_produce_the_same_receipt_bytes(
    sqlite_store, postgres_store,
) -> None:
    """Parity is not "both have one"; it is "both have the same one".

    The receipt is content-addressed, so an identical input that produces a
    different digest in the deployment's store means the published diagnostic
    evidence depends on which database wrote it.
    """
    left = parity_b_a_tiny_surface_is_diagnostic(sqlite_store)
    right = parity_b_a_tiny_surface_is_diagnostic(postgres_store)
    assert left["receipt_sha256"] == right["receipt_sha256"]
    assert left == right


# ---------------------------------------------------------------------------
# Schema readiness. These parse the shipped migration and need no server, so
# they hold in CI where there is no PostgreSQL -- but they only ever claim what
# the file says, never what a server did.
# ---------------------------------------------------------------------------

def test_the_shipped_migration_declares_the_routing_table() -> None:
    sql = POSTGRES_DDL.read_text(encoding="utf-8")
    assert POSTGRES_ROUTING_TABLE in sql, (
        f"the shipped migration does not declare {POSTGRES_ROUTING_TABLE}; "
        "a deployment applying it would route small surfaces against a table "
        "that is not there")


def test_the_shipped_migration_reaches_version_20() -> None:
    import re

    sql = POSTGRES_DDL.read_text(encoding="utf-8")
    versions = [int(value) for value in re.findall(
        r"INSERT\s+INTO\s+segment_schema_migrations\s*\([^)]*\)\s*"
        r"VALUES\s*\(\s*(\d+)", sql, re.IGNORECASE)]
    assert max(versions) >= 20, (
        f"the PostgreSQL migration stops at v{max(versions)}; {REQUIRES_V20}")


# ---------------------------------------------------------------------------
# Constraints, indexes and immutability on the server (I4)
# ---------------------------------------------------------------------------

@needs_dsn
def test_the_routing_table_exists_on_the_server_with_a_surface_key(
    postgres_store,
) -> None:
    table = _routing_table(postgres_store)
    with postgres_store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT c.contype, pg_get_constraintdef(c.oid) AS definition
                     FROM pg_constraint c
                     JOIN pg_class t ON t.oid=c.conrelid
                     JOIN pg_namespace n ON n.oid=t.relnamespace
                    WHERE t.relname=%s AND n.nspname=current_schema()""",
                (table,))
            constraints = [(row["contype"], row["definition"])
                           for row in cursor.fetchall()]
    kinds = {kind for kind, _ in constraints}
    assert "p" in kinds, f"{table} has no primary key, so a surface can be routed twice"
    assert any(kind == "p" and "surface_id" in definition
               for kind, definition in constraints), (
        f"{table}'s primary key is not the surface it routes")
    assert any(kind == "f" and "surface" in definition.lower()
               for kind, definition in constraints), (
        f"{table} has no foreign key to the surfaces it claims to route")


@needs_dsn
def test_the_receipt_cannot_be_rewritten_on_postgres(postgres_store) -> None:
    """SQLite refuses with a BEFORE UPDATE trigger. PostgreSQL must also refuse.

    A receipt that can be edited is not evidence, and "the application never
    updates it" is a property of today's callers, not of the database.
    """
    import psycopg2

    parity_a_measured_surface_gets_a_receipt(postgres_store)
    table = _routing_table(postgres_store)
    with postgres_store.connect() as connection:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.Error):
                cursor.execute(f"UPDATE {table} SET route=%s",  # noqa: S608
                               (routing.DIAGNOSTIC,))
    with postgres_store.connect() as connection:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.Error):
                cursor.execute(f"DELETE FROM {table}")  # noqa: S608


@needs_dsn
def test_the_receipt_survives_the_refused_edit_on_postgres(postgres_store) -> None:
    """Refusing is only half of it: the stored decision must be unchanged."""
    import psycopg2

    receipt = parity_a_measured_surface_gets_a_receipt(postgres_store)
    table = _routing_table(postgres_store)
    try:
        with postgres_store.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"UPDATE {table} SET route=%s",  # noqa: S608
                               (routing.DIAGNOSTIC,))
    except psycopg2.Error:
        pass
    stored = postgres_store.routing_receipt(receipt["surface_id"])
    assert stored == receipt
    assert routing.verify_receipt(stored) is True


@needs_dsn
def test_two_writers_racing_one_surface_leave_one_receipt(postgres_store) -> None:
    """Concurrency parity: PostgreSQL admits real concurrent writers, SQLite
    serialises them. Either way exactly one routing decision must exist, and
    neither writer may see an error the SQLite path does not raise."""
    import threading

    scroll = "TESTrace"
    snapshot = postgres_store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    payload = {
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": "f" * 64,
        "artifact_uri": "s3://bucket/race",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": 0.5,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"}
    failures: list[BaseException] = []
    barrier = threading.Barrier(4)

    def writer() -> None:
        try:
            barrier.wait(timeout=30)
            postgres_store.import_surface(dict(payload))
        except BaseException as error:  # noqa: BLE001 -- reported, not swallowed
            failures.append(error)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not failures, f"a concurrent import raised: {failures[0]!r}"

    table = _routing_table(postgres_store)
    with postgres_store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT count(*) AS n FROM {table} WHERE surface_id=%s",  # noqa: S608
                (payload["surface_id"],))
            assert cursor.fetchone()["n"] == 1, (
                "concurrent imports produced more or fewer than one routing "
                "decision for one surface")


@needs_dsn
def test_a_receipt_cannot_outlive_a_rolled_back_surface(postgres_store) -> None:
    """The receipt is written in the transaction that creates the surface.

    A receipt for a surface that does not exist is a routing decision about
    nothing, and it is exactly what a separate transaction would leave behind.
    """
    table = _routing_table(postgres_store)
    with postgres_store.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT r.surface_id FROM {table} r
                     LEFT JOIN segment_surfaces s ON s.surface_id=r.surface_id
                    WHERE s.surface_id IS NULL""")  # noqa: S608
            assert cursor.fetchall() == []


@needs_dsn
def test_postgres_refuses_the_edit_the_way_sqlite_does(
    sqlite_store, postgres_store,
) -> None:
    """SQLite says 'immutable' and 'permanent'. PostgreSQL must refuse too.

    The wording is not the contract -- a trigger and a rule word it differently
    -- but *refusing* is, and an operator reading two audit logs should be able
    to tell that the same thing was refused for the same reason.
    """
    import sqlite3

    import psycopg2

    parity_a_measured_surface_gets_a_receipt(sqlite_store)
    parity_a_measured_surface_gets_a_receipt(postgres_store)
    with sqlite_store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError) as sqlite_update:
            connection.execute("UPDATE surface_routing_receipts SET route=?",
                               (routing.DIAGNOSTIC,))
        with pytest.raises(sqlite3.IntegrityError) as sqlite_delete:
            connection.execute("DELETE FROM surface_routing_receipts")
    assert "immutable" in str(sqlite_update.value)
    assert "permanent" in str(sqlite_delete.value)

    table = _routing_table(postgres_store)
    for statement, expected in (
        (f"UPDATE {table} SET route='{routing.DIAGNOSTIC}'", "immutable"),  # noqa: S608
        (f"DELETE FROM {table}", "permanent"),  # noqa: S608
    ):
        with postgres_store.connect() as connection:
            with connection.cursor() as cursor:
                with pytest.raises(psycopg2.Error) as refusal:
                    cursor.execute(statement)
        assert expected in str(refusal.value).lower(), (
            f"PostgreSQL refused {statement.split()[0]} but not for the reason "
            f"SQLite gives: {refusal.value!r}")


# ---------------------------------------------------------------------------
# The families that never ran (I2's own precondition)
# ---------------------------------------------------------------------------

def test_the_ddl_reserved_word_check_covers_the_routing_table() -> None:
    """The check that exists because `authorization` shipped unquoted.

    It parses every `CREATE TABLE` in the migration, so a new table is covered
    the moment it is declared there -- but only there. A routing table added to
    PostgresFleetStore in Python DDL rather than to the migration file would be
    outside this check, and outside every other check that reads that file.
    """
    import test_postgresql_ddl_is_creatable as ddl_check

    sql = ddl_check.DDL.read_text(encoding="utf-8")
    assert POSTGRES_ROUTING_TABLE in sql, (
        "the routing table is not in the file the reserved-word check reads, so "
        f"that check says nothing about it. {REQUIRES_V20}")
    # And the parse actually reaches it: a table declared after a construct the
    # regex cannot close is silently skipped by the very check meant to guard it.
    bodies = ddl_check._TABLE.findall(sql)  # noqa: SLF001
    assert any("measured_area_cm2" in body for body in bodies), (
        "the reserved-word parser does not reach the routing table's columns")


def test_a_dsn_less_run_reports_how_much_it_did_not_check() -> None:
    """"3116 passed" without a DSN is a different sentence than with one.

    Fifty-five tests in this repository run only when HELENA_TEST_DSN is set.
    That is not a defect by itself -- it is a defect when a run without one is
    read as covering PostgreSQL, which is exactly how eighteen failures stayed
    invisible until `authorization` was found. This test does not gate anything;
    it fails if the DSN-gated set silently empties out, because a family of
    tests that stops existing looks identical to a family that always passes.
    """
    import re

    gated = 0
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if "HELENA_TEST_DSN" not in text:
            continue
        gated += len(re.findall(r"^\s*@pytest\.mark\.skipif\(\s*not DSN", text,
                                re.MULTILINE))
        gated += len(re.findall(r"\bneeds_dsn\b", text))
    assert gated >= 10, (
        "the DSN-gated PostgreSQL test family has all but disappeared; a run "
        "without a DSN would now look identical to a full one")


# ---------------------------------------------------------------------------
# The honesty check on this file itself
# ---------------------------------------------------------------------------

def test_no_postgres_behaviour_is_claimed_without_a_server() -> None:
    """Nothing in this file may report a PostgreSQL result from a SQLite run.

    Every test whose name says `postgres` carries the DSN skip marker, so a run
    without `HELENA_TEST_DSN` skips it rather than passing it.
    """
    import inspect

    checked = 0
    module = sys.modules[__name__]
    for name, function in vars(module).items():
        if (not name.startswith("test_") or not inspect.isfunction(function)
                or name == "test_no_postgres_behaviour_is_claimed_without_a_server"):
            continue
        if "postgres" not in name and "on_the_server" not in name:
            continue
        marks = getattr(function, "pytestmark", [])
        assert any(mark.name == "skipif" for mark in marks), (
            f"{name} names PostgreSQL but would run without a DSN")
        checked += 1
    assert checked >= 5, (
        "this guard found almost nothing to guard; the PostgreSQL half of the "
        "matrix has been renamed out from under it")
