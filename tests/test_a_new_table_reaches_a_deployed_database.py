"""A table added to the DDL has to arrive on databases that already exist.

`initialize()` reads the highest version the file records and, if the database
has recorded that version, replays nothing. So a table added to the *base*
block -- the statements before the first version row -- is created on fresh
databases and on no other. Tests pass, because tests build fresh databases.

That is not a hypothesis. `segment_preflight_jobs` went into the base block,
the deployment was already at version 21, and the preflight worker restarted
against `relation "segment_preflight_jobs" does not exist` until it was moved
into a version of its own.

The base block is frozen: it is what version 21 shipped. Anything new needs a
new version, which is the only thing an existing database will replay.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

DDL = ROOT / "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql"

# What the base block contained when the deployment recorded version 21. Not a
# style rule -- every name here is a table that existing databases already have,
# and a name added to this list is a table they will never get.
RELEASED_BASE_TABLES = {
    "segment_schema_migrations",
    "segment_source_snapshots",
    "segment_surfaces",
    "segment_tasks",
    "segment_campaign_budget_admissions",
    "segment_campaign_decisions",
    "segment_campaign_resume_authorizations",
    "segment_campaign_resume_principal_attestations",
    "segment_worker_capabilities",
    "segment_attempts",
    "segment_artifact_sets",
    "segment_qc_jobs",
    "segment_events",
    "human_review_events",
    "segment_probe_runs",
    "segment_probe_trials",
    "segment_probe_attempts",
    "segment_probe_artifact_sets",
    "segment_probe_evaluations",
    "segment_probe_decisions",
    "segment_probe_promotions",
    "surface_flattenings",
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
}


def _without_comments(sql: str) -> str:
    """A `--` line can contain the words CREATE TABLE and mean nothing by them.

    One does: a note reading "CREATE TABLE IF NOT EXISTS cannot add these
    bindings to an existing table". Scanning the raw text reports a table named
    `cannot`.
    """
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def _base_block_tables() -> set[str]:
    sql = _without_comments(DDL.read_text())
    base = sql[: sql.index("INSERT INTO segment_schema_migrations")]
    return set(re.findall(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\w+)", base))


def test_the_scan_finds_the_base_block_it_is_meant_to_check() -> None:
    """A check that matches nothing passes forever."""
    assert len(_base_block_tables()) > 30
    assert "segment_surfaces" in _base_block_tables()


def test_no_table_was_added_to_the_block_deployed_databases_skip() -> None:
    added = _base_block_tables() - RELEASED_BASE_TABLES
    assert not added, (
        f"{sorted(added)} sit in the base block, which every database at the "
        "file's top version replays nothing of. Move them to a new "
        "`INSERT INTO segment_schema_migrations` block at the end of the file "
        "so existing deployments create them."
    )


def test_nothing_released_was_renamed_or_dropped() -> None:
    """The other direction: existing databases keep the table under its old
    name whatever the file now says, so a rename here is a silent divergence."""
    missing = RELEASED_BASE_TABLES - _base_block_tables()
    assert not missing, f"{sorted(missing)} disappeared from the base block"


DSN = os.environ.get("HELENA_TEST_DSN")


# The version the deployment had recorded when its preflight worker could not
# find the table. A constant, not `max(version)`: rewinding to "one below the
# top" replays the whole file, and the base block recreates everything -- so the
# simulation would pass with the table in the base block, which is the bug. The
# state that tells the two apart is the real one: this version recorded, the
# table absent.
DEPLOYED_VERSION = 21


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_an_existing_database_gains_the_preflight_queue() -> None:
    """The upgrade itself, on a server: a database in the state gpu-1 was in
    must end up with the table. This is the check the static ones stand in for."""
    import uuid

    from fleet.postgres_store import PostgresFleetStore

    schema = f"upgrade_{uuid.uuid4().hex}"

    class _ScopedStore(PostgresFleetStore):
        def connect(self):
            connection = super().connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
            return connection

    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")
    try:
        store = _ScopedStore(DSN)
        store.initialize()

        # Rewind to the deployment's state: every table of version 21 present,
        # nothing newer recorded, the queue absent.
        with store.connect() as connection, connection.cursor() as cursor:
            cursor.execute("DROP TABLE segment_preflight_jobs")
            cursor.execute("DELETE FROM segment_schema_migrations WHERE version>%s",
                           (DEPLOYED_VERSION,))
            cursor.execute("SELECT max(version) AS v FROM segment_schema_migrations")
            assert cursor.fetchone()["v"] == DEPLOYED_VERSION, (
                "the rewind did not reach the state being simulated")

        store.initialize()

        with store.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('segment_preflight_jobs') IS NOT NULL AS there")
            assert cursor.fetchone()["there"], (
                "initialize() ran against a database missing the queue and left "
                "it missing -- which is what the deployment did"
            )
        # And the worker's first call against it works.
        store.enqueue_candidate_preflight({
            "mission_id": "upgrade-check", "sample_id": "PHerc0139",
            "source_snapshot_id": "snap", "parameters": {},
        })
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")
