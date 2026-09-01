"""An outage the fleet calls recoverable must not end the job.

The preflight worker already classifies a source outage as recoverable:

    # The sources did not answer, or answered unusably. May recover.
    "PREFLIGHT_SOURCE_UNAVAILABLE"

and then calls `fail_preflight`, which is terminal. The classification existed;
the consequence did not. So a dropped connection to S3 ended a seventy-minute
measurement and left a FAILED row that a human had to notice and re-drive.

The segmentation lane has had the answer to this the whole time --
`requeue_qc_unavailable`, `requeue_provider_unavailable`,
`requeue_source_unavailable` -- each bounded by `maximum_requeues` and delayed by
`retry_after`. The preflight lane has enqueue/claim/heartbeat/finalize/fail and
nothing between "it worked" and "it is over". This gives it the same shape as its
neighbours rather than a new one.

Bounded on purpose, in two places: the per-read retry in `fleet.retrying` covers
a blip inside one measurement, and this covers the outage that outlives the read.
Neither is unbounded -- a source that is genuinely gone still has to surface as
gone, and `maximum_requeues` is what keeps a requeue loop from hiding it forever.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.store import FleetStore  # noqa: E402

DSN = os.environ.get("HELENA_TEST_DSN")


REQUEST = {
    "schema": "campaignx.candidate_preflight_request.v1",
    "mission_id": "mission-control",
    "sample_id": "PHerc0139",
    "source_snapshot_id": "snap-1",
    "parameters": {"level": 2, "radius_l0_voxels": 64},
}


@pytest.fixture()
def store(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    return store


def _enqueue(store) -> str:
    return store.enqueue_candidate_preflight(dict(REQUEST))["preflight_job_id"]


def _receipt(detail: str = "ServerDisconnected") -> dict:
    return {"schema": "campaignx.preflight_outage.v1", "error": detail}


def test_an_outage_returns_the_job_to_pending(store) -> None:
    job_id = _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)

    outcome = store.requeue_preflight_source_unavailable(
        claim["preflight_job_id"], claim["lease_token"], _receipt(),
        retry_delay_seconds=30, maximum_requeues=3)

    assert outcome["status"] == "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE"
    row = store.preflight_job(job_id)
    assert row["state"] == "PENDING", "a recoverable outage ended the job"
    assert row["retry_after"], "the job would be re-claimed immediately"


def test_the_job_waits_before_it_can_be_taken_again(store) -> None:
    """Re-claiming instantly just re-reads a source that is still down."""
    _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)
    store.requeue_preflight_source_unavailable(
        claim["preflight_job_id"], claim["lease_token"], _receipt(),
        retry_delay_seconds=300, maximum_requeues=3)

    assert store.claim_preflight("worker-b", 60) is None


def test_the_job_is_taken_up_again_once_the_wait_is_over(store) -> None:
    _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)
    store.requeue_preflight_source_unavailable(
        claim["preflight_job_id"], claim["lease_token"], _receipt(),
        retry_delay_seconds=0, maximum_requeues=3)

    again = store.claim_preflight("worker-b", 60)
    assert again is not None, "the job never came back"
    assert again["request"] == REQUEST, "the frozen request did not survive"


def test_an_outage_that_does_not_lift_still_becomes_a_failure(store) -> None:
    """The half that matters as much: a bucket that is genuinely gone must not
    hide behind an endless requeue. The budget is what makes it surface."""
    job_id = _enqueue(store)
    for _ in range(2):
        claim = store.claim_preflight("worker-a", 60)
        assert claim is not None
        outcome = store.requeue_preflight_source_unavailable(
            claim["preflight_job_id"], claim["lease_token"], _receipt(),
            retry_delay_seconds=0, maximum_requeues=2)
        assert outcome["status"] == "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE"

    claim = store.claim_preflight("worker-a", 60)
    assert claim is not None
    spent = store.requeue_preflight_source_unavailable(
        claim["preflight_job_id"], claim["lease_token"], _receipt(),
        retry_delay_seconds=0, maximum_requeues=2)

    assert spent["status"] == "PREFLIGHT_SOURCE_UNAVAILABLE"
    row = store.preflight_job(job_id)
    assert row["state"] == "FAILED"
    assert row["reason_code"] == "PREFLIGHT_SOURCE_UNAVAILABLE"
    assert store.claim_preflight("worker-b", 60) is None


def test_the_attempts_stay_visible(store) -> None:
    """Preserve the queue history: an operator has to be able to see that this
    job has already been through an outage, not just its latest state."""
    job_id = _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)
    store.requeue_preflight_source_unavailable(
        claim["preflight_job_id"], claim["lease_token"], _receipt(),
        retry_delay_seconds=0, maximum_requeues=3)

    row = store.preflight_job(job_id)
    assert row["requeues"] == 1
    assert row["attempts"] >= 1


def test_a_stale_lease_cannot_requeue(store) -> None:
    _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)

    with pytest.raises(RuntimeError):
        store.requeue_preflight_source_unavailable(
            claim["preflight_job_id"], "not-the-token", _receipt(),
            retry_delay_seconds=0, maximum_requeues=3)


def test_the_outage_is_recorded_where_an_operator_reads_it(store) -> None:
    job_id = _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)
    store.requeue_preflight_source_unavailable(
        claim["preflight_job_id"], claim["lease_token"],
        _receipt("ServerDisconnected"), retry_delay_seconds=0,
        maximum_requeues=3)

    row = store.preflight_job(job_id)
    assert "ServerDisconnected" in str(row.get("detail") or row.get("receipt") or "")


def test_a_finished_preflight_is_not_requeued(store) -> None:
    """A COMPLETED measurement is a result; nothing may send it back."""
    _enqueue(store)
    claim = store.claim_preflight("worker-a", 60)
    store.finalize_preflight(claim["preflight_job_id"], claim["lease_token"],
                             {"schema": "campaignx.preflight_receipt.v1"})

    with pytest.raises(RuntimeError):
        store.requeue_preflight_source_unavailable(
            claim["preflight_job_id"], claim["lease_token"], _receipt(),
            retry_delay_seconds=0, maximum_requeues=3)


# The deployment runs PostgreSQL, so SQLite passing proves the contract and not
# the thing that will actually execute. The two stores have to behave the same.


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_the_deployed_store_requeues_an_outage_the_same_way() -> None:
    import uuid

    from fleet.postgres_store import PostgresFleetStore

    schema = f"outage_{uuid.uuid4().hex}"

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
        job_id = store.enqueue_candidate_preflight(dict(REQUEST))["preflight_job_id"]

        claim = store.claim_preflight("worker-a", 60)
        outcome = store.requeue_preflight_source_unavailable(
            claim["preflight_job_id"], claim["lease_token"], _receipt(),
            retry_delay_seconds=300, maximum_requeues=2)

        assert outcome["status"] == "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE"
        row = store.preflight_job(job_id)
        assert row["state"] == "PENDING"
        assert row["requeues"] == 1
        assert store.claim_preflight("worker-b", 60) is None, (
            "the delay does not hold on the store that will actually run")

        # And the bound is real here too.
        for _ in range(2):
            with store.connect() as connection, connection.cursor() as cursor:
                cursor.execute("UPDATE segment_preflight_jobs SET retry_after=NULL")
            claim = store.claim_preflight("worker-a", 60)
            assert claim is not None
            spent = store.requeue_preflight_source_unavailable(
                claim["preflight_job_id"], claim["lease_token"], _receipt(),
                retry_delay_seconds=0, maximum_requeues=2)
        assert spent["status"] == "PREFLIGHT_SOURCE_UNAVAILABLE"
        assert store.preflight_job(job_id)["state"] == "FAILED"
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_a_deployed_database_gains_the_retry_columns() -> None:
    """The columns are added in a version block, not the base block: a database
    that already records the top version replays nothing of the base block, so
    a column added there would never reach the deployment."""
    import uuid

    from fleet.postgres_store import PostgresFleetStore

    schema = f"outagecols_{uuid.uuid4().hex}"

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

        # Rewind to a deployment that has version 23 and neither column.
        with store.connect() as connection, connection.cursor() as cursor:
            cursor.execute("ALTER TABLE segment_preflight_jobs DROP COLUMN retry_after")
            cursor.execute("ALTER TABLE segment_preflight_jobs DROP COLUMN requeues")
            cursor.execute("DELETE FROM segment_schema_migrations WHERE version>23")
            cursor.execute("SELECT max(version) AS v FROM segment_schema_migrations")
            assert cursor.fetchone()["v"] == 23, "the rewind did not reach v23"

        store.initialize()

        with store.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT column_name FROM information_schema.columns
                    WHERE table_schema=%s AND table_name='segment_preflight_jobs'""",
                (schema,),
            )
            columns = {row["column_name"] for row in cursor.fetchall()}
        assert {"retry_after", "requeues"} <= columns, (
            "initialize() left a deployed database without the retry columns")
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")
