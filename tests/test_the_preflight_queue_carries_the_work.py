"""The preflight is work, so it goes in a queue.

It used to run inside the panel's HTTP handler. On the deployment that returned
503: the measurement asks M7 through the MCP seed service, which listens on
loopback for workers and holds a token the panel has neither of. The panel was
being asked to be a worker.

So it becomes a job: enqueued where the source lock is checked, executed where
the sources are reachable, and polled for. The lifecycle is `qc_jobs`', because
that queue already works and a second shape for the same idea is a second thing
to get wrong.

Two contract points this file settles, both raised by the worker that consumes
it. The claimed state is named CLAIMED. And a failure carries a detail beside
its reason code -- an operator reading FAILED and nothing else has to go to a
worker's stdout to learn why, which is exactly where evidence stops being
evidence.
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


def _request(**overrides) -> dict:
    request = {
        "mission_id": "control-fl-pherc0139",
        "sample_id": "PHerc0139",
        "source_snapshot_id": "snap-0139",
        "parameters": {"max_candidates": 8, "minimum_separation_voxels": 16},
    }
    request.update(overrides)
    return request


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _stores(tmp_path):
    """Both control planes, so the contract is one contract."""
    yield "sqlite", FleetStore(tmp_path / "fleet.sqlite")
    if DSN:
        from fleet.postgres_store import PostgresFleetStore

        yield "postgresql", PostgresFleetStore(DSN)


# -- enqueue -----------------------------------------------------------------

def test_a_job_is_enqueued_pending(store) -> None:
    out = store.enqueue_candidate_preflight(_request())
    assert out["state"] == "PENDING"
    assert out["created"] is True
    assert store.preflight_job(out["preflight_job_id"])["state"] == "PENDING"


def test_enqueue_is_idempotent_on_the_request(store) -> None:
    """An ambiguous POST is resolved by a readback, not by a second job.

    The panel client raises on a mutation whose response it could not read, and
    the caller is told to read state rather than retry. That is only safe if the
    second enqueue of the same request is the same job.
    """
    first = store.enqueue_candidate_preflight(_request())
    second = store.enqueue_candidate_preflight(_request())
    assert second["preflight_job_id"] == first["preflight_job_id"]
    assert second["created"] is False


def test_a_different_request_is_a_different_job(store) -> None:
    first = store.enqueue_candidate_preflight(_request())
    other = store.enqueue_candidate_preflight(
        _request(parameters={"max_candidates": 4, "minimum_separation_voxels": 16}))
    assert other["preflight_job_id"] != first["preflight_job_id"]


# -- claim -------------------------------------------------------------------

def test_claiming_hands_over_one_job_with_a_lease(store) -> None:
    enqueued = store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    assert claimed["preflight_job_id"] == enqueued["preflight_job_id"]
    assert claimed["lease_token"]
    assert claimed["request"]["sample_id"] == "PHerc0139"
    assert store.preflight_job(claimed["preflight_job_id"])["state"] == "CLAIMED"
    assert store.claim_preflight("worker-b", 60) is None


def test_an_expired_lease_returns_the_job_to_the_queue(store) -> None:
    store.enqueue_candidate_preflight(_request())
    first = store.claim_preflight("worker-a", 1)
    import time

    time.sleep(1.2)
    second = store.claim_preflight("worker-b", 60)
    assert second is not None
    assert second["preflight_job_id"] == first["preflight_job_id"]
    assert second["lease_token"] != first["lease_token"]


def test_a_heartbeat_needs_the_lease_that_holds_it(store) -> None:
    store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    store.heartbeat_preflight(claimed["preflight_job_id"], claimed["lease_token"], 60)
    with pytest.raises(Exception):
        store.heartbeat_preflight(claimed["preflight_job_id"], "not-the-token", 60)


# -- terminal ----------------------------------------------------------------

def test_finalizing_stores_the_receipt(store) -> None:
    store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    receipt = {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
               "raw_m7_cells": 8, "usable_cells": 0, "ink_used": False}
    store.finalize_preflight(claimed["preflight_job_id"], claimed["lease_token"], receipt)

    done = store.preflight_job(claimed["preflight_job_id"])
    assert done["state"] == "COMPLETED"
    assert done["receipt"] == receipt


def test_a_failure_carries_its_reason_and_its_detail(store) -> None:
    """FAILED alone sends an operator to a worker's stdout to learn why."""
    store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    store.fail_preflight(claimed["preflight_job_id"], claimed["lease_token"],
                         "PREFLIGHT_PROVIDER_NOT_CONFIGURED",
                         detail="VC_MCP_URL and VC_MCP_AUTH_TOKEN are required")

    failed = store.preflight_job(claimed["preflight_job_id"])
    assert failed["state"] == "FAILED"
    assert failed["reason_code"] == "PREFLIGHT_PROVIDER_NOT_CONFIGURED"
    assert "VC_MCP_URL" in failed["detail"]
    assert failed["attempts"] >= 1


def test_a_failed_job_does_not_block_a_fresh_attempt(store) -> None:
    """Idempotence is about not doing the same work twice, not about refusing
    to ever try again.

    The control enqueues the frozen request, so its digest is the same every
    run. A transient source outage failed the first job; the next run got that
    FAILED job handed back as its answer, read the stale reason code, and
    reported a boundary failure without anything having been measured. Nothing
    could ever have cleared it: the request is frozen, and a terminal job is
    never claimed again.
    """
    first = store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    store.fail_preflight(claimed["preflight_job_id"], claimed["lease_token"],
                         "PREFLIGHT_SOURCE_UNAVAILABLE", detail="the bucket refused")

    again = store.enqueue_candidate_preflight(_request())

    assert again["created"] is True
    assert again["state"] == "PENDING"
    assert again["preflight_job_id"] != first["preflight_job_id"]
    assert store.claim_preflight("worker-b", 60) is not None, (
        "the new attempt is not claimable, so nothing will run it")
    # The failure is still on the record: a retry is not an erasure.
    failed = store.preflight_job(first["preflight_job_id"])
    assert failed["state"] == "FAILED"
    assert failed["reason_code"] == "PREFLIGHT_SOURCE_UNAVAILABLE"
    assert "bucket" in (failed["detail"] or "")


def test_a_database_made_before_this_is_upgraded_without_losing_a_row(tmp_path) -> None:
    """`CREATE TABLE IF NOT EXISTS` does not remove a constraint from a table
    that is already there, and SQLite cannot drop one in place. A database made
    before the fix would keep the UNIQUE that made a failure permanent."""
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    FleetStore(path).initialize()
    with sqlite3.connect(path) as raw:
        raw.executescript(
            """DROP TABLE preflight_jobs;
CREATE TABLE preflight_jobs (
  preflight_job_id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL,
  sample_id TEXT NOT NULL,
  source_snapshot_id TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('PENDING','CLAIMED','COMPLETED','FAILED')),
  request_json TEXT NOT NULL,
  request_sha256 TEXT NOT NULL,
  worker_id TEXT, lease_token TEXT, lease_expires_at TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  receipt_json TEXT, reason_code TEXT, detail TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(mission_id, sample_id, request_sha256)
);
INSERT INTO preflight_jobs VALUES('old-1','control-fl-pherc0139','PHerc0139',
  'snap-0139','FAILED','{}','deadbeef',NULL,NULL,NULL,1,NULL,
  'PREFLIGHT_SOURCE_UNAVAILABLE','the bucket refused','t0','t1');""")

    upgraded = FleetStore(path)
    upgraded.initialize()

    kept = upgraded.preflight_job("old-1")
    assert kept["state"] == "FAILED"
    assert kept["reason_code"] == "PREFLIGHT_SOURCE_UNAVAILABLE"
    assert "bucket" in (kept["detail"] or "")
    # And the constraint that made the failure permanent is gone.
    first = upgraded.enqueue_candidate_preflight(_request())
    claimed = upgraded.claim_preflight("worker-a", 60)
    upgraded.fail_preflight(claimed["preflight_job_id"], claimed["lease_token"],
                            "PREFLIGHT_SOURCE_UNAVAILABLE", detail="again")
    again = upgraded.enqueue_candidate_preflight(_request())
    assert again["created"] is True
    assert again["preflight_job_id"] != first["preflight_job_id"]


def test_a_completed_job_still_answers_the_same_request(store) -> None:
    """The other terminal state is an answer, and re-asking must not re-measure."""
    first = store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    store.finalize_preflight(claimed["preflight_job_id"], claimed["lease_token"],
                             {"schema": "campaignx.segment_candidate_coverage_preflight.v1"})

    again = store.enqueue_candidate_preflight(_request())

    assert again["created"] is False
    assert again["preflight_job_id"] == first["preflight_job_id"]
    assert store.claim_preflight("worker-b", 60) is None


def test_a_terminal_job_is_not_claimed_again(store) -> None:
    store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    store.fail_preflight(claimed["preflight_job_id"], claimed["lease_token"],
                         "PREFLIGHT_SOURCE_UNAVAILABLE", detail="the bucket refused")
    assert store.claim_preflight("worker-b", 60) is None


def test_a_terminal_write_needs_the_lease(store) -> None:
    store.enqueue_candidate_preflight(_request())
    claimed = store.claim_preflight("worker-a", 60)
    with pytest.raises(Exception):
        store.finalize_preflight(claimed["preflight_job_id"], "not-the-token", {})
    assert store.preflight_job(claimed["preflight_job_id"])["state"] == "CLAIMED"


def test_an_unknown_job_reads_as_absent(store) -> None:
    assert store.preflight_job("no-such-job") is None


# -- parity ------------------------------------------------------------------

@pytest.fixture
def postgres() -> FleetStore:
    """A schema of its own, per test.

    `claim_preflight` takes the oldest pending job on the server -- it has no
    mission filter, because a worker takes whatever is next. Two tests sharing
    one schema claim each other's work, and this server outlives the suite, so
    yesterday's leftovers are in the queue too. A unique mission id is not
    enough: it makes the *rows* distinct without making the *queue* private.
    """
    import uuid

    if not DSN:
        pytest.skip("set HELENA_TEST_DSN to a throwaway PostgreSQL")
    from fleet.postgres_store import PostgresFleetStore

    schema = f"queue_{uuid.uuid4().hex}"

    class _Scoped(PostgresFleetStore):
        def connect(self):
            connection = super().connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
            return connection

    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")
    scoped = _Scoped(DSN)
    scoped.initialize()
    try:
        yield scoped
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_postgresql_carries_the_same_lifecycle(postgres) -> None:
    """The deployment runs PostgreSQL, and it is the store nobody had run."""
    request = _request()

    enqueued = postgres.enqueue_candidate_preflight(request)
    assert enqueued["state"] == "PENDING"
    assert postgres.enqueue_candidate_preflight(request)["created"] is False

    claimed = postgres.claim_preflight("worker-pg", 60)
    assert claimed["preflight_job_id"] == enqueued["preflight_job_id"]
    postgres.heartbeat_preflight(
        claimed["preflight_job_id"], claimed["lease_token"], 60)
    postgres.finalize_preflight(
        claimed["preflight_job_id"], claimed["lease_token"],
        {"schema": "campaignx.segment_candidate_coverage_preflight.v1",
         "ink_used": False})
    assert postgres.preflight_job(
        claimed["preflight_job_id"])["state"] == "COMPLETED"


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_postgresql_lets_a_failed_request_be_asked_again(postgres) -> None:
    """The half that was actually wrong on the deployment, on the deployed store.

    Excluding the failed row from the lookup is not enough on its own -- it has
    to stop occupying the constraint too, and only that half is a schema
    question.
    """
    request = _request()

    first = postgres.enqueue_candidate_preflight(request)
    claimed = postgres.claim_preflight("worker-pg", 60)
    postgres.fail_preflight(claimed["preflight_job_id"], claimed["lease_token"],
                            "PREFLIGHT_SOURCE_UNAVAILABLE", detail="the bucket refused")

    again = postgres.enqueue_candidate_preflight(request)

    assert again["created"] is True
    assert again["preflight_job_id"] != first["preflight_job_id"]
    assert postgres.claim_preflight("worker-pg-2", 60) is not None
    assert postgres.preflight_job(first["preflight_job_id"])["state"] == "FAILED"
