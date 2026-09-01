"""Deciding what the GPUs do first, without touching what they decide.

Two GTX 1660s do one surface QC an hour each: the frozen profile screens six
replicas of about nine thousand 64px tiles through a 26-frame TimeSformer. The
queue is FIFO, and forty-seven jobs for PHerc826 queued three days ago sat in
front of a development control that was needed now -- twenty-six GPU-hours of
somebody else's scroll.

The mechanism already existed. `claim_qc` skips a job whose `retry_after` is in
the future, and the job stays PENDING with its history, so deferring is not a
state anybody has to invent and not a row anybody has to move. What was missing
was a way to ask for it that leaves a record.

What this is not: it does not reorder anything, drop anything, or change any
verdict. A deferred job is claimed the moment it is released or its deferral
expires, and it is screened by the same frozen profile it always was. The only
thing that changes is which hour the GPU spends on it -- which is why the reason
and the principal are recorded rather than left to whoever remembers.
"""

from __future__ import annotations

import os
import sys
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.store import FleetStore  # noqa: E402

DSN = os.environ.get("HELENA_TEST_DSN")
LATER = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()


def _surface(store: FleetStore, surface_id: str, sample_id: str) -> None:
    """A surface with a QC job waiting on it, inserted the way QC finds them."""
    store.register_snapshot({
        "source_snapshot_id": "snap-1", "sample_id": sample_id,
        "ct_uri": "https://example.invalid/ct.zarr", "ct_sha256": "b" * 64,
        "m7_uri": "https://example.invalid/m7.zarr", "m7_sha256": "c" * 64,
        "shape_xyz": [64, 64, 64], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    with store.connect() as connection:
        connection.execute(
            "INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,"
            "artifact_sha256,artifact_uri,bbox_xyz_json,sample_points_json,"
            "area_cm2,state,physical_qc_state,geometry_qc_state,payload_json,"
            "created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (surface_id, "snap-1", sample_id, "campaign-x",
             hashlib.sha256(surface_id.encode()).hexdigest(),
             "s3://bucket/surface", "{}", "[]", 1.5, "GROWN", "UNVALIDATED",
             "GEOMETRY_UNMEASURED", "{}", "2026-08-03T06:00:00+00:00"))
        connection.execute(
            "INSERT INTO qc_jobs(qc_job_id,surface_id,profile_id,state,payload_json,"
            "created_at,updated_at) VALUES(?,?,?,'PENDING',?,?,?)",
            (f"qc-{surface_id}", surface_id, "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0",
             "{}", "2026-08-03T06:00:00+00:00", "2026-08-03T06:00:00+00:00"))
        connection.commit()


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    _surface(fleet, "surface-826-a", "PHerc826")
    _surface(fleet, "surface-826-b", "PHerc826")
    _surface(fleet, "surface-control", "PHerc0139")
    return fleet


def _claim_all(store: FleetStore) -> list[str]:
    """Every surface a worker would be handed, in the order it would get them."""
    seen = []
    while (claimed := store.claim_qc("worker-a", 60)) is not None:
        seen.append(claimed["surface"]["surface_id"])
    return seen


def test_without_a_deferral_the_oldest_goes_first(store) -> None:
    """The behaviour being changed, pinned first."""
    assert _claim_all(store)[0].startswith("surface-826")


def test_a_deferred_sample_is_skipped_and_the_next_one_runs(store) -> None:
    deferred = store.defer_qc_jobs(
        "PHerc826", until=LATER,
        reason="a development control needs the GPUs before a three-day-old backlog",
        by="tester")

    assert deferred["deferred"] == 2
    assert _claim_all(store) == ["surface-control"]


def test_a_deferred_job_is_still_pending_and_keeps_its_place(store) -> None:
    """Deferring is not dropping. Nothing is deleted, no state is invented, and
    the job is claimed the moment it comes back."""
    store.defer_qc_jobs("PHerc826", until=LATER, reason="control first", by="tester")
    assert _claim_all(store) == ["surface-control"]

    store.release_qc_jobs("PHerc826", by="tester")

    assert sorted(_claim_all(store)) == ["surface-826-a", "surface-826-b"]


def test_a_claimed_job_is_left_alone(store) -> None:
    """A worker holding a lease is not interrupted: deferral decides what starts
    next, never what is already running."""
    running = store.claim_qc("worker-busy", 600)
    assert running["surface"]["surface_id"].startswith("surface-826")

    deferred = store.defer_qc_jobs("PHerc826", until=LATER, reason="control first",
                                   by="tester")

    assert deferred["deferred"] == 1, "the running job was deferred out from under a worker"


def test_the_decision_is_on_the_record(store) -> None:
    """Which hour a GPU spends on whose scroll is a decision, and a decision
    nobody can read afterwards is indistinguishable from a queue that reordered
    itself."""
    store.defer_qc_jobs("PHerc826", until=LATER, reason="control first", by="lime")

    with store.connect() as connection:
        rows = connection.execute(
            "SELECT event_type,payload_json FROM events WHERE event_type LIKE 'qc.%'"
        ).fetchall()
    assert rows, "nothing recorded who deferred what, or why"
    payload = rows[0]["payload_json"]
    assert "PHerc826" in payload and "lime" in payload and "control first" in payload


def test_deferring_needs_a_reason(store) -> None:
    with pytest.raises(ValueError):
        store.defer_qc_jobs("PHerc826", until=LATER, reason="  ", by="tester")


def test_a_deferral_has_to_end(store) -> None:
    """An unbounded hold is a delete with better manners."""
    with pytest.raises(ValueError):
        store.defer_qc_jobs("PHerc826", until=None, reason="forever", by="tester")


def test_an_expired_deferral_needs_no_release(store) -> None:
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store.defer_qc_jobs("PHerc826", until=past, reason="briefly", by="tester")

    assert len(_claim_all(store)) == 3


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_postgresql_defers_the_same_way(tmp_path) -> None:
    """The deployment runs PostgreSQL, and this is a deployment decision."""
    import uuid

    from fleet.postgres_store import PostgresFleetStore

    schema = f"qcdefer_{uuid.uuid4().hex}"

    class _Scoped(PostgresFleetStore):
        def connect(self):
            connection = super().connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
            return connection

    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")
    try:
        postgres = _Scoped(DSN)
        postgres.initialize()
        # segment_surfaces has a foreign key onto the snapshots that SQLite does
        # not enforce, so the twin's fixture is not enough here.
        postgres.register_snapshot({
            "source_snapshot_id": "snap-1", "sample_id": "PHerc826",
            "ct_uri": "https://example.invalid/ct.zarr", "ct_sha256": "b" * 64,
            "m7_uri": "https://example.invalid/m7.zarr", "m7_sha256": "c" * 64,
            "shape_xyz": [64, 64, 64], "voxel_size_um": 9.362,
            "coordinate_frame": "ct_l0_xyz",
        })
        with postgres.connect() as connection, connection.cursor() as cursor:
            for surface_id, sample_id in (("s-826", "PHerc826"),
                                          ("s-ctl", "PHerc0139")):
                cursor.execute(
                    # physical_qc_state is NOT NULL here, unlike the SQLite
                    # twin, so this INSERT never ran until a DSN was present.
                    "INSERT INTO segment_surfaces(surface_id,source_snapshot_id,"
                    "sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz,"
                    "sample_points,area_cm2,state,physical_qc_state,payload) VALUES"
                    "(%s,%s,%s,'campaign-x',%s,'s3://b/s','{}'::jsonb,'[]'::jsonb,"
                    "1.5,'GROWN','UNVALIDATED','{}'::jsonb)",
                    # Distinct digests: (snapshot, artifact_sha256) is unique
                    # here, so two surfaces sharing one hash cannot both exist.
                    (surface_id, "snap-1", sample_id,
                     hashlib.sha256(surface_id.encode()).hexdigest()))
                cursor.execute(
                    "INSERT INTO segment_qc_jobs(qc_job_id,surface_id,profile_id,"
                    "state,payload) VALUES(%s,%s,'p','PENDING','{}'::jsonb)",
                    (f"qc-{surface_id}", surface_id))

        assert postgres.defer_qc_jobs(
            "PHerc826", until=LATER, reason="control first", by="tester",
        )["deferred"] == 1

        claimed = postgres.claim_qc("worker-pg", 60)
        assert claimed is not None
        assert claimed["surface"]["surface_id"] == "s-ctl"
        assert postgres.claim_qc("worker-pg-2", 60) is None
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")
