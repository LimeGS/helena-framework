"""A SELECT naming a column that does not exist fails only when it runs.

The control finally reached P4 and the geometry orientation proof answered

    HTTP 500 Internal Server Error

on a query, not a policy:

    psycopg.errors.UndefinedColumn: column "requested_by_job_id" does not exist
      job_store.py, in flattened_sheet

`surface_flattenings` has flattening_id, surface_id, profile_id, state,
artifact_uri, artifact_sha256, area_ratio, payload and created_at.
`requested_by_job_id` is a key inside the jsonb payload, not a column of its
own, so this read could never have succeeded -- P4 is simply the first boundary
any run ever reached, and it found the defect on its first attempt.

Nothing but a real server catches this. A test double answers whatever the code
asks it for, which is exactly how a query can name a column nobody ever created
and still look tested. So this is gated on HELENA_TEST_DSN and skipped -- not
quietly passed -- when there is no PostgreSQL to run against.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

DSN = os.environ.get("HELENA_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")

SURFACE = "19468577-4cdb-5f28-80b6-84c9346d8fdc"
PROFILE = "flatten-abf-v1@1.0.0"
JOB = "p3-90765882984e47"


@pytest.fixture()
def store():
    """An ink job store on its own schema, over the real segmentation tables."""
    from fleet.postgres_store import PostgresFleetStore
    from job_store import InkJobStore

    schema = f"sheet_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")

    class _Scoped(PostgresFleetStore):
        def connect(self):
            connection = super().connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
            return connection

    _Scoped(DSN).initialize()

    class _ScopedInk(InkJobStore):
        def _connect(self):
            connection = super()._connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}")
            return connection

    ink = _ScopedInk(DSN)
    ink.initialize()
    try:
        yield ink, _Scoped(DSN), schema
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


def _surface(fleet) -> None:
    """surface_flattenings has a foreign key onto segment_surfaces, so the sheet
    cannot exist without the surface it flattened."""
    fleet.register_snapshot({
        "source_snapshot_id": "snap-1", "sample_id": "PHerc0139",
        "ct_uri": "https://example.invalid/ct.zarr", "ct_sha256": "b" * 64,
        "m7_uri": "https://example.invalid/m7.zarr", "m7_sha256": "c" * 64,
        "shape_xyz": [64, 64, 64], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    with fleet.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO segment_surfaces(surface_id,source_snapshot_id,sample_id,"
            "owner,artifact_sha256,artifact_uri,bbox_xyz,sample_points,area_cm2,"
            "state,physical_qc_state,payload) VALUES"
            "(%s,'snap-1','PHerc0139','campaign-x',%s,'s3://b/s','{}'::jsonb,"
            "'[]'::jsonb,1.5,'GROWN','UNVALIDATED','{}'::jsonb)",
            (SURFACE, "d" * 64))


def _flattening(fleet, schema: str, *, state: str = "FLATTENED") -> None:
    payload = {
        "surface_id": SURFACE, "profile_id": PROFILE, "state": state,
        "requested_by_job_id": JOB,
        "artifact_sha256": "a" * 64, "receipt_sha256": "b" * 64,
        "artifact_uri": "s3://bucket/flat/" + SURFACE,
    }
    with fleet.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO surface_flattenings(flattening_id,surface_id,profile_id,"
            "state,artifact_uri,artifact_sha256,area_ratio,payload) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
            (str(uuid.uuid4()), SURFACE, PROFILE, state,
             payload["artifact_uri"], payload["artifact_sha256"], 0.95,
             json.dumps(payload)))


def test_the_sheet_reads_back_the_job_that_produced_it(store) -> None:
    """The read the orientation proof makes, against a real server."""
    ink, fleet, schema = store
    _surface(fleet)
    _flattening(fleet, schema)

    sheet = ink.flattened_sheet(SURFACE, PROFILE)

    assert sheet["requested_by_job_id"] == JOB, (
        "the proof matches this against the P3 job it was given; without it the "
        "sheet cannot be tied to the run that made it")
    assert sheet["artifact_sha256"] == "a" * 64
    assert sheet["state"] == "FLATTENED"


def test_a_failed_flattening_is_refused_rather_than_rendered(store) -> None:
    """Unchanged behaviour, kept honest: rendering a failed flattening renders
    whatever was left behind."""
    ink, fleet, schema = store
    _surface(fleet)
    _flattening(fleet, schema, state="FLATTENING_FAILED")

    with pytest.raises(RuntimeError, match="not FLATTENED"):
        ink.flattened_sheet(SURFACE, PROFILE)


def test_a_surface_nobody_flattened_says_so(store) -> None:
    ink, _fleet, _schema = store

    with pytest.raises(RuntimeError, match="no flattened sheet"):
        ink.flattened_sheet(SURFACE, PROFILE)
