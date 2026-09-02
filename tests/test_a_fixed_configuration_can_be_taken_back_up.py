"""A configuration that has been fixed must be able to move the queue again.

Blocking a QC job on a configuration error is terminal on purpose: a wrong
HELENA_QC_PROFILE_SHA256 fails identically on every retry, and the fleet once
spent two days proving that -- 3118 receipts, zero surfaces measured. What was
missing was the other half. Once the operator fixed the pin there was no way
back.

Measured on a clean deployment: a stale profile pin blocked the only QC job;
after the pin was corrected the job stayed BLOCKED_CONFIGURATION with its error
still quoting the old hash, so a fixed deployment read as unfixed. Nothing
downstream could run, and P3 then found no CT-supported surface and reported
success having flattened nothing -- four phases from the actual cause. Getting
out of it took an UPDATE typed into psql.

So this file holds two things in place at once, and they pull against each
other:

  * a person who fixed something can put the jobs back, through an endpoint that
    knows who asked and what they fixed;
  * nothing else can. No claim, no sweep, no timer moves a blocked job. The
    terminal state is what stopped the spin, and a requeue that anything but a
    person can trigger is the spin with a new name.

The stale error is the third: it is composed from `result`, so a requeue that
leaves the blocked attempt's receipt in place reproduces exactly the symptom
that sent the operator four phases away from the cause.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

DSN = os.environ.get("HELENA_TEST_DSN")

STORES = {
    name: (ROOT / f"framework/stages/01-segmentation/fleet/{name}.py").read_text()
    for name in ("postgres_store", "store")
}


# --------------------------------------------------------------------------
# The store: a blocked job can be put back, by somebody who says what they fixed
# --------------------------------------------------------------------------

def _blocked_sqlite_qc(tmp_path, suffix, error="surface-QC profile hash differs"):
    """One QC job, claimed and then blocked the way a misconfigured worker does.

    Through the real store rather than an INSERT, because the state under test is
    the one `block_qc_configuration` leaves behind -- including the receipt it
    writes, which is what the requeue then has to clear.
    """
    from fleet.store import FleetStore

    store = FleetStore(tmp_path / f"fleet-{suffix}.sqlite")
    store.initialize()
    source_id = store.register_snapshot({
        "sample_id": "PHercFIXED",
        "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64,
        "shape_xyz": [32, 32, 32],
        "voxel_size_um": 1.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    store.enqueue_imported_surface_qc(
        {
            "surface_id": f"surface-{suffix}",
            "source_snapshot_id": source_id,
            "sample_id": "PHercFIXED",
            "artifact_sha256": "2" * 64,
            "artifact_uri": f"fixture://surface-{suffix}",
            "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
            # A QC enqueue needs a routing decision, and an unmeasured surface
            # has none. The subject here is the job's state, not what a surface
            # may be, so the fixture supplies the measurement rather than the
            # boundary being relaxed for it.
            "area_cm2": 0.5,
            "geometry_qc_state": "GEOMETRY_CERTIFIED",
        },
        profile_id="requeue-test@1.0.0",
    )
    claim = store.claim_qc("gpu-1", 60)
    assert claim is not None
    store.block_qc_configuration(claim["qc_job_id"], claim["lease_token"], {
        "schema": "campaignx.test.raw_qc_receipt.v1",
        "status": "BLOCKED_CONFIGURATION",
        "qc_job_id": claim["qc_job_id"],
        "surface_id": claim["surface_id"],
        "error": error,
        "no_scientific_conclusion": True,
    })
    return store, claim["qc_job_id"]


def _job(store, qc_job_id):
    with store.connect() as connection:
        return connection.execute(
            "SELECT * FROM qc_jobs WHERE qc_job_id=?", (qc_job_id,)).fetchone()


def test_a_blocked_job_can_be_taken_back_up_once_the_pin_is_fixed(tmp_path):
    """The gap itself: before this, the only way out was psql."""
    store, qc_job_id = _blocked_sqlite_qc(tmp_path, "back-up")
    assert store.claim_qc("gpu-1", 60) is None, "the fixture is not blocked"

    record = store.requeue_blocked_qc_jobs(
        "PHercFIXED", fixed="corrected HELENA_QC_PROFILE_SHA256 on gpu-1",
        by="tester")

    assert record["requeued"] == 1
    assert record["qc_job_ids"] == [qc_job_id]
    claim = store.claim_qc("gpu-1", 60)
    assert claim is not None and claim["qc_job_id"] == qc_job_id, (
        "the job went back to PENDING and no worker can take it, which is the "
        "blocked state with a friendlier name"
    )


def test_the_requeue_does_not_reach_another_scroll(tmp_path):
    """Scope is the whole request. A requeue that unblocked everything would be
    the psql UPDATE again, with a login page in front of it."""
    store, qc_job_id = _blocked_sqlite_qc(tmp_path, "scope")

    record = store.requeue_blocked_qc_jobs(
        "PHerc1667", fixed="fixed the pin on the other scroll", by="tester")

    assert record["requeued"] == 0
    assert record["qc_job_ids"] == []
    assert _job(store, qc_job_id)["state"] == "BLOCKED_CONFIGURATION"


def test_the_stale_error_does_not_survive_the_requeue(tmp_path):
    """The symptom that cost four phases of searching.

    `error` and `last_status` are composed from `result`; there are no columns
    for them. A requeue that moves the state and leaves the receipt shows a
    corrected deployment quoting the hash it no longer pins, which reads as a
    fix that did not take.
    """
    pytest.importorskip("fastapi")
    from panel.app import _qc_diagnostic_fields

    store, qc_job_id = _blocked_sqlite_qc(
        tmp_path, "stale", error="surface-QC profile hash differs: expected dead00ff")
    blocked = json.loads(_job(store, qc_job_id)["result_json"])
    assert _qc_diagnostic_fields(blocked)["error"].endswith("dead00ff"), (
        "the fixture does not reproduce the error an operator would read"
    )

    store.requeue_blocked_qc_jobs(
        "PHercFIXED", fixed="repinned the profile", by="tester")

    result_json = _job(store, qc_job_id)["result_json"]
    assert result_json is None, (
        "the blocked attempt's receipt is still the job's result, and the panel "
        "composes a job's error and last_status out of exactly that, so a queued "
        "job goes on reporting the hash the deployment no longer pins"
    )
    reported = _qc_diagnostic_fields(json.loads(result_json) if result_json else None)
    assert reported["error"] is None and reported["last_status"] is None

    # Cleared from the job, not from the record. The receipt is the evidence
    # that a worker refused, and the requeue is the evidence somebody acted.
    with store.connect() as connection:
        events = {row["event_type"]: json.loads(row["payload_json"])
                  for row in connection.execute(
                      "SELECT event_type,payload_json FROM events").fetchall()}
    assert "dead00ff" in events["QC_BLOCKED_CONFIGURATION"]["error"]
    assert events["qc.requeued_after_fix"]["fixed"] == "repinned the profile"
    assert events["qc.requeued_after_fix"]["by"] == "tester"
    assert events["qc.requeued_after_fix"]["qc_job_ids"] == [qc_job_id]


@pytest.mark.parametrize("fixed", ["", "   ", None])
def test_a_requeue_that_does_not_say_what_was_fixed_is_refused(fixed, tmp_path):
    """Undoing a terminal state without a reason on the record makes the state
    unreliable rather than terminal: the next reader cannot tell a corrected
    deployment from somebody clearing the board."""
    store, qc_job_id = _blocked_sqlite_qc(tmp_path, "reason")

    with pytest.raises(ValueError):
        store.requeue_blocked_qc_jobs("PHercFIXED", fixed=fixed, by="tester")

    assert _job(store, qc_job_id)["state"] == "BLOCKED_CONFIGURATION"


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_postgresql_takes_the_same_job_back_up():
    """The deployment runs PostgreSQL, and the night this was needed was a
    PostgreSQL deployment. The SQLite twin cannot prove the statement that
    actually runs: it has different column names and no RETURNING to read the
    moved rows out of."""
    import uuid

    from fleet.postgres_store import PostgresFleetStore

    schema = f"qcrequeue_{uuid.uuid4().hex}"

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
            "source_snapshot_id": "snap-1", "sample_id": "PHercFIXED",
            "ct_uri": "https://example.invalid/ct.zarr", "ct_sha256": "b" * 64,
            "m7_uri": "https://example.invalid/m7.zarr", "m7_sha256": "c" * 64,
            "shape_xyz": [64, 64, 64], "voxel_size_um": 9.362,
            "coordinate_frame": "ct_l0_xyz",
        })
        with postgres.connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                # physical_qc_state is NOT NULL here, unlike the SQLite twin.
                "INSERT INTO segment_surfaces(surface_id,source_snapshot_id,"
                "sample_id,owner,artifact_sha256,artifact_uri,bbox_xyz,"
                "sample_points,area_cm2,state,physical_qc_state,payload) VALUES"
                "('s-blocked','snap-1','PHercFIXED','campaign-x',%s,'s3://b/s',"
                "'{}'::jsonb,'[]'::jsonb,1.5,'GROWN','UNVALIDATED','{}'::jsonb)",
                (hashlib.sha256(b"s-blocked").hexdigest(),))
            cursor.execute(
                "INSERT INTO segment_qc_jobs(qc_job_id,surface_id,profile_id,"
                "state,payload,result) VALUES('qc-blocked','s-blocked','p',"
                "'BLOCKED_CONFIGURATION','{}'::jsonb,%s::jsonb)",
                ('{"status":"BLOCKED_CONFIGURATION",'
                 '"error":"profile hash differs: expected dead00ff"}',))

        assert postgres.claim_qc("gpu-1", 60) is None

        record = postgres.requeue_blocked_qc_jobs(
            "PHercFIXED", fixed="repinned the profile", by="tester")

        assert record["requeued"] == 1
        assert record["qc_job_ids"] == ["qc-blocked"]
        with postgres.connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT state,result FROM segment_qc_jobs "
                           "WHERE qc_job_id='qc-blocked'")
            job = cursor.fetchone()
        assert job["state"] == "PENDING"
        assert job["result"] is None, "the old hash is still the job's error"
        claimed = postgres.claim_qc("gpu-1", 60)
        assert claimed is not None and claimed["qc_job_id"] == "qc-blocked"
    finally:
        with bootstrap.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


def test_both_stores_offer_it():
    """The deployment runs PostgreSQL and the tests run SQLite. An operator
    reaching for this on the day it is needed must not find it only here."""
    from fleet.postgres_store import PostgresFleetStore
    from fleet.store import FleetStore

    for store in (PostgresFleetStore, FleetStore):
        assert callable(getattr(store, "requeue_blocked_qc_jobs", None)), (
            store.__name__
        )


# --------------------------------------------------------------------------
# The property that must not regress: only a person moves a blocked job
# --------------------------------------------------------------------------

def test_claiming_never_takes_a_blocked_job_however_often_it_is_tried(tmp_path):
    """The two days of spinning, in the form a worker would reproduce it.

    `claim_qc` also sweeps expired leases back to PENDING, so this asks
    repeatedly rather than once: a sweep written to recover stuck work is the
    likeliest way for BLOCKED_CONFIGURATION to quietly become claimable again.
    """
    store, qc_job_id = _blocked_sqlite_qc(tmp_path, "never-claimed")

    for _ in range(5):
        assert store.claim_qc("gpu-1", 60) is None
        assert store.claim_qc("gpu-2", 60, profile_id="requeue-test@1.0.0") is None

    assert _job(store, qc_job_id)["state"] == "BLOCKED_CONFIGURATION"


@pytest.mark.parametrize("name", sorted(STORES))
def test_only_the_operator_route_moves_a_job_out_of_blocked(name):
    """Read as source across both stores, because this is a statement about
    what does *not* exist: a sweep, a reaper or a retry that quietly selects
    BLOCKED_CONFIGURATION would leave every other test in this file passing."""
    methods = {
        chunk.split("(")[0]: " ".join(chunk.split())
        for chunk in STORES[name].split("\n    def ")[1:]
    }
    unblocking = {method for method, body in methods.items()
                  if "WHERE state='BLOCKED_CONFIGURATION'" in body}
    assert unblocking == {"requeue_blocked_qc_jobs"}, (
        f"{name} moves jobs out of BLOCKED_CONFIGURATION in {sorted(unblocking)}; "
        "the state stops the retry loop only while a person is the sole way out"
    )
    requeue = methods["requeue_blocked_qc_jobs"]
    assert "state='PENDING'" in requeue
    assert "result=NULL" in requeue or "result_json=NULL" in requeue, (
        f"{name} requeues without clearing the blocked attempt's receipt"
    )


def test_nothing_in_the_fleet_calls_the_requeue_by_itself():
    """A worker, a reaper or a scheduled task calling this is the retry loop
    with an audit record attached. The panel route is the only caller: it is
    reached by a request, from somebody who has signed in."""
    callers = set()
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {".claude", "tests", "vendor", "workspace"}:
            continue
        text = path.read_text(errors="ignore")
        if "requeue_blocked_qc_jobs" in text and "def requeue_blocked_qc_jobs" \
                not in text:
            callers.add(str(relative))
    assert callers == {"panel/app.py"}, (
        f"requeue_blocked_qc_jobs is called from {sorted(callers)}; only an "
        "operator's request may take a blocked job back up"
    )


# --------------------------------------------------------------------------
# The panel: the way a person actually reaches it
# --------------------------------------------------------------------------

class _Store:
    def __init__(self, requeued=1) -> None:
        self.calls: list[tuple] = []
        self._requeued = requeued

    def requeue_blocked_qc_jobs(self, sample_id, *, fixed, by):
        self.calls.append((sample_id, fixed, by))
        ids = [f"job-{i}" for i in range(self._requeued)]
        return {"sample_id": sample_id, "requeued": self._requeued,
                "qc_job_ids": ids, "fixed": fixed, "by": by}


@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import panel.app as panel_app
    from framework.contracts import auth

    store = _Store()
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(panel_app, "fleet_store", lambda: store)
    monkeypatch.setattr(panel_app, "mission_scrolls", lambda mission: {"PHerc826"})
    auth.create_user(panel_app.AUTH_ROOT, "tester", "a-long-enough-one")
    http = TestClient(panel_app.app)
    assert http.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one"}).status_code == 200
    http.store = store
    return http


def test_a_fixed_deployment_can_be_unblocked_through_the_panel(client) -> None:
    response = client.post("/api/segmentation/qc-jobs/requeue", json={
        "mission_id": "first-letters",
        "sample_id": "PHerc826",
        "fixed": "corrected HELENA_QC_PROFILE_SHA256 on gpu-1",
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["requeued"] == 1
    sample, fixed, by = client.store.calls[0]
    assert sample == "PHerc826"
    assert by == "tester", "the requeue was not attributed to whoever asked"
    assert "HELENA_QC_PROFILE_SHA256" in fixed


def test_a_requeue_that_says_nothing_about_the_fix_is_refused(client) -> None:
    for body in ({"mission_id": "first-letters", "sample_id": "PHerc826"},
                 {"mission_id": "first-letters", "sample_id": "PHerc826",
                  "fixed": ""}):
        response = client.post("/api/segmentation/qc-jobs/requeue", json=body)
        assert response.status_code == 422, response.text
    assert client.store.calls == []


def test_a_requeue_outside_the_mission_never_reaches_the_store(client) -> None:
    """Scope is checked before the mutation, like every other write route."""
    outside = client.post("/api/segmentation/qc-jobs/requeue", json={
        "mission_id": "first-letters", "sample_id": "PHerc1667",
        "fixed": "repinned the profile"})
    assert outside.status_code == 409, outside.text

    unscoped = client.post("/api/segmentation/qc-jobs/requeue", json={
        "fixed": "repinned the profile"})
    assert unscoped.status_code == 409, unscoped.text
    assert client.store.calls == []


def test_a_scope_with_nothing_blocked_says_so(client, monkeypatch) -> None:
    """A bare `requeued: 0` reads as a failed request. It is usually a true and
    useful answer -- this is the first thing somebody tries when a scroll looks
    stuck -- so it has to say that the scroll is not blocked, which is what
    sends the reader on to the real cause instead of retrying this."""
    client.store._requeued = 0

    response = client.post("/api/segmentation/qc-jobs/requeue", json={
        "mission_id": "first-letters", "sample_id": "PHerc826",
        "fixed": "repinned the profile"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["requeued"] == 0
    assert "PHerc826" in body["detail"]
    assert "blocked" in body["detail"]


def test_the_route_does_not_answer_without_a_session(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import panel.app as panel_app

    store = _Store()
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(panel_app, "fleet_store", lambda: store)
    anonymous = TestClient(panel_app.app)

    assert anonymous.post("/api/segmentation/qc-jobs/requeue", json={
        "mission_id": "first-letters", "sample_id": "PHerc826",
        "fixed": "repinned the profile"}).status_code in {401, 403}
    assert store.calls == []


def test_a_reason_made_of_spaces_is_a_bad_request():
    """min_length=1 accepted "   ", which reached the store and came back as a
    500 from the ValueError there -- the caller's bad request reported as a
    server fault. Found by probing the live route."""
    from panel.app import QcRequeueRequest

    import pydantic

    for blank in ("   ", "\t", "\n  "):
        with pytest.raises(pydantic.ValidationError):
            QcRequeueRequest(sample_id="PHercFIXED", fixed=blank)

    # And a real reason keeps its meaning, without the padding.
    assert QcRequeueRequest(sample_id="PHercFIXED",
                            fixed="  repinned the profile  ").fixed == (
        "repinned the profile")
