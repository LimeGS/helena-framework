"""The PostgreSQL control plane, against a real PostgreSQL.

    HELENA_TEST_DSN=postgresql://…/throwaway python3 -m pytest tests/integration

Skipped without it. These need a database because what they hold is SQL: the
ordering that stops a permanently failing surface from starving a backlog, the
admissibility filter, the conflict rule that makes a re-run a no-op, and the
row access itself.

That last one is why this file exists. Two queries written this week -- coverage
and no_seed_cells -- read their rows positionally against a cursor that yields
mappings, so both raised KeyError the first time a real database answered them.
The suite was green through both, because nothing here had ever touched
PostgreSQL. One reached the browser as "no cells attempted", which is what an
unexplored scroll looks like, and the other as "the fleet refused this replan",
which is what a policy refusal looks like.

The database is left as the test found it: every test works under its own
sample_id, and the fixture drops the schema between modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.common import content_sha256, stable_id

DSN = os.environ.get("HELENA_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")


@pytest.fixture(scope="module")
def store():
    from fleet.postgres_store import PostgresFleetStore

    fleet = PostgresFleetStore(DSN)
    fleet.initialize()
    return fleet


@pytest.fixture
def scroll() -> str:
    """A sample id of this test's own, so tests do not read each other's rows."""
    return f"TEST{uuid.uuid4().hex[:10]}"


@pytest.fixture
def snapshot(store, scroll) -> str:
    return store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })


def cell(index: int, *, centre: int) -> dict:
    return {"cell_id": f"r{index:05d}c00000a00000",
            "center_xyz": {"x": centre, "y": centre, "z": centre},
            "bounds_xyz": [[centre - 128] * 3, [centre + 128] * 3]}


def task(snapshot: str, scroll: str, index: int, *, grid: str, policy: str) -> dict:
    """A task as the bootstrap writes one. The envelope and the catalogue hash
    are not optional: they are what a receipt traces a surface back through."""
    return {"source_snapshot_id": snapshot, "sample_id": scroll,
            "grid_version": grid, "policy_version": policy,
            "priority": 1.0, "ink_used": False,
            "parameter_envelope": {"profile_ids": ["vc3d-m7-growth-v1"],
                                   "ink_used": False, "parameters": {}},
            "catalog_snapshot_sha256": "c" * 64,
            **cell(index, centre=1024 * (index + 1))}


# --------------------------------------------------------------------------
# Identity, which is what makes a re-run safe
# --------------------------------------------------------------------------

def test_the_same_ground_under_the_same_policy_inserts_once(store, snapshot, scroll):
    """Task identity is (snapshot, grid, cell, policy) behind ON CONFLICT DO
    NOTHING, so a second bootstrap over the same ground is a no-op by design --
    and anything that reports it as work done is lying."""
    tasks = [task(snapshot, scroll, i, grid="g1", policy="p1") for i in range(3)]
    assert store.create_tasks(tasks) == (3, 3)
    assert store.create_tasks(tasks) == (0, 3)


def test_a_new_policy_makes_the_same_cell_a_new_question(store, snapshot, scroll):
    store.create_tasks([task(snapshot, scroll, 0, grid="g1", policy="p1")])
    inserted, _ = store.create_tasks([task(snapshot, scroll, 0, grid="g1", policy="p2")])
    assert inserted == 1


# --------------------------------------------------------------------------
# The claim, which is what makes two workers safe
# --------------------------------------------------------------------------

def test_two_workers_cannot_claim_the_same_task(store, snapshot, scroll):
    store.create_tasks([task(snapshot, scroll, 0, grid="claim", policy="p1")])
    first = store.claim("worker-a", 60)
    second = store.claim("worker-b", 60)
    assert first is not None
    # Whatever the second worker got, it is not the same task.
    assert second is None or second["task_id"] != first["task_id"]


def test_a_lease_is_a_token_and_not_a_name(store, snapshot, scroll):
    """The token is what a worker proves ownership with; a worker id alone would
    let a stale process finish a task somebody else is running."""
    store.create_tasks([task(snapshot, scroll, 0, grid="lease", policy="p1")])
    claimed = store.claim("worker-a", 60)
    assert claimed and claimed.get("lease_token")
    with pytest.raises(Exception):
        store.heartbeat(claimed["task_id"], claimed["attempt_id"], "not-the-token", 60)


# --------------------------------------------------------------------------
# The backlogs, and the ordering that keeps them moving
# --------------------------------------------------------------------------

def imported_surface(store, snapshot, scroll, *, name: str, geometry: str,
                     physical: str, uri: str, certify: bool = True) -> str:
    """One surface. `certify=False` leaves it never-measured in the sense the
    backlog means: no verdict has been recorded against it at all, which is what
    the ordering distinguishes from a verdict of GEOMETRY_UNMEASURED."""
    # A distinct digest per surface: (snapshot, artifact_sha256) is unique, which
    # is the deduplication rule doing its job -- two surfaces of one snapshot
    # with identical bytes are one surface.
    digest = hashlib.sha256(f"{scroll}-{name}".encode()).hexdigest()
    surface = store.import_surface({
        "surface_id": f"{scroll}-{name}", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": digest, "artifact_uri": uri,
        "bbox_xyz": [[0, 0, 0], [10, 10, 10]], "area_cm2": 1.0,
        "state": "QC_SCREENED", "physical_qc_state": physical})
    if certify:
        store.record_geometry_certification(surface, geometry, {"schema": "test"})
    return surface


def test_flattening_takes_only_what_both_gates_admit(store, snapshot, scroll):
    """Geometry says the shape is a plausible lamina; the physical axis says
    there is papyrus there. P3 read only the first for as long as it existed."""
    good = imported_surface(store, snapshot, scroll, name="ok",
                            geometry="GEOMETRY_CERTIFIED", physical="CT_SUPPORTED",
                            uri="s3://bucket/ok")
    imported_surface(store, snapshot, scroll, name="unvalidated",
                     geometry="GEOMETRY_CERTIFIED", physical="UNVALIDATED",
                     uri="s3://bucket/unvalidated")
    imported_surface(store, snapshot, scroll, name="uncertified",
                     geometry="GEOMETRY_UNMEASURED", physical="CT_SUPPORTED",
                     uri="s3://bucket/uncertified")

    admitted = store.surfaces_awaiting_flattening("flatten-abf-v1@1.0.0", limit=10,
                                                  sample_id=scroll)
    assert [row["surface_id"] for row in admitted] == [good]

    # The override exists for comparing against what the old gate allowed.
    relaxed = store.surfaces_awaiting_flattening("flatten-abf-v1@1.0.0", limit=10,
                                                 sample_id=scroll,
                                                 require_physical_qc=False)
    assert len(relaxed) == 2


def test_a_surface_that_fails_every_time_does_not_starve_the_queue(store, snapshot, scroll):
    """Ordered by created_at alone, a surface whose flattening fails sits at the
    head of the queue forever: with a limit of five and two such surfaces, forty
    per cent of every run went to them."""
    stubborn = imported_surface(store, snapshot, scroll, name="stubborn",
                                geometry="GEOMETRY_CERTIFIED", physical="CT_SUPPORTED",
                                uri="/artifacts/gone")
    fresh = imported_surface(store, snapshot, scroll, name="fresh",
                             geometry="GEOMETRY_CERTIFIED", physical="CT_SUPPORTED",
                             uri="s3://bucket/fresh")
    store.record_flattening({"surface_id": stubborn, "profile_id": "flatten-abf-v1@1.0.0",
                             "state": "FLATTENING_FAILED", "error": "gone"})
    waiting = store.surfaces_awaiting_flattening("flatten-abf-v1@1.0.0", limit=10,
                                                 sample_id=scroll)
    assert [row["surface_id"] for row in waiting][0] == fresh, \
        "the surface nobody has tried has to come first"
    assert stubborn in [row["surface_id"] for row in waiting], \
        "and the failure is still retried, because a fetch can fail on the network"


def test_the_geometry_backlog_puts_the_never_measured_first(store, snapshot, scroll):
    tried = imported_surface(store, snapshot, scroll, name="tried",
                             geometry="GEOMETRY_UNMEASURED", physical="UNVALIDATED",
                             uri="s3://bucket/tried")
    store.record_geometry_certification(tried, "GEOMETRY_UNMEASURED",
                                        {"reason": "ARTIFACT_UNAVAILABLE"})
    never = imported_surface(store, snapshot, scroll, name="never",
                             geometry="GEOMETRY_UNMEASURED", physical="UNVALIDATED",
                             uri="s3://bucket/never", certify=False)
    pending = store.surfaces_without_geometry_verdict(limit=10, sample_id=scroll)
    assert pending and pending[0]["surface_id"] == never


# --------------------------------------------------------------------------
# The reads the panel draws pages from
# --------------------------------------------------------------------------

def test_coverage_counts_cells_and_derives_the_grid_step(store, snapshot, scroll):
    """Written this week and wrong the first time a real database answered it:
    the rows are mappings and it read them positionally."""
    store.create_tasks([task(snapshot, scroll, i, grid="cov", policy="p1")
                        for i in range(3)])
    report = store.coverage(scroll)
    grid = next(g for g in report["grids"] if g["grid_version"] == "cov")
    assert grid["cells_attempted"] == 3
    # Centres are 1024 apart in the fixture, and the step comes from them rather
    # than from the grid version's name or the candidate search box.
    assert grid["grid_step_xyz"] == [1024.0, 1024.0, 1024.0]
    assert grid["cells_in_volume"] and grid["fraction_attempted"] is not None
    assert report["non_claims"]


def test_no_seed_cells_reads_its_rows_by_name(store, snapshot, scroll):
    """The other one. It surfaced in the panel as "the fleet refused this
    replan", which reads like a policy refusal and was a KeyError."""
    assert store.no_seed_cells(sample_id=scroll, limit=5) == []


def test_a_local_artifact_is_findable_and_repointable(store, snapshot, scroll):
    """Thirty per cent of this corpus was published to a worker's disk before
    the rule that a worker is disposable."""
    stranded = imported_surface(store, snapshot, scroll, name="local",
                                geometry="GEOMETRY_UNMEASURED", physical="UNVALIDATED",
                                uri="/artifacts/surfaces/PHerc826/x")
    imported_surface(store, snapshot, scroll, name="published",
                     geometry="GEOMETRY_UNMEASURED", physical="UNVALIDATED",
                     uri="s3://bucket/published")
    local = store.surfaces_on_local_artifacts(limit=10, sample_id=scroll)
    assert [row["surface_id"] for row in local] == [stranded]

    store.repoint_surface_artifact(
        stranded, "s3://bucket/recovered",
        hashlib.sha256(f"{scroll}-local".encode()).hexdigest())
    assert store.surfaces_on_local_artifacts(limit=10, sample_id=scroll) == []


def test_a_geometry_verdict_does_not_touch_the_other_axis(store, snapshot, scroll):
    """The two are orthogonal on purpose: a surface can be CT_SUPPORTED and
    GEOMETRY_REJECTED_BRIDGE at once, and that combination is exactly the one
    the campaign had no way to express."""
    surface = imported_surface(store, snapshot, scroll, name="axes",
                               geometry="GEOMETRY_UNMEASURED", physical="CT_SUPPORTED",
                               uri="s3://bucket/axes")
    store.record_geometry_certification(surface, "GEOMETRY_REJECTED_BRIDGE",
                                        {"schema": "test"})
    rows = [row for row in store.surfaces_for_snapshot(snapshot)
            if row["surface_id"] == surface]
    assert rows and rows[0]["physical_qc_state"] == "CT_SUPPORTED"


# --------------------------------------------------------------------------
# Finalisation, where a grown surface becomes a row
# --------------------------------------------------------------------------

def grown(store, snapshot, scroll, *, name: str, points: list) -> dict:
    """The surface a worker hands back, in the shape finalize takes."""
    return {"surface_id": f"{scroll}-{name}",
            "source_snapshot_id": snapshot, "sample_id": scroll,
            "owner": "worker-a",
            "artifact_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "artifact_uri": f"s3://bucket/{name}",
            "bbox_xyz": [[0, 0, 0], [10, 10, 10]], "sample_points": points,
            "area_cm2": 1.5,
            "geometry_qc_state": "GEOMETRY_CERTIFIED"}


def claim_one(store, snapshot, scroll, *, grid: str) -> dict:
    """Claim the task this call created, and not merely a pending one.

    `claim` takes whatever the queue holds, which is what a worker wants and
    what a test cannot rely on: these tests share one database, so a task an
    earlier test left pending was handed back here and the surface built from
    this test's snapshot was then finalized against it. The failure read as
    "surface receipt is not bound to its task", which was true and told you
    nothing about why.
    """
    wanted = stable_id("task", {"sample_id": scroll, "grid_version": grid})
    store.create_tasks([
        {**task(snapshot, scroll, 0, grid=grid, policy="p1"), "task_id": wanted}
    ])
    claimed = store.claim("worker-a", 60, task_id=wanted)
    assert claimed, "nothing to claim"
    return claimed


def prepare_finalization(store, claimed: dict, surface: dict, staging: str):
    locked_plan = {
        "schema": "test.locked_plan.v1",
        "task_id": claimed["task_id"],
    }
    locked_plan_sha256 = content_sha256(locked_plan)
    bound_surface = {
        **surface,
        "schema": "campaignx.segment_fleet_surface.v1",
        "task_id": claimed["task_id"],
        "attempt_id": claimed["attempt_id"],
        "locked_plan_sha256": locked_plan_sha256,
        "ink_used": False,
    }
    store.transition(
        claimed["task_id"],
        claimed["attempt_id"],
        claimed["lease_token"],
        "RUNNING",
        locked_plan=locked_plan,
    )
    artifact_set_id = store.add_artifact_set(
        claimed["task_id"],
        claimed["attempt_id"],
        claimed["lease_token"],
        {
            "schema": "campaignx.segmentation_artifact_set.v1",
            "task_id": claimed["task_id"],
            "attempt_id": claimed["attempt_id"],
            "locked_plan_sha256": locked_plan_sha256,
            "files": {},
            "artifact_sha256": bound_surface["artifact_sha256"],
            "bbox_xyz": bound_surface["bbox_xyz"],
            "sample_points": bound_surface["sample_points"],
            "area_cm2": bound_surface["area_cm2"],
            "ink_used": False,
        },
        staging,
    )
    store.transition(
        claimed["task_id"],
        claimed["attempt_id"],
        claimed["lease_token"],
        "FINALIZING",
    )
    return bound_surface, artifact_set_id


def test_a_finalised_surface_needs_a_versioned_qc_profile(store, snapshot, scroll):
    """An unversioned profile id would make two verdicts from two different
    gates indistinguishable in the record."""
    claimed = claim_one(store, snapshot, scroll, grid="final-a")
    with pytest.raises(ValueError):
        store.finalize(claimed["task_id"], claimed["attempt_id"],
                       claimed["lease_token"],
                       grown(store, snapshot, scroll, name="a", points=[[0, 0, 0]]),
                       "artifact-set", "unversioned-profile")


def test_finalization_rejects_another_attempts_artifact(
    store, snapshot, scroll
):
    first = claim_one(store, snapshot, scroll, grid="owner-a")
    first_surface, first_artifact = prepare_finalization(
        store,
        first,
        grown(store, snapshot, scroll, name="owner-a", points=[[0, 0, 0]]),
        "s3://staging/owner-a",
    )
    second = claim_one(store, snapshot, scroll, grid="owner-b")
    _, second_artifact = prepare_finalization(
        store,
        second,
        grown(store, snapshot, scroll, name="owner-b", points=[[1, 1, 1]]),
        "s3://staging/owner-b",
    )
    with pytest.raises(RuntimeError, match="different attempt"):
        store.finalize(
            first["task_id"],
            first["attempt_id"],
            first["lease_token"],
            first_surface,
            second_artifact,
            "surface-qc@1.0.0",
        )
    assert store.finalize(
        first["task_id"],
        first["attempt_id"],
        first["lease_token"],
        first_surface,
        first_artifact,
        "surface-qc@1.0.0",
    )["status"] == "QC_PENDING"


def test_a_surface_that_repeats_another_is_recorded_as_one(store, snapshot, scroll):
    """Two seeds can grow the same lamina. Counting both would inflate coverage
    and spend a GPU twice on one sheet."""
    points = [[float(x), 0.0, 0.0] for x in range(40)]

    first = claim_one(store, snapshot, scroll, grid="dedup-a")
    first_surface, first_artifact = prepare_finalization(
        store,
        first,
        {**grown(store, snapshot, scroll, name="one", points=points),
         "artifact_sha256": hashlib.sha256(b"one").hexdigest()},
        "s3://staging/one",
    )
    outcome = store.finalize(
        first["task_id"], first["attempt_id"], first["lease_token"],
        first_surface, first_artifact, "surface-qc@1.0.0")
    assert outcome["status"] == "QC_PENDING"
    assert store.finalize(
        first["task_id"], first["attempt_id"], first["lease_token"],
        first_surface, first_artifact, "surface-qc@1.0.0"
    ) == outcome

    second = claim_one(store, snapshot, scroll, grid="dedup-b")
    second_surface, second_artifact = prepare_finalization(
        store,
        second,
        {**grown(store, snapshot, scroll, name="two", points=points),
         "artifact_sha256": hashlib.sha256(b"two").hexdigest()},
        "s3://staging/two",
    )
    repeat = store.finalize(
        second["task_id"], second["attempt_id"], second["lease_token"],
        second_surface, second_artifact, "surface-qc@1.0.0")
    assert repeat["status"] == "DUPLICATE_SURFACE"
    assert repeat.get("duplicate_of")


def test_a_geometry_rejection_blocks_the_qc_job_it_would_have_queued(store, snapshot, scroll):
    """A surface the geometry gate rejected must not reach the ink model, and
    the queue records the refusal rather than dropping the job silently."""
    claimed = claim_one(store, snapshot, scroll, grid="reject")
    rejected_surface, rejected_artifact = prepare_finalization(
        store,
        claimed,
        {**grown(store, snapshot, scroll, name="rej", points=[[1.0, 1.0, 1.0]]),
         "artifact_sha256": hashlib.sha256(b"rej").hexdigest(),
         "geometry_qc_state": "GEOMETRY_REJECTED_BRIDGE"},
        "s3://staging/rej",
    )
    outcome = store.finalize(
        claimed["task_id"], claimed["attempt_id"], claimed["lease_token"],
        rejected_surface, rejected_artifact, "surface-qc@1.0.0")
    assert outcome["geometry_qc_state"] == "GEOMETRY_REJECTED_BRIDGE"


# --------------------------------------------------------------------------
# The QC queue, which is a second lease on the same rules
# --------------------------------------------------------------------------

def test_a_qc_job_is_claimed_once_and_owned_by_its_token(store, snapshot, scroll):
    # Its own profile id: claim_qc takes whatever is pending for a profile, and
    # these tests share one database.
    profile = f"surface-qc-{scroll.lower()}@1.0.0"
    # Certified, because the subject here is the lease and not the gate: an
    # uncertified surface gets a job that waits on geometry, which is correct
    # and leaves nothing to claim. This asked for `certify=False` and then for a
    # claimable job, which the geometry gate stopped being able to mean.
    surface = imported_surface(store, snapshot, scroll, name="qc",
                               geometry="GEOMETRY_CERTIFIED", physical="UNVALIDATED",
                               uri="s3://bucket/qc")
    store.enqueue_imported_surface_qc(
        {"surface_id": surface, "source_snapshot_id": snapshot, "sample_id": scroll,
         "artifact_sha256": hashlib.sha256(f"{scroll}-qc".encode()).hexdigest(),
         "artifact_uri": "s3://bucket/qc",
         "bbox_xyz": [[0, 0, 0], [1, 1, 1]], "area_cm2": 1.0},
        profile_id=profile)

    claimed = store.claim_qc("qc-worker", 60, profile_id=profile)
    assert claimed and claimed["surface_id"] == surface
    assert store.claim_qc("other-worker", 60, profile_id=profile) is None

    with pytest.raises(Exception):
        store.heartbeat_qc(claimed["qc_job_id"], "not-the-token", 60)
    store.heartbeat_qc(claimed["qc_job_id"], claimed["lease_token"], 60)


def test_a_qc_verdict_writes_the_physical_axis_and_not_the_geometric(store, snapshot, scroll):
    profile = f"surface-qc-{scroll.lower()}@1.0.0"
    surface = imported_surface(store, snapshot, scroll, name="verdict",
                               geometry="GEOMETRY_CERTIFIED", physical="UNVALIDATED",
                               uri="s3://bucket/verdict")
    store.enqueue_imported_surface_qc(
        {"surface_id": surface, "source_snapshot_id": snapshot, "sample_id": scroll,
         "artifact_sha256": hashlib.sha256(f"{scroll}-verdict".encode()).hexdigest(),
         "artifact_uri": "s3://bucket/verdict",
         "bbox_xyz": [[0, 0, 0], [1, 1, 1]], "area_cm2": 1.0},
        profile_id=profile)
    claimed = store.claim_qc("qc-worker", 60, profile_id=profile)
    # The result carries its own evidence or it is not a verdict: a QC outcome
    # with nothing durable behind it is an opinion in a column.
    outcome = "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL"
    store.finalize_qc(claimed["qc_job_id"], claimed["lease_token"], outcome, {
        "schema": "campaignx.segment_qc_result.v1", "surface_id": surface,
        "outcome": outcome, "evidence_manifest_sha256": "b" * 64,
        "evidence_uri": "s3://bucket/qc-evidence", "ink_used": False})
    row = next(r for r in store.surfaces_for_snapshot(snapshot)
               if r["surface_id"] == surface)
    assert row["physical_qc_state"] == "CT_SUPPORTED"
    # The geometry verdict is untouched: the axes are independent by design.
    assert row["geometry_qc_state"] == "GEOMETRY_CERTIFIED"


# --------------------------------------------------------------------------
# Secrets, which must not come back out
# --------------------------------------------------------------------------

def test_only_a_credential_a_worker_reads_can_be_stored(store):
    """The panel writes these from a browser, so the set of names it can write
    is the set a worker consumes -- anything else is a place to smuggle
    configuration past review."""
    with pytest.raises(ValueError):
        store.set_secret("TEST_KEY", "s3cr3t-value", "tester")


def test_a_secret_is_readable_by_the_fleet_and_never_by_a_status_view(store):
    store.set_secret("AWS_DEFAULT_REGION", "us-east-1", "tester")
    status = {row["name"]: row for row in store.secret_status()}
    assert "AWS_DEFAULT_REGION" in status
    assert "us-east-1" not in json.dumps(status), "a status view must not carry the value"
    assert store.secrets()["AWS_DEFAULT_REGION"] == "us-east-1"
    assert store.forget_secret("AWS_DEFAULT_REGION") is True
    assert "AWS_DEFAULT_REGION" not in store.secrets()
