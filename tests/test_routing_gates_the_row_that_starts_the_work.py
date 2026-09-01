"""The size gate has to hold where the row is written, in both stores.

PHerc0268 -- 0.01983222455087575 cm2, 14x14, 132 triangles, 84 finite
coordinates -- was GEOMETRY_CERTIFIED on 2026-08-02, reached the ink screen, and
came back EMPTY. A guard in the QC worker would not have stopped it: by the time
a worker looks, the QC row that lets the work start already exists, and a row
that exists is a row something eventually claims.

So the routing decision is resolved, persisted, and read back inside the same
transaction that creates the surface and its QC job. Four boundaries create
those rows -- `finalize`, `enqueue_imported_surface_qc`, the promotion inside
`record_geometry_certification`, and `register_merged_surface`, which is the P8
merge lane writing a surface nobody grew -- and each of them requires a receipt
that verifies and still agrees with the surface's measured area.

Everything here runs twice, against SQLite and against PostgreSQL, because the
deployment runs PostgreSQL and the gate that exists in only one store is not a
gate.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing as routing  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

DSN = os.environ.get("HELENA_TEST_DSN")

# Frozen in docs/first-letters/first-letters-hybrid-20260802/evidence.json.
PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6"
)


class _IsolatedPostgresStore(PostgresFleetStore):
    """One throwaway schema per test, so the live runs cannot see each other."""

    def __init__(self, database_url: str, schema: str):
        super().__init__(database_url)
        self._test_schema = schema

    def connect(self):
        connection = super().connect()
        with connection.cursor() as cursor:
            # The name is a UUID hex string with a fixed prefix.
            cursor.execute(f"SET search_path TO {self._test_schema}, public")
        return connection


@pytest.fixture
def sqlite_store(tmp_path) -> FleetStore:
    store = FleetStore(tmp_path / "fleet.sqlite3")
    store.initialize()
    return store


@pytest.fixture
def postgres_store():
    if not DSN:
        pytest.skip(
            "HELENA_TEST_DSN is not set; the PostgreSQL half of the size gate "
            "was not run"
        )
    schema = f"routing_gate_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    store = _IsolatedPostgresStore(DSN, schema)
    try:
        store.initialize()
        yield store
    finally:
        with bootstrap.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA {schema} CASCADE")


@pytest.fixture(params=["sqlite_store", "postgres_store"],
                ids=["sqlite", "postgresql"])
def store(request):
    """Both stores, so a gate can never exist in only one of them."""
    return request.getfixturevalue(request.param)


# -- shared fixtures for one surface's worth of evidence ---------------------

def _snapshot(store, scroll: str) -> str:
    return store.register_snapshot({
        "sample_id": scroll, "ct_uri": f"file:///{scroll}/ct",
        "m7_uri": f"file:///{scroll}/m7", "shape_xyz": [64, 64, 64],
        "voxel_size_um": 9.362, "coordinate_frame": "ct_l0_xyz"})


def _finalize(store, source_id: str, scroll: str, cell: str, *, area: float,
              digest: str, geometry: str = "GEOMETRY_CERTIFIED") -> dict:
    """Grow one surface through the real finalization boundary."""
    store.create_tasks([{
        "source_snapshot_id": source_id, "sample_id": scroll, "cell_id": cell,
        "grid_version": "g1", "policy_version": "p1",
        "bounds_xyz": [[0, 0, 0], [14, 14, 3]], "center_xyz": [7.0, 7.0, 1.5],
        "priority": 1.0, "parameter_envelope": {},
        "catalog_snapshot_sha256": "c" * 64}])
    task = store.claim(worker_id="w1", lease_seconds=600)
    assert task is not None
    locked_plan = {"schema": "test.locked_plan.v1", "task_id": task["task_id"]}
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"],
                     "RUNNING", locked_plan=locked_plan)
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": task["task_id"], "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan), "files": {},
        "artifact_sha256": digest, "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "sample_points": [[0.0, 0.0, 0.0]], "area_cm2": area,
        "ink_used": False}
    artifact_set_id = store.add_artifact_set(
        task["task_id"], task["attempt_id"], task["lease_token"], manifest,
        "file:///staging")
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"],
                     "FINALIZING")
    surface = {
        "schema": "campaignx.segment_fleet_surface.v1",
        "surface_id": f"{scroll}:{cell}", "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"file:///artifacts/{cell}",
        "bbox_xyz": manifest["bbox_xyz"], "sample_points": manifest["sample_points"],
        "area_cm2": area, "geometry_qc_state": geometry,
        "task_id": task["task_id"], "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan), "ink_used": False}
    outcome = store.finalize(task["task_id"], task["attempt_id"],
                             task["lease_token"], surface, artifact_set_id,
                             f"surface-qc-{cell}@1.0.0")
    return {"surface_id": surface["surface_id"], "outcome": outcome,
            "qc_profile_id": f"surface-qc-{cell}@1.0.0"}


# -- C3: finalization ---------------------------------------------------------

def test_finalization_records_the_routing_decision_with_the_surface(store) -> None:
    scroll = "PHercFINAL"
    source_id = _snapshot(store, scroll)
    grown = _finalize(store, source_id, scroll, "cell-normal", area=0.5,
                      digest="a" * 64)

    receipt = store.routing_receipt(grown["surface_id"])
    assert receipt is not None, (
        "a finalized surface exists with no routing decision recorded, so the "
        "gate is a downstream opinion again")
    assert routing.verify_receipt(receipt) is True
    assert receipt["route"] == routing.STANDARD
    assert receipt["measured_area_cm2"] == 0.5


def test_the_pherc0268_surface_never_gets_a_claimable_job(store) -> None:
    """The 2026-08-02 failure, at the boundary that produced it."""
    scroll = "PHerc0268"
    source_id = _snapshot(store, scroll)
    grown = _finalize(store, source_id, scroll, "cell-0268",
                      area=PHERC0268_AREA_CM2, digest=PHERC0268_ARTIFACT_SHA256)

    receipt = store.routing_receipt(grown["surface_id"])
    assert receipt["route"] == routing.DIAGNOSTIC
    assert receipt["is_absence_evidence"] is False
    assert routing.enters_standard_qc(receipt) is False
    assert store.claim_qc("qc-worker", 60,
                          profile_id=grown["qc_profile_id"]) is None, (
        "a two-square-millimetre surface reached the physical-QC queue")
    # Preserved, not dropped: being too small to judge is a reason to keep it.
    assert store.surface_artifact(grown["surface_id"])["artifact_sha256"] == (
        PHERC0268_ARTIFACT_SHA256)


def test_a_normal_finalized_surface_still_reaches_physical_qc(store) -> None:
    """The gate must not quarantine the work it was not written for."""
    scroll = "PHercOK"
    source_id = _snapshot(store, scroll)
    grown = _finalize(store, source_id, scroll, "cell-ok", area=0.5,
                      digest="b" * 64)

    claimed = store.claim_qc("qc-worker", 60, profile_id=grown["qc_profile_id"])
    assert claimed is not None and claimed["surface_id"] == grown["surface_id"]


def test_the_finalized_receipt_cannot_be_rewritten_or_removed(store) -> None:
    scroll = "PHercIMMUTABLE"
    source_id = _snapshot(store, scroll)
    grown = _finalize(store, source_id, scroll, "cell-imm",
                      area=PHERC0268_AREA_CM2, digest="c" * 64)
    before = store.routing_receipt(grown["surface_id"])
    assert before["route"] == routing.DIAGNOSTIC

    if isinstance(store, FleetStore):
        import sqlite3
        with store.connect() as connection:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute(
                    "UPDATE surface_routing_receipts SET route=? WHERE surface_id=?",
                    (routing.STANDARD, grown["surface_id"]))
            with pytest.raises(sqlite3.IntegrityError, match="permanent"):
                connection.execute(
                    "DELETE FROM surface_routing_receipts WHERE surface_id=?",
                    (grown["surface_id"],))
    else:
        import psycopg2
        for statement, message in (
            ("UPDATE segment_surface_routing_receipts SET route=%s "
             "WHERE surface_id=%s", "immutable"),
            ("DELETE FROM segment_surface_routing_receipts "
             "WHERE surface_id=%s", "permanent"),
        ):
            with store.connect() as connection:
                with connection.cursor() as cursor:
                    arguments = ((routing.STANDARD, grown["surface_id"])
                                 if "UPDATE" in statement
                                 else (grown["surface_id"],))
                    with pytest.raises(psycopg2.Error, match=message):
                        cursor.execute(statement, arguments)
            # psycopg2 leaves the connection aborted; each attempt gets its own.

    assert store.routing_receipt(grown["surface_id"]) == before


# -- C3: geometry certification -----------------------------------------------

def _import(store, source_id: str, scroll: str, name: str, *, area,
            digest: str) -> str:
    return store.import_surface({
        "surface_id": f"{scroll}:{name}", "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"s3://bucket/{name}",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": area,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})


def test_certification_never_promotes_a_diagnostic_surface_into_the_queue(
    store,
) -> None:
    """A geometry verdict is not a size verdict, and cannot overrule one.

    PHerc0268 was GEOMETRY_CERTIFIED. This is the exact place that certificate
    used to turn into a claimable job.
    """
    scroll = "PHercCERT"
    source_id = _snapshot(store, scroll)
    surface_id = _import(store, source_id, scroll, "tiny",
                         area=PHERC0268_AREA_CM2,
                         digest=PHERC0268_ARTIFACT_SHA256)
    profile = "surface-qc-cert@1.0.0"
    store.enqueue_imported_surface_qc({
        "surface_id": surface_id, "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/tiny", "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "area_cm2": PHERC0268_AREA_CM2}, profile_id=profile)

    store.record_geometry_certification(
        surface_id, "GEOMETRY_CERTIFIED", {"schema": "test"},
        requested_by_job_id="p2-cert", profile_id="geometry-test@1",
        profile_sha256="6" * 64)

    assert store.claim_qc("qc-worker", 60, profile_id=profile) is None, (
        "a geometry certificate promoted a surface below the floor into the "
        "physical-QC queue")
    assert store.routing_receipt(surface_id)["route"] == routing.DIAGNOSTIC


def test_certification_still_releases_a_surface_above_the_floor(store) -> None:
    scroll = "PHercRELEASE"
    source_id = _snapshot(store, scroll)
    surface_id = _import(store, source_id, scroll, "big", area=0.5,
                         digest="d" * 64)
    profile = "surface-qc-release@1.0.0"
    store.enqueue_imported_surface_qc({
        "surface_id": surface_id, "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": "d" * 64,
        "artifact_uri": "s3://bucket/big", "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "area_cm2": 0.5}, profile_id=profile)
    assert store.claim_qc("too-early", 60, profile_id=profile) is None

    store.record_geometry_certification(
        surface_id, "GEOMETRY_CERTIFIED", {"schema": "test"},
        requested_by_job_id="p2-release", profile_id="geometry-test@1",
        profile_sha256="6" * 64)

    claimed = store.claim_qc("qc-worker", 60, profile_id=profile)
    assert claimed is not None and claimed["surface_id"] == surface_id


# -- C4: the direct QC enqueue ------------------------------------------------

def test_a_direct_enqueue_without_a_measured_area_fails_closed(store) -> None:
    """No measurement, no route, no job. Not a job that quietly proceeds."""
    scroll = "PHercNOAREA"
    source_id = _snapshot(store, scroll)

    with pytest.raises((RuntimeError, ValueError), match="area|rout"):
        store.enqueue_imported_surface_qc({
            "surface_id": f"{scroll}:unmeasured",
            "source_snapshot_id": source_id, "sample_id": scroll,
            "artifact_sha256": "e" * 64, "artifact_uri": "s3://bucket/u",
            "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
            "geometry_qc_state": "GEOMETRY_CERTIFIED"},
            profile_id="surface-qc-noarea@1.0.0")

    assert store.claim_qc("qc-worker", 60,
                          profile_id="surface-qc-noarea@1.0.0") is None


def test_a_direct_enqueue_of_a_tiny_surface_is_never_claimable(store) -> None:
    scroll = "PHercDIRECT"
    source_id = _snapshot(store, scroll)
    profile = "surface-qc-direct@1.0.0"
    result = store.enqueue_imported_surface_qc({
        "surface_id": f"{scroll}:tiny", "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/tiny", "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "area_cm2": PHERC0268_AREA_CM2,
        "geometry_qc_state": "GEOMETRY_CERTIFIED"}, profile_id=profile)

    assert result["qc_state"] == "SMALL_SURFACE_DIAGNOSTIC"
    assert store.claim_qc("qc-worker", 60, profile_id=profile) is None


def test_a_direct_enqueue_above_the_floor_is_unchanged(store) -> None:
    scroll = "PHercDIRECTOK"
    source_id = _snapshot(store, scroll)
    profile = "surface-qc-directok@1.0.0"
    result = store.enqueue_imported_surface_qc({
        "surface_id": f"{scroll}:big", "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": "f" * 64,
        "artifact_uri": "s3://bucket/big", "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "area_cm2": 0.5, "geometry_qc_state": "GEOMETRY_CERTIFIED"},
        profile_id=profile)

    assert result["qc_state"] == "PENDING"
    claimed = store.claim_qc("qc-worker", 60, profile_id=profile)
    assert claimed is not None


def test_reconciling_an_area_away_from_its_receipt_fails_closed(store) -> None:
    """The receipt is immutable, so the measurement behind it cannot move.

    `enqueue_imported_surface_qc` may replace artifact metadata on an
    unvalidated surface, `area_cm2` included. A surface imported at half a
    square centimetre and reconciled down to two square millimetres would
    otherwise keep a STANDARD receipt describing an area it no longer has.
    """
    scroll = "PHercDRIFT"
    source_id = _snapshot(store, scroll)
    surface_id = _import(store, source_id, scroll, "drift", area=0.5,
                         digest="1" * 64)
    assert store.routing_receipt(surface_id)["route"] == routing.STANDARD

    with pytest.raises((RuntimeError, ValueError), match="rout|area"):
        store.enqueue_imported_surface_qc({
            "surface_id": surface_id, "source_snapshot_id": source_id,
            "sample_id": scroll, "artifact_sha256": "1" * 64,
            "artifact_uri": "s3://bucket/drift",
            "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
            "area_cm2": PHERC0268_AREA_CM2},
            profile_id="surface-qc-drift@1.0.0")

    assert store.claim_qc("qc-worker", 60,
                          profile_id="surface-qc-drift@1.0.0") is None
    assert store.routing_receipt(surface_id)["measured_area_cm2"] == 0.5


# -- P8: the surface a merge writes -------------------------------------------
#
# PostgreSQL only, because InkJobStore has no SQLite twin -- and PostgreSQL is
# what the deployment runs, so this is the store the gate has to hold in.

@pytest.fixture
def merge_store(postgres_store):
    """The ink control plane on the same throwaway schema as the segment one."""
    import importlib.util

    if not DSN:
        pytest.skip("HELENA_TEST_DSN is not set")
    spec = importlib.util.spec_from_file_location(
        "p8_job_store",
        ROOT / "framework/stages/03-ink/fleet/job_store.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = postgres_store._test_schema  # noqa: SLF001

    class _IsolatedInkStore(module.InkJobStore):
        def _connect(self):
            connection = super()._connect()
            with connection.cursor() as cursor:
                cursor.execute(f"SET search_path TO {schema}, public")
            return connection

    store = _IsolatedInkStore(DSN)
    store.initialize()
    return store


def _merged_surface(scroll: str, source_id: str, parents: list[dict], *,
                    area, surface_id: str) -> dict:
    surface = {
        "surface_id": surface_id, "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": "e" * 64,
        "artifact_uri": "file:///artifacts/merged",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "sample_points": [[0.0, 0.0, 0.0]], "state": "MERGED",
        "physical_qc_state": "UNVALIDATED",
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
        "parent_surface_ids": [row["surface_id"] for row in parents],
    }
    if area is not None:
        surface["area_cm2"] = area
    return surface


def _merge_job(merge_store, job_id: str, scroll: str) -> str:
    """The P8 job row the merged surface's lineage points back at."""
    with merge_store._connect() as connection:  # noqa: SLF001
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ink_jobs(job_id,sample_id,profile_id,parameters,"
                "state) VALUES(%s,%s,%s,%s::jsonb,'running') "
                "ON CONFLICT(job_id) DO NOTHING",
                (job_id, scroll, "vc3d-tifxyz-merge@1.0.0",
                 json.dumps({"lane": "vc3d-tifxyz-merge"})))
    return job_id


def _merge_parents(store, source_id: str, scroll: str) -> list[dict]:
    """Two parents that each clear the floor, so only the merge is in question."""
    parents = []
    for name, digest in (("p-a", "1" * 64), ("p-b", "2" * 64)):
        surface_id = _import(store, source_id, scroll, name, area=0.5,
                             digest=digest)
        assert store.routing_receipt(surface_id)["route"] == routing.STANDARD
        parents.append({"surface_id": surface_id, "artifact_sha256": digest})
    return parents


def test_a_merged_surface_under_the_floor_never_gets_a_claimable_job(
    postgres_store, merge_store,
) -> None:
    """The fourth door, and the one that ends at a technician's bench.

    `claim_qc` takes any PENDING row and never revisits the route, so a merged
    sheet enqueued PENDING is a merged sheet somebody eventually inspects. The
    parents were each gated at admission and each cleared the floor; the sheet
    they were stitched into is a new surface with its own area, and 0.0198 cm2
    is 0.0198 cm2 however it was assembled.
    """
    scroll = "PHercMERGE"
    source_id = _snapshot(postgres_store, scroll)
    parents = _merge_parents(postgres_store, source_id, scroll)
    _merge_job(merge_store, "p8-tiny", scroll)

    registration = merge_store.register_merged_surface(
        _merged_surface(scroll, source_id, parents, area=PHERC0268_AREA_CM2,
                        surface_id="merged-tiny"),
        parents, job_id="p8-tiny", qc_profile_id="surface-qc-merged@1.0.0")

    assert registration["route"] == routing.DIAGNOSTIC
    assert registration["qc_state"] == "SMALL_SURFACE_DIAGNOSTIC"
    assert postgres_store.claim_qc(
        "qc-worker", 60, profile_id="surface-qc-merged@1.0.0") is None

    stored = postgres_store.routing_receipt("merged-tiny")
    assert routing.verify_receipt(stored) is True
    assert routing.enters_standard_qc(stored) is False
    assert stored["measured_area_cm2"] == PHERC0268_AREA_CM2
    # The lane and the store are one decision, not two: the receipt the store
    # holds is the receipt the router builds from that surface.
    assert stored == routing.receipt_for_surface(
        _merged_surface(scroll, source_id, parents, area=PHERC0268_AREA_CM2,
                        surface_id="merged-tiny"))


def test_a_merged_surface_above_the_floor_still_reaches_physical_qc(
    postgres_store, merge_store,
) -> None:
    """The inversion: routing every merge out is as wrong as routing none out."""
    scroll = "PHercMERGEBIG"
    source_id = _snapshot(postgres_store, scroll)
    parents = _merge_parents(postgres_store, source_id, scroll)
    _merge_job(merge_store, "p8-big", scroll)

    registration = merge_store.register_merged_surface(
        _merged_surface(scroll, source_id, parents, area=0.75,
                        surface_id="merged-big"),
        parents, job_id="p8-big", qc_profile_id="surface-qc-mbig@1.0.0")

    assert registration["route"] == routing.STANDARD
    assert registration["qc_state"] == "PENDING"
    claimed = postgres_store.claim_qc("qc-worker", 60,
                                      profile_id="surface-qc-mbig@1.0.0")
    assert claimed is not None and claimed["surface_id"] == "merged-big"


def test_a_merge_with_no_measured_area_writes_no_qc_row_at_all(
    postgres_store, merge_store,
) -> None:
    """Unmeasured is not a route. It is a refusal, before any row exists."""
    scroll = "PHercMERGENONE"
    source_id = _snapshot(postgres_store, scroll)
    parents = _merge_parents(postgres_store, source_id, scroll)
    _merge_job(merge_store, "p8-none", scroll)

    with pytest.raises(RuntimeError, match="no measured area"):
        merge_store.register_merged_surface(
            _merged_surface(scroll, source_id, parents, area=None,
                            surface_id="merged-unmeasured"),
            parents, job_id="p8-none", qc_profile_id="surface-qc-mnone@1.0.0")

    assert postgres_store.routing_receipt("merged-unmeasured") is None
    assert postgres_store.claim_qc(
        "qc-worker", 60, profile_id="surface-qc-mnone@1.0.0") is None


def test_registering_the_same_merge_twice_keeps_the_first_decision(
    postgres_store, merge_store,
) -> None:
    """A replay is idempotent, and the receipt is permanent, so it must not
    collide with itself on the second attempt."""
    scroll = "PHercMERGEREPLAY"
    source_id = _snapshot(postgres_store, scroll)
    parents = _merge_parents(postgres_store, source_id, scroll)
    _merge_job(merge_store, "p8-replay", scroll)
    surface = _merged_surface(scroll, source_id, parents, area=PHERC0268_AREA_CM2,
                              surface_id="merged-replay")

    first = merge_store.register_merged_surface(
        surface, parents, job_id="p8-replay",
        qc_profile_id="surface-qc-mreplay@1.0.0")
    second = merge_store.register_merged_surface(
        surface, parents, job_id="p8-replay",
        qc_profile_id="surface-qc-mreplay@1.0.0")
    assert first == second
    assert second["qc_state"] == "SMALL_SURFACE_DIAGNOSTIC"


# -- parity between the two stores -------------------------------------------

def test_both_stores_write_the_same_receipt_bytes(sqlite_store,
                                                  postgres_store) -> None:
    """The receipt is evidence; two stores disagreeing about it is two records."""
    receipts = []
    for store in (sqlite_store, postgres_store):
        scroll = "PHercPARITY"
        source_id = _snapshot(store, scroll)
        surface_id = _import(store, source_id, scroll, "parity",
                             area=PHERC0268_AREA_CM2,
                             digest=PHERC0268_ARTIFACT_SHA256)
        receipts.append(store.routing_receipt(surface_id))

    assert routing.verify_receipt(receipts[0]) is True
    assert routing.verify_receipt(receipts[1]) is True
    assert json.dumps(receipts[0], sort_keys=True) == json.dumps(
        receipts[1], sort_keys=True)
    assert receipts[0]["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert receipts[1]["measured_area_cm2"] == PHERC0268_AREA_CM2


def test_the_stores_gate_through_the_shared_router_and_not_a_string(
) -> None:
    """A forged receipt has to fail exactly like a missing one.

    Both stores must ask `enters_standard_qc`, which verifies the digest before
    it looks at the route. An inline `route == 'STANDARD_QC_PENDING'` would
    admit a receipt whose route was edited to say so.
    """
    for name in ("store.py", "postgres_store.py"):
        source = (ROOT / "framework/stages/01-segmentation/fleet" / name
                  ).read_text()
        assert "enters_standard_qc" in source, (
            f"{name} does not gate through the shared router")
        assert '!= "STANDARD_QC_PENDING"' not in source, (
            f"{name} compares the route as a bare string, so a forged receipt "
            "passes the gate")


def test_a_forged_route_does_not_pass_the_router() -> None:
    forged = {"schema": routing.RECEIPT_SCHEMA, "surface_id": "s",
              "route": routing.STANDARD, "measured_area_cm2": 0.001,
              "minimum_area_cm2": 0.1, "policy_version": "1.0.0",
              "profile_id": routing.PROFILE_ID, "measurement": {},
              "read_set": {}, "preserved": True, "is_absence_evidence": False,
              "receipt_sha256": "0" * 64}
    assert routing.verify_receipt(forged) is False
    assert routing.enters_standard_qc(forged) is False
    assert routing.enters_canonical_downstream(forged) is False


def test_the_postgres_schema_declares_the_receipt_table() -> None:
    """Parsed from the DDL, so it holds where there is no server."""
    sql = (ROOT / "framework/stages/01-segmentation/fleet/migrations"
           / "001_postgresql.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS segment_surface_routing_receipts" in sql
    assert "VALUES (20," in sql, "the migration ledger does not record v20"
    for column in ("surface_id", "route", "measured_area_cm2",
                   "minimum_area_cm2", "policy_version", "profile_id",
                   "receipt_sha256", "receipt", "created_at"):
        assert column in sql
    assert "a surface routing receipt is immutable" in sql
    assert "a surface routing receipt is permanent" in sql
