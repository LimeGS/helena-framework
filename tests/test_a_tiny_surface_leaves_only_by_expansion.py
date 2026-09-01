"""Leaving the diagnostic path is a new surface, resolved where the row is made.

The policy sentence is short: promotion in place is PROHIBITED, and the original
stays diagnostic permanently. A tiny surface leaves only through a new versioned
grow or resume whose *new* surface independently clears the floor and passes
every standard gate.

Short sentences need a mechanism. The one here is an expansion authority: which
diagnostic surface is being continued, under which routing decision, from which
policy version to which. It is resolved inside the transaction that creates the
successor -- not before it, where the answer could change underneath, and not
after, where the row already exists -- then persisted, read back, and compared
with whatever the caller asserted. A caller that asserts a different authority
than the catalogue resolves is refused rather than believed.

Both stores, because the deployment runs PostgreSQL and a rule that holds only
in SQLite is a rule nothing in production obeys.
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

from fleet import surface_expansion as expansion  # noqa: E402
from fleet import surface_routing as routing  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

DSN = os.environ.get("HELENA_TEST_DSN")

PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")


class _SchemaPostgresStore(PostgresFleetStore):
    def __init__(self, database_url: str, schema: str):
        super().__init__(database_url)
        self._test_schema = schema

    def connect(self):
        connection = super().connect()
        with connection.cursor() as cursor:
            # A literal prefix plus a UUID hex; no caller-supplied text.
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
        pytest.skip("HELENA_TEST_DSN is not set; the PostgreSQL half was not run")
    schema = f"expansion_{uuid.uuid4().hex}"
    bootstrap = PostgresFleetStore(DSN)
    with bootstrap.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {schema}")
    store = _SchemaPostgresStore(DSN, schema)
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
    return request.getfixturevalue(request.param)


def _snapshot(store, scroll: str) -> str:
    return store.register_snapshot({
        "sample_id": scroll, "ct_uri": f"file:///{scroll}/ct",
        "m7_uri": f"file:///{scroll}/m7", "shape_xyz": [1024, 1024, 2048],
        "voxel_size_um": 9.362, "coordinate_frame": "ct_l0_xyz"})


def _import(store, snapshot: str, scroll: str, name: str, *, area, digest: str,
            **extra) -> str:
    return store.import_surface({
        "surface_id": f"{scroll}:{name}", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"s3://bucket/{name}",
        "bbox_xyz": [[0, 0, 0], [64, 64, 8]], "area_cm2": area,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED", **extra})


def _diagnostic_original(store, scroll: str) -> tuple[str, str]:
    snapshot = _snapshot(store, scroll)
    original = _import(store, snapshot, scroll, "original",
                       area=PHERC0268_AREA_CM2, digest=PHERC0268_ARTIFACT_SHA256)
    assert store.routing_receipt(original)["route"] == routing.DIAGNOSTIC
    return snapshot, original


# -- the pure contract --------------------------------------------------------

def test_the_resume_shape_is_validated_before_anything_is_resolved() -> None:
    assert expansion.resume_shape({"surface_id": "s"}) is None
    for broken in ({"resumes_surface": ""}, {"resumes_surface": "   "},
                   {"resumes_surface": 7}, {"resumes_surface": None},
                   {"resumes_surface": ["a"]}):
        with pytest.raises(ValueError, match="resum"):
            expansion.resume_shape(broken)
    shape = expansion.resume_shape(
        {"resumes_surface": "prior", "policy_version": "resume-v2",
         "resume_from": "s3://bucket/prior"})
    assert shape["expands_surface_id"] == "prior"
    assert shape["new_policy_version"] == "resume-v2"
    assert shape["resume_from"] == "s3://bucket/prior"


def test_an_authority_that_promotes_in_place_cannot_be_built() -> None:
    with pytest.raises(ValueError, match="in place|itself"):
        expansion.build_authority(
            expands_surface_id="same", successor_surface_id="same",
            predecessor_route=routing.DIAGNOSTIC,
            predecessor_receipt_sha256="a" * 64,
            prior_policy_version="p1", new_policy_version="p2",
            resume_from=None)


def test_an_authority_verifies_only_against_its_own_bytes() -> None:
    authority = expansion.build_authority(
        expands_surface_id="prior", successor_surface_id="next",
        predecessor_route=routing.DIAGNOSTIC,
        predecessor_receipt_sha256="a" * 64,
        prior_policy_version="p1", new_policy_version="p2",
        resume_from="s3://bucket/prior")
    assert expansion.verify_authority(authority) is True
    assert authority["in_place"] == "PROHIBITED"
    assert authority["original_surface"] == "REMAINS_DIAGNOSTIC_PERMANENTLY"
    forged = {**authority, "expands_surface_id": "somebody-else"}
    assert expansion.verify_authority(forged) is False


def test_leaving_the_diagnostic_path_needs_both_halves() -> None:
    """A diagnostic predecessor and a successor that clears the floor itself."""
    authority = expansion.build_authority(
        expands_surface_id="prior", successor_surface_id="next",
        predecessor_route=routing.DIAGNOSTIC,
        predecessor_receipt_sha256="a" * 64,
        prior_policy_version="p1", new_policy_version="p2", resume_from=None)
    policy = routing.load_policy()
    big = routing.build_receipt(surface_id="next", area_cm2=0.5, policy=policy,
                                measurement={}, read_set={})
    small = routing.build_receipt(surface_id="next", area_cm2=0.02,
                                  policy=policy, measurement={}, read_set={})
    assert expansion.leaves_diagnostic(authority, big) is True
    assert expansion.leaves_diagnostic(authority, small) is False
    # A tampered authority answers no, whatever the successor measured.
    assert expansion.leaves_diagnostic(
        {**authority, "predecessor_route": routing.DIAGNOSTIC,
         "expands_surface_id": "somebody-else"}, big) is False
    assert expansion.leaves_diagnostic(
        {**authority, "authority_sha256": "0" * 64}, big) is False


# -- resolved where the successor is created ---------------------------------

def test_an_expansion_records_the_diagnostic_surface_it_continues(store) -> None:
    scroll = "PHercLINEAGE"
    snapshot, original = _diagnostic_original(store, scroll)

    expanded = _import(store, snapshot, scroll, "resumed", area=1.5,
                       digest="6" * 64, resumes_surface=original)

    receipt = store.routing_receipt(expanded)
    assert receipt["read_set"]["expands_surface_id"] == original, (
        "the expansion's own receipt does not say which diagnostic surface it "
        "continues, so the promotion contract is not auditable from the receipt")
    authority = store.expansion_authority(expanded)
    assert authority is not None
    assert expansion.verify_authority(authority) is True
    assert authority["expands_surface_id"] == original
    assert authority["predecessor_route"] == routing.DIAGNOSTIC
    assert authority["predecessor_receipt_sha256"] == (
        store.routing_receipt(original)["receipt_sha256"])
    assert expansion.leaves_diagnostic(authority, receipt) is True


def test_the_original_stays_diagnostic_after_a_successful_expansion(store) -> None:
    scroll = "PHercSTAYS"
    snapshot, original = _diagnostic_original(store, scroll)
    before = store.routing_receipt(original)

    _import(store, snapshot, scroll, "resumed", area=1.5, digest="6" * 64,
            resumes_surface=original)

    assert store.routing_receipt(original) == before
    assert store.routing_receipt(original)["route"] == routing.DIAGNOSTIC
    assert routing.load_policy()["promotion"]["in_place"] == "PROHIBITED"


def test_an_expansion_that_is_still_too_small_is_still_diagnostic(store) -> None:
    """The authority permits the attempt. The measurement decides the outcome."""
    scroll = "PHercSTILLSMALL"
    snapshot, original = _diagnostic_original(store, scroll)

    smaller = _import(store, snapshot, scroll, "resumed", area=0.03,
                      digest="7" * 64, resumes_surface=original)

    receipt = store.routing_receipt(smaller)
    assert receipt["route"] == routing.DIAGNOSTIC
    assert expansion.leaves_diagnostic(
        store.expansion_authority(smaller), receipt) is False


def test_an_expansion_of_itself_is_refused(store) -> None:
    scroll = "PHercSELF"
    snapshot, original = _diagnostic_original(store, scroll)
    with pytest.raises((RuntimeError, ValueError), match="in place|itself"):
        store.import_surface({
            "surface_id": original, "source_snapshot_id": snapshot,
            "sample_id": scroll, "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
            "artifact_uri": "s3://bucket/self", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
            "area_cm2": 9.99, "resumes_surface": original})
    assert store.routing_receipt(original)["route"] == routing.DIAGNOSTIC
    assert store.routing_receipt(original)["measured_area_cm2"] == (
        PHERC0268_AREA_CM2)


def test_an_expansion_of_a_surface_that_does_not_exist_is_refused(store) -> None:
    scroll = "PHercGHOST"
    snapshot = _snapshot(store, scroll)
    with pytest.raises((RuntimeError, ValueError), match="unknown|expan|resum"):
        _import(store, snapshot, scroll, "resumed", area=1.5, digest="6" * 64,
                resumes_surface=f"{scroll}:never-existed")
    assert store.routing_receipt(f"{scroll}:resumed") is None


def test_an_expansion_of_an_unrouted_surface_is_refused(store) -> None:
    """An expansion claims something about the surface it continues.

    A surface with no routing decision has nothing to claim about, so an
    expansion of one is refused rather than recorded against nothing.
    """
    scroll = "PHercUNROUTED"
    snapshot = _snapshot(store, scroll)
    unrouted = _import(store, snapshot, scroll, "unmeasured", area=None,
                       digest="8" * 64)
    assert store.routing_receipt(unrouted) is None

    with pytest.raises((RuntimeError, ValueError), match="rout|decision"):
        _import(store, snapshot, scroll, "resumed", area=1.5, digest="9" * 64,
                resumes_surface=unrouted)


def test_an_asserted_authority_that_differs_from_the_catalogue_is_refused(
    store,
) -> None:
    """The compare step. A caller may assert; it may not decide."""
    scroll = "PHercASSERT"
    snapshot, original = _diagnostic_original(store, scroll)
    forged = expansion.build_authority(
        expands_surface_id=original, successor_surface_id=f"{scroll}:resumed",
        predecessor_route=routing.STANDARD,  # it is not
        predecessor_receipt_sha256="0" * 64,
        prior_policy_version=None, new_policy_version=None, resume_from=None)

    with pytest.raises((RuntimeError, ValueError), match="authority|differ"):
        _import(store, snapshot, scroll, "resumed", area=1.5, digest="6" * 64,
                resumes_surface=original, expansion_authority=forged)

    assert store.expansion_authority(f"{scroll}:resumed") is None


def test_a_matching_asserted_authority_is_accepted(store) -> None:
    scroll = "PHercMATCH"
    snapshot, original = _diagnostic_original(store, scroll)
    resolved = store.resolve_expansion_authority(
        successor_surface_id=f"{scroll}:resumed",
        source={"resumes_surface": original})

    expanded = _import(store, snapshot, scroll, "resumed", area=1.5,
                       digest="6" * 64, resumes_surface=original,
                       expansion_authority=resolved)

    assert store.expansion_authority(expanded) == resolved


def test_the_authority_is_immutable_once_written(store) -> None:
    scroll = "PHercIMMUTABLE"
    snapshot, original = _diagnostic_original(store, scroll)
    expanded = _import(store, snapshot, scroll, "resumed", area=1.5,
                       digest="6" * 64, resumes_surface=original)
    before = store.expansion_authority(expanded)

    if isinstance(store, FleetStore):
        import sqlite3
        with store.connect() as connection:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                connection.execute(
                    "UPDATE surface_expansion_authorities SET expands_surface_id=?"
                    " WHERE successor_surface_id=?", ("other", expanded))
            with pytest.raises(sqlite3.IntegrityError, match="permanent"):
                connection.execute(
                    "DELETE FROM surface_expansion_authorities "
                    "WHERE successor_surface_id=?", (expanded,))
    else:
        import psycopg2
        for statement, arguments, message in (
            ("UPDATE segment_surface_expansion_authorities "
             "SET expands_surface_id=%s WHERE successor_surface_id=%s",
             ("other", expanded), "immutable"),
            ("DELETE FROM segment_surface_expansion_authorities "
             "WHERE successor_surface_id=%s", (expanded,), "permanent"),
        ):
            with store.connect() as connection:
                with connection.cursor() as cursor:
                    with pytest.raises(psycopg2.Error, match=message):
                        cursor.execute(statement, arguments)

    assert store.expansion_authority(expanded) == before


def test_an_ordinary_surface_records_no_expansion(store) -> None:
    """The mechanism must not attach itself to work it was not written for."""
    scroll = "PHercORDINARY"
    snapshot = _snapshot(store, scroll)
    ordinary = _import(store, snapshot, scroll, "plain", area=0.5,
                       digest="a" * 64)
    assert store.expansion_authority(ordinary) is None
    assert store.routing_receipt(ordinary)["read_set"][
        "expands_surface_id"] is None


# -- the finalization transaction --------------------------------------------

def _resume_task(store, source_id: str, scroll: str, *, cell: str,
                 policy_version: str, resumes: str, resume_from: str) -> dict:
    store.create_tasks([{
        "source_snapshot_id": source_id, "sample_id": scroll, "cell_id": cell,
        "grid_version": "g1", "policy_version": policy_version,
        "bounds_xyz": [[0, 0, 0], [64, 64, 8]], "center_xyz": [32.0, 32.0, 4.0],
        "priority": 1.0, "parameter_envelope": {},
        "catalog_snapshot_sha256": "c" * 64,
        "resumes_surface": resumes, "resume_from": resume_from,
        "corrections": "none"}])
    task = store.claim(worker_id="w1", lease_seconds=600)
    assert task is not None
    return task


def _finalize(store, task: dict, source_id: str, scroll: str, *, surface_id: str,
              area: float, digest: str) -> dict:
    locked_plan = {"schema": "test.locked_plan.v1", "task_id": task["task_id"]}
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"],
                     "RUNNING", locked_plan=locked_plan)
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": task["task_id"], "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan), "files": {},
        "artifact_sha256": digest, "bbox_xyz": [[0, 0, 0], [64, 64, 8]],
        "sample_points": [[0.0, 0.0, 0.0]], "area_cm2": area, "ink_used": False}
    artifact_set_id = store.add_artifact_set(
        task["task_id"], task["attempt_id"], task["lease_token"], manifest,
        "file:///staging")
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"],
                     "FINALIZING")
    surface = {
        "schema": "campaignx.segment_fleet_surface.v1",
        "surface_id": surface_id, "source_snapshot_id": source_id,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"file:///artifacts/{surface_id}",
        "bbox_xyz": manifest["bbox_xyz"],
        "sample_points": manifest["sample_points"], "area_cm2": area,
        "geometry_qc_state": "GEOMETRY_CERTIFIED", "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan), "ink_used": False}
    return store.finalize(task["task_id"], task["attempt_id"],
                          task["lease_token"], surface, artifact_set_id,
                          f"surface-qc-{surface_id}@1.0.0")


def test_a_resumed_grow_resolves_its_authority_inside_finalization(store) -> None:
    scroll = "PHercGROWN"
    snapshot, original = _diagnostic_original(store, scroll)
    task = _resume_task(store, snapshot, scroll, cell=f"resume-{original}"[:32],
                        policy_version="resume-corrections-v1",
                        resumes=original, resume_from="s3://bucket/original")

    outcome = _finalize(store, task, snapshot, scroll,
                        surface_id=f"{scroll}:grown", area=0.42,
                        digest="b" * 64)

    assert outcome["status"] == "QC_PENDING"
    authority = store.expansion_authority(f"{scroll}:grown")
    assert authority is not None and expansion.verify_authority(authority)
    assert authority["expands_surface_id"] == original
    assert authority["new_policy_version"] == "resume-corrections-v1"
    receipt = store.routing_receipt(f"{scroll}:grown")
    assert receipt["route"] == routing.STANDARD
    assert receipt["read_set"]["expands_surface_id"] == original
    assert expansion.leaves_diagnostic(authority, receipt) is True
    # And the original is exactly as it was measured.
    assert store.routing_receipt(original)["route"] == routing.DIAGNOSTIC


def test_the_task_stamps_the_authority_and_finalization_compares_it(
    store,
) -> None:
    """Resolved twice under a lock, and the two must agree.

    Task creation resolves it; finalization resolves it again inside its own
    transaction and refuses if the answer moved. A stamp nobody rechecks is a
    claim about a catalogue as it was some time ago.
    """
    scroll = "PHercSTAMP"
    snapshot, original = _diagnostic_original(store, scroll)
    task = _resume_task(store, snapshot, scroll, cell="resume-stamp",
                        policy_version="resume-corrections-v1",
                        resumes=original, resume_from="s3://bucket/original")

    stamped = task["expansion_authority"]
    assert expansion.verify_authority(stamped) is True
    assert stamped["expands_surface_id"] == original

    _finalize(store, task, snapshot, scroll, surface_id=f"{scroll}:grown",
              area=0.42, digest="b" * 64)
    persisted = store.expansion_authority(f"{scroll}:grown")
    assert persisted["expands_surface_id"] == stamped["expands_surface_id"]
    assert persisted["predecessor_receipt_sha256"] == (
        stamped["predecessor_receipt_sha256"])


def test_a_resume_under_the_predecessors_own_policy_version_is_refused(
    store,
) -> None:
    """"A new versioned attempt" is the contract, so the version has to be new."""
    scroll = "PHercSAMEVERSION"
    snapshot = _snapshot(store, scroll)
    original = _import(store, snapshot, scroll, "original",
                       area=PHERC0268_AREA_CM2,
                       digest=PHERC0268_ARTIFACT_SHA256,
                       policy_version="resume-corrections-v1")

    with pytest.raises((RuntimeError, ValueError), match="version"):
        _resume_task(store, snapshot, scroll, cell="resume-same",
                     policy_version="resume-corrections-v1",
                     resumes=original, resume_from="s3://bucket/original")


# -- parity ------------------------------------------------------------------

def test_both_stores_resolve_the_same_authority(sqlite_store,
                                                postgres_store) -> None:
    authorities = []
    for one in (sqlite_store, postgres_store):
        scroll = "PHercPARITY"
        snapshot, original = _diagnostic_original(one, scroll)
        expanded = _import(one, snapshot, scroll, "resumed", area=1.5,
                           digest="6" * 64, resumes_surface=original)
        authorities.append(one.expansion_authority(expanded))

    assert json.dumps(authorities[0], sort_keys=True) == json.dumps(
        authorities[1], sort_keys=True)
    assert expansion.verify_authority(authorities[0]) is True


def test_the_public_postgres_wrapper_delegates_to_the_transactional_resolver(
) -> None:
    """Read at its source: a second implementation is a second set of rules.

    A public wrapper that re-derived the authority outside a transaction would
    answer from a catalogue that could move between the read and the write,
    which is exactly the failure the transactional resolver exists to prevent.
    """
    source = (ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py"
              ).read_text()
    start = source.index("def resolve_expansion_authority")
    body = source[start:][: source[start:].index("\n    def ")]
    assert "self._resolve_expansion_authority(" in body, (
        "the public PostgreSQL wrapper does not delegate to the transactional "
        "resolver")
    for duplicated in ("build_authority(", "SELECT route", "surface_routing."):
        assert duplicated not in body, (
            f"the public wrapper restates {duplicated} instead of delegating")


def test_the_postgres_schema_declares_the_expansion_table() -> None:
    sql = (ROOT / "framework/stages/01-segmentation/fleet/migrations"
           / "001_postgresql.sql").read_text()
    assert (
        "CREATE TABLE IF NOT EXISTS segment_surface_expansion_authorities" in sql)
    assert "VALUES (21," in sql
    assert "a surface expansion authority is immutable" in sql
    assert "a surface expansion authority is permanent" in sql
