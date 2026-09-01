"""Detectors for the mutations that would quietly re-open the PHerc0268 path.

A test that passes on the broken code is worse than no test: it is a green tick
where a gate used to be. So each check here was written against a deliberate
break -- the floor comparison flipped, a digest field dropped, the queue gate
inverted, the route strings swapped -- and kept only if that break turned it red.
The mutations no detector caught are reported rather than papered over.

Four surfaces of the same decision:

* **route transitions** -- STANDARD_QC_PENDING and SMALL_SURFACE_DIAGNOSTIC, the
  boundary between them, and every field the digest has to cover for a stored
  receipt to mean anything.
* **expansion** -- the only way out of the diagnostic path is a new versioned
  grow whose *new* surface clears the floor on its own measurement. Nothing
  promotes a surface in place, because that would be claiming the measurement
  was wrong, and the measurement is the evidence.
* **P8** -- a merge produces a new surface, so it produces a new routing
  question, and a merged sheet under the floor is as diagnostic as a grown one.
* **review** -- a person calling a surface good is not a measurement. It may not
  move a surface across the floor.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
for extra in (STAGE, ROOT / "framework/stages/03-ink/fleet"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fleet import surface_routing as routing  # noqa: E402
from fleet.store import (  # noqa: E402
    QC_SMALL_SURFACE_DIAGNOSTIC,
    FleetStore,
)

PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")
FLOOR = 0.10


@pytest.fixture
def policy() -> dict:
    return routing.load_policy()


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _surface(store, *, name: str, area, digest: str = "a" * 64) -> tuple[str, str]:
    scroll = f"TEST{name}"
    snapshot = store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    payload = {
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"s3://bucket/{name}",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"}
    if area is not None:
        payload["area_cm2"] = area
    return snapshot, store.import_surface(payload)


def _receipt(policy, *, surface_id="s-1", area=PHERC0268_AREA_CM2) -> dict:
    return routing.build_receipt(
        surface_id=surface_id, area_cm2=area, policy=policy,
        measurement={"grid_xy": [14, 14], "triangles": 132},
        read_set={"source_snapshot_id": "snap-1",
                  "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
                  "geometry_qc_state": "GEOMETRY_CERTIFIED"})


# ---------------------------------------------------------------------------
# Route transitions
# ---------------------------------------------------------------------------

def test_the_floor_is_closed_at_exactly_the_threshold(policy) -> None:
    """`>=` and `>` differ on one value, and that value is the policy.

    A surface measuring exactly 0.10 cm2 is at the floor the effort allocator
    already spends against, so it takes the standard path. Flipping the
    comparison quarantines it, and quarantining a surface for being exactly at
    the threshold is the sort of thing nobody notices for a campaign.
    """
    assert routing.route(FLOOR, policy=policy)[0] == routing.STANDARD
    assert routing.route(FLOOR - 1e-12, policy=policy)[0] == routing.DIAGNOSTIC
    # And it is the *policy's* floor, not a number that happens to equal it.
    assert policy["minimum_area_cm2"] == FLOOR


def test_the_two_routes_are_not_interchangeable(policy) -> None:
    """Swapping the two return values must not leave the module self-consistent."""
    assert routing.route(1.0, policy=policy)[0] != routing.route(
        PHERC0268_AREA_CM2, policy=policy)[0]
    assert routing.enters_standard_qc(_receipt(policy, area=1.0)) is True
    assert routing.enters_standard_qc(
        _receipt(policy, area=PHERC0268_AREA_CM2)) is False
    assert set(routing.ROUTES) == {routing.STANDARD, routing.DIAGNOSTIC}


@pytest.mark.parametrize("field,forged", [
    ("route", routing.STANDARD),
    ("measured_area_cm2", 5.0),
    ("minimum_area_cm2", 0.001),
    ("policy_version", "9.9.9"),
    ("profile_id", "small-surface-routing@0.0.1"),
    ("surface_id", "somebody-elses-surface"),
    ("preserved", False),
    ("is_absence_evidence", True),
    ("measurement", {"triangles": 999999}),
    ("read_set", {"artifact_sha256": "0" * 64}),
])
def test_the_digest_covers_every_field_the_decision_turns_on(
    policy, field, forged,
) -> None:
    """One case per digest field, so dropping any one of them turns this red.

    A single "tampering is detected" test passes while nine of ten fields are
    uncovered, because it only ever alters the tenth.
    """
    receipt = _receipt(policy)
    assert field in routing._DIGEST_FIELDS, (  # noqa: SLF001
        f"{field} left the digest; the receipt no longer covers it")
    assert routing.verify_receipt({**receipt, field: forged}) is False, (
        f"changing {field} left the receipt verifying")


def test_admission_asks_the_digest_and_not_the_route_string(policy) -> None:
    """`route == STANDARD` is the mutation; `verify and route == STANDARD` is not."""
    forged = {**_receipt(policy), "route": routing.STANDARD}
    assert forged["route"] == routing.STANDARD
    assert routing.enters_standard_qc(forged) is False
    assert routing.enters_canonical_downstream(forged) is False
    for absent in ({}, {"route": routing.STANDARD}, None, "STANDARD_QC_PENDING"):
        assert routing.enters_standard_qc(absent) is False
        assert routing.enters_canonical_downstream(absent) is False


def test_an_unmeasured_area_never_resolves_to_a_route(policy) -> None:
    """Every fallback here is a decision the data does not support."""
    for bad in (None, -1.0, float("nan"), float("inf"), "0.5", True, False,
                [0.5], {"area": 0.5}):
        with pytest.raises(ValueError):
            routing.route(bad, policy=policy)
        with pytest.raises(ValueError):
            routing.build_receipt(surface_id="s", area_cm2=bad, policy=policy,
                                  measurement={}, read_set={})


def test_the_queue_gate_turns_on_the_stored_route(store) -> None:
    """Inverting or deleting the gate in `store.py` must turn this red."""
    _, tiny = _surface(store, name="gatetiny", area=PHERC0268_AREA_CM2,
                       digest=PHERC0268_ARTIFACT_SHA256)
    store.enqueue_imported_surface_qc({
        "surface_id": tiny, "source_snapshot_id": _snapshot_of(store, tiny),
        "sample_id": "TESTgatetiny",
        "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/gatetiny", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": PHERC0268_AREA_CM2, "geometry_qc_state": "GEOMETRY_CERTIFIED",
    }, profile_id="surface-qc-gate@1.0.0")
    assert _qc_state(store, tiny) == QC_SMALL_SURFACE_DIAGNOSTIC
    assert store.claim_qc("w", 60, profile_id="surface-qc-gate@1.0.0") is None


def test_the_queue_gate_does_not_quarantine_a_normal_surface(store) -> None:
    """The other half of the inversion: `!=` flipped to `==` must also be caught."""
    _, big = _surface(store, name="gatebig", area=0.5, digest="c" * 64)
    store.record_geometry_certification(
        big, "GEOMETRY_CERTIFIED", {"schema": "test"},
        requested_by_job_id="p2", profile_id="geometry-test@1",
        profile_sha256="6" * 64)
    store.enqueue_imported_surface_qc({
        "surface_id": big, "source_snapshot_id": _snapshot_of(store, big),
        "sample_id": "TESTgatebig", "artifact_sha256": "c" * 64,
        "artifact_uri": "s3://bucket/gatebig", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": 0.5,
    }, profile_id="surface-qc-gate-ok@1.0.0")
    assert _qc_state(store, big) == "PENDING"
    claimed = store.claim_qc("w", 60, profile_id="surface-qc-gate-ok@1.0.0")
    assert claimed is not None and claimed["surface_id"] == big


def test_a_route_never_changes_after_it_is_written(store) -> None:
    """The database refuses, so no future caller has to remember not to."""
    _, surface = _surface(store, name="frozen", area=0.5, digest="b" * 64)
    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE surface_routing_receipts SET route=? WHERE surface_id=?",
                (routing.DIAGNOSTIC, surface))
        with pytest.raises(sqlite3.IntegrityError, match="permanent"):
            connection.execute(
                "DELETE FROM surface_routing_receipts WHERE surface_id=?",
                (surface,))
    assert store.routing_receipt(surface)["route"] == routing.STANDARD


def _qc_state(store, surface_id: str) -> str | None:
    with store.connect() as connection:
        row = connection.execute(
            "SELECT state FROM qc_jobs WHERE surface_id=?", (surface_id,)
        ).fetchone()
    return row["state"] if row is not None else None


def _snapshot_of(store, surface_id: str) -> str:
    with store.connect() as connection:
        row = connection.execute(
            "SELECT source_snapshot_id FROM surfaces WHERE surface_id=?",
            (surface_id,)).fetchone()
    return row["source_snapshot_id"]


# ---------------------------------------------------------------------------
# Expansion: the only way off the diagnostic path
# ---------------------------------------------------------------------------

def test_the_expanded_surface_is_measured_on_its_own_evidence(store) -> None:
    """A resume produces a new surface, and the new surface is routed afresh.

    Not inherited from the parent in either direction: an expansion that
    happened to stay small is still diagnostic, and one that cleared the floor
    is standard *because it measured that*, not because a resume happened.
    """
    _, original = _surface(store, name="expandsrc", area=PHERC0268_AREA_CM2,
                           digest=PHERC0268_ARTIFACT_SHA256)
    assert store.routing_receipt(original)["route"] == routing.DIAGNOSTIC

    snapshot = _snapshot_of(store, original)
    grown = store.import_surface({
        "surface_id": "TESTexpandsrc-s-resumed", "source_snapshot_id": snapshot,
        "sample_id": "TESTexpandsrc", "artifact_sha256": "9" * 64,
        "artifact_uri": "s3://bucket/expanded",
        "bbox_xyz": [[0, 0, 0], [64, 64, 8]], "area_cm2": 0.42,
        "resumes_surface": original,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    still_small = store.import_surface({
        "surface_id": "TESTexpandsrc-s-resumed-2", "source_snapshot_id": snapshot,
        "sample_id": "TESTexpandsrc", "artifact_sha256": "8" * 64,
        "artifact_uri": "s3://bucket/expanded-2",
        "bbox_xyz": [[0, 0, 0], [20, 20, 4]], "area_cm2": 0.03,
        "resumes_surface": original,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})

    assert store.routing_receipt(grown)["route"] == routing.STANDARD
    assert store.routing_receipt(still_small)["route"] == routing.DIAGNOSTIC
    # And the original is untouched by either.
    assert store.routing_receipt(original)["route"] == routing.DIAGNOSTIC


def test_an_expansion_cannot_promote_the_surface_it_resumes(store) -> None:
    """"Promotion in place is PROHIBITED" is a policy sentence and a database rule."""
    _, original = _surface(store, name="noinplace", area=PHERC0268_AREA_CM2,
                           digest=PHERC0268_ARTIFACT_SHA256)
    before = store.routing_receipt(original)
    snapshot = _snapshot_of(store, original)
    store.import_surface({
        "surface_id": "TESTnoinplace-s-resumed", "source_snapshot_id": snapshot,
        "sample_id": "TESTnoinplace", "artifact_sha256": "7" * 64,
        "artifact_uri": "s3://bucket/noinplace-resumed",
        "bbox_xyz": [[0, 0, 0], [64, 64, 8]], "area_cm2": 1.5,
        "resumes_surface": original,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    assert store.routing_receipt(original) == before
    assert routing.load_policy()["promotion"]["in_place"] == "PROHIBITED"
    assert (routing.load_policy()["promotion"]["original_surface"]
            == "REMAINS_DIAGNOSTIC_PERMANENTLY")


def test_re_importing_the_original_with_a_larger_area_does_not_promote_it(
    store,
) -> None:
    """The cheapest in-place promotion there is, and it must not work.

    `import_surface` is ON CONFLICT DO NOTHING, so the surface does not change
    -- and if the receipt were rewritten from the new payload the two would
    disagree, with the receipt saying the surface is something it is not.
    """
    _, original = _surface(store, name="reimport", area=PHERC0268_AREA_CM2,
                           digest=PHERC0268_ARTIFACT_SHA256)
    snapshot = _snapshot_of(store, original)
    # The attempt is now refused outright rather than quietly having its area
    # dropped. Both endings leave the surface and its receipt untouched, which
    # is what this test is for; the refusal additionally denies the caller the
    # false impression that a 9.99 cm2 surface was stored.
    with pytest.raises((RuntimeError, ValueError), match="conflict|differs|refus"):
        store.import_surface({
            "surface_id": original, "source_snapshot_id": snapshot,
            "sample_id": "TESTreimport",
            "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
            "artifact_uri": "s3://bucket/reimport",
            "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": 9.99,
            "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    receipt = store.routing_receipt(original)
    assert receipt["route"] == routing.DIAGNOSTIC
    assert receipt["measured_area_cm2"] == PHERC0268_AREA_CM2


def test_an_expansion_records_which_diagnostic_surface_it_continues(store) -> None:
    _, original = _surface(store, name="lineage", area=PHERC0268_AREA_CM2,
                           digest=PHERC0268_ARTIFACT_SHA256)
    snapshot = _snapshot_of(store, original)
    expanded = store.import_surface({
        "surface_id": "TESTlineage-s-resumed", "source_snapshot_id": snapshot,
        "sample_id": "TESTlineage", "artifact_sha256": "6" * 64,
        "artifact_uri": "s3://bucket/lineage-resumed",
        "bbox_xyz": [[0, 0, 0], [64, 64, 8]], "area_cm2": 1.5,
        "resumes_surface": original,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    receipt = store.routing_receipt(expanded)
    assert receipt["read_set"].get("expands_surface_id") == original, (
        "the expansion's own receipt does not say which diagnostic surface it "
        "continues, so the promotion contract is not auditable from the receipt")


# ---------------------------------------------------------------------------
# P8: a merge makes a surface, so a merge makes a routing question
# ---------------------------------------------------------------------------

def test_a_p8_merge_is_admitted_only_when_it_names_its_own_surface() -> None:
    """The binding a mutation would drop: publication vs receipt surface_id."""
    from ink_worker import merge_result_from_receipt

    job = {"phase": "P8", "parameters": {"lane": "vc3d-tifxyz-merge"}}
    receipt = {"schema": "campaignx.vc3d_tifxyz_merge_receipt.v1",
               "status": "PASS", "surface_id": "merged-1",
               "artifact_uri": "s3://bucket/merged", "artifact_sha256": "1" * 64,
               # A merged surface carries its own measured area, so a receipt
               # standing in for a real one carries it too.
               "area_cm2": 0.42, "parents": ["a", "b"]}
    publication = {
        "schema": "campaignx.vc3d_merge_evidence_publication.v1",
        "evidence_uri": "s3://bucket/evidence", "evidence_sha256": "2" * 64,
        "registration": {"surface_id": "merged-1"}}
    assert merge_result_from_receipt(
        job, receipt, publication)["surface_id"] == "merged-1"

    for broken, match in (
        ({**publication, "registration": {"surface_id": "somebody-else"}},
         "different registered surface"),
        (None, "no EVIDENCE_PUBLICATION"),
    ):
        with pytest.raises(RuntimeError, match=match):
            merge_result_from_receipt(job, receipt, broken)
    with pytest.raises(RuntimeError, match="does not record PASS"):
        merge_result_from_receipt(job, {**receipt, "status": "FAIL"}, publication)


def test_a_merged_surface_under_the_floor_is_routed_diagnostic(store) -> None:
    """A merge is not an exemption. The floor is about size, and so is a merge."""
    scroll = "TESTmerged"
    snapshot = store.register_snapshot({
        "sample_id": scroll, "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    merged = store.import_surface({
        "surface_id": "merged-tiny", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": "3" * 64,
        "artifact_uri": "s3://bucket/merged-tiny",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": 0.02,
        "parents": ["parent-a", "parent-b"],
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    receipt = store.routing_receipt(merged)
    assert receipt["route"] == routing.DIAGNOSTIC
    assert routing.enters_canonical_downstream(receipt) is False


def test_a_p8_merge_reports_the_route_of_the_surface_it_produced() -> None:
    from ink_worker import merge_result_from_receipt

    job = {"phase": "P8", "parameters": {"lane": "vc3d-tifxyz-merge"}}
    receipt = {"schema": "campaignx.vc3d_tifxyz_merge_receipt.v1",
               "status": "PASS", "surface_id": "merged-tiny",
               "artifact_uri": "s3://bucket/merged", "artifact_sha256": "1" * 64,
               "area_cm2": 0.02, "parents": ["a", "b"]}
    publication = {
        "schema": "campaignx.vc3d_merge_evidence_publication.v1",
        "evidence_uri": "s3://bucket/evidence", "evidence_sha256": "2" * 64,
        "registration": {"surface_id": "merged-tiny"}}
    result = merge_result_from_receipt(job, receipt, publication)
    assert result.get("route") == routing.DIAGNOSTIC, (
        "P8 published a 0.02 cm2 merged surface with no routing decision")
    # And the gate answers, from the router's predicates rather than from the
    # string above, so dropping the digest check leaves this red.
    assert result["enters_canonical_downstream"] is False
    assert result["enters_standard_qc"] is False
    assert result["area_cm2"] == 0.02
    assert routing.verify_receipt(result["routing_receipt"]) is True


def _merge_documents(area, **receipt_changes) -> tuple[dict, dict, dict]:
    job = {"phase": "P8", "parameters": {"lane": "vc3d-tifxyz-merge"}}
    receipt = {"schema": "campaignx.vc3d_tifxyz_merge_receipt.v1",
               "status": "PASS", "surface_id": "merged-x",
               "artifact_uri": "s3://bucket/merged", "artifact_sha256": "1" * 64,
               "parents": ["a", "b"], **receipt_changes}
    if area is not None:
        receipt["area_cm2"] = area
    publication = {
        "schema": "campaignx.vc3d_merge_evidence_publication.v1",
        "evidence_uri": "s3://bucket/evidence", "evidence_sha256": "2" * 64,
        "registration": {"surface_id": "merged-x"}}
    return job, receipt, publication


def test_a_merge_that_measured_nothing_is_refused_rather_than_left_unrouted(
    policy,
) -> None:
    """The mutation this catches is the whole original defect, restored.

    Making the area optional -- reporting a route when one can be computed and
    staying quiet when it cannot -- looks careful and reinstates the exact door:
    a PASS merge whose area never reached the receipt goes to the queue result
    with no decision on it. The lane always measures, so a receipt without an
    area is truncated, forged, or written by something that did not measure, and
    all three fail closed.
    """
    from ink_worker import merge_result_from_receipt

    for missing in (None, "0.5", True, -1.0, float("nan"), float("inf")):
        job, receipt, publication = _merge_documents(missing)
        with pytest.raises(RuntimeError, match="cannot be routed"):
            merge_result_from_receipt(job, receipt, publication)


def test_a_merge_receipt_cannot_carry_a_route_its_own_area_refutes(policy) -> None:
    """A route written into a document does not outrank the area beside it.

    This is `agrees_with_measurement` applied at the P8 boundary: the cheapest
    promotion available to a merge lane is to publish a real 0.02 cm2 sheet and
    staple a STANDARD receipt for some other area to it.
    """
    from ink_worker import merge_result_from_receipt

    honest = routing.build_receipt(
        surface_id="merged-x", area_cm2=0.02, policy=policy,
        measurement={"decided_at": "P8_MERGE_RESULT_PROJECTION"},
        read_set={"artifact_sha256": "1" * 64})
    job, receipt, publication = _merge_documents(0.02, routing_receipt=honest)
    assert merge_result_from_receipt(
        job, receipt, publication)["route"] == routing.DIAGNOSTIC

    # A perfectly valid receipt -- correct digest, verifies -- for a surface
    # that measured 5 cm2, attached to a merge that measured 0.02.
    forged = routing.build_receipt(
        surface_id="merged-x", area_cm2=5.0, policy=policy,
        measurement={"decided_at": "P8_MERGE_RESULT_PROJECTION"},
        read_set={"artifact_sha256": "1" * 64})
    assert routing.verify_receipt(forged) is True
    assert routing.enters_standard_qc(forged) is True
    job, receipt, publication = _merge_documents(0.02, routing_receipt=forged)
    with pytest.raises(RuntimeError, match="not the decision"):
        merge_result_from_receipt(job, receipt, publication)

    # And one whose digest no longer covers what it says.
    job, receipt, publication = _merge_documents(
        0.02, routing_receipt={**honest, "route": routing.STANDARD})
    with pytest.raises(RuntimeError, match="not the decision"):
        merge_result_from_receipt(job, receipt, publication)

    # Somebody else's surface, correctly signed for that surface.
    other = routing.build_receipt(
        surface_id="a-different-surface", area_cm2=0.02, policy=policy,
        measurement={"decided_at": "P8_MERGE_RESULT_PROJECTION"},
        read_set={"artifact_sha256": "1" * 64})
    job, receipt, publication = _merge_documents(0.02, routing_receipt=other)
    with pytest.raises(RuntimeError, match="not the decision"):
        merge_result_from_receipt(job, receipt, publication)


def test_a_merged_surface_over_the_floor_takes_the_standard_path() -> None:
    """The other half of the inversion: routing must not quarantine every merge."""
    from ink_worker import merge_result_from_receipt

    job, receipt, publication = _merge_documents(FLOOR)
    result = merge_result_from_receipt(job, receipt, publication)
    assert result["route"] == routing.STANDARD
    assert result["enters_canonical_downstream"] is True
    assert result["enters_standard_qc"] is True


# ---------------------------------------------------------------------------
# Review: a person is not a measurement
# ---------------------------------------------------------------------------

def test_a_human_verdict_does_not_move_a_surface_across_the_floor(
    store, policy,
) -> None:
    """APPROVED on a 2 mm2 surface is an opinion about a diagnostic surface.

    The review is stored in the surface payload. Whatever it says, the routing
    receipt is unchanged and the surface is still not admissible.
    """
    _, surface = _surface(store, name="reviewed", area=PHERC0268_AREA_CM2,
                          digest=PHERC0268_ARTIFACT_SHA256)
    before = store.routing_receipt(surface)
    with store.connect() as connection:
        connection.execute(
            "UPDATE surfaces SET payload_json=json_set(payload_json,'$.human_review',"
            "json('{\"verdict\":\"APPROVED\",\"by\":\"someone\"}')) "
            "WHERE surface_id=?", (surface,))
        connection.commit()
    after = store.routing_receipt(surface)
    assert after == before
    assert routing.enters_standard_qc(after) is False
    assert routing.enters_canonical_downstream(after) is False


def test_the_review_verdicts_are_a_closed_set() -> None:
    """Adding a fifth verdict is how "PROMOTED" would arrive."""
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    assert set(panel_app.HUMAN_REVIEW_VERDICTS) == {
        "APPROVED", "DEFECTIVE", "REVIEWED", "INSPECT"}
    for verdict in panel_app.HUMAN_REVIEW_VERDICTS:
        assert "PROMOT" not in verdict and "ROUTE" not in verdict, (
            f"{verdict} reads as a routing action; a human verdict is not one")


def test_the_review_endpoint_refuses_a_verdict_outside_that_set() -> None:
    """Naming the closed set is not enforcing it, and only one line does.

    Deleting `if request.verdict not in HUMAN_REVIEW_VERDICTS` was the one
    mutation in this file's matrix that the entire 3094-test suite did not
    notice: `ReviewRequest.verdict` is a free `str`, so with that line gone any
    sixteen-character string is stored on a surface as a human verdict. This is
    the detector for it -- the refusal happens before the control-plane lookup,
    so it holds with or without a database.
    """
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app
    from fastapi import HTTPException

    class _Http:
        state = type("S", (), {"username": "someone"})()

    for verdict in ("PROMOTED", "STANDARD_QC", "approved", "", "ADMIT"):
        request = panel_app.ReviewRequest.model_construct(
            verdict=verdict, note="", vetting_packet_sha256=None,
            p7_job_id=None)
        with pytest.raises(HTTPException) as refusal:
            panel_app.api_review("some-surface", request, _Http())
        assert refusal.value.status_code == 400, (
            f"{verdict!r} was accepted as a human verdict")


def test_reviewing_a_diagnostic_surface_says_that_it_is_diagnostic() -> None:
    sys.path.insert(0, str(ROOT / "panel"))
    import inspect

    import app as panel_app

    source = inspect.getsource(panel_app.api_review)
    assert "routing" in source or "SMALL_SURFACE_DIAGNOSTIC" in source, (
        "the review endpoint has no idea the surface it is recording a verdict "
        "on was routed out of the acceptance path")


class _AcceptingControlPlane:
    """A psycopg stand-in that accepts one payload update."""

    rowcount = 1

    def connect(self, *_args, **_kwargs):
        return self

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def execute(self, *_args):
        return self

    def commit(self):
        return None


def _reviewed(monkeypatch, store, surface_id: str) -> dict:
    """One APPROVED verdict, recorded through the real handler."""
    import json as json_module

    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    monkeypatch.setattr(panel_app, "DSN", "postgresql://fixture.invalid/fleet")
    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: store)
    monkeypatch.setitem(sys.modules, "psycopg", _AcceptingControlPlane())
    request = panel_app.ReviewRequest(verdict="APPROVED", note="looks fine")
    http = type("H", (), {"state": type("S", (), {"username": "someone"})()})()
    return json_module.loads(bytes(panel_app.api_review(
        surface_id, request, http).body))


def test_an_approved_verdict_on_a_tiny_surface_comes_back_saying_it_is_tiny(
    store, monkeypatch,
) -> None:
    """The sentence the reviewer was never shown, checked where it is written.

    APPROVED on 0.0198 cm2 and APPROVED on five square centimetres filed as the
    same row. The verdict still does not move the surface -- that is the
    database's job and it is tested above -- but the record now says which of
    the two this was.
    """
    _, tiny = _surface(store, name="reviewtiny", area=PHERC0268_AREA_CM2,
                       digest=PHERC0268_ARTIFACT_SHA256)
    before = store.routing_receipt(tiny)
    review = _reviewed(monkeypatch, store, tiny)

    assert review["verdict"] == "APPROVED"
    routing_said = review["surface_routing"]
    assert routing_said["route"] == routing.DIAGNOSTIC
    assert routing_said["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert routing_said["receipt_sha256"] == before["receipt_sha256"]
    assert "never available to it" in routing_said["advisory"]
    # Sanitized, because the receipt beside a verdict is still a public one.
    assert "read_set" not in routing_said and "measurement" not in routing_said
    assert PHERC0268_ARTIFACT_SHA256 not in json.dumps(review)
    # And reading is all it did.
    assert store.routing_receipt(tiny) == before


def test_the_same_verdict_on_a_normal_surface_does_not_cry_diagnostic(
    store, monkeypatch,
) -> None:
    """The other half: an advisory that fires on everything says nothing."""
    _, big = _surface(store, name="reviewbig", area=0.5, digest="d" * 64)
    routing_said = _reviewed(monkeypatch, store, big)["surface_routing"]
    assert routing_said["route"] == routing.STANDARD
    assert "never available to it" not in routing_said["advisory"]
    assert routing.DIAGNOSTIC not in routing_said["advisory"]


def test_a_verdict_on_an_unrouted_surface_says_nobody_classified_it(
    store, monkeypatch,
) -> None:
    """"No receipt" is reported as no receipt, never as the standard path."""
    routing_said = _reviewed(
        monkeypatch, store, "a-surface-nobody-routed")["surface_routing"]
    assert routing_said["route"] is None
    assert "no verified routing receipt is stored" in routing_said["advisory"]
