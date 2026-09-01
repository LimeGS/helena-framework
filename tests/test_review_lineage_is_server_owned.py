"""A surface may be named by a client; what it descends from may not be.

On 2026-08-02 a 0.0198 cm2 surface of PHerc0268 was GEOMETRY_CERTIFIED and
reached the ink screen. Every gate it walked past shared one shape: something
outside the server said which surface this was and what it was downstream of,
and the server agreed. These tests hold the opposite line. The client names a
job; the server walks the persisted rows and decides which surface that job is
about, refusing a chain that is missing, forked, or contradicted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import review_lineage  # noqa: E402
from fleet import surface_routing  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.store import FleetStore  # noqa: E402


class Jobs:
    """The persisted job rows, and nothing a caller may add to them."""

    def __init__(self, rows: dict[str, dict]):
        self.rows = rows

    def job(self, job_id):
        return self.rows.get(job_id)


def chain(**overrides) -> dict[str, dict]:
    """One honest P7 -> P5 -> P4 -> P3 chain over `surface-real`."""
    rows = {
        "p3": {
            "job_id": "p3", "phase": "P3", "state": "succeeded",
            "mission_id": "m", "sample_id": "PHerc0139",
            "parameters": {"surface_id": "surface-real"},
            "result": {"surfaces": [{
                "surface_id": "surface-real", "requested_by_job_id": "p3",
                "artifact_id": "flat-real", "artifact_sha256": "4" * 64,
                "source_artifact_sha256": "3" * 64,
            }]},
        },
        "p4": {
            "job_id": "p4", "phase": "P4", "state": "succeeded",
            "mission_id": "m", "sample_id": "PHerc0139",
            "parameters": {"flattened_surface": "surface-real", "p3_job_id": "p3",
                           "flattening_id": "flat-real",
                           "flattened_artifact_sha256": "4" * 64},
            "result": {"layer_stack": {"artifact_sha256": "a" * 64}},
        },
        "p5": {
            "job_id": "p5", "phase": "P5", "state": "succeeded",
            "mission_id": "m", "sample_id": "PHerc0139",
            "parameters": {"layer_stack": "p4"},
            "result": {"physical_normalization": {
                "p4_job_id": "p4", "p4_layer_artifact_sha256": "a" * 64}},
        },
        "p7": {
            "job_id": "p7", "phase": "P7", "state": "succeeded",
            "mission_id": "m", "sample_id": "PHerc0139",
            "parameters": {"surface_id": "surface-real", "screening_of": "p5"},
            "result": {},
        },
    }
    for key, value in overrides.items():
        rows[key] = value
    return rows


def resolve(rows, **kwargs):
    return review_lineage.resolve_surface_origin(Jobs(rows), **kwargs)


def rejection(rows, **kwargs) -> str:
    with pytest.raises(review_lineage.SurfaceOriginRejected) as error:
        resolve(rows, **kwargs)
    return error.value.reason_code


# --- C6: the origin is walked, never handed in -----------------------------

def test_p3_p5_and_p7_origins_are_walked_from_the_persisted_rows() -> None:
    rows = chain()
    for phase, job_id in (("P3", "p3"), ("P4", "p4"), ("P5", "p5"), ("P7", "p7")):
        origin = resolve(rows, phase=phase, job_id=job_id,
                         mission_id="m", sample_id="PHerc0139")
        assert origin["surface_id"] == "surface-real"
        assert origin["p3_job_id"] == "p3"
    p7 = resolve(rows, phase="P7", job_id="p7",
                 mission_id="m", sample_id="PHerc0139")
    assert p7["p5_job_id"] == "p5"
    assert p7["p4_job_id"] == "p4"
    assert p7["flattened_artifact_id"] == "flat-real"
    assert p7["flattened_artifact_sha256"] == "4" * 64
    assert p7["p4_layer_artifact_sha256"] == "a" * 64
    assert p7["chain_sha256"] == content_sha256(
        {key: value for key, value in p7.items() if key != "chain_sha256"})


def test_a_client_asserted_surface_is_checked_against_the_walk_never_used() -> None:
    rows = chain()
    origin = resolve(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139", asserted_surface_id="surface-real")
    assert origin["surface_id"] == "surface-real"
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139",
                     asserted_surface_id="surface-tiny") == \
        "CLIENT_SURFACE_ASSERTION_REJECTED"
    # And the decoy is not reachable by naming it, even when the job's own
    # result carries a row for it.
    rows["p3"]["result"]["surfaces"].append({
        "surface_id": "surface-tiny", "requested_by_job_id": "p3",
        "artifact_id": "flat-tiny", "artifact_sha256": "7" * 64,
        "source_artifact_sha256": "8" * 64,
    })
    assert resolve(rows, phase="P3", job_id="p3", mission_id="m",
                   sample_id="PHerc0139")["flattened_artifact_id"] == "flat-real"
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139",
                     asserted_surface_id="surface-tiny") == \
        "CLIENT_SURFACE_ASSERTION_REJECTED"


def test_a_p3_with_no_resolvable_surface_fails_closed() -> None:
    rows = chain()
    rows["p3"]["parameters"] = {}
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139") == "P3_SURFACE_UNRESOLVABLE"
    rows["p3"]["parameters"] = {"surface_id": "   "}
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139") == "P3_SURFACE_UNRESOLVABLE"
    # It also does not fall back to the single row the result happens to hold.
    rows["p3"]["parameters"] = {}
    rows["p3"]["result"] = {"surfaces": [{
        "surface_id": "surface-real", "requested_by_job_id": "p3",
        "artifact_id": "flat-real", "artifact_sha256": "4" * 64}]}
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139") == "P3_SURFACE_UNRESOLVABLE"


def test_a_conflicting_chain_is_refused_rather_than_silently_preferred() -> None:
    # A second P3 exists for a different surface. P4 names the honest P3 but
    # flattens the other surface: two stories, and neither may be picked.
    rows = chain()
    rows["p4"]["parameters"]["flattened_surface"] = "surface-tiny"
    assert rejection(rows, phase="P4", job_id="p4", mission_id="m",
                     sample_id="PHerc0139") == "LINEAGE_SURFACE_CONFLICT"
    assert rejection(rows, phase="P7", job_id="p7", mission_id="m",
                     sample_id="PHerc0139") == "LINEAGE_SURFACE_CONFLICT"

    # P7 claims a surface its own P5 chain never produced.
    rows = chain()
    rows["p7"]["parameters"]["surface_id"] = "surface-tiny"
    assert rejection(rows, phase="P7", job_id="p7", mission_id="m",
                     sample_id="PHerc0139") == "LINEAGE_SURFACE_CONFLICT"

    # P5 normalization names a different P4 than the one it consumed.
    rows = chain()
    rows["p5"]["result"]["physical_normalization"]["p4_job_id"] = "p4-decoy"
    assert rejection(rows, phase="P5", job_id="p5", mission_id="m",
                     sample_id="PHerc0139") == "LINEAGE_EDGE_AMBIGUOUS"


def test_a_decoy_flattening_row_is_refused_rather_than_preferred() -> None:
    rows = chain()
    rows["p3"]["result"]["surfaces"].append({
        "surface_id": "surface-real", "requested_by_job_id": "p3",
        "artifact_id": "flat-decoy", "artifact_sha256": "9" * 64,
        "source_artifact_sha256": "3" * 64,
    })
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139") == "P3_FLATTENED_LINEAGE_AMBIGUOUS"
    rows = chain()
    rows["p3"]["result"] = {"surfaces": []}
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139") == "P3_FLATTENED_LINEAGE_AMBIGUOUS"
    # A row another job produced is not this job's evidence.
    rows = chain()
    rows["p3"]["result"]["surfaces"][0]["requested_by_job_id"] = "p3-other"
    assert rejection(rows, phase="P3", job_id="p3", mission_id="m",
                     sample_id="PHerc0139") == "P3_FLATTENED_LINEAGE_AMBIGUOUS"


def test_every_hop_must_exist_succeed_and_stay_in_scope() -> None:
    rows = chain()
    assert rejection(rows, phase="P7", job_id="absent", mission_id="m",
                     sample_id="PHerc0139") == "JOB_NOT_FOUND"
    assert rejection(rows, phase="P5", job_id="p7", mission_id="m",
                     sample_id="PHerc0139") == "JOB_PHASE_MISMATCH"
    for broken, reason in (
        ({"state": "running"}, "JOB_NOT_SUCCEEDED"),
        ({"mission_id": "other"}, "JOB_SCOPE_MISMATCH"),
        ({"sample_id": "PHerc0268"}, "JOB_SCOPE_MISMATCH"),
    ):
        rows = chain()
        rows["p3"].update(broken)
        assert rejection(rows, phase="P7", job_id="p7", mission_id="m",
                         sample_id="PHerc0139") == reason
    for dropped in ("screening_of",):
        rows = chain()
        rows["p7"]["parameters"].pop(dropped)
        assert rejection(rows, phase="P7", job_id="p7", mission_id="m",
                         sample_id="PHerc0139") == "LINEAGE_EDGE_MISSING"
    rows = chain()
    rows["p4"]["parameters"].pop("p3_job_id")
    assert rejection(rows, phase="P4", job_id="p4", mission_id="m",
                     sample_id="PHerc0139") == "LINEAGE_EDGE_MISSING"
    rows = chain()
    rows["p5"]["parameters"].pop("layer_stack")
    assert rejection(rows, phase="P5", job_id="p5", mission_id="m",
                     sample_id="PHerc0139") == "LINEAGE_EDGE_MISSING"


# --- C5: the review intent, the adjudication, and the route ----------------

def routing_receipt(area_cm2: float, surface_id: str = "surface-real") -> dict:
    return surface_routing.build_receipt(
        surface_id=surface_id, area_cm2=area_cm2,
        policy=surface_routing.load_policy(),
        measurement={}, read_set={})


def adjudication() -> dict:
    return {"verdict": "PASS", "overall": {"pass": True},
            "verdict_sha256": "b" * 64, "card_sha256": "c" * 64,
            "config_hash": "d" * 64}


def locked_event(rows=None, *, area_cm2: float = 0.5, intent: str = "INSPECT",
                 receipt: dict | None = None, **overrides) -> dict:
    origin = resolve(rows or chain(), phase="P7", job_id="p7",
                     mission_id="m", sample_id="PHerc0139")
    event = review_lineage.build_review_event(
        origin=origin, intent=intent, note=None,
        routing_receipt=receipt or routing_receipt(area_cm2,
                                                   origin["surface_id"]),
        adjudication=adjudication(), vetting_packet_sha256="e" * 64,
        author="tester", at="2026-08-04T00:00:00+00:00",
        review_event_id="review-one", extra={})
    event.update(overrides)
    if "event_sha256" not in overrides:
        event["event_sha256"] = content_sha256(
            {key: value for key, value in event.items() if key != "event_sha256"})
    return event


def test_the_review_intent_is_a_server_owned_enum() -> None:
    assert review_lineage.REVIEW_INTENTS == ("INSPECT",)
    for refused in ("APPROVED", "inspect", "", None, "DEFECTIVE"):
        with pytest.raises(review_lineage.SurfaceOriginRejected) as error:
            review_lineage.require_review_intent(refused)
        assert error.value.reason_code == "REVIEW_INTENT_NOT_ALLOWED"
    assert review_lineage.require_review_intent("INSPECT") == "INSPECT"


def test_only_a_passing_adjudication_may_be_routed_for_inspection() -> None:
    for broken in (
        {"verdict": "FAIL"},
        {"overall": {"pass": False}},
        {"verdict_sha256": "not-a-hash"},
        {"card_sha256": None},
        {"config_hash": ""},
    ):
        with pytest.raises(review_lineage.SurfaceOriginRejected) as error:
            review_lineage.require_passing_adjudication({**adjudication(), **broken})
        assert error.value.reason_code == "ADJUDICATION_NOT_PASSING"
    assert review_lineage.require_passing_adjudication(adjudication())["card_sha256"] \
        == "c" * 64


def test_an_exact_standard_route_is_required_for_review() -> None:
    standard = routing_receipt(0.5)
    assert review_lineage.require_standard_route(standard)["route"] == \
        surface_routing.STANDARD
    for refused in (None, {}, routing_receipt(0.0198222),
                    {**standard, "route": surface_routing.STANDARD,
                     "receipt_sha256": "0" * 64}):
        with pytest.raises(review_lineage.SurfaceOriginRejected) as error:
            review_lineage.require_standard_route(refused)
        assert error.value.reason_code == "SURFACE_ROUTE_NOT_STANDARD"


# --- C5: the store refuses anything the resolver did not produce -----------

@pytest.fixture
def store(tmp_path):
    store = FleetStore(tmp_path / "reviews.sqlite")
    store.initialize()
    return store


def certified(store, surface_id: str, area_cm2: float) -> None:
    store.register_snapshot({
        "schema": "campaignx.p0_frozen_source.v1", "sample_id": "PHerc0139",
        "source_snapshot_id": "snap", "ct_uri": "https://ct.invalid/ct",
        "m7_uri": "https://m7.invalid/m7", "shape_xyz": [10, 10, 10],
        "voxel_size_um": 7.91})
    store.import_surface({
        "surface_id": surface_id, "source_snapshot_id": "snap",
        "sample_id": "PHerc0139", "bbox_xyz": [0, 0, 0, 1, 1, 1],
        "area_cm2": area_cm2,
        "artifact_sha256": content_sha256(surface_id)})


def test_a_direct_insert_that_bypassed_the_resolver_fails_closed(store) -> None:
    certified(store, "surface-real", 0.5)
    hand_built = {
        "schema": "campaignx.human_review_event.v1",
        "review_event_id": "review-forged", "p7_job_id": "p7",
        "intent": "INSPECT", "mission_id": "m", "sample_id": "PHerc0139",
        "surface_id": "surface-real", "verdict_sha256": "b" * 64,
        "card_sha256": "c" * 64, "config_sha256": "d" * 64,
        "vetting_packet_sha256": "e" * 64, "by": "tester",
        "at": "2026-08-04T00:00:00+00:00",
    }
    hand_built["event_sha256"] = content_sha256(hand_built)
    with pytest.raises(ValueError, match="review lineage"):
        store.insert_human_review(hand_built)
    assert store.human_reviews("p7") == []


def test_a_resolver_built_review_is_accepted_and_stays_idempotent(store) -> None:
    certified(store, "surface-real", 0.5)
    event = locked_event(receipt=store.routing_receipt("surface-real"))
    assert store.insert_human_review(event) == event
    assert store.insert_human_review(event) == event
    assert store.human_reviews("p7") == [event]


def test_the_store_refuses_a_review_of_a_small_surface_diagnostic(store) -> None:
    certified(store, "surface-real", 0.01983222455087575)
    event = locked_event(receipt=store.routing_receipt("surface-real"))
    with pytest.raises(ValueError, match="route"):
        store.insert_human_review(event)
    assert store.human_reviews("p7") == []


def test_the_store_refuses_a_review_of_an_unrouted_surface(store) -> None:
    event = locked_event()
    with pytest.raises(ValueError, match="route"):
        store.insert_human_review(event)
    assert store.human_reviews("p7") == []


def test_the_store_refuses_a_lock_that_does_not_cover_the_event(store) -> None:
    certified(store, "surface-real", 0.5)
    certified(store, "surface-other", 0.5)
    stored = store.routing_receipt("surface-real")
    for tampered in (
        {"surface_id": "surface-other"},
        {"p3_job_id": "p3-decoy"},
        {"verdict_sha256": "f" * 64},
        {"review_lineage_sha256": "0" * 64},
    ):
        event = locked_event(receipt=stored, **tampered)
        with pytest.raises(ValueError, match="review lineage"):
            store.insert_human_review(event)
    assert store.human_reviews("p7") == []


def test_the_store_refuses_an_intent_outside_the_frozen_enum(store) -> None:
    certified(store, "surface-real", 0.5)
    stored = store.routing_receipt("surface-real")
    for refused in ("APPROVED", "DEFECTIVE", "inspect"):
        with pytest.raises(ValueError, match="intent"):
            store.insert_human_review(locked_event(intent=refused, receipt=stored))
    assert store.human_reviews("p7") == []


def test_the_postgres_store_applies_the_same_three_refusals() -> None:
    import inspect

    from fleet.postgres_store import PostgresFleetStore

    source = inspect.getsource(PostgresFleetStore.insert_human_review)
    assert "require_review_intent" in source
    assert "verify_review_lineage_lock" in source
    assert "require_standard_route" in source
    assert "routing_receipt" in source
