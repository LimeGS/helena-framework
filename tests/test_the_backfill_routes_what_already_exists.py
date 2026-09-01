"""Surfaces that existed before the routing did still have to be routed.

gpu-1 holds 316 surfaces and no routing receipts: 293 above the floor, 2 below
it, and 21 with no measured area at all. Every gate built for Task 8 fails
closed on a missing receipt, so deploying without this would stop 87 pending QC
jobs -- not corrupt them, stop them -- and refuse every flattening.

The backfill decides nothing new. It runs the same frozen router over surfaces
that already exist and writes the receipt each one earns. Two things it must not
do, and both are asserted here rather than described:

* it must not invent a measurement. A surface with no area cannot be routed, and
  guessing one is the failure this whole task exists to prevent, arriving
  through the repair rather than through the pipeline.
* it must not re-decide a surface that already has a receipt. The receipt is
  immutable and is the record of what the surface was when it first existed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing as routing  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _snapshot(store, scroll: str) -> str:
    return store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})


def _unrouted_surface(store, *, name: str, area, digest: str) -> str:
    """A surface as a control plane that predates the routing holds one.

    Inserted directly, because that is what the historical state *is*: a row
    written by a version of import_surface that had no receipt to write. The
    first attempt at this helper imported normally and then deleted the receipt,
    which the immutability trigger correctly refused -- a receipt cannot be
    removed, so a surface that never had one cannot be made by taking one away.
    """
    scroll = f"TEST{name}"
    snapshot = _snapshot(store, scroll)
    surface_id = f"{scroll}-s"
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,
                   artifact_sha256,artifact_uri,bbox_xyz_json,area_cm2,state,
                   physical_qc_state,payload_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
            (surface_id, snapshot, scroll, "imported", digest,
             f"s3://bucket/{name}", "[[0,0,0],[14,14,3]]", area,
             "QC_SCREENED", "UNVALIDATED", "{}"))
        connection.commit()
    assert store.routing_receipt(surface_id) is None
    return surface_id


def test_it_routes_a_surface_that_predates_routing(store) -> None:
    surface = _unrouted_surface(store, name="old", area=0.5, digest="a" * 64)
    assert store.routing_receipt(surface) is None

    summary = store.backfill_routing_receipts(apply=True)

    receipt = store.routing_receipt(surface)
    assert receipt is not None and routing.verify_receipt(receipt) is True
    assert receipt["route"] == routing.STANDARD
    assert summary["routed"] == 1


def test_a_tiny_surface_is_routed_diagnostic_by_the_backfill(store) -> None:
    """The two under the floor on gpu-1, one of which is PHerc0268."""
    surface = _unrouted_surface(store, name="tiny", area=PHERC0268_AREA_CM2,
                                digest=PHERC0268_ARTIFACT_SHA256)
    store.backfill_routing_receipts(apply=True)

    receipt = store.routing_receipt(surface)
    assert receipt["route"] == routing.DIAGNOSTIC
    assert receipt["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert receipt["is_absence_evidence"] is False
    assert routing.enters_standard_qc(receipt) is False


def test_a_surface_with_no_area_is_reported_and_not_invented(store) -> None:
    """The twenty-one. A repair that guesses is the failure it was repairing."""
    surface = _unrouted_surface(store, name="noarea", area=None, digest="b" * 64)

    summary = store.backfill_routing_receipts(apply=True)

    assert store.routing_receipt(surface) is None, (
        "the backfill invented a measurement for a surface that has none")
    assert summary["unroutable"] == 1
    assert surface in summary["unroutable_surface_ids"]
    assert summary["routed"] == 0


def test_a_dry_run_writes_nothing(store) -> None:
    surface = _unrouted_surface(store, name="dry", area=0.5, digest="c" * 64)

    summary = store.backfill_routing_receipts(apply=False)

    assert store.routing_receipt(surface) is None
    assert summary["would_route"] == 1
    assert summary["applied"] is False


def test_running_it_twice_changes_nothing(store) -> None:
    """Idempotent, and the receipt is immutable, so a second pass must skip."""
    surface = _unrouted_surface(store, name="twice", area=0.5, digest="d" * 64)
    store.backfill_routing_receipts(apply=True)
    first = store.routing_receipt(surface)

    second_summary = store.backfill_routing_receipts(apply=True)

    assert second_summary["routed"] == 0
    assert second_summary["already_routed"] == 1
    assert store.routing_receipt(surface) == first


def test_it_does_not_touch_a_surface_that_already_has_one(store) -> None:
    """A routed surface keeps the decision made when it first existed."""
    scroll = "TESTkeep"
    snapshot = _snapshot(store, scroll)
    surface = store.import_surface({
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": "e" * 64,
        "artifact_uri": "s3://bucket/keep",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": 0.5,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    before = store.routing_receipt(surface)
    assert before is not None

    store.backfill_routing_receipts(apply=True)

    assert store.routing_receipt(surface) == before


def test_the_summary_is_evidence_and_adds_up(store) -> None:
    _unrouted_surface(store, name="s1", area=0.5, digest="1" * 64)
    _unrouted_surface(store, name="s2", area=PHERC0268_AREA_CM2, digest="2" * 64)
    _unrouted_surface(store, name="s3", area=None, digest="3" * 64)

    summary = store.backfill_routing_receipts(apply=True)

    assert summary["schema"] == "campaignx.small_surface_routing_backfill.v1"
    assert summary["policy_version"] == routing.load_policy()["policy_version"]
    assert summary["profile_id"] == routing.PROFILE_ID
    assert summary["considered"] == 3
    assert summary["routed"] == 2
    assert summary["unroutable"] == 1
    assert summary["by_route"] == {routing.STANDARD: 1, routing.DIAGNOSTIC: 1}
    # Every surface is accounted for exactly once, or the summary is a story.
    assert (summary["routed"] + summary["unroutable"]
            + summary["already_routed"]) == summary["considered"]
