"""The routing happens where the surface is created, or it happens too late.

This is the PHerc0268 case from the 2026-08-02 bounded campaign, by its frozen
identity rather than by a plausible number: a 0.01983222455087575 cm2 surface --
about two square millimetres, 84 finite coordinates, 132 triangles -- was
GEOMETRY_CERTIFIED, entered physical QC, and came back INK_SCREEN_INSUFFICIENT
with liveness EMPTY.

The dossier records that carefully as an empty output for a screening path and
not a finding about the scroll. The wording is right and the routing was not:
the platform sent it there, and an EMPTY over two square millimetres files next
to an EMPTY over five square centimetres with nothing distinguishing them.

So the check is at finalization, inside the transaction that creates the
surface. A guard in a downstream worker would be a second opinion arriving after
the row that lets the work start already exists.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing as routing  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

# Frozen in docs/first-letters/first-letters-hybrid-20260802/evidence.json.
PHERC0268 = {
    "area_cm2": 0.01983222455087575,
    "artifact_sha256": "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6",
    "grid_shape_y_x": [14, 14],
    "valid_triangle_count": 132,
    "finite_coordinate_count": 84,
}


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _surface(store, *, name: str, area: float, digest: str) -> tuple[str, str]:
    scroll = f"TEST{name}"
    snapshot = store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    surface = store.import_surface({
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": digest,
        "artifact_uri": f"s3://bucket/{name}",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": area,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    return snapshot, surface


# -- the receipt is written where the surface is ----------------------------

def test_a_routing_receipt_is_written_for_every_surface(store) -> None:
    _, surface = _surface(store, name="normal", area=0.5, digest="a" * 64)
    receipt = store.routing_receipt(surface)
    assert receipt is not None, "a surface exists with no routing decision recorded"
    assert routing.verify_receipt(receipt) is True
    assert receipt["route"] == routing.STANDARD


def test_the_pherc0268_surface_is_routed_diagnostic(store) -> None:
    _, surface = _surface(store, name="0268", area=PHERC0268["area_cm2"],
                          digest=PHERC0268["artifact_sha256"])
    receipt = store.routing_receipt(surface)
    assert receipt["route"] == routing.DIAGNOSTIC
    assert receipt["measured_area_cm2"] == PHERC0268["area_cm2"]
    assert receipt["read_set"]["artifact_sha256"] == PHERC0268["artifact_sha256"]
    assert receipt["is_absence_evidence"] is False


def _row(store, surface_id: str) -> dict:
    with store.connect() as connection:
        row = connection.execute(
            "SELECT * FROM surfaces WHERE surface_id=?", (surface_id,)).fetchone()
    return dict(row) if row is not None else None


def test_the_receipt_cannot_be_rewritten(store) -> None:
    """Immutable: the decision is evidence, and evidence that can be edited is not.

    Asserted against the database, not against a missing method -- an earlier
    version of this test called a helper that did not exist yet and passed on
    the AttributeError, which is a green test that checks nothing.
    """
    _, surface = _surface(store, name="immutable", area=0.5, digest="b" * 64)
    import sqlite3

    with store.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE surface_routing_receipts SET route=? WHERE surface_id=?",
                (routing.DIAGNOSTIC, surface))
        with pytest.raises(sqlite3.IntegrityError, match="permanent"):
            connection.execute(
                "DELETE FROM surface_routing_receipts WHERE surface_id=?", (surface,))

    assert store.routing_receipt(surface)["route"] == routing.STANDARD


# -- and it decides what physical QC may touch ------------------------------

def test_a_diagnostic_surface_gets_no_claimable_qc_job(store) -> None:
    """The failure this exists to stop: a two-square-millimetre ink screen."""
    _, surface = _surface(store, name="tiny", area=PHERC0268["area_cm2"],
                          digest=PHERC0268["artifact_sha256"])
    store.enqueue_imported_surface_qc({
        "surface_id": surface, "source_snapshot_id": _row(store, surface)["source_snapshot_id"],
        "sample_id": _row(store, surface)["sample_id"],
        "artifact_sha256": PHERC0268["artifact_sha256"],
        "artifact_uri": "s3://bucket/tiny", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": PHERC0268["area_cm2"],
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
    }, profile_id="surface-qc@1.0.0")

    assert store.claim_qc("qc-worker", 60, profile_id="surface-qc@1.0.0") is None, (
        "a surface below the floor reached the physical-QC queue, which is "
        "exactly what PHerc0268 did on 2026-08-02"
    )


def test_a_normal_surface_still_reaches_physical_qc(store) -> None:
    """The routing must not quarantine the work it was not written for."""
    _, surface = _surface(store, name="ok", area=0.5, digest="c" * 64)
    store.record_geometry_certification(
        surface, "GEOMETRY_CERTIFIED", {"schema": "test"},
        requested_by_job_id="p2-ok", profile_id="geometry-test@1",
        profile_sha256="6" * 64)
    store.enqueue_imported_surface_qc({
        "surface_id": surface,
        "source_snapshot_id": _row(store, surface)["source_snapshot_id"],
        "sample_id": _row(store, surface)["sample_id"],
        "artifact_sha256": hashlib.sha256(b"ok").hexdigest(),
        "artifact_uri": "s3://bucket/ok", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": 0.5,
    }, profile_id="surface-qc-ok@1.0.0")

    claimed = store.claim_qc("qc-worker", 60, profile_id="surface-qc-ok@1.0.0")
    assert claimed is not None and claimed["surface_id"] == surface


def test_the_diagnostic_surface_is_kept_not_dropped(store) -> None:
    _, surface = _surface(store, name="kept", area=PHERC0268["area_cm2"],
                          digest=PHERC0268["artifact_sha256"])
    stored = _row(store, surface)
    assert stored is not None
    assert stored["artifact_sha256"] == PHERC0268["artifact_sha256"]
    assert store.routing_receipt(surface)["preserved"] is True
