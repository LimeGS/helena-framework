"""Replaying the routing decision, reading it back, and dying halfway through.

Three questions, and they are not the same question.

*Replay* -- the same surface arriving twice. The decision is made once, when the
surface first exists, so a second arrival must find the first decision rather
than make a new one. A recovery bootstrap re-imports every surface it recovers,
so this is a path that runs in anger, not a hypothetical.

*Conflicting replay* -- the same surface arriving twice with different bytes.
The stored surface does not change, so the receipt must not either; and a caller
whose payload was discarded should not be told it was accepted.

*Failpoint* -- the process dying between the surface and its decision. A surface
with no routing decision is the PHerc0268 state exactly: nothing says it is too
small, so nothing stops it. That makes "no receipt" the state the queue gate has
to be most careful about, and it was the state it was least careful about.
C4 closed it: import_surface is transactional, the queue gate requires a
decision instead of consulting one, and finalize routes every surface it
creates. These now run green and stay as the regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
for extra in (STAGE, ROOT, ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fleet import surface_routing as routing  # noqa: E402
from fleet.store import QC_SMALL_SURFACE_DIAGNOSTIC, FleetStore  # noqa: E402

PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")

@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _snapshot(store, sample: str = "PHerc268") -> str:
    return store.register_snapshot({
        "sample_id": sample, "ct_uri": f"https://example.invalid/{sample}/ct.zarr",
        "m7_uri": f"https://example.invalid/{sample}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})


def _payload(snapshot: str, **overrides) -> dict:
    return {
        "surface_id": "replay-1", "source_snapshot_id": snapshot,
        "sample_id": "PHerc268", "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/replay",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": PHERC0268_AREA_CM2,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED", **overrides}


def _receipt_rows(store) -> list[dict]:
    with store.connect() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM surface_routing_receipts ORDER BY surface_id")]


def _qc_state(store, surface_id: str) -> str | None:
    with store.connect() as connection:
        row = connection.execute(
            "SELECT state FROM qc_jobs WHERE surface_id=?", (surface_id,)
        ).fetchone()
    return row["state"] if row is not None else None


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def test_replaying_the_same_import_leaves_one_decision(store) -> None:
    """The path a recovery bootstrap takes, four times over."""
    snapshot = _snapshot(store)
    identities = {store.import_surface(_payload(snapshot)) for _ in range(4)}
    assert identities == {"replay-1"}
    rows = _receipt_rows(store)
    assert len(rows) == 1
    assert rows[0]["route"] == routing.DIAGNOSTIC


def test_a_replay_reads_back_byte_identical(store) -> None:
    """Idempotent means the same bytes, not merely the same verdict."""
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    first = store.routing_receipt("replay-1")
    store.import_surface(_payload(snapshot))
    second = store.routing_receipt("replay-1")
    assert first == second
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert routing.verify_receipt(second) is True


def test_a_replay_does_not_disturb_the_created_at_of_the_first(store) -> None:
    """The decision has one time, and it is the time it was made."""
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    created = _receipt_rows(store)[0]["created_at"]
    store.import_surface(_payload(snapshot))
    assert _receipt_rows(store)[0]["created_at"] == created


def test_the_qc_enqueue_replays_without_a_second_job(store) -> None:
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    enqueue = {
        "surface_id": "replay-1", "source_snapshot_id": snapshot,
        "sample_id": "PHerc268", "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/replay", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": PHERC0268_AREA_CM2, "geometry_qc_state": "GEOMETRY_CERTIFIED"}
    first = store.enqueue_imported_surface_qc(
        enqueue, profile_id="surface-qc-replay@1.0.0")
    second = store.enqueue_imported_surface_qc(
        enqueue, profile_id="surface-qc-replay@1.0.0")
    assert first["status"] == "ENQUEUED"
    assert second["status"] == "ALREADY_ENQUEUED"
    assert first["qc_job_id"] == second["qc_job_id"]
    assert second["qc_state"] == QC_SMALL_SURFACE_DIAGNOSTIC
    assert store.claim_qc("w", 60, profile_id="surface-qc-replay@1.0.0") is None


# ---------------------------------------------------------------------------
# Conflicting replay
# ---------------------------------------------------------------------------

def test_a_conflicting_replay_does_not_change_the_stored_decision(store) -> None:
    """The important half, and it holds: the first decision stands.

    A replay claiming a fiftyfold larger area does not promote the surface,
    because the surface did not change either.
    """
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    before = store.routing_receipt("replay-1")
    # The replay is now refused rather than discarded, and the property this
    # test exists for is unchanged by that: whichever way the call ends, the
    # stored decision is the first one. Asserted after the refusal so the
    # refusal cannot be what leaves it looking right.
    with pytest.raises((RuntimeError, ValueError), match="conflict|differs|refus"):
        store.import_surface(_payload(snapshot, area_cm2=1.0,
                                      artifact_sha256="9" * 64))
    assert store.routing_receipt("replay-1") == before
    assert before["route"] == routing.DIAGNOSTIC
    assert before["measured_area_cm2"] == PHERC0268_AREA_CM2


def test_the_surface_and_its_receipt_never_disagree(store) -> None:
    """The invariant a conflicting replay could break, checked directly."""
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    with pytest.raises((RuntimeError, ValueError), match="conflict|differs|refus"):
        store.import_surface(_payload(snapshot, area_cm2=1.0))
    with store.connect() as connection:
        row = connection.execute(
            "SELECT area_cm2,artifact_sha256 FROM surfaces WHERE surface_id=?",
            ("replay-1",)).fetchone()
    receipt = store.routing_receipt("replay-1")
    assert receipt["measured_area_cm2"] == row["area_cm2"]
    assert receipt["read_set"]["artifact_sha256"] == row["artifact_sha256"]


def test_a_conflicting_replay_tells_the_caller_it_was_refused(store) -> None:
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    with pytest.raises((RuntimeError, ValueError), match="conflict|differs|refus"):
        store.import_surface(_payload(snapshot, area_cm2=1.0,
                                      artifact_sha256="9" * 64))


# ---------------------------------------------------------------------------
# Failpoints
# ---------------------------------------------------------------------------

def test_a_surface_that_reached_the_queue_gate_carries_its_decision(
    store,
) -> None:
    """The property every failpoint below is about, stated once."""
    snapshot = _snapshot(store)
    store.import_surface(_payload(snapshot))
    assert store.routing_receipt("replay-1") is not None
    assert _receipt_rows(store)[0]["surface_id"] == "replay-1"


def test_a_failed_routing_decision_takes_the_surface_with_it(
    store, monkeypatch,
) -> None:
    snapshot = _snapshot(store)

    def explode(**_kwargs):
        raise RuntimeError("the router died after the surface landed")

    monkeypatch.setattr(routing, "build_receipt", explode)
    with pytest.raises(RuntimeError):
        store.import_surface(_payload(snapshot))
    with store.connect() as connection:
        surfaces = connection.execute(
            "SELECT surface_id FROM surfaces").fetchall()
    assert surfaces == [], (
        "a surface is committed with no routing decision; nothing about it says "
        "it is too small, so nothing stops it")


def test_a_surface_with_no_decision_cannot_be_claimed(store) -> None:
    snapshot = _snapshot(store)
    surface = store.import_surface(_payload(snapshot, area_cm2=None))
    assert store.routing_receipt(surface) is None

    store.enqueue_imported_surface_qc({
        "surface_id": surface, "source_snapshot_id": snapshot,
        "sample_id": "PHerc268", "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/replay", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": PHERC0268_AREA_CM2, "geometry_qc_state": "GEOMETRY_CERTIFIED",
    }, profile_id="surface-qc-absent@1.0.0")
    assert store.claim_qc("w", 60, profile_id="surface-qc-absent@1.0.0") is None, (
        "a 0.0198 cm2 surface was handed to a QC worker because no routing "
        "receipt existed to say it was too small")


def test_a_grown_surface_is_routed_when_it_is_finalized(tmp_path) -> None:
    from test_geometry_certification import (
        _prepare_finalization,
        _snapshot as certification_snapshot,
        _surface as certification_surface,
        _task,
    )

    store = FleetStore(tmp_path / "grown.sqlite")
    store.initialize()
    source_id = certification_snapshot(store)
    task = _task(store, source_id, "cell-0001")
    surface = certification_surface(source_id, "tiny", "GEOMETRY_CERTIFIED")
    surface["area_cm2"] = PHERC0268_AREA_CM2
    bound, artifact_set_id = _prepare_finalization(store, task, surface)
    result = store.finalize(
        task["task_id"], task["attempt_id"], task["lease_token"], bound,
        artifact_set_id, "geometry-screen-v1@1")
    assert result["status"] == "QC_PENDING"

    receipt = store.routing_receipt(bound["surface_id"])
    assert receipt is not None, (
        "a grown surface of 0.0198 cm2 was finalized with no routing decision")
    assert receipt["route"] == routing.DIAGNOSTIC
    assert _qc_state(store, bound["surface_id"]) == QC_SMALL_SURFACE_DIAGNOSTIC


def test_a_grown_surface_above_the_floor_is_unaffected_by_any_of_this(
    tmp_path,
) -> None:
    """The regression guard for whoever closes the three xfails above.

    Routing the finalizer must not change what a normal surface does, which is
    the failure mode of every gate added late.
    """
    from test_geometry_certification import (
        _prepare_finalization,
        _snapshot as certification_snapshot,
        _surface as certification_surface,
        _task,
    )

    store = FleetStore(tmp_path / "grown-ok.sqlite")
    store.initialize()
    source_id = certification_snapshot(store)
    task = _task(store, source_id, "cell-0002")
    surface = certification_surface(source_id, "normal", "GEOMETRY_CERTIFIED")
    surface["area_cm2"] = 1.0
    bound, artifact_set_id = _prepare_finalization(store, task, surface)
    result = store.finalize(
        task["task_id"], task["attempt_id"], task["lease_token"], bound,
        artifact_set_id, "geometry-screen-v1@1")
    assert result["status"] == "QC_PENDING"
    assert _qc_state(store, bound["surface_id"]) == "PENDING"


def test_the_readback_of_an_unknown_surface_is_absence_not_a_guess(store) -> None:
    """`None` means "no decision recorded", and nothing else may be inferred."""
    assert store.routing_receipt("no-such-surface") is None
    assert routing.enters_standard_qc(store.routing_receipt("no-such-surface")) is False
    assert routing.enters_canonical_downstream(
        store.routing_receipt("no-such-surface")) is False
