"""A claim is a fact about the past. The worker has to ask again.

The queue decided this surface was standard physical-QC work at the moment it
was enqueued. Between that moment and the moment a card is pointed at the
papyrus there is a wave of P1 finalizations, a resume, a regrow, a re-measure --
and a worker that acts on the enqueue-time decision is acting on a decision
somebody may have superseded hours ago.

That is how PHerc0268 got measured: 0.01983222455087575 cm2, 14x14, 132
triangles, geometry-certified, routed nowhere in particular, screened for ink,
and filed as EMPTY next to an EMPTY over five square centimetres.

So immediately before any I/O the worker re-reads the two facts that decide
whether this is work at all -- the routing receipt and the canonical lineage --
and refuses without touching the surface if either has moved. Refusal is
terminal and is not a scientific verdict: the surface is left exactly as it was,
because being the wrong size to ask is not an answer about what is written.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing as routing  # noqa: E402
from fleet.qc_worker import FixtureQcExecutor, SurfaceQcWorker  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

PROFILE = "fixture-surface-qc@1.0.0"

# tests/fixtures/first-letters-hybrid-20260802/evidence.json
PHERC0268_AREA = 0.01983222455087575
PHERC0268_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6"
)


class CountingExecutor(FixtureQcExecutor):
    """A fixture executor that remembers whether anybody pointed it at anything."""

    def __init__(self) -> None:
        super().__init__()
        self.executions = 0

    def execute(self, claim, attempt_dir):
        self.executions += 1
        return super().execute(claim, attempt_dir)


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _surface(store: FleetStore, *, area: float | None, digest: str,
             sample: str = "PHerc0268") -> str:
    snapshot = store.register_snapshot({
        "sample_id": sample,
        "ct_uri": f"https://example.invalid/{sample}/ct.zarr",
        "m7_uri": f"https://example.invalid/{sample}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    return store.import_surface({
        "surface_id": f"{sample}-{digest[:8]}", "source_snapshot_id": snapshot,
        "sample_id": sample, "artifact_sha256": digest,
        "artifact_uri": f"s3://bucket/{digest[:8]}",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        **({} if area is None else {"area_cm2": area}),
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED",
    })


def _pending_qc_job(store: FleetStore, surface_id: str) -> str:
    """Put a claimable QC job on this surface, whatever routing thinks.

    Written directly rather than through an enqueue path on purpose: the point
    under test is that the worker does not inherit the queue's opinion. Every
    way a PENDING row can come to exist over a surface routed elsewhere -- an
    enqueue that predates routing, a re-measure after the fact, an operator
    replaying a job -- arrives at the worker looking exactly like this one.
    """
    from fleet.common import utc_now

    qc_job_id = f"qc-{surface_id}"
    now = utc_now()
    with store.connect() as connection:
        connection.execute(
            """INSERT INTO qc_jobs(qc_job_id,surface_id,profile_id,state,
               payload_json,created_at,updated_at)
               VALUES(?,?,?,'PENDING','{}',?,?)""",
            (qc_job_id, surface_id, PROFILE, now, now),
        )
        connection.commit()
    return qc_job_id


def _worker(store: FleetStore, tmp_path: Path,
            executor: CountingExecutor) -> SurfaceQcWorker:
    return SurfaceQcWorker(
        store, "gpu-0", executor, tmp_path / "runs", profile_id=PROFILE,
    )


def _job_state(store: FleetStore, qc_job_id: str) -> str:
    with store.connect() as connection:
        return connection.execute(
            "SELECT state FROM qc_jobs WHERE qc_job_id=?", (qc_job_id,),
        ).fetchone()["state"]


def _surface_row(store: FleetStore, surface_id: str) -> dict:
    with store.connect() as connection:
        return dict(connection.execute(
            "SELECT * FROM surfaces WHERE surface_id=?", (surface_id,),
        ).fetchone())


# -- the surface that must never be measured --------------------------------

def test_a_diagnostic_surface_is_refused_before_any_io(store, tmp_path):
    surface_id = _surface(store, area=PHERC0268_AREA, digest=PHERC0268_SHA256)
    assert store.routing_receipt(surface_id)["route"] == routing.DIAGNOSTIC
    qc_job_id = _pending_qc_job(store, surface_id)
    before = _surface_row(store, surface_id)

    executor = CountingExecutor()
    runs = tmp_path / "runs"
    receipt = _worker(store, tmp_path, executor).run_one()

    assert executor.executions == 0, "the ink screen ran on a 0.02 cm2 surface"
    assert not runs.exists(), "the worker wrote an attempt directory anyway"
    assert receipt["status"] == "BLOCKED_MISROUTED_SURFACE"
    assert receipt["reason_code"] == "SMALL_SURFACE_DIAGNOSTIC_NOT_QC_WORK"
    assert receipt["no_scientific_conclusion"] is True
    assert _job_state(store, qc_job_id) == "BLOCKED_CONFIGURATION"
    assert _surface_row(store, surface_id) == before


def test_the_refusal_is_not_a_verdict_about_the_papyrus(store, tmp_path):
    """Being too small to ask is not an answer, and must not read like one."""
    surface_id = _surface(store, area=PHERC0268_AREA, digest=PHERC0268_SHA256)
    _pending_qc_job(store, surface_id)
    receipt = _worker(store, tmp_path, CountingExecutor()).run_one()

    assert "physical_qc_state" not in receipt
    assert receipt.get("ink_used") is not True
    assert _surface_row(store, surface_id)["physical_qc_state"] == "UNVALIDATED"


# -- and the ones that must still be measured -------------------------------

def test_a_standard_surface_is_measured_exactly_as_before(store, tmp_path):
    surface_id = _surface(store, area=0.5, digest="a" * 64, sample="PHercBig")
    assert store.routing_receipt(surface_id)["route"] == routing.STANDARD
    qc_job_id = _pending_qc_job(store, surface_id)

    executor = CountingExecutor()
    receipt = _worker(store, tmp_path, executor).run_one()

    assert executor.executions == 1
    assert receipt["status"] == "COMPLETED"
    assert _job_state(store, qc_job_id) == "COMPLETED"


def test_an_unmeasured_surface_is_refused(store, tmp_path):
    """Nothing can route it, and a verdict over an unmeasured area is not one."""
    surface_id = _surface(store, area=None, digest="b" * 64, sample="PHercOld")
    assert store.routing_receipt(surface_id) is None
    _pending_qc_job(store, surface_id)

    executor = CountingExecutor()
    receipt = _worker(store, tmp_path, executor).run_one()

    assert executor.executions == 0
    assert receipt["reason_code"] == "ROUTING_UNDECIDABLE_NO_MEASURED_AREA"


def test_a_surface_with_no_receipt_is_routed_where_it_is_used(store, tmp_path):
    """Today only imports write a receipt, so the fleet's own grows have none.

    A rule of "no receipt, no opinion" would leave this gate holding for exactly
    the surfaces nobody grew, so the worker asks the router directly from the
    area the row is carrying. PHerc0268 was a grown surface with a measured area
    and no receipt anywhere, and this is the read that stops it.
    """
    surface_id = _surface(store, area=PHERC0268_AREA, digest=PHERC0268_SHA256)
    with store.connect() as connection:
        connection.execute(
            "DELETE FROM surface_routing_receipts WHERE 1=0")  # keep the trigger honest
    _pending_qc_job(store, surface_id)

    class Unrecorded(type(store)):
        def routing_receipt(self, wanted):
            return None

    executor = CountingExecutor()
    receipt = _worker(Unrecorded(store.path), tmp_path, executor).run_one()

    assert executor.executions == 0
    assert receipt["reason_code"] == "SMALL_SURFACE_DIAGNOSTIC_NOT_QC_WORK"


# -- asked now, not at enqueue ----------------------------------------------

def test_a_route_written_after_the_claim_still_stops_the_worker(store, tmp_path):
    """The whole point: the receipt appears between the claim and the I/O."""
    surface_id = _surface(store, area=0.5, digest="c" * 64, sample="PHercLate")
    _pending_qc_job(store, surface_id)

    class LateRouting(type(store)):
        """A control plane where the surface is re-measured mid-flight."""

        def claim_qc(self, *args, **kwargs):
            claim = super().claim_qc(*args, **kwargs)
            self._routes = {surface_id: {
                **routing.build_receipt(
                    surface_id=surface_id, area_cm2=PHERC0268_AREA,
                    policy=routing.load_policy(), measurement={}, read_set={},
                ),
            }}
            return claim

        def routing_receipt(self, wanted):
            return getattr(self, "_routes", {}).get(wanted) or super(
            ).routing_receipt(wanted)

    late = LateRouting(store.path)
    executor = CountingExecutor()
    receipt = _worker(late, tmp_path, executor).run_one()

    assert executor.executions == 0
    assert receipt["reason_code"] == "SMALL_SURFACE_DIAGNOSTIC_NOT_QC_WORK"


def test_a_forged_receipt_fails_closed_like_a_missing_one(store, tmp_path):
    """A receipt whose digest does not verify is not a receipt."""
    surface_id = _surface(store, area=PHERC0268_AREA, digest="d" * 64,
                          sample="PHercForged")
    _pending_qc_job(store, surface_id)

    class Forged(type(store)):
        def routing_receipt(self, wanted):
            receipt = super().routing_receipt(wanted)
            return {**receipt, "route": routing.STANDARD}

    executor = CountingExecutor()
    receipt = _worker(Forged(store.path), tmp_path, executor).run_one()

    assert executor.executions == 0
    assert receipt["reason_code"] == "ROUTING_RECEIPT_UNVERIFIED"


def test_a_store_that_cannot_answer_is_refused_by_name(store, tmp_path):
    """PostgreSQL has no routing receipts yet, and silence is not a yes."""
    surface_id = _surface(store, area=0.5, digest="e" * 64, sample="PHercNoRoute")
    _pending_qc_job(store, surface_id)

    class Unrouted(type(store)):
        routing_receipt = None

    executor = CountingExecutor()
    receipt = _worker(Unrouted(store.path), tmp_path, executor).run_one()

    assert executor.executions == 0
    assert receipt["reason_code"] == "ROUTING_UNAVAILABLE_ON_THIS_CONTROL_PLANE"


# -- lineage, re-read at the same moment ------------------------------------

def test_lineage_is_resolved_again_before_the_surface_is_touched(store, tmp_path):
    """The claim resolved lineage. That was then; this is the read that counts."""
    surface_id = _surface(store, area=0.5, digest="f" * 64, sample="PHercLineage")
    _pending_qc_job(store, surface_id)
    reads: list[str] = []

    class Watched(type(store)):
        def surface_artifact(self, wanted, *, boundary="P2_EXECUTION_RESOLUTION"):
            reads.append(boundary)
            return super().surface_artifact(wanted, boundary=boundary)

    executor = CountingExecutor()
    _worker(Watched(store.path), tmp_path, executor).run_one()

    assert reads == ["PHYSICAL_QC_CLAIM_RESOLUTION"]
    assert executor.executions == 1


def test_lineage_that_turns_noncanonical_after_the_claim_stops_the_worker(
    store, tmp_path,
):
    surface_id = _surface(store, area=0.5, digest="0" * 64, sample="PHercTurned")
    _pending_qc_job(store, surface_id)

    class Turned(type(store)):
        def claim_qc(self, *args, **kwargs):
            claim = super().claim_qc(*args, **kwargs)
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT payload_json FROM surfaces WHERE surface_id=?",
                    (surface_id,)).fetchone()
                import json
                payload = json.loads(row["payload_json"])
                payload["allow_unvalidated"] = False
                payload["authoritative_lineage"] = {
                    "schema": "campaignx.authoritative_surface_lineage.v1",
                    "namespace": "NONCANONICAL_DISCOVERY",
                    "surface_id": surface_id, "canonical": False,
                }
                connection.execute(
                    "UPDATE surfaces SET payload_json=? WHERE surface_id=?",
                    (json.dumps(payload), surface_id))
                connection.commit()
            return claim

    executor = CountingExecutor()
    from fleet import canonical_lineage

    with pytest.raises(canonical_lineage.CanonicalLineageRejected) as caught:
        _worker(Turned(store.path), tmp_path, executor).run_one()

    assert caught.value.reason_code == "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    assert executor.executions == 0
