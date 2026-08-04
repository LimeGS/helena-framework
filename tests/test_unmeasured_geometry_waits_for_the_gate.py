"""The geometry gate comes before the model, including when there is no verdict.

The stage says geometry is certified before a surface reaches the ink model, and
the hard cases were handled: a rejected surface keeps a durable FAILED QC job, so
claim_qc -- which takes only PENDING -- can never hand it over.

Unmeasured was the gap. It is neither certified nor rejected, and the job state
was chosen with `"FAILED" if geometry_rejected else "PENDING"`, so a surface whose
geometry had never been measured got a claimable job and the coupled ink/CT
adapter could spend model time on it before the gate it is supposed to be behind.

On the live control plane this had not happened: 60 certified, 11 unmeasured, and
every PENDING or CLAIMED job belonged to a certified surface -- because ten of the
eleven were imported, which skips finalization, and the eleventh is a fixture.
Certification runs at finalization, so the exposure was "unless it doesn't".

WAITING_GEOMETRY is that third state, and the promotion is half the fix: without
it, holding a job back would strand the surface instead of gating it.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.store import (  # noqa: E402
    FleetStore,
    QC_WAITING_GEOMETRY,
    qc_job_state_for,
)

PG = (ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py").read_text()
SQLITE = (ROOT / "framework/stages/01-segmentation/fleet/store.py").read_text()


def test_the_three_situations_get_three_answers() -> None:
    assert qc_job_state_for("GEOMETRY_CERTIFIED") == "PENDING"
    assert qc_job_state_for("GEOMETRY_UNMEASURED") == QC_WAITING_GEOMETRY
    assert qc_job_state_for(None) == QC_WAITING_GEOMETRY, (
        "an absent verdict is not a verdict; the default is unmeasured"
    )
    for rejection in ("GEOMETRY_REJECTED_BRIDGE", "GEOMETRY_REJECTED_LAMINA_SWITCH",
                      "GEOMETRY_REJECTED_DISTORTION", "GEOMETRY_REJECTED_COVERAGE"):
        assert qc_job_state_for(rejection) == "FAILED"


def test_waiting_is_not_claimable() -> None:
    """claim_qc's predicate is the whole gate, so it has to stay PENDING-only."""
    assert "WHERE state='PENDING' AND (retry_after IS NULL" in PG, (
        "claim_qc no longer selects on PENDING alone; the gate has moved and "
        "WAITING_GEOMETRY may now be claimable"
    )
    assert QC_WAITING_GEOMETRY not in PG.split("def claim_qc")[1].split("def ")[0], (
        "claim_qc mentions the waiting state, which would defeat the point"
    )


def test_both_stores_decide_it_the_same_way() -> None:
    """The deployment runs PostgreSQL and the tests run SQLite.

    Two copies of a scientific gate that disagree is worse than one that is
    wrong, because the tests would pass while the fleet did something else.
    """
    for store in (PG, SQLITE):
        assert "qc_job_state_for(geometry_state)" in store
        assert '"FAILED" if geometry_rejected else "PENDING"' not in store


def test_the_imported_path_is_gated_too() -> None:
    """The path this file's own docstring said imports take, and never checked.

    enqueue_imported_surface_qc is how a catalogue surface gets a QC job, and both
    stores created it with a hardcoded PENDING. Imports carry no geometry verdict
    at all, so the gate held on the path that measures geometry and leaked on the
    path that never does -- which is the wrong way round, and an audit found it
    while this test sat one function away noting the fact in prose.
    """
    for store in (PG, SQLITE):
        body = store[store.index("def enqueue_imported_surface_qc"):]
        body = body[: body.index("\n    def ")]
        assert "qc_job_state_for(geometry_state)" in body, (
            "the imported-surface path creates its QC job without asking geometry"
        )
        assert "'PENDING'" not in body, (
            "a hardcoded PENDING remains on the imported path"
        )


def test_a_verdict_promotes_what_was_waiting_on_it() -> None:
    """Otherwise the gate is a trap: held back, and never let through."""
    # The queries are f-strings, so the source carries the symbol rather than
    # its value -- which is the point of the symbol.
    for store in (PG, SQLITE):
        certified = store.split("def record_geometry_certification")[1]
        certified = certified[: certified.index("\n    def ")]
        assert 'GEOMETRY_CERTIFIED"' in certified
        assert "SET state='PENDING'" in certified, (
            "nothing promotes a waiting job when certification arrives, so a "
            "certified surface would never reach QC"
        )
        assert "{QC_WAITING_GEOMETRY}" in certified, (
            "the promotion does not name the waiting state, so it is updating "
            "something else"
        )
        # And a rejection has to sweep up the waiting ones too, or a job sits
        # waiting forever on a surface that was refused.
        assert "IN ('PENDING','{QC_WAITING_GEOMETRY}')" in certified


def _certified_surface(store, scroll: str) -> tuple[str, str]:
    snapshot = store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    surface = store.import_surface({
        "surface_id": f"{scroll}-s", "source_snapshot_id": snapshot,
        "sample_id": scroll,
        "artifact_sha256": hashlib.sha256(scroll.encode()).hexdigest(),
        "artifact_uri": "s3://bucket/s", "bbox_xyz": [[0, 0, 0], [10, 10, 10]],
        "area_cm2": 1.0, "state": "QC_SCREENED",
        "physical_qc_state": "UNVALIDATED"})
    store.record_geometry_certification(surface, "GEOMETRY_CERTIFIED",
                                        {"schema": "test"})
    return snapshot, surface


def _enqueue(store, snapshot: str, scroll: str, surface: str, profile: str) -> dict:
    return store.enqueue_imported_surface_qc({
        "surface_id": surface, "source_snapshot_id": snapshot,
        "sample_id": scroll,
        "artifact_sha256": hashlib.sha256(f"{scroll}-qc".encode()).hexdigest(),
        "artifact_uri": "s3://bucket/s", "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "area_cm2": 1.0}, profile_id=profile)


def test_certifying_before_enqueuing_does_not_strand_the_job(tmp_path) -> None:
    """The promotion runs on certification, so it cannot fix a later job.

    Both orders have to arrive at a claimable job. Enqueue-then-certify was
    covered: the certification promotes what is waiting. Certify-then-enqueue
    was not, and it stranded the surface -- the state was read from the payload,
    an import's payload never carries one, so a surface the gate had already
    passed got a job that waits for a verdict that has come and gone.
    """
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshot, surface = _certified_surface(store, "TESTstranded")

    _enqueue(store, snapshot, "TESTstranded", surface, "surface-qc@1.0.0")

    claimed = store.claim_qc("qc-worker", 60, profile_id="surface-qc@1.0.0")
    assert claimed is not None and claimed["surface_id"] == surface, (
        "a certified surface's QC job is not claimable, so it is stranded: the "
        "gate is holding back a surface that already passed it"
    )


def test_the_answer_names_the_state_it_wrote(tmp_path) -> None:
    """It said PENDING whatever it wrote, which is a caller-visible lie."""
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    scroll = "TESThonest"
    snapshot = store.register_snapshot({
        "sample_id": scroll,
        "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })
    # No surface imported first and no verdict anywhere: this is the unmeasured
    # case, and the honest answer is that the job waits.
    answer = _enqueue(store, snapshot, scroll, f"{scroll}-new", "surface-qc@1.0.0")
    assert answer["qc_state"] == QC_WAITING_GEOMETRY, (
        f"enqueue answered {answer['qc_state']} for an unmeasured surface"
    )
    assert store.claim_qc("qc-worker", 60, profile_id="surface-qc@1.0.0") is None
