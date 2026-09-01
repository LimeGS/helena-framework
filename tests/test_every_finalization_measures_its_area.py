"""Before the size gate may fail closed, size has to be a thing that exists.

The routing receipt is written from `area_cm2`. While a finalization could
produce a surface without one, refusing an unrouted surface would quarantine it
for a gap in the *measurement* path rather than for anything about its size --
the same class of error as sending two square millimetres to the ink screen,
pointed the other way.

So this closes the gap first. `inspect_tifxyz` already raises rather than
returning an unmeasured area, and every production finalization goes through it;
what was missing was the boundary check, so a caller that skipped the finalizer
could still write a surface with no area at all and no routing decision.

The check lives in `validate_finalization_evidence`, which is the one function
both stores call, so SQLite and PostgreSQL cannot disagree about it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.common import content_sha256  # noqa: E402
from fleet.store import FleetStore, validate_finalization_evidence  # noqa: E402


def _evidence(area: object, *, manifest_area: object = ...) -> dict:
    """One finalization's worth of cross-bound receipts, area left open."""
    locked_plan_sha256 = "a" * 64
    artifact_sha256 = "b" * 64
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": "t1", "attempt_id": "a1",
        "locked_plan_sha256": locked_plan_sha256, "files": {},
        "artifact_sha256": artifact_sha256,
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "sample_points": [[0.0, 0.0, 0.0]],
        "area_cm2": area if manifest_area is ... else manifest_area,
        "ink_used": False,
    }
    surface = {
        "schema": "campaignx.segment_fleet_surface.v1",
        "surface_id": "s1", "task_id": "t1", "attempt_id": "a1",
        "source_snapshot_id": "src", "sample_id": "PHercTEST",
        "locked_plan_sha256": locked_plan_sha256,
        "artifact_sha256": artifact_sha256,
        "artifact_uri": "file:///artifacts/s1", "ink_used": False,
        "bbox_xyz": manifest["bbox_xyz"],
        "sample_points": manifest["sample_points"],
        "area_cm2": area,
    }
    return {
        "task_id": "t1", "attempt_id": "a1",
        "task_state": "FINALIZING", "attempt_state": "FINALIZING",
        "task_source_snapshot_id": "src", "task_sample_id": "PHercTEST",
        "attempt_locked_plan_sha256": locked_plan_sha256,
        "artifact_attempt_id": "a1", "artifact_state": "UPLOADED",
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": content_sha256(manifest),
        "surface": surface,
    }


def test_a_measured_finalization_is_accepted() -> None:
    validate_finalization_evidence(**_evidence(0.5))


def test_a_zero_area_surface_is_still_a_measurement() -> None:
    """Zero is a measured area. It routes diagnostic; it is not a gap."""
    validate_finalization_evidence(**_evidence(0.0))


@pytest.mark.parametrize(
    "area", [None, "0.5", float("nan"), float("inf"), -0.5, True],
    ids=["missing", "string", "nan", "infinite", "negative", "boolean"],
)
def test_an_unmeasured_finalization_is_refused(area: object) -> None:
    with pytest.raises(ValueError, match="area"):
        validate_finalization_evidence(**_evidence(area))


def test_the_manifest_and_the_surface_must_agree_on_a_real_area() -> None:
    """An area that exists on one side only is not a measurement either."""
    with pytest.raises(ValueError, match="area"):
        validate_finalization_evidence(**_evidence(0.5, manifest_area=None))


def test_the_finalizer_measures_every_surface_it_builds() -> None:
    """The one production producer, read at its source.

    `inspect_tifxyz` raises rather than returning an unmeasured area, and
    `finalize_surface` copies its `area_cm2` into both the manifest and the
    surface. That is what makes the boundary check above a check on callers
    rather than a new requirement the fleet cannot meet.
    """
    finalizer = (ROOT / "framework/stages/01-segmentation/fleet/finalizer.py"
                 ).read_text()
    assert '"area_cm2": inspection["area_cm2"]' in finalizer
    assert "**inspection," in finalizer, (
        "the artifact manifest no longer carries the inspection, so the "
        "manifest and surface areas can drift apart")


def test_a_store_refuses_an_unmeasured_finalization(tmp_path: Path) -> None:
    """And it refuses before writing anything, not after."""
    store = FleetStore(tmp_path / "fleet.sqlite3")
    store.initialize()
    source_id = store.register_snapshot({
        "sample_id": "PHercTEST", "ct_uri": "file:///ct", "m7_uri": "file:///m7",
        "shape_xyz": [10, 10, 10], "voxel_size_um": 7.91,
        "coordinate_frame": "ct_l0_xyz"})
    store.create_tasks([{
        "source_snapshot_id": source_id, "sample_id": "PHercTEST",
        "cell_id": "cell-unmeasured", "grid_version": "g1",
        "policy_version": "p1", "bounds_xyz": [[0, 0, 0], [1, 1, 1]],
        "center_xyz": [0.5, 0.5, 0.5], "priority": 1.0,
        "parameter_envelope": {}, "catalog_snapshot_sha256": "c" * 64}])
    task = store.claim(worker_id="w1", lease_seconds=600)
    locked_plan = {"schema": "test.locked_plan.v1", "task_id": task["task_id"]}
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"],
                     "RUNNING", locked_plan=locked_plan)
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": task["task_id"], "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan), "files": {},
        "artifact_sha256": "d" * 64, "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "sample_points": [[0.0, 0.0, 0.0]], "area_cm2": None, "ink_used": False}
    artifact_set_id = store.add_artifact_set(
        task["task_id"], task["attempt_id"], task["lease_token"], manifest,
        "file:///staging")
    store.transition(task["task_id"], task["attempt_id"], task["lease_token"],
                     "FINALIZING")
    surface = {
        "schema": "campaignx.segment_fleet_surface.v1",
        "surface_id": "unmeasured-surface",
        "source_snapshot_id": source_id, "sample_id": "PHercTEST",
        "artifact_sha256": "d" * 64, "artifact_uri": "file:///artifacts/u",
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]], "sample_points": [[0.0, 0.0, 0.0]],
        "area_cm2": None, "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "locked_plan_sha256": content_sha256(locked_plan), "ink_used": False}

    with pytest.raises(ValueError, match="area"):
        store.finalize(task["task_id"], task["attempt_id"], task["lease_token"],
                       surface, artifact_set_id, "surface-qc@1.0.0")

    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) AS n FROM surfaces").fetchone()["n"] == 0
        assert connection.execute(
            "SELECT COUNT(*) AS n FROM qc_jobs").fetchone()["n"] == 0
