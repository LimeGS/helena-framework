"""A mission with no P0 selection is empty, never an alias for the fleet."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_read_scope_keeps_global_and_empty_distinct(monkeypatch):
    import panel.app as app

    monkeypatch.setattr(app, "mission_scrolls", lambda mission: set())
    assert app.read_scope(None, None) is None
    assert app.read_scope("brand-new", None) == set()
    assert app.read_scope("brand-new", "PHerc0826") == set()


def test_empty_mission_segmentation_does_not_fall_back_to_global(monkeypatch):
    import panel.app as app

    seen = []

    def fleet(samples=None):
        seen.append(samples)
        count = 99 if samples is None else 0
        return {
            "available": True, "tasks": count, "attempts": count,
            "surfaces": count, "imported": count, "leased": 0,
            "stale_leases": 0, "task_states": [], "workers": [],
            "surfaces_by_sample": ([{"sample_id": "PHerc826", "count": 99}]
                                   if samples is None else []),
        }

    monkeypatch.setattr(app, "fleet_status", fleet)
    monkeypatch.setattr(app, "public_segments", lambda: {
        "total": 40, "by_sample": {"PHerc0826": 40}, "origin": "fixture",
    })
    state = app.segmentation_state(samples=set())
    assert seen == [set()]
    assert state["public"]["total"] == 0
    assert state["private"]["total"] == 0
    assert state["queue"]["tasks"] == 0
    assert state["private"]["by_sample"] == []


def test_empty_mission_phases_have_no_progress_or_input(monkeypatch):
    import panel.app as app
    from fastapi.responses import JSONResponse

    monkeypatch.setattr(app, "DSN", "")
    monkeypatch.setattr(app, "mission_scrolls", lambda mission: set())
    monkeypatch.setattr(app, "index_runs", lambda mission_id=None: [])
    monkeypatch.setattr(app, "api_flattening", lambda sample=None, mission=None:
                        JSONResponse({"available": True, "certified": 0,
                                      "flattened": 0, "awaiting": 0, "rows": []}))
    monkeypatch.setattr(app, "render_status",
                        lambda sample_id=None, mission_id=None, phase="P4": {
                            {"P4": "renders", "P5": "screenings"}.get(phase, "jobs")
                            + "_succeeded": 0,
                            {"P4": "renders", "P5": "screenings"}.get(phase, "jobs")
                            + "_failed": 0,
                            {"P4": "renders", "P5": "screenings"}.get(phase, "jobs")
                            + "_queued": 0,
                        })

    p1 = app.phase_state("P1", mission_id="brand-new")
    assert p1["state"] == {"surfaces": 0, "area_cm2": 0.0,
                           "tasks": 0, "attempts": 0}
    assert p1["artefacts"] == [] and p1["input_available"] is False

    p2 = app.phase_state("P2", mission_id="brand-new")
    assert p2["state"] == {"surfaces": 0, "certified": 0,
                           "unmeasured": 0, "rejected": 0}
    assert p2["input_available"] is False

    p3 = app.phase_state("P3", mission_id="brand-new")
    assert p3["state"] == {"certified": 0, "flattened": 0, "awaiting": 0}
    assert p3["input_available"] is False

    p5 = app.phase_state("P5", mission_id="brand-new")
    assert "lanes" not in p5["state"]
    assert p5["state"]["receipts"] == 0


def test_mutations_require_a_p0_scroll_and_mission_membership(monkeypatch):
    import panel.app as app

    monkeypatch.setattr(app, "mission_scrolls", lambda mission: set())
    with pytest.raises(app.HTTPException) as empty:
        app.require_write_sample("brand-new", None, "P3 flattening")
    assert empty.value.status_code == 409
    assert "Select and freeze a scroll in P0" in str(empty.value.detail)

    monkeypatch.setattr(app, "mission_scrolls", lambda mission: {"PHerc826"})
    assert app.require_write_sample("one-scroll", None, "P2 certification") == "PHerc826"
    with pytest.raises(app.HTTPException) as outside:
        app.require_write_sample("one-scroll", "PHerc0841", "P4 job")
    assert outside.value.status_code == 409
    assert "not selected" in str(outside.value.detail)

    with pytest.raises(app.HTTPException) as historical:
        app.require_write_sample("unfiled", "PHerc0826", "P1 segmentation")
    assert historical.value.status_code == 409
    assert "read-only" in str(historical.value.detail)


def test_unfiled_is_receipt_scoped_and_unknown_mission_is_404(monkeypatch):
    import panel.app as app

    original_mission_scrolls = app.mission_scrolls
    monkeypatch.setattr(app, "mission_scrolls",
                        lambda mission: {"PHerc826"} if mission == "unfiled" else set())
    assert app.read_scope("unfiled") == {"PHerc826"}
    assert app.read_scope("unfiled", "PHerc0826") == {"PHerc826"}
    assert app.read_scope("unfiled", "PHerc0841") == set()

    def missing(_root, _mission):
        raise app.mission_contract.MissionError("no such mission")

    monkeypatch.setattr(app, "mission_scrolls", original_mission_scrolls)
    monkeypatch.setattr(app.mission_contract, "resolve", missing)
    with pytest.raises(app.HTTPException) as unknown:
        app.mission_scrolls("does-not-exist")
    assert unknown.value.status_code == 404


def test_segmentation_and_artifact_writes_validate_manifest_membership(
        monkeypatch, tmp_path):
    import panel.app as app

    checked: list[tuple[str | None, str | None, str]] = []

    def reject(mission, sample, operation):
        checked.append((mission, sample, operation))
        raise app.HTTPException(409, "outside the frozen P0 selection")

    monkeypatch.setattr(app, "require_write_sample", reject)

    run = app.SegmentationRunRequest(
        sample_id="PHerc0841", mission_id="only-826", backend="vc3d")
    with pytest.raises(app.HTTPException) as segmentation:
        app.api_queue_segmentation(run, None)
    assert segmentation.value.status_code == 409
    assert checked[-1] == ("only-826", "PHerc0841", "P1 segmentation")

    monkeypatch.setattr(app, "mission_directory", lambda _mission: tmp_path)
    artifact = app.ArtifactRequest(
        phase="P0", sample_id="PHerc0841", kind="frozen-source",
        path="does-not-matter.json")
    with pytest.raises(app.HTTPException) as registration:
        app.api_register_artifact("only-826", artifact)
    assert registration.value.status_code == 409
    assert checked[-1] == ("only-826", "PHerc0841", "artifact registration")


def test_foreign_subject_cannot_resurrect_receipts_or_jobs(monkeypatch):
    import panel.app as app

    monkeypatch.setattr(app, "DSN", "present-for-queue-branch")
    monkeypatch.setattr(app, "mission_scrolls", lambda _mission: {"PHerc826"})
    # Even a corrupt historical receipt bearing this mission id is outside P0.
    monkeypatch.setattr(app, "index_runs", lambda mission_id=None: [
        SimpleNamespace(sample_id="PHerc0841")])
    monkeypatch.setattr(app, "job_store", lambda: pytest.fail(
        "an empty/foreign scope must not query the job ledger"))

    state = app.phase_state(
        "P4", mission_id="only-826", subject="PHerc0841")
    assert state["jobs"] == []
    assert state["state"]["renders_succeeded"] == 0

    subjects = json.loads(app.api_subjects(mission="only-826").body)
    assert [row["sample_id"] for row in subjects["subjects"]] == ["PHerc826"]
    assert subjects["subjects"][0]["runs"] == 0


def test_phase_summary_propagates_an_unknown_mission(monkeypatch):
    import panel.app as app

    def unknown(_mission):
        raise app.HTTPException(404, "no such mission")

    monkeypatch.setattr(app, "mission_scrolls", unknown)
    with pytest.raises(app.HTTPException) as missing:
        app.api_phase_summary(mission="does-not-exist")
    assert missing.value.status_code == 404


def test_mission_resolve_rejects_path_aliases(tmp_path):
    from framework.contracts import mission

    mission.create(tmp_path, mission_id="only-826", name="Only 826",
                   scrolls=["PHerc0826"])
    with pytest.raises(mission.MissionError):
        mission.resolve(tmp_path, "alias/../only-826")


def test_same_scroll_is_not_a_mission_boundary():
    """P1 reads the task's mission id; sample equality alone may never scope it."""
    source = (ROOT / "panel/app.py").read_text(encoding="utf-8")
    runs = source[source.index("def segmentation_runs("):
                  source.index("\ndef scoped_queue(")]
    queue = source[source.index("def scoped_queue("):
                   source.index("\ndef segmentation_state(")]
    no_seed = source[source.index("def api_no_seed("):
                     source.index("\n\nGEOMETRY_MEANING")]
    assert "t.mission_id" in runs
    assert "t.mission_id" in queue
    assert "t.mission_id" in no_seed


def test_postgres_task_identity_includes_mission():
    migration = (ROOT / "framework/stages/01-segmentation/fleet/migrations/001_postgresql.sql").read_text()
    assert "mission_id text NOT NULL DEFAULT 'unfiled'" in migration
    assert "UNIQUE(mission_id, source_snapshot_id, grid_version, cell_id, policy_version)" in migration
    assert "VALUES (11, 'mission-bound segmentation task identity')" in migration


def test_same_cell_can_be_queued_once_per_mission(tmp_path):
    """A completed task in mission A is not mission B's run."""
    stage = ROOT / "framework/stages/01-segmentation"
    sys.path.insert(0, str(stage))
    from fleet.generator import DEFAULT_ENVELOPE
    from fleet.store import FleetStore

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store.register_snapshot({
        "sample_id": "PHerc826", "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64, "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64, "shape_xyz": [512, 512, 512],
        "voxel_size_um": 7.91, "coordinate_frame": "ct_l0_xyz",
    })
    base = {
        "source_snapshot_id": source_id, "sample_id": "PHerc826",
        "cell_id": "same-cell", "grid_version": "grid-v1",
        "policy_version": "policy-v1",
        "bounds_xyz": [[0, 0, 0], [256, 256, 256]],
        "center_xyz": {"x": 128, "y": 128, "z": 128},
        "priority": 1.0, "parameter_envelope": DEFAULT_ENVELOPE,
        "catalog_snapshot_sha256": "2" * 64, "ink_used": False,
    }
    assert store.create_tasks([{**base, "mission_id": "mission-a"}]) == (1, 1)
    assert store.create_tasks([{**base, "mission_id": "mission-a"}]) == (0, 1)
    assert store.create_tasks([{**base, "mission_id": "mission-b"}]) == (1, 1)


def test_a_mission_page_counts_the_selected_scrolls_surfaces(monkeypatch):
    """The P1 tile said "none grown yet" beside a table of seventeen.

    Under a mission the fleet's per-scroll rows are dropped, since they are not
    the mission's, and the selected scroll's count fell through to nothing. The
    scoped query is restricted to that scroll already; it has to be the answer.
    """
    import panel.app as app

    def fleet(samples=None):
        return {"available": True, "tasks": 0, "attempts": 0, "surfaces": 0,
                "imported": 0, "leased": 0, "stale_leases": 0, "task_states": [],
                "workers": [], "surfaces_by_sample": []}

    asked = []

    def scoped(samples, mission_id=None):
        asked.append((samples, mission_id))
        return {"tasks": 144, "leased": 0, "stale_leases": 0, "attempts": 144,
                "by_state": {}, "surfaces": 17, "area_cm2": 7.86, "imported": 0,
                "imported_area_cm2": 0.0, "certified": 17,
                "certified_area_cm2": 7.86, "ct_supported": 4,
                "ct_supported_area_cm2": 0.91}

    monkeypatch.setattr(app, "fleet_status", fleet)
    monkeypatch.setattr(app, "scoped_queue", scoped)
    monkeypatch.setattr(app, "public_segments", lambda: {
        "total": 0, "by_sample": {}, "origin": "fixture"})
    scroll = app.stored_scroll("PHerc826") or "PHerc826"
    state = app.segmentation_state(sample="PHerc826", samples={scroll},
                                   mission_id="segmentation-control")
    assert asked == [({scroll}, "segmentation-control")]
    assert state["private"]["total"] == 17
    assert state["private"]["for_sample"]["count"] == 17
    assert state["private"]["for_sample"]["area_cm2"] == 7.86

    outside = app.segmentation_state(sample="PHerc826", samples={"PHerc0139"},
                                     mission_id="ink-control")
    assert outside["private"]["for_sample"] is None
