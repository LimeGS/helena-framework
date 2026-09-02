"""The segmentation control passes on what the chain produced, not on bytes.

A grow is not deterministic: three runs of one control, same seed, same frozen
profile, same deployment, gave three different surfaces. So this control's rows
pass on outcome -- at least one surface produced, certified, CT-supported and
flattened within a bounded budget -- and record which ones with their digests.
These tests drive it with a scripted panel: what it asks, in what order, and
what a missing outcome does to the rows after it.
"""

from __future__ import annotations

import json
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

import run_public_segmentation_control as control  # noqa: E402
from panel_client import PanelError  # noqa: E402

SHA = {"a": "a" * 64, "b": "b" * 64, "c": "c" * 64, "sheet": "5" * 64}


class ReachableSource:
    def __init__(self, reachable=True, catalogued=True):
        self.ok, self.catalogued = reachable, catalogued

    def catalogue_entry(self, sample_id):
        if not self.catalogued:
            return None
        return {"sample_id": sample_id, "ct_uri": "https://bucket/ct.zarr",
                "surface_prediction_uri": "https://bucket/m7.zarr", "voxel_size_um": 9.362}

    def reachable(self, uri):
        return {k: (200 if self.ok else 403) for k in control.PublicSource.KEYS}


class ScriptedPanel:
    """A deployment that grows what it is told to, and records every call."""

    def __init__(self, *, surfaces, qc_jobs=(), p3_state="succeeded", p3_result=None,
                 inserted=48, scoped_to="m1", surfaces_after_batch=1, covered=()):
        # The surfaces exist once `surfaces_after_batch` batches have been
        # queued: a scroll whose first tiling finds nothing and whose second
        # does is the case the budget exists for.
        self.surfaces, self.qc_jobs = surfaces, list(qc_jobs)
        self.scoped_to, self.surfaces_after_batch = scoped_to, surfaces_after_batch
        self.covered = set(covered)
        self.p3_state, self.p3_result, self.inserted = p3_state, p3_result, inserted
        self.calls: list[tuple[str, str, dict | None]] = []
        self.selected: dict[str, str] = {}
        self.p3_jobs: list[dict] = []
        self.p3_queued = 0
        self.batches = 0
        self.polls = 0

    def _visible(self):
        return list(self.surfaces) if self.batches >= self.surfaces_after_batch else []

    def call(self, method, path, body=None, *, timeout=None):
        self.calls.append((method, path, body))
        if path.startswith("/api/missions/m1/artifacts/freeze-p0"):
            return {"artifacts": [{"artifact_id": "p0:PHercX:abc", "content_sha256": "0" * 64}]}
        if path == "/api/missions/m1":
            return {"mission_id": "m1", "scrolls": ["PHercX"]}
        if path == "/api/missions/m1/selection" and method == "GET":
            return {"current": {"choices": dict(self.selected)}}
        if path == "/api/missions/m1/selection":
            self.selected[next(iter(body["choices"]))] = next(iter(body["choices"].values()))
            return {"index": 0}
        if path == "/api/segmentation/runs":
            self.batches += 1
            if body["grid_step"] in self.covered:
                raise PanelError("POST", path, 409,
                                 '{"detail":"nothing was queued: all cells already have a task"}')
            return {"inserted": min(self.inserted, body["max_tasks"]), "generated": 48,
                    "backend": "vc3d", "planner": "cost-aware-v2"}
        if path.startswith("/api/fleet"):
            self.polls += 1
            # Settled after the first poll: everything terminal.
            return {"scoped_to": self.scoped_to,
                    "task_states": [{"state": "NO_SEED", "count": 40},
                                    {"state": "QC_PENDING", "count": len(self._visible())}]}
        if path.startswith("/api/segmentation/segments"):
            return {"segments": self._visible()}
        if path.startswith("/api/segmentation/qc-jobs"):
            return {"jobs": list(self.qc_jobs)}
        if path == "/api/flattening/run":
            self.p3_queued += 1
            return {"job_id": "p3-1"}
        if path.startswith("/api/jobs?phase=P3"):
            return {"jobs": list(self.p3_jobs)}
        raise AssertionError(f"unexpected call {method} {path}")

    def wait_until(self, predicate, *, minutes, tick):
        return predicate()

    def wait_for_job(self, job_id, *, minutes):
        return {"job_id": job_id, "state": self.p3_state, "result": self.p3_result or {}}


def _flattened(sid, sheet=None, source=None):
    """A P3 job's result, in the shape the queue reports it: the sheet under
    result.surfaces, one entry per surface, digest beside it."""
    return {"surfaces": [{"state": "FLATTENED", "surface_id": sid,
                          "artifact_id": "flat-" + sid, "artifact_uri": "/artifacts/flat/" + sid,
                          "profile_id": "flatten-abf-v1@1.0.0", "profile_file_sha256": "9" * 64,
                          "receipt_sha256": "7" * 64,
                          "artifact_sha256": sheet or SHA["sheet"],
                          "source_artifact_sha256": source or SHA[sid]}],
            "exit_code": 0}


def _surface(sid, geometry="GEOMETRY_CERTIFIED", physical="CT_SUPPORTED"):
    return {"surface_id": sid, "artifact_sha256": SHA[sid], "area_cm2": 1.5,
            "geometry_qc_state": geometry, "physical_qc_state": physical}


def _run(panel, tmp_path, source=None):
    return control.run_public_segmentation_control(
        panel, sample_id="PHercX", mission_id="m1", output=tmp_path / "out",
        max_tasks=144, grid_steps=(896, 1024, 768), minutes=1,
        source=source or ReachableSource(), clock=lambda: 0.0, tick=0)


def test_a_chain_that_produced_a_supported_surface_passes_every_row(tmp_path):
    panel = ScriptedPanel(
        surfaces=[_surface("a"), _surface("b", physical="INK_SCREEN_INSUFFICIENT")],
        qc_jobs=[{"state": "COMPLETED", "profile_id": "surface-qc@1.0.0",
                  "profile_sha256": "e" * 64}],
        p3_result=_flattened("a"))
    receipt = _run(panel, tmp_path)

    assert receipt["control_state"] == "CONTROL_PASS", receipt["stages"]
    assert receipt["schema"] == control.SEGMENTATION_SCHEMA
    by = {r["boundary"]: r for r in receipt["stages"]}
    assert by["GROW"]["resource_identity"]["through"] == "helena-queue"
    assert by["GROW"]["counts"]["surfaces"] == 2
    assert by["GEOMETRY"]["counts"] == {"certified": 2, "of": 2,
                                        "geometry_states": ["GEOMETRY_CERTIFIED"]}
    assert by["PHYSICAL_QC"]["counts"]["supported"] == 1
    assert by["PHYSICAL_QC"]["output_hashes"] == {"a": SHA["a"]}
    assert by["PHYSICAL_QC"]["resource_identity"]["profiles"] == [
        {"profile_id": "surface-qc@1.0.0", "profile_sha256": "e" * 64}]
    assert by["FLATTEN"]["output_hashes"]["sheet_sha256"] == SHA["sheet"]
    assert by["FLATTEN"]["input_artifacts"] == [{"surface_id": "a", "artifact_sha256": SHA["a"]}]
    assert by["FLATTEN"]["resource_identity"]["queued_this_run"] is True
    assert by["FLATTEN"]["resource_identity"]["artifact_id"] == "flat-a"
    assert (tmp_path / "out/PUBLIC_SEGMENTATION_CONTROL.json").is_file()
    assert (tmp_path / "out/SURFACES.json").is_file()


def test_the_calls_go_in_the_order_a_stranger_would_make_them(tmp_path):
    panel = ScriptedPanel(surfaces=[_surface("a")], p3_result=_flattened("a"))
    _run(panel, tmp_path)
    writes = [(m, p) for m, p, _ in panel.calls if m == "POST"]
    assert writes == [
        ("POST", "/api/missions/m1/artifacts/freeze-p0"),
        ("POST", "/api/missions/m1/selection"),
        ("POST", "/api/segmentation/runs"),
        ("POST", "/api/flattening/run"),
    ], "a supported surface after the first tiling should stop the spending"
    selection = next(b for m, p, b in panel.calls
                     if m == "POST" and p == "/api/missions/m1/selection")
    assert selection["choices"] == {"P0/PHercX": "p0:PHercX:abc"}
    flatten = next(b for m, p, b in panel.calls if p == "/api/flattening/run")
    assert flatten["surface_id"] == "a" and flatten["limit"] == 1


def test_a_grow_that_produced_nothing_owns_the_outcome(tmp_path):
    panel = ScriptedPanel(surfaces=[])
    receipt = _run(panel, tmp_path)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "GROW"
    by = {r["boundary"]: r for r in receipt["stages"]}
    assert by["GROW"]["reason_code"] == "NO_SURFACE_WITHIN_BUDGET"
    assert by["GROW"]["counts"]["task_states"] == {"NO_SEED": 40, "QC_PENDING": 0}
    # Every tiling was tried before giving up, and the receipt says so.
    assert [b["grid_step"] for b in by["GROW"]["resource_identity"]["batches"]] == [
        896, 1024, 768]
    assert by["GROW"]["resource_identity"]["inserted_total"] == 144
    assert all(r["terminal_state"] == "NOT_RUN_PREREQUISITE"
               for r in receipt["stages"][3:])
    # Nothing was flattened: the chain stopped where the evidence stopped.
    assert not any(p == "/api/flattening/run" for _, p, _ in panel.calls)


def test_surfaces_without_physical_support_do_not_reach_flattening(tmp_path):
    panel = ScriptedPanel(surfaces=[_surface("a", physical="INK_SCREEN_INSUFFICIENT"),
                                    _surface("b", physical="UNVALIDATED")])
    receipt = _run(panel, tmp_path)
    assert receipt["first_nonpassing_boundary"] == "PHYSICAL_QC"
    by = {r["boundary"]: r for r in receipt["stages"]}
    assert by["GEOMETRY"]["terminal_state"] == "PASS"
    assert by["PHYSICAL_QC"]["reason_code"] == "NO_CT_SUPPORTED_SURFACE"
    assert by["PHYSICAL_QC"]["counts"]["physical_states"] == [
        "INK_SCREEN_INSUFFICIENT", "UNVALIDATED"]
    assert by["FLATTEN"]["terminal_state"] == "NOT_RUN_PREREQUISITE"


def test_an_unreachable_source_asks_the_deployment_nothing(tmp_path):
    panel = ScriptedPanel(surfaces=[_surface("a")])
    receipt = _run(panel, tmp_path, source=ReachableSource(reachable=False))
    assert receipt["first_nonpassing_boundary"] == "PUBLIC_SOURCE"
    assert receipt["stages"][0]["reason_code"] == "PUBLIC_SOURCE_UNREACHABLE"
    assert receipt["stages"][0]["resource_identity"]["credentials_used"] is False
    assert panel.calls == []


def test_a_scroll_outside_the_frozen_catalogue_fails_rather_than_waits(tmp_path):
    panel = ScriptedPanel(surfaces=[_surface("a")])
    receipt = _run(panel, tmp_path, source=ReachableSource(catalogued=False))
    assert receipt["control_state"] == "CONTROL_FAILED"
    assert receipt["stages"][0]["reason_code"] == "SCROLL_NOT_IN_FROZEN_CATALOGUE"


def test_a_flattening_that_published_no_sheet_is_not_a_pass(tmp_path):
    panel = ScriptedPanel(surfaces=[_surface("a")], p3_state="succeeded", p3_result={})
    receipt = _run(panel, tmp_path)
    assert receipt["first_nonpassing_boundary"] == "FLATTEN"
    assert receipt["stages"][5]["reason_code"] == "FLATTEN_DID_NOT_PUBLISH"


def test_the_receipt_is_content_addressed_and_rereadable(tmp_path):
    panel = ScriptedPanel(surfaces=[_surface("a")], p3_result=_flattened("a"))
    receipt = _run(panel, tmp_path)
    written = json.loads((tmp_path / "out/PUBLIC_SEGMENTATION_CONTROL.json").read_text())
    assert written["content_sha256"] == receipt["content_sha256"]
    again = control.evaluate_survival_matrix(written)
    assert again["content_sha256"] == written["content_sha256"]


def test_fleet_counts_that_are_not_this_missions_are_refused_not_read(tmp_path):
    """/api/fleet accepted `?mission=` and ignored it for a long time. A control
    that read the whole fleet's 173 NO_SEED rows as its own would settle on
    somebody else's work; it stops and says whose numbers it was handed."""
    panel = ScriptedPanel(surfaces=[_surface("a")], scoped_to=None)
    with pytest.raises(RuntimeError, match="did not scope to m1"):
        _run(panel, tmp_path)


def test_the_budget_is_spent_across_tilings_until_a_surface_is_supported(tmp_path):
    """The first run of this control: one 48-task tiling, one certified
    surface, CT screen insufficient. A second tiling is not a second control;
    it is the rest of the same budget."""
    panel = ScriptedPanel(surfaces=[_surface("a")], surfaces_after_batch=2,
                          p3_result=_flattened("a"))
    receipt = _run(panel, tmp_path)
    assert receipt["control_state"] == "CONTROL_PASS"
    batches = {r["boundary"]: r for r in receipt["stages"]}["GROW"]["resource_identity"]["batches"]
    assert [b["grid_step"] for b in batches] == [896, 1024]
    assert [b["supported_so_far"] for b in batches] == [0, 1]
    # And the second batch asked for what was left of the budget, not for 48 again.
    asked = [b["max_tasks"] for m, p, b in panel.calls if p == "/api/segmentation/runs"]
    assert asked == [144, 96]


def test_a_second_run_on_the_same_mission_does_not_reselect(tmp_path):
    """The deployment refuses a selection that is already current with a 400,
    and the second run of this control on one mission stopped at INTAKE on
    exactly that. Agreeing with the deployment is not a refusal."""
    panel = ScriptedPanel(surfaces=[_surface("a")], p3_result=_flattened("a"))
    panel.selected = {"P0/PHercX": "p0:PHercX:abc"}
    receipt = _run(panel, tmp_path)
    assert receipt["control_state"] == "CONTROL_PASS"
    assert not any(m == "POST" and p == "/api/missions/m1/selection"
                   for m, p, _ in panel.calls)


def test_a_tiling_already_covered_is_skipped_not_a_refusal(tmp_path):
    """The queue answers 409 when every cell a tiling covers already has a
    task. The second run of this control stopped on that with four tilings
    and most of its budget unspent; it moves to the next grid now."""
    panel = ScriptedPanel(surfaces=[_surface("a")], surfaces_after_batch=2,
                          covered={896}, p3_result=_flattened("a"))
    receipt = _run(panel, tmp_path)
    assert receipt["control_state"] == "CONTROL_PASS"
    batches = {r["boundary"]: r for r in receipt["stages"]}["GROW"]["resource_identity"]["batches"]
    assert [(b["grid_step"], b["inserted"]) for b in batches] == [(896, 0), (1024, 48)]
    assert "already_covered" in batches[0]


def test_any_other_refusal_of_the_queue_still_stops_the_control(tmp_path):
    class Refusing(ScriptedPanel):
        def call(self, method, path, body=None, *, timeout=None):
            if path == "/api/segmentation/runs":
                raise PanelError("POST", path, 503, "the control plane is away")
            return super().call(method, path, body, timeout=timeout)

    receipt = _run(Refusing(surfaces=[_surface("a")]), tmp_path)
    assert receipt["first_nonpassing_boundary"] == "GROW"
    assert receipt["stages"][2]["reason_code"] == "QUEUE_REFUSED"


def test_a_sheet_from_another_surface_is_not_this_surfaces_evidence(tmp_path):
    """The sheet names the surface it came from by digest. A sheet whose source
    digest is not the supported surface's is somebody else's evidence."""
    panel = ScriptedPanel(surfaces=[_surface("a")], p3_result=_flattened("a", source=SHA["b"]))
    receipt = _run(panel, tmp_path)
    assert receipt["first_nonpassing_boundary"] == "FLATTEN"
    assert receipt["stages"][5]["reason_code"] == "SHEET_LINEAGE_MISMATCH"


def test_a_second_run_on_the_same_mission_reuses_what_it_holds(tmp_path):
    """The first run of this control on a fresh machine reached FLATTEN and
    read the wrong field of a job that had succeeded. A second run on that
    mission does not re-spend the budget: GROW counts the mission's own tasks,
    QC is already measured, and the sheet the mission already published is
    the same evidence whether this run queued it or the run before did. The
    receipt says which was the case."""
    panel = ScriptedPanel(surfaces=[_surface("a")], inserted=0)
    panel.selected = {"P0/PHercX": "p0:PHercX:abc"}
    panel.p3_jobs = [{"job_id": "p3-earlier", "state": "succeeded",
                      "result": _flattened("a")}]
    # The mission already holds tasks: the fleet reports them, and the queue
    # answers 409 for every tiling because every cell is covered.
    panel.covered = {896, 1024, 768}
    receipt = _run(panel, tmp_path)

    assert receipt["control_state"] == "CONTROL_PASS", receipt["stages"]
    by = {r["boundary"]: r for r in receipt["stages"]}
    assert by["GROW"]["reason_code"] == "SURFACES_HELD_BY_THE_MISSION"
    assert by["GROW"]["resource_identity"]["queued_this_run"] == 0
    assert by["GROW"]["resource_identity"]["mission_task_states"]["NO_SEED"] == 40
    assert by["FLATTEN"]["resource_identity"]["job_id"] == "p3-earlier"
    assert by["FLATTEN"]["resource_identity"]["queued_this_run"] is False
    assert panel.p3_queued == 0
