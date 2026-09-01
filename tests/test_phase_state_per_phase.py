"""Each phase page reporting its own numbers, and reporting them at all.

Three faults, all of which showed the same way -- a phase reading zero while the
control plane held fifty-two surfaces:

* P1 and P2 shared one branch, so two phases answering two different questions
  printed one number, and "certified" was counted off `state` and
  `physical_qc_state` (which hold QC_SCREENED and CT_SUPPORTED) while the verdict
  lives in `geometry_qc_state`.
* P3 had no branch at all and fell through to the fleet counters.
* The panel image carried `workspace/catalog/eligible_volumes.json` as a
  dangling symlink, so `catalog_sample_id` read no catalog, translated no scroll
  name, and every mission-scoped query matched no row. This one is invisible in
  the source and only exists in the built image, which is why the last test here
  reads the Containerfile.
"""

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

# QC_SCREENED and CT_SUPPORTED are what the fleet actually writes, and neither
# says anything about geometry. A surface is certified or it is not, and only one
# column knows.
SURFACES = [
    {"surface_id": "s-1", "state": "QC_SCREENED", "physical_qc_state": "CT_SUPPORTED",
     "geometry_qc_state": "GEOMETRY_CERTIFIED", "area_cm2": 1.5, "created_at": None},
    {"surface_id": "s-2", "state": "QC_SCREENED", "physical_qc_state": "CT_SUPPORTED",
     "geometry_qc_state": "GEOMETRY_CERTIFIED", "area_cm2": 2.5, "created_at": None},
    {"surface_id": "s-3", "state": "QC_SCREENED", "physical_qc_state": "CT_SUPPORTED",
     "geometry_qc_state": "GEOMETRY_UNMEASURED", "area_cm2": 1.0, "created_at": None},
    {"surface_id": "s-4", "state": "QC_SCREENED", "physical_qc_state": "CT_SUPPORTED",
     "geometry_qc_state": "GEOMETRY_REJECTED_SEAM", "area_cm2": 1.0, "created_at": None},
]


@pytest.fixture
def app_module(monkeypatch):
    import panel.app as module
    from fastapi.responses import JSONResponse

    monkeypatch.setattr(
        module, "subject_surfaces",
        lambda sample, mission_id=None: list(SURFACES),
    )
    monkeypatch.setattr(module, "fleet_status",
                        lambda samples=None: {"available": True, "tasks": 231,
                                              "surfaces": 4, "attempts": 270,
                                              "stale_leases": 0})
    monkeypatch.setattr(module, "api_flattening",
                        lambda sample=None, mission=None: JSONResponse(
                            {"available": True, "certified": 2, "flattened": 1,
                             "awaiting": 1, "rows": []}))
    monkeypatch.setattr(module, "render_status",
                        lambda sample_id=None, mission_id=None, phase="P4": {
                            f"{'renders' if phase == 'P4' else 'screenings'}_succeeded": 2,
                            f"{'renders' if phase == 'P4' else 'screenings'}_failed": 4,
                            f"{'renders' if phase == 'P4' else 'screenings'}_queued": 0})
    monkeypatch.setattr(module, "index_runs", lambda mission_id=None: [])
    monkeypatch.setattr(module, "mission_scrolls", lambda mission: {"PHerc826"})
    return module


def state_of(module, phase: str, **scope) -> dict:
    return module.phase_state(phase, mission_id="m", **scope)["state"]


def test_p2_counts_the_column_the_verdict_lives_in(app_module):
    state = state_of(app_module, "P2", subject="PHerc826")
    assert state["certified"] == 2
    assert state["unmeasured"] == 1
    assert state["rejected"] == 1
    assert state["surfaces"] == 4


def test_p1_and_p2_do_not_answer_the_same_question(app_module):
    """Sharing a branch is how a bug in one becomes a bug in both, and how a
    reader loses any way to tell which phase a number describes."""
    p1 = state_of(app_module, "P1", subject="PHerc826")
    p2 = state_of(app_module, "P2", subject="PHerc826")
    assert "certified" not in p1, "verdicts are P2's answer"
    assert p1["surfaces"] == 4 and p1["area_cm2"] == 6.0
    assert p1 != p2


def test_p3_reports_sheets_rather_than_falling_through_to_the_fleet(app_module):
    state = state_of(app_module, "P3", subject="PHerc826")
    assert state == {"certified": 2, "flattened": 1, "awaiting": 1}


def test_p4_reports_what_it_rendered(app_module):
    """From the job queue. Counting `tiff_dir` in run receipts cannot see a
    render queued through the panel, so the phase read zero the same afternoon
    it produced two layer stacks."""
    state = state_of(app_module, "P4", subject="PHerc826")
    assert state["renders_succeeded"] == 2
    assert state["renders_failed"] == 4
    # No legacy receipt for this mission, so no tile claiming zero of something
    # the reader has no reason to think about.
    assert "layer_stacks_referenced" not in state


def test_p5_counts_its_queue_and_not_only_receipts(app_module, monkeypatch):
    """A job queued without a mission files its receipt under `unfiled`, so a
    screening that ran this afternoon read as zero runs on the mission's page
    while its probability map sat on disk with a receipt beside it."""
    monkeypatch.setattr(app_module, "ink_profiles", lambda: [{"profile_id": "x"}])
    state = state_of(app_module, "P5", subject="PHerc826")
    assert state["receipts"] == 0
    assert state["screenings_succeeded"] == 2
    assert state["screenings_failed"] == 4


def test_p6_counts_liveness_from_the_p5_queue(app_module, monkeypatch):
    """P6 runs inside P5, so its verdict must follow the queued P5 result.

    A mission-scoped screening does not become a legacy filesystem receipt;
    reading only ``index_runs`` leaves P6 at zero after a successful ALIVE job.
    """
    class Store:
        def jobs(self, **filters):
            if filters.get("phase") != "P5":
                return []
            return [
                {
                    "job_id": "p5-1",
                    "sample_id": "PHerc826",
                    "phase": "P5",
                    "state": "succeeded",
                    "output_dir": "/runs/p5-1",
                    "result": {"liveness": {"verdict": "ALIVE"}},
                }
            ]

    monkeypatch.setattr(app_module, "DSN", "postgresql://unused")
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    state = state_of(app_module, "P6", subject="PHerc826")

    assert state == {
        "runs_with_liveness": 1,
        "verdicts": {"ALIVE": 1},
        "runs_without_liveness": 0,
    }
    detail = app_module.phase_state("P6", mission_id="m", subject="PHerc826")
    assert detail["input_available"] is True
    assert detail["blocked"] is None


def test_p7_separates_execution_success_from_scientific_outcome(app_module):
    counts = app_module.adjudication_outcome_counts([
        {"adjudication": {"verdict": "PASS"}},
        {"adjudication": {"verdict": "FAIL"}},
        {},
    ])

    assert counts == {
        "adjudications_passed": 1,
        "adjudications_refuted": 1,
        "adjudications_without_verdict": 1,
    }


def test_p7_can_use_an_alive_map_from_a_queued_p5(app_module, monkeypatch):
    class Store:
        def jobs(self, **filters):
            if filters.get("phase") == "P5":
                return [{
                    "job_id": "p5-1", "sample_id": "PHerc826",
                    "phase": "P5", "state": "succeeded",
                    "output_dir": "/runs/p5-1",
                    "result": {"liveness": {"verdict": "ALIVE"}},
                }]
            return []

    monkeypatch.setattr(app_module, "DSN", "postgresql://unused")
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    detail = app_module.phase_state("P7", mission_id="m", subject="PHerc826")

    assert detail["state"]["maps_available"] == 1
    assert detail["state"]["runs_screenable"] == 1
    assert detail["input_available"] is True
    assert detail["blocked"] is None


def test_p9_can_use_the_radial_table_from_a_successful_p8(app_module, monkeypatch):
    class Store:
        def jobs(self, **filters):
            if filters.get("phase") == "P8":
                return [{
                    "job_id": "p8-order", "sample_id": "PHerc826",
                    "phase": "P8", "state": "succeeded",
                    "parameters": {"out_path": "/runs/wrap_radial.json"},
                    "result": {},
                }]
            return []

    monkeypatch.setattr(app_module, "DSN", "postgresql://unused")
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    detail = app_module.phase_state("P9", mission_id="m", subject="PHerc826")

    assert detail["input_available"] is True
    assert detail["blocked"] is None


def test_successful_public_mesh_p8_is_not_reported_as_blocked(
        app_module, monkeypatch):
    """P8 can enter from public meshes without producing a P1 surface first.

    Once that lane succeeds, the panel has definitive evidence that its input
    existed. Reporting the same phase as both successful and blocked makes the
    navigation rail contradict its own job ledger.
    """
    class Store:
        def jobs(self, **filters):
            if filters.get("phase") == "P8":
                return [{
                    "job_id": "p8-public", "sample_id": "PHerc826",
                    "phase": "P8", "state": "succeeded",
                    "parameters": {
                        "lane": "column-atlas",
                        "out_path": "/runs/wrap_radial.json",
                    },
                    "result": {"artifact_id": "p8:PHerc826:public"},
                }]
            return []

    monkeypatch.setattr(app_module, "DSN", "postgresql://unused")
    monkeypatch.setattr(app_module, "job_store", lambda: Store())
    monkeypatch.setattr(app_module, "subject_surfaces", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(app_module, "render_status", lambda *_args, **_kwargs: {
        "assemblies_succeeded": 1,
        "assemblies_failed": 0,
        "assemblies_queued": 0,
    })

    detail = app_module.phase_state("P8", mission_id="m", subject="PHerc826")

    assert detail["state"]["assemblies_succeeded"] == 1
    assert detail["input_available"] is True
    assert detail["blocked"] is None


def test_a_mission_with_no_subject_is_still_scoped(app_module, monkeypatch):
    """Unscoped, these three answered for the whole fleet on a mission's page --
    which is the same defect, in the direction that looks like progress."""
    seen: list[tuple[set[str] | None, str | None]] = []
    monkeypatch.setattr(
        app_module, "scoped_queue",
        lambda samples=None, mission_id=None: seen.append((samples, mission_id)) or {
            "tasks": 1, "surfaces": 1, "attempts": 1,
            "area_cm2": 1.0, "stale_leases": 0,
        },
    )
    state_of(app_module, "P1")
    assert seen == [({"PHerc826"}, "m")]


def test_a_queued_job_names_the_scroll_the_way_the_control_plane_does():
    """The queue held one scroll under two names -- PHerc0826 straight from the
    bucket beside PHerc826 from the catalog -- because enqueue stored what the
    caller sent. A mission-scoped phase then counted only half its own renders:
    P4 rendered four layer stacks and reported two."""
    import panel.app as module

    assert module.stored_scroll("PHerc0826") == "PHerc826"
    source = (ROOT / "panel/app.py").read_text()
    enqueue = source[source.index("def api_enqueue("):]
    enqueue = enqueue[:enqueue.index("\n@app.")]
    assert "require_write_sample(" in enqueue
    assert "sample_id=job_sample" in enqueue, \
        "api_enqueue must normalize and authorize the sample before queueing"


def test_p8_merge_publication_store_is_owned_by_the_deployment(app_module, monkeypatch):
    """A browser or API client cannot redirect immutable merge evidence to an
    arbitrary path.  P8's publication root is infrastructure configuration."""
    captured = {}

    class Store:
        def enqueue(self, **kwargs):
            captured.update(kwargs)
            return "merge-job"

    monkeypatch.setattr(app_module, "RECONSTRUCTION_STORE", "s3://helena/reconstruction-v1")
    monkeypatch.setattr(app_module, "require_write_sample",
                        lambda mission, sample, operation: sample)
    monkeypatch.setattr(app_module, "module_disabled", lambda phase, module: False)
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    request = app_module.JobRequest(
        sample_id="PHerc826", phase="P8", mission_id="golden-run",
        profile_id="vc3d-tifxyz-merge@1.0.0",
        parameters={"lane": "vc3d-tifxyz-merge"},
    )
    response = app_module.api_enqueue(
        request, SimpleNamespace(state=SimpleNamespace(username="tester")))

    assert response.status_code == 201
    assert captured["server_parameters"]["artifact_store"] == \
        "s3://helena/reconstruction-v1"
    assert "artifact_store" not in captured["parameters"]


def test_p4_render_publication_store_is_owned_by_the_deployment(app_module, monkeypatch):
    """A browser or API client cannot redirect a rendered layer stack.

    The form hides ``artifact_store``, but the HTTP boundary still receives an
    arbitrary parameters object.  P4 must therefore enforce the deployment's
    durable store exactly as P8 does for immutable merge evidence.
    """
    captured = {}

    class Store:
        def enqueue(self, **kwargs):
            captured.update(kwargs)
            return "render-job"

    monkeypatch.setattr(app_module, "RENDER_STORE", "s3://helena/layer-stacks-v1")
    monkeypatch.setattr(app_module, "require_write_sample",
                        lambda mission, sample, operation: sample)
    monkeypatch.setattr(app_module, "module_disabled", lambda phase, module: False)
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    request = app_module.JobRequest(
        sample_id="PHerc826", phase="P4", mission_id="golden-run",
        parameters={
            "lane": "vc-render-tifxyz",
            "segmentation": "/surfaces/s-1",
            "volume": "/volumes/scroll.zarr",
            "scale": 1.0,
            "group_idx": 0,
        },
    )
    response = app_module.api_enqueue(
        request, SimpleNamespace(state=SimpleNamespace(username="tester")))

    assert response.status_code == 201
    # As the server's own value, not merged into the request's: the queue is
    # told which half each parameter came from rather than having to trust that
    # the panel got there first.
    assert captured["server_parameters"]["artifact_store"] == \
        "s3://helena/layer-stacks-v1"
    assert "artifact_store" not in captured["parameters"]


def test_a_client_that_names_a_publication_store_is_refused(app_module, monkeypatch):
    """Refused, not quietly corrected.

    The form hides the field and the panel used to overwrite whatever arrived
    in its place. Overwriting is the right destination and the wrong signal: a
    caller redirecting a render to a bucket it owns learns nothing, and neither
    does anyone reading the logs afterwards.
    """
    monkeypatch.setattr(app_module, "RENDER_STORE", "s3://helena/layer-stacks-v1")
    monkeypatch.setattr(app_module, "require_write_sample",
                        lambda mission, sample, operation: sample)
    monkeypatch.setattr(app_module, "module_disabled", lambda phase, module: False)

    request = app_module.JobRequest(
        sample_id="PHerc826", phase="P4", mission_id="golden-run",
        parameters={
            "lane": "vc-render-tifxyz",
            "segmentation": "/surfaces/s-1",
            "volume": "/volumes/scroll.zarr",
            "scale": 1.0,
            "group_idx": 0,
            "artifact_store": "s3://attacker/exfil",
        },
    )
    with pytest.raises(app_module.HTTPException) as refused:
        app_module.api_enqueue(
            request, SimpleNamespace(state=SimpleNamespace(username="tester")))
    assert refused.value.status_code == 409
    assert "artifact_store" in str(refused.value.detail)


def test_p4_local_override_cannot_smuggle_a_publication_store(app_module, monkeypatch):
    """Local-only mode means keep the worker-local result, not trust a URI.

    When no durable destination is configured, ``allow_local_layers`` is the
    explicit escape hatch.  It must not also let a client retain an arbitrary
    artifact store that the deployment did not authorize.
    """
    captured = {}

    class Store:
        def enqueue(self, **kwargs):
            captured.update(kwargs)
            return "local-render-job"

    monkeypatch.setattr(app_module, "RENDER_STORE", "")
    monkeypatch.setattr(app_module, "require_write_sample",
                        lambda mission, sample, operation: sample)
    monkeypatch.setattr(app_module, "module_disabled", lambda phase, module: False)
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    request = app_module.JobRequest(
        sample_id="PHerc826", phase="P4", mission_id="golden-run",
        parameters={
            "lane": "vc-render-tifxyz",
            "segmentation": "/surfaces/s-1",
            "volume": "/volumes/scroll.zarr",
            "scale": 1.0,
            "group_idx": 0,
            "allow_local_layers": True,
            "artifact_store": "s3://untrusted/destination",
        },
    )
    # The escape hatch is still an escape hatch; the store that rode along with
    # it is refused outright rather than dropped, so the caller learns the
    # field was never theirs.
    with pytest.raises(app_module.HTTPException) as refused:
        app_module.api_enqueue(
            request, SimpleNamespace(state=SimpleNamespace(username="tester")))
    assert refused.value.status_code == 409
    assert "artifact_store" in str(refused.value.detail)

    request.parameters.pop("artifact_store")
    response = app_module.api_enqueue(
        request, SimpleNamespace(state=SimpleNamespace(username="tester")))
    assert response.status_code == 201
    assert captured["parameters"]["allow_local_layers"] is True
    assert "artifact_store" not in captured["parameters"]
    assert "artifact_store" not in (captured["server_parameters"] or {})


def test_p5_probability_map_store_is_owned_by_the_deployment(app_module, monkeypatch):
    """P7 may run on another worker, so P5's map destination is not client input."""
    captured = {}

    class Store:
        def enqueue(self, **kwargs):
            captured.update(kwargs)
            return "screening-job"

    monkeypatch.setattr(app_module, "INK_STORE", "s3://helena/ink-maps-v1")
    monkeypatch.setattr(app_module, "require_write_sample",
                        lambda mission, sample, operation: sample)
    monkeypatch.setattr(app_module, "module_disabled", lambda phase, module: False)
    monkeypatch.setattr(app_module, "ink_profiles", lambda: [{
        "profile_id": "ink-profile@1", "method_id": "ink-method",
    }])
    monkeypatch.setattr(app_module, "registry_entries", lambda: {
        "ink-method": {"validation_status": "VALIDATED"},
    })
    monkeypatch.setattr(app_module, "job_store", lambda: Store())

    request = app_module.JobRequest(
        sample_id="PHerc826", phase="P5", mission_id="golden-run",
        profile_id="ink-profile@1",
        parameters={
            "tiff_dir": "/runs/layers",
            "checkpoint": "/models/model.safetensors",
            "source_pixel_um": 2.399,
        },
    )
    response = app_module.api_enqueue(
        request, SimpleNamespace(state=SimpleNamespace(username="tester")))

    assert response.status_code == 201
    assert captured["server_parameters"]["artifact_store"] == "s3://helena/ink-maps-v1"
    assert "artifact_store" not in captured["parameters"]


def test_a_p8_derived_surface_remains_visible_to_its_mission_in_p3():
    """Both readers scope surfaces the way the shared predicate does.

    The three ways a surface belongs to a mission -- grown by its tasks, derived
    by its ink jobs, uploaded into it -- live in `surface_mission_predicate`
    now, so what these two must not do is write a fourth definition inline. The
    derivation branch is asserted where it is defined.
    """
    source = (ROOT / "panel/app.py").read_text()
    flattening = source[source.index("def api_flattening("):
                        source.index("\n@app.get(\"/api/geometry\")")]
    subjects = source[source.index("def subject_surfaces("):
                      source.index("\n@app.get(\"/api/subjects\")")]
    for query in (flattening, subjects):
        assert "surface_mission_predicate(" in query
    predicate = source[source.index("def surface_mission_predicate("):]
    predicate = predicate[:predicate.index("\n\n\n")]
    assert "surface_derivations" in predicate
    assert "d.child_surface_id" in predicate
    assert "j.mission_id" in predicate


def test_the_panel_image_carries_what_the_catalog_symlinks_point_at():
    """The frozen catalog is three symlinks into the phase-0 archive, and COPY
    keeps a symlink as a symlink. The image had a dangling link where the
    catalog should be, so the panel translated no scroll name and every
    mission-scoped page in the deployment read zero -- with the source correct
    and the tests green."""
    containerfile = (ROOT / "containers/images/Containerfile.panel").read_text()
    # Every COPY source: a token that is neither the instruction, a line
    # continuation, nor the destination inside the image.
    copied = [Path(token) for line in containerfile.replace("\\\n", " ").splitlines()
              if line.strip().startswith("COPY")
              for token in line.split()[1:-1] if not token.startswith("/")]
    for link in sorted((ROOT / "workspace/catalog").iterdir()):
        if not link.is_symlink():
            continue
        target = Path(link.readlink())
        assert not target.is_absolute(), f"{link.name} points outside the repository"
        inside = (link.parent / target).resolve().relative_to(ROOT)
        assert any(inside == source or source in inside.parents for source in copied), \
            f"nothing in Containerfile.panel copies {inside}, which {link.name} needs"
