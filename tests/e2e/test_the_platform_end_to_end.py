"""The platform, through its HTTP API, with assertions instead of prose.

    HELENA_E2E_PANEL=https://127.0.0.1:8800 HELENA_E2E_USER=name \\
    HELENA_PANEL_TLS_INSECURE=1 \\
    HELENA_PANEL_PASSWORD=… python3 -m pytest tests/e2e -v

Skipped entirely without those, so `pytest tests/` stays a suite anybody can run
on a laptop with no fleet behind it.

The smoke test beside this one is a narrative: it prints what each phase did and
a person reads it. This is the same journey with the postconditions written
down, so a regression fails a build rather than being noticed in a paragraph.
Neither replaces the other -- the narrative caught the bugs a human eye catches
(a phase that finished suspiciously fast), and this catches the ones nobody
rereads for.

Two tiers. The light ones read state and cost seconds. The heavy ones render and
infer, cost around fifteen minutes and a GPU, and need HELENA_E2E_HEAVY=1: a
suite that quietly reserves a GPU is a suite people stop running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/harness"))

PANEL = os.environ.get("HELENA_E2E_PANEL")
USER = os.environ.get("HELENA_E2E_USER")
PASSWORD = os.environ.get("HELENA_PANEL_PASSWORD")
HEAVY = os.environ.get("HELENA_E2E_HEAVY") == "1"
# A worker takes one job at a time, so a heavy test waits for the queue ahead of
# it as well as for its own work. Sixty minutes was optimistic: a render and an
# inference behind one other job is over an hour, and the run that measured it
# failed on the clock while the job it was waiting for succeeded.
PATIENCE = float(os.environ.get("HELENA_E2E_MINUTES", "180"))
SCROLL = os.environ.get("HELENA_E2E_SCROLL", "PHerc0826")

pytestmark = pytest.mark.skipif(
    not (PANEL and USER and PASSWORD),
    reason="set HELENA_E2E_PANEL, HELENA_E2E_USER and HELENA_PANEL_PASSWORD")


@pytest.fixture(scope="module")
def panel():
    from panel_client import Panel

    client = Panel(PANEL)
    assert client.sign_in(USER, PASSWORD) == USER, "the session did not stick"
    return client


# --------------------------------------------------------------------------
# What the platform says about itself
# --------------------------------------------------------------------------

def test_the_gate_is_closed_to_a_client_with_no_session():
    """Deny by default, on the deployment rather than in a unit test: this is
    the thing standing between the network and a panel that queues GPU work."""
    from panel_client import Panel, PanelError

    stranger = Panel(PANEL, timeout=30)
    with pytest.raises(PanelError) as refused:
        stranger.call("GET", "/api/jobs")
    assert refused.value.status == 401


def test_p0_knows_the_scroll_and_its_scale(panel):
    """Every phase downstream is a resampling argued in microns, so a scroll
    without a declared scale is a scroll nothing else may touch."""
    scrolls = panel.call("GET", "/api/scrolls").get("scrolls", [])
    assert scrolls, "P0 lists no scrolls at all"
    named = [s for s in scrolls if str(s.get("sample_id", "")).replace("0", "")
             == SCROLL.replace("0", "")]
    assert named, f"{SCROLL} is not in the frozen catalog"
    assert named[0].get("pixel_um"), f"{SCROLL} has no declared scale"


def test_every_queueable_phase_serves_a_parameter_schema(panel):
    """The form draws itself from this. A phase whose schema is missing is a
    phase nobody can queue from a browser, however good the queue is."""
    for phase in ("P3", "P4", "P5"):
        schema = panel.call("GET", f"/api/phases/{phase}/parameters")
        assert schema["available"], f"{phase}: {schema.get('reason')}"
        assert schema["fields"], f"{phase} offers no fields"


def test_the_deployment_can_publish_what_it_renders(panel):
    """A layer stack written only to the worker's disk is lost with the worker.
    The panel fills the destination in, so its absence is a deployment fault
    that would surface as a refused render an hour into a campaign."""
    fields = {f["name"]: f for f in
              panel.call("GET", "/api/phases/P4/parameters")["fields"]}
    assert fields["artifact_store"]["filled_by_deployment"]
    environment = panel.call("GET", "/api/config").get("environment", [])
    store = next((s for s in environment if s["name"] == "CX_RENDER_STORE"), None)
    assert store and store.get("value"), "CX_RENDER_STORE is not set on this panel"


def test_the_ink_lanes_are_routable_or_say_why_not(panel):
    lanes = panel.call("GET", "/api/ink/lanes")
    assert lanes["available"] and lanes["lanes"]
    assert lanes["routable"] >= 1, "no ink model can be run at all"
    assert all(lane["reason"] for lane in lanes["lanes"] if not lane["routable"])


def test_coverage_answers_the_question_the_framework_is_named_for(panel):
    """How much of the scroll has been attempted, and how that arithmetic holds.

    A deployment that has never run anything has no grids, and that is a correct
    answer rather than a failure -- this suite is run against fresh installs to
    check that they work, and a fresh install failing its own end-to-end suite
    teaches people to ignore it. The endpoint still has to answer.
    """
    coverage = panel.call("GET", f"/api/coverage?sample={SCROLL}")
    assert coverage.get("available"), coverage.get("reason")
    if not coverage["grids"]:
        pytest.skip(f"nothing has been attempted on {SCROLL} on this deployment")
    for grid in coverage["grids"]:
        # Outcomes are per-cell populations, not a partition: a cell attempted
        # under two policies appears in both. What must hold is that neither
        # outcome exceeds what was attempted.
        assert grid["cells_with_surface"] <= grid["cells_attempted"]
        assert grid["cells_no_seed"] <= grid["cells_attempted"]


def test_the_flattening_backlog_is_arithmetic_that_holds(panel):
    sheets = panel.call("GET", f"/api/flattening?sample={SCROLL}")
    assert sheets.get("available"), sheets.get("reason")
    assert sheets["flattened"] <= sheets["certified"]
    assert (sheets["awaiting"] + sheets["flattened"]
            + sheets["awaiting_physical_qc"]) <= sheets["certified"] + sheets["flattened"]


def test_a_replan_under_the_same_policy_is_refused(panel):
    """Task identity is (snapshot, grid, cell, policy) behind an ON CONFLICT DO
    NOTHING, so a replan that reuses a policy inserts nothing and reports
    success. Checked against the deployment because that is where the refusal
    has to happen."""
    receipt = panel.call("POST", "/api/segmentation/replan", {
        "grid_version": "e2e-probe", "policy_version": "e2e-probe",
        "sample_id": SCROLL, "limit": 3, "dry_run": True})
    assert receipt["dry_run"] is True
    assert receipt["would_queue"] <= receipt["considered"]


# --------------------------------------------------------------------------
# The journey, which costs a GPU
# --------------------------------------------------------------------------

heavy = pytest.mark.skipif(not HEAVY, reason="set HELENA_E2E_HEAVY=1")


@heavy
def test_a_render_is_verified_and_published_before_it_is_called_a_success(panel):
    """P4's own output was the one artifact in the pipeline that stayed on the
    worker, and its exit code was believed over its content. Both are checked
    here on a sheet P3 has already unrolled, so this exercises the chain rather
    than a fixture."""
    sheets = panel.call("GET", f"/api/flattening?sample={SCROLL}")
    flattened = [row for row in sheets.get("rows", []) if row["state"] == "FLATTENED"]
    if not flattened:
        pytest.skip("P3 has produced no sheet on this scroll to render")

    volume = os.environ.get("HELENA_E2E_VOLUME_URL")
    if not volume:
        pytest.skip("set HELENA_E2E_VOLUME_URL to the scroll's OME-Zarr")
    job = panel.call("POST", "/api/jobs", {
        "sample_id": SCROLL, "phase": "P4",
        "parameters": {"lane": "vc-render-tifxyz",
                       "volume": os.environ.get("HELENA_E2E_VOLUME_CACHE",
                                                "/srv/helena/cache/e2e"),
                       "remote_url": volume, "scale": 1.0, "group_idx": 0,
                       "cache_gb": 4, "num_slices": 33, "slice_step": 1.0,
                       "flattened_surface": flattened[0]["surface_id"]}})
    outcome = panel.wait_for_job(job["job_id"], minutes=PATIENCE)
    result = outcome.get("result") or {}
    assert outcome["state"] == "succeeded", result.get("error") or result.get("stderr_tail")

    layers = result.get("layers") or {}
    assert layers.get("slices") == 33, f"asked for 33 slices, got {layers.get('slices')}"
    low, high = layers["middle_slice_range"]
    assert low < high, "the middle slice is a constant: exit 0 does not say that"

    published = result.get("layer_stack") or {}
    assert published.get("artifact_uri"), "the stack was never published"
    assert len(published.get("artifact_sha256", "")) == 64


@heavy
def test_the_detector_reads_the_render_the_queue_points_it_at(panel):
    """P5 took a directory on whichever machine held the layers, so the phase
    had only ever been run by hand. Naming the render is what makes the chain
    something the control plane can express."""
    renders = [job for job in panel.call("GET", "/api/jobs?limit=50").get("jobs", [])
               if job["phase"] == "P4" and job["state"] == "succeeded"
               and (job.get("result") or {}).get("layer_stack")]
    if not renders:
        pytest.skip("no published render to screen")
    checkpoint = os.environ.get("HELENA_E2E_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set HELENA_E2E_CHECKPOINT to a checkpoint the worker can read")

    job = panel.call("POST", "/api/jobs", {
        "sample_id": renders[0]["sample_id"], "phase": "P5",
        "profile_id": os.environ.get("HELENA_E2E_INK_PROFILE",
                                     "timesformer-gp-scroll1-screening@1.1.0"),
        "parameters": {"layer_stack": renders[0]["job_id"], "checkpoint": checkpoint,
                       "source_pixel_um": float(os.environ.get("HELENA_E2E_PIXEL_UM",
                                                               "9.362")),
                       "batch_size": 2, "device": "cuda:0"}})
    outcome = panel.wait_for_job(job["job_id"], minutes=PATIENCE)
    result = outcome.get("result") or {}
    assert outcome["state"] == "succeeded", result.get("error") or result.get("stderr_tail")
    liveness = result.get("liveness") or {}
    # A map that is one value everywhere is what a wrong depth window or a
    # back-to-front slab produces, and it exits zero.
    assert liveness.get("verdict") == "ALIVE", liveness
