"""P1 has two backends now, and only one of them is a queued job.

The seeded grower is planned: the fleet's bootstrap decides which cells are
uncovered, applies a candidate policy and picks a seed inside one. There is no
command line to build, which is why P1 was never in the queue.

A spiral fit decides nothing of the sort. It is told a scroll, a slab of it, a
scale and a winding direction, and fits every winding at once -- which is a job
with parameters, a runner and a runtime. So it goes through the queue like P4
and P5 do, and this checks the three things that would make it fail after being
claimed rather than before: the command, the runtime, and the parameters that
describe a fit nobody could interpret.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import (  # noqa: E402
    PHASE_LANES, PHASE_PARAMETERS, PHASE_REQUIRED, PHASE_RUNNERS, JobRejected,
    command_for, lane_image, phase_parameter_schema, runtime_image_for,
    validate_parameters,
)

PROFILE_ID = "spiral-fitter-v1@0.4.0"
# spiral-fitter-v1@0.4.1 joined the same lane as an A/B sibling, not a
# replacement -- see test_grow_track_patches_is_a_queued_p1_lane_too.py and
# tests/test_a_patches_enabled_fit_does_not_require_patches_nobody_has.py for
# what makes it a different profile rather than the same one renamed.
PROFILE_ID_0_4_1 = "spiral-fitter-v1@0.4.1"
RUNNER = PHASE_RUNNERS["P1"]
FIT = {"scroll_name": "PHerc0172", "dataset_path": "/artifacts/spiral/PHerc0172",
       "z_begin": 500, "z_end": 9000, "voxel_size_um": 7.91,
       "spiral_outward_sense": "CCW"}
# Where a fit publishes is the deployment's answer, so it reaches the queue as
# the server's own value rather than as part of the request.
PUBLISHES_TO = {"artifact_store": "/artifacts"}


def job(**overrides):
    parameters = validate_parameters(
        {**FIT, **PUBLISHES_TO, **overrides.pop("parameters", {})}, "P1",
        server_owned=PUBLISHES_TO)
    return {"job_id": "j-1", "sample_id": "PHerc0172", "mission_id": "m-1",
            "phase": "P1", "profile_id": PROFILE_ID,
            "parameters": parameters, **overrides}


def argv_of(**overrides) -> list[str]:
    return [str(token) for token in
            command_for(job(**overrides), runner=RUNNER, output_dir="/runs/p1-1")]


# -- the lane --------------------------------------------------------------

def test_p1_is_a_phase_with_a_runner_now():
    assert "P1" in PHASE_PARAMETERS and "P1" in PHASE_REQUIRED
    assert (ROOT / RUNNER).is_file(), "the lane names a runner that is not here"


def test_the_lane_declares_the_image_the_fitter_needs():
    """fit_spiral.py is Python with torch and zarr behind it. The worker that
    claims segmentation carries compiled binaries and cannot import any of it."""
    spec = PHASE_LANES["P1"]["spiral-fit"]
    assert lane_image(spec) == "helena-villa-python"
    assert runtime_image_for(job()) == "helena-villa-python"
    assert spec["gpu_required"] is True


def test_a_worker_in_the_wrong_runtime_refuses_before_the_lease(monkeypatch):
    """Without this the fit runs in the segment worker, dies on an import
    several minutes in, and is reported as whatever the traceback said."""
    import ink_worker

    monkeypatch.setattr(ink_worker, "RUNTIME_IMAGE", "helena-worker-cpp")
    with pytest.raises(RuntimeError, match="helena-villa-python"):
        ink_worker.require_runtime(job())

    monkeypatch.setattr(ink_worker, "RUNTIME_IMAGE", "helena-villa-python")
    ink_worker.require_runtime(job())  # must not raise


def test_the_lane_pins_the_frozen_profile_and_the_form_can_read_it():
    """The queue refuses a job whose profile is not on the lane's list, so a
    form with no way to learn the list can only send nothing and be refused."""
    schema = phase_parameter_schema("P1")
    lanes = {lane["id"]: lane for lane in schema["lanes"]}
    assert lanes["spiral-fit"]["profiles"] == [PROFILE_ID, PROFILE_ID_0_4_1]


def test_a_job_naming_another_profile_is_refused():
    with pytest.raises(JobRejected, match="accepts profiles"):
        command_for(job(profile_id="spiral-fitter-v1@0.1.0"),
                    runner=RUNNER, output_dir="/runs/p1-1")


def test_a_job_naming_the_patches_enabled_sibling_is_accepted():
    """0.4.1 shares the spiral-fit lane with 0.4.0; only the profile id
    differs on the command line -- the runner resolves everything else
    (config_overrides, inputs) from the frozen profile itself."""
    argv = argv_of(profile_id=PROFILE_ID_0_4_1)
    assert argv[argv.index("--profile-id") + 1] == PROFILE_ID_0_4_1


def test_a_job_can_name_where_grown_patches_are():
    """spiral-fitter-v1@0.4.1's own reason to exist: without this flag,
    unverified_patches has no path override and the fit runs exactly as
    0.4.0 would -- silently."""
    argv = argv_of(profile_id=PROFILE_ID_0_4_1,
                   parameters={"unverified_patches_dir": "unverified_patches"})
    assert argv[argv.index("--unverified-patches-dir") + 1] == "unverified_patches"


def test_a_verified_patches_directory_rides_the_same_channel():
    """Upstream's other patch role, named the same way: one directory under
    the dataset root, passed through as-is, and the profile -- not the job --
    decides whether it supervises the fit."""
    argv = command_for(job(parameters={"verified_patches_dir": "verified_patches"}),
                       runner=RUNNER, output_dir="/runs/p1-x")
    assert argv[argv.index("--verified-patches-dir") + 1] == "verified_patches"


def test_the_form_does_not_ask_for_what_the_server_already_owns():
    """artifact_store was required and server-owned at once: a caller who read
    the form's own required list and sent it got refused as smuggled, and one
    who left it out -- correctly -- had nowhere left to learn that from this
    table. It is filled_by_deployment, so it is never something to type in."""
    schema = phase_parameter_schema("P1")
    fields = {f["name"]: f for f in schema["fields"]}
    assert fields["artifact_store"]["filled_by_deployment"] is True
    assert fields["artifact_store"]["required"] is False


def test_supplying_artifact_store_from_the_request_is_still_refused():
    """The field being server-owned did not change -- only the form's claim
    that a caller must supply it."""
    complete = {**FIT, **PUBLISHES_TO}
    with pytest.raises(JobRejected, match="server's to decide"):
        validate_parameters(complete, "P1", server_owned=())


# -- the command -----------------------------------------------------------

def test_the_command_carries_the_whole_scroll_binding():
    argv = argv_of()
    for flag, value in (("--scroll-name", "PHerc0172"),
                        ("--dataset-path", "/artifacts/spiral/PHerc0172"),
                        ("--z-begin", "500"), ("--z-end", "9000"),
                        ("--voxel-um", "7.91"), ("--winding-sense", "CCW")):
        assert argv[argv.index(flag) + 1] == value


def test_the_scroll_and_the_mission_come_off_the_job_not_the_parameters():
    """Which scroll a fitted winding is filed under is not a knob a request may
    set, for the reason P2 and P3 read theirs off the job row."""
    assert "sample" not in PHASE_PARAMETERS["P1"]
    assert "mission_id" not in PHASE_PARAMETERS["P1"]
    argv = argv_of()
    assert argv[argv.index("--sample") + 1] == "PHerc0172"
    assert argv[argv.index("--mission-id") + 1] == "m-1"
    assert argv[argv.index("--requested-by-job-id") + 1] == "j-1"


def test_the_connection_string_never_travels_in_the_command():
    argv = argv_of()
    assert argv[argv.index("--db") + 1] == "postgres-env://CX_DB"


def test_a_job_with_no_sample_is_refused_rather_than_filed_nowhere():
    with pytest.raises(JobRejected, match="immutable sample_id"):
        command_for({**job(), "sample_id": ""}, runner=RUNNER, output_dir="/o")


def test_a_job_with_no_profile_is_refused():
    """The fitter's upstream settings come from the profile. A run that
    named none would record a configuration nobody chose."""
    with pytest.raises(JobRejected):
        command_for({**job(), "profile_id": None}, runner=RUNNER, output_dir="/o")


def test_a_dry_run_reaches_the_runner():
    """It preflights the dataset and writes spiral-scroll.json, then stops
    before the GPU, which is the only way to find out that a dataset is
    incomplete without paying."""
    assert "--dry-run" in argv_of(parameters={"dry_run": True})


# -- the parameters --------------------------------------------------------

@pytest.mark.parametrize("bad,expected", [
    ({"spiral_outward_sense": "clockwise"}, "CW or CCW"),
    ({"z_begin": 9000, "z_end": 500}, "above z_begin"),
    ({"voxel_size_um": 0}, "greater than zero"),
    ({"dataset_path": "relative/path"}, "absolute path"),
    ({"dataset_path": "/artifacts/../etc"}, "absolute path"),
])
def test_a_fit_nobody_could_interpret_is_refused_at_the_queue(bad, expected):
    with pytest.raises(JobRejected, match=expected):
        validate_parameters({**FIT, **PUBLISHES_TO, **bad}, "P1",
                            server_owned=PUBLISHES_TO)


@pytest.mark.parametrize("name", sorted(PHASE_REQUIRED["P1"]))
def test_five_of_the_six_have_no_default_at_all(name):
    """Defaulting them to Scroll 1's values would let a forgotten field fit the
    wrong scroll under the right name -- which produces windings, costs a
    GPU-day, and is not detectable from the output."""
    # Built from everything a complete P1 job carries, the server's own value
    # included: `artifact_store` is required and has no default either, and
    # dropping it only from the request half would leave it supplied.
    complete = {**FIT, **PUBLISHES_TO}
    incomplete = {key: value for key, value in complete.items() if key != name}
    with pytest.raises(JobRejected, match="missing required"):
        validate_parameters(incomplete, "P1", server_owned=PUBLISHES_TO)


def test_the_winding_sense_is_the_one_that_may_be_left_out():
    """It has a profile default, because CW is upstream's and a fit made with
    the wrong sense is visibly wrong rather than quietly so."""
    without = {key: value for key, value in FIT.items()
               if key != "spiral_outward_sense"}
    assert "spiral_outward_sense" not in validate_parameters(
        {**without, **PUBLISHES_TO}, "P1", server_owned=PUBLISHES_TO)
    assert "--winding-sense" not in argv_of(
        parameters={"spiral_outward_sense": None})


# --------------------------------------------------------------------------
# And through the panel, which is where a person actually meets it
# --------------------------------------------------------------------------

@pytest.fixture
def panel(tmp_path, monkeypatch):
    pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
    pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    # No control plane behind it, which is what a fresh deployment looks like.
    monkeypatch.setattr(module, "DSN", "")
    client = TestClient(module.app)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return module, client


def test_the_p1_page_offers_a_run_form_now(panel):
    """P1 was "observed rather than driven" for as long as its only backend was
    planned elsewhere. A phase with a runner has to say so, or the Run tab does
    not appear and the lane is unreachable from a browser."""
    module, client = panel
    assert "P1" in module.QUEUEABLE_PHASES


def test_the_form_is_served_with_every_field_and_the_frozen_profile(panel):
    _, client = panel
    body = client.get("/api/phases/P1/parameters").json()

    assert body["available"], body.get("reason")
    fields = {field["name"]: field for field in body["fields"]}
    assert set(fields) == set(PHASE_PARAMETERS["P1"])
    for name in ("scroll_name", "dataset_path", "z_begin", "z_end",
                 "voxel_size_um"):
        assert fields[name]["required"], f"{name} must be asked for"
        assert fields[name]["note"], "the guide draws itself from these"
    # Not required: it is the one with a default, and the default is upstream's.
    assert not fields["spiral_outward_sense"]["required"]
    lanes = {lane["id"]: lane for lane in body["lanes"]}
    assert lanes["spiral-fit"]["profiles"] == [PROFILE_ID, PROFILE_ID_0_4_1]


def test_the_segmentation_page_lists_it_and_says_where_it_runs(panel):
    """It used to be absent, then present and refused as "not implemented".
    Neither is true now, and a page that answers "can we run the method
    upstream recommends" with silence is the worse of the two."""
    module, _ = panel
    spiral = next(backend for backend in module.SEGMENTATION_BACKENDS
                  if backend["id"] == "spiral")

    assert spiral["runs_from"] == {"phase": "P1", "lane": "spiral-fit",
                                   "profile_id": PROFILE_ID}
    # Not adoptable *on that form*: it plans a seeded grow and a fit has no seed.
    assert spiral["adoptable"] is False


def test_the_grow_bootstrap_points_at_the_lane_instead_of_refusing_flatly(panel):
    """A 501 saying "not implemented" was right until it was not. The refusal
    now names the phase, the lane and the profile, because the run is possible
    and the person asking is one page away from it."""
    _, client = panel
    response = client.post("/api/segmentation/runs", json={
        "sample_id": "PHerc0172", "backend": "spiral",
        "planner": "deterministic-v2", "mission_id": "m-1"})

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["lane"] == "spiral-fit"
    assert detail["profile_id"] == PROFILE_ID
    assert "P1" in detail["how"]
