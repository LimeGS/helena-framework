"""The ink worker's decisions, without a GPU, a queue or a subprocess.

What is worth holding here is what the worker decides before and after the
process it starts: which runner a lane gets, what the renderer is allowed to
inherit, whether a stack is worth calling a success, where it goes afterwards
and what is cleaned up. Every one of those was wrong at least once this week,
and none of them needs a card to test.

The subprocess itself is deliberately not mocked. A test that replaces
vc_render_tifxyz with a stub proves the stub was called, and this pipeline's
failures have all been in the arguments, the inputs and the outputs -- never in
whether Python can start a process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The stack fixtures are real TIFFs, because what is verified is the content of
# a render and a stub would verify the stub. Skipped where the worker's own
# dependency is not installed, like the panel image.
pytest.importorskip("tifffile", reason="the ink worker's runtime carries this")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402
from ink_worker import RenderNotUsable  # noqa: E402


# --------------------------------------------------------------------------
# Which runner, and what it may inherit
# --------------------------------------------------------------------------

def test_the_lane_profile_chooses_the_runner():
    """The queue built one argv for every ink job while the profiles named their
    own adapter, so every TimeSformer lane ran the ResNet runner's flags."""
    timesformer = ink_worker.runner_for(
        {"phase": "P5", "profile_id": "timesformer-gp-scroll1-screening@1.1.0"})
    canonical = ink_worker.runner_for(
        {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0"})
    assert timesformer.name == "run_ink_timesformer.py"
    assert canonical.name == "run_ink_canonical2um.py"


def test_a_phase_with_no_runner_is_refused_rather_than_guessed():
    """The refusal now comes from the lane table, which is the thing that knows
    what a phase can run. JobRejected is the queue's own word for "this request
    does not describe a runnable job"."""
    from job_store import JobRejected

    with pytest.raises((RuntimeError, JobRejected)):
        ink_worker.runner_for({"phase": "P404"})


def test_the_renderer_does_not_inherit_the_private_bucket(monkeypatch):
    """The CT is public and served anonymously. Signing that request with keys
    for another bucket returns 400 one second into a render, on a URL that
    answers 200 to curl."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    assert "AWS_ACCESS_KEY_ID" not in ink_worker.runner_environment({"phase": "P4"})
    # P5 keeps them: its checkpoint may come from the private bucket.
    assert "AWS_ACCESS_KEY_ID" in ink_worker.runner_environment(
        {"phase": "P5", "profile_id": "x", "parameters": {}})


def test_a_vendored_runner_gets_its_architecture_on_the_path():
    """It imports the upstream model from beside itself, which is where that
    code sits in the recipe's own directory and not where it sits here."""
    environment = ink_worker.runner_environment(
        {"phase": "P5", "profile_id": "ink-canonical-2um-screening@1.1.0",
         "parameters": {"upstream_dir": "/models/canonical"}})
    assert environment["PYTHONPATH"].startswith("/models/canonical:")


# --------------------------------------------------------------------------
# Whether the render is worth calling a success
# --------------------------------------------------------------------------

def stack(directory: Path, slices: int = 5, *, constant: bool = False) -> Path:
    import numpy
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(slices):
        plane = (numpy.zeros((4, 4), dtype=numpy.uint16) if constant
                 else numpy.arange(16, dtype=numpy.uint16).reshape(4, 4) + index)
        tifffile.imwrite(directory / f"{index:02d}.tif", plane)
    return directory


def test_a_stack_of_constants_is_refused(tmp_path):
    with pytest.raises(RenderNotUsable) as refused:
        ink_worker.verify_layer_stack(stack(tmp_path / "l", constant=True), {})
    assert "no signal" in str(refused.value)


def test_the_slice_count_must_be_the_one_that_was_asked_for(tmp_path):
    with pytest.raises(RenderNotUsable):
        ink_worker.verify_layer_stack(stack(tmp_path / "l", 5), {"num_slices": 33})


def test_a_real_stack_is_described(tmp_path):
    described = ink_worker.verify_layer_stack(stack(tmp_path / "l", 7), {"num_slices": 7})
    assert described["slices"] == 7 and described["bytes"] > 0
    assert described["middle_slice_range"][0] < described["middle_slice_range"][1]


# --------------------------------------------------------------------------
# What happens to it afterwards
# --------------------------------------------------------------------------

def test_publishing_verifies_what_arrives(tmp_path):
    published = ink_worker.publish_layer_stack(
        stack(tmp_path / "l", 3), store_spec=str(tmp_path / "store"),
        sample_id="PHerc826", job_id="p4-x")
    assert published["files"] == 3 and len(published["artifact_sha256"]) == 64
    fetched = ink_worker.fetch_artifact_set(published["artifact_uri"], tmp_path / "back")
    assert len(fetched["files"]) == 3

    # An artifact that arrives with a different digest is a different artifact.
    manifest = json.loads((Path(published["artifact_uri"]) / "ARTIFACT_SET.json").read_text())
    manifest["files"]["01.tif"]["sha256"] = "0" * 64
    (Path(published["artifact_uri"]) / "ARTIFACT_SET.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError) as refused:
        ink_worker.fetch_artifact_set(published["artifact_uri"], tmp_path / "again")
    assert "not that artifact" in str(refused.value)


# --------------------------------------------------------------------------
# The window the detector is given
# --------------------------------------------------------------------------

class Store:
    def __init__(self, render):
        self._render = render

    def job(self, job_id):
        return self._render


def test_a_map_is_not_computed_on_a_failed_render(tmp_path):
    store = Store({"job_id": "p4-1", "phase": "P4", "state": "failed", "result": {}})
    with pytest.raises(RuntimeError) as refused:
        ink_worker.resolve_layer_stack(
            store, {"parameters": {"layer_stack": "p4-1"}}, tmp_path / "stack")
    assert "failed" in str(refused.value)


def test_the_depth_window_is_centred_on_the_stack_when_nothing_was_asked(tmp_path):
    """A lane's depth centres are written for the stack depth its author had.
    The GP Scroll1 lane says 25, 32 and 39, which are positions in a 62-layer
    volume, and on a 33-slice render they fall off the end."""
    job = {"profile_id": "timesformer-gp-scroll1-screening@1.1.0",
           "parameters": {"source_pixel_um": 9.362, "source_slice_um": 9.362}}
    ink_worker.fit_depth_to_stack(job, stack(tmp_path / "l", 33))
    assert job["parameters"]["depth_centers"] == "16"


def test_a_stack_too_shallow_for_the_model_is_refused(tmp_path):
    """Ten slices for a window that needs twenty-two is not a centring problem,
    and centring it silently would hand the model padding."""
    job = {"profile_id": "timesformer-gp-scroll1-screening@1.1.0",
           "parameters": {"source_pixel_um": 9.362, "source_slice_um": 9.362}}
    with pytest.raises(RuntimeError) as refused:
        ink_worker.fit_depth_to_stack(job, stack(tmp_path / "l", 10))
    assert "too shallow" in str(refused.value)


def test_an_integer_flag_is_never_given_a_half(tmp_path):
    """The canonical lane's --depth-center is an integer, and a 62-frame window
    fits a 62-slice stack at 30.5 and nowhere else. Rounding it would hand the
    runner a window half a slice past the end."""
    job = {"profile_id": "ink-canonical-2um-screening@1.1.0",
           "parameters": {"source_pixel_um": 2.399, "source_slice_um": 2.399}}
    ink_worker.fit_depth_to_stack(job, stack(tmp_path / "l", 63))
    assert job["parameters"]["depth_center"] == 31
    assert "depth_centers" not in job["parameters"]


def test_every_phase_is_given_a_directory_that_exists(tmp_path):
    """Some runners create their own output directory and some write a file into
    it. vet_map read the map, screened it, found the shapes and then died on
    FileNotFoundError for the verdict, because nothing had made the directory."""
    source = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    run_job = source[source.index("def run_job("):]
    run_job = run_job[:run_job.index("\ndef ")]
    assert "output.mkdir(parents=True, exist_ok=True)" in run_job
    # Before the command is built, not after: a builder that reads the directory
    # would see it missing.
    assert run_job.index("output.mkdir") < run_job.index("command_for(")
