"""A lane whose runtime is not the worker's says so, in the table.

Three of the methods this platform now registers cannot run in the worker
that claims their phase:

    9 um ink   P5   claimed by helena-ink-0   needs helena-ink-9um
    lasagna    P3   claimed by helena-ink-0   needs helena-villa-python
    spiral     P1   claimed by helena-segment needs helena-villa-python

Two phases, two workers, three lanes, and not one of them runs the image it
needs. Solving that three times, in three places, is how a platform ends up
with three different answers to one question -- so the answer goes where the
lanes already are: a lane declares its image, or declares nothing and runs
where it always did.

This is deliberately only the declaration. *How* a lane in a different image
gets executed -- a sibling container, a second worker that carries that
runtime, something else -- is a separate decision with its own tradeoffs, and
nothing here forecloses it. What it does is make the fact discoverable from
the registry instead of from whichever worker happened to fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import (  # noqa: E402
    INK_ADAPTERS, JobRejected, lane_image, runtime_image_for,
)

NINE_UM = "framework/stages/03-ink/scripts/run_ink_9um.py"


# -- the declaration -------------------------------------------------------

def test_the_9um_adapter_declares_the_image_it_needs():
    """Its runner is koine_machines, which is not in the ink worker: that
    image pins torch 2.5.1 and this lane's lock pins 2.10.0."""
    assert INK_ADAPTERS[NINE_UM]["image"] == "helena-ink-9um"


def test_an_adapter_that_runs_where_it_is_claimed_declares_nothing():
    """Absence is the answer for every lane that was already fine, and it has
    to stay the answer -- adding a field to all of them would be inventing a
    fact about lanes nobody had to think about."""
    generic = "framework/stages/03-ink/scripts/run_ink.py"
    assert "image" not in INK_ADAPTERS[generic]


# -- reading it ------------------------------------------------------------

def test_the_image_of_a_lane_that_needs_one_is_reported():
    assert lane_image(INK_ADAPTERS[NINE_UM]) == "helena-ink-9um"


def test_the_image_of_a_lane_that_does_not_is_none():
    assert lane_image({"receipt": "X.json"}) is None


def test_a_job_on_an_ordinary_lane_needs_no_image():
    job = {"phase": "P5", "profile_id": "timesformer-gp-scroll1-screening@1.1.0"}
    assert runtime_image_for(job) is None


def test_a_job_on_the_9um_lane_reports_its_image():
    job = {"phase": "P5", "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0"}
    assert runtime_image_for(job) == "helena-ink-9um"


def test_a_job_whose_lane_cannot_be_resolved_does_not_raise_here():
    """This answers "which image", not "is this job valid". The queue already
    refuses an unknown lane by name, with its own reason, and a second refusal
    here would report the wrong one first."""
    assert runtime_image_for({"phase": "P5", "profile_id": "no-such-lane@9.9.9"}) is None


# -- what it does not do ---------------------------------------------------

def test_declaring_an_image_does_not_change_the_command():
    """The argv is what runs inside whichever runtime is chosen. Wrapping it
    is the execution decision this deliberately leaves open."""
    from job_store import command_for

    job = {"phase": "P5", "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0",
           "sample_id": "PHerc0139",
           "parameters": {"tiff_dir": "/layers", "checkpoint": "/models/step.pth",
                          "source_pixel_um": 2.399}}
    argv = command_for(job, runner=NINE_UM, output_dir="/runs/p5")
    assert argv[0] == "python3"
    assert "docker" not in argv, (
        "command_for started deciding how to execute, which is not its job")


# -- and what the worker does with it --------------------------------------

def test_a_worker_that_cannot_reach_the_image_says_which_one(monkeypatch):
    """Before the lease is claimed and the render is paid for.

    Without this the worker runs the lane's argv in its own runtime, where
    the runner does not exist, and the job fails on an import several minutes
    in -- reported as whatever the traceback happened to say. The lane
    declared the image; refusing by name is what makes that declaration worth
    having.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"phase": "P5", "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0"}
    monkeypatch.setattr(ink_worker, "RUNTIME_IMAGE", "helena-worker-gpu")

    with pytest.raises(RuntimeError, match="helena-ink-9um"):
        ink_worker.require_runtime(job)


def test_a_worker_already_in_that_image_proceeds(monkeypatch):
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"phase": "P5", "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0"}
    monkeypatch.setattr(ink_worker, "RUNTIME_IMAGE", "helena-ink-9um")
    ink_worker.require_runtime(job)  # must not raise


def test_an_ordinary_lane_runs_wherever_it_is_claimed(monkeypatch):
    """A worker that does not know its own image must not start refusing work
    it has always done."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"phase": "P5", "profile_id": "timesformer-gp-scroll1-screening@1.1.0"}
    monkeypatch.setattr(ink_worker, "RUNTIME_IMAGE", None)
    ink_worker.require_runtime(job)  # must not raise


def test_an_unknown_runtime_does_not_block_a_lane_that_needs_one(monkeypatch):
    """A worker with no HELENA_RUNTIME_IMAGE set cannot prove it is in the
    wrong image, and guessing would refuse every 9 um job on a host that was
    configured correctly but not labelled."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    job = {"phase": "P5", "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0"}
    monkeypatch.setattr(ink_worker, "RUNTIME_IMAGE", None)
    ink_worker.require_runtime(job)  # must not raise


def test_naming_the_composed_image_as_the_runtime_is_refused_at_startup():
    """The quiet half of a confusion one suffix wide.

    HELENA_RUNTIME_IMAGE names the lane image a worker carries, not the
    composed image it runs as. Backwards, the worker starts, refuses every job
    for that lane by name, and looks exactly like a worker with nothing to do.

    Its loud sibling -- HELENA_INK_IMAGE pointed at the lane image -- ran for
    27 hours reporting `can't open file '.../ink_worker.py'`, which is true and
    says nothing about which of the two variables is wrong. The lane image
    answers that one itself now.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "framework/stages/03-ink/fleet"))
    import ink_worker

    for lane in ("helena-ink-9um", "helena-villa-python"):
        assert ink_worker.misnamed_runtime(lane) is None, (
            f"{lane} is a lane image and is exactly what belongs here")
        refusal = ink_worker.misnamed_runtime(f"{lane}-worker")
        assert refusal and lane in refusal, (
            f"{lane}-worker is the composed image and must be refused by name")

    # The ordinary worker is not a lane, so its own name is not the mistake.
    assert ink_worker.misnamed_runtime("helena-worker-gpu") is None
    assert ink_worker.misnamed_runtime(None) is None


def test_the_lane_image_explains_what_it_is_not():
    """A missing file is a true error and a useless one. The guard turns 27
    hours of crash-loop into one line naming both variables."""
    import subprocess
    from pathlib import Path

    guard = (Path(__file__).resolve().parents[1]
             / "containers/images/lane-entrypoint.sh")
    refused = subprocess.run(  # noqa: S603 - a script in this repository
        ["sh", str(guard), "python3", "/workspace/campaign-x/framework/"
         "stages/03-ink/fleet/ink_worker.py", "--host-id", "gpu-1"],
        capture_output=True, text=True, check=False)
    assert refused.returncode == 3
    assert "helena-worker-gpu" in refused.stderr
    assert "HELENA_INK_IMAGE" in refused.stderr

    # And it does not stand in the way of using the lane image as a lane image.
    allowed = subprocess.run(  # noqa: S603 - a script in this repository
        ["sh", str(guard), "echo", "the lane runs"],
        capture_output=True, text=True, check=False)
    assert allowed.returncode == 0
    assert allowed.stdout.strip() == "the lane runs"
