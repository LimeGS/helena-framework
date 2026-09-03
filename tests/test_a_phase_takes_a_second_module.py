"""Plugging a new way of doing a phase, without editing the queue.

The claim this framework makes is that it orchestrates other people's tools.
That claim is only true if adding a tool is adding a tool -- a row in a table --
rather than a patch to the code that builds command lines. It was true for P4
and P5, which have had lanes for a while, and false for every other phase: a
second assembler for P8 meant editing `command_for`.

These drive a synthetic lane end to end. Nothing is mocked and no real tool is
invented: the point is that the queue routes to whatever the lane names, and
that it refuses what it does not recognise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import job_store  # noqa: E402
from job_store import JobRejected  # noqa: E402


@pytest.fixture
def registered():
    """A lane that exists only for this test, removed afterwards."""
    job_store.register_lane("P8", "test-assembler", {
        "name": "a second assembler",
        "runner": "framework/stages/05-reconstruction/scripts/evaluate_r6_direct_geometry.py",
        "required": ("scroll", "out_path"),
        "flags": {"scroll": "--scroll", "out_path": "--out",
                  "work_dir": "--work", "subsample": "--subsample"},
        "defaults": {"work_dir": lambda out: f"{out}/scratch"},
        "note": "registered by a test",
    })
    yield "test-assembler"
    job_store.PHASE_LANES["P8"].pop("test-assembler", None)


def job(lane: str | None = None, **parameters):
    if lane:
        parameters["lane"] = lane
    return {"phase": "P8", "parameters": parameters}


# --------------------------------------------------------------------------
# It routes
# --------------------------------------------------------------------------

def test_a_new_lane_builds_its_own_command(registered):
    """No branch in `command_for` knows this lane exists."""
    argv = job_store.command_for(
        job(registered, scroll="PHerc0139", out_path="/out/page.json"),
        runner="ignored", output_dir="/run/p8-1")
    assert "evaluate_r6_direct_geometry.py" in argv[1]
    assert argv[2:] == ["--scroll", "PHerc0139", "--out", "/out/page.json",
                        "--work", "/run/p8-1/scratch"]


def test_the_worker_starts_what_the_lane_names(registered):
    """The runner comes from the lane, not from a per-phase table, so two lanes
    of one phase can be two different programs."""
    import importlib

    worker = importlib.import_module("ink_worker")
    resolved = worker.runner_for(
        job(registered, scroll="PHerc0139", out_path="/out/page.json"))
    assert resolved.name == "evaluate_r6_direct_geometry.py"


def test_the_form_offers_it_without_being_told(registered):
    """The parameter schema the panel draws its form from reads the same table,
    so a lane appears in the interface the moment it is registered."""
    schema = job_store.phase_parameter_schema("P8")
    assert registered in [lane["id"] for lane in schema["lanes"]]


# --------------------------------------------------------------------------
# It refuses
# --------------------------------------------------------------------------

def test_an_unknown_lane_is_refused_rather_than_defaulted():
    """A job that asked for one assembler and silently got another produces a
    result nobody can interpret afterwards."""
    with pytest.raises(JobRejected) as refused:
        job_store.command_for(job("no-such-lane", scroll="PHerc0139"),
                              runner="x", output_dir="/run/x")
    assert "no-such-lane" in str(refused.value)


def test_a_missing_required_parameter_is_refused(registered):
    with pytest.raises(JobRejected) as refused:
        job_store.command_for(job(registered, scroll="PHerc0139"),
                              runner="x", output_dir="/run/x")
    assert "out_path" in str(refused.value)


def test_a_lane_with_no_runner_is_refused_rather_than_run():
    """mesh-relations shipped with flags that its own runner's argparse does
    not accept -- a real file at the declared path, so the job was claimed and
    only failed once a worker actually ran it. `not_wired` makes that refusal
    happen here instead, before a lease is ever spent."""
    with pytest.raises(job_store.JobRejected) as refused:
        job_store.command_for(
            {"phase": "P8", "parameters": {"lane": "mesh-relations",
                                           "scroll": "PHerc0139",
                                           "out_path": "/run/x"}},
            runner="x", output_dir="/run/x")
    assert "mesh-relations" in str(refused.value)
    assert "not runnable" in str(refused.value)


def test_a_lane_cannot_shadow_another_silently(registered):
    with pytest.raises(ValueError):
        job_store.register_lane("P8", registered, {"runner": "elsewhere.py"})


# --------------------------------------------------------------------------
# What was there before still runs
# --------------------------------------------------------------------------

def test_every_phase_that_could_be_queued_still_has_a_lane():
    for phase in job_store.PHASE_RUNNERS:
        assert job_store.PHASE_LANES.get(phase), f"{phase} lost its lane"


def test_the_default_lane_is_the_one_that_always_ran():
    """A job that names no lane must behave exactly as it did before lanes
    existed, or this refactor quietly re-routed live work."""
    argv = job_store.command_for(
        {"phase": "P8", "parameters": {"scroll": "PHerc0139",
                                       "out_path": "/out/atlas.json"}},
        runner="framework/vendored/pherc0139-column-atlas-gh/scripts/wrap_order.py",
        output_dir="/run/p8-2")
    assert "wrap_order.py" in argv[1]
    assert "--scroll" in argv and "PHerc0139" in argv
