"""A P5 job may name a surface volume that already exists.

The 9 um lane takes either a P4 layer stack, which it pools, or a ready ~9.6 um
isotropic OME-Zarr, which it reads as it is. The queue could only express the
first: P5's contract was "exactly one of tiff_dir or layer_stack", and
command_for passed --tiff-dir unconditionally.

That is why the public control ran its ink step in a local subprocess. Its
input is a surface volume in the open-data bucket -- the whole point of it
being reproducible without a credential -- and there was no way to say so to
the queue. So the control proved the tooling and could say nothing about
Helena, which is the half a reviewer is being asked to trust.

Three ways of naming one input, exactly one of them given. What each lane
accepts is still the lane's own business: one handed a surface volume it has no
flag for is refused by name rather than having the parameter quietly dropped
and being pointed at nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import EXACTLY_ONE_OF, JobRejected, command_for, validate_parameters  # noqa: E402

NINE_UM = "ink-9um-hybrid-3d2d-screening@1.0.0"
PUBLIC_VOLUME = (
    "s3://vesuvius-challenge-open-data/PHerc0139/segments/"
    "20260112000000-w043_2026011217/surface-volumes/"
    "9.362um-1.2m-113keV-volume-20250728140407.zarr")


def a_job(**parameters):
    return {"phase": "P5", "sample_id": "PHerc0139", "profile_id": NINE_UM,
            "job_id": "p5-test", "parameters": parameters}


# -- what the queue will accept ---------------------------------------------


def test_a_surface_volume_is_a_third_way_to_name_the_input() -> None:
    names = {name for rule in EXACTLY_ONE_OF["P5"] for name in rule["names"]}

    assert names == {"tiff_dir", "layer_stack", "surface_volume"}


def test_naming_a_ready_volume_alone_is_accepted() -> None:
    clean = validate_parameters(
        {"surface_volume": PUBLIC_VOLUME, "checkpoint": "/models/c.pth"}, "P5")

    assert clean["surface_volume"] == PUBLIC_VOLUME


def test_naming_a_volume_and_a_stack_is_still_refused() -> None:
    """A lane handed two inputs has no way to say which one its map came from."""
    with pytest.raises(JobRejected) as refused:
        validate_parameters({"surface_volume": PUBLIC_VOLUME,
                             "tiff_dir": "/layers",
                             "checkpoint": "/models/c.pth"}, "P5")

    assert "exactly one" in str(refused.value)


def test_naming_none_of_the_three_is_still_refused() -> None:
    with pytest.raises(JobRejected):
        validate_parameters({"checkpoint": "/models/c.pth"}, "P5")


# -- what reaches the adapter ------------------------------------------------


def test_the_command_carries_the_volume_and_no_empty_tiff_dir() -> None:
    """--tiff-dir was passed unconditionally, and the adapter refuses a run
    that names both an input to pool and one already pooled."""
    argv = command_for(a_job(surface_volume=PUBLIC_VOLUME,
                             checkpoint="/models/c.pth"),
                       runner="/r/run_ink_9um.py", output_dir="/out")

    assert "--surface-volume" in argv
    assert argv[argv.index("--surface-volume") + 1] == PUBLIC_VOLUME
    assert "--tiff-dir" not in argv


def test_a_stack_still_reaches_the_adapter_as_it_always_did() -> None:
    argv = command_for(a_job(tiff_dir="/layers", checkpoint="/models/c.pth",
                             source_pixel_um=9.362),
                       runner="/r/run_ink_9um.py", output_dir="/out")

    assert argv[argv.index("--tiff-dir") + 1] == "/layers"
    assert "--surface-volume" not in argv


def test_a_lane_with_no_such_flag_refuses_rather_than_dropping_it() -> None:
    """The canonical 2 um lane cannot read a surface volume. Passing one and
    silently ignoring it would point the detector at nothing -- or at whatever a
    stale directory holds, which is the failure this refusal exists to prevent.
    """
    job = a_job(surface_volume=PUBLIC_VOLUME, checkpoint="/models/c.pth")
    job["profile_id"] = "ink-canonical-2um-screening@1.0.0"

    with pytest.raises(JobRejected) as refused:
        command_for(job, runner="/r/run_ink.py", output_dir="/out")

    assert "surface_volume" in str(refused.value)
