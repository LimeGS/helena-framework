"""grow-track-patches shares P1 with spiral-fit, and shares almost nothing else.

PHASE_REQUIRED was phase-wide because P1 had exactly one lane for as long as
it existed: the scroll binding spiral-fit needs. A second P1 lane that needs
none of it -- this one grows patches from an already-staged dataset, it does
not fit a scroll -- would have inherited that requirement anyway, and every
job naming it would have been refused at enqueue for missing scroll_name/
z_begin/z_end/voxel_size_um it has no argparse flag for at all. This file is
about the seam that makes that not true: validate_parameters reads a lane's
own `required` when the named lane is not the phase's historical one, and
otherwise falls back to PHASE_REQUIRED exactly as spiral-fit always has.

It is also about the two declared-facts tests this repository already has a
name for: gpu_required (test_a_lane_asks_for_a_card_only_if_it_uses_one.py's
own coverage picks this lane up automatically once it is registered, because
it iterates every PHASE_LANES entry) and "exactly one of" (job_store's own
EXACTLY_ONE_OF table, which this lane uses the same way P4/P7/P9 already do).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import (  # noqa: E402
    EXACTLY_ONE_OF, PHASE_LANES, PHASE_PARAMETERS, PHASE_REQUIRED,
    JobRejected, command_for, lane_image, phase_parameter_schema,
    runtime_image_for, validate_parameters,
)

PROFILE_ID = "grow-track-patches-v1@0.1.0"
LANE = PHASE_LANES["P1"]["grow-track-patches"]


def job(**overrides):
    parameters = validate_parameters(
        {"lane": "grow-track-patches", "dataset_path": "/artifacts/spiral/PHerc0172",
         "random_count": 4, **overrides.pop("parameters", {})}, "P1",
        server_owned=())
    return {"job_id": "j-2", "sample_id": "PHerc0172", "mission_id": "m-1",
            "phase": "P1", "profile_id": PROFILE_ID,
            "parameters": parameters, **overrides}


def argv_of(**overrides) -> list[str]:
    return [str(token) for token in
            command_for(job(**overrides), runner=LANE["runner"],
                       output_dir="/runs/p1-2")]


# -- the lane's own declared facts -------------------------------------------

def test_the_lane_is_registered_with_a_real_runner():
    assert (ROOT / LANE["runner"]).is_file()
    assert LANE["profiles"] == (PROFILE_ID,)


def test_the_lane_needs_the_fitter_image_but_no_card():
    """Same image as spiral-fit -- the three scripts this lane runs are
    siblings of fit_spiral.py in the one image that carries spiral-fitting/
    -- but no GPU: verified by reading every import in all three scripts and
    the two tracks.py functions they call (see run_grow_track_patches.py's
    own module docstring and the profile's notes.cpu_only)."""
    assert lane_image(LANE) == "helena-villa-python"
    assert runtime_image_for(job()) == "helena-villa-python"
    assert LANE["gpu_required"] is False


def test_the_gpu_card_test_picks_this_lane_up_on_its_own():
    """Not a duplicate of test_a_lane_asks_for_a_card_only_if_it_uses_one.py
    -- a check that this lane is the kind of thing that test's own `lanes()`
    helper (which iterates every PHASE_LANES entry) actually sees, so that
    file's coverage is real rather than accidentally skipping a lane
    registered after it was written."""
    assert ("P1", "grow-track-patches") in [
        (phase, lane_id)
        for phase in PHASE_LANES for lane_id in PHASE_LANES[phase]]


# -- the split requirement ----------------------------------------------------

def test_p1_still_requires_the_scroll_binding_by_default():
    """spiral-fit's own contract, unchanged: a job naming no lane (or
    spiral-fit explicitly) still falls back to PHASE_REQUIRED["P1"]."""
    assert PHASE_REQUIRED["P1"] == (
        "scroll_name", "dataset_path", "z_begin", "z_end", "voxel_size_um")
    with pytest.raises(JobRejected, match="missing required"):
        validate_parameters({"dataset_path": "/a"}, "P1", server_owned=())


def test_grow_track_patches_needs_only_the_dataset():
    """Everything spiral-fit's own six require is a fact about a scroll
    binding this lane never makes -- it reads an already-staged dataset and
    grows patches from its tracks, nothing else."""
    assert LANE["required"] == ("dataset_path",)
    validate_parameters(
        {"lane": "grow-track-patches", "dataset_path": "/artifacts/spiral/PHerc0172",
         "random_count": 1}, "P1", server_owned=())  # must not raise


def test_grow_track_patches_still_refuses_a_missing_dataset_path():
    with pytest.raises(JobRejected, match="missing required"):
        validate_parameters(
            {"lane": "grow-track-patches", "random_count": 1}, "P1", server_owned=())


def test_an_unknown_p1_lane_is_refused_at_enqueue_not_defaulted():
    """Belt and suspenders, from two directions: this file's own P1-specific
    fallback (parameters.get("lane") not registered -> the flat
    PHASE_REQUIRED tuple, which "/a" alone still fails) and job_store's own
    pre-existing generic one ("Lane-local requirements are authoritative for
    phases whose implementations consume different shapes", a few lines
    below in validate_parameters) -- both refuse before a worker ever burns
    an attempt on a lane nobody registered. The generic one's own lane_for()
    call reports it first, as "unknown P1 lane"."""
    with pytest.raises(JobRejected, match="unknown P1 lane"):
        validate_parameters(
            {"lane": "not-a-real-lane", "dataset_path": "/a"}, "P1", server_owned=())


# -- exactly one of seeds / random_count -------------------------------------

def test_the_lane_declares_its_exactly_one_of_rule():
    rule = EXACTLY_ONE_OF["P1"][0]
    assert rule["lane"] == "grow-track-patches"
    assert set(rule["names"]) == {"seeds", "random_count"}


def test_neither_seeds_nor_random_count_is_refused():
    with pytest.raises(JobRejected, match="exactly one"):
        validate_parameters(
            {"lane": "grow-track-patches", "dataset_path": "/a"}, "P1", server_owned=())


def test_both_seeds_and_random_count_is_also_refused():
    with pytest.raises(JobRejected, match="exactly one"):
        validate_parameters(
            {"lane": "grow-track-patches", "dataset_path": "/a",
             "seeds": [1, 2], "random_count": 3}, "P1", server_owned=())


def test_spiral_fit_itself_is_not_subject_to_the_grow_track_patches_rule():
    """The rule is lane-scoped in EXACTLY_ONE_OF (it carries "lane":
    "grow-track-patches"); a spiral-fit job naming neither seeds nor
    random_count -- it has no use for either -- must not be refused by a
    check that belongs to a different lane on the same phase."""
    validate_parameters(
        {"scroll_name": "PHerc0172", "dataset_path": "/a", "z_begin": 500,
         "z_end": 9000, "voxel_size_um": 7.91}, "P1", server_owned=())  # must not raise


# -- the command ---------------------------------------------------------------

def test_the_command_carries_the_dataset_and_the_random_count():
    argv = argv_of()
    assert argv[argv.index("--dataset-path") + 1] == "/artifacts/spiral/PHerc0172"
    assert argv[argv.index("--random-count") + 1] == "4"
    assert argv[argv.index("--out") + 1] == "/runs/p1-2"


def test_seeds_travel_as_one_json_flag_not_repeated_tokens():
    """declarative_argv's json_flags mechanism: one flag, one JSON value --
    matching what run_grow_track_patches.py's own --seeds-json parses."""
    argv = argv_of(parameters={"seeds": [3, 7, 11], "random_count": None})
    assert argv[argv.index("--seeds-json") + 1] == "[3,7,11]"
    assert "--random-count" not in argv


def test_dataset_path_is_still_an_absolute_path_without_dotdot():
    """PATH_PARAMETERS applies phase-wide; this lane's dataset_path is the
    same parameter spiral-fit's is, checked the same way."""
    with pytest.raises(JobRejected, match="absolute path"):
        validate_parameters(
            {"lane": "grow-track-patches", "dataset_path": "relative/path",
             "random_count": 1}, "P1", server_owned=())


def test_the_form_lists_the_lane_and_its_own_exactly_one_of_rule():
    schema = phase_parameter_schema("P1")
    lanes = {lane["id"]: lane for lane in schema["lanes"]}
    assert lanes["grow-track-patches"]["profiles"] == [PROFILE_ID]
    assert lanes["grow-track-patches"]["required"] == ["dataset_path"]
    rules = schema["exactly_one_of"]
    assert any(rule["lane"] == "grow-track-patches"
              and set(rule["names"]) == {"seeds", "random_count"}
              for rule in rules)


def test_every_new_p1_parameter_is_documented_for_the_form():
    for name in ("unverified_patches_dir", "verified_patches_dir", "seeds", "random_count"):
        assert name in PHASE_PARAMETERS["P1"]
