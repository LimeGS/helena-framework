"""Every option the grow binary accepts, and the ones that only mean something on a resume.

`vc_grow_seg_from_seed` takes twelve options. The fleet drove five, so resuming a
surface, inpainting holes and skipping the overlap check were unreachable through
the framework even though the binary has always supported them.

The failure mode being guarded is quiet: `--rewind-gen` on a fresh grow is
accepted by the command line and does nothing, so a plan that asked to rewind
would grow from scratch and the receipt would record a command that reads as
though it had not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "framework" / "stages" / "01-segmentation"))

from fleet.executor import RESUME_OPTIONS, optional_grow_flags  # noqa: E402

BASE = {"generations": 35, "step_size": 20, "min_area_cm": 0.0, "use_cuda": False}


def test_a_plain_seeded_grow_adds_nothing():
    """A plan written before these options existed must produce the same command.

    The receipts bind the command, so a flag that appears on its own would make
    every earlier receipt look like it ran something else.
    """
    assert optional_grow_flags(BASE, {}) == []


def test_the_standalone_toggles_are_passed():
    assert optional_grow_flags({**BASE, "inpaint": True}, {}) == ["--inpaint"]
    assert optional_grow_flags({**BASE, "skip_overlap_check": True}, {}) == [
        "--skip-overlap-check"
    ]


def test_a_resume_carries_its_path_and_options():
    flags = optional_grow_flags(
        {**BASE, "rewind_gen": 12, "resume_generations": 8, "resume_opt": "local"},
        {"resume_from": "/surfaces/fleet-0001", "corrections": "/plans/fix.json"},
    )
    assert flags[:2] == ["--resume", "/surfaces/fleet-0001"]
    assert "--correct" in flags and "/plans/fix.json" in flags
    assert flags[flags.index("--rewind-gen") + 1] == "12"
    assert flags[flags.index("--resume-generations") + 1] == "8"
    assert flags[flags.index("--resume-opt") + 1] == "local"


@pytest.mark.parametrize("key, value", [
    ("rewind_gen", 4), ("resume_generations", 9), ("resume_opt", "global"),
])
def test_resume_options_without_a_resume_are_refused(key, value):
    """Silently ignored is the one outcome worse than an error here."""
    with pytest.raises(ValueError, match="resume"):
        optional_grow_flags({**BASE, key: value}, {})


def test_an_unknown_resume_option_is_refused():
    with pytest.raises(ValueError, match="resume_opt"):
        optional_grow_flags(
            {**BASE, "resume_opt": "partial"}, {"resume_from": "/surfaces/x"}
        )
    assert RESUME_OPTIONS == ("skip", "local", "global")


def test_the_profile_bounds_every_option_the_executor_reads():
    """A parameter the executor honours but the profile does not describe is
    unreachable: the plan builder validates against the profile."""
    import json

    # v1 stays byte-identical: a source-lock plan binds its sha256. The new
    # options live in v2, which a plan opts into by naming it.
    profile = json.loads(
        (ROOT / "framework/stages/01-segmentation/fleet/profiles/vc3d-m7-growth-v2.json")
        .read_text()
    )
    declared = set(profile["parameters"])
    honoured = {"inpaint", "skip_overlap_check", "rewind_gen", "resume_generations",
                "resume_opt"}
    assert honoured <= declared, f"undeclared: {sorted(honoured - declared)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
