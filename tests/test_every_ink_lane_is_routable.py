"""Every ink lane P5 offers, and what actually runs it.

The queue built one argv for every P5 job -- `--profile <id> --upstream-dir
<dir>`, which is run_ink.py's CLI -- while the lane profiles have always
named their own adapter. The TimeSformer adapter takes `--ink-profile <path>`
and has no `--upstream-dir` at all, so every TimeSformer lane queued through the
API ran the wrong script. It went unnoticed because the single P5 job that ever
ran used the canonical lane, whose adapter is the one that was hardcoded.

The rule these tests hold: a profile in the lane directory is either routable or
refused with a reason. Silently routing it to whichever runner came first is
what this cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import (  # noqa: E402
    INK_ADAPTERS, JobRejected, command_for, ink_adapter, ink_lane_inventory,
)

PROFILES = sorted((ROOT / "framework/profiles/03-ink").glob("*.json"))


def profile_ids() -> list[str]:
    return [json.loads(path.read_text())["profile_id"] for path in PROFILES
            if "profile_id" in json.loads(path.read_text())]


@pytest.mark.parametrize("profile_id", profile_ids())
def test_a_lane_either_routes_or_says_why_not(profile_id):
    try:
        adapter, spec, _ = ink_adapter(profile_id)
    except JobRejected as refused:
        assert str(refused), "a refusal has to carry its reason"
        return
    assert (ROOT / adapter).is_file(), f"{profile_id} names an adapter that is not here"
    assert spec["needs"], f"{profile_id} routes to an adapter that declares no inputs"


@pytest.mark.parametrize("profile_id", profile_ids())
def test_a_routable_lane_builds_a_command_its_adapter_accepts(profile_id):
    """The failure this replaces was invisible until a GPU had been reserved:
    the command was built, the job was claimed, and the script rejected the
    flags."""
    try:
        adapter, spec, _ = ink_adapter(profile_id)
    except JobRejected:
        return
    parameters = {"tiff_dir": "/stack", "checkpoint": "/models/m.safetensors",
                  "upstream_dir": "/models", "source_pixel_um": 9.362,
                  "config": "/c.json", "villa_python_root": "/villa",
                  "input_manifest": "/m.json"}
    argv = command_for({"phase": "P5", "profile_id": profile_id,
                        "sample_id": "PHerc826", "parameters": parameters},
                       runner=str(ROOT / adapter), output_dir="/runs/p5-1")
    source = (ROOT / adapter).read_text()
    for token in argv:
        if token.startswith("--"):
            assert f'"{token}"' in source, f"{adapter} takes no {token}"


def test_an_adapter_nobody_taught_the_queue_is_refused(tmp_path, monkeypatch):
    import job_store

    lane = tmp_path / "made-up-lane.json"
    lane.write_text(json.dumps({"profile_id": "made-up@1.0.0",
                                "checkpoint_sha256": "0" * 64,
                                "adapter": "framework/stages/03-ink/scripts/nope.py"}))
    monkeypatch.setattr(job_store, "ink_profile_path", lambda profile_id: lane)
    with pytest.raises(JobRejected) as refused:
        job_store.ink_adapter("made-up@1.0.0")
    assert "no command for" in str(refused.value)


def test_the_inventory_covers_the_profiles_and_the_registry():
    """Three populations that were never listed together: the lane profiles, the
    adapters that can execute one, and the registry's record of what each
    checkpoint is worth. A method with no profile cannot be queued no matter how
    good it is, and nothing said which ones those were."""
    lanes = ink_lane_inventory()
    assert len(lanes) >= len(PROFILES)
    routable = [lane for lane in lanes if lane["routable"]]
    assert routable, "no ink lane can be run at all"
    assert all(lane["adapter"] in INK_ADAPTERS for lane in routable)
    # Everything that cannot be run says why, whether it is a profile with no
    # adapter or a checkpoint with no profile.
    assert all(lane["reason"] for lane in lanes if not lane["routable"])
