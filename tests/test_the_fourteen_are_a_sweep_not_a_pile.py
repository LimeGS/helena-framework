"""Fourteen checkpoints on disk are storage; fourteen that a run can name are a
measurement.

The gap this closes is small and total: `run_ink_9um.py` compares the
checkpoint it loaded against the digest its profile declares and exits 3 when
they differ. That is the right behaviour -- a receipt that named one checkpoint
while another produced the map would be worse than no receipt -- but it meant
the thirteen checkpoints without a profile could not be run at all. They could
be downloaded, listed, and verified, and then nothing.

So each gets a profile, and each profile gets a registry entry, which is the
platform's own vocabulary rather than a new one: the registry already carries
`pherc1667-cross-segment-iteration0` beside `pherc1667-iteration5`, two
training iterations of one method registered separately, for exactly this
reason. A step is a method whose behaviour has to be measured, not a version of
a settled one -- the canonical entry's own notes say so.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "framework/profiles/03-ink"
REGISTRY = json.loads(
    (ROOT / "framework/registries/method-capabilities-0.1.0.json").read_text())
MANIFEST = json.loads(
    (ROOT / "framework/registries/ink-weights-0.1.0.json").read_text())
CANONICAL = "ink-9um-hybrid-3d2d-screening@1.0.0"

sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
from job_store import ink_adapter, ink_profile_path  # noqa: E402


def nine_um_entries():
    return [e for e in MANIFEST["entries"] if e["repo"] == "scrollprize/ink_9um"]


def test_every_one_of_the_fourteen_has_a_profile_that_pins_exactly_it():
    by_digest = {}
    for path in sorted(PROFILES.glob("*.json")):
        spec = json.loads(path.read_text())
        if spec.get("checkpoint_sha256"):
            by_digest.setdefault(spec["checkpoint_sha256"], []).append(spec["profile_id"])
    for entry in nine_um_entries():
        owners = by_digest.get(entry["sha256"], [])
        assert len(owners) == 1, f"{entry['upstream_path']}: {owners or 'no profile'}"


def test_the_canonical_lane_keeps_its_name(entries=None):
    """Thirteen siblings must not have renamed or shadowed the lane that
    everything already queued refers to."""
    spec = json.loads((PROFILES / "ink-9um-hybrid-3d2d-screening-1.0.0.json").read_text())
    assert spec["profile_id"] == CANONICAL
    assert spec["method_id"] == "ink-9um-hybrid-3d2d@1.0.0"
    canonical = [e for e in nine_um_entries()
                 if e["upstream_path"] == "hybrid_3d2d-seed42/step-075000.pth"]
    assert canonical[0]["sha256"] == spec["checkpoint_sha256"]


def test_each_sibling_is_registered_and_names_its_own_checkpoint():
    """The platform's rule, not a new one: no profile may carry a checkpoint the
    method registry does not know."""
    known = {e["method_id"]: e.get("known_checkpoint_sha256")
             for e in REGISTRY["entries"]}
    for path in sorted(PROFILES.glob("ink-9um-hybrid-3d2d-seed*-step*.json")):
        spec = json.loads(path.read_text())
        assert spec["method_id"] in known, spec["profile_id"]
        assert known[spec["method_id"]] == spec["checkpoint_sha256"], spec["profile_id"]


def test_a_sibling_differs_from_the_canonical_lane_in_the_checkpoint_and_nothing_else():
    """A comparison across the fourteen only means something if one thing varies.

    Same adapter, same patch size, same normalization, same execution defaults --
    including batch_size 1, which the canonical profile records as the
    difference between a live map and an empty one.
    """
    canonical = json.loads((PROFILES / "ink-9um-hybrid-3d2d-screening-1.0.0.json").read_text())
    varies = {"profile_id", "method_id", "checkpoint_sha256", "notes"}
    for path in sorted(PROFILES.glob("ink-9um-hybrid-3d2d-seed*-step*.json")):
        spec = json.loads(path.read_text())
        for key in canonical:
            if key in varies:
                continue
            assert spec[key] == canonical[key], f"{spec['profile_id']} differs in {key}"
        assert spec["checkpoint_sha256"] != canonical["checkpoint_sha256"]


def test_the_profile_id_says_which_checkpoint_ran():
    """A receipt names a profile. If the id did not carry the seed and the step,
    reading a sweep would mean matching hashes by hand."""
    for entry in nine_um_entries():
        seed, step = re.match(r"hybrid_3d2d-seed(\d+)/step-(\d+)\.pth$",
                              entry["upstream_path"]).groups()
        spec = next(json.loads(p.read_text()) for p in sorted(PROFILES.glob("*.json"))
                    if json.loads(p.read_text()).get("checkpoint_sha256") == entry["sha256"])
        if spec["profile_id"] == CANONICAL:
            continue  # the canonical lane predates the naming and keeps its id
        assert f"seed{seed}" in spec["profile_id"]
        assert f"step{int(step)}" in spec["profile_id"]


def test_the_queue_routes_every_sibling_to_the_9um_runner():
    """Discovered by glob, so a new file is picked up -- but only if it routes."""
    for path in sorted(PROFILES.glob("ink-9um-hybrid-3d2d-seed*-step*.json")):
        spec = json.loads(path.read_text())
        assert ink_profile_path(spec["profile_id"]) == path
        runner, _contract, _found = ink_adapter(spec["profile_id"])
        assert runner.endswith("run_ink_9um.py"), (spec["profile_id"], runner)


def test_the_sweep_names_all_fourteen_and_a_profile_for_each(tmp_path):
    """The sweep is the thing that makes fourteen checkpoints one measurement."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/harness/sweep_ink_9um_control.py"),
         "--models-root", str(tmp_path / "absent"),
         "--output", str(tmp_path / "out"), "--dry-run"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    runs = json.loads((tmp_path / "out" / "sweep.json").read_text())["runs"]
    assert len(runs) == 14
    assert all(r["profile_id"] for r in runs), [r for r in runs if not r["profile_id"]]
    assert len({r["profile_id"] for r in runs}) == 14
    assert len({r["sha256"] for r in runs}) == 14
