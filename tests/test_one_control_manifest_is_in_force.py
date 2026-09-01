"""The panel and the fleet have to be validating against the same manifest.

The control manifest's filename was written out by hand in four places in
panel/app.py and once more in the fleet, so "which manifest is in force" was
five independent facts that happened to agree. Bumping the manifest moved some
and not others, and the failure that produces is quiet and expensive: the panel
seals the P0 artifact under one profile while the run declares another, and the
run dies at

    409 selected P0 contains a partial or tampered control marker

hours later, having already spent the GPU time. That refusal is correct -- it is
the check doing its job -- which is precisely why the two sides must not be
allowed to drift in the first place.

So the version is asserted to be one thing. This does not care which version is
current; it cares that everybody is looking at the same one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "panel/app.py"
GENERATOR = ROOT / "framework/stages/01-segmentation/fleet/generator.py"
# Everything that resolves the manifest at runtime. Tests may legitimately pin
# an older version as a fixture; these cannot, because a disagreement here is a
# P0 sealed under one profile and verified against another.
RUNTIME = (
    PANEL,
    GENERATOR,
    ROOT / "framework/stages/03-ink/fleet/ink_worker.py",
)
PROFILES = ROOT / "framework/profiles/01-segmentation"

VERSION = re.compile(r"first-letters-control-policy-(\d+\.\d+\.\d+)\.json")
POLICY_ID = re.compile(r"first-letters-control-policy@(\d+\.\d+\.\d+)")


def test_the_panel_names_exactly_one_manifest_version() -> None:
    found = set(VERSION.findall(PANEL.read_text(encoding="utf-8")))
    assert len(found) == 1, (
        f"panel/app.py loads {sorted(found)}; a P0 sealed under one and checked "
        "against another is the 409 this exists to prevent")


def test_the_fleet_pins_the_version_the_panel_loads() -> None:
    panel_version = set(VERSION.findall(PANEL.read_text(encoding="utf-8")))
    fleet_version = set(POLICY_ID.findall(GENERATOR.read_text(encoding="utf-8")))

    assert fleet_version, "the fleet pins no control policy at all"
    assert panel_version == fleet_version, (
        f"the panel validates against {sorted(panel_version)} while the fleet "
        f"pins {sorted(fleet_version)}")


def test_every_runtime_file_names_the_same_version() -> None:
    """The guard that missed a site the first time.

    Written against the panel and the fleet, it passed while the ink worker
    still named 1.1.0 -- a third runtime resolver nobody had thought to check.
    Naming the files explicitly is the point: a new one added tomorrow is a
    deliberate line here, not a silent sixth copy.
    """
    named = {}
    for path in RUNTIME:
        text = path.read_text(encoding="utf-8")
        found = set(VERSION.findall(text)) | set(POLICY_ID.findall(text))
        if found:
            named[path.name] = found

    assert named, "no runtime file resolves the control manifest at all"
    versions = set().union(*named.values())
    assert len(versions) == 1, (
        f"runtime files disagree about which manifest is in force: {named}")


def test_the_manifest_in_force_exists_and_declares_that_version() -> None:
    version = next(iter(VERSION.findall(PANEL.read_text(encoding="utf-8"))))
    path = PROFILES / f"first-letters-control-policy-{version}.json"
    assert path.exists(), f"{path.name} is named but not present"
    assert json.loads(path.read_text(encoding="utf-8"))["profile_id"].endswith(
        "@" + version)


def test_the_loader_returns_that_manifest() -> None:
    """Reading the file is not the same as the panel choosing it."""
    pytest.importorskip("fastapi")
    import panel.app as panel_app

    version = next(iter(VERSION.findall(PANEL.read_text(encoding="utf-8"))))
    assert panel_app.first_letters_control_policy()["profile_id"].endswith(
        "@" + version)
