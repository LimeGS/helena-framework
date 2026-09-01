"""One directory this process cannot open must not answer for all the others.

The panel said "No mission yet. Use New mission to name the scrolls you are
attempting" while fifteen missions sat on disk. The cause was one run directory
left at 0700 by a job that had run as root, after the panel was moved to run as
1000:1000. `discover()` probed it for a manifest, the `PermissionError`
propagated out of the loop, and the whole listing became an empty list.

Empty is the worst possible answer here, because it is indistinguishable from
the true one. A new installation has no missions either, and the page says the
same sentence in both cases -- so the failure reads as a fresh install rather
than as a permissions problem, and the obvious next action is to create a
sixteenth mission on top of fifteen invisible ones.

The directory in question did not even hold a mission. It held a probability
map. It hid every mission on the box anyway.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.contracts import mission as mission_contract  # noqa: E402


def write_mission(root: Path, mission_id: str) -> Path:
    directory = root / mission_id
    directory.mkdir(parents=True)
    (directory / mission_contract.MANIFEST_NAME).write_text(json.dumps({
        "schema": mission_contract.SCHEMA,
        "mission_id": mission_id,
        "name": mission_id,
        "description": "",
        "state": "active",
        "scrolls": ["PHerc826"],
        "scrolls_frozen_at_utc": None,
        "created_at_utc": "2026-08-01T00:00:00Z",
        "created_by": "test",
        "amendments": [],
        "non_claims": list(mission_contract.DEFAULT_NON_CLAIMS),
    }))
    return directory


@pytest.fixture(name="runs")
def _runs(tmp_path):
    for name in ("golden-run", "platform-soak-20260801", "pherc0826-descubrimiento"):
        write_mission(tmp_path, name)
    return tmp_path


def test_every_mission_is_listed_when_nothing_is_locked(runs):
    found = {m["mission_id"] for m in mission_contract.discover(runs)}
    assert {"golden-run", "platform-soak-20260801", "pherc0826-descubrimiento"} <= found


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 0700 directory")
def test_an_unreadable_directory_hides_only_itself(runs):
    locked = runs / "public-control-queued"
    locked.mkdir()
    (locked / "probability.npy").write_bytes(b"not a mission")
    locked.chmod(0o000)
    try:
        found = {m["mission_id"] for m in mission_contract.discover(runs)}
    finally:
        locked.chmod(0o700)  # so tmp_path teardown can remove it
    assert {"golden-run", "platform-soak-20260801", "pherc0826-descubrimiento"} <= found


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 0700 directory")
def test_what_could_not_be_read_is_reported_rather_than_swallowed(runs):
    """Silently skipping is the other way to be wrong: the operator is the one
    who can fix a permission, and they cannot fix what they are not told."""
    locked = runs / "public-control-queued"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        missions = mission_contract.discover(runs)
    finally:
        locked.chmod(0o700)
    reported = [m for m in missions if m["mission_id"] == "unreadable"]
    assert len(reported) == 1, "an unreadable directory vanished without a word"
    assert "public-control-queued" in " ".join(reported[0]["unreadable"])
    assert "chmod o+rx" in reported[0]["description"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a 0700 directory")
def test_a_mission_whose_runs_are_unreadable_does_not_claim_zero(runs):
    """`run_count` 0 means "nothing has run in this mission", which is what
    unfreezes its scroll selection. Guessing it from an unreadable directory
    would let a mission that has produced results be edited as a draft."""
    mission = runs / "golden-run"
    locked = mission / "a-run"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        found = {m["mission_id"]: m for m in mission_contract.discover(runs)}
    finally:
        locked.chmod(0o700)
    assert found["golden-run"]["run_count"] is None
    assert found["golden-run"]["run_count_unreadable"] is True
    assert found["golden-run"]["selection_frozen"] is False
