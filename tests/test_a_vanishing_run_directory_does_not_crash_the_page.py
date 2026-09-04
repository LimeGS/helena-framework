"""index_runs walked the whole runs tree on every call to `/api/state`, and a
directory a concurrent job removed between being listed and being stat'd --
routine when a fleet is cleaning up after itself, and it was, on gpu-1, while
this was live -- raised FileNotFoundError with no handler above it. The
exception came from inside Path.rglob's own generator, which cannot be
resumed past: the whole request died with a 500, on every page that reads
mission state, for as long as anything anywhere under the runs root was
being deleted while the walk ran.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")

import panel.app as module  # noqa: E402


def test_a_directory_that_vanishes_before_stat_is_skipped_not_raised(tmp_path):
    root = tmp_path / "runs"
    (root / "mission-a" / "run-1").mkdir(parents=True)
    (root / "mission-a" / "run-2").mkdir(parents=True)

    real_stat = os.stat

    def flaky_stat(path, *args, **kwargs):
        if os.path.basename(str(path)) == "run-2":
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(module.os, "stat", side_effect=flaky_stat):
        stamps = module._run_directory_stamps(root)

    names = {name for name, _mtime in stamps}
    assert "run-2" not in names, "a vanished directory must be skipped, not raise"
    assert "mission-a" in names and "run-1" in names, (
        "the race must not take siblings down with it")


def test_a_directory_that_vanishes_before_the_walk_descends_is_skipped(tmp_path):
    """The same race one level earlier: os.walk itself fails to scandir a
    directory that was listed by its parent and removed before the walk
    reached it. onerror has to swallow this rather than let it propagate."""
    root = tmp_path / "runs"
    (root / "mission-a").mkdir(parents=True)
    ghost = root / "mission-a" / "gone-before-descent"
    ghost.mkdir()
    ghost.rmdir()  # listed once below via a stub, never actually present

    real_walk = os.walk

    def walk_with_a_ghost(top, onerror=None, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, onerror=onerror, **kwargs):
            if dirpath == str(root / "mission-a") and "gone-before-descent" not in dirnames:
                dirnames.append("gone-before-descent")
            yield dirpath, dirnames, filenames

    import unittest.mock as mock
    with mock.patch.object(module.os, "walk", side_effect=walk_with_a_ghost):
        stamps = module._run_directory_stamps(root)  # must not raise

    names = {name for name, _mtime in stamps}
    assert "gone-before-descent" not in names
    assert "mission-a" in names


def test_index_runs_still_reads_a_mission_that_did_not_race(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    mission_dir = root / "mission-a"
    run_dir = mission_dir / "run-1"
    run_dir.mkdir(parents=True)
    (mission_dir / module.mission_contract.MANIFEST_NAME).write_text("{}")
    (run_dir / "SOME_RECEIPT.json").write_text(
        '{"lane": "x", "state": "SCREENED"}')

    monkeypatch.setattr(module, "RUNS", root)
    monkeypatch.setattr(module, "_cache", {"stamp": None, "runs": []})
    runs = module.index_runs(force=True)
    assert any(r.mission_id == "mission-a" for r in runs)


def test_scan_skips_a_run_directory_that_vanished_between_listing_and_reading(tmp_path):
    """The second half of the same race, one level later: _scan lists a
    mission's run directories, then a concurrent cleanup removes one before
    _scan gets to glob its receipts."""
    mission_dir = tmp_path / "mission-a"
    surviving = mission_dir / "run-1"
    surviving.mkdir(parents=True)
    (surviving / "SOME_RECEIPT.json").write_text('{"lane": "x", "state": "SCREENED"}')
    (mission_dir / "run-2-about-to-vanish").mkdir()

    real_glob = Path.glob

    def flaky_glob(self, pattern):
        if self.name == "run-2-about-to-vanish":
            raise FileNotFoundError(str(self))
        return real_glob(self, pattern)

    import unittest.mock as mock
    with mock.patch.object(Path, "glob", flaky_glob):
        runs = module._scan(mission_dir, "mission-a")

    assert len(runs) == 1
    assert runs[0].mission_id == "mission-a"


def test_scan_returns_nothing_when_the_directory_itself_is_already_gone(tmp_path):
    gone = tmp_path / "never-existed"
    assert module._scan(gone, "mission-a") == []
