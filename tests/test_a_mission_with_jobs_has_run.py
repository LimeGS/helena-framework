"""A mission that has queued work has produced work.

`selection_frozen` was decided from the mission directory alone -- a run being a
directory with a receipt in it -- and every phase that goes through the queue
writes its run somewhere else entirely. So a mission that had certified a
surface, rendered a stack and screened a map still told the person looking at P0
"nothing has run in this mission yet, so the selection is still a draft", and
would have taken a scroll in or out with no reason recorded against the work that
had already read the old selection.

That is the one thing the freeze exists to prevent: a selection that moves
quietly makes every earlier result unreadable, because "we screened everything"
means something else when everything changed halfway through.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")


class _Store:
    """A queue holding jobs for one mission."""

    def __init__(self, missions: dict[str, int]):
        self.missions = missions
        self.asked: list[dict] = []

    def jobs(self, *, limit=100, mission_id=None, **_rest):
        self.asked.append({"mission_id": mission_id, "limit": limit})
        if mission_id is not None:
            return [{"job_id": f"p2-{mission_id}"}] * min(
                self.missions.get(mission_id, 0), limit)
        rows = []
        for mission, count in self.missions.items():
            rows += [{"mission_id": mission}] * count
        return rows[:limit]


def _mission(root: Path, mission_id: str) -> None:
    directory = root / mission_id
    directory.mkdir(parents=True)
    (directory / "MISSION.json").write_text(json.dumps({
        "schema": "campaignx.mission.v1",
        "mission_id": mission_id,
        "name": mission_id,
        "scrolls": ["PHerc0139"],
        "state": "ACTIVE",
        "created_at_utc": "2026-08-27T00:00:00Z",
    }))


def _missions(monkeypatch, tmp_path, store) -> dict[str, dict]:
    import panel.app as app

    monkeypatch.setattr(app, "RUNS", tmp_path)
    monkeypatch.setattr(app, "DSN", "configured")
    monkeypatch.setattr(app, "job_store", lambda: store)
    body = json.loads(bytes(app.api_missions().body))
    return {m["mission_id"]: m for m in body["missions"]}


def test_a_queued_job_freezes_the_selection(monkeypatch, tmp_path):
    _mission(tmp_path, "qa-web")
    found = _missions(monkeypatch, tmp_path, _Store({"qa-web": 3}))
    assert found["qa-web"]["job_count"] == 3
    assert found["qa-web"]["selection_frozen"] is True


def test_a_mission_the_recent_jobs_do_not_mention_is_still_asked_about(
        monkeypatch, tmp_path):
    """The display count sees the newest jobs; freezing asks about this one.

    A mission whose work is older than the last five hundred jobs on a busy
    fleet is a mission that has run, and reading the freeze off that window
    would unfreeze it again.
    """
    _mission(tmp_path, "older")
    store = _Store({"older": 0})
    store.missions["older"] = 0
    # Nothing in the window, one row when asked about this mission.
    store.jobs = lambda limit=100, mission_id=None, **rest: (  # type: ignore[assignment]
        [{"job_id": "p8-old"}] if mission_id == "older" else [])
    found = _missions(monkeypatch, tmp_path, store)
    assert found["older"]["job_count"] == 0
    assert found["older"]["selection_frozen"] is True


def test_a_mission_with_no_work_is_still_a_draft(monkeypatch, tmp_path):
    _mission(tmp_path, "fresh")
    found = _missions(monkeypatch, tmp_path, _Store({}))
    assert found["fresh"]["job_count"] == 0
    assert found["fresh"]["selection_frozen"] is False


def test_a_queue_that_cannot_be_asked_does_not_invent_a_freeze(
        monkeypatch, tmp_path):
    """An unreachable queue leaves the filesystem's answer standing.

    Guessing "frozen" would demand a reason nobody can give a reason for, and
    guessing "draft" is the silent-change this exists to stop -- so it says what
    the receipts say and nothing more.
    """
    _mission(tmp_path, "fresh")

    class _Broken:
        def jobs(self, **_query):
            raise RuntimeError("no route to host")

    found = _missions(monkeypatch, tmp_path, _Broken())
    assert found["fresh"]["selection_frozen"] is False
