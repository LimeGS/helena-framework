"""A requeue that nobody can see is an unexplained ninety-minute wait.

The status route answers 200 "still running" for anything that is not COMPLETED
or FAILED, which is what lets a requeued job keep its poller waiting instead of
reporting an outage as a result. That is the correct behaviour and it has a
cost: from outside, a job that lost an hour to a dropped connection and started
over is indistinguishable from one that has simply been slow.

So the envelope carries what happened. Not a new state and not a decision --
just the two fields that tell those two situations apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")


class _Queue:
    def __init__(self, job: dict):
        self.job = job

    def preflight_job(self, _job_id):
        return dict(self.job)


def _answer(monkeypatch, job: dict):
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "fleet_store_read_only", lambda: _Queue(job))
    monkeypatch.setattr(panel_app, "_preflight_queue",
                        lambda store, _name: store.preflight_job)
    monkeypatch.setattr(panel_app, "read_scope",
                        lambda _mission, _sample: {"PHerc0139"})
    response = panel_app.api_segmentation_preflight_status(
        "pf-1", control={})
    import json as _json
    return _json.loads(bytes(response.body))


PENDING = {
    "preflight_job_id": "pf-1",
    "state": "PENDING",
    "attempts": 2,
    "requeues": 1,
    "retry_after": "2026-08-08T01:30:00Z",
    "request": {"mission_id": "mission-control", "sample_id": "PHerc0139"},
}


def test_a_requeued_job_reports_that_it_was_requeued(monkeypatch) -> None:
    answer = _answer(monkeypatch, PENDING)

    assert answer["job_state"] == "PENDING"
    assert answer["requeues"] == 1, (
        "a job that lost an hour to an outage polls identically to a slow one")
    assert answer["retry_after"] == "2026-08-08T01:30:00Z"


def test_a_job_that_never_hit_an_outage_says_nothing_extra(monkeypatch) -> None:
    """No noise on the ordinary path: zero requeues is the normal case and does
    not need a field explaining itself."""
    plain = dict(PENDING, requeues=0, retry_after=None)
    answer = _answer(monkeypatch, plain)

    assert answer.get("requeues") in (0, None)
    assert not answer.get("retry_after")


def test_a_control_plane_without_the_columns_still_answers(monkeypatch) -> None:
    """A store that predates the requeue must not turn a poll into a 500."""
    old = {k: v for k, v in PENDING.items() if k not in ("requeues", "retry_after")}
    answer = _answer(monkeypatch, old)

    assert answer["job_state"] == "PENDING"
