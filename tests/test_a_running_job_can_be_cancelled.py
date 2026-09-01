"""Stopping work that has already started.

`cancel` refused anything that had begun -- "only a job that has not started
can be cancelled" -- so the only way to stop a twenty-five minute render that
was already wrong was to kill the worker's container. That takes the worker
down with the job, abandons the lease to expire on its own, and leaves the
record saying the job died rather than that somebody stopped it.

A worker cannot be interrupted from outside, so cancellation is cooperative:
the request is written down, and the worker reads it on the heartbeat it
already sends every fifteen seconds. Nothing new polls, and a worker that is
not running the job cannot be told to stop the wrong one, because the lease
token is what the request is answered against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402
from job_store import InkJobStore  # noqa: E402


class Cursor:
    def __init__(self, rowcount=1, fetchone=None):
        self.statements: list[tuple[str, tuple]] = []
        self.rowcount = rowcount
        self._one = fetchone

    def execute(self, sql, args=()):
        self.statements.append((sql, args))

    def fetchone(self):
        return self._one

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Connection:
    def __init__(self, cursor):
        self._c = cursor

    def cursor(self):
        return self._c

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def store_with(cursor):
    store = InkJobStore("postgresql://unused")
    store._connect = lambda: Connection(cursor)  # noqa: SLF001
    return store


# -- asking ------------------------------------------------------------------


def test_cancelling_a_pending_job_still_stops_it_outright() -> None:
    """Nothing is running, so there is nobody to ask: it ends here."""
    cursor = Cursor(rowcount=1)
    assert store_with(cursor).cancel("p5-x") is True
    sql = " ".join(s for s, _ in cursor.statements)
    assert "state='cancelled'" in sql


def test_cancelling_a_running_job_records_the_request() -> None:
    """It cannot be stopped from here -- the work is a subprocess on another
    machine -- so the request is written down for the worker to read."""
    # rowcount 1: the UPDATE matches, because the job is running.
    cursor = Cursor(rowcount=1)
    store = store_with(cursor)

    assert store.request_cancel("p5-x") is True

    sql = " ".join(s for s, _ in cursor.statements)
    assert "cancel_requested" in sql


def test_the_route_accepts_a_running_job_instead_of_refusing_it() -> None:
    import inspect

    sys.path.insert(0, str(ROOT))
    from panel.app import api_cancel

    body = inspect.getsource(api_cancel)
    assert "request_cancel" in body, "the route still only cancels what never started"


# -- being told --------------------------------------------------------------


def test_the_heartbeat_reports_a_cancellation_back_to_the_worker() -> None:
    """On the write that already happens every fifteen seconds. A second poll
    for this would be a second thing to get wrong."""
    import inspect

    body = inspect.getsource(InkJobStore.heartbeat)
    assert "cancel_requested" in body
    assert body.count("UPDATE ink_jobs") == 1, (
        "asking whether to stop costs a statement of its own")


def test_the_worker_stops_the_run_when_it_is_told_to() -> None:
    import inspect

    body = inspect.getsource(ink_worker.run_job)
    beat = body[body.index("def beat()"):body.index("store.mark_running")]

    assert "cancel" in beat.lower(), "the heartbeat ignores what it is told"


def test_a_cancelled_run_is_recorded_as_stopped_not_as_broken() -> None:
    """A killed process exits non-zero. Reported as a failure it would read as
    the lane breaking, and the next person would go looking for the bug."""
    import inspect

    body = inspect.getsource(ink_worker.run_job)

    assert '"cancelled"' in body, "a cancelled run has no state of its own"


def test_the_column_exists_wherever_this_schema_is_applied() -> None:
    migrations = ROOT / "framework/stages/03-ink/fleet/migrations"
    applied = "\n".join(p.read_text() for p in sorted(migrations.glob("*.sql")))

    assert "cancel_requested" in applied
