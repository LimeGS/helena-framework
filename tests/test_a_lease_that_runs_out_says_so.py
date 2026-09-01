"""A job that dies by lease expiry leaves a reason behind.

When a worker stops reporting, the queue recycles its job to `pending` and, once
the attempts are gone, flips it to `failed`. Neither step writes `result`. The
event payload is `{"worker_id": ...}` -- the worker that noticed, not the one
that held the lease -- so a job that failed this way is indistinguishable from
one nobody ever claimed, and the attempt it burned is invisible.

That cost a real diagnosis: a P3 flatten job sat silent for 26 minutes, expired,
and left nothing to read. The failure was a URI scheme the lane could not
fetch, which the record could have named and did not.

The segmentation lane already writes `{"status": "LEASE_EXPIRED",
"failure_class": "LEASE_EXHAUSTION"}` into its attempt row. This is that, for
the ink queue.

What is deliberately *not* recorded is whether inference ran. A job can expire
anywhere, including after a model has been loaded and used, and writing
`ink_used: False` would be inferring a negative from the absence of a report --
the same mistake in miniature that the whole lane is built to refuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import InkJobStore  # noqa: E402

EXPIRED = ("p3-af8a53db1aa54c", "gpu-1.worker", 1, 1,
           "2026-08-13T02:29:40Z", "running")


class Cursor:
    """Enough of a cursor to walk `claim` past its two queries.

    Which of the two is being answered comes from the statement, not from which
    fetch method was called. Both of them read with `fetchall` now -- the
    candidate query looks at several rows so a worker can skip past another
    runtime's work -- and a cursor that answered by method handed the recycled
    leases back as pending jobs the moment that changed.
    """

    def __init__(self, *, recycled=(), pending=None) -> None:
        self.statements: list[tuple[str, tuple]] = []
        self._recycled = list(recycled)
        self._pending = pending

    def execute(self, sql: str, args: tuple = ()) -> None:
        self.statements.append((sql, args))

    def _asking_for_candidates(self) -> bool:
        return bool(self.statements) and self.statements[-1][0].lstrip().startswith("SELECT")

    def fetchall(self) -> list:
        if self._asking_for_candidates():
            return [self._pending] if self._pending else []
        return self._recycled

    def fetchone(self):
        return self._pending

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Connection:
    def __init__(self, cursor: Cursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def run_claim(cursor: Cursor) -> Cursor:
    store = InkJobStore("postgresql://unused")
    store._connect = lambda: Connection(cursor)  # noqa: SLF001
    store._job_surface_ids = lambda *a, **k: []  # noqa: SLF001
    store._controlled_mission = lambda *a, **k: False  # noqa: SLF001
    store._require_surface_ids_cursor = lambda *a, **k: None  # noqa: SLF001
    store.claim(worker_id="watcher", host_id="h", phases=None)
    return cursor


def events(cursor: Cursor, kind: str) -> list[dict]:
    found = []
    for sql, args in cursor.statements:
        if "INSERT INTO ink_job_events" in sql and args[1] == kind:
            found.append(json.loads(args[2]))
    return found


def test_the_recycled_job_records_why_it_came_back() -> None:
    cursor = run_claim(Cursor(recycled=[EXPIRED]))

    (expiry,) = events(cursor, "lease_expired")
    assert expiry["failure_class"] == "LEASE_EXHAUSTION"
    assert expiry["status"] == "LEASE_EXPIRED"


def test_the_record_names_the_worker_that_held_it_not_the_one_that_noticed() -> None:
    """`{"worker_id": worker_id}` named the claimant, which is the one party
    guaranteed not to have been involved in the failure."""
    cursor = run_claim(Cursor(recycled=[EXPIRED]))

    (expiry,) = events(cursor, "lease_expired")
    assert expiry["held_by"] == "gpu-1.worker"
    assert expiry["detected_by"] == "watcher"


def test_the_burnt_attempt_is_visible() -> None:
    """A hang costs an attempt whether or not anything is written down. What
    made it invisible is that nothing was."""
    cursor = run_claim(Cursor(recycled=[EXPIRED]))

    (expiry,) = events(cursor, "lease_expired")
    assert expiry["attempt"] == 1
    assert expiry["attempts_remaining"] == 0


def test_the_expiry_query_reads_the_lease_before_it_clears_it() -> None:
    """The old worker and attempt count are gone the instant the row is reset,
    so they have to come out of the same statement."""
    cursor = run_claim(Cursor(recycled=[EXPIRED]))

    recycle = next(sql for sql, _ in cursor.statements
                   if "state='pending'" in sql and "lease_expires_at IS NOT NULL" in sql)
    assert "RETURNING" in recycle
    for column in ("worker_id", "attempts", "lease_expires_at"):
        assert f"expired.{column}" in recycle, (
            f"{column} is read from the row after it was cleared")


def test_an_exhausted_job_fails_with_a_reason() -> None:
    """The gap that mattered most: a `failed` job whose `result` is empty."""
    spent = ("p3-af8a53db1aa54c", "PHerc0139", "profile", {}, 1, 1,
             "P3", "flatten", "control-fl-pherc0139-dev-20260806")
    cursor = run_claim(Cursor(pending=spent))

    failure = next(
        (sql, args) for sql, args in cursor.statements
        if "state='failed'" in sql)
    assert "result=" in failure[0], (
        "the job is failed without recording why, so it reads like one nobody "
        "ever claimed")
    recorded = json.loads(next(a for a in failure[1] if isinstance(a, str)
                               and a.startswith("{")))
    assert recorded["failure_class"] == "LEASE_EXHAUSTION"
    assert "LEASE_EXHAUSTION" in recorded["error"]


def test_the_exhausted_record_claims_nothing_about_inference() -> None:
    """A job can expire after a model has been loaded and used. Writing
    `ink_used: False` would infer a negative from a missing report."""
    spent = ("p3-af8a53db1aa54c", "PHerc0139", "profile", {}, 1, 1,
             "P3", "flatten", "m")
    cursor = run_claim(Cursor(pending=spent))

    failure = next((sql, args) for sql, args in cursor.statements
                   if "state='failed'" in sql)
    recorded = json.loads(next(a for a in failure[1] if isinstance(a, str)
                               and a.startswith("{")))
    assert "ink_used" not in recorded
