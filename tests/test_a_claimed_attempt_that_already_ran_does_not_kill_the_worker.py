"""The segmentation worker exited on a directory that was already there.

Found by running a spiral fit: 48 tasks queued through the API, the fleet
started growing them, and `docker inspect` showed the worker on its third
restart in ten minutes. Each crash was the same line --

    FileExistsError: [Errno 17] File exists:
    '/artifacts/attempts/a37113ea-.../7ae7aaad-...'

-- from `attempt_dir.mkdir(parents=True, exist_ok=False)`, which sits before
the try block in run_one, so nothing caught it.

exist_ok=False is right and stays. An attempt id is
`stable_id("attempt", {task_id, attempt_number})`, so a directory that is
already there belongs to this same identity, and writing into it would
attribute a second run's output to the first one's evidence.

What was wrong is who paid for it. The process exited, restarted, claimed the
next task and died on that one too, leaving every claimed task to expire its
lease -- one attempt's collision took down the worker and stalled the queue
behind it. It is recorded against the attempt now, the way the other refusals
in run_one already are, and the worker goes on to the next task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.worker import SegmentWorker  # noqa: E402


class _HandsOutOneTask:
    """Enough store to reach the collision and record what came of it."""

    def __init__(self, task):
        self.task = task
        self.terminal = []

    def claim(self, *arguments, **named):
        handed, self.task = self.task, None
        return handed

    def mark_terminal(self, task_id, attempt_id, lease_token, state, result):
        self.terminal.append((task_id, attempt_id, lease_token, state, result))


def _worker(store, run_root):
    return SegmentWorker(
        store, "worker-under-test", None, None, None, run_root,
        run_root / "artifacts", "fixture-surface-qc@1.0.0", lease_seconds=60)


TASK = {"task_id": "task-1", "attempt_id": "attempt-1",
        "lease_token": "a-lease", "sample_id": "PHerc826"}


def test_the_worker_survives_an_attempt_directory_that_exists(tmp_path):
    store = _HandsOutOneTask(dict(TASK))
    run_root = tmp_path / "runs"
    # The dead run's directory, with nothing recorded in it: exactly what a
    # worker killed mid-attempt leaves behind.
    (run_root / TASK["task_id"] / TASK["attempt_id"]).mkdir(parents=True)

    receipt = _worker(store, run_root).run_one()

    assert receipt is not None, "the worker returned nothing instead of a refusal"
    assert receipt["status"] == "POLICY_REJECTED"
    assert receipt["reason"] == "ATTEMPT_DIRECTORY_ALREADY_EXISTS"
    assert receipt["ink_used"] is False


def test_the_collision_is_recorded_against_the_attempt(tmp_path):
    store = _HandsOutOneTask(dict(TASK))
    run_root = tmp_path / "runs"
    attempt = run_root / TASK["task_id"] / TASK["attempt_id"]
    attempt.mkdir(parents=True)

    _worker(store, run_root).run_one()

    assert len(store.terminal) == 1, "nothing was written to the control plane"
    task_id, attempt_id, lease, state, result = store.terminal[0]
    assert (task_id, attempt_id, lease) == (
        TASK["task_id"], TASK["attempt_id"], TASK["lease_token"])
    assert state == "POLICY_REJECTED"
    assert result["failure_class"] == "CONFIGURATION_BLOCK"
    # And on disk beside the dead run, so the directory says why it stopped.
    assert (attempt / "TERMINAL_RECEIPT.json").is_file()


def test_a_receipt_the_earlier_run_left_is_not_overwritten(tmp_path):
    """If the earlier attempt did record, that record is what happened here."""
    store = _HandsOutOneTask(dict(TASK))
    run_root = tmp_path / "runs"
    attempt = run_root / TASK["task_id"] / TASK["attempt_id"]
    attempt.mkdir(parents=True)
    (attempt / "TERMINAL_RECEIPT.json").write_text('{"status": "GROW_FAILED"}')

    _worker(store, run_root).run_one()

    assert (attempt / "TERMINAL_RECEIPT.json").read_text() == (
        '{"status": "GROW_FAILED"}')


def test_an_empty_queue_is_still_nothing_to_do(tmp_path):
    """The branch added is for a collision, not for every claim."""
    store = _HandsOutOneTask(None)
    assert _worker(store, tmp_path / "runs").run_one() is None
