"""The control enqueues its preflight and waits, and every ending is a reason.

The measurement moved to a worker because it reads M7 through a service that
lives where workers live. So the runner no longer receives an answer in the
response: it gets a handle and polls.

The endings are the point. A caller that cannot tell "still running" from "the
source is gone" eventually records an outage as a pending job and waits forever,
and a control that never reports is worse than one that reports INCOMPLETE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

import run_first_letters_positive_control as control  # noqa: E402
from panel_client import PanelError  # noqa: E402

HANDLE = {"preflight_job_id": "pf-1"}


class _Panel:
    """A panel that answers a scripted sequence, then repeats the last answer."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def call(self, method, path, body=None):
        self.calls.append((method, path))
        answer = self.answers[0] if len(self.answers) == 1 else self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _clock():
    """A clock that advances a second per reading, so waits are deterministic."""
    ticks = iter(range(0, 100_000))
    return lambda: next(ticks)


def test_a_completed_job_gives_back_its_receipt() -> None:
    panel = _Panel([
        {"job_state": "PENDING"},
        {"job_state": "CLAIMED"},
        {"job_state": "COMPLETED", "status": "COMPLETE", "counts": {"raw_m7": 8}},
    ])
    receipt, reason = control._await_preflight(
        panel, HANDLE, clock=_clock(), sleep=lambda _s: None)
    assert reason == ""
    assert receipt["counts"] == {"raw_m7": 8}
    assert len(panel.calls) == 3, "it stopped polling as soon as it was terminal"


def test_a_failed_job_reports_the_panel_s_reason_and_does_not_wait() -> None:
    """503 is terminal, and its reason code is the finding."""
    refused = PanelError("GET", "/api/segmentation/preflight/pf-1", 503,
                         '{"detail": {"reason_code": "PREFLIGHT_SOURCE_UNAVAILABLE"}}')
    receipt, reason = control._await_preflight(
        _Panel([refused]), HANDLE, clock=_clock(), sleep=lambda _s: None)
    assert receipt is None
    assert reason == "PREFLIGHT_SOURCE_UNAVAILABLE"


def test_a_frozen_root_object_mismatch_is_not_reported_as_an_outage() -> None:
    """Two different findings arrive as 503; the code distinguishes them."""
    refused = PanelError("GET", "/api/segmentation/preflight/pf-1", 503,
                         '{"detail": {"reason_code": "FROZEN_ROOT_OBJECT_EVIDENCE_MISSING"}}')
    _, reason = control._await_preflight(
        _Panel([refused]), HANDLE, clock=_clock(), sleep=lambda _s: None)
    assert reason == "FROZEN_ROOT_OBJECT_EVIDENCE_MISSING"


def test_a_vanished_job_says_so_rather_than_waiting_for_it() -> None:
    refused = PanelError("GET", "/api/segmentation/preflight/pf-1", 404, "gone")
    receipt, reason = control._await_preflight(
        _Panel([refused]), HANDLE, clock=_clock(), sleep=lambda _s: None)
    assert receipt is None
    assert reason == "PREFLIGHT_JOB_NOT_FOUND"


def test_a_job_that_never_finishes_is_bounded() -> None:
    receipt, reason = control._await_preflight(
        _Panel([{"job_state": "CLAIMED"}]), HANDLE, clock=_clock(),
        wait_seconds=3, sleep=lambda _s: None)
    assert receipt is None
    assert reason == "PREFLIGHT_DID_NOT_FINISH_WITHIN_THE_WAIT"


def test_a_handle_without_an_id_is_refused_before_polling() -> None:
    panel = _Panel([{"job_state": "COMPLETED"}])
    receipt, reason = control._await_preflight(
        panel, {}, clock=_clock(), sleep=lambda _s: None)
    assert receipt is None
    assert reason == "PREFLIGHT_HANDLE_MISSING"
    assert panel.calls == [], "it asked the panel about a job it could not name"


def test_an_unexpected_refusal_is_raised_rather_than_swallowed() -> None:
    """A 500 is not a preflight finding; treating it as one would hide it."""
    refused = PanelError("GET", "/api/segmentation/preflight/pf-1", 500, "boom")
    with pytest.raises(PanelError):
        control._await_preflight(_Panel([refused]), HANDLE, clock=_clock(),
                                 sleep=lambda _s: None)


MEASURED_PREFLIGHT_SECONDS = 4184  # 2026-08-06 23:26:10 -> 2026-08-07 00:35:54


def test_the_wait_outlasts_a_measurement_that_actually_happened() -> None:
    """The bound was 1800 and the measurement takes 4184.

    1800 was picked before any real preflight had finished. The first one that
    did took an hour and nine minutes -- COMPLETE, 4200 raw M7 candidates -- so
    every first run on a fresh revision failed at P1 on the clock and needed a
    second one to read the result. Three runs in a row went that way before the
    pattern was named.

    The deployed revision is part of the preflight's request identity, so a new
    revision re-measures by design: this is not a rare path, it is every first
    run.
    """
    assert control.PREFLIGHT_WAIT_SECONDS > MEASURED_PREFLIGHT_SECONDS, (
        "the wait is shorter than a measurement that has actually completed")


def test_the_wait_is_bounded_by_default() -> None:
    """A default of "forever" is how a harness hangs a pipeline overnight.

    Room over the one measurement there is, not room for anything: 4184 seconds
    observed once is thin evidence for a ceiling, and a wait that cannot end is
    worse than one that ends too early -- the early one reports a timeout an
    operator can read, and this program has already learned what an unbounded
    wait costs.
    """
    assert MEASURED_PREFLIGHT_SECONDS < control.PREFLIGHT_WAIT_SECONDS <= 4 * 3600
    assert 0 < control.PREFLIGHT_POLL_SECONDS <= 60


def test_the_envelope_never_overwrites_a_field_of_the_measurement() -> None:
    """A preflight receipt has a `state` of its own.

    The status route used to merge the queue's state over the receipt under the
    same key, so the measurement's field silently became the job's. The digest
    that names the receipt then depended on how it was fetched, and a test
    comparing two hashes is what noticed. The envelope is `job_state` now, and
    the receipt keeps everything it arrived with.
    """
    answer = {"job_state": "COMPLETED", "preflight_job_id": "pf-1",
              "attempts": 1, "mission_id": "m", "sample_id": "s",
              "state": "COMPLETE", "counts": {"raw_m7": 8}}
    measurement = control._preflight_measurement(answer)

    assert measurement["state"] == "COMPLETE", (
        "the queue's state replaced the measurement's own")
    assert measurement["counts"] == {"raw_m7": 8}
    for field in control.PREFLIGHT_ENVELOPE_FIELDS:
        assert field not in measurement, f"{field} is transport, not measurement"
    assert "state" not in control.PREFLIGHT_ENVELOPE_FIELDS, (
        "stripping `state` would remove a field the receipt owns")
