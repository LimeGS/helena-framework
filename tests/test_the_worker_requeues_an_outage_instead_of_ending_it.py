"""The worker has to actually take the recoverable lane.

The store now has `requeue_preflight_source_unavailable`, but a queue method
nothing calls is not a retry mechanism -- it is dead code that reads like one.
This is the half that makes the previous half true: when the worker's own
classifier says PREFLIGHT_SOURCE_UNAVAILABLE, the job goes back to the queue;
when it says anything else, it stays terminal exactly as before.

The distinction is the whole point. A missing setting or a receipt that is not
ink-blind will fail identically on the next worker, so requeuing those would
just burn the budget and delay the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import preflight_worker  # noqa: E402


class _Store:
    """Records which lane the worker chose."""

    def __init__(self) -> None:
        self.failed: list[dict] = []
        self.requeued: list[dict] = []

    def fail_preflight(self, preflight_job_id, lease_token, reason_code, detail=None):
        self.failed.append({"preflight_job_id": preflight_job_id,
                            "reason_code": reason_code, "detail": detail})
        return {"preflight_job_id": preflight_job_id, "state": "FAILED"}

    def requeue_preflight_source_unavailable(self, preflight_job_id, lease_token,
                                             receipt, *, retry_delay_seconds,
                                             maximum_requeues):
        self.requeued.append({"preflight_job_id": preflight_job_id,
                              "receipt": receipt,
                              "retry_delay_seconds": retry_delay_seconds,
                              "maximum_requeues": maximum_requeues})
        return {"status": "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE",
                "preflight_job_id": preflight_job_id, "state": "PENDING",
                "retry_after": "2026-08-07T12:00:00Z", "requeues": 1}


CLAIM = {"preflight_job_id": "pf-1", "lease_token": "token-1",
         "mission_id": "mission-control", "sample_id": "PHerc0139",
         "source_snapshot_id": "snap-1", "request": {"parameters": {}}}


def _worker(store):
    worker = preflight_worker.CandidatePreflightWorker.__new__(
        preflight_worker.CandidatePreflightWorker)
    worker.store = store
    worker.worker_id = "worker-a"
    return worker


def test_a_source_outage_goes_back_to_the_queue() -> None:
    store = _Store()
    outcome = _worker(store)._fail(
        dict(CLAIM), "PREFLIGHT_SOURCE_UNAVAILABLE",
        ConnectionError("ServerDisconnected"))

    assert store.requeued, "the recoverable lane was never taken"
    assert not store.failed, "a recoverable outage was still ended terminally"
    assert outcome["status"] == "REQUEUED"
    assert outcome["no_scientific_conclusion"] is True
    assert outcome["ink_used"] is False


def test_the_requeue_is_bounded() -> None:
    """Unbounded, this hides a source that is genuinely gone."""
    store = _Store()
    _worker(store)._fail(dict(CLAIM), "PREFLIGHT_SOURCE_UNAVAILABLE",
                         ConnectionError("ServerDisconnected"))

    asked = store.requeued[0]
    assert asked["maximum_requeues"] >= 1
    assert asked["maximum_requeues"] < 100, "that is not a bound"
    assert asked["retry_delay_seconds"] > 0, "it would re-read a source still down"


def test_the_outage_sentence_is_redacted_before_it_becomes_durable() -> None:
    store = _Store()
    _worker(store)._fail(
        dict(CLAIM), "PREFLIGHT_SOURCE_UNAVAILABLE",
        ConnectionError("https://reader:hunter2@example.invalid/ct.zarr dropped"))

    assert "hunter2" not in str(store.requeued[0]["receipt"])


@pytest.mark.parametrize("reason_code", [
    "PREFLIGHT_PROVIDER_NOT_CONFIGURED",
    "PREFLIGHT_RECEIPT_NOT_INK_BLIND",
    "PREFLIGHT_SOURCE_SNAPSHOT_UNKNOWN",
    "PREFLIGHT_REQUEST_UNUSABLE",
])
def test_a_failure_that_will_not_heal_stays_terminal(reason_code: str) -> None:
    """These fail identically on the next worker. Requeuing them spends the
    budget and delays the report."""
    store = _Store()
    outcome = _worker(store)._fail(dict(CLAIM), reason_code, "no")

    assert store.failed, f"{reason_code} stopped being terminal"
    assert not store.requeued
    assert outcome["reason_code"] == reason_code


def test_a_store_without_the_lane_still_fails_terminally() -> None:
    """A control plane that predates the requeue must not crash the worker;
    the old behaviour is the correct fallback."""

    class _Old(_Store):
        requeue_preflight_source_unavailable = None

    store = _Old()
    store.requeue_preflight_source_unavailable = None
    outcome = _worker(store)._fail(dict(CLAIM), "PREFLIGHT_SOURCE_UNAVAILABLE",
                                   ConnectionError("ServerDisconnected"))

    assert store.failed, "the worker had no fallback and lost the job"
    assert outcome["reason_code"] == "PREFLIGHT_SOURCE_UNAVAILABLE"


def test_a_lost_lease_is_still_stood_down_not_requeued() -> None:
    """Somebody else owns the job now and gets to decide how it ends."""

    class _NotOurs(_Store):
        def requeue_preflight_source_unavailable(self, *args, **kwargs):
            raise RuntimeError("this preflight job is not held by that lease")

    store = _NotOurs()
    outcome = _worker(store)._fail(dict(CLAIM), "PREFLIGHT_SOURCE_UNAVAILABLE",
                                   ConnectionError("ServerDisconnected"))

    assert outcome["status"] == "STOOD_DOWN"
    assert not store.failed, "it vandalised a job it no longer held"
