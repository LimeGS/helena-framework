"""Run the bounded candidate preflight on a host that can reach the sources.

The measurement used to happen inside the panel's request handler. It cannot:
``run_control_region_preflight`` reaches the m7 prediction through the vc3d MCP
seed service, that service listens on loopback for the workers, and the panel
has neither VC_MCP_URL nor VC_MCP_AUTH_TOKEN. Every real call raised the
provider's own "VC_MCP_URL and VC_MCP_AUTH_TOKEN are required" and the endpoint
answered 503 PREFLIGHT_SOURCE_UNAVAILABLE -- an outage reported for a source
that was never down.

So the panel enqueues and a worker on a host with the sources executes. This
module is that worker. It is deliberately the same shape as ``SurfaceQcWorker``
and it keeps that worker's hard-won distinction:

  * an outage may recover, so the job is failed with an outage reason and the
    worker goes back for the next one;
  * a missing setting will not recover, so the worker fails the job it holds
    and STOPS. The alternative is one misconfigured host consuming the whole
    queue at one attempt each -- the incident that produced 3118 receipts and
    zero measurements, with the counters moved around.

Nothing is written to this host's disk. The receipt is the control plane's, and
the host is replaceable.

The preflight is ink-blind and read-only. It counts candidate availability in a
bounded region and makes no ink, text, letter or absence claim; the finishing
gate below refuses to persist any receipt that says otherwise.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Protocol

from framework.contracts import qc_diagnostics

from . import candidate_preflight
from .common import utc_now

# Every reason a preflight job can end without a measurement. `fail_preflight`
# takes a reason code and nothing else, so this is the entire channel between a
# failure on a worker and a person reading the queue.
PREFLIGHT_REASON_CODES = frozenset({
    # The worker is missing a setting. Terminal for the worker.
    "PREFLIGHT_PROVIDER_NOT_CONFIGURED",
    # The sources did not answer, or answered unusably. May recover.
    "PREFLIGHT_SOURCE_UNAVAILABLE",
    # This control plane has no such snapshot. One bad row, not a bad worker.
    "PREFLIGHT_SOURCE_SNAPSHOT_UNKNOWN",
    # The job carries no usable frozen parameters.
    "PREFLIGHT_REQUEST_UNUSABLE",
    # The executor produced something that is not an ink-blind, read-only
    # measurement. Terminal for the worker: it will do it again next time.
    "PREFLIGHT_RECEIPT_NOT_INK_BLIND",
})

# Reasons that mean "this worker cannot do this kind of work at all". They stop
# the loop rather than only ending the job.
# How a source outage is survived at the job level. The per-read retry in
# `fleet.retrying` covers a blip inside one measurement; this covers an outage
# that outlives the read. Both are bounded, and this bound is what keeps a
# source that is genuinely gone from hiding behind an endless requeue.
#
# One, not three. A preflight takes about seventy minutes, and the control
# harness stops waiting for it at `PREFLIGHT_WAIT_SECONDS`; a budget of three
# entitles a job to four attempts inside a window that fits two, so the extra
# requeues could never be used and the run would report a timeout instead of the
# outage that caused it. A bound that cannot be reached is not a bound.
# `tests/test_the_requeue_budget_fits_the_wait_that_bounds_it.py` holds the two
# constants to the same arithmetic, so neither can drift without the other.
PREFLIGHT_OUTAGE_RETRY_SECONDS = 60
PREFLIGHT_MAXIMUM_REQUEUES = 1

_TERMINAL_FOR_WORKER = frozenset({
    "PREFLIGHT_PROVIDER_NOT_CONFIGURED",
    "PREFLIGHT_RECEIPT_NOT_INK_BLIND",
})

# States that prove the job is no longer this worker's to run. The claimed
# state's own name is not frozen in the store contract, so the gate is written
# against the states that are -- and it fails closed on a missing row.
_NOT_OURS_ANY_MORE = frozenset({"PENDING", "COMPLETED", "FAILED", "CANCELLED"})

PREFLIGHT_RECEIPT_SCHEMA = "campaignx.segment_candidate_coverage_preflight.v1"


class PreflightFailure(RuntimeError):
    """A failure that already knows its own reason code.

    Whether it also stops the worker is decided by `_TERMINAL_FOR_WORKER` and
    not by the class: an unknown snapshot is one bad row and a missing setting
    is a bad host, and both arrive here carrying a code.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class PreflightConfigurationError(PreflightFailure):
    """Something a person has to change before any job here can run.

    Kept apart from every other failure for one reason: it will fail the same
    way next time, on this job and on every other one this worker claims.
    """


class PreflightSourceUnavailable(RuntimeError):
    """A transient CT/m7/MCP/S3 outage. Not a statement about the papyrus."""


class PreflightJobMoved(RuntimeError):
    """The job or its lease stopped being ours between the claim and the I/O."""


class PreflightExecutor(Protocol):
    """One ink-blind, read-only candidate-availability measurement."""

    def execute(self, snapshot: dict[str, Any], request: dict[str, Any],
                frozen_root_objects: dict[str, Any] | None = None,
                ) -> dict[str, Any]: ...


class ControlRegionPreflightExecutor:
    """The function the panel used to call, on a host that can reach the sources.

    Not a reimplementation: the measurement is exactly
    ``run_control_region_preflight``, moved.
    """

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider

    def _configured_provider(self) -> Any:
        """Refuse before the call, so the failure names a setting.

        Letting the provider raise mid-call works -- the classifier below reads
        its sentence -- but it arrives dressed as a source failure, which is the
        exact confusion that made the panel answer 503 for a service that was
        up. Asking first means the reason code is right by construction.
        """
        if self._provider is not None:
            return self._provider
        from .worker import McpSeedProvider  # noqa: PLC0415

        provider = McpSeedProvider()
        missing = [name for name, value in
                   (("VC_MCP_URL", provider.url),
                    ("VC_MCP_AUTH_TOKEN", provider.token)) if not value]
        if missing:
            raise PreflightConfigurationError(
                "PREFLIGHT_PROVIDER_NOT_CONFIGURED",
                f"the seed provider needs {' and '.join(missing)}; this host "
                "cannot reach the m7 prediction without them",
            )
        return provider

    def execute(self, snapshot: dict[str, Any], request: dict[str, Any],
                frozen_root_objects: dict[str, Any] | None = None,
                ) -> dict[str, Any]:
        # Through the module, not the name imported above. The panel called it
        # this way, so every test that replaces the measurement at its external
        # boundary replaces `candidate_preflight.run_control_region_preflight`.
        # A name bound at import ignores that, and the tests then measured for
        # real against a stand-in provider and reported the confusion as a
        # receipt that was not ink-blind.
        return candidate_preflight.run_control_region_preflight(
            snapshot, request, provider=self._configured_provider(),
            frozen_root_objects=frozen_root_objects)


def classify_preflight_failure(error: BaseException) -> str:
    """Name what went wrong, in the only vocabulary the job has.

    Structural first: a typed error carries its own reason code. The sentence
    match is a backstop for the provider's plain RuntimeError, which is what
    the deployment actually raises today and which names the two settings a
    person has to set.
    """
    if isinstance(error, PreflightFailure):
        return error.reason_code
    text = str(error)
    if "VC_MCP_URL" in text and "VC_MCP_AUTH_TOKEN" in text:
        return "PREFLIGHT_PROVIDER_NOT_CONFIGURED"
    return "PREFLIGHT_SOURCE_UNAVAILABLE"


def _ink_blind_and_read_only(receipt: object, *, depth: int = 0) -> str | None:
    """Return why this receipt may not be persisted, or None if it may.

    Recursive, because the flag at the top is not the promise. The promise is
    that nothing anywhere in the document carries an ink signal, and a
    candidate the provider handed back is as much part of the record as the
    summary above it.
    """
    if depth > 24:
        return "the receipt nests deeper than an ink-blindness check can read"
    if isinstance(receipt, dict):
        for key, value in receipt.items():
            name = str(key)
            if name == "ink_used":
                if value is not False:
                    return f"ink_used is {value!r} rather than False"
                continue
            if "ink" in name.lower():
                return f"the receipt carries an ink-bearing field: {name}"
            reason = _ink_blind_and_read_only(value, depth=depth + 1)
            if reason is not None:
                return reason
        return None
    if isinstance(receipt, (list, tuple)):
        for item in receipt:
            reason = _ink_blind_and_read_only(item, depth=depth + 1)
            if reason is not None:
                return reason
    return None


def _refusal(receipt: object) -> str | None:
    """The full finishing gate: ink-blind, read-only, and the right schema."""
    if not isinstance(receipt, dict):
        return "the executor did not return a receipt object"
    if receipt.get("schema") != PREFLIGHT_RECEIPT_SCHEMA:
        return f"the receipt schema is {receipt.get('schema')!r}"
    if receipt.get("ink_used") is not False:
        return f"ink_used is {receipt.get('ink_used')!r} rather than False"
    if receipt.get("growth_allowed") is not False:
        return "the receipt says growth was allowed; the preflight grows nothing"
    if receipt.get("state_mutation") != "NONE":
        return (f"the receipt says state_mutation is "
                f"{receipt.get('state_mutation')!r}; the preflight writes nothing")
    return _ink_blind_and_read_only(receipt)


class PreflightLeaseHeartbeat:
    """Keep the lease alive while the provider is being walked.

    A control-region preflight over a locked surface can visit thousands of
    regions; the lease has to outlive that or the job returns to PENDING under
    a worker that is still measuring it.
    """

    def __init__(self, store: Any, claim: dict[str, Any], lease_seconds: int):
        self.store = store
        self.claim = claim
        self.lease_seconds = lease_seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"fleet-preflight-heartbeat-{claim['preflight_job_id']}",
            daemon=True,
        )

    def _run(self) -> None:
        interval = max(10.0, self.lease_seconds / 3.0)
        while not self.stop_event.wait(interval):
            try:
                self.store.heartbeat_preflight(
                    self.claim["preflight_job_id"],
                    self.claim["lease_token"],
                    self.lease_seconds,
                )
            except BaseException as error:
                self.error = error
                self.stop_event.set()
                return

    def __enter__(self) -> "PreflightLeaseHeartbeat":
        self.thread.start()
        return self

    def ensure(self) -> None:
        if self.error is not None:
            raise PreflightJobMoved(
                "the preflight worker lost its lease heartbeat") from self.error

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.ensure()


def _sanitized_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in claim.items() if key != "lease_token"}


def _safe_detail(detail: object, fallback: str) -> str:
    """Normalize, then redact, every failure at the durable receipt boundary.

    `sanitize_error` only reads strings, and it wants the "Type: message" shape
    to recover the exception type -- so an exception is spelled out here rather
    than handed over as an object it would silently replace with the fallback.
    A provider sentence can carry a token, a DSN or an internal host name, and
    this is the last place before it becomes durable.
    """
    raw = (f"{type(detail).__name__}: {detail}"
           if isinstance(detail, BaseException) else str(detail))
    return qc_diagnostics.safe_message(raw, fallback)


class CandidatePreflightWorker:
    """Claim one queued candidate preflight, run it, finalize it or fail it."""

    def __init__(
        self,
        store: Any,
        worker_id: str,
        *,
        executor: PreflightExecutor | None = None,
        lease_seconds: int = 900,
        max_consecutive_failures: int = 5,
        alarm_after: tuple[int, ...] = (5, 25, 100, 500),
    ) -> None:
        self.store = store
        self.worker_id = worker_id
        self.executor: PreflightExecutor = (
            executor if executor is not None else ControlRegionPreflightExecutor())
        self.lease_seconds = lease_seconds
        self.max_consecutive_failures = max_consecutive_failures
        self.alarm_after = frozenset(alarm_after)

    # -- the check that has to happen immediately before the I/O ----------
    def _still_ours(self, claim: dict[str, Any]) -> None:
        """Re-read the job's state and re-prove the lease, now.

        ``claim_preflight`` answered both questions, and that answer became
        history the moment it returned. Between then and here a lease can
        expire, the row can go back to PENDING and another worker can take it;
        a worker acting on the claim alone runs the whole preflight and
        finalizes over somebody else's result.

        Two reads because they answer different questions. ``preflight_job``
        says what state the row is in and says nothing about who holds it -- a
        row reading CLAIMED is claimed by somebody, not necessarily by us.
        ``heartbeat_preflight`` is the contract method that takes the lease
        token, so it is the only one that can answer "is this still mine", and
        it is asked here rather than only from the background thread.
        """
        job = self.store.preflight_job(claim["preflight_job_id"])
        if job is None:
            raise PreflightJobMoved("the preflight job is no longer on this queue")
        state = str(job.get("state", ""))
        if state in _NOT_OURS_ANY_MORE:
            raise PreflightJobMoved(
                f"the preflight job moved to {state} after it was claimed")
        try:
            self.store.heartbeat_preflight(
                claim["preflight_job_id"], claim["lease_token"], self.lease_seconds)
        except PreflightJobMoved:
            raise
        except Exception as error:
            raise PreflightJobMoved(
                "the preflight lease is no longer held by this worker") from error

    def _snapshot(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Resolve the frozen source this job names, through the control plane.

        By ``source_snapshot_id`` and not by "the newest snapshot for this
        sample": the job was enqueued against one immutable source and a
        preflight measured against a different one answers a question nobody
        asked.
        """
        wanted = str(claim.get("source_snapshot_id") or "")
        sample = claim.get("sample_id")
        rows = self.store.snapshots({str(sample)} if sample else None) or []
        for row in rows:
            if str(row.get("source_snapshot_id")) == wanted:
                return dict(row)
        raise PreflightFailure(
            "PREFLIGHT_SOURCE_SNAPSHOT_UNKNOWN",
            "this control plane has no source snapshot for this preflight job",
        )

    @staticmethod
    def _request(claim: dict[str, Any]) -> dict[str, Any]:
        request = claim.get("request")
        if not isinstance(request, dict) or not request:
            raise PreflightFailure(
                "PREFLIGHT_REQUEST_UNUSABLE",
                "the preflight job carries no frozen parameters",
            )
        # The measurement is handed the frozen parameters, not the envelope they
        # arrive in. The panel enqueues identity, source lock and parameters
        # together; passing the whole document meant every field the measurement
        # reads was one level too deep, and the deployment's first real job died
        # on `KeyError: 'region_center_xyz'` -- reported as a source outage,
        # because a missing key looks like one to the classifier.
        parameters = request.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            raise PreflightFailure(
                "PREFLIGHT_REQUEST_UNUSABLE",
                "the preflight job carries no frozen parameters",
            )
        # A courier, not an author. The parameters are the panel's frozen
        # control policy and a worker that edits them forges the run.
        return parameters

    def _fail(self, claim: dict[str, Any], reason_code: str,
              detail: object) -> dict[str, Any]:
        if reason_code not in PREFLIGHT_REASON_CODES:
            reason_code = "PREFLIGHT_SOURCE_UNAVAILABLE"
        # Redacted once, and stored as well as returned. The returned copy lives
        # in this process; the queue's is the one an operator reads, and without
        # it a real failure said PREFLIGHT_SOURCE_UNAVAILABLE and nothing more.
        safe = _safe_detail(detail, "the preflight failure had no safe detail")
        if reason_code == "PREFLIGHT_SOURCE_UNAVAILABLE":
            requeued = self._requeue_outage(claim, safe)
            if requeued is not None:
                return requeued
        try:
            stored = self.store.fail_preflight(
                claim["preflight_job_id"], claim["lease_token"], reason_code,
                detail=safe)
        except Exception as rejected:
            # `fail_preflight` fails closed, so the ordinary way to be refused
            # here is that the lease is no longer ours -- somebody else owns
            # the job and gets to decide how it ends.
            return self._stood_down(claim, rejected)
        return {
            "schema": "campaignx.segment_preflight_worker_receipt.v1",
            **(stored or {}),
            "status": ("BLOCKED_CONFIGURATION"
                       if reason_code in _TERMINAL_FOR_WORKER else "FAILED"),
            "reason_code": reason_code,
            "preflight_job_id": claim["preflight_job_id"],
            "worker_id": self.worker_id,
            "error": safe,
            "generated_at_utc": utc_now(),
            "no_scientific_conclusion": True,
            "ink_used": False,
            "worker_stopped": reason_code in _TERMINAL_FOR_WORKER,
        }

    def _requeue_outage(self, claim: dict[str, Any],
                        safe: str) -> dict[str, Any] | None:
        """Give the job back to the queue instead of ending it.

        This is the consequence the classification never had. The worker has
        always said a source outage "may recover" and then called the terminal
        path anyway, so one dropped connection ended a measurement that had run
        for over an hour.

        Returns None when this store has no such lane, which is the honest
        fallback for a control plane that predates it: the old terminal
        behaviour, rather than a worker that crashes and loses the job.
        """
        requeue = getattr(self.store, "requeue_preflight_source_unavailable", None)
        if requeue is None:
            return None
        receipt = {
            "schema": "campaignx.segment_preflight_outage.v1",
            "error": safe,
            "worker_id": self.worker_id,
            "generated_at_utc": utc_now(),
            "no_scientific_conclusion": True,
            "ink_used": False,
        }
        try:
            stored = requeue(
                claim["preflight_job_id"], claim["lease_token"], receipt,
                retry_delay_seconds=PREFLIGHT_OUTAGE_RETRY_SECONDS,
                maximum_requeues=PREFLIGHT_MAXIMUM_REQUEUES)
        except Exception as rejected:
            # Same as the terminal path: the ordinary way to be refused here is
            # that the lease is no longer ours.
            return self._stood_down(claim, rejected)
        status = str((stored or {}).get("status"))
        if status != "RETRYABLE_PREFLIGHT_SOURCE_UNAVAILABLE":
            # The budget is spent, and the store has already made the job
            # terminal and released the lease. Falling through to
            # `fail_preflight` here would be refused for a lease we no longer
            # hold and would report a stand-down that did not happen.
            return {
                "schema": "campaignx.segment_preflight_worker_receipt.v1",
                **(stored or {}),
                "status": "FAILED",
                "reason_code": "PREFLIGHT_SOURCE_UNAVAILABLE",
                "preflight_job_id": claim["preflight_job_id"],
                "worker_id": self.worker_id,
                "error": safe,
                "generated_at_utc": utc_now(),
                "no_scientific_conclusion": True,
                "ink_used": False,
                "worker_stopped": False,
            }
        return {
            "schema": "campaignx.segment_preflight_worker_receipt.v1",
            **stored,
            "status": "REQUEUED",
            "reason_code": "PREFLIGHT_SOURCE_UNAVAILABLE",
            "preflight_job_id": claim["preflight_job_id"],
            "worker_id": self.worker_id,
            "error": safe,
            "generated_at_utc": utc_now(),
            "no_scientific_conclusion": True,
            "ink_used": False,
            "worker_stopped": False,
        }

    def _stood_down(self, claim: dict[str, Any],
                    refusal: BaseException) -> dict[str, Any]:
        """Walk away without touching the job.

        The lease is not ours, so ``fail_preflight`` would be rejected by a
        store that fails closed and would be vandalism on one that did not.
        Whoever holds it now gets to finish it.
        """
        return {
            "schema": "campaignx.segment_preflight_worker_receipt.v1",
            "status": "STOOD_DOWN",
            "reason_code": "PREFLIGHT_JOB_MOVED",
            "preflight_job_id": claim["preflight_job_id"],
            "worker_id": self.worker_id,
            "error": _safe_detail(refusal, "the stand-down had no safe detail"),
            "generated_at_utc": utc_now(),
            "no_scientific_conclusion": True,
            "ink_used": False,
        }

    def _measure(self, claim: dict[str, Any]) -> dict[str, Any]:
        # Before the snapshot, before the provider, before anything this worker
        # could be said to have measured.
        self._still_ours(claim)
        request = self._request(claim)
        snapshot = self._snapshot(claim)
        # The frozen roots stay in the envelope, beside the parameters rather
        # than among them: they are what the receipt binds to, not something the
        # measurement is free to choose.
        envelope = claim.get("request")
        roots = envelope.get("frozen_root_objects") if isinstance(envelope, dict) else None
        with PreflightLeaseHeartbeat(self.store, claim, self.lease_seconds) as beat:
            receipt = self.executor.execute(snapshot, request, frozen_root_objects=roots)
            beat.ensure()
        refusal = _refusal(receipt)
        if refusal is not None:
            return self._fail(claim, "PREFLIGHT_RECEIPT_NOT_INK_BLIND", refusal)
        finalized = self.store.finalize_preflight(
            claim["preflight_job_id"], claim["lease_token"], receipt)
        return {
            "schema": "campaignx.segment_preflight_worker_receipt.v1",
            **(finalized or {}),
            "status": "COMPLETED",
            "preflight_job_id": claim["preflight_job_id"],
            "worker_id": self.worker_id,
            "claim": _sanitized_claim(claim),
            "preflight_status": receipt.get("status"),
            "ink_used": False,
            "generated_at_utc": utc_now(),
        }

    def run_one(self) -> dict[str, Any] | None:
        claim = self.store.claim_preflight(self.worker_id, self.lease_seconds)
        if claim is None:
            return None
        # One handler around the whole attempt rather than one per step. The
        # lease heartbeat can fail on the way out of its own context manager,
        # after the body has already returned, and a handler inside the block
        # cannot see that -- which is how the failure that must not crash the
        # loop crashes the loop.
        try:
            return self._measure(claim)
        except PreflightJobMoved as refusal:
            return self._stood_down(claim, refusal)
        except Exception as error:
            return self._fail(claim, classify_preflight_failure(error), error)

    def _alarm(self, consecutive: int, last: dict[str, Any]) -> None:
        print(json.dumps({
            "schema": "campaignx.segment_preflight_worker_alarm.v1",
            "status": "NO_PREFLIGHT_MEASURED",
            "worker_id": self.worker_id,
            "consecutive_failures": consecutive,
            "last_reason_code": str(last.get("reason_code", "")),
            "generated_at_utc": utc_now(),
            "note": "every claim since this worker last succeeded has failed. "
                    "A run of failures this long is usually not an outage.",
        }, sort_keys=True), file=sys.stderr, flush=True)

    def run(
        self,
        max_jobs: int | None = None,
        *,
        idle_exit: bool = True,
        poll_seconds: float = 10.0,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        consecutive = 0
        while max_jobs is None or len(results) < max_jobs:
            result = self.run_one()
            if result is None:
                if idle_exit:
                    break
                time.sleep(poll_seconds)
                continue
            results.append(result)
            if result.get("status") == "COMPLETED":
                consecutive = 0
                continue
            if result.get("worker_stopped"):
                # Terminal for this worker. Claiming again would fail the next
                # job in the same second and empty the queue into FAILED with
                # nothing measured and nobody told why.
                break
            # A stand-down counts too. Losing a race is ordinary and the
            # counter resets on the first success, but a worker that claims
            # and stands down forever is the spin wearing a politer word.
            consecutive += 1
            if consecutive in self.alarm_after:
                self._alarm(consecutive, result)
            if (self.max_consecutive_failures
                    and consecutive >= self.max_consecutive_failures):
                self._alarm(consecutive, result)
                results[-1] = {**results[-1], "worker_stopped": True}
                break
        return results


def preflight_worker_environment_report() -> dict[str, Any]:
    """Say whether this host can run a preflight, without saying the token.

    For the operator who has to answer "why did that job come back
    PREFLIGHT_PROVIDER_NOT_CONFIGURED" from a shell on the wrong machine.
    """
    return {
        "schema": "campaignx.segment_preflight_worker_environment.v1",
        "vc_mcp_url_set": bool(os.environ.get("VC_MCP_URL")),
        "vc_mcp_auth_token_set": bool(os.environ.get("VC_MCP_AUTH_TOKEN")),
        "generated_at_utc": utc_now(),
    }
