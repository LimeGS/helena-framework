from __future__ import annotations

import copy
import threading
from typing import Any, Protocol


class DiscoveryPreRunRetryable(RuntimeError):
    """Provider preparation failed before any discovery execution began."""


class DiscoveryPostRunningControlError(RuntimeError):
    """A post-RUNNING terminal state could not be confirmed durably."""


class PreparedDiscoveryProvider(Protocol):
    def execute(self, provider_request: dict[str, Any]) -> bytes: ...


class DiscoveryProvider(Protocol):
    def prepare(self) -> PreparedDiscoveryProvider: ...


class _ClaimHeartbeat:
    def __init__(
        self, *, store: Any, run_handle: Any, lease_seconds: int,
        interval_seconds: float,
    ):
        self.store = store
        self.run_handle = run_handle
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"first-letters-discovery-heartbeat-{run_handle.run_id}",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.store.heartbeat_first_letters_discovery_evidence_run(
                    run_handle=self.run_handle,
                    lease_seconds=self.lease_seconds,
                )
            except BaseException as error:  # captured for the owner thread
                self._error = error
                self._stop.set()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("active discovery claim heartbeat failed") from self._error


def _completed_result(
    *, run_handle: Any, evidence: dict[str, Any], recovered: bool,
) -> dict[str, Any]:
    return {
        "schema": "campaignx.first_letters_discovery_controller_result.v1",
        "mode": "shadow", "state": "COMPLETED",
        "run_id": run_handle.run_id,
        "evidence_set_id": evidence["evidence_set_id"],
        "canonical_admission": "PROHIBITED",
        "recovered_after_response_loss": recovered,
    }


def _incomplete_result(
    *, run_handle: Any, reason: str,
) -> dict[str, Any]:
    return {
        "schema": "campaignx.first_letters_discovery_controller_result.v1",
        "mode": "shadow", "state": "CONTROL_INCOMPLETE",
        "run_id": run_handle.run_id, "evidence_set_id": None,
        "canonical_admission": "PROHIBITED",
        "incomplete_reason": reason,
    }


class FirstLettersDiscoveryWorker:
    """Execute one noncanonical item exclusively through public A3 claims."""

    def __init__(
        self, *, store: Any, provider: DiscoveryProvider,
        pre_run_attempts: int = 1,
        heartbeat_interval_seconds: float | None = None,
    ):
        if (isinstance(pre_run_attempts, bool)
                or not isinstance(pre_run_attempts, int)
                or pre_run_attempts < 1 or pre_run_attempts > 3):
            raise ValueError("pre-run attempts must be within 1..3")
        self.store = store
        self.provider = provider
        self.pre_run_attempts = pre_run_attempts
        self.heartbeat_interval_seconds = heartbeat_interval_seconds

    def _prepare(self) -> PreparedDiscoveryProvider:
        for attempt in range(1, self.pre_run_attempts + 1):
            try:
                prepared = self.provider.prepare()
            except DiscoveryPreRunRetryable:
                if attempt == self.pre_run_attempts:
                    raise
                continue
            if not callable(getattr(prepared, "execute", None)):
                raise ValueError("prepared discovery provider has no execute method")
            return prepared
        raise RuntimeError("unreachable pre-run preparation state")

    def _terminal_result(self, run_handle: Any, reason: str) -> dict[str, Any]:
        """Confirm one durable terminal state; never invent an incomplete one."""

        last_error: BaseException | None = None
        for attempt in range(2):
            try:
                status = (
                    self.store
                    .mark_first_letters_discovery_evidence_run_incomplete(
                        run_handle=run_handle, reason=reason,
                    )
                )
            except BaseException as error:
                last_error = error
                try:
                    status = (
                        self.store
                        .read_first_letters_discovery_evidence_run_status(
                            run_handle.run_id
                        )
                    )
                except BaseException as read_error:
                    raise DiscoveryPostRunningControlError(
                        "discovery terminal state could not be confirmed"
                    ) from read_error
            state = status.get("state")
            if state == "CONTROL_INCOMPLETE":
                if status.get("incomplete_reason") != reason:
                    raise DiscoveryPostRunningControlError(
                        "discovery terminal readback reason is inconsistent"
                    ) from last_error
                return _incomplete_result(
                    run_handle=run_handle, reason=reason,
                )
            if state == "COMPLETED":
                try:
                    evidence = (
                        self.store.read_first_letters_discovery_evidence_run(
                            run_handle.run_id
                        )
                    )
                except BaseException as read_error:
                    raise DiscoveryPostRunningControlError(
                        "discovery completed state could not be read exactly"
                    ) from read_error
                return _completed_result(
                    run_handle=run_handle, evidence=evidence, recovered=True,
                )
            if state != "RUNNING" or attempt == 1:
                break
            # An exact RUNNING readback proves that retrying only the terminal
            # write is safe.  Provider execution is never retried.
        raise DiscoveryPostRunningControlError(
            "discovery terminal state could not be confirmed"
        ) from last_error

    def _heartbeat_interval(self, lease_seconds: int) -> float:
        interval = self.heartbeat_interval_seconds
        if interval is None:
            interval = max(1.0, lease_seconds / 3.0)
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or interval <= 0 or interval >= lease_seconds):
            raise ValueError("heartbeat interval must be positive and below lease")
        return float(interval)

    def run_job(self, *, job_id: str, lease_seconds: int) -> dict[str, Any]:
        """Claim and revalidate an immutable job before provider preparation."""

        from .discovery_bridge import (
            validate_first_letters_discovery_job_claim,
        )

        interval = self._heartbeat_interval(lease_seconds)
        claim = self.store.claim_first_letters_discovery_job(
            job_id=job_id, lease_seconds=lease_seconds,
        )
        claim = validate_first_letters_discovery_job_claim(claim)
        self.store.revalidate_first_letters_discovery_job_claim(claim=claim)
        prepared = self._prepare()
        return self._execute_claimed(
            run_handle=claim._run_handle, prepared=prepared,
            lease_seconds=lease_seconds, interval=interval,
        )

    def run_item(
        self, *, lease_seconds: int, reservation_id: str, item_id: str,
        profile_bytes: bytes,
    ) -> dict[str, Any]:
        del lease_seconds, reservation_id, item_id, profile_bytes
        raise ValueError("DISCOVERY_JOB_ID_REQUIRED")

    def _execute_claimed(
        self, *, run_handle: Any, prepared: PreparedDiscoveryProvider,
        lease_seconds: int, interval: float,
    ) -> dict[str, Any]:
        try:
            self.store.start_first_letters_discovery_evidence_run(
                run_handle=run_handle
            )
        except BaseException:
            # A lost start response may already have committed RUNNING.  The
            # terminal API plus exact readback distinguishes that case without
            # ever invoking the provider.
            return self._terminal_result(
                run_handle, "START_RESPONSE_AMBIGUOUS_AFTER_RUNNING"
            )
        # Renew immediately so a slow provider receives the full configured
        # lease after its claim/start transactions have completed.
        try:
            self.store.heartbeat_first_letters_discovery_evidence_run(
                run_handle=run_handle, lease_seconds=lease_seconds,
            )
        except BaseException:
            return self._terminal_result(
                run_handle, "ACTIVE_CLAIM_HEARTBEAT_FAILED"
            )
        heartbeat = _ClaimHeartbeat(
            store=self.store, run_handle=run_handle,
            lease_seconds=lease_seconds, interval_seconds=interval,
        )
        try:
            heartbeat.start()
        except BaseException:
            return self._terminal_result(
                run_handle, "ACTIVE_CLAIM_HEARTBEAT_FAILED"
            )
        try:
            provider_response_bytes = prepared.execute(
                copy.deepcopy(run_handle.provider_request)
            )
            if not isinstance(provider_response_bytes, bytes):
                raise ValueError("discovery provider response must be exact bytes")
        except BaseException:
            heartbeat.stop()
            return self._terminal_result(
                run_handle, "PROVIDER_RESPONSE_AMBIGUOUS_AFTER_RUNNING"
            )
        heartbeat.stop()
        try:
            heartbeat.raise_if_failed()
            # One final renewal closes the provider/complete hand-off.  The
            # completion transaction then owns the row lock while the claimed
            # executor measures and commits.
            self.store.heartbeat_first_letters_discovery_evidence_run(
                run_handle=run_handle, lease_seconds=lease_seconds,
            )
        except BaseException:
            return self._terminal_result(
                run_handle, "ACTIVE_CLAIM_HEARTBEAT_FAILED"
            )
        try:
            evidence = self.store.complete_first_letters_discovery_evidence_run(
                run_handle=run_handle,
                provider_response_bytes=provider_response_bytes,
            )
        except BaseException:
            try:
                evidence = self.store.read_first_letters_discovery_evidence_run(
                    run_handle.run_id
                )
            except KeyError:
                reason = "COMPLETION_FAILED_AFTER_RUNNING"
                return self._terminal_result(run_handle, reason)
            except BaseException:
                reason = "COMPLETION_READBACK_AMBIGUOUS_AFTER_RUNNING"
                return self._terminal_result(run_handle, reason)
            return _completed_result(
                run_handle=run_handle, evidence=evidence, recovered=True,
            )
        return _completed_result(
            run_handle=run_handle, evidence=evidence, recovered=False,
        )
