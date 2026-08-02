from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from .common import file_sha256, read_json, utc_now, write_json_atomic
from .store import QC_OUTCOME_STATES


class QcExecutor(Protocol):
    """One scientific surface-QC execution; it never decides letter identity."""

    def execute(self, claim: dict[str, Any], attempt_dir: Path) -> dict[str, Any]: ...


class QcLeaseHeartbeat:
    def __init__(self, store: Any, claim: dict[str, Any], lease_seconds: int):
        self.store = store
        self.claim = claim
        self.lease_seconds = lease_seconds
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=f"fleet-qc-heartbeat-{claim['qc_job_id']}",
            daemon=True,
        )

    def _run(self) -> None:
        interval = max(10.0, self.lease_seconds / 3.0)
        while not self.stop_event.wait(interval):
            try:
                self.store.heartbeat_qc(
                    self.claim["qc_job_id"],
                    self.claim["lease_token"],
                    self.lease_seconds,
                )
            except BaseException as error:
                self.error = error
                self.stop_event.set()
                return

    def __enter__(self) -> "QcLeaseHeartbeat":
        self.thread.start()
        return self

    def ensure(self) -> None:
        if self.error is not None:
            raise RuntimeError("QC worker lost its lease heartbeat") from self.error

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.ensure()


def _sanitized_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in claim.items() if key != "lease_token"}


def _final_result(claim: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    outcome = execution.get("outcome")
    if outcome not in QC_OUTCOME_STATES:
        raise RuntimeError(f"QC executor returned an unsupported outcome: {outcome!r}")
    manifest_path = Path(str(execution.get("evidence_manifest_path", ""))).resolve()
    if not manifest_path.is_file():
        raise RuntimeError("QC executor did not produce an evidence manifest")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("QC evidence manifest is not a JSON object")
    if manifest.get("surface_id") != claim["surface_id"]:
        raise RuntimeError("QC evidence manifest belongs to another surface")
    if manifest.get("outcome") != outcome:
        raise RuntimeError("QC evidence manifest outcome differs from executor result")
    ink_used = execution.get("ink_used")
    if not isinstance(ink_used, bool) or manifest.get("ink_used") is not ink_used:
        raise RuntimeError("QC executor and evidence manifest disagree about ink use")
    evidence_uri = execution.get("evidence_uri")
    if not isinstance(evidence_uri, str) or not evidence_uri.strip():
        raise RuntimeError("QC executor did not publish a durable evidence URI")
    return {
        "schema": "campaignx.segment_qc_result.v1",
        "surface_id": claim["surface_id"],
        "outcome": outcome,
        "evidence_manifest_sha256": file_sha256(manifest_path),
        "evidence_uri": evidence_uri,
        "ink_used": ink_used,
        "completed_at_utc": utc_now(),
        "executor_receipt": execution,
    }


# sysexits.h EX_CONFIG, the code the adapter uses to say "this is a setting".
EX_CONFIG = 78


class QcConfigurationError(RuntimeError):
    """The adapter refused because something is configured wrong.

    Kept apart from every other failure for one reason: it will fail the same
    way next time. Requeuing it produces a worker that claims a job, fails in a
    second, requeues and claims it again -- which is what two GPUs did for two
    days over a profile hash nobody had updated, while the queue showed those
    jobs as PENDING and the cards showed as busy.
    """


def _adapter_complaint(output: str) -> str:
    """The adapter's own words for what is misconfigured.

    It writes a campaignx.segment_qc_configuration_error.v1 line; the rest of
    its output is a traceback nobody needs to read to know what to change.
    """
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            report = json.loads(line)
        except ValueError:
            continue
        if report.get("status") == "BLOCKED_CONFIGURATION" and report.get("error"):
            return str(report["error"])
    return "the adapter reported a configuration error and did not say which"


class SubprocessQcExecutor:
    """Invoke a fail-closed adapter implementing the Helena Framework QC contract."""

    def __init__(self, executable: Path | str, *, timeout_seconds: int = 7200):
        self.executable = Path(executable).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)

    def execute(self, claim: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
        input_path = attempt_dir / "QC_INPUT.json"
        output_dir = attempt_dir / "scientific-output"
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(
            input_path,
            {
                "schema": "campaignx.segment_qc_input.v1",
                "qc_job": _sanitized_claim(claim),
                "policy": {
                    "no_automatic_ink_acceptance": True,
                    "no_automatic_letter_acceptance": True,
                    "retained_means_review_only": True,
                },
            },
        )
        command = [
            *( [sys.executable] if self.executable.suffix == ".py" else [] ),
            str(self.executable),
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
        ]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        (attempt_dir / "scientific-executor.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode == EX_CONFIG:
            # The adapter prints a JSON line naming what is wrong. Carry that up
            # rather than the exit code alone: "exit code 78" is not something
            # anybody can act on, and the point of this path is that a person
            # acts on it.
            raise QcConfigurationError(_adapter_complaint(completed.stdout))
        if completed.returncode:
            raise RuntimeError(
                f"scientific QC adapter failed with exit code {completed.returncode}"
            )
        result_path = output_dir / "QC_EXECUTOR_RESULT.json"
        if not result_path.is_file():
            raise RuntimeError("scientific QC adapter omitted QC_EXECUTOR_RESULT.json")
        result = read_json(result_path)
        if not isinstance(result, dict):
            raise RuntimeError("scientific QC adapter result is not a JSON object")
        return result


class FixtureQcExecutor:
    """Explicitly non-scientific executor used only by tests and demos."""

    def __init__(self, outcome: str = "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL"):
        if outcome not in QC_OUTCOME_STATES:
            raise ValueError(outcome)
        self.outcome = outcome

    def execute(self, claim: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
        evidence = attempt_dir / "fixture-evidence"
        evidence.mkdir(parents=True, exist_ok=False)
        manifest = evidence / "EVIDENCE_MANIFEST.json"
        ink_used = self.outcome != "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS"
        write_json_atomic(
            manifest,
            {
                "schema": "campaignx.segment_qc_evidence_manifest.v1",
                "surface_id": claim["surface_id"],
                "outcome": self.outcome,
                "ink_used": ink_used,
                "fixture_only": True,
            },
        )
        return {
            "outcome": self.outcome,
            "ink_used": ink_used,
            "evidence_manifest_path": str(manifest),
            "evidence_uri": manifest.as_uri(),
            "fixture_only": True,
        }


class SurfaceQcWorker:
    def __init__(
        self,
        store: Any,
        worker_id: str,
        executor: QcExecutor,
        run_root: Path,
        *,
        lease_seconds: int = 900,
        retry_delay_seconds: int = 300,
        profile_id: str | None = None,
        alarm_after: tuple[int, ...] = (5, 25, 100, 500),
    ):
        self.store = store
        self.worker_id = worker_id
        self.executor = executor
        self.run_root = Path(run_root)
        self.lease_seconds = lease_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.profile_id = profile_id
        # Widening steps rather than every failure: the point is to be noticed,
        # and a line per failure is how 3118 of them went unread.
        self.alarm_after = frozenset(alarm_after)

    def run_one(self) -> dict[str, Any] | None:
        claim = self.store.claim_qc(
            self.worker_id, self.lease_seconds, profile_id=self.profile_id
        )
        if claim is None:
            return None
        attempt_dir = (
            self.run_root
            / claim["qc_job_id"]
            / f"{utc_now().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:12]}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(attempt_dir / "CLAIMED_QC_JOB.json", _sanitized_claim(claim))
        with QcLeaseHeartbeat(self.store, claim, self.lease_seconds) as heartbeat:
            try:
                execution = self.executor.execute(claim, attempt_dir)
                heartbeat.ensure()
                result = _final_result(claim, execution)
                write_json_atomic(attempt_dir / "QC_RESULT.json", result)
                finalized = self.store.finalize_qc(
                    claim["qc_job_id"],
                    claim["lease_token"],
                    result["outcome"],
                    result,
                )
                receipt = {
                    "schema": "campaignx.segment_qc_worker_receipt.v1",
                    **finalized,
                    "attempt_dir": str(attempt_dir),
                }
                write_json_atomic(attempt_dir / "QC_WORKER_RECEIPT.json", receipt)
                return receipt
            except QcConfigurationError as error:
                heartbeat.ensure()
                blocked = {
                    "schema": "campaignx.segment_qc_blocked_configuration.v1",
                    "status": "BLOCKED_CONFIGURATION",
                    "qc_job_id": claim["qc_job_id"],
                    "surface_id": claim["surface_id"],
                    "error": str(error),
                    "generated_at_utc": utc_now(),
                    "no_scientific_conclusion": True,
                }
                write_json_atomic(
                    attempt_dir / "BLOCKED_CONFIGURATION_RECEIPT.json", blocked)
                return self.store.block_qc_configuration(
                    claim["qc_job_id"], claim["lease_token"], blocked)
            except BaseException as error:
                heartbeat.ensure()
                retry = {
                    "schema": "campaignx.segment_qc_retryable_outage.v1",
                    "status": "RETRYABLE_QC_UNAVAILABLE",
                    "qc_job_id": claim["qc_job_id"],
                    "surface_id": claim["surface_id"],
                    "error": f"{type(error).__name__}: {error}",
                    "generated_at_utc": utc_now(),
                    "no_scientific_conclusion": True,
                }
                write_json_atomic(attempt_dir / "RETRYABLE_QC_RECEIPT.json", retry)
                return self.store.requeue_qc_unavailable(
                    claim["qc_job_id"],
                    claim["lease_token"],
                    retry,
                    retry_delay_seconds=self.retry_delay_seconds,
                )

    def run(
        self,
        max_jobs: int | None = None,
        *,
        idle_exit: bool = True,
        poll_seconds: float = 10.0,
    ) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        # A worker that fails every claim looks identical to a busy one from the
        # outside: the queue shows PENDING, the card shows utilisation. The
        # blocked-configuration path above covers the failure that is knowably
        # permanent; this covers the rest, because "retryable" repeated without
        # a single success is its own kind of stuck.
        consecutive_retryable = 0
        while max_jobs is None or len(completed) < max_jobs:
            result = self.run_one()
            if result is None:
                if idle_exit:
                    break
                time.sleep(poll_seconds)
                continue
            if result.get("status") == "RETRYABLE_QC_UNAVAILABLE":
                consecutive_retryable += 1
                if consecutive_retryable in self.alarm_after:
                    print(json.dumps({
                        "schema": "campaignx.segment_qc_worker_alarm.v1",
                        "status": "NO_SURFACE_MEASURED",
                        "worker_id": self.worker_id,
                        "consecutive_retryable_failures": consecutive_retryable,
                        "last_error": str(result.get("error", "")),
                        "generated_at_utc": utc_now(),
                        "note": "every claim since this worker last succeeded has "
                                "failed and been requeued. An outage this long is "
                                "usually not an outage.",
                    }, sort_keys=True), file=sys.stderr, flush=True)
            else:
                consecutive_retryable = 0
            completed.append(result)
        return completed
