from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

from framework.contracts import qc_diagnostics

from . import surface_routing
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


class QcMisroutedSurface(RuntimeError):
    """The queue offered this surface; the surface's own routing says no.

    Terminal like QcConfigurationError and for the same reason -- it will decide
    the same way next time -- but it is not a configuration mistake and it is
    emphatically not a measurement. A surface below the effort floor is not bad,
    not empty and not answered; it is too small for the question, and the record
    has to say only that.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class QcAdapterExecutionError(RuntimeError):
    def __init__(
        self,
        exit_code: int,
        error_type: str | None,
        safe_error: str | None,
    ) -> None:
        super().__init__(
            f"scientific QC adapter failed with exit code {exit_code}"
        )
        self.exit_code = exit_code
        self.error_type = error_type
        self.safe_error = safe_error


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
            _error_type, safe_error = qc_diagnostics.sanitize_error(
                str(report["error"])
            )
            if safe_error:
                return safe_error
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
            error_type, safe_error = qc_diagnostics.extract_last_python_exception(
                completed.stdout
            )
            raise QcAdapterExecutionError(
                completed.returncode,
                error_type,
                safe_error,
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


# What a card that is full actually says, across the layers that report it.
# Matched on the message rather than the exception type because it arrives as a
# subprocess's stderr, where the type is gone and only the text survives.
_GPU_EXHAUSTION_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)

GPU_MEMORY_EXHAUSTED = "GPU_MEMORY_EXHAUSTED"


def is_gpu_exhaustion(text: object) -> bool:
    """Whether this failure is a card with no room on it.

    Deliberately narrow. Calling every failure GPU exhaustion would produce
    the same unreadable receipt with a new name on it, which is the problem
    this exists to fix -- `RETRYABLE_QC_UNAVAILABLE` with "command failed with
    exit code 1" was equally true of a full card, a dead bucket and a syntax
    error, and telling them apart took reading 7,463 receipts and the logs
    underneath them.
    """
    lowered = str(text).lower()
    return any(marker in lowered for marker in _GPU_EXHAUSTION_MARKERS)


def parse_gpu_memory(text: str) -> list[tuple[int, int]]:
    """(free, total) MiB per card, from nvidia-smi's own CSV."""
    rows: list[tuple[int, int]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"unexpected nvidia-smi memory row: {line!r}")
        try:
            rows.append((int(fields[0]), int(fields[1])))
        except ValueError as error:
            raise RuntimeError(f"invalid GPU memory value: {line!r}") from error
    return rows


def gpu_memory(*, nvidia_smi: str = "nvidia-smi") -> list[tuple[int, int]] | None:
    """Free and total VRAM per card, or None when that cannot be read.

    `memory.free`, not `memory.total`. The GPU preflight in
    run_gpu_tier_supervisor asks for total, which is why a shared card was
    invisible to it: a 6 GiB GTX 1660 passes a 6 GiB minimum while 4.8 GiB of
    it belongs to a llama.cpp server.

    None rather than an empty list or zeroes: a host with no nvidia-smi has
    not told us there is no room, it has told us nothing, and those must not
    read the same to the caller.
    """
    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        rows = parse_gpu_memory(completed.stdout)
    except RuntimeError:
        return None
    return rows or None


def _with_reason(result: dict[str, Any], reason_code: str | None) -> dict[str, Any]:
    """Carry the worker's own reason out with the store's queue result.

    The store owns the queue transition and the shape it reports; which
    failure this was is an observation about the attempt, so it travels beside
    that rather than widening the store's contract.
    """
    if reason_code and isinstance(result, dict):
        result = {**result, "reason_code": reason_code}
    return result


def _retryable_error(error: BaseException) -> str:
    if isinstance(error, QcAdapterExecutionError):
        raw = error.safe_error or (
            "QcAdapterExecutionError: scientific QC adapter failed "
            f"with exit code {error.exit_code}"
        )
    else:
        raw = f"{type(error).__name__}: {error}"
    _error_type, safe_error = qc_diagnostics.sanitize_error(raw)
    return safe_error or "RuntimeError: retryable QC failure had no safe detail"


def _persisted_error(raw: object, fallback: str) -> str:
    """Normalize every error again at the durable receipt/store boundary."""
    return qc_diagnostics.safe_message(raw, fallback)


# What a failed attempt is allowed to leave behind. Receipts, logs and the small
# JSON that says what happened; not the render.
#
# A render is reproducible from the CT and the surface. The receipt is not, and
# it is the entire record that this attempt happened at all -- so the receipt
# stays and the bulk goes. Measured on gpu-1 on 2026-08-22: 7,463 failed
# attempts had kept 194.1 GB of renders belonging to attempts whose own
# receipts say "no_scientific_conclusion": true, on a disk whose exhaustion
# stopped the QC container, the fleet and the deployment pipeline alike.
_KEEP_SUFFIXES = (".json", ".log", ".txt", ".stdout", ".stderr")
_KEEP_BYTES = 256 * 1024


def _discard_bulk_output(attempt_dir: Path) -> dict[str, int]:
    """Remove a failed attempt's bulk output, keeping what explains it.

    Best-effort by construction, and deliberately after the receipt is already
    written: a cleanup that fails must cost disk, never the record of why the
    attempt failed. Nothing here may propagate -- the caller is on its way to
    requeue the job, and a delete that raises would turn a retryable outage
    into an unhandled error.
    """
    removed = kept = 0
    try:
        for path in sorted(attempt_dir.rglob("*"), reverse=True):
            try:
                if path.is_dir():
                    # Empty by now if everything under it went.
                    try:
                        path.rmdir()
                    except OSError:
                        pass
                    continue
                if path.suffix in _KEEP_SUFFIXES and path.stat().st_size <= _KEEP_BYTES:
                    kept += 1
                    continue
                path.unlink()
                removed += 1
            except OSError:
                # One unreadable file is not a reason to keep the other 158 MB.
                continue
    except OSError:
        pass
    return {"removed": removed, "kept": kept}


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
        stop_after_retryable: int | None = None,
        minimum_free_vram_mib: int | None = None,
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
        # None keeps the historical behaviour for every caller that never asked
        # for a brake. A deployment that wants one sets it; see `run`.
        self.stop_after_retryable = stop_after_retryable
        # None keeps the historical behaviour. A deployment that shares its
        # cards with anything else sets this to what a screening pass needs.
        self.minimum_free_vram_mib = minimum_free_vram_mib

    def _vram_shortfall(self) -> tuple[int, int] | None:
        """The best card's (free, total) when it is below the floor, else None.

        The best card rather than the first: a job runs on one GPU, so a full
        card beside an empty one is not an outage.

        Fails open. A host whose nvidia-smi cannot be read has told us nothing,
        not that there is no room, and a worker that refuses every job because
        a binary is missing has replaced an outage with a worse one.
        """
        if self.minimum_free_vram_mib is None:
            return None
        cards = gpu_memory()
        if not cards:
            return None
        free, total = max(cards, key=lambda row: row[0])
        if free >= self.minimum_free_vram_mib:
            return None
        return free, total

    def _current_surface(self, surface_id: str) -> dict[str, Any]:
        """Read the surface row again, through the lineage boundary, now.

        claim_qc resolved lineage too. That read is history the moment it
        returns, and the boundary that matters is the one immediately before a
        card is pointed at the papyrus. The same read carries the measurement
        the route is decided from, so both facts come from one instant rather
        than from two reads a worker could be preempted between.
        """
        reader = getattr(self.store, "surface_artifact", None)
        if not callable(reader):
            raise QcMisroutedSurface(
                "LINEAGE_UNRESOLVABLE_ON_THIS_CONTROL_PLANE",
                "this control plane cannot resolve the surface's lineage",
            )
        resolved = reader(surface_id, boundary="PHYSICAL_QC_CLAIM_RESOLUTION")
        return dict((resolved or {}).get("payload") or {})

    def _current_route(self, surface_id: str, surface: dict[str, Any]) -> None:
        """Ask, now, whether this surface is standard physical-QC work.

        Not the claim, which carries the queue's opinion from whenever the job
        was enqueued, and not the job state, which is that opinion written down.
        Between enqueue and here a surface can be re-measured, resumed, regrown
        or imported again.

        A stored receipt is the answer where there is one. Where there is not,
        the router is asked directly, from the area this row is carrying right
        now and under the same frozen policy -- because today only imports write
        a receipt and nothing the fleet finalizes has one, so a rule of "no
        receipt, no opinion" would make this gate hold for exactly the surfaces
        nobody grew. What the stored receipt adds is proof the decision was made
        once, at creation, and not re-litigated by whoever is claiming; it is
        strictly better and it is what should exist. Until it does, refusing to
        decide is the one option that puts a 0.02 cm2 surface on the ink screen.

        An unmeasured surface is refused either way. Nothing can route it, and a
        physical-QC verdict over an area nobody measured is not a measurement.
        """
        reader = getattr(self.store, "routing_receipt", None)
        if not callable(reader):
            raise QcMisroutedSurface(
                "ROUTING_UNAVAILABLE_ON_THIS_CONTROL_PLANE",
                "this control plane cannot say how this surface was routed",
            )
        receipt = reader(surface_id)
        if receipt is not None and not surface_routing.verify_receipt(receipt):
            raise QcMisroutedSurface(
                "ROUTING_RECEIPT_UNVERIFIED",
                "this surface's routing receipt does not match its own digest",
            )
        if receipt is None:
            try:
                receipt = surface_routing.build_receipt(
                    surface_id=surface_id, area_cm2=surface.get("area_cm2"),
                    policy=surface_routing.load_policy(),
                    measurement={"decided_at": "PHYSICAL_QC_CLAIM_RESOLUTION"},
                    read_set={"artifact_sha256": surface.get("artifact_sha256")},
                )
            except ValueError as unmeasured:
                raise QcMisroutedSurface(
                    "ROUTING_UNDECIDABLE_NO_MEASURED_AREA", str(unmeasured),
                ) from unmeasured
        # Through the router's own predicate rather than a route string compared
        # here: one place decides what physical QC admits, and it re-verifies the
        # digest, so a forged receipt fails exactly like a missing one.
        if not surface_routing.enters_standard_qc(receipt):
            raise QcMisroutedSurface(
                "SMALL_SURFACE_DIAGNOSTIC_NOT_QC_WORK",
                f"{receipt.get('measured_area_cm2')} cm2 is below the "
                f"{receipt.get('minimum_area_cm2')} cm2 floor: this surface is "
                "diagnostic evidence and physical QC would say nothing about it",
            )

    def _blocked_misrouted(
        self, claim: dict[str, Any], refusal: QcMisroutedSurface,
    ) -> dict[str, Any]:
        """Stop the job where it stands, and leave the surface alone.

        Terminal, because claim_qc takes only PENDING and a requeue here would
        spin a card forever on a surface nothing will ever admit. The receipt
        goes to the control plane and not to this host's disk: the host is
        replaceable and the finding is not.
        """
        blocked = {
            "schema": "campaignx.segment_qc_misrouted_surface.v1",
            "status": "BLOCKED_MISROUTED_SURFACE",
            "reason_code": refusal.reason_code,
            "qc_job_id": claim["qc_job_id"],
            "surface_id": claim["surface_id"],
            "error": _persisted_error(
                f"{refusal.reason_code}: {refusal}",
                "the routing refusal had no safe detail",
            ),
            "generated_at_utc": utc_now(),
            "no_scientific_conclusion": True,
            "is_absence_evidence": False,
        }
        stored = self.store.block_qc_configuration(
            claim["qc_job_id"], claim["lease_token"], blocked
        )
        return {**(stored or {}), **blocked}

    def run_one(self) -> dict[str, Any] | None:
        claim = self.store.claim_qc(
            self.worker_id, self.lease_seconds, profile_id=self.profile_id
        )
        if claim is None:
            return None
        # Before the attempt directory, before the executor, before anything
        # this worker could be said to have done to the surface.
        surface_id = str(claim["surface_id"])
        try:
            self._current_route(surface_id, self._current_surface(surface_id))
        except QcMisroutedSurface as refusal:
            return self._blocked_misrouted(claim, refusal)
        attempt_dir = (
            self.run_root
            / claim["qc_job_id"]
            / f"{utc_now().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:12]}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(attempt_dir / "CLAIMED_QC_JOB.json", _sanitized_claim(claim))
        with QcLeaseHeartbeat(self.store, claim, self.lease_seconds) as heartbeat:
            room = self._vram_shortfall()
            if room is not None:
                # Refused before the executor, which is the whole point. The
                # render that preceded ink screening ran to completion --
                # minutes of GPU and a 158 MB layer stack, exit code 0 -- and
                # only then did inference find there was no room. Reading free
                # VRAM first costs milliseconds.
                heartbeat.ensure()
                free, total = room
                retry = {
                    "schema": "campaignx.segment_qc_retryable_outage.v1",
                    "status": "RETRYABLE_QC_UNAVAILABLE",
                    "reason_code": GPU_MEMORY_EXHAUSTED,
                    "qc_job_id": claim["qc_job_id"],
                    "surface_id": claim["surface_id"],
                    "error": (
                        f"RuntimeError: {free} MiB free of {total} MiB on the "
                        f"best card, below the {self.minimum_free_vram_mib} MiB "
                        "this worker requires"),
                    "gpu_memory_free_mib": free,
                    "gpu_memory_total_mib": total,
                    "minimum_free_vram_mib": self.minimum_free_vram_mib,
                    "generated_at_utc": utc_now(),
                    "no_scientific_conclusion": True,
                }
                write_json_atomic(attempt_dir / "RETRYABLE_QC_RECEIPT.json", retry)
                return _with_reason(
                    self.store.requeue_qc_unavailable(
                        claim["qc_job_id"], claim["lease_token"], retry,
                        retry_delay_seconds=self.retry_delay_seconds,
                    ),
                    GPU_MEMORY_EXHAUSTED,
                )
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
                    "error": _persisted_error(
                        str(error),
                        "the configuration error had no safe detail",
                    ),
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
                    "error": _persisted_error(
                        _retryable_error(error),
                        "RuntimeError: retryable QC failure had no safe detail",
                    ),
                    "generated_at_utc": utc_now(),
                    "no_scientific_conclusion": True,
                }
                # The preflight below is a floor, not a guarantee: a shared
                # card can fill between the check and the allocation. Naming it
                # here too means the receipt says which failure this was
                # wherever it was noticed.
                if is_gpu_exhaustion(retry["error"]):
                    retry["reason_code"] = GPU_MEMORY_EXHAUSTED
                write_json_atomic(attempt_dir / "RETRYABLE_QC_RECEIPT.json", retry)
                _discard_bulk_output(attempt_dir)
                return _with_reason(
                    self.store.requeue_qc_unavailable(
                        claim["qc_job_id"],
                        claim["lease_token"],
                        retry,
                        retry_delay_seconds=self.retry_delay_seconds,
                    ),
                    retry.get("reason_code"),
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
                if (self.stop_after_retryable is not None
                        and consecutive_retryable >= self.stop_after_retryable):
                    # Stop claiming. The job itself stays requeued -- this
                    # worker is saying it cannot do the work, not deciding
                    # anything about the surface.
                    print(json.dumps({
                        "schema": "campaignx.segment_qc_worker_stopped.v1",
                        "status": "STOPPED_AFTER_CONSECUTIVE_RETRYABLE",
                        "worker_id": self.worker_id,
                        "consecutive_retryable_failures": consecutive_retryable,
                        "last_error": str(result.get("error", "")),
                        "generated_at_utc": utc_now(),
                        "note": "claiming stopped rather than continuing to "
                                "retry. A worker that fails every claim and "
                                "keeps claiming spends a card and a disk on "
                                "producing nothing.",
                    }, sort_keys=True), file=sys.stderr, flush=True)
                    completed.append(result)
                    break
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
