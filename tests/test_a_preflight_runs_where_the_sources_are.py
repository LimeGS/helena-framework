"""The preflight has to run on a host that can reach the sources.

It ran inside the panel's request handler, which is the one process in the
deployment that cannot do it. `run_control_region_preflight` reaches the m7
prediction through the vc3d MCP seed service, and that service listens on
loopback for the workers that need it. The panel has no VC_MCP_URL and no
VC_MCP_AUTH_TOKEN, so every real call raised the provider's own
"VC_MCP_URL and VC_MCP_AUTH_TOKEN are required" and the endpoint answered
503 PREFLIGHT_SOURCE_UNAVAILABLE -- a source outage, reported for a source
that was never down.

So the measurement moves to a queued job and a worker on a host that has the
sources. This file is about the worker half, and it holds four things:

  * a provider or source failure is a reason code on the job, not a traceback
    out of the loop and not an unbounded retry;
  * a worker that is missing its own configuration STOPS. It fails the job it
    is holding and does not claim another, because the alternative is one
    misconfigured host quietly consuming the entire queue at one attempt each
    -- the 3118-receipt incident with the counters moved around;
  * the state and the lease are re-read immediately before the provider is
    called. A check at claim time is a check about the past;
  * the preflight is ink-blind. It counts candidates in a bounded region and
    it must never ask for, receive, or record ink signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.preflight_worker import (  # noqa: E402
    CandidatePreflightWorker,
    ControlRegionPreflightExecutor,
    PreflightConfigurationError,
    PreflightSourceUnavailable,
)
from fleet.worker import SourceProviderUnavailable  # noqa: E402

SNAPSHOT = {
    "source_snapshot_id": "snap-1",
    "sample_id": "PHerc1667",
    "m7_uri": "https://example.invalid/m7.zarr",
    "ct_uri": "https://example.invalid/ct.zarr",
}

REQUEST = {
    "region_center_xyz": {"x": 1.0, "y": 2.0, "z": 3.0},
    "region_radius_xyz": {"x": 8.0, "y": 8.0, "z": 8.0},
    "known_coordinate_xyz": {"x": 1.0, "y": 2.0, "z": 3.0},
    "tolerance_ct_l0_voxels": 4.0,
    "m7_threshold": 0.5,
    "max_candidates": 8,
    "packet_candidate_limit": 8,
    "minimum_separation_voxels": 16,
    "minimum_cell_clearance_voxels": 2.0,
    "minimum_volume_clearance_voxels": 2.0,
    "seed_region_policy": "fixed-v1",
    "ct_material_support_gate": {"level": 5},
}


def _receipt(**overrides):
    return {
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "scope": "CONTROL_REGION",
        "status": "COMPLETE",
        "sample_id": SNAPSHOT["sample_id"],
        "source_snapshot_id": SNAPSHOT["source_snapshot_id"],
        "state_mutation": "NONE",
        "growth_allowed": False,
        "ink_used": False,
        "counts": {"raw_m7": 3},
        **overrides,
    }


class RecordingStore:
    """A control plane that records which finishing move the worker chose.

    It implements exactly the frozen store contract and nothing else, so a
    worker that reaches for a method nobody agreed to build fails here rather
    than on the deployment.
    """

    def __init__(self, jobs, *, snapshot=SNAPSHOT, job_state="CLAIMED"):
        self._jobs = list(jobs)
        self._snapshot = snapshot
        self.job_state = job_state
        self.calls: list[str] = []
        self.finalized: list[dict] = []
        self.failed: list[str] = []
        self.details: list[object] = []
        self.claims = 0
        self.heartbeats = 0
        self.heartbeat_error: BaseException | None = None

    # -- the frozen preflight contract -----------------------------------
    def claim_preflight(self, worker_id, lease_seconds):
        self.calls.append("claim")
        self.claims += 1
        if not self._jobs:
            return None
        return dict(self._jobs.pop(0))

    def heartbeat_preflight(self, preflight_job_id, lease_token, lease_seconds):
        self.calls.append("heartbeat")
        self.heartbeats += 1
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        return {"preflight_job_id": preflight_job_id, "state": self.job_state}

    def finalize_preflight(self, preflight_job_id, lease_token, receipt):
        self.calls.append("finalize")
        self.finalized.append(receipt)
        return {"preflight_job_id": preflight_job_id, "state": "COMPLETED"}

    def fail_preflight(self, preflight_job_id, lease_token, reason_code,
                       detail=None):
        self.calls.append("fail")
        self.failed.append(reason_code)
        self.details.append(detail)
        return {"preflight_job_id": preflight_job_id, "state": "FAILED",
                "reason_code": reason_code, "detail": detail}

    def preflight_job(self, preflight_job_id):
        self.calls.append("read")
        return {"preflight_job_id": preflight_job_id, "state": self.job_state}

    # -- what the worker needs to resolve the snapshot --------------------
    def snapshots(self, samples=None):
        self.calls.append("snapshots")
        return [dict(self._snapshot)] if self._snapshot else []


def _envelope(parameters=None):
    """What the panel actually enqueues.

    The frozen parameters are nested under `parameters`, beside the identity and
    the source lock. These fixtures used to be the parameters themselves, flat,
    which is the shape the panel passed when it ran the measurement in the
    request -- so every test agreed with every other test and none of them
    agreed with the deployment, where the first real job died on
    `KeyError: 'region_center_xyz'`.
    """
    return {
        "schema": "campaignx.segment_control_region_preflight_request.v1",
        "scope": "CONTROL_REGION",
        "mission_id": "mission-1",
        "sample_id": SNAPSHOT["sample_id"],
        "source_snapshot_id": SNAPSHOT["source_snapshot_id"],
        "snapshot": dict(SNAPSHOT),
        "parameters": dict(REQUEST if parameters is None else parameters),
        # The objects the manifest froze as each source's identity. They live in
        # the envelope and nowhere else, and the receipt is refused without them.
        "frozen_root_objects": {
            "ct": [{"path": ".zattrs", "sha256": "1" * 64},
                   {"path": "0/.zarray", "sha256": "2" * 64}],
            "m7": [{"path": ".zattrs", "sha256": "3" * 64},
                   {"path": "0/.zarray", "sha256": "4" * 64}],
        },
    }


def _job(job_id="pfj-1", **overrides):
    return {
        "preflight_job_id": job_id,
        "lease_token": f"token-{job_id}",
        "mission_id": "mission-1",
        "sample_id": SNAPSHOT["sample_id"],
        "source_snapshot_id": SNAPSHOT["source_snapshot_id"],
        "request": _envelope(),
        **overrides,
    }


class Executor:
    """Stands in for the run_control_region_preflight call."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls: list[tuple[dict, dict]] = []
        self.frozen_roots: list[object] = []

    def execute(self, snapshot, request, frozen_root_objects=None):
        self.calls.append((snapshot, request))
        self.frozen_roots.append(frozen_root_objects)
        if self.error is not None:
            raise self.error
        return self.result if self.result is not None else _receipt()


def _worker(store, executor, **kwargs):
    return CandidatePreflightWorker(
        store, "preflight-worker-1", executor=executor, **kwargs)


# --------------------------------------------------------------------------
# A provider failure is a reason code, not a traceback and not a retry loop
# --------------------------------------------------------------------------

def test_a_missing_provider_configuration_fails_the_job_and_stops_the_worker():
    """The exact failure the panel hit, on a host that also lacks the settings.

    Two things have to be true at once. The job cannot be left holding a lease
    nobody will release, so it is failed with a reason code that names the
    cause. And the worker cannot go back for another one: it is missing a
    setting, so it will fail the next job in the same second, and the next, and
    the queue will empty into FAILED with nothing measured and nothing said.
    """
    store = RecordingStore([_job("pfj-1"), _job("pfj-2")])
    executor = Executor(
        error=RuntimeError("VC_MCP_URL and VC_MCP_AUTH_TOKEN are required"))

    results = _worker(store, executor).run(max_jobs=2, idle_exit=False)

    assert store.failed == ["PREFLIGHT_PROVIDER_NOT_CONFIGURED"], (
        "a worker missing VC_MCP_URL/VC_MCP_AUTH_TOKEN did not say so on the job"
    )
    assert "finalize" not in store.calls
    assert store.claims == 1, (
        f"the worker claimed {store.claims} jobs while missing its own "
        "configuration; every one of them fails the same way, so this is the "
        "spin with the queue draining instead of standing still"
    )
    assert results[-1]["status"] == "BLOCKED_CONFIGURATION"
    assert results[-1]["worker_stopped"] is True


def test_a_configuration_error_raised_structurally_stops_the_worker_too():
    """The executor is allowed to know it is misconfigured before it tries.

    Matching the provider's sentence is a backstop, not the mechanism. An
    executor that checks its own settings raises the typed error, and that path
    has to reach the same terminal stop, or the structural check silently
    degrades into an ordinary retry.
    """
    store = RecordingStore([_job("pfj-1"), _job("pfj-2")])
    executor = Executor(error=PreflightConfigurationError(
        "PREFLIGHT_PROVIDER_NOT_CONFIGURED", "the seed provider has no URL"))

    _worker(store, executor).run(max_jobs=2, idle_exit=False)

    assert store.failed == ["PREFLIGHT_PROVIDER_NOT_CONFIGURED"]
    assert store.claims == 1


@pytest.mark.parametrize("error", [
    SourceProviderUnavailable("MCP returned no valid source read evidence"),
    OSError("connection reset by peer"),
    TimeoutError("the object store did not answer"),
])
def test_a_source_outage_fails_the_job_without_killing_the_loop(error):
    """S3 being down for a minute is what the bounded attempt count is for.

    The distinction against the case above is the whole point: an outage may
    recover, so the job is failed with an outage reason and the worker goes
    back for the next one. A misconfiguration will not recover, and the worker
    stops.
    """
    store = RecordingStore([_job("pfj-1"), _job("pfj-2")])
    executor = Executor(error=error)

    results = _worker(store, executor).run(max_jobs=2, idle_exit=False)

    assert store.failed == ["PREFLIGHT_SOURCE_UNAVAILABLE"] * 2
    assert store.claims == 2, "a recoverable outage stopped the worker"
    assert all(row["status"] == "FAILED" for row in results)
    assert all(row.get("worker_stopped") is not True for row in results)


def test_a_failed_job_records_what_went_wrong_and_not_only_that_it_did():
    """The reason code names a category; the detail names the event.

    The store takes a `detail` beside the code precisely so an operator does not
    have to go to a worker's stdout, which is not evidence and which a restart
    erases. `_fail` accepted a detail and dropped it on the way to the store, so
    the deployment's first real preflight failure read
    `PREFLIGHT_SOURCE_UNAVAILABLE` and nothing else -- and the worker's log had
    already been rotated away by the time anybody asked.
    """
    store = RecordingStore([_job("pfj-1")])
    executor = Executor(error=OSError("the surface bucket refused the read"))

    result = _worker(store, executor).run_one()

    assert store.failed == ["PREFLIGHT_SOURCE_UNAVAILABLE"]
    assert store.details and store.details[0], (
        "the job was failed with no detail, so the only account of why lives in "
        "a container's stdout"
    )
    assert "the surface bucket refused the read" in str(store.details[0])
    assert result["reason_code"] == "PREFLIGHT_SOURCE_UNAVAILABLE"


def test_a_detail_never_carries_the_token_that_reached_the_service():
    """The detail is read by whoever reads the queue, which is a wider audience
    than the worker's own stderr."""
    store = RecordingStore([_job("pfj-1")])
    executor = Executor(error=OSError("401 from http://host/mcp token=s3cr3t-value"))

    _worker(store, executor).run_one()

    assert "s3cr3t-value" not in str(store.details[0])


def test_the_loop_stops_when_nothing_it_claims_ever_succeeds():
    """"Retryable" repeated forever with no success between is its own spin."""
    store = RecordingStore([_job(f"pfj-{index}") for index in range(40)])
    executor = Executor(error=OSError("nope"))

    # A bounded max_jobs on purpose. With the guard gone the loop would empty
    # the queue and then wait for work forever, and a test that hangs reports
    # nothing; this way removing the guard fails on the count.
    results = _worker(store, executor, max_consecutive_failures=5).run(
        max_jobs=40, idle_exit=False)

    assert store.claims == 5, (
        f"the worker claimed {store.claims} jobs and measured none of them"
    )
    assert results[-1]["worker_stopped"] is True


def test_an_unknown_source_snapshot_is_the_jobs_problem_and_not_the_workers():
    """A job naming a snapshot this control plane does not have is one bad row.

    It must not stop a worker that can still run every other job in the queue.
    """
    store = RecordingStore([_job("pfj-1"), _job("pfj-2")], snapshot=None)
    executor = Executor()

    _worker(store, executor).run(max_jobs=2, idle_exit=False)

    assert store.failed == ["PREFLIGHT_SOURCE_SNAPSHOT_UNKNOWN"] * 2
    assert store.claims == 2
    assert executor.calls == [], "the provider was called without a source"


# --------------------------------------------------------------------------
# Ask again before doing anything
# --------------------------------------------------------------------------

def test_the_worker_asks_again_before_it_calls_the_provider():
    """The job moved between the claim and here, so nothing is measured.

    A lease expires, the row returns to PENDING, another worker claims it. The
    first worker still holds a token and, if it never looks again, still runs
    the whole preflight and finalizes over somebody else's result.
    """
    store = RecordingStore([_job("pfj-1")], job_state="PENDING")
    executor = Executor()

    result = _worker(store, executor).run_one()

    assert executor.calls == [], (
        "the worker called the provider for a job that had gone back to PENDING"
    )
    assert "finalize" not in store.calls
    assert result["status"] == "STOOD_DOWN"
    assert result["reason_code"] == "PREFLIGHT_JOB_MOVED"


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED"])
def test_a_job_somebody_already_finished_is_not_run_again(state):
    store = RecordingStore([_job("pfj-1")], job_state=state)
    executor = Executor()

    _worker(store, executor).run_one()

    assert executor.calls == []
    assert "finalize" not in store.calls
    assert "fail" not in store.calls, (
        "the worker overwrote the outcome of a job it no longer owned"
    )


def test_the_worker_refuses_when_the_lease_is_no_longer_its_own():
    """The state read alone cannot tell whose job it is.

    A row that says CLAIMED says nothing about which worker holds it. The lease
    token is the only thing that does, and heartbeat_preflight is the contract
    method that checks it, so it is asked immediately before the provider call
    rather than only from the background thread.
    """
    store = RecordingStore([_job("pfj-1")])
    store.heartbeat_error = RuntimeError("preflight lease is not held by this worker")
    executor = Executor()

    result = _worker(store, executor).run_one()

    assert executor.calls == []
    assert "finalize" not in store.calls
    assert result["status"] == "STOOD_DOWN"


def test_the_recheck_happens_after_the_claim_and_before_the_provider():
    """Ordering is the whole guarantee; a re-check after the I/O is a log line."""
    store = RecordingStore([_job("pfj-1")])
    executor = Executor()

    _worker(store, executor).run_one()

    order = [call for call in store.calls
             if call in {"claim", "read", "heartbeat", "finalize"}]
    assert order.index("claim") < order.index("read") < order.index("finalize")
    assert order.index("heartbeat") < order.index("finalize")
    assert executor.calls, "nothing was measured"


def test_a_heartbeat_that_dies_mid_measurement_does_not_crash_the_loop():
    """The lease keeper reports on the way out of its own context manager.

    So it reports after the body has returned, and a handler written inside the
    block cannot see it. That is the failure which must not crash the loop,
    crashing the loop.
    """
    from fleet import preflight_worker

    store = RecordingStore([_job("pfj-1"), _job("pfj-2")])
    original = preflight_worker.PreflightLeaseHeartbeat.__exit__

    def dying_exit(self, *exc):
        original(self, *exc)
        raise preflight_worker.PreflightJobMoved("the lease heartbeat stopped")

    preflight_worker.PreflightLeaseHeartbeat.__exit__ = dying_exit
    try:
        results = _worker(store, Executor()).run(max_jobs=2, idle_exit=False)
    finally:
        preflight_worker.PreflightLeaseHeartbeat.__exit__ = original

    assert all(row["status"] == "STOOD_DOWN" for row in results)
    assert "finalize" not in store.calls


def test_a_refused_fail_preflight_is_a_stand_down_rather_than_a_crash():
    """`fail_preflight` fails closed, so being refused means it is not ours."""
    class Refusing(RecordingStore):
        def fail_preflight(self, preflight_job_id, lease_token, reason_code):
            self.calls.append("fail")
            raise RuntimeError("this preflight lease is not held by this worker")

    store = Refusing([_job("pfj-1")])

    result = _worker(store, Executor(error=OSError("nope"))).run_one()

    assert result["status"] == "STOOD_DOWN"


def test_a_worker_that_only_ever_stands_down_stops_too():
    """Losing a race is ordinary; losing every race forever is the spin."""
    store = RecordingStore(
        [_job(f"pfj-{index}") for index in range(40)], job_state="PENDING")

    results = _worker(store, Executor(), max_consecutive_failures=5).run(
        max_jobs=40, idle_exit=False)

    assert store.claims == 5
    assert results[-1]["worker_stopped"] is True


# --------------------------------------------------------------------------
# The measurement, and what it is allowed to contain
# --------------------------------------------------------------------------

def test_a_successful_preflight_is_finalized_with_its_receipt():
    store = RecordingStore([_job("pfj-1")])
    executor = Executor()

    result = _worker(store, executor).run_one()

    assert store.finalized and store.finalized[0]["status"] == "COMPLETE"
    assert result["status"] == "COMPLETED"
    snapshot, request = executor.calls[0]
    assert snapshot["source_snapshot_id"] == "snap-1", (
        "the snapshot did not come from the store by source_snapshot_id"
    )
    assert request == REQUEST, "the frozen parameters were not the job's request"


def test_every_field_the_measurement_requires_is_in_what_the_worker_hands_it():
    """Read the requirement off the measurement rather than restating it.

    Both sides of this contract were tested and neither test knew the other's
    shape: the worker's fixtures were the flat parameters, the panel enqueues
    them nested. Deriving the field list from the source means a field added to
    the measurement, or a level added to the envelope, fails here.
    """
    import re
    from pathlib import Path

    from fleet.preflight_worker import CandidatePreflightWorker

    source = Path(__file__).resolve().parents[1] / (
        "framework/stages/01-segmentation/fleet/candidate_preflight.py")
    body = source.read_text()
    start = body.index("def run_control_region_preflight(")
    required = set(re.findall(r'request\["([a-z_0-9]+)"\]', body[start:]))
    assert len(required) > 8, "the field list was not parsed out of the source"

    handed = CandidatePreflightWorker._request(_job("pfj-1"))

    assert required <= set(handed), (
        "the measurement reads fields the worker does not hand it: "
        f"{sorted(required - set(handed))}"
    )


def test_the_frozen_root_objects_travel_with_the_parameters():
    """The receipt has to name the volume it read, and the only place those
    objects exist is the envelope the panel enqueued. Handing the measurement
    the parameters alone leaves it unable to bind, and the panel then refuses a
    complete measurement with FROZEN_ROOT_OBJECT_EVIDENCE_MISSING."""
    store = RecordingStore([_job("pfj-1")])
    executor = Executor()

    _worker(store, executor).run_one()

    assert executor.frozen_roots[0] == _envelope()["frozen_root_objects"]


def test_an_envelope_that_locks_no_roots_passes_none_rather_than_inventing_any():
    store = RecordingStore([_job("pfj-1", request={
        **_envelope(), "frozen_root_objects": None})])
    executor = Executor()

    _worker(store, executor).run_one()

    assert executor.frozen_roots[0] is None


def test_a_request_without_frozen_parameters_is_unusable_not_an_outage():
    """A malformed request will not fix itself, and calling it an outage means
    the worker retries it until the attempt budget is gone."""
    store = RecordingStore([_job("pfj-1", request={"scope": "CONTROL_REGION"})])

    result = _worker(store, Executor()).run_one()

    assert store.failed == ["PREFLIGHT_REQUEST_UNUSABLE"]
    assert result["status"] in {"FAILED", "BLOCKED_CONFIGURATION"}


def test_the_frozen_parameters_reach_the_provider_unaltered():
    """The worker is a courier for the request. Editing it forges the run."""
    store = RecordingStore([_job("pfj-1")])
    executor = Executor()

    _worker(store, executor).run_one()

    _snapshot, request = executor.calls[0]
    assert request == REQUEST
    assert request is not None


@pytest.mark.parametrize("receipt", [
    _receipt(ink_used=True),
    _receipt(ink_used="no"),
    _receipt(growth_allowed=True),
    _receipt(state_mutation="SURFACES_WRITTEN"),
])
def test_a_receipt_that_is_not_ink_blind_and_read_only_is_never_finalized(receipt):
    """The one thing this pipeline may not produce.

    The preflight measures candidate availability. A receipt claiming ink was
    used, or that growth was allowed, or that state moved, is either a wrong
    executor or a wrong contract, and persisting it would put an ink claim into
    the control plane under a schema that promises there is none.
    """
    store = RecordingStore([_job("pfj-1")])
    executor = Executor(result=receipt)

    _worker(store, executor).run_one()

    assert "finalize" not in store.calls, (
        f"a receipt with {receipt['ink_used']=} {receipt['growth_allowed']=} "
        f"{receipt['state_mutation']=} was persisted as an ink-blind preflight"
    )
    assert store.failed == ["PREFLIGHT_RECEIPT_NOT_INK_BLIND"]


def test_a_receipt_that_is_not_ink_blind_stops_the_worker():
    """It will be wrong for every other job too, and it is the worst failure."""
    store = RecordingStore([_job("pfj-1"), _job("pfj-2")])

    _worker(store, Executor(result=_receipt(ink_used=True))).run(
        max_jobs=2, idle_exit=False)

    assert store.claims == 1


def test_the_worker_never_names_ink_anywhere_in_what_it_persists():
    """Beyond the flag: no key the worker adds may carry an ink signal."""
    store = RecordingStore([_job("pfj-1")])

    _worker(store, Executor()).run_one()

    persisted = store.finalized[0]
    assert persisted["ink_used"] is False
    for key in persisted:
        assert "ink" not in key.lower() or key == "ink_used", key


def test_the_worker_source_never_requests_ink():
    """A source-level gate, because a future edit is how this gets lost."""
    source = (ROOT / "framework/stages/01-segmentation/fleet/preflight_worker.py"
              ).read_text(encoding="utf-8")
    for forbidden in ("ink_signal", "read_ink", "ink_probability",
                      "request_ink", "ink_threshold"):
        assert forbidden not in source, forbidden


# --------------------------------------------------------------------------
# The default executor is the function the panel used to call
# --------------------------------------------------------------------------

def test_the_default_executor_calls_run_control_region_preflight(monkeypatch):
    """Not a reimplementation of the measurement. The same function, moved."""
    from fleet import candidate_preflight, preflight_worker

    seen = {}

    def fake(snapshot, request, **kwargs):
        seen["args"] = (snapshot, request)
        return _receipt()

    # Patched on candidate_preflight, not on preflight_worker: the executor
    # reaches the measurement through the module, so this one seam serves every
    # caller. A name imported into the worker at module load would ignore the
    # patch and measure for real -- which it did, in the e2e tests, until the
    # executor was changed to look it up.
    assert callable(candidate_preflight.run_control_region_preflight)
    monkeypatch.setattr(candidate_preflight, "run_control_region_preflight", fake)
    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1:8099/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "token")

    result = ControlRegionPreflightExecutor().execute(SNAPSHOT, dict(REQUEST))

    assert seen["args"][0] == SNAPSHOT
    assert seen["args"][1] == REQUEST
    assert result["ink_used"] is False


def test_the_default_executor_refuses_before_it_calls_an_unconfigured_provider(
        monkeypatch):
    """Structural, so the failure names a setting rather than an outage."""
    monkeypatch.delenv("VC_MCP_URL", raising=False)
    monkeypatch.delenv("VC_MCP_AUTH_TOKEN", raising=False)

    with pytest.raises(PreflightConfigurationError) as refused:
        ControlRegionPreflightExecutor().execute(SNAPSHOT, dict(REQUEST))

    assert refused.value.reason_code == "PREFLIGHT_PROVIDER_NOT_CONFIGURED"
    assert "VC_MCP_URL" in str(refused.value)


def test_source_unavailable_is_distinguishable_from_configuration():
    assert not issubclass(PreflightSourceUnavailable, PreflightConfigurationError)
    assert not issubclass(PreflightConfigurationError, PreflightSourceUnavailable)


# --------------------------------------------------------------------------
# The fleet CLI runs it
# --------------------------------------------------------------------------

def test_the_fleet_cli_offers_a_preflight_worker_loop():
    from fleet import cli

    parser = cli.build_parser(ROOT)
    args = parser.parse_args([
        "preflight-worker", "run", "--db", "/tmp/x.sqlite",
        "--worker-id", "gpu-1", "--max-jobs", "1",
    ])
    assert args.handler is cli.command_preflight_worker_run
    assert args.worker_id == "gpu-1"


def _cli_args(tmp_path, **overrides):
    import argparse

    return argparse.Namespace(**{
        "db": str(tmp_path / "fleet.sqlite"), "worker_id": "gpu-1",
        "lease_seconds": 900, "max_jobs": 1, "watch": False,
        "poll_seconds": 1.0, "max_consecutive_failures": 5, **overrides})


def test_the_cli_loop_claims_executes_and_finalizes(monkeypatch, tmp_path):
    """End to end through the CLI handler with a recording control plane.

    The handler builds the real ControlRegionPreflightExecutor -- there is no
    seam for a fixture, deliberately -- so the sources are configured and only
    the measurement itself is replaced.
    """
    from fleet import candidate_preflight, cli

    store = RecordingStore([_job("pfj-1")])
    store.initialize = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "open_fleet_store", lambda db: store)
    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1:8099/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "token")
    monkeypatch.setattr(
        candidate_preflight, "run_control_region_preflight",
        lambda snapshot, request, **kwargs: _receipt())

    exit_code = cli.command_preflight_worker_run(_cli_args(tmp_path))

    assert exit_code == 0
    assert store.finalized


def test_the_cli_exits_on_the_configuration_code_when_the_worker_stops(
        monkeypatch, tmp_path, capsys):
    """A supervisor that restarts on any non-zero exit is the spin, one level up.

    EX_CONFIG is what the QC adapter already uses to mean "this is a setting",
    so the same code means the same thing here and a restart policy can tell a
    misconfigured host from a busy one.
    """
    from fleet import cli, qc_worker

    store = RecordingStore([_job("pfj-1"), _job("pfj-2")])
    store.initialize = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "open_fleet_store", lambda db: store)
    monkeypatch.delenv("VC_MCP_URL", raising=False)
    monkeypatch.delenv("VC_MCP_AUTH_TOKEN", raising=False)

    exit_code = cli.command_preflight_worker_run(_cli_args(tmp_path, max_jobs=2))

    assert exit_code == qc_worker.EX_CONFIG
    assert store.failed == ["PREFLIGHT_PROVIDER_NOT_CONFIGURED"]
    assert store.claims == 1
    report = json.loads(capsys.readouterr().out)
    assert report["environment"]["vc_mcp_url_set"] is False
    assert report["ink_used"] is False


def test_the_cli_report_never_prints_the_auth_token(monkeypatch, tmp_path, capsys):
    """The environment report exists to answer "can this host run one at all"."""
    from fleet import candidate_preflight, cli

    store = RecordingStore([_job("pfj-1")])
    store.initialize = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "open_fleet_store", lambda db: store)
    monkeypatch.setenv("VC_MCP_URL", "http://private-seed-host:8099/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "do-not-print-this")
    monkeypatch.setattr(
        candidate_preflight, "run_control_region_preflight",
        lambda snapshot, request, **kwargs: _receipt())

    cli.command_preflight_worker_run(_cli_args(tmp_path))

    printed = capsys.readouterr().out
    assert "do-not-print-this" not in printed
    assert "private-seed-host" not in printed
    assert json.loads(printed)["environment"]["vc_mcp_auth_token_set"] is True


# --------------------------------------------------------------------------
# What a failure is allowed to say
# --------------------------------------------------------------------------

def test_a_provider_failure_is_redacted_before_it_becomes_durable():
    """A provider sentence carries hosts, DSNs and tokens often enough."""
    store = RecordingStore([_job("pfj-1")])
    executor = Executor(error=SourceProviderUnavailable(
        "MCP refused at http://private-seed-host:8099/mcp with "
        "Authorization: Token seed-secret for "
        "postgresql://alice:hunter2@private-db/campaignx " + "x" * 700))

    result = _worker(store, executor).run_one()

    error = result["error"]
    assert len(error) <= 500
    for forbidden in ("private-seed-host", "seed-secret", "hunter2",
                      "private-db"):
        assert forbidden not in error
    assert result["no_scientific_conclusion"] is True
    assert result["ink_used"] is False


def test_every_reason_code_the_worker_can_emit_is_declared():
    """`fail_preflight` takes a reason code and nothing else.

    An undeclared code is a job that ended for a reason nobody can look up.
    """
    from fleet.preflight_worker import PREFLIGHT_REASON_CODES

    store = RecordingStore([_job("pfj-1")])
    _worker(store, Executor(error=OSError("nope"))).run_one()
    assert set(store.failed) <= PREFLIGHT_REASON_CODES


# --------------------------------------------------------------------------
# Both stores have to offer the contract the worker calls
# --------------------------------------------------------------------------

def test_both_stores_offer_the_preflight_contract():
    """The deployment runs PostgreSQL and the tests run SQLite.

    A worker calling a method one of them lacks fails at the moment it is most
    needed, which is on the host that actually has the sources.
    """
    from fleet.postgres_store import PostgresFleetStore
    from fleet.store import FleetStore

    for store in (PostgresFleetStore, FleetStore):
        for method in ("enqueue_candidate_preflight", "claim_preflight",
                       "heartbeat_preflight", "finalize_preflight",
                       "fail_preflight", "preflight_job"):
            assert callable(getattr(store, method, None)), \
                f"{store.__name__}.{method}"


def test_a_real_sqlite_queue_carries_a_job_from_enqueue_to_completed(tmp_path):
    """The worker against a real control plane rather than a double."""
    from fleet.store import FleetStore

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = store.register_snapshot({
        "sample_id": SNAPSHOT["sample_id"],
        "ct_uri": SNAPSHOT["ct_uri"],
        "m7_uri": SNAPSHOT["m7_uri"],
        "shape_xyz": [32, 32, 32],
        "voxel_size_um": 1.0,
        "coordinate_frame": "ct_l0_xyz",
    })
    enqueued = store.enqueue_candidate_preflight({
        **_envelope(),
        "source_snapshot_id": source_id,
    })
    assert enqueued["created"] is True

    worker = CandidatePreflightWorker(
        store, "preflight-worker-1", executor=Executor())
    result = worker.run_one()

    assert result["status"] == "COMPLETED"
    assert store.preflight_job(enqueued["preflight_job_id"])["state"] == "COMPLETED"
