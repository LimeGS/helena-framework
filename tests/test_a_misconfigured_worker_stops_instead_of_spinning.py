"""A failure that cannot resolve itself must not be retried.

On 2026-07-29 a rename changed the ink adapter's filename, which changed the ink
profile, which changed the surface-QC profile that pins it by hash. gpu-1's
HELENA_QC_PROFILE_SHA256 was never updated, so the adapter refused every job
with "surface-QC profile hash differs". The worker caught it as an outage and
requeued it. Both GPUs then claimed, failed in a second, requeued and reclaimed
for two days: 3118 receipts, zero surfaces measured.

Nothing was visibly wrong. The jobs sat PENDING, which is what a job waiting for
a free worker looks like, and the cards reported utilisation, which is what work
looks like. The queue could not distinguish "not started yet" from "started
three thousand times".

So the distinction is the fix, and it has three parts:

  * the adapter says which kind of failure it had, because the worker sees an
    exit code and nothing else;
  * a configuration failure is terminal -- claim_qc takes only PENDING, so
    BLOCKED_CONFIGURATION is never picked up again;
  * it is not a scientific verdict. A wrong hash says nothing about the papyrus,
    and the surface row must come through untouched.

The last one is the one worth being careful about: the easy implementation
routes this through finalize_qc, and then an operator's mistake is recorded as a
measurement of a scroll.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.qc_worker import (  # noqa: E402
    EX_CONFIG,
    QcConfigurationError,
    SurfaceQcWorker,
    _adapter_complaint,
)

ADAPTER = ROOT / "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py"


# --------------------------------------------------------------------------
# The adapter says which kind of failure it had
# --------------------------------------------------------------------------

def test_the_adapter_exits_with_a_code_that_means_configuration(tmp_path):
    """End to end through a real process, because the exit code is the whole
    channel: everything the worker learns about a config error crosses here.

    Run with no HELENA_QC_* set at all, which is the simplest configuration
    error there is.
    """
    payload = {"schema": "campaignx.segment_qc_input.v1",
               "qc_job": {"surface_id": "s", "profile_id": "p@1.0.0",
                          "source": {"sample_id": "x"}, "surface": {}}}
    request = tmp_path / "QC_INPUT.json"
    request.write_text(json.dumps(payload))
    output = tmp_path / "out"
    output.mkdir()

    finished = subprocess.run(
        [sys.executable, str(ADAPTER), "--input", str(request), "--output", str(output)],
        capture_output=True, text=True, timeout=180,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert finished.returncode == EX_CONFIG, (
        f"exited {finished.returncode} with {finished.stderr[-400:]!r}; a "
        "configuration error is indistinguishable from a crash at any other code"
    )
    report = json.loads(finished.stderr.strip().splitlines()[-1])
    assert report["status"] == "BLOCKED_CONFIGURATION"
    assert report["no_scientific_conclusion"] is True
    assert "HELENA_QC" in report["error"], report["error"]


def test_the_worker_repeats_what_the_adapter_said_rather_than_the_exit_code():
    """"exit code 78" is not something anybody can act on."""
    said = _adapter_complaint(
        'Traceback (most recent call last):\n'
        '{"schema": "campaignx.segment_qc_configuration_error.v1", '
        '"status": "BLOCKED_CONFIGURATION", "error": "profile hash differs", '
        '"no_scientific_conclusion": true}')
    assert said == "profile hash differs"
    # And when the adapter says nothing useful, say that, rather than "".
    assert "did not say which" in _adapter_complaint("boom\n")


def _claim():
    return {"qc_job_id": "job-1", "surface_id": "surface-1",
            "lease_token": "token-1", "profile_id": "p@1.0.0"}


def test_the_executor_turns_exit_78_into_a_configuration_error(tmp_path):
    """The join between the two halves, and the piece nothing else covers.

    Every other test here either drives a fake executor or runs the adapter
    alone. If SubprocessQcExecutor stopped recognising the exit code, both would
    still pass and the fleet would be back to requeuing forever -- so this runs
    a stub adapter through the real executor.
    """
    from fleet.qc_worker import SubprocessQcExecutor

    stub = tmp_path / "adapter.py"
    stub.write_text(
        "import sys, json\n"
        "print(json.dumps({'status': 'BLOCKED_CONFIGURATION',\n"
        "                  'error': 'the pin and the profile disagree'}))\n"
        "sys.exit(78)\n")
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(QcConfigurationError) as refused:
        SubprocessQcExecutor(stub).execute(_claim(), attempt)
    assert "the pin and the profile disagree" in str(refused.value)


def test_any_other_non_zero_exit_stays_an_ordinary_failure(tmp_path):
    """Only 78 means configuration. A crash is still a crash, and still retried."""
    from fleet.qc_worker import SubprocessQcExecutor

    stub = tmp_path / "adapter.py"
    stub.write_text("import sys\nsys.exit(1)\n")
    attempt = tmp_path / "attempt"
    attempt.mkdir()

    with pytest.raises(RuntimeError) as failed:
        SubprocessQcExecutor(stub).execute(_claim(), attempt)
    assert not isinstance(failed.value, QcConfigurationError)
    assert "exit code 1" in str(failed.value)


# --------------------------------------------------------------------------
# A configuration failure is terminal, and is not a verdict
# --------------------------------------------------------------------------

class Recording:
    """A store that records which finishing move the worker chose."""

    def __init__(self, claim):
        self._claim = claim
        self.calls: list[str] = []

    def claim_qc(self, *a, **k):
        return self._claim

    def heartbeat_qc(self, *a, **k):
        self.calls.append("heartbeat")

    def finalize_qc(self, *a, **k):
        self.calls.append("finalize")
        return {"status": "COMPLETED"}

    def requeue_qc_unavailable(self, *a, **k):
        self.calls.append("requeue")
        return {"status": "RETRYABLE_QC_UNAVAILABLE"}

    def block_qc_configuration(self, qc_job_id, lease_token, receipt):
        self.calls.append("block")
        return {"status": "BLOCKED_CONFIGURATION", "qc_job_id": qc_job_id,
                "surface_id": receipt["surface_id"], "error": receipt["error"]}


class Raises:
    def __init__(self, error):
        self.error = error

    def execute(self, claim, attempt_dir):
        raise self.error


def _worker(store, error, tmp_path):
    return SurfaceQcWorker(store, "worker-1", Raises(error), tmp_path)


def test_a_configuration_error_blocks_the_job(tmp_path):
    """The regression itself. Reverting the fix requeues, and the fleet spins."""
    store = Recording(_claim())
    result = _worker(store, QcConfigurationError("profile hash differs"),
                     tmp_path).run_one()

    assert "requeue" not in store.calls, (
        "a configuration error was requeued; the job returns to PENDING and is "
        "claimed again, forever, because it will fail the same way every time"
    )
    assert "block" in store.calls
    assert result["status"] == "BLOCKED_CONFIGURATION"
    assert result["error"] == "profile hash differs", (
        "the job is blocked without saying what to change"
    )


def test_a_configuration_error_is_not_a_scientific_verdict(tmp_path):
    """finalize_qc writes a surface state and a physical_qc_state.

    Routing a misconfiguration through it would record an operator's mistake as
    a measurement of a scroll, which is worse than the retry loop it replaced.
    """
    store = Recording(_claim())
    _worker(store, QcConfigurationError("checkpoint hash differs"), tmp_path).run_one()
    assert "finalize" not in store.calls

    receipt = json.loads(
        next(tmp_path.rglob("BLOCKED_CONFIGURATION_RECEIPT.json")).read_text())
    assert receipt["no_scientific_conclusion"] is True
    assert receipt["status"] == "BLOCKED_CONFIGURATION"
    assert "outcome" not in receipt


def test_a_genuine_outage_still_requeues(tmp_path):
    """The other half. S3 being down for a minute is exactly what retry is for,
    and narrowing that would trade one failure mode for its opposite."""
    store = Recording(_claim())
    result = _worker(store, OSError("connection reset by peer"), tmp_path).run_one()

    assert "requeue" in store.calls
    assert "block" not in store.calls
    assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
    assert next(tmp_path.rglob("RETRYABLE_QC_RECEIPT.json")).is_file()


@pytest.mark.parametrize("error", [
    KeyboardInterrupt(), MemoryError(), RuntimeError("adapter failed with exit code 1"),
])
def test_anything_that_is_not_a_configuration_error_keeps_the_old_behaviour(
        error, tmp_path):
    store = Recording(_claim())
    _worker(store, error, tmp_path).run_one()
    assert store.calls[-1] == "requeue"


# --------------------------------------------------------------------------
# The store: blocked means blocked
# --------------------------------------------------------------------------

def test_a_blocked_job_is_not_claimable_again():
    """claim_qc's predicate is the whole guarantee. If BLOCKED_CONFIGURATION
    were claimable this would be the retry loop with a new name."""
    for name in ("postgres_store", "store"):
        source = (ROOT / f"framework/stages/01-segmentation/fleet/{name}.py").read_text()
        claim = source.split("def claim_qc")[1].split("\n    def ")[0]
        assert "BLOCKED_CONFIGURATION" not in claim, (
            f"{name}.claim_qc mentions the blocked state"
        )

        blocking = source.split("def block_qc_configuration")[1].split("\n    def ")[0]
        # Past the docstring, which discusses the very things forbidden below --
        # the first version of this test matched its own prose, and the second
        # split on every triple quote and tore the SQL in half. Drop exactly the
        # signature and the docstring, keep the rest joined.
        parts = blocking.split('"""')
        statements = '"""'.join(parts[2:])
        assert "state='BLOCKED_CONFIGURATION'" in statements
        assert "retry_after=NULL" in statements, (
            "a blocked job keeps a retry time, which reads as pending-with-a-delay"
        )
        # The surface is somebody else's fact and must come through untouched.
        for forbidden in ("UPDATE surfaces", "UPDATE segment_surfaces"):
            assert forbidden not in statements, (
                f"{name} writes a surface state for a configuration error, which "
                "turns an operator's mistake into a measurement"
            )


def test_both_stores_offer_it():
    """The deployment runs PostgreSQL and the tests run SQLite. A worker calling
    a method one of them lacks fails at the moment it is most needed."""
    from fleet.postgres_store import PostgresFleetStore
    from fleet.store import FleetStore

    for store in (PostgresFleetStore, FleetStore):
        assert callable(getattr(store, "block_qc_configuration", None)), store.__name__


# --------------------------------------------------------------------------
# Failing every claim is loud
# --------------------------------------------------------------------------

def test_a_worker_that_never_succeeds_says_so(tmp_path, capsys):
    """For the failures that are genuinely retryable but never recover.

    Blocking covers what is knowably permanent. This covers the rest: a hundred
    requeues with no success between them is not an outage anybody is waiting
    out, and from outside it is indistinguishable from a busy worker.
    """
    store = Recording(_claim())
    worker = _worker(store, OSError("nope"), tmp_path)
    worker.run(max_jobs=6, idle_exit=False)

    alarms = [json.loads(line) for line in capsys.readouterr().err.splitlines()
              if line.startswith("{")]
    assert alarms, "six consecutive failures and the worker said nothing"
    assert alarms[0]["status"] == "NO_SURFACE_MEASURED"
    assert alarms[0]["consecutive_retryable_failures"] == 5
    assert alarms[0]["worker_id"] == "worker-1"


def test_the_alarm_resets_when_work_succeeds(tmp_path, capsys):
    """Otherwise it fires on any deployment that has ever had a bad afternoon."""
    store = Recording(_claim())
    worker = _worker(store, OSError("nope"), tmp_path)
    worker.alarm_after = frozenset({3})

    # Two failures, a success, then two more: never three in a row.
    outcomes = ["r", "r", "ok", "r", "r"]
    calls = iter(outcomes)

    def run_one():
        return ({"status": "COMPLETED"} if next(calls) == "ok"
                else {"status": "RETRYABLE_QC_UNAVAILABLE"})

    worker.run_one = run_one  # type: ignore[method-assign]
    worker.run(max_jobs=len(outcomes), idle_exit=False)
    assert not capsys.readouterr().err.strip(), "the alarm fired on a working worker"


# --------------------------------------------------------------------------
# Somebody is told
# --------------------------------------------------------------------------

def test_the_panel_counts_jobs_blocked_on_configuration():
    """A terminal state nobody looks at is the retry loop with better manners.

    The whole incident was invisible: the jobs sat in PENDING, which is what a
    healthy queue with no free worker looks like, and the GPUs reported
    utilisation. Blocking them stops the spin; counting them is what makes it
    somebody's problem. The count is only ever non-zero when a person has to
    change a setting.
    """
    app = (ROOT / "panel/app.py").read_text()
    status = app[app.index("def fleet_status("):]
    status = status[: status.index("\n\ndef ")]
    assert "BLOCKED_CONFIGURATION" in status, (
        "the fleet status does not count blocked QC jobs, so a misconfigured "
        "fleet looks exactly like an idle one"
    )
    assert "qc_blocked_on_configuration" in status
