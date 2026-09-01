"""A retry that keeps its own render is a disk that fills, and it did.

Measured on gpu-1 on 2026-08-22, not inferred. /ssd/vc3d/runs/surface-qc-v2
held 7,774 attempt directories, 7,463 of them carrying
RETRYABLE_QC_RECEIPT.json -- 194.1 GB of renders belonging to attempts whose
own receipts say "no_scientific_conclusion": true. 199.8 GB of the tree was
.tif layer stacks, roughly 158 MB written afresh on every retry of the same
job, failing the same way since July.

It filled a 477 GB disk, and everything else on the host stopped: the QC
container wedged with "No space left on device", and the CI pipeline that
deploys this platform hung for an hour on a first job that produced no log
output at all.

Two things were wrong and both are here.

The attempt directory: a failed attempt's bulk output is not evidence. Its
receipt is. Nothing ever removed the renders, so every retry cost another
158 MB permanently.

The loop: `run` already counted consecutive retryable failures and already
said, in its own comment, that "retryable repeated without a single success is
its own kind of stuck". It printed an alarm at 5, 25, 100 and 500 and then
kept claiming. An alarm nobody is awake to read is not a brake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "tests"))

from fleet.qc_worker import FixtureQcExecutor, SurfaceQcWorker  # noqa: E402
from fleet.store import FleetStore  # noqa: E402
from test_a_qc_worker_asks_again_before_it_measures import (  # noqa: E402
    PHERC0268_SHA256, PROFILE, _pending_qc_job, _surface,
)


class FailingExecutor(FixtureQcExecutor):
    """Writes a render the size of a real one, then fails the way the real
    executor failed on gpu-1 all through July."""

    def __init__(self, payload_bytes: int = 4096) -> None:
        super().__init__()
        self.executions = 0
        self.payload_bytes = payload_bytes

    def execute(self, claim, attempt_dir):
        self.executions += 1
        output = Path(attempt_dir) / "scientific-output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "layers.tif").write_bytes(b"\0" * self.payload_bytes)
        (output / "ct-render.log").write_text("rendering\n")
        raise RuntimeError("scientific QC adapter failed with exit code 1")


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _job(store: FleetStore, digest: str = PHERC0268_SHA256) -> str:
    surface = _surface(store, area=5.0, digest=digest)
    return _pending_qc_job(store, surface)


def _attempt_dirs(run_root: Path) -> list[Path]:
    return [p for p in run_root.rglob("*") if p.is_dir()
            and (p / "RETRYABLE_QC_RECEIPT.json").is_file()]


def _bytes_under(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# -- the attempt directory --------------------------------------------------

def test_a_failed_attempt_keeps_its_receipt(store, tmp_path):
    """The receipt is the whole record that this happened. It stays."""
    _job(store)
    worker = SurfaceQcWorker(store, "gpu-0", FailingExecutor(),
                             tmp_path / "runs", profile_id=PROFILE)
    worker.run_one()

    attempts = _attempt_dirs(tmp_path / "runs")
    assert len(attempts) == 1
    receipt = attempts[0] / "RETRYABLE_QC_RECEIPT.json"
    assert receipt.is_file(), "the failure left no record of itself"


def test_a_failed_attempt_does_not_keep_its_render(store, tmp_path):
    """194.1 GB of renders belonging to attempts that reached no conclusion.
    A render is reproducible from the CT and the surface; the receipt is not,
    so the receipt is what a failed attempt is allowed to cost."""
    _job(store)
    worker = SurfaceQcWorker(store, "gpu-0", FailingExecutor(payload_bytes=200_000),
                             tmp_path / "runs", profile_id=PROFILE)
    worker.run_one()

    attempts = _attempt_dirs(tmp_path / "runs")
    assert not (attempts[0] / "scientific-output" / "layers.tif").exists(), (
        "the failed attempt kept its layer stack; this is the 158 MB that was "
        "written afresh on every one of 7,463 retries")
    assert _bytes_under(attempts[0]) < 100_000, (
        "the failed attempt is still carrying bulk output")


def test_the_logs_of_a_failed_attempt_survive_the_cleanup(store, tmp_path):
    """Diagnosing why it failed is the point of keeping anything at all."""
    _job(store)
    worker = SurfaceQcWorker(store, "gpu-0", FailingExecutor(),
                             tmp_path / "runs", profile_id=PROFILE)
    worker.run_one()

    attempt = _attempt_dirs(tmp_path / "runs")[0]
    kept = {p.name for p in attempt.rglob("*") if p.is_file()}
    assert "ct-render.log" in kept, (
        "the render log is small and is the only account of what the adapter "
        "was doing when it failed")


def test_a_successful_attempt_is_untouched(store, tmp_path):
    """Cleanup belongs to the failure path only. A measured surface keeps
    everything it produced."""
    _job(store)
    worker = SurfaceQcWorker(store, "gpu-0", FixtureQcExecutor(),
                             tmp_path / "runs", profile_id=PROFILE)
    worker.run_one()

    assert not _attempt_dirs(tmp_path / "runs"), "this attempt should not have failed"
    produced = list((tmp_path / "runs").rglob("QC_RESULT.json"))
    assert produced, "the successful attempt kept no result"


# -- the loop ---------------------------------------------------------------

def test_the_worker_stops_claiming_once_every_claim_has_failed(store, tmp_path):
    """`run` counted these failures and printed at 5, 25, 100 and 500, then
    kept going. On gpu-1 it kept going 7,463 times."""
    for n in range(6):
        _job(store, digest=f"{n:08x}" + "0" * 56)
    executor = FailingExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE, stop_after_retryable=3)

    worker.run(idle_exit=True)

    assert executor.executions == 3, (
        f"the worker made {executor.executions} attempts after being told to "
        "stop at 3 consecutive retryable failures")


def test_one_success_clears_the_count(store, tmp_path):
    """The brake is for a worker that never succeeds. An intermittent outage
    is what "retryable" is for and must not trip it."""
    class FlakyExecutor(FailingExecutor):
        def execute(self, claim, attempt_dir):
            self.executions += 1
            if self.executions == 2:
                return FixtureQcExecutor.execute(self, claim, attempt_dir)
            return FailingExecutor.execute(self, claim, attempt_dir)

    for n in range(6):
        _job(store, digest=f"{n:08x}" + "0" * 56)
    executor = FlakyExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE, stop_after_retryable=3)

    worker.run(idle_exit=True)

    assert executor.executions > 3, (
        "a success in the middle must reset the count, or any flaky outage "
        "stops the fleet")


def test_the_brake_is_off_by_default_for_callers_that_did_not_ask(store, tmp_path):
    """Existing callers constructed this worker without the parameter. Adding
    a default brake would change what they do without their asking."""
    worker = SurfaceQcWorker(store, "gpu-0", FailingExecutor(),
                             tmp_path / "runs", profile_id=PROFILE)
    assert worker.stop_after_retryable is None
