"""A card somebody else is using is a schedulable fact, not a mystery.

On gpu-1, llama.cpp held 9,264 MiB of 12,288 across two GTX 1660s. Every QC
job that reached ink screening then died with

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 MiB.
    GPU 0 has a total capacity of 5.62 GiB of which 1.19 MiB is free.

and the fleet recorded it as `RETRYABLE_QC_UNAVAILABLE`, error "command failed
with exit code 1: /opt/conda/bin/python3". That sentence is true of a full
card, a dead S3 bucket and a syntax error alike. Working out which took
reading 7,463 receipts and then the logs underneath them.

Two things follow, and both are here.

Say it. A receipt that names GPU exhaustion, with the free and total MiB it
saw, is diagnosable at a glance instead of by archaeology.

Do not pay for the render first. The render ran to completion -- minutes of
GPU, a 158 MB layer stack, exit code 0 -- and only then did inference find
there was no room. Reading free VRAM first costs milliseconds and refuses
before any of that.

The existing GPU preflight in run_gpu_tier_supervisor checks memory.total,
which is why this was invisible: a 6 GiB card passes a 6 GiB minimum while
4.8 GiB of it belongs to somebody else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "tests"))

from fleet import qc_worker as qc  # noqa: E402
from fleet.qc_worker import SurfaceQcWorker  # noqa: E402
from fleet.store import FleetStore  # noqa: E402
from test_a_failing_qc_worker_does_not_fill_the_disk import (  # noqa: E402
    FailingExecutor,
)
from test_a_qc_worker_asks_again_before_it_measures import (  # noqa: E402
    PHERC0268_SHA256, PROFILE, _pending_qc_job, _surface,
)

# The real message, from /ssd/vc3d/runs/surface-qc-v2 on 2026-08-21.
REAL_OOM = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 MiB. "
    "GPU 0 has a total capacity of 5.62 GiB of which 1.19 MiB is free."
)


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _job(store: FleetStore, digest: str = PHERC0268_SHA256) -> str:
    return _pending_qc_job(store, _surface(store, area=5.0, digest=digest))


# -- recognising it --------------------------------------------------------

@pytest.mark.parametrize("text", [
    REAL_OOM,
    "torch.cuda.OutOfMemoryError: CUDA out of memory",
    "RuntimeError: CUDA error: out of memory",
    "cuBLAS error: CUBLAS_STATUS_ALLOC_FAILED",
])
def test_the_shapes_a_full_card_actually_reports_are_recognised(text):
    assert qc.is_gpu_exhaustion(text)


@pytest.mark.parametrize("text", [
    "RuntimeError: command failed with exit code 1: /opt/conda/bin/python3",
    "FileNotFoundError: /artifacts/surfaces/PHerc826/abc",
    "OSError: [Errno 28] No space left on device",
    "RuntimeError: published QC evidence failed verification",
])
def test_an_unrelated_failure_is_not_called_a_full_card(text):
    assert not qc.is_gpu_exhaustion(text), (
        "calling everything GPU exhaustion is the same unreadable receipt "
        "with a new name on it")


# -- reading the card ------------------------------------------------------

def test_free_memory_is_read_not_total():
    """nvidia-smi is asked for memory.free. The supervisor's existing
    preflight asks for memory.total, which is what let a card with 1 MiB free
    pass a 6 GiB minimum."""
    rows = "1290, 6144\n950, 6144\n"
    assert qc.parse_gpu_memory(rows) == [(1290, 6144), (950, 6144)]


def test_a_host_with_no_nvidia_smi_reports_nothing_rather_than_zero():
    """Zero free would refuse every job on a CPU fixture host. Not knowing is
    not the same as knowing there is no room."""
    assert qc.gpu_memory(nvidia_smi="/nonexistent/nvidia-smi") is None


# -- refusing before the render -------------------------------------------

def test_a_job_is_refused_before_the_executor_runs_when_the_card_is_full(
        store, tmp_path, monkeypatch):
    _job(store)
    monkeypatch.setattr(qc, "gpu_memory", lambda **_: [(1290, 6144), (950, 6144)])
    executor = FailingExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE, minimum_free_vram_mib=2048)

    result = worker.run_one()

    assert executor.executions == 0, (
        "the render ran anyway; the whole point is not paying for it")
    assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
    assert result.get("reason_code") == "GPU_MEMORY_EXHAUSTED"


def test_the_refusal_records_what_it_saw(store, tmp_path, monkeypatch):
    """The number is the difference between "retry later" and "somebody else
    is using this card"."""
    _job(store)
    monkeypatch.setattr(qc, "gpu_memory", lambda **_: [(1290, 6144), (950, 6144)])
    worker = SurfaceQcWorker(store, "gpu-0", FailingExecutor(), tmp_path / "runs",
                             profile_id=PROFILE, minimum_free_vram_mib=2048)

    worker.run_one()

    receipt = next((tmp_path / "runs").rglob("RETRYABLE_QC_RECEIPT.json"))
    import json
    body = json.loads(receipt.read_text())
    assert body["reason_code"] == "GPU_MEMORY_EXHAUSTED"
    assert body["gpu_memory_free_mib"] == 1290, "the best card, not the first"
    assert body["gpu_memory_total_mib"] == 6144
    assert body["minimum_free_vram_mib"] == 2048


def test_room_on_any_one_card_is_enough(store, tmp_path, monkeypatch):
    """A job runs on one GPU. A full card beside an empty one is not an
    outage."""
    _job(store)
    monkeypatch.setattr(qc, "gpu_memory", lambda **_: [(120, 6144), (5000, 6144)])
    executor = FailingExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE, minimum_free_vram_mib=2048)

    worker.run_one()

    assert executor.executions == 1, "there was room and the job did not run"


def test_an_unreadable_card_does_not_stop_the_work(store, tmp_path, monkeypatch):
    """Fail open. A worker that refuses every job because nvidia-smi is
    missing has replaced an outage with a worse one."""
    _job(store)
    monkeypatch.setattr(qc, "gpu_memory", lambda **_: None)
    executor = FailingExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE, minimum_free_vram_mib=2048)

    worker.run_one()

    assert executor.executions == 1


def test_the_check_is_off_unless_a_deployment_asks(store, tmp_path):
    worker = SurfaceQcWorker(store, "gpu-0", FailingExecutor(), tmp_path / "runs",
                             profile_id=PROFILE)
    assert worker.minimum_free_vram_mib is None


# -- naming it after the fact ---------------------------------------------

def test_an_oom_that_escapes_the_executor_is_still_named(store, tmp_path):
    """The preflight is a floor, not a guarantee: the card can fill between
    the check and the allocation, which is exactly what a shared host does."""
    class OomExecutor(FailingExecutor):
        def execute(self, claim, attempt_dir):
            self.executions += 1
            raise RuntimeError(REAL_OOM)

    _job(store)
    worker = SurfaceQcWorker(store, "gpu-0", OomExecutor(), tmp_path / "runs",
                             profile_id=PROFILE)

    result = worker.run_one()

    assert result.get("reason_code") == "GPU_MEMORY_EXHAUSTED", (
        "the receipt still says only that a command exited 1")
