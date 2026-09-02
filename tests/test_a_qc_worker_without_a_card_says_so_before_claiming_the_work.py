"""A QC worker whose GPU would not initialise kept claiming and failing.

Seen on a rented deployment during the segmentation control. The container
had been started with the card and had it; later the host's cgroup state
changed underneath it, and inside the container `nvidia-smi` answered
`Failed to initialize NVML: Unknown Error` and torch saw no device. The
worker's own heartbeat said `cuda_available: false` -- and it went on claiming
QC jobs, ran the adapter, and each one came back

    RETRYABLE_QC_UNAVAILABLE  "command failed with exit code 1: <path>"

which is also what a full card and a dead bucket look like. A restart of the
container was the cure. The worker's part is to say which card is missing, in
the driver's own words, before it spends a lease finding out -- exactly what it
already does for a card with no room.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from fleet import qc_worker as module  # noqa: E402
from fleet.qc_worker import FixtureQcExecutor, SurfaceQcWorker  # noqa: E402
from fleet.store import FleetStore  # noqa: E402
from test_a_qc_worker_asks_again_before_it_measures import (  # noqa: E402
    PHERC0268_SHA256, PROFILE, _pending_qc_job, _surface,
)

NVML = "Failed to initialize NVML: Unknown Error"


class CountingExecutor(FixtureQcExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.executions = 0

    def execute(self, claim, attempt_dir):
        self.executions += 1
        return super().execute(claim, attempt_dir)


@pytest.fixture
def store(tmp_path) -> FleetStore:
    fleet = FleetStore(tmp_path / "fleet.sqlite")
    fleet.initialize()
    return fleet


def _fake_nvidia_smi(tmp_path: Path, *, exit_code: int, says: str) -> str:
    script = tmp_path / "nvidia-smi"
    script.write_text(f"#!/bin/sh\necho '{says}' >&2\nexit {exit_code}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


# -- the driver's words are recognised ---------------------------------------

@pytest.mark.parametrize("said", [
    NVML, "RuntimeError: Found no NVIDIA driver on your system",
    "CUDA_ERROR_NO_DEVICE", "no CUDA-capable device is detected",
])
def test_the_words_a_missing_card_uses_are_known(said):
    assert module.is_gpu_unavailable(said)


def test_a_full_card_is_not_a_missing_card():
    assert not module.is_gpu_unavailable("CUDA out of memory. Tried to allocate")
    assert module.is_gpu_exhaustion("CUDA out of memory. Tried to allocate")


def test_a_present_nvidia_smi_that_cannot_initialise_is_the_finding(tmp_path):
    binary = _fake_nvidia_smi(tmp_path, exit_code=255, says=NVML)
    assert NVML in (module.gpu_probe_failure(nvidia_smi=binary) or "")


def test_a_working_card_and_a_missing_binary_both_say_nothing(tmp_path):
    """Fails open, like gpu_memory(): a host with no nvidia-smi has not said
    it has no card, and refusing every job over an absent binary would
    replace an outage with a worse one."""
    working = _fake_nvidia_smi(tmp_path, exit_code=0, says="GPU 0: something")
    assert module.gpu_probe_failure(nvidia_smi=working) is None
    assert module.gpu_probe_failure(nvidia_smi=str(tmp_path / "absent")) is None


def test_some_other_failure_of_nvidia_smi_is_not_called_a_missing_card(tmp_path):
    grumbling = _fake_nvidia_smi(tmp_path, exit_code=2, says="unknown argument")
    assert module.gpu_probe_failure(nvidia_smi=grumbling) is None


# -- the worker refuses before the executor -----------------------------------

def test_the_job_goes_back_with_the_reason_and_the_executor_never_runs(
        store, tmp_path, monkeypatch):
    surface = _surface(store, area=5.0, digest=PHERC0268_SHA256)
    job_id = _pending_qc_job(store, surface)
    executor = CountingExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE)
    monkeypatch.setattr(worker, "_gpu_unavailable", lambda: NVML)

    outcome = worker.run_one()

    assert executor.executions == 0, "the executor ran on a worker with no card"
    # The store owns the queue result's shape; the worker's reason travels
    # beside it, the way the VRAM floor's does.
    assert outcome["reason_code"] == module.GPU_UNAVAILABLE
    receipts = list((tmp_path / "runs").rglob("RETRYABLE_QC_RECEIPT.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "RETRYABLE_QC_UNAVAILABLE"
    assert receipt["reason_code"] == module.GPU_UNAVAILABLE
    assert NVML in receipt["error"], "the receipt does not say which card is missing"
    assert receipt["no_scientific_conclusion"] is True
    # Not lost and not terminal: the job is back in the queue behind its retry
    # delay, so there is nothing for this worker to claim until it passes.
    assert worker.run_one() is None


def test_a_worker_with_a_card_is_not_slowed_down(store, tmp_path, monkeypatch):
    surface = _surface(store, area=5.0, digest=PHERC0268_SHA256)
    _pending_qc_job(store, surface)
    executor = CountingExecutor()
    worker = SurfaceQcWorker(store, "gpu-0", executor, tmp_path / "runs",
                             profile_id=PROFILE)
    monkeypatch.setattr(worker, "_gpu_unavailable", lambda: None)
    worker.run_one()
    assert executor.executions == 1


# -- and when the failure reaches the receipt anyway, it carries the cause ----

def test_the_adapter_keeps_the_commands_last_words(tmp_path):
    import campaignx_surface_qc_adapter as adapter

    script = tmp_path / "fails.sh"
    script.write_text("#!/bin/sh\necho starting\necho '" + NVML + "'\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    with pytest.raises(RuntimeError) as failed:
        adapter.run_logged([str(script)], tmp_path / "out" / "cmd.log")
    assert "exit code 1" in str(failed.value)
    assert NVML in str(failed.value), "the receipt would say only 'exit code 1'"
    assert module.is_gpu_unavailable(str(failed.value))
