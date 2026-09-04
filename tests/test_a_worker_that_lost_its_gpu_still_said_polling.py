"""helena-ink-0 lost its GPU passthrough and kept saying POLLING anyway.

nvidia-smi inside the container started answering "No devices were found" --
a passthrough glitch, not a crash -- while helena-ink-9um, on the same host
and carrying the identical DeviceRequests, kept seeing its card the whole
time. The worker process never died; its "do I have a GPU" answer had been
decided once, at startup, and never asked again, so it silently stopped
claiming P5 work. Six jobs sat pending for five hours. `docker ps` said "Up",
the fleet row said POLLING, and nothing anywhere said a card was missing --
`docker restart` was the entire fix, because nothing had broken that a fresh
probe would not have caught on its own.

What is worth testing here is the three pieces the incident showed had never
been written: a probe that asks fresh, on every poll, rather than once; a
worker row that can say a GPU is gone while still honestly saying POLLING,
instead of being silent about the one and correct about the other; and a
claim that reads the same fresh answer it records, rather than a stale one
from a different, once-a-minute probe.
"""

from __future__ import annotations

import inspect
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402
import job_store  # noqa: E402


def _fake_nvidia_smi(tmp_path: Path, *, exit_code: int, stdout: str = "",
                     stderr: str = "") -> str:
    script = tmp_path / "nvidia-smi"
    script.write_text(
        "#!/bin/sh\n"
        + (f"echo '{stdout}'\n" if stdout else "")
        + (f"echo '{stderr}' >&2\n" if stderr else "")
        + f"exit {exit_code}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


# --------------------------------------------------------------------------
# The probe itself: what nvidia-smi says, asked fresh
# --------------------------------------------------------------------------

def test_a_device_nvidia_smi_lists_is_visible(tmp_path):
    present = _fake_nvidia_smi(
        tmp_path, exit_code=0,
        stdout="GPU-3d0b1af2-0000-0000-0000-000000000000")
    assert ink_worker.worker_gpu_visible(nvidia_smi=present) is True


def test_no_devices_found_is_not_visible_even_though_the_binary_ran(tmp_path):
    """The exact failure from the incident: the binary is present, runs, and
    says there is nothing to use -- which DeviceRequests never learns."""
    broken = _fake_nvidia_smi(
        tmp_path, exit_code=6, stderr="No devices were found")
    assert ink_worker.worker_gpu_visible(nvidia_smi=broken) is False


def test_an_empty_listing_with_exit_zero_is_not_visible(tmp_path):
    """Defensive: a clean exit with nothing on stdout is not proof of a card
    either, whatever the driver meant by it."""
    quiet = _fake_nvidia_smi(tmp_path, exit_code=0, stdout="")
    assert ink_worker.worker_gpu_visible(nvidia_smi=quiet) is False


def test_a_missing_binary_is_not_a_missing_card():
    """None, not False: a worker with no nvidia-smi on its PATH at all has
    never claimed a GPU, and is not the worker this column is about."""
    assert ink_worker.worker_gpu_visible(
        nvidia_smi=str(Path("/definitely/not/here/nvidia-smi"))) is None


def test_a_hung_driver_call_is_treated_as_not_visible(monkeypatch):
    """A timeout is a SubprocessError, not an OSError from a missing binary --
    the binary is present and something is wrong with it, which is "not
    proven live" rather than "never claimed one"."""
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr(ink_worker.subprocess, "run", fake_run)
    assert ink_worker.worker_gpu_visible() is False


# --------------------------------------------------------------------------
# Persisted where the fleet page already reads
# --------------------------------------------------------------------------

class Recorder:
    """The cursor, remembering what it was asked."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []

    def execute(self, sql: str, args: tuple = ()) -> None:
        self.statements.append((sql, args))

    def fetchall(self) -> list:
        return []

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Connection:
    def __init__(self, cursor: Recorder) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_the_claim_writes_gpu_visible_onto_the_workers_row():
    """ink_workers gained gpu_visible in migration 008 -- the same row
    last_poll_at already lives on, written by the same heartbeat, so a reader
    does not need a second table or a second timestamp to know when the
    probe last ran."""
    store = job_store.InkJobStore("postgresql://unused")
    recorder = Recorder()
    store._connect = lambda: Connection(recorder)  # noqa: SLF001

    store.claim(worker_id="helena-ink-0", host_id="gpu-1", has_gpu=False,
               gpu_visible=False)

    sql, args = next(
        s for s in recorder.statements if s[0].lstrip().startswith("INSERT"))
    assert "gpu_visible" in sql
    assert False in args, (
        "the worker's row does not carry whether its GPU is visible")


def test_a_worker_that_never_claimed_a_gpu_writes_no_opinion():
    """None survives the round trip rather than being coerced to False --
    a CPU-only worker has nothing to probe, and is not the same fact as a
    GPU worker whose card just went missing."""
    store = job_store.InkJobStore("postgresql://unused")
    recorder = Recorder()
    store._connect = lambda: Connection(recorder)  # noqa: SLF001

    store.claim(worker_id="cpu-runner", host_id="cpu-1", has_gpu=False)

    sql, args = next(
        s for s in recorder.statements if s[0].lstrip().startswith("INSERT"))
    assert "gpu_visible" in sql
    assert None in args


# --------------------------------------------------------------------------
# Read back on the fleet page: POLLING and gpu-dead are not the same state
# --------------------------------------------------------------------------

def test_a_worker_can_be_polling_and_have_lost_its_gpu_at_once():
    """The exact shape of the incident: last_poll_at kept advancing right
    through it, so `state` alone said this worker was fine. `gpu_visible` is
    the fact `state` could not carry."""
    now = datetime.now(timezone.utc)

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def execute(self, *_args): pass
        def fetchall(self):
            return [
                ("helena-ink-0", "gpu-1", "helena-worker-gpu", ["P4", "P5"],
                 now - timedelta(seconds=3), now - timedelta(hours=5), 3.0,
                 False),
                ("helena-ink-9um", "gpu-1", "helena-ink-9um", ["P5"],
                 now - timedelta(seconds=2), now - timedelta(seconds=8), 2.0,
                 True),
            ]

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def cursor(self): return Cursor()

    store = job_store.InkJobStore("postgresql://unused")
    store._connect = lambda **_kwargs: Connection()

    blind, sighted = store.workers()

    assert blind["worker_id"] == "helena-ink-0"
    assert blind["state"] == "POLLING", (
        "a lost GPU must not be mistaken for a dead worker -- it is still "
        "looking, just not for GPU work")
    assert blind["gpu_visible"] is False
    assert sighted["gpu_visible"] is True


# --------------------------------------------------------------------------
# The loop: eligibility and the recorded row read the same fresh probe
# --------------------------------------------------------------------------

def test_the_loop_asks_the_probe_before_every_claim():
    """Not host_state()'s once-a-minute reading, and not something decided
    once before the while loop starts -- both are exactly what let
    helena-ink-0 keep claiming nothing, silently, for five hours."""
    body = inspect.getsource(ink_worker.main)
    while_loop = body[body.index("while True:"):]
    assert "worker_gpu_visible()" in while_loop, (
        "the loop body never re-probes the GPU")


def test_the_claim_receives_the_same_probe_it_will_record():
    """One fresh answer, used for both eligibility and the row the panel
    reads -- not two probes that can disagree with each other."""
    body = inspect.getsource(ink_worker.main)
    call = body[body.index("job = store.claim("):]
    call = call[:call.index("except")]
    assert "has_gpu=bool(gpu_visible)" in call
    assert "gpu_visible=gpu_visible" in call
