"""What a long job is doing while it does it.

The worker ran its adapter with `capture_output=True`, which buffers both pipes
until the process exits. For a P5 job that is twenty-six minutes in which the
only observable facts are "it started" and, much later, "it finished" -- nothing
in `docker logs`, nothing in the control plane, nothing on the panel. Watching
p5-b7cd63a5e68f4c meant reading the size of a zarr on disk and the GPU's
utilisation, neither of which is progress and both of which are guesses.

Two properties make it not blind, and they are separate:

  * the output is echoed as it arrives, so a person on the host sees it live;
  * the most recent line reaches the control plane, so the panel does too.

The awkward part is the second half of the first. tqdm draws its bar by writing
carriage returns, not newlines: reading the pipe a line at a time yields nothing
at all until the bar is finished, which is exactly when the information stops
being useful.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402


# -- cutting the stream ------------------------------------------------------


def test_a_carriage_return_ends_a_line() -> None:
    """Or a progress bar is invisible until it stops being interesting."""
    lines, rest = ink_worker.split_progress("Infer: 1%\rInfer: 2%\rInfer: 3")

    assert lines == ["Infer: 1%", "Infer: 2%"]
    assert rest == "Infer: 3"


def test_newlines_still_end_lines() -> None:
    lines, rest = ink_worker.split_progress("Wrote a.tif\nWrote b.tif\npartial")

    assert lines == ["Wrote a.tif", "Wrote b.tif"]
    assert rest == "partial"


def test_the_two_mix_and_empty_pieces_are_dropped() -> None:
    """`\\r\\n` is two separators in a row, not an empty line of output."""
    lines, rest = ink_worker.split_progress("a\r\nb\n\rc")

    assert lines == ["a", "b"]
    assert rest == "c"


def test_a_chunk_with_no_separator_is_all_remainder() -> None:
    """Held, not emitted: half a line reported as a whole one is a lie about
    what the process said."""
    assert ink_worker.split_progress("Infer:  63%|") == ([], "Infer:  63%|")


# -- running the thing -------------------------------------------------------


def test_output_arrives_before_the_process_exits() -> None:
    """The whole point. `capture_output=True` returns everything at once, at
    the end, which is the one moment progress is worth nothing."""
    seen: list[tuple[float, str]] = []
    started = time.monotonic()

    completed = ink_worker.run_streaming(
        [sys.executable, "-c",
         "import sys, time\n"
         "for i in range(3):\n"
         "    sys.stderr.write(f'step {i}\\r'); sys.stderr.flush(); time.sleep(0.4)\n"
         "sys.stdout.write('done\\n')\n"],
        timeout=30, env=None,
        on_line=lambda source, line: seen.append((time.monotonic() - started, line)))

    assert completed.returncode == 0
    assert [line for _, line in seen if line.startswith("step")] == [
        "step 0", "step 1", "step 2"]
    # The first line has to arrive while the process is still running. If the
    # implementation regressed to buffering, every timestamp would sit at the end.
    first = next(at for at, line in seen if line == "step 0")
    assert first < 1.0, f"the first line took {first:.2f}s: this is still buffered"


def test_the_tails_are_what_they_always_were() -> None:
    """Everything downstream reads completed.stdout and completed.stderr. The
    receipt, the P3 lineage parse and the failure payload all depend on the two
    staying separate and complete."""
    completed = ink_worker.run_streaming(
        [sys.executable, "-c",
         "import sys\n"
         "sys.stdout.write('out-one\\nout-two\\n')\n"
         "sys.stderr.write('err-one\\n')\n"],
        timeout=30, env=None, on_line=lambda source, line: None)

    assert completed.stdout.splitlines() == ["out-one", "out-two"]
    assert completed.stderr.splitlines() == ["err-one"]


def test_which_pipe_a_line_came_from_is_reported() -> None:
    """A traceback and a progress bar are both "the last thing it said". Which
    stream a line came from is what tells them apart."""
    seen: list[tuple[str, str]] = []

    ink_worker.run_streaming(
        [sys.executable, "-c",
         "import sys; sys.stdout.write('o\\n'); sys.stderr.flush(); sys.stderr.write('e\\n')"],
        timeout=30, env=None,
        on_line=lambda source, line: seen.append((source, line)))

    assert ("stdout", "o") in seen
    assert ("stderr", "e") in seen


def test_a_run_that_overruns_still_raises_TimeoutExpired() -> None:
    """The worker has a branch for this that writes "timed out after Ns". A
    different exception would land in the generic handler and report the
    traceback instead of the fact."""
    with pytest.raises(subprocess.TimeoutExpired):
        ink_worker.run_streaming(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1, env=None, on_line=lambda source, line: None)


def test_a_timed_out_child_is_not_left_running() -> None:
    """A killed job whose process survives holds the GPU it was using."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        ink_worker.run_streaming(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1, env=None, on_line=lambda source, line: None)

    # If the child were merely abandoned, the reader threads would keep its
    # pipes open and this call would not have returned for thirty seconds.
    assert time.monotonic() - started < 10


# -- and the last line reaches the control plane -----------------------------


def test_the_worker_carries_progress_into_the_heartbeat() -> None:
    """The heartbeat already writes to this row on a timer. Progress rides on
    the write that is happening anyway rather than adding one of its own."""
    source = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    beat = source[source.index("    def beat()"):]
    beat = beat[:beat.index("\n    store.mark_running")]

    assert "progress=" in beat, "the heartbeat renews the lease and says nothing else"


def test_the_store_writes_progress_in_that_same_statement() -> None:
    import job_store

    heartbeat = job_store.InkJobStore.heartbeat
    import inspect

    body = inspect.getsource(heartbeat)
    assert "progress" in body
    assert body.count("UPDATE ink_jobs") == 1, (
        "progress costs a second statement; it should ride the lease renewal")


def test_a_job_row_carries_its_progress_outward() -> None:
    """Written and never read would be the same blindness with more columns."""
    import inspect

    import job_store

    assert "progress" in inspect.getsource(job_store.InkJobStore.jobs), (
        "the API cannot show what the query does not select")


def test_a_new_attempt_does_not_inherit_the_last_one_s_progress() -> None:
    """A retry showing where the previous attempt got to is worse than showing
    nothing: it reads as this attempt's position."""
    import inspect

    import job_store

    assert "progress=NULL" in inspect.getsource(job_store.InkJobStore.mark_running)


def test_the_column_exists_wherever_this_schema_is_applied() -> None:
    migrations = ROOT / "framework/stages/03-ink/fleet/migrations"
    applied = "\n".join(p.read_text() for p in sorted(migrations.glob("*.sql")))

    assert "ADD COLUMN IF NOT EXISTS progress" in applied, (
        "the heartbeat writes a column no deployment has")
