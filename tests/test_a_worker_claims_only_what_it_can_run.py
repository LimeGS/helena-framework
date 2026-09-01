"""A worker claiming a phase whose binary its image does not carry.

The ink image carries a three-tool VC3D bundle -- grow, render, mcp -- and no
vc_flatten. It claimed P3 anyway, failed five surfaces on
`FileNotFoundError: /opt/campaignx/vc3d/bin/vc_flatten`, and left the queue
looking like flattening was broken rather than misrouted. Nothing in the queue
knew which runtime could run what, so every worker claimed everything.

The failure mode is worse than a job that waits: the job is consumed, marked
failed, and the phase reports a real-looking verdict on work that never ran.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import InkJobStore, lane_for  # noqa: E402


class Recorder:
    """The cursor, remembering what it was asked."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []

    def execute(self, sql: str, args: tuple = ()) -> None:
        self.statements.append((sql, args))

    def fetchall(self) -> list:
        return []

    def fetchone(self) -> None:
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


def claim_with(phases):
    store = InkJobStore("postgresql://unused")
    recorder = Recorder()
    store._connect = lambda: Connection(recorder)  # noqa: SLF001
    store.claim(worker_id="w", host_id="h", phases=phases)
    return next(s for s in recorder.statements if s[0].lstrip().startswith("SELECT"))


def test_a_worker_with_phases_asks_only_for_those():
    sql, args = claim_with(["P2", "P3"])
    assert "phase = ANY(" in sql
    assert ["P2", "P3"] in args


def test_a_worker_without_phases_still_claims_everything():
    """A single-runtime deployment has one worker and no routing to do, and a
    filter that defaults to something would silently strand its queue."""
    sql, args = claim_with(None)
    # The predicate is still in the statement -- one query, not two -- but NULL
    # makes it true for every row.
    assert "phase = ANY(" in sql
    # Only the phase arguments are NULL. The capability arguments that follow
    # are this worker's own hardware and are always bound: a claim that left
    # them NULL would be a worker declining to say what it has, which is how
    # the same misrouting happens through the other column.
    phase_args = [a for a in args if a is None or a == "h" or isinstance(a, list)]
    assert len(phase_args) >= 3, args


def test_a_worker_says_what_hardware_it_has() -> None:
    """ink_jobs has carried gpu_required and minimum_vram_gb since it was
    written, and the claim read neither.

    A CPU-only host would take a job that needs a card, fail it on the missing
    device, burn an attempt and leave the queue reporting a real-looking
    verdict on work that never ran -- the same failure this file was opened
    for, through the other column. The segmentation queue has always filtered
    on its worker's capabilities; this one only looked like it did.
    """
    sql, args = claim_with(None)
    assert "gpu_required=false OR" in sql, (
        "the claim ignores whether the job needs a GPU"
    )
    assert "minimum_vram_gb <=" in sql, "the claim ignores how much VRAM it needs"
    # False and 0.0 are what a CPU-only host binds, so both must survive being
    # falsy -- a truthiness test here would drop exactly the host that matters.
    assert any(a is False or a == 0.0 or a is True or isinstance(a, float)
               for a in args), args


def test_cpu_certify_and_flatten_jobs_are_claimable_by_their_runner() -> None:
    """The P2/P3 fleet runner intentionally has no GPU runtime.

    Leaving the generic lane default in place marks these jobs GPU-only, so the
    worker that carries ``vc_flatten`` can never claim them.
    """
    for phase in ("P2", "P3"):
        _lane_id, lane = lane_for({"phase": phase, "parameters": {}})
        assert lane.get("gpu_required") is False, phase


def test_cpu_p8_jobs_are_claimable_by_the_worker_that_accepts_p8() -> None:
    """gpu-1-fleet-runner advertises P8 but intentionally has no GPU.

    Column-atlas and mesh-relations only download meshes and measure geometry.
    Marking either GPU-only leaves a valid job pending forever because the GPU
    ink worker advertises P4/P5/P7/P9, not P8.
    """
    for lane_id in ("column-atlas", "mesh-relations", "vc3d-tifxyz-merge"):
        _selected, lane = lane_for({
            "phase": "P8", "parameters": {"lane": lane_id},
        })
        assert lane.get("gpu_required") is False, lane_id


def test_cpu_p8_worker_carries_column_atlas_python_runtime() -> None:
    """A claimable lane must also be executable in the claiming image.

    ``column-atlas`` imports PIL before doing any work.  The CPU routing fix
    exposed that the fleet image did not install Pillow, so a correctly
    claimed job immediately burned its only attempt with ModuleNotFoundError.
    Keep both the dependency pin and the image's build-time import smoke tied
    to the lane that needs them.
    """
    requirements = (
        ROOT / "framework/stages/01-segmentation/fleet/requirements-worker.txt"
    ).read_text()
    containerfile = (ROOT / "containers/images/Containerfile.worker-cpp").read_text()

    assert "Pillow==12.3.0" in requirements
    assert "import PIL" in containerfile


# -- the same failure, through the runtime rather than the phase -------------
#
# Three lanes need an image their claiming worker may not be running. The
# worker already refuses those at execution, by name, which is a good message
# and a burned attempt: on a host running both a general ink worker and the
# 9 um one, the general worker wins the race, refuses, and the job is failed
# before the worker that could have run it ever polls. Observed exactly that on
# gpu-1 -- p5-4a8e0ba41a784d, claimed by gpu-1-ink0, failed on its only attempt
# while gpu-1-ink9um sat idle ten seconds away.
#
# So this is the same fix as `phases`, one column over: what a worker cannot
# run, it does not claim.


class Batch(Recorder):
    """A cursor with rows to hand out."""

    def __init__(self, rows: list[tuple]) -> None:
        super().__init__()
        self.rows = rows

    def fetchall(self):
        # The lease-recycling statement runs first and expects no rows; the
        # candidate SELECT is the one that wants these.
        if self.statements and self.statements[-1][0].lstrip().startswith("SELECT"):
            return self.rows
        return []


def candidate(job_id: str, profile_id: str | None, parameters: dict) -> tuple:
    # job_id, sample_id, profile_id, parameters, attempts, max_attempts,
    # phase, component, mission_id
    return (job_id, "PHerc0332", profile_id, parameters, 0, 3, "P5", None, None)


def candidates_for(runtime, rows):
    store = InkJobStore("postgresql://unused")
    batch = Batch(rows)
    store._connect = lambda: Connection(batch)  # noqa: SLF001
    return store._runnable_candidate(rows, runtime)  # noqa: SLF001


def test_a_worker_skips_a_lane_that_declares_another_image() -> None:
    nine_um = candidate("p5-9um", "ink-9um-hybrid-3d2d-screening@1.0.0", {})
    ordinary = candidate("p5-plain", None, {})

    picked = candidates_for("helena-worker-gpu", [nine_um, ordinary])

    assert picked is not None and picked[0] == "p5-plain", (
        "the general worker claimed the 9 um job it cannot run")


def test_the_worker_that_carries_the_image_takes_it() -> None:
    nine_um = candidate("p5-9um", "ink-9um-hybrid-3d2d-screening@1.0.0", {})

    assert candidates_for("helena-ink-9um", [nine_um])[0] == "p5-9um"


def test_a_specialist_worker_leaves_the_ordinary_work_alone() -> None:
    """The other direction, which was backwards.

    A job that declares no image needs the ordinary one. The rule was "anything
    that does not need a *different* image", so the 9 um worker -- built around
    one lane's frozen environment -- claimed canonical and timesformer runs and
    failed each in about two seconds, while the worker that could run them sat
    idle beside it.
    """
    ordinary = candidate("p5-plain", None, {})
    canonical = candidate("p5-canon", "ink-canonical-2um-screening@1.1.0", {})

    assert candidates_for("helena-ink-9um", [ordinary, canonical]) is None


def test_a_specialist_still_takes_its_own_lane_from_a_mixed_queue() -> None:
    ordinary = candidate("p5-plain", None, {})
    nine_um = candidate("p5-9um", "ink-9um-hybrid-3d2d-screening@1.0.0", {})

    picked = candidates_for("helena-ink-9um", [ordinary, nine_um])

    assert picked is not None and picked[0] == "p5-9um"


def test_which_images_count_as_specialist_comes_from_the_lanes() -> None:
    """Read from the declarations routing already uses, so a new lane with its
    own image is a specialist the moment it is registered -- not when somebody
    remembers to add it to a list here."""
    from job_store import lane_runtime_images

    images = lane_runtime_images()
    assert "helena-ink-9um" in images
    assert "helena-worker-gpu" not in images


def test_a_queue_holding_only_another_runtime_s_work_hands_out_nothing() -> None:
    """None, not the job. The job waits for the worker that can run it rather
    than being consumed by one that cannot."""
    nine_um = candidate("p5-9um", "ink-9um-hybrid-3d2d-screening@1.0.0", {})

    assert candidates_for("helena-worker-gpu", [nine_um]) is None


def test_an_unlabelled_worker_still_claims_everything() -> None:
    """Same reasoning as phases: a single-runtime deployment has no routing to
    do, and a worker that does not know its own image must not strand a queue.
    require_runtime stays the backstop at execution."""
    nine_um = candidate("p5-9um", "ink-9um-hybrid-3d2d-screening@1.0.0", {})

    assert candidates_for(None, [nine_um])[0] == "p5-9um"


def test_the_claim_asks_for_more_than_one_candidate() -> None:
    """It has to look past the first row to skip one. A LIMIT of 1 makes the
    filter above unreachable for any queue whose head is another runtime's."""
    sql, args = claim_with(["P5"])

    assert "LIMIT %s" in sql, "the candidate count is hard-coded"
    assert any(isinstance(a, int) and a > 1 for a in args), (
        f"the claim still asks for one row: {args}")


def test_the_worker_tells_the_queue_which_image_it_is() -> None:
    """The filter is on the queue; a worker that knows its image and does not
    say so leaves it switched off, which is how this was already true of
    require_runtime and still burned a job."""
    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    call = worker[worker.index("job = store.claim("):]
    call = call[:call.index("except")]      # the whole call, parens and all

    assert "runtime=RUNTIME_IMAGE" in call, (
        "the worker claims without saying which image it is")
