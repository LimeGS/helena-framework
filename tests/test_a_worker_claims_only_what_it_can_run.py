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
    containerfile = (ROOT / "containers/images/Containerfile.worker").read_text()

    assert "Pillow==12.3.0" in requirements
    assert "import PIL" in containerfile
