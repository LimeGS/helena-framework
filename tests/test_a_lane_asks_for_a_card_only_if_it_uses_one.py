"""A GPU requirement is a claim about the work, not a habit of the table.

`gpu_required` does one thing: it decides which workers may claim the job.
`AND (gpu_required=false OR <this worker has a card>=true)` in the claim query,
and nothing else. It is not a resource reservation and it does not make a
runner faster.

The lane table defaulted it to True, so four lanes asked for a card they never
touch. vc_render_tifxyz is OpenMP over streamed Zarr chunks -- upstream
volume-cartographer offers no CUDA path for rendering at all, its single CUDA
option in the whole project being lasagna's maxflow -- and the chunk-gather
lane, the vetting card and the plate composer import no torch between them: a
fetch-and-write loop, statistics over a probability map that already exists, and
an image pipeline.

It cost nothing while the only worker advertising those phases needed a card for
P5 anyway. It would have cost the first time somebody wanted to render, screen
or compose on a machine without one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import PHASE_LANES, INK_ADAPTERS  # noqa: E402

# The two that compute on a card, and what makes each of them do it.
GPU_LANES = {
    ("P1", "spiral-fit"): "fit_spiral.py runs a torch fit; the runner launches "
                          "it through the villa interpreter",
    ("P5", "ink-adapter"): "every ink adapter imports torch and runs a "
                           "checkpoint",
}


def lanes():
    return [(phase, lane, spec)
            for phase in sorted(PHASE_LANES)
            for lane, spec in PHASE_LANES[phase].items()]


def test_only_the_lanes_that_compute_on_a_card_ask_for_one():
    asking = {(phase, lane) for phase, lane, spec in lanes()
              if spec.get("gpu_required", True)}

    assert asking == set(GPU_LANES), (
        "a lane's GPU requirement changed. It decides which workers may claim "
        "the job, so adding one narrows where the phase can run and removing "
        "one lets a card-less worker take it: say which, and why, here."
    )


@pytest.mark.parametrize("phase,lane", sorted(
    (phase, lane) for phase, lane, spec in lanes()
    if not spec.get("gpu_required", True)))
def test_a_cpu_lane_declares_it_rather_than_inheriting_a_default(phase, lane):
    """Explicit, because the table's default is True: a lane that silently
    inherits it is a lane nobody decided about."""
    assert "gpu_required" in PHASE_LANES[phase][lane]


@pytest.mark.parametrize("phase,lane", [
    ("P4", "vc-render-tifxyz"), ("P4", "chunk-gather"),
    ("P7", "vetting-card"), ("P9", "official-plates"),
])
def test_the_four_that_were_corrected_have_no_gpu_path_in_their_runner(phase, lane):
    """Evidence rather than a restatement of the table. If one of these grows a
    torch import it has become GPU work, and its lane has to say so again."""
    runner = ROOT / PHASE_LANES[phase][lane]["runner"]
    if not runner.is_file():
        pytest.skip(f"{runner} is not vendored here")
    source = runner.read_text(encoding="utf-8", errors="replace")

    assert "import torch" not in source, f"{runner.name} imports torch"
    assert "cupy" not in source, f"{runner.name} reaches for cupy"


def test_the_ink_adapters_are_the_reason_p5_asks_for_a_card():
    """P5's requirement is on the phase's lane, and what justifies it is in the
    adapters underneath it."""
    torch_users = [
        relative for relative in INK_ADAPTERS
        if (ROOT / relative).is_file()
        and "import torch" in (ROOT / relative).read_text(
            encoding="utf-8", errors="replace")
    ]
    assert torch_users, "no ink adapter imports torch; why does P5 need a card?"


def test_a_gpu_requirement_is_about_claiming_not_about_speed():
    """Kept as a property because it is the thing that was misunderstood: the
    flag reaches exactly one place, the claim filter."""
    source = (ROOT / "framework/stages/03-ink/fleet/job_store.py").read_text(
        encoding="utf-8")
    assert "AND (gpu_required=false OR %s=true)" in source
