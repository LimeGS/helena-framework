"""ink-3d-dino-guided-diagnostic was accepted at the queue and refused only
when a worker claimed it and called ink_adapter() to build its argv -- a
burned lease and attempt for a job the adapter's own declaration already
says cannot run: it reads a patch manifest and a villa python root, not a
job shape this queue can build one for.

INK_ADAPTERS already carried "unroutable" for exactly this lane. It was
checked in ink_adapter(), which only ink_worker.py called, at claim time.
enqueue() now calls it too, so the same fact refuses the request instead
of the attempt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import InkJobStore, JobRejected  # noqa: E402


def enqueue(**overrides):
    store = InkJobStore("postgresql://unused")
    body = {"sample_id": "PHerc0332", "phase": "P5", "mission_id": "m1",
            "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0",
            "parameters": {"checkpoint": "/models/m/model.ckpt",
                           "tiff_dir": "/layers", "source_pixel_um": 9.362}}
    body.update(overrides)
    return store.enqueue(**body)


def test_an_unroutable_profile_is_refused_before_a_lease_exists():
    with pytest.raises(JobRejected, match="patch manifest"):
        enqueue(profile_id="ink-3d-dino-guided-diagnostic@1.0.0",
               parameters={})


def test_a_routable_profile_still_reaches_the_mission_check():
    """The new call must not swallow or precede the checks that already ran
    -- a profile that routes fine still needs a mission, same as before."""
    with pytest.raises(JobRejected, match="mission"):
        enqueue(mission_id=None)
