"""Bringing your own artifact is allowed. The evidence has to say you did.

Helena is operated by more than one person, and a phase is not always run by
whoever ran the one before it. Somebody flattening on their own machine, or
screening a stack a colleague rendered last week, is ordinary work -- not an
attempt to smuggle something past the record. The platform already allows it:
outside the campaign's own control scroll, P5 takes a `tiff_dir` and runs.

What it did not do is say so. P4 handed a bare path records
`{"kind": "surface_tifxyz", "path": ...}` -- "this came from somewhere I was
told about". P5 handed a bare `tiff_dir` recorded nothing at all:
`rendered_from` stayed None, and a result from an imported stack was
indistinguishable in the evidence from one whose provenance nobody wrote down.

That distinction is the whole point of allowing the import. A researcher
certifying part of a pipeline is entitled to certify exactly the part they
ran -- and a reader is entitled to see where the chain begins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker


def _job(**parameters):
    return {"job_id": "p5-test", "phase": "P5", "sample_id": "PHerc0332",
            "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0",
            "parameters": parameters}


def test_a_supplied_stack_is_recorded_as_supplied():
    note = ink_worker.supplied_input_note(_job(tiff_dir="/data/layers"))
    assert note is not None, (
        "a stack brought from outside left no trace of having been brought")
    assert note["kind"] == "supplied_layer_stack"
    assert note["path"] == "/data/layers"


def test_the_note_says_the_chain_begins_here():
    """Not a warning and not a demerit: a statement of scope. What follows is
    certified; what came before is the supplier's to account for."""
    note = ink_worker.supplied_input_note(_job(tiff_dir="/data/layers"))
    claim = note["non_claim"].lower()
    assert "not produced by this platform" in claim
    assert "begins" in claim


def test_a_platform_rendered_stack_is_not_marked_supplied():
    """`layer_stack` names a P4 this platform ran, and that branch records the
    full lineage. Marking it supplied would be false."""
    assert ink_worker.supplied_input_note(_job(layer_stack="p4-abc")) is None


def test_a_stack_that_is_both_is_not_marked_supplied():
    """The worker resolves `layer_stack` into `tiff_dir` before running, so by
    the time this is asked both are set. The lineage is what counts."""
    assert ink_worker.supplied_input_note(
        _job(layer_stack="p4-abc", tiff_dir="/staged")) is None


def test_a_job_with_neither_gets_no_note():
    assert ink_worker.supplied_input_note(_job()) is None


def test_only_p5_is_answered_here():
    """P4 already records its own bare-path case, and inventing a second
    vocabulary for the same fact is how two records come to disagree."""
    job = _job(tiff_dir="/data/layers")
    job["phase"] = "P4"
    assert ink_worker.supplied_input_note(job) is None
