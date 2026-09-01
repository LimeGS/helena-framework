"""An interrupted control run has to be able to finish.

A point under a policy is one task forever. The platform says so, and it is
right: that is what makes a task identity mean something. The control's policy
version encodes the deployed revision, so re-running the same revision after an
interrupted run cannot re-place the seed -- and the first run had already done
the growing.

That happened. A run was cut off mid-flight; the surface finished growing and
reached QC_PENDING; the next run asked for the seed, got

    409 nothing was queued: all 1 of these points already have a task
        under policy first-letters-control@1.0.0-...

and reported a failure about a job that had in fact succeeded. Worse, had it
carried on, it would have timed out anyway: the wait skips every attempt that
existed before the seed was placed, and on a resume the attempt it is waiting
for is exactly one of those.

Resuming is not a weaker check. Every provenance field is verified from the
attempt the fleet actually ran -- policy version, grid version, snapshot,
seed origin, author, coordinates -- and the P0 binding persisted on the task is
checked here as well, which the creation response never proved about a task
somebody else created.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

import run_first_letters_positive_control as control  # noqa: E402
from panel_client import PanelError  # noqa: E402

BODY = {
    "sample_id": "PHerc0139",
    "points": [{"x": 1.0, "y": 2.0, "z": 3.0}],
    "policy_version": "first-letters-control@1.0.0-d59c2a511df6",
    "grid_version": "first-letters-control-manual-v1",
    "expected_p0_artifact_id": "p0:PHerc0139:392cd753c0bd",
    "expected_p0_artifact_sha256": "a" * 64,
}

ALREADY = PanelError(
    "POST", "/api/segmentation/manual-seeds", 409,
    '{"detail":{"detail":"nothing was queued: all 1 of these points already '
    'have a task under policy first-letters-control@1.0.0-d59c2a511df6",'
    '"why":"A task is identified by volume, grid version, cell and policy '
    'version, so the same point under the same policy is one task forever."}}')


class _Panel:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def call(self, method, path, body=None):
        self.calls.append((method, path))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_a_seed_placed_now_is_reported_as_placed_now() -> None:
    created = {"receipt": {"seed_origin": "human"}, "submitted_by": "tester"}
    panel = _Panel(created)

    placed = control._place_control_seed(panel, BODY)

    assert placed.created is created
    assert placed.resumed is False


def test_a_seed_a_previous_run_placed_is_a_resume_not_a_failure() -> None:
    placed = control._place_control_seed(_Panel(ALREADY), BODY)

    assert placed.resumed is True
    assert placed.created is None


def test_a_resume_does_not_skip_the_attempt_it_is_waiting_for() -> None:
    """The wait excludes the attempts that existed beforehand. On a resume the
    one that did the growing is among them, so excluding them waits forever."""
    fresh = control._place_control_seed(_Panel({"receipt": {}}), BODY)
    resumed = control._place_control_seed(_Panel(ALREADY), BODY)

    assert fresh.excluded_attempts({"attempt-1"}) == {"attempt-1"}
    assert resumed.excluded_attempts({"attempt-1"}) == set()


def test_any_other_conflict_is_still_a_failure() -> None:
    """Only "this already exists" is a resume. A 409 about a drifted binding or
    a rejected policy is the platform refusing, and it must keep refusing."""
    drifted = PanelError("POST", "/api/segmentation/manual-seeds", 409,
                         '{"detail":"persisted control P0 binding drifted"}')
    with pytest.raises(PanelError):
        control._place_control_seed(_Panel(drifted), BODY)


def test_a_refusal_that_is_not_a_conflict_is_raised_unchanged() -> None:
    with pytest.raises(PanelError):
        control._place_control_seed(
            _Panel(PanelError("POST", "/x", 500, "boom")), BODY)


# -- the P0 binding, and what P1 actually carries -------------------------------
#
# This check was written to require `control_p0_artifact_id` on a resumed task,
# on the belief that the persisted control binding stood in for the creation
# response. It does not: the panel attaches that binding only for P4, P5 and P7
# (`control_binding_applicable`), so a P1 seed task never carries one. The unit
# tests passed because the fixtures here supplied the field -- the belief was
# checked against itself, and the deployment refused a resumed control with
# MANUAL_SEED_PROVENANCE_MISSING for want of evidence that phase never produces.
#
# What ties a resumed task to this run is verified either way, from the attempt
# the fleet ran: policy version (which encodes the deployed revision and the
# locks), grid version, source snapshot, seed origin, author and the coordinates.
# The P0 is bound through its snapshot, which is compared already.

def test_a_task_that_carries_no_binding_is_the_ordinary_case() -> None:
    """P1 seed tasks have no control binding. Requiring one refuses every
    resume, which is what it did."""
    assert control._binding_matches(BODY, {}) is True


def test_a_binding_that_disagrees_is_still_refused() -> None:
    """When a task does carry one -- P4, P5, P7 -- it has to be this run's."""
    assert control._binding_matches(BODY, {
        "control_p0_artifact_id": "p0:PHerc0139:something-else",
        "control_p0_artifact_sha256": "a" * 64,
    }) is False


def test_a_binding_that_agrees_passes() -> None:
    assert control._binding_matches(BODY, {
        "control_p0_artifact_id": "p0:PHerc0139:392cd753c0bd",
        "control_p0_artifact_sha256": "a" * 64,
    }) is True


def test_a_half_written_binding_is_refused() -> None:
    """One field without the other is a partial claim, and the platform's own
    `control_job_binding` fails closed on exactly that."""
    assert control._binding_matches(BODY, {
        "control_p0_artifact_id": "p0:PHerc0139:392cd753c0bd",
    }) is False
