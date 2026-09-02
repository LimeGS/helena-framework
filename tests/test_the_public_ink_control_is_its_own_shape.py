"""A second control, on public inputs, evaluated by the same proven rule.

The nine-boundary control cannot answer what the Vesuvius reviewers asked
for. It stops at P1 -- FROZEN_ROOT_OBJECT_EVIDENCE_MISSING, seven boundaries
never reached -- and even repaired it would still read surfaces out of a
private bucket, so "public input surfaces" and "a run from a clean
installation" stay out of reach by construction.

The post's own recommended tooling can answer it: the surface volume is in the
open-data bucket, the checkpoint is a public non-gated HuggingFace repo whose
digest this platform verified byte for byte, and nothing in the chain needs a
credential.

So this is a second control rather than a repair of the first. What it shares
with the first is the evaluation rule -- first non-pass owns the outcome, rows
after it are normalised to prerequisite-not-reached -- because that rule is
proven and re-derived by the panel before anything trusts a receipt.

What it must NOT share is the boundary list. That list being a fixed constant
is a safety property: a receipt cannot declare its own shape and pass
trivially. Two controls therefore mean two schemas, each pinned to its own
boundaries, and neither convertible into the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

from run_first_letters_positive_control import (  # noqa: E402
    BOUNDARIES, SCHEMA, evaluate_survival_matrix,
)

PUBLIC_SCHEMA = "campaignx.public_ink_stage_survival.v1"
PUBLIC_BOUNDARIES = (
    "PUBLIC_SOURCE", "SCALE", "CHECKPOINT", "INK", "LIVENESS", "HUMAN_REVIEW",
)


def rows(boundaries, state="PASS"):
    return [{"boundary": b, "terminal_state": state, "reason_code": "OK",
             "elapsed_seconds": 0.0, "resource_identity": {},
             "input_artifacts": [], "output_hashes": {}, "counts": {}}
            for b in boundaries]


# -- the shared rule -------------------------------------------------------

def test_the_public_control_is_evaluated_by_the_same_rule():
    receipt = evaluate_survival_matrix(
        {"schema": PUBLIC_SCHEMA, "stages": rows(PUBLIC_BOUNDARIES)})
    assert receipt["control_state"] == "CONTROL_PASS"
    assert receipt["content_sha256"]


def test_the_first_non_pass_still_owns_the_outcome():
    stages = rows(PUBLIC_BOUNDARIES)
    stages[3]["terminal_state"] = "INCOMPLETE"
    receipt = evaluate_survival_matrix({"schema": PUBLIC_SCHEMA, "stages": stages})
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "INK"
    # and nothing after it can resurrect the run
    assert [r["terminal_state"] for r in receipt["stages"][4:]] == [
        "NOT_RUN_PREREQUISITE"] * 2


# -- the shapes cannot be confused ----------------------------------------

def test_a_public_receipt_cannot_be_read_as_a_first_letters_one():
    """The safety property. If one schema accepted the other's boundaries, a
    six-row receipt could be published as a nine-boundary control PASS."""
    with pytest.raises(ValueError, match="boundary"):
        evaluate_survival_matrix({"schema": SCHEMA, "stages": rows(PUBLIC_BOUNDARIES)})


def test_a_first_letters_receipt_cannot_be_read_as_a_public_one():
    with pytest.raises(ValueError, match="boundary"):
        evaluate_survival_matrix({"schema": PUBLIC_SCHEMA, "stages": rows(BOUNDARIES)})


def test_a_receipt_with_no_schema_is_refused():
    """Dispatch is on the declared schema, so an absent one has no boundary
    list to check against and must not fall back to either."""
    with pytest.raises(ValueError, match="schema"):
        evaluate_survival_matrix({"stages": rows(PUBLIC_BOUNDARIES)})


def test_an_unknown_schema_is_refused_by_name():
    with pytest.raises(ValueError, match="not-a-control"):
        evaluate_survival_matrix({"schema": "not-a-control", "stages": rows(BOUNDARIES)})


def test_the_nine_boundary_control_still_evaluates_exactly_as_before():
    """Every existing receipt and the panel's own re-derivation depend on this."""
    receipt = evaluate_survival_matrix({"schema": SCHEMA, "stages": rows(BOUNDARIES)})
    assert receipt["control_state"] == "CONTROL_PASS"
    assert [r["boundary"] for r in receipt["stages"]] == list(BOUNDARIES)


# -- the segmentation control is a third shape, and not the other two ---------

SEGMENTATION_SCHEMA = "campaignx.public_segmentation_stage_survival.v1"
SEGMENTATION_BOUNDARIES = (
    "PUBLIC_SOURCE", "INTAKE", "GROW", "GEOMETRY", "PHYSICAL_QC", "FLATTEN",
)


def test_the_segmentation_control_is_evaluated_by_the_same_rule():
    receipt = evaluate_survival_matrix(
        {"schema": SEGMENTATION_SCHEMA, "stages": rows(SEGMENTATION_BOUNDARIES)})
    assert receipt["control_state"] == "CONTROL_PASS"
    assert receipt["first_nonpassing_boundary"] is None


def test_a_grow_that_produced_nothing_owns_the_segmentation_outcome():
    stages = rows(SEGMENTATION_BOUNDARIES)
    stages[2]["terminal_state"] = "INCOMPLETE"
    stages[2]["reason_code"] = "NO_SURFACE_WITHIN_BUDGET"
    receipt = evaluate_survival_matrix(
        {"schema": SEGMENTATION_SCHEMA, "stages": stages})
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "GROW"
    assert [r["terminal_state"] for r in receipt["stages"][3:]] == [
        "NOT_RUN_PREREQUISITE"] * 3


def test_a_segmentation_receipt_cannot_be_read_as_an_ink_one():
    with pytest.raises(ValueError):
        evaluate_survival_matrix(
            {"schema": PUBLIC_SCHEMA, "stages": rows(SEGMENTATION_BOUNDARIES)})
    with pytest.raises(ValueError):
        evaluate_survival_matrix(
            {"schema": SEGMENTATION_SCHEMA, "stages": rows(PUBLIC_BOUNDARIES)})
