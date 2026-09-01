"""A run that dies must not take its evidence with it.

The control crossed five boundaries for the first time -- P0, P1, P2, QC and P3,
the last of them the one that had never passed -- and then the next call was
refused:

    GET /api/geometry/orientation-proof?... -> HTTP 409
    {"detail": "selected P0 contains a partial or tampered control marker"}

Nothing caught it. The process died on the traceback, no receipt was written,
and every boundary it had just crossed went unrecorded. Hours of GPU time
produced a stack trace.

This is the second time. The first was fixed by wrapping five calls in
`except PanelError`, with this comment left beside them:

    A refusal is a boundary result, not an exception. Catching only the
    ambiguous cases let every 4xx escape: the run died, wrote no receipt, and
    the boundaries it had already crossed went unrecorded.

That fix was instance-by-instance, and instance-by-instance is why it was
incomplete: 17 of the 23 `panel.call` sites are still unguarded, and the one
that fired was simply the first that any run had ever reached. Guarding the
eighteenth would leave the nineteenth.

So the guarantee moves to where it cannot be forgotten: whatever escapes, the
receipt is written, the crossed boundaries keep their results, and the boundary
that died is recorded as INCOMPLETE with the reason. A run may fail. It may not
fail silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

import run_first_letters_positive_control as harness  # noqa: E402
from panel_client import PanelError  # noqa: E402


def _rows() -> list[dict]:
    return harness._stage_rows({"profile_id": "p", "profile_locks": []})


def test_the_helper_records_the_boundary_that_died() -> None:
    """The shape the wrapper produces, checked directly."""
    rows = _rows()
    harness._set_result(rows[0], state=harness.PASS, reason="EXACT_RUNTIME_BINDING",
                        started=0.0, clock=lambda: 1.0)

    refusal = PanelError("GET", "/api/geometry/orientation-proof", 409,
                         '{"detail": "selected P0 contains a partial or tampered '
                         'control marker"}')
    harness._record_unexpected_refusal(rows, refusal)

    assert rows[0]["terminal_state"] == "PASS", "a crossed boundary lost its result"
    died = [row for row in rows if row["terminal_state"] == harness.INCOMPLETE]
    assert died, "the boundary that died was not recorded at all"
    assert "409" in str(died[0]) or "UNEXPECTED" in str(died[0].get("reason_code"))


def test_the_receipt_says_what_refused_it_not_only_that_something_did() -> None:
    """Half a fix is a reason code with no reason.

    The wrapper recorded UNEXPECTED_REFUSAL and the run finally left a receipt
    -- and the receipt said nothing about what had happened, because
    `_set_result` keeps only resource_identity, output_hashes and counts from an
    observation and drops everything else. For a designed boundary the reason
    code carries the meaning. For an *unexpected* refusal the detail is the
    whole information, and without it the next one sends somebody back to the
    panel's logs to find out what the receipt was for.

    The refusal that motivated this: 409 "P3 result lacks hash-bound flattened
    lineage", which named the defect precisely and appeared nowhere in the
    receipt."""
    rows = _rows()
    harness._record_unexpected_refusal(rows, PanelError(
        "GET", "/api/geometry/orientation-proof", 409,
        '{"detail": "P3 result lacks hash-bound flattened lineage"}'))

    died = next(row for row in rows
                if row["terminal_state"] == harness.INCOMPLETE)
    recorded = json.dumps(died)
    assert "hash-bound flattened lineage" in recorded, (
        "the receipt records that a boundary died but not what refused it")
    assert "409" in recorded


def test_the_first_unrun_boundary_is_the_one_marked() -> None:
    """It must land on the boundary the run actually reached, not the first row:
    a receipt that blames P0 for a P4 refusal is worse than no receipt."""
    rows = _rows()
    for index in range(5):
        harness._set_result(rows[index], state=harness.PASS, reason="OK",
                            started=0.0, clock=lambda: 1.0)

    harness._record_unexpected_refusal(
        rows, PanelError("GET", "/x", 409, "{}"))

    assert rows[5]["terminal_state"] == harness.INCOMPLETE
    assert rows[4]["terminal_state"] == "PASS"


def test_a_run_that_raises_still_returns_a_matrix(monkeypatch) -> None:
    """End to end: the boundary work explodes, and the caller still gets a
    receipt carrying what was crossed."""
    crossed = {}

    def explode(*_args, **_kwargs):
        raise PanelError("GET", "/api/geometry/orientation-proof", 409,
                         '{"detail": "selected P0 contains a partial or tampered '
                         'control marker"}')

    monkeypatch.setattr(harness, "_run_boundaries", explode, raising=False)

    manifest = {
        "schema": "campaignx.first_letters_control_manifest.v1",
        "profile_id": "first-letters-control-policy@1.2.0",
        "profile_locks": [],
        "source_locks": {},
        "control_cohort": {"control_id": "c", "scroll_id": "PHerc0139"},
        "safety": {"allow_unvalidated": False,
                   "discovery_inputs_are_content_blind": True},
    }

    result = harness.run_positive_control(
        object(), manifest, mission_id="m", deployed_revision="rev",
        submitted_by="tester")

    assert result.get("stages"), "the run returned no boundaries at all"
    assert result.get("control_state"), "the run returned no verdict"
    crossed = [s for s in result["stages"]
               if s.get("terminal_state") == harness.INCOMPLETE]
    assert crossed, "the refusal was not recorded on any boundary"


def test_a_keyboard_interrupt_is_not_swallowed(monkeypatch) -> None:
    """Writing the receipt must not turn an operator's Ctrl-C, or an OOM kill,
    into a quiet 'INCOMPLETE' that reads like a measurement."""

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(harness, "_run_boundaries", interrupt, raising=False)

    manifest = {
        "schema": "campaignx.first_letters_control_manifest.v1",
        "profile_id": "p", "profile_locks": [], "source_locks": {},
        "control_cohort": {"control_id": "c", "scroll_id": "PHerc0139"},
        "safety": {"allow_unvalidated": False,
                   "discovery_inputs_are_content_blind": True},
    }

    with pytest.raises(KeyboardInterrupt):
        harness.run_positive_control(
            object(), manifest, mission_id="m", deployed_revision="rev",
            submitted_by="tester")
