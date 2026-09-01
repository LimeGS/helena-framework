"""A narration failure must cost a line, never a boundary's own verdict.

The independent review that follows every major task here found the reverse
was true: `_set_result` calls `_announce` as its own last, unguarded
statement, so a print or progress-posting failure on a boundary that had
already PASSED propagated into `run_positive_control`'s outer refusal
handler -- which then blamed the *next*, never-attempted boundary, exactly
the misattribution `_record_unexpected_refusal`'s own docstring says must
never happen. The same gap let a failure on the final "run finished" line
lose an already-computed, fully passing receipt outright.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
sys.path.insert(0, str(ROOT / "tests"))

import run_first_letters_positive_control as control  # noqa: E402
from test_first_letters_positive_control import (  # noqa: E402
    ScriptedPanel, manifest, run,
)


def test_announce_never_raises_even_when_printing_fails(monkeypatch, capsys):
    def broken_print(*args, **kwargs):
        raise BrokenPipeError("stdout is gone")

    monkeypatch.setattr(control, "print", broken_print, raising=False)
    control._announce("this must not raise")  # no reporter attached either


def test_announce_never_raises_when_classification_itself_is_broken(monkeypatch):
    def broken_classifier(_message):
        raise ValueError("boom")

    monkeypatch.setattr(control, "_classify_announcement", broken_classifier)
    reporter = control._ProgressReporter(panel=None, mission_id="m", run_id="r")
    token = control._REPORTER.set(reporter)
    try:
        control._announce("P0 -> PASS (X) after 1s")  # must not raise
    finally:
        control._REPORTER.reset(token)


def test_a_passing_boundary_stays_passing_even_when_its_own_announce_fails(monkeypatch):
    """The bug, reproduced directly: P4's row is correct; only the print after
    it is broken. Before the fix this blamed P5 -- untouched -- instead."""
    panel = ScriptedPanel(manifest())

    def flaky_print(*args, **kwargs):
        text = args[0] if args else ""
        if "P4 -> PASS" in str(text):
            raise BrokenPipeError("stdout is gone")
        return builtins.print(*args, **kwargs)

    monkeypatch.setattr(control, "print", flaky_print, raising=False)
    receipt = run(panel)

    by_boundary = {row["boundary"]: row for row in receipt["stages"]}
    assert by_boundary["P4"]["terminal_state"] == "PASS", (
        "P4 genuinely passed; a broken print on its own announce line must "
        "not overwrite its correct verdict")
    assert by_boundary["P5"]["terminal_state"] == "PASS", (
        "P5 was never at fault -- the failure was P4's print, not P5's own "
        "work -- so P5 must reach and report its own real verdict rather "
        "than being blamed for a narration glitch that happened before it")
    assert receipt["control_state"] == "CONTROL_PASS", (
        "the run actually passed end to end; a print failure on one "
        "announce line must not turn a passing control into a failing one")


def test_a_broken_final_announce_does_not_lose_a_passing_receipt(monkeypatch):
    panel = ScriptedPanel(manifest())

    def flaky_print(*args, **kwargs):
        text = args[0] if args else ""
        if "control run finished" in str(text):
            raise BrokenPipeError("stdout is gone")
        return builtins.print(*args, **kwargs)

    monkeypatch.setattr(control, "print", flaky_print, raising=False)
    receipt = run(panel)  # must return, not raise
    assert receipt["control_state"] == "CONTROL_PASS"


def test_a_broken_reporter_post_does_not_touch_the_receipt(monkeypatch):
    """The network side already had its own try/except; this confirms the
    outer guard in _announce does not change that behavior, only widens it."""
    panel = ScriptedPanel(manifest())

    def broken_post(self, message, **fields):
        raise RuntimeError("network is down")

    monkeypatch.setattr(control._ProgressReporter, "post", broken_post)
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_PASS"


def _panel_with_a_recording_opener():
    """A real Panel whose HTTP layer records the timeout it was asked for,
    without making a network call."""
    from panel_client import Panel

    panel = Panel("https://panel.example", insecure=True)
    calls = []

    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
        def read(self):
            return b"{}"

    def recording_open(request, timeout=None):
        calls.append(timeout)
        return _Response()

    panel.http.open = recording_open
    return panel, calls


def test_call_accepts_a_per_call_timeout_override():
    panel, calls = _panel_with_a_recording_opener()
    assert panel.timeout == 3600  # the harness's own default, unrelated to this call
    panel.call("GET", "/api/session", timeout=7)
    assert calls == [7], "an explicit timeout must reach the HTTP layer as given"


def test_call_falls_back_to_the_panel_s_own_timeout_when_none_is_given():
    panel, calls = _panel_with_a_recording_opener()
    panel.call("GET", "/api/session")
    assert calls == [panel.timeout], "unchanged default behavior for every other caller"


def test_a_heartbeat_post_uses_a_short_timeout_distinct_from_the_wait_it_reports_on():
    """The bug: a heartbeat fired from inside a 60-minute wait_until poll used
    the harness's full 3600s socket timeout -- so one slow response could by
    itself consume the whole boundary's own timeout budget."""
    panel, calls = _panel_with_a_recording_opener()
    reporter = control._ProgressReporter(panel=panel, mission_id="m", run_id="r")
    reporter.post("P1 still waiting for the grow to finish (900s)", event="heartbeat")
    assert calls, "the heartbeat did not reach the panel at all"
    assert calls[0] < 60, (
        f"a heartbeat POST used timeout={calls[0]}s -- indistinguishable from "
        "the boundary's own multi-hour wait budget, so a merely slow (not "
        "failing) panel response can silently consume that entire budget")
