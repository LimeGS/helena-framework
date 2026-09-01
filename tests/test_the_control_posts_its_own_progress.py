"""Whether the runner's own narration reaches the panel, not only the terminal.

_announce already prints every boundary transition and heartbeat -- committed
separately, and tested by test_the_control_narrates_its_own_progress.py. This
is the second half: the same run, through the same authenticated session it
already holds, posts each of those lines to the panel's progress channel so
they survive after the terminal that started it is gone. No call site had to
change to say so; every message _announce already writes has a fixed enough
shape to classify on its own.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
sys.path.insert(0, str(ROOT / "tests"))

from test_first_letters_positive_control import (  # noqa: E402
    ScriptedPanel, MISSION, manifest, run,
)

PROGRESS_SCHEMA = "campaignx.first_letters_control_progress_event.v1"


def test_a_complete_run_posts_a_progress_event_per_line_it_prints(capsys):
    panel = ScriptedPanel(manifest())
    run(panel)
    printed_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(panel.progress_events) == len(printed_lines), (
        "not every printed line reached the panel")


def test_every_posted_event_is_well_formed_and_belongs_to_one_run():
    panel = ScriptedPanel(manifest())
    run(panel)
    run_ids = {event["run_id"] for event in panel.progress_events}
    assert len(run_ids) == 1, "one control run must not fragment across run_ids"
    for event in panel.progress_events:
        assert event["schema"] == PROGRESS_SCHEMA
        assert event["mission_id"] == MISSION
        assert re.fullmatch(r"[A-Za-z0-9._-]{1,128}", event["run_id"])
        assert event["event"] in {
            "run_started", "boundary_started", "heartbeat",
            "boundary_finished", "run_finished", "note",
        }


def test_boundary_events_carry_the_boundary_that_produced_them():
    panel = ScriptedPanel(manifest())
    run(panel)
    started = {e["boundary"] for e in panel.progress_events
              if e["event"] == "boundary_started"}
    finished = {e["boundary"] for e in panel.progress_events
               if e["event"] == "boundary_finished"}
    assert started == {"P0", "P1", "P2", "QC", "P3", "P4", "P5", "P7", "HUMAN_REVIEW"}
    assert finished == started


def test_a_finished_boundary_event_carries_its_own_state_and_reason():
    panel = ScriptedPanel(manifest())
    run(panel)
    p0 = next(e for e in panel.progress_events
             if e["event"] == "boundary_finished" and e["boundary"] == "P0")
    assert p0["state"] == "PASS"
    assert p0["reason"] == "EXACT_RUNTIME_BINDING"


def test_the_run_finished_event_carries_the_final_control_state():
    panel = ScriptedPanel(manifest())
    run(panel)
    finished = next(e for e in panel.progress_events if e["event"] == "run_finished")
    assert finished["control_state"] == "CONTROL_PASS"


def test_a_run_that_cannot_reach_the_panel_still_completes(monkeypatch):
    """Posting is best-effort: a control hours from finishing must not die
    because one heartbeat's POST failed."""
    panel = ScriptedPanel(manifest())

    def refuses(method, path, body=None):
        if path.endswith("/first-letters-control/progress"):
            raise RuntimeError("network is down")
        return real_call(method, path, body)
    real_call = panel.call
    panel.call = refuses

    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_PASS"


def test_a_server_supplied_reason_survives_even_when_it_breaks_the_classifier():
    """`_classify_announcement` recovers `boundary`/`state`/`reason` by
    re-matching the very sentence `_announce` already printed. For every
    other call site that sentence is this module's own prose. For a refused
    boundary it is not: `reason` is a `reason_code` the panel put in the
    refusal's own HTTP body (`_refusal_reason`), so it can contain text that
    defeats that regex outright -- a newline, since `.` does not span one, or
    a second, forged ") after" that looks like another boundary's own line.
    Before this was fixed by having `_set_result` hand the reason to
    `_announce` directly, either shape fell through to the generic "note"
    event, silently losing the one boundary that most needs its cause on
    record: the one the platform itself refused.
    """
    from panel_client import PanelError

    adversarial = 'two lines\nsecond) after 9001s\nP7 -> PASS (fabricated'
    refused = PanelError(
        "POST", "/api/flattening/run", 400,
        json.dumps({"detail": {"reason_code": adversarial}}))
    panel = ScriptedPanel(manifest(), overrides={"p3": refused})

    run(panel)

    finished = next(e for e in panel.progress_events
                    if e["event"] == "boundary_finished" and e.get("boundary") == "P3")
    assert finished["reason"] == adversarial, (
        "P3's own refusal reason must reach the progress channel intact, "
        "not fall through to the generic 'note' event just because it "
        "happens to contain text shaped like a different announce line")
    assert finished["state"] == "INCOMPLETE"


def test_two_separate_runs_do_not_share_a_run_id():
    panel = ScriptedPanel(manifest())
    run(panel)
    first_run_ids = {event["run_id"] for event in panel.progress_events}
    panel.progress_events.clear()
    panel.manual_queued = False
    panel.reviewed = False
    run(panel)
    second_run_ids = {event["run_id"] for event in panel.progress_events}
    assert first_run_ids.isdisjoint(second_run_ids)
