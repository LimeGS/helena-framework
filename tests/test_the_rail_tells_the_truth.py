"""What the sidebar says a phase is doing, and whether that is true.

The rail is the first thing anybody looks at and the last thing anybody
verifies. Every state it can show is derived in one function, so these tests
drive that function directly rather than the markup.

Four of them cover states the rail could not previously express or expressed
wrongly: queued is not running, cancelled is a state at all, an old failure is
not the current one, and a busy phase does not hide a quiet one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import panel.app as panel  # noqa: E402


def job(state: str, phase: str = "P4") -> dict:
    return {"state": state, "phase": phase, "sample_id": "PHerc0826"}


RUNNABLE = {"id": "P4", "runnable_here": True, "runner": "ink_worker.py"}


def status(jobs: list[dict], **detail) -> str:
    base = {"blocked": None, "artefacts": [], "input_available": True}
    return panel.phase_status(RUNNABLE, {**base, **detail}, panel.phase_work(jobs))


def why(jobs: list[dict], **detail) -> str:
    base = {"blocked": None, "artefacts": [], "input_available": True}
    return panel.phase_status_reason(RUNNABLE, {**base, **detail}, panel.phase_work(jobs))


# --------------------------------------------------------------------------
# Work happening now
# --------------------------------------------------------------------------

def test_a_worker_executing_reads_as_running():
    assert status([job("running")]) == "running"
    assert status([job("leased")]) == "running"


def test_queued_is_not_running():
    """A job nobody has claimed and a job a worker is executing were both
    counted as active, and the rail pulsed for both. A phase routed to an image
    with no runner for it therefore pulsed "running now" indefinitely while
    nothing at all was happening -- which is the exact failure the rail exists
    to surface."""
    assert status([job("pending")]) == "queued"
    assert "no worker has claimed" in why([job("pending")])


def test_running_outranks_queued():
    assert status([job("pending"), job("running")]) == "running"


# --------------------------------------------------------------------------
# How the last attempt ended
# --------------------------------------------------------------------------

def test_a_cancelled_attempt_is_a_state_the_rail_can_show():
    """`cancelled` is terminal in the queue and the rail had no word for it, so
    stopping a job left the phase showing whatever it showed before."""
    assert status([job("cancelled")]) == "stopped"
    assert "cancelled" in why([job("cancelled")])


def test_an_old_failure_is_not_the_current_state():
    """Counting every failure in the window kept a phase red for as long as one
    old failure stayed in it, however many times it had succeeded since. Jobs
    arrive newest first, so the newest terminal row is the answer."""
    assert status([job("succeeded"), job("failed")]) == "done"
    assert status([job("failed"), job("succeeded")]) == "failed"


def test_a_success_with_no_artefact_still_reads_as_done():
    assert status([job("succeeded")], artefacts=[]) == "done"


# --------------------------------------------------------------------------
# Whether it could start at all
# --------------------------------------------------------------------------

def test_prerequisites_met_and_never_run_is_ready():
    assert status([], artefacts=[], input_available=True) == "ready"
    assert "prerequisites are met" in why([], artefacts=[])


def test_prerequisites_not_met_is_blocked_or_waiting():
    """Two causes, one meaning. Blocked names what is missing; waiting means
    nothing upstream has arrived at all."""
    assert status([], blocked="needs a certified surface") == "blocked"
    assert why([], blocked="needs a certified surface") == "needs a certified surface"
    assert status([], input_available=False) == "waiting"
    assert "prerequisites are not met" in why([], input_available=False)


def test_a_phase_with_nothing_to_run_is_not_blocked():
    """A committed catalog and a check inside another phase both have no
    command. Calling either "elsewhere" points at a machine that does not
    exist."""
    committed = {"id": "P0", "runnable_here": False}
    assert panel.phase_status(committed, {"blocked": None, "artefacts": []},
                              panel.phase_work([])) == "no-run"
    remote = {"id": "P6", "runnable_here": False, "runner": "somewhere.py"}
    assert panel.phase_status(remote, {"blocked": None, "artefacts": []},
                              panel.phase_work([])) == "elsewhere"


# --------------------------------------------------------------------------
# The query behind all of it
# --------------------------------------------------------------------------

def test_a_busy_phase_cannot_hide_a_quiet_one():
    """The rail fetched the newest fifty jobs of any phase and then kept the
    ones matching. On any day where one phase produced fifty jobs more
    recently, every other phase saw an empty list and reported itself idle
    while it was running. The filter belongs in SQL."""
    source = (ROOT / "framework/stages/03-ink/fleet/job_store.py").read_text()
    signature = source[source.index("def jobs(self"):source.index("def jobs(self") + 400]
    assert "phase: str | None" in signature
    assert 'conditions.append("phase = %s")' in source

    panel_source = (ROOT / "panel/app.py").read_text()
    helper = panel_source[panel_source.index("def scoped_jobs("):][:650]
    assert "phase=phase" in helper
    assert "sample_id=subject" in helper
    assert "queued = scoped_jobs(phase_id" in panel_source


def test_every_status_the_server_can_return_is_drawn():
    """A state with no entry in the front end renders as undefined, which is a
    blank mark and no colour -- indistinguishable from a phase nobody has
    touched."""
    app_tsx = (ROOT / "panel/web/src/App.tsx").read_text()
    css = (ROOT / "panel/web/src/styles.css").read_text()
    for state in ("running", "queued", "failed", "stopped", "done",
                  "ready", "blocked", "waiting", "elsewhere", "no-run"):
        key = f'"{state}"' if "-" in state else state
        assert f"{key}:" in app_tsx or f"{state}:" in app_tsx, state
        assert f".rail-item.is-{state}" in css, f"{state} has no colour"


def test_only_the_two_live_states_animate():
    """Six things pulsing is a sidebar people stop reading. And every animation
    is switched off under prefers-reduced-motion."""
    css = (ROOT / "panel/web/src/styles.css").read_text()
    rail = css[css.index(".rail-status {"):css.index("/* A unit that belongs")]
    animated = {line.split(".rail-item.is-")[1].split(" ")[0]
                for line in rail.splitlines() if "animation:rail" in line}
    assert animated == {"running", "queued", "failed"}
    assert "prefers-reduced-motion" in rail
    assert "animation:none !important" in rail


def _states_in_the_rail() -> set[str]:
    """The states App.tsx has words for, which is the vocabulary of the rail."""
    app_tsx = (ROOT / "panel/web/src/App.tsx").read_text()
    block = app_tsx[app_tsx.index("const STATUS"):]
    block = block[:block.index("};")]
    return set(re.findall(r'^\s*"?([\w-]+)"?:\s*\{ label:', block, re.MULTILINE))


def test_every_state_has_a_mark_drawn_for_it():
    """The marks are drawn now, not typeset, and this is what that has to hold.

    They were Unicode glyphs from four blocks, at four optical sizes in one
    column: a maths operator arrived twice the height of a black circle and the
    middle dot a fifth of it. This test used to read a font-size off .rail-status
    for that reason. There is no font-size there any more -- each mark is a CSS
    shape in a fixed box -- so the thing worth protecting changed shape too.

    A state with no `.mark-*` rule renders an empty box: present, aligned, and
    saying nothing. That is the failure this catches, and it is the one that
    happens when somebody adds a state to STATUS and stops there.
    """
    css = (ROOT / "panel/web/src/styles.css").read_text()

    box = css[css.index(".mark {"):]
    box = box[:box.index("}")]
    for axis in ("width", "height"):
        size = float(re.search(rf"{axis}:(\d+(?:\.\d+)?)px", box).group(1))
        assert size >= 10, f"the mark box is {size}px {axis}; it was a smudge at ten"

    drawn = set(re.findall(r"\.mark-([\w-]+)::(?:before|after)", css))
    missing = _states_in_the_rail() - drawn
    assert not missing, f"no shape is drawn for {sorted(missing)}"


def test_the_guide_shows_the_same_marks_the_rail_does():
    """The key in the guide is a second copy of the vocabulary. A state shown in
    one place and not the other teaches somebody an incomplete alphabet.

    This assertion went quiet rather than red when the marks stopped being
    glyphs: it compared two regexes over `mark: "\u2026"` and `<dt>x</dt>`, and once
    both stopped matching it was comparing empty set to empty set and passing.
    Hence the length checks -- a vacuous pass is worse than a failure, because
    nobody comes to look.
    """
    # The key was a <dl> in the User guide, which the handbook replaced. It is a
    # table in the handbook's Missions page now, one row per mark, so what this
    # reads is the generated content rather than a route file.
    # The generated module is structured blocks rather than Markdown, so the
    # table's first column arrives as `strong` spans. Intersecting with the rail
    # is what makes the comparison below meaningful in the direction that
    # matters: a state the handbook never names is missing from `keyed` and
    # fails, which is the case somebody is left unable to read a mark.
    handbook = (ROOT / "panel/web/src/routes/handbook-content.ts").read_text()
    bolded = set(re.findall(r'"kind":\s*"strong",\s*"text":\s*"([\w-]+)"', handbook))
    keyed = bolded & _states_in_the_rail()
    rail = _states_in_the_rail()

    assert len(rail) >= 10, f"only found {sorted(rail)} in STATUS"
    assert len(keyed) >= 10, f"only found {sorted(keyed)} in the handbook"
    assert rail == keyed, f"rail {sorted(rail)} vs handbook {sorted(keyed)}"
