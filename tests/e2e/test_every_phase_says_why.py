"""Every phase reports a state, and every state is justified by the data.

The existing end-to-end suite exercises P0, P1, P3, P4 and P5, which are the
phases that have produced something. The other five had no coverage at all, and
the reason is real: you cannot write a production end-to-end test for a phase
that has never run. P7 and P9 are blocked waiting on inputs, P8 has never been
asked, and P6 is not a job at all.

So this asserts the property those five can actually be held to: a phase's state
has to be justified. `blocked` must name what is missing, `done` must have
artefacts to point at, and a `no-run` phase must not also advertise a button.

The claim that a phase is ready is checked against evidence rather than against
its own sentence. Each phase's detail carries `input_available`, computed from
whether the upstream artefacts exist, and the two have to agree: `ready` and
`done` require it true, `blocked` requires it false. A phase that flips to ready
while its inputs are absent is caught there, which the "its prerequisites are
met" text alone could never do.

Run against a deployment, like the rest of tests/e2e:

    HELENA_E2E_PANEL=https://host:8800 HELENA_E2E_USER=name \\
    HELENA_PANEL_TLS_INSECURE=1 HELENA_PANEL_PASSWORD=… \\
    python3 -m pytest tests/e2e/test_every_phase_says_why.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/harness"))

PANEL = os.environ.get("HELENA_E2E_PANEL")
USER = os.environ.get("HELENA_E2E_USER")
PASSWORD = os.environ.get("HELENA_PANEL_PASSWORD")
MISSION = os.environ.get("HELENA_E2E_MISSION", "test")
SCROLL = os.environ.get("HELENA_E2E_SCROLL", "PHerc0826")

pytestmark = pytest.mark.skipif(
    not (PANEL and USER and PASSWORD),
    reason="set HELENA_E2E_PANEL, HELENA_E2E_USER and HELENA_PANEL_PASSWORD")

# The vocabulary the rail draws and the guide documents. A status outside it is a
# status nobody has a mark for, which means the page renders a blank.
KNOWN = {"running", "queued", "failed", "stopped", "blocked",
         "ready", "done", "waiting", "elsewhere", "no-run"}

# States that are a claim about the world rather than about work in flight, and
# therefore have to be explained.
MUST_EXPLAIN = {"blocked", "waiting", "no-run", "elsewhere", "ready", "done"}


@pytest.fixture(scope="module")
def phases():
    from panel_client import Panel

    client = Panel(PANEL)
    assert client.sign_in(USER, PASSWORD) == USER, "the session did not stick"
    summary = client.call(
        "GET", f"/api/phase-summary?mission={MISSION}&subject={SCROLL}")
    found = summary["phases"]
    assert len(found) == 10, f"expected ten phases, got {len(found)}"
    return {phase["id"]: phase for phase in found}


def test_every_phase_reports_a_status_the_page_can_draw(phases):
    for phase_id, phase in phases.items():
        assert phase["status"] in KNOWN, (
            f"{phase_id} reports {phase['status']!r}, which the rail has no mark "
            "for and will render as an empty box"
        )


@pytest.mark.parametrize("phase_id", [f"P{n}" for n in range(10)])
def test_the_state_is_explained(phases, phase_id):
    """A state without a reason is a state nobody can act on.

    This is the whole coverage story for P2, P6, P7, P8 and P9: none of them can
    be exercised end to end today, and all of them can be held to explaining
    themselves.
    """
    phase = phases[phase_id]
    if phase["status"] not in MUST_EXPLAIN:
        pytest.skip(f"{phase_id} is {phase['status']}, which is work in flight")
    why = (phase.get("why") or "").strip()
    assert why, f"{phase_id} is {phase['status']} and says nothing about why"
    assert len(why) > 15, f"{phase_id} explains itself with {why!r}"


def test_a_blocked_phase_names_what_it_is_waiting_for(phases):
    """Blocked is a claim about a missing input, so it has to name the input.

    P7 needs a probability map and a bounding box; P9 needs published ink maps.
    A blocked phase that cannot say what it lacks is indistinguishable from one
    that is broken.
    """
    blocked = {pid: p for pid, p in phases.items()
               if p["status"] in {"blocked", "waiting"}}
    assert blocked, (
        "no phase is blocked, which on a pipeline that has not run end to end "
        "means a gate stopped asking"
    )
    for phase_id, phase in blocked.items():
        why = (phase.get("why") or "").lower()
        assert any(word in why for word in
                   ("need", "requires", "no ", "missing", "before", "without")), (
            f"{phase_id} is blocked and its reason names no missing input: "
            f"{phase.get('why')!r}"
        )


@pytest.mark.parametrize("phase_id", [f"P{n}" for n in range(10)])
def test_the_status_agrees_with_whether_the_inputs_exist(phase_id):
    """The gap this file was written with, and the reason it is worth having.

    A status is a sentence the summary writes. `input_available` is a fact the
    detail computes from whether the upstream artefacts are there. When they
    disagree, one of them is lying, and the interesting direction is a phase
    calling itself ready with nothing to read: that is a gate that stopped
    asking, and it is invisible on the phase's own page.
    """
    from panel_client import Panel

    client = Panel(PANEL)
    client.sign_in(USER, PASSWORD)
    summary = client.call(
        "GET", f"/api/phase-summary?mission={MISSION}&subject={SCROLL}")
    status = {p["id"]: p["status"] for p in summary["phases"]}[phase_id]
    detail = client.call(
        "GET", f"/api/phase/{phase_id}?mission={MISSION}&subject={SCROLL}")
    available = detail.get("input_available")

    if status in {"ready", "done"}:
        assert available is True, (
            f"{phase_id} reports {status} and its inputs are not available. "
            "Either the gate opened without them or the summary is guessing."
        )
    if status == "blocked":
        assert available is False, (
            f"{phase_id} says blocked while its inputs are available, so it is "
            "refusing work it could do"
        )


def test_a_finished_phase_has_something_to_show(phases):
    """`done` is a claim that artefacts exist. It has to be able to point at them."""
    from panel_client import Panel

    client = Panel(PANEL)
    client.sign_in(USER, PASSWORD)
    for phase_id, phase in phases.items():
        if phase["status"] != "done":
            continue
        detail = client.call(
            "GET", f"/api/phase/{phase_id}?mission={MISSION}&subject={SCROLL}")
        artefacts = detail.get("artefacts") or detail.get("artifacts") or []
        rows = detail.get("state") or {}
        assert artefacts or rows, (
            f"{phase_id} says done and its page carries neither artefacts nor "
            "state rows"
        )


def test_a_phase_that_cannot_be_queued_says_so_rather_than_offering_a_button(phases):
    """P0 and P6 are not jobs, and the difference has to be visible.

    P0 is a frozen decision committed to the repository; P6 runs inside P5. Both
    report `no-run`, and a `no-run` phase that also advertises itself as
    queueable would give an operator a button that cannot do anything.
    """
    for phase_id, phase in phases.items():
        if phase["status"] != "no-run":
            continue
        assert not phase.get("queueable"), (
            f"{phase_id} reports no-run and queueable at once, so the page shows "
            "a control with nothing behind it"
        )
