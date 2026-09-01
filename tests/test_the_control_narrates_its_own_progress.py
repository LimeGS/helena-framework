"""Whether the control says what it is doing as it happens, not only at the end.

A boundary can run over an hour -- P1's discovery-and-grow measured at 66
minutes, P2's finalization at 60, in the run this was written against. The
runner printed nothing until it returned, so a terminal watching it for that
long had no way to tell "slow" from "hung": the only evidence was whether the
process was still alive. This is the first of two things that fixes -- stdout
today, a channel the panel and its UI can read back later.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
sys.path.insert(0, str(ROOT / "tests"))

from test_first_letters_positive_control import (  # noqa: E402
    ScriptedPanel, manifest, run,
)

BOUNDARIES_IN_ORDER = ("P0", "P1", "P2", "QC", "P3", "P4", "P5", "P7", "HUMAN_REVIEW")


def test_every_boundary_announces_starting_and_finishing_in_order(capsys):
    panel = ScriptedPanel(manifest())
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_PASS"  # the fixture this narrates
    out = capsys.readouterr().out

    positions = []
    for boundary in BOUNDARIES_IN_ORDER:
        start = re.search(rf"\b{boundary}\b.*start", out)
        finish = re.search(rf"\b{boundary}\b.*(PASS|FAILED|INCOMPLETE)", out)
        assert start, f"no 'starting' line printed for {boundary}"
        assert finish, f"no finishing line printed for {boundary}"
        assert start.start() < finish.start(), (
            f"{boundary} finished before it started, in the output")
        positions.append(finish.start())
    assert positions == sorted(positions), (
        "boundaries did not finish in their declared order")


def test_a_long_wait_is_not_silent(capsys):
    """The waits this exists for: P1's grow, P2's physical-QC readback, a job,
    human review. QC itself is a synchronous read of what P2 already fetched --
    it never polls, so it earns no heartbeat of its own.

    ScriptedPanel resolves every wait on its first poll, standing in for "the
    predicate became true eventually" rather than "instantly" -- so a run
    through it still ticks once per wait, which is where a heartbeat has to
    come from.
    """
    panel = ScriptedPanel(manifest())
    run(panel)
    out = capsys.readouterr().out
    assert re.search(r"P1.*wait", out, re.I), "no heartbeat while P1 waited"
    assert re.search(r"P2.*wait", out, re.I), \
        "no heartbeat while P2 waited on a terminal QC readback"
    assert re.search(r"P3.*wait", out, re.I), "no heartbeat while a job was polled"
    assert re.search(r"HUMAN_REVIEW.*wait", out, re.I), \
        "no heartbeat while human review waited"
