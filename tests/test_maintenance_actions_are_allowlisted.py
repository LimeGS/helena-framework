"""`action` selects a subcommand; it never supplies one.

One endpoint dispatches three of the fleet's maintenance commands, so the field
naming them reaches a subprocess argv. The allowlist is the whole boundary: with
it, `action` picks a key from a dict of three; without it, it would be a string
concatenated into a command line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / "panel/app.py").read_text()


def _handler() -> str:
    start = APP.index('@app.post("/api/segmentation/maintenance")')
    return APP[start:][: APP[start:].index("\n@app.", 1)]


def test_the_action_is_looked_up_not_interpolated() -> None:
    handler = _handler()
    assert "MAINTENANCE.get(request.action)" in handler, (
        "the action is not resolved through the allowlist"
    )
    # The refusal comes before anything is built, and before the DSN is read.
    assert handler.index("MAINTENANCE.get") < handler.index("argv = [")
    assert "not a maintenance action" in handler


def test_the_allowlist_only_names_commands_the_fleet_has() -> None:
    block = APP[APP.index("MAINTENANCE = {"):]
    block = block[: block.index("}")]
    actions = set(re.findall(r'"([a-z-]+)":', block))
    assert actions == {"republish", "coverage", "certify"}, actions

    # Run it, do not grep for it. The first version of this test asserted the
    # action's name appeared in cli.py, which it did for "qc" -- a parser with two
    # subcommands, so `qc --db …` exited 2 on argparse and the button was dead
    # from the day it shipped. An audit found it; this would have.
    script = (Path(__file__).resolve().parents[1]
              / "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py")
    for action in sorted(actions):
        result = subprocess.run(
            [sys.executable, str(script), action, "--help"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": str(script.parents[3].parent)})
        assert result.returncode == 0, (
            f"`{action} --help` exits {result.returncode}: the panel offers an "
            f"action the fleet cannot run as invoked.\n"
            f"{(result.stderr or result.stdout)[:300]}"
        )
        # A parser whose own help lists subcommands takes one, and the panel does
        # not pass one -- which is exactly how the dead button looked.
        assert "{" not in result.stdout.split("\n")[0], (
            f"`{action}` takes a subcommand and the panel invokes it bare: "
            f"{result.stdout.splitlines()[0]}"
        )


def test_republish_refuses_rather_than_inventing_a_destination() -> None:
    """It moves surfaces into object storage; a wrong destination strands them."""
    handler = _handler()
    assert 'os.environ.get("ARTIFACT_ROOT")' in handler
    assert "409" in handler
