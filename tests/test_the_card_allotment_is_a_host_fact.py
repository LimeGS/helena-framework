"""How many cards this host gives Helena is a property of the host.

`surface-qc.compose.yaml` already treats the card as a per-host fact -- "which
card is a per-host fact and lives in the env file, not here" -- and everything
that differs between instances is derived from `HELENA_QC_DEVICE`. The number of
instances was not: `deploy-platform.sh` defaulted to `0 1` in the script itself,
so a host that should leave a card free had no way to say so except by editing
a file shared with every other host, and an eight-card rig would have had to
edit it too.

The second half matters more than it looks. The deploy loop only ever brought
stacks *up*. Shrinking the list left the instance on the dropped card running
from the previous deploy -- unmanaged, still claiming work, still holding the
card the operator asked to free -- and the deploy would report success. Saying
"use one card" has to mean the other card ends up free, not merely unmentioned.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1]
          / "containers/deploy-platform.sh")
TEXT = SCRIPT.read_text(encoding="utf-8")


def resolution() -> str:
    """The lines that decide how many QC instances this host runs."""
    start = TEXT.index("devices=")
    return TEXT[max(0, start - 900):start + 400]


def test_the_device_list_can_come_from_the_host() -> None:
    """Set in the environment it still wins -- CI passes it that way -- but a
    host that says nothing must be able to answer from its own env file rather
    than from a default shared with every other machine."""
    block = resolution()
    assert "HELENA_QC_DEVICES" in block
    assert "surface-qc.env" in block, (
        "the device list is never read from the host's env file, so the only "
        "way to change it is editing a file every host shares")


def test_a_host_that_names_one_card_gets_one_instance() -> None:
    devices = next(l for l in TEXT.splitlines() if l.strip().startswith("devices="))
    assert "HELENA_QC_DEVICES" in devices, devices


def test_dropping_a_card_stops_what_was_running_on_it() -> None:
    """A list that shrinks has to take the old instance down with it, or the
    freed card is still occupied by a container nothing is managing."""
    assert "down" in TEXT, "the deploy never brings a QC stack down"
    stanza = TEXT[TEXT.index("for device in $devices"):]
    preceding = TEXT[:TEXT.index("for device in $devices")]
    assert "helena-qc-" in preceding or "helena-qc-" in stanza[:2000]
    assert re.search(r"compose\s+-p\s+\"?helena-qc-\$?\{?\w+", TEXT), (
        "nothing addresses a QC project by name to stop it")


def test_the_teardown_only_touches_cards_no_longer_asked_for() -> None:
    """Bringing down a card that is still in the list would restart QC on every
    deploy for no reason, and interrupt whatever it was measuring."""
    assert "still in the list" in TEXT or "not in the list" in TEXT or (
        "no longer" in TEXT), (
        "the teardown does not say which projects it spares")
