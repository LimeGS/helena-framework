"""Credentials a worker needs, held where the workers already look.

They lived in a file on one host's tmpfs. Lost on every reboot, absent on every
other machine, placed by hand each time -- so surface QC and the ink worker both
refuse to start after a restart until somebody remembers. A worker is ephemeral
and has to be able to start from nothing but a database URL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.cli import adopt_fleet_secrets  # noqa: E402


class Store:
    def __init__(self, held=None, fails=False):
        self.held, self.fails = held or {}, fails

    def secrets(self):
        if self.fails:
            raise RuntimeError("relation \"fleet_secrets\" does not exist")
        return dict(self.held)


def test_a_worker_picks_up_what_the_control_plane_holds(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    adopted = adopt_fleet_secrets(Store({"AWS_ACCESS_KEY_ID": "AKIAEXAMPLE"}))
    assert adopted == ["AWS_ACCESS_KEY_ID"]
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAEXAMPLE"


def test_an_environment_already_set_wins(monkeypatch):
    """An operator who exported a key for one run means it. A stored value
    overriding it silently is a debugging session about which one is in use."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "FROM-THE-OPERATOR")
    assert adopt_fleet_secrets(Store({"AWS_ACCESS_KEY_ID": "FROM-THE-TABLE"})) == []
    assert os.environ["AWS_ACCESS_KEY_ID"] == "FROM-THE-OPERATOR"


def test_an_older_control_plane_does_not_stop_the_worker(capsys):
    """Segmentation does not need object storage to run. A missing table must
    not be the reason a host cannot claim work."""
    assert adopt_fleet_secrets(Store(fails=True)) == []
    assert "unavailable" in capsys.readouterr().err


def test_only_vetted_names_can_be_stored():
    """This is read straight into a process environment, so a name nobody
    vetted is a way to set PATH or LD_PRELOAD from a web form."""
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.postgres_store import PostgresFleetStore

    assert "AWS_SECRET_ACCESS_KEY" in PostgresFleetStore.FLEET_SECRET_NAMES
    for forbidden in ("PATH", "LD_PRELOAD", "PYTHONPATH", "CX_DB"):
        assert forbidden not in PostgresFleetStore.FLEET_SECRET_NAMES
