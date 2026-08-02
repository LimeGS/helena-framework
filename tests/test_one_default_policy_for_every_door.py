""""Default P1" is a scientific policy, not a property of the entry point.

An audit found four answers to one question. The stage contract's
active_runtime_profile named the cost-aware v2 planner. The panel's API defaulted
to `deterministic` -- the history-blind v1 planner, the one that will re-pick a
recipe that already failed on this cell. The browser took whichever seeder came
first in a Python list. The CLI deferred to whatever the worker was started with.

So a run queued from the API and the same button in the browser used different
policies, and neither matched the contract. That is not a missing feature; it
means "we ran P1 with its default" does not identify what ran.

The declaration lives in one place now -- the seeder marked default in the
queue's own list -- and this checks that the contract, the API and the form all
read it rather than each carrying an opinion.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()
LAUNCHER = (ROOT / "panel/web/src/routes/Segmentation.tsx").read_text()
STAGE = json.loads((ROOT / "framework/stages/01-segmentation/stage.json").read_text())


def _declared_default() -> str:
    """The one seeder the queue marks as the fleet's default."""
    flagged = re.findall(r'\{"id": "([\w-]+)",[^}]*?"default": True', APP, re.DOTALL)
    assert len(flagged) == 1, f"exactly one seeder is the default, found {flagged}"
    return flagged[0]


def test_the_declared_default_is_the_one_the_stage_contract_runs() -> None:
    """The contract names a runtime profile; the default has to be that planner.

    If the stage moves to another profile this fails, which is the point: the
    two are the same decision written in two files, and they drifted before.
    """
    profile = STAGE["planner_contract"]["active_runtime_profile"]
    default = _declared_default()
    assert default in profile, (
        f"the queue's default seeder is {default!r} and the stage runs "
        f"{profile!r}; one of them is wrong"
    )


def test_the_api_does_not_carry_its_own_default() -> None:
    """It said "deterministic", which is the history-blind v1 planner."""
    assert 'planner: str = Field("deterministic")' not in APP, (
        "the API is back to defaulting to the history-blind planner"
    )
    assert "default_factory=lambda: DEFAULT_SEEDER" in APP


def test_the_form_reads_the_declared_default_not_the_first_row() -> None:
    """`planners[0]` made the form's default drift with list order."""
    assert "planners.find((p) => p.default)" in LAUNCHER
    assert 'useState(planners[0]?.id ?? "deterministic-v2")' not in LAUNCHER


def test_a_v2_policy_with_a_v1_seeder_is_refused_before_it_is_queued() -> None:
    """The worker already refuses it; the form should not build it.

    The generator derives planner_contract_version from the policy, so
    adaptive-geometry-history-v2 makes a v2 task, and a v1 planner cannot read a
    v2 packet. Queueing it means the operator finds out one lease later, from a
    failed attempt instead of from the control that offered the combination.
    """
    assert '"adaptive-geometry-history-v2"' in APP
    assert 'request.planner.endswith("-v2")' in APP, (
        "nothing stops a v1 seeder being queued against the v2 policy"
    )
    # And the refusal the worker makes is still the one being anticipated here.
    planner = (ROOT / "framework/stages/01-segmentation/fleet/planner.py").read_text()
    assert "planner v2 task has an unsupported candidate selection policy" in planner
