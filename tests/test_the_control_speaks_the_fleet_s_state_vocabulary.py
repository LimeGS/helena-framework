"""The control has to wait for states the fleet actually reaches.

P1 waited for an attempt to reach `SUCCEEDED` and treated anything else as
unfinished. The fleet has no such state. A grow that works ends at `QC_PENDING`
-- which is terminal, and which QC never changes, because finalizing QC updates
the *surface* (`QC_SCREENED`, `CT_SUPPORTED`) and leaves the attempt alone. So
P1 could not pass: not slowly, not after QC, not ever. Three surfaces grew to
1.58 cm² on the deployment and the control reported GROW_NONTERMINAL_TIMEOUT.

The same shape sat in the QC boundary, where the readback treated a job as
finished on `{COMPLETED, FAILED, CANCELLED}`. `CANCELLED` is not a state any QC
job reaches, and `BLOCKED_CONFIGURATION` -- which is -- was missing, so a worker
that could not be configured would have been waited on until the timeout instead
of reported.

The runner is deliberately stdlib-only: it speaks HTTP to a panel and must stay
importable from a bare checkout, so it cannot import the fleet's constants. The
coupling belongs here instead. These tests read the vocabulary out of the store
and fail when the runner's copy drifts -- naming what to change, rather than
letting the drift be discovered by a control that hangs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

import run_first_letters_positive_control as control  # noqa: E402

STORE = (ROOT / "framework/stages/01-segmentation/fleet/store.py").read_text()


def _fleet_terminal_states() -> set[str]:
    body = re.search(r"^TERMINAL_STATES\s*=\s*\((.*?)\)", STORE, re.S | re.M)
    assert body, "TERMINAL_STATES is not where this test looks for it"
    return set(re.findall(r'"([A-Z_]+)"', body.group(1)))


def _fleet_qc_job_states() -> set[str]:
    """Every state the store ever writes onto a QC job."""
    return set(re.findall(r"qc_jobs SET state='([A-Z_]+)'", STORE))


# -- the vocabulary ------------------------------------------------------------

def test_the_scan_finds_the_vocabulary_it_is_meant_to_check() -> None:
    """A check that matches nothing passes forever."""
    assert len(_fleet_terminal_states()) > 10
    assert {"PENDING", "CLAIMED", "COMPLETED"} <= _fleet_qc_job_states()


def test_the_control_waits_for_every_state_an_attempt_can_end_in() -> None:
    """Missing one is not a wrong answer, it is no answer: the wait runs out."""
    missing = _fleet_terminal_states() - set(control.ATTEMPT_TERMINAL_STATES)
    assert not missing, (
        f"an attempt can end in {sorted(missing)} and the control would wait out "
        "its timeout instead of reading the result"
    )


def test_the_control_invents_no_attempt_state() -> None:
    """`SUCCEEDED`, `FAILED` and `ARTIFACT_INVALID` were waited for by name and
    none of them is a state the fleet ever sets."""
    invented = set(control.ATTEMPT_TERMINAL_STATES) - _fleet_terminal_states()
    assert not invented, f"the control waits for states that do not exist: {sorted(invented)}"


def test_a_finished_grow_is_the_state_the_fleet_finalizes_one_with() -> None:
    assert control.GROWN_STATE == "QC_PENDING"
    assert control.GROWN_STATE in _fleet_terminal_states(), (
        "the state that means the grow worked is not one the fleet can reach"
    )


def test_the_qc_readback_knows_every_way_a_job_can_stop() -> None:
    running = {"PENDING", "CLAIMED"}
    terminal = _fleet_qc_job_states() - running
    missing = terminal - set(control.QC_JOB_TERMINAL_STATES)
    assert not missing, (
        f"a QC job can stop at {sorted(missing)} and the readback would keep waiting"
    )
    invented = set(control.QC_JOB_TERMINAL_STATES) - _fleet_qc_job_states()
    assert not invented, f"the readback waits for {sorted(invented)}, which no job reaches"


# -- the other boundaries' vocabularies ----------------------------------------
#
# These were audited and are right; they are pinned so they stay right. The
# control crosses three different stages and each has its own words -- the
# segmentation fleet has no `succeeded`, the ink stage has nothing else -- so
# "which vocabulary is this?" is a question every comparison has to answer, and
# getting it wrong reads as a stuck pipeline rather than as a typo.

RUNNER = (ROOT / "scripts/harness/run_first_letters_positive_control.py").read_text()


def test_p5_speaks_the_ink_stage_s_words_not_the_fleet_s() -> None:
    """Lowercase, and a different set: the ink stage really does have
    `succeeded`, which the segmentation fleet never writes."""
    ink = (ROOT / "framework/stages/03-ink/fleet/job_store.py").read_text()
    assert '"succeeded"' in ink or "'succeeded'" in ink
    assert 'p5.get("job_state") != "succeeded"' in RUNNER


def test_p5_liveness_asks_for_a_verdict_the_adapter_can_give() -> None:
    adapter = (ROOT / "framework/stages/04-validation/scripts/"
               "campaignx_surface_qc_adapter.py").read_text()
    verdicts = set(re.findall(r'verdict not in \{([^}]*)\}', adapter))
    assert verdicts, "the adapter's verdict set is not where this test looks"
    assert "ALIVE" in " ".join(verdicts)
    assert '"verdict") != "ALIVE"' in RUNNER


def test_the_review_the_control_asks_for_is_one_the_panel_accepts() -> None:
    panel = (ROOT / "panel/app.py").read_text()
    accepted = re.search(r"HUMAN_REVIEW_VERDICTS\s*=\s*\(([^)]*)\)", panel)
    assert accepted, "HUMAN_REVIEW_VERDICTS is not where this test looks for it"
    assert "INSPECT" in accepted.group(1)
    assert '"verdict": "INSPECT"' in RUNNER


def test_p2_matches_every_geometry_rejection_rather_than_one_name() -> None:
    """There is no `GEOMETRY_REJECTED`; there are four of them. The prefix match
    is what makes that safe, and replacing it with an equality would pass every
    test written against a fixture and fail on the deployment."""
    store_states = set(re.findall(r'"(GEOMETRY_REJECTED_[A-Z_]+)"', STORE))
    assert len(store_states) >= 3, "the geometry rejections moved"
    assert 'startswith("GEOMETRY_REJECTED")' in RUNNER
    assert '== "GEOMETRY_REJECTED"' not in RUNNER


# -- what the boundary decides -------------------------------------------------

def _run(state: str, **overrides) -> dict:
    run = {"state": state, "attempt_id": "a-1", "surface_id": "s-1",
           "artifact_sha256": "a" * 64, "area_cm2": 1.58}
    run.update(overrides)
    return run


def test_a_grown_surface_finishes_the_wait_and_counts_as_grown() -> None:
    run = _run("QC_PENDING")
    assert control._grow_finished(run) is True
    assert control._grow_succeeded(run) is True


def test_a_failed_grow_finishes_the_wait_and_does_not_count() -> None:
    """It has to end the wait: a failure reported at the timeout reads as an
    outage, and the receipt then says the wrong thing about why."""
    for state in ("GROW_FAILED", "NO_SEED", "POLICY_REJECTED", "FINALIZATION_FAILED"):
        run = _run(state, surface_id=None, artifact_sha256=None)
        assert control._grow_finished(run) is True, state
        assert control._grow_succeeded(run) is False, state


def test_a_duplicate_is_terminal_but_is_not_this_run_s_grow() -> None:
    """DUPLICATE_SURFACE carries somebody else's surface. The control is
    evidence that *this* seed grew."""
    run = _run("DUPLICATE_SURFACE")
    assert control._grow_finished(run) is True
    assert control._grow_succeeded(run) is False


def test_a_running_attempt_does_not_finish_the_wait() -> None:
    for state in ("RUNNING", "CLAIMED", "FINALIZING"):
        assert control._grow_finished(_run(state)) is False, state


def test_a_grown_state_without_an_artifact_is_not_a_grow() -> None:
    """Fails closed: the state alone is not the evidence, the artifact is."""
    assert control._grow_succeeded(_run("QC_PENDING", artifact_sha256=None)) is False
    assert control._grow_succeeded(_run("QC_PENDING", surface_id=None)) is False
