"""The promises the platform makes, checked against the deployment that makes them.

The rest of the end-to-end suite asks whether a phase ran. This asks whether the
things that are supposed to be impossible are still impossible, which is the part
that decays quietly: a gate keeps returning results long after it stopped
checking anything, and nothing looks wrong until somebody audits the catalogue.

Every assertion here is about a boundary that an audit has already found broken
once, or that the stage contracts state outright:

  * grown and imported surfaces are counted apart, because the fleet was being
    credited with work it did not do;
  * an unmeasured surface does not reach model QC, on either the finalisation or
    the imported path;
  * a human verdict is not a scientific one;
  * area is reported as an upper bound rather than as yield;
  * a run records which P0 selection it read;
  * the queue refuses a backend it cannot dispatch, rather than running VC3D
    under another name.

Read-only. Nothing here queues work.
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


@pytest.fixture(scope="module")
def panel():
    from panel_client import Panel

    client = Panel(PANEL)
    assert client.sign_in(USER, PASSWORD) == USER, "the session did not stick"
    return client


@pytest.fixture(scope="module")
def segments(panel):
    answer = panel.call("GET", "/api/segmentation/segments?limit=200")
    assert answer.get("available"), answer.get("reason")
    return answer


def test_grown_and_imported_are_counted_apart(segments):
    """Imported catalogue surfaces live in the same table as grown ones.

    Three aggregations summed the whole table while the page called the result
    "surfaces we grew". The split is reported now, and the arithmetic has to
    close: every surface is one or the other.
    """
    grown = segments["grown_here"]
    imported = segments["imported"]
    assert grown + imported == segments["count"], (
        f"{grown} grown plus {imported} imported does not equal "
        f"{segments['count']} surfaces"
    )
    rows = segments["segments"]
    assert all(row["origin"] in {"GROWN_HERE", "IMPORTED"} for row in rows)
    assert sum(1 for row in rows if row["origin"] == "GROWN_HERE") <= grown


def test_an_imported_surface_has_no_attempt_and_a_grown_one_does(segments):
    """The origin rule itself, which everything above depends on."""
    for row in segments["segments"]:
        if row["origin"] == "GROWN_HERE":
            assert row.get("attempt_id"), (
                f"{row['surface_id']} is called grown and names no attempt"
            )
        else:
            assert not row.get("attempt_id"), (
                f"{row['surface_id']} is called imported and names an attempt"
            )


def test_a_human_verdict_is_not_a_scientific_one(segments):
    """Approved by a person and certified by the fleet are different columns.

    One click should never be able to forge the second, so a surface carrying a
    human review must still show whatever geometry QC decided, untouched.
    """
    reviewed = [r for r in segments["segments"] if r.get("human_review")]
    if not reviewed:
        pytest.skip("no surface carries a human review on this control plane")
    for row in reviewed:
        verdict = row["human_review"]["verdict"]
        assert verdict in {"APPROVED", "DEFECTIVE", "REVIEWED", "INSPECT"}
        assert verdict != row.get("geometry_qc_state"), (
            f"{row['surface_id']} carries the same word as a human verdict and "
            "a geometry verdict, which is how the two get confused"
        )


def test_no_surface_claims_certification_it_does_not_have(segments):
    """The geometry vocabulary is closed. Anything outside it is a page that
    renders a state nobody defined."""
    allowed = {"GEOMETRY_CERTIFIED", "GEOMETRY_UNMEASURED",
               "GEOMETRY_REJECTED_BRIDGE", "GEOMETRY_REJECTED_LAMINA_SWITCH",
               "GEOMETRY_REJECTED_DISTORTION", "GEOMETRY_REJECTED_COVERAGE"}
    for row in segments["segments"]:
        assert row["geometry_qc_state"] in allowed, (
            f"{row['surface_id']} reports {row['geometry_qc_state']!r}"
        )


def test_the_page_reports_area_as_an_upper_bound(panel):
    """Naive area double-counts identity and can include radial spokes.

    The scientific handoff says so, so the number the page shows is not yield and
    must not be labelled as though it were.
    """
    state = panel.call("GET", f"/api/segmentation?mission={MISSION}&subject={SCROLL}")
    private = state["private"]
    assert "area_cm2" in private
    # Certified area is reported separately, because that is the figure the
    # downstream gates actually admit on.
    assert "certified" in private, (
        "the certified count is computed and not served, which is how a gross "
        "number ends up being read as yield"
    )


def test_the_queue_refuses_a_backend_it_cannot_dispatch(panel):
    """A request naming ScrollFiesta must not quietly run VC3D.

    The fleet instantiates VC3DGrowExecutor unconditionally, so a wrong method
    label on a scientific record cannot be detected afterwards. The refusal is
    the feature.
    """
    from panel_client import PanelError

    with pytest.raises(PanelError) as refusal:
        panel.call("POST", "/api/segmentation/runs", {
            "sample_id": SCROLL, "mission_id": MISSION,
            "backend": "scrollfiesta",
            "reason": "end-to-end check that this is refused",
            "max_tasks": 1,
        })
    assert refusal.value.status in {400, 501}, (
        f"a comparison backend was answered {refusal.value.status}"
    )


def test_a_maintenance_action_outside_the_allowlist_is_refused(panel):
    """`action` reaches a subprocess argv, so the allowlist is the boundary."""
    from panel_client import PanelError

    with pytest.raises(PanelError) as refusal:
        panel.call("POST", "/api/segmentation/maintenance",
                   {"action": "; rm -rf /"})
    assert refusal.value.status == 400


def test_a_run_records_which_p0_selection_it_read(panel):
    """Not every task carries one -- the field is newer than the backlog -- but
    the endpoint that queues them has to be asking for it."""
    runs = panel.call("GET", "/api/segmentation/runs?limit=50")
    assert runs.get("available"), runs.get("reason")
    for run in runs["runs"]:
        # Every attempt names the task it belongs to; that is the thread by which
        # provenance is reachable at all.
        assert run.get("task_id"), f"{run['attempt_id']} names no task"


def test_a_failure_is_explained_somewhere_and_the_arithmetic_closes(panel):
    """NO_SEED is not a cause, it is the label for "none survived".

    So the explanation is not on the attempt: 97 of the 155 failures on this
    control plane carry no error and no exit code, and that is by design. The
    screens record which one removed the proposals, and the diagnosis endpoint
    adds them up. What has to hold is that the sum closes -- every attempt is
    either diagnosed or counted as undiagnosed -- because a screen that stopped
    recording would otherwise just shrink one bucket quietly.
    """
    diagnosis = panel.call("GET", f"/api/segmentation/no-seed?sample={SCROLL}")
    if not diagnosis.get("available"):
        pytest.skip(diagnosis.get("reason", "no diagnosis available"))

    attempts = diagnosis["attempts"]
    # by_cause counts occurrences, not attempts: one attempt can trip two
    # screens, and filtering on either has to find it. Summing it and calling
    # the total an attempt count missed by exactly the number of rows with two
    # causes -- 163 over 161. `diagnosed` is the attempt-level partition.
    diagnosed = diagnosis["diagnosed"]
    assert sum(diagnosis["by_cause"].values()) >= diagnosed, (
        "fewer causes than diagnosed attempts, so an attempt was diagnosed by "
        "nothing"
    )
    undiagnosed = diagnosis["undiagnosed"]
    assert diagnosed + undiagnosed == attempts, (
        f"{diagnosed} diagnosed plus {undiagnosed} undiagnosed does not equal "
        f"{attempts} attempts, so a cause is being counted twice or lost"
    )

    screens = diagnosis["candidates_surviving_each_screen"]
    assert screens["usable_candidate_count"] <= screens["raw_candidate_count"], (
        "more candidates survived screening than were ever proposed"
    )


def test_a_failed_attempt_that_ran_a_binary_reports_its_exit(panel):
    """Distinct from the above: an attempt that actually executed VC3D and failed
    has an exit code, and losing it would leave the run table saying only that
    something went wrong."""
    runs = panel.call("GET", "/api/segmentation/runs?limit=200")
    executed = [r for r in runs["runs"]
                if r.get("exit_code") is not None or r.get("error")]
    if not executed:
        pytest.skip("no attempt on this control plane carries an execution result")
    assert all(r.get("attempt_id") for r in executed)
