"""The First Letters gates, checked against the deployment that enforces them.

Four gates decide whether a campaign may queue: the positive control on the
exact deployed revision, the source-locked candidate preflight, the derived
task budget, and the candidate-starvation pause. Each was enforced somewhere,
none of them was reported anywhere, and the queue path checked only one.

What is asserted here is the part that decays quietly. A readiness endpoint
that keeps answering after it stopped consulting the evidence looks identical
to one that works, and the campaign that finds out is the one that queued eight
hundred tasks against a revision whose control nobody had re-run.

Read-only with one deliberate exception: a queue POST that must be refused. It
is issued only when the readiness answer already says the campaign is blocked,
so a refusal is the expected outcome and nothing is created either way.
"""

from __future__ import annotations

import os
import re
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

# Nothing in a readiness answer may offer a way around an acceptance gate.
FORBIDDEN = ("allow_unvalidated", "bypass", "override", "force_queue",
             "skip_gate", "accept_anyway")

# Affirmative absence, not the frozen policy's explicit refusal to claim it.
NEVER_SAID = re.compile(
    r"contains no ink|has no ink|there is no ink|ink absent|absence of ink"
    r"|no ink was found|holds no text|contains no letters", re.IGNORECASE)

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
def readiness(panel):
    return panel.call(
        "GET", f"/api/missions/{MISSION}/first-letters-readiness")


def test_the_readiness_endpoint_answers_on_this_deployment(readiness):
    assert readiness["schema"] == "campaignx.first_letters_readiness.v1"
    assert readiness["mission_id"] == MISSION
    assert isinstance(readiness["controlled"], bool)
    assert isinstance(readiness["blockers"], list)
    assert re.fullmatch(r"[0-9a-f]{64}", readiness["readiness_sha256"])


def test_readiness_binds_the_exact_deployed_revision(readiness):
    """A campaign gate that does not name the running code gates nothing."""
    if readiness["controlled"] is not True:
        pytest.skip(f"{MISSION} carries no First Letters campaign binding")
    assert re.fullmatch(r"[0-9a-f]{40}", readiness["deployed_revision"] or "")
    control = readiness["control"]
    assert control["evidence_status"] in {
        "CURRENT", "STALE", "MISSING", "INVALID"}
    if control["evidence_status"] == "CURRENT":
        assert control["bound_deployed_revision"] == readiness["deployed_revision"]


def test_every_blocker_names_the_evidence_that_clears_it(readiness):
    if readiness["controlled"] is not True:
        pytest.skip(f"{MISSION} carries no First Letters campaign binding")
    for blocker in readiness["blockers"]:
        assert blocker["code"] and blocker["scope"] and blocker["detail"], blocker
    if readiness["blockers"]:
        assert readiness["queue_admitted"] is False
        assert readiness["allowed_actions"], (
            "a blocked campaign that offers no next action is a dead end")


def test_readiness_offers_no_acceptance_gate_bypass(panel, readiness):
    del readiness
    raw = panel.call(
        "GET", f"/api/missions/{MISSION}/first-letters-readiness")
    said = repr(raw).lower()
    for forbidden in FORBIDDEN:
        assert forbidden not in said, f"readiness offers {forbidden}"


def test_readiness_never_infers_absence_of_ink(readiness):
    match = NEVER_SAID.search(repr(readiness))
    assert match is None, f"readiness claimed absence: {match.group(0)!r}"
    if readiness["controlled"] is True:
        assert any("not evidence" in claim.lower()
                   for claim in readiness["non_claims"])


def test_small_surface_diagnostics_never_become_a_content_claim(readiness):
    if readiness["controlled"] is not True:
        pytest.skip(f"{MISSION} carries no First Letters campaign binding")
    diagnostics = readiness["small_surfaces"]
    if diagnostics.get("available") is not True:
        pytest.skip(f"small-surface routing is not deployed: {diagnostics.get('reason')}")
    assert diagnostics["is_absence_evidence"] is False
    assert diagnostics["ink_claim"] == "NONE_MADE"
    assert diagnostics["promotion_in_place"] == "PROHIBITED"
    for surface in diagnostics.get("surfaces") or []:
        assert surface["route"] in {
            "STANDARD_QC_PENDING", "SMALL_SURFACE_DIAGNOSTIC", None}
        assert surface.get("is_absence_evidence") in {False, None}


def test_a_blocked_campaign_refuses_a_p1_wave_on_this_deployment(panel, readiness):
    """The one mutation here, and only where the answer already says refused."""
    from panel_client import PanelError

    if readiness["controlled"] is not True:
        pytest.skip(f"{MISSION} carries no First Letters campaign binding")
    if not readiness["blockers"]:
        pytest.skip("this campaign is not blocked, so nothing must be refused")
    with pytest.raises(PanelError) as refusal:
        panel.call("POST", "/api/segmentation/runs", {
            "sample_id": SCROLL, "mission_id": MISSION, "backend": "vc3d",
            "max_tasks": 1,
            "reason": "end-to-end check that a blocked campaign is refused",
        })
    assert refusal.value.status == 409, (
        f"a blocked campaign was answered {refusal.value.status}")
    assert readiness["blockers"][0]["code"] in refusal.value.body


def test_an_ordinary_mission_keeps_its_own_workflow(panel):
    """The gate must be invisible to every mission with no campaign binding."""
    missions = panel.call("GET", "/api/missions")["missions"]
    ordinary = [row["mission_id"] for row in missions
                if row.get("campaign_kind") != "FIRST_LETTERS_DISCOVERY"
                and row["mission_id"] != "unfiled"]
    if not ordinary:
        pytest.skip("this deployment has no mission outside the campaign")
    answer = panel.call(
        "GET", f"/api/missions/{ordinary[0]}/first-letters-readiness")
    assert answer["controlled"] is False
    assert answer["blockers"] == []
    assert answer["allowed_actions"] == []
    assert "no First Letters campaign binding" in answer["reason"]
