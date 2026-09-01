"""The first gate to become a lane: publishing a control that belongs to no campaign.

A development control is not a campaign. Its mission carries no campaign
binding -- that is what lets it place its own seeds instead of going through the
campaign's budget authority -- and the publish route then refused its receipt
because stage-survival evidence "belongs to a First Letters mission". The run
happened, the result existed, and there was nowhere to put it.

This is the safest gate to move first, because the record it writes is inert
where it matters: control evidence is read back per mission
(`_latest_first_letters_control_evidence(mission_id)`), so a receipt published
on a mission with no campaign binding cannot answer any campaign's readiness
gate. It is a record of a development run, and it says so.

CERTIFIED is unchanged: a campaign's control still needs a campaign mission, and
asking to publish onto a generic one still returns 409. The lane has to be
declared.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/harness"))

from framework.contracts import execution_mode  # noqa: E402

pytest.importorskip("fastapi")


def _receipt(mission_id: str) -> dict:
    """The shape the publish route checks before it stores anything."""
    import run_first_letters_positive_control as control  # noqa: PLC0415

    receipt = {
        "schema": "campaignx.first_letters_stage_survival.v1",
        "mission_id": mission_id,
        "allow_unvalidated": False,
        "control_pass_is_independent_validation": False,
        "automatic_letter_acceptance": False,
        "ink_blind_discovery": True,
        "bindings": {"deployed_revision": "c" * 40},
        "stages": [],
        "control_state": "CONTROL_INCOMPLETE",
        "first_nonpassing_boundary": "P0",
    }
    receipt["content_sha256"] = control.canonical_sha256(receipt)
    return receipt


def test_a_certified_publish_still_needs_a_campaign_mission(monkeypatch, tmp_path) -> None:
    """The default path is byte-for-byte what it was."""
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _m: {})
    with pytest.raises(panel_app.HTTPException) as refused:
        panel_app._publish_first_letters_control(
            "dev-control", _receipt("dev-control"), published_by="tester")
    assert refused.value.status_code == 409


def test_an_exploratory_publish_records_the_run_it_actually_made(
        monkeypatch, tmp_path) -> None:
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _m: {})
    monkeypatch.setattr(panel_app, "_first_letters_control_root",
                        lambda _m: tmp_path / "control")
    monkeypatch.setattr(panel_app, "_control_survival_evaluator",
                        lambda: lambda receipt: receipt)

    stored = panel_app._publish_first_letters_control(
        "dev-control", _receipt("dev-control"), published_by="tester",
        mode=execution_mode.EXPLORATORY)

    assert stored["control_state"] == "CONTROL_INCOMPLETE"
    written = list((tmp_path / "control").glob("*.json"))
    assert written, "the run happened and nothing recorded it"


def test_the_receipt_bytes_are_not_touched_by_the_lane(monkeypatch, tmp_path) -> None:
    """The receipt is the runner's document and its hash has to stay
    reproducible from its own bytes. The lane is recorded beside it."""
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _m: {})
    monkeypatch.setattr(panel_app, "_first_letters_control_root",
                        lambda _m: tmp_path / "control")
    monkeypatch.setattr(panel_app, "_control_survival_evaluator",
                        lambda: lambda receipt: receipt)
    original = _receipt("dev-control")

    panel_app._publish_first_letters_control(
        "dev-control", dict(original), published_by="tester",
        mode=execution_mode.EXPLORATORY)

    written = json.loads(next((tmp_path / "control").glob("*.json")).read_text())
    assert written == original, "the lane changed the receipt's own bytes"


def test_the_attestation_says_the_run_certifies_nothing(monkeypatch, tmp_path) -> None:
    import panel.app as panel_app

    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _m: {})
    monkeypatch.setattr(panel_app, "_first_letters_control_root",
                        lambda _m: tmp_path / "control")
    monkeypatch.setattr(panel_app, "_control_survival_evaluator",
                        lambda: lambda receipt: receipt)

    panel_app._publish_first_letters_control(
        "dev-control", _receipt("dev-control"), published_by="tester",
        mode=execution_mode.EXPLORATORY)

    attestation = json.loads(
        next((tmp_path / "control" / "attestations").glob("*.json")).read_text())
    assert attestation["certified"] is False
    assert attestation["execution_mode"] == execution_mode.EXPLORATORY
    assert any("First Letters mission" in reason
               for reason in attestation["uncertified_because"])
    assert execution_mode.is_certified(attestation) is False


def test_a_campaign_mission_publishes_certified_as_before(monkeypatch, tmp_path) -> None:
    import panel.app as panel_app

    monkeypatch.setattr(panel_app.mission_contract,
                        "is_first_letters_discovery_manifest", lambda _m: True)
    monkeypatch.setattr(panel_app, "_mission_campaign_manifest", lambda _m: {"x": 1})
    monkeypatch.setattr(panel_app, "_first_letters_control_root",
                        lambda _m: tmp_path / "control")
    monkeypatch.setattr(panel_app, "_control_survival_evaluator",
                        lambda: lambda receipt: receipt)

    panel_app._publish_first_letters_control(
        "campaign", _receipt("campaign"), published_by="tester")

    attestation = json.loads(
        next((tmp_path / "control" / "attestations").glob("*.json")).read_text())
    assert attestation["certified"] is True
    assert execution_mode.is_certified(attestation) is True
