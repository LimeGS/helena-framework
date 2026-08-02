"""BL-11: every decision threshold declares the sample it actually rests on.

BL-10 verified one hand-written declaration for one frozen profile.  These tests
cover the generalisation: the rule must quantify over the profiles, must not be
satisfiable by a declaration that overstates its evidence, and must keep the
underpowered-threshold inventory visible rather than let it drift.

The last two tests are the scientific ones: they recompute the measured numbers
in the v3 1.1.0 declaration from the receipt it binds, so the declaration cannot
claim a recall the evidence does not support.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/audits/audit_business_logic.py"
SPEC = importlib.util.spec_from_file_location("calibration_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PROFILES = ROOT / "framework/profiles"
GATE_V3_DECLARATION_1_0_0 = (
    PROFILES / "validation/ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json"
)
GATE_V3_DECLARATION_1_1_0 = (
    PROFILES / "validation/ct-fiber-localization-gate-v3-calibration-declaration-1.1.0.json"
)
WINDOW_ROUTER_DECLARATION = (
    PROFILES / "validation/ct-fiber-supported-window-router-v4.1-calibration-declaration-1.0.0.json"
)
DEPTH_CONCENTRATION_DECLARATION = (
    PROFILES / "06-discovery/ct-depth-concentration-priority-calibration-declaration-1.0.0.json"
)
STRICT_SCREEN_DECLARATION = (
    PROFILES / "03-ink/strict-text-like-screen-calibration-declaration-1.0.0.json"
)
ALL_DECLARATIONS = (
    GATE_V3_DECLARATION_1_0_0,
    GATE_V3_DECLARATION_1_1_0,
    WINDOW_ROUTER_DECLARATION,
    DEPTH_CONCENTRATION_DECLARATION,
    STRICT_SCREEN_DECLARATION,
)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def patch_document(monkeypatch, target: Path, mutate) -> None:
    """Serve a mutated copy of one document to the audit, leaving disk alone."""

    original = MODULE.load_object

    def loader(path: Path) -> dict[str, object]:
        document = original(path)
        if Path(path).resolve() == target.resolve():
            document = copy.deepcopy(document)
            mutate(document)
        return document

    monkeypatch.setattr(MODULE, "load_object", loader)


def run() -> dict[str, object]:
    return MODULE.run_audit(ROOT)


def check(report: dict[str, object], check_id: str) -> dict[str, object]:
    return next(item for item in report["checks"] if item["check_id"] == check_id)


@needs_campaign_evidence
def test_coverage_rule_passes_on_the_current_repository() -> None:
    assert check(run(), "BL-11")["status"] == "PASS"


@needs_campaign_evidence
def test_every_declaration_is_discovered_and_structurally_valid() -> None:
    evidence = check(run(), "BL-11")["evidence"]
    assert f"declarations={len(ALL_DECLARATIONS)}" in evidence
    for path in ALL_DECLARATIONS:
        assert path.is_file(), path
        document = load(path)
        assert document["schema"] == MODULE.CALIBRATION_DECLARATION_SCHEMA
        assert document["status"] == "METADATA_ONLY_NO_THRESHOLD_CHANGE"
        assert document["audit_policy"]["threshold_changes_from_this_declaration"] is False


def test_the_frozen_1_0_0_declaration_was_not_edited() -> None:
    """The 1.1.0 declaration must add fields in parallel, never mutate 1.0.0."""

    digest = hashlib.sha256(GATE_V3_DECLARATION_1_0_0.read_bytes()).hexdigest()
    assert digest == (
        "40467cd81ba697fef9e69f0b9b8105c7b59ecb32056c2eca4ac42bc0ac3e0d42"
    ), "the frozen 1.0.0 calibration declaration was modified"
    original = load(GATE_V3_DECLARATION_1_0_0)
    updated = load(GATE_V3_DECLARATION_1_1_0)
    assert original["declaration_id"].endswith("@1.0.0")
    assert updated["declaration_id"].endswith("@1.1.0")
    assert original["bound_profile"] == updated["bound_profile"]
    # Every sample count carried forward unchanged: the new version reports, it
    # does not revise.
    by_feature = {row["feature"]: row["calibration"] for row in original["requirement_calibration"]}
    for row in updated["requirement_calibration"]:
        before = by_feature[row["feature"]]
        after = row["calibration"]
        assert after["positive_n"] == before["positive_n"]
        assert after["negative_n"] == before["negative_n"]
        assert after["sources"] == before["sources"]
        assert after["independent_validation"] is False


@needs_campaign_evidence
def test_a_threshold_carrying_profile_cannot_go_undeclared(monkeypatch) -> None:
    exemptions = dict(MODULE.CALIBRATION_DECLARATION_EXEMPTIONS)
    exemptions.pop("framework/profiles/01-segmentation/hybrid-scrollfiesta-vc3d-0.1.0.json")
    monkeypatch.setattr(MODULE, "CALIBRATION_DECLARATION_EXEMPTIONS", exemptions)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "no calibration declaration" in result["evidence"][0]


@needs_campaign_evidence
def test_a_dead_exemption_fails(monkeypatch) -> None:
    """Exempting a profile that *is* declared must fail, not pass quietly."""

    exemptions = dict(MODULE.CALIBRATION_DECLARATION_EXEMPTIONS)
    exemptions[
        "framework/profiles/06-discovery/ct-depth-concentration-priority-0.1.0.json"
    ] = "test: this profile is in fact declared"
    monkeypatch.setattr(MODULE, "CALIBRATION_DECLARATION_EXEMPTIONS", exemptions)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "exemption is dead" in result["evidence"][0]


@needs_campaign_evidence
def test_a_stale_exemption_fails(monkeypatch) -> None:
    exemptions = dict(MODULE.CALIBRATION_DECLARATION_EXEMPTIONS)
    exemptions["framework/profiles/validation/does-not-exist.json"] = "test: stale"
    monkeypatch.setattr(MODULE, "CALIBRATION_DECLARATION_EXEMPTIONS", exemptions)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "no longer applies" in result["evidence"][0]


@needs_campaign_evidence
def test_a_declaration_cannot_claim_independent_validation(monkeypatch) -> None:
    def mutate(document: dict[str, object]) -> None:
        document["requirement_calibration"][0]["calibration"]["independent_validation"] = True

    patch_document(monkeypatch, WINDOW_ROUTER_DECLARATION, mutate)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "overstates independent validation" in result["evidence"][0]


@needs_campaign_evidence
def test_a_declaration_cannot_skip_a_decision_parameter(monkeypatch) -> None:
    """Declaring seven of eight window parameters must fail, not pass."""

    def mutate(document: dict[str, object]) -> None:
        document["requirement_calibration"].pop()

    patch_document(monkeypatch, WINDOW_ROUTER_DECLARATION, mutate)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "do not match the bound decision parameters" in result["evidence"][0]


@needs_campaign_evidence
def test_an_underpowered_threshold_cannot_quietly_inflate_its_sample(monkeypatch) -> None:
    def mutate(document: dict[str, object]) -> None:
        for row in document["requirement_calibration"]:
            row["calibration"]["positive_n"] = 500
            row["calibration"]["negative_n"] = 500

    patch_document(monkeypatch, WINDOW_ROUTER_DECLARATION, mutate)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "underpowered-threshold inventory changed" in result["evidence"][0]


def test_the_zero_sample_window_parameters_are_recorded_as_such() -> None:
    """The operational default router's window numbers rest on n=0 either side."""

    document = load(WINDOW_ROUTER_DECLARATION)
    rows = document["requirement_calibration"]
    assert len(rows) == 8
    for row in rows:
        assert row["calibration"]["positive_n"] == 0, row["feature"]
        assert row["calibration"]["negative_n"] == 0, row["feature"]
    inventory = [
        item for item in MODULE.LOW_SAMPLE_THRESHOLD_INVENTORY
        if "supported-window-router" in item
    ]
    assert len(inventory) == 8
    assert all("positive_n=0,negative_n=0" in item for item in inventory)


def test_the_strict_screen_declaration_tracks_the_real_constants(monkeypatch) -> None:
    """A screen threshold cannot change without the declaration failing."""

    binding = load(STRICT_SCREEN_DECLARATION)["bound_implementation"]
    source = ROOT / binding["path"]
    assert source.is_file()
    text = source.read_text(encoding="utf-8")
    for name, value in binding["constants"].items():
        assert f"{name} = {value}" in text

    def mutate(document: dict[str, object]) -> None:
        document["bound_implementation"]["constants"]["STRICT_SCREEN_MINIMUM_CANDIDATES"] = 3

    patch_document(monkeypatch, STRICT_SCREEN_DECLARATION, mutate)
    result = check(run(), "BL-11")
    assert result["status"] == "FAIL"
    assert "no longer defines it that way" in result["evidence"][0]


def bound_evidence(document: dict[str, object], role: str) -> dict[str, object]:
    binding = next(
        item for item in document["evidence_bindings"] if item["role"] == role
    )
    path = ROOT / binding["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    return json.loads(path.read_text(encoding="utf-8"))


@needs_campaign_evidence
def test_declared_joint_recall_is_reproducible_from_the_bound_receipt() -> None:
    """Recompute 15/33 from the decisions the declaration hash-binds."""

    document = load(GATE_V3_DECLARATION_1_1_0)
    decisions = bound_evidence(
        document, "EIGHT_TERM_GATE_DECISIONS_ON_OWN_CALIBRATION_CONTROLS"
    )
    declared = document["measured_joint_recall"]

    assert len(decisions) == declared["evaluated_control_count"]
    retained = [record for record in decisions if record["retained"]]
    assert len(retained) == declared["retained_control_count"]
    assert declared["joint_recall"] == pytest.approx(len(retained) / len(decisions))
    assert declared["joint_recall"] == pytest.approx(15 / 33)

    for group in declared["by_group"]:
        rows = [r for r in decisions if r["group_id"] == group["group_id"]]
        assert len(rows) == group["evaluated"]
        hits = sum(1 for r in rows if r["retained"])
        assert hits == group["retained"]
        assert group["recall"] == pytest.approx(hits / len(rows))


@needs_campaign_evidence
def test_declared_inert_requirements_are_reproducible_from_the_bound_receipt() -> None:
    """Six of eight requirements never rejected a control; recompute it."""

    document = load(GATE_V3_DECLARATION_1_1_0)
    decisions = bound_evidence(
        document, "EIGHT_TERM_GATE_DECISIONS_ON_OWN_CALIBRATION_CONTROLS"
    )
    declared = document["measured_term_discrimination"]

    rejections: dict[str, int] = {}
    for record in decisions:
        for item in record["checks"]:
            rejections.setdefault(item["feature"], 0)
            if not item["passed"]:
                rejections[item["feature"]] += 1

    assert rejections == declared["rejections_by_feature"]
    inert = sorted(name for name, count in rejections.items() if count == 0)
    assert inert == sorted(declared["inert_requirements"])
    assert len(inert) == declared["inert_requirement_count"] == 6
    assert declared["discriminating_requirement_count"] == 2
    # Only entropy and top3_fraction ever rejected anything, and entropy alone
    # accounts for every downranked control.
    assert rejections["depth_profile_entropy"] == 18
    assert rejections["depth_profile_top3_fraction"] == 8
    assert rejections["depth_profile_entropy"] == len(decisions) - sum(
        1 for record in decisions if record["retained"]
    )

    # Each per-requirement row restates its own measured rejection count.
    for row in document["requirement_calibration"]:
        assert (
            row["calibration"]["measured_rejections_on_calibration_controls"]
            == rejections[row["feature"]]
        )


@needs_campaign_evidence
def test_depth_concentration_declaration_matches_its_evidence() -> None:
    """15 positives and 6 visually-audited negatives, recomputed."""

    document = load(DEPTH_CONCENTRATION_DECLARATION)
    row = document["requirement_calibration"][0]
    assert row["feature"] == "depth_profile_top3_fraction"

    decisions = bound_evidence(document, "POSITIVE_CONTROL_DECISIONS")
    values = [
        item["value"]
        for record in decisions
        if record["group_id"] == "PHerc0139-public-positive"
        for item in record["checks"]
        if item["feature"] == "depth_profile_top3_fraction"
    ]
    assert len(values) == row["calibration"]["positive_n"] == 15
    assert min(values) == pytest.approx(row["calibration"]["positive_observed_range"][0])
    assert max(values) == pytest.approx(row["calibration"]["positive_observed_range"][1])

    assessments = bound_evidence(document, "NEGATIVE_NON_TEXT_DETERMINATION")["assessments"]
    assert len(assessments) == row["calibration"]["negative_n"] == 6
    assert all(
        item["outcome"] == "GEOMETRY_CORRELATED_NO_TEXT_MORPHOLOGY_OBSERVED"
        for item in assessments
    )
    # Every negative sits below the threshold and every positive above it.
    threshold = row["calibration"]["declared_threshold"]
    assert threshold == 0.18
    assert max(row["calibration"]["negative_observed_range"]) < threshold < min(values)
