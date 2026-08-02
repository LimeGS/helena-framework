from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/audits/audit_business_logic.py"
SPEC = importlib.util.spec_from_file_location("business_logic_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def check(report: dict[str, object], check_id: str) -> dict[str, object]:
    return next(item for item in report["checks"] if item["check_id"] == check_id)


# No business-logic check may fail on this repository.
#
# History, kept because it is the reason BL-SCHEMA-ENFORCE exists: when the rule
# was added it failed, and the failure was a real finding rather than a defect of
# the rule.  framework/registries/method-capabilities-0.1.0.json violated its own
# committed contract, framework/contracts/schemas/
# method-capability-registry-v1.schema.json, in nine places across the three
# newest methods -- two integration_status values, three validation_status values,
# and three null source_url fields.  Nothing had ever validated the registry
# against that schema.  It was resolved on the schema side by *widening* the two
# enums to admit the states the registry already recorded and by admitting a null
# source_url for internal methods, each with a $comment giving the reason;
# widening an enum cannot invalidate a previously valid document, and no
# constraint was removed.  The registry itself was not edited to fit the schema.
#
# The assertion below is a subset check against an empty set: any failing check
# breaks the suite.  If a failure is ever accepted here again it must carry the
# same kind of written justification, and test_accepted_failures_keep_their_
# documented_cause must pin its exact cause so it cannot drift into a different
# failure.
ACCEPTED_FAILING_CHECKS: set[str] = set()
EXPECTED_CHECK_IDS = [
    "BL-01",
    "BL-02",
    "BL-03",
    "BL-04",
    "BL-05",
    "BL-06",
    "BL-07",
    "BL-08",
    "BL-09",
    "BL-10",
    "BL-11",
    "BL-GEN-PROMOTION",
    "BL-GEN-SELECTION",
    "BL-COVERAGE",
    "BL-SCHEMA-ENFORCE",
]
EXPECTED_REPOSITORY_ONLY_SKIPS = {
    "BL-02",
    "BL-04",
    "BL-GEN-SELECTION",
    "BL-COVERAGE",
}
EXPECTED_REPOSITORY_ONLY_PARTIAL = {
    "BL-05",
    "BL-08",
    "BL-09",
    "BL-10",
    "BL-11",
    "BL-GEN-PROMOTION",
    "BL-SCHEMA-ENFORCE",
}
EXPECTED_EXTERNAL_EVIDENCE_SCHEMAS = {
    "ink-method-routing-policy-v1.schema.json",
    "ink-volumetric-patch-input-v1.schema.json",
    "segmentation-artifact-set-v1.schema.json",
    "segmentation-locked-plan-v1.schema.json",
    "segmentation-locked-plan-v2.schema.json",
    "segmentation-planner-packet-v2.schema.json",
    "segmentation-proposal-v1.schema.json",
    "segmentation-proposal-v2.schema.json",
    "segmentation-regional-attempt-history-v1.schema.json",
    "segmentation-task-v1.schema.json",
}


@needs_campaign_evidence
def test_current_repository_business_logic_audit_has_no_new_failure() -> None:
    report = MODULE.run_audit(ROOT)
    assert [item["check_id"] for item in report["checks"]] == EXPECTED_CHECK_IDS
    failing = {item["check_id"] for item in report["checks"] if item["status"] == "FAIL"}
    assert failing <= ACCEPTED_FAILING_CHECKS, json.dumps(report, indent=2)


def test_audit_declares_the_same_roster_this_test_expects() -> None:
    """The audit's frozen roster and this file's expectation must not diverge.

    They are two independent declarations on purpose: the audit enforces its own
    roster at run time (so the CLI in CI catches a dropped rule), and this asserts
    that the enforced roster is the one the suite was written against.
    """

    assert list(MODULE.EXPECTED_CHECK_ROSTER) == EXPECTED_CHECK_IDS
    assert [check_id for check_id, _summary, _function in MODULE.CHECKS] == EXPECTED_CHECK_IDS
    assert len(set(EXPECTED_CHECK_IDS)) == len(EXPECTED_CHECK_IDS)


def test_repository_only_audit_is_explicit_and_green_without_the_data_release() -> None:
    report = MODULE.run_audit(ROOT, repository_only=True)
    statuses = {
        item["check_id"]: item["status"]
        for item in report["checks"]
        if item["check_id"] in EXPECTED_CHECK_IDS
    }

    assert report["status"] == "PASSED_REPOSITORY_ONLY", json.dumps(report, indent=2)
    assert report["mode"] == "REPOSITORY_ONLY"
    assert {
        check_id for check_id, status in statuses.items() if status == "SKIP"
    } == EXPECTED_REPOSITORY_ONLY_SKIPS
    assert {
        check_id for check_id, status in statuses.items() if status == "PARTIAL"
    } == EXPECTED_REPOSITORY_ONLY_PARTIAL
    assert set(MODULE.REPOSITORY_ONLY_SKIPS) == EXPECTED_REPOSITORY_ONLY_SKIPS
    assert set(MODULE.REPOSITORY_ONLY_OVERRIDES) == EXPECTED_REPOSITORY_ONLY_PARTIAL
    assert any(
        "does not inspect the external campaign evidence release" in statement
        for statement in report["non_claims"]
    )


def test_full_audit_does_not_silently_downgrade_without_campaign_evidence() -> None:
    if (ROOT / MODULE.CAMPAIGN_ROOT).is_dir():
        pytest.skip("this checkout has the external campaign evidence overlaid")

    report = MODULE.run_audit(ROOT)
    assert report["mode"] == "FULL_WITH_CAMPAIGN_EVIDENCE"
    assert report["status"] == "FAILED"
    assert check(report, "BL-02")["status"] == "FAIL"
    assert "required file is missing" in check(report, "BL-02")["evidence"][0]


def test_repository_only_cli_reports_its_reduced_scope_and_exits_zero() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--repository-only"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "PASSED_REPOSITORY_ONLY"
    assert report["mode"] == "REPOSITORY_ONLY"


def test_repository_only_mode_still_rejects_an_unsafe_source_mutation(
    monkeypatch,
) -> None:
    original = MODULE.load_object

    def promote(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-physical-priority-router-v4.3.json":
            value = copy.deepcopy(value)
            value["operational_decision"]["v4_3_allowed_as_default"] = True
        return value

    monkeypatch.setattr(MODULE, "load_object", promote)
    report = MODULE.run_audit(ROOT, repository_only=True)
    assert report["status"] == "FAILED"
    assert check(report, "BL-09")["status"] == "FAIL"
    assert "permitted as a default" in check(report, "BL-09")["evidence"][0]


def test_external_evidence_schema_inventory_is_closed_and_stale_checked(
    monkeypatch,
) -> None:
    assert (
        set(MODULE.EXTERNAL_EVIDENCE_SCHEMA_BINDINGS)
        == EXPECTED_EXTERNAL_EVIDENCE_SCHEMAS
    )

    bindings = dict(MODULE.EXTERNAL_EVIDENCE_SCHEMA_BINDINGS)
    bindings["missing-v1.schema.json"] = "synthetic stale entry"
    monkeypatch.setattr(MODULE, "EXTERNAL_EVIDENCE_SCHEMA_BINDINGS", bindings)
    report = MODULE.run_audit(ROOT, repository_only=True)
    assert report["status"] == "FAILED"
    schema_check = check(report, "BL-SCHEMA-ENFORCE")
    assert schema_check["status"] == "FAIL"
    assert "refer to missing schemas" in schema_check["evidence"][0]


def test_a_silently_dropped_rule_fails_the_audit(monkeypatch) -> None:
    """Deleting a rule from CHECKS must go red, not shrink the roster quietly.

    BL-COVERAGE only names BL-08, BL-09 and BL-GEN-PROMOTION, so without this the
    other twelve rules could each be removed with no check ever failing.
    """

    complete = MODULE.CHECKS  # capture once: setattr below would compound otherwise
    for dropped in ("BL-05", "BL-SCHEMA-ENFORCE", "BL-11"):
        reduced = tuple(item for item in complete if item[0] != dropped)
        monkeypatch.setattr(MODULE, "CHECKS", reduced)
        report = MODULE.run_audit(ROOT)
        assert report["status"] == "FAILED", dropped
        roster = check(report, "BL-ROSTER")
        assert roster["status"] == "FAIL"
        assert f"removed={dropped}" in roster["evidence"][0]


def test_an_undeclared_rule_cannot_join_the_roster_silently(monkeypatch) -> None:
    """A rule added without updating the frozen roster is also a failure."""

    def always_passes(root: Path) -> tuple[str, ...]:
        return ("synthetic",)

    monkeypatch.setattr(
        MODULE,
        "CHECKS",
        MODULE.CHECKS + (("BL-99", "synthetic undeclared rule", always_passes),),
    )
    report = MODULE.run_audit(ROOT)
    assert report["status"] == "FAILED"
    assert "undeclared=BL-99" in check(report, "BL-ROSTER")["evidence"][0]


def test_accepted_failures_keep_their_documented_cause() -> None:
    """An accepted failure may not quietly turn into a different failure.

    ``ACCEPTED_FAILING_CHECKS`` is empty today, so this asserts the invariant that
    keeps it honest: every accepted failure must have its cause pinned here.
    """

    causes: dict[str, tuple[str, ...]] = {}
    assert set(causes) == ACCEPTED_FAILING_CHECKS, (
        "every accepted failing check must pin the evidence fragments that "
        "identify its cause"
    )
    report = MODULE.run_audit(ROOT)
    for check_id, fragments in causes.items():
        item = check(report, check_id)
        if item["status"] == "PASS":
            continue
        evidence = item["evidence"][0]
        for fragment in fragments:
            assert fragment in evidence, evidence


def test_registry_conforms_to_its_own_committed_schema() -> None:
    """Regression pin for the drift BL-SCHEMA-ENFORCE found.

    The schema was widened to admit the states the registry records.  This pins
    the outcome directly, so a future narrowing of the enums (or a registry entry
    inventing another undeclared state) fails here as well as inside the rule.
    """

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (ROOT / "framework/contracts/schemas/method-capability-registry-v1.schema.json")
        .read_text(encoding="utf-8")
    )
    instance = json.loads(
        (ROOT / "framework/registries/method-capabilities-0.1.0.json").read_text(
            encoding="utf-8"
        )
    )
    validator = jsonschema.validators.validator_for(schema)(schema)
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in validator.iter_errors(instance)
    ]
    assert not errors, errors


@needs_campaign_evidence
def test_routing_control_hash_drift_fails_closed(monkeypatch) -> None:
    original = MODULE.load_object

    def drift(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "INK_METHOD_ROUTING_POLICY_0.1.0.json":
            value = copy.deepcopy(value)
            value["lanes"][0]["control_receipt"]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(MODULE, "load_object", drift)
    report = MODULE.run_audit(ROOT)
    assert report["status"] == "FAILED"
    assert check(report, "BL-02")["status"] == "FAIL"
    assert "SHA-256 drift" in check(report, "BL-02")["evidence"][0]


def test_scrollfiesta_cannot_silently_become_default(monkeypatch) -> None:
    original = MODULE.load_object

    def promote(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "stage.json" and path.parent.name == "01-segmentation":
            value = copy.deepcopy(value)
            value["backend_images"]["default"] = "helena-scrollfiesta"
        return value

    monkeypatch.setattr(MODULE, "load_object", promote)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-03")["status"] == "FAIL"
    assert "VC3D is no longer" in check(report, "BL-03")["evidence"][0]


@needs_campaign_evidence
def test_r6_cannot_be_represented_as_externally_generalized(monkeypatch) -> None:
    original = MODULE.load_object

    def overclaim(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "RUN_STATE.json" and "legacy-phases/phase2" in path.as_posix():
            value = copy.deepcopy(value)
            value["external_generalization_claim"] = True
        return value

    monkeypatch.setattr(MODULE, "load_object", overclaim)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-04")["status"] == "FAIL"
    assert "claims external generalization" in check(report, "BL-04")["evidence"][0]


def test_first_letters_claim_requires_resolvable_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "EVIDENCE_MANIFEST.json"
    manifest.write_text('{"schema": "test"}', encoding="utf-8")
    receipt = tmp_path / "ADJUDICATION_RECEIPT.json"
    receipt.write_text('{"schema": "test"}', encoding="utf-8")
    manifest_hash = MODULE.sha256_file(manifest)
    receipt_hash = MODULE.sha256_file(receipt)

    # A claim that resolves and re-hashes both bound files is accepted.
    MODULE.require_claim_evidence(
        tmp_path,
        tmp_path,
        {
            "evidence_manifest_sha256": manifest_hash,
            "evidence_manifest_path": "EVIDENCE_MANIFEST.json",
            "adjudication_receipt_sha256": receipt_hash,
            "adjudication_receipt_path": "ADJUDICATION_RECEIPT.json",
        },
        label="claim",
    )

    # Two well-formed hex strings that correspond to no file are not evidence.
    with pytest.raises(MODULE.AuditFailure):
        MODULE.require_claim_evidence(
            tmp_path,
            tmp_path,
            {"evidence_manifest_sha256": "a" * 64, "adjudication_receipt_sha256": "b" * 64},
            label="claim",
        )

    # A declared hash without a declared path is a failure.
    with pytest.raises(MODULE.AuditFailure):
        MODULE.require_claim_evidence(
            tmp_path,
            tmp_path,
            {
                "evidence_manifest_sha256": manifest_hash,
                "evidence_manifest_path": "EVIDENCE_MANIFEST.json",
                "adjudication_receipt_sha256": receipt_hash,
            },
            label="claim",
        )

    # A resolvable path whose content drifted from the declared hash is a failure.
    with pytest.raises(MODULE.AuditFailure):
        MODULE.require_claim_evidence(
            tmp_path,
            tmp_path,
            {
                "evidence_manifest_sha256": "a" * 64,
                "evidence_manifest_path": "EVIDENCE_MANIFEST.json",
                "adjudication_receipt_sha256": receipt_hash,
                "adjudication_receipt_path": "ADJUDICATION_RECEIPT.json",
            },
            label="claim",
        )


@needs_campaign_evidence
def test_first_letters_claim_with_invented_hashes_fails_closed(monkeypatch) -> None:
    """A confirmed claim carrying two invented hashes must fail the whole audit."""

    original = MODULE.load_json

    def inject(path: Path):
        value = original(path)
        if path.name == "FIRST_LETTERS_REVIEW_QUEUE.json" and "first-letters-review-v2" in path.as_posix():
            value = copy.deepcopy(value)
            value["candidates"][0]["claim_state"] = "FIRST_LETTERS_CONFIRMED"
            value["candidates"][0]["evidence_manifest_sha256"] = "a" * 64
            value["candidates"][0]["adjudication_receipt_sha256"] = "b" * 64
        return value

    monkeypatch.setattr(MODULE, "load_json", inject)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-05")["status"] == "FAIL"
    assert "FIRST_LETTERS_CONFIRMED" in check(report, "BL-05")["evidence"][0]


@needs_campaign_evidence
def test_unreadable_campaign_json_fails_outside_first_letters(monkeypatch) -> None:
    """An unreadable JSON document anywhere in the campaign tree is a failure."""

    original = MODULE.load_json

    def corrupt(path: Path):
        if path.name == "INK_METHOD_ROUTING_POLICY_0.1.0.json":
            raise MODULE.AuditFailure(f"cannot read JSON {path}: injected")
        return original(path)

    monkeypatch.setattr(MODULE, "load_json", corrupt)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-05")["status"] == "FAIL"
    assert "cannot read JSON" in check(report, "BL-05")["evidence"][0]


def test_v42_priority_router_cannot_discard_evidence(monkeypatch) -> None:
    original = MODULE.load_object

    def permit_discard(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-texture-priority-router-v4.2.json":
            value = copy.deepcopy(value)
            value["routing"]["automatic_discard"] = True
        return value

    monkeypatch.setattr(MODULE, "load_object", permit_discard)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-08")["status"] == "FAIL"
    assert "automatic evidence discard" in check(report, "BL-08")["evidence"][0]


def test_v43_priority_router_cannot_become_default(monkeypatch) -> None:
    original = MODULE.load_object

    def promote(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-physical-priority-router-v4.3.json":
            value = copy.deepcopy(value)
            value["operational_decision"]["v4_3_allowed_as_default"] = True
        return value

    monkeypatch.setattr(MODULE, "load_object", promote)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-09")["status"] == "FAIL"
    assert "permitted as a default" in check(report, "BL-09")["evidence"][0]


def test_v43_priority_router_cannot_discard_evidence(monkeypatch) -> None:
    original = MODULE.load_object

    def permit_discard(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-physical-priority-router-v4.3.json":
            value = copy.deepcopy(value)
            value["routing"]["automatic_discard"] = True
        return value

    monkeypatch.setattr(MODULE, "load_object", permit_discard)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-09")["status"] == "FAIL"
    assert "automatic evidence discard" in check(report, "BL-09")["evidence"][0]


@needs_campaign_evidence
def test_v3_gate_calibration_declaration_cannot_overstate_independence(
    monkeypatch,
) -> None:
    original = MODULE.load_object

    def overclaim(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json":
            value = copy.deepcopy(value)
            value["requirement_calibration"][0]["calibration"][
                "independent_validation"
            ] = True
        return value

    monkeypatch.setattr(MODULE, "load_object", overclaim)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-10")["status"] == "FAIL"
    assert "overstates independent validation" in check(report, "BL-10")["evidence"][0]


@needs_campaign_evidence
def test_v3_gate_calibration_declaration_cannot_hide_low_sample_counts(
    monkeypatch,
) -> None:
    original = MODULE.load_object

    def hide_low_sample(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json":
            value = copy.deepcopy(value)
            value["requirement_calibration"][0]["calibration"]["negative_n"] = 5
        return value

    monkeypatch.setattr(MODULE, "load_object", hide_low_sample)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-10")["status"] == "FAIL"
    assert "low-sample calibration flags changed" in check(report, "BL-10")["evidence"][0]


def test_composite_qc_profile_rejects_checkpoint_drift(monkeypatch) -> None:
    original = MODULE.load_object

    def drift(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "surface-qc-gp-scroll1-ct-fiber-v3-1.0.0.json":
            value = copy.deepcopy(value)
            value["ink_lane"]["checkpoint_sha256"] = "f" * 64
        return value

    monkeypatch.setattr(MODULE, "load_object", drift)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-06")["status"] == "FAIL"
    assert "checkpoint differs" in check(report, "BL-06")["evidence"][0]


def test_distributed_segmentation_cannot_promote_sqlite(monkeypatch) -> None:
    original = MODULE.load_object

    def drift(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "stage.json" and path.parent.name == "01-segmentation":
            value = copy.deepcopy(value)
            value["distributed_runtime"]["authoritative_control_plane"] = "sqlite"
        return value

    monkeypatch.setattr(MODULE, "load_object", drift)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-07")["status"] == "FAIL"
    assert "not PostgreSQL" in check(report, "BL-07")["evidence"][0]


def test_distributed_segmentation_planner_must_remain_ink_blind(monkeypatch) -> None:
    original = MODULE.load_object

    def drift(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "stage.json" and path.parent.name == "01-segmentation":
            value = copy.deepcopy(value)
            value["planner_contract"]["ink_blind"] = False
        return value

    monkeypatch.setattr(MODULE, "load_object", drift)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-07")["status"] == "FAIL"
    assert "ink_blind" in check(report, "BL-07")["evidence"][0]


def test_distributed_fixtures_must_remain_outside_scientific_qc(monkeypatch) -> None:
    original = MODULE.load_object

    def drift(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "stage.json" and path.parent.name == "01-segmentation":
            value = copy.deepcopy(value)
            value["distributed_runtime"]["fixture_isolation"] = "ALLOW_QC"
        return value

    monkeypatch.setattr(MODULE, "load_object", drift)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-07")["status"] == "FAIL"
    assert "fixtures can enter scientific QC" in check(report, "BL-07")["evidence"][0]


@needs_campaign_evidence
def test_v42_v3_result_cannot_delete_its_per_scroll_metrics(monkeypatch) -> None:
    """all(...) over a deleted collection is True; BL-08 must not accept that."""

    original = MODULE.load_object

    def erase(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "MULTISCROLL_TRANSFER_V3_RESULT.json":
            value = copy.deepcopy(value)
            value.pop("metrics_by_scroll", None)
        return value

    monkeypatch.setattr(MODULE, "load_object", erase)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-08")["status"] == "FAIL"
    assert "metrics_by_scroll" in check(report, "BL-08")["evidence"][0]


@needs_campaign_evidence
def test_v43_v4_result_cannot_delete_its_per_scroll_metrics(monkeypatch) -> None:
    original = MODULE.load_object

    def erase(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "MULTISCROLL_TRANSFER_V4_RESULT.json":
            value = copy.deepcopy(value)
            value.pop("metrics_by_scroll", None)
        return value

    monkeypatch.setattr(MODULE, "load_object", erase)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-09")["status"] == "FAIL"
    assert "metrics_by_scroll" in check(report, "BL-09")["evidence"][0]


@needs_campaign_evidence
def test_v43_v4_result_cannot_report_a_subset_of_the_frozen_scrolls(monkeypatch) -> None:
    original = MODULE.load_object

    def drop_scroll(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "MULTISCROLL_TRANSFER_V4_RESULT.json":
            value = copy.deepcopy(value)
            value["metrics_by_scroll"].pop("PHercMAN5", None)
        return value

    monkeypatch.setattr(MODULE, "load_object", drop_scroll)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-09")["status"] == "FAIL"
    assert "frozen source plan" in check(report, "BL-09")["evidence"][0]


def test_v3_gate_calibration_cannot_drop_its_evidence_bindings(monkeypatch) -> None:
    original = MODULE.load_object

    def erase(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-localization-gate-v3-calibration-declaration-1.0.0.json":
            value = copy.deepcopy(value)
            value["evidence_bindings"] = []
        return value

    monkeypatch.setattr(MODULE, "load_object", erase)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-10")["status"] == "FAIL"
    assert "binds no evidence" in check(report, "BL-10")["evidence"][0]


@needs_campaign_evidence
def test_v47_cannot_be_promoted_after_a_failed_transfer(monkeypatch) -> None:
    """BL-GEN-PROMOTION covers v4.4-v4.7, which no path-pinned rule reaches."""

    original = MODULE.load_json

    def promote(path: Path):
        value = original(path)
        if path.name == "SURFACE_CALIBRATION_TRANSFER_V8_RESULT.json":
            value = copy.deepcopy(value)
            value["promotion_decision"] = "PROMOTE_V47"
        return value

    monkeypatch.setattr(MODULE, "load_json", promote)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-GEN-PROMOTION")["status"] == "FAIL"
    evidence = check(report, "BL-GEN-PROMOTION")["evidence"][0]
    assert "ct-fiber-semantic-priority-router@4.7.0" in evidence
    assert "PROMOTE_V47" in evidence


@needs_campaign_evidence
def test_failed_router_cannot_be_declared_default(monkeypatch) -> None:
    original = MODULE.load_json

    def promote(path: Path):
        value = original(path)
        if path.name == "V42_PROMOTION_DECISION.json":
            value = copy.deepcopy(value)
            value["policy"]["default_router_remains"] = "ct-fiber-texture-priority-router@4.2.0"
        return value

    monkeypatch.setattr(MODULE, "load_json", promote)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-GEN-PROMOTION")["status"] == "FAIL"
    assert "declared default" in check(report, "BL-GEN-PROMOTION")["evidence"][0]


@needs_campaign_evidence
def test_threshold_may_not_be_selected_on_the_reported_evaluation(monkeypatch) -> None:
    original = MODULE.load_json

    def declare(path: Path):
        value = original(path)
        if path.name == "SURFACE_CALIBRATION_TRANSFER_V8_RESULT.json":
            value = copy.deepcopy(value)
            value["threshold_selected_on"] = ["PHercParis4"]
        return value

    monkeypatch.setattr(MODULE, "load_json", declare)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-GEN-SELECTION")["status"] == "FAIL"
    assert "selected its threshold on the same data" in check(
        report, "BL-GEN-SELECTION"
    )["evidence"][0]


def test_profile_without_profile_id_is_not_skipped(monkeypatch) -> None:
    """BL-01 no longer walks past an unidentified profile."""

    original = MODULE.load_object

    def strip_identity(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "timesformer-gp-scroll1-screening-1.0.0.json":
            value = copy.deepcopy(value)
            value.pop("profile_id", None)
        return value

    monkeypatch.setattr(MODULE, "load_object", strip_identity)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-01")["status"] == "FAIL"
    assert "not a declared exception" in check(report, "BL-01")["evidence"][0]


def test_exempt_profile_cannot_lose_its_frozen_kind(monkeypatch) -> None:
    original = MODULE.load_object

    def drift(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "ct-fiber-localization-gate-v3-candidate-coverage.json":
            value = copy.deepcopy(value)
            value["kind"] = "campaignx.ct_surface_localization_gate.v4"
        return value

    monkeypatch.setattr(MODULE, "load_object", drift)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-01")["status"] == "FAIL"
    assert "lost its frozen identity" in check(report, "BL-01")["evidence"][0]


def test_profile_cannot_bind_an_unregistered_checkpoint(monkeypatch) -> None:
    """The second BL-01 escape hatch: no method_id meant no checkpoint check."""

    original = MODULE.load_object

    def smuggle(path: Path) -> dict[str, object]:
        value = original(path)
        if path.name == "surface-qc-gp-scroll1-ct-fiber-v3-1.0.0.json":
            value = copy.deepcopy(value)
            value["ink_lane"]["checkpoint_sha256"] = "1" * 64
        return value

    monkeypatch.setattr(MODULE, "load_object", smuggle)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-01")["status"] == "FAIL"
    assert "checkpoint the registry does not know" in check(report, "BL-01")["evidence"][0]


@needs_campaign_evidence
def test_router_findings_directory_without_a_rule_fails(monkeypatch) -> None:
    coverage = dict(MODULE.ROUTER_FINDINGS_RULE_COVERAGE)
    coverage.pop("ct-priority-router-v47")
    monkeypatch.setattr(MODULE, "ROUTER_FINDINGS_RULE_COVERAGE", coverage)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-COVERAGE")["status"] == "FAIL"
    assert "no declared business-logic rule" in check(report, "BL-COVERAGE")["evidence"][0]


def test_schema_enforcement_rejects_an_undeclared_unused_schema(monkeypatch) -> None:
    exemptions = dict(MODULE.SCHEMA_ENFORCEMENT_EXEMPTIONS)
    exemptions.pop("stage-manifest-v1.schema.json")
    monkeypatch.setattr(MODULE, "SCHEMA_ENFORCEMENT_EXEMPTIONS", exemptions)
    report = MODULE.run_audit(ROOT)
    assert check(report, "BL-SCHEMA-ENFORCE")["status"] == "FAIL"
    evidence = check(report, "BL-SCHEMA-ENFORCE")["evidence"][0]
    assert "stage-manifest-v1.schema.json validates no artifact" in evidence


def test_unexpected_exception_is_a_failed_check_not_a_false_pass(monkeypatch) -> None:
    def explode(root: Path) -> tuple[str, ...]:
        raise ValueError("unexpected")

    monkeypatch.setattr(
        MODULE,
        "CHECKS",
        (("BL-X", "synthetic unexpected failure", explode),),
    )
    report = MODULE.run_audit(ROOT)
    assert report["status"] == "FAILED"
    assert report["checks"][0]["evidence"] == ("ValueError: unexpected",)
