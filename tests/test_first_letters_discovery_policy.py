"""Scientific contract for the First Letters discovery-recovery campaign.

These tests are the admission oracle for the two immutable policy documents.
They deliberately exercise malformed copies as well as the checked-in profiles:
a future consumer must reject the same cases before it runs a control or spends
campaign compute.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = (
    ROOT / "framework/profiles/01-segmentation/first-letters-control-policy-1.0.0.json"
)
CAMPAIGN_PATH = (
    ROOT
    / "framework/profiles/01-segmentation/first-letters-campaign-decision-policy-1.0.0.json"
)
SUCCESSOR_CAMPAIGN_PATH = (
    ROOT
    / "framework/profiles/01-segmentation/first-letters-campaign-decision-policy-1.1.0.json"
)
ACTIVE_SUCCESSOR_CAMPAIGN_PATH = (
    ROOT
    / "framework/profiles/01-segmentation/first-letters-campaign-decision-policy-1.2.0.json"
)
PROBE_PROFILE_PATH = (
    ROOT
    / "framework/stages/01-segmentation/fleet/profiles/vc3d-m7-probe-v1.json"
)
RUNBOOK_PATH = ROOT / "framework/stages/01-segmentation/SEED_PROBE_RUNBOOK.md"
CATALOG_PATH = ROOT / "workspace/catalog/eligible_volumes.json"

VERSIONED_ID = re.compile(r"(?:@\d+\.\d+\.\d+|-v\d+(?:\.\d+)*)$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_READ_SET_FIELDS = {"object_key", "sha256", "bytes"}
PROMOTION_ARTIFACT_FIELDS = {
    "artifact_id",
    "content_sha256",
    "receipt_id",
    "receipt_sha256",
    "namespace",
    "deployed_revision",
}
PROMOTION_CANDIDATE_FIELDS = {
    "candidate_id",
    "candidate_evidence_sha256",
    "provider_response_sha256",
    "coordinate_ct_l0_xyz",
    "coordinate_frame",
}
PROMOTION_LINEAGE_BINDINGS = {
    "parent_discovery_artifact_id",
    "parent_discovery_artifact_sha256",
    "parent_candidate_id",
    "parent_candidate_evidence_sha256",
    "promotion_receipt_sha256",
    "new_full_grow_attempt_id",
}
PROMOTION_POLICY_BINDINGS = {
    "control_receipt_id",
    "control_receipt_sha256",
    "preflight_receipt_id",
    "preflight_receipt_sha256",
    "budget_receipt_id",
    "budget_receipt_sha256",
    "source_snapshot_id",
    "ct_read_set_manifest_sha256",
    "m7_read_set_manifest_sha256",
    "grid_version",
    "m7_level_set_iso_value",
    "ct_material_support_policy",
    "clearance_policy",
    "discovery_policy_id",
    "growth_profile_id",
    "deployed_revision",
}
INK_INFORMED_TOKENS = {
    "glyph",
    "human",
    "ink",
    "letter",
    "ocr",
    "p5",
    "p7",
    "review",
    "text",
    "transcription",
}


class PolicyError(ValueError):
    """The immutable policy is unsafe or scientifically incomplete."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(document: dict[str, Any], key: str, context: str) -> Any:
    value = document.get(key)
    if value is None or value == "" or value == [] or value == {}:
        raise PolicyError(f"{context} missing {key}")
    return value


def require_sha256(value: object, context: str) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PolicyError(f"{context} missing sha256")


def finite_coordinates(value: Any, context: str = "coordinates") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_coordinates(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_coordinates(child, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise PolicyError(f"{context} contains non-finite coordinate")


def normalized_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}


def reject_ink_informed_discovery(value: Any, context: str = "discovery") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if normalized_tokens(str(key)) & INK_INFORMED_TOKENS:
                raise PolicyError(f"ink-informed discovery field: {context}.{key}")
            reject_ink_informed_discovery(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_ink_informed_discovery(child, f"{context}[{index}]")
    elif isinstance(value, str):
        if normalized_tokens(value) & INK_INFORMED_TOKENS:
            raise PolicyError(f"ink-informed discovery value: {context}")


def validate_profile_locks(locks: list[dict[str, Any]]) -> None:
    if not locks:
        raise PolicyError("missing profile locks")
    for lock in locks:
        profile_id = str(require(lock, "profile_id", "profile lock"))
        if not VERSIONED_ID.search(profile_id):
            raise PolicyError(f"unversioned profile: {profile_id}")
        require_sha256(lock.get("sha256"), profile_id)
        require(lock, "path", profile_id)


def validate_chunked_source_lock(lock: dict[str, Any], source_name: str) -> None:
    whole_store = require(lock, "whole_store_digest", f"{source_name} source lock")
    if whole_store != {"status": "UNAVAILABLE_NOT_INVENTED", "sha256": None}:
        raise PolicyError(f"{source_name} source lock invents a whole-store digest")

    execution_lock = require(
        lock, "execution_content_lock", f"{source_name} source lock"
    )
    if execution_lock.get("required_receipt_schema") != (
        "campaignx.first_letters_source_read_set.v1"
    ):
        raise PolicyError(f"{source_name} source lock has wrong read-set schema")
    manifest = require(
        execution_lock, "read_set_manifest", f"{source_name} execution content lock"
    )
    if set(manifest.get("required_fields", [])) != SOURCE_READ_SET_FIELDS:
        raise PolicyError(f"{source_name} read-set manifest fields are incomplete")
    if manifest.get("all_objects_read_must_appear_exactly_once") is not True:
        raise PolicyError(f"{source_name} read-set manifest permits missing objects")
    if manifest.get("canonical_manifest_sha256_required") is not True:
        raise PolicyError(f"{source_name} read-set manifest is not content bound")
    if (
        manifest.get("canonical_order") != "LEXICOGRAPHIC_OBJECT_KEY"
        or execution_lock.get("source_change_after_read") != "CONTROL_INCOMPLETE_STALE"
    ):
        raise PolicyError(
            f"{source_name} read-set canonicalization or staleness is unsafe"
        )
    if execution_lock.get("missing_or_unhashed_read") != "CONTROL_INCOMPLETE":
        raise PolicyError(f"{source_name} read-set failure is not fail closed")
    expected_completion = {
        "ct": "EVERY_OBJECT_READ_HAS_SHA256_AND_BYTES_AND_THE_CANONICAL_MANIFEST_SHA256_IS_RECORDED",
        "m7": "EVERY_OBJECT_READ_HAS_SHA256_AND_BYTES_THE_CANONICAL_MANIFEST_SHA256_IS_RECORDED_AND_PROVIDER_REQUEST_AND_RESPONSE_BYTES_ARE_HASH_BOUND",
    }
    if execution_lock.get("complete_if") != expected_completion[source_name]:
        raise PolicyError(
            f"{source_name} source lock does not require materialized read-set hashes"
        )
    if source_name == "m7":
        provider = require(
            execution_lock, "provider_exchange", "m7 execution content lock"
        )
        for key in (
            "request_sha256_required",
            "response_sha256_required",
            "response_bytes_required",
        ):
            if provider.get(key) is not True:
                raise PolicyError(f"m7 provider exchange missing {key}")


def validate_positive_evidence(
    evidence_items: list[dict[str, Any]], source_locks: dict[str, Any]
) -> None:
    required_relationship_by_role = {
        "PUBLIC_COMMUNITY_ATTRIBUTION": (
            "public_evidence",
            "SCHOLARLY_PAPER_WITH_SUPPLEMENTARY_INFORMATION",
        ),
        "SOURCE_LOCKED_PUBLIC_SURFACE": (
            "community_surface",
            "TIFXYZ_SURFACE",
        ),
        "SOURCE_LOCKED_PUBLIC_POSITIVE_MAP": (
            "official_positive_map",
            "INK_PROBABILITY_MAP",
        ),
    }
    evidence_roles = [
        str(require(item, "role", "positive evidence")) for item in evidence_items
    ]
    if len(set(evidence_roles)) != len(evidence_roles):
        raise PolicyError("duplicate positive evidence role")
    evidence_by_role = dict(zip(evidence_roles, evidence_items, strict=True))
    for item in evidence_items:
        if "source_lock" in item and item["source_lock"] not in source_locks:
            raise PolicyError(
                f"positive evidence references unknown source lock: {item['source_lock']}"
            )
    for role, (
        expected_lock,
        expected_artifact_kind,
    ) in required_relationship_by_role.items():
        item = evidence_by_role.get(role)
        if item is None or item.get("source_lock") != expected_lock:
            raise PolicyError(f"{role} does not resolve to {expected_lock}")
        if source_locks[expected_lock].get("artifact_kind") != expected_artifact_kind:
            raise PolicyError(
                f"{role} source lock has the wrong declared artifact kind"
            )

    if source_locks["public_evidence"].get("stable_id") != "arXiv:2606.29085v1":
        raise PolicyError("PUBLIC_COMMUNITY_ATTRIBUTION has the wrong source identity")

    repository_evidence = evidence_by_role.get("REPOSITORY_SOURCE_LOCKED_VALIDATION")
    if repository_evidence is None:
        raise PolicyError("missing repository source-locked validation evidence")
    path = ROOT / str(require(repository_evidence, "path", "repository evidence"))
    require_sha256(repository_evidence.get("sha256"), "repository evidence")
    if not path.is_file():
        raise PolicyError("repository evidence path does not exist")
    if hashlib.sha256(path.read_bytes()).hexdigest() != repository_evidence["sha256"]:
        raise PolicyError("repository evidence sha256 does not match")


def validate_promotion_contract(discovery_artifacts: dict[str, Any]) -> None:
    promotion = require(discovery_artifacts, "promotion", "discovery artifacts")
    if not isinstance(promotion, dict) or promotion.get("schema") != (
        "campaignx.first_letters_discovery_promotion_contract.v1"
    ):
        raise PolicyError("wrong promotion schema")

    artifact = require(promotion, "required_discovery_artifact", "promotion")
    if artifact.get("schema") != "campaignx.first_letters_discovery_artifact.v1":
        raise PolicyError("wrong discovery artifact schema")
    if set(artifact.get("required_fields", [])) != PROMOTION_ARTIFACT_FIELDS:
        raise PolicyError("promotion artifact fields are incomplete")
    if artifact.get("namespace") != "NONCANONICAL_DISCOVERY":
        raise PolicyError("promotion artifact namespace is not isolated")

    candidate = require(promotion, "selected_candidate", "promotion")
    if set(candidate.get("required_fields", [])) != PROMOTION_CANDIDATE_FIELDS:
        raise PolicyError("promotion candidate fields are incomplete")
    if candidate.get("must_resolve_inside_discovery_artifact") is not True:
        raise PolicyError("promotion candidate may not resolve to its artifact")

    lineage = require(promotion, "lineage", "promotion")
    if set(lineage.get("required_bindings", [])) != PROMOTION_LINEAGE_BINDINGS:
        raise PolicyError("promotion lineage bindings are incomplete")
    if lineage.get("immutable") is not True:
        raise PolicyError("promotion lineage is mutable")

    policy_bindings = require(promotion, "policy_bindings", "promotion")
    if set(policy_bindings.get("required", [])) != PROMOTION_POLICY_BINDINGS:
        raise PolicyError("promotion policy bindings are incomplete")

    admission = require(promotion, "admission", "promotion")
    if admission.get("mutation") != "CREATE_NEW_NORMAL_FULL_GROW_ATTEMPT":
        raise PolicyError("promotion does not create a new full grow")
    if admission.get("reuse_discovery_artifact_as_canonical") is not False:
        raise PolicyError("promotion permits canonical discovery reuse")
    if admission.get("require_normal_admission_gates") is not True:
        raise PolicyError("promotion bypasses normal admission gates")
    if admission.get("ambiguous_write") != (
        "CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK"
    ):
        raise PolicyError("promotion ambiguous-write handling is unsafe")


def validate_control(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != "campaignx.first_letters_control_manifest.v1":
        raise PolicyError("wrong control schema")
    profile_id = str(require(document, "profile_id", "control"))
    if not VERSIONED_ID.search(profile_id):
        raise PolicyError("unversioned control policy")

    cohort = require(document, "control_cohort", "control")
    control_scroll = str(require(cohort, "scroll_id", "control cohort"))
    evaluation_scrolls = require(cohort, "evaluation_scroll_ids", "control cohort")
    if control_scroll in evaluation_scrolls:
        raise PolicyError("control/evaluation regions overlap")
    if len(evaluation_scrolls) != len(set(evaluation_scrolls)):
        raise PolicyError("duplicate evaluation region")

    locks = require(document, "source_locks", "control")
    for name in (
        "public_evidence",
        "ct",
        "m7",
        "community_surface",
        "official_positive_map",
    ):
        lock = require(locks, name, "source locks")
        require(lock, "uri", f"{name} source lock")
        if name in {"ct", "m7"}:
            for metadata in require(lock, "metadata", f"{name} source lock"):
                require_sha256(metadata.get("sha256"), f"{name} metadata")
            validate_chunked_source_lock(lock, name)
        elif name == "community_surface":
            for artifact in require(lock, "artifacts", "community surface lock"):
                require_sha256(artifact.get("sha256"), "community surface artifact")
        elif name == "public_evidence":
            require(lock, "stable_id", "public evidence lock")
            for artifact in require(lock, "artifacts", "public evidence lock"):
                require(artifact, "uri", "public evidence artifact")
                require(artifact, "role", "public evidence artifact")
                require(artifact, "coverage", "public evidence artifact")
                if (
                    artifact.get("role")
                    != "VERSIONED_PAPER_WITH_SUPPLEMENTARY_INFORMATION"
                    or artifact.get("uri") != "https://arxiv.org/pdf/2606.29085v1"
                    or artifact.get("coverage")
                    != "MAIN_TEXT_AND_SUPPLEMENTARY_INFORMATION"
                ):
                    raise PolicyError(
                        "public evidence does not lock the versioned paper and supplementary information"
                    )
                if not isinstance(artifact.get("bytes"), int) or artifact["bytes"] <= 0:
                    raise PolicyError("public evidence artifact missing bytes")
                require_sha256(artifact.get("sha256"), "public evidence artifact")
        else:
            require_sha256(lock.get("sha256"), "official positive map")
            if not isinstance(lock.get("bytes"), int) or lock["bytes"] <= 0:
                raise PolicyError("official positive map missing bytes")

    known_region = require(document, "known_region", "control")
    finite_coordinates(known_region)
    require(known_region, "coordinate_frame", "known region")
    require(known_region, "voxel_size_um", "known region")
    require(known_region, "control_tolerance_ct_l0_voxels", "known region")

    checks = require(document, "checks", "control")
    if set(checks) != {"DISCOVERY_CONTROL", "PIPELINE_CONTROL"}:
        raise PolicyError("control responsibilities are not split")
    for check_name, check in checks.items():
        require(check, "expected_outcome", check_name)
        require(check, "pass_requirements", check_name)
        require(check, "incomplete_outcome", check_name)
        require(check, "failed_outcome", check_name)

    discovery = checks["DISCOVERY_CONTROL"]
    reject_ink_informed_discovery(require(discovery, "inputs", "DISCOVERY_CONTROL"))
    if (
        discovery["expected_outcome"]
        != "AT_LEAST_ONE_POST_CT_POST_CLEARANCE_CANDIDATE_IN_REGION"
    ):
        raise PolicyError("missing discovery expected outcome")
    required_content_bindings = {
        "CT_READ_SET_CONTENT_BOUND",
        "M7_READ_SET_CONTENT_BOUND",
        "M7_PROVIDER_REQUEST_RESPONSE_CONTENT_BOUND",
    }
    if not required_content_bindings <= set(discovery["pass_requirements"]):
        raise PolicyError("discovery control lacks materialized content bindings")
    required_incomplete_causes = {
        "MISSING_OR_UNHASHED_CT_READ_SET",
        "MISSING_OR_UNHASHED_M7_READ_SET",
        "MISSING_OR_UNHASHED_M7_PROVIDER_REQUEST_OR_RESPONSE",
    }
    if not required_incomplete_causes <= set(
        require(discovery, "incomplete_causes", "DISCOVERY_CONTROL")
    ):
        raise PolicyError("discovery control lacks fail-closed content semantics")

    pipeline = checks["PIPELINE_CONTROL"]
    if pipeline.get("seed_origin") not in {"human", "source_locked_community_surface"}:
        raise PolicyError("pipeline control lacks provenance-marked seed")
    required_stages = {
        "CANONICAL_FULL_GROW",
        "GEOMETRY_CERTIFICATION",
        "PHYSICAL_QC",
        "P3_FLATTENING",
        "P4_RENDERING",
        "P5_LIVE_OUTPUT",
        "P7_ROUTING",
        "HUMAN_REVIEW_ROUTING",
    }
    if set(pipeline["pass_requirements"]) != required_stages:
        raise PolicyError("pipeline control omits an acceptance stage")

    positive_evidence = require(document, "positive_evidence", "control")
    evidence_roles = {str(item.get("role")) for item in positive_evidence}
    if evidence_roles == {"SYNTHETIC_PROBABILITY_MAP_TEST"}:
        raise PolicyError("synthetic probability-map test is not a positive control")
    if (
        not {"PUBLIC_COMMUNITY_ATTRIBUTION", "SOURCE_LOCKED_PUBLIC_SURFACE"}
        <= evidence_roles
    ):
        raise PolicyError("control lacks public source-locked positive evidence")
    validate_positive_evidence(positive_evidence, locks)

    validate_profile_locks(require(document, "profile_locks", "control"))
    return document


def validate_campaign(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != "campaignx.first_letters_campaign_policy.v1":
        raise PolicyError("wrong campaign schema")
    profile_id = str(require(document, "profile_id", "campaign policy"))
    if not VERSIONED_ID.search(profile_id):
        raise PolicyError("unversioned campaign policy")
    if document.get("ink_blind") is not True:
        raise PolicyError("campaign discovery must be ink blind")
    reject_ink_informed_discovery(
        require(document, "discovery_inputs", "campaign policy")
    )

    budget = require(document, "task_budget", "campaign policy")
    if budget.get("target_detection_probability") != 0.95:
        raise PolicyError("wrong target detection probability")
    sampled = require(budget, "sampled_preflight", "task budget")
    if sampled.get("lower_bound") != "ONE_SIDED_CLOPPER_PEARSON":
        raise PolicyError("sampled preflight lacks conservative lower bound")
    if sampled.get("confidence_level") != 0.95:
        raise PolicyError("wrong lower-bound confidence level")
    if (
        sampled.get("zero_usable_cells")
        != "NO_TASK_BUDGET_REQUIRE_MORE_PREFLIGHT_OR_NEW_SOURCE"
    ):
        raise PolicyError("sampled zero rule invents a budget")
    if (
        require(budget, "compute_cap", "task budget").get("source")
        != "REQUIRED_FROZEN_INPUT"
    ):
        raise PolicyError("compute cap is not content-bound")

    stopping = require(document, "candidate_starvation", "campaign policy")
    if stopping.get("minimum_scientific_terminal_attempts") != 8:
        raise PolicyError("wrong starvation denominator")
    if stopping.get("pause_no_m7_count") != 7:
        raise PolicyError("wrong starvation numerator")
    if stopping.get("consecutive_zero_raw_m7_scroll_budgets") != 2:
        raise PolicyError("wrong cross-scroll starvation rule")
    required_exclusions = {
        "CANCELLED",
        "CONFIGURATION_BLOCK",
        "LEASE_EXHAUSTION",
        "PUBLICATION_FAILURE",
        "SOURCE_FAILURE",
        "WORKER_FAILURE",
        "FIXTURE_ONLY",
    }
    if (
        set(stopping.get("excluded_from_scientific_denominator", []))
        != required_exclusions
    ):
        raise PolicyError("platform failures contaminate scientific denominator")

    discovery_artifacts = require(document, "discovery_artifacts", "campaign policy")
    validate_promotion_contract(discovery_artifacts)

    validate_profile_locks(require(document, "profile_locks", "campaign policy"))
    return document


@pytest.fixture(name="control")
def control_fixture() -> dict[str, Any]:
    return validate_control(load_json(CONTROL_PATH))


@pytest.fixture(name="campaign")
def campaign_fixture() -> dict[str, Any]:
    return validate_campaign(load_json(CAMPAIGN_PATH))


def test_checked_in_policies_are_complete(
    control: dict[str, Any], campaign: dict[str, Any]
) -> None:
    assert control["profile_id"] == "first-letters-control-policy@1.0.0"
    assert campaign["profile_id"] == "first-letters-campaign-decision-policy@1.0.0"
    assert control["source_locks"]["m7"]["level_set_iso_value"] == 0.2
    assert (
        control["checks"]["DISCOVERY_CONTROL"]["inputs"]["m7_level_set_iso_value"]
        == 0.2
    )


def test_campaign_policy_1_0_0_bytes_remain_sha256_7180c214_and_old_receipts_validate():
    assert hashlib.sha256(CAMPAIGN_PATH.read_bytes()).hexdigest() == (
        "7180c214d5032f4d5ed107c005c0c560a0d3246c240a73d6b9b1a89bef4ff41a"
    )
    assert validate_campaign(load_json(CAMPAIGN_PATH))["profile_id"] == (
        "first-letters-campaign-decision-policy@1.0.0"
    )


def test_campaign_policy_1_1_0_names_exact_predecessor_and_task6_contracts():
    successor = validate_campaign(load_json(SUCCESSOR_CAMPAIGN_PATH))
    assert successor["profile_id"] == (
        "first-letters-campaign-decision-policy@1.1.0"
    )
    assert successor["predecessor"] == {
        "profile_id": "first-letters-campaign-decision-policy@1.0.0",
        "sha256":
            "7180c214d5032f4d5ed107c005c0c560a0d3246c240a73d6b9b1a89bef4ff41a",
    }
    task6 = successor["task6_discovery_isolation"]
    assert task6["production_select"] == (
        "DORMANT_UNTIL_TASK9_CURRENT_CONTROL_READINESS_AND_WAVE_GATE"
    )
    assert task6["promotion"] == (
        "FRESH_ORDINARY_NORMAL_FULL_GROW_CHILD_NOT_RESUME"
    )
    assert task6["compute_ledger"] == (
        "ONE_MISSION_LEDGER_BASELINE_ALTERNATIVE_ADAPTIVE_EXACT_24_UNITS_PER_ITEM"
    )
    assert task6["canonicalization"] == (
        "DISCOVERY_ARTIFACTS_PROHIBITED_AT_EVERY_CANONICAL_AND_SURFACE_BOUNDARY"
    )
    assert task6["allow_unvalidated"] is False
    assert task6["contracts"] == {
        "adaptive": "campaignx.first_letters_discovery_adaptive.v1",
        "benchmark_decision":
            "campaignx.first_letters_seed_probe_benchmark_decision.v2",
        "benchmark_execution_manifest":
            "campaignx.first_letters_seed_probe_benchmark_execution_manifest.v2",
        "canonical_lineage": "campaignx.authoritative_surface_lineage.v1",
        "compute_cap": "campaignx.first_letters_discovery_compute_cap.v1",
        "compute_reservation":
            "campaignx.first_letters_discovery_compute_reservation.v1",
        "coordinate": "campaignx.ct_l0_xyz_coordinate.v1",
        "discovery_receipt": "campaignx.first_letters_discovery_receipt.v1",
        "experimental_arm":
            "campaignx.first_letters_experimental_arm_admission.v1",
        "normal_full_grow":
            "campaignx.first_letters_normal_full_grow_profile.v1",
        "promotion":
            "campaignx.first_letters_discovery_promotion_authority.v1",
        "work_binding":
            "campaignx.first_letters_discovery_work_binding.v1",
    }


def test_campaign_policy_1_2_0_preserves_history_and_seals_ranked_queue_authority():
    """New controlled admissions must name a new immutable policy document."""
    assert hashlib.sha256(CAMPAIGN_PATH.read_bytes()).hexdigest() == (
        "7180c214d5032f4d5ed107c005c0c560a0d3246c240a73d6b9b1a89bef4ff41a"
    )
    assert hashlib.sha256(SUCCESSOR_CAMPAIGN_PATH.read_bytes()).hexdigest() == (
        "790f04465c76f61fa7da8a350e2fa9be16ef50e9578ad46ef06258d8bd385a78"
    )
    successor = validate_campaign(load_json(ACTIVE_SUCCESSOR_CAMPAIGN_PATH))
    assert successor["profile_id"] == (
        "first-letters-campaign-decision-policy@1.2.0"
    )
    assert successor["predecessor"] == {
        "profile_id": "first-letters-campaign-decision-policy@1.1.0",
        "sha256": "790f04465c76f61fa7da8a350e2fa9be16ef50e9578ad46ef06258d8bd385a78",
    }
    assert successor["task6_discovery_isolation"]["allow_unvalidated"] is False
    queue = successor["task_budget"]["queue_execution"]
    assert queue["candidate_rank"] == 1
    assert queue["reconsider_covered"] is False


def test_probe_profile_v1_bytes_remain_sha256_219a0208():
    assert hashlib.sha256(PROBE_PROFILE_PATH.read_bytes()).hexdigest() == (
        "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33"
    )


def test_runbook_states_select_is_task9_gated_and_promotion_is_fresh_not_resume():
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8").lower()
    assert "task 9 current control/readiness/wave gate" in runbook
    assert "fresh ordinary normal full-grow child" in runbook
    assert "not a resume" in runbook
    assert "one common mission compute ledger" in runbook


def test_runbook_forbids_multiscale_arm_content_inputs_unvalidated_and_direct_canonicalization():
    runbook = RUNBOOK_PATH.read_text(encoding="utf-8").lower()
    assert "never combine scores across scales or experimental arms" in runbook
    assert "content-informed inputs are prohibited" in runbook
    assert "allow_unvalidated=false" in runbook
    assert "never directly canonicalized" in runbook


def test_the_control_scroll_is_not_a_catalogued_target(
    control: dict[str, Any],
) -> None:
    """It fixed the catalogue at thirteen and required the evaluation cohort to
    equal it. Those are different sets that happened to coincide: the cohort is
    frozen for an experiment, the catalogue is what this platform can intake,
    and the open-data bucket has twenty-six volumes with everything P0 and P1
    need. Growing the catalogue broke a test about the control.

    What the control needs is that its own scroll is not among the targets --
    PHerc0139 proves the pipeline finds ink, and a scroll cannot be both the
    proof and the thing being scored. The cohort still has to be intakeable,
    which is a containment and not an equality.
    """
    catalog = load_json(CATALOG_PATH)
    catalogued = {entry["sample_id"] for entry in catalog["entries"]}
    cohort = set(control["control_cohort"]["evaluation_scroll_ids"])

    assert control["control_cohort"]["scroll_id"] == "PHerc0139"
    assert "PHerc0139" not in catalogued
    assert cohort <= catalogued, (
        f"the cohort names scrolls the catalogue lacks: {sorted(cohort - catalogued)}")


def test_profile_locks_match_the_checked_in_bytes(
    control: dict[str, Any], campaign: dict[str, Any]
) -> None:
    locks = control["profile_locks"] + campaign["profile_locks"]
    for lock in locks:
        path = ROOT / lock["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == lock["sha256"], path
        locked_document = load_json(path)
        if "profile_id" in locked_document:
            assert locked_document["profile_id"] == lock["profile_id"], path


@pytest.mark.parametrize(
    "source_name", ["public_evidence", "ct", "m7", "community_surface"]
)
def test_missing_source_lock_is_rejected(
    control: dict[str, Any], source_name: str
) -> None:
    malformed = copy.deepcopy(control)
    del malformed["source_locks"][source_name]
    with pytest.raises(PolicyError, match="source locks missing"):
        validate_control(malformed)


@pytest.mark.parametrize("check_name", ["DISCOVERY_CONTROL", "PIPELINE_CONTROL"])
def test_missing_expected_outcome_is_rejected(
    control: dict[str, Any], check_name: str
) -> None:
    malformed = copy.deepcopy(control)
    del malformed["checks"][check_name]["expected_outcome"]
    with pytest.raises(PolicyError, match="missing expected_outcome"):
        validate_control(malformed)


def test_overlapping_control_and_evaluation_regions_are_rejected(
    control: dict[str, Any],
) -> None:
    malformed = copy.deepcopy(control)
    malformed["control_cohort"]["evaluation_scroll_ids"].append("PHerc0139")
    with pytest.raises(PolicyError, match="overlap"):
        validate_control(malformed)


def test_non_finite_control_coordinates_are_rejected(control: dict[str, Any]) -> None:
    malformed = copy.deepcopy(control)
    malformed["known_region"]["anchor_ct_l0_xyz"][2] = float("nan")
    with pytest.raises(PolicyError, match="non-finite"):
        validate_control(malformed)


@pytest.mark.parametrize("document_name", ["control", "campaign"])
def test_unversioned_profile_is_rejected(
    control: dict[str, Any], campaign: dict[str, Any], document_name: str
) -> None:
    document = copy.deepcopy(control if document_name == "control" else campaign)
    document["profile_locks"][0]["profile_id"] = "floating-latest"
    validator = validate_control if document_name == "control" else validate_campaign
    with pytest.raises(PolicyError, match="unversioned profile"):
        validator(document)


@pytest.mark.parametrize("field", ["ink_score", "glyph_candidates", "human_review"])
def test_ink_informed_discovery_fields_are_rejected(
    control: dict[str, Any], field: str
) -> None:
    malformed = copy.deepcopy(control)
    malformed["checks"]["DISCOVERY_CONTROL"]["inputs"][field] = "forbidden"
    with pytest.raises(PolicyError, match="ink-informed discovery"):
        validate_control(malformed)


def test_synthetic_probability_map_alone_is_rejected(control: dict[str, Any]) -> None:
    malformed = copy.deepcopy(control)
    malformed["positive_evidence"] = [
        {
            "role": "SYNTHETIC_PROBABILITY_MAP_TEST",
            "path": "tests/test_lane_liveness.py",
        }
    ]
    with pytest.raises(PolicyError, match="synthetic probability-map"):
        validate_control(malformed)


def test_missing_pipeline_stage_is_control_incomplete(control: dict[str, Any]) -> None:
    pipeline = control["checks"]["PIPELINE_CONTROL"]
    assert pipeline["incomplete_outcome"] == "CONTROL_INCOMPLETE"
    assert pipeline["failed_outcome"] == "CONTROL_FAILED"
    malformed = copy.deepcopy(control)
    malformed["checks"]["PIPELINE_CONTROL"]["pass_requirements"].remove("P4_RENDERING")
    with pytest.raises(PolicyError, match="omits an acceptance stage"):
        validate_control(malformed)


def test_campaign_zero_and_platform_failure_rules_are_fail_closed(
    campaign: dict[str, Any],
) -> None:
    assert campaign["task_budget"]["census"]["zero_usable_cells"] == (
        "DO_NOT_QUEUE_CURRENT_SOURCE"
    )
    assert campaign["candidate_starvation"]["no_seed_counts_as_no_m7_only_when"] == (
        "RECORDED_RAW_M7_CANDIDATE_COUNT_EQUALS_ZERO"
    )
    assert campaign["discovery_artifacts"]["canonical_admission"] == "PROHIBITED"
    assert campaign["allow_unvalidated"] is False


def test_scholarly_positive_evidence_has_a_versioned_byte_lock(
    control: dict[str, Any],
) -> None:
    public_evidence = control["source_locks"]["public_evidence"]
    artifact = public_evidence["artifacts"][0]
    assert artifact["role"] == "VERSIONED_PAPER_WITH_SUPPLEMENTARY_INFORMATION"
    assert artifact["uri"] == "https://arxiv.org/pdf/2606.29085v1"
    assert artifact["coverage"] == "MAIN_TEXT_AND_SUPPLEMENTARY_INFORMATION"
    assert artifact["bytes"] == 38794737
    assert artifact["sha256"] == (
        "99d894c12970530d528d1b7559273bb783c0da4c67fabe12abe59710d321e77b"
    )


def test_public_evidence_without_a_byte_hash_is_rejected(
    control: dict[str, Any],
) -> None:
    malformed = copy.deepcopy(control)
    malformed["source_locks"]["public_evidence"]["artifacts"] = [
        {
            "role": "VERSIONED_PAPER_WITH_SUPPLEMENTARY_INFORMATION",
            "uri": "https://arxiv.org/pdf/2606.29085v1",
            "coverage": "MAIN_TEXT_AND_SUPPLEMENTARY_INFORMATION",
            "bytes": 1,
        }
    ]
    with pytest.raises(PolicyError, match="public evidence artifact missing sha256"):
        validate_control(malformed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "UNVERSIONED_LANDING_PAGE"),
        ("uri", "https://arxiv.org/abs/2606.29085"),
        ("coverage", "MAIN_TEXT_ONLY"),
    ],
)
def test_public_evidence_without_versioned_paper_and_supplement_is_rejected(
    control: dict[str, Any], field: str, value: str
) -> None:
    malformed = copy.deepcopy(control)
    malformed["source_locks"]["public_evidence"]["artifacts"][0][field] = value
    with pytest.raises(PolicyError, match="paper and supplementary information"):
        validate_control(malformed)


@pytest.mark.parametrize("source_name", ["ct", "m7"])
def test_chunked_source_requires_a_fail_closed_execution_read_set_contract(
    control: dict[str, Any], source_name: str
) -> None:
    source = control["source_locks"][source_name]
    assert source["whole_store_digest"] == {
        "status": "UNAVAILABLE_NOT_INVENTED",
        "sha256": None,
    }
    execution_lock = source["execution_content_lock"]
    assert execution_lock["required_receipt_schema"] == (
        "campaignx.first_letters_source_read_set.v1"
    )
    assert set(execution_lock["read_set_manifest"]["required_fields"]) == (
        SOURCE_READ_SET_FIELDS
    )
    assert (
        execution_lock["read_set_manifest"]["all_objects_read_must_appear_exactly_once"]
        is True
    )
    assert (
        execution_lock["read_set_manifest"]["canonical_manifest_sha256_required"]
        is True
    )
    assert execution_lock["missing_or_unhashed_read"] == "CONTROL_INCOMPLETE"
    if source_name == "m7":
        provider = execution_lock["provider_exchange"]
        assert provider["request_sha256_required"] is True
        assert provider["response_sha256_required"] is True
        assert provider["response_bytes_required"] is True


@pytest.mark.parametrize("source_name", ["ct", "m7"])
def test_chunked_source_without_execution_read_set_contract_is_rejected(
    control: dict[str, Any], source_name: str
) -> None:
    malformed = copy.deepcopy(control)
    malformed["source_locks"][source_name].pop("execution_content_lock", None)
    with pytest.raises(
        PolicyError, match=f"{source_name} source lock missing execution_content_lock"
    ):
        validate_control(malformed)


@pytest.mark.parametrize("source_name", ["ct", "m7"])
def test_chunked_source_cannot_defer_read_set_materialization(
    control: dict[str, Any], source_name: str
) -> None:
    malformed = copy.deepcopy(control)
    malformed["source_locks"][source_name]["execution_content_lock"][
        "complete_if"
    ] = "DECLARED_FOR_LATER"
    with pytest.raises(
        PolicyError, match="does not require materialized read-set hashes"
    ):
        validate_control(malformed)


@pytest.mark.parametrize("source_name", ["ct", "m7"])
def test_chunked_source_read_set_must_be_canonical_and_fail_closed_on_change(
    control: dict[str, Any], source_name: str
) -> None:
    malformed = copy.deepcopy(control)
    execution_lock = malformed["source_locks"][source_name]["execution_content_lock"]
    execution_lock["read_set_manifest"]["canonical_order"] = "ARBITRARY"
    execution_lock["source_change_after_read"] = "IGNORE"
    with pytest.raises(PolicyError, match="read-set canonicalization or staleness"):
        validate_control(malformed)


def test_discovery_control_requires_materialized_ct_m7_and_provider_hashes(
    control: dict[str, Any],
) -> None:
    discovery = control["checks"]["DISCOVERY_CONTROL"]
    assert {
        "CT_READ_SET_CONTENT_BOUND",
        "M7_READ_SET_CONTENT_BOUND",
        "M7_PROVIDER_REQUEST_RESPONSE_CONTENT_BOUND",
    } <= set(discovery["pass_requirements"])
    assert {
        "MISSING_OR_UNHASHED_CT_READ_SET",
        "MISSING_OR_UNHASHED_M7_READ_SET",
        "MISSING_OR_UNHASHED_M7_PROVIDER_REQUEST_OR_RESPONSE",
    } <= set(discovery["incomplete_causes"])


def test_official_positive_map_lock_is_required(control: dict[str, Any]) -> None:
    malformed = copy.deepcopy(control)
    del malformed["source_locks"]["official_positive_map"]
    with pytest.raises(PolicyError, match="source locks missing official_positive_map"):
        validate_control(malformed)


def test_dangling_positive_evidence_source_lock_is_rejected(
    control: dict[str, Any],
) -> None:
    malformed = copy.deepcopy(control)
    malformed["positive_evidence"][1]["source_lock"] = "does_not_exist"
    with pytest.raises(PolicyError, match="unknown source lock"):
        validate_control(malformed)


def test_duplicate_positive_evidence_role_cannot_hide_wrong_preceding_lock(
    control: dict[str, Any],
) -> None:
    malformed = copy.deepcopy(control)
    valid_index = next(
        index
        for index, item in enumerate(malformed["positive_evidence"])
        if item["role"] == "SOURCE_LOCKED_PUBLIC_POSITIVE_MAP"
    )
    malformed["positive_evidence"].insert(
        valid_index,
        {
            "role": "SOURCE_LOCKED_PUBLIC_POSITIVE_MAP",
            "source_lock": "community_surface",
            "observation": "Wrong existing lock must not be hidden by ordering.",
            "independent_validation": False,
        },
    )
    with pytest.raises(PolicyError, match="duplicate positive evidence role"):
        validate_control(malformed)


@pytest.mark.parametrize(
    ("source_lock", "wrong_artifact_kind"),
    [
        ("public_evidence", "TIFXYZ_SURFACE"),
        ("community_surface", "INK_PROBABILITY_MAP"),
        ("official_positive_map", "TIFXYZ_SURFACE"),
    ],
)
def test_positive_evidence_role_requires_matching_declared_artifact_kind(
    control: dict[str, Any], source_lock: str, wrong_artifact_kind: str
) -> None:
    malformed = copy.deepcopy(control)
    malformed["source_locks"][source_lock]["artifact_kind"] = wrong_artifact_kind
    with pytest.raises(PolicyError, match="artifact kind"):
        validate_control(malformed)


def test_public_attribution_role_requires_the_frozen_scholarly_identity(
    control: dict[str, Any],
) -> None:
    malformed = copy.deepcopy(control)
    malformed["source_locks"]["public_evidence"]["stable_id"] = "arXiv:other-v1"
    with pytest.raises(PolicyError, match="source identity"):
        validate_control(malformed)


def test_repository_positive_evidence_is_byte_locked(control: dict[str, Any]) -> None:
    evidence = next(
        item
        for item in control["positive_evidence"]
        if item["role"] == "REPOSITORY_SOURCE_LOCKED_VALIDATION"
    )
    path = ROOT / evidence["path"]
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]


def test_campaign_promotion_contract_binds_artifact_candidate_and_lineage(
    campaign: dict[str, Any],
) -> None:
    promotion = campaign["discovery_artifacts"]["promotion"]
    assert promotion["schema"] == (
        "campaignx.first_letters_discovery_promotion_contract.v1"
    )
    assert set(promotion["required_discovery_artifact"]["required_fields"]) == (
        PROMOTION_ARTIFACT_FIELDS
    )
    assert set(promotion["selected_candidate"]["required_fields"]) == (
        PROMOTION_CANDIDATE_FIELDS
    )
    assert set(promotion["lineage"]["required_bindings"]) == (
        PROMOTION_LINEAGE_BINDINGS
    )
    assert set(promotion["policy_bindings"]["required"]) == (PROMOTION_POLICY_BINDINGS)
    assert promotion["admission"]["mutation"] == ("CREATE_NEW_NORMAL_FULL_GROW_ATTEMPT")
    assert promotion["admission"]["reuse_discovery_artifact_as_canonical"] is False
    assert promotion["admission"]["require_normal_admission_gates"] is True


def test_incomplete_campaign_promotion_contract_is_rejected(
    campaign: dict[str, Any],
) -> None:
    malformed = copy.deepcopy(campaign)
    malformed["discovery_artifacts"]["promotion"] = {
        "schema": "campaignx.first_letters_discovery_promotion_contract.v1",
        "required_discovery_artifact": {
            "schema": "campaignx.first_letters_discovery_artifact.v1",
            "required_fields": sorted(PROMOTION_ARTIFACT_FIELDS - {"content_sha256"}),
            "namespace": "NONCANONICAL_DISCOVERY",
        },
        "selected_candidate": {
            "required_fields": sorted(PROMOTION_CANDIDATE_FIELDS),
            "must_resolve_inside_discovery_artifact": True,
        },
        "lineage": {
            "required_bindings": sorted(PROMOTION_LINEAGE_BINDINGS),
            "immutable": True,
        },
        "policy_bindings": {"required": sorted(PROMOTION_POLICY_BINDINGS)},
        "admission": {
            "mutation": "CREATE_NEW_NORMAL_FULL_GROW_ATTEMPT",
            "reuse_discovery_artifact_as_canonical": False,
            "require_normal_admission_gates": True,
            "ambiguous_write": "CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK",
        },
    }
    with pytest.raises(PolicyError, match="promotion artifact fields"):
        validate_campaign(malformed)
