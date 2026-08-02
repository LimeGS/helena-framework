#!/usr/bin/env python3
"""Verify the locally frozen Helena Framework Phase 0–2 closure evidence.

This is an integrity audit, not a scientific promotion.  A successful exit
means that Phase 0 and Phase 1 are internally consistent and that the recorded
Phase 2 terminal state matches its immutable evidence.  It keeps
``allowed`` false while blocked and recognizes the narrowly scoped
``COMPLETED_LOCAL_HOLDOUT_V1_ONLY`` and ``COMPLETED_LOCAL_FUNCTIONAL_ONLY``
transitions without presenting either as H1 or independent validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
LOCAL_BLOCKED = "BLOCKED_LOCAL_HOLDOUT_V1_MODEL_FREEZE"
LOCAL_COMPLETED = "COMPLETED_LOCAL_HOLDOUT_V1_ONLY"
R6_LOCAL_COMPLETED = "COMPLETED_LOCAL_FUNCTIONAL_ONLY"
LOCAL_SCOPE = "LOCAL_PIPELINE_CONTINUATION_ONLY"
LOCAL_ARTIFACT_SCOPE = "LOCAL_HOLDOUT_V1_ONLY"
LOCAL_AMENDMENT = "phase2/PHASE2_CONTRACT_AMENDMENT_014.md"
LOCAL_BASE_AMENDMENT = "phase2/PHASE2_CONTRACT_AMENDMENT_013.md"
LOCAL_PLAN = "phase2/RELATION_V2_LOCAL_HOLDOUT_V1_PLAN.md"
LOCAL_V1_TERMINAL = "BLOCKED_LOCAL_HOLDOUT_V1_LABEL_NOISE"
LOCAL_V2_AMENDMENT = "phase2/PHASE2_CONTRACT_AMENDMENT_018.md"
LOCAL_V2_PLAN = (
    "phase2/RELATION_V2_LOCAL_HOLDOUT_V2_SUPPORT_FEASIBILITY_RECOVERY_PLAN.md"
)
R6_AMENDMENT = "phase2/PHASE2_CONTRACT_AMENDMENT_020.md"
R6_PLAN = "phase2/RELATION_V2_R6_DIRECT_GEOMETRY_PLAN.md"
R6_POLICY = "phase2/benchmark/r6_direct_geometry/R6_DIRECT_GEOMETRY_POLICY.json"
R6_RESULT = "phase2/benchmark/r6_direct_geometry/R6_LOCAL_FUNCTIONAL_RESULTS.json"
R6_CLOSEOUT = (
    "phase2/benchmark/r6_direct_geometry/R6_LOCAL_FUNCTIONAL_CLOSEOUT.json"
)
R6_SIGN_POSTMORTEM = (
    "phase2/benchmark/r6_direct_geometry/R5_SIGN_TARGET_ORDER_POSTMORTEM.json"
)
AMENDMENT_011_PATH = "phase2/PHASE2_CONTRACT_AMENDMENT_011.md"
AMENDMENT_011_SHA256 = (
    "dd6a7f61fda782a615490327e0fb7fa02f59e6fee3842a1d14eedced27492a07"
)
AMENDMENT_011_STATUS = "DRAFT_REQUIRES_EXPLICIT_USER_AUTHORIZATION"
PRE_STAGE_REFACTOR_COMMIT = "ef47833f402cc5173a1cc095b3e917568e27bc4b"
sys.path.insert(0, str(ROOT))
import campaign_x as cx  # noqa: E402
from scripts.harness.stage_script_registry import resolve_stage_script  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def no_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicate_keys,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_historical_blob(commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise ValueError(
            f"historical implementation blob unavailable: {commit}:{relative}"
        )
    return hashlib.sha256(completed.stdout).hexdigest()


def markdown_contract_status(path: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("**Status:**")
    ]
    if len(lines) != 1:
        raise ValueError(f"expected one contract status: {path.relative_to(ROOT)}")
    prefix = "**Status:** `"
    suffix = "`"
    line = lines[0]
    if not line.startswith(prefix) or not line.endswith(suffix):
        raise ValueError(f"contract status syntax differs: {path.relative_to(ROOT)}")
    return line[len(prefix) : -len(suffix)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_hash_map(hash_map: Any, label: str) -> None:
    require(
        isinstance(hash_map, dict) and hash_map, f"Phase 2 {label} hash map missing"
    )
    for relative, expected in hash_map.items():
        require(
            isinstance(relative, str) and isinstance(expected, str),
            f"Phase 2 malformed {label} hash entry",
        )
        path = ROOT / relative
        # Historical receipts bind the byte content under the former flat
        # ``scripts/<name>`` layout.  Repository organization may change the
        # location, but never the bound bytes.  Resolve only a missing legacy
        # basename through the unique stage registry; all other paths remain
        # exact and fail closed.
        if not path.is_file() and Path(relative).parent == Path("scripts"):
            path = resolve_stage_script(ROOT, Path(relative).name)
        require(path.is_file(), f"Phase 2 {label} input missing: {relative}")
        actual = sha256_file(path)
        if actual != expected and Path(relative).parent == Path("scripts"):
            # Stage ownership changed the root-discovery line in some active
            # implementations.  Immutable Phase-2 receipts continue to bind
            # the original blob, retained in Git at the exact pre-refactor
            # checkpoint.  This is not a hash alias: the historical bytes are
            # loaded and hashed on every audit.
            actual = sha256_historical_blob(PRE_STAGE_REFACTOR_COMMIT, relative)
        require(actual == expected, f"Phase 2 {label} hash mismatch: {relative}")


def verify_local_holdout_transition(
    results: dict[str, Any],
    state: dict[str, Any],
    relation: dict[str, Any],
    local_config: dict[str, Any] | None = None,
    local_v2_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify either the fail-closed pre-run state or the exact local PASS."""

    require(
        relation.get("status") == "BLOCKED_PENDING_INDEPENDENT_H1"
        and relation.get("eligible") is False,
        "Historical relation/H1 state was overwritten by the local route",
    )
    local_state = state.get("relation_v2_local_holdout_v1")
    local_results = results.get("relation_v2_local_holdout_v1")
    require(
        isinstance(local_state, dict) and isinstance(local_results, dict),
        "LOCAL_HOLDOUT_V1 state block is missing",
    )
    require(
        local_state == local_results,
        "RUN_STATE and PHASE2_RESULTS disagree on LOCAL_HOLDOUT_V1",
    )
    overall = state.get("overall")
    require(results.get("overall") == overall, "Phase 2 terminal state disagreement")

    if isinstance(overall, str) and overall.startswith("BLOCKED_LOCAL_HOLDOUT_V2_"):
        v2_state = state.get("relation_v2_local_holdout_v2")
        v2_results = results.get("relation_v2_local_holdout_v2")
        require(
            isinstance(v2_state, dict)
            and isinstance(v2_results, dict)
            and v2_state == v2_results,
            "RUN_STATE and PHASE2_RESULTS disagree on LOCAL_HOLDOUT_V2",
        )
        require(
            local_state.get("status") == LOCAL_V1_TERMINAL
            and v2_state.get("v19_terminal_status") == LOCAL_V1_TERMINAL
            and v2_state.get("v19_execution_claim_created") is False
            and v2_state.get("v19_r5_output_read") is False,
            "LOCAL_HOLDOUT_V1 terminal preclaim failure was overwritten",
        )
        require(
            v2_state.get("status") == overall
            and v2_state.get("protocol") == "LOCAL_HOLDOUT_V2"
            and v2_state.get("amendment") == LOCAL_V2_AMENDMENT
            and v2_state.get("plan") == LOCAL_V2_PLAN
            and v2_state.get("authorization_received") is True
            and v2_state.get("authorization_committed") is True
            and v2_state.get("real_data_execution_enabled") is True
            and v2_state.get("amendment_011_activated") is False
            and v2_state.get("h0_reused") is False
            and v2_state.get("independent_h1_validated") is False
            and v2_state.get("external_generalization_claim") is False
            and v2_state.get("complete") is False
            and v2_state.get("eligible") is False,
            "LOCAL_HOLDOUT_V2 blocked-state governance differs",
        )
        require(
            state.get("complete") is False
            and results.get("complete") is False
            and state.get("eligible") is False
            and results.get("eligible") is False,
            "Blocked LOCAL_HOLDOUT_V2 incorrectly permits Phase 3",
        )
        require(
            isinstance(local_v2_config, dict)
            and local_v2_config.get("status") == overall
            and local_v2_config.get("validation_protocol") == "LOCAL_HOLDOUT_V2"
            and isinstance(local_v2_config.get("governance"), dict)
            and local_v2_config["governance"].get("required_contract_amendment")
            == LOCAL_V2_AMENDMENT
            and local_v2_config["governance"].get("required_plan") == LOCAL_V2_PLAN
            and local_v2_config["governance"].get("amendment_011_activated") is False,
            "LOCAL_HOLDOUT_V2 config governance identity differs",
        )
        return {
            "status": overall,
            "eligible": False,
            "validation_scope": None,
            "activation_state": "ACTIVATED",
        }

    if overall == R6_LOCAL_COMPLETED:
        v2_state = state.get("relation_v2_local_holdout_v2")
        v2_results = results.get("relation_v2_local_holdout_v2")
        r6_state = state.get("relation_v2_r6")
        r6_results = results.get("relation_v2_r6")
        require(
            isinstance(v2_state, dict)
            and isinstance(v2_results, dict)
            and v2_state == v2_results
            and v2_state.get("status") == "BLOCKED_LOCAL_HOLDOUT_V2_IMPLEMENTATION",
            "Historical LOCAL_HOLDOUT_V2 activation block changed",
        )
        require(
            isinstance(r6_state, dict)
            and isinstance(r6_results, dict)
            and r6_state == r6_results,
            "RUN_STATE and PHASE2_RESULTS disagree on R6",
        )
        expected_transition = {
            "overall": R6_LOCAL_COMPLETED,
            "complete": True,
            "completion_kind": R6_LOCAL_COMPLETED,
            "eligible": True,
            "validation_scope": LOCAL_SCOPE,
            "independent_h1_validated": False,
            "external_generalization_claim": False,
        }
        for document, label in ((state, "RUN_STATE"), (results, "PHASE2_RESULTS")):
            require(
                all(
                    document.get(key) == value
                    for key, value in expected_transition.items()
                ),
                f"{label} R6 local-functional transition fields differ",
            )
        require(
            r6_state.get("status") == "PASSED_R6_LOCAL_FUNCTIONAL"
            and r6_state.get("scope") == LOCAL_SCOPE
            and r6_state.get("amendment") == R6_AMENDMENT
            and r6_state.get("plan") == R6_PLAN
            and r6_state.get("closeout") == R6_CLOSEOUT
            and r6_state.get("policy") == R6_POLICY
            and r6_state.get("result") == R6_RESULT
            and r6_state.get("sign_target_postmortem") == R6_SIGN_POSTMORTEM
            and r6_state.get("h0_reused") is False
            and r6_state.get("h1_opened") is False
            and r6_state.get("independent_h1_validated") is False
            and r6_state.get("external_generalization_claim") is False
            and r6_state.get("first_letters_eligible") is True,
            "R6 local-functional registration differs",
        )
        amendment_path = ROOT / R6_AMENDMENT
        policy_path = ROOT / R6_POLICY
        result_path = ROOT / R6_RESULT
        closeout_path = ROOT / R6_CLOSEOUT
        postmortem_path = ROOT / R6_SIGN_POSTMORTEM
        require(
            markdown_contract_status(amendment_path)
            == "AUTHORIZED_BY_STANDING_USER_DIRECTION",
            "R6 Amendment 020 is not authorized",
        )
        require(
            all(
                path.is_file()
                for path in (
                    ROOT / R6_PLAN,
                    policy_path,
                    result_path,
                    closeout_path,
                    postmortem_path,
                )
            ),
            "R6 closeout evidence is missing",
        )
        r6_result = read_json(result_path)
        r6_closeout = read_json(closeout_path)
        metrics = r6_result.get("metrics")
        require(
            r6_result.get("status") == "PASSED_R6_LOCAL_FUNCTIONAL"
            and r6_result.get("passed") is True
            and r6_result.get("independent_h1_validated") is False
            and r6_result.get("external_generalization_claim") is False
            and isinstance(metrics, dict)
            and metrics.get("pair_count") == 1600
            and metrics.get("same_precision") == 1.0
            and metrics.get("same_recall") == 1.0
            and metrics.get("adjacent_as_same") == 0.0
            and metrics.get("accepted_relative_sign_accuracy") == 1.0
            and metrics.get("candidate_recall_at_12") == 1.0
            and metrics.get("topology_ray_agreement_rate") == 1.0
            and metrics.get("physical_swap_consistency") == 1.0
            and metrics.get("pointcollections_roundtrip") == 1.0,
            "R6 local-functional metrics or limitations differ",
        )
        require(
            r6_closeout.get("status") == R6_LOCAL_COMPLETED
            and r6_closeout.get("completion_kind") == R6_LOCAL_COMPLETED
            and r6_closeout.get("validation_scope") == LOCAL_SCOPE
            and r6_closeout.get("artifact_scope")
            == "R6_LOCAL_FUNCTIONAL_ONLY"
            and r6_closeout.get("h0_reused") is False
            and r6_closeout.get("h1_opened") is False
            and r6_closeout.get("independent_h1_validated") is False
            and r6_closeout.get("external_generalization_claim") is False,
            "R6 closeout limitations differ",
        )
        return {
            "status": R6_LOCAL_COMPLETED,
            "eligible": True,
            "validation_scope": LOCAL_SCOPE,
            "activation_state": "R6_LOCAL_FUNCTIONAL_RELEASE",
        }

    if overall == LOCAL_BLOCKED:
        require(
            isinstance(local_config, dict)
            and isinstance(local_config.get("governance"), dict),
            "LOCAL_HOLDOUT_V1 activation config is missing",
        )
        governance = local_config["governance"]
        protected_data = local_config.get("protected_data")
        state_activation = (
            local_state.get("authorization_committed"),
            local_state.get("real_data_execution_enabled"),
        )
        config_activation = (
            governance.get("authorization_committed"),
            governance.get("real_data_execution_enabled"),
        )
        require(
            state.get("eligible") is False
            and results.get("eligible") is False
            and local_state.get("complete") is False
            and local_state.get("eligible") is False,
            "Blocked LOCAL_HOLDOUT_V1 incorrectly permits Phase 3",
        )
        require(
            local_state.get("status") == LOCAL_BLOCKED
            and local_state.get("authorization_received") is True
            and local_state.get("amendment") == LOCAL_AMENDMENT
            and local_state.get("plan") == LOCAL_PLAN
            and local_state.get("amendment_011_activated") is False
            and local_state.get("h0_reused") is False
            and local_state.get("independent_h1_validated") is False
            and local_state.get("external_generalization_claim") is False,
            "Blocked LOCAL_HOLDOUT_V1 governance flags are not fail-closed",
        )
        require(
            local_config.get("status") == LOCAL_BLOCKED
            and isinstance(protected_data, dict)
            and protected_data.get("amendment_011_activated") is False
            and protected_data.get("h0_reused") is False
            and protected_data.get("h0_class_counts_allowed") is False
            and protected_data.get("h0_label_bearing_inputs_allowed") is False
            and protected_data.get("h0_metrics_allowed") is False
            and protected_data.get("h0_predictions_allowed") is False
            and protected_data.get("identifier_only_quarantine_inventory_required")
            is True
            and protected_data.get("independent_holdout_artifacts_allowed") is False
            and protected_data.get("independent_holdout_nonce_created") is False
            and governance.get("required_contract_amendment") == LOCAL_AMENDMENT
            and governance.get("exact_authorization_phrase_source") == LOCAL_AMENDMENT
            and governance.get("base_contract_amendment") == LOCAL_BASE_AMENDMENT
            and governance.get("required_plan") == LOCAL_PLAN
            and governance.get("authorization_received") is True
            and governance.get("amendment_011_activated") is False
            and governance.get("amendment_011_sha256") == AMENDMENT_011_SHA256
            and governance.get("additional_paid_compute_authorized") is False,
            "LOCAL_HOLDOUT_V1 config governance identity differs",
        )
        require(
            all(
                type(value) is bool for value in (*state_activation, *config_activation)
            )
            and state_activation in {(False, False), (True, True)}
            and config_activation == state_activation,
            "LOCAL_HOLDOUT_V1 activation flags are split-brain",
        )
        return {
            "status": "BLOCKED_LOCAL_HOLDOUT_V1_MODEL_FREEZE",
            "eligible": False,
            "validation_scope": None,
            "activation_state": (
                "ACTIVATED" if state_activation == (True, True) else "SAFETY"
            ),
        }

    require(overall == LOCAL_COMPLETED, "Unrecognized Phase 2 terminal state")
    expected_transition = {
        "complete": True,
        "overall": LOCAL_COMPLETED,
        "completion_kind": LOCAL_COMPLETED,
        "eligible": True,
        "validation_scope": LOCAL_SCOPE,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
    }
    for document, label in ((state, "RUN_STATE"), (results, "PHASE2_RESULTS")):
        require(
            all(
                document.get(key) == value for key, value in expected_transition.items()
            ),
            f"{label} local-only transition fields differ",
        )
    require(
        local_state.get("status") == "PASSED_LOCAL_HOLDOUT_V1"
        and local_state.get("validation_protocol") == "LOCAL_HOLDOUT_V1"
        and local_state.get("validation_authority") == "LOCAL_SELF_VALIDATION"
        and local_state.get("independent_h1_validated") is False
        and local_state.get("external_generalization_claim") is False
        and local_state.get("transition") == expected_transition,
        "LOCAL_HOLDOUT_V1 PASS block overstates or changes the transition",
    )
    limitations = local_state.get("limitations")
    require(
        isinstance(limitations, dict)
        and limitations.get("artifact_scope") == LOCAL_ARTIFACT_SCOPE
        and limitations.get("limitation_banner_required") is True
        and limitations.get("public_or_external_deployment_claim_allowed") is False
        and limitations.get("independent_validation_claim_allowed") is False
        and limitations.get("external_generalization_claim_allowed") is False,
        "LOCAL_HOLDOUT_V1 Phase-3 limitations are incomplete",
    )
    closeout_path = (
        ROOT / "phase2/benchmark/local_holdout_v1/LOCAL_HOLDOUT_V1_CLOSEOUT.json"
    )
    require(closeout_path.is_file(), "LOCAL_HOLDOUT_V1 closeout artifact is missing")
    require(
        read_json(closeout_path) == local_state,
        "LOCAL_HOLDOUT_V1 closeout differs from registered state",
    )
    return {
        "status": "COMPLETED_LOCAL_HOLDOUT_V1_ONLY",
        "eligible": True,
        "validation_scope": LOCAL_SCOPE,
    }


def write_receipt_if_changed(path: Path, receipt: dict[str, Any]) -> None:
    """Keep a frozen audit receipt byte-stable while its asserted state is stable."""
    stable = {key: value for key, value in receipt.items() if key != "generated_at_utc"}
    if path.is_file():
        try:
            previous = read_json(path)
            previous_stable = {
                key: value
                for key, value in previous.items()
                if key != "generated_at_utc"
            }
            if previous_stable == stable:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def audit() -> dict[str, Any]:
    phase0 = ROOT / "phase0"
    eligible = read_json(phase0 / "eligible_volumes.json")
    ledger = read_json(phase0 / "target_contamination_ledger.json")
    provenance = read_json(phase0 / "official_page_snapshots" / "provenance.json")
    coordinate = read_json(phase0 / "coordinate_contracts" / "ct_l0_xyz.json")
    entries = eligible.get("entries")
    ledger_entries = ledger.get("entries")
    require(
        eligible.get("schema_version") == 1, "Phase 0 eligible inventory schema changed"
    )
    require(
        isinstance(entries, list) and len(entries) == len(cx.SAMPLES),
        "Phase 0 eligible cohort is not 13 volumes",
    )
    require(
        isinstance(ledger_entries, list) and len(ledger_entries) == len(cx.SAMPLES),
        "Phase 0 contamination ledger is incomplete",
    )
    sample_ids = {entry.get("sample_id") for entry in entries}
    require(
        sample_ids == set(cx.SAMPLES),
        "Phase 0 cohort does not match the frozen sample set",
    )
    ledger_by_sample = {entry.get("sample_id"): entry for entry in ledger_entries}
    for entry in entries:
        sample = entry["sample_id"]
        require(
            entry.get("target_allowed") is True
            and entry.get("training_allowed") is True,
            f"Phase 0 target policy invalid for {sample}",
        )
        require(
            entry.get("shape_zyx") and len(entry["shape_zyx"]) == 3,
            f"Phase 0 Zarr shape missing for {sample}",
        )
        metadata = read_json(phase0 / "volume_metadata" / f"{sample}.json")
        require(
            metadata.get("ct_uri") == entry.get("ct_uri"),
            f"Phase 0 CT URI mismatch for {sample}",
        )
        require(
            metadata.get("stored_array_order") == "zyx",
            f"Phase 0 storage order mismatch for {sample}",
        )
        require(
            metadata.get("consumer_coordinate_order") == "xyz",
            f"Phase 0 coordinate order mismatch for {sample}",
        )
        require(sample in ledger_by_sample, f"Phase 0 ledger missing {sample}")
    ph1203 = ledger_by_sample.get("PHerc1203", {})
    require(
        ph1203.get("access_status") == "FORBIDDEN",
        "PHerc1203 higher-resolution firewall is not active",
    )
    require(
        ph1203.get("higher_resolution_sibling_uri") == cx.FORBIDDEN_1203,
        "PHerc1203 forbidden sibling changed",
    )
    require(
        coordinate.get("contract", "").startswith("All Phase 1 MCP inputs"),
        "Phase 0 coordinate contract missing",
    )
    sources = provenance.get("sources")
    require(
        isinstance(sources, list) and len(sources) == len(cx.SAMPLES) + 1,
        "Phase 0 page provenance is incomplete",
    )
    for source in sources:
        relative = source.get("path")
        require(
            isinstance(relative, str) and "/" not in relative and ".." not in relative,
            "unsafe Phase 0 provenance path",
        )
        snapshot = phase0 / "official_page_snapshots" / relative
        require(snapshot.is_file(), f"Phase 0 snapshot missing: {relative}")
        require(
            sha256_file(snapshot) == source.get("sha256"),
            f"Phase 0 snapshot hash mismatch: {relative}",
        )
    return {
        "status": "PASSED_PHASE0_FREEZE_INTEGRITY",
        "eligible_volume_count": len(entries),
        "source_snapshot_count": len(sources),
    }


def expected_phase1_slots(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for entry in entries:
        for axial_slot in range(8):
            for structural in ("outer", "middle", "inner"):
                slots.append(
                    {
                        "seed_id": f"{entry['sample_id']}-a{axial_slot + 1:02d}-{structural}",
                        "sample_id": entry["sample_id"],
                        "axial_stratum": axial_slot + 1,
                        "prediction_uri": entry["surface_prediction_uri"],
                        "voxel_size_um": entry["voxel_size_um"],
                        "candidate_region": cx.planned_region(
                            entry["shape_zyx"], axial_slot, structural
                        ),
                    }
                )
    return slots


def audit() -> dict[str, Any]:
    phase1 = ROOT / "phase1"
    plan = read_json(phase1 / "seed_plan.json")
    closeout = read_json(phase1 / "PHASE1_AUTOMATED_CLOSEOUT.json")
    targets = read_json(phase1 / "AUTOMATED_PROVISIONAL_TARGETS.json")
    dashboard = read_json(
        phase1 / "qc/live-baseline-sparse-fixed-20260715/manifest.json"
    )
    observations = read_json(
        phase1
        / "qc/live-baseline-sparse-fixed-20260715/llm_visual_triage/observations.json"
    )
    prefill = read_json(
        phase1
        / "qc/live-baseline-sparse-fixed-20260715/model_full_qc/model_prefill.json"
    )
    eligible = read_json(ROOT / "phase0/eligible_volumes.json")["entries"]
    slots = plan.get("slots")
    require(
        isinstance(slots, list) and len(slots) == 312,
        "Phase 1 has not retained 312 frozen slots",
    )
    require(
        plan.get("seed_count") == 312
        and len({slot.get("seed_id") for slot in slots}) == 312,
        "Phase 1 slot IDs are invalid",
    )
    require(
        slots == expected_phase1_slots(eligible),
        "Phase 1 frozen plan differs from the Phase 0-derived deterministic plan",
    )
    require(
        closeout.get("status") == "CLOSED_AT_AUTOMATED_SCREENING_GATE",
        "Phase 1 automated closeout status changed",
    )
    execution = closeout.get("execution", {})
    screening = closeout.get("model_screening", {})
    require(
        execution.get("frozen_slots") == 312, "Phase 1 closeout slot count mismatch"
    )
    require(
        execution.get("successful_ct_candidates") == 226,
        "Phase 1 success count mismatch",
    )
    require(
        execution.get("genuine_no_candidate_slots") == 86,
        "Phase 1 NO_CANDIDATE count mismatch",
    )
    require(
        execution.get("all_successes_have_3x3_neighbor_ct") is True,
        "Phase 1 3x3 CT coverage invariant failed",
    )
    require(
        execution["successful_ct_candidates"] + execution["genuine_no_candidate_slots"]
        == execution["frozen_slots"],
        "Phase 1 terminal counts do not close",
    )
    require(
        len(observations.get("assessments", []))
        == screening.get("visual_triage_assessments")
        == 226,
        "Phase 1 visual triage count mismatch",
    )
    require(
        len(prefill.get("assessments", []))
        == screening.get("multiview_prefill_assessments")
        == 174,
        "Phase 1 multiview pre-QC count mismatch",
    )
    candidates = targets.get("candidates", [])
    candidate_ids = [candidate.get("seed_id") for candidate in candidates]
    require(
        targets.get("status") == "AUTOMATED_SCREENING_SHORTLIST",
        "Phase 1 shortlist status changed",
    )
    require(
        targets.get("candidate_count")
        == screening.get("automated_provisional_target_count")
        == len(candidates)
        == 78,
        "Phase 1 shortlist count mismatch",
    )
    require(len(set(candidate_ids)) == 78, "Phase 1 shortlist contains duplicate seeds")
    require(
        set(candidate_ids).issubset(
            {assessment.get("seed_id") for assessment in prefill.get("assessments", [])}
        ),
        "Phase 1 shortlist contains a seed without multiview pre-QC",
    )
    require(
        dashboard.get("kind") == "campaign_x_phase1_qc_dashboard_v1",
        "Phase 1 QC dashboard manifest changed",
    )
    return {
        "status": "PASSED_PHASE1_AUTOMATED_CLOSEOUT_INTEGRITY",
        "frozen_slots": 312,
        "successful_ct_candidates": 226,
        "provisional_targets": 78,
    }


def audit() -> dict[str, Any]:
    phase2 = ROOT / "phase2"
    amendment_011 = ROOT / AMENDMENT_011_PATH
    require(
        amendment_011.is_file()
        and sha256_file(amendment_011) == AMENDMENT_011_SHA256
        and markdown_contract_status(amendment_011) == AMENDMENT_011_STATUS,
        "Amendment 011 bytes or status changed",
    )
    results = read_json(phase2 / "PHASE2_RESULTS.json")
    state = read_json(phase2 / "RUN_STATE.json")
    relation = read_json(phase2 / "RELATION_V2_RESULTS.json")
    local_config = read_json(phase2 / "configs/relation_v2_local_holdout_v1.json")
    local_v2_config = read_json(phase2 / "configs/relation_v2_local_holdout_v2.json")
    recovery = phase2 / "benchmark/paris4_relation_v2_r1_recovery"
    r2_recovery = phase2 / "benchmark/paris4_relation_v2_r2_adjacency_recovery"
    coverage = read_json(recovery / "GEOMETRY_COVERAGE_AUDIT.json")
    r1 = read_json(recovery / "model/R1_NESTED_CV_RESULTS_V2.json")
    calibration = read_json(recovery / "model/CALIBRATION_RESULTS_R1_V2.json")
    r2_path = r2_recovery / "model/R2_NESTED_CV_RESULTS_V1.json"
    r2 = read_json(r2_path)
    r5_root = phase2 / "benchmark/paris4_relation_v2_r5_conservative"
    r5_audit_path = r5_root / "R5_DEVELOPMENT_GATE_AUDIT.json"
    r5_freeze_path = r5_root / "R5_MODEL_FREEZE.json"
    r5_audit = read_json(r5_audit_path)
    r5_freeze = read_json(r5_freeze_path)
    quarantine = phase2 / "benchmark/paris4_relation_v2/H0_QUARANTINE.json"
    local_transition = verify_local_holdout_transition(
        results, state, relation, local_config, local_v2_config
    )
    m9 = state.get("milestones", {}).get("M9_RELATION_V2", {})
    require(
        m9.get("r1_status")
        == r1.get("status")
        == calibration.get("status")
        == "NO_SAFE_R1_SELECTION",
        "Phase 2 R1 terminal status mismatch",
    )
    require(
        r1.get("terminal_state") == "BLOCKED_RELATION_V2_DEVELOPMENT",
        "Phase 2 R1 terminal state mismatch",
    )
    require(
        r1.get("selected_arm") is None and m9.get("r1_model_frozen") is False,
        "Phase 2 incorrectly records a frozen R1 model",
    )
    require(
        r1.get("holdout_opened") is False
        and m9.get("h1_opened") is False
        and relation.get("h1_opened") is False,
        "Phase 2 H1 guard failed",
    )
    require(
        m9.get("h0_reused_for_recovery") is False
        and m9.get("h0_quarantine_enforced") is True,
        "Phase 2 H0 quarantine guard failed",
    )
    require(
        coverage.get("status") == "PASSED_GEOMETRY_COVERAGE_PREFERRED",
        "Phase 2 recovery coverage did not pass",
    )
    require(
        coverage.get("geometry_ready_pair_count") == 5994
        and coverage.get("geometry_ready_pair_rate") == 0.9981681931723564,
        "Phase 2 recovery coverage count mismatch",
    )
    require(
        r1.get("geometry_coverage", {}).get("audit_sha256")
        == sha256_file(recovery / "GEOMETRY_COVERAGE_AUDIT.json"),
        "Phase 2 coverage audit hash mismatch",
    )
    require(
        r1.get("geometry_provenance", {}).get("sealed_h0_quarantine_sha256")
        == sha256_file(quarantine),
        "Phase 2 H0 quarantine hash mismatch",
    )
    verify_hash_map(r1.get("source_sha256"), "source")
    verify_hash_map(r1.get("implementation_sha256"), "implementation")
    require(
        r1.get("passed_seed_count") == 0
        and r1.get("fold_seed_passes") == {"0": 0, "1": 0, "2": 0, "3": 0},
        "Phase 2 R1 seed gate mismatch",
    )

    # R2 is separately pre-registered and may only consume the sealed R0/R1
    # development artifacts.  Its result stores semantic source names rather
    # than paths, so resolve every one here instead of accepting an unchecked
    # digest map.
    expected_r2_sources = {
        "development_edges": phase2
        / "benchmark/paris4_relation_v2/DEVELOPMENT_EDGES.jsonl",
        "h0_quarantine": phase2 / "benchmark/paris4_relation_v2/H0_QUARANTINE.json",
        "r0_features": phase2
        / "benchmark/paris4_relation_v2/development/R0_FEATURES.npz",
        "r0_results": phase2 / "benchmark/paris4_relation_v2/R0_NESTED_CV_RESULTS.json",
        "r1_coverage": recovery / "GEOMETRY_COVERAGE_AUDIT.json",
        "r1_features": recovery / "development/R1_FEATURES.npz",
        "r1_manifest": recovery / "development/R1_FEATURE_MANIFEST.json",
        "r1_source_lock": recovery / "GEOMETRY_SOURCE_LOCK.json",
        "r1_transform_lock": recovery / "GEOMETRY_TRANSFORM_LOCK.json",
    }
    r2_sources = r2.get("source_sha256")
    require(
        isinstance(r2_sources, dict) and set(r2_sources) == set(expected_r2_sources),
        "Phase 2 R2 source lock keys mismatch",
    )
    for name, source in expected_r2_sources.items():
        require(
            source.is_file(), f"Phase 2 R2 source missing: {source.relative_to(ROOT)}"
        )
        require(
            r2_sources.get(name) == sha256_file(source),
            f"Phase 2 R2 source hash mismatch: {name}",
        )
    verify_hash_map(r2.get("implementation_sha256"), "R2 implementation")
    require(
        m9.get("r2_status")
        == r2.get("status")
        == relation.get("r2", {}).get("status")
        == "NO_SAFE_R2_SELECTION",
        "Phase 2 R2 terminal status mismatch",
    )
    require(
        r2.get("terminal_state") == "BLOCKED_RELATION_V2_DEVELOPMENT",
        "Phase 2 R2 terminal state mismatch",
    )
    require(
        r2.get("selected_arm") is None
        and m9.get("r2_model_frozen") is False
        and relation.get("r2", {}).get("model_frozen") is False,
        "Phase 2 incorrectly records a frozen R2 model",
    )
    require(
        r2.get("h0_reused") is False
        and r2.get("h1_opened") is False
        and r2.get("holdout_opened") is False,
        "Phase 2 R2 holdout guard failed",
    )
    require(
        r2.get("geometry_ready_pair_count") == 5994,
        "Phase 2 R2 geometry-ready count mismatch",
    )
    expected_arms = {"R2_LINEAR_ADJACENCY", "R2_HGB_ADJACENCY"}
    arms = r2.get("arms")
    require(
        isinstance(arms, list) and {arm.get("arm") for arm in arms} == expected_arms,
        "Phase 2 R2 arm set mismatch",
    )
    for arm in arms:
        require(
            arm.get("status") == "NO_SAFE_R2_SELECTION" and arm.get("passed") is False,
            f"Phase 2 R2 arm state mismatch: {arm.get('arm')}",
        )
        require(
            arm.get("passed_seed_count") == 0
            and arm.get("fold_seed_passes") == {"0": 0, "1": 0, "2": 0, "3": 0},
            f"Phase 2 R2 seed gate mismatch: {arm.get('arm')}",
        )
    require(
        relation.get("r2", {}).get("result_evidence") == str(r2_path.relative_to(ROOT)),
        "Phase 2 R2 evidence path mismatch",
    )
    require(
        relation.get("r2", {}).get("result_sha256") == sha256_file(r2_path),
        "Phase 2 R2 result hash mismatch",
    )
    r5_state = state.get("relation_v2_r5", {})
    r5_summary = relation.get("r5", {})
    require(
        r5_audit.get("kind")
        == "campaign_x_phase2_relation_v2_r5_development_gate_audit_v1",
        "Phase 2 R5 audit kind changed",
    )
    require(
        r5_audit.get("status") == "PASSED_R5_DEVELOPMENT"
        and r5_audit.get("passed") is True,
        "Phase 2 R5 development did not pass",
    )
    require(
        r5_audit.get("fold_seed_passes") == {"0": 3, "1": 3, "2": 3}
        and r5_audit.get("passed_seed_count") == 3,
        "Phase 2 R5 outer/seed gate mismatch",
    )
    require(
        r5_audit.get("consensus", {}).get("passed") is True
        and r5_audit.get("consensus", {}).get("monotonic_subset") is True,
        "Phase 2 R5 consensus gate mismatch",
    )
    outer = r5_audit.get("outer_cells")
    require(
        isinstance(outer, list) and len(outer) == 9,
        "Phase 2 R5 outer inventory changed",
    )
    require(
        all(
            cell.get("monotonic_subset") is True
            and cell.get("report", {}).get("passed") is True
            and cell.get("report", {}).get("point", {}).get("adjacent_accepted") == 0
            for cell in outer
        ),
        "Phase 2 R5 outer monotonic/exact gate mismatch",
    )
    require(
        r5_audit.get("model_bytes_reused") is True
        and r5_audit.get("new_joblib_count") == 0,
        "Phase 2 R5 changed model bytes",
    )
    require(
        r5_audit.get("protected_holdouts") == {"h0_reused": False, "h1_opened": False},
        "Phase 2 R5 holdout guard failed",
    )
    require(
        r5_freeze.get("kind")
        == "campaign_x_phase2_relation_v2_r5_wrapper_model_freeze_v1",
        "Phase 2 R5 freeze kind changed",
    )
    require(
        r5_freeze.get("status") == "FROZEN_R5_WRAPPER_READY_FOR_H1_PREFLIGHT"
        and r5_freeze.get("complete") is False,
        "Phase 2 R5 freeze status changed",
    )
    require(
        r5_freeze.get("model_bytes_reused") is True
        and r5_freeze.get("new_joblib_count") == 0,
        "Phase 2 R5 freeze changed model bytes",
    )
    require(
        r5_freeze.get("protected_holdouts") == {"h0_reused": False, "h1_opened": False},
        "Phase 2 R5 freeze holdout guard failed",
    )
    models = r5_freeze.get("models")
    require(
        isinstance(models, dict) and len(models) == 3,
        "Phase 2 R5 model inventory changed",
    )
    for model_path, digest in models.items():
        require(
            sha256_file(ROOT / model_path) == digest,
            f"Phase 2 R5 model hash mismatch: {model_path}",
        )
    require(
        r5_state.get("development_audit_sha256")
        == r5_summary.get("development_audit_sha256")
        == sha256_file(r5_audit_path),
        "Phase 2 R5 audit hash linkage failed",
    )
    require(
        r5_state.get("model_freeze_sha256")
        == r5_summary.get("model_freeze_sha256")
        == sha256_file(r5_freeze_path),
        "Phase 2 R5 freeze hash linkage failed",
    )
    require(
        state.get("relation_v2_h1", {}).get("h1_opened") is False
        and state.get("relation_v2_h1", {}).get("h1_v2_contract_status")
        == "AUTHORIZED_BY_USER",
        "Phase 2 H1-v2 authorization/holdout state is unsafe",
    )
    cost_events = state.get("relation_v2_geometry_recovery_cost_events", [])
    require(
        cost_events
        and cost_events[-1].get("planned_phase") == "R5_MODEL_FREEZE"
        and cost_events[-1].get("stage") == "PREFLIGHT"
        and cost_events[-1].get("status") == "PASSED_COST_GATE",
        "Phase 2 final cost gate missing",
    )
    return {
        "status": (
            "PASSED_PHASE2_R6_LOCAL_FUNCTIONAL_INTEGRITY"
            if local_transition["status"] == R6_LOCAL_COMPLETED
            else (
                "PASSED_PHASE2_LOCAL_HOLDOUT_V1_ONLY_INTEGRITY"
                if local_transition["eligible"]
                else (
                    "PASSED_PHASE2_R5_FREEZE_INTEGRITY_PENDING_LOCAL_HOLDOUT_V2"
                    if str(local_transition["status"]).startswith(
                        "BLOCKED_LOCAL_HOLDOUT_V2_"
                    )
                    else "PASSED_PHASE2_R5_FREEZE_INTEGRITY_PENDING_LOCAL_HOLDOUT_V1"
                )
            )
        ),
        "terminal_state": local_transition["status"],
        "geometry_ready_pairs": coverage["geometry_ready_pair_count"],
        "eligible": local_transition["eligible"],
        "validation_scope": local_transition["validation_scope"],
    }


def build_audit() -> dict[str, Any]:
    phase0 = audit()
    phase1 = audit()
    phase2 = audit()
    allowed = phase2["eligible"] is True
    return {
        "kind": "campaign_x_phase0_phase1_phase2_closure_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "status": (
            "PHASE0_PHASE1_PHASE2_R6_LOCAL_FUNCTIONAL_INTEGRITY_PASSED"
            if phase2["terminal_state"] == R6_LOCAL_COMPLETED
            else (
                "PHASE0_PHASE1_PHASE2_LOCAL_HOLDOUT_V1_ONLY_INTEGRITY_PASSED"
                if allowed
                else (
                    "PHASE0_PHASE1_INTEGRITY_PASSED_PHASE2_PENDING_LOCAL_HOLDOUT_V2"
                    if str(phase2["terminal_state"]).startswith(
                        "BLOCKED_LOCAL_HOLDOUT_V2_"
                    )
                    else "PHASE0_PHASE1_INTEGRITY_PASSED_PHASE2_PENDING_LOCAL_HOLDOUT_V1"
                )
            )
        ),
        "allowed": allowed,
        "phase0": phase0,
        "phase1": phase1,
        "phase2": phase2,
        "next_permitted_action": (
            "Continue to Phase 3 under LOCAL_PIPELINE_CONTINUATION_ONLY; every downstream artifact must preserve the R6 local-functional limitation and external generalization claims remain prohibited."
            if phase2["terminal_state"] == R6_LOCAL_COMPLETED
            else (
                "Request explicit Phase-3 authorization; every downstream artifact must remain LOCAL_HOLDOUT_V1_ONLY and external generalization claims remain prohibited."
                if allowed
                else (
                    "Finish LOCAL_HOLDOUT_V2 preclaim recovery and execute R5 exactly once only after every gate passes; Phase 3 remains blocked."
                    if str(phase2["terminal_state"]).startswith(
                        "BLOCKED_LOCAL_HOLDOUT_V2_"
                    )
                    else "Finish the clean LOCAL_HOLDOUT_V1 safety and activation commits, then execute the frozen local protocol exactly once; Phase 3 remains blocked."
                )
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "phase2/PHASE2_CLOSURE_AUDIT.json"
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    try:
        audit = build_audit()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "status": "FAILED_CLOSURE_INTEGRITY",
                    "reason": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    if not args.no_write:
        write_receipt_if_changed(args.output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
