"""Regression checks for the three Stage 01 fleet backlog closeouts.

* P0 — the deterministic/LLM planner lane comparison harness.
* P4 — the recipe yield table that closes the learning loop's missing input.
* A5 — the effort-allocation policy and per-roll budget review.

The hard constraint tested here, and not only documented, is that an A5
recommendation is about where to spend the next GPU slot and never a statement
about what a roll contains.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
SCRIPTS = STAGE / "scripts"
for _path in (str(SCRIPTS), str(STAGE), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import helena_build_recipe_yield as recipe_yield  # noqa: E402
import helena_compare_planner_lanes as lane_comparison  # noqa: E402
import helena_review_campaign_budget as budget_review  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.planner import task_packet_for_planner  # noqa: E402

from evidence import needs_campaign_evidence


POLICY_PATH = (
    ROOT
    / "framework/profiles/01-segmentation/segmentation-effort-allocation-policy-v1-1.0.0.json"
)

ENVELOPE = {
    "ink_used": False,
    "maximum_candidate_count": 8,
    "parameters": {
        "generations": {"type": "integer", "default": 35, "minimum": 20, "maximum": 45},
        "min_area_cm": {"type": "number", "default": 0.0, "minimum": 0.0, "maximum": 0.0},
        "step_size": {"type": "integer", "default": 20, "minimum": 12, "maximum": 24},
        "use_cuda": {"type": "boolean", "default": False, "const": False},
    },
    "profile_ids": ["vc3d-m7-growth-v1"],
}

DEFAULT_PARAMETERS = {
    "generations": 35,
    "min_area_cm": 0.0,
    "step_size": 20,
    "use_cuda": False,
}


def seed(index: int, score: float, clearance: float) -> dict:
    return {
        "candidate_id": f"c{index:02d}",
        "x": 100 + index,
        "y": 200 + index,
        "z": 300 + index,
        "score": score,
        "cell_interior_clearance_voxels": clearance,
        "volume_interior_clearance_voxels": clearance * 2,
    }


def packet(
    *,
    policy: str | None = "score-cell-volume-clearance-v1",
    seeds: list[dict] | None = None,
) -> dict:
    candidates = seeds if seeds is not None else [seed(1, 1.0, 40.0), seed(2, 1.0, 10.0)]
    task = {
        "task_id": "task-0001",
        "attempt_id": "attempt-0001",
        "sample_id": "PHercTEST",
        "cell_id": "cell-a",
        "bounds_xyz": [[0, 0, 0], [1024, 1024, 1024]],
        "center_xyz": {"x": 512, "y": 512, "z": 512},
        "catalog_snapshot_sha256": "0" * 64,
        "parameter_envelope": ENVELOPE,
        "source": {
            "source_snapshot_id": "snapshot-0001",
            "ct_uri": "fixture://ct",
            "m7_uri": "fixture://m7",
            "shape_xyz": [1024, 1024, 1024],
            "voxel_size_um": 9.362,
            "coordinate_frame": "ct_l0_xyz",
        },
    }
    if policy is not None:
        task["candidate_selection_policy"] = policy
    return task_packet_for_planner(task, candidates, None, contract_version="v1")


def proposal_for(packet_value: dict, candidate: dict, parameters: dict) -> dict:
    return {
        "schema": "campaignx.segmentation_proposal.v1",
        "task_id": packet_value["task_id"],
        "attempt_id": packet_value["attempt_id"],
        "selected_seed": {key: candidate[key] for key in ("candidate_id", "x", "y", "z")},
        "profile_id": "vc3d-m7-growth-v1",
        "parameters": dict(parameters),
        "hypothesis": "fixture",
        "alternatives_rejected": [
            {"candidate_id": row["candidate_id"], "reason": "fixture"}
            for row in packet_value["candidate_seeds"]
            if row["candidate_id"] != candidate["candidate_id"]
        ],
        "ink_used": False,
    }


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_dir(
    root: Path,
    name: str,
    packet_value: dict,
    proposal: dict | None,
    *,
    lane_marker: str | None = "opencode.stdout.log",
) -> Path:
    directory = root / name
    write(directory / "PLANNER_PACKET.json", packet_value)
    if proposal is not None:
        write(directory / "SEGMENTATION_PROPOSAL.json", proposal)
    if lane_marker is not None:
        (directory / lane_marker).parent.mkdir(parents=True, exist_ok=True)
        (directory / lane_marker).write_text("fixture", encoding="utf-8")
    return directory


# --------------------------------------------------------------------------
# P0 — deterministic versus LLM planner lane
# --------------------------------------------------------------------------


def test_lane_comparison_counts_agreement_and_disagreement(tmp_path: Path) -> None:
    legacy = packet(policy=None)
    top, other = legacy["candidate_seeds"]
    run_dir(tmp_path, "agree", legacy, proposal_for(legacy, top, DEFAULT_PARAMETERS))
    run_dir(
        tmp_path,
        "different-seed",
        legacy,
        proposal_for(legacy, other, DEFAULT_PARAMETERS),
    )
    run_dir(
        tmp_path,
        "different-parameters",
        legacy,
        proposal_for(legacy, top, {**DEFAULT_PARAMETERS, "generations": 41}),
    )

    report = lane_comparison.build_report([tmp_path], tmp_path)
    totals = report["totals"]
    assert totals["packet_count"] == 3
    assert totals["comparable_packet_count"] == 3
    assert totals["identical_seed_and_parameters_count"] == 1
    assert totals["identical_seed_count"] == 2
    assert totals["identical_profile_and_parameters_count"] == 2
    rows = {Path(row["packet_path"]).parent.name: row for row in report["rows"]}
    assert rows["different-seed"]["same_seed"] is False
    assert rows["different-seed"]["same_parameters"] is True
    assert rows["different-parameters"]["same_parameters"] is False
    assert rows["agree"]["same_decision"] is True


def test_lane_comparison_excludes_runs_without_provider_evidence(tmp_path: Path) -> None:
    legacy = packet(policy=None)
    top = legacy["candidate_seeds"][0]
    run_dir(
        tmp_path,
        "unattributed",
        legacy,
        proposal_for(legacy, top, DEFAULT_PARAMETERS),
        lane_marker=None,
    )
    run_dir(
        tmp_path,
        "fallback",
        legacy,
        proposal_for(legacy, top, DEFAULT_PARAMETERS),
        lane_marker="deterministic-fallback/DETERMINISTIC_FALLBACK_RECEIPT.json",
    )
    report = lane_comparison.build_report([tmp_path], tmp_path)
    assert report["totals"]["comparable_packet_count"] == 0
    assert report["totals"]["excluded_packet_count"] == 2
    lanes = {row["model_lane"]["lane"] for row in report["rows"]}
    assert lanes == {"UNATTRIBUTED", "DETERMINISTIC_FALLBACK"}
    # A proposal identical to the deterministic one must not be counted as
    # agreement when nothing proves a model produced it.
    assert all(row["same_decision"] is None for row in report["rows"])


def test_lane_comparison_reports_how_much_freedom_the_validator_left(tmp_path: Path) -> None:
    enforced = packet(policy="score-cell-volume-clearance-v1")
    top = enforced["candidate_seeds"][0]
    run_dir(tmp_path, "enforced", enforced, proposal_for(enforced, top, DEFAULT_PARAMETERS))
    report = lane_comparison.build_report([tmp_path], tmp_path)
    space = report["rows"][0]["decision_space"]
    # Eight listed candidates, exactly one legal choice.
    assert space["validator_allowed_seed_count"] == 1
    assert space["seed_choice_is_free"] is False
    assert space["seed_forced_by"] == "CANDIDATE_SELECTION_POLICY"
    # generations 20..45 (26) x step_size 12..24 (13) x one const x one degenerate
    # number x one profile.
    assert space["parameter_and_profile_tuple_count"] == 338


def test_lane_comparison_skips_the_opencode_sandbox_duplicate(tmp_path: Path) -> None:
    legacy = packet(policy=None)
    top = legacy["candidate_seeds"][0]
    directory = run_dir(tmp_path, "run", legacy, proposal_for(legacy, top, DEFAULT_PARAMETERS))
    write(directory / "opencode-planner-sandbox" / "PLANNER_PACKET.json", legacy)
    report = lane_comparison.build_report([tmp_path], tmp_path)
    assert report["totals"]["packet_count"] == 1


def test_lane_comparison_declares_the_gpu_hour_question_out_of_scope(tmp_path: Path) -> None:
    legacy = packet(policy=None)
    top = legacy["candidate_seeds"][0]
    run_dir(tmp_path, "run", legacy, proposal_for(legacy, top, DEFAULT_PARAMETERS))
    report = lane_comparison.build_report([tmp_path], tmp_path)
    assert any("GPU-hour" in item for item in report["out_of_scope"])
    assert report["totals"]["deterministic_provider_call_count"] == 0
    assert report["ink_used"] is False
    assert any("validator" in claim for claim in report["non_claims"])


@needs_campaign_evidence
def test_lane_comparison_over_the_real_packet_corpus_is_internally_consistent() -> None:
    report = lane_comparison.build_report([ROOT / "workspace"], ROOT)
    totals = report["totals"]
    assert totals["packet_count"] >= 1
    assert totals["comparable_packet_count"] <= totals["packet_count"]
    assert (
        totals["identical_seed_and_parameters_count"]
        <= totals["identical_profile_and_parameters_count"]
    )
    assert (
        totals["identical_seed_where_choice_was_free_count"]
        <= totals["free_seed_choice_packet_count"]
    )
    for row in report["rows"]:
        if row["comparable"]:
            assert row["model_lane"]["lane_evidence"]
            assert row["deterministic"]["provider_calls"] == 0


# --------------------------------------------------------------------------
# P4 — recipe yield table
# --------------------------------------------------------------------------


def archive_receipt(path: Path, *, scope: str, area: float, voxel: float = 9.362) -> None:
    write(
        path,
        {
            "kind": recipe_yield.ARCHIVE_RECEIPT_KIND,
            "status": "PASSED",
            "area_cm2": area,
            "plan_query_scope": scope,
            "profile": {
                "generations": 35,
                "step_size": 20,
                "voxelsize": voxel,
                "min_area_cm": 0.0,
                "use_cuda": False,
            },
            "archived_surface": {"path": "library/tifxyz/PHercTEST/PHercTEST-g01"},
        },
    )


def fleet_receipt(directory: Path, *, area: float, policy: str | None) -> None:
    write(
        directory / "GROWTH_RECEIPT.json",
        {
            "schema": "campaignx.segment_fleet_growth_receipt.v1",
            "exit_code": 0,
            "profile": {
                "generations": 35,
                "step_size": 20,
                "voxelsize": 9.362,
                "min_area_cm": 0.0,
                "use_cuda": False,
            },
        },
    )
    write(directory / "ARTIFACT_SET.json", {"area_cm2": area})
    discovery: dict = {"provider": "vc3d-mcp"}
    if policy is not None:
        discovery["seed_region_policy"] = policy
    write(
        directory / "CLAIMED_TASK.json",
        {"sample_id": "PHercTEST", "candidate_discovery": discovery},
    )


def test_recipe_yield_groups_by_the_requested_key(tmp_path: Path) -> None:
    for index in range(6):
        archive_receipt(
            tmp_path / f"a{index}" / "GROWTH_RECEIPT.json",
            scope="ONE_FIXED_INTERLEAVED_REGION_PER_SCROLL",
            area=1.58 + index * 0.001,
        )
    for index in range(6):
        archive_receipt(
            tmp_path / f"b{index}" / "GROWTH_RECEIPT.json",
            scope="FIXED_PREDECLARED_STRATA_PER_SCROLL",
            area=1.58 + index * 0.001,
        )
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    keys = {group["recipe_key"] for group in report["groups"]}
    assert keys == {
        "ONE_FIXED_INTERLEAVED_REGION_PER_SCROLL|generations=35|step_size=20",
        "FIXED_PREDECLARED_STRATA_PER_SCROLL|generations=35|step_size=20",
    }
    assert report["grouping_key"] == ["seed_region_policy", "generations", "step_size"]
    assert report["totals"]["attempt_count"] == 12


def test_recipe_yield_uses_the_worker_default_seed_region_policy(tmp_path: Path) -> None:
    fleet_receipt(tmp_path / "declared", area=10.0, policy="m7-recenter-z-v1")
    fleet_receipt(tmp_path / "undeclared", area=11.0, policy=None)
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    policies = {row["seed_region_policy"] for row in report["attempts"]}
    assert policies == {"m7-recenter-z-v1", recipe_yield.FLEET_DEFAULT_SEED_REGION_POLICY}
    sources = {row["seed_region_policy_source"] for row in report["attempts"]}
    assert sources == {"CLAIMED_TASK", "WORKER_DEFAULT"}
    assert all(row["area_source"] == "ARTIFACT_SET.json" for row in report["attempts"])


def test_recipe_yield_reports_insufficient_signal_with_numbers(tmp_path: Path) -> None:
    for index in range(6):
        archive_receipt(
            tmp_path / f"a{index}" / "GROWTH_RECEIPT.json",
            scope="POLICY_A",
            area=1.50 + (index % 2) * 0.20,
        )
    for index in range(6):
        archive_receipt(
            tmp_path / f"b{index}" / "GROWTH_RECEIPT.json",
            scope="POLICY_B",
            area=1.51 + (index % 2) * 0.20,
        )
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    assessment = report["assessment"]
    assert assessment["verdict"] == "INSUFFICIENT_TO_DISCRIMINATE"
    assert assessment["dominant_recipe"] is None
    assert assessment["between_recipe_mean_spread_cm2"] == pytest.approx(0.01, abs=1e-9)
    assert assessment["pooled_within_recipe_stdev_cm2"] > 0.01
    assert "cm2" in assessment["reason"]


def test_recipe_yield_never_promotes_separation_to_dominance(tmp_path: Path) -> None:
    for index in range(6):
        archive_receipt(
            tmp_path / f"a{index}" / "GROWTH_RECEIPT.json", scope="POLICY_A", area=1.0
        )
    for index in range(6):
        archive_receipt(
            tmp_path / f"b{index}" / "GROWTH_RECEIPT.json", scope="POLICY_B", area=9.0
        )
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    assessment = report["assessment"]
    assert assessment["verdict"] == "SEPARATION_OBSERVED_NOT_CAUSAL"
    assert assessment["dominant_recipe"] is None
    assert assessment["highest_mean_recipe_key"].startswith("POLICY_B")
    within = assessment["within_execution_harness_comparisons"]
    assert all(row["dominant_recipe"] is None for row in within)


def test_recipe_yield_separates_recipe_effects_from_the_execution_harness(
    tmp_path: Path,
) -> None:
    for index in range(6):
        archive_receipt(
            tmp_path / f"a{index}" / "GROWTH_RECEIPT.json", scope="POLICY_A", area=1.0
        )
    for index in range(6):
        fleet_receipt(tmp_path / f"f{index}", area=20.0, policy="fixed-v1")
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    assessment = report["assessment"]
    assert assessment["seed_region_policy_confounded_with_execution_harness"] is True
    harnesses = {
        row["execution_harness"] for row in assessment["within_execution_harness_comparisons"]
    }
    assert harnesses == {"GEOMETRY_RECOVERY_ARCHIVE", "SEGMENT_FLEET"}
    assert all(
        row["verdict"] == "SINGLE_COMPARABLE_RECIPE"
        for row in assessment["within_execution_harness_comparisons"]
    )


def test_recipe_yield_does_not_change_the_planner_v2_contract(tmp_path: Path) -> None:
    for index in range(6):
        archive_receipt(
            tmp_path / f"a{index}" / "GROWTH_RECEIPT.json", scope="POLICY_A", area=1.0
        )
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    assert report["consumed_by_planner_v2"] is False
    history = {
        "schema": "campaignx.segmentation_regional_attempt_history.v1",
        "ink_used": False,
        "attempts": [],
    }
    history["history_sha256"] = content_sha256(
        {key: value for key, value in history.items()}
    )
    built = task_packet_for_planner(
        {
            "task_id": "t",
            "attempt_id": "a",
            "sample_id": "PHercTEST",
            "cell_id": "cell",
            "bounds_xyz": [[0, 0, 0], [10, 10, 10]],
            "center_xyz": {"x": 5, "y": 5, "z": 5},
            "catalog_snapshot_sha256": "0" * 64,
            "parameter_envelope": ENVELOPE,
            "candidate_selection_policy": "adaptive-geometry-history-v2",
                "source": {
                    "source_snapshot_id": "s",
                    "ct_uri": "fixture://ct",
                    "m7_uri": "fixture://m7",
                "shape_xyz": [10, 10, 10],
                "voxel_size_um": 9.362,
            },
        },
        [seed(1, 1.0, 1.0)],
        None,
        contract_version="v2",
        regional_attempt_history=history,
    )
    rendered = json.dumps(built)
    assert "recipe_yield" not in rendered
    assert "area_cm2" not in rendered


def test_recipe_yield_declares_its_non_claims(tmp_path: Path) -> None:
    archive_receipt(tmp_path / "a" / "GROWTH_RECEIPT.json", scope="POLICY_A", area=1.0)
    report = recipe_yield.build_report([tmp_path], tmp_path, tmp_path / "absent.sqlite")
    joined = " ".join(report["non_claims"])
    assert "association" in joined
    assert "never a measured zero" in joined
    assert report["ink_used"] is False
    assert any("planner v2 packet contract" in item for item in report["out_of_scope"])


# --------------------------------------------------------------------------
# A5 — effort-allocation policy and budget review
# --------------------------------------------------------------------------


def test_effort_allocation_policy_thresholds_are_frozen() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["profile_id"] == "segmentation-effort-allocation-policy-v1@1.0.0"
    assert policy["ink_blind"] is True
    thresholds = policy["thresholds"]
    assert thresholds["minimum_attempts_before_any_pause"] == 5
    assert thresholds["minimum_guard_relation_count"] == 30
    assert thresholds["needs_new_approach_maximum_guard_pass_rate"] == 0.4
    assert thresholds["continue_minimum_guard_pass_rate"] == 0.6
    assert thresholds["minimum_unique_area_per_attempt_cm2"] == 0.1
    assert (
        thresholds["needs_new_approach_maximum_guard_pass_rate"]
        < thresholds["continue_minimum_guard_pass_rate"]
    )
    for forbidden in policy["forbidden_inputs"]:
        assert "ink" in forbidden or "text" in forbidden or "glyph" in forbidden or "human" in forbidden


def guard(passed: int, failed: int) -> dict:
    total = passed + failed
    return {
        "evaluation_path": "fixture",
        "generated_at_utc": "2026-07-25T00:00:00Z",
        "evaluation_status": "FAILED",
        "scope": "PRIVATE_LOCAL_FUNCTIONAL_FIELD_TEST_ONLY",
        "guard_passed_count": passed,
        "guard_failed_count": failed,
        "guard_relation_count": total,
        "guard_pass_rate": passed / total if total else None,
    }


@pytest.fixture(name="policy")
def policy_fixture() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def test_recommendation_covers_all_three_outcomes(policy: dict) -> None:
    assert budget_review.recommend(
        policy, attempt_count=40, guard=guard(90, 10), area_per_attempt=1.0
    )["recommendation"] == "CONTINUE"
    assert budget_review.recommend(
        policy, attempt_count=40, guard=guard(50, 50), area_per_attempt=1.0
    )["recommendation"] == "PAUSE_PENDING_REVIEW"
    assert budget_review.recommend(
        policy, attempt_count=40, guard=guard(20, 80), area_per_attempt=1.0
    )["recommendation"] == "NEEDS_NEW_APPROACH"


def test_a_barely_tried_roll_is_never_paused(policy: dict) -> None:
    decision = budget_review.recommend(
        policy, attempt_count=2, guard=None, area_per_attempt=None
    )
    assert decision["recommendation"] == "CONTINUE"
    assert decision["primary_cause"] == "INSUFFICIENT_EFFORT_SPENT"


def test_a_roll_that_never_reached_the_relation_guard_needs_a_new_approach(
    policy: dict,
) -> None:
    decision = budget_review.recommend(
        policy, attempt_count=40, guard=None, area_per_attempt=5.0
    )
    assert decision["recommendation"] == "NEEDS_NEW_APPROACH"
    assert decision["primary_cause"] == "NO_RELATION_GUARD_EVALUATION"


def test_thin_guard_evidence_pauses_instead_of_reclassifying(policy: dict) -> None:
    decision = budget_review.recommend(
        policy, attempt_count=40, guard=guard(2, 8), area_per_attempt=1.0
    )
    assert decision["recommendation"] == "PAUSE_PENDING_REVIEW"
    assert decision["primary_cause"] == "GUARD_EVIDENCE_TOO_THIN"


def test_marginal_area_can_only_downgrade(policy: dict) -> None:
    healthy = guard(90, 10)
    assert budget_review.recommend(
        policy, attempt_count=40, guard=healthy, area_per_attempt=0.01
    )["recommendation"] == "PAUSE_PENDING_REVIEW"
    # A large area never rescues a failing relation guard.
    assert budget_review.recommend(
        policy, attempt_count=40, guard=guard(10, 90), area_per_attempt=10_000.0
    )["recommendation"] == "NEEDS_NEW_APPROACH"


def budget_fixture(tmp_path: Path) -> Path:
    for index in range(6):
        fleet_receipt(tmp_path / "healthy" / f"a{index}", area=5.0, policy="fixed-v1")
    write(
        tmp_path / "healthy" / "fit_ab" / "AB_EVALUATION.json",
        {
            "sample_id": "PHercTEST",
            "generated_at_utc": "2026-07-25T00:00:00Z",
            "status": "PASSED",
            "guard": {"passed_count": 90, "failed_count": 10},
        },
    )
    for index in range(6):
        directory = tmp_path / "silent" / f"a{index}"
        fleet_receipt(directory, area=5.0, policy="fixed-v1")
        write(
            directory / "CLAIMED_TASK.json",
            {
                "sample_id": "PHercSILENT",
                "candidate_discovery": {"seed_region_policy": "fixed-v1"},
            },
        )
    return tmp_path


def test_budget_review_recommends_per_roll(tmp_path: Path, policy: dict) -> None:
    root = budget_fixture(tmp_path)
    report = budget_review.build_report([root], root, root / "absent.sqlite", policy)
    rows = {row["sample_id"]: row for row in report["rolls"]}
    assert set(rows) == {"PHercTEST", "PHercSILENT"}
    assert rows["PHercTEST"]["recommendation"] == "CONTINUE"
    assert rows["PHercSILENT"]["recommendation"] == "NEEDS_NEW_APPROACH"
    assert rows["PHercSILENT"]["primary_cause"] == "NO_RELATION_GUARD_EVALUATION"
    assert report["totals"]["rolls_without_relation_guard_evaluation"] == ["PHercSILENT"]


def test_budget_review_never_scores_missing_evidence_as_zero(
    tmp_path: Path, policy: dict
) -> None:
    root = budget_fixture(tmp_path)
    report = budget_review.build_report([root], root, root / "absent.sqlite", policy)
    silent = next(row for row in report["rolls"] if row["sample_id"] == "PHercSILENT")
    assert silent["relation_guard"] is None
    assert silent["relation_guard_evaluation_count"] == 0
    assert silent["no_seed_cause_breakdown"] == "UNAVAILABLE"
    assert "never scored as zero" in report["no_seed_evidence"]["coverage_note"]


def test_budget_review_is_an_effort_decision_and_never_a_content_claim(
    tmp_path: Path, policy: dict
) -> None:
    root = budget_fixture(tmp_path)
    report = budget_review.build_report([root], root, root / "absent.sqlite", policy)

    assert report["decides"].startswith("how much further segmentation effort")
    assert "ink" in report["never_decides"]
    joined = " ".join(report["non_claims"]).lower()
    assert "effort allocation only" in joined
    assert "no ink" in joined
    assert "no recommendation retires a roll" in joined
    assert report["ink_used"] is False

    # No paused or retired roll may carry any ink-derived measurement, and the
    # report as a whole must never contain one.
    rendered = json.dumps(report).lower()
    for forbidden in ("ink_score", "glyph_like", "text_like", "letters_found", "ink_activation"):
        assert forbidden not in rendered

    for row in report["rolls"]:
        assert row["recommendation"] in {
            "CONTINUE",
            "PAUSE_PENDING_REVIEW",
            "NEEDS_NEW_APPROACH",
        }
        assert row["primary_cause"] in policy["decision_order"]


def test_budget_review_rejects_a_policy_that_is_not_ink_blind(tmp_path: Path) -> None:
    weakened = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    weakened["ink_blind"] = False
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(weakened), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ink-blind"):
        budget_review.load_policy(ROOT, path)


@needs_campaign_evidence
def test_budget_review_over_the_real_corpus_flags_the_stalled_roll() -> None:
    policy_value = budget_review.load_policy(ROOT, None)
    report = budget_review.build_report(
        [ROOT / "workspace"],
        ROOT,
        ROOT
        / "workspace/catalog/geometry_surface_catalog_v4/GEOMETRY_SURFACE_CATALOG.sqlite",
        policy_value,
    )
    rows = {row["sample_id"]: row for row in report["rolls"]}
    assert "PHerc257" in rows
    floor = policy_value["thresholds"]["minimum_unique_area_per_attempt_cm2"]
    stalled = [
        row
        for row in report["rolls"]
        if row["relation_guard"] is None
        and row["attempt_count"]
        >= policy_value["thresholds"]["minimum_attempts_before_any_pause"]
    ]
    # A roll that never reached a checkable stage is flagged, and it is flagged
    # for that reason rather than for producing little area.
    assert stalled, "expected at least one roll with no relation-guard evaluation"
    for row in stalled:
        assert row["recommendation"] == "NEEDS_NEW_APPROACH"
        assert row["primary_cause"] == "NO_RELATION_GUARD_EVALUATION"
        assert row["area_per_attempt_cm2"] > floor
    assert sum(report["totals"]["recommendation_counts"].values()) == report["totals"][
        "roll_count"
    ]
