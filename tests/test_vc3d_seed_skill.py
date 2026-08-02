from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents/skills/vc3d-select-seeds"
COMPARE_PATH = SKILL / "scripts/compare_strategies.py"
READINESS_PATH = SKILL / "scripts/production_readiness.py"
RECOVERY_PATH = (
    ROOT
    / "framework/stages/01-segmentation/scripts/build_geometry_recovery_v1.py"
)
MCP_SERVER_PATH = ROOT / "framework/stages/01-segmentation/mcp/server.py"
STAGE_ROOT = ROOT / "framework/stages/01-segmentation"
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))

from fleet.seed_probe import validate_seed_probe_benchmark_receipt


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


COMPARE = load_module("vc3d_seed_compare", COMPARE_PATH)
READINESS = load_module("vc3d_seed_readiness", READINESS_PATH)
RECOVERY = load_module("geometry_recovery_builder", RECOVERY_PATH)
MCP_SERVER = load_module("helena_seed_mcp_server", MCP_SERVER_PATH)


def benchmark_spec() -> dict:
    cells = [
        {
            "cell_id": f"cell-{index:03d}",
            "sample_id": f"PHerc{(index % 3) + 1:03d}",
            "independence_block_id": f"block-{index:03d}",
        }
        for index in range(60)
    ]
    return {
        "schema": "campaignx.seed_probe_benchmark_spec.v1",
        "benchmark_id": "seed-probe-test-v1",
        "frozen_at_utc": "2026-08-01T00:00:00Z",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "baseline": {
            "policy_version": "deterministic-v2-baseline-v1",
            "planner": "deterministic-v2",
            "seed_probe_mode": "off",
        },
        "closed_loop": {
            "policy_version": "seed-probe-select-benchmark-v1",
            "planner": "deterministic-v2",
            "seed_probe_mode": "select",
        },
        "minimum_cells": 40,
        "maximum_cells": 60,
        "minimum_scrolls": 3,
        "minimum_relative_yield_improvement": 0.10,
        "maximum_relative_reviewer_rate_regression": 0.0,
        "maximum_incremental_compute_wall_hours_per_cell": 0.25,
        "maximum_new_incorrect_lamina_rate_upper_bound": 0.05,
        "paired_test_alpha": 0.05,
        "minimum_pairs_per_scroll": 5,
        "review_protocol_id": "blind-single-lamina-review-v1",
        "cells": cells,
    }


def result_record(cell: dict, arm: str) -> dict:
    closed = arm == "closed_loop"
    return {
        "cell_id": cell["cell_id"],
        "sample_id": cell["sample_id"],
        "independence_block_id": cell["independence_block_id"],
        "arm": arm,
        "policy_version": (
            "seed-probe-select-benchmark-v1"
            if closed
            else "deterministic-v2-baseline-v1"
        ),
        "planner": "deterministic-v2",
        "seed_probe_mode": "select" if closed else "off",
        "source_content_lock_sha256": "a" * 64,
        "candidate_set_sha256": "b" * 64,
        "vc3d_binary_sha256": "c" * 64,
        "full_grow_profile_sha256": "d" * 64,
        "full_grow_envelope_sha256": "e" * 64,
        "rng_seed": f"matched-{cell['cell_id']}",
        "worker_tier": "cpu-standard-v1",
        "compute_device": "cpu",
        "review_protocol_id": "blind-single-lamina-review-v1",
        "reviewer_blinded": True,
        "canonical_area_cm2": 1.0,
        "usable_nonduplicate_single_lamina_area_cm2": 0.8 if closed else 0.5,
        "compute_wall_hours": 2.1 if closed else 2.0,
        "reviewer_minutes": 8.0,
        "geometry_rejected": False,
        "incorrect_lamina": False,
        "abstained": False,
        "geometry_qc_passed": True,
        "physical_qc_passed": True,
        "dedup_passed": True,
        "single_lamina_confirmed": True,
        "canonical_output_invariant_ok": True,
        "lease_invariant_ok": True,
        "replay_invariant_ok": True,
        "budget_invariant_ok": True,
    }


def benchmark_results(spec: dict) -> dict:
    return {
        "schema": "campaignx.seed_probe_benchmark_results.v1",
        "benchmark_id": spec["benchmark_id"],
        "started_at_utc": "2026-08-02T00:00:00Z",
        "records": [
            result_record(cell, arm)
            for cell in spec["cells"]
            for arm in ("baseline", "closed_loop")
        ],
    }


def resize_benchmark_spec(spec: dict, count: int) -> dict:
    resized = copy.deepcopy(spec)
    resized["cells"] = resized["cells"][:count]
    resized["minimum_cells"] = count
    resized["maximum_cells"] = count
    return resized


def benchmark_record(results: dict, cell_id: str, arm: str) -> dict:
    return next(
        row
        for row in results["records"]
        if row["cell_id"] == cell_id and row["arm"] == arm
    )


def compare_receipt(spec: dict, results: dict) -> dict:
    return COMPARE.compare(
        spec,
        results,
        spec_hash=COMPARE.content_sha256(spec),
        results_hash=COMPARE.content_sha256(results),
    )


def receipt_check(receipt: dict, check_id: str) -> dict:
    return next(row for row in receipt["checks"] if row["check_id"] == check_id)


def test_matched_benchmark_can_emit_an_approval_receipt() -> None:
    spec = benchmark_spec()
    results = benchmark_results(spec)
    receipt = compare_receipt(spec, results)
    assert receipt["status"] == "APPROVED_SELECT"
    assert receipt["paired_cell_count"] == 60
    assert receipt["scroll_count"] == 3
    assert receipt["authorized_sample_ids"] == [
        "PHerc001",
        "PHerc002",
        "PHerc003",
    ]
    authorization = validate_seed_probe_benchmark_receipt(receipt)
    assert authorization["authorized_sample_ids"] == receipt[
        "authorized_sample_ids"
    ]
    assert all(row["status"] == "PASS" for row in receipt["checks"])
    assert receipt["metrics"]["relative_yield_change"] > 0.10
    assert (
        receipt["receipt_sha256"]
        == COMPARE.content_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "receipt_sha256"
            }
        )
    )


def test_benchmark_refuses_an_rng_mismatch() -> None:
    spec = benchmark_spec()
    results = benchmark_results(spec)
    results = copy.deepcopy(results)
    results["records"][1]["rng_seed"] = "different-rng"
    receipt = compare_receipt(spec, results)
    assert receipt["status"] == "NOT_APPROVED"
    identity = next(
        row
        for row in receipt["checks"]
        if row["check_id"] == "MATCHED_EXECUTION_IDENTITY"
    )
    assert identity["status"] == "FAIL"
    assert identity["evidence"]["mismatches"][0]["differing_fields"] == [
        "rng_seed"
    ]


def test_shadow_is_not_accepted_as_the_causal_comparison_arm() -> None:
    spec = benchmark_spec()
    spec["closed_loop"]["seed_probe_mode"] = "shadow"
    with pytest.raises(COMPARE.BenchmarkError, match="must be select"):
        COMPARE.validate_spec(spec)


@pytest.mark.parametrize("bound", [0.0, 1.01])
def test_benchmark_refuses_an_impossible_new_harm_rate_bound(bound: float) -> None:
    spec = benchmark_spec()
    spec["maximum_new_incorrect_lamina_rate_upper_bound"] = bound
    with pytest.raises(
        COMPARE.BenchmarkError,
        match="maximum_new_incorrect_lamina_rate_upper_bound",
    ):
        COMPARE.validate_spec(spec)


def test_aggregate_gain_compatible_with_chance_is_not_approved() -> None:
    spec = resize_benchmark_spec(benchmark_spec(), 40)
    spec["maximum_new_incorrect_lamina_rate_upper_bound"] = 0.08
    results = benchmark_results(spec)
    for index, cell in enumerate(spec["cells"]):
        baseline = benchmark_record(results, cell["cell_id"], "baseline")
        closed = benchmark_record(results, cell["cell_id"], "closed_loop")
        baseline["compute_wall_hours"] = 2.0
        closed["compute_wall_hours"] = 2.0
        closed["usable_nonduplicate_single_lamina_area_cm2"] = (
            0.8 if index < 21 else 0.45
        )
    receipt = compare_receipt(spec, results)
    assert receipt["status"] == "NOT_APPROVED"
    assert receipt_check(receipt, "YIELD_IMPROVEMENT")["status"] == "PASS"
    paired = receipt_check(receipt, "PAIRED_SUPERIORITY")
    assert paired["status"] == "FAIL"
    assert paired["evidence"]["closed_loop_margin_wins"] == 21
    assert paired["evidence"]["one_sided_p_value"] == pytest.approx(
        0.43731465619057417
    )


def test_one_scroll_cannot_hide_regressions_on_two_others() -> None:
    spec = benchmark_spec()
    for index, cell in enumerate(spec["cells"]):
        cell["sample_id"] = (
            "PHerc001"
            if index < 50
            else "PHerc002"
            if index < 55
            else "PHerc003"
        )
    results = benchmark_results(spec)
    for cell in spec["cells"]:
        baseline = benchmark_record(results, cell["cell_id"], "baseline")
        closed = benchmark_record(results, cell["cell_id"], "closed_loop")
        baseline["compute_wall_hours"] = 2.0
        closed["compute_wall_hours"] = 2.0
        closed["usable_nonduplicate_single_lamina_area_cm2"] = (
            1.0 if cell["sample_id"] == "PHerc001" else 0.1
        )
    receipt = compare_receipt(spec, results)
    assert receipt["status"] == "NOT_APPROVED"
    assert receipt_check(receipt, "YIELD_IMPROVEMENT")["status"] == "PASS"
    scrolls = receipt_check(receipt, "SCROLL_NONREGRESSION")
    assert scrolls["status"] == "FAIL"
    assert scrolls["evidence"]["yield_regressed_scrolls"] == [
        "PHerc002",
        "PHerc003",
    ]


def test_new_lamina_harm_cannot_be_netted_against_a_baseline_harm() -> None:
    spec = benchmark_spec()
    results = benchmark_results(spec)
    baseline_harm = benchmark_record(results, "cell-000", "baseline")
    closed_harm = benchmark_record(results, "cell-001", "closed_loop")
    for row in (baseline_harm, closed_harm):
        row["incorrect_lamina"] = True
        row["single_lamina_confirmed"] = False
        row["physical_qc_passed"] = False
        row["usable_nonduplicate_single_lamina_area_cm2"] = 0.0
    receipt = compare_receipt(spec, results)
    assert receipt["metrics"]["absolute_incorrect_lamina_rate_change"] == 0.0
    safety = receipt_check(receipt, "NEW_INCORRECT_LAMINA_SAFETY")
    assert receipt["status"] == "NOT_APPROVED"
    assert safety["status"] == "FAIL"
    assert safety["evidence"]["new_harm_cell_ids"] == ["cell-001"]


def test_forty_zero_harm_pairs_do_not_prove_a_five_percent_safety_bound() -> None:
    spec = resize_benchmark_spec(benchmark_spec(), 40)
    results = benchmark_results(spec)
    receipt = compare_receipt(spec, results)
    safety = receipt_check(receipt, "NEW_INCORRECT_LAMINA_SAFETY")
    assert receipt["status"] == "NOT_APPROVED"
    assert safety["status"] == "FAIL"
    assert safety["evidence"]["one_sided_upper_bound"] == pytest.approx(
        0.07215752450551463
    )


def test_abstention_or_geometry_rejection_cannot_report_usable_area() -> None:
    spec = benchmark_spec()
    results = benchmark_results(spec)
    benchmark_record(results, "cell-000", "closed_loop")["abstained"] = True
    with pytest.raises(COMPARE.BenchmarkError, match="abstained but reports"):
        compare_receipt(spec, results)

    results = benchmark_results(spec)
    benchmark_record(results, "cell-000", "closed_loop")[
        "geometry_rejected"
    ] = True
    with pytest.raises(COMPARE.BenchmarkError, match="geometry rejection"):
        compare_receipt(spec, results)


def test_benchmark_decision_metrics_are_deterministic() -> None:
    spec = benchmark_spec()
    results = benchmark_results(spec)
    first = compare_receipt(spec, results)
    second = compare_receipt(spec, results)
    assert first["status"] == second["status"]
    assert first["checks"] == second["checks"]
    assert first["metrics"] == second["metrics"]


def test_skill_code_readiness_passes_without_claiming_live_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELENA_DISABLE_SEED_PROBE_V1", raising=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(READINESS_PATH),
            "--root",
            str(ROOT),
            "--mode",
            "shadow",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "READY"
    assert receipt["scope"] == "CODE"
    assert {
        row["status"] for row in receipt["checks"] if row["check_id"].startswith("LIVE_")
    } == {"SKIP"}


def test_readiness_requires_every_published_seed_probe_contract() -> None:
    required_contracts = {
        "framework/contracts/schemas/m7-seed-candidate-evidence-v1.schema.json",
        "framework/contracts/schemas/seed-probe-artifact-set-v1.schema.json",
        "framework/contracts/schemas/seed-probe-benchmark-execution-v1.schema.json",
        "framework/contracts/schemas/seed-probe-decision-v1.schema.json",
        "framework/contracts/schemas/seed-probe-evaluation-v1.schema.json",
        "framework/contracts/schemas/seed-probe-locked-plan-v1.schema.json",
        "framework/contracts/schemas/seed-probe-policy-v1.schema.json",
        "framework/contracts/schemas/seed-probe-promotion-v1.schema.json",
        "framework/contracts/schemas/source-content-lock-v1.schema.json",
    }
    assert required_contracts <= set(READINESS.REQUIRED_RUNTIME_FILES)


def test_select_readiness_fails_closed_without_locks_or_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(READINESS_PATH),
            "--root",
            str(ROOT),
            "--mode",
            "select",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "NOT_READY"
    assert {
        "SELECT_IMMUTABLE_SOURCES",
        "SELECT_ROLLOUT_FLAG",
        "SELECT_BENCHMARK_APPROVAL",
        "SELECT_BENCHMARK_SCOPE",
        "SELECT_REVIEW_OWNER",
    }.issubset(receipt["failed_check_ids"])


def test_production_worker_gate_uses_a_recent_postgresql_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: dict[str, str] = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, query: str) -> None:
            executed["query"] = query

        def fetchone(self):
            return (2,)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return Cursor()

    def connect(database_url: str, *, connect_timeout: int):
        assert database_url == "postgresql://redacted.example/test"
        assert connect_timeout == 5
        return Connection()

    monkeypatch.setenv(
        "VC3D_SKILL_TEST_DATABASE_URL",
        "postgresql://redacted.example/test",
    )
    monkeypatch.setitem(
        sys.modules,
        "psycopg2",
        SimpleNamespace(connect=connect),
    )
    assert (
        READINESS.fresh_probe_worker_count(
            "postgres-env://VC3D_SKILL_TEST_DATABASE_URL"
        )
        == 2
    )
    assert "updated_at >= now() - interval '20 minutes'" in executed["query"]


def test_bundled_mcp_threshold_is_declared_and_recovery_reads_its_xyz() -> None:
    properties = MCP_SERVER.TOOLS[0]["inputSchema"]["properties"]
    assert properties["threshold"]["minimum"] == 0.0
    assert properties["threshold"]["maximum"] == 1.0
    expected = {"x": 11, "y": 22, "z": 33}
    assert RECOVERY.candidate_coordinate({"ct_l0_coordinate": expected}) == expected
    assert RECOVERY.candidate_coordinate(expected) == expected
    assert RECOVERY.candidate_coordinate({"coordinate": expected}) == expected


def test_active_cost_aware_profile_declares_probe_winner_route() -> None:
    profile = json.loads(
        (
            ROOT
            / "framework/profiles/01-segmentation/"
            "segmentation-planner-cost-aware-v2-1.0.0.json"
        ).read_text(encoding="utf-8")
    )
    assert profile["decision_order"][0] == "DETERMINISTIC_PROBE_WINNER"
