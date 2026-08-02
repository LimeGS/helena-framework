from __future__ import annotations

import copy
import hashlib
import itertools
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256, stable_id
from fleet.executor import FixtureGrowExecutor, resolve_execution_rng
from fleet.generator import (
    SEED_PROBE_CONTINUATION_ENVELOPE,
    generate_seed_probe_benchmark_arm_tasks,
    generate_tasks_for_snapshot,
)
from fleet.planner import (
    DeterministicPlanner,
    task_packet_for_planner,
    validate_and_lock,
)
from fleet.seed_probe import (
    BENCHMARK_AUTHORIZATION_SCHEMA,
    BENCHMARK_RNG_PROTOCOL,
    build_seed_probe_benchmark_execution,
    default_seed_probe_policy,
    validate_seed_probe_benchmark_spec,
    validate_seed_probe_task_contract,
)
from fleet.store import FleetStore


SAMPLES = ("PHercA", "PHercB", "PHercC")


def _authorization() -> dict:
    indices = list(itertools.product(range(8), repeat=3))[:40]
    cells = [
        {
            "sample_id": SAMPLES[index % len(SAMPLES)],
            "cell_id": "r%05dc%05da%05d" % grid_index,
            "independence_block_id": f"block-{index:03d}",
        }
        for index, grid_index in enumerate(indices)
    ]
    return validate_seed_probe_benchmark_spec(
        {
            "schema": "campaignx.seed_probe_benchmark_spec.v1",
            "benchmark_id": "isolated-paired-runtime-v1",
            "frozen_at_utc": "2026-07-30T12:00:00Z",
            "execution_scope": "ISOLATED_NONPRODUCTION",
            "baseline": {
                "policy_version": "paired-off-v1",
                "planner": "deterministic-v2",
                "seed_probe_mode": "off",
            },
            "closed_loop": {
                "policy_version": "paired-select-v1",
                "planner": "deterministic-v2",
                "seed_probe_mode": "select",
            },
            "minimum_cells": 40,
            "maximum_cells": 40,
            "minimum_scrolls": 3,
            "minimum_relative_yield_improvement": 0.1,
            "maximum_relative_reviewer_rate_regression": 0.0,
            "maximum_incremental_compute_wall_hours_per_cell": 0.25,
            "maximum_new_incorrect_lamina_rate_upper_bound": 0.05,
            "paired_test_alpha": 0.05,
            "minimum_pairs_per_scroll": 5,
            "review_protocol_id": "blind-single-lamina-review-v1",
            "cells": cells,
        }
    )


def _snapshots(store: FleetStore) -> list[dict]:
    for sample_id in SAMPLES:
        ct_version = f"ct-version-{sample_id}"
        m7_version = f"m7-version-{sample_id}"
        ct_uri = f"fixture://ct/{ct_version}"
        m7_uri = f"fixture://m7/{m7_version}"
        store.register_snapshot(
            {
                "sample_id": sample_id,
                "ct_uri": ct_uri,
                "ct_sha256": "0" * 64,
                "m7_uri": m7_uri,
                "m7_sha256": "1" * 64,
                "shape_xyz": [512, 512, 512],
                "voxel_size_um": 9.362,
                "coordinate_frame": "ct_l0_xyz",
                "source_content_lock": {
                    "schema": "campaignx.source_content_lock.v1",
                    "status": "VERIFIED_IMMUTABLE",
                    "verification_method": "fixture-sha256-v1",
                    "verified_at_utc": "2026-07-30T00:00:00Z",
                    "ct_uri": ct_uri,
                    "ct_sha256": "0" * 64,
                    "ct_version_id": ct_version,
                    "m7_uri": m7_uri,
                    "m7_sha256": "1" * 64,
                    "m7_version_id": m7_version,
                },
            }
        )
    return store.snapshots(set(SAMPLES))


def _generation_options() -> dict:
    return {
        "catalog_snapshot_sha256": "2" * 64,
        "grid_step": 64,
        "query_radius": 32,
        "clearance": 0.0,
        "volume_edge_margin": 32,
        "candidate_interior_clearance": 0,
        "selection_strategy": "max-clearance-v1",
        "max_tasks": 40,
        "grid_version": "paired-grid-v1",
    }


def _task_id(task: dict) -> str:
    return stable_id(
        "task",
        {
            key: task[key]
            for key in (
                "source_snapshot_id",
                "grid_version",
                "cell_id",
                "policy_version",
            )
        },
    )


def test_both_arms_generate_exact_cohort_with_common_pair_rng(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshots = _snapshots(store)
    authorization = _authorization()
    closed_policy = default_seed_probe_policy(
        mode="select",
        benchmark_authorization=authorization,
    )

    baseline = generate_seed_probe_benchmark_arm_tasks(
        store,
        snapshots,
        benchmark_execution_authorization=authorization,
        arm="baseline",
        generation_options=_generation_options(),
    )
    closed = generate_seed_probe_benchmark_arm_tasks(
        store,
        snapshots,
        benchmark_execution_authorization=authorization,
        arm="closed_loop",
        generation_options=_generation_options(),
        seed_probe=closed_policy,
    )

    expected = {
        (cell["sample_id"], cell["cell_id"])
        for cell in authorization["cells"]
    }
    assert {(row["sample_id"], row["cell_id"]) for row in baseline} == expected
    assert {(row["sample_id"], row["cell_id"]) for row in closed} == expected
    assert len(baseline) == len(closed) == 40
    baseline_by_pair = {
        (row["sample_id"], row["cell_id"]): row for row in baseline
    }
    closed_by_pair = {
        (row["sample_id"], row["cell_id"]): row for row in closed
    }
    for identity in expected:
        off = baseline_by_pair[identity]
        select = closed_by_pair[identity]
        assert _task_id(off) != _task_id(select)
        assert (
            off["benchmark_execution"]["pair_rng_seed"]
            == select["benchmark_execution"]["pair_rng_seed"]
        )
        assert (
            off["benchmark_execution"]["full_grow_envelope_sha256"]
            == select["benchmark_execution"]["full_grow_envelope_sha256"]
            == content_sha256(SEED_PROBE_CONTINUATION_ENVELOPE)
        )


def test_execution_contract_propagates_and_retry_rng_is_stable(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshots = _snapshots(store)
    authorization = _authorization()
    baseline = generate_seed_probe_benchmark_arm_tasks(
        store,
        snapshots,
        benchmark_execution_authorization=authorization,
        arm="baseline",
        generation_options=_generation_options(),
    )
    closed = generate_seed_probe_benchmark_arm_tasks(
        store,
        snapshots,
        benchmark_execution_authorization=authorization,
        arm="closed_loop",
        generation_options=_generation_options(),
        seed_probe=default_seed_probe_policy(
            mode="select",
            benchmark_authorization=authorization,
        ),
    )
    store.create_tasks([*baseline, *closed])

    assert store.claim("ordinary-worker", 60) is None
    claim = store.claim(
        "benchmark-worker",
        60,
        capabilities={
            "benchmark_spec_sha256": authorization[
                "benchmark_spec_sha256"
            ]
        },
    )
    assert claim is not None
    candidate = {
        "candidate_id": "c01",
        **claim["center_xyz"],
        "score": 1.0,
        "cell_interior_clearance_voxels": 32.0,
        "volume_interior_clearance_voxels": 64.0,
    }
    packet = task_packet_for_planner(
        claim,
        [candidate],
        contract_version="v2",
        regional_attempt_history=store.regional_attempt_history(claim),
    )
    proposal = DeterministicPlanner(contract_version="v2").propose(
        packet, tmp_path
    )
    locked = validate_and_lock(packet, proposal)
    assert packet["benchmark_execution"] == claim["benchmark_execution"]
    assert locked["benchmark_execution"] == claim["benchmark_execution"]

    pair = (claim["sample_id"], claim["cell_id"])
    closed_task = next(
        task
        for task in closed
        if (task["sample_id"], task["cell_id"]) == pair
    )
    closed_locked = {
        **copy.deepcopy(locked),
        "task_id": _task_id(closed_task),
        "attempt_id": "closed-attempt",
        "benchmark_execution": closed_task["benchmark_execution"],
    }
    executor = FixtureGrowExecutor()
    off_receipt = executor.execute(locked, tmp_path / "off")["receipt"]
    closed_receipt = executor.execute(
        closed_locked, tmp_path / "closed"
    )["receipt"]
    retry_locked = {**copy.deepcopy(locked), "attempt_id": "retry-attempt"}
    retry_receipt = executor.execute(
        retry_locked, tmp_path / "retry"
    )["receipt"]
    assert {
        off_receipt["rng_seed"],
        closed_receipt["rng_seed"],
        retry_receipt["rng_seed"],
    } == {claim["benchmark_execution"]["pair_rng_seed"]}
    assert off_receipt["rng_protocol"] == BENCHMARK_RNG_PROTOCOL

    invalid = copy.deepcopy(locked)
    invalid["benchmark_execution"]["pair_rng_seed"] = "0" * 16
    with pytest.raises(ValueError, match="RNG binding"):
        resolve_execution_rng(invalid)
    with pytest.raises(ValueError, match="standalone"):
        resolve_execution_rng(
            {
                "attempt_id": "legacy-attempt",
                "execution_rng_seed": "0" * 16,
            }
        )
    legacy = resolve_execution_rng({"attempt_id": "legacy-attempt"})
    assert legacy["rng_seed"] == hashlib.sha256(
        b"legacy-attempt"
    ).hexdigest()[:16]


def test_store_collision_and_production_sample_scope_fail_closed(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshots = _snapshots(store)
    authorization = _authorization()
    baseline = generate_seed_probe_benchmark_arm_tasks(
        store,
        snapshots,
        benchmark_execution_authorization=authorization,
        arm="baseline",
        generation_options=_generation_options(),
    )
    store.create_tasks(baseline)
    collision = copy.deepcopy(baseline[0])
    other_authorization = {
        **copy.deepcopy(authorization),
        "benchmark_id": "other-benchmark",
        "benchmark_spec_sha256": "f" * 64,
    }
    collision["benchmark_execution"] = (
        build_seed_probe_benchmark_execution(
            other_authorization,
            arm="baseline",
            sample_id=collision["sample_id"],
            cell_id=collision["cell_id"],
            parameter_envelope=collision["parameter_envelope"],
        )
    )
    with pytest.raises(ValueError, match="different benchmark_execution"):
        store.create_tasks([collision])

    production_authorization = {
        "schema": BENCHMARK_AUTHORIZATION_SCHEMA,
        "benchmark_id": "approved-v1",
        "decision_receipt_sha256": "a" * 64,
        "spec_sha256": "b" * 64,
        "results_sha256": "c" * 64,
        "paired_cell_count": 40,
        "scroll_count": 3,
        "authorized_sample_ids": list(SAMPLES),
        "execution_scope": "ISOLATED_NONPRODUCTION",
    }
    unauthorized_snapshot = {
        **snapshots[0],
        "sample_id": "PHercOUTSIDE",
    }
    with pytest.raises(ValueError, match="production-approved select cohort"):
        generate_tasks_for_snapshot(
            store,
            unauthorized_snapshot,
            seed_probe=default_seed_probe_policy(
                mode="select",
                benchmark_authorization=production_authorization,
                review_owner="segmentation-review@campaign-x",
            ),
            policy_version="approved-select-v1",
            planner="deterministic-v2",
            **_generation_options(),
        )

    runtime_task = copy.deepcopy(baseline[0])
    runtime_task.pop("benchmark_execution")
    runtime_task["sample_id"] = "PHercOUTSIDE"
    runtime_task["source"] = unauthorized_snapshot
    runtime_task["seed_probe"] = default_seed_probe_policy(
        mode="select",
        benchmark_authorization=production_authorization,
        review_owner="segmentation-review@campaign-x",
    )
    runtime_task["resource_requirements"]["seed_probe_required"] = True
    with pytest.raises(ValueError, match="authorized non-regressing"):
        validate_seed_probe_task_contract(runtime_task)
