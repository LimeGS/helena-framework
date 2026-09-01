from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import tifffile
from jsonschema import Draft202012Validator
from referencing import Registry, Resource


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.common import content_sha256, file_sha256
from fleet.artifact_store import LocalArtifactStore
from fleet.cli import build_parser, command_bootstrap, command_probe_gc
import fleet.cli as cli_module
from fleet.executor import FixtureGrowExecutor
from fleet.generator import SEED_PROBE_CONTINUATION_ENVELOPE
from fleet.planner import (
    CostAwareSegmentationPlanner,
    PlannerScientificViolation,
)
from fleet.seed_probe import (
    PROBE_EVALUATION_PROFILE_SHA256,
    PROBE_PROFILE_SHA256,
    build_seed_probe_benchmark_execution,
    build_probe_locked_plan,
    decide_probe_run,
    default_seed_probe_policy,
    load_seed_probe_benchmark_receipt,
    load_seed_probe_benchmark_spec,
    normalize_seed_probe_policy,
    validate_seed_probe_benchmark_receipt,
    validate_seed_probe_benchmark_spec,
    normalize_source_content_lock,
    validate_seed_probe_task_contract,
    validate_probe_continuation_contract,
    verify_probe_evaluation_profile,
    verify_probe_growth_profile,
)
from fleet.store import FleetStore
from fleet.worker import RecordedSeedProvider, SegmentWorker
import fleet.worker as worker_module


def _validate_contract(instance: dict, schema_name: str) -> None:
    schema = json.loads(
        (
            ROOT
            / "framework/contracts/schemas"
            / schema_name
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(instance)


def _source(store: FleetStore) -> str:
    ct_uri = "fixture://ct/fixture-ct-v1"
    m7_uri = "fixture://m7/fixture-m7-v1"
    return store.register_snapshot(
        {
            "sample_id": "PHercPROBE",
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
                "ct_version_id": "fixture-ct-v1",
                "m7_uri": m7_uri,
                "m7_sha256": "1" * 64,
                "m7_version_id": "fixture-m7-v1",
            },
        }
    )


def _task(
    source_id: str,
    *,
    mode: str = "select",
    top_k: int = 2,
    cell_id: str = "probe-cell",
) -> dict:
    return {
        "source_snapshot_id": source_id,
        "sample_id": "PHercPROBE",
        "cell_id": cell_id,
        "grid_version": "probe-grid-v1",
        "policy_version": f"seed-probe-{mode}-k{top_k}-v1",
        "bounds_xyz": [[128, 128, 128], [384, 384, 384]],
        "center_xyz": {"x": 256, "y": 256, "z": 256},
        "priority": 1.0,
        "parameter_envelope": copy.deepcopy(
            SEED_PROBE_CONTINUATION_ENVELOPE
        ),
        "catalog_snapshot_sha256": "2" * 64,
        "candidate_selection_policy": "adaptive-geometry-history-v2",
        "planner_contract_version": "v2",
        "planner": "cost-aware-v2",
        "recorded_candidates": [
            {
                "candidate_id": "c01",
                "coordinate": {"x": 256, "y": 256, "z": 256},
                "score": 0.9,
            },
            {
                "candidate_id": "c02",
                "coordinate": {"x": 220, "y": 220, "z": 220},
                "score": 0.8,
            },
        ],
        "seed_probe": default_seed_probe_policy(
            mode=mode,
            top_k=top_k,
            generations=12,
            step_size=12,
        ),
        "resource_requirements": {
            "gpu_required": False,
            "minimum_vram_gb": 0.0,
            "seed_probe_required": True,
        },
        "ink_used": False,
    }


def _candidate(rank: int = 1) -> dict:
    return {
        "candidate_rank": rank,
        "candidate_id": f"c{rank:02d}",
        "x": 256 - rank,
        "y": 256,
        "z": 256,
        "score": 1.0 - rank / 10.0,
        "cell_interior_clearance_voxels": 100.0,
        "volume_interior_clearance_voxels": 200.0,
        "source": None,
    }


def _probe_fingerprint(tmp_path: Path) -> dict:
    return {
        "executor": "fixture",
        "fixture_only": True,
        "probe_namespace": LocalArtifactStore(
            tmp_path / "artifacts"
        ).probe_namespace_identity(),
    }


def _benchmark_approval_receipt() -> dict:
    checks = [
        {
            "check_id": check_id,
            "status": "PASS",
            "detail": f"{check_id} passed",
            "evidence": {},
        }
        for check_id in (
            "MATCHED_EXECUTION_IDENTITY",
            "INVARIANTS",
            "YIELD_IMPROVEMENT",
            "PAIRED_SUPERIORITY",
            "REVIEWER_RATE",
            "COMPUTE_BUDGET",
            "NEW_INCORRECT_LAMINA_SAFETY",
            "SCROLL_COVERAGE",
            "SCROLL_NONREGRESSION",
        )
    ]
    authorized_sample_ids = ["PHercA", "PHercB", "PHercPROBE"]
    receipt = {
        "schema": "campaignx.seed_probe_benchmark_decision.v1",
        "benchmark_id": "paired-seed-probe-v1",
        "status": "APPROVED_SELECT",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "spec_sha256": "a" * 64,
        "results_sha256": "b" * 64,
        "paired_cell_count": 40,
        "scroll_count": 3,
        "authorized_sample_ids": authorized_sample_ids,
        "arms": {
            "baseline": {
                "policy_version": "paired-baseline-v1",
                "planner": "deterministic-v2",
                "seed_probe_mode": "off",
            },
            "closed_loop": {
                "policy_version": "paired-select-v1",
                "planner": "deterministic-v2",
                "seed_probe_mode": "select",
            },
        },
        "metrics": {
            "per_scroll": {
                sample_id: {"relative_yield_change": 0.1}
                for sample_id in authorized_sample_ids
            }
        },
        "checks": checks,
        "generated_at_utc": "2026-07-30T12:00:00Z",
        "non_claims": ["approval is not proof that a seed is correct"],
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def _benchmark_spec() -> dict:
    cells = [
        {
            "cell_id": "probe-cell",
            "sample_id": "PHercPROBE",
            "independence_block_id": "block-000",
        }
    ]
    cells.extend(
        {
            "cell_id": f"cell-{index:03d}",
            "sample_id": (
                "PHercPROBE"
                if index < 5
                else ("PHercA" if index % 2 else "PHercB")
            ),
            "independence_block_id": f"block-{index:03d}",
        }
        for index in range(1, 40)
    )
    return {
        "schema": "campaignx.seed_probe_benchmark_spec.v1",
        "benchmark_id": "paired-seed-probe-v1",
        "frozen_at_utc": "2026-07-30T10:00:00Z",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "baseline": {
            "policy_version": "paired-baseline-v1",
            "planner": "deterministic-v2",
            "seed_probe_mode": "off",
        },
        "closed_loop": {
            "policy_version": "paired-select-v1",
            "planner": "deterministic-v2",
            "seed_probe_mode": "select",
        },
        "minimum_cells": 40,
        "maximum_cells": 60,
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


def _decision_run(*verdicts: str) -> dict:
    states = {
        "ELIGIBLE": "SUCCEEDED",
        "REJECTED": "REJECTED",
        "UNMEASURED": "UNMEASURED",
        "FAILED": "FAILED",
    }
    trials = []
    for rank, verdict in enumerate(verdicts, start=1):
        evaluation = (
            None
            if verdict == "FAILED"
            else {
                "verdict": verdict,
                "geometry_qc_state": (
                    "GEOMETRY_CERTIFIED"
                    if verdict == "ELIGIBLE"
                    else (
                        "GEOMETRY_REJECTED_BRIDGE"
                        if verdict == "REJECTED"
                        else "GEOMETRY_UNMEASURED"
                    )
                ),
            }
        )
        trials.append(
            {
                "probe_trial_id": f"trial-{rank}",
                "candidate_rank": rank,
                "candidate": _candidate(rank),
                "state": states[verdict],
                "evaluation": evaluation,
            }
        )
    return {
        "probe_run_id": "probe-run",
        "policy": default_seed_probe_policy(
            mode="select", top_k=len(verdicts)
        ),
        "trials": trials,
    }


def _claim_parent(store: FleetStore, task_value: dict) -> dict:
    store.create_tasks([task_value])
    capabilities = {"seed_probe_v1": True}
    benchmark_execution = task_value.get("benchmark_execution")
    if benchmark_execution is not None:
        capabilities["benchmark_spec_sha256"] = benchmark_execution[
            "benchmark_spec_sha256"
        ]
    claim = store.claim(
        "probe-worker",
        60,
        capabilities=capabilities,
    )
    assert claim is not None
    return claim


class OneEligibleOneRejectedExecutor:
    """Fixture VC3D output with one deterministic lamina-switch defect."""

    def __init__(self) -> None:
        self.fixture = FixtureGrowExecutor()

    def execute(self, locked_plan: dict, attempt_dir: Path) -> dict:
        grown = self.fixture.execute(locked_plan, attempt_dir)
        if locked_plan["selected_seed"]["candidate_id"] == "c02":
            surface = Path(grown["surface_dir"])
            z_path = surface / "z.tif"
            z = np.asarray(tifffile.imread(z_path), dtype=np.float32)
            # Twelve times the grid step, not twelve voxels. The gate calls a
            # jump a lamina switch when it exceeds `step_discontinuity_factor`
            # (8) times the step, so a constant was only ever a defect while the
            # fixture's step happened to be one voxel.
            z[z.shape[0] // 2 :, :] += 12.0 * FixtureGrowExecutor.grid_step_voxels()
            tifffile.imwrite(z_path, z)
        return grown


class AlwaysFailingProbeExecutor:
    def execute(self, _locked_plan: dict, _attempt_dir: Path) -> dict:
        raise RuntimeError("fixture probe worker failure")


class RejectingV2Planner:
    contract_version = "v2"

    def propose(self, *_args, **_kwargs):
        raise PlannerScientificViolation("fixture planner policy refusal")


def test_frozen_policy_and_profile_are_schema_valid_and_hash_linked() -> None:
    policy = default_seed_probe_policy(mode="shadow")
    policy_schema = json.loads(
        (
            ROOT
            / "framework/contracts/schemas/seed-probe-policy-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(policy_schema).validate(policy)
    profile_path = (
        ROOT
        / "framework/stages/01-segmentation/fleet/profiles/"
        "vc3d-m7-probe-v1.json"
    )
    assert file_sha256(profile_path) == PROBE_PROFILE_SHA256
    assert verify_probe_growth_profile()["profile_id"] == "vc3d-m7-probe-v1"
    evaluation_profile = verify_probe_evaluation_profile()
    assert (
        file_sha256(
            ROOT
            / "framework/stages/01-segmentation/fleet/profiles/"
            "tifxyz-geometry-gate-probe-v1.json"
        )
        == PROBE_EVALUATION_PROFILE_SHA256
    )
    assert evaluation_profile["profile_id"] == "tifxyz-geometry-gate-probe-v1"

    with pytest.raises(ValueError, match="top_k"):
        normalize_seed_probe_policy({**policy, "top_k": 4})
    with pytest.raises(ValueError, match="total_probe_generations"):
        normalize_seed_probe_policy(
            {**policy, "maximum_total_probe_generations": 25}
        )
    with pytest.raises(ValueError, match="ink-blind"):
        normalize_seed_probe_policy({**policy, "ink_used": True})
    with pytest.raises(ValueError, match="use_cuda is frozen"):
        normalize_seed_probe_policy(
            {
                **policy,
                "probe_parameters": {
                    **policy["probe_parameters"],
                    "use_cuda": True,
                },
            }
        )


def test_select_benchmark_receipt_is_exact_and_bound_to_policy(
    tmp_path: Path,
) -> None:
    receipt = _benchmark_approval_receipt()
    receipt_path = tmp_path / "approval.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    authorization = load_seed_probe_benchmark_receipt(receipt_path)
    assert authorization == validate_seed_probe_benchmark_receipt(receipt)
    assert (
        authorization["decision_receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    policy = default_seed_probe_policy(
        mode="select",
        benchmark_authorization=authorization,
        review_owner="segmentation-review@campaign-x",
    )
    _validate_contract(policy, "seed-probe-policy-v1.schema.json")
    assert policy["benchmark_authorization"] == authorization
    assert policy["review_owner"] == "segmentation-review@campaign-x"
    assert "approval.json" not in json.dumps(policy)

    with pytest.raises(ValueError, match="review_owner"):
        default_seed_probe_policy(
            mode="select",
            benchmark_authorization=authorization,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update({"paired_cell_count": 39}),
            "paired_cell_count",
        ),
        (
            lambda value: value["checks"][0].update({"status": "FAIL"}),
            "did not PASS",
        ),
        (
            lambda value: value["arms"]["closed_loop"].update(
                {"planner": "cost-aware-v2"}
            ),
            "deterministic-v2",
        ),
    ],
)
def test_select_benchmark_receipt_rejects_semantic_bypasses(
    mutate,
    message: str,
) -> None:
    receipt = _benchmark_approval_receipt()
    mutate(receipt)
    receipt["receipt_sha256"] = content_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match=message):
        validate_seed_probe_benchmark_receipt(receipt)


def test_select_benchmark_receipt_rejects_canonical_hash_tampering() -> None:
    receipt = _benchmark_approval_receipt()
    receipt["metrics"]["invented"] = True
    with pytest.raises(ValueError, match="canonical SHA-256"):
        validate_seed_probe_benchmark_receipt(receipt)


def test_preregistered_spec_authorizes_only_its_isolated_select_cohort(
    tmp_path: Path,
) -> None:
    spec = _benchmark_spec()
    spec_path = tmp_path / "benchmark-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    authorization = load_seed_probe_benchmark_spec(spec_path)
    assert authorization["benchmark_spec_sha256"] == content_sha256(spec)
    assert authorization["execution_scope"] == "ISOLATED_NONPRODUCTION"

    store = FleetStore(tmp_path / "benchmark.sqlite")
    store.initialize()
    task = _task(_source(store))
    task["policy_version"] = "paired-select-v1"
    task["planner"] = "deterministic-v2"
    task["seed_probe"] = default_seed_probe_policy(
        mode="select",
        benchmark_authorization=authorization,
    )
    task["benchmark_execution"] = build_seed_probe_benchmark_execution(
        authorization,
        arm="closed_loop",
        sample_id=task["sample_id"],
        cell_id=task["cell_id"],
        parameter_envelope=task["parameter_envelope"],
    )
    _validate_contract(
        task["seed_probe"], "seed-probe-policy-v1.schema.json"
    )
    claim = _claim_parent(store, task)
    assert (
        validate_seed_probe_task_contract(claim)["benchmark_authorization"]
        == authorization
    )

    outside = copy.deepcopy(claim)
    outside["cell_id"] = "not-preregistered"
    with pytest.raises(ValueError, match="cell_id differs from its task"):
        validate_seed_probe_task_contract(outside)
    wrong_arm = copy.deepcopy(claim)
    wrong_arm["policy_version"] = "another-policy"
    with pytest.raises(ValueError, match="policy_version differs from its task"):
        validate_seed_probe_task_contract(wrong_arm)


@pytest.mark.parametrize("bound", [0.0, 1.01])
def test_preregistered_spec_rejects_invalid_lamina_safety_bound(
    bound: float,
) -> None:
    spec = _benchmark_spec()
    spec["maximum_new_incorrect_lamina_rate_upper_bound"] = bound
    with pytest.raises(
        ValueError,
        match="maximum_new_incorrect_lamina_rate_upper_bound",
    ):
        validate_seed_probe_benchmark_spec(spec)


def test_nonfixture_programmatic_select_has_no_unapproved_bypass() -> None:
    ct_version = "ct-version-0001"
    m7_version = "m7-version-0001"
    ct_uri = f"s3://fixture-bucket/ct?versionId={ct_version}"
    m7_uri = f"s3://fixture-bucket/m7?versionId={m7_version}"
    task = {
        "sample_id": "PHercREAL",
        "cell_id": "real-cell",
        "policy_version": "unapproved-select-v1",
        "planner": "deterministic-v2",
        "parameter_envelope": copy.deepcopy(
            SEED_PROBE_CONTINUATION_ENVELOPE
        ),
        "seed_probe": default_seed_probe_policy(mode="select"),
        "source": {
            "ct_uri": ct_uri,
            "ct_sha256": "0" * 64,
            "m7_uri": m7_uri,
            "m7_sha256": "1" * 64,
            "source_content_lock": {
                "schema": "campaignx.source_content_lock.v1",
                "status": "VERIFIED_IMMUTABLE",
                "verification_method": "immutable-uri-manifest-sha256-v1",
                "verified_at_utc": "2026-07-30T00:00:00Z",
                "ct_uri": ct_uri,
                "ct_sha256": "0" * 64,
                "ct_version_id": ct_version,
                "m7_uri": m7_uri,
                "m7_sha256": "1" * 64,
                "m7_version_id": m7_version,
            },
        },
    }
    with pytest.raises(ValueError, match="requires a production approval"):
        validate_seed_probe_task_contract(task)


def _bootstrap_args(
    tmp_path: Path,
    *,
    mode: str,
    benchmark_receipt: Path | None = None,
    review_owner: str | None = None,
) -> argparse.Namespace:
    arguments = [
        "bootstrap",
        "--db",
        str(tmp_path / "fleet.sqlite"),
        "--eligible",
        str(tmp_path / "eligible.json"),
        "--catalog",
        str(tmp_path / "catalog.json"),
        "--receipt",
        str(tmp_path / "bootstrap.json"),
        "--seed-probe-mode",
        mode,
    ]
    if benchmark_receipt is not None:
        arguments.extend(
            ["--seed-probe-benchmark-receipt", str(benchmark_receipt)]
        )
    if review_owner is not None:
        arguments.extend(["--seed-probe-review-owner", review_owner])
    return build_parser(ROOT).parse_args(arguments)


def test_cli_select_requires_both_flag_and_explicit_benchmark_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "approval.json"
    receipt_path.write_text(
        json.dumps(_benchmark_approval_receipt()), encoding="utf-8"
    )
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    with pytest.raises(RuntimeError, match="rollout-gated"):
        command_bootstrap(
            _bootstrap_args(
                tmp_path, mode="select", benchmark_receipt=receipt_path
            )
        )

    monkeypatch.setenv("HELENA_ENABLE_SEED_PROBE_SELECT", "true")
    with pytest.raises(RuntimeError, match="environment flag alone"):
        command_bootstrap(_bootstrap_args(tmp_path, mode="select"))


def test_cli_select_binds_validated_approval_without_persisting_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _benchmark_approval_receipt()
    receipt_path = tmp_path / "approval.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    captured: dict = {}

    monkeypatch.setenv("HELENA_ENABLE_SEED_PROBE_SELECT", "1")
    monkeypatch.setattr(cli_module, "open_fleet_store", lambda _database: object())

    def fake_bootstrap(_store, _eligible, _catalog, **kwargs):
        captured.update(kwargs)
        return {"tasks": {}, "status": {}}

    monkeypatch.setattr(cli_module, "bootstrap_queue", fake_bootstrap)
    with pytest.raises(RuntimeError, match="review-owner"):
        command_bootstrap(
            _bootstrap_args(
                tmp_path, mode="select", benchmark_receipt=receipt_path
            )
        )
    assert (
        command_bootstrap(
            _bootstrap_args(
                tmp_path,
                mode="select",
                benchmark_receipt=receipt_path,
                review_owner="segmentation-review@campaign-x",
            )
        )
        == 0
    )
    authorization = captured["seed_probe"]["benchmark_authorization"]
    assert authorization["decision_receipt_sha256"] == receipt["receipt_sha256"]
    assert (
        captured["seed_probe"]["review_owner"]
        == "segmentation-review@campaign-x"
    )
    assert str(receipt_path) not in json.dumps(captured["seed_probe"])


def test_cli_shadow_remains_backward_compatible_without_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    monkeypatch.setattr(cli_module, "open_fleet_store", lambda _database: object())

    def fake_bootstrap(_store, _eligible, _catalog, **kwargs):
        captured.update(kwargs)
        return {"tasks": {}, "status": {}}

    monkeypatch.setattr(cli_module, "bootstrap_queue", fake_bootstrap)
    with pytest.raises(RuntimeError, match="only valid"):
        command_bootstrap(
            _bootstrap_args(
                tmp_path,
                mode="shadow",
                review_owner="must-not-be-silently-ignored",
            )
        )
    assert command_bootstrap(_bootstrap_args(tmp_path, mode="shadow")) == 0
    assert captured["seed_probe"]["mode"] == "shadow"
    assert "benchmark_authorization" not in captured["seed_probe"]


def test_select_source_lock_binds_hashes_to_the_uris_actually_read(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "source-lock.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    lock = normalize_source_content_lock(claim["source"])
    source_lock_schema = json.loads(
        (
            ROOT
            / "framework/contracts/schemas/source-content-lock-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator(source_lock_schema).validate(lock)

    run = store.prepare_probe_run(
        claim,
        [_candidate(1), _candidate(2)],
        claim["seed_probe"],
        _probe_fingerprint(tmp_path),
    )
    plan = build_probe_locked_plan(
        claim, run["trials"][0], claim["seed_probe"]
    )
    plan_schema = json.loads(
        (
            ROOT
            / "framework/contracts/schemas/"
            "seed-probe-locked-plan-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    registry = Registry().with_resource(
        source_lock_schema["$id"], Resource.from_contents(source_lock_schema)
    )
    Draft202012Validator(plan_schema, registry=registry).validate(plan)
    assert plan["source"]["ct_sha256"] == "0" * 64
    assert plan["source"]["m7_sha256"] == "1" * 64
    assert plan["source"]["source_content_lock"] == lock

    mutable = copy.deepcopy(claim["source"])
    mutable["m7_uri"] = "fixture://m7/mutable"
    with pytest.raises(ValueError, match="does not match"):
        normalize_source_content_lock(mutable)

    fake_digest = copy.deepcopy(claim["source"])
    fake_digest["m7_sha256"] = "x"
    fake_digest["source_content_lock"]["m7_sha256"] = "x"
    with pytest.raises(ValueError, match="lowercase m7 SHA-256"):
        normalize_source_content_lock(fake_digest)


def test_select_admission_rejects_a_probe_that_cannot_resume_full_growth(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "admission.sqlite")
    store.initialize()
    task = _task(_source(store))
    task["seed_probe"]["probe_parameters"]["step_size"] = 8
    with pytest.raises(ValueError, match="step_size.*full envelope"):
        store.create_tasks([task])
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        assert (
            connection.execute("SELECT COUNT(*) FROM probe_runs").fetchone()[0]
            == 0
        )


def test_sqlite_upgrade_refuses_to_invent_bindings_for_legacy_promotions(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-v7.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """CREATE TABLE probe_promotions (
                 promotion_id TEXT PRIMARY KEY,
                 decision_id TEXT NOT NULL UNIQUE,
                 winner_trial_id TEXT NOT NULL,
                 winner_probe_artifact_set_id TEXT NOT NULL,
                 continuation_task_id TEXT UNIQUE,
                 canonical_artifact_set_id TEXT,
                 surface_id TEXT,
                 state TEXT NOT NULL,
                 receipt_json TEXT NOT NULL,
                 receipt_sha256 TEXT NOT NULL,
                 created_at TEXT NOT NULL,
                 updated_at TEXT NOT NULL
               )"""
        )
        connection.execute(
            """INSERT INTO probe_promotions VALUES(
                 'legacy-promotion','legacy-decision','legacy-trial',
                 'legacy-artifact','legacy-task',NULL,NULL,'CONTINUING',
                 '{}',?, '2026-01-01T00:00:00Z',
                 '2026-01-01T00:00:00Z')""",
            ("a" * 64,),
        )
    with pytest.raises(RuntimeError, match="cannot auto-bind legacy"):
        FleetStore(database).initialize()
    with sqlite3.connect(database) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(probe_promotions)"
            )
        }
        assert {
            "continuation_attempt_id",
            "continuation_contract_sha256",
            "continuation_locked_plan_sha256",
        }.issubset(columns)
        assert connection.execute(
            "SELECT state FROM probe_promotions"
        ).fetchone()[0] == "CONTINUING"


def test_probe_run_uses_authoritative_source_and_policy(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "authority.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    fingerprint = _probe_fingerprint(tmp_path)

    forged_source = copy.deepcopy(claim)
    forged_source["source"]["source_snapshot_id"] = "forged-source"
    with pytest.raises(ValueError, match="authoritative parent source"):
        store.prepare_probe_run(
            forged_source,
            [_candidate()],
            claim["seed_probe"],
            fingerprint,
        )

    forged_policy = copy.deepcopy(claim["seed_probe"])
    forged_policy["maximum_attempts_per_candidate"] = 1
    forged_policy["maximum_total_probe_generations"] = (
        forged_policy["top_k"]
        * forged_policy["probe_parameters"]["generations"]
    )
    forged_task = copy.deepcopy(claim)
    forged_task["seed_probe"] = forged_policy
    with pytest.raises(ValueError, match="authoritative parent task"):
        store.prepare_probe_run(
            forged_task,
            [_candidate()],
            forged_policy,
            fingerprint,
        )
    with store.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM probe_runs").fetchone()[0]
            == 0
        )


def test_worker_rejects_candidate_uri_drift_before_reading_a_provider(
    tmp_path: Path,
) -> None:
    class NeverReadProvider:
        def __init__(self) -> None:
            self.called = False

        def discover(self, _task_value: dict) -> list[dict]:
            self.called = True
            raise AssertionError("candidate provider must not be read")

    store = FleetStore(tmp_path / "uri-drift.sqlite")
    store.initialize()
    task = _task(_source(store))
    task["candidate_discovery"] = {
        "provider": "vc3d-mcp",
        "prediction_uri": "fixture://m7/not-the-locked-source",
        "prediction_space": "ct_l0_xyz",
    }
    store.create_tasks([task])
    provider = NeverReadProvider()
    result = SegmentWorker(
        store,
        "probe-worker",
        provider,
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    ).run_one()
    assert result is not None
    assert result["status"] == "POLICY_REJECTED"
    assert provider.called is False
    with store.connect() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM probe_runs").fetchone()[0]
            == 0
        )


def test_ranked_select_rejects_before_provider_or_probe_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted rank-two select task cannot read candidates or begin probing."""
    class CountingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def discover(self, _task_value: dict) -> dict:
            self.calls += 1
            return {"candidates": [], "fixture": True, "ink_used": False}

    coordinator_calls = 0

    class ForbiddenCoordinator:
        def __init__(self, *_args, **_kwargs) -> None:
            nonlocal coordinator_calls
            coordinator_calls += 1
            raise AssertionError("ranked select must not construct a probe coordinator")

    store = FleetStore(tmp_path / "ranked-select.sqlite")
    store.initialize()
    persisted = _task(_source(store), mode="select")
    persisted["candidate_rank"] = 2
    assert store.create_tasks([persisted]) == (1, 1)
    provider = CountingProvider()
    monkeypatch.setattr(worker_module, "SeedProbeCoordinator", ForbiddenCoordinator)

    result = SegmentWorker(
        store,
        "probe-worker",
        provider,
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    ).run_one()

    assert result is not None
    assert result["status"] == "POLICY_REJECTED"
    assert result["reason"] == "INVALID_SEED_PROBE_TASK_CONTRACT"
    assert provider.calls == 0
    assert coordinator_calls == 0
    with pytest.raises(ValueError, match="select requires candidate_rank 1"):
        validate_seed_probe_task_contract(persisted)


@pytest.mark.parametrize(
    ("verdicts", "action", "winner"),
    [
        (("ELIGIBLE", "REJECTED"), "CONTINUE_WINNER", "trial-1"),
        (("ELIGIBLE", "ELIGIBLE"), "HUMAN_REVIEW", None),
        (("REJECTED", "REJECTED"), "REJECT_ALL", None),
        (("ELIGIBLE", "UNMEASURED"), "HUMAN_REVIEW", None),
        (("ELIGIBLE", "FAILED"), "HUMAN_REVIEW", None),
    ],
)
def test_probe_decision_matrix_abstains_unless_one_result_is_unambiguous(
    verdicts: tuple[str, ...], action: str, winner: str | None
) -> None:
    decision = decide_probe_run(_decision_run(*verdicts))
    assert decision["action"] == action
    assert decision["winner_trial_id"] == winner
    assert decision["ink_used"] is False


def test_probe_capability_is_required_at_claim_time(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store))])

    assert store.claim("legacy-worker", 60) is None
    claim = store.claim(
        "probe-worker",
        60,
        capabilities={"seed_probe_v1": True},
    )
    assert claim is not None
    assert claim["resource_requirements"]["seed_probe_required"] is True


def test_select_task_cannot_finalize_an_ordinary_plan_before_a_decision(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "select-finalize-bypass.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    locked_plan = {
        "schema": "test.locked_plan.v1",
        "task_id": claim["task_id"],
        "attempt_id": claim["attempt_id"],
    }
    locked_plan_sha256 = content_sha256(locked_plan)
    store.transition(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        "RUNNING",
        locked_plan=locked_plan,
    )
    surface = {
        "schema": "campaignx.segment_fleet_surface.v1",
        "surface_id": "surface-select-bypass",
        "task_id": claim["task_id"],
        "attempt_id": claim["attempt_id"],
        "source_snapshot_id": claim["source_snapshot_id"],
        "sample_id": claim["sample_id"],
        "owner": "malformed-worker",
        "artifact_sha256": "9" * 64,
        "artifact_uri": "fixture://surface/select-bypass",
        "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
        "sample_points": [[0.0, 0.0, 0.0]],
        "area_cm2": 1.0,
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
        "locked_plan_sha256": locked_plan_sha256,
        "ink_used": False,
    }
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "task_id": claim["task_id"],
        "attempt_id": claim["attempt_id"],
        "locked_plan_sha256": locked_plan_sha256,
        "files": {},
        "artifact_sha256": surface["artifact_sha256"],
        "bbox_xyz": surface["bbox_xyz"],
        "sample_points": surface["sample_points"],
        "area_cm2": surface["area_cm2"],
        "ink_used": False,
    }
    artifact_set_id = store.add_artifact_set(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        manifest,
        "fixture://staging/select-bypass",
    )
    store.transition(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        "FINALIZING",
    )
    with pytest.raises(RuntimeError, match="persisted probe decision"):
        store.finalize(
            claim["task_id"],
            claim["attempt_id"],
            claim["lease_token"],
            surface,
            artifact_set_id,
            "surface-qc@1.0.0",
        )
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM qc_jobs").fetchone()[0] == 0


def test_probe_gc_is_a_no_delete_dry_run_without_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "gc.sqlite"
    FleetStore(database).initialize()
    receipt = tmp_path / "PROBE_GC.json"
    result = command_probe_gc(
        SimpleNamespace(
            db=str(database),
            artifact_root=str(tmp_path / "artifacts"),
            limit=100,
            apply=False,
            expect_candidate_set_sha256=None,
            receipt=receipt,
        )
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "DRY_RUN"
    assert output["candidate_count"] == 0
    assert len(output["candidate_set_sha256"]) == 64
    assert output["deleted_count"] == 0
    assert json.loads(receipt.read_text())["mode"] == "DRY_RUN"
    assert not (tmp_path / "artifacts").exists()
    with pytest.raises(RuntimeError, match="exact.*candidate-set"):
        command_probe_gc(
            SimpleNamespace(
                db=str(database),
                artifact_root=str(tmp_path / "artifacts"),
                limit=100,
                apply=True,
                expect_candidate_set_sha256=None,
                receipt=None,
            )
        )


def test_probe_publication_rejects_path_components(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="unsafe probe sample_id"):
        store.publish_probe(
            tmp_path / "source",
            "../../escaped",
            "run",
            "trial",
            "artifact",
            {"artifact_sha256": "a" * 64, "files": {}},
        )
    assert not (tmp_path / "escaped").exists()


def test_parent_task_has_one_immutable_probe_budget_across_retries(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    candidates = [_candidate()]
    fingerprint = _probe_fingerprint(tmp_path)
    first = store.prepare_probe_run(
        claim, candidates, claim["seed_probe"], fingerprint
    )
    replay = store.prepare_probe_run(
        claim, candidates, claim["seed_probe"], fingerprint
    )
    assert replay["probe_run_id"] == first["probe_run_id"]

    with store.connect() as connection:
        connection.execute(
            "UPDATE tasks SET lease_expires_at=? WHERE task_id=?",
            ("2000-01-01T00:00:00Z", claim["task_id"]),
        )
    replacement = store.claim(
        "replacement-worker",
        60,
        task_id=claim["task_id"],
        capabilities={"seed_probe_v1": True},
    )
    assert replacement is not None
    recovered = store.prepare_probe_run(
        replacement,
        candidates,
        replacement["seed_probe"],
        fingerprint,
    )
    assert recovered["probe_run_id"] == first["probe_run_id"]

    drifted_candidates = [{**_candidate(), "score": 0.123}]
    with pytest.raises(RuntimeError, match="immutable probe run"):
        store.prepare_probe_run(
            replacement,
            drifted_candidates,
            replacement["seed_probe"],
            fingerprint,
        )
    with pytest.raises(RuntimeError, match="immutable probe run"):
        store.prepare_probe_run(
            replacement,
            candidates,
            replacement["seed_probe"],
            {**fingerprint, "build": "rolling-deploy"},
        )

    with store.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM probe_runs WHERE task_id=?",
                (claim["task_id"],),
            ).fetchone()[0]
            == 1
        )
        unique_columns = [
            [
                column["name"]
                for column in connection.execute(
                    f"PRAGMA index_info({index['name']})"
                )
            ]
            for index in connection.execute("PRAGMA index_list(probe_runs)")
            if bool(index["unique"])
        ]
    assert ["task_id"] in unique_columns


def test_probe_store_recomputes_plan_and_reserves_exact_namespace_uri(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    run = store.prepare_probe_run(
        claim,
        [_candidate()],
        claim["seed_probe"],
        _probe_fingerprint(tmp_path),
    )
    trial = store.claim_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        run["probe_run_id"],
        "probe-worker",
        60,
        {"seed_probe_v1": True},
    )
    assert trial is not None
    plan = build_probe_locked_plan(claim, trial, claim["seed_probe"])
    forged = copy.deepcopy(plan)
    forged["parameters"]["generations"] = 999
    with pytest.raises(ValueError, match="authoritative"):
        store.transition_probe_trial(
            claim["task_id"],
            claim["attempt_id"],
            claim["lease_token"],
            trial["probe_trial_id"],
            trial["probe_attempt_id"],
            trial["lease_token"],
            "RUNNING",
            locked_plan=forged,
        )
    with pytest.raises(ValueError, match="only direct probe transition"):
        store.transition_probe_trial(
            claim["task_id"],
            claim["attempt_id"],
            claim["lease_token"],
            trial["probe_trial_id"],
            trial["probe_attempt_id"],
            trial["lease_token"],
            "EVALUATING",
            locked_plan=plan,
        )
    store.transition_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        trial["probe_trial_id"],
        trial["probe_attempt_id"],
        trial["lease_token"],
        "RUNNING",
        locked_plan=plan,
    )
    required = ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json")
    files = {
        name: {"sha256": str(index) * 64, "size_bytes": index}
        for index, name in enumerate(required, start=1)
    }
    manifest = {
        "schema": "campaignx.seed_probe_artifact_set.v1",
        "probe_run_id": run["probe_run_id"],
        "probe_trial_id": trial["probe_trial_id"],
        "locked_plan_sha256": content_sha256(plan),
        "files": files,
        "artifact_sha256": content_sha256(files),
        "noncanonical": True,
        "ink_used": False,
    }
    artifact_set_id = store.reserve_probe_artifact(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        trial["probe_trial_id"],
        trial["probe_attempt_id"],
        trial["lease_token"],
        manifest,
    )
    with store.connect() as connection:
        reserved = connection.execute(
            """SELECT artifact_uri,state FROM probe_artifact_sets
                WHERE probe_artifact_set_id=?""",
            (artifact_set_id,),
        ).fetchone()
    expected_uri = str(
        (
            tmp_path
            / "artifacts"
            / "probes"
            / claim["sample_id"]
            / run["probe_run_id"]
            / trial["probe_trial_id"]
            / manifest["artifact_sha256"]
        ).resolve()
    )
    assert tuple(reserved) == (expected_uri, "RESERVED")
    with pytest.raises(ValueError, match="reserved namespace URI"):
        store.complete_probe_trial(
            claim["task_id"],
            claim["attempt_id"],
            claim["lease_token"],
            trial["probe_trial_id"],
            trial["probe_attempt_id"],
            trial["lease_token"],
            artifact_set_id,
            f"https://attacker.invalid/{manifest['artifact_sha256']}",
            {},
            {
                "verdict": "UNMEASURED",
                "ink_used": False,
                "profile_sha256": "a" * 64,
            },
            {},
        )


def test_probe_credentials_cannot_be_composed_across_parent_tasks(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = _source(store)
    store.create_tasks(
        [
            _task(source_id, cell_id="probe-cell-a"),
            _task(source_id, cell_id="probe-cell-b"),
        ]
    )
    first_parent = store.claim(
        "parent-a", 60, capabilities={"seed_probe_v1": True}
    )
    second_parent = store.claim(
        "parent-b", 60, capabilities={"seed_probe_v1": True}
    )
    assert first_parent is not None and second_parent is not None
    fingerprint = _probe_fingerprint(tmp_path)
    first_run = store.prepare_probe_run(
        first_parent,
        [_candidate()],
        first_parent["seed_probe"],
        fingerprint,
    )
    second_run = store.prepare_probe_run(
        second_parent,
        [_candidate()],
        second_parent["seed_probe"],
        fingerprint,
    )

    with pytest.raises(RuntimeError, match="leased parent task"):
        store.claim_probe_trial(
            first_parent["task_id"],
            first_parent["attempt_id"],
            first_parent["lease_token"],
            second_run["probe_run_id"],
            "probe-worker",
            60,
            {"seed_probe_v1": True},
        )
    with pytest.raises(RuntimeError, match="leased parent task"):
        store.begin_probe_promotion(
            first_parent["task_id"],
                first_parent["attempt_id"],
                first_parent["lease_token"],
                second_run["probe_run_id"],
                {},
            )

    child = store.claim_probe_trial(
        second_parent["task_id"],
        second_parent["attempt_id"],
        second_parent["lease_token"],
        second_run["probe_run_id"],
        "probe-worker",
        60,
        {"seed_probe_v1": True},
    )
    assert child is not None
    child_plan = build_probe_locked_plan(
        second_parent, child, second_parent["seed_probe"]
    )
    with pytest.raises(RuntimeError, match="stale trial lease"):
        store.transition_probe_trial(
            first_parent["task_id"],
            first_parent["attempt_id"],
            first_parent["lease_token"],
            child["probe_trial_id"],
            child["probe_attempt_id"],
            child["lease_token"],
            "RUNNING",
            locked_plan=child_plan,
        )

    files = {
        name: {"sha256": str(index) * 64, "size_bytes": index}
        for index, name in enumerate(
            ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json"),
            start=1,
        )
    }
    manifest = {
        "schema": "campaignx.seed_probe_artifact_set.v1",
        "probe_run_id": second_run["probe_run_id"],
        "probe_trial_id": child["probe_trial_id"],
        "locked_plan_sha256": "a" * 64,
        "files": files,
        "artifact_sha256": content_sha256(files),
        "noncanonical": True,
        "ink_used": False,
    }
    with pytest.raises(RuntimeError, match="stale trial lease"):
        store.reserve_probe_artifact(
            first_parent["task_id"],
            first_parent["attempt_id"],
            first_parent["lease_token"],
            child["probe_trial_id"],
            child["probe_attempt_id"],
            child["lease_token"],
            manifest,
        )
    with pytest.raises(RuntimeError, match="stale trial lease"):
        store.complete_probe_trial(
            first_parent["task_id"],
            first_parent["attempt_id"],
            first_parent["lease_token"],
            child["probe_trial_id"],
            child["probe_attempt_id"],
            child["lease_token"],
            "unreserved-artifact",
            "fixture://artifact",
            {},
            {
                "verdict": "UNMEASURED",
                "ink_used": False,
                "profile_sha256": "a" * 64,
            },
            {},
        )
    with pytest.raises(RuntimeError, match="stale trial lease"):
        store.fail_probe_trial(
            first_parent["task_id"],
            first_parent["attempt_id"],
            first_parent["lease_token"],
            child["probe_trial_id"],
            child["probe_attempt_id"],
            child["lease_token"],
            {"status": "CROSS_TASK_FORGERY", "ink_used": False},
            retryable=False,
        )

    store.fail_probe_trial(
        second_parent["task_id"],
        second_parent["attempt_id"],
        second_parent["lease_token"],
        child["probe_trial_id"],
        child["probe_attempt_id"],
        child["lease_token"],
        {"status": "FIXTURE_FAILURE", "ink_used": False},
        retryable=False,
    )
    decision = decide_probe_run(
        store.probe_run(second_run["probe_run_id"])
    )
    with pytest.raises(RuntimeError, match="leased parent task"):
        store.record_probe_decision(
            first_parent["task_id"],
            first_parent["attempt_id"],
            first_parent["lease_token"],
            second_run["probe_run_id"],
            decision,
        )
    recorded = store.record_probe_decision(
        second_parent["task_id"],
        second_parent["attempt_id"],
        second_parent["lease_token"],
        second_run["probe_run_id"],
        decision,
    )
    assert recorded["action"] == "HUMAN_REVIEW"
    assert first_run["task_id"] == first_parent["task_id"]


def test_expired_probe_lease_retries_only_that_trial_and_rejects_stale_owner(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    policy = claim["seed_probe"]
    run = store.prepare_probe_run(
        claim,
        [_candidate()],
        policy,
        _probe_fingerprint(tmp_path),
    )
    first = store.claim_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        run["probe_run_id"],
        "probe-worker",
        60,
        {"seed_probe_v1": True},
    )
    assert first is not None
    with store.connect() as connection:
        connection.execute(
            "UPDATE probe_trials SET lease_expires_at=? "
            "WHERE probe_trial_id=?",
            ("2000-01-01T00:00:00Z", first["probe_trial_id"]),
        )
    second = store.claim_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        run["probe_run_id"],
        "probe-worker",
        60,
        {"seed_probe_v1": True},
    )
    assert second is not None
    assert second["attempt_number"] == 2
    assert second["probe_trial_id"] == first["probe_trial_id"]
    first_plan = build_probe_locked_plan(claim, first, policy)
    with pytest.raises(RuntimeError, match="stale trial lease"):
        store.transition_probe_trial(
            claim["task_id"],
            claim["attempt_id"],
            claim["lease_token"],
            first["probe_trial_id"],
            first["probe_attempt_id"],
            first["lease_token"],
            "RUNNING",
            locked_plan=first_plan,
        )


def test_replayed_probe_bytes_are_idempotent_across_attempt_ids(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    run = store.prepare_probe_run(
        claim,
        [_candidate()],
        claim["seed_probe"],
        _probe_fingerprint(tmp_path),
    )
    first = store.claim_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        run["probe_run_id"],
        "probe-worker",
        60,
        {"seed_probe_v1": True},
    )
    assert first is not None
    first_plan = build_probe_locked_plan(claim, first, claim["seed_probe"])
    store.transition_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        first["probe_trial_id"],
        first["probe_attempt_id"],
        first["lease_token"],
        "RUNNING",
        locked_plan=first_plan,
    )
    files = {
        name: {"sha256": str(index) * 64, "size_bytes": index}
        for index, name in enumerate(
            ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json"),
            start=1,
        )
    }

    def manifest() -> dict:
        return {
            "schema": "campaignx.seed_probe_artifact_set.v1",
            "probe_run_id": run["probe_run_id"],
            "probe_trial_id": first["probe_trial_id"],
            "locked_plan_sha256": content_sha256(first_plan),
            "files": files,
            "artifact_sha256": content_sha256(files),
            "noncanonical": True,
            "ink_used": False,
        }

    first_id = store.reserve_probe_artifact(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        first["probe_trial_id"],
        first["probe_attempt_id"],
        first["lease_token"],
        manifest(),
    )
    with store.connect() as connection:
        connection.execute(
            "UPDATE probe_trials SET state='PENDING',worker_id=NULL,"
            "lease_token=NULL,lease_expires_at=NULL,"
            "active_probe_attempt_id=NULL WHERE probe_trial_id=?",
            (first["probe_trial_id"],),
        )
        connection.execute(
            "UPDATE probe_attempts SET state='LEASE_EXPIRED' "
            "WHERE probe_attempt_id=?",
            (first["probe_attempt_id"],),
        )
    second = store.claim_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        run["probe_run_id"],
        "probe-worker",
        60,
        {"seed_probe_v1": True},
    )
    assert second is not None
    store.transition_probe_trial(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        second["probe_trial_id"],
        second["probe_attempt_id"],
        second["lease_token"],
        "RUNNING",
        locked_plan=first_plan,
    )
    second_id = store.reserve_probe_artifact(
        claim["task_id"],
        claim["attempt_id"],
        claim["lease_token"],
        second["probe_trial_id"],
        second["probe_attempt_id"],
        second["lease_token"],
        manifest(),
    )
    assert second_id == first_id


@pytest.mark.parametrize("mode", ["shadow", "select"])
def test_fixture_probe_never_catalogues_micro_growth_and_full_grow_finishes(
    tmp_path: Path, mode: str
) -> None:
    store = FleetStore(tmp_path / f"{mode}.sqlite")
    store.initialize()
    top_k = 1 if mode == "shadow" else 2
    store.create_tasks([_task(_source(store), mode=mode, top_k=top_k)])
    worker = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        (
            FixtureGrowExecutor()
            if mode == "shadow"
            else OneEligibleOneRejectedExecutor()
        ),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    )

    result = worker.run_one()
    assert result is not None
    assert result["status"] == "QC_PENDING"
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM artifact_sets").fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM probe_artifact_sets"
            ).fetchone()[0]
            == top_k
        )
        assert connection.execute("SELECT COUNT(*) FROM qc_jobs").fetchone()[0] == 1
        promotions = [
            dict(row)
            for row in connection.execute(
                "SELECT state,canonical_artifact_set_id,surface_id "
                "FROM probe_promotions"
            )
        ]
        locked_plan = json.loads(
            connection.execute(
                "SELECT locked_plan_json FROM attempts"
            ).fetchone()[0]
        )
        emitted_contracts = {
            "seed-probe-artifact-set-v1.schema.json": [
                json.loads(row["manifest_json"])
                for row in connection.execute(
                    "SELECT manifest_json FROM probe_artifact_sets"
                )
            ],
            "seed-probe-evaluation-v1.schema.json": [
                json.loads(row["result_json"])
                for row in connection.execute(
                    "SELECT result_json FROM probe_evaluations"
                )
            ],
            "seed-probe-decision-v1.schema.json": [
                json.loads(row["receipt_json"])
                for row in connection.execute(
                    "SELECT receipt_json FROM probe_decisions"
                )
            ],
            "seed-probe-promotion-v1.schema.json": [
                json.loads(row["receipt_json"])
                for row in connection.execute(
                    "SELECT receipt_json FROM probe_promotions"
                )
            ],
        }
    for schema_name, instances in emitted_contracts.items():
        if schema_name == "seed-probe-promotion-v1.schema.json" and mode == "shadow":
            assert instances == []
            continue
        assert instances
        for instance in instances:
            _validate_contract(instance, schema_name)
    if mode == "select":
        assert len(promotions) == 1
        assert promotions[0]["state"] == "PROMOTED"
        assert promotions[0]["canonical_artifact_set_id"]
        assert promotions[0]["surface_id"]
        assert locked_plan["resume_from"]
        assert locked_plan["seed_probe_decision"]["action"] == "CONTINUE_WINNER"
        promotion_receipt = emitted_contracts[
            "seed-probe-promotion-v1.schema.json"
        ][0]
        continuation_contract = promotion_receipt["continuation_contract"]
        validate_probe_continuation_contract(
            continuation_contract,
            locked_plan,
            task_id=locked_plan["task_id"],
            attempt_id=locked_plan["attempt_id"],
        )
        wrong_seed = copy.deepcopy(locked_plan)
        wrong_seed["selected_seed"]["x"] += 1
        with pytest.raises(RuntimeError, match="selected probe seed"):
            validate_probe_continuation_contract(
                continuation_contract,
                wrong_seed,
                task_id=locked_plan["task_id"],
                attempt_id=locked_plan["attempt_id"],
            )
        wrong_resume = copy.deepcopy(locked_plan)
        wrong_resume["resume_artifact"]["artifact_uri"] = (
            "fixture://probe/not-the-winner"
        )
        with pytest.raises(RuntimeError, match="selected probe artifact"):
            validate_probe_continuation_contract(
                continuation_contract,
                wrong_resume,
                task_id=locked_plan["task_id"],
                attempt_id=locked_plan["attempt_id"],
            )
        receipt = json.loads(
            next(
                (tmp_path / "runs").rglob("COST_AWARE_PLANNER_RECEIPT.json")
            ).read_text(encoding="utf-8")
        )
        assert receipt["route"] == "DETERMINISTIC_PROBE_WINNER"
        assert receipt["provider_call_count"] == 0
    else:
        assert promotions == []
        assert "resume_from" not in locked_plan


def test_select_promotion_recovers_an_exact_finalizer_dependency_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FleetStore(tmp_path / "promotion-recovery.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    artifact_root = tmp_path / "artifacts"
    run_root = tmp_path / "runs"

    def worker() -> SegmentWorker:
        return SegmentWorker(
            store,
            "probe-worker",
            RecordedSeedProvider(),
            CostAwareSegmentationPlanner(
                cache_root=tmp_path / "planner-cache"
            ),
            OneEligibleOneRejectedExecutor(),
            run_root,
            artifact_root,
            "fixture-surface-qc@1.0.0",
            lease_seconds=60,
            seed_probe_support=True,
        )

    original_finalizer = worker_module.finalize_surface

    def missing_numpy(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'numpy'")

    monkeypatch.setattr(worker_module, "finalize_surface", missing_numpy)
    failed = worker().run_one()
    assert failed is not None
    assert failed["status"] == "FINALIZATION_FAILED"
    with store.connect() as connection:
        task = connection.execute(
            "SELECT task_id,active_attempt_id FROM tasks"
        ).fetchone()
        first_attempt_id = task["active_attempt_id"]
        probe_attempt_count = connection.execute(
            "SELECT COUNT(*) FROM probe_attempts"
        ).fetchone()[0]
        assert connection.execute(
            "SELECT state FROM probe_promotions"
        ).fetchone()[0] == "FAILED"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "PROMOTION_FAILED"

    recovered = store.recover_terminal_finalizer_dependency(
        task["task_id"],
        first_attempt_id,
        retry_delay_seconds=1,
    )
    assert recovered["status"] == "RECOVERED_FINALIZER_DEPENDENCY"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT state FROM probe_promotions"
        ).fetchone()[0] == "CONTINUING"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "CONTINUING"
        assert connection.execute(
            "SELECT state FROM probe_artifact_sets "
            "WHERE probe_artifact_set_id=("
            "SELECT winner_probe_artifact_set_id FROM probe_promotions)"
        ).fetchone()[0] == "WINNER_RETAINED"
        connection.execute(
            "UPDATE tasks SET retry_after=? WHERE task_id=?",
            ("2000-01-01T00:00:00Z", task["task_id"]),
        )

    monkeypatch.setattr(worker_module, "finalize_surface", original_finalizer)
    completed = worker().run_one()
    assert completed is not None
    assert completed["status"] == "QC_PENDING"
    with store.connect() as connection:
        promotion = connection.execute(
            """SELECT state,continuation_attempt_id,
                      continuation_locked_plan_sha256
                 FROM probe_promotions"""
        ).fetchone()
        assert promotion["state"] == "PROMOTED"
        assert promotion["continuation_attempt_id"] != first_attempt_id
        assert len(promotion["continuation_locked_plan_sha256"]) == 64
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM probe_attempts"
            ).fetchone()[0]
            == probe_attempt_count
        )
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 1
        assert (
            connection.execute("SELECT COUNT(*) FROM artifact_sets").fetchone()[0]
            == 1
        )


def test_transient_winner_materialization_requeues_and_reuses_probe_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FleetStore(tmp_path / "materialization-retry.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    original_materialize = LocalArtifactStore.materialize_probe
    calls = 0

    def flaky_materialize(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("object-store read timed out")
        return original_materialize(self, *args, **kwargs)

    monkeypatch.setattr(
        LocalArtifactStore, "materialize_probe", flaky_materialize
    )

    def worker() -> SegmentWorker:
        return SegmentWorker(
            store,
            "probe-worker",
            RecordedSeedProvider(),
            CostAwareSegmentationPlanner(
                cache_root=tmp_path / "planner-cache"
            ),
            OneEligibleOneRejectedExecutor(),
            tmp_path / "runs",
            tmp_path / "artifacts",
            "fixture-surface-qc@1.0.0",
            lease_seconds=60,
            source_retry_delay_seconds=1,
            seed_probe_support=True,
        )

    retryable = worker().run_one()
    assert retryable is not None
    assert retryable["status"] == "RETRYABLE_PROBE_ARTIFACT_UNAVAILABLE"
    with store.connect() as connection:
        task_id = connection.execute("SELECT task_id FROM tasks").fetchone()[0]
        assert connection.execute(
            "SELECT state FROM tasks"
        ).fetchone()[0] == "PENDING"
        assert connection.execute(
            "SELECT state FROM attempts"
        ).fetchone()[0] == "RETRYABLE_PROBE_ARTIFACT_UNAVAILABLE"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "CONTINUATION_QUEUED"
        assert connection.execute(
            "SELECT COUNT(*) FROM probe_promotions"
        ).fetchone()[0] == 0
        probe_attempt_count = connection.execute(
            "SELECT COUNT(*) FROM probe_attempts"
        ).fetchone()[0]
        connection.execute(
            "UPDATE tasks SET retry_after=? WHERE task_id=?",
            ("2000-01-01T00:00:00Z", task_id),
        )

    completed = worker().run_one()
    assert completed is not None and completed["status"] == "QC_PENDING"
    with store.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM probe_attempts"
            ).fetchone()[0]
            == probe_attempt_count
        )
        assert connection.execute(
            "SELECT state FROM probe_promotions"
        ).fetchone()[0] == "PROMOTED"


def test_corrupt_winner_materialization_stops_in_review_without_a_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FleetStore(tmp_path / "materialization-review.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])

    def corrupt_materialize(*_args, **_kwargs):
        raise RuntimeError("published artifact hash mismatch")

    monkeypatch.setattr(
        LocalArtifactStore, "materialize_probe", corrupt_materialize
    )
    result = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        OneEligibleOneRejectedExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    ).run_one()
    assert result is not None
    assert result["status"] == "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE"
    assert result["failure_class"] == "SOURCE_FAILURE"
    assert result["reason"] == "WINNER_ARTIFACT_MATERIALIZATION_FAILED"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT state FROM tasks"
        ).fetchone()[0] == "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "REVIEW_PENDING"
        assert {
            row[0]
            for row in connection.execute(
                "SELECT state FROM probe_artifact_sets"
            )
        } == {"REVIEW_RETAINED"}
        assert connection.execute(
            "SELECT COUNT(*) FROM probe_promotions"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0


def test_terminal_planner_failure_closes_an_unpromoted_select_winner(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "planner-terminal.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    result = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        RejectingV2Planner(),
        OneEligibleOneRejectedExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    ).run_one()
    assert result is not None and result["status"] == "POLICY_REJECTED"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "CONTINUATION_FAILED"
        assert {
            row[0]
            for row in connection.execute(
                "SELECT state FROM probe_artifact_sets"
            )
        } == {"REVIEW_RETAINED"}
        assert connection.execute(
            "SELECT COUNT(*) FROM probe_promotions"
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM events "
            "WHERE event_type='PROBE_CONTINUATION_FAILED'"
        ).fetchone()[0] == 1


def test_transient_finalizer_outage_preserves_select_promotion_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FleetStore(tmp_path / "finalization-retry.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    original_finalizer = worker_module.finalize_surface
    calls = 0

    def flaky_finalizer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("S3 promotion timed out")
        return original_finalizer(*args, **kwargs)

    monkeypatch.setattr(worker_module, "finalize_surface", flaky_finalizer)

    def worker() -> SegmentWorker:
        return SegmentWorker(
            store,
            "probe-worker",
            RecordedSeedProvider(),
            CostAwareSegmentationPlanner(
                cache_root=tmp_path / "planner-cache"
            ),
            OneEligibleOneRejectedExecutor(),
            tmp_path / "runs",
            tmp_path / "artifacts",
            "fixture-surface-qc@1.0.0",
            lease_seconds=60,
            seed_probe_support=True,
            finalization_retry_delay_seconds=1,
        )

    retryable = worker().run_one()
    assert retryable is not None
    assert retryable["status"] == "RETRYABLE_FINALIZATION_UNAVAILABLE"
    with store.connect() as connection:
        task_id = connection.execute("SELECT task_id FROM tasks").fetchone()[0]
        first_attempt_id = connection.execute(
            "SELECT attempt_id FROM attempts"
        ).fetchone()[0]
        assert connection.execute(
            "SELECT state FROM tasks"
        ).fetchone()[0] == "PENDING"
        assert connection.execute(
            "SELECT state FROM attempts"
        ).fetchone()[0] == "RETRYABLE_FINALIZATION_UNAVAILABLE"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "CONTINUING"
        promotion = connection.execute(
            "SELECT state,continuation_attempt_id FROM probe_promotions"
        ).fetchone()
        assert tuple(promotion) == ("CONTINUING", first_attempt_id)
        probe_attempt_count = connection.execute(
            "SELECT COUNT(*) FROM probe_attempts"
        ).fetchone()[0]
        connection.execute(
            "UPDATE tasks SET retry_after=? WHERE task_id=?",
            ("2000-01-01T00:00:00Z", task_id),
        )

    completed = worker().run_one()
    assert completed is not None and completed["status"] == "QC_PENDING"
    with store.connect() as connection:
        promotion = connection.execute(
            "SELECT state,continuation_attempt_id FROM probe_promotions"
        ).fetchone()
        assert promotion["state"] == "PROMOTED"
        assert promotion["continuation_attempt_id"] != first_attempt_id
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM probe_attempts"
            ).fetchone()[0]
            == probe_attempt_count
        )
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 1


def test_finalization_retry_budget_exhaustion_closes_the_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FleetStore(tmp_path / "finalization-cap.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])

    def unavailable_finalizer(*_args, **_kwargs):
        raise TimeoutError("database connection timed out")

    monkeypatch.setattr(
        worker_module, "finalize_surface", unavailable_finalizer
    )
    result = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        OneEligibleOneRejectedExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
        finalization_max_requeues=0,
    ).run_one()
    assert result is not None
    assert result["status"] == "FINALIZATION_FAILED"
    assert result["reason"] == "TRANSIENT_FINALIZATION_RETRY_BUDGET_EXHAUSTED"
    with store.connect() as connection:
        assert connection.execute(
            "SELECT state FROM tasks"
        ).fetchone()[0] == "FINALIZATION_FAILED"
        assert connection.execute(
            "SELECT state FROM probe_promotions"
        ).fetchone()[0] == "FAILED"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "PROMOTION_FAILED"
        assert {
            row[0]
            for row in connection.execute(
                "SELECT state FROM probe_artifact_sets"
            )
        } == {"REVIEW_RETAINED"}


def test_finalization_retry_budget_allows_exactly_the_configured_requeues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FleetStore(tmp_path / "finalization-cap-two.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])

    def unavailable_finalizer(*_args, **_kwargs):
        raise TimeoutError("object store connection timed out")

    monkeypatch.setattr(
        worker_module, "finalize_surface", unavailable_finalizer
    )
    worker = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        OneEligibleOneRejectedExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
        finalization_retry_delay_seconds=1,
        finalization_max_requeues=2,
    )

    statuses: list[str] = []
    for run_index in range(3):
        result = worker.run_one()
        assert result is not None
        statuses.append(result["status"])
        if run_index < 2:
            with store.connect() as connection:
                connection.execute(
                    "UPDATE tasks SET retry_after=?",
                    ("2000-01-01T00:00:00Z",),
                )

    assert statuses == [
        "RETRYABLE_FINALIZATION_UNAVAILABLE",
        "RETRYABLE_FINALIZATION_UNAVAILABLE",
        "FINALIZATION_FAILED",
    ]
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM attempts "
            "WHERE state='RETRYABLE_FINALIZATION_UNAVAILABLE'"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM attempts "
            "WHERE state='FINALIZATION_FAILED'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT state FROM probe_promotions"
        ).fetchone()[0] == "FAILED"
        assert connection.execute(
            "SELECT state FROM probe_runs"
        ).fetchone()[0] == "PROMOTION_FAILED"


def test_probe_gc_expires_only_due_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "retention.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    artifact_root = tmp_path / "artifacts"
    result = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        OneEligibleOneRejectedExecutor(),
        tmp_path / "runs",
        artifact_root,
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    ).run_one()
    assert result is not None and result["status"] == "QC_PENDING"
    canonical_uri = Path(result["surface"]["artifact_uri"])
    assert canonical_uri.is_dir()

    with store.connect() as connection:
        states = {
            row["state"]
            for row in connection.execute(
                "SELECT state FROM probe_artifact_sets"
            )
        }
        assert states == {"LOSER_RETAINED", "PROMOTED_RETAINED"}
        connection.execute(
            "UPDATE probe_artifact_sets "
            "SET retain_until='2000-01-01T00:00:00Z'"
        )

    due = store.probe_artifacts_due_for_gc(limit=100)
    assert len(due) == 2
    first = due[0]
    deletion = LocalArtifactStore(artifact_root).delete_probe(
        first["artifact_uri"], first["manifest"]
    )
    assert deletion["deleted"] is True
    store.mark_probe_artifact_expired(
        first["probe_artifact_set_id"], first["artifact_uri"]
    )

    with store.connect() as connection:
        expired = connection.execute(
            "SELECT state,deleted_at FROM probe_artifact_sets "
            "WHERE probe_artifact_set_id=?",
            (first["probe_artifact_set_id"],),
        ).fetchone()
        assert tuple(expired) == ("EXPIRED", expired["deleted_at"])
        assert expired["deleted_at"] is not None
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 1
    assert canonical_uri.is_dir()
    assert len(store.probe_artifacts_due_for_gc(limit=100)) == 1


def test_two_eligible_select_probes_abstain_without_canonical_surface(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "review.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    worker = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    )

    result = worker.run_one()
    assert result is not None
    assert result["status"] == "PROBE_REVIEW_PENDING"
    assert result["decision"]["reason"] == "MULTIPLE_GEOMETRY_ELIGIBLE_PROBES"
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM surfaces").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM artifact_sets").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM probe_artifact_sets"
            ).fetchone()[0]
            == 2
        )
        assert connection.execute("SELECT COUNT(*) FROM qc_jobs").fetchone()[0] == 0


def test_failed_probe_trials_are_operational_not_scientific_review(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "failed-probes.sqlite")
    store.initialize()
    store.create_tasks([_task(_source(store), mode="select", top_k=2)])
    result = SegmentWorker(
        store,
        "probe-worker",
        RecordedSeedProvider(),
        CostAwareSegmentationPlanner(cache_root=tmp_path / "planner-cache"),
        AlwaysFailingProbeExecutor(),
        tmp_path / "runs",
        tmp_path / "artifacts",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        seed_probe_support=True,
    ).run_one()
    assert result is not None
    assert result["status"] == "PROBE_TECHNICAL_FAILURE"
    assert result["failure_class"] == "WORKER_FAILURE"
    with store.connect() as connection:
        assert connection.execute("SELECT state FROM tasks").fetchone()[0] == (
            "PROBE_TECHNICAL_FAILURE")


def test_select_does_not_call_a_single_available_candidate_a_comparison() -> None:
    run = _decision_run("ELIGIBLE", "REJECTED")
    run["trials"] = run["trials"][:1]
    decision = decide_probe_run(run)
    assert decision["action"] == "HUMAN_REVIEW"
    assert decision["reason"] == "INSUFFICIENT_CANDIDATES_FOR_COMPARISON"


def test_shadow_does_not_overstate_a_single_preflight_as_a_winner() -> None:
    run = _decision_run("ELIGIBLE", "REJECTED")
    run["trials"] = run["trials"][:1]
    run["policy"] = default_seed_probe_policy(mode="shadow", top_k=1)
    decision = decide_probe_run(run)
    assert decision["action"] == "HUMAN_REVIEW"
    assert decision["winner_trial_id"] is None
    assert decision["reason"] == "INSUFFICIENT_CANDIDATES_FOR_COMPARISON"


def test_store_rejects_a_forged_probe_winner(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "forged.sqlite")
    store.initialize()
    claim = _claim_parent(store, _task(_source(store)))
    run = store.prepare_probe_run(
        claim,
        [_candidate(1), _candidate(2)],
        default_seed_probe_policy(mode="select", top_k=2),
        _probe_fingerprint(tmp_path),
    )
    with store.connect() as connection:
        for trial in run["trials"]:
            evaluation = {
                "verdict": "ELIGIBLE",
                "geometry_qc_state": "GEOMETRY_CERTIFIED",
            }
            connection.execute(
                "UPDATE probe_trials SET state='SUCCEEDED' "
                "WHERE probe_trial_id=?",
                (trial["probe_trial_id"],),
            )
            connection.execute(
                "INSERT INTO probe_attempts("
                "probe_attempt_id,probe_trial_id,attempt_number,worker_id,"
                "state,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?)",
                (
                    f"attempt-{trial['candidate_rank']}",
                    trial["probe_trial_id"],
                    1,
                    "fixture",
                    "COMPLETED",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO probe_artifact_sets("
                "probe_artifact_set_id,probe_trial_id,probe_attempt_id,"
                "manifest_json,manifest_sha256,artifact_uri,state,created_at"
                ") VALUES(?,?,?,?,?,?,?,?)",
                (
                    f"artifact-{trial['candidate_rank']}",
                    trial["probe_trial_id"],
                    f"attempt-{trial['candidate_rank']}",
                    "{}",
                    "a" * 64,
                    "fixture://probe",
                    "AVAILABLE",
                    "2026-01-01T00:00:00Z",
                ),
            )
            connection.execute(
                "INSERT INTO probe_evaluations("
                "evaluation_id,probe_trial_id,probe_artifact_set_id,"
                "profile_id,profile_sha256,verdict,result_json,"
                "result_sha256,created_at"
                ") VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    f"evaluation-{trial['candidate_rank']}",
                    trial["probe_trial_id"],
                    f"artifact-{trial['candidate_rank']}",
                    "tifxyz-geometry-gate-probe-v1",
                    "b" * 64,
                    "ELIGIBLE",
                    json.dumps(evaluation),
                    content_sha256(evaluation),
                    "2026-01-01T00:00:00Z",
                ),
            )
    snapshot = store.probe_run(run["probe_run_id"])
    expected = decide_probe_run(snapshot)
    assert expected["action"] == "HUMAN_REVIEW"
    forged = {
        **expected,
        "action": "CONTINUE_WINNER",
        "winner_trial_id": snapshot["trials"][0]["probe_trial_id"],
        "reason": "FORGED",
    }
    with pytest.raises(RuntimeError, match="persisted terminal evidence"):
        store.record_probe_decision(
            claim["task_id"],
            claim["attempt_id"],
            claim["lease_token"],
            run["probe_run_id"],
            forged,
        )
