from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet import cli as cli_module
from fleet.cli import (
    _worker,
    build_parser,
    command_bootstrap,
    command_seed_probe_benchmark_arm,
)
from fleet.store import FleetStore


def _authorization() -> dict:
    return {
        "benchmark_id": "isolated-benchmark-v1",
        "benchmark_spec_sha256": "a" * 64,
        "cells": [
            {"sample_id": "scroll-a", "cell_id": "cell-a"},
            {"sample_id": "scroll-b", "cell_id": "cell-b"},
            {"sample_id": "scroll-c", "cell_id": "cell-c"},
        ],
    }


class _BenchmarkStore:
    path = "isolated.sqlite"

    def __init__(self) -> None:
        self.created: list[list[dict]] = []

    def initialize(self) -> None:
        pass

    def status(self) -> dict:
        return {"tasks": {}, "surfaces": 0}

    def snapshots(self, samples: set[str]) -> list[dict]:
        assert samples == {"scroll-a", "scroll-b", "scroll-c"}
        return [{"sample_id": sample} for sample in sorted(samples)]

    def create_tasks(self, tasks: list[dict]) -> tuple[int, int]:
        self.created.append(tasks)
        return len(tasks), len(tasks)


def _benchmark_args(tmp_path: Path, arm: str) -> list[str]:
    return [
        "seed-probe-benchmark-arm",
        "--db",
        str(tmp_path / "isolated.sqlite"),
        "--confirm-isolated-nonproduction",
        "--benchmark-spec",
        str(tmp_path / "spec.json"),
        "--arm",
        arm,
        "--eligible",
        str(tmp_path / "eligible.json"),
        "--catalog",
        str(tmp_path / "catalog.json"),
        "--grid-version",
        "causal-grid-v1",
    ]


@pytest.mark.parametrize("arm", ["baseline", "closed_loop"])
def test_isolated_benchmark_cli_loads_only_preregistered_spec_and_writes_one_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    store = _BenchmarkStore()
    captured: dict = {}
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    monkeypatch.setattr(cli_module, "open_fleet_store", lambda _db: store)
    monkeypatch.setattr(
        cli_module, "load_seed_probe_benchmark_spec", lambda _path: _authorization()
    )
    monkeypatch.setattr(
        cli_module,
        "load_seed_probe_benchmark_receipt",
        lambda _path: (_ for _ in ()).throw(AssertionError("production receipt")),
    )

    def import_sources(store_arg, eligible, samples, *, verify):
        assert store_arg is store
        assert samples == {"scroll-a", "scroll-b", "scroll-c"}
        assert verify is True
        assert eligible == tmp_path / "eligible.json"
        return {sample: f"source-{sample}" for sample in samples}

    monkeypatch.setattr(cli_module, "bootstrap_sources", import_sources)
    monkeypatch.setattr(
        cli_module,
        "import_catalog",
        lambda store_arg, catalog, sources: (
            {"imported": 0, "catalog": str(catalog), "sources": sources}
        ),
    )

    def generate(store_arg, snapshots, **kwargs):
        captured["store"] = store_arg
        captured["snapshots"] = snapshots
        captured.update(kwargs)
        return [
            {"sample_id": cell["sample_id"], "cell_id": cell["cell_id"]}
            for cell in kwargs["benchmark_execution_authorization"]["cells"]
        ]

    monkeypatch.setattr(
        cli_module, "generate_seed_probe_benchmark_arm_tasks", generate
    )
    monkeypatch.setattr(
        cli_module,
        "default_seed_probe_policy",
        lambda **kwargs: {"isolated": True, **kwargs},
    )

    (tmp_path / "eligible.json").write_text("[]", encoding="utf-8")
    (tmp_path / "catalog.json").write_text("{}", encoding="utf-8")
    args = build_parser(ROOT).parse_args(_benchmark_args(tmp_path, arm))
    assert command_seed_probe_benchmark_arm(args) == 0

    assert len(store.created) == 1  # exactly one transactional arm insertion
    assert {
        (task["sample_id"], task["cell_id"]) for task in store.created[0]
    } == {("scroll-a", "cell-a"), ("scroll-b", "cell-b"), ("scroll-c", "cell-c")}
    assert captured["arm"] == arm
    assert captured["benchmark_execution_authorization"] == _authorization()
    assert captured["generation_options"]["max_tasks"] == 1
    assert captured["generation_options"]["queued_reason"] == (
        "isolated-seed-probe-causal-benchmark-v1"
    )
    if arm == "baseline":
        assert captured["seed_probe"] is None
    else:
        assert captured["seed_probe"]["mode"] == "select"
        assert captured["seed_probe"]["benchmark_authorization"] == _authorization()


def test_isolated_benchmark_cli_refuses_production_select_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HELENA_ENABLE_SEED_PROBE_SELECT", "1")
    (tmp_path / "eligible.json").write_text("[]", encoding="utf-8")
    (tmp_path / "catalog.json").write_text("{}", encoding="utf-8")
    args = build_parser(ROOT).parse_args(_benchmark_args(tmp_path, "baseline"))
    with pytest.raises(RuntimeError, match="separate process"):
        command_seed_probe_benchmark_arm(args)


def test_production_select_requires_and_binds_review_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = [
        "bootstrap",
        "--db",
        str(tmp_path / "production.sqlite"),
        "--eligible",
        str(tmp_path / "eligible.json"),
        "--catalog",
        str(tmp_path / "catalog.json"),
        "--receipt",
        str(tmp_path / "bootstrap-receipt.json"),
        "--seed-probe-mode",
        "select",
        "--seed-probe-benchmark-receipt",
        str(tmp_path / "decision.json"),
    ]
    monkeypatch.setenv("HELENA_ENABLE_SEED_PROBE_SELECT", "1")
    monkeypatch.setattr(
        cli_module, "load_seed_probe_benchmark_receipt", lambda _path: {"ok": True}
    )
    with pytest.raises(RuntimeError, match="seed-probe-review-owner"):
        command_bootstrap(build_parser(ROOT).parse_args(base))

    captured: dict = {}
    monkeypatch.setattr(cli_module, "open_fleet_store", lambda _db: object())

    def policy(**kwargs):
        captured.update(kwargs)
        return {"policy": "captured"}

    monkeypatch.setattr(cli_module, "default_seed_probe_policy", policy)
    monkeypatch.setattr(
        cli_module,
        "bootstrap_queue",
        lambda _store, _eligible, _catalog, **kwargs: (
            captured.update(kwargs) or {"tasks": {}, "status": {}}
        ),
    )
    args = build_parser(ROOT).parse_args(
        [*base, "--seed-probe-review-owner", "review-ops@example.test"]
    )
    assert command_bootstrap(args) == 0
    assert captured["review_owner"] == "review-ops@example.test"


def test_isolated_benchmark_cli_requires_operator_isolation_confirmation(
    tmp_path: Path,
) -> None:
    arguments = _benchmark_args(tmp_path, "baseline")
    arguments.remove("--confirm-isolated-nonproduction")
    with pytest.raises(SystemExit):
        build_parser(ROOT).parse_args(arguments)


def test_isolated_benchmark_cli_rejects_preexisting_tasks_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NonemptyStore(_BenchmarkStore):
        def status(self) -> dict:
            return {"tasks": {"PENDING": 1}, "surfaces": 0}

    store = NonemptyStore()
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    monkeypatch.setattr(cli_module, "open_fleet_store", lambda _db: store)
    monkeypatch.setattr(
        cli_module, "load_seed_probe_benchmark_spec", lambda _path: _authorization()
    )
    (tmp_path / "eligible.json").write_text("[]", encoding="utf-8")
    (tmp_path / "catalog.json").write_text("{}", encoding="utf-8")
    args = build_parser(ROOT).parse_args(_benchmark_args(tmp_path, "baseline"))
    with pytest.raises(RuntimeError, match="no pre-existing tasks"):
        command_seed_probe_benchmark_arm(args)


def test_isolated_benchmark_worker_advertises_only_loaded_spec_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    monkeypatch.setattr(
        cli_module, "load_seed_probe_benchmark_spec", lambda _path: _authorization()
    )
    args = build_parser(ROOT).parse_args(
        [
            "worker",
            "run",
            "--db",
            str(tmp_path / "worker.sqlite"),
            "--worker-id",
            "benchmark-worker",
            "--run-root",
            str(tmp_path / "runs"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--allow-local-artifacts",
            "--qc-profile-id",
            "fixture-qc@v1",
            "--fixture-grow",
            "--allow-fixture",
            "--planner",
            "deterministic-v2",
            "--isolated-benchmark-spec",
            str(tmp_path / "spec.json"),
            "--confirm-isolated-nonproduction",
        ]
    )
    worker = _worker(args)
    assert worker.worker_capabilities["benchmark_spec_sha256"] == "a" * 64
    assert worker.worker_capabilities["seed_probe_v1"] is False


def test_isolated_benchmark_worker_requires_isolation_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    args = build_parser(ROOT).parse_args(
        [
            "worker",
            "run",
            "--db",
            str(tmp_path / "worker.sqlite"),
            "--worker-id",
            "benchmark-worker",
            "--run-root",
            str(tmp_path / "runs"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--allow-local-artifacts",
            "--qc-profile-id",
            "fixture-qc@v1",
            "--fixture-grow",
            "--allow-fixture",
            "--planner",
            "deterministic-v2",
            "--isolated-benchmark-spec",
            str(tmp_path / "spec.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="confirm-isolated-nonproduction"):
        _worker(args)


def test_isolated_benchmark_cli_bootstraps_exact_cohort_on_fresh_sqlite_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The isolated command imports a fresh DB but queues no extra cells."""

    monkeypatch.delenv("HELENA_ENABLE_SEED_PROBE_SELECT", raising=False)
    sample_counts = {"scroll-a": 14, "scroll-b": 13, "scroll-c": 13}
    cells = []
    for sample_id, count in sample_counts.items():
        for index in range(count):
            r, c, a = index // 64, (index // 8) % 8, index % 8
            cells.append(
                {
                    "sample_id": sample_id,
                    "cell_id": f"r{r:05d}c{c:05d}a{a:05d}",
                    "independence_block_id": f"{sample_id}-block-{index}",
                }
            )
    spec = {
        "schema": "campaignx.seed_probe_benchmark_spec.v1",
        "benchmark_id": "fresh-sqlite-causal-v1",
        "frozen_at_utc": "2026-07-30T00:00:00Z",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "baseline": {
            "policy_version": "benchmark-baseline-v1",
            "planner": "deterministic-v2",
            "seed_probe_mode": "off",
        },
        "closed_loop": {
            "policy_version": "benchmark-closed-loop-v1",
            "planner": "deterministic-v2",
            "seed_probe_mode": "select",
        },
        "minimum_cells": 40,
        "maximum_cells": 40,
        "minimum_scrolls": 3,
        "minimum_relative_yield_improvement": 0.0,
        "maximum_relative_reviewer_rate_regression": 0.0,
        "maximum_incremental_compute_wall_hours_per_cell": 1.0,
        "maximum_new_incorrect_lamina_rate_upper_bound": 0.1,
        "paired_test_alpha": 0.05,
        "minimum_pairs_per_scroll": 5,
        "review_protocol_id": "blind-review-v1",
        "cells": cells,
    }
    eligible = {
        "entries": [
            {
                "sample_id": sample_id,
                "ct_uri": f"fixture://ct/{sample_id}/fixture-ct-{sample_id}-v1",
                "ct_sha256": "0" * 64,
                "surface_prediction_uri": (
                    f"fixture://m7/{sample_id}/fixture-m7-{sample_id}-v1"
                ),
                "surface_prediction_sha256": "1" * 64,
                "shape_zyx": [1024, 1024, 1024],
                "voxel_size_um": 9.362,
                "source_content_lock": {
                    "schema": "campaignx.source_content_lock.v1",
                    "status": "VERIFIED_IMMUTABLE",
                    "verification_method": "fixture-sha256-v1",
                    "verified_at_utc": "2026-07-30T00:00:00Z",
                    "ct_uri": (
                        f"fixture://ct/{sample_id}/fixture-ct-{sample_id}-v1"
                    ),
                    "ct_sha256": "0" * 64,
                    "ct_version_id": f"fixture-ct-{sample_id}-v1",
                    "m7_uri": (
                        f"fixture://m7/{sample_id}/fixture-m7-{sample_id}-v1"
                    ),
                    "m7_sha256": "1" * 64,
                    "m7_version_id": f"fixture-m7-{sample_id}-v1",
                },
            }
            for sample_id in sample_counts
        ]
    }
    spec_path = tmp_path / "spec.json"
    eligible_path = tmp_path / "eligible.json"
    catalog_path = tmp_path / "catalog.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    eligible_path.write_text(json.dumps(eligible), encoding="utf-8")
    catalog_path.write_text("{}", encoding="utf-8")
    db_path = tmp_path / "fresh-isolated.sqlite"

    args = build_parser(ROOT).parse_args(
        [
            "seed-probe-benchmark-arm",
            "--db",
            str(db_path),
            "--confirm-isolated-nonproduction",
            "--benchmark-spec",
            str(spec_path),
            "--arm",
            "baseline",
            "--eligible",
            str(eligible_path),
            "--catalog",
            str(catalog_path),
            "--grid-version",
            "causal-grid-v1",
            "--grid-step",
            "128",
            "--clearance",
            "0",
        ]
    )
    assert command_seed_probe_benchmark_arm(args) == 0

    store = FleetStore(db_path)
    assert store.status()["source_snapshots"] == 3
    assert store.status()["tasks"] == {"PENDING": 40}
    with store.connect() as connection:
        payloads = [
            json.loads(row[0])
            for row in connection.execute("SELECT payload_json FROM tasks")
        ]
    assert {
        (payload["sample_id"], payload["cell_id"]) for payload in payloads
    } == {(cell["sample_id"], cell["cell_id"]) for cell in cells}
    assert {payload["benchmark_execution"]["arm"] for payload in payloads} == {
        "baseline"
    }
