from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(ROOT))

from fleet.executor import FixtureGrowExecutor, InsufficientGpuMemoryError, log_reports_gpu_oom
from fleet import cli as fleet_cli
from fleet import planner as planner_module
from fleet.artifact_store import S3ArtifactStore
from fleet.common import artifact_manifest, content_sha256
from fleet.ct_support import apply_ct_material_support_gate
from fleet.finalizer import find_duplicate, inspect_tifxyz
from fleet.generator import (
    DEFAULT_ENVELOPE,
    _axis_centers,
    bootstrap_adaptive_retry_task,
    bootstrap_seed_positive_recovery_queue,
)
from fleet.cli import build_parser, command_demo, command_survey, seed_positive_count
from fleet.planner import COST_AWARE_DIRECT_MODEL, DEFAULT_FUSION_JUDGE, DEFAULT_FUSION_PANEL, DEFAULT_FUSION_REASONING, DEFAULT_OPENCODE_MODEL, PLANNER_PROMPT_V2, PLANNER_SANDBOX_CONFIG, CostAwareSegmentationPlanner, DeterministicPlanner, OpenCodePlanner, OpenRouterFusionPlanner, PlannerProviderUnavailable, candidate_rank_key, extract_json, normalize_candidates, screen_candidates, task_packet_for_planner, validate_and_lock
from fleet.qc_worker import FixtureQcExecutor, SurfaceQcWorker
from fleet.store import FleetStore
from fleet.worker import McpSeedProvider, RecordedSeedProvider, SegmentWorker, SourceProviderUnavailable


def source(store: FleetStore) -> str:
    return store.register_snapshot({
        "sample_id": "PHercTEST",
        "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64,
        "shape_xyz": [512, 512, 512],
        "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
    })


def task(source_id: str, cell_id: str = "cell-a", seed: tuple[int, int, int] = (256, 256, 256)) -> dict:
    return {
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "cell_id": cell_id,
        "grid_version": "test-grid-v1",
        "policy_version": "ink-blind-test-v1",
        "bounds_xyz": [[128, 128, 128], [384, 384, 384]],
        "center_xyz": {"x": 256, "y": 256, "z": 256},
        "priority": 1.0,
        "parameter_envelope": DEFAULT_ENVELOPE,
        "catalog_snapshot_sha256": "2" * 64,
        "recorded_candidates": [
            {"candidate_id": "c01", "coordinate": dict(zip("xyz", seed, strict=True)), "score": 0.9},
            {"candidate_id": "c02", "coordinate": {"x": 200, "y": 200, "z": 200}, "score": 0.4},
        ],
        "ink_used": False,
    }


def test_bounded_worker_can_opt_in_to_expected_terminal_outcomes() -> None:
    parser = build_parser(ROOT)
    default = parser.parse_args(
        [
            "worker",
            "run",
            "--db",
            "fixture.sqlite",
            "--worker-id",
            "worker",
            "--run-root",
            "/tmp/run",
            "--artifact-root",
            "/tmp/artifacts",
            "--qc-profile-id",
            "fixture-qc",
        ]
    )
    bounded = parser.parse_args(
        [
            "worker",
            "run",
            "--db",
            "fixture.sqlite",
            "--worker-id",
            "worker",
            "--run-root",
            "/tmp/run",
            "--artifact-root",
            "/tmp/artifacts",
            "--qc-profile-id",
            "fixture-qc",
            "--terminal-outcomes-exit-zero",
        ]
    )
    assert default.terminal_outcomes_exit_zero is False
    assert bounded.terminal_outcomes_exit_zero is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("NO_SEED", 0),
        ("POLICY_REJECTED", 0),
        ("BLOCKED_SOURCE_UNAVAILABLE", 0),
        ("GROW_FAILED", 2),
        ("FINALIZATION_FAILED", 2),
        ("RETRYABLE_PROVIDER_UNAVAILABLE", 2),
    ],
)
def test_bounded_worker_exit_status_distinguishes_terminal_from_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected: int,
) -> None:
    class FakeStore:
        @staticmethod
        def status() -> dict:
            return {"tasks": {status: 1}}

    class FakeWorker:
        worker_capabilities = {"schema": "fixture"}
        store = FakeStore()

        @staticmethod
        def run(**_kwargs: object) -> list[dict]:
            return [{"status": status}]

    monkeypatch.setattr(fleet_cli, "_worker", lambda _args: FakeWorker())
    result = fleet_cli.command_worker_run(
        SimpleNamespace(
            max_jobs=1,
            watch=False,
            poll_seconds=0.0,
            terminal_outcomes_exit_zero=True,
            worker_id="fixture-worker",
        )
    )
    assert result == expected
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == status


def test_adaptive_retry_creates_novel_v2_task_without_reopening_source(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    original = task(source_id)
    original["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "prediction_space": "ct_l0_xyz",
        "region": {
            "center": original["center_xyz"],
            "radius": {axis: 64 for axis in "xyz"},
        },
        "seed_region_policy": "m7-recenter-z-chunk-safe-v1",
        "minimum_cell_interior_clearance_voxels": 32,
    }
    store.create_tasks([original])
    claimed = store.claim("fixture-worker", 60)
    assert claimed is not None
    store.mark_terminal(
        claimed["task_id"],
        claimed["attempt_id"],
        claimed["lease_token"],
        "NO_SEED",
        {"status": "NO_SEED", "ink_used": False},
    )

    receipt = bootstrap_adaptive_retry_task(
        store,
        claimed["task_id"],
        grid_version="test-grid-adaptive-r01",
        policy_version="ink-blind-adaptive-v2-r01",
    )
    assert store.task_packet(claimed["task_id"])["state"] == "NO_SEED"
    adaptive = store.task_packet(receipt["task_id"])
    assert adaptive["state"] == "PENDING"
    assert adaptive["planner_contract_version"] == "v2"
    assert adaptive["candidate_selection_policy"] == "adaptive-geometry-history-v2"
    assert (
        adaptive["candidate_discovery"]["seed_region_policy"]
        == "m7-chunk-safe-merge-interior-v2"
    )
    assert adaptive["candidate_discovery"]["minimum_cell_interior_clearance_voxels"] == 0
    assert adaptive["adaptive_source_task_id"] == claimed["task_id"]

    with pytest.raises(RuntimeError, match="identity already exists"):
        bootstrap_adaptive_retry_task(
            store,
            claimed["task_id"],
            grid_version="test-grid-adaptive-r01",
            policy_version="ink-blind-adaptive-v2-r01",
        )


def test_seed_positive_recovery_bootstrap_is_geometry_only_and_preserves_catalogue(
    tmp_path: Path,
) -> None:
    source_store = FleetStore(tmp_path / "source.sqlite")
    source_store.initialize()
    source_id = source(source_store)
    surface_id = source_store.import_surface({
        "surface_id": "known-surface",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "owner": "campaign-x",
        "artifact_sha256": "a" * 64,
        "artifact_uri": "s3://fixture/known-surface/",
        "bbox_xyz": [[10, 10, 10], [20, 20, 20]],
        "sample_points": [[10, 10, 10], [20, 20, 20]],
        "area_cm2": 1.25,
        "state": "QC_SCREENED",
        "physical_qc_state": "CT_SUPPORTED",
        "ink_used": False,
    })
    assert surface_id == "known-surface"

    source_tasks = [task(source_id, cell_id) for cell_id in ("cell-positive", "cell-empty")]
    for task_value in source_tasks:
        task_value["candidate_discovery"] = {
            "prediction_uri": "fixture://m7",
            "prediction_space": "ct_l0_xyz",
            "region": {
                "center": task_value["center_xyz"],
                "radius": {"x": 128, "y": 128, "z": 128},
            },
            "seed_region_policy": "m7-recenter-z-chunk-safe-v1",
            "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
            "minimum_cell_interior_clearance_voxels": 32,
            "minimum_volume_interior_clearance_voxels": 64,
        }
    inserted, seen = source_store.create_tasks(source_tasks)
    assert (inserted, seen) == (2, 2)

    attempt_root = tmp_path / "source-attempts"
    raw_counts = {"cell-positive": 8, "cell-empty": 0}
    for _ in source_tasks:
        claimed = source_store.claim("fixture-worker", 60)
        assert claimed is not None
        attempt_dir = attempt_root / claimed["task_id"] / claimed["attempt_id"]
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "SEED_SCREEN.json").write_text(
            json.dumps({
                "raw_candidate_count": raw_counts[claimed["cell_id"]],
                "eligible_candidate_count": 0,
                "ink_used": False,
            }),
            encoding="utf-8",
        )
        source_store.mark_terminal(
            claimed["task_id"],
            claimed["attempt_id"],
            claimed["lease_token"],
            "NO_SEED",
            {"status": "NO_SEED", "ink_used": False},
        )

    output_store = FleetStore(tmp_path / "recovery.sqlite")
    gate = {
        "policy": "ome-zarr-nearby-material-v1",
        "level": 5,
        "radius_l0_voxels": 192,
        "minimum_nonzero_voxels": 1,
    }
    receipt = bootstrap_seed_positive_recovery_queue(
        source_store,
        output_store,
        attempt_root,
        grid_version="test-grid-merged-interior-v2",
        policy_version="ink-blind-test-merged-interior-v2",
        ct_material_support_gate=gate,
    )

    assert receipt["selected_task_count"] == 1
    assert receipt["selected_raw_candidate_count"] == 8
    assert receipt["skipped_zero_candidate_count"] == 1
    assert receipt["copied_surface_count"] == 1
    assert receipt["ink_used"] is False
    assert receipt["status"]["tasks"] == {"PENDING": 1}
    with output_store.connect() as connection:
        task_row = connection.execute("SELECT payload_json FROM tasks").fetchone()
        surface_row = connection.execute(
            "SELECT state,physical_qc_state FROM surfaces WHERE surface_id='known-surface'"
        ).fetchone()
    recovered_task = json.loads(task_row["payload_json"])
    assert recovered_task["grid_version"] == "test-grid-merged-interior-v2"
    assert recovered_task["policy_version"] == "ink-blind-test-merged-interior-v2"
    assert recovered_task["candidate_discovery"]["seed_region_policy"] == (
        "m7-chunk-safe-merge-interior-v2"
    )
    assert recovered_task["candidate_discovery"]["ct_material_support_gate"] == gate
    assert recovered_task["recovery_basis"] == "M7_RAW_POSITIVE_INTERIOR_REJECTED"
    assert recovered_task["ink_used"] is False
    assert tuple(surface_row) == ("QC_SCREENED", "CT_SUPPORTED")


def test_tifxyz_inspection_excludes_negative_sentinels_and_keeps_boundary_triangle(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    surface = tmp_path / "surface"
    surface.mkdir()
    rows, columns = np.indices((3, 3), dtype=np.float64)
    arrays = [columns, rows, np.zeros((3, 3), dtype=np.float64)]
    for array in arrays:
        array[0, 0] = -1.0
    for axis, array in zip("xyz", arrays, strict=True):
        tifffile.imwrite(surface / f"{axis}.tif", array)
    (surface / "meta.json").write_text("{}\n", encoding="utf-8")

    inspection = inspect_tifxyz(surface, voxel_size_um=10_000.0)

    assert inspection["finite_coordinate_count"] == 8
    assert inspection["valid_triangle_count"] == 7
    assert inspection["bbox_xyz"] == [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]
    assert inspection["area_cm2"] == pytest.approx(3.5)
    assert all(min(point) >= 0.0 for point in inspection["sample_points"])


def worker(store: FleetStore, root: Path, worker_id: str = "worker-a") -> SegmentWorker:
    return SegmentWorker(
        store,
        worker_id,
        RecordedSeedProvider(),
        DeterministicPlanner(),
        FixtureGrowExecutor(),
        root / "runs",
        root / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
    )


class OomGrowExecutor:
    def execute(self, locked_plan: dict, _attempt_dir: Path) -> dict:
        raise InsufficientGpuMemoryError(
            "fixture CUDA out of memory",
            {
                "schema": "campaignx.segment_fleet_growth_receipt.v1",
                "status": "RETRY_ON_LARGER_GPU",
                "task_id": locked_plan["task_id"],
                "attempt_id": locked_plan["attempt_id"],
                "ink_used": False,
            },
        )


def test_resource_aware_claim_skips_tasks_that_do_not_fit_worker_vram(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    legacy = task(source_id, "legacy")
    legacy["priority"] = 1.0
    large = task(source_id, "large")
    large["priority"] = 100.0
    large["resource_requirements"] = {"gpu_required": True, "minimum_vram_gb": 12.0}
    store.create_tasks([legacy, large])

    cpu_claim = store.claim("cpu-worker", 60)
    assert cpu_claim is not None
    assert cpu_claim["cell_id"] == "legacy"
    assert cpu_claim["resource_requirements"]["gpu_required"] is False

    assert store.claim(
        "gtx-1660", 60, capabilities={"cuda_available": True, "gpu_model": "GTX 1660", "gpu_vram_gb": 6.0}
    ) is None
    large_claim = store.claim(
        "rtx-3090", 60, capabilities={"cuda_available": True, "gpu_model": "RTX 3090", "gpu_vram_gb": 24.0}
    )
    assert large_claim is not None
    assert large_claim["cell_id"] == "large"
    assert large_claim["resource_requirements"]["minimum_vram_gb"] == 12.0
    assert store.status()["workers"]["rtx-3090"]["gpu_model"] == "RTX 3090"


def test_gpu_oom_requeues_for_larger_gpu_without_geometry_failure(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id, "oom")])
    segment_worker = SegmentWorker(
        store,
        "gtx-1660",
        RecordedSeedProvider(),
        DeterministicPlanner(),
        OomGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        worker_capabilities={
            "cuda_available": True,
            "gpu_model": "GTX 1660",
            "gpu_vram_gb": 6.0,
            "cuda_device_index": 0,
        },
    )

    result = segment_worker.run_one()
    assert result is not None
    assert result["status"] == "RETRY_ON_LARGER_GPU"
    assert result["minimum_vram_gb"] == 7.0
    status = store.status()
    assert status["tasks"] == {"PENDING": 1}
    assert status["attempts"] == {"RETRY_ON_LARGER_GPU": 1}
    assert store.claim(
        "same-gtx-1660", 60, capabilities={"cuda_available": True, "gpu_vram_gb": 6.0}
    ) is None
    retry = store.claim(
        "larger-gpu", 60, capabilities={"cuda_available": True, "gpu_vram_gb": 12.0}
    )
    assert retry is not None
    assert retry["resource_requirements"]["minimum_vram_gb"] == 7.0


def test_gpu_oom_detection_is_explicit_and_does_not_match_generic_failures() -> None:
    assert log_reports_gpu_oom("CUDA error: out of memory") is True
    assert log_reports_gpu_oom("VC3D failed with exit code 1") is False


class FixtureCtSupportSampler:
    def __init__(self, nonzero_by_x: dict[int, int]):
        self.nonzero_by_x = nonzero_by_x

    def sample(self, _uri: str, coordinate_xyz: dict[str, int], *, level: int, radius_l0_voxels: int) -> dict:
        nonzero = self.nonzero_by_x.get(coordinate_xyz["x"], 0)
        return {
            "level": level,
            "scale_zyx": [32.0, 32.0, 32.0],
            "center_zyx": [8, 8, 8],
            "radius_zyx": [6, 6, 6],
            "shape_zyx": [13, 13, 13],
            "voxel_count": 2197,
            "nonzero_voxel_count": nonzero,
            "nonzero_fraction": nonzero / 2197,
            "mean": 1.0 if nonzero else 0.0,
            "standard_deviation": 1.0 if nonzero else 0.0,
            "maximum": 1.0 if nonzero else 0.0,
        }


def test_ct_material_support_gate_retains_only_nearby_scanned_material() -> None:
    task_value = task("source")
    task_value["source"] = {"ct_uri": "fixture://ct"}
    task_value["candidate_discovery"] = {
        "ct_material_support_gate": {
            "policy": "ome-zarr-nearby-material-v1",
            "level": 5,
            "radius_l0_voxels": 192,
            "minimum_nonzero_voxels": 1,
        }
    }
    response = {"candidates": task_value["recorded_candidates"], "ink_used": False}

    filtered, receipt = apply_ct_material_support_gate(
        response,
        task_value,
        FixtureCtSupportSampler({200: 11}),
    )

    assert [row["candidate_id"] for row in filtered["candidates"]] == ["c02"]
    assert receipt["status"] == "COMPLETED_INK_BLIND"
    assert receipt["retained_candidate_count"] == 1
    assert receipt["rejected_candidate_count"] == 1
    assert receipt["ink_used"] is False


def test_ct_material_support_gate_is_noop_when_not_frozen_in_task() -> None:
    task_value = task("source")
    response = {"candidates": task_value["recorded_candidates"], "ink_used": False}
    filtered, receipt = apply_ct_material_support_gate(response, task_value)
    assert filtered is response
    assert receipt["status"] == "NOT_CONFIGURED"


def test_worker_stops_before_planner_and_grow_when_ct_neighborhood_is_empty(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    task_value = task(source_id)
    task_value["candidate_discovery"] = {
        "ct_material_support_gate": {
            "policy": "ome-zarr-nearby-material-v1",
            "level": 5,
            "radius_l0_voxels": 192,
            "minimum_nonzero_voxels": 1,
        }
    }
    store.create_tasks([task_value])
    result = SegmentWorker(
        store,
        "worker-ct-gate",
        RecordedSeedProvider(),
        DeterministicPlanner(),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        ct_support_sampler=FixtureCtSupportSampler({}),
    ).run_one()

    assert result is not None and result["status"] == "NO_SEED"
    assert "CT-material-support" in result["reason"]
    assert result["raw_candidate_count"] == 2
    assert result["post_ct_candidate_count"] == 0
    assert result["no_seed_cause_counts"]["CT_MATERIAL_SUPPORT_REJECTED"] == 2
    assert result["primary_causes"] == ["CT_MATERIAL_SUPPORT_REJECTED"]
    attempt = next((tmp_path / "runs").glob("*/*"))
    support = json.loads((attempt / "CT_MATERIAL_SUPPORT_SCREEN.json").read_text())
    assert support["retained_candidate_count"] == 0
    diagnosis = json.loads(
        (attempt / "NO_SEED_CAUSAL_DIAGNOSIS.json").read_text()
    )
    assert diagnosis["m7_raw_candidate_count"] == 2
    assert diagnosis["ct_support_rejected_candidate_count"] == 2
    assert diagnosis["non_claim"].startswith("NO_SEED identifies")
    assert not (attempt / "surface").exists()


def test_task_insertion_is_idempotent_and_claim_is_atomic(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    assert store.create_tasks([task(source_id)]) == (1, 1)
    assert store.create_tasks([task(source_id)]) == (0, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda name: store.claim(name, 60), ("worker-a", "worker-b")))
    assert sum(claim is not None for claim in claims) == 1
    claimed = next(claim for claim in claims if claim is not None)
    assert claimed["attempt_number"] == 1
    assert claimed["lease_token"]


def test_specific_claim_is_atomic_and_does_not_take_higher_priority_task(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    first, second = task(source_id, "cell-a"), task(source_id, "cell-b")
    first["priority"], second["priority"] = 10.0, 1.0
    store.create_tasks([first, second])
    pending = store.pending_tasks(2)
    assert [row["cell_id"] for row in pending] == ["cell-a", "cell-b"]
    claimed = store.claim("worker-a", 60, task_id=pending[1]["task_id"])
    assert claimed is not None and claimed["cell_id"] == "cell-b"
    assert store.pending_tasks(1)[0]["cell_id"] == "cell-a"


def test_m7_survey_count_ignores_provider_errors() -> None:
    assert seed_positive_count([
        {"candidate_count": 0},
        {"candidate_count": 3},
        {"candidate_count": None, "error": "network"},
    ]) == 1


def test_task_specific_survey_reads_only_requested_pending_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    first, second = task(source_id, "cell-a"), task(source_id, "cell-b")
    store.create_tasks([first, second])
    requested = store.pending_tasks(2)[1]

    class Provider:
        def discover(self, value: dict) -> dict:
            assert value["task_id"] == requested["task_id"]
            return {"candidates": []}

    monkeypatch.setattr("fleet.cli.McpSeedProvider", Provider)
    args = type("Args", (), {
        "db": tmp_path / "fleet.sqlite", "output": tmp_path / "survey.json", "limit": 99,
        "parallelism": 1, "progress": None, "task_id": requested["task_id"],
    })()
    assert command_survey(args) == 0
    receipt = json.loads((tmp_path / "survey.json").read_text())
    assert receipt["surveyed_count"] == 1
    assert receipt["requested_task_id"] == requested["task_id"]
    assert receipt["results"][0]["task_id"] == requested["task_id"]
    assert store.task_packet(requested["task_id"])["state"] == "PENDING"


def test_mcp_provider_uses_a_safe_survey_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, _url: str, _token: str):
            self.request_id = None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, _arguments: dict, request_id: str) -> dict:
            self.request_id = request_id
            return {"result": {"structuredContent": {"candidates": []}}}

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    provider = McpSeedProvider()
    task_value = task("source")
    task_value["task_id"] = "task-1"
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "region": {"center": {"x": 1, "y": 2, "z": 3}, "radius": {"x": 4, "y": 4, "z": 4}},
    }
    result = provider.discover(task_value)
    assert result["candidates"] == []


def test_mcp_http_503_is_retryable_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, _url: str, _token: str):
            return None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, _arguments: dict, _request_id: str) -> dict:
            raise HTTPError("http://fixture/mcp", 503, "Service Unavailable", {}, None)

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    task_value = task("source")
    task_value["task_id"] = "task-503"
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "region": {"center": {"x": 1, "y": 2, "z": 3}, "radius": {"x": 4, "y": 4, "z": 4}},
    }
    with pytest.raises(SourceProviderUnavailable, match="503"):
        McpSeedProvider().discover(task_value)


def test_mcp_recenter_policy_requeries_at_median_m7_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class Client:
        def __init__(self, _url: str, _token: str):
            return None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, arguments: dict, _request_id: str) -> dict:
            calls.append(arguments)
            if len(calls) == 1:
                return {"result": {"structuredContent": {"candidates": [
                    {"candidate_id": "a", "ct_l0_coordinate": {"x": 10, "y": 20, "z": 100}},
                    {"candidate_id": "b", "ct_l0_coordinate": {"x": 10, "y": 20, "z": 120}},
                    {"candidate_id": "c", "ct_l0_coordinate": {"x": 10, "y": 20, "z": 140}},
                ]}}}
            return {"result": {"structuredContent": {"candidates": [
                {"candidate_id": "safe", "ct_l0_coordinate": {"x": 15, "y": 25, "z": 120}, "combined_score": 1.0},
            ]}}}

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    task_value = task("source")
    task_value["task_id"] = "task-recenter"
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "prediction_space": "ct_l0_xyz",
        "region": {"center": {"x": 10, "y": 20, "z": 1}, "radius": {"x": 128, "y": 128, "z": 128}},
        "seed_region_policy": "m7-recenter-z-v1",
        "recenter_probe_max_candidates": 100,
        "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
    }
    result = McpSeedProvider().discover(task_value)
    assert len(calls) == 2
    assert calls[0]["region"]["center"]["z"] == 1
    assert calls[1]["region"] == {"center": {"x": 10, "y": 20, "z": 120}, "radius": {"x": 64, "y": 64, "z": 64}}
    assert result["effective_candidate_region"] == calls[1]["region"]
    assert result["initial_probe"]["median_z_ct_l0"] == 120


def test_mcp_recenter_xyz_policy_requeries_at_median_m7_coordinate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class Client:
        def __init__(self, _url: str, _token: str):
            return None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, arguments: dict, _request_id: str) -> dict:
            calls.append(arguments)
            if len(calls) == 1:
                return {"result": {"structuredContent": {"candidates": [
                    {"candidate_id": "a", "ct_l0_coordinate": {"x": 10, "y": 20, "z": 100}},
                    {"candidate_id": "b", "ct_l0_coordinate": {"x": 30, "y": 40, "z": 120}},
                    {"candidate_id": "c", "ct_l0_coordinate": {"x": 50, "y": 60, "z": 140}},
                ]}}}
            return {"result": {"structuredContent": {"candidates": [
                {"candidate_id": "safe", "ct_l0_coordinate": {"x": 30, "y": 40, "z": 120}, "combined_score": 1.0},
            ]}}}

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    task_value = task("source")
    task_value["task_id"] = "task-recenter-xyz"
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "prediction_space": "ct_l0_xyz",
        "region": {"center": {"x": 1, "y": 2, "z": 3}, "radius": {"x": 128, "y": 128, "z": 128}},
        "seed_region_policy": "m7-recenter-xyz-v1",
        "recenter_probe_max_candidates": 100,
        "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
    }
    result = McpSeedProvider().discover(task_value)
    expected = {"center": {"x": 30, "y": 40, "z": 120}, "radius": {"x": 64, "y": 64, "z": 64}}
    assert calls[1]["region"] == expected
    assert result["effective_candidate_region"] == expected
    assert result["initial_probe"]["median_coordinate_ct_l0"] == {"x": 30, "y": 40, "z": 120}


def test_mcp_chunk_safe_recenter_uses_eight_fixed_subqueries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class Client:
        def __init__(self, _url: str, _token: str):
            return None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, arguments: dict, _request_id: str) -> dict:
            calls.append(arguments)
            if len(calls) <= 8:
                z = int(arguments["region"]["center"]["z"])
                return {"result": {"structuredContent": {"candidates": [{
                    "candidate_id": f"seed-{len(calls)}",
                    "ct_l0_coordinate": {"x": 10, "y": 20, "z": z},
                    "combined_score": 1.0,
                }]}}}
            return {"result": {"structuredContent": {"candidates": [{
                "candidate_id": "safe",
                "ct_l0_coordinate": {"x": 10, "y": 20, "z": 84},
                "combined_score": 1.0,
            }]}}}

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    task_value = task("source")
    task_value["task_id"] = "task-chunk-safe"
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "prediction_space": "ct_l0_xyz",
        "region": {"center": {"x": 10, "y": 20, "z": 20}, "radius": {"x": 128, "y": 128, "z": 128}},
        "seed_region_policy": "m7-recenter-z-chunk-safe-v1",
        "recenter_probe_max_candidates": 100,
        "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
    }
    result = McpSeedProvider().discover(task_value)
    assert len(calls) == 9
    assert {tuple(call["region"]["center"].values()) for call in calls[:8]} == {
        (x, y, z) for x in (-54, 74) for y in (-44, 84) for z in (-44, 84)
    }
    assert all(call["region"]["radius"] == {"x": 64, "y": 64, "z": 64} for call in calls)
    assert result["effective_candidate_region"]["center"]["z"] == 84
    assert result["initial_probe"]["subquery_count"] == 8
    assert len(result["initial_probe"]["subqueries"]) == 8


def test_mcp_chunk_safe_merge_preserves_interior_candidates_without_final_edge_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class Client:
        def __init__(self, _url: str, _token: str):
            return None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, arguments: dict, _request_id: str) -> dict:
            calls.append(arguments)
            center = arguments["region"]["center"]
            # Mimic the production failure mode: the best point is on the
            # lower face of every 64-voxel subquery.  Some of those faces are
            # nevertheless safely inside the original 128-voxel task cube.
            coordinate = {axis: int(center[axis]) - 64 for axis in "xyz"}
            return {"result": {"structuredContent": {"candidates": [{
                "candidate_id": f"seed-{len(calls)}",
                "ct_l0_coordinate": coordinate,
                "combined_score": 1.0,
            }]}}}

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    task_value = task("source")
    task_value["task_id"] = "task-chunk-safe-merge"
    task_value["source"] = {"shape_xyz": [1024, 1024, 1024]}
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "prediction_space": "ct_l0_xyz",
        "region": {
            "center": {"x": 512, "y": 512, "z": 512},
            "radius": {"x": 128, "y": 128, "z": 128},
        },
        "seed_region_policy": "m7-chunk-safe-merge-interior-v2",
        "recenter_probe_max_candidates": 100,
        "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
        "minimum_cell_interior_clearance_voxels": 32,
        "minimum_volume_interior_clearance_voxels": 128,
    }

    result = McpSeedProvider().discover(task_value)
    screened = screen_candidates(result, task_value)

    assert len(calls) == 8
    assert result["effective_candidate_region"] == task_value["candidate_discovery"]["region"]
    assert result["initial_probe"]["merge_used_as_final_candidate_set"] is True
    assert result["initial_probe"]["merged_candidate_count"] == 8
    assert screened["raw_candidate_count"] == 8
    assert screened["usable_candidate_count"] > 0
    assert screened["best_candidate"]["cell_interior_clearance_voxels"] >= 32


def test_mcp_empty_structured_response_is_retryable_source_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        def __init__(self, _url: str, _token: str):
            return None

        def initialize(self) -> None:
            return None

        def call(self, _name: str, _arguments: dict, _request_id: str) -> dict:
            return {"result": {"structuredContent": {}}}

    monkeypatch.setenv("VC_MCP_URL", "http://127.0.0.1/mcp")
    monkeypatch.setenv("VC_MCP_AUTH_TOKEN", "fixture-token")
    monkeypatch.setattr("campaign_x.McpClient", Client)
    task_value = task("source")
    task_value["task_id"] = "task-empty-mcp"
    task_value["candidate_discovery"] = {
        "prediction_uri": "fixture://m7",
        "region": {"center": {"x": 1, "y": 2, "z": 3}, "radius": {"x": 4, "y": 4, "z": 4}},
    }
    with pytest.raises(SourceProviderUnavailable, match="candidate array"):
        McpSeedProvider().discover(task_value)


def test_screen_uses_effective_recentered_region_and_caps_planner_packet() -> None:
    task_value = task("source")
    task_value["source"] = {"shape_xyz": [1000, 1000, 1000]}
    task_value["parameter_envelope"] = {**DEFAULT_ENVELOPE, "maximum_candidate_count": 1}
    task_value["candidate_discovery"] = {
        "minimum_cell_interior_clearance_voxels": 32,
        "minimum_volume_interior_clearance_voxels": 64,
    }
    response = {
        "effective_candidate_region": {"center": {"x": 500, "y": 500, "z": 500}, "radius": {"x": 64, "y": 64, "z": 64}},
        "candidates": [
            {"candidate_id": "best", "ct_l0_coordinate": {"x": 500, "y": 500, "z": 500}, "combined_score": 1.0},
            {"candidate_id": "second", "ct_l0_coordinate": {"x": 510, "y": 510, "z": 510}, "combined_score": 0.5},
        ],
    }
    screen = screen_candidates(response, task_value)
    assert screen["eligible_candidate_count"] == 2
    assert screen["usable_candidate_count"] == 1
    assert screen["best_candidate"]["candidate_id"] == "best"
    assert screen["best_candidate"]["cell_interior_clearance_voxels"] == pytest.approx(64.0)


def test_planner_packet_preserves_recenter_query_provenance(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    response = {
        "effective_candidate_region": {"center": {"x": 256, "y": 256, "z": 200}, "radius": {"x": 64, "y": 64, "z": 64}},
        "initial_probe": {"policy": "m7-recenter-z-v1", "median_z_ct_l0": 200, "recentered": True},
    }
    packet = task_packet_for_planner(claimed, [], response)
    assert packet["m7_query"] == response


def test_source_failure_requeues_only_after_cooldown(tmp_path: Path) -> None:
    class UnavailableProvider:
        def discover(self, _task: dict) -> dict:
            raise SourceProviderUnavailable("transient MCP source failure: HTTP Error 503")

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    result = SegmentWorker(
        store,
        "worker-a",
        UnavailableProvider(),
        DeterministicPlanner(),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        source_retry_delay_seconds=60,
    ).run_one()
    assert result is not None and result["status"] == "RETRYABLE_SOURCE_UNAVAILABLE"
    assert store.status()["tasks"] == {"PENDING": 1}
    assert store.pending_tasks(1) == []
    with store.connect() as connection:
        attempt_state = connection.execute("SELECT state FROM attempts").fetchone()[0]
    assert attempt_state == "RETRYABLE_SOURCE_UNAVAILABLE"


def test_planner_provider_failure_requeues_not_policy_rejects(tmp_path: Path) -> None:
    class UnavailablePlanner:
        def propose(self, _packet: dict, _attempt_dir: Path) -> dict:
            raise PlannerProviderUnavailable("ResourceExhausted")

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    result = SegmentWorker(
        store,
        "worker-a",
        RecordedSeedProvider(),
        UnavailablePlanner(),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
        provider_retry_delay_seconds=60,
    ).run_one()
    assert result is not None and result["status"] == "RETRYABLE_PROVIDER_UNAVAILABLE"
    assert store.status()["tasks"] == {"PENDING": 1}
    with store.connect() as connection:
        attempt_state = connection.execute("SELECT state FROM attempts").fetchone()[0]
    assert attempt_state == "RETRYABLE_PROVIDER_UNAVAILABLE"


def test_historical_planner_outage_recovery_is_narrow_and_auditable(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    store.mark_terminal(
        claimed["task_id"],
        claimed["attempt_id"],
        claimed["lease_token"],
        "POLICY_REJECTED",
        {"error": "PlannerProviderUnavailable: ResourceExhausted"},
    )
    recovered = store.recover_terminal_provider_outage(
        claimed["task_id"], claimed["attempt_id"], retry_delay_seconds=60
    )
    assert recovered["status"] == "RECOVERED_PROVIDER_UNAVAILABLE"
    assert store.status()["tasks"] == {"PENDING": 1}
    with store.connect() as connection:
        attempt_state = connection.execute("SELECT state FROM attempts").fetchone()[0]
    assert attempt_state == "RECOVERED_PROVIDER_UNAVAILABLE"


def test_terminal_mcp_auth_outage_recovery_is_exact_and_auditable(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    receipt = {
        "status": "BLOCKED_SOURCE_UNAVAILABLE",
        "error": "HTTPError: HTTP Error 401: Unauthorized",
    }
    store.mark_terminal(
        claimed["task_id"], claimed["attempt_id"], claimed["lease_token"],
        "BLOCKED_SOURCE_UNAVAILABLE", receipt,
    )
    recovered = store.recover_terminal_mcp_auth_outage(
        claimed["task_id"], claimed["attempt_id"], retry_delay_seconds=1
    )
    assert recovered["status"] == "RECOVERED_MCP_AUTH_OUTAGE"
    assert store.status()["tasks"] == {"PENDING": 1}
    with store.connect() as connection:
        state = connection.execute("SELECT state FROM attempts").fetchone()[0]
    assert state == "RECOVERED_MCP_AUTH_OUTAGE"


def test_terminal_mcp_auth_recovery_rejects_nonmatching_401(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    store.mark_terminal(
        claimed["task_id"], claimed["attempt_id"], claimed["lease_token"],
        "BLOCKED_SOURCE_UNAVAILABLE",
        {"status": "BLOCKED_SOURCE_UNAVAILABLE", "error": "HTTPError: HTTP Error 401: remote object denied"},
    )
    with pytest.raises(RuntimeError, match="exact loopback MCP"):
        store.recover_terminal_mcp_auth_outage(
            claimed["task_id"], claimed["attempt_id"], retry_delay_seconds=1
        )


def test_terminal_finalizer_dependency_recovery_is_exact_and_auditable(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    receipt = {
        "status": "FINALIZATION_FAILED",
        "error": "ModuleNotFoundError: No module named 'numpy'",
    }
    store.mark_terminal(
        claimed["task_id"], claimed["attempt_id"], claimed["lease_token"],
        "FINALIZATION_FAILED", receipt,
    )
    recovered = store.recover_terminal_finalizer_dependency(
        claimed["task_id"], claimed["attempt_id"], retry_delay_seconds=1
    )
    assert recovered["status"] == "RECOVERED_FINALIZER_DEPENDENCY"
    assert store.status()["tasks"] == {"PENDING": 1}
    with store.connect() as connection:
        state = connection.execute("SELECT state FROM attempts").fetchone()[0]
    assert state == "RECOVERED_FINALIZER_DEPENDENCY"


def test_terminal_finalizer_dependency_recovery_rejects_other_errors(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    store.mark_terminal(
        claimed["task_id"], claimed["attempt_id"], claimed["lease_token"],
        "FINALIZATION_FAILED",
        {"status": "FINALIZATION_FAILED", "error": "RuntimeError: corrupt TIFXYZ"},
    )
    with pytest.raises(RuntimeError, match="exact missing finalizer dependency"):
        store.recover_terminal_finalizer_dependency(
            claimed["task_id"], claimed["attempt_id"], retry_delay_seconds=1
        )


def test_m7_screen_uses_combined_score_and_rejects_boundary_seed() -> None:
    task_value = task("source")
    task_value["bounds_xyz"] = [[100, 100, 100], [300, 300, 300]]
    task_value["source"] = {
        "shape_xyz": [1000, 1000, 1000],
        "voxel_size_um": 9.362,
    }
    task_value["candidate_discovery"] = {
        "minimum_cell_interior_clearance_voxels": 32,
        "minimum_volume_interior_clearance_voxels": 64,
    }
    response = {"candidates": [
        {"candidate_id": "edge", "ct_l0_coordinate": {"x": 110, "y": 150, "z": 150}, "combined_score": 1.0},
        {"candidate_id": "safe", "ct_l0_coordinate": {"x": 180, "y": 180, "z": 180}, "combined_score": 0.7},
    ]}
    screen = screen_candidates(response, task_value)
    assert screen["raw_candidate_count"] == 2
    assert screen["usable_candidate_count"] == 1
    assert screen["best_candidate"]["candidate_id"] == "safe"
    assert screen["best_candidate"]["score"] == pytest.approx(0.7)
    diagnostic = screen["rejection_diagnostics"]
    assert diagnostic["rejection_counts"] == {
        "MALFORMED_COORDINATE_OR_SCORE": 0,
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": 1,
        "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": 0,
    }
    assert diagnostic["clearance_policy"][
        "minimum_cell_interior_clearance_um"
    ] == pytest.approx(32 * 9.362)
    assert normalize_candidates(response, task_value)[0]["cell_interior_clearance_voxels"] == pytest.approx(80.0)


def test_candidate_screen_reports_each_no_seed_clearance_cause() -> None:
    task_value = task("source")
    task_value["bounds_xyz"] = [[0, 0, 0], [300, 300, 300]]
    task_value["source"] = {
        "shape_xyz": [1000, 1000, 1000],
        "voxel_size_um": 9.362,
    }
    task_value["candidate_discovery"] = {
        "minimum_cell_interior_clearance_voxels": 32,
        "minimum_volume_interior_clearance_voxels": 64,
    }
    response = {
        "candidates": [
            {
                "candidate_id": "edge",
                "coordinate": {"x": 10, "y": 100, "z": 100},
                "score": 1.0,
            },
            {
                "candidate_id": "malformed",
                "coordinate": {"x": "not-a-number", "y": 100, "z": 100},
                "score": 0.9,
            },
            {
                "candidate_id": "safe",
                "coordinate": {"x": 150, "y": 150, "z": 150},
                "score": 0.8,
            },
        ]
    }
    diagnostic = screen_candidates(response, task_value)["rejection_diagnostics"]
    assert diagnostic["raw_candidate_count"] == 3
    assert diagnostic["retained_before_packet_limit_count"] == 1
    assert diagnostic["rejected_candidate_count"] == 2
    assert diagnostic["rejection_counts"] == {
        "MALFORMED_COORDINATE_OR_SCORE": 1,
        "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": 1,
        "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": 1,
    }
    assert diagnostic["clearance_policy"][
        "minimum_volume_interior_clearance_um"
    ] == pytest.approx(64 * 9.362)


def test_new_candidate_policy_prefers_interior_seed_and_rejects_lexical_tie(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    task_value = task(source_id)
    task_value["candidate_selection_policy"] = "score-cell-volume-clearance-v1"
    store.create_tasks([task_value])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    candidates = [
        {"candidate_id": "c01", "x": 260, "y": 260, "z": 260, "score": 1.0, "cell_interior_clearance_voxels": 4, "volume_interior_clearance_voxels": 260},
        {"candidate_id": "c02", "x": 256, "y": 256, "z": 256, "score": 1.0, "cell_interior_clearance_voxels": 128, "volume_interior_clearance_voxels": 256},
    ]
    assert sorted(candidates, key=candidate_rank_key)[0]["candidate_id"] == "c02"
    packet = task_packet_for_planner(claimed, candidates)
    proposal = DeterministicPlanner().propose(packet, tmp_path)
    assert proposal["selected_seed"]["candidate_id"] == "c02"
    validate_and_lock(packet, proposal)
    proposal["selected_seed"] = {"candidate_id": "c01", "x": 260, "y": 260, "z": 260}
    with pytest.raises(ValueError, match="score/cell/volume-clearance"):
        validate_and_lock(packet, proposal)


def test_volume_edge_margin_keeps_generated_centres_out_of_source_rim() -> None:
    assert _axis_centers(1024, radius=64, step=256, volume_edge_margin=256) == [256, 512, 767]
    assert _axis_centers(512, radius=64, step=256, volume_edge_margin=256) == []


def test_planner_rejects_seed_not_returned_by_mcp(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    candidates = [{"candidate_id": "c01", "x": 256, "y": 256, "z": 256, "score": 0.9}]
    packet = task_packet_for_planner(claimed, candidates)
    proposal = DeterministicPlanner().propose(packet, tmp_path)
    proposal["selected_seed"]["x"] += 1
    with pytest.raises(ValueError, match="not one exact MCP candidate"):
        validate_and_lock(packet, proposal)


def test_vertical_slice_promotes_once_and_deduplicates_second_grow(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id, "cell-a")])
    first = worker(store, tmp_path, "worker-a").run_one()
    assert first is not None and first["status"] == "QC_PENDING"
    assert store.status()["surfaces"] == 1
    store.create_tasks([task(source_id, "cell-b")])
    second = worker(store, tmp_path, "worker-b").run_one()
    assert second is not None and second["status"] == "DUPLICATE_SURFACE"
    assert second["duplicate_of"] == first["surface"]["surface_id"]
    assert second["duplicate_diagnostics"]["rule"] == "BIDIRECTIONAL_POINT_OVERLAP"
    assert store.status()["surfaces"] == 1
    assert store.status()["qc_jobs"] == 1


def test_fixture_surface_is_catalogued_but_never_enqueued_for_scientific_qc(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    sample_id = "PHercDISTRIBUTEDTEST-fixture-isolation"
    source_id = store.register_snapshot(
        {
            "sample_id": sample_id,
            "ct_uri": "fixture://ct",
            "ct_sha256": "0" * 64,
            "m7_uri": "fixture://m7",
            "m7_sha256": "1" * 64,
            "shape_xyz": [512, 512, 512],
            "voxel_size_um": 9.362,
            "coordinate_frame": "ct_l0_xyz",
            "fixture_only": True,
        }
    )
    fixture_task = task(source_id)
    fixture_task.update(
        {
            "sample_id": sample_id,
            "policy_version": "fixture-only-v1",
            "fixture_only": True,
        }
    )
    store.create_tasks([fixture_task])

    result = worker(store, tmp_path).run_one()

    assert result is not None and result["status"] == "FIXTURE_ONLY"
    assert result["surface"]["fixture_only"] is True
    assert store.status()["surfaces"] == 1
    assert store.status()["qc_jobs"] == 0
    surface = store.surfaces_for_snapshot(source_id)[0]
    assert surface["state"] == "FIXTURE_ONLY"
    assert surface["physical_qc_state"] == "NOT_APPLICABLE_FIXTURE"
    assert store.claim_qc("qc-must-stay-idle", 60) is None


def test_fixture_surface_backfill_is_rejected_before_qc_enqueue(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    with pytest.raises(ValueError, match="fixture-only"):
        store.enqueue_imported_surface_qc(
            {
                "surface_id": "fixture-surface",
                "source_snapshot_id": source_id,
                "sample_id": "PHercTEST",
                "artifact_sha256": "f" * 64,
                "artifact_uri": "fixture://surface",
                "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
                "fixture_only": True,
            },
            profile_id="fixture-surface-qc@1.0.0",
        )


def test_qc_job_is_claimed_once_and_finalized_with_durable_evidence(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    growth = worker(store, tmp_path).run_one()
    assert growth is not None and growth["status"] == "QC_PENDING"

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda worker_id: store.claim_qc(worker_id, 60), ("qc-a", "qc-b")))
    live_claims = [claim for claim in claims if claim is not None]
    assert len(live_claims) == 1
    claim = live_claims[0]
    assert claim["surface"]["artifact_uri"]
    assert claim["source"]["ct_uri"] == "fixture://ct"

    with pytest.raises(RuntimeError, match="heartbeat rejected"):
        store.heartbeat_qc(claim["qc_job_id"], "wrong-token", 60)
    store.heartbeat_qc(claim["qc_job_id"], claim["lease_token"], 60)
    result = {
        "schema": "campaignx.segment_qc_result.v1",
        "surface_id": claim["surface_id"],
        "outcome": "CT_SUPPORTED_RETAINED_FOR_REVIEW",
        "evidence_manifest_sha256": "a" * 64,
        "evidence_uri": "fixture://evidence/manifest.json",
        "ink_used": True,
    }
    finalized = store.finalize_qc(
        claim["qc_job_id"], claim["lease_token"], result["outcome"], result
    )
    assert finalized["surface_state"] == "QC_REVIEW_PENDING"
    assert finalized["physical_qc_state"] == "CT_SUPPORTED_REVIEW"
    assert store.status()["qc_job_states"] == {"COMPLETED": 1}
    with pytest.raises(RuntimeError, match="stale lease"):
        store.finalize_qc(
            claim["qc_job_id"], claim["lease_token"], result["outcome"], result
        )


def test_historical_surface_backfill_reconciles_once_then_becomes_immutable(
    tmp_path: Path,
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.import_surface(
        {
            "surface_id": "campaign-x:PHercTEST:legacy-01",
            "source_snapshot_id": source_id,
            "sample_id": "PHercTEST",
            "owner": "campaign-x",
            "artifact_sha256": "1" * 64,
            "artifact_uri": "legacy/PHercTEST/legacy-01",
            "bbox_xyz": [[1, 2, 3], [4, 5, 6]],
            "area_cm2": 1.25,
            "state": "PRESERVED_TIFXYZ_CANDIDATE",
            "physical_qc_state": "UNVALIDATED",
        }
    )
    payload = {
        "surface_id": "campaign-x:PHercTEST:legacy-01",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "owner": "campaign-x",
        "artifact_sha256": "2" * 64,
        "artifact_uri": "/verified/PHercTEST/legacy-01",
        "bbox_xyz": [[1, 2, 3], [4, 5, 6]],
        "area_cm2": 1.25,
    }

    first = store.enqueue_imported_surface_qc(
        payload, profile_id="fixture-surface-qc@1.0.0"
    )
    second = store.enqueue_imported_surface_qc(
        payload, profile_id="fixture-surface-qc@1.0.0"
    )

    assert first["status"] == "ENQUEUED"
    assert first["reconciliation"] == "RECONCILED_UNVALIDATED"
    assert second == {
        "status": "ALREADY_ENQUEUED",
        "surface_id": payload["surface_id"],
        "qc_job_id": first["qc_job_id"],
        # An imported surface has no geometry verdict, so its QC job waits. This
        # asserted PENDING, and then claimed the job successfully -- which is the
        # geometry bypass written down as an expectation. Model QC comes after the
        # geometry gate whichever path put the surface here.
        "qc_state": "WAITING_GEOMETRY",
    }
    assert store.claim_qc("qc-backfill", 60) is None, (
        "an unmeasured import is claimable, so the gate is decorative"
    )

    # And the gate is not a trap: a verdict promotes the job and then it claims.
    store.record_geometry_certification(payload["surface_id"], "GEOMETRY_CERTIFIED",
                                        {"schema": "test", "note": "measured"})
    claim = store.claim_qc("qc-backfill", 60)
    assert claim is not None, "certification did not release the waiting job"
    assert claim["surface"]["artifact_sha256"] == "2" * 64
    assert claim["surface"]["artifact_uri"] == payload["artifact_uri"]
    changed = {**payload, "artifact_uri": "/different"}
    with pytest.raises(RuntimeError, match="after its QC job exists"):
        store.enqueue_imported_surface_qc(
            changed, profile_id="fixture-surface-qc@1.0.0"
        )


def test_historical_surface_backfill_inserts_missing_surface(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    payload = {
        "surface_id": "campaign-x:PHercTEST:new-history",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "owner": "campaign-x",
        "artifact_sha256": "3" * 64,
        "artifact_uri": "/verified/PHercTEST/new-history",
        "bbox_xyz": [[10, 20, 30], [40, 50, 60]],
        "area_cm2": 1.5,
    }

    result = store.enqueue_imported_surface_qc(
        payload, profile_id="fixture-surface-qc@1.0.0"
    )

    assert result["status"] == "ENQUEUED"
    assert result["reconciliation"] == "INSERTED"
    assert store.status()["surfaces"] == 1
    # WAITING_GEOMETRY, not PENDING. An imported surface carries no geometry
    # verdict, and claim_qc takes only PENDING -- so PENDING here meant the ink
    # model could claim a surface nothing had measured, which is the boundary the
    # stage states. record_geometry_certification promotes it when a verdict
    # arrives. This expectation encoded the bypass.
    assert store.status()["qc_job_states"] == {"WAITING_GEOMETRY": 1}


def test_qc_finalization_rejects_unattributed_or_malformed_evidence(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    assert worker(store, tmp_path).run_one()["status"] == "QC_PENDING"
    claim = store.claim_qc("qc-a", 60)
    assert claim is not None
    malformed = {
        "schema": "campaignx.segment_qc_result.v1",
        "surface_id": claim["surface_id"],
        "outcome": "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
        "evidence_manifest_sha256": "not-a-digest",
        "evidence_uri": "",
        "ink_used": False,
    }
    with pytest.raises(ValueError, match="SHA-256"):
        store.finalize_qc(
            claim["qc_job_id"], claim["lease_token"], malformed["outcome"], malformed
        )
    assert store.status()["qc_job_states"] == {"CLAIMED": 1}


def test_surface_qc_worker_completes_fixture_vertical_slice_without_leaking_lease(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    growth = worker(store, tmp_path).run_one()
    assert growth is not None and growth["status"] == "QC_PENDING"

    receipt = SurfaceQcWorker(
        store,
        "qc-fixture",
        FixtureQcExecutor("CT_SUPPORTED_RETAINED_FOR_REVIEW"),
        tmp_path / "qc-runs",
        lease_seconds=60,
    ).run_one()

    assert receipt is not None and receipt["status"] == "COMPLETED"
    assert receipt["outcome"] == "CT_SUPPORTED_RETAINED_FOR_REVIEW"
    assert store.status()["qc_job_states"] == {"COMPLETED": 1}
    claimed_receipt = next((tmp_path / "qc-runs").glob("*/*/CLAIMED_QC_JOB.json"))
    assert "lease_token" not in claimed_receipt.read_text(encoding="utf-8")


def test_surface_qc_worker_requeues_operational_failure_without_scientific_result(tmp_path: Path) -> None:
    class BrokenExecutor:
        def execute(self, _claim: dict, _attempt_dir: Path) -> dict:
            raise RuntimeError("temporary renderer outage")

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    assert worker(store, tmp_path).run_one()["status"] == "QC_PENDING"

    receipt = SurfaceQcWorker(
        store,
        "qc-broken",
        BrokenExecutor(),
        tmp_path / "qc-runs",
        lease_seconds=60,
        retry_delay_seconds=0,
    ).run_one()

    assert receipt is not None and receipt["status"] == "RETRYABLE_QC_UNAVAILABLE"
    # Back to PENDING, not held: this job was claimable before the executor
    # broke, so requeueing it is a retry and not a new admission.
    assert store.status()["qc_job_states"] == {"PENDING": 1}
    retry = next((tmp_path / "qc-runs").glob("*/*/RETRYABLE_QC_RECEIPT.json"))
    payload = json.loads(retry.read_text(encoding="utf-8"))
    assert payload["no_scientific_conclusion"] is True


def test_concurrent_finalizers_register_only_one_surface(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id, "cell-a"), task(source_id, "cell-b")])
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(
                lambda name: worker(store, tmp_path, name).run_one(),
                ("worker-a", "worker-b"),
            )
        )
    assert sorted(receipt["status"] for receipt in receipts if receipt) == [
        "DUPLICATE_SURFACE",
        "QC_PENDING",
    ]
    assert store.status()["surfaces"] == 1
    assert store.status()["qc_jobs"] == 1


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}

    def upload_file(self, path: str, bucket: str, key: str, ExtraArgs: dict) -> None:
        self.objects[(bucket, key)] = {
            "body": Path(path).read_bytes(),
            "metadata": dict(ExtraArgs.get("Metadata", {})),
        }

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, Metadata: dict, **_kwargs) -> None:
        self.objects[(Bucket, Key)] = {"body": bytes(Body), "metadata": dict(Metadata)}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        value = self.objects[(Bucket, Key)]
        return {"ContentLength": len(value["body"]), "Metadata": value["metadata"]}

    def copy_object(self, *, Bucket: str, Key: str, CopySource: dict, **_kwargs) -> None:
        source = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.objects[(Bucket, Key)] = {
            "body": source["body"],
            "metadata": dict(source["metadata"]),
        }


def test_s3_artifact_store_stages_verifies_and_promotes_idempotently(tmp_path: Path) -> None:
    surface = tmp_path / "surface"
    surface.mkdir()
    for name, payload in {
        "x.tif": b"x-grid",
        "y.tif": b"y-grid",
        "z.tif": b"z-grid",
        "meta.json": b"{}\n",
    }.items():
        (surface / name).write_bytes(payload)
    files = artifact_manifest(surface, ("x.tif", "y.tif", "z.tif", "meta.json"))
    manifest = {"files": files, "artifact_sha256": content_sha256(files)}
    client = _FakeS3()
    store = S3ArtifactStore("s3://fixture-bucket/fleet-v1", client=client)
    staged = store.stage(surface, "attempt-1", manifest)
    first = store.promote(staged, "PHercTEST", "surface-1", manifest)
    second = store.promote(staged, "PHercTEST", "surface-1", manifest)
    assert first == second
    assert first["artifact_uri"] == "s3://fixture-bucket/fleet-v1/surfaces/PHercTEST/surface-1"
    assert len(client.objects) == 10


def test_s3_artifact_store_refuses_existing_hash_mismatch(tmp_path: Path) -> None:
    surface = tmp_path / "surface"
    surface.mkdir()
    for name in ("x.tif", "y.tif", "z.tif", "meta.json"):
        (surface / name).write_bytes(name.encode())
    files = artifact_manifest(surface, ("x.tif", "y.tif", "z.tif", "meta.json"))
    manifest = {"files": files, "artifact_sha256": content_sha256(files)}
    client = _FakeS3()
    store = S3ArtifactStore("s3://fixture-bucket/fleet-v1", client=client)
    staged = store.stage(surface, "attempt-1", manifest)
    destination = ("fixture-bucket", "fleet-v1/surfaces/PHercTEST/surface-1/x.tif")
    client.objects[destination] = {"body": b"wrong", "metadata": {"sha256": "wrong"}}
    with pytest.raises(RuntimeError, match="mismatch"):
        store.promote(staged, "PHercTEST", "surface-1", manifest)


def test_opencode_json_extraction_accepts_plain_or_fenced() -> None:
    assert extract_json('{"ok":true}') == {"ok": True}
    assert extract_json('result\n```json\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_repairs_only_redundant_array_opener_and_requires_schema() -> None:
    malformed = (
        '{"schema":"campaignx.segmentation_proposal.v2",'
        '"alternatives_rejected":[{"candidate_id":"c1","reason":"one"},'
        '[{"candidate_id":"c2","reason":"two"}],"ink_used":false}'
    )
    parsed = extract_json(
        malformed,
        expected_schema="campaignx.segmentation_proposal.v2",
    )
    assert [row["candidate_id"] for row in parsed["alternatives_rejected"]] == [
        "c1",
        "c2",
    ]

    with pytest.raises(ValueError, match="matching schema"):
        extract_json(
            '{"candidate_id":"nested-only"}',
            expected_schema="campaignx.segmentation_proposal.v2",
        )


def test_opencode_adapter_places_prompt_before_greedy_file_option(tmp_path: Path) -> None:
    assert DEFAULT_OPENCODE_MODEL is None
    historical_model = "openrouter/historical-fixture-model"
    executable = tmp_path / "fake-opencode"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "print(json.dumps({'argv': sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = OpenCodePlanner(
        str(executable),
        ROOT,
        model=historical_model,
    ).propose({}, tmp_path)
    argv = result["argv"]
    assert argv[0] == "run"
    assert argv[1].startswith("You are the ink-blind Helena Framework Stage 01 segmentation planner")
    assert "constraints.ink_used" in argv[1]
    assert "downstream validator rejects" in argv[1]
    sandbox = tmp_path / "opencode-planner-sandbox"
    assert argv[argv.index("--dir") + 1] == str(sandbox.resolve())
    assert argv[argv.index("--file") + 1] == str((sandbox / "PLANNER_PACKET.json").resolve())
    assert json.loads((sandbox / ".opencode" / "opencode.json").read_text()) == PLANNER_SANDBOX_CONFIG
    assert PLANNER_SANDBOX_CONFIG["permission"]["edit"] == "deny"
    assert PLANNER_SANDBOX_CONFIG["permission"]["bash"] == "deny"
    assert argv[argv.index("--model") + 1] == historical_model


def test_opencode_v2_uses_bounded_adaptive_prompt(tmp_path: Path) -> None:
    executable = tmp_path / "fake-opencode-v2"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "print(json.dumps({'argv': sys.argv[1:]}))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    planner = OpenCodePlanner(
        str(executable),
        ROOT,
        model=DEFAULT_OPENCODE_MODEL,
        contract_version="v2",
    )
    result = planner.propose({}, tmp_path)
    prompt = result["argv"][1]
    assert prompt == PLANNER_PROMPT_V2
    assert "Never average, offset" in prompt
    assert "Do not repeat an exact failed recipe" in prompt
    assert '"history_considered"' in prompt
    assert planner.contract_version == "v2"
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = planner.propose(
        {"schema": "campaignx.segmentation_planner_packet.v1"}, legacy_dir
    )
    assert legacy["argv"][1].startswith(
        "You are the ink-blind Helena Framework Stage 01 segmentation planner"
    )
    assert "history_considered" not in legacy["argv"][1]


def test_opencode_provider_exhaustion_is_retryable_not_a_bad_proposal(tmp_path: Path) -> None:
    executable = tmp_path / "provider-unavailable-opencode"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stderr.write('{\\\"metadata\\\":{\\\"error_type\\\":\\\"provider_unavailable\\\"},\\\"message\\\":\\\"ResourceExhausted\\\"}\\n')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    with pytest.raises(PlannerProviderUnavailable):
        OpenCodePlanner(str(executable), ROOT, model=DEFAULT_OPENCODE_MODEL).propose({}, tmp_path)


def test_provider_failure_requeues_only_after_cooldown(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    store.requeue_provider_unavailable(
        claimed["task_id"],
        claimed["attempt_id"],
        claimed["lease_token"],
        {"status": "RETRYABLE_PROVIDER_UNAVAILABLE"},
        retry_delay_seconds=60,
    )
    assert store.status()["tasks"] == {"PENDING": 1}
    assert store.pending_tasks(1) == []
    with store.connect() as connection:
        attempt_state = connection.execute("SELECT state FROM attempts WHERE attempt_id=?", (claimed["attempt_id"],)).fetchone()[0]
    assert attempt_state == "RETRYABLE_PROVIDER_UNAVAILABLE"


def test_close_parallel_sheet_is_not_collapsed_as_duplicate(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    points = [[float(x), float(y), 100.0] for x in range(8) for y in range(8)]
    store.import_surface({
        "surface_id": "known-sheet",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "owner": "test",
        "artifact_sha256": "3" * 64,
        "bbox_xyz": [[0.0, 0.0, 100.0], [7.0, 7.0, 100.0]],
        "sample_points": points,
        "state": "IMPORTED",
    })
    shifted = [[x, y, z + 2.0] for x, y, z in points]
    duplicate, diagnostics = find_duplicate(store, source_id, "4" * 64, shifted)
    assert duplicate is None
    assert diagnostics["comparisons"][0]["median_voxels"] == pytest.approx(2.0)


def test_ink_directed_or_out_of_envelope_proposals_fail_closed(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id)])
    claimed = store.claim("worker-a", 60)
    assert claimed is not None
    packet = task_packet_for_planner(claimed, [{"candidate_id": "c01", "x": 256, "y": 256, "z": 256, "score": 1.0}])
    proposal = DeterministicPlanner().propose(packet, tmp_path)
    proposal["ink_used"] = True
    with pytest.raises(ValueError, match="ink-directed"):
        validate_and_lock(packet, proposal)
    proposal = DeterministicPlanner().propose(packet, tmp_path)
    proposal["parameters"]["generations"] = 999
    with pytest.raises(ValueError, match="above maximum"):
        validate_and_lock(packet, proposal)


def _v2_proposal(packet: dict, candidate_id: str = "c02") -> dict:
    selected = next(row for row in packet["candidate_seeds"] if row["candidate_id"] == candidate_id)
    envelope = packet["parameter_envelope"]
    parameters = {
        name: rule.get("const", rule.get("default"))
        for name, rule in envelope["parameters"].items()
    }
    parameters["generations"] = 30
    return {
        # 1 is what every run does unless it is deliberately asking for one of
        # m7's alternatives; the field records which rung the seed came from.
        "candidate_rank": 1,
        "candidate_rank": 1,
    "schema": "campaignx.segmentation_proposal.v2",
        "task_id": packet["task_id"],
        "attempt_id": packet["attempt_id"],
        "selected_seed": {key: selected[key] for key in ("candidate_id", "x", "y", "z")},
        "profile_id": envelope["profile_ids"][0],
        "parameters": parameters,
        "history_considered": [
            row["history_id"]
            for row in packet["regional_attempt_history"]["attempts"]
        ],
        "hypothesis": "A different in-cell seed and bounded generation count may avoid the prior geometry failure.",
        "parameter_rationale": {
            name: "Frozen constant." if "const" in rule else "Bounded geometry variation."
            for name, rule in envelope["parameters"].items()
        },
        "variation_summary": "Uses a different listed seed and fewer generations than the prior failed recipe.",
        "alternatives_rejected": [
            {"candidate_id": row["candidate_id"], "reason": "Lower clearance for this bounded retry."}
            for row in packet["candidate_seeds"]
            if row["candidate_id"] != candidate_id
        ],
        "ink_used": False,
    }


def _empty_v2_history(task_row: dict) -> dict:
    body = {
        "schema": "campaignx.segmentation_regional_attempt_history.v1",
        "task_id": task_row["task_id"],
        "source_snapshot_id": task_row["source_snapshot_id"],
        "attempt_count": 0,
        "attempts": [],
        "ink_used": False,
    }
    return {**body, "history_sha256": content_sha256(body)}


def test_fusion_planner_locks_max_reasoning_panel_and_records_exact_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    adaptive = task(source_id)
    adaptive["planner_contract_version"] = "v2"
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    store.create_tasks([adaptive])
    claimed = store.claim("worker-fusion", 60)
    assert claimed is not None
    candidates = normalize_candidates(
        {"candidates": claimed["recorded_candidates"]},
        claimed,
    )
    packet = task_packet_for_planner(
        claimed,
        candidates,
        contract_version="v2",
        regional_attempt_history=_empty_v2_history(claimed),
    )
    proposal = _v2_proposal(packet, candidate_id="c01")
    proposal["parameters"]["generations"] = 35
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            response = {
                "id": "gen-fusion-test",
                "model": DEFAULT_FUSION_JUDGE,
                "choices": [{"message": {"role": "assistant", "content": json.dumps(proposal)}}],
                "usage": {
                    "prompt_tokens": 1234,
                    "completion_tokens": 567,
                    "total_tokens": 1801,
                    "cost": 0.123456,
                },
            }
            return json.dumps(response).encode()

    def fake_urlopen(request: object, timeout: int) -> FakeResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    monkeypatch.setattr(planner_module.urllib.request, "urlopen", fake_urlopen)
    run_dir = tmp_path / "fusion"
    result = OpenRouterFusionPlanner().propose(packet, run_dir)
    assert result == proposal
    request = captured["request"]
    body = json.loads(request.data)
    plugin = body["plugins"][0]
    assert body["model"] == "openrouter/fusion"
    assert body["tool_choice"] == "required"
    assert plugin["analysis_models"] == list(DEFAULT_FUSION_PANEL)
    assert plugin["model"] == DEFAULT_FUSION_JUDGE
    assert plugin["reasoning"] == DEFAULT_FUSION_REASONING
    assert plugin["reasoning"]["effort"] == "max"
    assert plugin["temperature"] == 0
    cost = json.loads((run_dir / "FUSION_COST_RECEIPT.json").read_text())
    assert cost["response_reported_cost_usd"] == pytest.approx(0.123456)
    assert cost["total_cost_usd"] is None
    assert cost["cost_complete"] is False
    assert cost["usage"]["total_tokens"] == 1801
    all_bytes = b"".join(path.read_bytes() for path in run_dir.iterdir() if path.is_file())
    assert b"secret-test-key" not in all_bytes


def test_cost_aware_planner_uses_zero_cost_deterministic_route_without_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    adaptive = task(source_id)
    adaptive["planner_contract_version"] = "v2"
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    store.create_tasks([adaptive])
    claimed = store.claim("worker-cost-aware", 60)
    assert claimed is not None
    packet = task_packet_for_planner(
        claimed,
        normalize_candidates({"candidates": claimed["recorded_candidates"]}, claimed),
        contract_version="v2",
        regional_attempt_history=_empty_v2_history(claimed),
    )

    def provider_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provider should not run for a no-history packet")

    monkeypatch.setattr(planner_module.urllib.request, "urlopen", provider_must_not_run)
    proposal = CostAwareSegmentationPlanner(cache_root=tmp_path / "cache").propose(
        packet, tmp_path / "run"
    )
    validate_and_lock(packet, proposal)
    receipt = json.loads(
        (tmp_path / "run/COST_AWARE_PLANNER_RECEIPT.json").read_text()
    )
    assert receipt["route"] == "DETERMINISTIC_NO_HISTORY"
    assert receipt["provider_call_count"] == 0


def test_cost_aware_runtime_profile_matches_code() -> None:
    profile = json.loads(
        (
            ROOT
            / "framework/profiles/01-segmentation/segmentation-planner-cost-aware-v2-1.0.0.json"
        ).read_text()
    )
    assert profile["direct_model"] == COST_AWARE_DIRECT_MODEL
    assert profile["decision_order"][0] == "DETERMINISTIC_PROBE_WINNER"
    assert profile["decision_order"][1] == "DETERMINISTIC_NO_HISTORY"
    assert profile["decision_order"][3] == "DIRECT_OPUS"
    assert profile["max_completion_tokens"] == 4096


def test_cost_aware_planner_calls_opus_first_with_compact_view_then_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    adaptive = task(source_id)
    adaptive["planner_contract_version"] = "v2"
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    store.create_tasks([adaptive])
    claimed = store.claim("worker-cost-aware", 60)
    assert claimed is not None
    history = _empty_v2_history(claimed)
    history["attempt_count"] = 1
    history["attempts"] = [
        {
            "history_id": "history-no-seed",
            "outcome": "NO_SEED",
            "selected_seed": None,
            "profile_id": None,
            "parameters": None,
            "recipe_sha256": None,
        }
    ]
    history["history_sha256"] = content_sha256(
        {key: value for key, value in history.items() if key != "history_sha256"}
    )
    packet = task_packet_for_planner(
        claimed,
        normalize_candidates({"candidates": claimed["recorded_candidates"]}, claimed),
        contract_version="v2",
        regional_attempt_history=history,
    )
    proposal = _v2_proposal(packet, candidate_id="c01")
    proposal["history_considered"] = ["history-no-seed"]
    calls: list[dict] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "gen-opus-direct",
                    "model": COST_AWARE_DIRECT_MODEL,
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(proposal),
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 700,
                        "completion_tokens": 300,
                        "total_tokens": 1000,
                        "cost": 0.0123,
                    },
                }
            ).encode()

    def fake_urlopen(request: object, timeout: int) -> FakeResponse:
        calls.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    monkeypatch.setattr(planner_module.urllib.request, "urlopen", fake_urlopen)
    planner = CostAwareSegmentationPlanner(cache_root=tmp_path / "cache")
    result = planner.propose(packet, tmp_path / "first")
    assert result == proposal
    assert len(calls) == 1
    assert calls[0]["model"] == COST_AWARE_DIRECT_MODEL
    prompt = calls[0]["messages"][0]["content"]
    assert "campaignx.segmentation_planner_decision_view.v1" in prompt
    assert "m7_query" not in prompt
    assert calls[0]["max_tokens"] == 4096
    first_receipt = json.loads(
        (tmp_path / "first/COST_AWARE_PLANNER_RECEIPT.json").read_text()
    )
    assert first_receipt["route"] == "DIRECT_OPUS"
    assert first_receipt["provider_call_count"] == 1

    cached = planner.propose(packet, tmp_path / "second")
    assert cached == proposal
    assert len(calls) == 1
    second_receipt = json.loads(
        (tmp_path / "second/COST_AWARE_PLANNER_RECEIPT.json").read_text()
    )
    assert second_receipt["route"] == "VALIDATED_CACHE"
    assert second_receipt["cache_hit"] is True


def test_worker_retries_three_operational_proposals_then_uses_v2_fallback(
    tmp_path: Path,
) -> None:
    class MalformedPlanner:
        contract_version = "v2"

        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def propose(
            self,
            _packet: dict,
            _run_dir: Path,
            repair_feedback: str | None = None,
        ) -> dict:
            self.calls.append(repair_feedback)
            return {"schema": "campaignx.segmentation_proposal.v2"}

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    adaptive = task(source_id)
    adaptive["planner_contract_version"] = "v2"
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    store.create_tasks([adaptive])
    planner = MalformedPlanner()
    result = SegmentWorker(
        store,
        "worker-repair",
        RecordedSeedProvider(),
        planner,
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
    ).run_one()
    assert result is not None and result["status"] == "QC_PENDING"
    assert len(planner.calls) == 3
    assert planner.calls[0] is None
    assert all("proposal fields differ" in str(value) for value in planner.calls[1:])
    attempt_dir = next((tmp_path / "runs").glob("*/*"))
    fallback = json.loads(
        (
            attempt_dir
            / "deterministic-fallback"
            / "DETERMINISTIC_FALLBACK_RECEIPT.json"
        ).read_text()
    )
    assert fallback["status"] == "DETERMINISTIC_FALLBACK_LOCKED"
    assert fallback["operational_attempt_count"] == 3
    assert json.loads((attempt_dir / "SEGMENTATION_PLAN.json").read_text())[
        "schema"
    ].endswith(".v2")


def test_planner_v2_reads_regional_failures_and_locks_bounded_variation(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    prior = task(source_id, "cell-a")
    store.create_tasks([prior])
    claimed_prior = store.claim("worker-prior", 60)
    assert claimed_prior is not None
    candidates = [
        {"candidate_id": "c01", "x": 256, "y": 256, "z": 256, "score": 1.0},
        {"candidate_id": "c02", "x": 200, "y": 200, "z": 200, "score": 0.8},
    ]
    prior_packet = task_packet_for_planner(claimed_prior, candidates)
    prior_proposal = DeterministicPlanner().propose(prior_packet, tmp_path)
    prior_plan = validate_and_lock(prior_packet, prior_proposal)
    store.transition(
        claimed_prior["task_id"],
        claimed_prior["attempt_id"],
        claimed_prior["lease_token"],
        "LOCKED_READY",
        proposal=prior_proposal,
        locked_plan=prior_plan,
    )
    store.mark_terminal(
        claimed_prior["task_id"],
        claimed_prior["attempt_id"],
        claimed_prior["lease_token"],
        "GROW_FAILED",
        {"status": "GROW_FAILED", "ink_used": False},
    )

    adaptive = task(source_id, "cell-a")
    adaptive["policy_version"] = "ink-blind-adaptive-v2"
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    store.create_tasks([adaptive])
    claimed = store.claim("worker-v2", 60)
    assert claimed is not None
    history = store.regional_attempt_history(claimed)
    assert history["attempt_count"] == 1
    assert history["attempts"][0]["outcome"] == "GROW_FAILED"
    assert "error" not in history["attempts"][0]
    packet = task_packet_for_planner(
        claimed,
        candidates,
        contract_version="v2",
        regional_attempt_history=history,
    )
    proposal = _v2_proposal(packet)
    locked = validate_and_lock(packet, proposal)
    assert locked["schema"] == "campaignx.segmentation_locked_plan.v2"
    assert locked["selected_seed"]["candidate_id"] == "c02"
    assert locked["parameters"]["generations"] == 30
    assert locked["regional_attempt_history_sha256"] == history["history_sha256"]
    schema_root = ROOT / "framework/contracts/schemas"
    for name in (
        "segmentation-regional-attempt-history-v1.schema.json",
        "segmentation-planner-packet-v2.schema.json",
        "segmentation-proposal-v2.schema.json",
        "segmentation-locked-plan-v2.schema.json",
    ):
        Draft202012Validator.check_schema(json.loads((schema_root / name).read_text()))
    Draft202012Validator(
        json.loads((schema_root / "segmentation-regional-attempt-history-v1.schema.json").read_text())
    ).validate(history)
    Draft202012Validator(
        json.loads((schema_root / "segmentation-proposal-v2.schema.json").read_text())
    ).validate(proposal)
    Draft202012Validator(
        json.loads((schema_root / "segmentation-locked-plan-v2.schema.json").read_text())
    ).validate(locked)


def test_planner_v2_rejects_history_omission_replay_and_out_of_range(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    store.create_tasks([task(source_id, "cell-a")])
    prior = store.claim("worker-prior", 60)
    assert prior is not None
    candidates = [
        {"candidate_id": "c01", "x": 256, "y": 256, "z": 256, "score": 1.0},
        {"candidate_id": "c02", "x": 200, "y": 200, "z": 200, "score": 0.8},
    ]
    prior_packet = task_packet_for_planner(prior, candidates)
    prior_proposal = DeterministicPlanner().propose(prior_packet, tmp_path)
    prior_plan = validate_and_lock(prior_packet, prior_proposal)
    store.transition(prior["task_id"], prior["attempt_id"], prior["lease_token"], "LOCKED_READY", proposal=prior_proposal, locked_plan=prior_plan)
    store.mark_terminal(prior["task_id"], prior["attempt_id"], prior["lease_token"], "GROW_FAILED", {"status": "GROW_FAILED", "ink_used": False})
    adaptive = task(source_id, "cell-a")
    adaptive["policy_version"] = "adaptive-v2"
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    store.create_tasks([adaptive])
    claimed = store.claim("worker-v2", 60)
    assert claimed is not None
    history = store.regional_attempt_history(claimed)
    packet = task_packet_for_planner(claimed, candidates, contract_version="v2", regional_attempt_history=history)

    omitted = _v2_proposal(packet)
    omitted["history_considered"] = []
    with pytest.raises(ValueError, match="every regional history ID"):
        validate_and_lock(packet, omitted)

    replay = _v2_proposal(packet, "c01")
    replay["parameters"]["generations"] = 35
    with pytest.raises(ValueError, match="repeats an exact failed regional recipe"):
        validate_and_lock(packet, replay)

    outside = _v2_proposal(packet)
    outside["parameters"]["step_size"] = 999
    with pytest.raises(ValueError, match="above maximum"):
        validate_and_lock(packet, outside)


class _V2FixturePlanner:
    contract_version = "v2"

    def propose(self, packet: dict, _run_dir: Path) -> dict:
        return _v2_proposal(packet)


def test_worker_v2_persists_history_packet_before_fixture_grow(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    source_id = source(store)
    adaptive = task(source_id)
    adaptive["candidate_selection_policy"] = "adaptive-geometry-history-v2"
    adaptive["planner_contract_version"] = "v2"
    store.create_tasks([adaptive])
    segment_worker = SegmentWorker(
        store,
        "worker-v2",
        RecordedSeedProvider(),
        _V2FixturePlanner(),
        FixtureGrowExecutor(),
        tmp_path / "runs",
        tmp_path / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
    )
    result = segment_worker.run_one()
    assert result is not None and result["status"] == "QC_PENDING"
    attempt_dir = next((tmp_path / "runs").glob("*/*"))
    assert json.loads((attempt_dir / "PLANNER_PACKET.json").read_text())["schema"].endswith(".v2")
    assert json.loads((attempt_dir / "REGIONAL_ATTEMPT_HISTORY.json").read_text())["attempt_count"] == 0
    assert json.loads((attempt_dir / "SEGMENTATION_PLAN.json").read_text())["schema"].endswith(".v2")


def test_opencode_v2_demo_builds_a_v2_packet_and_plan(tmp_path: Path) -> None:
    fake_opencode = tmp_path / "fake-opencode"
    fake_opencode.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path

packet = json.loads(Path("PLANNER_PACKET.json").read_text())
selected, rejected = packet["candidate_seeds"]
parameters = {
    name: rule.get("const", rule.get("default"))
    for name, rule in packet["parameter_envelope"]["parameters"].items()
}
proposal = {
    "schema": "campaignx.segmentation_proposal.v2",
    "task_id": packet["task_id"],
    "attempt_id": packet["attempt_id"],
    "selected_seed": {
        key: selected[key] for key in ("candidate_id", "x", "y", "z")
    },
    "profile_id": packet["parameter_envelope"]["profile_ids"][0],
    "parameters": parameters,
    "history_considered": [
        item["history_id"]
        for item in packet["regional_attempt_history"]["attempts"]
    ],
    "hypothesis": "Use the listed seed with the stronger geometric score.",
    "parameter_rationale": {
        name: "Use the bounded default for this no-history fixture."
        for name in parameters
    },
    "variation_summary": "No regional failure history existed.",
    "alternatives_rejected": [{
        "candidate_id": rejected["candidate_id"],
        "reason": "Lower listed geometric score and clearance."
    }],
    "ink_used": False,
}
print(json.dumps(proposal, separators=(",", ":")))
""",
        encoding="utf-8",
    )
    fake_opencode.chmod(0o755)
    demo_root = tmp_path / "demo-v2"
    args = type(
        "Args",
        (),
        {
            "root": demo_root,
            "repo_root": ROOT,
            "planner": "opencode-v2",
            "opencode": str(fake_opencode),
            "model": "fixture/model",
            "planner_timeout": 30,
        },
    )()

    assert command_demo(args) == 0
    attempt_dir = next((demo_root / "runs").glob("*/*"))
    assert json.loads((attempt_dir / "PLANNER_PACKET.json").read_text())["schema"].endswith(".v2")
    assert json.loads((attempt_dir / "SEGMENTATION_PROPOSAL.json").read_text())["schema"].endswith(".v2")
    assert json.loads((attempt_dir / "SEGMENTATION_PLAN.json").read_text())["schema"].endswith(".v2")
    receipt = json.loads((demo_root / "DEMO_RECEIPT.json").read_text())
    assert receipt["result"]["status"] == "FIXTURE_ONLY"
    assert receipt["status"]["qc_jobs"] == 0


def test_planner_v2_prompt_makes_rationale_string_contract_explicit() -> None:
    assert "EVERY rationale value MUST be a non-empty JSON" in PLANNER_PROMPT_V2
    assert "string enclosed in quotes" in PLANNER_PROMPT_V2
    assert 'NEVER use `"generations":35`' in PLANNER_PROMPT_V2
    assert "every `parameter_rationale` value is a quoted string" in PLANNER_PROMPT_V2
