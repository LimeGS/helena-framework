from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .common import utc_now, write_json_atomic
from .executor import FixtureGrowExecutor
from .generator import DEFAULT_ENVELOPE
from .planner import DeterministicPlanner
from .store_factory import open_fleet_store, store_identity
from .worker import RecordedSeedProvider, SegmentWorker


def _task(source_id: str, sample_id: str, cell_id: str, run_id: str) -> dict[str, Any]:
    return {
        "source_snapshot_id": source_id,
        "sample_id": sample_id,
        "cell_id": cell_id,
        "grid_version": f"distributed-probe-{run_id}",
        "policy_version": "fixture-only-v1",
        "bounds_xyz": [[128, 128, 128], [384, 384, 384]],
        "center_xyz": {"x": 256, "y": 256, "z": 256},
        "priority": 1.0,
        "parameter_envelope": DEFAULT_ENVELOPE,
        "catalog_snapshot_sha256": "2" * 64,
        "recorded_candidates": [
            {
                "candidate_id": "fixture-seed",
                "coordinate": {"x": 256, "y": 256, "z": 256},
                "score": 1.0,
            }
        ],
        "fixture_only": True,
        "ink_used": False,
    }


def run_probe(
    database: str,
    artifact_root: str,
    work_root: Path,
    run_id: str,
) -> dict[str, Any]:
    store = open_fleet_store(database)
    store.initialize()
    sample_id = f"PHercDISTRIBUTEDTEST-{run_id}"
    source_id = store.register_snapshot(
        {
            "sample_id": sample_id,
            "ct_uri": f"fixture://ct/{run_id}",
            "ct_sha256": "0" * 64,
            "m7_uri": f"fixture://m7/{run_id}",
            "m7_sha256": "1" * 64,
            "shape_xyz": [512, 512, 512],
            "voxel_size_um": 9.362,
            "coordinate_frame": "ct_l0_xyz",
            "fixture_only": True,
        }
    )
    inserted, seen = store.create_tasks(
        [
            _task(source_id, sample_id, "cell-a", run_id),
            _task(source_id, sample_id, "cell-b", run_id),
        ]
    )
    if (inserted, seen) != (2, 2):
        raise RuntimeError(f"probe tasks are not new: inserted={inserted}, seen={seen}")

    def execute(worker_id: str) -> dict[str, Any] | None:
        worker = SegmentWorker(
            store,
            worker_id,
            RecordedSeedProvider(),
            DeterministicPlanner(),
            FixtureGrowExecutor(),
            work_root / run_id / "runs",
            artifact_root,
            "fixture-surface-qc@1.0.0",
            lease_seconds=60,
        )
        return worker.run_one()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(execute, (f"{run_id}-gpu0", f"{run_id}-gpu1")))
    statuses = sorted(result["status"] for result in results if result is not None)
    surfaces = store.surfaces_for_snapshot(source_id)
    if statuses != ["DUPLICATE_SURFACE", "FIXTURE_ONLY"]:
        raise RuntimeError(f"unexpected terminal states: {statuses}")
    if len(surfaces) != 1:
        raise RuntimeError(f"atomic catalogue expected one surface, found {len(surfaces)}")
    receipt = {
        "schema": "campaignx.segment_fleet_distributed_probe.v1",
        "run_id": run_id,
        "generated_at_utc": utc_now(),
        "database": store_identity(store),
        "artifact_root": artifact_root,
        "source_snapshot_id": source_id,
        "statuses": statuses,
        "catalogued_surface_id": surfaces[0]["surface_id"],
        "catalogued_surface_count": len(surfaces),
        "artifact_uris": sorted(
            {result["surface"]["artifact_uri"] for result in results if result}
        ),
        "fixture_only": True,
        "scientific_geometry": False,
        "pass": True,
    }
    receipt_path = work_root / run_id / "DISTRIBUTED_PROBE_RECEIPT.json"
    write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fixture-only two-worker PostgreSQL+artifact-store integration probe."
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.db, args.artifact_root, args.work_root, args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
