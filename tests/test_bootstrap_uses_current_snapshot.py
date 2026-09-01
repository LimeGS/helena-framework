from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.generator import _current_bootstrap_snapshots, bootstrap_queue
from fleet.store import FleetStore


def test_bootstrap_queues_only_the_current_eligible_snapshot(tmp_path: Path) -> None:
    """A current catalog source must not fan a bootstrap out to its history."""
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    old_snapshot_id = store.register_snapshot({
        "sample_id": "PHerc826",
        "ct_uri": "fixture://ct/PHerc826-historical",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7/PHerc826-historical",
        "m7_sha256": "1" * 64,
        "shape_xyz": [512, 512, 512],
        "voxel_size_um": 7.91,
        "coordinate_frame": "ct_l0_xyz",
    })
    eligible_path = tmp_path / "eligible.json"
    eligible_path.write_text(json.dumps({"entries": [{
        "sample_id": "PHerc826",
        "ct_uri": "fixture://ct/PHerc826-current",
        "ct_sha256": "2" * 64,
        "surface_prediction_uri": "fixture://m7/PHerc826-current",
        "surface_prediction_sha256": "3" * 64,
        "shape_zyx": [512, 512, 512],
        "voxel_size_um": 7.91,
    }]}), encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "campaign_surfaces": [],
        "public_surfaces": [],
    }), encoding="utf-8")

    receipt = bootstrap_queue(
        store,
        eligible_path,
        catalog_path,
        samples={"PHerc826"},
        grid_step=128,
        query_radius=64,
        clearance=0.0,
        volume_edge_margin=64,
        candidate_interior_clearance=64,
        selection_strategy="max-clearance-v1",
        max_tasks_per_sample=4,
        grid_version="current-snapshot-regression-v1",
        policy_version="current-snapshot-regression-v1",
        verify_sources=False,
    )

    current_snapshot_id = receipt["sources"]["PHerc826"]
    with store.connect() as connection:
        task_snapshot_ids_with_repetition = [
            row[0]
            for row in connection.execute(
                "SELECT source_snapshot_id FROM tasks ORDER BY task_id"
            )
        ]
    task_snapshot_ids = set(task_snapshot_ids_with_repetition)

    assert receipt["tasks"] == {
        "PHerc826": {"generated": 4, "inserted": 4}
    }
    assert task_snapshot_ids == {current_snapshot_id}
    assert len(task_snapshot_ids_with_repetition) == 4
    assert {row["source_snapshot_id"] for row in store.snapshots({"PHerc826"})} == {
        old_snapshot_id,
        current_snapshot_id,
    }


@pytest.mark.parametrize(
    ("case", "rows", "sources", "message", "identifiers"),
    [
        (
            "missing",
            [{
                "source_snapshot_id": "stored-snapshot",
                "sample_id": "sample-a",
                "ct_uri": "postgresql://worker:password@ct.example/source",
                "m7_uri": "s3://worker:password@m7.example/source",
            }],
            {"sample-a": "current-snapshot"},
            "current source snapshot .* is absent",
            ("current-snapshot", "sample-a"),
        ),
        (
            "duplicate",
            [
                {
                    "source_snapshot_id": "duplicate-snapshot",
                    "sample_id": "sample-a",
                    "ct_uri": "postgresql://worker:password@ct.example/source",
                    "m7_uri": "s3://worker:password@m7.example/source",
                },
                {
                    "source_snapshot_id": "duplicate-snapshot",
                    "sample_id": "sample-a",
                    "ct_uri": "postgresql://worker:password@ct.example/other",
                    "m7_uri": "s3://worker:password@m7.example/other",
                },
            ],
            {"sample-a": "duplicate-snapshot"},
            "source snapshot id .* appears more than once",
            ("duplicate-snapshot",),
        ),
        (
            "sample mismatch",
            [{
                "source_snapshot_id": "mismatched-snapshot",
                "sample_id": "stored-sample",
                "ct_uri": "postgresql://worker:password@ct.example/source",
                "m7_uri": "s3://worker:password@m7.example/source",
            }],
            {"requested-sample": "mismatched-snapshot"},
            "belongs to .* not .*",
            ("mismatched-snapshot", "stored-sample", "requested-sample"),
        ),
    ],
    ids=["missing", "duplicate", "sample-mismatch"],
)
def test_current_bootstrap_snapshots_fail_closed_without_source_secrets(
    case: str,
    rows: list[dict[str, str]],
    sources: dict[str, str],
    message: str,
    identifiers: tuple[str, ...],
) -> None:
    """Reject stale catalog bindings without echoing snapshot endpoint data."""
    with pytest.raises(ValueError, match=message) as error:
        _current_bootstrap_snapshots(rows, sources)

    error_message = str(error.value).lower()
    assert all(identifier in error_message for identifier in identifiers)
    assert all(
        row[uri_field] not in error_message
        for row in rows
        for uri_field in ("ct_uri", "m7_uri")
    )
    assert all(secret not in error_message for secret in (
        "ct_uri", "m7_uri", "postgresql", "worker", "password",
    ))
