from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.store import FleetStore  # noqa: E402


def test_sqlite_geometry_backlog_queries_the_real_surfaces_table(tmp_path):
    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshot = store.register_snapshot({
        "sample_id": "S1", "ct_uri": "https://example/ct",
        "m7_uri": "https://example/m7", "shape_xyz": [2, 2, 2],
        "voxel_size_um": 1.0, "coordinate_frame": "ct_l0_xyz"})
    with store.connect() as connection:
        connection.execute(
            "INSERT INTO surfaces(surface_id,source_snapshot_id,sample_id,owner,"
            "artifact_sha256,artifact_uri,bbox_xyz_json,state,physical_qc_state,"
            "geometry_qc_state,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            ("surface-1", snapshot, "S1", "test", "a" * 64, "s3://surface-1",
             "[[0,0,0],[1,1,1]]", "QC_PENDING", "UNVALIDATED",
             "GEOMETRY_UNMEASURED", "{}", "2026-08-03T00:00:00+00:00"))
    assert store.surfaces_without_geometry_verdict(
        sample_id="S1", surface_id="surface-1") == [{
            "surface_id": "surface-1", "sample_id": "S1",
            "artifact_uri": "s3://surface-1", "artifact_sha256": "a" * 64,
            "state": "QC_PENDING", "geometry_qc_state": "GEOMETRY_UNMEASURED"}]
