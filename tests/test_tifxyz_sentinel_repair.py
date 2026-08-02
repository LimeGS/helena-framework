from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(STAGE))

from fleet.common import artifact_manifest, content_sha256
from fleet.store import FleetStore


SCRIPT = STAGE / "scripts/repair_tifxyz_sentinel_metadata.py"


def load_script():
    spec = importlib.util.spec_from_file_location("repair_tifxyz_sentinel_metadata", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_updates_only_derived_metadata_and_records_event(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    run = tmp_path / "run"
    surface = run / "attempts/task-1/attempt-1/surface"
    surface.mkdir(parents=True)
    rows, columns = np.indices((3, 3), dtype=np.float64)
    arrays = [columns, rows, np.zeros((3, 3), dtype=np.float64)]
    for array in arrays:
        array[0, 0] = -1.0
    for axis, array in zip("xyz", arrays, strict=True):
        tifffile.imwrite(surface / f"{axis}.tif", array)
    (surface / "meta.json").write_text("{}\n", encoding="utf-8")
    files = artifact_manifest(surface, ("x.tif", "y.tif", "z.tif", "meta.json"))
    (surface.parent / "ARTIFACT_SET.json").write_text(
        json.dumps({"files": files, "artifact_sha256": content_sha256(files)}) + "\n",
        encoding="utf-8",
    )

    store = FleetStore(run / "control/fleet.sqlite")
    store.initialize()
    source_id = store.register_snapshot({
        "sample_id": "PHercTEST",
        "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64,
        "shape_xyz": [32, 32, 32],
        "voxel_size_um": 10_000.0,
    })
    store.import_surface({
        "surface_id": "surface-1",
        "source_snapshot_id": source_id,
        "sample_id": "PHercTEST",
        "owner": "campaign-x",
        "artifact_sha256": content_sha256(files),
        "artifact_uri": "fixture://surface-1",
        "bbox_xyz": [[-1.0, -1.0, -1.0], [2.0, 2.0, 0.0]],
        "sample_points": [[-1.0, -1.0, -1.0]],
        "area_cm2": 99.0,
        "task_id": "task-1",
        "attempt_id": "attempt-1",
    })

    module = load_script()
    receipt_path = run / "control/TIFXYZ_SENTINEL_REPAIR.json"
    receipt = module.build_repair(store.path, run)
    receipt.update({"status": "PREPARED", "generated_at_utc": "test"})
    module.write_json_atomic(receipt_path, receipt)
    module.apply_repair(store.path, receipt, receipt_path)

    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT area_cm2,bbox_xyz_json,payload_json FROM surfaces WHERE surface_id='surface-1'"
        ).fetchone()
        assert row is not None
        assert row[0] == pytest.approx(3.5)
        assert json.loads(row[1]) == [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]
        assert json.loads(row[2])["geometry_inspection_policy"] == module.POLICY
        assert connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type=?", (module.EVENT,)
        ).fetchone()[0] == 1

    # Re-applying the same receipt is idempotent and does not duplicate events.
    module.apply_repair(store.path, receipt, receipt_path)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM events WHERE event_type=?", (module.EVENT,)
        ).fetchone()[0] == 1
