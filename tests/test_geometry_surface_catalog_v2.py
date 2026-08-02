from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/build_geometry_surface_catalog_v2.py"
SPEC = importlib.util.spec_from_file_location("geometry_surface_catalog_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_surface(root: Path, sample: str, seed: str, bbox: list[list[float]], area: float) -> None:
    directory = root / sample / seed
    directory.mkdir(parents=True)
    for axis in "xyz":
        (directory / f"{axis}.tif").write_bytes((axis + seed).encode())
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "area_cm2": area,
                "bbox": bbox,
                "seed": [1, 2, 3],
                "target_volume": "m7.zarr",
                "uuid": seed,
                "vc_gsfs_params": {"generations": 35},
            }
        )
    )


def test_archive_is_authoritative_and_receipts_are_optional(tmp_path: Path) -> None:
    root, archive = tmp_path / "repo", tmp_path / "surfaces"
    root.mkdir()
    make_surface(archive, "PHerc1", "one", [[0, 0, 0], [10, 10, 10]], 1.25)
    make_surface(archive, "PHerc1", "two", [[9, 9, 9], [20, 20, 20]], 2.5)

    receipt_dir = root / "workspace/runs/a"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "GROWTH_RECEIPT.json").write_text(
        json.dumps(
            {
                "kind": "campaign_x_geometry_growth_receipt",
                "sample_id": "PHerc1",
                "seed_id": "one",
                "status": "PASSED",
                "area_cm2": 1.25,
            }
        )
    )
    (receipt_dir / "GEOMETRY_RECOVERY_SCREEN_EXECUTION.json").write_text(
        json.dumps(
            {
                "receipts": [
                    {"sample_id": "PHerc1", "seed_id": "one", "state": "COMPLETED_DIAGNOSTIC_ONLY"}
                ]
            }
        )
    )
    public = root / "public.json"
    public.write_text(json.dumps({"downloaded_measured_count": 0, "surfaces": []}))
    database, summary = tmp_path / "catalog.sqlite", tmp_path / "summary.json"

    result = MODULE.build(root, archive, public, database, summary)

    assert result["campaign_surface_count"] == 2
    assert result["campaign_with_receipt_count"] == 1
    assert result["campaign_without_receipt_count"] == 1
    assert result["campaign_aabb_overlap_warning_count"] == 1
    assert [row["seed_id"] for row in result["campaign_surfaces"]] == ["one", "two"]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM campaign_surfaces").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM growth_receipts").fetchone()[0] == 1


def test_incomplete_archive_surface_fails_closed(tmp_path: Path) -> None:
    root, archive = tmp_path / "repo", tmp_path / "surfaces"
    root.mkdir()
    directory = archive / "PHerc1/incomplete"
    directory.mkdir(parents=True)
    (directory / "x.tif").write_bytes(b"x")
    try:
        MODULE.build(root, archive, None, tmp_path / "catalog.sqlite", tmp_path / "summary.json")
    except RuntimeError as error:
        assert "incomplete archived TIFXYZ" in str(error)
    else:
        raise AssertionError("incomplete archive entry was silently ignored")
