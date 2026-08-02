from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/helena_prepare_surface_qc_backfill.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("helena_prepare_surface_qc_backfill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_backfill_verifies_files_and_writes_fleet_manifest(
    tmp_path: Path,
) -> None:
    module = load_module()
    surface = tmp_path / "surfaces/PHercTEST/surface-01"
    surface.mkdir(parents=True)
    for name, payload in {
        "x.tif": b"x-grid",
        "y.tif": b"y-grid",
        "z.tif": b"z-grid",
        "meta.json": b"{}\n",
    }.items():
        (surface / name).write_bytes(payload)
    files = module.artifact_manifest(surface, module.REQUIRED)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "campaign_surfaces": [
                    {
                        "surface_id": "campaign-x:PHercTEST:surface-01",
                        "sample_id": "PHercTEST",
                        "owner": "campaign-x",
                        "archive_relative": "PHercTEST/surface-01",
                        "tifxyz_sha256": "a" * 64,
                        "files": files,
                        "area_cm2": 1.5,
                        "bbox_l0_xyz": [[1, 2, 3], [4, 5, 6]],
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "BACKFILL.json"

    result = module.prepare(
        catalog_path=catalog,
        surface_root=tmp_path / "surfaces",
        artifact_uri_root="/workspace/surfaces",
        output=output,
        write_artifact_manifests=True,
    )

    assert result["surface_count"] == 1
    assert result["gross_area_cm2"] == 1.5
    row = result["surfaces"][0]
    assert row["artifact_uri"] == "/workspace/surfaces/PHercTEST/surface-01"
    assert row["artifact_sha256"] == module.content_sha256(files)
    artifact = json.loads((surface / "ARTIFACT_SET.json").read_text())
    assert artifact["artifact_sha256"] == row["artifact_sha256"]
    assert artifact["legacy_tifxyz_sha256"] == "a" * 64
    assert json.loads(output.read_text())["manifest_sha256"] == result["manifest_sha256"]


def test_prepare_backfill_preserves_compatible_richer_artifact_manifest(
    tmp_path: Path,
) -> None:
    module = load_module()
    surface = tmp_path / "surfaces/PHercTEST/surface-rich"
    surface.mkdir(parents=True)
    for name, payload in {
        "x.tif": b"x-grid",
        "y.tif": b"y-grid",
        "z.tif": b"z-grid",
        "meta.json": b"{}\n",
    }.items():
        (surface / name).write_bytes(payload)
    files = module.artifact_manifest(surface, module.REQUIRED)
    artifact_sha256 = module.content_sha256(files)
    richer = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "artifact_sha256": artifact_sha256,
        "files": files,
        "attempt_id": "attempt-preserved",
        "sample_points": [[1.0, 2.0, 3.0]],
    }
    artifact_path = surface / "ARTIFACT_SET.json"
    artifact_path.write_text(json.dumps(richer, sort_keys=True) + "\n")
    original_bytes = artifact_path.read_bytes()
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "campaign_surfaces": [
                    {
                        "surface_id": "campaign-x:PHercTEST:surface-rich",
                        "sample_id": "PHercTEST",
                        "archive_relative": "PHercTEST/surface-rich",
                        "tifxyz_sha256": "b" * 64,
                        "files": files,
                        "area_cm2": 1.0,
                        "bbox_l0_xyz": [[1, 2, 3], [4, 5, 6]],
                    }
                ]
            }
        )
        + "\n"
    )

    result = module.prepare(
        catalog_path=catalog,
        surface_root=tmp_path / "surfaces",
        artifact_uri_root="/immutable/surfaces",
        output=tmp_path / "BACKFILL.json",
        write_artifact_manifests=True,
    )

    assert artifact_path.read_bytes() == original_bytes
    assert result["surfaces"][0]["artifact_manifest_sha256"] == module.content_sha256(
        richer
    )
