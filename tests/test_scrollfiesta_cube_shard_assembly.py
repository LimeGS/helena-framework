from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/helena_assemble_scrollfiesta_cube_shards.py"
BACKENDS = ROOT / "framework/stages/01-segmentation/backends"
import sys

sys.path.insert(0, str(BACKENDS))
from scrollfiesta.cube_stage import CubeStageError  # noqa: E402
from scrollfiesta.receipt import file_artifact  # noqa: E402


def load_module():
    spec = spec_from_file_location("scrollfiesta_shard_assembly", SCRIPT)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shard(tmp_path: Path, cube_id: str, x: int) -> Path:
    root = tmp_path / cube_id
    final_name = f"{cube_id}_step12_final"
    final = root / "dump" / cube_id / final_name
    final.mkdir(parents=True)
    obj = final / f"{final_name}_all.obj"
    obj.write_text(f"v {x} 0 0\n", encoding="utf-8")
    receipt = root / "CUBE_STAGE_RECEIPT.json"
    value = {
        "schema": "campaignx.scrollfiesta_compact_cube_stage.v1",
        "status": "SUCCEEDED",
        "complete_fraction": 1.0,
        "source": {"uri": "https://example.test/m7.zarr", "sha256_or_etag": "etag", "level": 0},
        "parameters": {
            "halo": 13,
            "threshold": "nonzero",
            "resolved_upstream_threshold": ">=1",
            "cube_edge_voxels": 128,
        },
        "results": [
            {"cube_id": cube_id, "returncode": 0, "produced_obj": True, "obj": file_artifact(obj)}
        ],
    }
    receipt.write_text(json.dumps(value), encoding="utf-8")
    return receipt


def request(tmp_path: Path, receipts: list[Path]) -> tuple[dict, Path]:
    value = {
        "schema": "campaignx.scrollfiesta_cube_shard_assembly_request.v1",
        "canonical_roi_level0_zyx": [0, 0, 0, 128, 128, 256],
        "level": 0,
        "cube_edge_voxels": 128,
        "shard_receipts": [str(path) for path in receipts],
        "output_dir": str(tmp_path / "assembled"),
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return value, path


def test_assembly_requires_and_hardlinks_the_exact_canonical_cube_union(tmp_path: Path) -> None:
    module = load_module()
    first = shard(tmp_path, "z00000_y00000_x00000", 0)
    second = shard(tmp_path, "z00000_y00000_x00128", 128)
    value, path = request(tmp_path, [first, second])

    receipt_path = module.assemble(value, path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["assembled_cube_count"] == 2
    assert receipt["complete_fraction"] == 1.0
    source = Path(first).parent / "dump/z00000_y00000_x00000/z00000_y00000_x00000_step12_final/z00000_y00000_x00000_step12_final_all.obj"
    assembled = Path(value["output_dir"]) / "dump/z00000_y00000_x00000/z00000_y00000_x00000_step12_final/z00000_y00000_x00000_step12_final_all.obj"
    assert source.stat().st_ino == assembled.stat().st_ino


def test_assembly_fails_closed_for_an_incomplete_union(tmp_path: Path) -> None:
    module = load_module()
    first = shard(tmp_path, "z00000_y00000_x00000", 0)
    value, path = request(tmp_path, [first])
    with pytest.raises(CubeStageError, match="incomplete shard union"):
        module.assemble(value, path)


def test_assembly_rejects_mixed_runtime_binaries(tmp_path: Path) -> None:
    module = load_module()
    first = shard(tmp_path, "z00000_y00000_x00000", 0)
    second = shard(tmp_path, "z00000_y00000_x00128", 128)
    first_value = json.loads(first.read_text(encoding="utf-8"))
    second_value = json.loads(second.read_text(encoding="utf-8"))
    first_value["runtime"] = {"cube_mesh": {"sha256": "a" * 64}}
    second_value["runtime"] = {"cube_mesh": {"sha256": "b" * 64}}
    first.write_text(json.dumps(first_value), encoding="utf-8")
    second.write_text(json.dumps(second_value), encoding="utf-8")
    value, path = request(tmp_path, [first, second])

    with pytest.raises(CubeStageError, match="runtime binaries disagree"):
        module.assemble(value, path)
