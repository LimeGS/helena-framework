from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/helena_run_vc3d_locked_plan.py"


def plan(path: Path) -> Path:
    value = {
        "schema": "campaignx.segment_fleet_locked_plan.v1",
        "task_id": "task",
        "attempt_id": "attempt",
        "sample_id": "PHerc-test",
        "backend_profile": "segmentation.vc3d-m7-grow@1.0.0",
        "roi_level0_zyx": [0, 0, 0, 128, 128, 128],
        "source": {
            "m7_uri": "https://example.test/m7.zarr",
            "m7_etag": "etag",
            "ct_uri": "https://example.test/ct.zarr",
            "ct_etag": "etag",
            "voxel_size_um": 9.362,
        },
        "selected_seed": {"x": 1, "y": 2, "z": 3},
        "parameters": {"generations": 1, "step_size": 20, "min_area_cm": 0.0, "use_cuda": False},
        "ink_used": False,
        "non_claim": "fixture",
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fake_binary(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "import numpy as np, tifffile\n"
        "out=pathlib.Path(sys.argv[sys.argv.index('--target-dir')+1])\n"
        "out.mkdir()\n"
        "a=np.arange(4,dtype=np.float32).reshape(2,2)\n"
        "[tifffile.imwrite(out/f'{axis}.tif',a) for axis in 'xyz']\n"
        "tifffile.imwrite(out/'generations.tif',"
        "np.ones((2,2),dtype=np.uint16))\n"
        "(out/'meta.json').write_text(json.dumps({'area_cm2':1.0}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_locked_plan_cli_produces_a_hashed_growth_receipt(tmp_path: Path) -> None:
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plan",
            str(plan(tmp_path / "plan.json")),
            "--output-dir",
            str(output),
            "--vc3d-binary",
            str(fake_binary(tmp_path / "vc3d")),
            "--minimum-free-gib",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((output / "GROWTH_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "GROW_SUCCEEDED"
    assert receipt["complete_tifxyz"] is True
    assert receipt["compute_device"] == "cpu"
    assert receipt["elapsed_seconds"] >= 0
    assert len(receipt["rng_seed"]) == 16
    assert receipt["rng_seed_source"] == "attempt-id-sha256-prefix-v1"
    assert receipt["started_at_utc"].endswith("Z")
    assert receipt["completed_at_utc"].endswith("Z")
    assert (output / "surface/generations.tif").is_file()
    assert json.loads((output / "VC3D_LOCKED_PLAN_EXECUTION.json").read_text())["status"] == "SUCCEEDED"


def test_locked_plan_cli_refuses_output_reuse(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--plan",
            str(plan(tmp_path / "plan.json")),
            "--output-dir",
            str(output),
            "--vc3d-binary",
            str(fake_binary(tmp_path / "vc3d")),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "immutable absolute output-dir required" in completed.stderr
