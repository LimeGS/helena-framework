from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/helena_measure_tifxyz_agreement.py"


def surface(path: Path, *, z_offset: float = 0.0) -> Path:
    path.mkdir()
    yy, xx = np.mgrid[0:3, 0:3]
    values = {
        "x": xx.astype(np.float32),
        "y": yy.astype(np.float32),
        "z": np.full((3, 3), z_offset, dtype=np.float32),
    }
    for axis, value in values.items():
        tifffile.imwrite(path / f"{axis}.tif", value)
    (path / "meta.json").write_text(json.dumps({"area_cm2": 123.0}), encoding="utf-8")
    return path


def run(tmp_path: Path, offset: float) -> dict:
    vc3d = surface(tmp_path / "vc3d")
    scrollfiesta = surface(tmp_path / "scrollfiesta", z_offset=offset)
    output = tmp_path / "agreement.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--vc3d",
            str(vc3d),
            "--scrollfiesta",
            str(scrollfiesta),
            "--roi",
            "-1",
            "-1",
            "-1",
            "2",
            "4",
            "4",
            "--voxel-size-um",
            "10",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_identical_tifxyz_surfaces_have_zero_distance_and_matching_normals(tmp_path: Path) -> None:
    result = run(tmp_path, 0.0)
    directed = result["directed"]["vc3d_to_scrollfiesta"]
    assert directed["distance_voxels"]["p95"] == 0.0
    assert directed["normal_dot_absolute"]["p05"] == 1.0
    assert result["surfaces"]["vc3d"]["summary"]["triangle_count_in_roi"] == 8


def test_tifxyz_agreement_reports_a_known_one_voxel_offset(tmp_path: Path) -> None:
    result = run(tmp_path, 1.0)
    directed = result["directed"]["vc3d_to_scrollfiesta"]
    assert directed["distance_voxels"]["p50"] == 1.0
    assert directed["distance_voxels"]["p95"] == 1.0

