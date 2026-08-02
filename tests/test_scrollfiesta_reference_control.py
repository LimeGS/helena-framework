from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/01-segmentation/scripts/helena_build_surface_reference_control.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("helena_reference_control", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def test_control_selection_is_reference_only_aligned_and_hashed(tmp_path: Path) -> None:
    reference = tmp_path / "known.tifxyz"
    reference.mkdir()
    yy, xx = np.mgrid[:17, :19]
    valid = (yy >= 2) & (yy <= 14) & (xx >= 3) & (xx <= 16)
    arrays = {
        "x": np.where(valid, 1000.0 + xx * 3.0, -1.0).astype(np.float32),
        "y": np.where(valid, 2000.0 + yy * 2.0, -1.0).astype(np.float32),
        "z": np.where(valid, 3000.0 + xx + yy, -1.0).astype(np.float32),
    }
    for name, array in arrays.items():
        tifffile.imwrite(reference / f"{name}.tif", array)
    (reference / "meta.json").write_text(json.dumps({"format": "tifxyz"}))

    first = MODULE.select_control(
        reference,
        sample_id="PHercControl",
        segment_id="known-01",
        cube_edge=128,
        cubes_per_axis=2,
        voxel_size_um=9.362,
    )
    second = MODULE.select_control(
        reference,
        sample_id="PHercControl",
        segment_id="known-01",
        cube_edge=128,
        cubes_per_axis=2,
        voxel_size_um=9.362,
    )

    assert first["anchor"] == second["anchor"]
    assert first["roi"] == second["roi"]
    assert first["selection_uses_candidate_output"] is False
    assert first["selection_uses_ink"] is False
    assert first["roi"]["reference_point_count"] > 0
    lower = first["roi"]["level0_zyx"][:3]
    upper = first["roi"]["level0_zyx"][3:]
    assert all(value % 128 == 0 for value in lower + upper)
    assert all(hi - lo == 256 for lo, hi in zip(lower, upper, strict=True))
    assert set(first["reference"]["files"]) == {"meta.json", "x.tif", "y.tif", "z.tif"}
    assert all(len(item["sha256"]) == 64 for item in first["reference"]["files"].values())


def test_control_selection_rejects_invalid_reference(tmp_path: Path) -> None:
    reference = tmp_path / "invalid.tifxyz"
    reference.mkdir()
    for axis in "xyz":
        tifffile.imwrite(reference / f"{axis}.tif", np.full((4, 4), -1.0, dtype=np.float32))
    (reference / "meta.json").write_text("{}")

    try:
        MODULE.select_control(
            reference,
            sample_id="PHercControl",
            segment_id="empty",
            cube_edge=128,
            cubes_per_axis=2,
            voxel_size_um=9.362,
        )
    except MODULE.ControlSelectionError as exc:
        assert "fewer than nine" in str(exc)
    else:
        raise AssertionError("invalid reference must fail closed")
