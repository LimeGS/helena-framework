from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/helena_compare_surface_to_tifxyz_reference.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("helena_reference_agreement", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def _fixture(tmp_path: Path, *, candidate_z: float) -> tuple[Path, Path]:
    reference = tmp_path / "reference.tifxyz"
    reference.mkdir()
    yy, xx = np.mgrid[:9, :9]
    for name, array in {
        "x": xx.astype(np.float32) * 4.0 + 10.0,
        "y": yy.astype(np.float32) * 4.0 + 20.0,
        "z": np.full_like(xx, 30.0, dtype=np.float32),
    }.items():
        tifffile.imwrite(reference / f"{name}.tif", array)
    (reference / "meta.json").write_text('{"format":"tifxyz"}\n')

    candidate = tmp_path / "candidate.obj"
    lines = []
    size = 33
    for row in range(size):
        for column in range(size):
            lines.append(f"v {10 + column} {20 + row} {candidate_z}")
    for row in range(size - 1):
        for column in range(size - 1):
            a = row * size + column + 1
            b = a + 1
            c = a + size
            d = c + 1
            lines.extend((f"f {a} {b} {d}", f"f {a} {d} {c}"))
    candidate.write_text("\n".join(lines) + "\n")
    return candidate, reference


def _compare(candidate: Path, reference: Path) -> dict:
    return MODULE.compare(
        candidate,
        reference,
        roi_level0_zyx=[0, 0, 0, 100, 100, 100],
        threshold_voxels=2.0,
        minimum_candidate_fidelity_fraction=0.98,
        minimum_reference_recovery_fraction=0.80,
        minimum_normal_dot=0.866,
        maximum_reference_support_radius_voxels=10.0,
        maximum_candidate_samples=100000,
    )


def test_matching_plane_passes_reference_control(tmp_path: Path) -> None:
    candidate, reference = _fixture(tmp_path, candidate_z=30.0)
    result = _compare(candidate, reference)
    assert result["status"] == "PASS"
    assert result["metrics"]["candidate_reference_point_to_plane_voxels"]["p95"] == 0.0
    assert result["metrics"]["reference_recovery_fraction"] == 1.0
    assert result["topology_or_flattening_override_permitted"] is False


def test_offset_plane_fails_fidelity_without_changing_other_gates(tmp_path: Path) -> None:
    candidate, reference = _fixture(tmp_path, candidate_z=33.0)
    result = _compare(candidate, reference)
    assert result["status"] == "FAIL"
    assert result["requirements"]["candidate_point_to_plane_p95"]["status"] == "FAIL"
    assert result["requirements"]["candidate_within_tolerance_fraction"]["status"] == "FAIL"
