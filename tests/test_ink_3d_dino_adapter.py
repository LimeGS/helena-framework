from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/03-ink/scripts/run_ink_3d_dino.py"
SPEC = importlib.util.spec_from_file_location("ink_3d_dino", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_manifest(tmp_path: Path, *, voxel_um: float = 2.4) -> Path:
    array_path = tmp_path / "patch.npy"
    np.save(array_path, np.zeros((256, 256, 256), dtype=np.uint8), allow_pickle=False)
    manifest = {
        "schema": "campaignx.ink_volumetric_patch_input.v1",
        "sample_id": "fixture",
        "array": {
            "path": "patch.npy",
            "sha256": MODULE.sha256_file(array_path),
            "format": "npy",
            "dtype": "uint8",
            "shape_zyx": [256, 256, 256],
        },
        "input_voxel_size_um": voxel_um,
        "source_volume": {"uri": "fixture", "array_path": "0", "voxel_size_um": voxel_um},
        "extraction": {"center_xyz": [128, 128, 128], "bounds_zyx": [[0, 256]] * 3, "resampled": False},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_manifest_validation_binds_array_and_scale(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest, array_path = MODULE.validate_manifest(manifest_path, model_voxel_um=2.4)
    assert manifest["sample_id"] == "fixture"
    assert array_path == (tmp_path / "patch.npy").resolve()


def test_manifest_validation_rejects_voxel_drift(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path, voxel_um=9.362)
    with pytest.raises(RuntimeError, match="incompatible with frozen model scale"):
        MODULE.validate_manifest(manifest_path, model_voxel_um=2.4)


def test_manifest_validation_rejects_array_tampering(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    with (tmp_path / "patch.npy").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="input array SHA-256 mismatch"):
        MODULE.validate_manifest(manifest_path, model_voxel_um=2.4)


def test_resampling_requires_receipt(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["extraction"]["resampled"] = True
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="resampling_receipt"):
        MODULE.validate_manifest(manifest_path, model_voxel_um=2.4)


def test_execution_profile_binds_method_profile_and_adapter() -> None:
    profile_path = ROOT / MODULE.PROFILE_RELATIVE_PATH
    profile, identity = MODULE.load_execution_profile(profile_path, repo_root=ROOT)
    assert profile["method_id"] == MODULE.METHOD_ID
    assert identity["sha256"] == MODULE.sha256_file(profile_path)
    assert identity["adapter_sha256"] == MODULE.sha256_file(SCRIPT)


def test_execution_profile_rejects_caller_substitution(tmp_path: Path) -> None:
    substitute = tmp_path / "profile.json"
    substitute.write_text(
        (ROOT / MODULE.PROFILE_RELATIVE_PATH).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="repository-pinned profile"):
        MODULE.load_execution_profile(substitute, repo_root=ROOT)
