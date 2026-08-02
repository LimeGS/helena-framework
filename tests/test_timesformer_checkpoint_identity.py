from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework"
    / "stages"
    / "03-ink"
    / "scripts"
    / "run_ink_timesformer.py"
)
SPEC = importlib.util.spec_from_file_location("timesformer_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "method_id": "model-a@1.0.0",
                        "known_checkpoint_sha256": "a" * 64,
                        "receipt_model_family_aliases": ["family-a"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_registered_hash_and_family_resolve(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    write_registry(registry)
    result = MODULE.resolve_checkpoint_identity(
        registry_path=registry,
        checkpoint_sha256="a" * 64,
        declared_model_family="family-a",
        allow_unregistered_checkpoint=False,
    )
    assert result["status"] == "REGISTERED_CHECKPOINT_FAMILY_MATCH"
    assert result["method_id"] == "model-a@1.0.0"


def test_known_hash_with_wrong_family_always_fails(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    write_registry(registry)
    with pytest.raises(RuntimeError, match="disagrees"):
        MODULE.resolve_checkpoint_identity(
            registry_path=registry,
            checkpoint_sha256="a" * 64,
            declared_model_family="wrong-family",
            allow_unregistered_checkpoint=True,
        )


def test_unknown_hash_requires_explicit_escape_hatch(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    write_registry(registry)
    with pytest.raises(RuntimeError, match="not registered"):
        MODULE.resolve_checkpoint_identity(
            registry_path=registry,
            checkpoint_sha256="b" * 64,
            declared_model_family="new",
            allow_unregistered_checkpoint=False,
        )
    result = MODULE.resolve_checkpoint_identity(
        registry_path=registry,
        checkpoint_sha256="b" * 64,
        declared_model_family="new",
        allow_unregistered_checkpoint=True,
    )
    assert result["status"] == "UNREGISTERED_CHECKPOINT_EXPLICITLY_ALLOWED"


def test_tiff_stack_accepts_only_native_uint8(tmp_path: Path) -> None:
    Image.fromarray(np.full((8, 9), 17, dtype=np.uint8)).save(tmp_path / "0.tif")
    Image.fromarray(np.full((8, 9), 23, dtype=np.uint8)).save(tmp_path / "1.tif")
    stack, files, ordering = MODULE.load_tiff_stack(tmp_path)
    assert stack.dtype == np.uint8
    assert stack.shape == (2, 8, 9)
    assert [path.name for path in files] == ["0.tif", "1.tif"]
    assert ordering == "NUMERIC_STEM_ASCENDING"


def test_tiff_stack_rejects_16_bit_instead_of_low_byte_cast(tmp_path: Path) -> None:
    Image.fromarray(np.full((8, 9), 1024, dtype=np.uint16)).save(tmp_path / "0.tif")
    with pytest.raises(RuntimeError, match="accepts only native uint8"):
        MODULE.load_tiff_stack(tmp_path)


def test_tile_filter_reports_discarded_fraction_explicitly() -> None:
    stack = np.zeros((2, 64, 64), dtype=np.uint8)
    _prediction, _valid, counts = MODULE.infer_map(
        stack,
        object(),
        device="cpu",
        tile_size=32,
        stride=16,
        tiling_offset=0,
        batch_size=8,
        min_valid_ratio=0.6,
    )
    assert counts["candidate_tiles"] == 0
    assert counts["discarded_tiles"] == counts["all_grid_tiles"]
    assert counts["candidate_tile_fraction"] == 0.0
    assert counts["discarded_tile_fraction"] == 1.0


def test_target_screening_policy_filters_physical_scale_before_diversity() -> None:
    policy = json.loads(
        (
            ROOT
            / "framework/policies/03-ink/target-screening-physical-scale-policy-1.0.0.json"
        ).read_text()
    )
    assert policy["policy_order"][0] == "physical_scale_compatibility"
    assert (
        policy["admissible_source_to_training_linear_factor"]["maximum"] == 1.25
    )
    development = {
        row.get("method_id", row.get("method_family"))
        for row in policy["public_development_only"]
    }
    assert "ink-3d-dino-guided@1.0.0" in development
    assert "fiber-ink-4class" in development
