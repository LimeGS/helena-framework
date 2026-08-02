from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/06-discovery/scripts"))

from run_pherc1667_iteration0 import (
    EXPECTED_DEPTH,
    derive_fragment_mask,
    eligible_coordinates,
    grid_positions,
    load_stack,
    ordered_tiff_files,
    preprocess_stack,
)


def write_layer(path: Path, value: int = 1) -> None:
    Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(path)


def test_numeric_tiffs_are_ordered_by_integer_stem(tmp_path: Path) -> None:
    write_layer(tmp_path / "10.tif")
    write_layer(tmp_path / "8.tif")
    write_layer(tmp_path / "9.tif")
    files, policy = ordered_tiff_files(tmp_path)
    assert [path.stem for path in files] == ["8", "9", "10"]
    assert policy == "NUMERIC_STEM_CONTIGUOUS_ASCENDING"


def test_numeric_tiffs_must_be_contiguous(tmp_path: Path) -> None:
    write_layer(tmp_path / "0.tif")
    write_layer(tmp_path / "2.tif")
    with pytest.raises(RuntimeError, match="contiguous"):
        ordered_tiff_files(tmp_path)


def test_load_stack_requires_exact_model_depth(tmp_path: Path) -> None:
    for index in range(EXPECTED_DEPTH - 1):
        write_layer(tmp_path / f"{index}.tif", index % 255)
    with pytest.raises(RuntimeError, match="exactly 62"):
        load_stack(tmp_path, reverse_layers=False)


def test_load_stack_can_reverse_physical_layer_order(tmp_path: Path) -> None:
    for index in range(EXPECTED_DEPTH):
        write_layer(tmp_path / f"{index}.tif", index)
    stack, files, policy = load_stack(tmp_path, reverse_layers=True)
    assert int(stack[0, 0, 0]) == EXPECTED_DEPTH - 1
    assert int(stack[-1, 0, 0]) == 0
    assert files[0].stem == str(EXPECTED_DEPTH - 1)
    assert policy.endswith("_THEN_REVERSED")


def test_grid_includes_terminal_edge_without_duplicates() -> None:
    assert grid_positions(600, 256, 128) == [0, 128, 256, 344]
    assert grid_positions(512, 256, 128) == [0, 128, 256]


def test_fragment_mask_is_any_nonzero_depth() -> None:
    stack = np.zeros((2, 3, 4), dtype=np.uint8)
    stack[1, 2, 3] = 8
    mask = derive_fragment_mask(stack)
    assert mask.sum() == 1
    assert mask[2, 3]


def test_albumentations_transform_clips_and_divides_by_255() -> None:
    stack = np.array([[[0, 100, 200, 255]]], dtype=np.uint8)
    transformed = preprocess_stack(stack, "clip-divide-255")
    np.testing.assert_allclose(
        transformed,
        np.array([[[0, 100 / 255, 200 / 255, 200 / 255]]], dtype=np.float32),
    )


def test_published_snippet_transform_is_explicitly_preserved() -> None:
    stack = np.array([[[0, 100, 200, 255]]], dtype=np.uint8)
    transformed = preprocess_stack(stack, "clip-float-no-scaling")
    np.testing.assert_array_equal(
        transformed,
        np.array([[[0, 100, 200, 200]]], dtype=np.float32),
    )


def test_eligible_coordinates_enforce_frozen_valid_ratio() -> None:
    mask = np.ones((512, 512), dtype=bool)
    mask[:128, :128] = False
    strict, considered = eligible_coordinates(
        mask,
        tile=256,
        stride=256,
        min_valid_ratio=1.0,
    )
    relaxed, _ = eligible_coordinates(
        mask,
        tile=256,
        stride=256,
        min_valid_ratio=0.75,
    )
    assert considered == 4
    assert strict == [(0, 256), (256, 0), (256, 256)]
    assert relaxed == [(0, 0), (0, 256), (256, 0), (256, 256)]
