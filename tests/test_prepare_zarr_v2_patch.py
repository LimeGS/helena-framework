from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/03-ink/scripts/prepare_zarr_v2_patch.py"
SPEC = importlib.util.spec_from_file_location("prepare_zarr_patch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_extract_patch_reassembles_cross_chunk_request(monkeypatch) -> None:
    source = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    metadata = {
        "zarr_format": 2,
        "shape": [8, 8, 8],
        "chunks": [4, 4, 4],
        "dtype": "<u2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
        "compressor": None,
    }

    def fake_fetch(url: str, *, allow_missing: bool = False):  # noqa: ARG001
        if url.endswith(".zarray"):
            return json.dumps(metadata).encode()
        z, y, x = (int(value) for value in url.rsplit("/", 3)[-3:])
        return source[z * 4 : (z + 1) * 4, y * 4 : (y + 1) * 4, x * 4 : (x + 1) * 4].tobytes(order="C")

    monkeypatch.setattr(MODULE, "fetch", fake_fetch)
    patch, receipt = MODULE.extract_patch(
        base_url="https://example.invalid/volume.zarr",
        array_path="0",
        center_xyz=(4, 4, 4),
        shape_zyx=(4, 4, 4),
    )
    assert np.array_equal(patch, source[2:6, 2:6, 2:6])
    assert receipt["chunks_fetched"] == 8
    assert receipt["chunks_missing_filled"] == 0


def test_extract_patch_fails_closed_on_missing_chunk(monkeypatch) -> None:
    metadata = {
        "zarr_format": 2,
        "shape": [4, 4, 4],
        "chunks": [4, 4, 4],
        "dtype": "|u1",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": "/",
        "compressor": None,
    }

    def fake_fetch(url: str, *, allow_missing: bool = False):
        if url.endswith(".zarray"):
            return json.dumps(metadata).encode()
        if allow_missing:
            return None
        raise RuntimeError("missing chunk")

    monkeypatch.setattr(MODULE, "fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="missing chunk"):
        MODULE.extract_patch(
            base_url="https://example.invalid/volume.zarr",
            array_path="0",
            center_xyz=(2, 2, 2),
            shape_zyx=(4, 4, 4),
        )

    patch, receipt = MODULE.extract_patch(
        base_url="https://example.invalid/volume.zarr",
        array_path="0",
        center_xyz=(2, 2, 2),
        shape_zyx=(4, 4, 4),
        allow_missing_chunks=True,
    )
    assert not patch.any()
    assert receipt["chunks_missing_filled"] == 1
