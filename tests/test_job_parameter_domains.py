"""Queue parameters preserve scientific choices instead of coercing nonsense.

The generic job API receives an untyped ``parameters`` object.  These tests
exercise the real queue validator with values JSON clients can send but no
runner can interpret honestly.  A rejection here saves a worker attempt and,
more importantly, prevents the receipt from recording one value while the argv
silently uses another.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import JobRejected, validate_parameters  # noqa: E402


P4 = {
    "lane": "vc-render-tifxyz",
    "segmentation": "/surfaces/s-1",
    "volume": "/volumes/scroll.zarr",
    "scale": 1.0,
    "group_idx": 0,
}
P4_SCROLL3 = {"lane": "scroll3-chunk-gather", "ppm": "/surfaces/s-1.ppm"}
P5 = {
    "tiff_dir": "/layers",
    "checkpoint": "/models/checkpoint.pt",
    "source_pixel_um": 9.362,
}
P7 = {"screening_of": "p5-source", "bbox": "0,0,64,64", "px_um": 2.399}
P8_COLUMN = {
    "lane": "column-atlas",
    "scroll": "PHerc0139",
    "out_path": "/runs/wrap-radial.json",
}


@pytest.mark.parametrize(
    ("phase", "base", "key", "value"),
    [
        ("P2", {}, "limit", 0),
        ("P3", {"artifact_store": "/artifacts/flattened-v1"}, "limit", 0),
        ("P4", P4, "scale", 0.0),
        ("P4", P4, "group_idx", -1),
        ("P4", P4, "cache_gb", 0),
        ("P4", P4, "num_slices", 0),
        ("P4", P4, "slice_step", 0.0),
        ("P4", P4_SCROLL3, "layers", 0),
        ("P4", P4_SCROLL3, "spacing", 0.0),
        ("P4", P4_SCROLL3, "concurrency", 0),
        ("P4", P4_SCROLL3, "stripe", 0),
        ("P4", P4_SCROLL3, "max_gb", 0.0),
        ("P5", P5, "source_slice_um", 0.0),
        ("P5", P5, "depth_center", -1),
        ("P5", P5, "stride", 0),
        ("P5", P5, "batch_size", 0),
        ("P5", P5, "tile_size", 0),
        ("P5", P5, "min_valid_ratio", 0.0),
        ("P5", P5, "min_valid_ratio", 1.01),
        ("P7", P7, "px_um", 0.0),
        ("P8", P8_COLUMN, "subsample", 0),
    ],
)
def test_nonphysical_parameter_domains_are_refused(phase, base, key, value):
    """Zero/negative sizes and out-of-range ratios never reach a runner."""
    with pytest.raises(JobRejected):
        validate_parameters({**base, key: value}, phase)


@pytest.mark.parametrize(
    ("phase", "base", "key"),
    [
        ("P4", P4, "scale"),
        ("P4", P4, "slice_step"),
        ("P4", P4_SCROLL3, "spacing"),
        ("P4", P4_SCROLL3, "max_gb"),
        ("P5", P5, "source_pixel_um"),
        ("P5", P5, "source_slice_um"),
        ("P5", P5, "min_valid_ratio"),
        ("P7", P7, "px_um"),
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_scientific_parameters_are_refused(phase, base, key, value):
    """NaN and infinity must not be serialized into an argv or PostgreSQL JSON."""
    with pytest.raises(JobRejected):
        validate_parameters({**base, key: value}, phase)


@pytest.mark.parametrize("value", [1.5, True])
def test_integer_parameters_are_not_silently_truncated(value):
    """A requested 1.5 slices is invalid, not a request for one slice."""
    with pytest.raises(JobRejected):
        validate_parameters({**P4, "num_slices": value}, "P4")


def test_boolean_text_cannot_reverse_the_normal_direction():
    """Python's bool('false') is true, the opposite of what the client wrote."""
    with pytest.raises(JobRejected):
        validate_parameters({**P4, "flip_normals": "false"}, "P4")


def test_a_string_cannot_be_coerced_into_an_artifact_id_list():
    """list('ab') used to become two merge parents named ``a`` and ``b``."""
    with pytest.raises(JobRejected):
        validate_parameters(
            {
                "lane": "vc3d-tifxyz-merge",
                "artifact_ids": "ab",
                "rows": [["a", "b"]],
                "reference_artifact_id": "a",
                "ransac_seed": 1729,
                "anchor_cap": 2000,
                "strip_cols": 0,
                "artifact_store": "/artifacts/reconstruction-v1",
            },
            "P8",
        )


def test_structured_json_cannot_be_coerced_into_a_path():
    """An array is not a volume path just because ``str`` can print it."""
    with pytest.raises(JobRejected):
        validate_parameters({**P4, "volume": ["/volumes/scroll.zarr"]}, "P4")


def test_valid_boundary_values_keep_their_meaning():
    """The checks preserve intentional zeroes and integral JSON numbers."""
    render = validate_parameters(
        {**P4, "group_idx": 0, "cache_gb": 4.0, "flip_normals": False}, "P4")
    assert render["group_idx"] == 0
    assert render["cache_gb"] == 4
    assert render["flip_normals"] is False

    screen = validate_parameters({**P5, "min_valid_ratio": 1.0}, "P5")
    assert screen["min_valid_ratio"] == 1.0

    merge = validate_parameters(
        {
            "lane": "vc3d-tifxyz-merge",
            "artifact_ids": ["a", "b"],
            "rows": [["a", "b"]],
            "reference_artifact_id": "a",
            "ransac_seed": 1729,
            "anchor_cap": 0,
            "strip_cols": 0,
            "artifact_store": "/artifacts/reconstruction-v1",
        },
        "P8",
    )
    assert merge["anchor_cap"] == 0 and merge["strip_cols"] == 0
