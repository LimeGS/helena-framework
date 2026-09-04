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

from job_store import JobRejected, command_for, validate_parameters  # noqa: E402


P4 = {
    "lane": "vc-render-tifxyz",
    "segmentation": "/surfaces/s-1",
    "volume": "/volumes/scroll.zarr",
    "scale": 1.0,
    "group_idx": 0,
}
P4_SCROLL3 = {"lane": "chunk-gather", "ppm": "/surfaces/s-1.ppm"}
P5 = {
    "tiff_dir": "/layers",
    "checkpoint": "/models/checkpoint.pt",
    "source_pixel_um": 9.362,
}
P7 = {"screening_of": "p5-source",
      "probability_map_artifact_sha256": "a" * 64,
      "probability_map_manifest_sha256": "b" * 64,
      "bbox": "0,0,64,64", "px_um": 2.399}
P8_COLUMN = {
    "lane": "column-atlas",
    "scroll": "PHerc0139",
    "out_path": "/runs/wrap-radial.json",
}


@pytest.mark.parametrize(
    ("phase", "base", "key", "value"),
    [
        ("P2", {}, "limit", 0),
        ("P3", {}, "limit", 0),
        ("P4", P4, "scale", 0.0),
        ("P4", P4, "group_idx", -1),
        ("P4", P4, "cache_gb", 0),
        ("P4", P4, "num_slices", 0),
        ("P4", P4, "slice_step", 0.0),
        ("P4", P4, "source_voxel_um", 0.0),
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
        ("P4", P4, "source_voxel_um"),
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
            server_owned=("artifact_store",),
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


@pytest.mark.parametrize("phase", ["P2", "P3"])
def test_worker_refuses_sample_parameter_that_disagrees_with_queue_identity(phase):
    parameters = {"sample": "PHerc9999"}
    with pytest.raises(JobRejected, match="sample"):
        command_for(
            {
                "job_id": f"{phase.lower()}-bound-sample",
                "phase": phase,
                "sample_id": "PHerc0139",
                "parameters": parameters,
            },
            runner="/fleet/surface_worker.py",
            output_dir="/runs/job",
        )

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
        server_owned=("artifact_store",),
    )
    assert merge["anchor_cap"] == 0 and merge["strip_cols"] == 0


def test_a_size_nothing_rejects_is_a_lease_taken_before_it_is_found_out():
    """Lower bounds alone are half a domain.

    Python integers do not stop being valid at any particular size, so
    `num_slices: 10**12` was a job the queue accepted, a worker leased, and a
    GPU host discovered.
    """
    from job_store import JobRejected, validate_parameters

    with pytest.raises(JobRejected, match="bounded above"):
        validate_parameters({"lane": "vc-render-tifxyz", "volume": "/v",
                             "segmentation": "/s", "scale": 1.0, "group_idx": 0,
                             "num_slices": 10 ** 12}, "P4")


def test_the_two_p1_numbers_that_were_in_neither_set():
    """`lasagna_scale` and the optimizer's seed had no sign check at all."""
    from job_store import JobRejected, validate_parameters

    fit = {"scroll_name": "PHerc0172", "dataset_path": "/artifacts/spiral",
           "z_begin": 500, "z_end": 9000, "voxel_size_um": 7.91,
           "artifact_store": "/artifacts"}
    for bad in ({"lasagna_scale": -1}, {"random_seed": -5}):
        with pytest.raises(JobRejected, match="cannot be negative"):
            validate_parameters({**fit, **bad}, "P1",
                                server_owned=("artifact_store",))


def test_an_explicit_zero_seed_reaches_the_command():
    """Dropped by truthiness, the run used upstream's default while the receipt
    recorded the zero -- two halves that both look like a job that ran as
    asked, disagreeing where nothing downstream can see it."""
    from job_store import command_for, validate_parameters

    parameters = validate_parameters(
        {"scroll_name": "PHerc0172", "dataset_path": "/artifacts/spiral",
         "z_begin": 500, "z_end": 9000, "voxel_size_um": 7.91,
         "artifact_store": "/artifacts", "random_seed": 0}, "P1",
        server_owned=("artifact_store",))
    argv = [str(token) for token in command_for(
        {"job_id": "p1-1", "sample_id": "PHerc0172", "phase": "P1",
         "profile_id": "spiral-fitter-v1@0.4.0", "parameters": parameters},
        runner="ignored", output_dir="/runs/p1-1")]
    assert "--random-seed" in argv
    assert argv[argv.index("--random-seed") + 1] == "0"
