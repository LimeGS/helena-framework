"""The 9 um lane, queued like every other, with the pooling inside it.

Two commits ago this lane existed and nothing could run it: its runner streams
an OME-Zarr surface volume at ~9 um isotropic and every P4 output here is a
numbered uint8 TIFF stack at 2.399 um. One commit ago the conversion existed.
What was still missing was the shape of the job.

The chain lives in the adapter rather than in the queue. A second job type
would mean a second state machine -- a prepare job whose output a later infer
job has to find, with its own lease, its own retry and its own way of being
half-done -- to express one lane that happens to need two steps. The adapter
already owns the Helena half of this lane, and pooling is part of that half:
the receipt that names the checkpoint should name the input it pooled too, and
that is only one receipt if it is one process.

So P5 hands this lane a `tiff_dir` exactly as it hands one to every other
lane, and the adapter pools it into a surface volume before calling upstream's
runner on it. A lane given a native ~9 um zarr skips the pooling, because
there is nothing to do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

import run_ink_9um as lane  # noqa: E402
from job_store import INK_ADAPTERS, JobRejected, command_for  # noqa: E402

PROFILE = "ink-9um-hybrid-3d2d-screening@1.0.0"
ADAPTER = "framework/stages/03-ink/scripts/run_ink_9um.py"


def _stack(directory: Path, *, slices: int = 8, size: int = 16) -> Path:
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(slices):
        tifffile.imwrite(directory / f"{index:02d}.tif",
                         rng.integers(0, 255, size=(size, size), dtype=np.uint8))
    return directory


# -- the queue -------------------------------------------------------------

def test_the_lane_is_no_longer_unroutable():
    assert "unroutable" not in INK_ADAPTERS[ADAPTER], (
        "the queue still refuses this lane by declaration")


def test_the_queue_builds_a_command_for_it(tmp_path):
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"tiff_dir": "/layers", "checkpoint": "/models/step.pth",
                          "source_pixel_um": 2.399}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert "--tiff-dir" in argv and "/layers" in argv
    assert "--checkpoint" in argv and "/models/step.pth" in argv
    assert "--source-pixel-um" in argv and "2.399" in argv
    assert "--profile" in argv


def test_the_queue_still_refuses_it_without_the_source_scale(tmp_path):
    """Pooling needs to know what it is pooling from. Guessing 2.4 would pool
    a native 9 um render into a 38 um one."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"tiff_dir": "/layers", "checkpoint": "/models/step.pth"}}
    with pytest.raises(JobRejected, match="source_pixel_um"):
        command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")


# -- the chain -------------------------------------------------------------

def test_a_tiff_dir_is_pooled_before_inference(tmp_path):
    layers = _stack(tmp_path / "layers")
    argv = lane.inference_command(
        lane.load_lane_profile(
            ROOT / "framework/profiles/03-ink/ink-9um-hybrid-3d2d-screening-1.0.0.json"),
        surface_volume=str(lane.prepared_surface_volume(
            layers, tmp_path / "work", source_voxel_um=2.399)),
        checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    volume = argv[3]
    assert volume.endswith(".zarr"), f"inference was pointed at {volume}"
    assert Path(volume).is_dir()


def test_the_pooled_input_carries_its_own_receipt(tmp_path):
    layers = _stack(tmp_path / "layers")
    volume = lane.prepared_surface_volume(layers, tmp_path / "work",
                                          source_voxel_um=2.399)
    receipt = json.loads((volume / "INK_9UM_INPUT_RECEIPT.json").read_text())
    assert receipt["xy_factor"] == 4 and receipt["z_factor"] == 4
    assert len(receipt["source_sha256"]) == 64


def test_a_native_scale_stack_is_not_pooled(tmp_path):
    layers = _stack(tmp_path / "layers")
    volume = lane.prepared_surface_volume(layers, tmp_path / "work",
                                          source_voxel_um=9.362)
    receipt = json.loads((volume / "INK_9UM_INPUT_RECEIPT.json").read_text())
    assert receipt["xy_factor"] == 1 and receipt["z_factor"] == 1


def test_a_scale_the_recipe_cannot_reach_refuses_before_any_gpu(tmp_path):
    layers = _stack(tmp_path / "layers")
    with pytest.raises(lane.IncompatibleSourceScale):
        lane.prepared_surface_volume(layers, tmp_path / "work",
                                     source_voxel_um=7.9)


# -- what the caller must say ---------------------------------------------

def test_naming_both_a_stack_and_a_volume_is_refused(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        lane.resolve_surface_volume(
            tiff_dir=tmp_path / "layers", surface_volume="s3://b/v.zarr",
            work_dir=tmp_path / "work", source_voxel_um=2.399)


def test_naming_neither_is_refused(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        lane.resolve_surface_volume(
            tiff_dir=None, surface_volume=None,
            work_dir=tmp_path / "work", source_voxel_um=None)


def test_a_ready_made_volume_is_passed_straight_through(tmp_path):
    """A native zarr needs no pooling and no working directory."""
    resolved = lane.resolve_surface_volume(
        tiff_dir=None, surface_volume="s3://bucket/segment.zarr",
        work_dir=tmp_path / "work", source_voxel_um=None)
    assert resolved == "s3://bucket/segment.zarr"
    assert not (tmp_path / "work").exists()
