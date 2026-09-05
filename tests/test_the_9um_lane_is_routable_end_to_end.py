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


def test_a_queued_job_can_ask_for_a_different_batch_size(tmp_path):
    """The profile pins 1 against a 6 GB card; nothing routed through the
    queue could ask this runner for more until batch_size joined the lane's
    flags."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"surface_volume": "/vol.zarr",
                          "checkpoint": "/models/step.pth", "batch_size": 8}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert "--batch-size" in argv and "8" in argv


def test_a_queued_job_can_ask_for_a_different_worker_count(tmp_path):
    """Not useful until the input is cached locally, but the queue is the
    only path a job has to ask for anything -- exposed the same way
    batch_size was."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"surface_volume": "/vol.zarr",
                          "checkpoint": "/models/step.pth", "num_workers": 8}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert "--num-workers" in argv and "8" in argv


def test_a_queued_job_can_ask_for_a_different_layer_window(tmp_path):
    """A band-position experiment (top/center/bottom thirds of a stack) had
    to be built as three separate on-disk layer directories before this,
    because the profile pinned a window nothing could override per job."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"surface_volume": "/vol.zarr",
                          "checkpoint": "/models/step.pth",
                          "layer_start": 2, "layer_end": 9}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert "--layer-start" in argv and "2" in argv
    assert "--layer-end" in argv and "9" in argv


def test_a_queued_job_can_pick_the_autocast_dtype(tmp_path):
    """bf16 is the profile's; a card that cannot do it has only the queue to
    say so through."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"surface_volume": "/vol.zarr",
                          "checkpoint": "/models/step.pth", "amp_dtype": "fp16"}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert argv[argv.index("--amp-dtype") + 1] == "fp16"


def test_a_queued_job_can_ask_for_the_shuffled_layer_control(tmp_path):
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc1447",
           "parameters": {"surface_volume": "/vol.zarr",
                          "checkpoint": "/models/step.pth", "shuffle_seeds": 8}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert argv[argv.index("--shuffle-seeds") + 1] == "8"


def test_the_queue_still_refuses_it_without_the_source_scale(tmp_path):
    """Pooling needs to know what it is pooling from. Guessing 2.4 would pool
    a native 9 um render into a 38 um one."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0139",
           "parameters": {"tiff_dir": "/layers", "checkpoint": "/models/step.pth"}}
    with pytest.raises(JobRejected, match="source_pixel_um"):
        command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")


def test_a_queued_job_can_ask_to_resample_from_a_different_scale(tmp_path):
    """4 of the 13 eligible scrolls were scanned at 8.64 um/116 keV, which the
    pooling guard refuses -- not an integer factor of 9.362 um. Opting in
    unblocks them at a measured ~4% correlation cost, the largest single
    surface unblock this lane has had."""
    job = {"phase": "P5", "profile_id": PROFILE, "sample_id": "PHerc0268",
           "parameters": {"tiff_dir": "/layers", "checkpoint": "/models/step.pth",
                          "source_pixel_um": 9.362, "resample_from_um": 8.640}}
    argv = command_for(job, runner=ADAPTER, output_dir="/runs/p5-9um")
    assert "--resample-from-um" in argv and "8.64" in argv


def test_without_it_the_guard_still_refuses_exactly_as_before(tmp_path):
    """The parameter is opt-in: a caller who does not name it gets the same
    refusal as today, not a silent resample."""
    layers = _stack(tmp_path / "layers")
    with pytest.raises(lane.IncompatibleSourceScale):
        lane.prepared_surface_volume(layers, tmp_path / "work",
                                     source_voxel_um=8.640)


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


def test_resampling_reaches_the_recipe_a_pool_could_not(tmp_path):
    """8.64 um is not within tolerance of either scale the pool knows and is
    refused on its own (proven above); resample_from_um is the opt-in that
    reaches the model's 9.362 um scale anyway, same as the job parameter
    pairing (source_pixel_um: 9.362, resample_from_um: 8.640) does."""
    layers = _stack(tmp_path / "layers")
    volume = lane.prepared_surface_volume(layers, tmp_path / "work",
                                          source_voxel_um=9.362,
                                          resample_from_um=8.640)
    receipt = json.loads((volume / "INK_9UM_INPUT_RECEIPT.json").read_text())
    assert receipt["resample_from_um"] == 8.640
    assert receipt["isotropic"] is False, (
        "a resample is XY-only and must not claim Z isotropy")


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
