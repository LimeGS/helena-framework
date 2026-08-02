"""P5 taking the render P4 published, rather than a path on one machine.

P5's `tiff_dir` is a directory on whichever host happens to hold the layers,
which is why the phase had only ever been run by hand against a stack somebody
downloaded: the queue could express "run the detector" but not "run it on that
render". Naming the render makes the chain P4 -> P5 something the control plane
can say, and makes what a probability map was computed from a matter of record
rather than of a path in an argv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402
from job_store import JobRejected, command_for, validate_parameters  # noqa: E402

P5 = {"checkpoint": "/models/gp-scroll1/model.ckpt", "upstream_dir": "/models/gp-scroll1",
      "source_pixel_um": 9.362}


def test_a_job_may_name_the_render_instead_of_a_directory():
    clean = validate_parameters({**P5, "layer_stack": "p4-8d338e5a4a9445"}, "P5")
    assert clean["layer_stack"] == "p4-8d338e5a4a9445"
    assert "tiff_dir" not in clean


def test_naming_both_is_refused():
    with pytest.raises(JobRejected) as refused:
        validate_parameters({**P5, "tiff_dir": "/layers", "layer_stack": "p4-1"}, "P5")
    assert "exactly one" in str(refused.value)


def test_naming_neither_is_refused():
    with pytest.raises(JobRejected) as refused:
        validate_parameters(P5, "P5")
    assert "exactly one" in str(refused.value)


def test_an_unfetched_stack_never_reaches_the_detector():
    job = {"phase": "P5", "profile_id": "timesformer-gp-scroll1-screening@1.1.0",
           "sample_id": "PHerc826",
           "parameters": {**P5, "layer_stack": "p4-1"}}
    with pytest.raises(JobRejected) as refused:
        command_for(job, runner="unused", output_dir="/runs/p5-1")
    assert "did not run" in str(refused.value)


class Store:
    def __init__(self, render):
        self._render = render

    def job(self, job_id):
        return self._render


def test_a_map_is_not_computed_on_a_failed_render(tmp_path):
    store = Store({"job_id": "p4-1", "phase": "P4", "state": "failed", "result": {}})
    with pytest.raises(RuntimeError) as refused:
        ink_worker.resolve_layer_stack(
            store, {"parameters": {"layer_stack": "p4-1"}}, tmp_path / "stack")
    assert "failed" in str(refused.value)


def test_a_render_that_published_nothing_cannot_be_read_here(tmp_path):
    store = Store({"job_id": "p4-1", "phase": "P4", "state": "succeeded",
                   "result": {"output_dir": "/ssd/campaignx/runs/p4-1"}})
    with pytest.raises(RuntimeError) as refused:
        ink_worker.resolve_layer_stack(
            store, {"parameters": {"layer_stack": "p4-1"}}, tmp_path / "stack")
    assert "published no layer stack" in str(refused.value)


def test_a_fetched_stack_is_verified_against_its_manifest(tmp_path):
    """An artifact that arrives with a different digest is a different artifact,
    and a probability map computed on it means nothing."""
    import json

    import numpy
    import tifffile

    source = tmp_path / "published"
    source.mkdir()
    for index in range(3):
        tifffile.imwrite(source / f"{index:02d}.tif",
                         numpy.full((4, 4), index, dtype=numpy.uint16))
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import file_sha256

    manifest = {"schema": "campaignx.layer_stack_artifact_set.v1",
                "files": {f"{i:02d}.tif": {"sha256": file_sha256(source / f"{i:02d}.tif"),
                                           "size_bytes": (source / f"{i:02d}.tif").stat().st_size}
                          for i in range(3)}}
    (source / "ARTIFACT_SET.json").write_text(json.dumps(manifest))

    fetched = ink_worker.fetch_artifact_set(str(source), tmp_path / "fetched")
    assert len(fetched["files"]) == 3

    manifest["files"]["01.tif"]["sha256"] = "0" * 64
    (source / "ARTIFACT_SET.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError) as refused:
        ink_worker.fetch_artifact_set(str(source), tmp_path / "again")
    assert "not that artifact" in str(refused.value)


def test_the_slice_pitch_comes_from_the_render_rather_than_a_person(tmp_path):
    """The TimeSformer adapter refuses to default the depth pitch, and it is
    right to: the campaign spans 8.64 and 9.362 um acquisitions and assuming
    either rescales the other by 8.4% in silence. But when the render is named
    it is not a guess -- P4 recorded the scale it sampled at and the step it
    took along the normal."""
    job = {"parameters": {"source_pixel_um": 9.362}}
    assert ink_worker.slice_pitch_from_render(
        job, {"parameters": {"scale": 1.0, "slice_step": 1.0}}) == 9.362
    # Two voxels between slices is twice the pitch, which is the case that makes
    # "the pixel size" the wrong answer.
    assert ink_worker.slice_pitch_from_render(
        job, {"parameters": {"scale": 1.0, "slice_step": 2.0}}) == 18.724
    # Nothing to derive it from stays nothing: the adapter's refusal is the
    # right outcome, and inventing a number here would defeat it.
    assert ink_worker.slice_pitch_from_render({"parameters": {}}, {}) is None
