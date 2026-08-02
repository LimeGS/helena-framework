"""The layer stack, which was the one artifact the pipeline left on a worker.

P1 publishes its surface, P3 publishes its sheet, and both record a URI and a
digest. P4 wrote 33 TIFFs into the runs directory of whichever host claimed the
job and recorded `output_dir` -- a path that means nothing on any other machine
and disappears with that one. The worker hosts are disposable by design, so the
product of the whole P0-P4 chain was the only thing in it that was not.

The second half is the check P3 already had and P4 did not: a renderer whose
surface fell outside the cached region writes the requested number of slices and
every one of them is a constant. Exit 0, 33 files, nothing in them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

import ink_worker  # noqa: E402
from ink_worker import RenderNotUsable, verify_layer_stack  # noqa: E402


def stack(directory: Path, slices: int = 5, *, constant: bool = False) -> Path:
    import numpy
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(slices):
        plane = (numpy.zeros((4, 4), dtype=numpy.uint16) if constant
                 else numpy.arange(16, dtype=numpy.uint16).reshape(4, 4) + index)
        tifffile.imwrite(directory / f"{index:02d}.tif", plane)
    return directory


def test_a_stack_of_constants_is_not_a_render(tmp_path):
    with pytest.raises(RenderNotUsable) as refused:
        verify_layer_stack(stack(tmp_path / "layers", constant=True), {})
    assert "no signal" in str(refused.value)


def test_the_slice_count_has_to_be_the_one_that_was_asked_for(tmp_path):
    """N slices at the wrong spacing is P4's documented way of failing, and a
    stack with the wrong number of them is the visible half of that."""
    with pytest.raises(RenderNotUsable) as refused:
        verify_layer_stack(stack(tmp_path / "layers", 5), {"num_slices": 33})
    assert "33" in str(refused.value)


def test_an_empty_directory_is_refused(tmp_path):
    (tmp_path / "layers").mkdir()
    with pytest.raises(RenderNotUsable):
        verify_layer_stack(tmp_path / "layers", {})


def test_a_real_stack_is_described_rather_than_just_accepted(tmp_path):
    described = verify_layer_stack(stack(tmp_path / "layers", 33), {"num_slices": 33})
    assert described["slices"] == 33
    assert described["shape"] == [4, 4]
    assert described["middle_slice_range"][0] < described["middle_slice_range"][1]
    assert described["bytes"] > 0


def test_scroll3_verification_follows_its_explicit_output_directory():
    """The legacy lane writes ``OUT/layers`` when an out_dir is supplied."""
    from ink_worker import rendered_layers_directory

    chosen = rendered_layers_directory(
        {"parameters": {
            "lane": "scroll3-chunk-gather",
            "out_dir": "/durable/p4-job",
        }},
        Path("/runs/p4-job"),
    )
    assert chosen == Path("/durable/p4-job/layers")


def test_publishing_records_a_uri_and_a_digest(tmp_path):
    """A local store here, an s3:// one in the deployment: the same call, and
    what matters is that the job row ends up with somewhere to look."""
    published = ink_worker.publish_layer_stack(
        stack(tmp_path / "layers", 3), store_spec=str(tmp_path / "store"),
        sample_id="PHerc826", job_id="p4-test")
    assert published["files"] == 3
    assert len(published["artifact_sha256"]) == 64
    assert (Path(published["artifact_uri"]) / "01.tif").is_file()
    assert (Path(published["artifact_uri"]) / "ARTIFACT_SET.json").is_file()


def test_a_p4_job_may_carry_where_it_publishes():
    """The panel fills this in from one deployment setting, and the queue has to
    accept it: it did not, and every render was refused with "unknown parameters
    for P4" one step after the fix that added it."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import validate_parameters

    clean = validate_parameters(
        {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr", "scale": 1.0,
         "group_idx": 0, "segmentation": "/surfaces/s-1",
         "artifact_store": "s3://bucket/layer-stacks-v1"}, "P4")
    assert clean["artifact_store"] == "s3://bucket/layer-stacks-v1"


def test_the_local_copy_goes_once_the_bytes_are_published(tmp_path, monkeypatch):
    """A stack is ~50 MB and a worker renders many. Without cleanup the volume
    fills, and on a host that also carries the control-plane database that means
    rendered output nobody needed locally takes the platform down."""
    layers = stack(tmp_path / "layers", 3)
    published = ink_worker.publish_layer_stack(
        layers, store_spec=str(tmp_path / "store"),
        sample_id="PHerc826", job_id="p4-cleanup")
    # What the worker does next, spelled out here because the run loop around it
    # needs a queue and a subprocess.
    import shutil

    shutil.rmtree(layers, ignore_errors=True)
    assert not layers.exists()
    assert (Path(published["artifact_uri"]) / "01.tif").is_file()


def test_the_normal_direction_is_a_choice_the_job_records():
    """Our layer 0 is the community's layer 85 and our 62 is their 23, each at
    r = 0.99 in descending order: the renderer's default traverses the normal
    the other way from theirs on this mesh. A depth-reversed slab is a correct
    render of the far side first, and it is not what an ink model was trained
    on -- with everything else identical, including their checkpoint, the map
    correlated 0.09 with theirs."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    base = {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr", "scale": 1.0,
            "group_idx": 0, "segmentation": "/surfaces/s-1"}
    plain = command_for({"phase": "P4", "parameters": validate_parameters(base, "P4")},
                        runner="unused", output_dir="/runs/job-9")
    assert "--flip-normals" not in plain
    flipped = command_for(
        {"phase": "P4",
         "parameters": validate_parameters({**base, "flip_normals": True}, "P4")},
        runner="unused", output_dir="/runs/job-9")
    assert "--flip-normals" in flipped


def test_scroll3_render_defaults_output_to_the_job_directory():
    """The lane declares only its PPM as required, so its output has a default.

    A minimal valid request used to pass queue validation and then fail in the
    worker's command builder with ``KeyError: out_dir`` before the renderer ran.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    parameters = validate_parameters(
        {"lane": "scroll3-chunk-gather", "ppm": "/surfaces/s-1.ppm"}, "P4")
    argv = command_for(
        {"phase": "P4", "sample_id": "PHerc0332", "parameters": parameters},
        runner="/workspace/render_scroll3.py", output_dir="/runs/p4-job")

    assert argv[argv.index("--out") + 1] == "/runs/p4-job"


def test_scroll3_render_refuses_a_different_scroll_before_building_argv():
    """The legacy renderer's CT volume and alignment are fixed to Scroll 3.

    Recording another sample on the job while reading PHerc0332 would create a
    plausible layer stack with false lineage, which is worse than a crash.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import JobRejected, command_for, validate_parameters

    parameters = validate_parameters(
        {"lane": "scroll3-chunk-gather", "ppm": "/surfaces/s-1.ppm"}, "P4")

    with pytest.raises(JobRejected, match="PHerc0332"):
        command_for(
            {"phase": "P4", "sample_id": "PHerc826", "parameters": parameters},
            runner="/workspace/render_scroll3.py", output_dir="/runs/p4-job")


def test_scroll3_render_refuses_a_different_scroll_before_queue_insertion():
    """A known-bad lineage contract must not consume a fleet attempt first."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import InkJobStore, JobRejected

    store = InkJobStore("postgresql://unused")
    store._connect = lambda: pytest.fail("an invalid job reached PostgreSQL")  # noqa: SLF001

    with pytest.raises(JobRejected, match="PHerc0332"):
        store.enqueue(
            sample_id="PHerc826",
            phase="P4",
            parameters={
                "lane": "scroll3-chunk-gather",
                "ppm": "/surfaces/s-1.ppm",
            },
        )
