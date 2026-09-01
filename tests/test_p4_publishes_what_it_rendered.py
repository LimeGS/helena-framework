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
sys.path.insert(0, str(ROOT / "scripts/harness"))
from run_first_letters_positive_control import _exact_numeric_tiff_inventory  # noqa: E402


def stack(directory: Path, slices: int = 5, *, constant: bool = False,
          padded: bool = True) -> Path:
    import numpy
    import tifffile

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(slices):
        plane = (numpy.zeros((4, 4), dtype=numpy.uint16) if constant
                 else numpy.arange(16, dtype=numpy.uint16).reshape(4, 4) + index)
        name = f"{index:02d}.tif" if padded else f"{index}.tif"
        tifffile.imwrite(directory / name, plane)
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
    described = verify_layer_stack(
        stack(tmp_path / "layers", 33, padded=False), {"num_slices": 33})
    assert described["slices"] == 33
    assert described["slice_indices"] == list(range(33))
    assert described["slice_filenames"] == [f"{index}.tif" for index in range(33)]
    assert described["shape"] == [4, 4]
    assert described["middle_slice_range"][0] < described["middle_slice_range"][1]
    assert described["bytes"] > 0


@pytest.mark.parametrize("names", [
    [f"{index}.tif" for index in range(32)],
    [f"{index}.tif" for index in range(1, 34)],
    [*(f"{index}.tif" for index in range(33) if index != 17), "33.tif"],
    [*(f"{index}.tif" for index in range(33)), "slice.tif"],
    [*(f"{index}.tif" for index in range(33)), "00.tif"],
])
def test_control_stack_requires_exact_distinct_numeric_indices_0_through_32(
        tmp_path, names):
    import numpy
    import tifffile

    directory = tmp_path / "layers"
    directory.mkdir()
    plane = numpy.arange(16, dtype=numpy.uint16).reshape(4, 4)
    for name in names:
        tifffile.imwrite(directory / name, plane)
    with pytest.raises(RenderNotUsable):
        verify_layer_stack(directory, {"num_slices": 33})


def test_publishing_preserves_actual_numeric_names_in_numeric_order(tmp_path):
    published = ink_worker.publish_layer_stack(
        stack(tmp_path / "layers", 33, padded=False),
        store_spec=str(tmp_path / "store"), sample_id="PHerc0139", job_id="p4-control")
    assert [row["object_key"] for row in published["objects"]] == [
        f"{index}.tif" for index in range(33)]
    assert published["slice_indices"] == list(range(33))
    import json
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import content_sha256

    stored = json.loads(
        (Path(published["artifact_uri"]) / "ARTIFACT_SET.json").read_text())
    assert stored["artifact_sha256"] == content_sha256(stored["files"])
    assert published["artifact_sha256"] == stored["artifact_sha256"]


def test_control_runner_rechecks_exact_stored_layer_inventory():
    objects = [
        {"object_key": f"{index}.tif", "sha256": "8" * 64, "bytes": 42}
        for index in range(33)]
    manifest = {
        "files": 33, "objects": objects, "slice_indices": list(range(33)),
        "slice_ordering": "NUMERIC_STEM_CONTIGUOUS_ASCENDING",
    }
    assert _exact_numeric_tiff_inventory(manifest, 33) is True
    assert _exact_numeric_tiff_inventory({
        **manifest, "files": 32, "objects": objects[:-1],
        "slice_indices": list(range(32)),
    }, 33) is False
    assert _exact_numeric_tiff_inventory({
        **manifest,
        "objects": [*objects[:-1], {**objects[-1], "object_key": "33.tif"}],
    }, 33) is False


def test_scroll3_verification_follows_its_explicit_output_directory():
    """The legacy lane writes ``OUT/layers`` when an out_dir is supplied."""
    from ink_worker import rendered_layers_directory

    chosen = rendered_layers_directory(
        {"parameters": {
            "lane": "chunk-gather",
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
    from job_store import JobRejected, validate_parameters

    publishes_to = {"artifact_store": "s3://bucket/layer-stacks-v1"}
    clean = validate_parameters(
        {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr", "scale": 1.0,
         "group_idx": 0, "segmentation": "/surfaces/s-1", **publishes_to}, "P4",
        server_owned=publishes_to)
    assert clean["artifact_store"] == "s3://bucket/layer-stacks-v1"

    # And refused when it is the request that says so.
    with pytest.raises(JobRejected, match="artifact_store"):
        validate_parameters(
            {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr",
             "scale": 1.0, "group_idx": 0, "segmentation": "/surfaces/s-1",
             "artifact_store": "s3://attacker/exfil"}, "P4")


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


def test_the_ppm_lane_defaults_output_to_the_job_directory():
    """The lane requires a PPM and a volume, so its output has a default.

    A minimal valid request used to pass queue validation and then fail in the
    worker's command builder with ``KeyError: out_dir`` before the renderer ran.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    parameters = validate_parameters(
        {"lane": "chunk-gather", "ppm": "/surfaces/s-1.ppm",
         "volume_key": "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"},
        "P4")
    argv = command_for(
        {"phase": "P4", "sample_id": "PHerc0332", "parameters": parameters},
        runner="/workspace/render_scroll.py", output_dir="/runs/p4-job")

    assert argv[argv.index("--out") + 1] == "/runs/p4-job"


def test_the_ppm_lane_refuses_a_job_that_does_not_say_which_volume():
    """This lane used to be Scroll 3 only, and refused any other sample.

    That was the right guard for a renderer that hardcoded PHerc0332's volume
    key, array shape and voxel size: recording another sample while reading
    PHerc0332 would have produced a plausible layer stack with false lineage,
    which is worse than a crash. render_scroll.py takes the volume as an
    argument, so the sample is no longer the thing that identifies the target.

    The guard moved rather than disappearing: `volume` is required and has no
    default, so a job that does not say which rescan it renders against is
    refused here instead of rendering against whichever one was compiled in.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import JobRejected, validate_parameters

    with pytest.raises(JobRejected, match="volume_key"):
        validate_parameters({"lane": "chunk-gather", "ppm": "/surfaces/s-1.ppm"}, "P4")


def test_the_ppm_lane_renders_a_scroll_that_is_not_scroll_three():
    """The point of the generalisation, stated as a job that used to be refused.

    PHerc826 with its own rescan named: accepted, and the volume reaches the
    renderer's argv. Before, this raised because the sample was not PHerc0332.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    volume = "PHercParis3/volumes/20260427095331-2.400um-0.2m-78keV-masked.zarr"
    parameters = validate_parameters(
        {"lane": "chunk-gather", "ppm": "/surfaces/s-1.ppm", "volume_key": volume}, "P4")
    argv = command_for(
        {"phase": "P4", "sample_id": "PHerc826", "parameters": parameters},
        runner="/workspace/render_scroll.py", output_dir="/runs/p4-job")

    assert "--volume" in argv, "the renderer is not told which rescan to read"
    assert argv[argv.index("--volume") + 1] == volume
    assert "--ppm" in argv


def test_the_renderer_is_told_the_voxel_size_it_would_otherwise_guess():
    """It reports "Voxel size: 1.0 (no metadata found; override with
    --voxel-size)" and carries on.

    `source_voxel_um` was already a P4 parameter, already filled by the
    deployment, and already used on this side to derive depth spacing and the
    P3/P4 lateral metric. It never reached the renderer, so on a volume with no
    metadata the two halves of one job used different numbers -- 1.0 and the
    real one -- and nothing compared them.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    from job_store import command_for, validate_parameters

    server = {"artifact_store": "/artifacts/layer-stacks-v1"}
    parameters = validate_parameters(
        {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr",
         "segmentation": "/surfaces/s-1", "scale": 1.0, "group_idx": 0,
         "source_voxel_um": 9.362, **server}, "P4", server_owned=server)
    argv = [str(token) for token in command_for(
        {"phase": "P4", "sample_id": "PHerc0826", "parameters": parameters},
        runner="ignored", output_dir="/runs/p4-1")]

    assert "--voxel-size" in argv
    assert argv[argv.index("--voxel-size") + 1] == "9.362"
    assert argv[argv.index("--voxel-unit") + 1] == "micrometer"

    # Absent, it stays absent: the renderer's own default is then a choice
    # nobody made here, and the receipt says so by not carrying the flag.
    without = validate_parameters(
        {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr",
         "segmentation": "/surfaces/s-1", "scale": 1.0, "group_idx": 0,
         **server}, "P4", server_owned=server)
    assert "--voxel-size" not in [str(t) for t in command_for(
        {"phase": "P4", "sample_id": "PHerc0826", "parameters": without},
        runner="ignored", output_dir="/runs/p4-1")]
