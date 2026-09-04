"""The 9 um lane, which must not be run by the generic runner.

The registry already carries the cost of getting this wrong once: the
canonical-2um recipe was routed through the generic runner, whose
clip(0,200)/200 normalisation and resampled depth axis are not that recipe's,
and the resulting map correlated r=0.079 with the community's published map
for the same recipe on the same segment -- against r=0.885 once the recipe's
own runner was used.

So this lane shells out to `koine_machines.inference.infer`, upstream's own
entrypoint, and this module owns only what Helena owns: the argv built from a
frozen profile, and the refusal to invent anything the profile did not say.

The second fact worth pinning here is the label-smoothing floor. These models
train with BCE label smoothing 0.5, so their most confident *no-ink* output
sits near 0.25 rather than 0. Anything that thresholds this map as if 0 meant
no-ink is reading a different quantity than the one the model emits.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

import run_ink_9um  # noqa: E402
from prepare_9um_isotropic_input import IncompatibleSourceScale  # noqa: E402
from run_ink_9um import (  # noqa: E402
    default_batch_size_for_host, detect_gpu_vram_mb, inference_command,
    load_lane_profile, preparation_command,
)

PROFILE = ROOT / "framework/profiles/03-ink/ink-9um-hybrid-3d2d-screening-1.0.0.json"


def test_the_shipped_profile_is_a_valid_lane_profile():
    profile = load_lane_profile(PROFILE)
    assert profile["profile_id"] == "ink-9um-hybrid-3d2d-screening@1.0.0"
    assert profile["method_id"] == "ink-9um-hybrid-3d2d@1.0.0"
    # The digest read from the upstream model repository, not asserted.
    assert profile["checkpoint_sha256"] == (
        "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab")


def test_the_command_is_upstreams_own_entrypoint_with_positional_arguments():
    """input_zarr, checkpoint and output_tiff are positional in upstream's
    parser; passing any of them as a flag value would not run."""
    argv = inference_command(
        load_lane_profile(PROFILE),
        surface_volume="s3://bucket/segment.zarr",
        checkpoint=Path("/models/step-075000.pth"),
        output_tiff=Path("/out/segment.tif"))
    assert argv[:3] == [sys.executable, "-m", "koine_machines.inference.infer"]
    assert argv[3] == "s3://bucket/segment.zarr"
    assert argv[4] == "/models/step-075000.pth"
    assert argv[5] == "/out/segment.tif"


def test_every_inference_setting_comes_from_the_profile():
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    execution = profile["default_execution"]
    assert argv[argv.index("--overlap") + 1] == str(execution["overlap"])
    assert argv[argv.index("--blend-mode") + 1] == execution["blend_mode"]
    assert argv[argv.index("--direction") + 1] == execution["direction"]
    assert argv[argv.index("--batch-size") + 1] == str(execution["batch_size"])


def test_a_layer_window_is_sent_only_when_the_profile_pins_one():
    """The models are sensitive to z offset and upstream defaults both bounds
    to None. A window invented here would be an unrecorded setting."""
    profile = load_lane_profile(PROFILE)
    profile["default_execution"].pop("layer_start", None)
    profile["default_execution"].pop("layer_end", None)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert "--layer-start" not in argv and "--layer-end" not in argv


def test_a_direction_upstream_does_not_offer_is_refused():
    profile = load_lane_profile(PROFILE)
    profile["default_execution"]["direction"] = "sideways"
    with pytest.raises(ValueError, match="direction"):
        inference_command(profile, surface_volume="v.zarr",
                          checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))


def test_the_profile_records_the_no_ink_floor_this_recipe_actually_emits():
    """0.25, not 0. A screen that assumes 0 is reading a different quantity,
    and the display rescale must not be applied before a quantitative read."""
    profile = load_lane_profile(PROFILE)
    assert profile["output_contract"]["no_ink_floor"] == 0.25
    comparability = " ".join(profile["output_contract"]["comparability"])
    assert "0.25" in comparability


def test_the_lane_declares_the_scale_it_was_trained_at():
    """Helena targets sit at 8.64/9.362 um. This lane is the first one
    whose training scale does not require upsampling to reach them, and the
    profile has to say so rather than leave it to be inferred."""
    profile = load_lane_profile(PROFILE)
    contract = profile["input_contract"]
    assert contract["physical_resampling_required"] is False
    assert 9.0 <= contract["training_pixel_um"] <= 9.7


def test_the_lane_does_not_ask_for_a_compiler_it_does_not_have():
    """Found on gpu-1: torch.compile's Inductor backend JIT-compiles kernels
    and needs a C compiler at *run* time --

        InductorError: RuntimeError: Failed to find C compiler.

    The image deliberately has none. build-essential lives in the build stage
    only, for the reason Containerfile.villa gives about its own toolchain: a
    compiler in a production image is attack surface that never executes.

    Upstream offers --no-compile, so the lane takes that rather than the
    image taking a compiler. It costs kernel-fusion speed and buys a runtime
    that cannot compile anything, which is the trade this platform already
    made everywhere else.
    """
    profile = load_lane_profile(PROFILE)
    assert profile["default_execution"]["compile"] is False
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert "--no-compile" in argv


def test_a_profile_that_wants_compilation_does_not_send_the_flag():
    """A deployment whose image does carry a toolchain can turn it back on,
    and absence of the flag is how upstream's own default is expressed."""
    profile = load_lane_profile(PROFILE)
    profile["default_execution"]["compile"] = True
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert "--no-compile" not in argv


def test_the_profile_records_the_depth_the_checkpoint_actually_asks_for():
    """Measured on gpu-1 by loading the checkpoint, not read off the card.

    The model card describes training where "the z window jitters over 17 of
    the 21 slices", and 21 is the number that reads like the input depth. It
    is not: the runner rebuilds the model from the checkpoint and reports

        Configured model ... roi_size=128 in_chans=17

    and selects seventeen source layers. A profile that says 21 is stating
    something about this lane that the checkpoint contradicts, and the
    checkpoint is the authority -- the same rule the registry applies to a
    digest over a filename.
    """
    profile = load_lane_profile(PROFILE)
    assert profile["input_contract"]["slices"] == 17
    provenance = " ".join(profile["input_contract"]["normalization_provenance"])
    assert "in_chans" in provenance or "17" in provenance


def test_the_lane_does_not_batch_that_runner_more_than_one_patch():
    """Measured on gpu-1 on 2026-08-23, by bisecting the flags this profile
    sends against the model card's own plain command, on the public
    PHerc0139 w043 surface volume:

        plain                     non-zero 100.00%  p50=0.259  p99=0.737
        --direction both          non-zero 100.00%  p50=0.259  p99=0.737
        --blend-mode hann         non-zero 100.00%  p50=0.259  p99=0.737
        --batch-size 4            non-zero   0.43%  p50=0.000  p99=0.000

    p50=0.259 is this recipe's documented no-ink floor, so the first three are
    producing real, correctly-calibrated maps and are bit-identical to each
    other. --batch-size 4 empties the map.

    At full resolution it emptied it completely: 11,006 patches selected,
    twelve minutes of GPU per direction, and a 6120x8120 output whose maximum
    value was zero. It exits 0 and writes a file of the right shape, which is
    exactly the failure the liveness gate exists to catch -- and did.

    The tutorial suggests --batch-size 4 and says to lower it if GPU memory is
    short. That advice is about memory; this is not a memory failure, and 4
    was carried into the profile from the tutorial rather than from a
    measurement here. Upstream's own default is 1.
    """
    profile = load_lane_profile(PROFILE)
    assert profile["default_execution"]["batch_size"] == 1
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert argv[argv.index("--batch-size") + 1] == "1"


def test_a_caller_may_ask_for_more_batch_than_the_profile_pins():
    """1 was measured against 4 emptying the map on a 6 GB card. Whether that
    is the model or the card is an open question the profile itself does not
    resolve -- a caller with more VRAM had no way to test it. Overlap, blend
    mode and direction stay the profile's regardless: nothing has measured
    what changing those costs, which batch_size now has, twice.
    """
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                             batch_size=8)
    assert argv[argv.index("--batch-size") + 1] == "8"
    assert argv[argv.index("--overlap") + 1] == str(profile["default_execution"]["overlap"])


def test_none_still_means_the_profiles_own_default():
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                             batch_size=None)
    assert argv[argv.index("--batch-size") + 1] == str(profile["default_execution"]["batch_size"])


def test_a_batch_size_under_one_is_refused():
    profile = load_lane_profile(PROFILE)
    with pytest.raises(ValueError, match="batch_size"):
        inference_command(profile, surface_volume="v.zarr",
                          checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                          batch_size=0)


def test_num_workers_is_absent_by_default():
    """No profile pins a worker count -- upstream's own default (4) has to
    apply, which only happens when this recipe sends nothing at all."""
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert "--num-workers" not in argv


def test_a_caller_may_ask_for_a_different_worker_count():
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                             num_workers=8)
    assert argv[argv.index("--num-workers") + 1] == "8"


def test_a_negative_worker_count_is_refused():
    profile = load_lane_profile(PROFILE)
    with pytest.raises(ValueError, match="num_workers"):
        inference_command(profile, surface_volume="v.zarr",
                          checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                          num_workers=-1)


# -- layer_start / layer_end as job parameters -------------------------------

def test_a_layer_window_is_absent_by_default():
    """The shipped profile pins neither edge -- upstream reads the whole
    stack, which only happens when this recipe sends neither flag."""
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert "--layer-start" not in argv and "--layer-end" not in argv


def test_a_caller_may_ask_for_a_different_window():
    """Added after a band-position experiment (top/center/bottom thirds) had
    to be built as three separate on-disk layer directories, because the
    profile pinned a window nothing could override per job."""
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                             layer_start=2, layer_end=9)
    assert argv[argv.index("--layer-start") + 1] == "2"
    assert argv[argv.index("--layer-end") + 1] == "9"


def test_a_caller_supplying_only_one_edge_is_refused():
    """A window with only one edge chosen is not a window either caller
    meant -- refused rather than silently inheriting the profile's other
    edge (here, no edge at all)."""
    profile = load_lane_profile(PROFILE)
    with pytest.raises(ValueError, match="layer_start and layer_end"):
        inference_command(profile, surface_volume="v.zarr",
                          checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                          layer_start=2)
    with pytest.raises(ValueError, match="layer_start and layer_end"):
        inference_command(profile, surface_volume="v.zarr",
                          checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                          layer_end=9)


def test_a_callers_window_overrides_a_profile_pinned_one():
    profile = load_lane_profile(PROFILE)
    profile["default_execution"]["layer_start"] = 0
    profile["default_execution"]["layer_end"] = 5
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                             layer_start=10, layer_end=15)
    assert argv[argv.index("--layer-start") + 1] == "10"
    assert argv[argv.index("--layer-end") + 1] == "15"


# -- batch_size default by host ---------------------------------------------

class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout


def test_detect_gpu_vram_mb_is_none_when_nvidia_smi_is_absent(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("no such file: nvidia-smi")
    monkeypatch.setattr(run_ink_9um.subprocess, "run", missing)
    assert detect_gpu_vram_mb() is None


def test_detect_gpu_vram_mb_is_none_on_a_nonzero_exit(monkeypatch):
    monkeypatch.setattr(run_ink_9um.subprocess, "run",
                        lambda *a, **k: _FakeCompletedProcess(returncode=1, stdout=""))
    assert detect_gpu_vram_mb() is None


def test_detect_gpu_vram_mb_takes_the_smallest_visible_card(monkeypatch):
    """A batch size has to fit every card this process can see, not just the
    biggest one -- two visible GPUs of different size means the smaller one
    decides."""
    monkeypatch.setattr(
        run_ink_9um.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="32768\n6144\n"))
    assert detect_gpu_vram_mb() == 6144


def test_default_batch_size_is_4_at_the_vram_threshold(monkeypatch):
    monkeypatch.setattr(run_ink_9um, "detect_gpu_vram_mb", lambda: 16_384)
    profile = load_lane_profile(PROFILE)
    assert default_batch_size_for_host(profile) == 4


def test_default_batch_size_falls_back_below_the_vram_threshold(monkeypatch):
    """One MiB under the line is still the profile's own pinned value --
    measured against the 6 GB card, not a guess about the ones in between."""
    monkeypatch.setattr(run_ink_9um, "detect_gpu_vram_mb", lambda: 16_383)
    profile = load_lane_profile(PROFILE)
    assert default_batch_size_for_host(profile) == profile["default_execution"]["batch_size"]


def test_default_batch_size_falls_back_when_vram_is_unreadable(monkeypatch):
    monkeypatch.setattr(run_ink_9um, "detect_gpu_vram_mb", lambda: None)
    profile = load_lane_profile(PROFILE)
    assert default_batch_size_for_host(profile) == profile["default_execution"]["batch_size"]


def test_inference_command_applies_the_hosts_default_when_none_is_given(monkeypatch):
    monkeypatch.setattr(run_ink_9um, "detect_gpu_vram_mb", lambda: 32_768)
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert argv[argv.index("--batch-size") + 1] == "4"


def test_an_explicit_batch_size_wins_without_even_probing_the_host(monkeypatch):
    """A caller who already named a batch has said what they want; probing
    the host anyway would be work with no effect and one more way this could
    fail on a host with no nvidia-smi at all."""
    probed = []
    monkeypatch.setattr(run_ink_9um, "detect_gpu_vram_mb",
                        lambda: probed.append(True) or 6_144)
    profile = load_lane_profile(PROFILE)
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"),
                             batch_size=8)
    assert argv[argv.index("--batch-size") + 1] == "8"
    assert probed == []


def test_the_upstream_runner_can_live_in_another_interpreter(monkeypatch):
    """A worker that claims from the queue needs psycopg; this lane's frozen
    lock has none, and adding one on top of `uv sync --frozen` would spend the
    property that lock exists for.

    So the two live in one image as two environments -- the same composition
    helena-gpu-runtime already is -- and the adapter has to call upstream with
    the lane venv's interpreter rather than its own. Unset, it uses its own,
    which is what a direct CLI run wants.
    """
    profile = load_lane_profile(PROFILE)
    monkeypatch.setenv("HELENA_INK_9UM_PYTHON", "/opt/villa/ink-venv/bin/python")
    argv = inference_command(profile, surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert argv[0] == "/opt/villa/ink-venv/bin/python"
    assert argv[1:3] == ["-m", "koine_machines.inference.infer"]


def test_without_that_setting_it_uses_its_own_interpreter(monkeypatch):
    monkeypatch.delenv("HELENA_INK_9UM_PYTHON", raising=False)
    argv = inference_command(load_lane_profile(PROFILE), surface_volume="v.zarr",
                             checkpoint=Path("c.pth"), output_tiff=Path("o.tif"))
    assert argv[0] == sys.executable


def test_the_pooling_step_runs_where_the_reader_lives(monkeypatch, tmp_path) -> None:
    """The pooling step writes the zarr the inference step reads.

    Both halves of this lane are one job on purpose -- one lease, one receipt,
    one record naming the checkpoint and the input it pooled. What that does not
    settle is which interpreter does the writing, and on a composed worker image
    it was the wrong one: `python3` is the half that can talk to the queue, and
    the pooling ran there, where zarr is not installed. Observed on gpu-1 as
    p5-72ec04c791f846, `ModuleNotFoundError: No module named 'zarr'`, after the
    worker had correctly claimed and started the job.

    A writer and a reader that disagree about zarr's major version would be the
    subtler version of the same bug, and there is a dispatch in the preparer for
    exactly that. One interpreter for the whole lane removes the question.
    """
    monkeypatch.setenv("HELENA_INK_9UM_PYTHON", "/opt/villa/ink-venv/bin/python")

    argv = preparation_command(
        tiff_dir=tmp_path / "layers", destination=tmp_path / "vol.zarr",
        source_voxel_um=9.362)

    assert argv[0] == "/opt/villa/ink-venv/bin/python"
    assert argv[1].endswith("prepare_9um_isotropic_input.py")
    assert "--source-voxel-um" in argv and "9.362" in argv


def test_pooling_falls_back_to_this_interpreter(monkeypatch, tmp_path) -> None:
    """Unset is a direct CLI run inside the lane image, where this interpreter
    is already the lane's."""
    monkeypatch.delenv("HELENA_INK_9UM_PYTHON", raising=False)

    argv = preparation_command(
        tiff_dir=tmp_path / "layers", destination=tmp_path / "vol.zarr",
        source_voxel_um=9.362)

    assert argv[0] == sys.executable


REFUSAL_EXIT = 3


def test_a_scale_this_recipe_cannot_reach_still_says_so(monkeypatch, tmp_path) -> None:
    """The refusal has to survive becoming a subprocess.

    Pooling used to run in this process, so IncompatibleSourceScale propagated
    and main turned it into exit 3 with its message -- refused before a GPU is
    claimed, which is the point of checking the scale at all. Across a process
    boundary an exception is an exit code and some stderr, and a parent that
    only reported "exit code 3" would have turned a sentence somebody can act on
    into a number.
    """
    import run_ink_9um

    class Refused:
        returncode = REFUSAL_EXIT
        stderr = b"2.4 um cannot be pooled to 9 um by an integer factor\n"

    monkeypatch.setattr(run_ink_9um.subprocess, "run", lambda *a, **k: Refused())

    with pytest.raises(IncompatibleSourceScale) as refusal:
        run_ink_9um.prepared_surface_volume(
            tmp_path / "layers", tmp_path, source_voxel_um=2.4)

    assert "integer factor" in str(refusal.value)


def test_pooling_that_fails_some_other_way_is_not_reported_as_a_scale_problem(
        monkeypatch, tmp_path) -> None:
    """A missing file and an unreachable scale are different answers, and
    exit 3 is the one the preparer reserves for the second."""
    import run_ink_9um

    class Broke:
        returncode = 1
        stderr = b"FileNotFoundError: /layers/00.tif\n"

    monkeypatch.setattr(run_ink_9um.subprocess, "run", lambda *a, **k: Broke())

    with pytest.raises(RuntimeError) as failure:
        run_ink_9um.prepared_surface_volume(
            tmp_path / "layers", tmp_path, source_voxel_um=9.362)

    assert not isinstance(failure.value, IncompatibleSourceScale)
    assert "00.tif" in str(failure.value)


# -- what --output means -----------------------------------------------------
#
# It meant "the TIFF upstream writes", and every other P5 lane means "the job's
# directory" -- run_ink.py mkdirs it, refuses it non-empty and writes several
# named files into it. The worker has one shape for all of them and passes the
# job directory, so this lane was handed a directory and used it as a filename.
#
# Two things came of that, both seen on gpu-1 as p5-9e70793966dc4d: twelve
# minutes of inference ended in `IsADirectoryError` at the write, and because
# the pooled volume is placed beside the output, 865 MB of surface-volume.zarr
# landed in the runs root instead of inside the job.


def test_the_output_is_a_directory_like_every_other_lane() -> None:
    """One convention across P5, rather than this lane's own."""
    import run_ink_9um

    parser_source = inspect.getsource(run_ink_9um.main)
    assert '"uint8 tiled TIFF upstream writes"' not in parser_source, (
        "--output still documents itself as a file")
    assert "args.output.parent" not in parser_source, (
        "something is still reaching outside the job directory")


def test_the_lane_names_its_own_file_inside_the_job_directory(
        monkeypatch, tmp_path, capsys) -> None:
    import run_ink_9um

    seen: dict = {}

    def fake_resolve(*, tiff_dir, surface_volume, work_dir, source_voxel_um):
        seen["work_dir"] = Path(work_dir)
        return "/somewhere/surface-volume.zarr"

    def fake_inference_command(profile, *, surface_volume, checkpoint, output_tiff,
                               batch_size=None, num_workers=None,
                               layer_start=None, layer_end=None):
        seen["output_tiff"] = Path(output_tiff)
        return ["true"]

    monkeypatch.setattr(run_ink_9um, "resolve_surface_volume", fake_resolve)
    monkeypatch.setattr(run_ink_9um, "inference_command", fake_inference_command)
    monkeypatch.setattr(run_ink_9um, "load_lane_profile", lambda path: {})

    job = tmp_path / "pherc0332-p5-abc123"
    monkeypatch.setattr(sys, "argv", [
        "run_ink_9um.py", "--profile", "p.json", "--checkpoint", "c.pth",
        "--output", str(job), "--tiff-dir", str(tmp_path / "layers"),
        "--source-pixel-um", "9.362", "--print-command",
    ])
    run_ink_9um.main()

    assert seen["output_tiff"].parent == job, (
        f"the map is written to {seen['output_tiff']}, outside the job directory")
    assert seen["work_dir"] == job, (
        f"the pooled volume goes to {seen['work_dir']}, not into the job")
    assert job.is_dir(), "the lane did not create the directory it was given"
