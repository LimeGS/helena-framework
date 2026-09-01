#!/usr/bin/env python3
"""The 9 um hybrid 3D-stem / 2D-U-Net ink lane, run by its own recipe.

Why this shells out instead of reimplementing the loop
------------------------------------------------------
The registry records what happened the last time a recipe was run by the
generic runner: ``ink-canonical-2um-screening@1.0.0`` was routed through
``run_ink.py``, whose clip(0,200)/200 normalisation and resampled depth axis
belong to a different recipe. On ground where the render matched the
community's at r=0.98, the resulting map correlated r=0.079 with their
published map -- against r=0.885 once the recipe's own runner was used.

This recipe's own runner is ``koine_machines.inference.infer``. It streams the
surface volume from Zarr, rebuilds the model *and its normalisation* from the
checkpoint, and does its own sliding-window blending. Reimplementing that here
would be re-deriving, from a model card, facts the checkpoint already carries.

What this module owns is the part that is Helena's: an argv built entirely
from a frozen profile, no setting invented, and a refusal to pass a direction
or a layer window upstream does not offer.

Scale
-----
This is the first ink lane whose training scale matches the campaign's own
targets. The 3D-DINO lane's registry note records that Helena targets at
8.64/9.362 um need 3.6x/3.90x linear upsampling to reach that model's scale,
"which creates no new spatial information". These models were trained at
~9.6 um isotropic, including on native 9.362 um segments, so a native render
runs without resampling at all.

Non-claims
----------
* A probability map is a routing signal, not proof of ink or letters.
* Nothing here calibrates the lane on a target cohort. The lane is registered
  as screening-only and target-uncalibrated until a campaign establishes
  otherwise with its own control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent))
from framework.contracts.lane_liveness import (  # noqa: E402
    assess_liveness, refuse_if_not_alive,
)

from prepare_9um_isotropic_input import (  # noqa: E402
    REFUSED_SCALE_EXIT, IncompatibleSourceScale,
)

LANE_PROFILE_SCHEMA = "campaignx.ink_lane_profile.v1"
RECEIPT_NAME = "INK_9UM_RECEIPT.json"
PREPARED_VOLUME_NAME = "surface-volume.zarr"
# What the map is called inside the job directory. With `direction: both`
# upstream writes a second one beside it, for the reverse surface normal.
OUTPUT_MAP_NAME = "ink.tif"
REVERSE_MAP_NAME = "ink_reverse.tif"

# What upstream writes: a uint8 tiled TIFF. The probability this lane publishes
# is that byte scale mapped back to [0, 1] -- and nothing else. The model card's
# display rescale (p - 0.25) / 0.5 is deliberately not applied: it is for
# viewing, and baking it in would publish a rescaled quantity under the name of
# the raw one.
UINT8_FULL_SCALE = 255.0

# Upstream's own choices, from ink-detection/koine_machines/inference/infer.py.
# Restated here so an unsupported value is refused before a GPU is claimed
# rather than after argparse rejects it at the far end of a queued job.
DIRECTIONS = ("forward", "reverse", "both")
BLEND_MODES = ("auto", "constant", "gaussian", "hann")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lane_profile(path: Path) -> dict[str, Any]:
    """A frozen declaration of how this lane runs, with its own hash."""
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema") != LANE_PROFILE_SCHEMA:
        raise ValueError(f"not an ink lane profile: {path}")
    for key in ("profile_id", "method_id", "checkpoint_sha256", "adapter",
                "input_contract", "output_contract", "default_execution",
                "safety"):
        if key not in profile:
            raise ValueError(f"ink lane profile is missing {key}: {path}")
    return profile


def lane_interpreter() -> str:
    """The interpreter this lane's steps run in.

    Its own by default, which is what a direct CLI run inside the lane image
    wants. A composed worker image carries two environments -- one with psycopg
    so it can claim from the queue, one holding this lane's frozen lock -- and
    points this at the second. Adding psycopg to the frozen one instead would
    spend the property that lock exists for.
    """
    return os.environ.get("HELENA_INK_9UM_PYTHON") or sys.executable


def preparation_command(*, tiff_dir: Path, destination: Path,
                        source_voxel_um: float) -> list[str]:
    """The argv that pools a P4 layer stack into the runner's surface volume.

    A subprocess in the lane's interpreter rather than a call in this one, for
    the same reason inference is: on a composed worker image `python3` is the
    half that can reach the queue, and the pooling ran there, where zarr is not
    installed. It failed after the worker had correctly claimed and started the
    job -- gpu-1, p5-72ec04c791f846.

    Which interpreter writes it matters beyond whether the import resolves. The
    preparer dispatches on zarr's major version because 2 and 3 disagree about
    how an array is created, and a writer and a reader on different majors is
    the quieter version of this same bug. One interpreter for the whole lane
    means the question cannot come up.
    """
    return [
        lane_interpreter(),
        str(Path(__file__).resolve().parent / "prepare_9um_isotropic_input.py"),
        "--layers", str(tiff_dir),
        "--output", str(destination),
        "--source-voxel-um", str(source_voxel_um),
    ]


def prepared_surface_volume(tiff_dir: Path, work_dir: Path, *,
                           source_voxel_um: float) -> Path:
    """Pool a P4 layer stack into the surface volume upstream's runner reads.

    In this job rather than in a second queued one. A prepare job whose output
    an infer job has to find would be a second state machine -- its own lease,
    its own retry, its own way of being half-done -- to express one lane that
    happens to take two steps. It also splits the record: the receipt that names
    the checkpoint should name the input it pooled, and that is one receipt only
    if it is one job.
    """
    destination = Path(work_dir) / PREPARED_VOLUME_NAME
    argv = preparation_command(tiff_dir=Path(tiff_dir), destination=destination,
                               source_voxel_um=float(source_voxel_um))
    completed = subprocess.run(  # noqa: S603 - argv is built here
        argv, check=False, stderr=subprocess.PIPE)
    if completed.returncode == 0:
        return destination

    # stderr, not the exit code alone. The scale check is the reason pooling
    # happens before a GPU is claimed, and its answer is a sentence somebody can
    # act on -- reporting it as a number would throw that away at the one
    # boundary it has to cross.
    said = (completed.stderr or b"").decode("utf-8", "replace").strip()
    if completed.returncode == REFUSED_SCALE_EXIT:
        raise IncompatibleSourceScale(said or "this scale cannot be pooled to 9 um")
    raise RuntimeError(
        f"pooling the layer stack failed with exit code {completed.returncode}"
        + (f": {said}" if said else ""))


def resolve_surface_volume(*, tiff_dir: Path | None,
                           surface_volume: str | None,
                           work_dir: Path,
                           source_voxel_um: float | None) -> str:
    """Whatever this run should point the runner at, from what it was given.

    Exactly one of the two, for the reason P5 already refuses naming both a
    `tiff_dir` and a `layer_stack`: a lane handed two inputs has no way to say
    which one its map came from.
    """
    if bool(tiff_dir) == bool(surface_volume):
        raise ValueError(
            "name exactly one of --tiff-dir (a P4 layer stack, pooled here) "
            "or --surface-volume (a ready ~9 um isotropic OME-Zarr)")
    if surface_volume:
        return str(surface_volume)
    if source_voxel_um is None:
        raise ValueError(
            "--source-pixel-um is required with --tiff-dir: pooling has to "
            "know what it is pooling from, and guessing 2.4 would turn a "
            "native 9 um render into a 38 um one")
    return str(prepared_surface_volume(
        Path(tiff_dir), Path(work_dir), source_voxel_um=source_voxel_um))


def inference_command(profile: dict[str, Any], *, surface_volume: str,
                      checkpoint: Path, output_tiff: Path) -> list[str]:
    """The exact argv, so the receipt can carry it and a run can be repeated.

    ``input_zarr``, ``checkpoint`` and ``output_tiff`` are positional in
    upstream's parser; sending any of them as a flag value would not run.
    """
    execution = profile["default_execution"]
    direction = execution["direction"]
    if direction not in DIRECTIONS:
        raise ValueError(
            f"unknown inference direction {direction!r}; upstream offers {list(DIRECTIONS)}")
    blend_mode = execution["blend_mode"]
    if blend_mode not in BLEND_MODES:
        raise ValueError(
            f"unknown blend mode {blend_mode!r}; upstream offers {list(BLEND_MODES)}")

    interpreter = lane_interpreter()
    argv = [
        interpreter, "-m", "koine_machines.inference.infer",
        str(surface_volume), str(checkpoint), str(output_tiff),
        "--overlap", str(execution["overlap"]),
        "--blend-mode", str(blend_mode),
        "--direction", str(direction),
        "--batch-size", str(int(execution["batch_size"])),
    ]
    # Absence of the flag is upstream's own default, so this is sent only to
    # turn compilation off -- see the profile's note for why this image wants
    # it off.
    if execution.get("compile") is False:
        argv.append("--no-compile")
    # Only when the profile pins one. These models are sensitive to z offset
    # and upstream defaults both bounds to None; a window chosen here would be
    # a setting no receipt records and no second run reproduces.
    if execution.get("layer_start") is not None:
        argv.extend(["--layer-start", str(int(execution["layer_start"]))])
    if execution.get("layer_end") is not None:
        argv.extend(["--layer-end", str(int(execution["layer_end"]))])
    return argv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--surface-volume",
                    help="a ready ~9 um isotropic OME-Zarr, path or URL")
    ap.add_argument("--tiff-dir", type=Path,
                    help="a P4 numbered TIFF layer stack, pooled here first")
    ap.add_argument("--source-pixel-um", type=float,
                    help="the scale --tiff-dir was rendered at")
    ap.add_argument("--sample-id", help="recorded in the receipt")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True,
                    help="the job's directory: the map, the pooled volume and "
                         "the receipt are all written inside it")
    ap.add_argument("--timeout-seconds", type=int, default=14400)
    ap.add_argument("--on-degenerate", choices=("fail", "warn"), default="fail",
                    help="fail refuses a map that carries no decision; warn is "
                         "the way through for a deliberate diagnostic run and "
                         "still leaves LANE_NOT_USABLE behind.")
    ap.add_argument("--print-command", action="store_true",
                    help="Write the argv and exit without running it.")
    args = ap.parse_args()

    profile = load_lane_profile(args.profile)

    # A directory, like every other P5 lane. It used to be the TIFF's own path,
    # and the worker has one shape for all of them: it passes the job's
    # directory. So this lane was handed a directory and used it as a filename,
    # which cost twelve minutes of inference and then IsADirectoryError at the
    # write -- and, because the pooled volume is placed beside the output, put
    # 865 MB of surface-volume.zarr in the runs root rather than inside the job.
    args.output.mkdir(parents=True, exist_ok=True)
    output_tiff = args.output / OUTPUT_MAP_NAME

    try:
        surface_volume = resolve_surface_volume(
            tiff_dir=args.tiff_dir, surface_volume=args.surface_volume,
            work_dir=args.output, source_voxel_um=args.source_pixel_um)
    except (ValueError, IncompatibleSourceScale) as refused:
        # Before a GPU is claimed: a scale this recipe cannot reach is not
        # something to discover after the model is resident.
        print(str(refused), file=sys.stderr)
        return 3
    argv = inference_command(profile, surface_volume=surface_volume,
                             checkpoint=args.checkpoint,
                             output_tiff=output_tiff)
    if args.print_command:
        print(json.dumps(argv))
        return 0

    # Before inference, not after. This ran at line 326 and compared at 356,
    # which meant a checkpoint that was not the profile's was discovered having
    # spent the whole run -- and with the fourteen ink_9um steps now individually
    # queueable, that cost multiplies by the size of the sweep. Hashing 138 MB
    # takes under a second; the run it saves takes minutes on a GPU.
    checkpoint_sha = sha256_file(args.checkpoint)
    declared = profile["checkpoint_sha256"]
    if not declared:
        print(f"profile {profile['profile_id']} pins no checkpoint_sha256, so "
              f"nothing can say whether {args.checkpoint} is the weights it "
              "means. Add the digest to the profile -- upstream's LFS metadata "
              "carries it.", file=sys.stderr)
        return 3
    if checkpoint_sha != declared:
        print(f"checkpoint digest {checkpoint_sha} is not the {declared} this "
              f"profile declares. Nothing was run.", file=sys.stderr)
        return 3

    started = time.time()
    completed = subprocess.run(argv, timeout=args.timeout_seconds)
    if completed.returncode != 0:
        print(f"koine_machines.inference.infer exited {completed.returncode}",
              file=sys.stderr)
        return completed.returncode
    if not output_tiff.is_file():
        # The same rule P3 applies to vc_flatten: an exit code is not evidence
        # that the output exists.
        print("inference reported success but wrote no output; a lane that "
              "believes an exit code over its own output publishes nothing",
              file=sys.stderr)
        return 3

    # ------------------------------------------------------------ Helena's half
    # Upstream owns the model and its normalisation. What it does not own is
    # whether the map it just wrote carries a decision at all -- a dead head
    # answers every input with the same number and exits zero.
    directory = args.output
    def read_map(path: Path) -> np.ndarray:
        values = np.asarray(tifffile.imread(path))
        if values.dtype == np.uint8:
            # Upstream writes a uint8 tiled TIFF, so the published probability is
            # that byte scale mapped back to [0, 1] and nothing else: 255 steps
            # across a range that in practice occupies about 0.22 to 0.81, which
            # is roughly 150 usable levels. Recorded in the receipt because a
            # reader comparing this lane against a float one is comparing
            # different resolutions, not just different models.
            values = values.astype(np.float32) / UINT8_FULL_SCALE
        return np.squeeze(values)

    probability = read_map(output_tiff)
    np.save(directory / "probability.npy", probability)

    # The reverse map was computed and was being thrown away.
    #
    # The profile pins `direction: both` and says why: "the surface-normal
    # direction that faces the ink is a measurement, and running one direction
    # only assumes an answer this lane has no way to check". Upstream duly
    # wrote both maps -- and this runner read `ink.tif`, published it, and left
    # `ink_reverse.tif` on the disk unread. Half of a deliberately paid-for
    # measurement, discarded silently, with nothing in the receipt to say which
    # side the published map came from.
    #
    # Both are published now, and the receipt carries how far apart they are.
    # It is not this runner's business to decide which side faces the ink --
    # that is the measurement the profile wanted -- but a reader cannot make
    # that call without the number, and neither can P7.
    reverse_tiff = directory / REVERSE_MAP_NAME
    reverse_summary: dict[str, Any] | None = None
    if reverse_tiff.is_file():
        reverse = read_map(reverse_tiff)
        np.save(directory / "probability_reverse.npy", reverse)
        reverse_summary = {
            "map": "probability_reverse.npy",
            "p50": float(np.percentile(reverse, 50)),
            "p99": float(np.percentile(reverse, 99)),
        }
        if reverse.shape == probability.shape:
            forward_flat = probability.ravel().astype(np.float64)
            reverse_flat = reverse.ravel().astype(np.float64)
            if forward_flat.std() > 0 and reverse_flat.std() > 0:
                reverse_summary["pearson_r_with_forward"] = float(
                    np.corrcoef(forward_flat, reverse_flat)[0, 1])
            reverse_summary["identical"] = bool(
                np.array_equal(probability, reverse))
        else:
            reverse_summary["shape_differs"] = [list(probability.shape),
                                                list(reverse.shape)]

    # No `valid` mask: unlike the tiled lanes this runner blends over the whole
    # plane, and `probability > 0` would silently drop the genuine no-ink floor
    # this recipe emits, which is 0.25 and not 0.
    liveness = assess_liveness(probability, valid=np.ones(probability.shape, bool))
    finite = probability.ravel()
    receipt = {
        "schema": "campaignx.ink_9um_screening_receipt.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "profile_id": profile["profile_id"],
        "method_id": profile["method_id"],
        # Already computed and already compared, above, before the run.
        "checkpoint_sha256": checkpoint_sha,
        "declared_checkpoint_sha256": profile["checkpoint_sha256"],
        "runner": "koine_machines.inference.infer",
        "pinned_source_revision": profile.get("pinned_source_revision"),
        "command": argv,
        "input": {
            "surface_volume": str(surface_volume),
            "sample_id": args.sample_id,
            "shape": [int(n) for n in probability.shape],
        },
        "liveness": liveness,
        # Which direction the published map came from, and what the other one
        # looked like. `direction: both` is a measurement the profile asks for
        # on purpose; a receipt that does not say which side it published makes
        # that measurement unusable by whoever reads it later.
        # From the profile, not from `execution`: that name belongs to
        # inference_command and does not exist here. Written as though it did,
        # this was a NameError on the last line of a lane that had already spent
        # its GPU time -- the receipt would have failed after the inference.
        "direction": profile["default_execution"]["direction"],
        "published_map": "probability.npy (forward)",
        "reverse": reverse_summary,
        # 255 steps across a range that occupies roughly 0.22 to 0.81 in
        # practice. Stated because a reader comparing this lane's map against a
        # float lane's is comparing resolutions as well as models.
        "value_quantisation": {
            "source_dtype": "uint8",
            "full_scale": UINT8_FULL_SCALE,
            "distinct_levels": int(np.unique(probability).size),
        },
        "no_ink_floor": profile["output_contract"]["no_ink_floor"],
        "statistics": {
            "p50": float(np.percentile(finite, 50)) if finite.size else 0.0,
            "p90": float(np.percentile(finite, 90)) if finite.size else 0.0,
            "p99": float(np.percentile(finite, 99)) if finite.size else 0.0,
            "max": float(finite.max()) if finite.size else 0.0,
        },
        "runtime_seconds": round(time.time() - started, 2),
        "non_claims": [
            "a probability map is a routing signal, not proof of ink, letters, or text",
            "this recipe's no-ink floor is 0.25, not 0; a screen calibrated "
            "against a 0 floor is reading a different quantity",
            "the published map is raw: the model card's display rescale is not applied",
        ],
    }
    (directory / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("statistics", "liveness",
                                              "runtime_seconds")}, indent=2))
    return refuse_if_not_alive(
        liveness, lane=profile["method_id"], output=directory,
        on_degenerate=args.on_degenerate)


if __name__ == "__main__":
    raise SystemExit(main())
