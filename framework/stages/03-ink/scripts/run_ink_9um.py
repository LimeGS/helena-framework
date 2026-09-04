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


def zarr_cache_root() -> Path | None:
    """Where a --surface-volume URL is mirrored locally, or None if disabled.

    Off unless a caller sets it. The fourteen ink_9um checkpoints run against
    the same PHerc0139 volume, restreamed from S3 in full on every one of
    them -- ~1.4 GB at ~7.5 MB/s, with `direction: both` reading it twice.
    Fourteen checkpoints times two directions is twenty-eight downloads of one
    volume. This does not change what is read, only where it is read from
    the second time onward.
    """
    raw = os.environ.get("HELENA_INK_9UM_ZARR_CACHE")
    return Path(raw) if raw else None


def parse_s3_https_url(url: str) -> tuple[str, str] | None:
    """(bucket, key prefix) from a virtual-hosted-style S3 HTTPS URL.

    None for anything else -- a local path, a different host, a scheme this
    was not written for. The cache is skipped rather than guessed at.
    """
    from urllib.parse import urlparse
    parsed = urlparse(str(url))
    if parsed.scheme not in ("https", "s3"):
        return None
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    host = parsed.netloc
    if ".s3." not in host and not host.startswith("s3."):
        return None
    bucket = host.split(".s3.")[0] if ".s3." in host else parsed.path.lstrip("/").split("/", 1)[0]
    if ".s3." in host:
        prefix = parsed.path.lstrip("/")
    else:
        parts = parsed.path.lstrip("/").split("/", 1)
        prefix = parts[1] if len(parts) > 1 else ""
    if not bucket or not prefix:
        return None
    return bucket, prefix.rstrip("/")


def _anonymous_s3_client():
    import boto3  # noqa: PLC0415
    from botocore import UNSIGNED  # noqa: PLC0415
    from botocore.config import Config  # noqa: PLC0415
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def resolve_zarr_revision(client, bucket: str, prefix: str) -> str | None:
    """A cheap signature for what this store currently holds: the ETag of its
    root metadata object. One HEAD request, not a walk of every chunk -- if
    the metadata is unchanged the array underneath is, by this project's own
    convention that a published volume is not mutated in place.
    """
    for name in ("zarr.json", ".zattrs", ".zgroup", ".zarray"):
        try:
            head = client.head_object(Bucket=bucket, Key=f"{prefix}/{name}")
        except Exception:  # noqa: BLE001 -- tried in order, next name or None
            continue
        return head.get("ETag", "").strip('"') or None
    return None


def mirror_zarr_to_local(url: str, cache_root: Path) -> str | None:
    """Best-effort local mirror of an S3-hosted OME-Zarr. None on anything
    that does not work out, so the caller falls back to the URL unchanged --
    a cache that can fail open is the only kind worth having here.

    Downloaded to a per-attempt temporary directory and moved into place with
    one `rename`, so a job reading the cache never sees a partial copy left by
    another job racing it, and two jobs racing just duplicate the download
    rather than corrupt anything.
    """
    parsed = parse_s3_https_url(url)
    if parsed is None:
        return None
    bucket, prefix = parsed
    try:
        client = _anonymous_s3_client()
        revision = resolve_zarr_revision(client, bucket, prefix)
        key = hashlib.sha256(f"{bucket}/{prefix}@{revision or 'unknown'}".encode()).hexdigest()[:24]
        final_dir = cache_root / key
        marker = final_dir / "_HELENA_CACHE_COMPLETE"
        if marker.is_file():
            return str(final_dir)

        cache_root.mkdir(parents=True, exist_ok=True)
        temp_dir = cache_root / f".tmp-{os.getpid()}-{key}"
        if temp_dir.exists():
            import shutil  # noqa: PLC0415
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True)

        paginator = client.get_paginator("list_objects_v2")
        objects = [obj["Key"] for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/")
                   for obj in page.get("Contents", ())]
        if not objects:
            return None

        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        def fetch(remote_key: str) -> None:
            relative = remote_key[len(prefix) + 1:]
            destination = temp_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, remote_key, str(destination))

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, objects))

        (temp_dir / "_HELENA_CACHE_COMPLETE").write_text(
            json.dumps({"source": url, "revision": revision, "objects": len(objects)}))
        os.replace(temp_dir, final_dir)  # atomic on the same filesystem
        return str(final_dir)
    except Exception:  # noqa: BLE001 -- a cache that cannot fetch reads the URL instead
        return None


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
        cache_root = zarr_cache_root()
        if cache_root is not None:
            cached = mirror_zarr_to_local(str(surface_volume), cache_root)
            if cached is not None:
                return cached
        return str(surface_volume)
    if source_voxel_um is None:
        raise ValueError(
            "--source-pixel-um is required with --tiff-dir: pooling has to "
            "know what it is pooling from, and guessing 2.4 would turn a "
            "native 9 um render into a 38 um one")
    return str(prepared_surface_volume(
        Path(tiff_dir), Path(work_dir), source_voxel_um=source_voxel_um))


# Measured 6 GB (gpu-1, empties the map at batch 4) and 32 GB (RTX 5090, 1/4/8
# all ALIVE and correlated r>0.99999). Nothing in between has been measured,
# so this is a guess at the midpoint, not a finding -- move it if a card near
# this line turns out to belong on the other side.
DEFAULT_BATCH_SIZE_VRAM_THRESHOLD_MB = 16_384


def detect_gpu_vram_mb() -> int | None:
    """This process's own GPU memory budget, in MiB, or None if unreadable.

    Deliberately not host-wide: a container sees only the card(s) its
    CUDA_VISIBLE_DEVICES grants it, and a batch size has to fit what this
    process can use, not what the host owns in total. Unlike
    framework.contracts.host_probe.host_state, which strips that variable so
    the panel can show a host's full inventory, this keeps it.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        totals = [int(line.strip()) for line in out.stdout.strip().splitlines()
                  if line.strip()]
        return min(totals) if totals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def default_batch_size_for_host(profile: dict[str, Any]) -> int:
    """4 on a card with enough VRAM to have been measured safe there, else the
    profile's own pinned batch_size -- the one value known to work
    everywhere, including a host this cannot read anything about.
    """
    vram_mb = detect_gpu_vram_mb()
    if vram_mb is not None and vram_mb >= DEFAULT_BATCH_SIZE_VRAM_THRESHOLD_MB:
        return 4
    return int(profile["default_execution"]["batch_size"])


def inference_command(profile: dict[str, Any], *, surface_volume: str,
                      checkpoint: Path, output_tiff: Path,
                      batch_size: int | None = None,
                      num_workers: int | None = None,
                      layer_start: int | None = None,
                      layer_end: int | None = None) -> list[str]:
    """The exact argv, so the receipt can carry it and a run can be repeated.

    ``input_zarr``, ``checkpoint`` and ``output_tiff`` are positional in
    upstream's parser; sending any of them as a flag value would not run.

    ``batch_size`` overrides this host's own detected default when given.
    Left None, ``default_batch_size_for_host`` picks 4 on a card with at
    least 16 GiB and falls back to the profile's own pinned 1 -- measured
    against a 6 GB card where 4 emptied the map -- on anything smaller or
    unreadable. That measurement is about the card, not the model, so a
    caller who knows their own hardware better than the guess can still
    override either way.

    ``num_workers`` overrides upstream's own --num-workers default (4) when
    given. Measured live against a caching run: the DataLoader workers sat at
    ~11% each while the main process pinned one core -- they are not the
    bottleneck today, so raising this alone does little. It becomes worth
    setting once HELENA_INK_9UM_ZARR_CACHE turns the workers' own read from an
    S3 stream into a local one they can outrun the model with.

    ``layer_start``/``layer_end`` override the profile's own pinned window
    (default_execution.layer_start/layer_end, both null by default -- upstream
    reads the whole stack). A caller who names one names both: a job asking
    for one edge of the window and inheriting the other from the profile is
    not a window anyone chose, and this lane refuses to guess the missing
    edge. Added after a band-position experiment (top/center/bottom thirds of
    a stack) had to be built as three separate on-disk layer directories,
    because the profile pinned a window nothing could override per job.

    Neither is a change to overlap, blend mode or direction: nothing has
    measured what changing those costs, and this stays the four knobs that do.
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
    resolved_batch_size = (default_batch_size_for_host(profile) if batch_size is None
                           else int(batch_size))
    if resolved_batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {resolved_batch_size}")
    if num_workers is not None and int(num_workers) < 0:
        raise ValueError(f"num_workers must be at least 0, got {num_workers}")

    interpreter = lane_interpreter()
    argv = [
        interpreter, "-m", "koine_machines.inference.infer",
        str(surface_volume), str(checkpoint), str(output_tiff),
        "--overlap", str(execution["overlap"]),
        "--blend-mode", str(blend_mode),
        "--direction", str(direction),
        "--batch-size", str(resolved_batch_size),
    ]
    if num_workers is not None:
        argv += ["--num-workers", str(int(num_workers))]
    # Absence of the flag is upstream's own default, so this is sent only to
    # turn compilation off -- see the profile's note for why this image wants
    # it off.
    if execution.get("compile") is False:
        argv.append("--no-compile")
    # A caller's window wins whole, not edge by edge -- see the docstring.
    if layer_start is not None or layer_end is not None:
        if layer_start is None or layer_end is None:
            raise ValueError(
                "layer_start and layer_end must be given together: a window "
                "with only one edge chosen is not a window either caller meant")
        resolved_layer_start, resolved_layer_end = int(layer_start), int(layer_end)
    else:
        resolved_layer_start = execution.get("layer_start")
        resolved_layer_end = execution.get("layer_end")
    # Only when pinned or asked for. These models are sensitive to z offset
    # and upstream defaults both bounds to None; a window chosen here would be
    # a setting no receipt records and no second run reproduces.
    if resolved_layer_start is not None:
        argv.extend(["--layer-start", str(int(resolved_layer_start))])
    if resolved_layer_end is not None:
        argv.extend(["--layer-end", str(int(resolved_layer_end))])
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
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the profile's default_execution.batch_size "
                         "(default 1, chosen against a 6 GB card; a card with "
                         "more VRAM may run more patches per step)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override upstream's own DataLoader worker count "
                         "(default 4); worth raising once the input is cached "
                         "locally and the workers are no longer S3-bound")
    ap.add_argument("--layer-start", type=int, default=None,
                    help="override the profile's default_execution.layer_start "
                         "-- must be given with --layer-end, never alone")
    ap.add_argument("--layer-end", type=int, default=None,
                    help="override the profile's default_execution.layer_end "
                         "-- must be given with --layer-start, never alone")
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
                             output_tiff=output_tiff,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             layer_start=args.layer_start,
                             layer_end=args.layer_end)
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
