#!/usr/bin/env python3
"""Claim ink jobs and run them. One process per host that has a GPU to offer.

    python3 -m framework.stages.03-ink.fleet.ink_worker \
        --host-id this-machine --runs-root /srv/helena/runs

The worker never receives a command from the queue -- it receives a profile and
validated parameters and builds the command itself. A job row cannot make this
process run something the framework does not already know how to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_store import (  # noqa: E402
    INK_ADAPTERS, PHASE_RUNNERS, InkJobStore, command_for,
    depth_centers_that_fit, ink_adapter, lane_for,
)


def receipt_names(job: dict) -> tuple[str, ...]:
    """The receipt filenames this job's runner may have written.

    Each ink adapter names its own receipt -- INK_PROFILE_RECEIPT,
    INK_SCREENING_RECEIPT, INK_CANONICAL_RECEIPT -- and the registry records
    which. The worker used to look only for the first, so a lane could compute
    a liveness verdict and have it silently dropped.

    The full set is the fallback, because a phase that is not P5 has no adapter
    in the registry and still writes a receipt.
    """
    lane_receipt = lane_for(job)[1].get("receipt")
    known = tuple(
        spec["receipt"] for spec in INK_ADAPTERS.values() if spec.get("receipt")
    )
    if lane_receipt:
        return (str(lane_receipt), *(name for name in known
                                     if name != lane_receipt))
    if str(job.get("phase")) == "P5":
        try:
            adapter = ink_adapter(job.get("profile_id"))[0]
        except Exception:  # noqa: BLE001 -- fall back to trying all of them
            return known
        named = INK_ADAPTERS.get(adapter, {}).get("receipt")
        if named:
            return (named, *(n for n in known if n != named))
    return known


def merge_result_from_receipt(
    job: dict,
    receipt: dict | None,
    publication: dict | None,
) -> dict:
    """Project a successful P8 merge into the durable queue result.

    The complete merge receipt and the evidence-publication record are kept as
    separate documents to avoid a circular digest: the receipt is a member of
    the evidence set, while the publication record names that set's digest.
    Both must exist before the worker can call the job successful.  Keeping
    them in ``ink_jobs.result`` makes the API sufficient to discover the new
    surface and its evidence without reading a worker-local directory.
    """
    if (str(job.get("phase")) != "P8"
            or (job.get("parameters") or {}).get("lane")
            != "vc3d-tifxyz-merge"):
        return {}
    if not isinstance(receipt, dict):
        raise RuntimeError("P8 merge produced no MERGE_RECEIPT.json")
    if receipt.get("schema") != "campaignx.vc3d_tifxyz_merge_receipt.v1":
        raise RuntimeError("P8 merge receipt has the wrong schema")
    if receipt.get("status") != "PASS":
        raise RuntimeError("P8 merge receipt does not record PASS")
    required_receipt = (
        "surface_id", "artifact_uri", "artifact_sha256", "parents",
    )
    missing = [name for name in required_receipt if not receipt.get(name)]
    if missing:
        raise RuntimeError(f"P8 merge receipt is missing {missing}")
    if not isinstance(publication, dict):
        raise RuntimeError("P8 merge produced no EVIDENCE_PUBLICATION.json")
    if publication.get("schema") != "campaignx.vc3d_merge_evidence_publication.v1":
        raise RuntimeError("P8 evidence publication has the wrong schema")
    required_publication = ("evidence_uri", "evidence_sha256", "registration")
    missing = [name for name in required_publication if not publication.get(name)]
    if missing:
        raise RuntimeError(f"P8 evidence publication is missing {missing}")
    registered = publication["registration"].get("surface_id")
    if registered != receipt["surface_id"]:
        raise RuntimeError(
            "P8 evidence publication names a different registered surface")
    return {
        "merge_receipt": receipt,
        "evidence_publication": publication,
        "surface_id": receipt["surface_id"],
        "artifact_uri": receipt["artifact_uri"],
        "artifact_sha256": receipt["artifact_sha256"],
        "evidence_uri": publication["evidence_uri"],
        "evidence_sha256": publication["evidence_sha256"],
        "parents": receipt["parents"],
    }


def runner_for(job: dict) -> Path:
    """Resolve the runner for this job, refusing one that has none.

    P5 dispatches on the lane profile: it names its adapter, there are two of
    them, and they do not take the same arguments. Every other phase has one.
    """
    phase = str(job.get("phase") or "P5")
    if phase == "P5":
        relative = ink_adapter(job.get("profile_id"))[0]
    else:
        # The lane names its own program. A phase gains a second implementation
        # by gaining a lane, which is a row in a table rather than an edit here.
        relative = lane_for(job)[1].get("runner") or PHASE_RUNNERS.get(phase)
    if relative is None:
        raise RuntimeError(f"phase {phase} has no runner registered")
    runner = ROOT / relative
    if not runner.exists():
        raise RuntimeError(f"runner for {phase} is not present at {runner}")
    return runner


# Measured by framework/contracts/host_probe, which a segmentation host also
# imports. Re-exported here because callers -- the panel included -- already
# reach for fleet.ink_worker.host_state.
from framework.contracts.host_probe import (  # noqa: E402
    cpu_and_memory, host_state, local_images,
)



def record_artifact(job: dict, output: Path, *, runs_root: Path) -> dict | None:
    """Register what this job produced, with what it read as lineage.

    Wrapped so nothing here can fail the job. The run is the result; this is the
    note about it, and a note that can abort the thing it describes is a bad
    note -- a job that produced a real map must not be marked failed because a
    mission directory was missing.

    A job with no mission has nowhere to record: the unfiled view is assembled
    from receipts and owns no artifacts.
    """
    mission_id = job.get("mission_id")
    if not mission_id:
        return None
    try:
        from framework.contracts import artifact as artifact_contract

        directory = Path(runs_root) / mission_id
        if not (directory / "MISSION.json").exists():
            return None
        return artifact_contract.record_run(
            directory,
            ROOT / "framework" / "contracts" / "pipeline_phases.json",
            phase=job.get("phase", "P5"),
            sample_id=job["sample_id"],
            output=output,
            produced_by=f"job:{job['job_id']}",
            note=f"profile {job.get('profile_id') or '?'}",
        )
    except Exception as exc:  # noqa: BLE001 -- bookkeeping never fails a run
        print(f"artifact not recorded: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return None


DEFAULT_FLATTENING_PROFILE = "flatten-abf-v1@1.0.0"


class RenderNotUsable(RuntimeError):
    """The renderer exited zero and what it wrote is not a layer stack."""


def rendered_layers_directory(job: dict, output: Path) -> Path:
    """Where this P4 lane actually writes the numbered TIFF stack."""
    parameters = job.get("parameters") or {}
    if parameters.get("lane") == "scroll3-chunk-gather":
        return Path(str(parameters.get("out_dir") or output)) / "layers"
    return Path(str(parameters.get("tif_output") or (output / "layers")))


def verify_layer_stack(directory: Path, parameters: dict) -> dict:
    """Look at what was rendered before calling the job a success.

    P3 parses its own output as a TIFXYZ before believing the exit code; P4
    believed the exit code alone. A render whose surface fell outside the cached
    region, or whose scale was wrong, writes the requested number of slices and
    every one of them is a constant -- exit 0, 33 files, nothing in them.

    This is not a quality measure. It separates "wrote a stack" from "wrote
    nothing worth reading", which is the difference the exit code was standing
    in for.
    """
    import numpy  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    slices = sorted(Path(directory).glob("*.tif"))
    if not slices:
        raise RenderNotUsable(f"the renderer wrote no .tif under {directory}")
    expected = parameters.get("num_slices")
    if expected and len(slices) != int(expected):
        raise RenderNotUsable(
            f"asked for {int(expected)} slices and found {len(slices)}")
    # The middle one: the ends of a stack can legitimately fall off the lamina,
    # and a stack whose centre is blank was not sampling the sheet at all.
    middle = tifffile.imread(slices[len(slices) // 2])
    low, high = float(numpy.min(middle)), float(numpy.max(middle))
    if low == high:
        raise RenderNotUsable(
            f"every voxel of the middle slice is {low}: the stack carries no "
            "signal, which exit 0 does not say")
    return {
        "slices": len(slices),
        "shape": [int(value) for value in middle.shape],
        "dtype": str(middle.dtype),
        "middle_slice_range": [low, high],
        "bytes": sum(path.stat().st_size for path in slices),
    }


def publish_artifact_set(directory: Path, names: list[str], *, schema: str,
                         store_spec: str, sample_id: str, key: str,
                         job_id: str) -> dict:
    """Publish a set of files with a manifest, and say where they went."""
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.artifact_store import open_artifact_store  # noqa: PLC0415
    from fleet.common import content_sha256, file_sha256  # noqa: PLC0415

    directory = Path(directory)
    files = {name: {"sha256": file_sha256(directory / name),
                    "size_bytes": (directory / name).stat().st_size}
             for name in names}
    manifest = {"schema": schema, "job_id": job_id, "files": files,
                "artifact_sha256": content_sha256(files)}
    store = open_artifact_store(store_spec)
    staged = store.stage(directory, f"{key.replace('/', '-')}-{job_id}", manifest)
    promoted = store.promote(staged, sample_id, key, manifest)
    return {"artifact_uri": promoted["artifact_uri"],
            "artifact_sha256": manifest["artifact_sha256"],
            "files": len(files)}


def publish_probability_map(directory: Path, *, store_spec: str, sample_id: str,
                            job_id: str) -> dict | None:
    """The maps P5 produced, where another host can read them.

    Same reason as the layer stack: a probability map on the disk of the worker
    that computed it is a result nobody else can check.
    """
    directory = Path(directory)
    names = sorted(path.name for path in directory.glob("*.npy"))
    receipt = directory / "INK_SCREENING_RECEIPT.json"
    if receipt.is_file():
        names.append(receipt.name)
    if not names:
        return None
    return publish_artifact_set(
        directory, names, schema="campaignx.ink_probability_map.v1",
        store_spec=store_spec, sample_id=sample_id,
        key=f"ink-maps/{job_id}", job_id=job_id)


def publish_layer_stack(directory: Path, *, store_spec: str, sample_id: str,
                        job_id: str) -> dict:
    """Put the stack where a worker's death does not take it with it.

    Everything else in the pipeline publishes: the grown surface, the flattened
    sheet. P4's stack -- the thing the detector actually eats -- was written to
    the runs directory of whichever host claimed the job and recorded as a local
    path, which means nothing on any other machine.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.artifact_store import open_artifact_store  # noqa: PLC0415
    from fleet.common import content_sha256, file_sha256  # noqa: PLC0415

    names = sorted(path.name for path in Path(directory).glob("*.tif"))
    return publish_artifact_set(
        directory, names, schema="campaignx.layer_stack_artifact_set.v1",
        store_spec=store_spec, sample_id=sample_id,
        key=f"layers/{job_id}", job_id=job_id)


def resolve_flattened_surface(store: InkJobStore, job: dict, destination: Path) -> str:
    """Fetch the sheet P3 unrolled and hand back the directory to render.

    P4 has always taken a path to a tifxyz, which in practice was a P1 surface --
    the curved patch -- because nothing produced a flattened sheet to point it
    at. Now something does, and a job can name the surface instead of a path:
    the sheet is looked up by (surface, profile), fetched from wherever P3
    published it, and rendered from there.
    """
    surface_id = str(job["parameters"]["flattened_surface"])
    profile_id = str(job["parameters"].get("flattening_profile")
                     or DEFAULT_FLATTENING_PROFILE)
    sheet = store.flattened_sheet(surface_id, profile_id)

    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.certifier import load_qc_adapter  # noqa: PLC0415

    load_qc_adapter().materialize_surface(
        str(sheet["artifact_uri"]), str(sheet["artifact_sha256"] or ""), destination)
    return str(destination)


def fetch_artifact_set(artifact_uri: str, destination: Path) -> dict:
    """Bring a published artifact set down, verifying every file against its
    manifest.

    The surface adapter does this for TIFXYZ and refuses anything else by
    schema, which is right: the provenance of a layer stack depends on knowing
    which kind of thing it is. A layer stack is 33 numbered slices, so it needs
    its own fetch, and it needs the same verification -- an artifact that
    arrives with a different digest is a different artifact.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import file_sha256  # noqa: PLC0415

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "s3":
        import boto3  # noqa: PLC0415

        client = boto3.client("s3")
        bucket, base = parsed.netloc, parsed.path.strip("/")
        manifest = json.loads(client.get_object(
            Bucket=bucket, Key=f"{base}/ARTIFACT_SET.json")["Body"].read())
        for name in manifest["files"]:
            client.download_file(bucket, f"{base}/{name}", str(destination / name))
    else:
        source = Path(parsed.path if parsed.scheme == "file" else artifact_uri)
        manifest = json.loads((source / "ARTIFACT_SET.json").read_text())
        for name in manifest["files"]:
            shutil.copy2(source / name, destination / name)
    for name, expected in manifest["files"].items():
        digest = file_sha256(destination / name)
        if digest != expected["sha256"]:
            raise RuntimeError(
                f"{name} arrived with digest {digest[:12]} and the manifest says "
                f"{expected['sha256'][:12]}: this is not that artifact")
    return manifest


def resolve_layer_stack(store: InkJobStore, job: dict, destination: Path) -> str:
    """Fetch the stack a P4 job published and hand back the directory to read.

    P5 took `tiff_dir`, a path on whichever machine happened to have the layers,
    which is why it had only ever been run by hand against a directory somebody
    downloaded. Naming the render instead makes the chain P4 -> P5 a thing the
    queue can express, and makes what a probability map was computed from a
    matter of record rather than of a path in an argv.
    """
    rendered_by = str(job["parameters"]["layer_stack"])
    render = store.job(rendered_by)
    if render is None:
        raise RuntimeError(f"no such render: {rendered_by}")
    if render.get("phase") != "P4":
        raise RuntimeError(f"{rendered_by} is a {render.get('phase')} job, not a render")
    if render.get("state") != "succeeded":
        raise RuntimeError(
            f"render {rendered_by} is {render.get('state')}; a probability map "
            "computed on a failed render is a map of whatever was left behind")
    published = (render.get("result") or {}).get("layer_stack") or {}
    if not published.get("artifact_uri"):
        raise RuntimeError(
            f"render {rendered_by} published no layer stack, so there is nothing "
            "to read on this machine")
    fetch_artifact_set(str(published["artifact_uri"]), destination)
    return str(destination)


def slice_pitch_from_render(job: dict, render: dict) -> float | None:
    """The depth pitch of a stack, from the render that produced it.

    The TimeSformer adapter refuses to default this, and it is right to: the
    campaign spans 8.64 and 9.362 um acquisitions and assuming either rescales
    the other by 8.4% in silence. But it is not a guess when the render is
    named -- P4 recorded the scale it sampled at and the step it took along the
    normal, and the pitch follows:

        pitch = source_pixel_um * scale * slice_step

    At scale 1.0 and a one-voxel step, which is every render this pipeline has
    made, that is the pixel size itself. Derived rather than typed, because a
    number the platform already knows and asks a person to retype is a number
    that will eventually be retyped wrong.
    """
    parameters = render.get("parameters") or {}
    lateral = job["parameters"].get("source_pixel_um")
    if lateral in (None, ""):
        return None
    try:
        scale = float(parameters.get("scale") or 1.0)
        step = float(parameters.get("slice_step") or 1.0)
        return float(lateral) * scale * step
    except (TypeError, ValueError):
        return None


def fit_depth_to_stack(job: dict, stack: Path) -> None:
    """Refuse, or centre, the detector window before a GPU is reserved.

    A lane profile's depth centres are written for the stack depth its author
    had: the GP Scroll1 lane says 25, 32 and 39, which are positions in a
    62-layer surface volume. Pointed at a 33-slice render they fall past the
    end, and the failure arrives as "depth positions extend beyond the source
    stack" after the job is claimed, the sheet fetched and the model loaded.

    With no centres asked for, this centres the window on the stack, which is
    the only defensible default: it is the deepest sampling the render supports.
    """
    _, spec, profile_path = ink_adapter(job.get("profile_id"))
    if profile_path is None:
        return
    # Each adapter names this differently and means something different by it:
    # one takes a comma-separated list of centres, the other a single integer.
    # Writing the wrong key is writing nothing, and the runner then falls back
    # to its own default -- which is how a 62-frame window came to be centred at
    # 31 of 62 layers and refused for half a slice.
    key = "depth_centers" if "depth_centers" in spec["flags"] else (
        "depth_center" if "depth_center" in spec["flags"] else None)
    if key is None:
        return
    profile = json.loads(Path(profile_path).read_text())
    contract = profile.get("input_contract") or {}
    frames = int(contract.get("frames") or 0)
    training_slice_um = float(contract.get("training_slice_um") or 0)
    source_slice_um = float(job["parameters"].get("source_slice_um")
                            or job["parameters"].get("source_pixel_um") or 0)
    slices = len(sorted(Path(stack).glob("*.tif")))
    if not (frames and training_slice_um and source_slice_um and slices):
        return

    asked = job["parameters"].get(key)
    centers = ([float(value) for value in str(asked).split(",")] if asked
               else [float(value) for value in
                     (profile.get("default_execution") or {}).get("depth_centers")
                     or [(slices - 1) / 2.0]])
    fitting = depth_centers_that_fit(
        slices, frames, centers, source_slice_um=source_slice_um,
        training_slice_um=training_slice_um)
    if key == "depth_center":
        # An integer flag cannot express 30.5, and a 62-frame window on a
        # 62-slice stack fits at no other centre. Round-tripping it as an int
        # would hand the runner a window half a slice past the end.
        fitting = [center for center in fitting if float(center).is_integer()]
    if fitting:
        job["parameters"][key] = (int(fitting[0]) if key == "depth_center"
                                  else ",".join(f"{c:g}" for c in fitting))
        return
    middle = (slices - 1) / 2.0
    if key == "depth_center":
        middle = float(round(middle))
    if depth_centers_that_fit(slices, frames, [middle],
                              source_slice_um=source_slice_um,
                              training_slice_um=training_slice_um):
        if asked:
            raise RuntimeError(
                f"depth centres {centers} do not fit a {slices}-slice stack "
                f"sampled at {source_slice_um} um for a model that wants "
                f"{frames} frames at {training_slice_um} um; {middle:g} does")
        job["parameters"][key] = (int(middle) if key == "depth_center"
                                  else f"{middle:g}")
        return
    raise RuntimeError(
        f"a {slices}-slice stack at {source_slice_um} um is too shallow for "
        f"{frames} frames at {training_slice_um} um: the window needs "
        f"{frames * training_slice_um / source_slice_um:.1f} slices. Render more "
        "slices in P4, or use a lane trained at a coarser pitch")


def canonical_probability_map(directory: Path) -> Path:
    """Select the aggregate map by contract, never by an arbitrary glob.

    Older lanes wrote ``probability.npy``.  The TimeSformer screening lane
    writes its aggregate as ``mean_probability.npy`` beside per-window maps
    and ``stability_std.npy``.  Those other arrays are evidence, but they are
    not interchangeable with the probability map P7 adjudicates.
    """
    directory = Path(directory)
    for name in ("probability.npy", "mean_probability.npy"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    arrays = ", ".join(path.name for path in sorted(directory.glob("*.npy")))
    found = arrays or "no .npy files"
    raise RuntimeError(
        f"no canonical probability map in {directory}: found {found}; expected "
        "probability.npy or mean_probability.npy")


def read_adjudication(output: Path) -> dict:
    """Read P7's measured outcome and reject an incomplete success receipt.

    A refutation is a valid scientific result, so ``overall.pass = false`` is
    recorded rather than turned into a failed queue job.  Missing, malformed or
    error receipts are operational failures: exit zero alone cannot establish
    that the adjudicator evaluated the map.
    """
    output = Path(output)
    verdict_path = output / "verdict.json"
    card_path = output / "VETTING_CARD.md"
    if not verdict_path.is_file():
        raise RuntimeError("P7 exited zero without verdict.json")
    if not card_path.is_file():
        raise RuntimeError("P7 exited zero without VETTING_CARD.md")
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, json.JSONDecodeError) as failure:
        raise RuntimeError(f"P7 wrote an unreadable verdict: {failure}") from failure
    if not isinstance(verdict, dict) or verdict.get("status") != "ok":
        raise RuntimeError(
            f"P7 verdict status is {verdict.get('status') if isinstance(verdict, dict) else 'invalid'}")
    overall = verdict.get("overall")
    if not isinstance(overall, dict) or type(overall.get("pass")) is not bool:
        raise RuntimeError("P7 verdict has no boolean overall.pass")
    checks = verdict.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("P7 verdict has no checks object")
    return {
        "verdict": "PASS" if overall["pass"] else "FAIL",
        "overall": overall,
        "checks": checks,
        "input": verdict.get("input"),
        "evaluator": verdict.get("evaluator"),
        "schema_version": verdict.get("schema_version"),
        "tool": verdict.get("tool"),
        "tool_version": verdict.get("tool_version"),
        "config_hash": verdict.get("config_hash"),
        "verdict_file": verdict_path.name,
        "card_file": card_path.name,
    }


def verify_plate_set(job: dict) -> dict:
    """Verify P9's complete, content-addressed plate inventory.

    The composer historically skipped wraps without a map and still exited
    zero.  A successful process is therefore insufficient evidence: the
    manifest must bind every PNG, the measured P8 order and the subject before
    the queue records a successful plate run.
    """
    from PIL import Image  # noqa: PLC0415

    parameters = job.get("parameters") or {}
    directory = Path(str(parameters.get("out_dir") or ""))
    manifest_path = directory / "PLATE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("P9 exited zero without PLATE_MANIFEST.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as failure:
        raise RuntimeError(f"P9 wrote an unreadable plate manifest: {failure}") from failure
    if not isinstance(manifest, dict):
        raise RuntimeError("P9 plate manifest is not an object")
    if manifest.get("schema") != "campaignx.p9_plate_set.v1":
        raise RuntimeError("P9 plate manifest has the wrong schema")
    if manifest.get("status") != "PASS":
        raise RuntimeError("P9 plate manifest does not record PASS")
    if str(manifest.get("sample_id") or "") != str(job.get("sample_id") or ""):
        raise RuntimeError("P9 plate manifest names a different sample")

    order_path = Path(str(parameters.get("order_path") or ""))
    if not order_path.is_file():
        raise RuntimeError("P9 has no readable measured order to bind")
    order_sha256 = hashlib.sha256(order_path.read_bytes()).hexdigest()
    if manifest.get("ordering_sha256") != order_sha256:
        raise RuntimeError("P9 plate manifest does not bind the measured P8 order")

    plates = manifest.get("plates")
    if not isinstance(plates, list) or not plates:
        raise RuntimeError("P9 plate manifest contains no plates")
    if manifest.get("plate_count") != len(plates):
        raise RuntimeError("P9 plate count disagrees with its inventory")

    declared: set[str] = set()
    verified = []
    total_bytes = 0
    for entry in plates:
        if not isinstance(entry, dict):
            raise RuntimeError("P9 plate inventory contains a non-object entry")
        name = str(entry.get("file") or "")
        if not name or Path(name).name != name or not name.endswith(".png"):
            raise RuntimeError(f"P9 plate inventory has unsafe filename {name!r}")
        if name in declared:
            raise RuntimeError(f"P9 plate inventory repeats {name}")
        declared.add(name)
        plate = directory / name
        if not plate.is_file():
            raise RuntimeError(f"P9 is missing plate {name}")
        size = plate.stat().st_size
        digest = hashlib.sha256(plate.read_bytes()).hexdigest()
        if entry.get("bytes") != size:
            raise RuntimeError(f"P9 plate {name} has the wrong byte size")
        if entry.get("sha256") != digest:
            raise RuntimeError(f"P9 plate {name} has the wrong SHA-256")
        try:
            with Image.open(plate) as image:
                image.verify()
            with Image.open(plate) as image:
                width, height = image.size
                image_format = image.format
        except Exception as failure:  # noqa: BLE001
            raise RuntimeError(f"P9 plate {name} is not a readable PNG: {failure}") from failure
        if image_format != "PNG" or entry.get("width") != width \
                or entry.get("height") != height:
            raise RuntimeError(f"P9 plate {name} dimensions or format disagree")
        total_bytes += size
        verified.append({"file": name, "wrap": entry.get("wrap"),
                         "sha256": digest, "bytes": size,
                         "width": width, "height": height})

    actual = {path.name for path in directory.glob("*.png")}
    if actual != declared:
        extra = sorted(actual - declared)
        missing = sorted(declared - actual)
        raise RuntimeError(
            f"P9 PNG inventory mismatch: extra={extra}, missing={missing}")
    return {
        "schema": manifest["schema"],
        "manifest_file": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "ordering_sha256": order_sha256,
        "plate_count": len(verified),
        "bytes": total_bytes,
        "plates": verified,
    }


def resolve_screened_map(store: InkJobStore, job: dict, destination: Path) -> str:
    """Fetch the map a P5 job produced, and refuse one that is not alive.

    P7 took `map_path`, a file on whichever machine held it, so the chain from a
    screening to its adjudication was a path somebody carried by hand. Naming
    the job makes it something the queue can express -- and makes the liveness
    verdict readable from the job row rather than from a receipt that happens to
    sit beside the file.
    """
    screening_id = str(job["parameters"]["screening_of"])
    screening = store.job(screening_id)
    if screening is None:
        raise RuntimeError(f"no such screening: {screening_id}")
    if screening.get("phase") != "P5":
        raise RuntimeError(
            f"{screening_id} is a {screening.get('phase')} job, not a screening")
    if screening.get("state") != "succeeded":
        raise RuntimeError(
            f"screening {screening_id} is {screening.get('state')}; adjudicating "
            "a failed screening adjudicates whatever was left behind")
    result = screening.get("result") or {}
    verdict = ((result.get("liveness") or {}).get("verdict"))
    if verdict != "ALIVE":
        # The gate P6 was written to be and never was: screening finds shapes in
        # noise perfectly well, so a map the lane could not read must not reach
        # an adjudication.
        raise RuntimeError(
            f"the map of {screening_id} is {verdict or 'unrecorded'}, not ALIVE: "
            "screening a map the lane could not read finds shapes in noise")
    published = (result.get("probability_map") or {}).get("artifact_uri")
    if published:
        fetch_artifact_set(str(published), destination)
        return str(canonical_probability_map(destination))
    local_output = Path(str(result.get("output_dir") or ""))
    try:
        return str(canonical_probability_map(local_output))
    except RuntimeError as failure:
        raise RuntimeError(
            f"screening {screening_id} published no map and has no canonical map "
            f"on this worker ({failure}); give P5 an artifact_store so its map "
            "outlives the machine that made it") from failure


def resolve_wrap_order(store: InkJobStore, job: dict) -> str:
    """Resolve the measured radial table a P9 plate run names."""
    ordering_id = str(job["parameters"]["ordering_of"])
    ordering = store.job(ordering_id)
    if ordering is None:
        raise RuntimeError(f"no such wrap-order job: {ordering_id}")
    if ordering.get("phase") != "P8":
        raise RuntimeError(
            f"{ordering_id} is a {ordering.get('phase')} job, not a wrap order")
    if ordering.get("state") != "succeeded":
        raise RuntimeError(
            f"wrap-order job {ordering_id} is {ordering.get('state')}")
    if (ordering.get("sample_id") and job.get("sample_id")
            and str(ordering["sample_id"]) != str(job["sample_id"])):
        raise RuntimeError(
            f"wrap-order job {ordering_id} is for {ordering['sample_id']}, not "
            f"{job['sample_id']}")
    path = Path(str((ordering.get("parameters") or {}).get("out_path") or ""))
    if not path.is_file():
        raise RuntimeError(
            f"wrap-order job {ordering_id} published no radial order on this worker")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as failure:
        raise RuntimeError(
            f"wrap-order job {ordering_id} wrote unreadable JSON: {failure}") from failure
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), dict) \
            or not payload["segments"]:
        raise RuntimeError(
            f"wrap-order job {ordering_id} has no measured segments")
    return str(path)


def runner_environment(job: dict) -> dict[str, str]:
    """The environment the runner subprocess gets.

    P4's renderer streams the CT from the public open-data bucket, which serves
    it anonymously. The worker's own credentials belong to the private bucket
    the campaign publishes artifacts to, and a signed request against a bucket
    those keys do not own comes back 400 -- "Error opening remote zarr: HTTP 400
    fetching .zattrs", one second into a render, on a URL that answers 200 to
    curl.

    So the renderer runs without them. The worker keeps them: it is what fetches
    the flattened sheet out of the private bucket a moment earlier. A volume in
    a private bucket would need this reconsidered, and there is not one.
    """
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    # Job identity is control-plane state, not a client parameter. Derived
    # artifacts bind their N->1 registration to this immutable queue row.
    if job.get("job_id"):
        environment["HELENA_JOB_ID"] = str(job["job_id"])
    if job.get("phase") == "P5":
        # A vendored runner that imports its architecture from beside itself
        # needs that directory on the path, and it is a job parameter because
        # which model code a lane runs is part of what its receipt has to say.
        spec = ink_adapter(job.get("profile_id"))[1]
        source = spec.get("pythonpath_from")
        upstream = job["parameters"].get(source) if source else None
        if upstream:
            environment["PYTHONPATH"] = f"{upstream}:{environment['PYTHONPATH']}"
    if job.get("phase") == "P4":
        for key in [k for k in environment if k.startswith("AWS_")]:
            del environment[key]
    return environment


def run_job(store: InkJobStore, job: dict, *, runs_root: Path, timeout: int) -> None:
    job_id, token = job["job_id"], job["lease_token"]
    output = runs_root / f"{job['sample_id'].lower()}-{job_id}"
    # Every phase is given a directory that exists. Some runners create their
    # own -- P5's does -- and some write a file into it and expect it to be
    # there: vet_map died on FileNotFoundError for its verdict, having read the
    # map, screened it and found the shapes, because nothing had made the
    # directory it was told to write into.
    output.mkdir(parents=True, exist_ok=True)
    rendered_from: dict[str, str] | None = None
    try:
        if job.get("phase") == "P4" and job["parameters"].get("flattened_surface"):
            # vc_render_tifxyz writes beside the segmentation it was given, so
            # the sheet is staged under this job's own output directory and the
            # rendered layers land with it.
            sheet = output / "flattened-surface"
            sheet.parent.mkdir(parents=True, exist_ok=True)
            job["parameters"]["segmentation"] = resolve_flattened_surface(
                store, job, sheet)
            rendered_from = {
                "kind": "flattened_sheet",
                "surface_id": str(job["parameters"]["flattened_surface"]),
                "profile_id": str(job["parameters"].get("flattening_profile")
                                  or DEFAULT_FLATTENING_PROFILE),
                # The one caveat worth carrying: a flattened sheet is a
                # resampling of the surface, so this stack is one interpolation
                # further from the scan than one rendered off the raw tifxyz.
                "non_claim": ("rendered on the flattened parametrisation, which "
                              "is resampled from the certified surface"),
            }
        elif job.get("phase") == "P4":
            rendered_from = {"kind": "surface_tifxyz",
                             "path": str(job["parameters"].get("segmentation"))}
        elif job.get("phase") == "P7" and job["parameters"].get("screening_of"):
            job["parameters"]["map_path"] = resolve_screened_map(
                store, job, runs_root / f"{job_id}-map")
            rendered_from = {"kind": "probability_map",
                             "screened_by": str(job["parameters"]["screening_of"]),
                             "non_claim": ("a screen is a verdict about shape, "
                                           "not a reading")}
        elif job.get("phase") == "P9" and job["parameters"].get("ordering_of"):
            job["parameters"]["order_path"] = resolve_wrap_order(store, job)
            rendered_from = {
                "kind": "measured_wrap_order",
                "ordered_by": str(job["parameters"]["ordering_of"]),
                "non_claim": ("a radial ordering composes plates; it does not "
                              "establish a reading"),
            }
        elif job.get("phase") == "P5" and job["parameters"].get("layer_stack"):
            # Beside the output directory, not inside it. P4's renderer writes
            # next to the surface it was given, so staging under the output is
            # right there; P5's runner owns its output and refuses to start if
            # anything is in it -- "refusing to overwrite non-empty output",
            # which is the correct behaviour meeting the wrong staging path.
            stack = runs_root / f"{job_id}-input"
            job["parameters"]["tiff_dir"] = resolve_layer_stack(store, job, stack)
            if not job["parameters"].get("source_slice_um"):
                pitch = slice_pitch_from_render(
                    job, store.job(str(job["parameters"]["layer_stack"])) or {})
                if pitch:
                    job["parameters"]["source_slice_um"] = pitch
            fit_depth_to_stack(job, stack)
            rendered_from = {"kind": "layer_stack",
                             "rendered_by": str(job["parameters"]["layer_stack"]),
                             "non_claim": ("a probability map is a screening "
                                           "output, not a reading")}
        argv = command_for(job, runner=str(runner_for(job)), output_dir=str(output))
    except Exception as exc:  # noqa: BLE001 -- a bad job must not stop the worker
        store.finish(job_id, token, state="failed",
                     result={"error": f"{type(exc).__name__}: {exc}"})
        return

    stop = threading.Event()

    def beat() -> None:
        # A long inference must not look abandoned. The lease is renewed while
        # the process lives, and stops the moment it does not.
        while not stop.wait(120):
            try:
                store.heartbeat(job_id, token)
            except RuntimeError:
                return

    store.mark_running(job_id, token, output_dir=str(output), command=argv)
    if rendered_from is not None:
        # On the record before the render starts. A layer stack whose provenance
        # is inferred from a path in an argv is a layer stack nobody can say the
        # geometry of afterwards.
        store.note(job_id, "rendered_from", rendered_from)
    heart = threading.Thread(target=beat, daemon=True)
    heart.start()

    started = time.time()
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
            env=runner_environment(job))
        tail = (completed.stdout or "")[-4000:]
        errors = (completed.stderr or "")[-4000:]
        # Whichever receipt this adapter writes. This was hardcoded to
        # run_ink.py's name, so the other three lanes could have computed a
        # liveness verdict and the worker would still have stored null -- the
        # file it looked for was never there.
        receipt = None
        for name in receipt_names(job):
            candidate = output / name
            if candidate.exists():
                receipt = json.loads(candidate.read_text())
                break
        evidence_publication = None
        publication_path = output / "EVIDENCE_PUBLICATION.json"
        if publication_path.exists():
            evidence_publication = json.loads(publication_path.read_text())
        result = {
            "exit_code": completed.returncode,
            "runtime_seconds": round(time.time() - started, 1),
            "stdout_tail": tail,
            "stderr_tail": errors,
            "statistics": (receipt or {}).get("statistics"),
            "liveness": (receipt or {}).get("liveness"),
            "output_dir": str(output),
        }
        # Exit 3 is the runner refusing a map that carries no decision. That is
        # a real outcome of the job, not a crash, and it is recorded as such.
        state = "succeeded" if completed.returncode == 0 else "failed"
        if state == "succeeded":
            try:
                result.update(merge_result_from_receipt(
                    job, receipt, evidence_publication))
                if job.get("phase") == "P7":
                    result["adjudication"] = read_adjudication(output)
                if job.get("phase") == "P9":
                    result["plate_set"] = verify_plate_set(job)
            except RuntimeError as failure:
                state = "failed"
                result["error"] = f"{type(failure).__name__}: {failure}"
        if completed.returncode == 3:
            result["refused"] = "DEGENERATE map: the lane produced no decision"
        # A P5 job with no verdict at all is not a pass. Storing null read
        # exactly like a lane that had been checked and found alive, which is
        # how three adapters went years without the gate and nobody saw it.
        if state == "succeeded" and job.get("phase") == "P5" and not result["liveness"]:
            state = "failed"
            result["refused"] = (
                "the lane recorded no liveness verdict, so nothing established "
                "that its map carries a decision. An unchecked map and a live "
                "one must not look the same.")
        if state == "succeeded" and job.get("phase") == "P4":
            # Verified and published before it is called a success. A stack that
            # exists only here is lost with this machine, and this machine is
            # meant to be disposable.
            try:
                layers = rendered_layers_directory(job, output)
                result["layers"] = verify_layer_stack(layers, job["parameters"])
                store_spec = job["parameters"].get("artifact_store")
                if store_spec:
                    result["layer_stack"] = publish_layer_stack(
                        layers, store_spec=str(store_spec),
                        sample_id=str(job["sample_id"]), job_id=job_id)
                    # And then let it go. A stack is ~50 MB and a worker renders
                    # many; without this the volume fills and takes down whatever
                    # else lives on it. Once the bytes are in object storage with
                    # a digest, the copy here is the disposable one. The receipt,
                    # the logs and the fetched sheet stay: they are small and
                    # they are the record.
                    if not job["parameters"].get("keep_local_layers"):
                        shutil.rmtree(layers, ignore_errors=True)
                        result["local_layers_removed"] = True
                elif not job["parameters"].get("allow_local_layers"):
                    raise RenderNotUsable(
                        "no artifact_store for this render: the stack would exist "
                        "only on this worker, and a worker is disposable. Set "
                        "CX_RENDER_STORE on the panel, or pass allow_local_layers "
                        "for a deliberate single-machine run")
            except Exception as failure:  # noqa: BLE001
                state = "failed"
                result["error"] = f"{type(failure).__name__}: {failure}"
        if state == "succeeded" and job.get("phase") == "P5":
            store_spec = job["parameters"].get("artifact_store")
            if store_spec:
                try:
                    published = publish_probability_map(
                        output, store_spec=str(store_spec),
                        sample_id=str(job["sample_id"]), job_id=job_id)
                    if published:
                        result["probability_map"] = published
                except Exception as failure:  # noqa: BLE001
                    # P7 may be claimed by another worker, so a local map is not
                    # a completed fleet result.  Fail honestly and let the job's
                    # retry policy decide whether publication is attempted again.
                    state = "failed"
                    result["error"] = f"{type(failure).__name__}: {failure}"
        if state == "succeeded":
            artifact_output = (Path(job["parameters"]["out_dir"])
                               if job.get("phase") == "P9" else output)
            registered = record_artifact(job, artifact_output, runs_root=runs_root)
            if registered:
                result["artifact_id"] = registered["artifact_id"]
                result["consumed"] = registered["inputs"]
        store.finish(job_id, token, state=state, result=result)
    except subprocess.TimeoutExpired:
        store.finish(job_id, token, state="failed",
                     result={"error": f"timed out after {timeout}s", "output_dir": str(output)})
    except Exception as exc:  # noqa: BLE001 -- a worker must not die on one job
        store.finish(job_id, token, state="failed",
                     result={"error": f"{type(exc).__name__}: {exc}", "output_dir": str(output)})
    finally:
        stop.set()
        # The fetched stack is a copy of something in object storage; keeping it
        # is how a worker's disk fills up.
        shutil.rmtree(runs_root / f"{job_id}-input", ignore_errors=True)
        shutil.rmtree(runs_root / f"{job_id}-map", ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host-id", default=socket.gethostname())
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--runs-root", type=Path, default=Path("/srv/helena/runs"))
    ap.add_argument("--dsn", default=os.environ.get("CX_DB", ""))
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--lease-seconds", type=int, default=3600)
    ap.add_argument("--job-timeout", type=int, default=21600)
    ap.add_argument("--once", action="store_true", help="claim at most one job and exit")
    # What this runtime can actually run. The ink image has no vc_flatten, so a
    # worker there must not claim P3; the segment image has the whole VC3D
    # toolchain and runs P2 and P3. Unset means every phase, which is what a
    # single-runtime deployment wants.
    ap.add_argument("--phases", default=os.environ.get("HELENA_WORKER_PHASES", ""),
                    help="comma-separated phases this worker may claim, e.g. P2,P3")
    args = ap.parse_args()

    if not args.dsn:
        print("no DSN: pass --dsn or set CX_DB", file=sys.stderr)
        return 2
    phases = [p.strip().upper() for p in args.phases.split(",") if p.strip()] or None
    worker_id = args.worker_id or f"{args.host_id}-{os.getpid()}"
    store = InkJobStore(args.dsn)
    store.initialize()
    args.runs_root.mkdir(parents=True, exist_ok=True)

    print(f"ink worker {worker_id} on {args.host_id}, runs -> {args.runs_root}, "
          f"phases: {','.join(phases) if phases else 'all'}", flush=True)
    last_state = 0.0
    # Probed once before the first claim, not only on the heartbeat: a worker
    # that claimed before its first probe would have no cards recorded and
    # would refuse every GPU job for the first minute of its life.
    last_probe = host_state(args.runs_root)
    while True:
        if time.time() - last_state > 60:
            try:
                last_probe = host_state(args.runs_root)
                store.record_host_state(args.host_id, last_probe)
            except Exception as exc:  # noqa: BLE001
                print(f"host state not recorded: {exc}", file=sys.stderr, flush=True)
            last_state = time.time()

        try:
            # What this machine actually has, measured rather than assumed.
            # A host with no card must not take a job that needs one: it fails
            # it, burns an attempt, and leaves the queue looking broken instead
            # of misrouted.
            cards = (last_probe or {}).get("gpus") or []
            job = store.claim(
                worker_id=worker_id, host_id=args.host_id,
                lease_seconds=args.lease_seconds, phases=phases,
                has_gpu=bool(cards),
                gpu_vram_gb=max((c.get("total_mb", 0) / 1024 for c in cards),
                                default=0.0))
        except Exception as exc:  # noqa: BLE001 -- the database may blink
            print(f"claim failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(args.poll_seconds)
            continue

        if job is None:
            if args.once:
                print("nothing to claim", flush=True)
                return 0
            time.sleep(args.poll_seconds)
            continue

        label = job.get("profile_id") or job.get("component") or "-"
        print(f"claimed {job['job_id']} [{job.get('phase', 'P5')}] "
              f"({job['sample_id']} / {label})", flush=True)
        run_job(store, job, runs_root=args.runs_root, timeout=args.job_timeout)
        print(f"finished {job['job_id']}", flush=True)
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
