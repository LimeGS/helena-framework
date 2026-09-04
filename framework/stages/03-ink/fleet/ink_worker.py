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
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from job_store import (  # noqa: E402
    INK_ADAPTERS, PHASE_RUNNERS, READ_PATH_PARAMETERS, WRITE_PATH_PARAMETERS,
    InkJobStore, command_for,
    runtime_image_for, depth_centers_that_fit, ink_adapter, ink_profile_path,
    lane_for,
)
from framework.contracts.slice_order import (  # noqa: E402
    SliceOrderError, ordered_tiff_files,
)


CONTROL_BINDING_FIELDS = (
    "control_p0_artifact_id",
    "control_p0_artifact_sha256",
    "control_p0_selection_version",
    "control_source_snapshot_id",
    "control_source_content_lock",
    "control_source_content_lock_sha256",
    "control_policy_sha256",
)


def persisted_control_binding(job: dict) -> dict | None:
    """Return one complete immutable server claim, or fail closed."""
    parameters = job.get("parameters") or {}
    present = [field for field in CONTROL_BINDING_FIELDS if field in parameters]
    if not present:
        return None
    if len(present) != len(CONTROL_BINDING_FIELDS):
        raise RuntimeError("job has a partial persisted control binding")
    binding = {field: parameters[field] for field in CONTROL_BINDING_FIELDS}
    content_lock = binding["control_source_content_lock"]
    if not isinstance(content_lock, dict):
        raise RuntimeError("job has a tampered persisted control binding")
    content_lock_sha = hashlib.sha256(json.dumps(
        content_lock, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    hashes = (
        binding["control_p0_artifact_sha256"],
        binding["control_source_content_lock_sha256"],
        binding["control_policy_sha256"],
    )
    if (not all(isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes)
            or not all(isinstance(binding[field], str) and binding[field]
                       for field in ("control_p0_artifact_id",
                                     "control_p0_selection_version",
                                     "control_source_snapshot_id"))
            or binding["control_source_content_lock_sha256"] != content_lock_sha):
        raise RuntimeError("job has a tampered persisted control binding")
    return binding


def verified_control_binding(job: dict) -> dict | None:
    """Bind worker control behavior to the exact policy hash it advertises."""
    binding = persisted_control_binding(job)
    if binding is None:
        return None
    policy_path = (
        ROOT / "framework/profiles/01-segmentation/"
        "first-letters-control-policy-1.3.0.json"
    )
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"worker cannot load bound control policy: {exc}") from exc

    def canonical_sha256(document: dict) -> str:
        return hashlib.sha256(json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")).hexdigest()

    locks = policy.get("source_locks") or {}
    expected_lock = {
        "control_profile_id": policy.get("profile_id"),
        "control_profile_sha256": canonical_sha256(policy),
        "ct_lock_sha256": canonical_sha256(locks.get("ct") or {}),
        "m7_lock_sha256": canonical_sha256(locks.get("m7") or {}),
    }
    if (binding["control_policy_sha256"] != canonical_sha256(policy)
            or binding["control_source_content_lock"] != expected_lock):
        raise RuntimeError("worker control policy differs from persisted binding")
    job["_verified_control_policy"] = policy
    return binding


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


def merged_surface_routing_receipt(receipt: dict) -> dict:
    """The routing decision the merged surface earns on its own measurement.

    A merge produces a *new* surface.  It has its own extent -- the merge lane
    measures it with ``fleet.finalizer.inspect_tifxyz`` over the merged x/y/z
    grids it just published, under the same frozen triangulation every other
    surface is measured with -- so it has its own routing question, and the
    answer is neither inherited from a parent nor implied by the merge having
    passed.  Two parents that each clear the floor can be stitched into a strip
    that does not, and the floor is a statement about how much papyrus there is
    to ask about, which stitching does not change.

    Decided here rather than read out of the merge receipt.  A route carried in
    a document is a claim; the area beside it is the evidence, and where the two
    disagree the document does not win.  This is the contract the stores already
    hold to in ``agrees_with_measurement``: re-decide, then require the carried
    receipt to be the receipt this area produces.  A merge receipt that records
    no usable area is refused, because every alternative -- defaulting to
    STANDARD, defaulting to DIAGNOSTIC, or omitting the decision the way this
    function used to -- is an answer the measurement did not give.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet import surface_routing  # noqa: PLC0415

    surface_id = str(receipt["surface_id"])
    policy = surface_routing.load_policy()
    try:
        decided = surface_routing.build_receipt(
            surface_id=surface_id, area_cm2=receipt.get("area_cm2"),
            policy=policy,
            measurement={"decided_at": "P8_MERGE_RESULT_PROJECTION"},
            read_set={"artifact_sha256": receipt.get("artifact_sha256")},
        )
    except ValueError as unmeasured:
        raise RuntimeError(
            f"P8 merge receipt records no usable area for {surface_id}, so the "
            f"surface the merge produced cannot be routed: {unmeasured}"
        ) from unmeasured

    carried = receipt.get("routing_receipt")
    if carried is None:
        return decided
    # The lane's own receipt is the one the catalogue stores, so it is the one
    # to report -- but only after it has been checked against the measurement
    # rather than trusted for having a digest.
    if not surface_routing.agrees_with_measurement(
        carried, receipt.get("area_cm2"), policy=policy,
    ) or carried.get("surface_id") != surface_id:
        raise RuntimeError(
            f"P8 merge receipt carries a routing receipt for {surface_id} that "
            "is not the decision its own measured area produces")
    return carried


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

    The route travels with them.  A merged sheet under the effort floor used to
    reach this result with no area and no routing decision at all, which is the
    door PHerc0268 walked through on the ink screen wearing different clothes:
    downstream reads a successful P8 result and finds nothing in it saying the
    standard path was never available to the surface it names.
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
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet import surface_routing  # noqa: PLC0415

    routing_receipt = merged_surface_routing_receipt(receipt)
    return {
        "merge_receipt": receipt,
        "evidence_publication": publication,
        "surface_id": receipt["surface_id"],
        "artifact_uri": receipt["artifact_uri"],
        "artifact_sha256": receipt["artifact_sha256"],
        "evidence_uri": publication["evidence_uri"],
        "evidence_sha256": publication["evidence_sha256"],
        "parents": receipt["parents"],
        "area_cm2": routing_receipt["measured_area_cm2"],
        # Read out of a receipt that has just been verified, never copied from
        # a route string somebody wrote down.  The two admissibility answers
        # come from the router's own predicates for the same reason: a forged
        # or corrupted receipt then fails exactly the way a missing one does.
        "route": routing_receipt["route"],
        "routing_receipt": routing_receipt,
        "enters_standard_qc": surface_routing.enters_standard_qc(routing_receipt),
        "enters_canonical_downstream":
            surface_routing.enters_canonical_downstream(routing_receipt),
    }


# Which image this worker is. Set by the compose that runs it; None on a host
# that never said, where this check cannot prove anything and stays quiet.
RUNTIME_IMAGE = os.environ.get("HELENA_RUNTIME_IMAGE") or None


def misnamed_runtime(runtime: str | None) -> str | None:
    """Why this worker's declared runtime cannot be the one it says, or None.

    HELENA_RUNTIME_IMAGE names the *lane* image a worker carries, not the
    composed image it runs as -- `helena-ink-9um`, not
    `helena-ink-9um-worker`. Getting it the other way round is quiet in the
    worst way: the worker starts, claims nothing that needs the lane, and looks
    exactly like a worker with nothing to do.

    The sibling mistake is loud and the lane image now explains it: pointing
    HELENA_INK_IMAGE at the lane gives a container with no repository in it.
    These are the same confusion, one suffix apart, in an invocation that sets
    both variables to nearly the same string.
    """
    name = (runtime or "").strip()
    if not name or not name.endswith("-worker"):
        return None
    from job_store import lane_runtime_images  # noqa: PLC0415

    lane = name[: -len("-worker")]
    if lane not in lane_runtime_images():
        return None
    return (
        f"HELENA_RUNTIME_IMAGE is {name!r}, which is the composed worker "
        f"image. It has to name the lane image that worker carries: {lane!r}. "
        "As it stands this worker will refuse every job for that lane by name "
        "and go on looking idle.")


def worker_code_revision() -> dict[str, str | None]:
    """Which build of this worker ran a job, recorded on every result.

    Four P4 jobs were finished `succeeded` with zero layers written, by a
    worker whose copy of this file predated the check that refuses exactly
    that. Nothing in the row said which build had run, so "the image is stale"
    was an inference from a second symptom rather than something the receipt
    could answer.

    The digest of this file is the part that usually works: the build arguments
    are only as true as the image that carries them, and a bind mount over the
    checkout has neither. It is None on a worker that cannot read itself.

    On the result of a job that *ran*. A job refused before the subprocess
    starts -- an unreadable input, a write outside its run directory, a missing
    upstream directory, a lineage refusal -- finishes through a path that
    records the refusal and nothing else. Do not promise a reader that every
    row carries this.
    """
    try:
        source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - a worker that cannot read itself
        source = None
    return {
        "image": RUNTIME_IMAGE,
        "build_revision": os.environ.get("BUILD_REVISION") or None,
        "worker_source_sha256": source,
    }


# How often a running job reports in. It was 120 s, which is the right cadence
# for renewing an hour-long lease and the wrong one for saying what a job is
# doing -- a twenty-six-minute render checked in thirteen times and said
# nothing on any of them. The write happens anyway; the progress rides it.
HEARTBEAT_SECONDS = 15
# How much of a line survives into the control plane. A progress bar is short;
# a traceback line can be arbitrarily long, and this column is read on every
# poll of the queue.
PROGRESS_LINE_CHARS = 400


def split_progress(buffer: str) -> tuple[list[str], str]:
    """Cut a chunk into finished lines and whatever is still half-written.

    On carriage returns as well as newlines, because that is how tqdm draws a
    bar: a reader that waits for a newline sees the whole thing at once, when
    the process ends, which is the one moment it is worth nothing.

    The remainder is held rather than emitted. Half a line reported as a whole
    one is a claim the process did not make.
    """
    pieces = re.split(r"[\r\n]", buffer)
    remainder = pieces.pop()
    return [piece for piece in pieces if piece], remainder


def run_streaming(argv: list[str], *, timeout: float | None, env: dict | None,
                  on_line, on_start=None) -> subprocess.CompletedProcess:
    """Run a child, echoing what it writes as it writes it.

    `capture_output=True` buffers both pipes until the process exits, so a long
    job is observable only as "started" and, much later, "finished" -- nothing
    in the host's logs and nothing in the control plane while it runs. Watching
    one meant reading the size of a file on disk and the GPU's utilisation,
    neither of which is progress.

    The return shape is subprocess.run's, deliberately: the receipt, the P3
    lineage parse and the failure payload all read `completed.stdout` and
    `completed.stderr`, and they need the two whole and separate. What is
    collected is the normalised lines rather than the raw bytes, which is what
    `text=True` used to do and what keeps a tail of progress bar readable.
    """
    process = subprocess.Popen(  # noqa: S603 - argv is built by command_for
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    # Handed out so a cancellation can reach it. Without this the only way to
    # stop a running job was to kill the whole worker.
    if on_start is not None:
        on_start(process)
    collected: dict[str, list[str]] = {"stdout": [], "stderr": []}

    def pump(name: str, pipe, echo) -> None:
        rest = ""
        try:
            while True:
                chunk = os.read(pipe.fileno(), 65536)
                if not chunk:
                    break
                lines, rest = split_progress(
                    rest + chunk.decode("utf-8", "replace"))
                for line in lines:
                    collected[name].append(line + "\n")
                    # The host's own log first: it is the copy that survives a
                    # control plane nobody can reach.
                    print(line, file=echo, flush=True)
                    try:
                        on_line(name, line)
                    except Exception:  # noqa: BLE001, S110
                        # Reporting progress must not be able to kill the job
                        # whose progress it is.
                        pass
        except (OSError, ValueError):
            return
        finally:
            if rest:
                collected[name].append(rest)

    threads = [
        threading.Thread(target=pump, args=("stdout", process.stdout, sys.stdout),
                         daemon=True),
        threading.Thread(target=pump, args=("stderr", process.stderr, sys.stderr),
                         daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Killed, not abandoned. A job whose process outlives its worker still
        # holds the card it was using.
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=5)
        raise
    for thread in threads:
        thread.join(timeout=30)
    return subprocess.CompletedProcess(
        argv, process.returncode,
        "".join(collected["stdout"]), "".join(collected["stderr"]))


def require_runtime(job: dict) -> None:
    """Refuse a lane whose runtime this worker is not. The lease and the attempt are already spent when this fires; the pre-claim skip is the cheap path.

    Three lanes now need an image their claiming worker does not run -- the
    9 um detector, lasagna, the spiral fitter -- because their dependencies
    cannot share one environment with the platform's. Without this the worker
    runs their argv in its own runtime, the runner is not importable, and the
    job dies several minutes in reporting whatever the traceback said rather
    than the one fact that explains it.

    Quiet when this worker does not know its own image: a host that was
    configured correctly but never labelled would otherwise have every one of
    those jobs refused, which is worse than the failure it prevents.
    """
    if RUNTIME_IMAGE is None:
        return
    needed = runtime_image_for(job)
    if needed and needed != RUNTIME_IMAGE:
        raise RuntimeError(
            f"this lane runs in {needed} and this worker is {RUNTIME_IMAGE}. "
            f"The lane declares its image because its dependencies cannot "
            f"share an environment with this one -- route the job to a worker "
            f"carrying {needed}.")


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
from framework.contracts.host_probe import (  # noqa: E402,F401
    cpu_and_memory, host_state, local_images,
)


def worker_gpu_visible(*, nvidia_smi: str = "nvidia-smi") -> bool | None:
    """Whether *this worker's own* visible device answers, asked fresh.

    helena-ink-0 kept polling for five hours with nvidia-smi inside the
    container reporting "No devices were found" -- a passthrough glitch, not a
    crash -- while helena-ink-9um, on the same host and carrying the identical
    DeviceRequests, kept seeing its card the whole time. The worker's own "do I
    have a GPU" answer had been decided once, at startup, and never asked
    again: six P5 jobs sat pending for five hours with nothing in any log --
    the process never died, `docker ps` said "Up", and the fleet row said
    POLLING, which is what a worker with nothing to claim looks like too.
    `docker restart` was the whole fix, because nothing had actually broken
    that a restart could not re-probe.

    host_state() cannot stand in for this. It strips CUDA_VISIBLE_DEVICES and
    NVIDIA_VISIBLE_DEVICES on purpose, because it wants the whole host's
    inventory for the Hosts page -- the question there is "what does this
    machine have". The question here is narrower and is the one that was
    missing: "can this process, right now, reach the device it was given". So
    those variables are kept rather than stripped, and this is called every
    poll rather than once a minute, so a lost device is caught within one
    polling interval instead of riding out however long it takes someone to
    notice a queue has stopped draining.

    Three answers, not two. `None` is a worker that has never claimed a GPU at
    all -- nvidia-smi is not even on its PATH, which is what a CPU-only ink
    worker looks like, if one is ever deployed. `False` is a worker that does
    claim one and cannot currently reach it: nvidia-smi is present and either
    ran and found nothing, or failed outright. `True` is the only proof that
    counts -- a device nvidia-smi listed just now. Config-level claims --
    DeviceRequests, an env var, a driver that answered a heartbeat ago -- are
    not asked here at all, because the whole point of asking again is that
    exactly those claims can lie.
    """
    try:
        probed = subprocess.run(
            [nvidia_smi, "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return None
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    if probed.returncode != 0:
        return False
    return bool(probed.stdout.strip())


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
    # Renamed from scroll3-chunk-gather when the renderer stopped being about
    # one scroll. A job queued under the old id keeps working: it is the lane
    # that changed, not where it writes.
    if parameters.get("lane") in ("chunk-gather", "scroll3-chunk-gather"):
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

    try:
        slices, slice_ordering = ordered_tiff_files(
            Path(directory), require_numeric=True, require_contiguous=True)
    except SliceOrderError as exc:
        raise RenderNotUsable(str(exc)) from exc
    expected = parameters.get("num_slices")
    if expected and len(slices) != int(expected):
        raise RenderNotUsable(
            f"asked for {int(expected)} slices and found {len(slices)}")
    indices = [int(path.stem) for path in slices]
    required_indices = list(range(int(expected) if expected else len(slices)))
    if indices != required_indices:
        raise RenderNotUsable(
            f"numeric TIFF indices must be exactly {required_indices[0]}.."
            f"{required_indices[-1]}, found {indices}")
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
        "slice_indices": indices,
        "slice_filenames": [path.name for path in slices],
        "slice_ordering": slice_ordering,
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
            "manifest_sha256": content_sha256(manifest),
            "files": len(files),
            "objects": [
                {"object_key": name, "sha256": files[name]["sha256"],
                 "bytes": files[name]["size_bytes"]}
                for name in names
            ]}


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
    try:
        paths, slice_ordering = ordered_tiff_files(
            Path(directory), require_numeric=True, require_contiguous=True)
    except SliceOrderError as exc:
        raise RenderNotUsable(str(exc)) from exc
    indices = [int(path.stem) for path in paths]
    if indices != list(range(len(paths))):
        raise RenderNotUsable(
            f"numeric TIFF indices must start at 0, found {indices}")
    published = publish_artifact_set(
        directory, [path.name for path in paths],
        schema="campaignx.layer_stack_artifact_set.v1",
        store_spec=store_spec, sample_id=sample_id,
        key=f"layers/{job_id}", job_id=job_id)
    return {**published, "slice_indices": indices,
            "slice_ordering": slice_ordering}


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
    lineage_guard = getattr(store, "require_surface_lineage", None)
    if callable(lineage_guard):
        lineage_guard(
            surface_id=surface_id, mission_id=job.get("mission_id"),
            boundary="P4_EXECUTION_RESOLUTION", allow_unvalidated=False,
        )
    sheet = store.flattened_sheet(surface_id, profile_id)
    expected_id = str(job["parameters"].get("flattening_id") or "")
    expected_job = str(job["parameters"].get("p3_job_id") or "")
    expected_sha = str(job["parameters"].get("flattened_artifact_sha256") or "")
    if (not expected_id or sheet.get("flattening_id") != expected_id
            or not expected_job or sheet.get("requested_by_job_id") != expected_job
            or not expected_sha or sheet.get("artifact_sha256") != expected_sha):
        raise RuntimeError(
            "flattened sheet does not match the exact P3 artifact/job identity")
    job["_flattened_sheet"] = {
        "artifact_id": sheet.get("flattening_id"),
        "surface_id": surface_id,
        "p3_job_id": sheet.get("requested_by_job_id"),
        "artifact_sha256": sheet.get("artifact_sha256"),
        "artifact_uri": sheet.get("artifact_uri"),
    }

    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.certifier import load_qc_adapter  # noqa: PLC0415

    load_qc_adapter().materialize_surface(
        str(sheet["artifact_uri"]), str(sheet["artifact_sha256"] or ""), destination)
    return str(destination)


def measure_p3_p4_lateral_metric(
        sheet: Path, layer_shape_yx: list[int], *, source_voxel_um: float,
        lineage: dict, policy: dict) -> dict:
    """Measure the exact one-grid-cell P3→P4 raster transform in physical units."""
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    base = {"schema": "campaignx.first_letters_p3_p4_lateral_metric.v1",
            "profile_id": policy.get("profile_id"),
            "profile_sha256": hashlib.sha256(json.dumps(
                policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False).encode()).hexdigest(),
            "lineage": lineage, "policy": policy,
            "source_voxel_um": source_voxel_um}
    def finish(status: str, reason: str, **values):
        receipt = {**base, **values, "status": status, "reason_code": reason}
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
        return receipt
    arrays = [np.asarray(tifffile.imread(Path(sheet) / f"{axis}.tif"),
                         dtype=np.float64) for axis in "xyz"]
    if len({array.shape for array in arrays}) != 1 or arrays[0].ndim != 2:
        return finish("UNPROVEN", "P3_TIFXYZ_GRID_INVALID")
    shape = list(arrays[0].shape)
    if shape != list(layer_shape_yx):
        return finish("UNPROVEN", "P4_RASTER_DIMENSION_MISMATCH",
                      tifxyz_shape_yx=shape, layer_shape_yx=list(layer_shape_yx))
    xyz = np.stack(arrays, axis=-1)
    valid = np.isfinite(xyz).all(axis=2) & (xyz >= 0).all(axis=2)
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.finalizer import triangulate_tifxyz_grid  # noqa: PLC0415
    mesh = triangulate_tifxyz_grid(xyz)
    possible_triangle_count = 2 * max(0, shape[0] - 1) * max(0, shape[1] - 1)
    valid_triangle_count = int(len(mesh["faces"]))
    triangle_coverage = valid_triangle_count / max(1, possible_triangle_count)
    masks = (valid[:, :-1] & valid[:, 1:], valid[:-1, :] & valid[1:, :])
    deltas = (xyz[:, 1:] - xyz[:, :-1], xyz[1:, :] - xyz[:-1, :])
    distances = [np.linalg.norm(delta, axis=2)[mask] * float(source_voxel_um)
                 for delta, mask in zip(deltas, masks, strict=True)]
    totals = [mask.size for mask in masks]
    valid_count = sum(len(values) for values in distances)
    edge_coverage = valid_count / max(1, sum(totals))
    if any(not len(values) for values in distances):
        return finish("UNPROVEN", "P3_P4_LATERAL_EDGES_MISSING",
                      valid_triangle_count=valid_triangle_count,
                      possible_triangle_count=possible_triangle_count,
                      valid_triangle_fraction=triangle_coverage,
                      valid_edge_fraction=edge_coverage)
    def stats(values):
        return {"count": int(len(values)), "min_um": float(np.min(values)),
                "median_um": float(np.median(values)),
                "p95_um": float(np.percentile(values, 95)),
                "max_um": float(np.max(values))}
    horizontal, vertical = map(stats, distances)
    medians = [horizontal["median_um"], vertical["median_um"]]
    all_values = np.concatenate(distances)
    lateral = float(np.median(all_values))
    distortion = max(horizontal["p95_um"], vertical["p95_um"]) / max(
        min(medians), np.finfo(float).eps)
    threshold = float(policy["maximum_uv_to_3d_distortion_ratio"])
    minimum_coverage = float(policy["minimum_valid_triangle_fraction"])
    proven = triangle_coverage >= minimum_coverage and distortion <= threshold
    return finish("PROVEN" if proven else "UNPROVEN",
                  "LATERAL_METRIC_PROVEN" if proven else "LATERAL_METRIC_POLICY_FAILED",
                  lateral_pixel_um=lateral,
                  valid_triangle_count=valid_triangle_count,
                  possible_triangle_count=possible_triangle_count,
                  valid_triangle_fraction=triangle_coverage,
                  valid_edge_fraction=edge_coverage,
                  observed_uv_to_3d_distortion_ratio=float(distortion),
                  raster_transform={"rule": "ONE_OUTPUT_PIXEL_PER_TIFXYZ_GRID_CELL",
                                    "shape_yx": shape},
                  measurements={"horizontal": horizontal, "vertical": vertical})


ARTIFACT_HTTP_TIMEOUT_SECONDS = 300


def _http_get(url: str, *, timeout: float = ARTIFACT_HTTP_TIMEOUT_SECONDS) -> bytes:
    """Read one published object, with a deadline.

    A read without one is how a job burns its whole lease and dies with nothing
    recorded -- no CPU, no output, no reason -- which is indistinguishable from
    a job nobody ever claimed.
    """

    request = urllib.request.Request(
        url, headers={"User-Agent": "Campaign-X-ink-worker/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


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
    from fleet.retrying import read_with_retry  # noqa: PLC0415

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "s3":
        import boto3  # noqa: PLC0415

        client = boto3.client("s3")
        bucket, base = parsed.netloc, parsed.path.strip("/")

        # `download_file` below retries on its own -- s3transfer carries
        # S3_RETRYABLE_DOWNLOAD_ERRORS -- but nothing covers this one. The body
        # is streamed after `get_object` has already returned, so botocore's
        # request-layer retry is behind it and s3transfer is not in front of it.
        # A disconnect here loses the whole fetch, and every file that would
        # have been verified against this manifest goes with it.
        def read_manifest() -> dict:
            return json.loads(client.get_object(
                Bucket=bucket, Key=f"{base}/ARTIFACT_SET.json")["Body"].read())

        manifest = read_with_retry(read_manifest)
        for name in manifest["files"]:
            client.download_file(bucket, f"{base}/{name}", str(destination / name))
    elif parsed.scheme in ("http", "https"):
        base = artifact_uri.rstrip("/")

        # Nothing sits in front of these the way s3transfer does for S3, so the
        # bounded retry that guards the manifest read above guards every object
        # here. It is bounded on purpose: a public host that is down should end
        # the job with a reason, not hold the lease until it expires.
        def fetch(name: str) -> bytes:
            return read_with_retry(lambda: _http_get(f"{base}/{name}"))

        manifest = json.loads(fetch("ARTIFACT_SET.json"))
        for name in manifest["files"]:
            (destination / name).write_bytes(fetch(name))
    elif parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        manifest = json.loads((source / "ARTIFACT_SET.json").read_text())
        for name in manifest["files"]:
            shutil.copy2(source / name, destination / name)
    elif "://" in artifact_uri:
        # Anything else must say which scheme it was. Falling through to the
        # local branch is what turned an `https://` URI into a lookup under the
        # worker's current directory: a URI the lane cannot read became a
        # filesystem accident. Keyed off `://` rather than a parsed scheme so a
        # relative path containing a colon stays a path.
        raise ValueError(
            f"artifact set URI scheme {parsed.scheme!r} cannot be read by this "
            f"lane: {artifact_uri}")
    else:
        source = Path(artifact_uri)
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


FLATTENING_RUN_SCHEMA = "campaignx.surface_flattening_run.v1"


def flattening_lineage(stdout: str, job_id: str) -> list[dict]:
    """What this job flattened, for its own terminal event.

    The control reads the job's `succeeded` payload as a second witness beside
    the flattening index: the index says a surface was flattened, this says
    *this job* flattened it. Without both, a row some other run wrote could
    satisfy this run's boundary. It refused the first control ever to reach P3
    with P3_CURRENT_JOB_EVIDENCE_MISSING, for want of a payload that carried
    nothing but the runner's exit code.

    Read off stdout because P3 has nowhere else to leave it: it works in a temp
    directory and publishes, so `output_dir` is empty and `receipt_names()`
    finds no file. Read off the whole of stdout rather than the stored 4000
    character tail, which one surface fits inside and several do not.

    Fails closed in both directions. A row naming another job is not adopted --
    a witness that can be borrowed is not a witness -- and a surface that did
    not flatten is not reported, because saying otherwise would claim the job
    produced something it did not. Anything unparseable is simply no evidence:
    every other phase runs through here, and none of them may be failed by the
    shape of their own logs.
    """
    start = stdout.find("{")
    if start < 0:
        return []
    try:
        receipt = json.loads(stdout[start:])
    except (ValueError, TypeError):
        return []
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != FLATTENING_RUN_SCHEMA):
        return []
    surfaces = receipt.get("surfaces")
    if not isinstance(surfaces, list):
        return []
    lineage = []
    for row in surfaces:
        if (not isinstance(row, dict)
                or row.get("requested_by_job_id") != job_id
                or row.get("state") != "FLATTENED"
                or not row.get("surface_id")
                or not row.get("receipt_sha256")
                or not row.get("artifact_sha256")):
            continue
        lineage.append({
            "surface_id": row["surface_id"],
            "requested_by_job_id": row["requested_by_job_id"],
            "receipt_sha256": row["receipt_sha256"],
            "artifact_sha256": row["artifact_sha256"],
            "artifact_uri": row.get("artifact_uri"),
            "profile_id": row.get("profile_id"),
            "state": row["state"],
            # The geometry orientation proof reads these off the same rows. The
            # first version of this carried only what the control harness
            # needed, so P3 passed and the very next boundary refused the same
            # job for "hash-bound flattened lineage" it could not see. They were
            # in the receipt all along.
            "source_artifact_sha256": row.get("source_artifact_sha256"),
            "profile_file_sha256": row.get("profile_file_sha256"),
            "artifact_id": row.get("artifact_id"),
            "flattening_id": row.get("flattening_id"),
        })
    return lineage


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
    binding = verified_control_binding(job)
    render_binding = persisted_control_binding(render)
    if binding is None and render_binding is not None:
        raise RuntimeError("P5 dropped its P4 persisted control binding")
    if binding is not None and render_binding != binding:
        raise RuntimeError("P5/P4 persisted control bindings disagree")
    if (binding is not None
            and persisted_control_binding({"parameters": render.get("result") or {}})
            != binding):
        raise RuntimeError("P4 result lacks its persisted control binding")
    published = (render.get("result") or {}).get("layer_stack") or {}
    if not published.get("artifact_uri"):
        raise RuntimeError(
            f"render {rendered_by} published no layer stack, so there is nothing "
            "to read on this machine")
    manifest = fetch_artifact_set(str(published["artifact_uri"]), destination)
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import content_sha256  # noqa: PLC0415

    try:
        paths, _ordering = ordered_tiff_files(
            destination, require_numeric=True, require_contiguous=True)
    except SliceOrderError as exc:
        raise RuntimeError(str(exc)) from exc
    objects = [
        {"object_key": path.name,
         "sha256": manifest["files"][path.name]["sha256"],
         "bytes": manifest["files"][path.name]["size_bytes"]}
        for path in paths
    ]
    artifact_sha = content_sha256(manifest.get("files") or {})
    manifest_sha = content_sha256(manifest)
    if (manifest.get("artifact_sha256") not in (None, artifact_sha)
            or published.get("artifact_sha256") not in (None, artifact_sha)):
        raise RuntimeError("the fetched layer manifest does not match the P4 artifact digest")
    if (binding is not None
            and (manifest.get("schema") != "campaignx.layer_stack_artifact_set.v1"
                 or manifest.get("job_id") != rendered_by
                 or published.get("manifest_sha256") != manifest_sha)):
        raise RuntimeError(
            "the fetched layer manifest digest does not match the exact P4 result")
    if published.get("objects") not in (None, objects):
        raise RuntimeError("the fetched layer objects do not match the P4 result inventory")
    if binding is not None:
        indices = [int(path.stem) for path in paths]
        if indices != list(range(33)) or published.get("files") != 33:
            raise RuntimeError("the First Letters P5 input is not exact slices 0..32")
    job["_source_layer_stack"] = {
        "schema": "campaignx.p5_source_layer_stack.v1",
        "p4_job_id": rendered_by,
        "artifact_uri": str(published["artifact_uri"]),
        "artifact_sha256": artifact_sha,
        "manifest_sha256": manifest_sha,
        "objects": objects,
    }
    metric = (render.get("result") or {}).get("lateral_metric") or {}
    if binding is not None:
        unhashed_metric = {key: value for key, value in metric.items()
                           if key != "receipt_sha256"}
        metric_sha = hashlib.sha256(json.dumps(
            unhashed_metric, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        lateral = metric.get("lateral_pixel_um")
        pitch = slice_pitch_from_render(job, render)
        if (metric.get("status") != "PROVEN"
                or metric.get("receipt_sha256") != metric_sha
                or not isinstance(lateral, (int, float))
                or not isinstance(pitch, (int, float))):
            raise RuntimeError("the First Letters P4 lateral metric is unproven")
        supplied_lateral = job["parameters"].get("source_pixel_um")
        supplied_pitch = job["parameters"].get("source_slice_um")
        if supplied_lateral not in (None, lateral) or supplied_pitch not in (None, pitch):
            raise RuntimeError("P5 physical spacing differs from the bound P4 evidence")
        job["parameters"]["source_pixel_um"] = float(lateral)
        job["parameters"]["source_slice_um"] = float(pitch)
        job["_source_lateral_metric"] = metric
    return str(destination)


def slice_pitch_from_render(job: dict, render: dict) -> float | None:
    """The depth pitch of a stack, from the render that produced it.

    The TimeSformer adapter refuses to default this, and it is right to: the
    campaign spans 8.64 and 9.362 um acquisitions and assuming either rescales
    the other by 8.4% in silence. But it is not a guess when the render is
    named -- P4 recorded the scale it sampled at and the step it took along the
    normal, and the pitch follows:

        pitch = source_voxel_um * scale * slice_step

    Lateral pixel size is a separate P3-to-P4 geometry measurement and is never
    substituted into this depth formula. Derived rather than typed, because a
    number the platform already knows and asks a person to retype is a number
    that will eventually be retyped wrong.
    """
    parameters = render.get("parameters") or {}
    source_voxel_um = parameters.get("source_voxel_um")
    if source_voxel_um in (None, ""):
        return None
    try:
        scale = float(parameters.get("scale") or 1.0)
        step = float(parameters.get("slice_step") or 1.0)
        return float(source_voxel_um) * scale * step
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
        "verdict_sha256": hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
        "card_file": card_path.name,
        "card_sha256": hashlib.sha256(card_path.read_bytes()).hexdigest(),
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


def screened_window_grid_alarm(job: dict) -> dict:
    """Measure the screened window's dominant period, or say why it could not.

    Never raises: this is a note attached to somebody else's verdict, and a
    screening that ran must not fail because the note could not be written.
    """
    parameters = job.get("parameters") or {}
    try:
        import numpy as np  # noqa: PLC0415

        sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))
        from grid_alarm import grid_alarm  # noqa: PLC0415

        path = Path(str(parameters["map_path"]))
        values = np.load(path) if path.suffix == ".npy" else None
        if values is None:
            return {"alarm": False,
                    "reason": f"the screened map is not a .npy this can read: {path.name}"}
        x0, y0, x1, y1 = (int(part) for part in str(parameters["bbox"]).split(","))
        window = values[y0:y1, x0:x1]
        return grid_alarm(
            window,
            px_um=float(parameters["px_um"]),
            render_cell_px=(float(parameters["render_cell_px"])
                            if parameters.get("render_cell_px") else None))
    except Exception as failure:  # noqa: BLE001 -- see the docstring
        return {"alarm": False,
                "reason": f"not measured: {type(failure).__name__}: {failure}"}


def objects_by_key(rows) -> list:
    """One artifact set's objects, in an order both writers agree on.

    The published manifest sorts its files; a P5 result records them as they
    were published. So the receipt is first in one and last in the other, and
    comparing the two lists as sequences called that a manifest/content mismatch
    for every map -- `INK_SCREENING_RECEIPT.json` sorts before the `.npy` files
    it describes, and P7 refused an ALIVE map in one second on the order of a
    list.

    Nothing is loosened by sorting: the same keys, the same digests and the same
    sizes are still compared. Only the sequence stops being part of the claim.
    """
    return sorted((row for row in (rows or []) if isinstance(row, dict)),
                  key=lambda row: str(row.get("object_key")))


def resolve_screened_map(store: InkJobStore, job: dict, destination: Path) -> dict:
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
    binding = verified_control_binding(job)
    screening_binding = persisted_control_binding(screening)
    if binding is None and screening_binding is not None:
        raise RuntimeError("P7 dropped its P5 persisted control binding")
    if binding is not None and screening_binding != binding:
        raise RuntimeError("P7/P5 persisted control bindings disagree")
    if (binding is not None
            and persisted_control_binding({"parameters": screening.get("result") or {}})
            != binding):
        raise RuntimeError("P5 result lacks its persisted control binding")
    result = screening.get("result") or {}
    verdict = ((result.get("liveness") or {}).get("verdict"))
    if verdict != "ALIVE":
        # The gate P6 was written to be and never was: screening finds shapes in
        # noise perfectly well, so a map the lane could not read must not reach
        # an adjudication.
        raise RuntimeError(
            f"the map of {screening_id} is {verdict or 'unrecorded'}, not ALIVE: "
            "screening a map the lane could not read finds shapes in noise")
    probability_map = result.get("probability_map") or {}
    published = probability_map.get("artifact_uri")
    expected_artifact = str(job["parameters"].get(
        "probability_map_artifact_sha256") or "")
    expected_manifest = str(job["parameters"].get(
        "probability_map_manifest_sha256") or "")
    if not published or not expected_artifact or not expected_manifest:
        raise RuntimeError(
            f"screening {screening_id} lacks exact probability-map content binding")
    manifest = fetch_artifact_set(str(published), destination)
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet.common import content_sha256  # noqa: PLC0415
    objects = [
        {"object_key": name, "sha256": item.get("sha256"),
         "bytes": item.get("size_bytes")}
        for name, item in (manifest.get("files") or {}).items()
    ]

    # Compared by content, not by the order two writers happened to list it in.
    #
    # The manifest sorts its files and the P5 result records them in the order
    # they were published, so the receipt came first in one and last in the
    # other -- and the equality below called that a "manifest/content mismatch"
    # for every map either side listed differently. Which is every map: the
    # receipt's name sorts before the .npy files it describes.
    #
    # Nothing is loosened. The same keys, the same digests and the same sizes
    # are still required; only the sequence stops being part of the claim.
    if (manifest.get("schema") != "campaignx.ink_probability_map.v1"
            or manifest.get("job_id") != screening_id
            or manifest.get("artifact_sha256") != expected_artifact
            or content_sha256(manifest.get("files") or {}) != expected_artifact
            or content_sha256(manifest) != expected_manifest
            or probability_map.get("artifact_sha256") != expected_artifact
            or probability_map.get("manifest_sha256") != expected_manifest
            or objects_by_key(probability_map.get("objects"))
               != objects_by_key(objects)):
        raise RuntimeError(
            f"screening {screening_id} probability-map manifest/content mismatch")
    return {"path": str(canonical_probability_map(destination)),
            "artifact_sha256": expected_artifact,
            "manifest_sha256": expected_manifest, "objects": objects}


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


def supplied_input_note(job: dict) -> dict[str, str] | None:
    """Record that this job's input was brought to it, when it was.

    Helena is operated by more than one person, and a phase is not always run
    by whoever ran the one before it. Screening a stack a colleague rendered
    last week is ordinary work, and the platform allows it: outside the
    campaign's own control scroll, P5 takes a `tiff_dir` and runs.

    What it did not do is say so. P4 handed a bare path already records
    `surface_tifxyz` with that path; P5 handed a bare `tiff_dir` recorded
    nothing, so a result from an imported stack was indistinguishable in the
    evidence from one whose provenance nobody wrote down.

    Saying so is what makes the import safe rather than a hole. A researcher
    certifying part of a pipeline is entitled to certify the part they ran --
    and a reader is entitled to see where the chain begins. This is a
    statement of scope, not a demerit.

    None when `layer_stack` names a P4 this platform ran: that branch records
    the real lineage, and marking it supplied would be false.
    """
    if str(job.get("phase") or "") != "P5":
        # P4 answers its own bare-path case in its own vocabulary. A second
        # name for the same fact is how two records come to disagree.
        return None
    parameters = job.get("parameters") or {}
    if parameters.get("layer_stack"):
        return None
    # Either way of naming an input this platform did not produce: a layer stack
    # carried in by hand, or a surface volume fetched from somewhere published.
    # The second is how the public control gets its input, and it is as much an
    # outside input as the first -- more plainly so, since anybody can fetch it.
    kinds = {"tiff_dir": "supplied_layer_stack",
             "surface_volume": "supplied_surface_volume"}
    supplied = next((key for key in kinds if parameters.get(key)), None)
    if supplied is None:
        return None
    return {
        "kind": kinds[supplied],
        "path": str(parameters[supplied]),
        "non_claim": (
            f"this {'layer stack' if supplied == 'tiff_dir' else 'surface volume'} "
            "was supplied to the job and was not produced by this platform, so "
            "the chain of evidence begins here: what follows is recorded, what "
            "came before it is not"),
    }


class WorkerRefused(RuntimeError):
    """This host cannot run this job, for a reason about the host.

    Distinct from a rejected job: the request is fine and another worker may
    well take it. A worker without the vendored upstream this lane imports is
    the case that exists.
    """


def upstream_root(spec: Mapping[str, Any]) -> str | None:
    """The vendored architecture directory this host carries, if it does.

    Read here rather than passed in from the queue, which is the whole point:
    the directory a runner imports its model code from is a property of the
    machine that was built to run it.
    """
    variable = (spec.get("upstream") or {}).get("variable")
    return (os.environ.get(variable) or "").strip() or None if variable else None


# Phases whose runner reads object storage whatever it publishes to: P1 fetches
# its lasagna volumes from the campaign bucket, and P8's default lane vendors a
# script that requires boto3 outright.
S3_READING_PHASES = frozenset({"P1", "P8"})


def runner_needs_object_storage(job: Mapping[str, Any]) -> bool:
    """Whether this runner has any business holding the worker's AWS keys.

    P4 was the only phase they were taken from, for a reason about P4 -- and
    that left every other runner holding credentials to the campaign's private
    bucket whether or not it touches one. A subprocess that never opens an S3
    URL does not need them, and the ones that do say so: either the phase reads
    object storage regardless, or this job publishes to an s3:// store.

    Read from the store rather than assumed, because that value is the
    server's now and no longer something a request can point anywhere.
    """
    if job.get("phase") == "P4":
        # Its own reason, and the opposite one: the renderer streams the CT
        # from the public open-data bucket anonymously, and a signed request
        # against a bucket these keys do not own comes back 400 one second
        # into a render. Never, even if the store is s3.
        return False
    if str(job.get("phase") or "") in S3_READING_PHASES:
        return True
    store = str((job.get("parameters") or {}).get("artifact_store") or "")
    return store.startswith("s3://")


# Roots a runner may write into beyond the ones derived per job. Colon
# separated, for a deployment that publishes somewhere the queue cannot infer --
# a P9 plate run with its own volume, say. Empty is the ordinary case.
WRITE_ROOTS_VARIABLE = "HELENA_JOB_WRITE_ROOTS"


def refuse_unreadable_inputs(job: Mapping[str, Any]) -> None:
    """Say which input this worker cannot see, before spending a lease on it.

    The panel and the worker are different containers, often on different
    hosts, and a path that exists for whoever queued the job says nothing about
    whether the process that opens it can. A render pointed at a directory
    outside the worker's mounts got one line -- `Error loading` -- from a
    renderer that then exited 0, and the hour after that went on TIFF tag
    types, a bbox, and everything except the file not being there.

    Checked here rather than at enqueue for the same reason it is worth
    checking at all: this is the only process that knows what it can reach.

    "Here" is after the claim, holding the lease -- so a bad path costs one
    attempt and the job reads `failed` with the reason in it. That is still far
    better than the alternative, which was a renderer saying `Error loading`
    and exiting zero; but it is not free, and this docstring used to claim it
    happened before a lease was spent. The only thing that genuinely filters
    before a claim is the runtime image.
    """
    parameters = job.get("parameters") or {}
    checked = set(READ_PATH_PARAMETERS)
    # `volume` is conditional, which is why it is not in the static set. With a
    # remote URL it is the renderer's own cache and absent is the ordinary first
    # run; without one it is an input that has to be there, and leaving it
    # unchecked left the exact failure this function exists for reachable by
    # another door -- `Error loading`, exit zero, nothing rendered.
    if not parameters.get("remote_url"):
        checked.add("volume")
    for key in sorted(checked):
        value = parameters.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.exists():
            raise WorkerRefused(
                f"{key} is {value!r}, which does not exist on this worker. It "
                "may well exist on the host: this process sees only what the "
                "container mounts, and that path is not one of them.")
        if not os.access(path, os.R_OK):
            raise WorkerRefused(
                f"{key} is {value!r}, which exists on this worker and is not "
                "readable by it. This is a permission, not a path.")


def refuse_writes_outside_the_job(job: Mapping[str, Any], *, runs_root: Path) -> None:
    """Refuse a job whose output would land somewhere that is not its own.

    validate_parameters bounds these to "absolute, no ..", which is a shape and
    not a place: every absolute path on the host satisfies it. The bound is this
    worker's run directory and the store it publishes to, and neither is known
    where the job is validated -- so it is checked here, where they are.

    On the resolved path, for the same reason the artifact endpoints are: a
    symlink inside the run directory points wherever it points.
    """
    parameters = job.get("parameters") or {}
    roots = [Path(runs_root).resolve()]
    store = str(parameters.get("artifact_store") or "")
    if store.startswith("/"):
        roots.append(Path(store).resolve())
    roots += [Path(extra).resolve()
              for extra in os.environ.get(WRITE_ROOTS_VARIABLE, "").split(":")
              if extra.strip()]
    for key in sorted(WRITE_PATH_PARAMETERS):
        value = parameters.get(key)
        if not value:
            continue
        target = Path(str(value)).resolve()
        if not any(target == root or root in target.parents for root in roots):
            raise WorkerRefused(
                f"{key} would write to {value!r}, which is outside this job's "
                f"run directory and outside where it publishes. Roots this "
                f"worker allows: {[str(root) for root in roots]}. A deployment "
                f"that means it can name more in {WRITE_ROOTS_VARIABLE}.")


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
        # needs that directory on the path. It used to be a job parameter,
        # which meant a request chose what the GPU host imported; it is the
        # worker's own now, and a worker that does not have one refuses the
        # lane rather than importing from wherever it was pointed.
        spec = ink_adapter(job.get("profile_id"))[1]
        upstream = spec.get("upstream") or {}
        if upstream.get("pythonpath"):
            root = upstream_root(spec)
            if not root:
                raise WorkerRefused(
                    f"this lane imports its architecture from a vendored "
                    f"upstream directory and this worker does not say where "
                    f"one is: set {upstream['variable']} on this host")
            environment["PYTHONPATH"] = f"{root}:{environment['PYTHONPATH']}"
    if not runner_needs_object_storage(job):
        for key in [k for k in environment if k.startswith("AWS_")]:
            del environment[key]
    return environment


def worker_failure_result(
        error: str, output: Path, control_binding: dict | None) -> dict:
    """Keep immutable classification on every terminal worker result."""
    result = {"error": error, "output_dir": str(output)}
    if control_binding is not None:
        result.update(control_binding)
    return result


def run_job(store: InkJobStore, job: dict, *, runs_root: Path, timeout: int) -> None:
    job_id, token = job["job_id"], job["lease_token"]
    lineage_guard = getattr(store, "require_job_canonical_lineage", None)
    if callable(lineage_guard):
        lineage_guard(job, execution=True)
    # And the size question, which lineage cannot see. Both are asked here
    # rather than at enqueue because a decision made when the job was queued is
    # a decision about a surface that may since have been re-measured, and
    # because everything below this line is I/O on the surface itself.
    route_guard = getattr(store, "require_job_standard_route", None)
    if callable(route_guard):
        route_guard(job)
    output = runs_root / f"{job['sample_id'].lower()}-{job_id}"
    # Every phase is given a directory that exists. Some runners create their
    # own -- P5's does -- and some write a file into it and expect it to be
    # there: vet_map died on FileNotFoundError for its verdict, having read the
    # map, screened it and found the shapes, because nothing had made the
    # directory it was told to write into.
    output.mkdir(parents=True, exist_ok=True)
    rendered_from: dict[str, str] | None = None
    control_binding: dict | None = None
    try:
        control_binding = verified_control_binding(job)
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
                "flattening_id": str(job["_flattened_sheet"]["artifact_id"]),
                "p3_job_id": str(job["_flattened_sheet"]["p3_job_id"]),
                "flattened_artifact_sha256": str(
                    job["_flattened_sheet"]["artifact_sha256"]),
                "profile_id": str(job["parameters"].get("flattening_profile")
                                  or DEFAULT_FLATTENING_PROFILE),
                **({"orientation_receipt_sha256": str(
                    job["parameters"]["orientation_receipt_sha256"])}
                   if job["parameters"].get("orientation_receipt_sha256") else {}),
                **({"flip_normals": job["parameters"]["flip_normals"]}
                   if "flip_normals" in job["parameters"] else {}),
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
            resolved_map = resolve_screened_map(
                store, job, runs_root / f"{job_id}-map")
            job["parameters"]["map_path"] = resolved_map["path"]
            rendered_from = {"kind": "probability_map",
                             "screened_by": str(job["parameters"]["screening_of"]),
                             "probability_map_artifact_sha256": resolved_map[
                                 "artifact_sha256"],
                             "probability_map_manifest_sha256": resolved_map[
                                 "manifest_sha256"],
                             **({"roi_receipt_sha256": str(
                                 job["parameters"]["roi_receipt_sha256"])}
                                if job["parameters"].get("roi_receipt_sha256") else {}),
                             "bbox": job["parameters"].get("bbox"),
                             "px_um": job["parameters"].get("px_um"),
                             **({"surface_id": job["parameters"]["surface_id"]}
                                if job["parameters"].get("surface_id") else {}),
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
        elif job.get("phase") == "P5" and not job["parameters"].get("layer_stack"):
            # Brought from outside. Allowed, and recorded as such.
            rendered_from = supplied_input_note(job)
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
                             **({
                                 "layer_stack_artifact_sha256": job[
                                     "_source_layer_stack"]["artifact_sha256"],
                                 "layer_stack_manifest_sha256": job[
                                     "_source_layer_stack"]["manifest_sha256"],
                                 "lateral_metric_receipt_sha256": job.get(
                                     "_source_lateral_metric", {}).get("receipt_sha256"),
                                 "source_pixel_um": job["parameters"].get(
                                     "source_pixel_um"),
                                 "source_slice_um": job["parameters"].get(
                                     "source_slice_um"),
                             } if job.get("_source_layer_stack") else {}),
                             "non_claim": ("a probability map is a screening "
                                           "output, not a reading")}
        if rendered_from is not None and control_binding is not None:
            rendered_from.update(control_binding)
        require_runtime(job)
        refuse_unreadable_inputs(job)
        refuse_writes_outside_the_job(job, runs_root=runs_root)
        argv = command_for(job, runner=str(runner_for(job)), output_dir=str(output),
                           upstream_root=upstream_root(
                               ink_adapter(job.get("profile_id"))[1]
                               if job.get("phase") == "P5" else {}))
    except Exception as exc:  # noqa: BLE001 -- a bad job must not stop the worker
        failure = {"error": f"{type(exc).__name__}: {exc}"}
        if control_binding is not None:
            failure.update(control_binding)
        store.finish(job_id, token, state="failed",
                     result=failure)
        return

    stop = threading.Event()
    # The child, once it exists, and whether somebody asked for it to stop. The
    # heartbeat is the only thing that hears a cancellation, and the only thing
    # positioned to act on it.
    running: dict[str, object] = {"process": None}
    cancelled: dict[str, float | None] = {"at": None}
    # The most recent thing the child said, and when. Written by the reader
    # threads, read by the heartbeat: one line, replaced rather than queued,
    # because what a watcher wants is where the job is now.
    latest: dict[str, object] = {"line": None, "source": None, "at": None}

    def remember(source: str, line: str) -> None:
        text = line.strip()
        if text:
            latest["line"] = text[:PROGRESS_LINE_CHARS]
            latest["source"] = source
            latest["at"] = time.time()

    def beat() -> None:
        # A long inference must not look abandoned. The lease is renewed while
        # the process lives, and stops the moment it does not -- and it carries
        # the job's latest line, so the queue is not the only thing a watcher
        # can see.
        while not stop.wait(HEARTBEAT_SECONDS):
            note = None
            if latest["line"] is not None:
                note = {
                    "line": latest["line"],
                    "source": latest["source"],
                    "at": datetime.fromtimestamp(
                        float(latest["at"] or 0), tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                }
            try:
                asked_to_stop = store.heartbeat(job_id, token, progress=note)
            except RuntimeError:
                return
            if asked_to_stop and not cancelled["at"]:
                # Terminate, not kill: the runner gets the chance to close what
                # it has open. The wait below is short because the point of
                # cancelling is that somebody is waiting for the card back.
                cancelled["at"] = time.time()
                child = running.get("process")
                if child is not None:
                    child.terminate()
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        child.kill()
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
        completed = run_streaming(
            argv, timeout=timeout, env=runner_environment(job),
            on_line=remember,
            on_start=lambda child: running.__setitem__("process", child))
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
        # A terminated child exits non-zero. Reported as a failure that would
        # read as the lane breaking, and the next person would go looking for a
        # bug instead of finding the person who stopped it.
        was_cancelled = cancelled["at"] is not None
        result = {
            "exit_code": completed.returncode,
            "runtime_seconds": round(time.time() - started, 1),
            "stdout_tail": tail,
            "stderr_tail": errors,
            "statistics": (receipt or {}).get("statistics"),
            "liveness": (receipt or {}).get("liveness"),
            # The 9 um lane's comparison of its published map against the one
            # `direction: both` also wrote -- p50/p99 of the reverse map, the
            # correlation between the two, and (when their shapes match) the
            # forward/reverse asymmetry-by-threshold block. None on every
            # other lane, and on a forward-only or reverse-only 9 um run: there
            # is no second map to have compared it against.
            "reverse": (receipt or {}).get("reverse"),
            "output_dir": str(output),
            "ran_by": worker_code_revision(),
        }
        # Where the bytes actually landed, when the job named somewhere else.
        #
        # `output_dir` is the run directory this worker made, and it is where
        # receipts and logs go -- but P8 takes `--out <file>` and P9 `--out
        # <dir>`, so a job that names one leaves that directory empty and the
        # record points at nothing. A P9 run that wrote 38 plates read as a job
        # with an empty output directory, and only the person who typed the path
        # knew otherwise.
        named = job.get("parameters") or {}
        destination = next(
            (str(named[name]) for name in ("out_dir", "out_path")
             if named.get(name)), None)
        if destination and destination != str(output):
            result["wrote_to"] = destination
        if control_binding is not None:
            result.update(control_binding)
        if job.get("phase") == "P3" and completed.returncode == 0:
            # `result` becomes the terminal event's payload, and the control
            # reads that event as its second witness that *this* job produced
            # the artifact the flattening index carries.
            lineage = flattening_lineage(completed.stdout or "", job_id)
            if lineage:
                result["surfaces"] = lineage
        if job.get("phase") == "P7":
            # Beside the card's verdict, never instead of it: whether the
            # strongest repetition in the screened window is the writing or the
            # grid the map was rendered on. Any structure metric over an
            # upsampled render carries a peak at the upsampling factor, and a
            # row-periodicity score that reads 0.85 on blank papyrus is reading
            # the mesh. The card is vendored byte for byte, so this rides
            # alongside rather than inside it.
            result["grid_alarm"] = screened_window_grid_alarm(job)
        if job.get("phase") == "P7" and rendered_from is not None:
            result["probability_map_input"] = {
                "screened_by": rendered_from["screened_by"],
                "artifact_sha256": rendered_from[
                    "probability_map_artifact_sha256"],
                "manifest_sha256": rendered_from[
                    "probability_map_manifest_sha256"],
            }
        if job.get("phase") == "P5" and isinstance(receipt, dict):
            physical = receipt.get("physical_normalization") or {}
            lane = receipt.get("lane") or {}
            checkpoint = receipt.get("checkpoint") or {}
            result.update({
                "physical_normalization": physical,
                "map_shape_yx": physical.get("target_shape_y_x")
                    or ((receipt.get("input") or {}).get("shape_model_y_x")),
                "checkpoint_sha256": checkpoint.get("sha256")
                    or lane.get("checkpoint_sha256"),
            })
            source_stack = job.get("_source_layer_stack")
            metric = job.get("_source_lateral_metric") or {}
            if source_stack and control_binding is not None:
                profile_path = ink_profile_path(str(job.get("profile_id") or ""))
                if profile_path is None:
                    raise RuntimeError("P5 profile cannot be resolved for normalization")
                profile_document = json.loads(profile_path.read_text(encoding="utf-8"))
                profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
                training_pixel_um = (profile_document.get("input_contract") or {}).get(
                    "training_pixel_um")
                normalization = {
                    "schema": "campaignx.first_letters_p5_normalization.v1",
                    "p4_job_id": source_stack["p4_job_id"],
                    "p4_layer_artifact_sha256": source_stack["artifact_sha256"],
                    "p4_layer_manifest_sha256": source_stack["manifest_sha256"],
                    "source_layer_objects": source_stack["objects"],
                    "lateral_metric_receipt_sha256": metric.get("receipt_sha256"),
                    "source_pixel_um": job["parameters"].get("source_pixel_um"),
                    "source_slice_um": job["parameters"].get("source_slice_um"),
                    "training_pixel_um": training_pixel_um,
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "profile_id": job.get("profile_id"),
                    "profile_sha256": profile_sha,
                }
                normalization["receipt_sha256"] = hashlib.sha256(json.dumps(
                    normalization, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
                result["adapter_physical_normalization"] = physical
                result["physical_normalization"] = normalization
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
                if control_binding is not None:
                    control = job.get("_verified_control_policy")
                    if not isinstance(control, dict):
                        raise RuntimeError("worker lost its verified control policy")
                    policy = control["checks"]["PIPELINE_CONTROL"][
                        "p3_p4_lateral_metric"]["policy"]
                    sheet = job.get("_flattened_sheet") or {}
                    layer_stack = result.get("layer_stack") or {}
                    metric = measure_p3_p4_lateral_metric(
                        output / "flattened-surface", result["layers"]["shape"],
                        source_voxel_um=float(job["parameters"]["source_voxel_um"]),
                        lineage={
                            "flattened_artifact_id": sheet.get("artifact_id"),
                            "flattened_artifact_sha256": sheet.get("artifact_sha256"),
                            "p3_job_id": sheet.get("p3_job_id"), "p4_job_id": job_id,
                            "p4_layer_artifact_sha256": layer_stack.get("artifact_sha256"),
                            "p4_layer_manifest_sha256": layer_stack.get("manifest_sha256"),
                            "p4_layer_objects": layer_stack.get("objects"),
                        }, policy=policy)
                    result["lateral_metric"] = metric
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
        if was_cancelled:
            state = "cancelled"
            result["stopped_by_request"] = True
        store.finish(job_id, token, state=state, result=result)
    except subprocess.TimeoutExpired:
        store.finish(job_id, token, state="failed",
                     result=worker_failure_result(
                         f"timed out after {timeout}s", output, control_binding))
    except Exception as exc:  # noqa: BLE001 -- a worker must not die on one job
        store.finish(job_id, token, state="failed",
                     result=worker_failure_result(
                         f"{type(exc).__name__}: {exc}", output, control_binding))
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

    misnamed = misnamed_runtime(RUNTIME_IMAGE)
    if misnamed:
        print(misnamed, file=sys.stderr, flush=True)
        return 3

    print(f"ink worker {worker_id} on {args.host_id}, runs -> {args.runs_root}, "
          f"phases: {','.join(phases) if phases else 'all'}"
          + (f", runtime: {RUNTIME_IMAGE}" if RUNTIME_IMAGE else ""), flush=True)
    last_state = 0.0
    # Probed once before the first claim, not only on the heartbeat: a worker
    # that claimed before its first probe would have no cards recorded and
    # would refuse every GPU job for the first minute of its life.
    last_probe = host_state(args.runs_root)
    # Printed only when it changes, not every poll: the worker that lost its
    # GPU for five hours logged nothing at all about it, and a line repeated
    # every ten seconds for five hours is a log nobody reads either. A sentinel
    # distinct from True, False and None -- gpu_visible's own three answers --
    # so the very first probe always prints once, whatever it finds.
    _UNKNOWN = object()
    last_gpu_visible: bool | None | object = _UNKNOWN
    while True:
        if time.time() - last_state > 60:
            try:
                last_probe = host_state(args.runs_root)
                store.record_host_state(args.host_id, last_probe)
            except Exception as exc:  # noqa: BLE001
                print(f"host state not recorded: {exc}", file=sys.stderr, flush=True)
            last_state = time.time()

        # This worker's own device, asked fresh on every poll -- not the
        # host-wide reading above, and not something decided once at startup.
        # helena-ink-0 once lost its container's GPU passthrough silently and
        # kept polling; has_gpu stayed whatever it was the one time it had been
        # computed, and nothing this worker wrote said a card was missing.
        gpu_visible = worker_gpu_visible()
        if gpu_visible != last_gpu_visible:
            print(
                "this worker claims no GPU (nvidia-smi is not on its PATH)"
                if gpu_visible is None else
                "this worker's GPU is no longer visible to nvidia-smi"
                if gpu_visible is False else
                "this worker's GPU is visible", file=sys.stderr, flush=True)
            last_gpu_visible = gpu_visible

        try:
            # What this machine actually has, measured rather than assumed.
            # A host with no card must not take a job that needs one: it fails
            # it, burns an attempt, and leaves the queue looking broken instead
            # of misrouted. gpu_vram_gb still reads the once-a-minute host
            # probe, the only one that carries VRAM; eligibility itself now
            # reads the fresher, per-poll, worker-scoped probe above rather
            # than that same once-a-minute reading.
            cards = (last_probe or {}).get("gpus") or []
            job = store.claim(
                worker_id=worker_id, host_id=args.host_id,
                lease_seconds=args.lease_seconds, phases=phases,
                has_gpu=bool(gpu_visible),
                # Recorded on ink_workers on every poll, so a worker whose card
                # has gone silent stops looking exactly like a worker with
                # nothing to claim -- today that row says nothing about a GPU
                # at all, healthy or not.
                gpu_visible=gpu_visible,
                # Which image this worker is, so a lane needing another one is
                # left for the worker that has it. require_runtime already
                # refused those by name -- but at execution, having taken the
                # job and spent an attempt on it.
                runtime=RUNTIME_IMAGE,
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
