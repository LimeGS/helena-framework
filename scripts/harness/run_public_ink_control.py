#!/usr/bin/env python3
"""The short chain: a positive control anybody can reproduce from public data.

Why a second control rather than a repair of the first
------------------------------------------------------
The nine-boundary campaign control stops at P1 with
FROZEN_ROOT_OBJECT_EVIDENCE_MISSING and seven boundaries never reached. Even
repaired it reads surfaces out of a private bucket, so two of the things asked
of it -- public input surfaces, and a run from a clean installation -- are out
of its reach however well it works.

The recommended tooling can reach both. The surface volume is in the open-data
bucket and answers without a credential; the checkpoint is a public, non-gated
repository whose digest this platform verified byte for byte. Nothing in this
chain needs an account.

The six boundaries
------------------
    PUBLIC_SOURCE  the volume is reachable and is what it declares
    SCALE          it is at the model's scale, or poolable to it by the
                   recipe the model card gives -- never by an invented factor
    CHECKPOINT     the model file's digest is the declared one
    INK            inference completed and wrote a map
    LIVENESS       the map carries a decision rather than one value everywhere
    HUMAN_REVIEW   it is routed for review and claims nothing on its own

Non-claims
----------
* This is not a reading, not an ink claim, and not a letter claim.
* Passing says this platform can drive the recommended tooling end to end on
  data anybody can obtain. It says nothing about whether the map is correct,
  and nothing about the nine-boundary campaign control, which is a different
  receipt with a different schema that this cannot be published as.
* A single checkpoint of the several upstream publishes, at one z window.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# <root>/scripts/harness/this file -- so the repository root is two levels up,
# not one. parents[1] is <root>/scripts, which contains no `framework`, and
# every test that imports this module runs inside one that has already put the
# root on sys.path, so the arithmetic was never exercised until it ran alone.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.contracts.lane_liveness import assess_liveness  # noqa: E402
from run_first_letters_positive_control import (  # noqa: E402
    FAILED, INCOMPLETE, NOT_RUN, PASS, PUBLIC_BOUNDARIES, PUBLIC_SCHEMA,
    evaluate_survival_matrix,
)

sys.path.insert(0, str(REPO_ROOT / "framework/stages/03-ink/scripts"))
from prepare_9um_isotropic_input import (  # noqa: E402
    IncompatibleSourceScale, MODEL_SCALE_UM, plan_pooling,
)

NON_CLAIMS = (
    "This is not a reading, not an ink claim, and not a letter claim.",
    "Passing says this platform drove the recommended tooling end to end on "
    "public data. It says nothing about whether the resulting map is correct.",
    "This is not the nine-boundary campaign control and cannot be published "
    "as one; that is a different receipt with a different schema.",
    "One checkpoint of the several upstream publishes, at one z window.",
)


def _row(boundary: str) -> dict[str, Any]:
    return {"boundary": boundary, "terminal_state": NOT_RUN,
            "reason_code": "PREREQUISITE_NOT_REACHED", "elapsed_seconds": 0.0,
            "resource_identity": {}, "input_artifacts": [],
            "output_hashes": {}, "counts": {}}


def _set(row: dict[str, Any], state: str, reason: str, **fields) -> None:
    row.update({"terminal_state": state, "reason_code": reason, **fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PublicSource:
    """Reads a public OME-Zarr's own metadata, with no credentials at all.

    Deliberately anonymous: a control that proves "public" while sending
    credentials has proved nothing about what a stranger can obtain.
    """

    def read_metadata(self, uri: str) -> dict[str, Any]:
        request = urllib.request.Request(uri.rstrip("/") + "/.zattrs")
        with urllib.request.urlopen(request, timeout=60) as response:
            attrs = json.loads(response.read())
        scales = attrs.get("multiscales") or [{}]
        axes = [a.get("name") for a in (scales[0].get("axes") or [])]
        voxel = None
        for dataset in scales[0].get("datasets") or []:
            for transform in dataset.get("coordinateTransformations") or []:
                if transform.get("type") == "scale" and transform.get("scale"):
                    voxel = float(transform["scale"][-1])
                    break
            if voxel is not None:
                break
        return {"axes": axes, "voxel_size_um": voxel,
                "canvas_size": attrs.get("canvas_size")}


def run_public_ink_control(
    *,
    surface_volume: str,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    output: Path,
    source: Any | None = None,
    inference: Callable[..., dict[str, Any]] | None = None,
    clock: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Run the six boundaries and return the receipt, whatever happened."""
    import time

    clock = clock or time.monotonic
    source = source or PublicSource()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [_row(name) for name in PUBLIC_BOUNDARIES]
    by = {row["boundary"]: row for row in rows}

    receipt: dict[str, Any] = {
        "schema": PUBLIC_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0)
        .isoformat().replace("+00:00", "Z"),
        "surface_volume": surface_volume,
        "stages": rows,
        "non_claims": list(NON_CLAIMS),
    }

    # -- PUBLIC_SOURCE -----------------------------------------------------
    started = clock()
    try:
        metadata = source.read_metadata(surface_volume)
    except Exception as failure:  # noqa: BLE001 -- a boundary result, not a crash
        _set(by["PUBLIC_SOURCE"], INCOMPLETE, "PUBLIC_SOURCE_UNREACHABLE",
             elapsed_seconds=max(0.0, clock() - started),
             detail=f"{type(failure).__name__}: {failure}"[:400])
        return evaluate_survival_matrix(receipt)
    voxel = metadata.get("voxel_size_um")
    if voxel is None:
        _set(by["PUBLIC_SOURCE"], INCOMPLETE, "PUBLIC_SOURCE_SCALE_UNDECLARED",
             elapsed_seconds=max(0.0, clock() - started))
        return evaluate_survival_matrix(receipt)
    _set(by["PUBLIC_SOURCE"], PASS, "PUBLIC_SOURCE_READ_ANONYMOUSLY",
         elapsed_seconds=max(0.0, clock() - started),
         resource_identity={"uri": surface_volume, "axes": metadata.get("axes"),
                            "canvas_size": metadata.get("canvas_size"),
                            "voxel_size_um": voxel,
                            "credentials_used": False})

    # -- SCALE -------------------------------------------------------------
    started = clock()
    try:
        plan = plan_pooling(float(voxel))
    except IncompatibleSourceScale as refused:
        _set(by["SCALE"], INCOMPLETE, "SOURCE_SCALE_UNREACHABLE_BY_THE_RECIPE",
             elapsed_seconds=max(0.0, clock() - started), detail=str(refused)[:400])
        return evaluate_survival_matrix(receipt)
    _set(by["SCALE"], PASS,
         "NATIVE_MODEL_SCALE" if plan.xy_factor == 1 else "POOLED_BY_THE_RECIPE",
         elapsed_seconds=max(0.0, clock() - started),
         counts={"xy_factor": plan.xy_factor, "z_factor": plan.z_factor},
         resource_identity={"source_voxel_um": float(voxel),
                            "model_scale_um": MODEL_SCALE_UM,
                            "output_voxel_um": plan.output_voxel_um})

    # -- CHECKPOINT --------------------------------------------------------
    started = clock()
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        _set(by["CHECKPOINT"], INCOMPLETE, "CHECKPOINT_MISSING",
             elapsed_seconds=max(0.0, clock() - started))
        return evaluate_survival_matrix(receipt)
    digest = sha256_file(checkpoint)
    if digest != expected_checkpoint_sha256:
        _set(by["CHECKPOINT"], FAILED, "CHECKPOINT_DIGEST_MISMATCH",
             elapsed_seconds=max(0.0, clock() - started),
             detail=f"{digest} is not the declared {expected_checkpoint_sha256}",
             output_hashes={"checkpoint_sha256": digest})
        return evaluate_survival_matrix(receipt)
    _set(by["CHECKPOINT"], PASS, "CHECKPOINT_IS_THE_DECLARED_ONE",
         elapsed_seconds=max(0.0, clock() - started),
         output_hashes={"checkpoint_sha256": digest},
         resource_identity={"bytes": checkpoint.stat().st_size})

    # -- INK ---------------------------------------------------------------
    started = clock()
    try:
        execution = (inference or _default_inference)(
            surface_volume=surface_volume, checkpoint=checkpoint, output=output)
    except Exception as failure:  # noqa: BLE001
        _set(by["INK"], INCOMPLETE, "INFERENCE_DID_NOT_COMPLETE",
             elapsed_seconds=max(0.0, clock() - started),
             detail=f"{type(failure).__name__}: {failure}"[:400])
        return evaluate_survival_matrix(receipt)
    probability_path = output / "probability.npy"
    if not probability_path.is_file():
        # The same rule P3 applies to vc_flatten: an exit code is not evidence
        # that the output exists.
        _set(by["INK"], INCOMPLETE, "INFERENCE_WROTE_NO_MAP",
             elapsed_seconds=max(0.0, clock() - started))
        return evaluate_survival_matrix(receipt)
    probability = np.load(probability_path)
    _set(by["INK"], PASS, "PROBABILITY_MAP_WRITTEN",
         elapsed_seconds=max(0.0, clock() - started),
         counts={"shape": [int(n) for n in probability.shape]},
         output_hashes={"probability_sha256": sha256_file(probability_path)},
         # Whatever the inference reported about itself, rather than one key
         # plucked out of it. It read `argv`, which a queued run has no
         # equivalent of, so the receipt of the run that most needed
         # identifying -- the one that went through the platform -- recorded
         # `command: null` and named nothing at all.
         resource_identity=dict(execution or {}))

    # -- LIVENESS ----------------------------------------------------------
    started = clock()
    liveness = assess_liveness(probability,
                               valid=np.ones(probability.shape, dtype=bool))
    if liveness["verdict"] != "ALIVE":
        _set(by["LIVENESS"], INCOMPLETE, f"MAP_{liveness['verdict']}",
             elapsed_seconds=max(0.0, clock() - started),
             detail=str(liveness.get("reason"))[:400], counts=liveness["metrics"])
        return evaluate_survival_matrix(receipt)
    _set(by["LIVENESS"], PASS, "MAP_CARRIES_A_DECISION",
         elapsed_seconds=max(0.0, clock() - started), counts=liveness["metrics"])

    # -- HUMAN_REVIEW ------------------------------------------------------
    started = clock()
    _set(by["HUMAN_REVIEW"], PASS, "ROUTED_TO_REVIEW_WITHOUT_A_CLAIM",
         elapsed_seconds=max(0.0, clock() - started),
         resource_identity={"map": str(probability_path),
                            "claims_nothing": True})
    return evaluate_survival_matrix(receipt)


def _default_inference(*, surface_volume: str, checkpoint: Path,
                       output: Path) -> dict[str, Any]:
    """Upstream's own runner, through this platform's lane adapter."""
    import subprocess

    adapter = REPO_ROOT / "framework/stages/03-ink/scripts/run_ink_9um.py"
    profile = (REPO_ROOT
               / "framework/profiles/03-ink/ink-9um-hybrid-3d2d-screening-1.0.0.json")
    # --output is a DIRECTORY to the adapter: it mkdirs the path and writes
    # ink.tif, probability.npy and the reverse pair inside it. Passing the tif
    # itself made a directory named ink.tif holding a second ink.tif, so the
    # inference ran its full twenty-six minutes and then the control looked for
    # probability.npy one level up and recorded INFERENCE_WROTE_NO_MAP. Both
    # halves believed they had succeeded, which is the expensive kind of
    # disagreement -- the adapter's own contract is the one to follow.
    argv = [sys.executable, str(adapter), "--profile", str(profile),
            "--surface-volume", surface_volume, "--checkpoint", str(checkpoint),
            "--output", str(output)]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=14400)
    if completed.returncode != 0:
        raise RuntimeError(
            f"the ink lane exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout or '')[-600:]}")
    # Both ways of running this answer the same question -- how was this map
    # made -- so both say which way. A receipt where only one path names itself
    # invites the reader to assume the other.
    return {"through": "local-subprocess", "argv": argv}


ARTIFACT_ROOT = "/artifacts/"


def _unpack_published_map(archive: bytes, output: Path) -> None:
    """Place what the panel handed back, without letting it choose where.

    These bytes arrive over HTTP from the host this control is testing, which
    is exactly the assumption not to make about a tar: it can carry absolute
    paths, `..`, symlinks and device nodes. Members are checked before anything
    is written, the same way the panel checks the ones a worker sends it.
    """
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            name = Path(member.name)
            if member.name.startswith("/") or ".." in name.parts:
                raise RuntimeError(f"unsafe path in the published artifact: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"only files and directories: {member.name}")
        bundle.extractall(output)  # noqa: S202 - members checked above


def queued_inference(panel: Any, *, mission_id: str, sample_id: str,
                     profile_id: str, parameters: dict[str, Any],
                     minutes: float = 180) -> Callable[..., dict[str, Any]]:
    """Drive the INK boundary through Helena, using nothing but its API.

    The control passes without this: it drives the recommended tooling end to
    end on public data. What it does not do is exercise Helena -- the local path
    shells out to the lane adapter, so a green receipt speaks for the tooling
    and says nothing about the queue, the worker, the routing or the
    publication, which is the half a reviewer is being asked to trust.

    Enqueue, wait, and fetch what the job published. No path on a worker's disk
    is read: a control that reaches into the machine it is testing is not
    testing the interface anybody else would use.
    """

    def run(*, surface_volume: str, checkpoint: Path, output: Path) -> dict[str, Any]:
        queued = panel.call("POST", "/api/jobs", {
            "sample_id": sample_id, "phase": "P5", "mission_id": mission_id,
            "profile_id": profile_id, "parameters": dict(parameters),
            "max_attempts": 1})
        job_id = queued["job_id"]
        finished = panel.wait_for_job(job_id, minutes=minutes)
        result = finished.get("result") or {}
        if finished.get("state") != "succeeded":
            raise RuntimeError(
                f"{job_id} ended {finished.get('state')}: "
                f"{result.get('error') or (result.get('stderr_tail') or '')[-400:]}")

        published = (result.get("probability_map") or {}).get("artifact_uri")
        if not published:
            raise RuntimeError(
                f"{job_id} reported success and published no map; an exit code "
                "is not evidence that the output exists")
        if not str(published).startswith(ARTIFACT_ROOT):
            raise RuntimeError(
                f"{job_id} published outside the artifact volume: {published}")

        key = str(published)[len(ARTIFACT_ROOT):]
        _unpack_published_map(panel.fetch(f"/api/artifacts/{key}"), Path(output))
        return {
            "through": "helena-queue",
            "job_id": job_id,
            "artifact_uri": published,
            "artifact_sha256": (result.get("probability_map") or {}).get(
                "artifact_sha256"),
            "manifest_sha256": (result.get("probability_map") or {}).get(
                "manifest_sha256"),
            "runtime_seconds": result.get("runtime_seconds"),
        }

    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface-volume", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--expected-checkpoint-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    # Through Helena, rather than beside it. Without --panel the ink step is a
    # local subprocess, which proves the tooling and nothing about the queue.
    ap.add_argument("--panel", help="drive the ink step through this panel's API")
    ap.add_argument("--user")
    ap.add_argument("--password", default=os.environ.get("HELENA_PANEL_PASSWORD"))
    ap.add_argument("--mission", help="the mission the queued job belongs to")
    ap.add_argument("--sample-id", default="PHerc0332")
    ap.add_argument("--profile-id",
                    default="ink-9um-hybrid-3d2d-screening@1.0.0")
    ap.add_argument("--tiff-dir",
                    help="queue a P4 layer stack instead of the volume above, "
                         "as the worker sees it; the scale is then required")
    ap.add_argument("--checkpoint-path",
                    help="the checkpoint, as the worker sees it")
    ap.add_argument("--source-pixel-um", type=float, default=9.362)
    # Defaults to the pixel size because the control's volume is isotropic --
    # its own key says 9.362um and the procedure calls it isotropic -- but it is
    # a separate figure and an anisotropic volume has to say so. Defaulting to
    # None and filling it in below keeps "not given" distinguishable from "given
    # and equal", which is what a receipt has to be able to show.
    ap.add_argument("--source-slice-um", type=float, default=None)
    args = ap.parse_args()

    inference = None
    if args.panel:
        for needed in ("user", "password", "mission", "checkpoint_path"):
            if not getattr(args, needed):
                ap.error(f"--panel needs --{needed.replace('_', '-')}")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from panel_client import Panel  # noqa: PLC0415

        panel = Panel(args.panel)
        panel.sign_in(args.user, args.password)
        inference = queued_inference(
            panel, mission_id=args.mission, sample_id=args.sample_id,
            profile_id=args.profile_id,
            # The same volume the boundaries above just verified, handed to
            # the queue as it is. Naming a path on the worker instead would put
            # a second input in the receipt and leave which one the map came
            # from a matter of trust. A stack is the alternative, and pooling
            # then needs the scale it was rendered at.
            # artifact_store is not sent: the panel owns it and refuses a
            # request that sets it with HTTP 409. requeue_timesformer_large.py
            # already carried that note and this did not, so the queued path
            # failed at the first POST against any deployment -- which is the
            # path a reviewer is asked to trust.
            parameters={"checkpoint": args.checkpoint_path,
                        # source_pixel_um on both paths. It was sent only with
                        # a tiff_dir, and the panel needs it either way: its
                        # catalogue has no entry for a scroll a stranger brings,
                        # so it refuses rather than guessing a scale --
                        # correctly, because a wrong micron silently pools the
                        # volume to the wrong size and the map still looks like
                        # a map. The control already had the figure and did not
                        # pass it.
                        "source_pixel_um": args.source_pixel_um,
                        "source_slice_um": (args.source_slice_um
                                            if args.source_slice_um is not None
                                            else args.source_pixel_um),
                        **({"tiff_dir": args.tiff_dir}
                           if args.tiff_dir else
                           {"surface_volume": args.surface_volume})})

    receipt = run_public_ink_control(
        surface_volume=args.surface_volume, checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        output=args.output, inference=inference)
    (args.output / "PUBLIC_INK_CONTROL.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in
                      ("control_state", "first_nonpassing_boundary",
                       "content_sha256")}, indent=2, sort_keys=True))
    return 0 if receipt["control_state"] == "CONTROL_PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
