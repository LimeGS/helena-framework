#!/usr/bin/env python3
"""Fit one scroll with the spiral fitter, and file what it produced.

This is the P1 lane's runner: the one program a queued spiral job turns into.
It does six things, in an order chosen so that everything cheap that can
refuse a run happens before anything expensive starts.

    1. resolve the frozen profile by id, from this repository
    2. build the scroll binding and write it as spiral-scroll.json
    3. check the dataset holds every input the profile names
    4. run the fit -- it writes a checkpoint, not a surface
    5. export that checkpoint to one flattened TIFXYZ
    6. publish the TIFXYZ and register it as a surface

Steps 1-3 cost a few milliseconds and catch the failures that otherwise appear
forty minutes into a GPU lease: a dataset missing its umbilicus, a profile
that predates this fitter's interface, a winding sense nobody set.

Steps 4 and 5 are two subprocesses, not one. At the commit this once ran
against, `fit_spiral.py` wrote TIFXYZ meshes directly and step 5 did not
exist. At 23adee04 the fitter is checkpoint-centric: a run writes
`checkpoint_fitted.ckpt` under its own output directory and stops there.
Turning that into a surface is `flatten_spiral_checkpoint.py`'s job --
upstream's own tool, not something reimplemented here -- and it does two
things a training run should not also be doing: reconstructing the combined
spiral surface on the GPU, then starting a private Lasagna service to flatten
it. Both steps are refused, run and reported the same way, so a fit that
finishes and an export that fails are two different receipts rather than one
run silently missing its output.

Step 6 is what makes this a P1 backend rather than a script that writes
meshes. The exported surface is registered exactly the way a grown surface
is -- same artifact store, same content address, same `import_surface` -- so
it enters P2's geometry certification and the lamina gate beside it without a
parallel path.

What it does not do
-------------------
It does not certify. It does not decide the fit was good. A surface that
comes out of here has a provenance, and every gate downstream applies to it
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
STAGE = ROOT / "framework/stages/01-segmentation"
PROFILE_DIR = ROOT / "framework/profiles/01-segmentation"
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(STAGE / "backends/spiral"))

import adapter  # noqa: E402

RECEIPT = "SPIRAL_FIT_RECEIPT.json"
CHECKPOINT_NAME = "checkpoint_fitted.ckpt"
TIFXYZ = ("x.tif", "y.tif", "z.tif")


def staged_manifest(dataset: Path) -> dict[str, Any] | None:
    """What the stager recorded about this dataset, when there is a stager."""
    path = Path(dataset) / "SPIRAL_DATASET.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def staging_verify(dataset: Path) -> dict[str, Any]:
    from stage_spiral_dataset import verify  # noqa: PLC0415

    return verify(Path(dataset))


# How long a cancelled subprocess gets to close what it has open.
TERMINATE_GRACE_SECONDS = 20.0


class SpiralRunRefused(RuntimeError):
    """The run cannot be made meaningful, said before it costs anything."""


def resolve_profile(profile_id: str) -> tuple[Path, dict[str, Any]]:
    """Find the frozen profile with this id, in this repository.

    By id rather than by path on purpose: a caller-supplied path is a way to
    run against a profile nobody froze, and the receipt would record an id it
    did not read.
    """
    for candidate in sorted(PROFILE_DIR.glob("*.json")):
        try:
            profile = adapter.load_spiral_profile(candidate)
        except ValueError:
            continue  # not a segmentation backend profile; the directory holds others
        if profile.get("profile_id") == profile_id:
            return candidate, profile
    raise SpiralRunRefused(
        f"no frozen profile in {PROFILE_DIR.relative_to(ROOT)} has id "
        f"{profile_id!r}")


def fitter_root() -> Path:
    """Where the image put upstream's `spiral-fitting`.

    The fallback is what the image sets, kept in step with it: lanes moved
    from /opt/villa to /opt/lanes/<name> when one worker had to carry more
    than one, and a fallback left behind points at a directory no image has.
    It only ever applies when the variable is unset, which is the case that
    reports the least, so it has to be right rather than merely present.
    """
    root = os.environ.get("VILLA_SPIRAL_ROOT") or "/opt/lanes/spiral/spiral-fitting"
    path = Path(root)
    if not (path / "fit_spiral.py").is_file():
        raise SpiralRunRefused(
            f"{path}/fit_spiral.py is not here. This lane runs in "
            "helena-villa-python, which is the image that carries the fitter; "
            "a worker in another runtime cannot run it.")
    return path


def lasagna_root(root: Path) -> Path:
    """Where upstream's own `flatten_spiral_checkpoint.py` looks for Lasagna
    when nothing names it: a sibling of the fitter's own root. Matches this
    image's layout (`/opt/villa/spiral-fitting` beside `/opt/villa/lasagna`)
    so this needs no environment variable of its own -- named explicitly
    anyway, on this platform's own rule that a setting typed nowhere is a
    setting no receipt records.
    """
    candidate = root.parent / "lasagna"
    if not (candidate / "fit_service.py").is_file():
        raise SpiralRunRefused(
            f"{candidate}/fit_service.py is not here. The export step starts "
            "a private Lasagna service from it to flatten the fitted surface.")
    return candidate


def survey_inputs(dataset: Path, profile: dict[str, Any],
                  layout: dict[str, Any]) -> dict[str, Any]:
    """What this dataset holds, split by what an absence costs.

    Read from upstream's fit_session.FIT_INPUT_CATALOG at the pinned commit,
    not the tutorial: several inputs are optional there (a missing point
    collection or fiber file loads as nothing and the fit runs), and this
    platform's own frozen profile additionally disables patches and outer-
    shell losses and points dense-spacing at gradient magnitude rather than
    the newer winding-model mode -- see the profile's `notes` for why. What
    remains required under that configuration is exactly what was required
    before this commit: the umbilicus, the tracks, and the three lasagna
    arrays.
    """
    def resolve(path: str) -> str:
        # `{lasagna_volume_name:nx}` names the volume set and which array of
        # it, because the three are one setting and three paths.
        for array in ("nx", "ny", "grad_mag"):
            token = "{lasagna_volume_name:%s}" % array
            if token in path:
                return path.replace(token, str(
                    layout["lasagna_volume_name"]).format(array=array))
        return path.format(**layout)

    def present(path: str) -> bool:
        # A path may still be a glob; a directory is a present .ome.zarr.
        if any(character in path for character in "*?["):
            return bool(list(dataset.glob(path)))
        return (dataset / path).exists()

    declared = profile.get("inputs") or {}
    missing_required = [resolve(entry["path"])
                        for entry in declared.get("required") or ()
                        if not present(resolve(entry["path"]))]
    absent_optional = [
        {"path": resolve(entry["path"]), "costs": entry.get("absence_costs")}
        for entry in declared.get("optional") or ()
        if not present(resolve(entry["path"]))
    ]
    return {
        "path": str(dataset),
        "missing_required": missing_required,
        "absent_optional": absent_optional,
        "degraded": bool(absent_optional),
    }


def find_surface(export_dir: Path) -> Path | None:
    """The exported TIFXYZ, identified by content -- x, y and z planes --
    rather than by the fixed name this runner asked the export step for,
    since a naming disagreement should surface as "not found" rather than a
    silent pass on the wrong directory.
    """
    if all((export_dir / name).is_file() for name in TIFXYZ):
        return export_dir
    return None


def register_fitted_surface(store: Any, artifact_store: Any, directory: Path, *,
                            sample_id: str, snapshot: dict[str, Any],
                            voxel_size_um: float, run_id: str,
                            lineage: dict[str, Any]) -> dict[str, Any]:
    """Publish the exported surface and record it as a surface.

    Deliberately the same three steps the fleet's finalizer takes for a grown
    surface: inspect, content-address, promote. A second way of getting a
    surface into the catalogue is a second set of rules about what a surface
    is, and the certification gate would then be measuring two things.

    Named for what it does now rather than kept as `register_winding`: at
    this commit one fit produces one combined, flattened surface spanning a
    range of windings (upstream's own `output_first_winding` through
    `model_gap_expander_num_windings - 1` by default), not one TIFXYZ per
    winding. The registration mechanics below did not need to change; only
    how many times a run calls this did, and what to call it.
    """
    from fleet.common import artifact_manifest, content_sha256, stable_id  # noqa: PLC0415
    from fleet.finalizer import inspect_tifxyz  # noqa: PLC0415

    # The staging id becomes one path component in the artifact store. Job ids
    # come from the queue and are safe, which is exactly why an unchecked one
    # would go unnoticed the day something else calls this.
    if run_id != Path(run_id).name or run_id in ("", ".", ".."):
        raise SpiralRunRefused(
            f"{run_id!r} is not a single safe name, and it is used as a path "
            "component in the artifact store")

    names = tuple(name for name in (*TIFXYZ, "meta.json")
                  if (directory / name).is_file())
    inspection = inspect_tifxyz(directory, voxel_size_um)
    files = artifact_manifest(directory, names)
    artifact_sha = content_sha256(files)
    manifest = {
        "schema": "campaignx.segmentation_artifact_set.v1",
        "files": files,
        "artifact_sha256": artifact_sha,
        **inspection,
        "ink_used": False,
        "produced_by": lineage,
    }
    surface_id = stable_id("surface", {
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "artifact_sha256": artifact_sha})
    staged = artifact_store.stage(directory, f"spiral-{run_id}-{surface_id}", manifest)
    promotion = artifact_store.promote(staged, sample_id, surface_id, manifest)
    payload = {
        "surface_id": surface_id,
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "sample_id": sample_id,
        # Not "imported": these bytes were produced here, by a run whose whole
        # provenance is on the record. Calling them imported would put them in
        # the bucket the catalogue keeps for surfaces from outside.
        "owner": "campaign-x",
        "artifact_sha256": artifact_sha,
        "artifact_uri": promotion["artifact_uri"],
        "bbox_xyz": inspection["bbox_xyz"],
        "sample_points": inspection.get("sample_points"),
        "area_cm2": inspection.get("area_cm2"),
        "state": "IMPORTED",
        "physical_qc_state": "UNVALIDATED",
        **lineage,
    }
    store.import_surface(payload)
    return {"surface_id": surface_id, "artifact_sha256": artifact_sha,
            "artifact_uri": promotion["artifact_uri"],
            "area_cm2": inspection.get("area_cm2"),
            "surface_dir": str(directory)}


def run_subprocess(argv: list[str], *, cwd: Path, environment: dict[str, str]) -> int:
    """Run one subprocess, and take it down when this process is asked to
    stop.

    Shared by the fit and the export step: the worker cancels a job by
    terminating its runner, and `subprocess.run` would leave either one
    running -- SIGTERM's default action ends this process, the child is
    reparented, and the card stays held by something no lease covers and
    nothing will ever reap. So the signal is caught and passed on.
    """
    child = subprocess.Popen(argv, cwd=str(cwd), env=environment)  # noqa: S603
    kill_after: list[float | None] = [None]

    def stop(_signal_number, _frame):  # noqa: ANN001 - a signal handler's shape
        # Terminate and note the deadline. Nothing else: a handler runs on the
        # main thread, which is inside `child.wait()`, and waiting again from
        # in here cannot reap the child -- the outer wait holds the lock, so
        # the inner one spins until its own timeout and cancellation takes
        # twenty seconds instead of one. The escalation is the loop's job.
        try:
            child.terminate()
        except ProcessLookupError:  # already gone, and already being reaped
            return
        kill_after[0] = time.monotonic() + TERMINATE_GRACE_SECONDS

    # The handler does not raise. Dying here would leave the run with no
    # receipt at all, and a cancelled run is exactly the case where the
    # record of what was asked for and what was written is worth having: the
    # wait below returns the negative signal number, the caller files it as
    # cancelled, and the receipt is written on the way out like any other
    # outcome.
    previous = {number: signal.signal(number, stop)
                for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            try:
                return child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if kill_after[0] is not None and time.monotonic() > kill_after[0]:
                    child.kill()
                    kill_after[0] = None
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


# Backward-compatible alias: `run_fit` is what the fit-cancellation contract
# was verified against, and the export step shares the identical contract
# rather than a copy of it.
run_fit = run_subprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True,
                        help="the frozen spiral profile this run is made against")
    parser.add_argument("--out", required=True, help="where this run writes")
    parser.add_argument("--sample", required=True,
                        help="the scroll as this platform names it, for the catalogue")
    # The scroll binding. Named after Helena's own parameters, which have not
    # changed shape even though the mechanism that delivers them has: see
    # adapter.py's module docstring.
    parser.add_argument("--scroll-name", help="the scroll as fit_spiral.py names it")
    parser.add_argument("--dataset-path", help="the fitter's dataset directory")
    parser.add_argument("--z-begin", type=int)
    parser.add_argument("--z-end", type=int)
    parser.add_argument("--voxel-um", type=float)
    parser.add_argument("--winding-sense", choices=sorted(adapter.WINDING_SENSES))
    # The dataset layout: which lasagna volume set, at which zarr group and
    # scale, and which tracks file. Defaulted to upstream's own values, so a
    # run that says nothing reproduces upstream.
    parser.add_argument("--lasagna-volume-name",
                        help="one name under lasagna_inputs/, carrying {array} "
                             "for normal_x, normal_y and gradient_magnitude")
    parser.add_argument("--normal-zarr-group",
                        help="which level inside those zarrs the fit reads")
    parser.add_argument("--lasagna-scale", type=int,
                        help="the divisor the shape check uses; derived as "
                             "2 ** group when the layout moves and this is not "
                             "stated")
    parser.add_argument("--tracks-file",
                        help="the .dbm under tracks/ the fit reads")
    parser.add_argument("--random-seed", type=int,
                        help="upstream's own optimizer_random_seed. Two fits "
                             "differing only in this are the only error bar "
                             "this geometry has, so the pair is named rather "
                             "than implied")
    parser.add_argument("--run-tag", help="upstream's own label for this run")
    parser.add_argument("--cache-dir", help="defaults to cache/ inside the run")
    # Deliberately not reachable from a queued job: the queue builds this argv
    # and never sends it. A parameter that named the fitter's location would
    # be a way to run arbitrary code under this profile's identity.
    parser.add_argument("--fitter-root",
                        help="the spiral-fitting checkout to run, when it is "
                             "not the image's own; used by the tests, never "
                             "by the queue")
    parser.add_argument("--artifact-store",
                        help="where the exported surface is published")
    parser.add_argument("--db", help="control plane the surface is registered in")
    parser.add_argument("--mission-id", help="the mission this surface belongs to")
    parser.add_argument("--requested-by-job-id",
                        help="the queue job this run answers")
    parser.add_argument("--dry-run", action="store_true",
                        help="preflight and stop before the fit")
    args = parser.parse_args(argv)

    requested_layout = {
        name: value for name, value in
        (("lasagna_volume_name", args.lasagna_volume_name),
         ("normal_zarr_group", args.normal_zarr_group),
         ("lasagna_scale", args.lasagna_scale),
         ("tracks_file", args.tracks_file)) if value is not None}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "campaignx.spiral_fit_receipt.v2",
        "profile_id": args.profile_id,
        "sample_id": args.sample,
        "mission_id": args.mission_id,
        "requested_by_job_id": args.requested_by_job_id,
        "dry_run": bool(args.dry_run),
    }

    def finish(code: int) -> int:
        (out_dir / RECEIPT).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return code

    try:
        profile_path, profile = resolve_profile(args.profile_id)
        adapter.require_runnable_profile(profile)
        receipt["profile_path"] = str(profile_path.relative_to(ROOT))

        binding = adapter.scroll_binding_for(profile, {
            "scroll_name": args.scroll_name, "dataset_path": args.dataset_path,
            "z_begin": args.z_begin, "z_end": args.z_end,
            "voxel_size_um": args.voxel_um,
            "spiral_outward_sense": args.winding_sense})
        receipt["binding"] = binding
        receipt["binding_sha256"] = adapter.binding_sha256(binding)

        dataset = Path(binding["dataset_path"])
        if not dataset.is_dir():
            raise SpiralRunRefused(
                f"the dataset directory {dataset} is not here. The fitter reads "
                "everything it needs from under this path and creates none of it.")
        # A staged dataset carries what it is. Its layout wins over a default,
        # because the level the stager fetched is the level the fit has to read
        # -- and the request wins over both, so a deliberate override still
        # works and is visible in the receipt beside what it overrode.
        staged = staged_manifest(dataset)
        if staged:
            receipt["dataset_manifest"] = {
                "scroll": staged.get("scroll"), "scan": staged.get("scan"),
                "files": len(staged.get("files") or {})}
            requested_layout = {**(staged.get("layout") or {}), **requested_layout}
        layout = adapter.validate_layout(requested_layout)
        receipt["dataset_layout"] = layout
        survey = survey_inputs(dataset, profile, layout)
        if staged:
            # Presence is not wholeness. A .dbm cut short exists, opens, and
            # yields fewer tracks; the size against the manifest is what says so,
            # and it costs a stat per file rather than a minute of hashing.
            report = staging_verify(dataset)
            survey["manifest_check"] = report
            if not report["whole"]:
                raise SpiralRunRefused(
                    f"{dataset} does not match the manifest it was staged with: "
                    f"{report['wrong'][:3]}. A file that is present and the "
                    "wrong length is the failure this check exists for.")
        else:
            survey["manifest_check"] = {
                "whole": None,
                "why": ("this dataset carries no SPIRAL_DATASET.json, so it was "
                        "assembled by hand and nothing here can say it is "
                        "whole; stage it with stage_spiral_dataset.py")}
        receipt["dataset"] = survey
        if survey["missing_required"]:
            raise SpiralRunRefused(
                f"{dataset} is missing {survey['missing_required']}. The fit "
                "opens these directly and raises on any of them; refused here "
                "it costs no lease.")
        if survey["degraded"]:
            # Not a refusal. Said in the receipt because a reader comparing two
            # fits needs to know they were not given the same evidence.
            print("running without " + ", ".join(
                entry["path"] for entry in survey["absent_optional"]),
                file=sys.stderr)

        root = Path(args.fitter_root) if args.fitter_root else fitter_root()

        scroll_spec_path = out_dir / "spiral-scroll.json"
        written = adapter.write_scroll_spec(binding, layout, scroll_spec_path)
        receipt["scroll_spec"] = written

        cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fit_dir = out_dir / "fit"
        fit_dir.mkdir(parents=True, exist_ok=True)
        # The seed is one of upstream's own Config keys, so it rides the
        # override channel -- and it is recorded, because a pair of fits
        # nobody can tell apart is not a pair.
        seeded = dict(profile)
        if args.random_seed is not None:
            # The name is read from the commit that is about to run, not from
            # a profile: upstream has renamed this key before, and a seed
            # override that reaches nothing produces two identical fits and
            # an agreement of zero -- which reads as perfect reproducibility.
            key = adapter.seed_key(root)
            seeded["config_overrides"] = {
                **(profile.get("config_overrides") or {}),
                key: int(args.random_seed)}
            receipt["random_seed"] = {"value": int(args.random_seed),
                                      "config_key": key}
        z_range_overrides = {"z_begin": int(binding["z_begin"]),
                             "z_end": int(binding["z_end"])}
        unknown_z_keys = set(z_range_overrides) - adapter.spiral_config_keys(root)
        if unknown_z_keys:  # pragma: no cover - defensive; verified at the pinned commit
            raise SpiralRunRefused(
                f"upstream's Config has no {sorted(unknown_z_keys)}; the z "
                "range can no longer be set through the override channel")
        seeded["config_overrides"] = {
            **z_range_overrides, **(seeded.get("config_overrides") or {})}
        environment = adapter.spiral_environment(
            seeded, fitter_root=root, out_dir=fit_dir, cache_dir=cache_dir,
            run_tag=args.run_tag)
        # Only the variables this run set. The rest of the environment is the
        # container's and putting it in a receipt would publish whatever
        # happens to be exported on the host.
        receipt["environment"] = {name: value for name, value in environment.items()
                                  if name.startswith(("FIT_SPIRAL_", "WANDB_"))}

        if args.dry_run:
            receipt["outcome"] = "PREFLIGHT_ONLY"
            receipt["note"] = ("everything up to the fit succeeded: the profile "
                               "resolves, the binding is complete, the dataset "
                               "holds every named input, and spiral-scroll.json "
                               "read back as what was written")
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return finish(0)

        interpreter = os.environ.get("VILLA_PYTHON") or sys.executable
        fit_argv = [interpreter, "fit_spiral.py",
                   "--dataset", str(dataset),
                   "--scroll-spec", str(scroll_spec_path),
                   "--cache", str(cache_dir)]
        returncode = run_subprocess(fit_argv, cwd=root, environment=environment)
        receipt["fit"] = {"argv": fit_argv, "returncode": returncode}
        if returncode != 0:
            receipt["outcome"] = ("FIT_CANCELLED" if returncode < 0
                                  else "FIT_FAILED")
            return finish(abs(returncode))

        checkpoint = fit_dir / CHECKPOINT_NAME
        if not checkpoint.is_file():
            receipt["outcome"] = "NO_CHECKPOINT"
            receipt["note"] = (
                f"the fit exited zero and wrote no {CHECKPOINT_NAME} under "
                f"{fit_dir}. Nothing is exported: a run with no checkpoint is "
                "not a run with an empty one.")
            return finish(1)

        # Step 5: the fitter writes checkpoints, not surfaces, at this commit.
        # Upstream's own flatten_spiral_checkpoint.py reconstructs the
        # combined spiral surface from the checkpoint and flattens it through
        # a private Lasagna service; neither step is reimplemented here.
        export_dir = out_dir / "export" / "spiral.tifxyz"
        export_argv = [interpreter, "flatten_spiral_checkpoint.py",
                       str(checkpoint), str(export_dir),
                       "--umbilicus", str(dataset / "umbilicus.json"),
                       "--lasagna-dir", str(lasagna_root(root)),
                       "--device", "cuda",
                       "--voxel-size-um", str(float(binding["voxel_size_um"]))]
        export_env = dict(environment)
        export_returncode = run_subprocess(export_argv, cwd=root, environment=export_env)
        receipt["export"] = {"argv": export_argv, "returncode": export_returncode}
        if export_returncode != 0:
            receipt["outcome"] = ("EXPORT_CANCELLED" if export_returncode < 0
                                  else "EXPORT_FAILED")
            receipt["note"] = (
                "the fit produced a checkpoint but flatten_spiral_checkpoint.py "
                "did not produce a surface from it; the checkpoint is left in "
                f"place at {checkpoint} for a retried export")
            return finish(abs(export_returncode))

        surface = find_surface(export_dir)
        receipt["surface_found"] = surface is not None
        if surface is None:
            receipt["outcome"] = "NO_SURFACE"
            receipt["note"] = (
                "the export exited zero and wrote no TIFXYZ (x.tif, y.tif, "
                f"z.tif) under {export_dir}. Nothing is registered: a run "
                "with no surface is not a run with an empty one.")
            return finish(1)

        if not (args.db and args.artifact_store):
            receipt["outcome"] = "FITTED_NOT_REGISTERED"
            receipt["surface"] = str(surface)
            receipt["note"] = ("no control plane and artifact store were given, "
                               "so the surface stays on this worker's disk and "
                               "P2 will never see it")
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return finish(0)

        from fleet.artifact_store import open_artifact_store  # noqa: PLC0415
        from fleet.store_factory import open_fleet_store  # noqa: PLC0415

        store = open_fleet_store(args.db)
        snapshots = list(store.snapshots({args.sample}))
        if not snapshots:
            raise SpiralRunRefused(
                f"{args.sample} has no registered source snapshot, so a fitted "
                "surface has nothing to hang its provenance on. Register the "
                "scroll's source before fitting it.")
        snapshot = snapshots[0]
        artifact_store = open_artifact_store(args.artifact_store)
        lineage = {
            "produced_by_backend": "spiral",
            "produced_by_profile_id": profile["profile_id"],
            "spiral_binding_sha256": receipt["binding_sha256"],
            "spiral_scroll_spec_sha256": written["sha256"],
            "requested_by_job_id": args.requested_by_job_id,
            "mission_id": args.mission_id,
        }
        registered = register_fitted_surface(
            store, artifact_store, surface,
            sample_id=args.sample, snapshot=snapshot,
            voxel_size_um=float(binding["voxel_size_um"]),
            run_id=str(args.requested_by_job_id or "adhoc"),
            lineage=lineage)
        receipt["registered"] = registered
        receipt["outcome"] = "REGISTERED"
        receipt["wrote_to"] = str(out_dir)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return finish(0)
    except (SpiralRunRefused, ValueError, adapter.ScrollSpecRefused) as refusal:
        receipt["outcome"] = "REFUSED"
        receipt["reason"] = f"{type(refusal).__name__}: {refusal}"
        print(receipt["reason"], file=sys.stderr)
        return finish(2)


if __name__ == "__main__":
    raise SystemExit(main())
