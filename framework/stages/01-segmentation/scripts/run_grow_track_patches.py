#!/usr/bin/env python3
"""Grow track-graph patches on an already-staged scroll, headlessly.

The published `verified_patches` villa ships (PHercParis4's own examples) are
for a scroll this platform does not target. Villa's own recipe for growing
patches on a different, already-tracked scroll is `Actions > Grow track
patches` in VC3D's GUI -- which blocks any queued or automated use. This
runner exists because that GUI action turns out to have a real headless
counterpart underneath it: `grow_track_graph.py`, in the same pinned
`spiral-fitting/` package `run_spiral_fit.py` already runs, with a complete
`argparse` CLI (`tracks`, `crossings`, `output`, `--seeds`/`--random-count`).
Nothing here is GUI-only.

What upstream's own CLI wants is not what this platform stages, though. Three
scripts sit between them, confirmed by reading each one rather than assumed
from its name:

    1. `convert_track_store.py` -- the *same* `tracks/{tracks_file}` DBM this
       platform already stages (`load_tracks_from_dbm` opens it with
       `dbm.open`, the same call the fit itself makes) into a packed,
       memory-mapped `.vctracks` directory beside it. Pure Python: `dbm`,
       `pickle`, `numpy`, no GPU.
    2. `build_track_crossings.py` -- the *same* DBM again, independently, into
       an uncompressed `.crossings.npz` CSR sidecar beside it. It opportunis-
       tically reads the `.vctracks` store from step 1 when one is already
       current (`_packed_store_if_current`), but does not require it. Its
       `source_ids` are computed by the identical `(key_ordinal << 32) |
       local_index` walk over `sorted(dbm.keys())` that step 1 uses, over the
       same source file -- confirmed by reading both, not assumed from the
       shared naming -- so `CrossingCsr`'s own alignment check against the
       packed store's `source_ids` passes. Uses a compiled `vc_spiral`
       kernel when the image provides one (it does: the same nanobind
       extension `fit_spiral.py`'s own crossing lookup uses) and falls back
       to a slower pure-Python path when it does not; neither path touches a
       GPU.
    3. `grow_track_graph.py` -- reads the packed store and the crossings
       sidecar (`PackedTracks`, `CrossingCsr`), grows one `.tifxyz` patch per
       requested seed by graph-topological rail consensus, and writes each
       under its `output` directory as `band-seed<N>-<timestamp>.tifxyz`.
       Pure `numpy`/stdlib -- no `torch`, no `cuda`, no subprocess to a
       compiled VC3D binary anywhere in the file.

None of the three needs a GPU. `tracks.py`, which the first two import for
`write_packed_track_store`/`build_crossing_partner_csr_disk_backed`, imports
`torch` at module level for functions elsewhere in that file, but the two
functions this runner actually calls are pure `numpy`/`dbm`/`pickle`/`struct`
and call no CUDA op -- confirmed by reading both functions in full. So `torch`
must be importable in the subprocess (it already is: this runs in
`helena-villa-python`, the same interpreter `fit_spiral.py` uses), but no
worker needs a card claimed to run this lane. See the frozen profile's own
`notes.gpu` for the citation trail.

Where the output goes
----------------------
Grown patches are not registered as surfaces. They are an *input* to a future
fit, not a certified, content-addressed artifact of their own -- there is no
`import_surface` call here, and P2 never sees them directly. They are written
straight into the staged dataset's own `unverified_patches_dir` (default
`unverified_patches/`, the same shared-per-scroll cache
`stage_spiral_dataset.py` already maintains for tracks and lasagna), so that a
subsequent `spiral-fitter-v1@0.4.1` run naming the same
`--unverified-patches-dir` finds them without a copy step. Upstream's own
`_load_patches_from_dir` treats every entry in that directory as one patch
(`os.listdir`), so nothing renames or repackages what `grow_track_graph.py`
already wrote.

They are deliberately called *unverified*, not *verified*: nothing here
attaches a human review, and `verified_patches` (per `fit_session.py`'s own
`FIT_INPUT_CATALOG`) is `required` the instant it is enabled, gates
`abs_winding.json`'s absolute-winding annotations, and is trusted outright by
the fit's own near-trusted-geometry masking of unverified patches. Calling
grown, unreviewed geometry "verified" would misuse all three.

What it does not do
--------------------
It does not decide whether patches help. A villa contributor (pmh47,
2026-08-24) flagged that track-graph patches are "currently strangely
unhelpful in the spiral" despite being high-quality patches in themselves --
this runner exists to make that measurable (against
`spiral-fitter-v1@0.4.1`), not to answer it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
STAGE = ROOT / "framework/stages/01-segmentation"
PROFILE_DIR = ROOT / "framework/profiles/01-segmentation"
sys.path.insert(0, str(STAGE))
sys.path.insert(0, str(STAGE / "backends/spiral"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapter  # noqa: E402
from run_spiral_fit import run_subprocess  # noqa: E402 -- the shared cancellation contract

RECEIPT = "GROW_TRACK_PATCHES_RECEIPT.json"
SCHEMA = "campaignx.grow_track_patches_profile.v1"

# grow_track_graph.py's own CLI flags this runner may pass through from a
# profile's `growth_overrides`, mapped to how the value becomes an argv
# token. `store_true` flags take no value; everything else is `str(value)`.
# Built from the pinned commit's own argparse block (grow_track_graph.py,
# `def main`), not assumed -- see the frozen profile's `notes.growth_overrides`
# for the verified default of each.
GROWTH_FLAGS: dict[str, str] = {
    "random_top_percent": "--random-top-percent",
    "min_valid_vertices": "--min-valid-vertices",
    "growth_min_span": "--growth-min-span",
    "min_connect": "--min-connect",
    "min_size": "--min-size",
    "max_size": "--max-size",
    "max_thick_cell_frac": "--max-thick-cell-frac",
    "reject_any_fold_fixes": "--reject-any-fold-fixes",
    "gate_tol": "--gate-tol",
    "resample_spacing": "--resample-spacing",
    "min_track_arclength": "--min-track-arclength",
    "output_spacing": "--output-spacing",
    "border_erode_vx": "--border-erode-vx",
    "workers": "--workers",
    "overwrite": "--overwrite",
}
GROWTH_FLAG_STORE_TRUE = frozenset({"reject_any_fold_fixes", "overwrite"})


class GrowRunRefused(RuntimeError):
    """The run cannot be made meaningful, said before it costs anything."""


def resolve_profile(profile_id: str) -> tuple[Path, dict[str, Any]]:
    """Find the frozen grow-track-patches profile with this id.

    Mirrors run_spiral_fit.resolve_profile's reasoning exactly, but against
    this lane's own schema: a caller-supplied path would let a request run
    against a profile nobody froze, and the receipt would record an id it did
    not read.
    """
    for candidate in sorted(PROFILE_DIR.glob("*.json")):
        try:
            document = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("schema") != SCHEMA:
            continue  # not a grow-track-patches profile; the directory holds others
        if document.get("profile_id") == profile_id:
            required = ("profile_id", "growth_overrides", "dataset_layout")
            missing = [key for key in required if key not in document]
            if missing:
                raise GrowRunRefused(
                    f"{candidate} is missing {missing}: not a runnable "
                    "grow-track-patches profile")
            return candidate, document
    raise GrowRunRefused(
        f"no frozen profile in {PROFILE_DIR.relative_to(ROOT)} has id "
        f"{profile_id!r} under schema {SCHEMA!r}")


def fitter_root() -> Path:
    """Where the image put upstream's `spiral-fitting`.

    Identical to run_spiral_fit.fitter_root: these three scripts are siblings
    of fit_spiral.py in the same pinned checkout, so the same image, the same
    fallback path and the same refusal apply unchanged.
    """
    root = os.environ.get("VILLA_SPIRAL_ROOT") or "/opt/lanes/spiral/spiral-fitting"
    path = Path(root)
    if not (path / "grow_track_graph.py").is_file():
        raise GrowRunRefused(
            f"{path}/grow_track_graph.py is not here. This lane runs in "
            "helena-villa-python, which is the image that carries the fitter; "
            "a worker in another runtime cannot run it.")
    return path


def _track_db_signature(tracks_dbm: Path) -> list[tuple[str, int, int]]:
    """Fingerprint the DBM's backing files, replicating `tracks.py`'s private
    `_tracks_db_signature` (name, size, mtime_ns per existing candidate file,
    sorted) so a `.vctracks` store already beside it can be trusted or
    distrusted without importing `tracks.py` (which pulls in torch/kornia at
    module level for code this runner never calls -- see the module
    docstring). Re-derive this from the pinned commit's `tracks.py` if it is
    ever out of date; a signature that no longer matches upstream's own only
    costs an unnecessary re-pack, never a wrong one, since a mismatch always
    falls back to reconverting from the DBM.
    """
    candidates = [tracks_dbm, *(tracks_dbm.with_name(tracks_dbm.name + suffix)
                                for suffix in (".db", ".dat", ".dir", ".bak", ".pag"))]
    seen: set[Path] = set()
    result: list[tuple[str, int, int]] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.is_file():
                continue
            seen.add(resolved)
            stat = candidate.stat()
            result.append((candidate.name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            continue
    return sorted(result)


def vctracks_is_current(tracks_dbm: Path) -> bool:
    """Whether `<tracks_dbm>.vctracks` already reflects this exact DBM.

    `write_packed_track_store` (via `convert_track_store.py`) refuses to
    overwrite an existing store unless told `--force`; it has no self-skip
    the way the crossings cache does. So this runner checks freshness itself,
    against the same `source_db_signature` the packed store's own
    `metadata.json` already carries -- reading it costs a stat and a small
    JSON parse, not a re-pack of a multi-gigabyte DBM.
    """
    store = tracks_dbm.with_name(tracks_dbm.name + ".vctracks")
    metadata_path = store / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = [list(item) for item in _track_db_signature(tracks_dbm)]
    return metadata.get("source_db_signature") == expected


def growth_argv(script: str, overrides: dict[str, Any]) -> list[str]:
    """Translate a profile's `growth_overrides` into `grow_track_graph.py`
    flags, refusing an unknown key before a GPU-less-but-still-real subprocess
    is spent on a config nobody meant.
    """
    unknown = sorted(set(overrides) - set(GROWTH_FLAGS))
    if unknown:
        raise GrowRunRefused(
            f"{script} has no growth override(s) {unknown}; this lane knows "
            f"{sorted(GROWTH_FLAGS)}")
    argv: list[str] = []
    for name, value in overrides.items():
        flag = GROWTH_FLAGS[name]
        if name in GROWTH_FLAG_STORE_TRUE:
            if value:
                argv.append(flag)
            continue
        argv += [flag, str(value)]
    return argv


def written_patch_dirs(patches_dir: Path) -> list[str]:
    if not patches_dir.is_dir():
        return []
    return sorted(entry.name for entry in patches_dir.iterdir() if entry.is_dir())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True,
                        help="the frozen grow-track-patches profile this run "
                             "is made against")
    parser.add_argument("--out", required=True,
                        help="where this run writes its own receipt")
    parser.add_argument("--dataset-path", required=True,
                        help="the staged spiral dataset directory (the same "
                             "one a spiral-fit job names)")
    parser.add_argument("--tracks-file",
                        help="the .dbm under tracks/ to grow patches from; "
                             "defaults to adapter.DEFAULT_TRACKS_FILE")
    parser.add_argument("--unverified-patches-dir",
                        help="one directory directly under the dataset root "
                             "grown patches are written into; defaults to "
                             "the profile's own dataset_layout default")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="explicit track rows to grow (one patch per "
                             "seed); exactly one of --seeds/--seeds-json or "
                             "--random-count is required")
    parser.add_argument("--seeds-json",
                        help="the same as --seeds, as a JSON array of ints -- "
                             "what the queue sends, since a lane's flags are "
                             "one value per flag (job_store.declarative_argv's "
                             "json_flags); do not pass both --seeds and "
                             "--seeds-json")
    parser.add_argument("--random-count", type=int,
                        help="grow this many patches from random unused "
                             "top-arclength seeds; exactly one of --seeds or "
                             "--random-count is required")
    parser.add_argument("--random-seed", type=int, default=0,
                        help="RNG seed for --random-count's selection")
    parser.add_argument("--fitter-root",
                        help="the spiral-fitting checkout to run, when it is "
                             "not the image's own; used by the tests, never "
                             "by the queue")
    parser.add_argument("--mission-id", help="the mission this run belongs to")
    parser.add_argument("--requested-by-job-id",
                        help="the queue job this run answers")
    parser.add_argument("--dry-run", action="store_true",
                        help="preflight and stop before growing anything")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "campaignx.grow_track_patches_receipt.v1",
        "profile_id": args.profile_id,
        "mission_id": args.mission_id,
        "requested_by_job_id": args.requested_by_job_id,
        "dry_run": bool(args.dry_run),
    }

    def finish(code: int) -> int:
        (out_dir / RECEIPT).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return code

    try:
        if args.seeds is not None and args.seeds_json is not None:
            raise GrowRunRefused(
                "pass --seeds or --seeds-json, not both -- they name the "
                "same thing two ways")
        seeds = args.seeds
        if args.seeds_json is not None:
            try:
                parsed = json.loads(args.seeds_json)
            except json.JSONDecodeError as malformed:
                raise GrowRunRefused(
                    f"--seeds-json is not valid JSON: {malformed}") from malformed
            if (not isinstance(parsed, list) or not parsed
                    or not all(isinstance(item, int) and not isinstance(item, bool)
                               for item in parsed)):
                raise GrowRunRefused(
                    f"--seeds-json must be a non-empty JSON array of ints, "
                    f"got {args.seeds_json!r}")
            seeds = parsed

        if (seeds is None) == (args.random_count is None):
            raise GrowRunRefused(
                "pass exactly one of --seeds/--seeds-json or --random-count; growing "
                "nothing and growing an explicit list plus a random quota "
                "are both requests this lane refuses to guess between")

        profile_path, profile = resolve_profile(args.profile_id)
        try:
            receipt["profile_path"] = str(profile_path.relative_to(ROOT))
        except ValueError:
            # Only reachable when a caller points --profile-id resolution at
            # a profile directory outside this repository (the tests do,
            # deliberately, to inject a profile without freezing one here).
            # The queue never does this: resolve_profile only ever globs
            # PROFILE_DIR, which is under ROOT.
            receipt["profile_path"] = str(profile_path)

        layout_defaults = profile.get("dataset_layout") or {}
        tracks_file = (args.tracks_file
                       or layout_defaults.get("tracks_file")
                       or adapter.DEFAULT_TRACKS_FILE)
        patches_dirname = (args.unverified_patches_dir
                           or layout_defaults.get("unverified_patches_dir"))
        if not patches_dirname:
            raise GrowRunRefused(
                "no unverified_patches_dir: pass --unverified-patches-dir or "
                "freeze one into the profile's dataset_layout -- there is no "
                "upstream default to fall back to (see adapter.py's "
                "DEFAULT_UNVERIFIED_PATCHES_DIR)")
        # Reuses adapter.validate_layout's own directory-name discipline
        # (no braces, no path separators, not . or ..) rather than a second
        # copy of the same three checks.
        checked = adapter.validate_layout({"unverified_patches_dir": patches_dirname,
                                           "tracks_file": tracks_file})
        tracks_file = checked["tracks_file"]
        patches_dirname = checked["unverified_patches_dir"]

        dataset = Path(args.dataset_path)
        if not dataset.is_dir():
            raise GrowRunRefused(
                f"the dataset directory {dataset} is not here; this lane "
                "reads an already-staged dataset and creates none of it")
        tracks_dbm = dataset / "tracks" / tracks_file
        if not tracks_dbm.is_file():
            raise GrowRunRefused(
                f"{tracks_dbm} is not here. This lane grows patches from the "
                "same tracks DBM a spiral fit reads; stage the dataset first "
                "with stage_spiral_dataset.py.")
        receipt["dataset_path"] = str(dataset)
        receipt["tracks_dbm"] = str(tracks_dbm)
        receipt["unverified_patches_dir"] = patches_dirname
        receipt["requested"] = ({"seeds": seeds} if seeds
                                else {"random_count": args.random_count,
                                      "random_seed": args.random_seed})

        root = Path(args.fitter_root) if args.fitter_root else fitter_root()
        interpreter = os.environ.get("VILLA_PYTHON") or sys.executable

        growth_overrides = profile.get("growth_overrides") or {}
        extra_growth_argv = growth_argv("grow_track_graph.py", growth_overrides)

        if args.dry_run:
            receipt["outcome"] = "PREFLIGHT_ONLY"
            receipt["note"] = ("the tracks DBM is here, the profile resolves, "
                               "and exactly one of --seeds/--random-count was "
                               "given; nothing has been converted, indexed or "
                               "grown")
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return finish(0)

        patches_dir = dataset / patches_dirname
        patches_dir.mkdir(parents=True, exist_ok=True)
        before = set(written_patch_dirs(patches_dir))

        # Step 1: pack the DBM, unless a store beside it already reflects
        # these exact bytes.
        vctracks = tracks_dbm.with_name(tracks_dbm.name + ".vctracks")
        reused_vctracks = vctracks_is_current(tracks_dbm)
        if reused_vctracks:
            receipt["vctracks"] = {"reused": True, "path": str(vctracks)}
        else:
            convert_argv = [interpreter, "convert_track_store.py",
                            str(tracks_dbm)]
            if vctracks.exists():
                convert_argv.append("--force")
            returncode = run_subprocess(convert_argv, cwd=root, environment=dict(os.environ))
            receipt["vctracks"] = {"reused": False, "argv": convert_argv,
                                   "returncode": returncode, "path": str(vctracks)}
            if returncode != 0:
                receipt["outcome"] = ("VCTRACKS_CANCELLED" if returncode < 0
                                      else "VCTRACKS_FAILED")
                return finish(abs(returncode))

        # Step 2: the crossings sidecar. build_track_crossings.py's own
        # build_cache() already skips the work when a valid one exists, so
        # this is unconditional -- rerunning it is cheap when there is
        # nothing to do.
        crossings_argv = [interpreter, "build_track_crossings.py", str(tracks_dbm)]
        crossings_returncode = run_subprocess(
            crossings_argv, cwd=root, environment=dict(os.environ))
        crossings_path = tracks_dbm.with_name(tracks_dbm.name + ".crossings.npz")
        receipt["crossings"] = {"argv": crossings_argv,
                                "returncode": crossings_returncode,
                                "path": str(crossings_path)}
        if crossings_returncode != 0:
            receipt["outcome"] = ("CROSSINGS_CANCELLED" if crossings_returncode < 0
                                  else "CROSSINGS_FAILED")
            return finish(abs(crossings_returncode))
        if not crossings_path.is_file():
            receipt["outcome"] = "NO_CROSSINGS"
            receipt["note"] = (
                f"build_track_crossings.py exited zero and wrote no "
                f"{crossings_path.name}; nothing is grown from a sidecar "
                "that is not there")
            return finish(1)

        # Step 3: grow.
        growth_argv_full = [interpreter, "grow_track_graph.py",
                            str(vctracks), str(crossings_path), str(patches_dir)]
        if seeds:
            growth_argv_full += ["--seeds", *[str(seed) for seed in seeds]]
        else:
            growth_argv_full += ["--random-count", str(args.random_count),
                                 "--random-seed", str(args.random_seed)]
        growth_argv_full += extra_growth_argv
        growth_returncode = run_subprocess(
            growth_argv_full, cwd=root, environment=dict(os.environ))
        after = set(written_patch_dirs(patches_dir))
        written = sorted(after - before)
        receipt["growth"] = {"argv": growth_argv_full,
                             "returncode": growth_returncode,
                             "written": written}
        receipt["patches_written"] = len(written)
        if growth_returncode != 0:
            if written:
                receipt["outcome"] = "GROWN_PARTIAL"
                receipt["note"] = (
                    f"grow_track_graph.py exited {growth_returncode} after "
                    f"writing {len(written)} of the requested patch(es); the "
                    "ones it wrote are left in place")
            else:
                receipt["outcome"] = ("GROWTH_CANCELLED" if growth_returncode < 0
                                      else "GROWTH_FAILED")
            return finish(abs(growth_returncode) if not written else 0)

        receipt["outcome"] = "GROWN"
        receipt["wrote_to"] = str(patches_dir)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return finish(0)
    except (GrowRunRefused, ValueError, adapter.ScrollSpecRefused) as refusal:
        receipt["outcome"] = "REFUSED"
        receipt["reason"] = f"{type(refusal).__name__}: {refusal}"
        print(receipt["reason"], file=sys.stderr)
        return finish(2)


if __name__ == "__main__":
    raise SystemExit(main())
