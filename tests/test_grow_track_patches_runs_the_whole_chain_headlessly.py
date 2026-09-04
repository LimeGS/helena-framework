"""Growing track-graph patches without VC3D's GUI.

The published `verified_patches` villa ships are PHercParis4's own examples,
which this campaign's target scrolls are ineligible for. Villa's own recipe
for growing patches on a different, already-tracked scroll is VC3D's GUI
action `Actions > Grow track patches` -- and a GUI-only step blocks any
queued or automated use.

It turns out not to be GUI-only underneath: `grow_track_graph.py`, in the
same pinned `spiral-fitting/` package `run_spiral_fit.py` already runs, has a
complete headless `argparse` CLI. What it wants is not what this platform
stages, though -- a packed `.vctracks` directory and an uncompressed
crossings `.npz`, not the `tracks/{tracks_file}` DBM this platform already
has. Three more of upstream's own scripts turn out to bridge that gap
(`convert_track_store.py`, `build_track_crossings.py`), confirmed by reading
each one rather than assumed from its name -- in particular that the two
scripts' `source_ids` arrays are computed by the identical walk over the same
DBM, which is the only reason `CrossingCsr`'s own alignment check against a
packed store ever passes.

The fakes below stand in for all three upstream scripts. They are not a
claim about what the real ones compute -- only about the files each leaves
behind, the argv each accepts and the exit codes each can return, which is
everything `run_grow_track_patches.py`'s own logic depends on. Every test
here is about that logic: what refuses before a subprocess runs, what reuses
an existing `.vctracks` store instead of re-packing a multi-gigabyte DBM,
where patches land, and what a receipt says either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/backends/spiral"))

import run_grow_track_patches as runner  # noqa: E402

PROFILE_ID = "grow-track-patches-v1@0.1.0"

CONVERT_TRACK_STORE = '''\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tracks_dbm")
parser.add_argument("-o", "--output")
parser.add_argument("--force", action="store_true")
parser.add_argument("--quiet", action="store_true")
args = parser.parse_args()

dbm_path = Path(args.tracks_dbm)
destination = Path(args.output) if args.output \\
    else dbm_path.with_name(dbm_path.name + ".vctracks")
if destination.exists() and not args.force:
    raise SystemExit(f"{destination} exists; pass --force")
destination.mkdir(parents=True, exist_ok=True)
stat = dbm_path.stat()
signature = [[dbm_path.name, stat.st_size, stat.st_mtime_ns]]
(destination / "metadata.json").write_text(json.dumps(
    {"version": 1, "source_db_signature": signature,
     "track_count": 1, "point_count": 1}))
(destination / "coordinates.i32").write_bytes(b"")
print(f"packed store ready at {destination}")
'''

BUILD_TRACK_CROSSINGS = '''\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tracks_dbm")
parser.add_argument("--force", action="store_true")
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--z-min", type=int)
parser.add_argument("--z-max", type=int)
parser.add_argument("--temp-dir")
args = parser.parse_args()

dbm_path = Path(args.tracks_dbm)
destination = dbm_path.with_name(dbm_path.name + ".crossings.npz")
destination.write_bytes(b"PK\\x03\\x04fake-crossings")
print(f"wrote {destination}")
'''

# Every flag grow_track_graph.py's own argparse block accepts at the pinned
# commit, so a call this runner builds parses cleanly even for the ones this
# test never overrides.
GROW_TRACK_GRAPH = '''\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tracks", type=Path)
parser.add_argument("crossings", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--seeds", type=int, nargs="+")
parser.add_argument("--random-count", type=int)
parser.add_argument("--random-seed", type=int, default=0)
parser.add_argument("--random-top-percent", type=float)
parser.add_argument("--min-valid-vertices", type=int)
parser.add_argument("--growth-min-span", type=float)
parser.add_argument("--min-connect", type=int)
parser.add_argument("--min-size", type=float)
parser.add_argument("--max-size", type=float)
parser.add_argument("--max-thick-cell-frac", type=float)
parser.add_argument("--reject-any-fold-fixes", action="store_true")
parser.add_argument("--gate-tol", type=float)
parser.add_argument("--resample-spacing", type=float)
parser.add_argument("--min-track-arclength", type=float)
parser.add_argument("--output-spacing", type=float)
parser.add_argument("--border-erode-vx", type=float)
parser.add_argument("--workers", type=int)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

if not args.tracks.is_dir():
    raise SystemExit(f"not a packed .vctracks directory: {args.tracks}")
if not args.crossings.is_file():
    raise SystemExit(f"missing crossings sidecar: {args.crossings}")

args.output.mkdir(parents=True, exist_ok=True)
targets = args.seeds if args.seeds else list(range(args.random_count))
# SPLIT_MARKER
for index, seed in enumerate(targets):
    patch = args.output / f"band-seed{seed}-{index:06d}.tifxyz"
    patch.mkdir()
    (patch / "x.tif").write_bytes(b"")
print(f"grew {len(targets)} patches")
'''


def truncated(script: str, replacement: str) -> str:
    preamble = script.split("# SPLIT_MARKER")[0]
    return preamble + replacement


def fitter(root: Path, *, grow: str = GROW_TRACK_GRAPH) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "convert_track_store.py").write_text(CONVERT_TRACK_STORE, encoding="utf-8")
    (root / "build_track_crossings.py").write_text(BUILD_TRACK_CROSSINGS, encoding="utf-8")
    (root / "grow_track_graph.py").write_text(grow, encoding="utf-8")
    return root


def dataset(root: Path, *, tracks_file: str = "2um_ds2_ps256_surf_v2.dbm") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "tracks").mkdir(exist_ok=True)
    (root / "tracks" / tracks_file).write_text("fake dbm bytes")
    return root


def run(tmp_path, *extra, dataset_dir=None, fitter_root=None, seed_args=("--random-count", "2")):
    return runner.main([
        "--profile-id", PROFILE_ID,
        "--out", str(tmp_path / "run"),
        "--dataset-path", str(dataset_dir or tmp_path / "ds"),
        "--fitter-root", str(fitter_root or tmp_path / "spiral"),
        *seed_args,
        *extra])


def receipt_of(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / runner.RECEIPT).read_text())


# -- before anything runs ----------------------------------------------------

def test_a_missing_tracks_dbm_is_refused_before_any_subprocess(tmp_path):
    (tmp_path / "ds").mkdir()
    fitter(tmp_path / "spiral")

    assert run(tmp_path) == 2
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "REFUSED"
    assert "tracks" in receipt["reason"]


def test_seeds_and_random_count_are_mutually_required(tmp_path):
    """Growing nothing and growing an explicit list plus a random quota are
    both requests this lane refuses to guess between."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, seed_args=()) == 2
    assert "exactly one" in receipt_of(tmp_path)["reason"]

    assert run(tmp_path, seed_args=("--random-count", "1", "--seeds", "3")) == 2
    assert "exactly one" in receipt_of(tmp_path)["reason"]


def test_seeds_and_seeds_json_together_is_refused(tmp_path):
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, seed_args=("--seeds", "1", "--seeds-json", "[2]")) == 2
    assert "not both" in receipt_of(tmp_path)["reason"]


def test_no_unverified_patches_dir_is_refused_rather_than_defaulted(tmp_path):
    """There is no upstream default the way tracks_file has one -- this lane
    does not invent 'unverified_patches' silently; the profile freezes it."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    bare_profile = {
        "schema": runner.SCHEMA, "profile_id": "grow-track-patches-v1@bare",
        "growth_overrides": {}, "dataset_layout": {}}
    (profile_dir / "bare.json").write_text(json.dumps(bare_profile))

    import importlib
    original_dir = runner.PROFILE_DIR
    try:
        runner.PROFILE_DIR = profile_dir
        code = runner.main([
            "--profile-id", "grow-track-patches-v1@bare",
            "--out", str(tmp_path / "run"),
            "--dataset-path", str(tmp_path / "ds"),
            "--fitter-root", str(tmp_path / "spiral"),
            "--random-count", "1"])
    finally:
        runner.PROFILE_DIR = original_dir
    assert code == 2
    assert "unverified_patches_dir" in receipt_of(tmp_path)["reason"]


def test_a_dry_run_touches_no_subprocess(tmp_path):
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run", "--unverified-patches-dir",
              "unverified_patches") == 0
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "PREFLIGHT_ONLY"
    assert not (tmp_path / "ds" / "unverified_patches").exists()
    assert "vctracks" not in receipt and "growth" not in receipt


# -- the whole chain, and where it lands -------------------------------------

def test_a_random_count_run_grows_that_many_patches_into_the_dataset(tmp_path):
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--random-count", "3", "--random-seed", "7")) == 0
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "GROWN"
    written = sorted((tmp_path / "ds" / "unverified_patches").iterdir())
    assert len(written) == 3
    assert receipt["patches_written"] == 3
    assert receipt["requested"] == {"random_count": 3, "random_seed": 7}
    assert receipt["wrote_to"] == str(tmp_path / "ds" / "unverified_patches")


def test_explicit_seeds_are_the_ones_grown(tmp_path):
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--seeds", "5", "12")) == 0
    receipt = receipt_of(tmp_path)
    written = {p.name for p in (tmp_path / "ds" / "unverified_patches").iterdir()}
    assert any(name.startswith("band-seed5-") for name in written)
    assert any(name.startswith("band-seed12-") for name in written)
    assert receipt["requested"] == {"seeds": [5, 12]}


def test_seeds_json_is_the_same_request_the_queue_sends(tmp_path):
    """job_store.declarative_argv's json_flags sends one flag, one JSON
    value -- not repeated --seeds tokens."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--seeds-json", "[9]")) == 0
    written = {p.name for p in (tmp_path / "ds" / "unverified_patches").iterdir()}
    assert any(name.startswith("band-seed9-") for name in written)


def test_an_existing_current_vctracks_store_is_reused_not_repacked(tmp_path):
    """convert_track_store.py refuses to overwrite an existing store without
    --force and has no self-skip; this runner has to know when not to call it
    at all, or every repeated run on a large DBM re-packs it for nothing."""
    ds = dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")
    tracks_dbm = ds / "tracks" / "2um_ds2_ps256_surf_v2.dbm"
    stat = tracks_dbm.stat()
    store = tracks_dbm.with_name(tracks_dbm.name + ".vctracks")
    store.mkdir()
    (store / "metadata.json").write_text(json.dumps({
        "version": 1,
        "source_db_signature": [[tracks_dbm.name, stat.st_size, stat.st_mtime_ns]],
        "track_count": 0, "point_count": 0}))
    marker = store / "not_touched_by_a_repack"
    marker.write_bytes(b"x")

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--random-count", "1")) == 0
    receipt = receipt_of(tmp_path)
    assert receipt["vctracks"] == {"reused": True, "path": str(store)}
    assert marker.is_file(), "a reused store must not be rebuilt"


def test_a_stale_vctracks_store_is_repacked_with_force(tmp_path):
    """A signature that does not match the DBM's own (name, size, mtime_ns)
    is distrusted, not merged with -- the store is fully rebuilt."""
    ds = dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")
    tracks_dbm = ds / "tracks" / "2um_ds2_ps256_surf_v2.dbm"
    store = tracks_dbm.with_name(tracks_dbm.name + ".vctracks")
    store.mkdir()
    (store / "metadata.json").write_text(json.dumps({
        "version": 1, "source_db_signature": [["not-the-real-file", 0, 0]],
        "track_count": 0, "point_count": 0}))

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--random-count", "1")) == 0
    receipt = receipt_of(tmp_path)
    assert receipt["vctracks"]["reused"] is False
    assert "--force" in receipt["vctracks"]["argv"]


def test_no_crossings_sidecar_is_refused_rather_than_grown_from_nothing(tmp_path):
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral", grow=GROW_TRACK_GRAPH)
    # A build_track_crossings.py that exits zero but writes nothing.
    (tmp_path / "spiral" / "build_track_crossings.py").write_text(
        "print('did nothing')\n", encoding="utf-8")

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--random-count", "1")) == 1
    assert receipt_of(tmp_path)["outcome"] == "NO_CROSSINGS"


def test_a_growth_failure_with_nothing_written_is_a_clean_failure(tmp_path):
    dataset(tmp_path / "ds")
    failing_grow = truncated(GROW_TRACK_GRAPH, 'raise SystemExit("boom")\n')
    fitter(tmp_path / "spiral", grow=failing_grow)

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--random-count", "2")) != 0
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "GROWTH_FAILED"
    assert receipt["patches_written"] == 0


def test_a_partial_growth_keeps_what_it_wrote(tmp_path):
    """One seed grown, then a failure -- the written patch is not discarded,
    and the receipt says this run did not get everything it asked for."""
    dataset(tmp_path / "ds")
    partial_grow = truncated(GROW_TRACK_GRAPH, '''\
(args.output / f"band-seed{targets[0]}-000000.tifxyz").mkdir()
raise SystemExit("ran out partway")
''')
    fitter(tmp_path / "spiral", grow=partial_grow)

    assert run(tmp_path, "--unverified-patches-dir", "unverified_patches",
              seed_args=("--seeds", "1", "2")) == 0
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "GROWN_PARTIAL"
    assert receipt["patches_written"] == 1


def test_growth_overrides_translate_to_the_flags_grow_track_graph_py_reads(tmp_path):
    ds = dataset(tmp_path / "ds")
    recording_grow = truncated(GROW_TRACK_GRAPH, '''\
(args.output / "SEEN.json").write_text(
    __import__("json").dumps({"min_connect": args.min_connect,
                              "resample_spacing": args.resample_spacing,
                              "overwrite": args.overwrite}))
for index, seed in enumerate(targets):
    (args.output / f"band-seed{seed}-{index:06d}.tifxyz").mkdir()
''')
    fitter(tmp_path / "spiral", grow=recording_grow)
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    tuned = {
        "schema": runner.SCHEMA, "profile_id": "grow-track-patches-v1@tuned",
        "growth_overrides": {"min_connect": 5, "resample_spacing": 2.5,
                             "overwrite": True},
        "dataset_layout": {"unverified_patches_dir": "unverified_patches"}}
    (profile_dir / "tuned.json").write_text(json.dumps(tuned))

    original_dir = runner.PROFILE_DIR
    try:
        runner.PROFILE_DIR = profile_dir
        code = runner.main([
            "--profile-id", "grow-track-patches-v1@tuned",
            "--out", str(tmp_path / "run"),
            "--dataset-path", str(ds),
            "--fitter-root", str(tmp_path / "spiral"),
            "--random-count", "1"])
    finally:
        runner.PROFILE_DIR = original_dir
    assert code == 0
    seen = json.loads((ds / "unverified_patches" / "SEEN.json").read_text())
    assert seen == {"min_connect": 5, "resample_spacing": 2.5, "overwrite": True}


def test_an_unknown_growth_override_key_is_refused_before_a_subprocess_runs():
    with pytest.raises(runner.GrowRunRefused, match="min_connect"):
        # A typo a profile author might actually make.
        runner.growth_argv("grow_track_graph.py", {"min_conect": 4})


# -- what this lane's own source does not do ---------------------------------

def test_the_runner_itself_imports_no_torch_or_cuda():
    """This lane's own claim, not upstream's: run_grow_track_patches.py never
    imports torch or cupy. (The subprocesses it launches are villa's own --
    see the profile's notes.cpu_only for the citation trail through their
    imports, which this repository does not vendor and so cannot grep here.)
    """
    source = (ROOT / "framework/stages/01-segmentation/scripts/"
             "run_grow_track_patches.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "cupy" not in source
