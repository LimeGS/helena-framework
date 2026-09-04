"""What the spiral runner does around the fit, which is most of the value.

The fit and the export are upstream's. What this platform adds is the order
things happen in: everything that can refuse a run happens before a GPU is
claimed, both subprocesses are cancellable the same way, and everything they
produce is filed the same way a grown surface is.

At 23adee04 that is two subprocesses, not one: fit_spiral.py writes a
checkpoint and stops, and flatten_spiral_checkpoint.py -- upstream's own
tool, not reimplemented here -- turns it into the one combined, flattened
TIFXYZ this platform registers. The fake scripts below stand in for both;
they are not a claim about what the real ones compute, only about the files
they leave behind and the exit codes they can return, which is everything
this runner's own logic depends on.

The registration half is the one worth guarding on its own terms. A fitted
surface that stays on the worker's disk is a surface P2 will never certify
and nobody will ever find -- which is the failure the whole artifact store
exists to prevent, arriving through a new door.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation/scripts"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

import run_spiral_fit as runner  # noqa: E402

PROFILE_ID = "spiral-fitter-v1@0.4.0"

# Upstream's shape at this commit: a lightweight config.py the adapter
# validates overrides against, and a headless fit_spiral.py that reads
# --dataset/--scroll-spec/--cache and writes a checkpoint where
# FIT_SPIRAL_RUN_DIR points, not a TIFXYZ directly.
FAKE_CONFIG = '''\
class Config:
    def __init__(self, overrides=None):
        self.z_begin = 4000
        self.z_end = 17000
        self.optimizer_random_seed = 1
        self.input_disable_patches = False
        self.input_use_tracks = False
        self.input_use_outer_shell = True
        self.dense_spacing_mode = "winding_model"
        for key, value in (overrides or {}).items():
            setattr(self, key, value)

    def as_dict(self):
        return vars(self).copy()
'''

# The split marker every "make this fitter fail a different way" test cuts
# at: everything above it always runs (the CLI parses, the manifest is read,
# the overrides are read), so a truncated variant still exercises argv
# parsing and manifest resolution before it diverges.
FIT_SPIRAL = '''\
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--scroll-spec", default=None)
parser.add_argument("--cache", default=None)
args = parser.parse_args()

spec_path = Path(args.scroll_spec) if args.scroll_spec \\
    else Path(args.dataset) / "spiral-scroll.json"
spec = json.loads(spec_path.read_text())
overrides = json.loads(os.environ.get("FIT_SPIRAL_CONFIG_OVERRIDES", "{}"))
run_dir = Path(os.environ["FIT_SPIRAL_RUN_DIR"])
run_dir.mkdir(parents=True, exist_ok=True)

# SPLIT_MARKER
(run_dir / "checkpoint_fitted.ckpt").write_bytes(
    b"PK\\x03\\x04" + json.dumps(spec).encode())
print("fitted", spec["name"], overrides.get("z_begin"), overrides.get("z_end"))
'''

FLATTEN = '''\
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--umbilicus", type=Path)
parser.add_argument("--lasagna-dir", type=Path)
parser.add_argument("--device", default="cuda")
parser.add_argument("--voxel-size-um", type=float, default=9.6)
args = parser.parse_args()
if not args.checkpoint.is_file():
    raise SystemExit(f"no checkpoint at {args.checkpoint}")

# SPLIT_MARKER
import numpy as np
import tifffile
args.output.mkdir(parents=True)
rows, columns = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
tifffile.imwrite(args.output / "x.tif", (100 + columns).astype(np.float32))
tifffile.imwrite(args.output / "y.tif", (200 + rows).astype(np.float32))
tifffile.imwrite(args.output / "z.tif", np.full((8, 8), 300.0, np.float32))
(args.output / "meta.json").write_text('{"schema": "test"}')
print("wrote", args.output)
'''


def dataset(root: Path, *, optional: bool = True,
            volumes: str = "las_008_{array}.ome.zarr") -> Path:
    """A dataset holding what the fit opens directly.

    `optional=False` is the case every scroll of ours is in: an umbilicus,
    tracks and lasagna, and none of the Paris 4 winding annotations. The
    physical layout is unchanged by the 23adee04 restructure -- verified
    against spiral-fitting/fit_session.py's conventional relative paths.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "umbilicus.json").write_text("{}")
    if optional:
        for name in ("abs_winding.json", "patch-overlap-pcls.json",
                     "relative_windings.json", "same_windings.json"):
            (root / name).write_text("{}")
    (root / "tracks").mkdir(exist_ok=True)
    (root / "tracks/2um_ds2_ps256_surf_v2.dbm").write_text("")
    for volume in ("nx", "ny", "grad_mag"):
        (root / "lasagna_inputs" / volumes.format(array=volume)).mkdir(
            parents=True, exist_ok=True)
    return root


def fitter(root: Path, *, fit_spiral: str = FIT_SPIRAL,
          flatten: str = FLATTEN) -> Path:
    """A fake spiral-fitting checkout: config.py, fit_spiral.py,
    flatten_spiral_checkpoint.py, and a sibling lasagna/fit_service.py so
    lasagna_root()'s existence check passes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.py").write_text(FAKE_CONFIG, encoding="utf-8")
    (root / "fit_spiral.py").write_text(fit_spiral, encoding="utf-8")
    (root / "flatten_spiral_checkpoint.py").write_text(flatten, encoding="utf-8")
    lasagna = root.parent / "lasagna"
    lasagna.mkdir(parents=True, exist_ok=True)
    (lasagna / "fit_service.py").write_text("", encoding="utf-8")
    return root


def truncated(script: str, replacement: str) -> str:
    """Everything before SPLIT_MARKER, plus a different ending -- the way to
    make a fake fitter or exporter fail a specific way while still exercising
    its own argv parsing and manifest/checkpoint reads."""
    preamble = script.split("# SPLIT_MARKER")[0]
    return preamble + replacement


def run(tmp_path, *extra, **kwargs):
    return runner.main([
        "--profile-id", PROFILE_ID,
        "--out", str(tmp_path / "run"),
        "--sample", "PHerc0172",
        "--scroll-name", "PHerc0172",
        "--dataset-path", str(kwargs.get("dataset", tmp_path / "ds")),
        "--z-begin", "500", "--z-end", "9000", "--voxel-um", "7.91",
        "--winding-sense", "CCW",
        "--fitter-root", str(kwargs.get("fitter_root", tmp_path / "spiral")),
        *extra])


def receipt_of(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / runner.RECEIPT).read_text())


# -- what happens before the GPU -------------------------------------------

def test_a_dataset_missing_a_required_input_is_named_rather_than_discovered(tmp_path):
    """Upstream finds this after loading the volume. Here it costs nothing."""
    directory = dataset(tmp_path / "ds")
    (directory / "umbilicus.json").unlink()
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run") == 2
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "REFUSED"
    assert "umbilicus.json" in receipt["reason"]
    assert receipt["dataset"]["missing_required"] == ["umbilicus.json"]


def test_a_dataset_with_no_paris_4_annotations_runs_and_says_what_it_lost(tmp_path):
    """The case this exists for. PHerc0826, 0211 and 0257 have an umbilicus,
    tracks and lasagna and none of the four winding files; 0.2.0 used to
    demand all nine and made the fitter unrunnable on every one of them."""
    dataset(tmp_path / "ds", optional=False)
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run") == 0
    survey = receipt_of(tmp_path)["dataset"]
    assert survey["missing_required"] == []
    assert survey["degraded"] is True
    absent = {entry["path"] for entry in survey["absent_optional"]}
    # New at this commit relative to the previous pin: "fibers", also
    # optional -- see the profile's inputs.optional entry for it.
    assert absent == {"abs_winding.json", "patch-overlap-pcls.json",
                      "relative_windings.json", "same_windings.json", "fibers"}
    # And the one whose absence is not just fewer constraints says so.
    winding = next(entry for entry in survey["absent_optional"]
                   if entry["path"] == "abs_winding.json")
    assert "absolute winding" in winding["costs"]


def test_the_lasagna_volumes_level_and_scale_move_together(tmp_path):
    """Upstream's conventional path names one volume set. The campaign
    measured scale 8 at 7.5% better than scale 4 for eight times the data and
    out of memory in 24 GB on a whole scroll -- so which level a run reads is
    a decision, now expressed as a spiral-scroll.json path override rather
    than a rewritten constant."""
    dataset(tmp_path / "ds", optional=False,
            volumes="PHerc0826_{array}.ome.zarr")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run", "--lasagna-volume-name",
               "PHerc0826_{array}.ome.zarr", "--normal-zarr-group", "3") == 0
    receipt = receipt_of(tmp_path)
    layout = receipt["dataset_layout"]
    assert layout["lasagna_volume_name"] == "PHerc0826_{array}.ome.zarr"
    # Derived, not stated: level 3 is scale 8, which is the relation a
    # lookup table got wrong by answering 4.
    assert layout["normal_zarr_group"] == "3" and layout["lasagna_scale"] == 8
    written = receipt["scroll_spec"]
    assert written["layout_is_upstream_default"] is False
    document = written["document"]
    assert document["normal_zarr_group"] == "3"
    assert document["lasagna_scale"] == 8
    assert document["paths"]["normal_x"].endswith(
        "lasagna_inputs/PHerc0826_nx.ome.zarr")
    assert document["paths"]["gradient_magnitude"].endswith(
        "lasagna_inputs/PHerc0826_grad_mag.ome.zarr")
    # The manifest actually on disk agrees with the receipt's copy of it.
    on_disk = json.loads((tmp_path / "run/spiral-scroll.json").read_text())
    assert on_disk == document

    # And a run that leaves the volume name at upstream's las_008 against this
    # PHerc0826-named dataset is refused by name rather than opening the
    # wrong volumes -- Helena's own preflight, unchanged by the restructure.
    assert run(tmp_path, "--dry-run") == 2
    assert "las_008_nx.ome.zarr" in receipt_of(tmp_path)["reason"]


def test_a_dataset_that_is_not_there_is_not_created(tmp_path):
    fitter(tmp_path / "spiral")
    assert run(tmp_path, "--dry-run", dataset=tmp_path / "nowhere") == 2
    assert "is not here" in receipt_of(tmp_path)["reason"]


def test_a_superseded_profile_cannot_be_queued_against(tmp_path):
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")
    assert runner.main([
        "--profile-id", "spiral-fitter-v1@0.3.0", "--out", str(tmp_path / "run"),
        "--sample", "PHerc0172", "--dataset-path", str(tmp_path / "ds"),
        "--dry-run"]) == 2
    assert "superseded" in receipt_of(tmp_path)["reason"]


def test_a_preflight_writes_the_manifest_and_stops(tmp_path):
    """The whole run except the fit: it is how a dataset gets checked without
    paying for a lease, and the manifest is proved readable before it
    matters."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run") == 0
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "PREFLIGHT_ONLY"
    assert receipt["scroll_spec"]["document"]["name"] == "PHerc0172"
    assert receipt["environment"]["WANDB_MODE"] == "disabled"
    # The image's own fit_spiral.py is never edited -- there is nothing left
    # to rewrite it into, unlike the previous pin's private copy.
    assert (tmp_path / "spiral/fit_spiral.py").read_text() == FIT_SPIRAL
    assert (tmp_path / "run/spiral-scroll.json").is_file()


def test_the_environment_in_the_receipt_is_only_what_this_run_set(tmp_path):
    """A receipt carrying the container's whole environment publishes whatever
    happens to be exported on the host."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")
    run(tmp_path, "--dry-run")

    assert set(receipt_of(tmp_path)["environment"]) <= {
        "FIT_SPIRAL_OUT_DIR", "FIT_SPIRAL_RUN_DIR", "FIT_SPIRAL_CACHE_DIR",
        "FIT_SPIRAL_RUN_TAG", "FIT_SPIRAL_CONFIG_OVERRIDES", "WANDB_MODE"}


def test_the_z_range_and_seed_ride_the_override_channel_not_the_manifest(tmp_path):
    """z_begin/z_end moved off the six-constant rebind and onto Config; this
    is the one place that shows up now, since the manifest itself carries
    neither."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run", "--random-seed", "7") == 0
    receipt = receipt_of(tmp_path)
    overrides = json.loads(receipt["environment"]["FIT_SPIRAL_CONFIG_OVERRIDES"])
    assert overrides["z_begin"] == 500 and overrides["z_end"] == 9000
    assert overrides["optimizer_random_seed"] == 7
    assert receipt["random_seed"] == {"value": 7, "config_key": "optimizer_random_seed"}
    assert "z_begin" not in receipt["scroll_spec"]["document"]


# -- and after the fit -------------------------------------------------------

def test_the_export_step_runs_after_a_successful_fit_and_is_registered(tmp_path):
    """Not "windings found" any more: one fit now writes one checkpoint, and
    the export step turns it into one combined, flattened surface."""
    pytest.importorskip("numpy")
    pytest.importorskip("tifffile")
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    code = run(tmp_path)
    receipt = receipt_of(tmp_path)
    assert code == 0, receipt
    assert receipt["outcome"] == "FITTED_NOT_REGISTERED"
    assert receipt["surface_found"] is True
    assert receipt["note"].startswith("no control plane")
    assert (tmp_path / "run/fit/checkpoint_fitted.ckpt").is_file()
    assert (tmp_path / "run/export/spiral.tifxyz/x.tif").is_file()


def test_a_fit_that_wrote_no_checkpoint_is_not_exported(tmp_path):
    """The failure this two-stage split adds a name for: a fit that exits
    zero without writing checkpoint_fitted.ckpt never reaches the export
    step at all."""
    dataset(tmp_path / "ds")
    directory = tmp_path / "spiral"
    fitter(directory, fit_spiral=truncated(FIT_SPIRAL, "print('fitted nothing')\n"))

    assert run(tmp_path) == 1
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "NO_CHECKPOINT"
    assert "export" not in receipt


def test_a_fit_that_failed_is_reported_with_its_code(tmp_path):
    dataset(tmp_path / "ds")
    directory = tmp_path / "spiral"
    fitter(directory, fit_spiral=truncated(FIT_SPIRAL, "raise SystemExit(3)\n"))

    assert run(tmp_path) == 3
    assert receipt_of(tmp_path)["outcome"] == "FIT_FAILED"


def test_an_export_that_wrote_no_surface_is_not_a_fit_with_an_empty_one(tmp_path):
    dataset(tmp_path / "ds")
    directory = tmp_path / "spiral"
    fitter(directory, flatten=truncated(FLATTEN, "print('exported nothing')\n"))

    assert run(tmp_path) == 1
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "NO_SURFACE"
    assert "Nothing is registered" in receipt["note"]
    # The checkpoint survives a failed export, so a retry does not refit.
    assert (tmp_path / "run/fit/checkpoint_fitted.ckpt").is_file()


def test_an_export_that_failed_is_reported_with_its_code(tmp_path):
    dataset(tmp_path / "ds")
    directory = tmp_path / "spiral"
    fitter(directory, flatten=truncated(FLATTEN, "raise SystemExit(4)\n"))

    assert run(tmp_path) == 4
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "EXPORT_FAILED"
    assert (tmp_path / "run/fit/checkpoint_fitted.ckpt").is_file()


# -- registration -------------------------------------------------------------

class Store:
    """A control plane that records one snapshot and every surface filed."""

    def __init__(self):
        self.imported: list[dict] = []

    def snapshots(self, samples=None):
        return [{"source_snapshot_id": "src-1", "sample_id": "PHerc0172",
                 "voxel_size_um": 7.91, "ct_uri": "s3://bucket/ct.zarr"}]

    def import_surface(self, payload):
        self.imported.append(payload)
        return payload["surface_id"]


def test_a_fitted_surface_is_filed_the_way_a_grown_one_is(tmp_path):
    """Same artifact store, same content address, same import_surface. A
    second route into the catalogue would be a second answer to what a
    surface is, and the certification gate would then be measuring two
    things."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from fleet.artifact_store import LocalArtifactStore  # noqa: PLC0415

    surface = tmp_path / "export/spiral.tifxyz"
    surface.mkdir(parents=True)
    rows, columns = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    tifffile.imwrite(surface / "x.tif", (100 + columns).astype(np.float32))
    tifffile.imwrite(surface / "y.tif", (200 + rows).astype(np.float32))
    tifffile.imwrite(surface / "z.tif", np.full((8, 8), 300.0, np.float32))
    (surface / "meta.json").write_text('{"schema": "test"}')

    store = Store()
    lineage = {"produced_by_backend": "spiral",
               "produced_by_profile_id": PROFILE_ID,
               "spiral_binding_sha256": "a" * 64,
               "spiral_scroll_spec_sha256": "b" * 64,
               "requested_by_job_id": "j-1", "mission_id": "m-1"}
    outcome = runner.register_fitted_surface(
        store, LocalArtifactStore(tmp_path / "artifacts"), surface,
        sample_id="PHerc0172", snapshot=store.snapshots()[0],
        voxel_size_um=7.91, run_id="j-1", lineage=lineage)

    payload = store.imported[0]
    assert payload["surface_id"] == outcome["surface_id"]
    assert payload["sample_id"] == "PHerc0172"
    assert payload["source_snapshot_id"] == "src-1"
    # Produced here, with its whole provenance on the record: calling it
    # imported would file it with the surfaces that came from outside.
    assert payload["owner"] == "campaign-x"
    assert payload["produced_by_backend"] == "spiral"
    assert payload["produced_by_profile_id"] == PROFILE_ID
    assert payload["bbox_xyz"] and payload["area_cm2"] > 0

    published = Path(outcome["artifact_uri"])
    assert (published / "x.tif").is_file()
    manifest = json.loads((published / "ARTIFACT_SET.json").read_text())
    assert manifest["artifact_sha256"] == payload["artifact_sha256"]
    assert set(manifest["files"]) == {"x.tif", "y.tif", "z.tif", "meta.json"}
    assert manifest["produced_by"]["spiral_scroll_spec_sha256"] == "b" * 64


# -- cancelling ---------------------------------------------------------------

def test_cancelling_the_runner_takes_the_subprocess_down_with_it(tmp_path):
    """The worker cancels a job by terminating its runner. Without this the
    subprocess is reparented and keeps the card, held by nothing the queue
    knows about and reaped by nothing at all.

    Shared by both subprocess stages: run_fit is an alias for the same
    run_subprocess this runner uses for the export step too, so this
    contract does not need proving twice.

    Driven by sending the signal to this process, which is what the worker
    does: the handler has to reach the child from wherever it runs.
    """
    import os
    import signal
    import threading
    import time

    script = tmp_path / "sleeper.py"
    script.write_text(
        "import time\n"
        "open('started', 'w').write('yes')\n"
        "time.sleep(120)\n", encoding="utf-8")

    threading.Timer(0.4, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    started = time.monotonic()
    code = runner.run_fit([sys.executable, "sleeper.py"], cwd=tmp_path,
                          environment=dict(os.environ))
    elapsed = time.monotonic() - started

    assert (tmp_path / "started").is_file(), "the child never ran"
    # Negative: killed by a signal rather than exiting on its own.
    assert code < 0, f"the fit outlived the cancellation, returncode {code}"
    # Promptly. A handler that waits on the child from inside itself cannot
    # reap it -- the outer wait holds the lock -- so it spins until its own
    # grace period expires and every cancellation costs twenty seconds.
    assert elapsed < runner.TERMINATE_GRACE_SECONDS / 2, (
        f"cancelling took {elapsed:.1f}s; the child was not reaped promptly")
    # And this process is still here to write a receipt about it.
    assert signal.getsignal(signal.SIGTERM) is not None
