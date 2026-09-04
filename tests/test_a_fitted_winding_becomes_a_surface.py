"""What the spiral runner does around the fit, which is most of the value.

The fit itself is upstream's. What this platform adds is the order things
happen in: everything that can refuse a run happens before a GPU is claimed,
and everything the fit produces is filed the same way a grown surface is.

The second half is the one worth guarding. A fitted winding that stays on the
worker's disk is a surface P2 will never certify and nobody will ever find --
which is the failure the whole artifact store exists to prevent, arriving
through a new door.
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

PROFILE_ID = "spiral-fitter-v1@0.3.0"

# Upstream's shape: constants at module level, a default_config the adapter
# parses, and -- here -- output written where FIT_SPIRAL_OUT_DIR points.
FITTER = '''\
import os
from pathlib import Path

dataset_path = '/ephemeral/paul/spiral/dataset'
scroll_name = 's1'
z_begin, z_end = 4000, 17000
voxel_size_um = 9.6
spiral_outward_sense = 'CW'
normal_nx_zarr_path = f'{dataset_path}/lasagna_inputs/las_008_nx.ome.zarr'
normal_ny_zarr_path = f'{dataset_path}/lasagna_inputs/las_008_ny.ome.zarr'
grad_mag_zarr_path = f'{dataset_path}/lasagna_inputs/las_008_grad_mag.ome.zarr'
tracks_dbm_path = f'{dataset_path}/tracks/2um_ds2_ps256_surf_v2.dbm'
normal_zarr_group = '4'
lasagna_scale = 4

default_config = {'disable_patches': False, 'random_seed': 1}

import numpy as np, tifffile
out = Path(os.environ['FIT_SPIRAL_OUT_DIR'])
for winding in ('w010', 'w011'):
    directory = out / 'meshes' / winding
    directory.mkdir(parents=True, exist_ok=True)
    rows, columns = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    tifffile.imwrite(directory / 'x.tif', (100 + columns).astype(np.float32))
    tifffile.imwrite(directory / 'y.tif', (200 + rows).astype(np.float32))
    tifffile.imwrite(directory / 'z.tif', np.full((8, 8), 300.0, np.float32))
    (directory / 'meta.json').write_text('{"winding": "%s"}' % winding)
print('fitted', scroll_name, z_begin, z_end)
'''


def dataset(root: Path, *, optional: bool = True,
            volumes: str = "las_008_{array}.ome.zarr") -> Path:
    """A dataset holding what the fit opens directly.

    `optional=False` is the case every scroll of ours is in: an umbilicus,
    tracks and lasagna, and none of the Paris 4 winding annotations.
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


def fitter(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    script = root / "fit_spiral.py"
    script.write_text(FITTER, encoding="utf-8")
    return script


def run(tmp_path, *extra, **kwargs):
    return runner.main([
        "--profile-id", PROFILE_ID,
        "--out", str(tmp_path / "run"),
        "--sample", "PHerc0172",
        "--scroll-name", "PHerc0172",
        "--dataset-path", str(kwargs.get("dataset", tmp_path / "ds")),
        "--z-begin", "500", "--z-end", "9000", "--voxel-um", "7.91",
        "--winding-sense", "CCW",
        "--script", str(kwargs.get("script", tmp_path / "spiral/fit_spiral.py")),
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
    tracks and lasagna and none of the four winding files; the profile used to
    demand all nine and made the fitter unrunnable on every one of them."""
    dataset(tmp_path / "ds", optional=False)
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run") == 0
    survey = receipt_of(tmp_path)["dataset"]
    assert survey["missing_required"] == []
    assert survey["degraded"] is True
    absent = {entry["path"] for entry in survey["absent_optional"]}
    assert absent == {"abs_winding.json", "patch-overlap-pcls.json",
                      "relative_windings.json", "same_windings.json"}
    # And the one whose absence is not just fewer constraints says so.
    winding = next(entry for entry in survey["absent_optional"]
                   if entry["path"] == "abs_winding.json")
    assert "absolute winding" in winding["costs"]


def test_the_lasagna_volumes_level_and_scale_move_together(tmp_path):
    """Upstream hardcodes scale 8 in three constants. The campaign measured it
    at 7.5% better than scale 4 for eight times the data, and out of memory in
    24 GB on a whole scroll -- so which one a run reads is a decision."""
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
    assert receipt["repin"]["layout_is_upstream_default"] is False
    rebound = Path(receipt["repin"]["rebound_script"]).read_text()
    assert "PHerc0826_nx.ome.zarr" in rebound and "las_008" not in rebound
    assert "lasagna_scale = 8" in rebound
    # And the dataset root is still referenced rather than frozen into three
    # paths. It used to be an f-string field; it is a concatenation now, because
    # an f-string is an interpolation context and the value being substituted
    # comes from the caller. The property is the same: `dataset_path` is still a
    # name the script resolves, not a literal baked in three times.
    assert "dataset_path + '/lasagna_inputs/PHerc0826_nx.ome.zarr'" in rebound
    assert "f'{dataset_path}" not in rebound and 'f"{dataset_path}' not in rebound

    # And a run that leaves the scale at upstream's 008 against this scale-4
    # dataset is refused by name rather than opening the wrong volumes.
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
        "--profile-id", "spiral-fitter-v1@0.2.0", "--out", str(tmp_path / "run"),
        "--sample", "PHerc0172", "--dataset-path", str(tmp_path / "ds"),
        "--dry-run"]) == 2
    assert "superseded" in receipt_of(tmp_path)["reason"]


def test_a_preflight_rebinds_and_stops(tmp_path):
    """The whole run except the fit: it is how a dataset gets checked without
    paying for a lease, and the rebind is proved before it matters."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    assert run(tmp_path, "--dry-run") == 0
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "PREFLIGHT_ONLY"
    assert receipt["repin"]["replaced"]["scroll_name"]["now"] == "PHerc0172"
    assert receipt["environment"]["WANDB_MODE"] == "disabled"
    # A private copy, so two scrolls can be fitted at once and the image keeps
    # the bytes its source lock verified.
    assert (tmp_path / "spiral/fit_spiral.py").read_text() == FITTER
    assert "PHerc0172" in (tmp_path / "run/fitter/fit_spiral.py").read_text()


def test_the_environment_in_the_receipt_is_only_what_this_run_set(tmp_path):
    """A receipt carrying the container's whole environment publishes whatever
    happens to be exported on the host."""
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")
    run(tmp_path, "--dry-run")

    assert set(receipt_of(tmp_path)["environment"]) <= {
        "FIT_SPIRAL_OUT_DIR", "FIT_SPIRAL_CACHE_DIR", "FIT_SPIRAL_RUN_TAG",
        "FIT_SPIRAL_CONFIG_OVERRIDES", "WANDB_MODE"}


# -- and after the fit -----------------------------------------------------

def test_the_windings_are_found_by_what_they_hold(tmp_path):
    """Not by a naming convention: the naming is upstream's, and the contract
    this platform depends on is the three coordinate planes."""
    pytest.importorskip("numpy")
    pytest.importorskip("tifffile")
    dataset(tmp_path / "ds")
    fitter(tmp_path / "spiral")

    code = run(tmp_path)
    receipt = receipt_of(tmp_path)
    assert code == 0, receipt
    assert receipt["outcome"] == "FITTED_NOT_REGISTERED"
    assert receipt["windings_found"] == 2
    assert receipt["note"].startswith("no control plane")


def test_a_fit_that_wrote_no_surface_is_not_a_fit_with_an_empty_one(tmp_path):
    dataset(tmp_path / "ds")
    directory = tmp_path / "spiral"
    fitter(directory)
    (directory / "fit_spiral.py").write_text(
        FITTER.split("import numpy")[0] + "print('fitted nothing')\n",
        encoding="utf-8")

    assert run(tmp_path) == 1
    receipt = receipt_of(tmp_path)
    assert receipt["outcome"] == "NO_SURFACE"
    assert "Nothing is registered" in receipt["note"]


def test_a_fit_that_failed_is_reported_with_its_code(tmp_path):
    dataset(tmp_path / "ds")
    directory = tmp_path / "spiral"
    fitter(directory)
    (directory / "fit_spiral.py").write_text(
        FITTER.split("import numpy")[0] + "raise SystemExit(3)\n", encoding="utf-8")

    assert run(tmp_path) == 3
    assert receipt_of(tmp_path)["outcome"] == "FIT_FAILED"


# -- registration ----------------------------------------------------------

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


def test_a_fitted_winding_is_filed_the_way_a_grown_one_is(tmp_path):
    """Same artifact store, same content address, same import_surface. A second
    route into the catalogue would be a second answer to what a surface is, and
    the certification gate would then be measuring two things."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from fleet.artifact_store import LocalArtifactStore  # noqa: PLC0415

    winding = tmp_path / "fit/meshes/w010"
    winding.mkdir(parents=True)
    rows, columns = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    tifffile.imwrite(winding / "x.tif", (100 + columns).astype(np.float32))
    tifffile.imwrite(winding / "y.tif", (200 + rows).astype(np.float32))
    tifffile.imwrite(winding / "z.tif", np.full((8, 8), 300.0, np.float32))
    (winding / "meta.json").write_text('{"winding": "w010"}')

    store = Store()
    lineage = {"produced_by_backend": "spiral",
               "produced_by_profile_id": PROFILE_ID,
               "spiral_binding_sha256": "a" * 64,
               "requested_by_job_id": "j-1", "mission_id": "m-1"}
    outcome = runner.register_winding(
        store, LocalArtifactStore(tmp_path / "artifacts"), winding,
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
    assert manifest["produced_by"]["spiral_binding_sha256"] == "a" * 64


# -- cancelling ------------------------------------------------------------

def test_cancelling_the_runner_takes_the_fit_down_with_it(tmp_path):
    """The worker cancels a job by terminating its runner. Without this the
    fitter is reparented and keeps the card, held by nothing the queue knows
    about and reaped by nothing at all.

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
