"""P4 rendering the sheet P3 unrolled, rather than the curved patch.

P4's contract says it consumes what P3 produces. Nothing produced it, so the
default lane was handed a path to a P1 surface instead -- a tifxyz following the
lamina through the scroll, curvature and all. The layer stack that came out was
a stack of curved patches, and the detector downstream was trained on pages.

Now a job can name a surface and let the worker fetch its flattened sheet. What
matters is that it cannot be ambiguous about which one it rendered.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import JobRejected, command_for, validate_parameters  # noqa: E402
import ink_worker  # noqa: E402

# A remote source, so `volume` is the renderer's own cache and its absence is
# the ordinary first run. Without `remote_url` the worker checks it, which is
# the point of that check.
VOLUME = {"lane": "vc-render-tifxyz", "volume": "/vol/scroll.zarr",
          "remote_url": "https://example.invalid/scroll.zarr",
          "scale": 1.0, "group_idx": 0}


def test_a_job_may_name_a_flattened_surface_instead_of_a_path():
    clean = validate_parameters(
        {**VOLUME,
         "flattened_surface": "0e79f232-6e29-51ad-8558-60be6b85a08d",
         "flattening_id": "flat-exact", "p3_job_id": "p3-exact",
         "flattened_artifact_sha256": "a" * 64},
        "P4")
    assert clean["flattened_surface"] == "0e79f232-6e29-51ad-8558-60be6b85a08d"
    assert "segmentation" not in clean


def test_a_flattened_surface_without_exact_p3_identity_is_refused():
    with pytest.raises(JobRejected, match="exact P3 flattening identity"):
        validate_parameters(
            {**VOLUME, "flattened_surface":
             "0e79f232-6e29-51ad-8558-60be6b85a08d"}, "P4")


def test_a_path_still_works():
    """The curved patch stays renderable. P3 is new, the corpus is not, and a
    comparison between the two needs both lanes to run."""
    clean = validate_parameters({**VOLUME, "segmentation": "/surfaces/s-1"}, "P4")
    assert clean["segmentation"] == "/surfaces/s-1"


def test_naming_both_is_refused():
    """Which one was rendered is the one thing a layer stack's provenance has to
    be certain of, and a job carrying both leaves it to whichever branch runs."""
    with pytest.raises(JobRejected) as refused:
        validate_parameters(
            {**VOLUME, "segmentation": "/surfaces/s-1",
             "flattened_surface": "s-1"}, "P4")
    assert "exactly one" in str(refused.value)


def test_naming_neither_is_refused():
    with pytest.raises(JobRejected) as refused:
        validate_parameters(VOLUME, "P4")
    assert "exactly one" in str(refused.value)


def test_an_unresolved_sheet_never_reaches_the_renderer():
    """The worker fetches the sheet and fills in `segmentation` before the
    command is built. If that step is ever skipped, the renderer must not be
    invoked with a missing argument or, worse, the wrong surface."""
    job = {"phase": "P4",
           "parameters": {**VOLUME, "flattened_surface": "s-1"}}
    with pytest.raises(JobRejected) as refused:
        command_for(job, runner="unused", output_dir="/tmp/out")
    assert "did not run" in str(refused.value)


def test_a_resolved_sheet_is_what_the_renderer_is_pointed_at():
    job = {"phase": "P4",
           "parameters": {**VOLUME, "flattened_surface": "s-1",
                          "segmentation": "/runs/job-9/flattened-surface"}}
    argv = command_for(job, runner="unused", output_dir="/tmp/out")
    assert "--segmentation" in argv
    assert argv[argv.index("--segmentation") + 1] == "/runs/job-9/flattened-surface"


def test_a_flattened_sheet_verifies_as_a_tifxyz_artifact_set():
    """The fetcher refused the sheet on its schema name.

    A grown surface and a flattened sheet are both immutable TIFXYZ artifact
    sets and are verified identically, but they keep separate schema names: the
    provenance of a layer stack depends on which of the two it came from, and
    one name for both would make that unanswerable from the artifact alone.
    """
    sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))
    from importlib.util import module_from_spec, spec_from_file_location

    path = ROOT / "framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py"
    spec = spec_from_file_location("qc_adapter_for_test", path)
    module = module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    assert "campaignx.flattened_artifact_set.v1" in module.TIFXYZ_MANIFEST_SCHEMAS
    assert "campaignx.segmentation_artifact_set.v1" in module.TIFXYZ_MANIFEST_SCHEMAS


def test_resolver_uses_original_surface_and_rejects_a_different_flattening(
        tmp_path):
    class Store:
        def flattened_sheet(self, surface_id, profile_id):
            assert surface_id == "surface-control"
            assert profile_id == "flatten-abf-v1@1.0.0"
            return {"flattening_id": "flat-other", "requested_by_job_id": "p3-current",
                    "artifact_sha256": "a" * 64, "artifact_uri": "s3://flat",
                    "state": "FLATTENED"}

    job = {"parameters": {"flattened_surface": "surface-control",
                           "flattening_id": "flat-current",
                           "p3_job_id": "p3-current",
                           "flattened_artifact_sha256": "a" * 64,
                           "flattening_profile": "flatten-abf-v1@1.0.0"}}
    with pytest.raises(RuntimeError, match="exact P3 artifact/job identity"):
        ink_worker.resolve_flattened_surface(Store(), job, tmp_path / "sheet")


def _metric_policy():
    return {"profile_id": "metric@1",
            "maximum_uv_to_3d_distortion_ratio": 1.25,
            "minimum_valid_triangle_fraction": 0.95}


def test_real_lateral_metric_measures_the_exact_tifxyz_grid(tmp_path):
    xx, yy = np.meshgrid(np.arange(3, dtype=np.float32),
                         np.arange(2, dtype=np.float32), indexing="xy")
    for axis, values in zip("xyz", (xx, yy, np.zeros_like(xx)), strict=True):
        tifffile.imwrite(tmp_path / f"{axis}.tif", values)
    receipt = ink_worker.measure_p3_p4_lateral_metric(
        tmp_path, [2, 3], source_voxel_um=2.0,
        lineage={"flattened_artifact_id": "flat", "p3_job_id": "p3",
                 "p4_job_id": "p4", "p4_layer_artifact_sha256": "a" * 64,
                 "p4_layer_manifest_sha256": "b" * 64}, policy=_metric_policy())
    assert receipt["status"] == "PROVEN"
    assert receipt["lateral_pixel_um"] == 2.0
    assert receipt["measurements"]["horizontal"]["count"] == 4
    assert receipt["measurements"]["vertical"]["count"] == 3


def test_lateral_metric_rejects_a_layer_grid_dimension_mismatch(tmp_path):
    for axis in "xyz":
        tifffile.imwrite(tmp_path / f"{axis}.tif", np.zeros((2, 3), dtype=np.float32))
    receipt = ink_worker.measure_p3_p4_lateral_metric(
        tmp_path, [3, 2], source_voxel_um=1.0, lineage={}, policy=_metric_policy())
    assert receipt["status"] == "UNPROVEN"
    assert receipt["reason_code"] == "P4_RASTER_DIMENSION_MISMATCH"


def test_lateral_metric_uses_frozen_triangle_coverage_not_edge_coverage(tmp_path):
    xx, yy = np.meshgrid(np.arange(10, dtype=np.float32),
                         np.arange(10, dtype=np.float32), indexing="xy")
    zz = np.zeros_like(xx)
    for row, column in ((3, 3), (6, 6)):
        xx[row, column] = yy[row, column] = zz[row, column] = -1
    for axis, values in zip("xyz", (xx, yy, zz), strict=True):
        tifffile.imwrite(tmp_path / f"{axis}.tif", values)

    receipt = ink_worker.measure_p3_p4_lateral_metric(
        tmp_path, [10, 10], source_voxel_um=1.0, lineage={},
        policy=_metric_policy())

    assert receipt["valid_triangle_count"] == 150
    assert receipt["possible_triangle_count"] == 162
    assert receipt["valid_triangle_fraction"] == pytest.approx(150 / 162)
    assert receipt["valid_edge_fraction"] == pytest.approx(172 / 180)
    assert receipt["status"] == "UNPROVEN"
    assert receipt["reason_code"] == "LATERAL_METRIC_POLICY_FAILED"


def test_worker_event_binds_the_orientation_proof_and_selected_boolean(
        tmp_path, monkeypatch):
    class Store:
        def __init__(self):
            self.events = []

        def mark_running(self, *args, **kwargs):
            return None

        def note(self, job_id, event_type, payload):
            self.events.append((job_id, event_type, payload))

        def heartbeat(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    def resolve(_store, active_job, sheet):
        active_job["_flattened_sheet"] = {
            "artifact_id": "flat-control", "surface_id": "surface-control",
            "p3_job_id": "p3-current", "artifact_sha256": "a" * 64}
        # The real one fetches the sheet to this path. The fake did not, and
        # the worker now checks that its inputs are there before it spends a
        # lease -- so a fake that returns a path to nothing describes a fetch
        # that silently produced nothing.
        Path(sheet).mkdir(parents=True, exist_ok=True)
        return str(sheet)
    monkeypatch.setattr(ink_worker, "resolve_flattened_surface", resolve)
    monkeypatch.setattr(ink_worker, "runner_for", lambda _job: "renderer")
    monkeypatch.setattr(ink_worker, "command_for",
                        lambda *_args, **_kwargs: ["renderer"])
    monkeypatch.setattr(ink_worker.subprocess, "run", lambda *_args, **_kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="failed"))
    store = Store()
    job = {
        "job_id": "p4-current", "lease_token": "lease", "phase": "P4",
        "sample_id": "PHerc0139",
        "parameters": {
            **VOLUME, "flattened_surface": "surface-control",
            "flattening_profile": "flatten-abf-v1@1.0.0",
            "orientation_receipt_sha256": "6" * 64,
            "flip_normals": False,
        },
    }
    ink_worker.run_job(store, job, runs_root=tmp_path, timeout=10)
    assert store.events == [("p4-current", "rendered_from", {
        "kind": "flattened_sheet",
        "surface_id": "surface-control",
            "flattening_id": "flat-control",
            "p3_job_id": "p3-current",
            "flattened_artifact_sha256": "a" * 64,
        "profile_id": "flatten-abf-v1@1.0.0",
        "orientation_receipt_sha256": "6" * 64,
        "flip_normals": False,
        "non_claim": "rendered on the flattened parametrisation, which is resampled "
                     "from the certified surface",
    })]
def test_the_cache_budget_reaches_the_renderer_as_an_integer():
    """vc_render_tifxyz refuses "4.0" for --cache-gb and dies on the argument
    before it reads a voxel, which is how the first end-to-end P4 run failed."""
    clean = validate_parameters(
        {**VOLUME, "segmentation": "/surfaces/s-1", "cache_gb": 4.0}, "P4")
    assert clean["cache_gb"] == 4
    argv = command_for({"phase": "P4", "parameters": clean},
                       runner="unused", output_dir="/tmp/out")
    assert argv[argv.index("--cache-gb") + 1] == "4"


def test_the_renderer_is_always_given_somewhere_to_write():
    """It refuses with "at least one of --zarr-output or --tif-output required"
    and renders nothing. The lane's own comment said it took no output flag,
    which went unnoticed for as long as nothing ran it."""
    argv = command_for(
        {"phase": "P4", "parameters": {**VOLUME, "segmentation": "/surfaces/s-1"}},
        runner="unused", output_dir="/runs/job-9")
    assert argv[argv.index("--tif-output") + 1] == "/runs/job-9/layers"


def test_a_zarr_render_also_writes_the_publishable_tiff_stack():
    """P4's verified/published contract and P5 both consume numbered TIFFs.

    A Zarr-only command could exit zero, then the worker looked for TIFFs and
    marked it failed. Zarr is an additional export, not a way around P4's
    actual output contract.
    """
    argv = command_for(
        {"phase": "P4", "parameters": {**VOLUME, "segmentation": "/surfaces/s-1",
                                       "zarr_output": "/runs/job-9/stack.zarr"}},
        runner="unused", output_dir="/runs/job-9")
    assert argv[argv.index("--tif-output") + 1] == "/runs/job-9/layers"
    assert argv[argv.index("--zarr-output") + 1] == "/runs/job-9/stack.zarr"


def test_depth_is_only_sent_when_it_was_chosen():
    """N slices at the wrong spacing is P4's documented way of failing, so an
    unset depth must leave the renderer's own default visible rather than have
    the queue pick a number nobody recorded choosing."""
    bare = command_for(
        {"phase": "P4", "parameters": {**VOLUME, "segmentation": "/s"}},
        runner="unused", output_dir="/runs/job-9")
    assert "--num-slices" not in bare
    chosen = command_for(
        {"phase": "P4", "parameters": {**VOLUME, "segmentation": "/s",
                                       "num_slices": 109, "slice_step": 1.0}},
        runner="unused", output_dir="/runs/job-9")
    assert chosen[chosen.index("--num-slices") + 1] == "109"
    assert chosen[chosen.index("--slice-step") + 1] == "1.0"


def test_the_renderer_does_not_inherit_the_private_buckets_credentials(monkeypatch):
    """The CT is public and served anonymously. Signing that request with keys
    for a different bucket returns 400 one second into a render, on a URL that
    answers 200 to curl -- while the worker still needs those keys to have
    fetched the flattened sheet a moment earlier."""
    sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
    import ink_worker

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HOME", "/root")
    assert "AWS_ACCESS_KEY_ID" not in ink_worker.runner_environment({"phase": "P4"})
    # P5 does lose them when it has nowhere to use them. This said its
    # checkpoint may come from the private bucket, which `checkpoint` being a
    # PATH_PARAMETER rules out -- it must be an absolute local path, so it is a
    # mount. The keys follow the store instead.
    assert "AWS_ACCESS_KEY_ID" not in ink_worker.runner_environment({"phase": "P5"})
    assert "AWS_ACCESS_KEY_ID" in ink_worker.runner_environment(
        {"phase": "P5", "parameters": {"artifact_store": "s3://helena/ink-maps-v1"}})
    assert ink_worker.runner_environment({"phase": "P4"})["HOME"] == "/root"
