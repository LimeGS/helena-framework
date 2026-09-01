"""Queue and lease ink jobs across however many hosts exist.

This deliberately mirrors the segmentation fleet in
``framework/stages/01-segmentation/fleet/postgres_store.py``: ``FOR UPDATE SKIP
LOCKED`` to claim, a hashed lease token so a stale worker cannot write over a
job that was taken from it, an attempt counter, and an event log. Adding a
second scheduling model next to a working one is how two schedulers end up
disagreeing about who owns a GPU.

What the panel may enqueue is constrained on purpose. A job names a *profile*
and a set of parameters that must validate against it; there is no free-form
command field. The worker builds the command line from the profile, so a caller
that can reach the API still cannot ask a GPU host to run arbitrary code.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import secrets
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Collection

MIGRATIONS = sorted((Path(__file__).resolve().parent / "migrations").glob("*.sql"))

CLAIMABLE_STATES = ("pending",)
TERMINAL_STATES = ("succeeded", "failed", "cancelled")


def _segmentation_fleet_module(name: str):
    """Load a shared segmentation-fleet module without colliding with this one."""

    package_name = "_helena_segmentation_fleet"
    if package_name not in sys.modules:
        package_root = Path(__file__).resolve().parents[2] / "01-segmentation/fleet"
        spec = importlib.util.spec_from_file_location(
            package_name, package_root / "__init__.py",
            submodule_search_locations=[str(package_root)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("segmentation fleet package cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{package_name}.{name}")


def _canonical_lineage_api():
    """Load the Task 6 shared guard without colliding with this fleet package."""

    return _segmentation_fleet_module("canonical_lineage")


def _surface_routing_api():
    """The frozen router itself, never a copy of its threshold or its strings."""

    return _segmentation_fleet_module("surface_routing")


def _qc_small_surface_diagnostic() -> str:
    """The QC-job state a routed-out surface parks in, from its one definition.

    Spelled the same as ``surface_routing.DIAGNOSTIC`` and deliberately not the
    same constant: one names a route and the other names a queue state, and the
    day either is renamed the other must not follow it silently.
    """

    return str(_segmentation_fleet_module("store").QC_SMALL_SURFACE_DIAGNOSTIC)


class RoutingRefused(RuntimeError):
    """This surface is not canonical downstream work, whatever the queue said.

    Not a failure of the job and emphatically not a result about the papyrus.
    A surface below the effort floor is diagnostic evidence; running a renderer
    or an ink screen over it produces an output that files next to a real one.
    """

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code

# Immutable server-derived control identity. These values are lineage only;
# command builders never translate them into runner flags.
CONTROL_BINDING_PARAMETERS = {
    "control_p0_artifact_id": str,
    "control_p0_artifact_sha256": str,
    "control_p0_selection_version": str,
    "control_source_snapshot_id": str,
    "control_source_content_lock": dict,
    "control_source_content_lock_sha256": str,
    "control_policy_sha256": str,
}

# Parameters whose value the server decides and a request may not carry.
#
# Where a run publishes is machine-room policy: `open_artifact_store` dispatches
# on the string, so any s3://bucket or http(s) host reaches it, and a worker
# publishing P1/P3/P5/P8 output still holds its own AWS credentials. A caller
# choosing that destination is a caller choosing where the evidence goes and
# whose keys pay for it.
#
# The control binding is the same shape: it says which P0 artifact and which
# source snapshot a job's lineage receipt claims, and a request that supplies
# its own is a request writing its own provenance.
#
# The panel already refused both. That refusal lived in the panel, and
# `enqueue` is reached by a harness, a script, or anything else holding a DSN
# -- the same reason the mission requirement is checked here rather than there.
SERVER_OWNED_PARAMETERS = frozenset({"artifact_store"}) | frozenset(
    CONTROL_BINDING_PARAMETERS)

# Parameters a job may carry. Anything else is rejected rather than passed
# through, so the surface the API exposes is this list and nothing more.
# One allowlist per phase. A job may carry exactly what its phase declares and
# nothing else, so widening the API is an explicit edit here rather than a
# consequence of somebody passing an extra key.
PHASE_PARAMETERS: dict[str, dict[str, type]] = {
    # Segmentation, for the backend that is a global fit rather than a grow.
    # The seeded grower is planned instead of parameterised -- cells, seeds and
    # a candidate policy, none of which is a job parameter -- and it keeps its
    # own bootstrap. The spiral fitter has no seed at all: it is told which
    # scroll, which slab of it, at what scale and which way the winding goes,
    # and those six are the whole decision.
    "P1": {"lane": str,
           # The six upstream assigns at module level. They are parameters here
           # because the rebind makes them so; see repin.py beside the adapter.
           "scroll_name": str, "dataset_path": str,
           "z_begin": int, "z_end": int, "voxel_size_um": float,
           "spiral_outward_sense": str,
           # Which lasagna scale and which tracks file the fit reads. Upstream
           # writes both as f-strings over dataset_path, so they are part of the
           # rebind rather than settings; they default to upstream's own values,
           # which is why they are not required.
           "lasagna_volume_name": str, "normal_zarr_group": str,
           "lasagna_scale": int, "tracks_file": str,
           # Two fits differing only in this are the only error bar this
           # geometry has; upstream accepts it as a config key, so it costs an
           # override rather than a source rewrite.
           "random_seed": int,
           "run_tag": str, "artifact_store": str, "dry_run": bool},
    # Both take a bounded batch. Neither takes a path: what they read comes from
    # the control plane, and where they publish is the fleet's artifact store.
    "P2": {"limit": int, "sample": str, "surface_id": str, "dry_run": bool},
    "P3": {"limit": int, "sample": str, "surface_id": str, "dry_run": bool,
           "binary": str, "profile": str, "artifact_store": str,
           "allow_unvalidated": bool},
    # Two renderers, and which one runs is part of what a layer stack means.
    # The official one takes any volume and eats a tifxyz directly; the vendored
    # one is built around one scroll and needs a PPM.
    "P4": {"lane": str,
           **CONTROL_BINDING_PARAMETERS,
           # vc_render_tifxyz
           "segmentation": str, "volume": str, "volume_key": str,
           "scale": float, "group_idx": int,
           # Or the sheet P3 unrolled, named rather than typed. The worker
           # resolves it out of surface_flattenings and fetches it, so the layer
           # stack is rendered on the flat parametrisation instead of the curved
           # patch -- which is what P4 was declared to consume all along.
           "flattened_surface": str, "flattening_profile": str,
           "flattening_id": str, "p3_job_id": str,
           "flattened_artifact_sha256": str,
           # int, not float: vc_render_tifxyz refuses "4.0" for --cache-gb and
           # the whole render dies on the argument before it reads a voxel. The
           # caster coerces 4.0 to 4, so a JSON number still works either way.
           "remote_url": str, "cache_gb": int,
           "source_voxel_um": float,
           # Where the layer stack goes, and how deep it is. num_slices and
           # slice_step are the depth decision this phase exists to make: N
           # slices at the wrong spacing hand the detector a slab of the wrong
           # thickness, which is P4's documented way of failing.
           "tif_output": str, "zarr_output": str,
           "num_slices": int, "slice_step": float,
           # Which way along the normal the slices go. The renderer's default
           # and the community's convention are opposite on the PHerc0139 mesh:
           # our layer 0 is their layer 85 and our 62 is their 23, each matching
           # at r = 0.99 in descending order. A depth-reversed slab is still a
           # correct render and it is the far side of the sheet first, which is
           # not what an ink model was trained on -- the map that came out
           # correlated 0.09 with the community's for the same recipe and the
           # same checkpoint. Explicit rather than defaulted: the direction is a
           # property of the mesh, and a receipt has to say which one it used.
           "flip_normals": bool,
           # Content address of the server-derived proof that selected the
           # boolean above. It is lineage, not a renderer command-line flag.
           "orientation_receipt_sha256": str,
           # Where the stack is published. P4's own output was the one artifact
           # in the pipeline that stayed on the worker that made it, which is
           # the machine most likely to be gone tomorrow.
           "artifact_store": str, "allow_local_layers": bool,
           "keep_local_layers": bool,
           # render_scroll3
           "ppm": str, "layers": int, "spacing": float,
           "concurrency": int, "stripe": int, "max_gb": float,
           "out_dir": str},
    "P5": {"tiff_dir": str, "layer_stack": str, "surface_volume": str,
           "surface_id": str,
           **CONTROL_BINDING_PARAMETERS,
           "checkpoint": str,
           # The TimeSformer adapter's own knobs, which the ResNet one has no
           # equivalent of. Both lists are the adapter table's, not a superset
           # somebody has to keep in their head.
           "depth_centers": str, "tiling_offsets": str, "tile_size": int,
           "source_slice_um": float, "model_family": str,
           "source_pixel_um": float, "depth_center": int, "stride": int,
           "model_config": str,
           "batch_size": int, "min_valid_ratio": float, "device": str,
           "on_degenerate": str, "artifact_store": str},
    "P7": {"map_path": str, "screening_of": str,
           **CONTROL_BINDING_PARAMETERS,
           "bbox": str, "px_um": float, "roi_receipt_sha256": str,
           "surface_id": str, "probability_map_artifact_sha256": str,
           "probability_map_manifest_sha256": str,
           # What the render repeats at, when the person queueing knows it: a
           # structure metric over an upsampled map peaks at the upsampling
           # factor, and naming the factor turns "something repeats here" into
           # "this is the grid". Optional -- without it the alarm falls back to
           # scale, which catches the same case less precisely.
           "render_cell_px": float,
           "out_dir": str},
    # wrap_order.py takes a scroll name and fetches public meshes itself; it
    # has no --segments flag, and the knob is --subsample, not axis_samples.
    "P8": {
           # Every P8 implementation names its lane.  The historical lanes use
           # scroll/out_path; vc3d-tifxyz-merge instead names immutable surface
           # artifact ids and a JSON grid.  There is deliberately no command or
           # binary parameter: those are properties of the frozen lane/profile.
           "lane": str,
           "scroll": str, "out_path": str, "work_dir": str, "subsample": int,
           "artifact_ids": list, "rows": list,
           "reference_artifact_id": str,
           "ransac_seed": int, "anchor_cap": int, "strip_cols": int,
           "artifact_store": str},
    # P9 composes the plates. make_plates.py fetches the official ink maps for
    # the scroll itself, so it takes no map: what it needs is where to write and
    # somewhere to stage what it downloads.
    "P9": {"scroll": str, "out_dir": str, "work_dir": str,
           "order_path": str, "ordering_of": str},
}
# What each parameter is, for a person. The queue has always known the names and
# the types; the panel kept its own copy of both plus the wording, which is a
# second list that agrees with this one until somebody adds a parameter to one of
# them. Every field the API accepts is described here, so a form drawn from this
# cannot be missing one.
#
# `filled_by_deployment` is not a field anyone types: where a render publishes is
# a property of the machine room, and the panel fills it in.
PARAMETER_HELP: dict[str, dict[str, Any]] = {
    # The spiral fitter's scroll selection. Upstream assigns all six at import
    # and none is one of its 105 config keys, so they are questions this form
    # asks and a source rewrite answers.
    "scroll_name": {
        "label": "Scroll, as the fitter names it",
        "placeholder": "s1",
        "note": "what fit_spiral.py calls this scroll internally; it need not "
                "match the catalogue name, and both are recorded"},
    "dataset_path": {
        "label": "Fitter dataset directory",
        "placeholder": "/artifacts/spiral/PHerc0172",
        "note": "holds the umbilicus, the tracks, the winding point collections "
                "and the lasagna normal volumes; the fit reads all of them from "
                "here and creates none of them"},
    "z_begin": {"label": "First slice", "placeholder": "4000",
                "note": "the bottom of the slab to fit, in CT slices"},
    "z_end": {"label": "Last slice", "placeholder": "17000",
              "note": "the top of the slab; a fit is a global winding solution "
                      "over this range, not a local patch"},
    "voxel_size_um": {
        "label": "Voxel size of that dataset", "placeholder": "9.6",
        "note": "microns per voxel of the volume the fitter reads, which may be "
                "a pooled copy rather than the scan's native scale"},
    "spiral_outward_sense": {
        "label": "Winding direction", "placeholder": "CW",
        "note": "which way the scroll winds outward, CW or CCW. A property of "
                "the scroll rather than a setting; the wrong one fits backwards"},
    "lasagna_volume_name": {
        "label": "Lasagna volumes", "placeholder": "las_008_{array}.ome.zarr",
        "note": "one name under lasagna_inputs/, carrying {array} for nx, ny "
                "and grad_mag; a set downloaded from the open-data bucket is "
                "named for its scroll instead of upstream's prepared one"},
    "normal_zarr_group": {
        "label": "Lasagna level", "placeholder": "4",
        "note": "which level inside those zarrs the fit reads. It uses one; "
                "downloading three is 57 times the data for nothing"},
    "lasagna_scale": {
        "label": "Lasagna scale", "placeholder": "4",
        "note": "the divisor the shape check uses. Left empty it is derived as "
                "2 to the level, which is the relation a downloaded pyramid "
                "satisfies; state it only to reproduce a prepared volume set "
                "that does not"},
    "tracks_file": {
        "label": "Tracks file", "placeholder": "2um_ds2_ps256_surf_v2.dbm",
        "note": "the .dbm under tracks/ the fit reads; upstream carries one "
                "alternative in a comment beside the constant"},
    "random_seed": {
        "label": "Seed", "placeholder": "1",
        "note": "two fits of one target differing only in this are the only "
                "error bar this geometry has: the fit publishes no uncertainty "
                "and its paper reports no run-to-run variability. Leave it empty "
                "for upstream's own, and queue a second job with another to get "
                "a pair"},
    "run_tag": {"label": "Run label",
                "note": "upstream's own tag for this fit, carried into its "
                        "checkpoints and diagnostics"},
    "limit": {"label": "Batch size", "note": "how many to consider in one run"},
    "sample": {"label": "Scroll",
               "note": "restrict the batch to one scroll; empty means every scroll in scope"},
    "surface_id": {"label": "Exact surface",
                   "note": "process only this immutable surface id; used for an explicit lineage continuation"},
    "dry_run": {"label": "Dry run", "note": "list what would be done and stop"},
    "allow_unvalidated": {
        "label": "Include surfaces the CT never confirmed",
        "note": "the default takes only surfaces the scan supports; this is the "
                "comparison against what the old gate admitted"},
    "binary": {"label": "vc_flatten binary",
               "note": "path to the flattening binary inside the worker image"},
    "profile": {"label": "Flattening profile",
                "note": "the profile whose parameters this run uses"},
    "artifact_store": {"label": "Where it publishes", "filled_by_deployment": True},
    "allow_local_layers": {"label": "Allow an unpublished stack",
                           "filled_by_deployment": True},
    "keep_local_layers": {"label": "Keep the local copy after publishing",
                          "filled_by_deployment": True},
    "lane": {"label": "Lane",
             "note": "which renderer runs; the default reads a tifxyz directly"},
    "segmentation": {"label": "Surface path",
                     "placeholder": "/surfaces/PHerc826/<id>",
                     "note": "a tifxyz on disk -- the curved surface"},
    "flattened_surface": {"label": "…or a flattened surface id",
                          "placeholder": "0e79f232-6e29-51ad-…",
                          "note": "a surface P3 unrolled; the worker fetches the sheet"},
    "flattening_profile": {"label": "Flattening profile of that sheet",
                           "note": "which flattening produced the sheet being rendered"},
    "flattening_id": {
        "label": "Exact flattened artifact",
        "note": "server-derived artifact id for the exact P3 sheet being rendered",
        "filled_by_deployment": True,
    },
    "flattened_artifact_sha256": {
        "label": "Flattened artifact content hash",
        "note": "server-derived content hash binding P4 to the exact P3 sheet",
        "filled_by_deployment": True,
    },
    "p3_job_id": {
        "label": "Source P3 job",
        "note": "server-derived flattening job that produced the named sheet",
        "filled_by_deployment": True,
    },
    "volume": {"label": "Volume", "placeholder": "/vol/scroll.zarr",
               "note": "where chunks are staged; with a remote URL it is a cache"},
    "volume_key": {
        "label": "Rescan volume key",
        "placeholder": "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr",
        "note": ("the masked OME-Zarr to render against, as a key under the "
                 "open-data bucket. Not `Volume` above, which is a path on "
                 "disk: this names which rescan, and the renderer reads the "
                 "array shape from its own .zarray. No default -- the wrong "
                 "rescan renders a plausible stack of the wrong scroll"),
    },
    "remote_url": {"label": "Remote OME-Zarr",
                   "placeholder": "https://…/PHerc0826/volumes/….zarr",
                   "note": "the volume to fetch from when it is not already on disk"},
    "source_voxel_um": {
        "label": "Source CT voxel size",
        "note": "server-derived physical voxel size used with render scale and "
                "slice step to derive depth spacing",
        "filled_by_deployment": True,
    },
    "scale": {"label": "Scale", "placeholder": "1.0",
              "note": "pixels per level-g voxel"},
    "group_idx": {"label": "Group index", "placeholder": "0",
                  "note": "resolution level to sample, 0 being full resolution"},
    "cache_gb": {"label": "Chunk cache (GB)", "placeholder": "4",
                 "note": "how much disk the staged chunks may use"},
    "num_slices": {"label": "Slices", "placeholder": "63",
                   "note": "N slices at the wrong spacing is this phase's way of "
                           "failing, so both this and the step are choices"},
    "slice_step": {"label": "Slice step", "placeholder": "1.0",
                   "note": "spacing along the normal, in voxels"},
    "flip_normals": {
        "label": "Reverse the direction along the normal",
        "note": "which way the slices go. On the PHerc0139 community mesh the "
                "renderer's default is the opposite of theirs, and a reversed "
                "slab is the far side of the sheet first: the ink map came out "
                "at r = 0.09 against theirs, and r = 0.885 with this on. "
                "Do NOT render twice to find out which: a lane that infers in "
                "both directions already gives you both answers from one "
                "render. Measured on the same window, flipped-render/forward "
                "and unflipped-render/reverse agree at r = 1.0000, so flipping "
                "here is exactly choosing which of P5's two outputs to read -- "
                "and rendering it a second time buys nothing but the render. "
                "The choice still matters: forward and reverse correlate at "
                "only r = 0.35 with each other. It is made by reading P5"},
    "orientation_receipt_sha256": {
        "label": "Orientation proof",
        "note": "server-derived content hash of the geometry and absolute-side "
                "proof that selected the normal direction",
        "filled_by_deployment": True,
    },
    "tif_output": {"label": "TIFF output directory",
                   "note": "where the numbered TIFFs go; defaults to layers/ in the run"},
    "zarr_output": {"label": "Zarr output",
                    "note": "also write a zarr; P4 still publishes numbered TIFFs for P5"},
    "ppm": {"label": "PPM file", "placeholder": "/path/to/segment.ppm",
            "note": "the per-pixel map the chunk-gather lane needs"},
    "out_dir": {"label": "Output directory",
                "note": "where this job writes; defaults to the run directory"},
    "layers": {"label": "Layers", "placeholder": "62",
               "note": "how many layers to write"},
    "spacing": {"label": "Spacing", "placeholder": "1.0",
                "note": "distance between sampled slices, in voxels"},
    "concurrency": {"label": "Concurrency", "placeholder": "8",
                    "note": "how many chunks to fetch at once"},
    "stripe": {"label": "Stripe",
               "note": "render one horizontal band only, for a cheap look first"},
    "max_gb": {"label": "Byte budget (GB)",
               "note": "upper bound on what this job may write"},
    "tiff_dir": {"label": "Layer stack directory",
                 "note": "a directory this worker can already see"},
    "layer_stack": {"label": "…or the render that produced one",
                    "placeholder": "p4-8d338e5a4a9445",
                    "note": "a P4 job; the worker fetches what it published, so "
                            "what a map was computed from is on the record"},
    "surface_volume": {"label": "…or a surface volume already at scale",
                       "placeholder": "s3://vesuvius-challenge-open-data/…/9.362um-….zarr",
                       "note": "a ~9.6 um isotropic OME-Zarr, read as it is rather "
                               "than pooled. The open-data bucket answers without a "
                               "credential, which is what lets a control be "
                               "reproduced from nothing but this field"},
    "checkpoint": {"label": "Checkpoint", "placeholder": "/models/…/model.safetensors",
                   "note": "the weights, on a path the worker can read; the digest is "
                           "verified before inference"},
    "source_pixel_um": {"label": "Source µm per pixel", "placeholder": "2.399",
                        "note": "stated, never defaulted: an 8.4% error moves the "
                                "recovered peak by tens of microns"},
    "model_config": {
        "label": "Model architecture config",
        "placeholder": "framework/registries/model-configs/<name>.json",
        "note": "for a checkpoint whose architecture is not the lane's frozen "
                "default. The TimeSformer runner has always accepted one and the "
                "large-scroll1 profile has always declared model_config_required, "
                "but there was no parameter to name the file -- so a profile with "
                "installed, digest-verified weights could not be run at all: the "
                "queue built a 6-head model and the checkpoint carries 8"},
    "source_slice_um": {"label": "Source µm per slice",
                        "note": "physical spacing between slices; this and the model's "
                                "training scale decide how deep the window really is"},
    "render_cell_px": {
        "label": "Render cell, in pixels",
        "note": "what the render repeats at -- mesh cell or upsampling factor. "
                "A structure score whose strongest period lands on it is "
                "measuring the grid: theirs read 0.85 on papyrus with no ink",
    },
    "roi_receipt_sha256": {
        "label": "Positive-control ROI proof",
        "note": "server-derived content hash binding the downstream-only "
                "letterform ROI and its exact transform into this P5 map",
        "filled_by_deployment": True,
    },
    "depth_center": {"label": "Depth centre",
                     "note": "which slice the model's window is centred on; left empty "
                             "the worker centres it on the stack"},
    "depth_centers": {"label": "Depth centres", "placeholder": "25,32,39",
                      "note": "several centres, combined by minimum -- a signal present "
                              "at only one depth disappears"},
    "tiling_offsets": {"label": "Tiling offsets",
                       "note": "shift the tile grid and combine, so a shape is not an "
                               "artefact of where the grid fell"},
    "tile_size": {"label": "Tile size",
                  "note": "edge of the square the model sees, in stack pixels"},
    "stride": {"label": "Stride",
               "note": "how far the tile moves; smaller overlaps more and costs more"},
    "batch_size": {"label": "Batch size",
                   "note": "tiles per forward pass; lower it if the card runs out of memory"},
    "min_valid_ratio": {"label": "Minimum valid ratio",
                        "note": "skip a tile with less real data than this, so padding is "
                                "never scored as signal"},
    "device": {"label": "Device", "placeholder": "auto",
               "note": "empty or 'auto' uses the card if the worker that claims "
                       "this has one and the CPU if it does not; 'cuda:0' "
                       "requires a card and refuses without one; 'cpu' says the "
                       "slower device is acceptable here"},
    "on_degenerate": {"label": "On a degenerate map",
                      "note": "what to do when the map carries no decision: fail the "
                              "job, or record it and continue"},
    "model_family": {"label": "Model family", "filled_by_deployment": True},
    "map_path": {"label": "Probability map", "placeholder": "/path/to/probability.npy",
                 "note": "a file this worker can already see"},
    "screening_of": {"label": "…or the screening that produced one",
                     "placeholder": "p5-01facc430f694c",
                     "note": "a P5 job; the worker fetches its map and reads the "
                             "liveness verdict from the job rather than from a "
                             "file beside a path"},
    "probability_map_artifact_sha256": {
        "label": "Probability-map content hash",
        "note": "server-derived aggregate hash binding P7 to the exact P5 map",
        "filled_by_deployment": True,
    },
    "probability_map_manifest_sha256": {
        "label": "Probability-map manifest hash",
        "note": "server-derived manifest hash binding P7 to the exact P5 inventory",
        "filled_by_deployment": True,
    },
    "bbox": {"label": "Bounding box", "placeholder": "x0,y0,x1,y1",
             "note": "restrict the screen to one region of the map"},
    "px_um": {"label": "µm per pixel", "placeholder": "2.399",
              "note": "so shape sizes can be argued in microns rather than pixels"},
    "scroll": {"label": "Scroll", "note": "the scroll this run belongs to"},
    "out_path": {"label": "Output path", "note": "the file this job writes"},
    "work_dir": {"label": "Work directory",
                 "note": "scratch for intermediates, and for anything this job "
                         "fetches; never the published output"},
    "subsample": {"label": "Subsample",
                  "note": "keep every Nth point; a coarser assembly that costs less"},
    "order_path": {"label": "Measured wrap order",
                   "note": "a wrap_radial.json this worker can already see"},
    "ordering_of": {"label": "…or the P8 job that measured it",
                    "placeholder": "p8-01facc430f694c",
                    "note": "the worker resolves the successful P8 output; P9 never "
                            "falls back to a transcribed order"},
    "artifact_ids": {
        "label": "Certified TIFXYZ artifacts",
        "note": "two or more content-addressed surface ids; the worker resolves "
                "and verifies them rather than accepting filesystem paths"},
    "rows": {
        "label": "Surface layout",
        "note": "a connected JSON row grid containing every input exactly once"},
    "reference_artifact_id": {
        "label": "Reference surface",
        "note": "one of the input ids, pinned as vc_merge_tifxyz --ref"},
    "ransac_seed": {
        "label": "RANSAC seed",
        "note": "a deterministic nonzero seed; zero is refused"},
    "anchor_cap": {
        "label": "Anchor cap",
        "note": "0 keeps every upstream anchor; positive values use VC3D's own decimator"},
    "strip_cols": {
        "label": "Strip columns",
        "note": "0 performs the canonical single-pass N-way blend"},
}

for _control_field in CONTROL_BINDING_PARAMETERS:
    PARAMETER_HELP[_control_field] = {
        "label": _control_field.replace("_", " "),
        "note": "immutable server-derived First Letters control identity",
        "filled_by_deployment": True,
    }

# Where the queue needs one of two names and not both. "required" cannot say it:
# neither is required and exactly one must be there.
EXACTLY_ONE_OF: dict[str, tuple[dict[str, Any], ...]] = {
    "P4": ({"lane": "vc-render-tifxyz",
            "names": ("segmentation", "flattened_surface")},),
    # Three ways of naming one input: a layer stack on disk, a published P4 job
    # whose stack is fetched, or a surface volume already at the model's scale.
    # The third was missing, so a public input -- a ready OME-Zarr in the
    # open-data bucket, which is what makes the control reproducible without a
    # credential -- could not be queued at all, and the control that used one
    # had to run beside this platform rather than through it.
    "P5": ({"names": ("tiff_dir", "layer_stack", "surface_volume")},),
    "P7": ({"names": ("map_path", "screening_of")},),
    "P9": ({"names": ("order_path", "ordering_of")},),
}


GPU_OBSERVATION_TTL_SECONDS = 3600


def merge_gpu_observations(stored: dict[str, Any], incoming: dict[str, Any],
                           *, now: float | None = None) -> list[dict[str, Any]]:
    """The cards this host has, from every worker that can see any of them.

    Keyed by uuid, because index is per-container: a worker pinned to the second
    card calls it index 0, and merging on index would collapse two cards into
    one. Each card carries when it was last seen, so one that is genuinely gone
    disappears an hour later rather than never.
    """
    moment = time.time() if now is None else now
    cards: dict[str, dict[str, Any]] = {}
    for card in (stored.get("gpus") or []):
        key = str(card.get("uuid") or card.get("name") or card.get("index"))
        seen = float(card.get("seen_at") or 0.0)
        if moment - seen <= GPU_OBSERVATION_TTL_SECONDS:
            cards[key] = card
    for card in (incoming.get("gpus") or []):
        key = str(card.get("uuid") or card.get("name") or card.get("index"))
        cards[key] = {**card, "seen_at": moment,
                      "seen_by": incoming.get("hostname")}
    return sorted(cards.values(), key=lambda card: str(card.get("uuid") or ""))


# Parameters that name another job, and the phase whose jobs they may name.
#
# `screening_of` is a P5 job id and `ordering_of` a P8 one: the worker fetches
# that job's output instead of being handed a path. The form had no way to know
# that, so both were free-text boxes for an opaque identifier -- and the id had
# to be found on another page and typed back in, with a refusal at the queue as
# the only check that it was the right kind of thing.
JOB_INPUT_PARAMETERS: dict[str, str] = {
    "screening_of": "P5",
    "ordering_of": "P8",
}


def phase_parameter_schema(phase: str) -> dict[str, Any]:
    """Every parameter this phase accepts, as a form can draw it.

    Served rather than duplicated: the panel used to carry its own list of these
    and every parameter added to the queue was invisible in the browser until
    somebody remembered to add it there too.
    """
    accepted = PHASE_PARAMETERS.get(phase, {})
    required = set(PHASE_REQUIRED.get(phase, ()))
    lane_of: dict[str, str | list[str]] = {}
    lanes = []
    if phase == "P4":
        for lane_id, lane in P4_LANES.items():
            lanes.append({"id": lane_id, "name": lane["name"], "note": lane["note"],
                          "validated": lane.get("validated"),
                          "required": list(lane["required"])})
            for name in lane["required"]:
                lane_of[name] = lane_id
    elif len(PHASE_LANES.get(phase) or {}) > 1 or any(
            lane.get("profiles") for lane in (PHASE_LANES.get(phase) or {}).values()):
        # Every phase that has grown a second way of doing it says so here, so
        # the form offers the choice and the guide documents it without either
        # of them being told about the new module.
        #
        # A single lane is described too when it pins its profiles. That is not
        # decoration: the queue refuses a job whose profile_id is not on the
        # lane's list, and a form with no way to learn the list can only send
        # nothing and be refused. P1 is the case -- one lane, one frozen
        # profile -- and the browser reads the id from here.
        lane_owners: dict[str, list[str]] = {}
        for lane_id, lane in PHASE_LANES[phase].items():
            lanes.append({"id": lane_id, "name": lane.get("name", lane_id),
                          "note": lane.get("note"), "validated": lane.get("validated"),
                          "required": list(lane.get("required", ())),
                          "profiles": list(lane.get("profiles", ()))})
            for name in lane.get("required", ()):
                lane_owners.setdefault(name, []).append(lane_id)
        for name, owners in lane_owners.items():
            # A field can belong to one lane or to several implementations that
            # share the same contract. The browser understands both forms.
            lane_of[name] = owners[0] if len(owners) == 1 else owners
    fields = []
    for name, kind in accepted.items():
        help_text = PARAMETER_HELP.get(name, {})
        fields.append({
            "name": name,
            "type": {int: "integer", float: "number", bool: "boolean",
                     list: "json"}.get(kind, "text"),
            "required": name in required or name in lane_of,
            "lane": lane_of.get(name) or help_text.get("lane"),
            "label": help_text.get("label", name.replace("_", " ")),
            "note": help_text.get("note"),
            "placeholder": help_text.get("placeholder"),
            "filled_by_deployment": bool(help_text.get("filled_by_deployment")),
            # Which phase's jobs this field may name, so a form can offer them
            # rather than ask for an id to be remembered.
            "names_a_job_from": JOB_INPUT_PARAMETERS.get(name),
        })
    return {"phase": phase, "fields": fields, "lanes": lanes,
            "exactly_one_of": [dict(rule) | {"names": list(rule["names"])}
                               for rule in EXACTLY_ONE_OF.get(phase, ())]}


PHASE_REQUIRED: dict[str, tuple[str, ...]] = {
    # Five of the six, and where the windings go. The winding sense is the one
    # with a profile default, because CW is upstream's and a fit made with the
    # wrong one is visibly wrong rather than quietly so. The other five have
    # none on purpose: defaulting them to Scroll 1's values would let a
    # forgotten field fit the wrong scroll under the right name.
    "P1": ("scroll_name", "dataset_path", "z_begin", "z_end", "voxel_size_um",
           "artifact_store"),
    "P2": (),
    # Flattening publishes, and a sheet published nowhere is a sheet the next
    # phase cannot read -- the same way a surface on a worker's disk was.
    "P3": ("artifact_store",),
    # Nothing is required of P4 as a phase: the two renderers need different
    # things, and vc_render_tifxyz writes beside the segmentation rather than to
    # an --out. What each lane needs is in P4_LANES and enforced there.
    "P4": (),
    # Not tiff_dir: a job may name the render that produced the stack instead,
    # and the worker fetches it. Not upstream_dir either: only one of the two
    # adapters takes it. Both are checked against the adapter the profile names.
    #
    # Not source_pixel_um either, any more. It is the scale a stack was rendered
    # at, and pooling needs it -- but a job naming a surface volume already at
    # the model's scale has nothing to pool, and demanding the number anyway
    # would be asking for a fact about an input this job does not have. Required
    # with a stack, checked below where which input was named is known.
    "P5": ("checkpoint",),
    # Not map_path: a job may name the P5 screening whose map it is about, and
    # the worker fetches it. Exactly one of the two, checked below.
    "P7": ("bbox", "px_um"),
    # P8 has three lanes with different contracts.  Their own `required`
    # declarations are enforced below; a global scroll/out_path requirement
    # would make the TIFXYZ merge impossible to enqueue.
    "P8": (),
    "P9": ("scroll", "out_dir"),
}
# Paths are checked for absoluteness and traversal; these are the keys that are
# paths in any phase.  `segments_dir` was in this set and is not a parameter of
# any phase, which is how `segmentation`, `volume` and `work_dir` -- which are
# -- stayed out of it: the set is hand-kept, so it drifts in both directions.
PATH_PARAMETERS = {"tiff_dir", "checkpoint", "ppm", "out_dir",
                   "map_path", "out_path", "dataset_path",
                   "tif_output", "zarr_output", "order_path",
                   "segmentation", "volume", "work_dir"}

# Of those, the ones a runner writes to rather than reads from.
#
# Absolute-and-no-`..` is a shape, not a bound: it admits every absolute path on
# the host. These are where the bytes land, so an unbounded one overwrites
# another mission's published artifacts or drops a file into the worker's own
# checkout -- and for P9 the artifact record points at `out_dir`, so the
# mission's index can be made to name any location on the box.
#
# Bounded at the worker rather than here, because the bound is the worker's run
# directory and the queue does not know it.
# Not `map_path`: P7 reads the probability map through it, and the worker sets
# it itself to wherever it fetched the map to. Its own docstring calls it an
# input, and treating it as a destination refused every P7 job whose map came
# from another worker -- which is all of them.
# Inputs that have to exist before the runner starts, so a worker can say
# whether it can see them rather than leaving a renderer to say `Error loading`
# and exit 0.
#
# Not `dataset_path`: P1 stages into it, so it is legitimately absent when the
# job begins. Not `volume` either -- the renderer streams the CT from a remote
# URL and uses that path as its own cache, so an absent one is the ordinary
# first run rather than a mistake.
READ_PATH_PARAMETERS = frozenset({
    "tiff_dir", "checkpoint", "ppm", "order_path", "segmentation",
    # The architecture a checkpoint was trained with. Read like any other input
    # so a worker that cannot see it refuses before claiming, rather than
    # loading the weights into the wrong shape and failing on a size mismatch.
    "model_config",
})

WRITE_PATH_PARAMETERS = frozenset({
    "out_dir", "out_path", "tif_output", "zarr_output", "work_dir",
})

# `binary` names the executable P3 hands to subprocess.run as argv[0], so it is
# not a path parameter, it is the choice of what runs.  An absolute path with no
# '..' is not a bound on that: it names any executable on the host.  A run may
# only ask for one of these.
ALLOWED_BINARIES = frozenset({"/opt/campaignx/vc3d/bin/vc_flatten"})

# A type says how a value is represented; these sets say whether it can describe
# real work.  Without the second half, zero is silently omitted by several argv
# builders (and the runner uses its default while the receipt records zero), and
# negative/non-finite sizes reach numpy, range(), or a GPU binary as failed jobs.
STRICTLY_POSITIVE_PARAMETERS = frozenset({
    "limit", "scale", "cache_gb", "num_slices", "slice_step",
    "layers", "spacing", "concurrency", "stripe", "max_gb",
    "source_voxel_um", "source_pixel_um", "source_slice_um", "stride", "batch_size",
    "tile_size", "min_valid_ratio", "px_um", "subsample",
    # Every micron figure a fit produces is derived from this one.
    "voxel_size_um",
})
NON_NEGATIVE_PARAMETERS = frozenset({
    "group_idx", "depth_center", "anchor_cap", "strip_cols",
    "z_begin", "z_end",
    # Both were in neither set, so a negative reached the fitter: the scale is
    # a power-of-two level and the seed is fed to the optimizer's RNG.
    "lasagna_scale", "random_seed", "optimizer_random_seed",
})

# And an upper bound, because a lower one alone is only half a domain. Every
# one of these sizes a loop, an allocation or a stream on a shared GPU host,
# and Python integers do not stop being valid at any particular size -- so
# `num_slices: 10**12` was a queued job that a worker would take a lease on and
# then discover. The numbers are generous: they bound the absurd, not the
# ambitious.
PARAMETER_CEILINGS: dict[str, float] = {
    "limit": 10_000,
    "layers": 4_096,
    "num_slices": 100_000,
    "slice_step": 10_000.0,
    "cache_gb": 4_096,
    "max_gb": 4_096,
    "concurrency": 256,
    "stripe": 65_536,
    "tile_size": 8_192,
    "batch_size": 4_096,
    "frames": 1_024,
    "stride": 8_192,
    "subsample": 1_000_000,
    "anchor_cap": 1_000_000,
    "strip_cols": 65_536,
    "spacing": 100_000.0,
    "scale": 1_024.0,
    "group_idx": 64,
    # A level, not a factor: 2**32 voxels is already past any scroll.
    "lasagna_scale": 32,
    # Microns. A scroll is millimetres across and these are per-pixel figures.
    "source_voxel_um": 10_000.0,
    "source_pixel_um": 10_000.0,
    "source_slice_um": 10_000.0,
    "voxel_size_um": 10_000.0,
    "px_um": 10_000.0,
    "min_valid_ratio": 1.0,
    # The scroll's own z extent in voxels.
    "z_begin": 100_000,
    "z_end": 100_000,
}

ALLOWED_PARAMETERS = PHASE_PARAMETERS["P5"]
REQUIRED_PARAMETERS = PHASE_REQUIRED["P5"]


class JobRejected(ValueError):
    """The request did not describe a runnable job."""


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def coerce_parameter(key: str, value: Any, caster: type) -> Any:
    """Normalize a scalar without changing the decision the client expressed."""
    if caster is bool:
        if type(value) is not bool:
            raise JobRejected(f"{key} is not a boolean: {value!r}")
        return value
    if caster is int:
        # bool is an int subclass, and int(1.5) truncates.  Both turn one JSON
        # decision into another before it reaches the receipt.
        if isinstance(value, bool):
            raise JobRejected(f"{key} is not an integer: {value!r}")
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            raise JobRejected(f"{key} is not an integer: {value!r}")
    elif caster is float:
        if isinstance(value, bool):
            raise JobRejected(f"{key} is not a number: {value!r}")
    elif caster is list:
        # list("ab") becoming two artifact ids is valid Python and invalid JSON
        # semantics.  Arrays are the only representation accepted here.
        if not isinstance(value, list):
            raise JobRejected(f"{key} is not an array: {value!r}")
        return list(value)
    elif caster is str:
        # str(["/volume"]) prints a plausible-looking value which no runner can
        # open.  Paths and identifiers arrive as JSON strings or are refused.
        if not isinstance(value, str):
            raise JobRejected(f"{key} is not a string: {value!r}")
        return value

    try:
        normalized = caster(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise JobRejected(f"{key} is not a {caster.__name__}: {value!r}") from exc
    if isinstance(normalized, float) and not math.isfinite(normalized):
        raise JobRejected(f"{key} must be finite: {value!r}")
    return normalized


def validate_parameters(parameters: dict[str, Any], phase: str = "P5", *,
                        server_owned: Collection[str] = ()) -> dict[str, Any]:
    """Coerce and bound every parameter against its phase, or refuse the job.

    `server_owned` names the keys the server itself supplied on this call.
    Anything in SERVER_OWNED_PARAMETERS that is not among them came from the
    request and is refused -- not dropped, because a job that publishes
    somewhere other than where it was told to publish should fail loudly rather
    than quietly do the right thing and teach nobody.
    """
    smuggled = sorted(SERVER_OWNED_PARAMETERS.intersection(parameters)
                      - set(server_owned))
    if smuggled:
        raise JobRejected(
            f"these are the server's to decide, not the request's: {smuggled}")
    allowed = PHASE_PARAMETERS.get(phase)
    if allowed is None:
        raise JobRejected(
            f"phase {phase} has no queueable parameters; "
            f"queueable phases are {sorted(PHASE_PARAMETERS)}")
    unknown = set(parameters) - set(allowed)
    if unknown:
        raise JobRejected(f"unknown parameters for {phase}: {sorted(unknown)}")
    missing = [k for k in PHASE_REQUIRED[phase] if parameters.get(k) in (None, "")]
    if missing:
        raise JobRejected(f"missing required parameters for {phase}: {missing}")

    clean: dict[str, Any] = {}
    for key, value in parameters.items():
        if value is None:
            continue
        caster = allowed[key]
        clean[key] = coerce_parameter(key, value, caster)

    for key, candidate in clean.items():
        if key in PATH_PARAMETERS:
            if not str(candidate).startswith("/") or ".." in Path(str(candidate)).parts:
                raise JobRejected(f"{key} must be an absolute path without '..': {candidate!r}")
        if key == "binary" and str(candidate) not in ALLOWED_BINARIES:
            raise JobRejected(
                f"binary is what P3 executes, not a path it reads: {candidate!r} "
                f"is not one of {sorted(ALLOWED_BINARIES)}")
        if key in STRICTLY_POSITIVE_PARAMETERS and candidate <= 0:
            raise JobRejected(f"{key} must be greater than zero: {candidate!r}")
        if key in NON_NEGATIVE_PARAMETERS and candidate < 0:
            raise JobRejected(f"{key} cannot be negative: {candidate!r}")
        ceiling = PARAMETER_CEILINGS.get(key)
        if ceiling is not None and candidate > ceiling:
            raise JobRejected(
                f"{key} is bounded above at {ceiling} and this job asks for "
                f"{candidate!r}; a size nothing rejects is a lease a worker "
                "takes before it finds out")
    if "source_pixel_um" in clean and not 0.01 <= clean["source_pixel_um"] <= 1000:
        raise JobRejected(f"source_pixel_um out of range: {clean['source_pixel_um']}")
    if "min_valid_ratio" in clean and clean["min_valid_ratio"] > 1:
        raise JobRejected(f"min_valid_ratio out of range: {clean['min_valid_ratio']}")
    if clean.get("on_degenerate") not in (None, "fail", "warn"):
        raise JobRejected("on_degenerate must be 'fail' or 'warn'")
    # `auto` is the default and means "the card if the worker that claims this
    # has one". An explicit `cuda` is a requirement the runner refuses to
    # silently downgrade, and `cpu` is a caller saying plainly that the slower
    # device is acceptable for this run.
    if (clean.get("device") not in (None, "auto", "cpu")
            and not str(clean.get("device", "")).startswith("cuda")):
        raise JobRejected("device must be 'auto', 'cpu' or start with 'cuda'")
    if phase == "P1":
        sense = clean.get("spiral_outward_sense")
        if sense not in (None, "", "CW", "CCW"):
            raise JobRejected(
                f"spiral_outward_sense must be CW or CCW, got {sense!r}. The "
                "winding direction is a property of the scroll, and the wrong "
                "one fits the spiral backwards.")
        if clean.get("z_end") is not None and clean.get("z_begin") is not None:
            if int(clean["z_end"]) <= int(clean["z_begin"]):
                raise JobRejected(
                    f"z_end ({clean['z_end']}) must be above z_begin "
                    f"({clean['z_begin']}); an empty slab fits nothing")
    if phase == "P7":
        named = [k for k in ("map_path", "screening_of") if clean.get(k) not in (None, "")]
        if len(named) != 1:
            raise JobRejected(
                "name exactly one map to screen: `map_path` for a file this "
                "worker can already see, or `screening_of` for the id of the P5 "
                "job that produced one"
                + (f"; got {named}" if named else ""))
        if clean.get("screening_of"):
            hashes = (clean.get("probability_map_artifact_sha256"),
                      clean.get("probability_map_manifest_sha256"))
            if not all(isinstance(value, str)
                       and re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in hashes):
                raise JobRejected(
                    "a named screening requires exact P5 probability-map identity: "
                    "probability_map_artifact_sha256 and "
                    "probability_map_manifest_sha256")
        if clean.get("map_path"):
            # A named screening carries its verdict in the job row, and the
            # worker checks it there when it fetches the map.
            refuse_dead_map(Path(str(clean["map_path"])))
    if phase == "P9":
        named = [k for k in ("order_path", "ordering_of")
                 if clean.get(k) not in (None, "")]
        if len(named) != 1:
            raise JobRejected(
                "name exactly one measured wrap order: `order_path` for a "
                "wrap_radial.json this worker can see, or `ordering_of` for "
                "the P8 job that produced it"
                + (f"; got {named}" if named else ""))
    if phase == "P4":
        lane_id = str(clean.get("lane") or DEFAULT_P4_LANE)
        lane = P4_LANES.get(lane_id)
        if lane is None:
            raise JobRejected(
                f"unknown P4 lane {lane_id!r}; lanes are {sorted(P4_LANES)}")
        missing = [k for k in lane["required"] if clean.get(k) in (None, "")]
        # A tifxyz can be named two ways and the lane needs exactly one of them.
        # Both is ambiguous about what was rendered, which is the one thing a
        # layer stack's provenance has to be certain of.
        if lane_id == "vc-render-tifxyz":
            named = [k for k in ("segmentation", "flattened_surface")
                     if clean.get(k) not in (None, "")]
            missing = [k for k in missing if k != "segmentation"]
            if len(named) != 1:
                raise JobRejected(
                    "name exactly one surface to render: `segmentation` for a "
                    "path to a tifxyz, or `flattened_surface` for a surface id "
                    "whose flattened sheet P3 published"
                    + (f"; got {named}" if named else ""))
            if clean.get("flattened_surface"):
                exact = (clean.get("flattening_id"), clean.get("p3_job_id"),
                         clean.get("flattened_artifact_sha256"))
                if (not all(isinstance(value, str) and value for value in exact)
                        or not re.fullmatch(
                            r"[0-9a-f]{64}",
                            str(clean.get("flattened_artifact_sha256") or ""))):
                    raise JobRejected(
                        "a flattened surface requires exact P3 flattening identity: "
                        "flattening_id, p3_job_id, and flattened_artifact_sha256")
        if missing:
            raise JobRejected(
                f"{lane['name']} needs {missing}. "
                # Both lanes render any scroll now; what separates them is the
                # input. The message used to say the PPM lane was Scroll 3 only,
                # which stopped being true when its renderer took the volume as
                # an argument instead of hardcoding one.
                + ("The other lane takes a tifxyz, or the surface id of a sheet "
                   "P3 flattened." if lane_id == "chunk-gather" else
                   "The other lane takes a PPM and the volume to render it "
                   "against."))
    if phase == "P5":
        # The same rule P4 has for its surface: a stack can be named two ways,
        # and which one a probability map was computed from is exactly what its
        # provenance has to be certain of.
        named = [k for k in ("tiff_dir", "layer_stack", "surface_volume")
                 if clean.get(k) not in (None, "")]
        if len(named) != 1:
            raise JobRejected(
                "name exactly one input: `tiff_dir` for a layer stack this "
                "worker can already see, `layer_stack` for the id of the P4 "
                "render that published one, or `surface_volume` for an "
                "OME-Zarr already at the model's scale"
                + (f"; got {named}" if named else ""))
        if named[0] != "surface_volume" and clean.get("source_pixel_um") in (None, ""):
            # Pooling has to know what it is pooling from. Guessing 2.4 would
            # turn a native 9 um render into a 38 um one, which is the failure
            # the adapter refuses at run time; refused here it costs no lease.
            raise JobRejected(
                "missing required parameters for P5: ['source_pixel_um'] -- the "
                "scale the layer stack was rendered at, which pooling needs")
    if phase == "P8":
        lane_id = str(clean.get("lane") or "column-atlas")
        if lane_id == "vc3d-tifxyz-merge":
            artifacts = clean.get("artifact_ids")
            rows = clean.get("rows")
            reference = clean.get("reference_artifact_id")
            if not isinstance(artifacts, list) or not 2 <= len(artifacts) <= 64:
                raise JobRejected(
                    "vc3d-tifxyz-merge needs between 2 and 64 artifact_ids")
            if any(not isinstance(value, str) or not value.strip()
                   for value in artifacts):
                raise JobRejected("every merge artifact id must be a non-empty string")
            if len(set(artifacts)) != len(artifacts):
                raise JobRejected("merge artifact_ids must be distinct")
            if reference not in artifacts:
                raise JobRejected(
                    "reference_artifact_id must name one of artifact_ids")
            if int(clean.get("ransac_seed") or 0) <= 0:
                raise JobRejected("ransac_seed must be a deterministic nonzero integer")
            if int(clean.get("anchor_cap") or 0) < 0:
                raise JobRejected("anchor_cap cannot be negative")
            if int(clean.get("strip_cols") or 0) < 0:
                raise JobRejected("strip_cols cannot be negative")
            if not isinstance(rows, list) or not rows:
                raise JobRejected("rows must be a non-empty JSON array")
            if any(not isinstance(row, list) or not row for row in rows):
                raise JobRejected("every rows entry must be a non-empty JSON array")
            width = len(rows[0])
            if any(len(row) != width for row in rows):
                raise JobRejected("rows must be rectangular; use null for empty cells")
            flattened: list[str] = []
            positions: dict[str, tuple[int, int]] = {}
            for row_index, row in enumerate(rows):
                for column_index, value in enumerate(row):
                    if value is None:
                        continue
                    if not isinstance(value, str) or not value.strip():
                        raise JobRejected("layout cells must be artifact ids or null")
                    flattened.append(value)
                    positions[value] = (row_index, column_index)
            if len(flattened) != len(set(flattened)):
                raise JobRejected("each artifact may appear in rows only once")
            if set(flattened) != set(artifacts):
                raise JobRejected(
                    "rows must contain every artifact_id exactly once and no others")
            # The upstream tool derives 4-neighbour edges from this grid.  A
            # disconnected declaration would otherwise run separate sheets and
            # call their shared output one assembly.
            visited = {flattened[0]}
            pending = [flattened[0]]
            by_position = {position: value for value, position in positions.items()}
            while pending:
                value = pending.pop()
                row_index, column_index = positions[value]
                for neighbour in ((row_index - 1, column_index),
                                  (row_index + 1, column_index),
                                  (row_index, column_index - 1),
                                  (row_index, column_index + 1)):
                    other = by_position.get(neighbour)
                    if other is not None and other not in visited:
                        visited.add(other)
                        pending.append(other)
            if visited != set(artifacts):
                raise JobRejected("rows describes a disconnected surface layout")

    # Lane-local requirements are authoritative for phases whose
    # implementations consume different shapes.  This also rejects an unknown
    # lane at enqueue time instead of letting a worker burn an attempt on it.
    if phase != "P4":
        _, lane = lane_for({"phase": phase, "parameters": clean})
        missing_for_lane = [name for name in lane.get("required", ())
                            if clean.get(name) in (None, "")]
        if missing_for_lane:
            raise JobRejected(
                f"{lane.get('name', phase)} needs {missing_for_lane}")
    return clean


def refuse_dead_map(map_path: Path) -> None:
    """P7 does not screen a map the lane could not read.

    The pipeline contract has claimed since it was written that P7 consumes "a
    probability map that passed P6", and until now nothing checked: a DEGENERATE
    map -- one whose values barely move -- could be screened, and screening finds
    shapes in noise perfectly well.

    The verdict is a field of the P5 receipt beside the map. A missing receipt is
    refused too: a map with no provenance is not a map that passed anything.
    """
    receipt = map_path.parent / "INK_PROFILE_RECEIPT.json"
    if not receipt.exists():
        raise JobRejected(
            f"no INK_PROFILE_RECEIPT.json beside {map_path.name}; a map with no "
            "receipt has no liveness verdict, and P7 does not screen one")
    try:
        verdict = (json.loads(receipt.read_text()).get("liveness") or {}).get("verdict")
    except (json.JSONDecodeError, OSError) as exc:
        raise JobRejected(f"{receipt} cannot be read: {exc}") from exc
    if verdict is None:
        raise JobRejected(
            f"{receipt.name} records no liveness verdict; it predates the check, "
            "so whether the lane read anything is unknown")
    if verdict != "ALIVE":
        raise JobRejected(
            f"the map is {verdict}, not ALIVE: screening a map the lane could not "
            "read finds shapes in noise. Re-run the lane, or screen a live map.")


# P4 renders a surface into a layer stack, and there are two ways to do it.
#
# The default is volume-cartographer's own renderer, which takes the volume as
# an argument and reads a tifxyz directly -- so it works for any scroll. It is a
# binary from the VC3D runtime, not a script in this repository, which is why
# a lane names an executable rather than a path.
#
# The vendored chunk-gather renderer stays as an alternative because it earns
# its place on Scroll 3: it computes the byte budget per stripe before
# downloading, checkpoints each stripe so a crash resumes, and validates at
# r = 0.89 against an official ink map. It is also pinned to PHerc0332 by
# construction -- it bridges the legacy segment frame through that rescan's
# published transform and asserts the landmarks agree -- so it renders Scroll 3
# and nothing else.
#
# They have not been compared against each other on the same surface. Until they
# are, r = 0.89 is a statement about the vendored lane only.
P4_LANES: dict[str, dict[str, Any]] = {
    "vc-render-tifxyz": {
        "name": "volume-cartographer renderer",
        "executable_env": "VC3D_RENDER_BINARY",
        "fallback": "/opt/campaignx/vc3d/bin/vc_render_tifxyz",
        "required": ("segmentation", "volume", "scale", "group_idx"),
        "any_scroll": True,
        "validated": None,
        "note": ("Takes the volume as an argument and reads a tifxyz directly -- "
                 "either a path to one, or the surface id of a sheet P3 flattened, "
                 "which the worker fetches. Rendering the flattened sheet puts "
                 "the layer stack on the unrolled parametrisation, so a page is "
                 "a page rather than a curved patch. It writes per-slice TIFFs "
                 "to --tif-output, which defaults to layers/ inside the job's "
                 "own run directory."),
    },
    "chunk-gather": {
        "name": "chunk-gather renderer (PPM, any rescan)",
        "script": "framework/vendored/scroll-streaming-tools/render_scroll.py",
        # `volume_key`, not `volume`: the other lane's `volume` is an absolute
        # path to a zarr on disk and is validated as one, and this is a key
        # under the open-data bucket. Two meanings behind one name would have
        # made the queue refuse every legitimate value here.
        #
        # Required and no default on purpose: rendering a segment against the
        # wrong rescan produces a plausible layer stack of the wrong scroll,
        # which no downstream check would catch.
        "required": ("ppm", "volume_key"),
        "any_scroll": True,
        # It was pinned to PHerc0332 because render_scroll3.py hardcoded that
        # rescan's volume key, array shape and voxel size -- and render_scroll2
        # was the same file with those three edited, which is how a second
        # scroll got added. render_scroll.py takes the key as an argument and
        # reads the shape and chunk edge from the volume's own .zarray, so the
        # lane stopped being about one scroll.
        "validated": "r = 0.89 against an official ink map, on PHerc0332",
        "note": ("Bridges a legacy segment frame through the target rescan's "
                 "published transform and asserts the landmarks agree, so a "
                 "volume that does not correspond to the PPM fails before any "
                 "stripe is written. Streams by stripe with a byte budget and "
                 "resumes from done.json. The r = 0.89 control was measured on "
                 "PHerc0332; another rescan inherits the method, not the "
                 "number."),
    },
}
DEFAULT_P4_LANE = "vc-render-tifxyz"


# Where each phase's work actually runs. Paths are relative to the repository
# root; the vendored ones came in with their provenance manifests.
# Paths inside the phase image, not on any host. A job that carried host paths
# would only run on the host that wrote it.
DEFAULT_VC_FLATTEN = "/opt/campaignx/vc3d/bin/vc_flatten"
DEFAULT_FLATTEN_PROFILE = ("/workspace/campaign-x/framework/profiles/02-flattening"
                           "/flatten-abf-v1-1.0.0.json")

# P5 has two adapters and they do not take the same arguments. The profile has
# always named its own -- `adapter` in the lane profile -- and the queue ignored
# it, building one argv for every ink job: --profile <id> --upstream-dir <dir>,
# which is run_ink.py's CLI. The TimeSformer adapter takes --ink-profile
# <path> and has no --upstream-dir at all, so every TimeSformer lane queued
# through the API ran the wrong script. Nobody noticed because the one P5 job
# that ever ran used the canonical lane, whose adapter is the one that was
# hardcoded.
# How many pending rows a claim looks at before giving up. More than one because
# a worker has to be able to skip past another runtime's work to reach its own;
# small because every row it looks at is locked for the length of the claim.
CLAIM_CANDIDATES = 8


# Where a vendored architecture lives is a fact about the host that carries it,
# not about the job. It was a P5 parameter typed into a form, which made the
# import path of the model code something a caller chose: run_ink.py does
# sys.path.insert(0, upstream_dir) and then imports the architecture from it,
# so whoever set the field decided what the GPU host executed.
#
# An adapter that needs one says so here and the worker reads the directory
# from its own environment. A lane needing a different upstream is a lane with
# its own image, which is how the 9 um lane already works.
UPSTREAM_ROOT_VARIABLE = "HELENA_INK_UPSTREAM_ROOT"


INK_ADAPTERS: dict[str, dict[str, Any]] = {
    "framework/stages/03-ink/scripts/run_ink.py": {
        "receipt": "INK_PROFILE_RECEIPT.json",
        "profile_flag": "--profile",
        "profile_as": "id",
        "needs": ("tiff_dir", "checkpoint", "source_pixel_um"),
        # The directory it inserts at sys.path[0], from the worker rather than
        # from the request.
        "upstream": {"variable": UPSTREAM_ROOT_VARIABLE,
                     "flag": "--upstream-dir"},
        "flags": {"source_pixel_um": "--source-pixel-um",
                  "depth_center": "--depth-center", "stride": "--stride",
                  "batch_size": "--batch-size",
                  "min_valid_ratio": "--min-valid-ratio",
                  "device": "--device", "on_degenerate": "--on-degenerate"},
    },
    "framework/stages/03-ink/scripts/run_ink_timesformer.py": {
        "receipt": "INK_SCREENING_RECEIPT.json",
        "profile_flag": "--ink-profile",
        "profile_as": "path",
        # No upstream_dir: this adapter carries its own architecture and takes
        # the training scale from the profile it is handed.
        "needs": ("tiff_dir", "checkpoint", "source_pixel_um"),
        "flags": {"source_pixel_um": "--source-pixel-um",
                  "source_slice_um": "--source-slice-um",
                  # Without this the queue can only ever build the GP lane's
                  # frozen architecture, and a profile pinning any other
                  # checkpoint fails at load with a size mismatch after the
                  # stack has already been fetched.
                  "model_config": "--model-config",
                  "depth_centers": "--depth-centers",
                  "tiling_offsets": "--tiling-offsets",
                  "stride": "--stride", "batch_size": "--batch-size",
                  "min_valid_ratio": "--min-valid-ratio",
                  "tile_size": "--tile-size", "device": "--device",
                  "model_family": "--model-family",
                  "on_degenerate": "--on-degenerate"},
    },
    # The 2 um canonical lane. Takes no profile at all: everything it needs is
    # a flag, and the profile is provenance rather than configuration.
    "framework/stages/03-ink/scripts/run_ink_canonical2um.py": {
        "receipt": "INK_CANONICAL_RECEIPT.json",
        "profile_flag": None,
        "profile_as": None,
        # It imports the upstream architecture from beside itself, which is
        # where that code sits in the recipe's own directory and not where it
        # sits here. The directory reaches it as PYTHONPATH rather than as a
        # flag: the runner is vendored byte for byte and stays that way, because
        # the whole point of this lane is running their model and not ours.
        "upstream": {"variable": UPSTREAM_ROOT_VARIABLE, "pythonpath": True},
        "needs": ("tiff_dir", "checkpoint", "source_pixel_um"),
        "flags": {"source_pixel_um": "--source-pixel-um",
                  "frames": "--frames", "tile_size": "--tile-size",
                  "stride": "--stride", "batch_size": "--batch-size",
                  "depth_center": "--depth-center",
                  "min_valid_ratio": "--min-valid-ratio", "device": "--device",
                  "on_degenerate": "--on-degenerate"},
    },
    # DINO-guided, which is a different shape of job: it reads a manifest of
    # patches rather than a layer stack directory, and it writes where it is
    # told rather than into an output directory. Declared so the queue refuses
    # it with that reason instead of routing it to a runner that would fail on
    # the first flag.
    "framework/stages/03-ink/scripts/run_ink_3d_dino.py": {
        "profile_flag": "--profile",
        "profile_as": "path",
        "needs": ("checkpoint", "config", "villa_python_root", "input_manifest"),
        "unroutable": ("this lane takes a patch manifest and a villa python root "
                       "rather than a layer stack; it is run from the CLI until "
                       "somebody gives it a job shape"),
        "flags": {},
    },
    # The 9 um lane. Its runner reads an OME-Zarr surface volume at ~9 um
    # isotropic and P4 writes a numbered TIFF stack at 2.399 um, so the adapter
    # pools the stack into that volume before calling upstream -- in the same
    # process, which is why this is one queued job and not two. `source_pixel_um`
    # is required rather than defaulted: pooling has to know what it is pooling
    # from, and assuming 2.4 would turn a native 9 um render into a 38 um one.
    "framework/stages/03-ink/scripts/run_ink_9um.py": {
        "receipt": "INK_9UM_RECEIPT.json",
        # Not the worker's runtime. koine_machines is not in the ink worker,
        # and cannot be: that image pins torch 2.5.1+cu124 and this lane's lock
        # pins 2.10.0, five major versions apart and --frozen on purpose.
        "image": "helena-ink-9um",
        "profile_flag": "--profile",
        "profile_as": "path",
        # Not tiff_dir or source_pixel_um: which input was named is settled by
        # EXACTLY_ONE_OF, and the coupling between them -- pooling has to know
        # what it is pooling from -- is the adapter's own check, which says it
        # far better than a missing-key list can.
        "needs": ("checkpoint",),
        "flags": {"source_pixel_um": "--source-pixel-um",
                  "surface_volume": "--surface-volume",
                  "on_degenerate": "--on-degenerate"},
    },
}
INK_PROFILE_DIR = "framework/profiles/03-ink"


def ink_profile_path(profile_id: str) -> Path | None:
    """The profile file a lane id names, from the frozen profile directory."""
    root = Path(__file__).resolve().parents[4] / INK_PROFILE_DIR
    for candidate in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            if json.loads(candidate.read_text()).get("profile_id") == profile_id:
                return candidate
        except (OSError, ValueError):
            continue
    return None


def ink_adapter(profile_id: str | None) -> tuple[str, dict[str, Any], Path | None]:
    """Which script runs this lane, and what it takes.

    The profile decides. A lane whose profile cannot be found falls back to the
    original runner rather than refusing: a job queued for a profile this
    checkout does not carry is a deployment problem, and it should fail where it
    runs with the real reason rather than at enqueue with a guess.
    """
    default = PHASE_RUNNERS["P5"]
    path = ink_profile_path(profile_id) if profile_id else None
    adapter = default
    if path is not None:
        try:
            adapter = json.loads(path.read_text()).get("adapter") or default
        except (OSError, ValueError):
            adapter = default
    if path is not None:
        declared = json.loads(path.read_text())
        if not declared.get("adapter") and not declared.get("checkpoint_sha256"):
            # A profile that names neither a script nor a checkpoint is a
            # declaration about the campaign -- a calibration note, a threshold
            # statement -- and not a lane. Routing one would run the default
            # adapter with no model behind it.
            raise JobRejected(
                f"{profile_id} declares no adapter and no checkpoint: it is a "
                "calibration declaration, not a runnable lane")
    spec = INK_ADAPTERS.get(adapter)
    if spec is None:
        # Falling back to the default runner here is how a TimeSformer lane came
        # to be handed --upstream-dir. A lane whose adapter nobody taught the
        # queue is a lane the queue cannot run, and it should say so.
        raise JobRejected(
            f"{profile_id} names the adapter {adapter}, which the queue has no "
            f"command for. Adapters it knows: {sorted(INK_ADAPTERS)}")
    if spec.get("unroutable"):
        raise JobRejected(f"{profile_id} cannot be queued: {spec['unroutable']}")
    return adapter, spec, path


def depth_window(center: float, frames: int, *, source_slice_um: float,
                 training_slice_um: float) -> tuple[float, float]:
    """The first and last source slice a detector window touches.

    The same arithmetic the TimeSformer adapter does: the model wants `frames`
    planes at its training pitch, so on a stack sampled at a different pitch the
    window spans frames * training/source slices around the centre.
    """
    half = (frames - 1) / 2.0 * training_slice_um / source_slice_um
    return center - half, center + half


def depth_centers_that_fit(slices: int, frames: int, centers: Sequence[float], *,
                           source_slice_um: float,
                           training_slice_um: float) -> list[float]:
    """Which of these centres a stack this deep can actually be sampled at."""
    fitting = []
    for center in centers:
        low, high = depth_window(float(center), frames,
                                 source_slice_um=source_slice_um,
                                 training_slice_um=training_slice_um)
        if low >= 0 and high <= slices - 1:
            fitting.append(float(center))
    return fitting


def ink_lane_inventory() -> list[dict[str, Any]]:
    """Every ink method this checkout knows, and whether P5 can run it.

    Three populations that were never listed together: the lane profiles, the
    adapters that can execute one, and the method registry's own record of what
    each checkpoint is worth. A method with no profile cannot be queued no
    matter how good it is, and nothing said which ones those were.
    """
    root = Path(__file__).resolve().parents[4]
    registry_path = root / "framework/registries/method-capabilities-0.1.0.json"
    try:
        registry = {entry.get("method_id"): entry
                    for entry in json.loads(registry_path.read_text()).get("entries", [])}
    except (OSError, ValueError):
        registry = {}

    lanes: list[dict[str, Any]] = []
    seen_methods: set[str] = set()
    profiles = sorted((root / INK_PROFILE_DIR).glob("*.json")) \
        if (root / INK_PROFILE_DIR).is_dir() else []
    for path in profiles:
        try:
            profile = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        profile_id = str(profile.get("profile_id") or path.stem)
        method_id = profile.get("method_id")
        seen_methods.add(str(method_id))
        entry = registry.get(method_id) or {}
        try:
            adapter, _, _ = ink_adapter(profile_id)
            routable, reason = True, None
        except JobRejected as refused:
            adapter, routable, reason = profile.get("adapter"), False, str(refused)
        lanes.append({
            "profile_id": profile_id, "method_id": method_id,
            "adapter": adapter, "routable": routable, "reason": reason,
            "checkpoint_sha256": profile.get("checkpoint_sha256"),
            "validation_status": entry.get("validation_status"),
            "training_pixel_um": (profile.get("input_contract") or {}).get(
                "training_pixel_um"),
        })
    for method_id, entry in registry.items():
        # A checkpoint the registry knows and no profile routes. Ink methods
        # only: the registry also carries segmentation and routing methods.
        if method_id in seen_methods or not entry.get("known_checkpoint_sha256"):
            continue
        lanes.append({
            "profile_id": None, "method_id": method_id, "adapter": None,
            "routable": False,
            "reason": "the registry knows this checkpoint and no lane profile "
                      "names it, so P5 has nothing to run it with",
            "checkpoint_sha256": entry.get("known_checkpoint_sha256"),
            "validation_status": entry.get("validation_status"),
            "training_pixel_um": None,
        })
    return sorted(lanes, key=lambda lane: (not lane["routable"],
                                           str(lane["profile_id"] or lane["method_id"])))


# ---------------------------------------------------------------------------
# Lanes: how a phase is made modular
#
# A lane is one way of doing a phase: which program runs, what it must be given,
# and how its parameters become flags. P4 and P5 have had this for a while --
# two renderers, several detectors -- and adding one there is a row in a table.
# Every other phase had a single runner and a hand-written branch in
# `command_for`, so a second implementation meant editing the queue.
#
# `PHASE_LANES` gives every phase the same shape. A lane declares:
#
#     runner    the program, relative to the repository root
#     required  parameters without which the job is refused
#     flags     parameter -> command-line flag, for the flat majority
#     fixed     leading arguments, e.g. a subcommand
#     build     an escape hatch for argv that is not a flat mapping
#
# Adding a module that does a phase differently is one entry here plus its
# parameters in PHASE_PARAMETERS. Nothing in the worker, the panel or this
# builder needs to know it exists: the form, the guide and the routing all read
# this table.
PHASE_LANES: dict[str, dict[str, dict[str, Any]]] = {}


def lane_image(spec: Mapping[str, Any] | None) -> str | None:
    """The runtime a lane needs, or None when the worker's own will do.

    Absent means "runs where it is claimed", which is true of every lane that
    existed before three of them did not. Making it explicit everywhere would
    invent a fact about lanes nobody had to think about.
    """
    return (spec or {}).get("image") or None


def lane_runtime_images() -> set[str]:
    """Every image a lane asks for that is not the ordinary worker's own.

    A worker running one of these is a specialist: it was built to carry that
    lane's frozen environment and nothing else. The set is read from the same
    declarations routing uses, so a new lane with its own image is a specialist
    the moment it is registered.
    """
    images = {lane_image(spec) for spec in INK_ADAPTERS.values()}
    for lanes in PHASE_LANES.values():
        images |= {lane_image(spec) for spec in lanes.values()}
    return {image for image in images if image}


def runtime_image_for(job: Mapping[str, Any]) -> str | None:
    """Which image this job's lane needs, if it is not the worker's own.

    Answers "which image", not "is this job valid". An unresolvable lane is
    refused by the queue with its own reason, and raising a second one here
    would report the wrong failure first.
    """
    try:
        if str(job.get("phase") or "P5") == "P5":
            return lane_image(ink_adapter(job.get("profile_id"))[1])
        return lane_image(lane_for(job)[1])
    except Exception:  # noqa: BLE001 -- see the docstring
        return None


def register_lane(phase: str, lane_id: str, spec: dict[str, Any]) -> None:
    """Add a way of doing a phase. Refuses to shadow one silently."""
    lanes = PHASE_LANES.setdefault(phase, {})
    if lane_id in lanes:
        raise ValueError(f"{phase} already has a lane called {lane_id!r}")
    lanes[lane_id] = spec


def lane_for(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Which lane this job runs on, and its spec.

    A job names one in `parameters["lane"]`; without that it gets the phase's
    default, which is the first registered. An unknown name is refused rather
    than silently defaulted -- a job that asked for a detector and got another
    one is the kind of result nobody can interpret afterwards.
    """
    phase = str(job.get("phase") or "P5")
    lanes = PHASE_LANES.get(phase) or {}
    if not lanes:
        raise JobRejected(f"phase {phase} has no lane registered")
    asked = (job.get("parameters") or {}).get("lane")
    if asked:
        if asked not in lanes:
            raise JobRejected(
                f"unknown {phase} lane {asked!r}; registered: {sorted(lanes)}")
        return str(asked), lanes[asked]
    default = next(iter(lanes))
    return default, lanes[default]


def validate_lane_profile(job: dict[str, Any], spec: dict[str, Any]) -> None:
    """A frozen lane may accept only explicitly versioned profiles.

    P5 already resolves profiles through its adapter registry.  This is the
    equivalent for non-P5 lanes: a caller cannot give the merge wrapper an
    arbitrary profile path, and omitting the profile cannot fall back to
    whichever file happens to be in an image.
    """

    accepted = tuple(spec.get("profiles") or ())
    if not accepted:
        return
    profile_id = job.get("profile_id")
    if profile_id not in accepted:
        raise JobRejected(
            f"{job.get('phase')} lane accepts profiles {list(accepted)}; "
            f"got {profile_id!r}")


def validate_lane_sample(job: dict[str, Any], lane_id: str,
                         spec: dict[str, Any]) -> None:
    """Refuse a lane whose embedded data belongs to another scroll."""
    accepted = tuple(spec.get("sample_ids") or ())
    if not accepted:
        return
    sample_id = job.get("sample_id")
    if sample_id not in accepted:
        raise JobRejected(
            f"{lane_id} is pinned to PHerc0332, but this job records "
            f"sample_id={sample_id!r}; use vc-render-tifxyz for other scrolls")


def declarative_argv(runner: str, spec: dict[str, Any],
                     parameters: dict[str, Any], output_dir: str) -> list[str]:
    """argv for a lane that is a flat mapping of parameters to flags.

    Which is most of them. The two that are not -- a subcommand, or a flag whose
    value is a file inside the job's directory -- declare `build` instead.
    """
    argv = ["python3", runner, *spec.get("fixed", ())]
    if spec.get("output_flag"):
        argv += [str(spec["output_flag"]), output_dir]
    for name in spec.get("required", ()):
        if name not in parameters:
            raise JobRejected(f"{name} is required on this lane")
    for name, flag in spec.get("flags", {}).items():
        value = parameters.get(name)
        if value is None and name in spec.get("defaults", {}):
            value = spec["defaults"][name](output_dir)
        if value is None or value is False:
            continue
        if name in set(spec.get("json_flags", ())):
            argv += [flag, json.dumps(value, sort_keys=True, separators=(",", ":"))]
        elif value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    return argv


PHASE_RUNNERS: dict[str, str] = {
    # The spiral fitter, which is Python at the villa source-lock commit and
    # runs in the image that carries it. Not the seeded grower: that one is
    # planned by the fleet's own bootstrap and has never been a queued job.
    "P1": "framework/stages/01-segmentation/scripts/run_spiral_fit.py",
    # P2 and P3 are subcommands of the fleet CLI rather than scripts of their
    # own. They are queued like everything else so the panel never has to run a
    # container: it writes a row, and a worker that already carries the runtime
    # -- vc_flatten, scipy, tifffile, boto3 -- claims it. A panel that ran these
    # itself would need the Docker socket, which is host control handed to a web
    # process to save a queue.
    "P2": "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py",
    "P3": "framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py",
    "P4": "framework/vendored/scroll-streaming-tools/render_scroll.py",
    "P5": "framework/stages/03-ink/scripts/run_ink.py",
    "P7": "framework/vendored/vetting-card/vet_map.py",
    "P8": "framework/vendored/pherc0139-column-atlas-gh/scripts/wrap_order.py",
    "P9": "framework/vendored/pherc0139-column-atlas-gh/scripts/make_plates.py",
}


# P4 already had a lane table of its own and P5 has one keyed by profile. They
# stay the source for their phases; this registers what is there rather than
# restating it, so there is still one place to add a renderer or a detector.
for _lane_id, _lane in P4_LANES.items():
    register_lane("P4", _lane_id, {
        "name": _lane["name"], "note": _lane.get("note"),
        "runner": _lane.get("script", PHASE_RUNNERS["P4"]),
        "sample_ids": _lane.get("sample_ids", ()),
        # Neither renderer touches a card. vc_render_tifxyz is OpenMP over
        # streamed Zarr chunks -- upstream volume-cartographer has no CUDA path
        # for rendering at all, its one CUDA option being lasagna's maxflow --
        # and the chunk-gather lane is a fetch-and-write loop. They inherited a
        # GPU requirement from this table's own default, which is a statement
        # about which workers may claim the job and was never a statement about
        # this work.
        "gpu_required": False,
        "build": "legacy",
    })

for _phase, _runner in PHASE_RUNNERS.items():
    # P4 has its own table and P1 its own declaration below: both need more
    # than a name and a runner -- an image, a GPU, the profiles it accepts.
    if _phase in ("P1", "P4"):
        continue
    register_lane(_phase, {
        "P2": "certify-batch", "P3": "flatten-batch",
        "P5": "ink-adapter", "P7": "vetting-card", "P8": "column-atlas",
        "P9": "official-plates",
    }[_phase], {
        "name": {
            "P2": "fleet certifier", "P3": "fleet flattener",
            "P5": "the profile's own adapter", "P7": "vetting card",
            "P8": "column atlas wrap order", "P9": "official plate composer",
        }[_phase],
        "runner": _runner,
        # Certification, flattening and the column-atlas relation measurement
        # run in the CPU-only fleet image. If they inherit the historical GPU
        # default, the only worker that advertises those phases rejects them
        # before it can claim them.
        #
        # The vetting card and the plate composer are here for a different
        # reason: they do run beside the detector, on a host that has a card,
        # and neither imports torch. One is statistics over a probability map
        # that already exists and the other is an image pipeline. Their GPU
        # requirement was inherited, not chosen, and it costs nothing today
        # only because the worker that claims them needs a card for P5 anyway.
        # It would cost the first time somebody wanted to screen or compose on
        # a machine without one.
        **({"gpu_required": False}
           if _phase in ("P2", "P3", "P7", "P8", "P9") else {}),
        **({"required": ("scroll", "out_path")} if _phase == "P8" else {}),
        # The bespoke builder below. Kept as the lane's own rule rather than
        # rewritten: these five predate the table and each encodes something a
        # flat flag mapping cannot say.
        "build": "legacy",
    })

# Segmentation's second backend. Upstream calls the spiral fitter their most
# powerful surface-recovery method and now recommends it; it is registered here
# rather than bolted onto the grower's bootstrap because the two decide
# different things. A grow is planned -- cells, a seed, a candidate policy. A
# fit is told which scroll and which slab of it, and nothing else.
#
# `image` is what makes it runnable at all: fit_spiral.py is Python with torch
# and zarr behind it, and the segment worker that claims P1 carries compiled
# binaries. A worker in the wrong runtime refuses by name before the lease,
# which is what the declaration is for.
register_lane("P1", "spiral-fit", {
    "name": "Spiral fitter, global winding fit",
    "runner": PHASE_RUNNERS["P1"],
    "image": "helena-villa-python",
    "gpu_required": True,
    "profiles": ("spiral-fitter-v1@0.3.0",),
    "required": ("scroll_name", "dataset_path", "z_begin", "z_end",
                 "voxel_size_um", "artifact_store"),
    # Named, so the worker collects it. A lane that writes a receipt and does
    # not declare its name has that receipt silently dropped.
    "receipt": "SPIRAL_FIT_RECEIPT.json",
    "build": "legacy",
    "note": ("Fits one global spiral through tracks, point collections and a "
             "lasagna normal field, and writes one TIFXYZ per winding. Those "
             "enter P2's certification gate exactly as a grown surface does. "
             "The scroll is selected by rewriting the six module constants "
             "upstream assigns at import; the receipt carries the digest of "
             "the script before and after."),
})

# P8's alternative, and the proof the table is the route rather than decoration:
# a different assembler for the same phase is a row, not an edit to the builder.
register_lane("P8", "mesh-relations", {
    "name": "relation-driven assembly",
    "runner": "framework/stages/05-reconstruction/scripts/evaluate_r6_direct_geometry.py",
    "required": ("scroll", "out_path"),
    "flags": {"scroll": "--scroll", "out_path": "--out", "work_dir": "--work",
              "subsample": "--subsample"},
    "defaults": {"work_dir": lambda out: str(Path(out) / "relations")},
    "gpu_required": False,
    "note": ("Assembles from measured relations between segments rather than "
             "from the published column atlas. It answers the same question -- "
             "who neighbours whom -- from geometry this platform produced, "
             "which is the arrangement somebody else can check."),
})

register_lane("P8", "vc3d-tifxyz-merge", {
    "name": "Volume Cartographer TIFXYZ merge",
    "runner": "framework/stages/05-reconstruction/scripts/run_vc3d_tifxyz_merge.py",
    "required": (
        "artifact_ids", "rows", "reference_artifact_id", "ransac_seed",
        "anchor_cap", "strip_cols", "artifact_store",
    ),
    "profiles": ("vc3d-tifxyz-merge@1.0.0",),
    "fixed": (
        "--db", "postgres-env://CX_DB",
        "--profile",
        "/workspace/campaign-x/framework/profiles/05-reconstruction/"
        "vc3d-tifxyz-merge-1.0.0.json",
    ),
    "flags": {
        "artifact_ids": "--artifact-ids-json",
        "rows": "--rows-json",
        "reference_artifact_id": "--reference-artifact-id",
        "ransac_seed": "--ransac-seed",
        "anchor_cap": "--anchor-cap",
        "strip_cols": "--strip-cols",
        "artifact_store": "--artifact-store",
    },
    "json_flags": ("artifact_ids", "rows"),
    "output_flag": "--output",
    "receipt": "MERGE_RECEIPT.json",
    "gpu_required": False,
    "note": (
        "Merges two or more certified, content-addressed TIFXYZ surfaces with "
        "the frozen upstream vc_merge_tifxyz binary. Inputs are resolved from "
        "the control plane; no caller-provided path or command is accepted."
    ),
})


def command_for(job: dict[str, Any], *, runner: str, output_dir: str,
                upstream_root: str | None = None) -> list[str]:
    """Build the command line for this job's phase from its validated parameters.

    The queue never carries a command. Each branch below is the only way a job
    of that phase can turn into a process, so widening what a caller can run is
    an edit here rather than a consequence of a request.

    `upstream_root` is the worker's own vendored-architecture directory, passed
    in rather than read from the environment here so this stays a function of
    its arguments and a test does not need a variable set to say what a job
    turns into.
    """
    phase = job.get("phase", "P5")
    parameters = job["parameters"]
    if phase in ("P2", "P3"):
        outer_sample = job.get("sample_id")
        parameter_sample = parameters.get("sample")
        if (not isinstance(outer_sample, str) or not outer_sample
                or parameter_sample != outer_sample):
            raise JobRejected(
                f"{phase} parameters.sample must equal immutable job sample_id")

    # A lane that declares its own flags is built from the table. Only the
    # phases whose argv is not a flat mapping fall through to the branches.
    lane_id, spec = lane_for(job)
    validate_lane_profile(job, spec)
    validate_lane_sample(job, lane_id, spec)
    if spec.get("build") != "legacy":
        return declarative_argv(spec.get("runner", runner), spec, parameters,
                                output_dir)

    if phase == "P1":
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise JobRejected("P1 command requires its immutable queue job_id")
        sample_id = job.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise JobRejected(
                "a fitted winding is filed under a scroll, so P1 requires the "
                "job's immutable sample_id")
        profile_id = job.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise JobRejected(
                "P1 runs against a frozen profile: the fitter's 105 upstream "
                "settings and this platform's overrides come from it, and a run "
                "that named none would record a configuration nobody chose")
        # The scroll is not a parameter and the mission is not a knob: both come
        # off the job row, for the reason P2 and P3 read theirs there.
        argv = ["python3", runner,
                "--profile-id", profile_id,
                "--out", output_dir,
                "--sample", sample_id,
                "--requested-by-job-id", job_id,
                "--scroll-name", str(parameters["scroll_name"]),
                "--dataset-path", str(parameters["dataset_path"]),
                "--z-begin", str(int(parameters["z_begin"])),
                "--z-end", str(int(parameters["z_end"])),
                "--voxel-um", str(float(parameters["voxel_size_um"])),
                "--artifact-store", str(parameters["artifact_store"]),
                # postgres-env:// so the connection string is read from the
                # worker's own environment rather than travelling in a job row.
                "--db", "postgres-env://CX_DB"]
        if parameters.get("spiral_outward_sense"):
            argv += ["--winding-sense", str(parameters["spiral_outward_sense"])]
        for name, flag in (("lasagna_volume_name", "--lasagna-volume-name"),
                           ("normal_zarr_group", "--normal-zarr-group"),
                           ("lasagna_scale", "--lasagna-scale"),
                           ("tracks_file", "--tracks-file"),
                           ("random_seed", "--random-seed")):
            # `is not None`, not truthiness. Seed 0 and scale 0 are values a
            # caller can mean, and dropping them sent the run with upstream's
            # default while the receipt recorded the zero -- the one shape of
            # disagreement nothing downstream can detect, because both halves
            # look like a job that ran as asked.
            if parameters.get(name) is not None and parameters.get(name) != "":
                argv += [flag, str(parameters[name])]
        if parameters.get("run_tag"):
            argv += ["--run-tag", str(parameters["run_tag"])]
        if job.get("mission_id"):
            argv += ["--mission-id", str(job["mission_id"])]
        if parameters.get("dry_run"):
            argv += ["--dry-run"]
        return argv

    if phase in ("P2", "P3"):
        requested_by_job_id = job.get("job_id")
        if not isinstance(requested_by_job_id, str) or not requested_by_job_id:
            raise JobRejected(f"{phase} command requires its immutable queue job_id")
        # postgres-env:// so the connection string is read from the worker's own
        # environment rather than travelling in a job row and an argv.
        argv = ["python3", runner,
                "certify" if phase == "P2" else "flatten",
                "--db", "postgres-env://CX_DB",
                "--requested-by-job-id", requested_by_job_id]
        if parameters.get("limit"):
            argv += ["--limit", str(int(parameters["limit"]))]
        if parameters.get("sample"):
            argv += ["--sample", str(parameters["sample"])]
        # The mission this job belongs to, so the backlog it works is that
        # mission's own surfaces. Without it a certify or flatten queued from
        # one mission takes surfaces from another -- or from the pre-mission
        # `unfiled` history -- and publishes a sheet nobody can find, because
        # every view in the panel is mission-scoped. Read from the job row
        # rather than the parameters: the mission is not a knob a request may
        # set, it is which project the work belongs to.
        if job.get("mission_id"):
            argv += ["--mission-id", str(job["mission_id"])]
        if phase == "P2" and parameters.get("surface_id"):
            argv += ["--surface-id", str(parameters["surface_id"])]
        if phase == "P3" and parameters.get("surface_id"):
            argv += ["--surface-id", str(parameters["surface_id"])]
        if parameters.get("dry_run"):
            argv += ["--dry-run"]
        if phase == "P3":
            if parameters.get("allow_unvalidated"):
                argv += ["--allow-unvalidated"]
            argv += [
                "--binary", parameters.get("binary", DEFAULT_VC_FLATTEN),
                "--profile", parameters.get("profile", DEFAULT_FLATTEN_PROFILE),
                "--artifact-store", parameters["artifact_store"],
            ]
        return argv

    if phase == "P4":
        lane_id = parameters.get("lane") or DEFAULT_P4_LANE
        lane = P4_LANES.get(lane_id)
        if lane is None:
            # Reachable when a job predates a lane being renamed, so it refuses
            # rather than raising KeyError out of a worker's command builder.
            raise JobRejected(f"unknown P4 lane {lane_id!r}; lanes are {sorted(P4_LANES)}")
        if lane_id == "chunk-gather":
            argv = ["python3", runner,
                    "--ppm", parameters["ppm"],
                    "--volume", parameters["volume_key"],
                    "--out", parameters.get("out_dir", output_dir)]
            for key, flag in (("layers", "--layers"), ("spacing", "--spacing"),
                              ("concurrency", "--conc"), ("stripe", "--stripe"),
                              ("max_gb", "--max-gb")):
                if key in parameters:
                    argv += [flag, str(parameters[key])]
            return argv

        # The default lane is a binary from the VC3D runtime, resolved through
        # the same environment variable the container already sets.
        #
        # It does take an output flag, contrary to what this comment said for as
        # long as nothing ran it: without --tif-output or --zarr-output it
        # refuses with "at least one of --zarr-output or --tif-output required"
        # and renders nothing. Defaulted rather than required, so a job that
        # names neither still lands its layers somewhere findable.
        executable = os.environ.get(lane["executable_env"]) or lane["fallback"]
        # `segmentation` is either what the caller passed or what the worker
        # resolved from `flattened_surface` before this ran. The renderer takes
        # a directory either way; which directory it was is on the job.
        if not parameters.get("segmentation"):
            raise JobRejected(
                "this job names a flattened surface and nothing resolved it to a "
                "directory; the worker fetches the sheet before building the "
                "command, so reaching here means that step did not run")
        argv = [executable,
                "--volume", parameters["volume"],
                "--segmentation", parameters["segmentation"],
                "--scale", str(parameters["scale"]),
                "--group-idx", str(int(parameters["group_idx"]))]
        if parameters.get("remote_url"):
            argv += ["--remote-url", parameters["remote_url"], "--prefetch-remote"]
        if parameters.get("cache_gb"):
            argv += ["--cache-gb", str(parameters["cache_gb"])]
        if parameters.get("zarr_output"):
            argv += ["--zarr-output", parameters["zarr_output"]]
        # The worker verifies and publishes TIFFs, and P5 consumes them.  Zarr
        # is an additional export; a Zarr-only command used to exit zero and
        # then fail P4's output contract because no publishable stack existed.
        argv += ["--tif-output",
                 parameters.get("tif_output") or str(Path(output_dir) / "layers")]
        # The physical scale, which the renderer otherwise reports as
        # "Voxel size: 1.0 (no metadata found; override with --voxel-size)" and
        # carries on. `source_voxel_um` is already a P4 parameter, already
        # filled by the deployment, and already used on this side to derive
        # depth spacing and the P3/P4 lateral metric -- it just never reached
        # the renderer, so the two halves of one job used different numbers
        # whenever the volume carried no metadata. The official rescreen path
        # has passed it all along.
        if parameters.get("source_voxel_um"):
            argv += ["--voxel-size", str(float(parameters["source_voxel_um"])),
                     "--voxel-unit", "micrometer"]
        # Depth. Left off the command when unset so the renderer's own defaults
        # apply and the receipt shows they were not chosen here.
        if parameters.get("flip_normals"):
            argv += ["--flip-normals"]
        for key, flag in (("num_slices", "--num-slices"),
                          ("slice_step", "--slice-step")):
            if parameters.get(key) is not None:
                argv += [flag, str(parameters[key])]
        return argv

    if phase == "P7":
        if parameters.get("screening_of") and not parameters.get("map_path"):
            # The worker fetches the named screening's map and fills this in.
            # Without that step the screen would run on nothing, or worse on
            # whatever a stale path holds.
            raise JobRejected(
                f"the map of {parameters['screening_of']} was named and the "
                "fetch did not run")
        # --out is "path to write the verdict JSON", a file and not a
        # directory: given one, vet_map wrote the verdict *as* the directory and
        # the card that followed died on "Not a directory". The caller names a
        # directory because that is what a job's output is; the two file names
        # are this builder's business.
        into = Path(parameters.get("out_dir", output_dir))
        return ["python3", runner, "--map", parameters["map_path"],
                "--bbox", parameters["bbox"], "--px-um", str(parameters["px_um"]),
                "--out", str(into / "verdict.json"),
                "--card", str(into / "VETTING_CARD.md")]

    if phase == "P8":
        # Every P8 job died here at argparse: --segments does not exist. The
        # runner takes the scroll and downloads the public meshes for it.
        argv = ["python3", runner,
                "--scroll", parameters["scroll"],
                "--out", parameters["out_path"],
                "--work", parameters.get("work_dir", str(Path(output_dir) / "meshes"))]
        if "subsample" in parameters:
            argv += ["--subsample", str(int(parameters["subsample"]))]
        return argv

    if phase == "P9":
        if parameters.get("ordering_of") and not parameters.get("order_path"):
            raise JobRejected(
                f"the radial order of {parameters['ordering_of']} was named and "
                "the fetch did not run")
        # It fetches the official ink maps for the scroll itself; the plate
        # sequence comes from the measured P8 radial table, never a copied list.
        return ["python3", runner,
                "--scroll", parameters["scroll"],
                "--out", parameters["out_dir"],
                "--work", parameters.get("work_dir",
                                         str(Path(output_dir) / "official-maps")),
                "--order", parameters["order_path"]]

    if phase != "P5":
        raise JobRejected(f"no command builder for phase {phase}")

    if parameters.get("layer_stack") and not parameters.get("tiff_dir"):
        # The worker fetches the published stack and fills tiff_dir in. If that
        # step is ever skipped, the detector must not be pointed at nothing --
        # or worse, at whatever a stale directory holds.
        raise JobRejected(
            f"the layer stack of {parameters['layer_stack']} was named and the "
            "fetch did not run")
    adapter, spec, profile_path = ink_adapter(job.get("profile_id"))
    if spec["profile_as"] == "path" and profile_path is None:
        raise JobRejected(
            f"{adapter} takes the lane profile as a file and "
            f"{job.get('profile_id')!r} is not in {INK_PROFILE_DIR}")
    argv = ["python3", runner]
    if spec["profile_flag"]:
        # A lane whose runner takes no profile is not a lane without one: the
        # profile still pins the checkpoint and the frames, it just does so by
        # being the thing that chose this runner.
        argv += [spec["profile_flag"],
                 str(profile_path) if spec["profile_as"] == "path"
                 else job["profile_id"]]
    argv += [
        "--sample-id", job["sample_id"],
        "--output", output_dir,
        "--checkpoint", parameters["checkpoint"],
    ]
    # Only the one that was named. This was unconditional, which is why a ready
    # surface volume could not be queued even once the contract allowed it: the
    # adapter refuses a run naming both an input to pool and one already pooled,
    # and rightly so -- a lane handed two has no way to say which its map came
    # from.
    if parameters.get("tiff_dir"):
        argv += ["--tiff-dir", parameters["tiff_dir"]]
        # Pooling has to know what it is pooling from: guessing 2.4 would turn a
        # native 9 um render into a 38 um one. The queue refuses this at enqueue
        # too, where it costs nothing; here is the backstop, because this is
        # what the worker builds and a job that reached it without the scale
        # would spend a lease discovering the same thing.
        if ("source_pixel_um" in spec["flags"]
                and parameters.get("source_pixel_um") in (None, "")):
            raise JobRejected(
                f"{adapter} needs ['source_pixel_um'] with a tiff_dir: the "
                "scale the stack was rendered at, which pooling reads")
    if parameters.get("surface_volume") and "surface_volume" not in spec["flags"]:
        # Dropped silently, this would point the detector at nothing -- or at
        # whatever a stale directory holds.
        raise JobRejected(
            f"{adapter} cannot read a surface_volume: it takes a layer stack "
            "and pools it. Queue this against a lane that declares "
            "--surface-volume, or name a tiff_dir.")
    missing = [key for key in spec["needs"] if parameters.get(key) in (None, "")]
    if missing:
        raise JobRejected(f"{adapter} needs {missing}")
    for key, flag in spec["flags"].items():
        if key in parameters:
            argv += [flag, str(parameters[key])]
    upstream = spec.get("upstream") or {}
    if upstream.get("flag"):
        if not upstream_root:
            raise JobRejected(
                f"{adapter} imports its architecture from a vendored upstream "
                f"directory and this worker does not say where one is: set "
                f"{upstream['variable']} on the host that runs this lane. It is "
                "not a job parameter -- the import path of the model code is "
                "not something a request gets to choose.")
        argv += [upstream["flag"], str(upstream_root)]
    # The model family the checkpoint belongs to comes from the profile, not
    # from whoever queued the job. The adapter checks it against the registry by
    # hash and refuses a known hash under the wrong family, and its own default
    # names a different model -- so an operator who omitted it got a rejection
    # about provenance for what was really a missing flag.
    if "model_family" in spec["flags"] and "model_family" not in parameters:
        family = ((json.loads(profile_path.read_text()).get("input_contract") or {})
                  .get("model_family") if profile_path else None)
        if family:
            argv += ["--model-family", str(family)]
    return argv


class InkJobStore:
    """Thin PostgreSQL store. Every mutation is one transaction."""

    def __init__(self, dsn: str):
        self.dsn = dsn

    # How long a statement may wait for a lock, and how long it may run.
    #
    # Neither was set, and `FOR UPDATE SKIP LOCKED` is not the protection it
    # looks like: it skips locked *rows* and waits like anything else for a
    # lock on the table. `initialize()` runs migrations whose CREATE TABLE and
    # ALTER TABLE take ACCESS EXCLUSIVE, so one stalled migration holds every
    # claim on the fleet.
    #
    # Three ink workers stopped claiming and went completely silent for
    # eighteen hours while `docker ps` said "Up 27 hours" -- not erroring in a
    # loop, which prints, but blocked inside a claim that could never return.
    # A bounded claim fails, says so, sleeps and tries again, which is a
    # recoverable outage instead of an invisible one.
    LOCK_TIMEOUT_MS = 10_000
    STATEMENT_TIMEOUT_MS = 120_000
    MIGRATION_STATEMENT_TIMEOUT_MS = 600_000
    # The name the advisory lock is derived from. A string rather than a magic
    # integer so a second lock added later is obviously a different one.
    MIGRATION_LOCK = "helena_ink_migrations"

    # No arguments, deliberately. Tests subclass this store and override
    # `_connect` to pin a search_path, calling `super()._connect()` -- so a
    # keyword here is a signature every override has to learn, and the ones
    # that did not learn it failed only against PostgreSQL, which is the half
    # that does not run locally.
    def _connect(self):
        import psycopg

        connection = psycopg.connect(self.dsn, connect_timeout=10)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"SET lock_timeout = {int(self.LOCK_TIMEOUT_MS)}")
                cursor.execute(
                    f"SET statement_timeout = {int(self.STATEMENT_TIMEOUT_MS)}")
            connection.commit()
        except Exception:
            connection.close()
            raise
        return connection

    def initialize(self) -> None:
        """Apply the migrations, once, whoever calls.

        This said migrations were "run deliberately, by one process", and that
        was simply not true: every worker calls it at startup, so a host that
        brings up three of them at once has three processes applying the same
        ALTER TABLE. Each takes an AccessExclusiveLock on the same relation, in
        an order nobody controls, and PostgreSQL kills one of them:

            DeadlockDetected: Process 33 waits for AccessExclusiveLock on
            relation 26218; blocked by process 32. Process 32 waits for
            AccessExclusiveLock on relation 26218; blocked by process 33.

        A worker that dies half way out of its own migration is a worker whose
        schema is in an unknown state, and it is the best explanation anyone has
        for three of them going silent for eighteen hours afterwards.

        So: an advisory lock around the whole block. The first process applies
        the migrations and the rest wait for it, which is what "one process"
        meant all along. Transaction-scoped, so it is released by the commit
        this `with` performs and cannot leak if the process dies holding it.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            # A schema change that times out half way is worse than one that is
            # waited for, and the wait now includes waiting for whoever got
            # here first -- so the lock timeout is raised to the same budget
            # rather than left at the ten seconds a claim gets.
            cursor.execute(
                "SET statement_timeout = "
                f"{int(self.MIGRATION_STATEMENT_TIMEOUT_MS)}")
            cursor.execute(
                f"SET lock_timeout = {int(self.MIGRATION_STATEMENT_TIMEOUT_MS)}")
            cursor.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (self.MIGRATION_LOCK,))
            for migration in MIGRATIONS:
                cursor.execute(migration.read_text())

    @staticmethod
    def _controlled_mission(cursor, mission_id: str | None) -> bool:
        if not mission_id:
            return False
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM segment_campaign_budget_admissions "
            "WHERE mission_id=%s)", (mission_id,),
        )
        row = cursor.fetchone()
        return bool(row and row[0])

    @staticmethod
    def _named_rows(cursor) -> list[dict[str, Any]]:
        """Rows keyed by the column names the cursor itself reports.

        A positional unpack is a promise that a SELECT list will never grow and
        never be reordered, and this particular SELECT list is the one the
        routing work widens: the route has to travel with the surface it
        describes. Growing it raises; reordering it does not -- seven columns in
        another order unpack cleanly and put the state where the digest belongs,
        after which the lineage is wrong in every field and no guard downstream
        can tell.

        `strict=True` because a description and a row of different lengths is a
        driver disagreeing with itself, and guessing which end to trim is how a
        surface ends up carrying its neighbour's facts.
        """
        columns = [description[0] for description in cursor.description]
        return [
            dict(row) if isinstance(row, Mapping)
            else dict(zip(columns, row, strict=True))
            for row in cursor.fetchall()
        ]

    @staticmethod
    def _lineage_from_surface_row(row: Mapping[str, Any], mission_id: str) -> dict[str, Any]:
        surface_id = row["surface_id"]
        source_snapshot_id = row["source_snapshot_id"]
        sample_id = row["sample_id"]
        artifact_sha256 = row["artifact_sha256"]
        artifact_uri = row["artifact_uri"]
        state = row["state"]
        payload = dict(row["payload"] or {})
        retained = payload.get("authoritative_lineage")
        if isinstance(retained, dict):
            return {**retained, "surface_id": surface_id,
                    "mission_id": mission_id}
        external = payload.get("canonical_external_admission")
        content = lambda value: hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
        return {
            "schema": "campaignx.authoritative_surface_lineage.v1",
            "mission_id": mission_id, "surface_id": surface_id,
            "namespace": payload.get("namespace") or "CANONICAL_SURFACE",
            "artifact_identity": payload.get("artifact_set_id") or
                f"surface:{surface_id}",
            "artifact_sha256": artifact_sha256, "artifact_uri": artifact_uri,
            "source_snapshot_id": source_snapshot_id,
            "source_binding_sha256": payload.get("source_binding_sha256") or
                content({"source_snapshot_id": source_snapshot_id,
                         "sample_id": sample_id}),
            "promotion_lineage_sha256": payload.get(
                "promotion_authority_sha256"),
            "promotion_lineage_kind": payload.get("promotion_lineage_kind"),
            "route_sha256": payload.get("route_sha256"),
            "surface_state": state,
            "canonical": payload.get("namespace") !=
                "NONCANONICAL_DISCOVERY",
            "external": payload.get("owner") == "imported",
            "external_admission_sha256": (
                content(external) if isinstance(external, dict) else None
            ),
            "ambiguous": False, "hash_conflict": False,
        }

    def _require_surface_ids_cursor(
        self, cursor, *, surface_ids: Sequence[str], mission_id: str | None,
        boundary: str, allow_unvalidated: Any = False,
    ) -> list[dict[str, Any]]:
        wanted = [str(value) for value in surface_ids]
        controlled = self._controlled_mission(cursor, mission_id)
        if not wanted:
            return []
        cursor.execute(
            """SELECT surface_id,source_snapshot_id,sample_id,artifact_sha256,
                      artifact_uri,state,payload FROM segment_surfaces
                 WHERE surface_id = ANY(%s)""",
            (wanted,),
        )
        by_id = {str(row["surface_id"]): row for row in self._named_rows(cursor)}
        api = _canonical_lineage_api()
        lineages = []
        for surface_id in wanted:
            lineage = (
                self._lineage_from_surface_row(by_id[surface_id], str(mission_id))
                if surface_id in by_id else None
            )
            api.require_canonical_lineage(
                boundary=boundary, controlled_mission=controlled,
                authoritative_lineage=lineage,
                allow_unvalidated=allow_unvalidated,
            )
            if lineage is not None:
                lineages.append(lineage)
        return lineages

    @staticmethod
    def _job_surface_ids(phase: str, parameters: dict[str, Any]) -> list[str]:
        if phase in {"P2", "P3", "P5", "P7"} and parameters.get("surface_id"):
            return [str(parameters["surface_id"])]
        if phase == "P4" and parameters.get("flattened_surface"):
            return [str(parameters["flattened_surface"])]
        if (phase == "P8" and parameters.get("lane") ==
                "vc3d-tifxyz-merge"):
            return [str(value) for value in parameters.get("artifact_ids") or []]
        return []

    def require_surface_lineage(
        self, *, surface_id: str, mission_id: str | None, boundary: str,
        allow_unvalidated: Any = False,
    ) -> dict[str, Any]:
        with self._connect() as connection, connection.cursor() as cursor:
            rows = self._require_surface_ids_cursor(
                cursor, surface_ids=[surface_id], mission_id=mission_id,
                boundary=boundary, allow_unvalidated=allow_unvalidated,
            )
        return rows[0] if rows else {}

    # The frozen name from the segmentation schema. Read here rather than
    # joined into the lineage query so a control plane that has not got it yet
    # fails one identifiable way instead of breaking every lineage read.
    ROUTING_RECEIPTS_TABLE = "segment_surface_routing_receipts"

    def _surface_routes(self, surface_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        with self._connect() as connection, connection.cursor() as cursor:
            try:
                cursor.execute(
                    f"""SELECT s.surface_id, s.area_cm2, s.artifact_sha256,
                               r.receipt AS routing_receipt
                          FROM segment_surfaces s
                          LEFT JOIN {self.ROUTING_RECEIPTS_TABLE} r
                            ON r.surface_id = s.surface_id
                         WHERE s.surface_id = ANY(%s)""",
                    (list(surface_ids),),
                )
            except Exception as error:  # noqa: BLE001 -- re-raised, named
                # 42P01 is undefined_table. Named rather than swallowed: a
                # control plane that cannot answer the routing question has to
                # say so, because the alternative is a renderer deciding that
                # silence meant yes.
                if getattr(error, "sqlstate", None) == "42P01":
                    raise RoutingRefused(
                        "ROUTING_UNAVAILABLE_ON_THIS_CONTROL_PLANE",
                        f"{self.ROUTING_RECEIPTS_TABLE} is not deployed here",
                    ) from error
                raise
            return {str(row["surface_id"]): row
                    for row in self._named_rows(cursor)}

    def require_job_standard_route(self, job: dict[str, Any]) -> None:
        """Refuse a job whose surface is not standard downstream work.

        Asked at execution, from the row as it is then. The queue decided this
        when the job was enqueued, and between then and now the surface can have
        been re-measured, resumed or regrown -- so the enqueue-time answer is a
        statement about the past and this one is not.

        A stored receipt is the answer where there is one. Where there is none
        the router is asked directly from the area on the row, under the same
        frozen policy, because today only imports write a receipt and nothing
        the fleet finalizes has one. Both paths go through the router's own
        predicate, so a forged receipt is refused exactly like a missing one.
        """
        phase = str(job.get("phase") or "P5")
        surface_ids = self._job_surface_ids(phase, job.get("parameters") or {})
        if not surface_ids:
            return
        routing = _surface_routing_api()
        policy = routing.load_policy()
        rows = self._surface_routes(surface_ids)
        for surface_id in surface_ids:
            row = rows.get(surface_id)
            if row is None:
                raise RoutingRefused(
                    "SURFACE_NOT_REGISTERED",
                    f"{surface_id} is not a registered surface")
            receipt = row.get("routing_receipt")
            if receipt is not None and not routing.verify_receipt(receipt):
                raise RoutingRefused(
                    "ROUTING_RECEIPT_UNVERIFIED",
                    f"{surface_id} carries a routing receipt that does not "
                    "match its own digest")
            if receipt is None:
                try:
                    receipt = routing.build_receipt(
                        surface_id=surface_id, area_cm2=row.get("area_cm2"),
                        policy=policy,
                        measurement={"decided_at": f"{phase}_EXECUTION_RESOLUTION"},
                        read_set={"artifact_sha256": row.get("artifact_sha256")},
                    )
                except ValueError as unmeasured:
                    raise RoutingRefused(
                        "ROUTING_UNDECIDABLE_NO_MEASURED_AREA", str(unmeasured),
                    ) from unmeasured
            if not routing.enters_canonical_downstream(receipt):
                raise RoutingRefused(
                    "SMALL_SURFACE_DIAGNOSTIC_NOT_CANONICAL_WORK",
                    f"{receipt.get('measured_area_cm2')} cm2 is below the "
                    f"{receipt.get('minimum_area_cm2')} cm2 floor: {surface_id} "
                    "is diagnostic evidence, and a phase run over it produces "
                    "an output that files next to a real one")

    def require_job_canonical_lineage(
        self, job: dict[str, Any], *, execution: bool,
    ) -> None:
        phase = str(job.get("phase") or "P5")
        suffix = "EXECUTION_RESOLUTION" if execution else "QUEUE_ADMISSION"
        boundary = f"{phase}_{suffix}"
        if phase == "P8" and execution:
            boundary = "P8_PARENT_MATERIALIZATION"
        surface_ids = self._job_surface_ids(phase, job.get("parameters") or {})
        with self._connect() as connection, connection.cursor() as cursor:
            controlled = self._controlled_mission(cursor, job.get("mission_id"))
            if controlled and not surface_ids and phase in {"P4", "P5", "P7", "P8"}:
                api = _canonical_lineage_api()
                api.require_canonical_lineage(
                    boundary=boundary, controlled_mission=True,
                    authoritative_lineage=None,
                    allow_unvalidated=(job.get("parameters") or {}).get(
                        "allow_unvalidated"),
                )
            self._require_surface_ids_cursor(
                cursor, surface_ids=surface_ids,
                mission_id=job.get("mission_id"), boundary=boundary,
                allow_unvalidated=(job.get("parameters") or {}).get(
                    "allow_unvalidated", False),
            )

    # -- enqueue ----------------------------------------------------------
    def enqueue(
        self,
        *,
        sample_id: str,
        parameters: dict[str, Any],
        phase: str = "P5",
        mission_id: str | None = None,
        profile_id: str | None = None,
        component: str | None = None,
        priority: int = 0,
        requested_host: str | None = None,
        max_attempts: int = 3,
        created_by: str = "panel",
        server_parameters: Mapping[str, Any] | None = None,
    ) -> str:
        """Queue one job.

        `server_parameters` carries the values the server decides -- where this
        publishes, which control binding it claims. They arrive as their own
        argument rather than merged into `parameters` by the caller, so that
        "the request supplied this" and "we supplied this" are different facts
        here rather than indistinguishable ones.
        """
        if not sample_id:
            raise JobRejected("sample_id is required")
        # A mission is the project that contains a run, and work does not exist
        # outside one. This was optional, and five P5 jobs went in without it in
        # a single afternoon: they ran on real data and published real
        # artifacts, and no view in the panel could show them, because every one
        # of those is mission-scoped and the gate renders nothing without a
        # mission. Work nobody can find is work nobody can review or attribute.
        #
        # Here rather than only in the panel: a harness, a script or anything
        # else holding a DSN reaches this method, and a rule one caller enforces
        # is a suggestion.
        if not str(mission_id or "").strip():
            raise JobRejected(
                "a job needs a mission: it is the project that contains the "
                "run, and nothing queued outside one can be found, reviewed or "
                "attributed afterwards")
        if str(mission_id).strip() == "unfiled":
            raise JobRejected(
                "'unfiled' is not a mission to queue into: it is a read-only "
                "view assembled from the receipts of runs that predate "
                "missions, so it describes what happened rather than naming a "
                "scope. Create a mission and queue into that.")
        if phase == "P5" and not profile_id:
            raise JobRejected("P5 jobs must name a profile_id")
        supplied = dict(server_parameters or {})
        clean = validate_parameters({**parameters, **supplied}, phase,
                                    server_owned=supplied)
        queued_job = {"phase": phase, "sample_id": sample_id,
                      "profile_id": profile_id, "parameters": clean}
        lane_id, lane = lane_for(queued_job)
        validate_lane_profile(queued_job, lane)
        validate_lane_sample(queued_job, lane_id, lane)
        # Preserve the historical GPU admission default for existing lanes.
        # The Villa TIFXYZ merge is explicitly CPU/high-memory work and must be
        # claimable by the segment/fleet-runner image that actually carries the
        # complete VC3D toolchain.
        gpu_required = bool(lane.get("gpu_required", True))
        minimum_vram_gb = int(lane.get("minimum_vram_gb", 0))
        job_id = f"{phase.lower()}-{uuid.uuid4().hex[:14]}"
        with self._connect() as connection, connection.cursor() as cursor:
            surface_ids = self._job_surface_ids(phase, clean)
            controlled = self._controlled_mission(cursor, mission_id)
            if controlled and not surface_ids and phase in {"P4", "P5", "P7", "P8"}:
                _canonical_lineage_api().require_canonical_lineage(
                    boundary=f"{phase}_QUEUE_ADMISSION",
                    controlled_mission=True, authoritative_lineage=None,
                    allow_unvalidated=clean.get("allow_unvalidated"),
                )
            self._require_surface_ids_cursor(
                cursor, surface_ids=surface_ids, mission_id=mission_id,
                boundary=f"{phase}_QUEUE_ADMISSION",
                allow_unvalidated=clean.get("allow_unvalidated", False),
            )
            cursor.execute(
                """INSERT INTO ink_jobs
                   (job_id, sample_id, profile_id, phase, component, mission_id,
                    parameters, priority, requested_host, max_attempts, created_by,
                    gpu_required, minimum_vram_gb)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (job_id, sample_id, profile_id, phase, component, mission_id,
                 json.dumps(clean), priority, requested_host, max_attempts, created_by,
                 gpu_required, minimum_vram_gb),
            )
            self._event(cursor, job_id, "enqueued",
                        {"phase": phase, "mission_id": mission_id,
                         "profile_id": profile_id, "component": component,
                         "parameters": clean, "by": created_by})
        return job_id

    # -- worker side ------------------------------------------------------
    def _runnable_candidate(self, rows: Sequence[tuple],
                            runtime: str | None) -> tuple | None:
        """The first of these rows this worker's image can actually run.

        `runtime_image_for` decides, rather than a second table mapping lanes to
        images: there is one answer to "which image does this lane need" and it
        already has a home. Rows this worker cannot run are left alone -- the
        transaction ends immediately, so their locks do too, and the worker that
        carries the image finds them on its next poll with every attempt intact.
        """
        # A specialist takes its own lane and nothing else.
        #
        # The rule was "anything that does not need a different image", which
        # reads correctly and is backwards for the worker carrying a lane's
        # frozen environment: a job needing no special image needs the ordinary
        # one, and the 9 um worker claimed canonical and timesformer runs it
        # could not execute -- failing them in two seconds each while the worker
        # that could run them sat idle.
        specialist = runtime in lane_runtime_images() if runtime else False
        for row in rows:
            if runtime is None:
                return row
            job_id, _sample, profile_id, parameters, *_rest = row
            phase = row[6]
            needed = runtime_image_for({
                "phase": phase, "profile_id": profile_id,
                "parameters": dict(parameters or {}),
            })
            if needed == runtime or (not needed and not specialist):
                return row
        return None

    def claim(self, *, worker_id: str, host_id: str, lease_seconds: int = 3600,
              phases: Sequence[str] | None = None,
              has_gpu: bool = True, gpu_vram_gb: float = 1e9,
              runtime: str | None = None) -> dict | None:
        """Take one pending job, or return None. Expired leases are recycled first.

        `phases` is what this worker can actually run. Without it every worker
        claims every phase, and a runtime missing one phase's binary takes that
        job anyway and fails it: the ink image carries a three-tool VC3D bundle
        with no vc_flatten, so it claimed P3, failed five surfaces on
        FileNotFoundError, and left the queue looking like flattening was
        broken rather than misrouted.

        `has_gpu` and `gpu_vram_gb` are what this machine has. ink_jobs has
        carried gpu_required and minimum_vram_gb since it was written and this
        query read neither, so a CPU-only host would claim a job that needs a
        card and fail it -- the same misrouting as above, by a different route.
        The segmentation queue has always filtered on its worker's capabilities;
        this one only looked like it did. The defaults are permissive because
        every caller today is a GPU host, and a default that silently claimed
        nothing would be a worse failure than the one being fixed.

        `runtime` is which image this worker is, and it is the same filter one
        column over. Three lanes need an image their claiming worker may not be
        running; the worker refuses those at execution, by name, which is a good
        message and a burned attempt. On a host running both a general ink
        worker and the 9 um one, the general worker wins the race, refuses, and
        the job is failed before the worker that could have run it ever polls --
        observed on gpu-1 as p5-4a8e0ba41a784d, claimed by gpu-1-ink0 and failed
        on its only attempt with gpu-1-ink9um idle ten seconds away.

        Which is why this looks at several rows rather than one. A queue whose
        head belongs to another runtime would otherwise hand this worker
        nothing at all, and a filter that can only see the first row is not a
        filter. Unset leaves it permissive, for the same reason `phases` does.
        """
        token = secrets.token_hex(24)
        with self._connect() as connection, connection.cursor() as cursor:
            # Read the lease in the same statement that clears it: the worker
            # that held it and the attempt it burned are gone the instant the
            # row is reset, and they are the whole of what makes a silent death
            # diagnosable afterwards.
            cursor.execute(
                """UPDATE ink_jobs SET state='pending', worker_id=NULL,
                       lease_token_hash=NULL, lease_expires_at=NULL, updated_at=now()
                   FROM (
                       SELECT job_id, worker_id, attempts, max_attempts,
                              lease_expires_at, state
                       FROM ink_jobs
                       WHERE state IN ('leased','running')
                         AND lease_expires_at IS NOT NULL AND lease_expires_at <= now()
                       FOR UPDATE SKIP LOCKED) AS expired
                   WHERE ink_jobs.job_id = expired.job_id
                   RETURNING ink_jobs.job_id, expired.worker_id, expired.attempts,
                             expired.max_attempts, expired.lease_expires_at,
                             expired.state""")
            for (recycled, held_by, attempts_used, ceiling, expired_at,
                 last_state) in cursor.fetchall():
                self._event(cursor, recycled, "lease_expired", {
                    "status": "LEASE_EXPIRED",
                    "failure_class": "LEASE_EXHAUSTION",
                    "held_by": held_by,
                    "detected_by": worker_id,
                    "expired_at": str(expired_at) if expired_at else None,
                    "last_state": last_state,
                    "attempt": attempts_used,
                    "attempts_remaining": max(int(ceiling or 0)
                                              - int(attempts_used or 0), 0),
                    "detail": "the lease ran out with no result recorded: the "
                              "worker stopped reporting before it finished",
                })

            # Written on every poll, claimed or not, because "this worker is
            # alive and looking" is the fact the fleet page could not state.
            # An idle worker and a blocked one showed the same thing, and the
            # blocked ones were blocked for eighteen hours.
            cursor.execute(
                """INSERT INTO ink_workers
                       (worker_id, host_id, runtime, last_poll_at, phases)
                   VALUES (%s, %s, %s, now(), %s)
                   ON CONFLICT (worker_id) DO UPDATE
                      SET host_id = EXCLUDED.host_id,
                          runtime = EXCLUDED.runtime,
                          phases = EXCLUDED.phases,
                          last_poll_at = now()""",
                (worker_id, host_id, runtime, list(phases) if phases else []),
            )

            cursor.execute(
                """SELECT job_id, sample_id, profile_id, parameters, attempts, max_attempts,
                          phase, component, mission_id
                   FROM ink_jobs
                   WHERE state='pending'
                     AND (requested_host IS NULL OR requested_host=%s)
                     AND (%s::text[] IS NULL OR phase = ANY(%s))
                     AND (gpu_required=false OR %s=true)
                     AND minimum_vram_gb <= %s
                   ORDER BY priority DESC, created_at
                   FOR UPDATE SKIP LOCKED LIMIT %s""",
                (host_id, list(phases) if phases else None,
                 list(phases) if phases else None, has_gpu, gpu_vram_gb,
                 CLAIM_CANDIDATES),
            )
            row = self._runnable_candidate(cursor.fetchall(), runtime)
            if row is not None:
                cursor.execute(
                    "UPDATE ink_workers SET last_claim_at = now() "
                    "WHERE worker_id = %s", (worker_id,))
            if row is None:
                return None
            (job_id, sample_id, profile_id, parameters, attempts, max_attempts,
             phase, component, mission_id) = row
            claim_job = {
                "phase": phase, "mission_id": mission_id,
                "parameters": dict(parameters or {}),
            }
            surface_ids = self._job_surface_ids(phase, claim_job["parameters"])
            controlled = self._controlled_mission(cursor, mission_id)
            boundary = (
                "P8_PARENT_MATERIALIZATION" if phase == "P8"
                else f"{phase}_EXECUTION_RESOLUTION"
            )
            if controlled and not surface_ids and phase in {"P4", "P5", "P7", "P8"}:
                _canonical_lineage_api().require_canonical_lineage(
                    boundary=boundary, controlled_mission=True,
                    authoritative_lineage=None,
                    allow_unvalidated=claim_job["parameters"].get(
                        "allow_unvalidated"),
                )
            self._require_surface_ids_cursor(
                cursor, surface_ids=surface_ids, mission_id=mission_id,
                boundary=boundary,
                allow_unvalidated=claim_job["parameters"].get(
                    "allow_unvalidated", False),
            )
            if attempts >= max_attempts:
                # Reaching this branch means every attempt ended without any
                # worker calling `finish` -- one that had would have left the
                # job failed or succeeded, not pending. So the class is known,
                # and a `failed` row with an empty `result` is the one thing
                # that made such a death read like a job nobody ever claimed.
                #
                # Nothing is claimed here about whether inference ran: a job
                # can expire after a model is loaded and used, and writing
                # `ink_used: False` would be inferring a negative from a
                # missing report.
                spent = {
                    "error": "LEASE_EXHAUSTION: every attempt's lease expired "
                             "and no worker ever recorded a result",
                    "failure_class": "LEASE_EXHAUSTION",
                    "status": "LEASE_EXPIRED",
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                }
                cursor.execute(
                    "UPDATE ink_jobs SET state='failed', result=%s::jsonb, "
                    "updated_at=now() WHERE job_id=%s",
                    (json.dumps(spent), job_id))
                self._event(cursor, job_id, "exhausted", spent)
                return None
            cursor.execute(
                """UPDATE ink_jobs SET state='leased', worker_id=%s, lease_token_hash=%s,
                       lease_expires_at=now() + (%s || ' seconds')::interval,
                       attempts=attempts+1, updated_at=now()
                   WHERE job_id=%s""",
                (worker_id, token_hash(token), str(lease_seconds), job_id),
            )
            self._event(cursor, job_id, "claimed",
                        {"worker_id": worker_id, "host_id": host_id, "attempt": attempts + 1})
        return {"job_id": job_id, "sample_id": sample_id, "profile_id": profile_id,
                "phase": phase, "component": component, "mission_id": mission_id,
                "parameters": parameters, "lease_token": token, "attempt": attempts + 1}

    def heartbeat(self, job_id: str, lease_token: str, lease_seconds: int = 3600,
                  *, progress: dict | None = None) -> bool:
        """Renew the lease, and say where the job is while doing it.

        One statement. Progress on its own write would be a second round trip
        every fifteen seconds to say something this one is already in a
        position to carry, and the two could then disagree about whether the
        lease is still held.

        COALESCE rather than an assignment: a beat with nothing new to report
        leaves the last line standing, so a job that has gone quiet still shows
        where it got to instead of blanking.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE ink_jobs
                   SET lease_expires_at=now() + (%s || ' seconds')::interval,
                       updated_at=now(),
                       progress=COALESCE(%s::jsonb, progress)
                   WHERE job_id=%s AND lease_token_hash=%s AND state IN ('leased','running')
                   RETURNING cancel_requested""",
                (str(lease_seconds),
                 json.dumps(progress) if progress is not None else None,
                 job_id, token_hash(lease_token)),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError(f"lease for {job_id} is no longer held")
        # Answered on the write that was happening anyway. A second statement to
        # ask "should I stop" would be a second thing that can disagree with the
        # first about whether this worker still holds the job.
        return row[0] is not None

    def mark_running(self, job_id: str, lease_token: str, *, output_dir: str, command: list[str]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE ink_jobs SET state='running', output_dir=%s,
                       progress=NULL, updated_at=now()
                   WHERE job_id=%s AND lease_token_hash=%s AND state='leased'""",
                (output_dir, job_id, token_hash(lease_token)),
            )
            if cursor.rowcount == 0:
                raise RuntimeError(f"lease for {job_id} is no longer held")
            self._event(cursor, job_id, "started", {"command": command, "output_dir": output_dir})

    def finish(self, job_id: str, lease_token: str, *, state: str, result: dict[str, Any]) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"{state} is not terminal")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """UPDATE ink_jobs SET state=%s, result=%s, lease_token_hash=NULL,
                       lease_expires_at=NULL, updated_at=now()
                   WHERE job_id=%s AND lease_token_hash=%s""",
                (state, json.dumps(result), job_id, token_hash(lease_token)),
            )
            if cursor.rowcount == 0:
                raise RuntimeError(f"lease for {job_id} is no longer held")
            self._event(cursor, job_id, state, result)

    # -- panel side -------------------------------------------------------
    def request_cancel(self, job_id: str, *, by: str = "panel") -> bool:
        """Ask the worker holding this job to stop.

        The work is a subprocess on another machine, so this cannot end it --
        it writes the request down, and the worker reads it on the heartbeat it
        already sends. Recorded as an event too, because who asked and when is
        part of why a run has no result.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ink_jobs SET cancel_requested=now(), updated_at=now() "
                "WHERE job_id=%s AND state IN ('leased','running') "
                "AND cancel_requested IS NULL",
                (job_id,))
            asked = cursor.rowcount > 0
            if asked:
                self._event(cursor, job_id, "cancel_requested", {"by": by})
        return asked

    def cancel(self, job_id: str, *, by: str = "panel") -> bool:
        """Cancel only what has not started. A running job keeps its lease."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE ink_jobs SET state='cancelled', updated_at=now() "
                "WHERE job_id=%s AND state='pending'",
                (job_id,))
            cancelled = cursor.rowcount > 0
            if cancelled:
                self._event(cursor, job_id, "cancelled", {"by": by})
        return cancelled

    def jobs(self, *, limit: int = 100, states: tuple[str, ...] | None = None,
             mission_id: str | None = None, phase: str | None = None,
             sample_id: str | None = None) -> list[dict]:
        """The newest jobs, newest first.

        `phase` and `sample_id` filter in SQL rather than in the caller. A caller
        that fetches the newest fifty rows overall and then keeps the ones
        matching its phase gets nothing at all whenever another phase has
        produced fifty jobs more recently -- which is every busy day, and which
        made quiet phases look idle while they were running.
        """
        conditions, params = [], []
        if states:
            conditions.append("state = ANY(%s)")
            params.append(list(states))
        if mission_id:
            conditions.append("mission_id = %s")
            params.append(mission_id)
        if phase:
            conditions.append("phase = %s")
            params.append(phase)
        if sample_id:
            conditions.append("sample_id = %s")
            params.append(sample_id)
        clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"""SELECT job_id, sample_id, profile_id, phase, component, mission_id,
                           parameters, state, priority,
                           requested_host, worker_id, attempts, max_attempts, output_dir,
                           result, progress, created_by, created_at, updated_at,
                           lease_expires_at
                    FROM ink_jobs {clause} ORDER BY created_at DESC LIMIT %s""",
                params,
            )
            columns = [c.name for c in cursor.description]
            return [
                {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                 for k, v in zip(columns, row)}
                for row in cursor.fetchall()
            ]

    def job(self, job_id: str) -> dict | None:
        """One job, with what it produced. How P5 finds the render it reads."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT job_id, sample_id, phase, profile_id, state, parameters, "
                "       result, output_dir, mission_id "
                "FROM ink_jobs WHERE job_id=%s", (job_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        return {"job_id": row[0], "sample_id": row[1], "phase": row[2],
                "profile_id": row[3], "state": row[4], "parameters": row[5],
                "result": row[6], "output_dir": row[7], "mission_id": row[8]}

    def note(self, job_id: str, event_type: str, payload: Any) -> None:
        """Record something about a job that is not a state change."""
        with self._connect() as connection, connection.cursor() as cursor:
            self._event(cursor, job_id, event_type, payload)

    def flattened_sheet(self, surface_id: str, profile_id: str) -> dict[str, Any]:
        """The sheet P3 unrolled for this surface under this profile.

        P4 lives in the ink store and flattenings live in the segmentation
        store, but both are the same database, so this is one read rather than a
        second control plane or a path copied between two of them.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            # `requested_by_job_id` is a key inside the jsonb payload, not a
            # column: surface_flattenings carries flattening_id, surface_id,
            # profile_id, state, artifact_uri, artifact_sha256, area_ratio,
            # payload and created_at. Selecting it as a column raised
            # UndefinedColumn on every call, so this read had never once
            # succeeded -- P4 is simply the first boundary a run ever reached.
            cursor.execute(
                "SELECT flattening_id, artifact_uri, artifact_sha256, state, "
                "area_ratio, payload->>'requested_by_job_id' "
                "FROM surface_flattenings WHERE surface_id=%s AND profile_id=%s",
                (surface_id, profile_id))
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"no flattened sheet for {surface_id} under {profile_id}; P3 has "
                "not unrolled this surface, or did so under another profile")
        if row[3] != "FLATTENED":
            raise RuntimeError(
                f"the flattening of {surface_id} is {row[3]}, not FLATTENED; "
                "rendering a failed flattening renders whatever was left behind")
        if not row[1]:
            raise RuntimeError(
                f"the flattened sheet for {surface_id} was never published, so "
                "there is nothing to fetch; re-run P3 with an --artifact-store")
        return {"flattening_id": row[0], "artifact_uri": row[1],
                "artifact_sha256": row[2], "state": row[3],
                "area_ratio": row[4], "requested_by_job_id": row[5]}

    def merge_surfaces(self, artifact_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Resolve immutable TIFXYZ inputs for the VC3D merge lane.

        `segment_surfaces.surface_id` is the platform's stable artifact id.  The
        join returns the source facts used to refuse a cross-scroll,
        cross-acquisition, cross-frame or cross-resolution assembly before any
        large artifact is downloaded.
        """

        wanted = [str(value) for value in artifact_ids]
        if len(wanted) != len(set(wanted)):
            raise RuntimeError("merge artifact ids must be distinct")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT s.surface_id, s.source_snapshot_id, s.sample_id,
                          s.artifact_sha256, s.artifact_uri, s.bbox_xyz,
                          s.sample_points, s.area_cm2, s.state,
                          s.physical_qc_state, s.geometry_qc_state, s.payload,
                          src.ct_uri, src.ct_sha256, src.m7_uri, src.m7_sha256,
                          src.shape_xyz, src.voxel_size_um,
                          src.coordinate_frame, src.payload
                     FROM segment_surfaces s
                     JOIN segment_source_snapshots src
                       ON src.source_snapshot_id=s.source_snapshot_id
                    WHERE s.surface_id = ANY(%s)""",
                (wanted,),
            )
            rows = cursor.fetchall()
        by_id = {row[0]: row for row in rows}
        missing = [value for value in wanted if value not in by_id]
        if missing:
            raise RuntimeError(f"unknown TIFXYZ artifact ids: {missing}")
        fields = (
            "surface_id", "source_snapshot_id", "sample_id",
            "artifact_sha256", "artifact_uri", "bbox_xyz", "sample_points",
            "area_cm2", "state", "physical_qc_state", "geometry_qc_state",
            "payload", "ct_uri", "ct_sha256", "m7_uri", "m7_sha256",
            "shape_xyz", "voxel_size_um", "coordinate_frame", "source_payload",
        )
        resolved = [dict(zip(fields, by_id[value], strict=True)) for value in wanted]
        api = _canonical_lineage_api()
        for parent in resolved:
            payload = dict(parent.get("payload") or {})
            api.require_canonical_lineage(
                boundary="P8_PARENT_MATERIALIZATION",
                controlled_mission=payload.get("controlled_first_letters") is True,
                authoritative_lineage=payload.get("authoritative_lineage"),
                allow_unvalidated=payload.get("allow_unvalidated"),
            )
        return resolved

    def register_merged_surface(
        self,
        surface: dict[str, Any],
        parents: Sequence[dict[str, str]],
        *,
        job_id: str,
        qc_profile_id: str,
    ) -> dict[str, Any]:
        """Insert one immutable derived surface and its complete N->1 lineage.

        Parent rows are locked and their digests rechecked in the same
        transaction as the insert.  A replay with identical output is
        idempotent; a row with the same identity but different bytes or parents
        is refused rather than replaced.
        """

        if not qc_profile_id:
            raise RuntimeError("a merged surface requires a frozen physical-QC profile")
        from fleet.common import stable_id

        api = _canonical_lineage_api()
        controlled = bool(surface.get("controlled_first_letters")) or any(
            bool(parent.get("controlled_first_letters"))
            or bool((parent.get("payload") or {}).get("controlled_first_letters"))
            for parent in parents
        )
        parent_lineage_hashes = []
        for parent in parents:
            parent_payload = dict(parent.get("payload") or {})
            lineage = parent.get("authoritative_lineage") or parent_payload.get(
                "authoritative_lineage")
            api.require_canonical_lineage(
                boundary="P8_PARENT_MATERIALIZATION",
                controlled_mission=controlled,
                authoritative_lineage=lineage,
                allow_unvalidated=(
                    parent.get("allow_unvalidated")
                    if "allow_unvalidated" in parent
                    else parent_payload.get("allow_unvalidated")
                ),
            )
            if isinstance(lineage, dict):
                parent_lineage_hashes.append(hashlib.sha256(json.dumps(
                    lineage, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False,
                ).encode()).hexdigest())
        output_lineage = {
            "schema": "campaignx.authoritative_surface_lineage.v1",
            "mission_id": surface.get("mission_id"),
            "surface_id": surface.get("surface_id"),
            "namespace": surface.get("namespace") or "CANONICAL_SURFACE",
            "artifact_identity": f"surface:{surface.get('surface_id')}",
            "artifact_sha256": surface.get("artifact_sha256"),
            "artifact_uri": surface.get("artifact_uri"),
            "source_snapshot_id": surface.get("source_snapshot_id"),
            "source_binding_sha256": hashlib.sha256(json.dumps({
                "source_snapshot_id": surface.get("source_snapshot_id"),
                "sample_id": surface.get("sample_id"),
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "promotion_lineage_sha256": None,
            "promotion_lineage_kind": None,
            "route_sha256": hashlib.sha256(json.dumps({
                "relationship": "vc3d-tifxyz-merge",
                "parents": parent_lineage_hashes,
                "job_id": job_id,
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "surface_state": surface.get("state", "MERGED"),
            "canonical": surface.get("namespace") != "NONCANONICAL_DISCOVERY",
            "external": False, "external_admission_sha256": None,
            "ambiguous": False, "hash_conflict": False,
        }
        api.require_canonical_lineage(
            boundary="P8_DERIVED_SURFACE_REGISTRATION",
            controlled_mission=controlled,
            authoritative_lineage=output_lineage,
            allow_unvalidated=surface.get("allow_unvalidated"),
        )
        if controlled:
            surface = {
                **surface, "controlled_first_letters": True,
                "authoritative_lineage": output_lineage,
                "allow_unvalidated": False,
            }

        parent_ids = [str(row["surface_id"]) for row in parents]
        expected = {str(row["surface_id"]): str(row["artifact_sha256"])
                    for row in parents}
        qc_job_id = stable_id(
            "qc-job", {"surface_id": surface["surface_id"],
                       "profile_id": qc_profile_id})
        # The merged sheet's own routing question, asked before its physical-QC
        # job is written. The parents were gated at admission by
        # require_job_standard_route and the merged surface is not any of them:
        # it is new, it has its own area, and stitching two sheets that each
        # clear the floor can produce one that does not.
        #
        # A merge whose area never arrived is refused here rather than enqueued
        # unrouted. This method's whole output is a claimable physical-QC job,
        # and physical QC on two square millimetres of papyrus is the thing the
        # floor exists to prevent.
        routing = _surface_routing_api()
        policy = routing.load_policy()
        try:
            decided = routing.receipt_for_surface(surface, policy=policy)
        except (ValueError, KeyError) as unmeasured:
            raise RuntimeError(
                f"{surface.get('surface_id')} has no measured area, so the "
                "merge that produced it cannot be routed and no physical-QC "
                f"job may be created for it: {unmeasured}") from unmeasured
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT surface_id, artifact_sha256 FROM segment_surfaces "
                "WHERE surface_id = ANY(%s) FOR SHARE",
                (parent_ids,),
            )
            observed = {row[0]: row[1] for row in cursor.fetchall()}
            if observed != expected:
                raise RuntimeError(
                    "a merge parent changed between materialization and registration")
            cursor.execute(
                """INSERT INTO segment_surfaces
                   (surface_id,source_snapshot_id,sample_id,owner,
                    artifact_sha256,artifact_uri,bbox_xyz,sample_points,
                    area_cm2,state,physical_qc_state,geometry_qc_state,payload)
                   VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT DO NOTHING""",
                (
                    surface["surface_id"], surface["source_snapshot_id"],
                    surface["sample_id"], surface.get("owner", "campaign-x"),
                    surface["artifact_sha256"], surface["artifact_uri"],
                    json.dumps(surface["bbox_xyz"]),
                    json.dumps(surface.get("sample_points")),
                    surface.get("area_cm2"), surface.get("state", "MERGED"),
                    surface.get("physical_qc_state", "UNVALIDATED"),
                    surface["geometry_qc_state"],
                    json.dumps(surface, sort_keys=True, separators=(",", ":")),
                ),
            )
            cursor.execute(
                "SELECT surface_id, artifact_sha256, artifact_uri, payload "
                "FROM segment_surfaces WHERE source_snapshot_id=%s "
                "AND artifact_sha256=%s",
                (surface["source_snapshot_id"], surface["artifact_sha256"]),
            )
            existing = cursor.fetchone()
            if existing is None:
                raise RuntimeError("merged surface insert produced no row")
            if str(existing[0]) != str(surface["surface_id"]):
                raise RuntimeError(
                    f"merged bytes already belong to a different surface {existing[0]}")
            existing_parents = set((existing[3] or {}).get("parent_surface_ids") or [])
            if existing_parents and existing_parents != set(parent_ids):
                raise RuntimeError(
                    "merged surface identity already exists with different parents")
            # Written in the same transaction as the surface, exactly as both
            # segmentation stores write it as a surface is created. DO NOTHING
            # because a replay of an identical merge must be idempotent and the
            # table refuses UPDATE outright; the answer that matters is the one
            # read back below, not the one just sent.
            cursor.execute(
                f"""INSERT INTO {self.ROUTING_RECEIPTS_TABLE}
                    (surface_id,route,measured_area_cm2,minimum_area_cm2,
                     policy_version,profile_id,receipt_sha256,receipt)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (surface_id) DO NOTHING""",
                (decided["surface_id"], decided["route"],
                 decided["measured_area_cm2"], decided["minimum_area_cm2"],
                 decided["policy_version"], decided["profile_id"],
                 decided["receipt_sha256"],
                 json.dumps(decided, sort_keys=True, separators=(",", ":"))),
            )
            cursor.execute(
                f"SELECT receipt FROM {self.ROUTING_RECEIPTS_TABLE} "
                "WHERE surface_id=%s",
                (surface["surface_id"],),
            )
            persisted = cursor.fetchone()
            routing_receipt = persisted[0] if persisted is not None else None
            # Re-decided against the area rather than trusted for having a
            # digest, which is what makes a receipt already in the table -- from
            # an earlier attempt, or from anywhere else -- unable to outrank the
            # surface it describes.
            if routing_receipt is None or not routing.agrees_with_measurement(
                routing_receipt, surface.get("area_cm2"), policy=policy,
            ):
                raise RuntimeError(
                    f"{surface['surface_id']} routing receipt did not persist "
                    "as the decision its measured area produces")
            # Asked positively through the router's own predicate: exactly a
            # verified STANDARD route yields a claimable job, so a forged or
            # corrupted receipt strands the job the same way a missing one does.
            # claim_qc takes any PENDING row and never revisits the route, so
            # this insert is the only thing standing between a two-square-
            # millimetre merged sheet and a physical-QC technician.
            qc_state = ("PENDING" if routing.enters_standard_qc(routing_receipt)
                        else _qc_small_surface_diagnostic())
            for ordinal, parent in enumerate(parents):
                cursor.execute(
                    """INSERT INTO surface_derivations
                       (child_surface_id,parent_surface_id,parent_artifact_sha256,
                        ordinal,relationship,job_id)
                       VALUES(%s,%s,%s,%s,'vc3d-tifxyz-merge',%s)
                       ON CONFLICT(child_surface_id,parent_surface_id) DO NOTHING""",
                    (surface["surface_id"], parent["surface_id"],
                     parent["artifact_sha256"], ordinal, job_id),
                )
                cursor.execute(
                    """SELECT parent_artifact_sha256, ordinal, relationship
                         FROM surface_derivations
                        WHERE child_surface_id=%s AND parent_surface_id=%s""",
                    (surface["surface_id"], parent["surface_id"]),
                )
                lineage = cursor.fetchone()
                expected_lineage = (
                    parent["artifact_sha256"], ordinal, "vc3d-tifxyz-merge")
                if lineage is None or tuple(lineage) != expected_lineage:
                    raise RuntimeError(
                        "merged surface lineage already exists with different facts")
            cursor.execute(
                """INSERT INTO segment_qc_jobs
                   (qc_job_id,surface_id,profile_id,state,payload,result,updated_at)
                   VALUES(%s,%s,%s,%s,%s::jsonb,NULL,now())
                   ON CONFLICT(surface_id,profile_id) DO NOTHING""",
                (qc_job_id, surface["surface_id"], qc_profile_id, qc_state,
                 json.dumps({
                     "derived_from_job": job_id,
                     "relationship": "vc3d-tifxyz-merge",
                     "artifact_sha256": surface["artifact_sha256"],
                     "routing_receipt_sha256": routing_receipt["receipt_sha256"],
                 }, sort_keys=True, separators=(",", ":"))),
            )
            cursor.execute(
                """SELECT qc_job_id,surface_id,profile_id,state
                     FROM segment_qc_jobs
                    WHERE surface_id=%s AND profile_id=%s""",
                (surface["surface_id"], qc_profile_id),
            )
            qc = cursor.fetchone()
            if qc is None or tuple(qc[:3]) != (
                    qc_job_id, surface["surface_id"], qc_profile_id):
                raise RuntimeError(
                    "merged surface physical-QC job conflicts with existing facts")
        return {
            "surface_id": surface["surface_id"],
            "artifact_sha256": surface["artifact_sha256"],
            "artifact_uri": surface["artifact_uri"],
            "parents": parent_ids,
            "qc_job_id": qc_job_id,
            "qc_profile_id": qc_profile_id,
            "qc_state": qc[3],
            "route": routing_receipt["route"],
            "routing_receipt": routing_receipt,
        }

    def events(self, job_id: str, limit: int = 50) -> list[dict]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT event_type, payload, created_at FROM ink_job_events "
                "WHERE job_id=%s ORDER BY event_id DESC LIMIT %s",
                (job_id, limit))
            return [{"event_type": t, "payload": p, "created_at": c.isoformat()}
                    for t, p, c in cursor.fetchall()]

    # -- hosts ------------------------------------------------------------
    def register_host(self, host_id: str, ssh_target: str, roles: list[str],
                      notes: str | None = None) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """INSERT INTO ink_hosts (host_id, ssh_target, roles, notes)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (host_id) DO UPDATE
                   SET ssh_target=EXCLUDED.ssh_target, roles=EXCLUDED.roles,
                       notes=EXCLUDED.notes""",
                (host_id, ssh_target, roles, notes))

    # How long a worker may go without polling before it is not idle any more.
    #
    # Six times the default ten-second poll. A worker that has missed six in a
    # row is not between jobs; the three that hung missed about six thousand.
    WORKER_SILENT_SECONDS = 60

    def workers(self, *, silent_after: int | None = None) -> list[dict[str, Any]]:
        """Every worker that has ever polled, and whether it still is.

        `state` is the distinction the fleet page could not draw: a worker with
        nothing to claim and a worker that cannot claim looked identical, and
        `docker ps` said "Up" for both.
        """
        threshold = int(silent_after or self.WORKER_SILENT_SECONDS)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT worker_id, host_id, runtime, phases,
                          last_poll_at, last_claim_at,
                          EXTRACT(EPOCH FROM (now() - last_poll_at))
                     FROM ink_workers ORDER BY worker_id""")
            return [{
                "worker_id": worker_id, "host_id": host_id, "runtime": runtime,
                "phases": list(phases or []),
                "last_poll_at": last_poll.isoformat() if last_poll else None,
                "last_claim_at": last_claim.isoformat() if last_claim else None,
                "seconds_since_poll": round(float(since), 1),
                "state": "POLLING" if float(since) <= threshold else "SILENT",
            } for (worker_id, host_id, runtime, phases, last_poll, last_claim,
                   since) in cursor.fetchall()]

    def record_host_state(self, host_id: str, state: dict[str, Any]) -> None:
        """What this machine can offer, merged across the workers that see it.

        A host runs more than one worker and they do not see the same hardware:
        one may be given a card and another not. Whichever wrote last would
        otherwise decide what the panel shows, and a worker with no card writes
        an empty GPU list over a real one -- leaving the Hosts page reporting no
        hardware on a machine that has just finished a render.

        Cards are merged by uuid and kept for an hour after they were last seen.
        A worker that sees none contributes none rather than erasing what
        another saw, and a card that is genuinely gone ages out.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT last_state FROM ink_hosts WHERE host_id=%s",
                           (host_id,))
            row = cursor.fetchone()
            merged = dict(state)
            merged["gpus"] = merge_gpu_observations(
                (row[0] if row else None) or {}, state)
            cursor.execute(
                "UPDATE ink_hosts SET last_seen_at=now(), last_state=%s WHERE host_id=%s",
                (json.dumps(merged), host_id))

    def hosts(self) -> list[dict]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT host_id, ssh_target, roles, enabled, notes, last_seen_at, last_state "
                "FROM ink_hosts ORDER BY host_id")
            return [
                {"host_id": h, "ssh_target": s, "roles": list(r or []), "enabled": e,
                 "notes": n, "last_seen_at": ls.isoformat() if ls else None, "last_state": st}
                for h, s, r, e, n, ls, st in cursor.fetchall()
            ]

    def set_host_roles(self, host_id: str, roles: list[str]) -> int:
        """Replace what a host is asked to do. Returns rows changed.

        Separate from register_host, whose upsert also writes ssh_target and
        notes: reusing it to change roles would need the caller to resend those,
        and a caller that resends them from a stale page overwrites them.
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE ink_hosts SET roles=%s WHERE host_id=%s",
                           (list(roles), host_id))
            return cursor.rowcount

    def set_host_enabled(self, host_id: str, enabled: bool) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE ink_hosts SET enabled=%s WHERE host_id=%s", (enabled, host_id))

    # -- internals --------------------------------------------------------
    @staticmethod
    def _event(cursor, job_id: str, event_type: str, payload: Any) -> None:
        cursor.execute(
            "INSERT INTO ink_job_events (job_id, event_type, payload) VALUES (%s,%s,%s)",
            (job_id, event_type, json.dumps(payload)))


def store_from_env() -> InkJobStore | None:
    dsn = os.environ.get("CX_DB", "")
    return InkJobStore(dsn) if dsn else None
