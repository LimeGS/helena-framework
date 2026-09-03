---
title: Step 3 — Flatten and render
summary: Unroll the surface, then sample the CT through it into a stack of layers.
---

Two phases, and the walkthrough runs both: flatten the geometry so the sheet is
a rectangle, then sample the volume through it to get something a detector can
read.

## P3 — flatten

The walkthrough calls the lasagna flattener optional but recommended. Helena
treats it the same way: a fit that is already close to flat can skip it, and
anything with real curvature should not.

Flatten from the **Flattening** page — `POST /api/flattening/run`. Like P2, P3
is not queueable from Command.

> **Trap** It refuses with a 409 if `CX_FLATTEN_STORE` has been emptied: a sheet
> published nowhere is one P4 cannot read. The default is
> `/artifacts/flattened-v1`, on the platform's own volume.

The panel sends `limit`, `dry_run`, `allow_unvalidated`, `sample_id` and
`mission_id`; `surface_id` is accepted by the route but not offered by the
form, and `profile` and `binary` are contract-only, with `binary` accepting
exactly one value. The fields:

| Field | What it is |
| --- | --- |
| `surface_id` | which certified surface to flatten |
| `profile` | the flattening profile — the frozen settings. The default, `flatten-abf-v1@1.0.0`, runs `vc_flatten` (LSCM then ABF++). A second profile, `flatten-lasagna-v1@1.0.0`, selects upstream's gradient-descent flattener instead — GPU only, and marked experimental rather than default: no frozen reference control has been run against it on this corpus |
| `binary` | which flattener runs; allowlisted to the `vc_flatten` binary, so it is the only value that is accepted |
| `limit` | how many surfaces, when you queue a batch |

> **Trap** P3 requires `GEOMETRY_CERTIFIED` and rejects anything else rather
> than degrading. That does not relax with `allow_unvalidated` -- the gate
> holds either way, because a sheet that crosses two laminae flattens into a
> rectangle of two different pieces of papyrus either way. If it refuses, the
> answer is at [P2](#/docs/phases/p2), not here.

The output is a flattened sheet, published with its digest. P4 names it rather
than a path.

## P4 — render the surface volume

Queue P4 from its own phase panel — unlike P3, it is one of the phases
`POST /api/jobs` accepts directly. The panel's **Renderer** picker chooses the
lane, and **Queue P4 job** runs it.

There are two lanes. The default, `vc-render-tifxyz`, is
volume-cartographer's own renderer: walk the surface, sample the CT at a
series of depths, write a numbered TIFF per depth. It takes the fields below.
A second lane, `chunk-gather`, takes `ppm` and `volume_key` instead — a PPM
plus the OME-Zarr key of the rescan to sample — and does not read a P3 sheet
at all; it streams by stripe, checkpoints through `done.json` so a crash
resumes, and correlates at r = 0.89 with an official ink map, measured on
PHerc0332. The two lanes have never been compared against each other on the
same surface, so that number describes `chunk-gather` only.

The walkthrough's numbers are 28 slices at 1-slice step, 16 GB of cache, scale 1
for 9 µm data. In Helena, for the default lane:

| Field | Blog equivalent | Notes |
| --- | --- | --- |
| `flattened_surface` | the segment | exactly one of this or `segmentation`; naming it makes Helena resolve and fill in `flattening_id`, `p3_job_id` and `flattened_artifact_sha256` for you |
| `volume` | — | **required**; where chunks are staged, and a cache when `remote_url` is set |
| `num_slices` | 28 | how deep the stack goes |
| `slice_step` | 1 | spacing along the normal, in **voxels**; a float |
| `scale` | 1 | pixels per level-g voxel |
| `group_idx` | 0 | resolution level; 0 is full |
| `cache_gb` | 16 | how much disk the staged chunks may use, in GB |
| `source_voxel_um` | — | filled by the deployment; see below |
| `flip_normals` | — | which side of the sheet is up |

> **Trap** `source_voxel_um` is the physical scale, and the renderer will guess
> `1.0` and carry on if nobody tells it — reporting *"Voxel size: 1.0 (no
> metadata found; override with --voxel-size)"* in its log. Helena resolves it
> from the frozen P0 catalogue and passes it, or refuses the job with a 409
> naming the scales the catalogue does hold if the scan is not in it — never
> silently defaulting to `1.0`. On a volume with no catalogue entry, a hand-run
> render and a Helena one will not agree unless you state the number yourself.

> **Trap** It refuses with a 409 if `CX_RENDER_STORE` is unset, the same way
> P3 refuses without `CX_FLATTEN_STORE`: a layer stack that exists only on the
> worker that rendered it is lost when that worker goes away. The default is
> `/artifacts/layer-stacks-v1`, on the platform's own volume. Pass
> `allow_local_layers` for a deliberate single-machine run that publishes
> nowhere instead.

## Reading the result

The walkthrough's test is whether you can see structure: direct ink, or the
characteristic raised, cracked texture. Before that, Helena checks the render is
a render at all:

- the stack has the number of slices that was asked for;
- the indices are contiguous from zero;
- **the middle slice is not a constant.** A render whose surface fell outside
  the cached region writes the requested number of files and every one of them
  is flat. Exit code zero, 28 files, nothing in them.

A job that fails any of those is `failed`, not `succeeded`. That check exists
because the failure looks exactly like success from the outside.

Next: [detect ink](#/docs/tutorial/detect-ink).
