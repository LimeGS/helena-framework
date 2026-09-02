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
| `profile` | the flattening profile — the frozen settings |
| `binary` | which flattener runs; allowlisted, and the default is the right one |
| `limit` | how many surfaces, when you queue a batch |

> **Trap** P3 requires `GEOMETRY_CERTIFIED` and rejects anything else rather
> than degrading. This is not a certification-run rule -- it holds in both
> modes, because a sheet that crosses two laminae flattens into a rectangle of
> two different pieces of papyrus either way. If it refuses, the answer is at
> [P2](#/docs/phases/p2), not here.

The output is a flattened sheet, published with its digest. P4 names it rather
than a path.

## P4 — render the surface volume

This is `vc_render_tifxyz`: walk the surface, sample the CT at a series of
depths, write a numbered TIFF per depth.

The walkthrough's numbers are 28 slices at 1-slice step, 16 GB of cache, scale 1
for 9 µm data. In Helena:

| Field | Blog equivalent | Notes |
| --- | --- | --- |
| `flattened_surface` | the segment | exactly one of this or `segmentation`; naming it also requires `flattening_id`, `p3_job_id` and `flattened_artifact_sha256` |
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
> metadata found)"* in its log. Helena passes it, so the render and the depth
> arithmetic use the same number. On a volume with no metadata, a hand-run
> render and a Helena one will not agree unless you pass it yourself.

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
