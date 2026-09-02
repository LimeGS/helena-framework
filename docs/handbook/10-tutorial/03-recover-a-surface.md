---
title: Step 2 — Recover a surface
summary: The spiral fitter, its three inputs, and the one that is not downloadable.
---

This is the step the walkthrough calls the hard part, and it is where a first
attempt usually stops. The goal is a mesh that follows one sheet of papyrus
through the scroll without jumping to the next.

## What the fitter needs

The spiral fitter fits a global spiral, so it needs to know where the centre of
the scroll is and roughly where the sheets go:

- **An umbilicus** — the X/Y centre of the scroll at a series of Z positions.
- **Tracks** — curves extracted from surface predictions.
- **Lasagna normal volumes** — the orientation field, as OME-Zarr.

> **Trap** The umbilicus is **not in any bucket**. It is human-annotated work —
> ours came from Aleksei Drobkov and bruniss — so it arrives as an artifact
> somebody uploads, not something a job downloads. A first run that assumes it
> will be fetched fails at the input survey, which is the cheapest place it
> could fail.

Tracks and the lasagna volumes are fetchable, and `stage_spiral_dataset.py`
fetches them: one lasagna level, into a cache shared per scroll rather than per
worker, and a missing chunk is counted separately from a network failure so
"the data is not there" and "the network blinked" do not look alike.

> **Trap** That is a **CLI step you run once per scroll**, not something a job
> does, and there is no panel route for it. A P1 job pointed at a directory with
> no `SPIRAL_DATASET.json` refuses at the input survey and says to stage it.

## Queue the fit

Queue P1 from its own phase panel. Two things refuse the job before any field
is read:

- **the profile.** `spiral-fitter-v1@0.3.0` is the only one this lane accepts;
  another, or none, is refused.
- **the worker.** This lane declares the `helena-villa-python` image. An
  ordinary GPU worker will not claim it — the row sits `pending`, looking
  exactly like a stuck queue.

The fields that decide the run:

| Field | What it is | Advice |
| --- | --- | --- |
| `scroll_name` | what the fitter calls this scroll internally | need not match the catalogue name; both are recorded |
| `dataset_path` | where the staged inputs live | staged for you; name the scroll's dataset root |
| `z_begin`, `z_end` | the slice range to fit | **start small.** 10000 to 11000 is a thousand slices |
| `voxel_size_um` | µm per voxel **of the staged dataset** | may be a pooled copy rather than P0's native scale |
| `spiral_outward_sense` | which way the spiral winds | a property of the scroll |
| `random_seed` | the optimizer's seed | set it if you intend to compare two fits |
| `normal_zarr_group` | which pyramid level of the normal volumes to read | one level; three is 57× the data |
| `lasagna_scale` | the **divisor** the shape check uses | left empty it is `2 ** normal_zarr_group` |

The profile disables patches, and that alone is this lane's override — the
other 104 upstream keys stay at their defaults, frozen for comparability. The
overrides come from the profile, not the job; `random_seed` is the only setting
a request can move.

> **Trap** The point collections are **optional** and their absence is not a
> refusal — one line to stderr and the fit continues. Without `abs_winding.json`
> the windings are **relative**: you get surfaces, and not which turn of the
> scroll each one is.

> **Cost** Fit the small z-range first and look at it. A whole scroll is hours,
> and every input mistake costs the whole of it.

## Then certify it

A fit produces one TIFXYZ per winding, and each becomes a surface. Certify them
with **certify** under Maintenance on the P1 panel, or with
`POST /api/geometry/certify`; the P2 panel shows the verdicts and has no
button of its own.

> **Trap** P2 is not one of the phases `POST /api/jobs` queues. The queueable
> set is P1, P4, P5, P7, P8 and P9; anything else gets a 400 saying the phase
> has no runner registered. P3 is the same, and goes through the Flattening
> page.

This is not a formality — it is the step that tells you whether the fit stayed
on one sheet.

What P2 can tell you that looking cannot:

- **Lamina switches** — the surface crossing from one sheet to the next, which
  is the failure the walkthrough warns about.
- **Bridge** — the sheet doubled onto itself: two parts of the mesh nearly
  coincident and parallel, within about 28 µm.
- **Fold-back area** — a certified surface may carry folded cells. Only a mesh
  folded almost everywhere, above 5%, is rejected, and that ceiling is
  uncalibrated.
- **Distortion and coverage** — self-intersection, and too little of the grid
  carrying data.

A surface that fails gets a named rejection, and [P3](#/docs/phases/p3) will not
flatten it. That refusal is the feature: flattening a sheet that crosses two
laminae produces a beautiful image of two different pieces of papyrus.

A third outcome is easy to miss: `GEOMETRY_UNMEASURED`, the **default state of
every surface** and the fail-closed verdict when a detector could not run. It is
not a pass.

> **Trap** Certification is not a claim that the fit followed the right lamina.
> P2 detects what a lamina switch produces *at the grid step of the artifact it
> measured*; on a coarse grid it cannot, and records `resolution_limited`.

## If the fit will not behave

The walkthrough's fallback is a local patch — *Create Segment (GrowPatch)* in
VC3D — rather than a global fit. That is a reasonable answer when the scroll's
geometry defeats the spiral in a region, and Helena will certify and flatten a
patch-derived surface the same way.

Next: [flatten and render](#/docs/tutorial/flatten-and-render).
