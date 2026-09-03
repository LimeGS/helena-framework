---
title: The pipeline, end to end
summary: Ten phases, what each one consumes and produces, and where the gates are.
---

A scroll enters as a stack of CT slices and leaves as a picture of a surface
with a claim about ink on it. Every phase in between takes an artifact and
produces another, and records the digest of both.

## The chain

| Phase | Does | Takes | Produces |
| --- | --- | --- | --- |
| **P0** | Volume intake | a public OME-Zarr scan | a frozen catalogue entry: scan id, µm per voxel, beam energy |
| **P1** | Segmentation | an m7 volume (seeded grow), or an umbilicus, tracks and normals (spiral fit) | a TIFXYZ surface — one per winding from a fit, one from a grow |
| **P2** | Geometry certification | a TIFXYZ surface | certified, certified-but-`resolution_limited`, a named rejection, or `GEOMETRY_UNMEASURED` |
| **P3** | Flattening | a certified surface | a flattened sheet |
| **P4** | Surface volume rendering | a flattened sheet or the certified surface itself, and the CT volume | a numbered TIFF stack, optionally also a Zarr copy |
| **P5** | Ink detection | a layer stack and an ink lane profile | a probability map |
| **P6** | Liveness | a probability map | `ALIVE`, `DEGENERATE` or `EMPTY` — whether the map carries a decision at all |
| **P7** | Screening and adjudication | a probability map and a claimed bounding box | candidate shapes, qualifying rows, and a pass or fail against the strict screen |
| **P8** | Reconstruction | published meshes, or two or more certified surfaces (merge lane) | a relation graph, or a merged surface (merge lane) |
| **P9** | Rendering and reading | the published maps and a measured P8 radial order | ordered plates |

Not every run touches all ten. A screening pass is P4→P5→P7. A geometry
campaign is P1→P2→P8. The chain is what constrains the order, not a schedule
anybody has to follow.

Three phases run more than one way, and each one names its own lanes: P1's
seeded grow and spiral fit above, P8's column-atlas and merge lane above, and
P4's two renderers — the default `vc-render-tifxyz`, and a chunk-gather lane
built around a legacy PPM instead of a sheet. See the [phase page](#/docs/phases/p4)
for what each lane needs.

## The gates, and what they refuse

Three places in the chain say no rather than continuing with a worse answer.
They are the reason a Helena result means something, and the reason a run
sometimes stops.

- **P2 certifies geometry.** An ordered cascade runs against the mesh — lamina
  switches, bridges, self-intersection, fold-back area, coverage — and the
  first thing that fires names the verdict — self-intersection and an
  over-ceiling fold-back both name `DISTORTION`, and a check that could not run
  names `GEOMETRY_UNMEASURED` rather than certifying. A surface that fails gets
  a named rejection, not a score, and [P3 refuses it](#/docs/phases/p3) rather
  than flattening something that crosses two sheets.
- **Liveness asks whether a map decided anything.** P6 is not a job you queue:
  the P5 adapter assesses its own map and writes the verdict into the receipt.
  A detector handed an unusable input still produces one: a flat field at 0.51,
  or a constant. A P5 job whose receipt carries no verdict is failed rather than
  passed, because an unchecked map and a live one must not look the same.
- **The lineage gate** runs at every queue admission in **both** modes. Its
  question is not "is this good" but "can this platform say where this came
  from". In exploration it refuses discovery-namespace lineage and passes
  everything else through unchanged; in certification it also demands a complete
  canonical document. See
  [certification and exploration](#/docs/start/what-helena-is).

> **Trap** A gate that passes is not a result that is right. P2 certifies that
> the mesh is geometrically coherent, not that it is on the papyrus you meant.
> The phase pages say what each verdict does and does not claim.

## What travels between phases

Artifacts, not files on somebody's disk. A phase publishes to the artifact store
with a manifest — every file, its size and its sha256 — and the next phase
fetches by that digest. Three consequences worth knowing:

- A worker is disposable. Anything left only on the machine that made it is lost
  when that machine goes away, which is why P4 refuses to call a render a
  success if there is nowhere to publish it.
- A phase can run on a different host from the one before it. P5 routinely reads
  a layer stack another worker rendered.
- A result carries its inputs' digests, so "which render did this map come
  from" is answerable years later, from the row alone.

## Where the work runs

The panel writes a row; it never runs the work. A worker on a GPU host claims
the job, runs it in the image that lane requires, and reports.

Only six phases are queued this way: P1, P4, P5, P7, P8 and P9. The other four
are not jobs you queue from the panel. P0 has no command at all. P2 runs
automatically when the fleet finalizes a surface, or through its own `certify`
backfill command. P3 runs from its own `flatten` command. P6 is not run; it is
a verdict the P5 adapter writes into its own receipt.

- Lanes declare the **runtime image** they need. A worker running something else
  refuses the job by name rather than taking it and failing at the first import.
- A worker that cannot see its inputs says which of them, and whether the
  problem is the path or the permission. That check runs after the claim, so it
  costs one attempt — still far better than a renderer that says `Error loading`
  and exits zero.
- Jobs are leased. A worker that dies mid-job has its lease expire and the job
  returns to the queue with the attempt counted — until `max_attempts`, three by
  default, after which it is failed as `LEASE_EXHAUSTION` rather than recycled
  forever.

See [the fleet](#/docs/panel/fleet) for reading all of that from the panel, and
[operations](#/docs/operations/deploys) for the machinery underneath.
