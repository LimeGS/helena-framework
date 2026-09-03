---
title: Vocabulary
summary: The nouns this platform uses, and the distinctions each one is protecting.
---

Most of these words are ordinary English doing a specific job. Where a term
looks like a synonym for another, it is not, and the difference is the point.

## Panel

**Phase rail** — the sidebar on every panel screen: one row per phase, P0
through P9, coloured by whether it is running, queued, blocked or done for the
mission and subject currently selected. Its ten rows draw on first paint; the
live status lands once the summary answers.

**Tile** — a titled summary card on the Mission page, such as fleet hardware
or run counts, holding one value and a tone (steady, busy, warn or alert) that
says how much attention it needs.

## Work

**Mission** — the project a piece of work belongs to. It names the scrolls being
attempted and whether the run is a
[certification or exploration run](#/docs/start/what-helena-is). Nothing may be queued
outside one: coverage, backlogs and counts are all mission-scoped, so a job
without one is work that cannot be found, reviewed or attributed. `unfiled` is a
read-only view of runs that predate missions — it is history, not a scope you
can queue into.

**Phase** — a step in the chain, P0 through P9. See
[the pipeline](#/docs/start/pipeline).

**Lane** — one way of doing a phase. P4 renders through `vc-render-tifxyz` or
`chunk-gather`; P8 merges, builds a column atlas, or computes mesh
relations. Naming a lane is choosing a method, and an unknown name is refused
rather than silently defaulted — a job that asked for one detector and got
another is a result nobody can interpret.

**Profile** — a frozen declaration of how a lane runs, with its own hash, so
"which settings produced this" has an answer that cannot drift. What it contains
depends on the stage: an **ink** profile pins the adapter, the checkpoint digest,
the training scale and the model family; a **segmentation** profile pins the
upstream module constants, the runtime image and the dataset layout, and carries
no checkpoint at all.

**Job** — one queued unit of work: a phase, a lane, a profile, a sample, and
parameters. A job becomes a process exactly one way, through the queue's own
command builder, which is why a parameter that does not exist in the contract is
refused instead of passed through.

**Attempt** — one try at a job. A job with three attempts has been leased three
times; the lease expiring without a result is itself recorded.

## Things that get made

**Surface** — a piece of papyrus recovered as geometry: a mesh in TIFXYZ form.
It has a `surface_id`, a sample, an area, and four independent QC states.

**Origin** — whether a surface was grown by this pipeline or brought in from
outside it: `GROWN_HERE` for a surface P1 produced here, `IMPORTED` for
everything else, including a segment published by someone else's run. The
panel shows the same split as "grown here" and "imported".

**TIFXYZ** — the surface format: `x.tif`, `y.tif`, `z.tif` plus metadata, where
each pixel of the grid carries a coordinate in the volume. A surface is a
mapping from a flat sheet to where that sheet sits in the scroll.

**Layer stack** — what P4 renders: a numbered TIFF per depth, sampled through
the surface. `00.tif` to `32.tif` is a 33-slice stack.

**Surface volume** — the same idea as a Zarr array rather than numbered TIFFs.
Some detectors want one, some the other.

**Probability map** — what P5 produces: one float per pixel, brighter meaning
more likely ink. It is a screening output, not a reading.

**Liveness verdict** — whether a probability map carries a decision at all:
`ALIVE`, `DEGENERATE`, or `EMPTY`. `DEGENERATE` is not a statement about ink;
it means the output head is untrained, collapsed, or fed input far outside its
training distribution, so nothing downstream may screen the map. A P5 job
whose receipt carries no liveness verdict is failed.

**Artifact** — anything published to the store with a manifest. Artifacts are
addressed by key and verified by digest.

## States, and why there are several

A surface carries **four independent** QC axes, and collapsing them is the
mistake they exist to prevent.

| Axis | Asks | States |
| --- | --- | --- |
| Geometry | is the mesh coherent? | `GEOMETRY_UNMEASURED` (the default, and the fail-closed verdict), `GEOMETRY_CERTIFIED`, or one of four rejections: `GEOMETRY_REJECTED_BRIDGE`, `GEOMETRY_REJECTED_LAMINA_SWITCH`, `GEOMETRY_REJECTED_DISTORTION`, `GEOMETRY_REJECTED_COVERAGE` |
| Physical | does the CT support it? | `UNVALIDATED` (no verdict, not a failure), `CT_SUPPORTED` and `CT_SUPPORTED_REVIEW` — the two P3 and P4 accept — plus `CT_INSUFFICIENT`, `INK_SCREEN_INSUFFICIENT`, `NOT_APPLICABLE_FIXTURE` |
| Lamina | does the scan resolve one sheet? | `LAMINA_UNMEASURED` (the default), `LAMINA_SINGLE_SHEET`, `LAMINA_FUSED`, `LAMINA_TOO_THIN`, `LAMINA_UNRESOLVED`, `LAMINA_INSUFFICIENT_COLUMNS` |
| Seed agreement | do two seeds agree? | `SEED_UNPAIRED` (the default), `SEED_AGREEMENT_MEASURED`, `SEED_AGREEMENT_UNMEASURED`, `SEED_OVERRIDE_DID_NOT_TAKE` |

A surface can be geometrically perfect and physically unsupported. It can be
both and still sit on a scan too coarse to tell one lamina from the next. Each
axis has its own page under [the phases](#/docs/phases/p2).

> **Trap** `SEED_OVERRIDE_DID_NOT_TAKE` is the one worth knowing by name: the
> second seed never reached the optimizer, so the two fits are the same fit and
> the agreement is near zero — which reads as perfect reproducibility. It is the
> only metric here whose failure looks like its best possible result.

> **Note** `SEED_UNPAIRED` is a state, not missing data. It means no second seed
> was run, which is a fact about the experiment rather than a gap in the record.

## Provenance

**Source snapshot** — the frozen identity of a volume: where it came from, its
digest, its scale. Everything downstream binds to it.

**Lineage** — the chain of digests from a result back to its source snapshot.
A surface may **never** state its own: an imported payload carrying a lineage,
a control flag or a namespace claiming canonicity is refused in both modes. What
certification adds is that the resolved document must also be *complete* —
right schema, canonical, a recognised surface state, no ambiguity or hash
conflict.

**Receipt** — what a phase writes about its own run: inputs, outputs, digests,
lane, parameters, and the revision of the code that ran.

**Content lock** — the digest of the source files a run used, so "was this the
code we think it was" is answerable after the fact.
