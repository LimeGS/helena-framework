---
title: Screening, Coverage and Compare
summary: Reading results: one region, a whole scroll, or two runs against each other.
---

## Screening

Adjudication, not viewing. All seven thresholds of the strict screen are **on
the controls** rather than compiled in, because watching what moves when they
move is the only way to tell a robust verdict from one balanced on a number.

If a verdict flips when you nudge a threshold by a few percent, that is the
finding — not the verdict.

The verdict is `PASSES_STRICT_SCREEN` or `DOES_NOT_PASS`, and the response
carries its own non-claims. The first: **passing the strict screen is not a
reading, and does not accept ink, text or letters.**

> **Note** The page starts at a threshold of 0.55 where the endpoint's own
> default is 0.5. Whichever you use, it is a choice — do not read the page's
> starting values as "the frozen ones".

Only runs that have a probability map are listed. The map card previews the
selected run at the chosen threshold, next to the table it is explaining.

See [P7](#/docs/phases/p7) for what a vetting card records.

## Coverage

How much of a scroll has been looked at, and with what result.

The framework is named for exploration and originally could not answer this:
coverage existed only as a ranking input inside the bootstrap — how far a
candidate cell is from the surfaces already grown. As a question a person asks,
it did not exist, and progress used to be read off a surface count instead,
which rises whether the fleet is finding new ground or re-treading old.

It is computed against **the mission's scrolls**, so a scroll you did not add
shows nowhere. That is the most common reason coverage looks empty when work has
clearly happened — and the candidate-availability preflight below appears only
when a mission **and** a scroll are both selected, which is the second.

### The states that stop a campaign

| Field | Values |
| --- | --- |
| campaign decision | `CONTINUE`, `PAUSE_CANDIDATE_STARVATION`, `CONTROL_INCOMPLETE` |
| evidence status | `COMPLETE`, `INCOMPLETE`, `IN_PROGRESS` |
| preflight evidence | `CURRENT`, `STALE`, `INVALID` |
| measurement kind | `CENSUS`, `ESTIMATE`, `INCOMPLETE_CENSUS`, `INCOMPLETE_ESTIMATE` |

The campaign decision card names how many scientific-terminal attempts
recorded zero raw M7 against how many were evaluated, which attempts triggered
the decision, and which were excluded as platform or control outcomes, with a
receipt hash for the evaluation; earlier evaluations stay visible below it as
history. A decision of `PAUSE_CANDIDATE_STARVATION` is the campaign telling you
it has run out of places worth looking, and it still allows creating a
materially changed strategy or closing the campaign — not queuing past it.

### Campaign gates

A mission bound to a First Letters campaign carries a **First Letters campaign
gates** card above everything else on this page; an ordinary mission draws
nothing here, because it has no campaign to gate. It names the deployed
revision, whether the mission was bound to a different one, and the positive
control's status — a full-pipeline run against a known fixture, tracked live
through the run's nine boundaries, `P0`, `P1`, `P2`, `QC`, `P3`, `P4`, `P5`,
`P7`, `HUMAN_REVIEW`, with a toggle for the full event log. It lists, per
scroll, the candidate preflight and the computed task budget, names every
blocker with the evidence that clears it, and every advisory worth knowing.

Below that, only the actions the server's evidence actually offers are drawn as
buttons — a future action code the page has not been taught renders as plain
text, never a button, so no new server string can become a way around a gate.
There is deliberately no control that accepts a blocked campaign, forces a
queue, or turns a stale control into a current one: the way past a blocker is
to produce the evidence it names. Accepting a computed budget authorizes
compute under a named cap; it does not choose the task count. Closing a
campaign asks for confirmation, archives the mission, and deletes nothing —
its receipts stay readable.

### Candidate availability preflight

A source-locked survey of what the current source exposes, ahead of any fleet
attempt — this can show a populated funnel while the grid table below it is
still empty, because "is there anything worth attempting here" is a question
that comes before "has anything been attempted". `INVALID` evidence shows only
the reason and no numbers: a preflight whose receipt does not verify is not
read from.

Otherwise it states whether the measurement is an exact census or a sampled
estimate, and whether either is incomplete, alongside the planned and achieved
sampling percentage, and a funnel: raw M7 candidates, then post-CT,
post-cell-clearance, post-volume-clearance, and packet-retained, next to cells
surveyed successfully out of cells attempted, source failures, and the eligible
cell population, exact or estimated. A table breaks the same funnel down by
spatial bin. Its own non-claim: candidate scarcity is not evidence of surface,
ink, text or letter absence.

### What has been looked at

Three tiles: cells attempted across every grid, cells that produced a surface,
and the mission's surface area in cm² — stated as an upper bound, because
overlap below the deduplication threshold is counted twice.

The per-grid table underneath is the one worth reading:

| Column | Means |
| --- | --- |
| grid | the grid version; cells under two different grid versions are not the same cells |
| step | the grid's cell spacing |
| attempted | cells this grid has attempted |
| of the volume | cells inside the scroll's volume, and the percentage attempted |
| with a surface | cells that produced a surface |
| no seed | cells that produced nothing to grow from |
| hit rate | with a surface ÷ attempted |

The **hit rate** is the column worth reading: on this control plane one grid
found a lamina in 30 of 30 cells and another in 1 of 128, and nothing had ever
put those two numbers beside each other. It is drawn green at 50% or better,
red below 10%, and it is not a quality signal — it says the planner found a
seed worth growing in that cell, and nothing about what grew there.

### Re-asking cells that gave no seed

A cell that ended `NO_SEED` recorded how many candidates the provider offered
and which screen removed them. This form re-queues a chosen set of those
causes under a new grid and policy version, through a different planner if one
is picked. Grid and policy version both default to a field pre-filled with
today's date, because a task's identity is (snapshot, grid, cell, policy): a
re-ask under the same policy as before inserts nothing and silently looks like
it worked. A dry run lists what would be queued without queuing it.

Re-asking is not evidence a lamina is there, and it changes neither the
prediction volume nor its threshold: a cell that failed for want of any raw M7
candidate fails the same way again unless what the source offers changes.

## Compare

Two ink runs, side by side. The fields worth checking come first: the contract
keys — schema, sample, lane, checkpoint digest, normalisation, clip value,
divisor — because a difference there is what makes any further comparison
meaningless. Every other statistic each run carries follows, sorted, with a
count of how many fields differ between the two. A shared threshold slider
drives both probability-map previews on one pan and zoom, so panning either one
compares the same place in both.

## Seed agreement, on the Segmentation panel

Not on Compare above — that section compares two ink runs, not two seeds.
Seed agreement is a column on the P1 panel: a state pill and **one** number,
the normal median. The decomposition, the percentiles and what the
normalisation divided by are in the surface's own detail, deliberately: four
scannable columns beside a cell holding a decomposition is a table nobody can
sweep.

What the measurement actually produces is more than three numbers — the normal
median, p90 and p99 and its value in sheet thicknesses; the lateral median and
p90; the total in voxels and microns with its convention; and all of it **per z
band**. The p90 must be read beside the median: the median's tail is not
measured by the median.

| State | Means |
| --- | --- |
| `SEED_AGREEMENT_MEASURED` | a pair was measured. Drawn **neutral** on purpose — a measured pair is a measurement, not a pass |
| `SEED_UNPAIRED` | one seed, so no error bar. Not the same as a small one, and cannot be defended |
| `SEED_AGREEMENT_UNMEASURED` | no cell had a normal; only a total exists |
| `SEED_OVERRIDE_DID_NOT_TAKE` | the one red pill |

> **Trap** An agreement below **0.1 voxels** is **refused**, not reported — the
> measurement raises rather than returning a number. Real pairs on this campaign
> sit at 9 to 17 voxels, so that close means the seed never reached the
> optimizer. It surfaces as `SEED_OVERRIDE_DID_NOT_TAKE`: the one metric here
> whose failure looks like its best possible result.

The unit matters and two figures nearly collide: the old total over the winding
pitch is 121/371 = 0.326, and the normal over the sheet thickness is 12/35.5 =
0.338. They are different quantities that round alike to one decimal, which is
why the unit is always stated.
