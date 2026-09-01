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

Only runs that have a probability map are listed.

See [P7](#/docs/phases/p7) for what a vetting card records.

## Coverage

How much of a scroll has been looked at, and with what result.

The framework is named for exploration and originally could not answer this:
coverage existed only as a ranking input inside the bootstrap — how far a
candidate cell is from the surfaces already grown. As a question a person asks,
it did not exist.

It is computed against **the mission's scrolls**, so a scroll you did not add
shows nowhere. That is the most common reason coverage looks empty when work has
clearly happened — and the preflight block appears only when a mission **and** a
scroll are both selected, which is the second.

The column worth reading is the **hit rate**. On this control plane one grid
found a lamina in 30 of 30 cells and another in 1 of 128, and nothing had ever
put those two numbers beside each other.

### The states that stop a campaign

| Field | Values |
| --- | --- |
| campaign decision | `CONTINUE`, `PAUSE_CANDIDATE_STARVATION`, `CONTROL_INCOMPLETE` |
| evidence status | `COMPLETE`, `INCOMPLETE`, `IN_PROGRESS` |
| preflight evidence | `CURRENT`, `STALE`, `INVALID` |
| measurement kind | `CENSUS`, `ESTIMATE`, `INCOMPLETE_CENSUS`, `INCOMPLETE_ESTIMATE` |

A decision of `PAUSE_CANDIDATE_STARVATION` is the campaign telling you it has
run out of places worth looking, and it comes with the actions it will still
allow.

## Seed agreement, on the Segmentation panel

Not on Compare — that page compares two ink runs. Seed agreement is a column on
the P1 panel: a state pill and **one** number, the normal median. The
decomposition, the percentiles and what the normalisation divided by are in the
surface's own detail, deliberately: four scannable columns beside a cell holding
a decomposition is a table nobody can sweep.

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
