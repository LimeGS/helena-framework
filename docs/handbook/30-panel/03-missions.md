---
title: Missions, scrolls and coverage
summary: Scoping work so it can be found, counted and attributed, and what the mission dashboard shows once you have.
---

## Missions

A mission names the scrolls being attempted. Nothing may be queued outside one,
and that rule lives in the **store**, not the panel — a script holding the DSN
hits it too.

The create form has two fields: a name and an id; scrolls are chosen in P0.
There is **no mode control**. A scroll that is in neither the eligible
catalogue nor the sources registered on this control plane is refused — at
creation through the API and at every amendment — with the names the
deployment does know, because a mission holding it could never grow anything
and would only say so at P1. Both spellings of a scroll are accepted:
`PHerc0826` as the bucket lists it and `PHerc826` as the catalogue keys it.

> **Trap** The certified/exploratory distinction is not a checkbox here, and it
> is not one thing. Three separate mechanisms answer "is this controlled":
> `campaign_kind` on the manifest (settable through the API only, never the UI);
> a **campaign budget admission** row per mission *and sample*, which is what
> the ink queue actually asks about; and `controlled_first_letters` on the
> surface row, which is what the surface boundaries ask. A mission with
> `campaign_kind` and no registered admission is gated as generic.

Separately, `execution_mode` is **per request**, not per mission. It defaults to
`CERTIFIED`, and a caller opts into `EXPLORATORY`, whose output carries a
non-claim and may not be read as evidence by a certified run.

> **Note** `unfiled` is a read-only view assembled from the receipts of runs
> that predate missions. It describes what happened; it is not a scope you can
> queue into.

### The selection freezes

As soon as any job exists for the mission, its scroll selection is marked
frozen. After that:

- **widening** it needs an explicit amend, with a **reason**, which is recorded;
- removing a scroll is its own route.

The form says so itself: every change to that selection is recorded with a
reason.

### Picking one

Nothing else in the panel renders until a mission is chosen. What greets you
first is a table of every mission: name and description, id, how many scrolls
it holds, how many runs and how many recently queued jobs, when it was
created, and its state — `active`, `paused` or `archived`, drawn as a pill, or
**pre-existing** for the one implicit entry, `unfiled`, which you cannot edit.
**New mission** opens the two-field form above it; past four missions a filter
box searches name, id and scroll. Clicking a row opens that mission and
replaces the table with everything scoped to it.

While a mission is open, the sidebar carries a small card naming it, with its
scroll and run counts underneath, and a **Change or create** button. That
button does not open a dropdown: there can be hundreds of missions, so it
clears the selection and returns to the table above instead.

The same sidebar carries a scroll selector for the mission that is open,
labelling each scroll with the furthest phase it has reached: `P0` with
nothing yet, `P1` once P1 has grown a surface for it (`P2` once one is
certified), `P5` once it has an ink run, `P7` once one of those runs has
produced a probability map. It reads "no scroll in this mission has anything
yet" until one does.

## The mission dashboard

Opening a mission shows four tiles, then two tables.

| Tile | What it shows |
|---|---|
| **Fleet hardware** | gpus, cpu cores and ram summed across enabled hosts (a disabled host is not counted), and how many of them last reported in |
| **Fleet** | this mission's surface count, the tasks that produced them, and stale leases — or the reason nothing is available |
| **Runs** | this mission's run count; underneath it, the number of ink lanes with a declared profile, which is a deployment-wide figure, not scoped to the mission |
| **Integrity** | how many receipts contradict their own declared contract, or that every receipt matches its contract |

**Scrolls in this mission** lists every scroll the mission has selected, with
scale and energy read from the frozen catalog:

| Column | Meaning |
|---|---|
| Scroll | the sample id |
| µm | pixel size; a ▲ marks a scroll with a finer scan available |
| keV | beam energy, when known |
| Runs | ink runs indexed for this scroll |
| Last lane | despite the name, the lane of the run with the **highest p90** for this scroll, not the most recent one — it links to that run |
| p90 | that run's p90 statistic |
| State | **Screened** once this scroll has any indexed ink run at all — not the pass/fail verdict from the strict screen, only whether ink has run |

Beside it, **Surfaces by scroll** repeats the fleet's per-scroll surface count
and area for this mission's scrolls, and **Findings** — shown only when there
are any — lists what Integrity found: normalization and clip contradictions in
a run's declared contract, and probability maps a liveness check did not read
as alive, each linking to its run.

## Scrolls

The catalogue of what this deployment can reach, with each scan's declared
scale. Adding a scroll to a mission is what makes coverage countable.

> **Note** The catalogue is not the only source. At least one scroll is absent
> from it on purpose and has its addresses in the control manifest instead — so
> a check against the catalogue alone once answered "no volume" for a scroll the
> platform could name. A scroll in neither is one you cannot start from.

## Coverage

Coverage answers "how much of this scroll has been looked at, and with what
result" — the candidate cells P1's bootstrap search has attempted, and the hit
rate among them. It is computed against the mission's scrolls, which is why a
scroll outside the mission shows nowhere. See
[Screening, Coverage and Compare](#/docs/panel/screening-and-coverage) for the
fields.

## Compare

Two **ink runs**, side by side: their contract keys — schema, sample, lane,
checkpoint digest, normalisation, clip value, divisor — and their statistics,
with two map viewers on a shared pan and zoom.

> **Trap** Seed agreement is **not** here. It is on the P1 Segmentation panel,
> as a state pill and one number per surface. This page has no distance, no
> decomposition and no seed in it at all.

## The phase rail

Each scroll carries a rail: one mark per phase, in pipeline order, saying where
that phase stands for that scroll. It is the densest thing on the page, and its
ten states are the whole alphabet — a mark you cannot read is a row you cannot
act on.

| Mark | Means |
|---|---|
| **running** | running now on a worker |
| **queued** | queued — no worker has claimed it yet |
| **failed** | the last attempt failed |
| **stopped** | the last attempt was cancelled |
| **done** | has produced something here |
| **ready** | prerequisites met — ready to run |
| **blocked** | prerequisites not met |
| **waiting** | prerequisites not met — nothing upstream yet |
| **elsewhere** | run somewhere other than this deployment |
| **no-run** | nothing to run — a committed artefact, or a check inside another phase |

Three of these are easy to confuse and mean different things. **blocked** and
**waiting** both say the prerequisites are not met, but waiting adds that
nothing upstream has been attempted yet, so there is nothing to chase. And
**no-run** is not a failure to run: it is a phase with nothing to do for this
scroll, which is why it is drawn differently from **done**.

> **Trap** **elsewhere** is the one that misleads. It means the work exists but
> was not produced here, so this deployment cannot show you its receipt — not
> that it is missing.
