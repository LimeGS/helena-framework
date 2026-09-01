---
title: Missions, scrolls and coverage
summary: Scoping work so it can be found, counted and attributed.
---

## Missions

A mission names the scrolls being attempted. Nothing may be queued outside one,
and that rule lives in the **store**, not the panel — a script holding the DSN
hits it too.

The create form has two fields: a name and an id. There is **no mode control**.

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

## Scrolls

The catalogue of what this deployment can reach, with each scan's declared
scale. Adding a scroll to a mission is what makes coverage countable.

> **Note** The catalogue is not the only source. At least one scroll is absent
> from it on purpose and has its addresses in the control manifest instead — so
> a check against the catalogue alone once answered "no volume" for a scroll the
> platform could name. A scroll in neither is one you cannot start from.

## Coverage

Coverage answers "how much of this scroll has been through which phase". It is
computed against the mission's scrolls, which is why a scroll outside the
mission shows nowhere.

## Compare

Two **ink runs**, side by side: their contract keys — schema, sample, lane,
checkpoint digest, normalisation, clip value, divisor — and their statistics,
with two map viewers on a shared pan and zoom.

> **Trap** Seed agreement is **not** here. It is on the P1 Segmentation panel,
> as a state pill and one number per surface. This page has no distance, no
> decomposition and no seed in it at all.
