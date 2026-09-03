---
title: Certification, Lineage and Audit
summary: Three pages that answer questions after the fact rather than doing the work.
---

## Certification

The P2 panel, under Mission. It has **surfaces and verdicts on them**, not tasks
and workers.

> **Note** Certification runs **automatically** when the fleet finalises a
> surface. Catching up on surfaces nobody measured is a maintenance action, not
> a control on this page — it is not how certification normally happens.

The first block counts the silence: how many surfaces carry no verdict at all.
That number is the one to watch, because certification is **fail-soft** — an
unmeasured surface is not a rejected one, and it will sit waiting rather than
announcing itself.

The two verdict tables have no controls: they read. To certify what is
unmeasured, use **certify** under Maintenance on the P1 panel, which asks you
to confirm and runs against the scroll selected in **P0** — inside a mission
it refuses until one is. A script uses `POST /api/geometry/certify`, which
takes `limit` (25 by default), `dry_run` and `surface_id`. See
[P2](#/docs/phases/p2) for what the verdicts mean.

The gate is **fail-soft in control flow and fail-closed in verdict**: a gate
that cannot load records `unmeasured`, which is not certification, rather than
discarding a segmentation that took hours. Dropping the second half of that
sentence inverts what it guarantees.

> **Trap** There is a second table here, collapsed, for the **physical** axis. A
> surface can be `CT_SUPPORTED` and rejected for bridging at the same time,
> because they answer different questions. The geometry verdict is not the
> verdict.

Below both tables is a third card that does have controls: **score against a
reference strip**. A strip records consecutive papyrus wraps as separate
labelled point sets, taken from a segment's own geometry where it spirals, so
it needs no annotation — mint one with `reference-strips/make_strip.py`, or
upload a `strip-v0` `.npz` here. It is **not ground truth**: a strip is derived
from a segmentation, so it cannot judge the segment it came from, and only its
optional CT cross-check — off by default, the one check that reaches the
network — appeals to anything outside that segmentation. Upload, then
**qualify** against four checks (a plumbing self-test, one that catches
mislabelled or shuffled wraps, a null baseline, and the CT cross-check) before
anything is scored against it. Scoring needs a surface's path handed in as
`predPath`; nothing in the panel supplies one yet, so from this page you can
upload and qualify, not score. It is the independent check regardless: the
gate above is the fleet grading its own output, and a strip is a reference the
grower did not write.

## Lineage

The record of what each phase produced, what it may consume, and which version
a mission currently uses — plus one thing that is not part of that record: a
**backfill** that registers artifacts made before the register existed. Dry
run first, then a button that writes.

It is phase-scoped, with a P0…P9 selector, and cannot be narrowed to a scroll.
Its three cards are what a phase will read, the selection history, and what that
phase produced. None of the three is read-only: **use this**, on a row of
either register, moves the mission's selection to that version; **affects**
traces what was already computed downstream of one, since replacing an input
does not erase the answers computed from it. **go back to this**, on the
history, does not rewind — it writes a new version equal to the old one, so a
mission that went forward, found a mistake and came back reads as three
decisions, not one that never happened. The two questions it exists for:

- **what did that read?**
- **when did the answer change?**

The lineage itself is built from the surface row, never from a document a
caller supplied — a surface that arrived describing its own place in the chain
would be answering its own audit, and that is refused in both modes.

> **Trap** One escape hatch does reach it: `allow_unvalidated` is a
> caller-supplied P3 parameter, threaded straight through — "include surfaces
> the CT never confirmed". A certification run refuses it; an exploration run
> honours it.

## Audit

Everything that changed something, and who changed it. **Reads are not here** —
a trail that records every page load is a trail nobody searches, and this
platform's questions are all about mutations: who queued that render, who
removed that scroll from the mission, who restored that configuration.

**Refusals are in the trail on purpose.** "Nobody did that" and "somebody tried
and was told no" are different answers, and 401s and 403s are drawn in red. That
is the most useful thing about it.

Only `POST`, `PUT`, `PATCH` and `DELETE` under `/api/` are recorded — a mutation
exposed as a GET would not be in the trail. It filters by user and by substring,
a **showing** control caps how far back a query looks (200 entries by default,
up to 2000), and the files are monthly `.jsonl` under the audit root.

Request bodies are never captured, on purpose: one route sets S3 credentials
and another sets a password, and a trail that recorded what was sent would be
the most sensitive file on the machine. What each row keeps is the timestamp,
an id, the user, the action, the outcome, how long it took, and the client
address.

> **Note** Machine tokens appear under their own name — `machine:<host>-segment`
> — rather than under whichever person's password would otherwise have been
> copied onto a worker host. That is most of the point of having them.
