---
title: Certification, Lineage and Audit
summary: Three pages that answer questions after the fact rather than doing the work.
---

## Certification

The P2 panel, under Mission. It has **surfaces and verdicts on them**, not tasks
and workers.

> **Note** Certification runs **automatically** when the fleet finalises a
> surface. The button here is for catching up on surfaces nobody measured — it
> is not how certification normally happens.

The first block counts the silence: how many surfaces carry no verdict at all.
That number is the one to watch, because certification is **fail-soft** — an
unmeasured surface is not a rejected one, and it will sit waiting rather than
announcing itself.

It runs against whatever scroll is selected in **P0**, and the button is
disabled until one is — there is no scroll picker here. Dry run defaults **on**,
and the only other control is a limit. See [P2](#/docs/phases/p2) for what the
verdicts mean.

The gate is **fail-soft in control flow and fail-closed in verdict**: a gate
that cannot load records `unmeasured`, which is not certification, rather than
discarding a segmentation that took hours. Dropping the second half of that
sentence inverts what it guarantees.

> **Trap** There is a second table here, collapsed, for the **physical** axis. A
> surface can be `CT_SUPPORTED` and rejected for bridging at the same time,
> because they answer different questions. The geometry verdict is not the
> verdict.

## Lineage

The audit trail, plus one thing that is not one: a **backfill** that registers
artifacts the record is missing — dry run first, then a button that writes.

It is phase-scoped, with a P0…P9 selector, and cannot be narrowed to a scroll.
Its three cards are what a phase will read, the selection history, and what that
phase produced. The two questions it exists for:

- **what did that read?**
- **when did the answer change?**

The lineage itself is built from the surface row, never from a document a
caller supplied — a surface that arrived describing its own place in the chain
would be answering its own audit, and that is refused in both modes.

> **Trap** One escape hatch does reach it: `allow_unvalidated` is a
> caller-supplied P2 parameter, threaded straight through — "include surfaces
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
and the files are monthly `.jsonl` under the audit root.

> **Note** Machine tokens appear under their own name — `machine:gpu-1-segment`
> — rather than under whichever person's password would otherwise have been
> copied onto a worker host. That is most of the point of having them.
