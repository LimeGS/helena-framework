---
title: Queueing work
summary: Where each phase is queued from, what the form will not let you do, why a lane is missing from the list, and what a queued job looks like afterwards.
---

> **Trap** There is no Command page. Queueing lives inside the phase that does
> the work: the P5 launcher is on the P5 panel, and `/command` redirects to
> Mission. Two phases are not queueable at all from the panel's job route.

## Which phases queue where

| Phase | How |
| --- | --- |
| P1, P4, P5, P7, P8, P9 | `POST /api/jobs`, from that phase's panel |
| **P2** | in the panel, **certify** under Maintenance on P1 — the P2 panel only reads |
| **P3** | `POST /api/flattening/run`, from the Flattening panel |

Anything outside the first row gets a 400 saying the phase has no runner
registered. That is the single most likely 400 a new reader meets.

> **Trap** `POST /api/geometry/certify` does queue a real P2 job, the same way
> `POST /api/jobs` does for the other phases — but no button in the panel calls
> it. The Maintenance **certify** action instead posts to `POST
> /api/segmentation/maintenance`, which runs the fleet's own certify command
> synchronously inside the request and hands back its receipt directly: no job
> id, no row in the Queue table, none of `pending`, `leased` or `running`.
> `/api/geometry/certify` is reachable only by calling the API directly.

## Mission and scroll scoping

Every route on this page needs a mission: `POST /api/jobs`, `POST
/api/geometry/certify` and `POST /api/flattening/run` all refuse a request
with no `mission_id`, one addressed to `unfiled` (a read-only view of
pre-mission receipts), or one whose mission has no scrolls yet.

With a mission selected, the form's Scroll field is filled from the phase
rail's current subject (`?mission=` and the subject in the URL) and disabled —
you cannot queue against a different scroll from this page. Typing a scroll by
hand only works with no mission selected, which is also the one case a submit
is guaranteed to be refused for missing a mission.

`GET /api/phases/{phase}/parameters` takes the same `mission` and `sample`
query as the phase page itself, and uses them for one thing: a field that
names another job — `screening_of`, `ordering_of` — is offered only that
mission and scroll's succeeded jobs, not every job on the deployment.

## What the form is doing for you

- **Only fields that phase accepts.** The queue's allowlist is the authority and
  refuses anything outside it — `unknown parameters for P5`. The launcher's own
  field list was written to match it by hand, so a field the queue gained is not
  automatically on the form: `model_config`, needed for a checkpoint whose
  architecture is not the timesformer lane's frozen default, has no field in
  the P5 launcher at all.
- **The lane list is live.** It comes from `/api/ink/lanes`, which decides
  routability by resolving each profile's adapter. A lane that is registered but
  unroutable, or disqualified in the registry, is simply not offered — with a
  reason in the payload the form does not show.
- **Deployment-owned fields are hidden.** Where a job publishes, and its seven
  control-binding fields, are the server's; a request that sets one is refused
  with `these are the server's to decide, not the request's`.

> **Trap** The other half of that: with the store setting **empty**, the panel
> refuses to queue at all. `CX_RENDER_STORE` for P4 (unless you pass
> `allow_local_layers`), `CX_INK_STORE` for P5, `CX_FLATTEN_STORE` for P3,
> `CX_RECONSTRUCTION_STORE` for P8, and `ARTIFACT_ROOT` (or
> `HELENA_SEGMENT_ARTIFACTS`) for P1. The four `CX_*` stores default to a path
> under `/artifacts`, the platform's own volume, and the platform compose sets
> `ARTIFACT_ROOT` to `/artifacts` directly — so this only happens to a
> deployment that blanked one. The symptom is "nothing will queue" and the
> cause is a blank field on the Configuration tab.

## Exactly-one-of, in four places

| Phase | Choose one |
| --- | --- |
| P4 | `segmentation` or `flattened_surface` |
| P5 | `layer_stack`, `tiff_dir` or `surface_volume` |
| P7 | `map_path` or `screening_of` |
| P9 | `order_path` or `ordering_of` |

Two of them demand exact upstream identity as well: `flattened_surface` needs
`flattening_id`, `p3_job_id` and a 64-hex artifact digest; `screening_of` needs
both of the P5 map's digests.

## Controls that are easy to miss

- **`requested_host`** — pin the job to one machine. Disabled hosts are greyed.
- **`priority`** — the claim orders by priority, then age.
- **`device`** — `auto`, `cpu` or `cuda*`; anything else is refused.
- **`on_degenerate`** — `fail` or `warn`. With `warn` a map that carries no
  decision **succeeds**, and P7 refuses it later.
- **`source_slice_um`** — the timesformer lane refuses without it, because the
  campaign spans 8.64 and 9.362 µm acquisitions.

On the P5 launcher, a dead Queue button says what is still needed (`needs a
scroll, a checkpoint, …`). The generic form behind P4, P7, P8 and P9 is not as
generous: it names an unmet exactly-one-of pair, but a plain required field
left blank, or no scroll chosen at all, just leaves the button disabled with
nothing beside it.

## After you queue

A successful submit returns a `job_id`; the panel shows it and the new row
appears in the Queue table on the next poll. A job's state moves `pending` →
`leased` (a worker holds the lease) → `running`, then `succeeded` or `failed`.
`cancelled` is a fourth resting state — the panel's cancel button only appears
while a job is still `pending`; once a worker has leased it, cancelling from
here is no longer offered. The Run tab's badge counts a phase's jobs in
`pending`, `leased` or `running`, so it reads nothing once every job at that
phase has finished.

## Paths, and when they are checked

Prefer naming an **id** over a path. `layer_stack` as a P4 job id resolves to the
published artifact and its digest; `tiff_dir` as a path is a claim about a
filesystem the worker may not share with you.

> **Trap** The path check runs on the worker, **after** it has taken the lease —
> so a bad path costs one attempt and the job reads `failed` with the reason in
> it. It names which of two: missing (a mount) or unreadable (a permission). It
> covers `tiff_dir`, `checkpoint`, `ppm`, `order_path`, `segmentation` and
> `model_config`, plus `volume` when there is no `remote_url`. `dataset_path`
> is deliberately excluded — P1 stages into it.

The only thing that genuinely filters before a claim is the runtime image.

## Dry run

`dry_run` is a parameter of **P1, P2 and P3 only**. The Flattening panel draws
it as a toggle that defaults **on**. P1 takes it as a plain request field on
`POST /api/jobs`, with no toggle in the panel. P2 takes it the same way on
`POST /api/geometry/certify` — but see the certify trap under **Which phases
queue where**: nothing in the panel calls that route, so a P2 dry run is only
reachable by calling the API directly; the panel's own **certify** action has
no `dry_run` of its own. On a scroll you have not run before, a dry run
through the panel — P1 or P3 — is the cheapest thing you can do.

## Ceilings

Every numeric parameter has a floor, and most have a ceiling. A size nothing
rejects is a lease a worker takes before it finds out.
