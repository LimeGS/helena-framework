---
title: Queueing work
summary: Where each phase is queued from, what the form will not let you do, and why a lane is missing from the list.
---

> **Trap** There is no Command page. Queueing lives inside the phase that does
> the work: the P5 launcher is on the P5 panel, and `/command` redirects to
> Mission. Two phases are not queueable at all from the panel's job route.

## Which phases queue where

| Phase | How |
| --- | --- |
| P1, P4, P5, P7, P8, P9 | `POST /api/jobs`, from that phase's panel |
| **P2** | `POST /api/geometry/certify`, from the Geometry panel |
| **P3** | `POST /api/flattening/run`, from the Flattening panel |

Anything outside the first row gets a 400 saying the phase has no runner
registered. That is the single most likely 400 a new reader meets.

## What the form is doing for you

- **Only fields that phase accepts.** The queue's allowlist is the authority and
  refuses anything outside it — `unknown parameters for P5`. The launcher's own
  field list was written to match it by hand, so a field the queue gained is not
  automatically on the form.
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
> `CX_RECONSTRUCTION_STORE` for P8. The symptom is "nothing will queue" and the
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

A dead Queue button has a reason beside it saying what is still needed.

## Paths, and when they are checked

Prefer naming an **id** over a path. `layer_stack` as a P4 job id resolves to the
published artifact and its digest; `tiff_dir` as a path is a claim about a
filesystem the worker may not share with you.

> **Trap** The path check runs on the worker, **after** it has taken the lease —
> so a bad path costs one attempt and the job reads `failed` with the reason in
> it. It names which of two: missing (a mount) or unreadable (a permission). It
> covers `tiff_dir`, `checkpoint`, `ppm`, `order_path` and `segmentation`, plus
> `volume` when there is no `remote_url`. `dataset_path` is deliberately
> excluded — P1 stages into it.

The only thing that genuinely filters before a claim is the runtime image.

## Dry run

`dry_run` is queueable for **P1, P2 and P3 only**, and on the phases that run it
the toggle defaults **on**. On a scroll you have not run before it is the
cheapest thing you can do.

## Ceilings

Every numeric parameter has a floor, and most have a ceiling. A size nothing
rejects is a lease a worker takes before it finds out.
