---
title: Intake and Configuration
summary: Where a mission gets its scrolls, and where the deployment's own knobs live.
---

## Intake

P0 is where a mission gets its scrolls, and where its selection is edited. **No
whole scroll is ever pulled onto a disk**, and that is worth saying plainly
because the interface would otherwise imply it.

The source is an OME-Zarr over HTTPS and every reader takes what it touches:
VC3D takes the URI, the renderers do sparse chunk-gather. What lands on disk is
derived and far smaller — a layer stack is gigabytes where the scan is hundreds.
P1 is the exception that proves it: it **stages** what it fits into
`dataset_path`, once per scroll.

### Scrolls

The table lists every top-level prefix the source bucket exposes, found by
layout rather than by an index: a prefix counts as a scroll only when it holds
`volumes/<timestamp>-<µm>um-…-<keV>keV.zarr/`. Prefixes with no `volumes/` are
named below the table rather than dropped silently. Its columns are the scroll
id, **µm** and **keV**, scan count, the earliest scan's own date, runs recorded
in this deployment, and whether the scroll sits in the current mission. A
filter box narrows by id, and **only this mission** narrows to the selection.

µm and keV come first from the frozen catalogue and, where it says nothing,
from the finest scan's own directory name; a blank cell means the scale is
unpinned, not merely unknown. The catalogue is a committed file covering 13
scrolls; the bucket itself currently lists around 45, so most rows get their
scale, if any, from the scan name instead. `CX_CATALOG_REFRESH` (on by
default) regenerates the catalogue from the bucket at startup and once a day,
so a newly published scroll can gain a declared scale without anyone editing
the file. Only a scroll the catalogue or another registered source can resolve
to a volume can actually be added to a mission — see
[Missions](#/docs/panel/missions).

**Refresh inventory** re-lists the bucket directly, bypassing the on-disk
cache that `CX_SCROLL_TTL` (a day by default) otherwise serves from. That is a
different cache, and a different setting, from the catalogue refresh above.
The **Inventory origin** tile in the phase header above names which bucket
answered and can **change** to a different one for this browser only; that
never touches `CX_SCROLL_SOURCE`, which is what Configuration persists.

The `unfiled` view has no selection to edit — its list describes runs that
predate missions, not a choice — so its checkboxes stay disabled.

### Changing the selection

Ticking an unselected scroll queues an addition; unticking a selected one
queues a removal. Nothing happens until **Apply**, which adds through
`/amend` and then removes through `/remove`. A mission with no work yet
applies straight away; once anything has run, every change needs a **reason**,
kept in the amendments table below, and a scroll that already produced work
here cannot be removed at all.

Amending re-registers a frozen-source artifact for every scroll now in the
selection (content-addressed, so an unchanged scroll keeps its version), which
is what "What P0 produced" below actually answers from. Removing does not
touch that registry. A selection assembled before that automatic freeze
existed, or made through the API, can show scrolls selected and nothing
produced there; **Record what P0 decided** catches it up on demand, and a
second press returns the same artifacts rather than minting new ones, because
pressing it twice is not a new decision.

If the voxel size is not pinned here, every micron figure downstream is
unanchored, and P5 resamples against a number nobody checked.

See [P0](#/docs/phases/p0).

## Configuration

The Configuration surface is the shell for seven tabs: Settings, Modules,
Models, Hosts, Users, Lineage and Audit log. The Settings tab itself has every
setting the deployment has, with its value, its default, and **where the value
came from** — precedence is override, then environment, then default.

That last column is the one that matters. A setting that looks right and is
coming from somewhere you did not expect is how a host ends up behaving
differently from its neighbour.

### Versions

Configuration is versioned, and there is always a version 0 — the built-in
defaults, the only version whose reason is not an action.

Restoring an old version writes a **new** version equal to it rather than
rewinding, so the fact that you went back stays in the record. Restoring rewrites
the override file to only the values that differ from **both** the default and
the environment, so a restore can silently drop an override that happens to
match an env var.

Two refusals: a commit that changes nothing, and restoring the version that is
already active.

### Two more sections, and they are not settings

- **Framework constants** rewrite a module-level constant **in the platform's
  own source**, in place, and nothing is committed. That is editing code from a
  web form, and it is not covered by the configuration version log.
- **Theme** is the one setting on the page that is not versioned.

### Credentials

The **Credentials** card is fleet secrets: a *separate* store in the control
plane with its own routes (`GET`/`PUT`/`DELETE /api/secrets`), not a setting
marked secret and not covered by configuration versions either. It holds six
names and nothing else — `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, `AWS_REGION`, `AWS_ENDPOINT_URL`.

Write-only: the card reports whether a value is set, its length and who set
it, never the value itself, and setting one needs a signed-in author. This is
not encryption at rest — anyone who can read the control-plane database can
read these too, the same as the database password every worker already
carries.

A worker adopts them when it next starts. An environment variable already set
on that host wins over the control plane's copy, deliberately, and the worker
logs which names it is shadowing so the mismatch does not pass unnoticed. See
[backups and object storage](#/docs/operations/backups-and-object-storage)
for what these credentials are for.

### Secrets

A setting marked secret never returns its value through `GET /api/config` — and
never its default either, because a default that is a real credential is still
one. The page shows whether a value is present, not what it is.

> **Trap** The **version history does not redact.** `GET /api/config/versions`
> returns each version's full settings map with no secret filter, and `CX_DB` is
> a Postgres DSN with the password in it. Anyone signed in can read through one
> GET what the config endpoint's redaction exists to protect.

> **Trap** Some settings only take effect on restart, and the page says which.
> A knob that reads as changed and is not yet in force is the shape of an
> afternoon spent debugging the wrong thing. Only a short list is rebound live —
> the repo, runs, cache, scroll source and TTL, catalogue, profile directory,
> registry and map size cap. Everything else marked for restart is written and
> reported, not pretended to be live.
