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

The selection freezes as soon as work exists for the mission, and widening a
frozen one needs a reason, which is recorded.

See [P0](#/docs/phases/p0).

## Configuration

The Configuration surface is the shell for seven tabs: Config, Hosts, Users,
Lineage, Audit, Modules and Models. The Config tab itself has every setting the
deployment has, with its value, its default, and **where the value came from** —
precedence is override, then environment, then default.

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
- **Fleet secrets** are a *separate* store in the control plane with its own
  routes. Workers pick one up when they next start. They are not the same thing
  as a setting marked secret.
- **Theme** is the one setting on the page that is not versioned.

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
