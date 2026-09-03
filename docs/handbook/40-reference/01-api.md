---
title: The HTTP API
summary: 133 endpoints, the ones worth knowing by hand, and the two credentials that reach them.
---

The panel's own pages use this API, so anything the panel can do is reachable
from a script. The forms are not doing anything privileged.

## Credentials

Two, and they are not interchangeable.

**A session** — a person, signed in, holding a cookie. Reaches everything. There
are no roles: signing in is the whole boundary, and any account can do anything
another one can.

**A machine token** — a worker. Presented as `Authorization: Bearer
helena-machine-…`, and it reaches **the artifact endpoints and nothing else**. A
worker publishing a surface needs no ability to queue GPU work or read somebody's
missions, so it does not have one.

```
GET    /api/session              who is signed in, and whether anybody can be
POST   /api/session              sign in
DELETE /api/session              sign out
POST   /api/session/bootstrap    create the first account
```

`GET /api/session` answers `bootstrap_available`: true only while no account
exists yet. `POST /api/session/bootstrap` is refused once one does — 409, sign
in and add users from there — and refused from anywhere but loopback: the first
account can only be claimed from a shell on the host itself, with
`curl -X POST localhost:8800/api/session/bootstrap` and a JSON body naming a
username and password.

> **Note** A machine token's writes are bounded by namespace: it PUTs into
> `staging/` and `probes/`, promotes from staging into `surfaces/` and
> `layers/`, and deletes only probe evidence. It cannot write over a published
> surface, and a promotion may not take one as its source.
>
> **Reads are deliberately not bounded.** A token can GET or HEAD any key,
> published surfaces included, because P5 legitimately reads what another worker
> published and narrowing that would need a mission the key does not carry.

## What the shape of a request is

Everything is JSON. Errors carry a `detail`, and the ones worth handling say
what to do rather than what went wrong. The spec is at `/api/openapi.json`,
and an `/api/` path that matches no route answers **404** as JSON rather than
the app shell, so a typo or a version skew fails as itself instead of as a 200
full of HTML.

Server-owned parameters are refused rather than ignored. Where a job publishes,
and its whole control binding, are decided by the server — a request that sets
one gets a 409 naming the fields, because a job that publishes somewhere other
than where it was told should fail loudly rather than quietly do the right thing
and teach nobody.

## Queueing work

```
GET  /api/phases                       the phases, and which are queueable
GET  /api/phases/{phase}/parameters    every field that phase accepts
GET  /api/modules                      lanes, profiles, backends and seeders per phase
GET  /api/lanes                        the ink lane profiles, with registry status
GET  /api/ink/lanes                    the lane/adapter/registry cross-table
POST /api/jobs                         queue one -- P1, P4, P5, P7, P8, P9 only
POST /api/geometry/certify             queue P2
POST /api/flattening/run               queue P3
POST /api/segmentation/runs            queue a seeded P1 grow
GET  /api/jobs                         list, filterable by state and mission
GET  /api/jobs/{id}/events             the job's own history
POST /api/jobs/{id}/cancel             stop one
```

> **Trap** `POST /api/jobs` refuses P2 and P3 with a 400 saying the phase has no
> runner registered. They have their own routes, above. P1 splits the same way:
> the spiral-fit lane goes through `POST /api/jobs`, but the vc3d seeded grow --
> the default -- goes through `POST /api/segmentation/runs`, which shells out to
> the fleet's own bootstrap so the panel does not carry a second copy of the
> seeding policy.

`/api/phases/{phase}/parameters` is the one to build against. It is the same
contract the queue validates with and the form renders from, so a field it does
not list does not exist.

> **Trap** `POST /api/segmentation/runs` answers 409 with `inserted: 0` when
> every cell the run would cover already has a task under the same grid and
> policy version. That is not a fault: name a new policy version to open a new
> experiment over the same ground, or a smaller grid step to reach ground the
> current tiling skipped.

> **Trap** Cancelling behaves differently depending on where the job is.
> **Queued**: cancelled outright, 200, state `cancelled`, no worker involved.
> **Running**: a request — 202, the transient state `cancelling`, and the worker
> stops on its next heartbeat, within fifteen seconds. **Neither**: a 409, not
> silence. Either way the end state is `cancelled`, not `failed`.

## Missions

```
GET  /api/missions                          every mission, with its job count
POST /api/missions                          create one, empty or with scrolls
GET  /api/missions/{mission_id}             its manifest, runs and scroll lists
POST /api/missions/{mission_id}/amend       widen a frozen selection
POST /api/missions/{mission_id}/remove      drop scrolls that have no work yet
POST /api/missions/{mission_id}/state       change its state
POST /api/missions/{mission_id}/artifacts/freeze-p0   register what P0 produced
```

`unfiled` is not a mission: it is a read-only view assembled from runs that
predate missions, and every write route above refuses it with a 409.
`freeze-p0` and `amend` are idempotent by content -- pressing either twice
without a changed decision returns the same artifacts rather than a new
version.

## Reading state

```
GET /api/state?mission=…        counts; host-wide without the parameter
GET /api/runs?mission=…         the run index, same rule
GET /api/ink/maps?mission=…     P5 screenings, from the queue
GET /api/segmentation/segments  surfaces, with their QC axes
GET /api/fleet                  workers, jobs, and who is silent
GET /api/build                  which revision this panel is
GET /api/audit                  every mutation, and every refusal
```

> **Note** The `mission` parameter is not cosmetic on the first two: without it
> a dashboard tile counted the whole host, so a mission holding one scroll
> reported another scroll's surfaces and tasks as its own.

`/api/runs` walks the legacy run-receipt tree on disk: a directory per run,
each holding a receipt file. `/api/ink/maps` reads the same P5 screenings from
the job queue instead. A screening filed under no mission lands in `unfiled`
and never reaches a receipt directory, so it shows up at `/api/ink/maps` and
nowhere in `/api/runs` -- use the queue-backed one unless the deployment has
no database configured, in which case it says so rather than an empty list.

`/api/fleet` carries `workers` with a `state` of `POLLING` or `SILENT`, and
`workers_silent` as its own list. That is the difference between a quiet fleet
and a stuck one — see [the queue page](#/docs/panel/fleet).

## Artifacts

```
PUT    /api/artifacts/{key}   receive a gzipped tar, place it atomically
POST   /api/artifacts/{key}   copy from another key, server-side
GET    /api/artifacts/{key}   hand it back, gzipped
HEAD   /api/artifacts/{key}   is it there
DELETE /api/artifacts/{key}   remove it
```

A PUT is unpacked beside its destination and moved into place, so a transfer
that dies halfway leaves nothing that looks like a published surface. A reader
downstream cannot tell "still uploading" from "finished" by looking, so it never
sees the first state.

> **Trap** Overwriting an existing key removes it **first**, so a re-PUT has a
> brief window in which the key is absent rather than old-or-new.

## Models

```
GET  /api/models              what the frozen profiles need, and what is installed
POST /api/models/download     fetch one checkpoint from Hugging Face
```

A checkpoint counts as installed when a file under the models root hashes to
what a profile declares, not by its filename. `.safetensors` downloads freely;
a `.bin`, `.pt` or `.pth` is a pickle that runs code the moment a worker loads
it, so those are fetched only when the request states the `expect_sha256` it
must have -- send the hash the profile or registry declares. A file that
arrives with the wrong hash is deleted, not installed, and the response says
so as a 409.

## Configuration and identity

```
GET    /api/config                             every setting, its source and its default
PUT    /api/config/env/{name}                  set one
GET    /api/config/versions                    the history
POST   /api/config/versions/{version_id}/restore   go back by writing a new version
GET    /api/machines                           machine tokens
POST   /api/machines                           mint one; returned once, never recoverable
DELETE /api/machines/{name}                    revoke
```

> **Note** `/api/config` never returns a secret's value, and never its default
> either — a default that is a real credential is still a credential. It returns
> `value_present` instead.

Restoring a configuration version writes a **new** version equal to the old one
rather than rewinding, so the fact that you went back stays in the record.

## Also here

```
GET/PUT/DELETE /api/secrets/{name}     the fleet secret store -- not settings
DELETE         /api/config/env/{name}  drop an override
PUT            /api/config/constant    rewrites a constant in the platform's source
GET/POST/DELETE /api/users             accounts
POST           /api/users/{username}/password
POST           /api/segmentation/qc-jobs/requeue   configuration-blocked QC back to PENDING, with `fixed`
GET/POST       /api/hosts              workers' hosts; POST provisions one, or answers 503 without the script
GET            /api/hosts/{host_id}/provision   how that went
```

> **Trap** `PUT /api/config/constant` is not a setting. It edits a module-level
> constant in the source tree in place, nothing is committed, and the change is
> outside the configuration version log.
