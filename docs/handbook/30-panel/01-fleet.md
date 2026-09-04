---
title: The queue, and which workers are alive
summary: Where the queue actually lives now, and how to tell a worker that is idle from one that is stuck.
---

> **Trap** There is no Fleet page. The panel has three top-level surfaces —
> Mission, Configuration, Documentation — and queueing belongs inside the phase
> that does the work. `Fleet.tsx` still exists in the tree and is imported by
> nothing; `/command` redirects to `/`.

What survives where:

| You want | It is on |
| --- | --- |
| the queue for a phase | that phase's panel, under Mission |
| fleet counts and hardware | the **Fleet** and **Fleet hardware** tiles on Mission |
| worker liveness | `GET /api/fleet` — and only there |
| a host's own hardware, roles and ssh target | [Configuration → Hosts](#/docs/panel/models-hosts-modules) |

**Fleet hardware** only adds up enabled hosts, so disabling one drops its cores
and cards from the total — but nothing in the claim reads that flag, and a
"disabled" host's own worker keeps claiming. See Hosts for what enabling and
disabling a host actually does.

`GET /api/fleet` layers `workers` and `workers_silent` onto the same counts
`/api/state`'s `fleet` field already carries — that field is where the Mission
tiles read from, so `tasks`, `surfaces` and `stale_leases` are on the page
already: `attempts`, `imported` (surfaces that arrived from a catalogue rather
than being grown by an attempt), `events`, `leased`, `task_states`,
`surfaces_by_sample`, and `qc_blocked_on_configuration` — QC jobs refused over
a configuration mismatch rather than a busy fleet — are not. Both endpoints
take `?mission=`, narrowing the counts to one mission's scrolls.

## Workers

Every worker that has ever polled has a row: its host, the runtime image it
carries, and when it last asked for work.

| State | Means |
| --- | --- |
| `POLLING` | it polled within sixty seconds — six missed polls is the threshold |
| `SILENT` | it has not, and something is wrong |

> **Trap** Nothing in the UI renders this yet. `GET /api/fleet` computes it and
> returns `workers_silent` as its own list; no page shows either. Until one
> does, this is an endpoint you curl.

The row is written by the **poll**, claimed or not — so a worker that is looking
and finding nothing still appears, and its absence is the signal. `last_claim_at`
is a separate column, and it is what distinguishes looking from taking.

A worker with nothing to claim and a worker that cannot claim used to look
identical, and `docker ps` says "Up" for both. Three of them once stopped for
eighteen hours that way.

The host's own heartbeat cannot answer this: it is written by a different branch
of the same loop, so it keeps reporting while the claim beside it is blocked,
and the host looks healthy because part of it is.

## Jobs

Jobs move `pending` → `leased` → `running` → `succeeded` / `failed` /
`cancelled`.

- A **lease** is a claim with a deadline. A worker that dies has its lease
  expire and the job returns to the queue with the attempt counted, until
  `max_attempts` — three by default — after which it is failed as
  `LEASE_EXHAUSTION` rather than recycled forever. The expiry is its own event,
  carrying `held_by`, `detected_by` and `attempts_remaining`, so "the worker
  stopped reporting" is distinguishable from "the job failed". It is recycled
  lazily, by the next worker to poll, not by a sweeper.
- **Cancel** depends on where the job is. A **queued** one is cancelled
  outright and synchronously — 200, state `cancelled`, no worker involved. A
  **running** one cannot be interrupted from outside: the request is written,
  the worker reads it on its next heartbeat (fifteen seconds) and stops. That
  returns 202 with the transient state `cancelling`. A job that is neither gets
  a 409. Either way the end state is `cancelled`, not `failed`.
- **Attempts** are bounded. A job that has burned its attempts stops rather than
  cycling.

## Reading a failure

A result from a job that **ran** carries `ran_by`: the image, `BUILD_REVISION`,
and the sha256 of `ink_worker.py`. That is what turns "the image was stale" from
an inference into a fact in the row.

A job refused **before** the subprocess starts — an unreadable input, a write
outside its run directory, a missing upstream directory, a lineage refusal —
carries the refusal and no `ran_by` at all. For those, compare the container's
image digest instead: `GET /api/hosts/{host_id}/images` reads what a host's
containers actually run over ssh and checks it against what its roles expect,
by digest rather than tag. It answers "not reachable" for a host with no ssh
target — the common case for one that only ever registered itself by
reporting.

## Runtime images

A lane can declare the image it needs. The claim skips a candidate whose lane
needs an image this worker does not run, so most of the time nothing is spent
on a job it cannot do. A worker that has never declared its own image claims
blind instead, and is only refused **by name** at execution — better than
failing at the first import with no explanation, but the lease and an attempt
are already gone by then. Either way, a job can sit pending while the only
worker that could run it is busy.

> **Note** `HELENA_RUNTIME_IMAGE` names the **lane** image a worker carries, not
> the composed image it runs as. Backwards, the worker starts, refuses every job
> for that lane, and looks exactly like a worker with nothing to do. It refuses
> to start now and says which name it needs — but only when the wrong name ends
> in `-worker` and stripping that yields a known lane. Any other wrong value is
> still silent.

The rule runs both ways, and the second direction surprises people: a worker
whose runtime **is** a lane image takes **only** that lane. It refuses jobs that
need no special image rather than claiming them, which is why the 9 µm worker
stopped taking canonical and timesformer runs and failing them in two seconds
each.

## Why a job sits pending with healthy workers

Five filters, and none of them is visible on the row:

- `requested_host` pins it to one machine;
- the worker's `phases` must include the job's;
- `gpu_required` against what the worker probed;
- `minimum_vram_gb` against the card it found;
- the runtime image, in both directions above.

The claim also looks at only eight candidate rows at a time, ordered by priority
then age — so a job behind a wall of higher-priority work is not stuck, it is
queued.
