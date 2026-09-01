---
title: When something is wrong
summary: The failures this platform has actually had, and what each one looked like from the panel.
---

Each of these happened. The symptom is what you would have seen; the cause is
what it was.

## A phase reports success and produced nothing

**Symptom** — `succeeded`, exit code 0, and an empty output directory.

**Cause** — an upstream tool that exits zero after failing to open its input.
The renderer prints one line, `Error loading`, and carries on.

**What the platform does now** — four checks: the count matches `num_slices`,
the filenames are numeric, the indices are exactly `0..n-1`, and the middle
slice is not a constant. A job that fails any of them is `failed`. A render with
nowhere to publish is failed too, unless `allow_local_layers` was passed.

If you see a success with no output, the worker is running code that predates
the check. Read `ran_by` in the result — and if it is not there, that is itself
the answer, because `ran_by` arrived with the same generation of the worker.
Compare the container's image digest instead.

## A job fails on a path that exists

**Symptom** — `Error loading /some/path`, and the path is right there.

**Cause** — the panel and the worker are different containers, often on
different hosts. The path exists on the host and is not mounted in the
container.

**What to do** — the refusal names which of **two** it is: missing, or
unreadable. Missing means a mount; unreadable means a permission. There is no
third.

It covers `tiff_dir`, `checkpoint`, `ppm`, `order_path`, `segmentation`, and
`volume` when there is no `remote_url`. `dataset_path` is excluded on purpose —
P1 stages into it. A job failing on something else will not produce this
refusal.

The sibling refusal you meet next is a **write** outside the run directory: the
allowed roots are named in the message, and `HELENA_JOB_WRITE_ROOTS` is how a
deployment widens them.

## Workers stop claiming and `docker ps` says Up

**Symptom** — the queue has pending work, nothing is running, containers look
healthy, and the logs simply stop.

**Cause** — a claim blocked on a lock. `FOR UPDATE SKIP LOCKED` skips locked
*rows* and waits like anything else for a lock on the *table*, and the
migrations take one.

**What the platform does now** — two things, and the first is the cause.

`initialize()` takes a transaction-scoped advisory lock around the whole
migration block. Every worker calls it at startup, so three starting together
deadlocked on `AccessExclusiveLock` against each other; a worker that dies half
way out of its own migration has a schema in an unknown state, which is the best
explanation anyone has for what followed. Migrations also get a 600-second
budget so a schema change waits rather than half-applying.

Second, every connection sets a lock timeout and a statement timeout, so a claim
fails, says so, sleeps and retries. A blocked call prints nothing, which is why
the logs stopped; a bounded one is noisy and recoverable.

`GET /api/fleet` separates `SILENT` from idle and returns `workers_silent` as
its own list. **Nothing in the UI shows it yet** — for now this is a curl.

## A lane refuses every job by name

**Symptom** — a worker is up and polling, and jobs for one lane sit pending.

**Cause** — `HELENA_RUNTIME_IMAGE` naming the composed worker image rather than
the lane image it carries.

**What to do** — it refuses to start now and says which name it needs. That
check only fires when the wrong value **ends in `-worker`** and stripping that
yields a known lane image; any other wrong value is still silent, with exactly
this symptom. On an older build, compare the variable against the lane image the
profile declares.

## A P5 lane refuses with "no upstream directory"

**Cause** — two lanes declare a vendored upstream tree, and that directory is a
property of the **host**, not the job. One of them puts it on `PYTHONPATH`, and
that is the one whose absence refuses. A worker that does not declare one
refuses rather than importing from wherever it was pointed.

**What to do** — set the variable named in the refusal on that host, pointing at
a path the container can see.

## Jobs sit `PENDING` and nothing is wrong with the workers

**Symptom** — a healthy fleet, and a queue that will not drain.

Five filters decide a claim and none is visible on the row: `requested_host`,
the worker's phase list, `gpu_required`, the VRAM floor, and the runtime image
in both directions — a worker whose runtime *is* a lane image takes **only**
that lane.

**And a sixth, which is not the queue at all**: QC jobs blocked on
configuration. Two GPUs once failed every claim for two days and the only place
it showed was jobs sitting in `PENDING`, which is also what a healthy queue with
no free worker looks like. The state is `BLOCKED_CONFIGURATION` — a profile hash
that does not match what the deployment pins, or a checkpoint that is not the one
the profile names — and it is served as its own count. **A number above zero is
always something to act on**, because these stay blocked until a person changes
a setting.

## Nothing will queue at all

**Symptom** — every attempt to queue a phase comes back 409.

**Cause** — a blank store setting. `CX_RENDER_STORE` for P4,
`CX_INK_STORE` for P5, `CX_FLATTEN_STORE` for P3, `CX_RECONSTRUCTION_STORE` for
P8. The panel refuses rather than queueing work whose output would have nowhere
to go.

**What to do** — the Configuration tab. The symptom and the cause are two pages
apart, which is why this entry exists.
