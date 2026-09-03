---
title: When something is wrong
summary: The failures this platform has actually had, plus what a fresh install turns up, and what each looked like from the panel.
---

Each of these happened, here or on a fresh install. The symptom is what you
would have seen; the cause is what it was.

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

It covers `tiff_dir`, `checkpoint`, `ppm`, `order_path`, `segmentation`,
`model_config`, and `volume` when there is no `remote_url`. `dataset_path` is
excluded on purpose — P1 stages into it. A job failing on something else will
not produce this refusal.

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

## A worker dies on its first QC job with `PermissionError`

**Symptom** — a fresh surface-QC worker claims its first job and dies on
`PermissionError: '/artifacts/qc-runtime/<job-id>'`.

**Cause** — the surface-QC compose file bind-mounts
`$HELENA_QC_RUN_ROOT/gpu<device>` onto `/artifacts/qc-runtime`, and Docker
creates a bind mount's source as `root:root` when it does not exist yet. The
runtime runs as uid 1000 and cannot write into a directory Docker just made
for it.

**What the platform does now** — the deploy creates and chowns each device's
run root before the surface-QC stacks come up, so Docker never gets to make
it first. An install still running an older deploy script will meet this on
its first job on any device it has not run before.

## A worker will not start over its database URL

**Symptom** — `database environment variable is not a PostgreSQL URL: <NAME>`
from a segmentation worker, or `<NAME> is not set to a PostgreSQL URL` from
`host_report.py`.

**Cause** — `postgres-env://NAME` is the preferred form for a fleet DSN: it
names the environment variable holding the URL, so the URL itself never lands
in argv or a process listing that any user on the host can read. The variable
either is not set on this worker, or holds something that is not a
`postgresql://` or `postgres://` URL.

**Fix** — set `NAME` to the real connection string in the worker's own
environment, not the panel's.

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
property of the **host**, not the job: one takes it as a `--upstream-dir` flag,
the other puts it on `PYTHONPATH`. Both read the same variable and both refuse
without it, rather than importing from wherever a worker happens to be
pointed.

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
a setting. Once the setting is fixed, `POST /api/segmentation/qc-jobs/requeue`
with the mission, the scroll and a `fixed` saying what changed sends those jobs
back to `PENDING`; the state is terminal on its own, and nothing retries it on
a timer.

## QC jobs keep retrying and never finish

**Symptom** — `RETRYABLE_QC_UNAVAILABLE` receipts pile up for one worker. The
job never reaches `BLOCKED_CONFIGURATION` or a verdict; it just requeues.

**Cause** — `nvidia-smi -L` fails in the driver's own words: `Failed to
initialize NVML: Unknown Error`, no CUDA-capable device, and the like. Seen
on a host whose cgroup state changed under a running container. This is a
card that cannot be initialised at all, distinct from a card with no room on
it, which is the same retryable state under `GPU_MEMORY_EXHAUSTED`.

**Fix** — restart the worker's container. The worker checks before the
executor runs, on purpose, so this reads as "GPU cannot be initialised"
instead of an executor failing with exit code 1 and no other explanation.

## Nothing will queue at all

**Symptom** — every attempt to queue a phase comes back 409.

**Cause** — a blank store setting. `CX_RENDER_STORE` for P4,
`CX_INK_STORE` for P5, `CX_FLATTEN_STORE` for P3, `CX_RECONSTRUCTION_STORE` for
P8. Each defaults to a path under `/artifacts`, the platform's own volume, so
this is a setting somebody emptied — or, for P1, a panel started without
`ARTIFACT_ROOT`, which the platform compose sets to `/artifacts`. The panel
refuses rather than queueing work whose output would have nowhere to go.

**What to do** — the Configuration tab. The symptom and the cause are two pages
apart, which is why this entry exists.

Three more 409s at enqueue have nothing to do with the store:

- **P4 needs a scale.** `source_voxel_um is required: the frozen catalogue has
  no entry for <sample>`. Left out, the renderer finds no metadata in the
  volume, reports `Voxel size: 1.0`, and renders anyway at the wrong scale —
  the slice count, shape and exit code all read the same as a correctly scaled
  render. Read the voxel size off the volume and pass `source_voxel_um` on the
  job.
- **A P2 tiling is already covered.** `nothing was queued: all N cells this run
  covers already have a task under grid <grid_version> and policy
  <policy_version>`. A task's identity is (volume, grid version, cell, policy
  version); re-running the same scroll with a different seeder writes into
  the identity that already exists and inserts nothing. Name a new policy
  version, or use a smaller grid step to reach ground the current tiling
  skipped.
- **The phase has no direct queue.** `phase <phase> has no runner registered`
  from `POST /api/jobs`. Only P1, P4, P5, P7, P8 and P9 queue through that
  endpoint; P2 and P3 are observed there, not driven — they have their own
  path (P2's fleet CLI, P3's `/api/flattening/run`).

## A credential set on the panel has no effect on a worker

**Symptom** — the worker's log says `set on the panel but overridden by this
worker's environment: <NAME>`. Configuration shows the credential as set, and
the worker keeps using something else.

**Cause** — a segmentation worker adopts credentials from the control plane's
stored secrets at startup, but an environment variable already set on that
worker wins over what the panel holds. Deliberate: an operator who exported a
key for one run means it, and a value from the panel silently overriding it
would be a debugging session about which credential is actually in use.

**Fix** — unset the variable on the worker to let the panel manage it, or
accept that this worker keeps its own value.

## No backup container on a fresh deployment

**Cause** — the backup service only starts with somewhere to put a backup.
Started without `HELENA_BACKUP_S3` it does not fail; it restarts forever
printing "no destination", a permanently red container on a host that was
never meant to ship its data anywhere. A development deployment is the
ordinary case.

**What the platform does now** — the deploy checks `HELENA_BACKUP_S3` before
bringing the stack up and, if it is blank, skips the backup profile and says
so on the console: `no HELENA_BACKUP_S3 on this host; the backup service is
not started`.

## The Runs tab is empty on a fresh install

**Symptom** — P5's Runs tab reads "No receipts on disk under CX_RUNS.
Screenings queued through the fleet are under Maps."

**Cause** — Runs indexes the legacy receipt tree on disk under `CX_RUNS`,
which a fresh install has none of. A screening queued through the fleet is
recorded in the database instead, and shows up under Maps, not here.

**Fix** — nothing to fix. Look at Maps for anything queued through the fleet.

## A host row appears with no ssh target

**Symptom** — a Hosts row for a machine nobody registered, its note reading
"registered by its own report; add an ssh target under Configuration -> Hosts
to provision it from the panel". `host_report.py` itself prints `registered
'<host>' by its own report` the first time.

**Cause** — a worker can run on a machine before anyone adds it in the panel.
Rather than refuse to record hardware for a host nobody has typed in yet,
`host_report.py` inserts the row on that host's first report, with no ssh
target.

**Fix** — nothing is broken. Add an ssh target under Configuration -> Hosts if
the panel should be able to provision or manage that machine from here.

## Registering a host 503s: "this deployment cannot provision hosts"

**Cause** — the panel image does not carry `containers/provision-host.sh`;
every deployment running from the panel image alone answers this the same
way, so it is not a fault in the request.

**Fix** — the host is registered regardless. Bring the worker up yourself with
the compose files and it will report itself; provisioning from the panel needs
the full checkout, not the panel image.

## Every API call 401s with "not signed in"

**Cause** — no session cookie matched a signed-in user. Ordinary before
anyone has an account on a fresh install, or after a cookie has expired.

**Fix** — sign in. The response carries `bootstrap_available`, which says
which case it is: true when no account exists yet, so the first request can
create the first admin instead of asking to sign in to nothing.

## The installer refuses to run

Checked before anything is built or started:

- **Disk space.** `Docker has <N> GB free at <path> and the build needs about
  6.` `docker system prune -af` reclaims images and build cache.
- **Port 8800 busy.** `port 8800 is already in use.` Set `HELENA_PORT` to
  another one.
- **A stack already running.** `this machine already runs a Helena stack.`
  The compose project is always named `helena`; running the installer again
  would recreate it, not start a second one. Use
  `containers/deploy-platform.sh` to update an existing deployment instead.
- **Volumes left behind.** Nothing is running, but Helena volumes from an
  earlier install are still there — possibly written by a different user,
  including root, whose leftover TLS material can make a fresh panel exit
  with a `PermissionError` that does not look like the cause. Set
  `HELENA_ADOPT_VOLUMES=1` to install over them anyway; otherwise the
  installer stops here.
