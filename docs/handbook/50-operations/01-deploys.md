---
title: Deploys and images
summary: How code reaches the fleet, and the two ways a deploy can lie about what it shipped.
---

## The path

A push to `staging` or `development` runs the pipeline. Five stages —
`prepare`, `test`, `build`, `deploy`, `verify` — on the host that branch owns:
each branch deploys the host its runner is tagged for.

The last one is the half people forget. `verify` is the smoke and end-to-end run
**against the deployment that just landed**, and e2e is deliberately excluded
from `test` because at that point there is no deployment to test against.

The **deploy and smoke** jobs prove they are on the machine they are for before
doing anything — a tag is a request; the check is the proof. The build jobs only
prove a daemon is there.

## Where configuration lives

`install.sh` asks what a machine should be — `--panel`, `--cpu` or `--gpu`, or
`HELENA_INSTALL` — brings the platform stack up from the checkout and then
calls `containers/deploy-platform.sh nogpu|gpu` for the workers. That script
seeds `config/*.env` in the checkout from `containers/compose/*.env.example`
on first run — platform, panel, segment, ink, surface-qc, and a `postgres.env`
it writes from the platform's own credentials — never overwriting a file that
exists, and fills in the two things no template can know: this host's name and
the workers' database URL. The directory is git-ignored and holds nothing
privileged. A host configured before this keeps `/etc/helena`, and the deploy
says so; `HELENA_ENV_DIR` points it anywhere else. Nothing on a host runs
outside a container: `provision-host.sh`, for a second machine joining a
fleet, copies the compose files and an env file to the target and starts the
tunnel and the worker as compose projects, and installs no units.

## What runs, and what only builds

Up to eleven containers on a host with cards, from five images. Three more
images exist and never run: they are parents, and a name in `docker images`
that is never a container is the thing that makes this look bigger than it is.

| Container | Does | Image |
| --- | --- | --- |
| `helena-postgres` | the queue, missions, receipts | `postgres:16-alpine` |
| `helena-panel` | the panel and its API | `helena-panel` |
| `helena-backup` | copies of the database | `helena-backup` |
| `helena-segment` | finds surfaces (P1) | `helena-worker-cpp` |
| `helena-fleet-runner` | P2, P3, P8 | `helena-worker-cpp` |
| `helena-preflight` | checks a job before it is queued | `helena-worker-cpp` |
| `helena-host-report` | reports this host's hardware | `helena-worker-cpp` |
| `helena-spiral` | the spiral fitter (P1) | `helena-worker-cpp` + lane |
| `helena-ink-0` | P4, P5, P7 | `helena-worker-gpu` |
| `helena-ink-9um` | P5 on the 9 µm lane | `helena-worker-gpu` + lane |
| `helena-gpu-runtime-<n>` | automatic QC, one per card | `helena-worker-gpu` |

Not all of them on every host. `helena-backup` starts only where
`HELENA_BACKUP_S3` is set; `helena-spiral` only where the `helena-villa-python`
lane image is present, which the deploy does not build; `helena-ink-9um` where
the 9 µm lane image is, which it does. A clean `install.sh --gpu` on one card
leaves nine running, and two more — `helena-prepare-volumes`, `helena-init` —
that ran once and exited. A host without cards runs the first seven, backup on
the same condition.

## The tree

```
postgres:16-alpine                         pulled, not built

node + python  -> helena-panel
postgres:16-alpine -> helena-backup                   the dump script on the client image

ubuntu:25.10   -> helena-villa -+-> helena-worker-cpp        P1 P2 P3 P8
                                |
                                +-(bundle stage)------+
pytorch/pytorch -> helena-ink --------------------------+-> helena-gpu-runtime
                                                             |
                                                             +-> helena-worker-gpu   P4 P5 P7
```

`helena-villa`, `helena-ink` and `helena-gpu-runtime` are build-only.

- **`helena-villa`** is volume-cartographer compiled from source, cloned at the
  commit the source lock pins and checked against its tree hash. It is
  **GPL-3.0**; see `NOTICE.md` before publishing an image built from it.
- **`helena-ink`** is the frozen six-replica TimeSformer environment: a public
  PyTorch base and pinned wheels, no code of ours.
- **`helena-gpu-runtime`** is the two of them joined — torch for P5's inference
  and the VC3D tools for P4's rendering, because one worker runs both phases.

The bundle stage takes three tools out of `helena-villa` with the closure `ldd`
reports for them: 289 MB rather than the 1.4 GB of the whole toolchain. It used
to be an image of its own, helena-vc3d, which no longer exists: its only
consumer was this build.

> **Trap** Basing the GPU runtime on `helena-ink` is the obvious simplification
> and it produces an image whose tools cannot start. Measured:
> `helena-villa` compiles on glibc 2.42 and its binaries ask for 2.38, while the
> PyTorch runtime image is 2.35 — `version GLIBC_2.38 not found`, on the first
> job, because a bundle carries its library closure and deliberately not glibc.
> The build checks this and says so.

## Lanes

A **lane** is a frozen upstream environment — its own interpreter, its own
virtualenv, its own source tree — that cannot share ours. Two exist:
`helena-ink-9um` (P5 at 9 µm) and `helena-villa-python` (the spiral fitter and
lasagna).

They are grafted into a worker at `/opt/lanes/<name>` by a second build target.
Each worker Containerfile has two: one for a host without the lane — `runtime`
in the GPU worker, `worker` in the CPU one — and `with_lane` for one with it.
BuildKit does not build the lane stage when it is not asked for, so a host
without the lane image never needs to have it.

One path per lane, and that matters: both used to land at `/opt/villa`, so two
lanes could not sit in one image, and there were four worker images for two
workers: a 9 µm worker image of 16.5 GB beside the 7.83 GB one it was built
from, differing by one directory. Neither of those two exists now.

> **Trap** A **lane** image and a **worker** image are both named in the same
> compose invocation, and pointing the wrong one at `HELENA_INK_IMAGE` is quiet. `HELENA_INK_IMAGE` is what runs;
> `HELENA_RUNTIME_IMAGE` is what it says it is. Pointed at the lane image, the
> container has no repository in it and crash-loops on a missing
> `ink_worker.py`; the lane image says so itself now rather than letting Python
> report a missing file.

## Two ways a deploy lies

- **A moving tag.** `:latest` in a deploy path means two hosts provisioned a
  week apart run different code and both report the same image. It is no longer
  pushed to the shared registry — it stays on the building daemon, where it is a
  convenience rather than a name several hosts pull. **`:local` is the same
  failure with another name**, and the spiral compose still defaults to it; the
  runtime compose is the counter-example worth copying, since it refuses to
  start without a digest.
- **A wrong base.** A composed image built on the wrong base passes every check
  the composition has and is missing something nobody thought to check. The
  composed worker images verify the repository is there **first**, before the
  dependency checks a wrong base would satisfy.

## Verifying a deploy

Two different questions, two different labels:

```bash
docker inspect <image> --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
docker inspect <image> --format '{{index .Config.Labels "org.opencontainers.image.base.name"}}'
```

The first answers "is the right commit deployed" — the pipeline asserts it too.
The second answers "was this built on the right base", which is a different
failure and the one that shipped a worker with no repository in it. The CI image
carries no base label, so that command comes back empty for it.

You can also compare a running worker's reported source digest against the
commit you think is deployed — every result from a job that **ran** carries it.

## Things that look like a broken pipeline and are not

- **A build that pulls instead of building.** If the commit's tag is already in
  the registry the job pulls and exits, because a second build of the same
  commit is not the same bytes — Docker builds are not reproducible by default,
  and one commit naming two different images is the thing being prevented.
- **A merge request that builds and publishes nothing.** Deliberate: an MR that
  pushed its own bytes for a sha left the post-merge deploy pulling the MR's
  image.

## Rollback

Before rewriting an env file, the deploy retags the outgoing image
`…:rollback-<commit>` and copies the env file to `<file>.bak-<commit>`, keeping
the last ten. That is the recovery path.

## "The platform" is seven compose projects

`helena` (postgres, prepare-volumes, init, panel, backup), `helena-segment`
(segment, fleet-runner, preflight), `helena-host-report`, `helena-ink-0`,
`helena-ink-9um`, `helena-spiral`, and one `helena-qc-N` per card. A worker
host that reaches the control plane over SSH runs an eighth, `helena-tunnel`:
`control-tunnel.compose.yaml`, the forward to the database's port as a
container with a restart policy, where a systemd unit used to be. The deploy
script exists precisely because an "all" that is not all is worse than no
"all", since it reports success.
