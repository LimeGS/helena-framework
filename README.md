<p align="center">
  <img src="brand/helena-logo-horizontal-dark-1500.png"
       alt="Helena Exploration Framework"
       width="620">
</p>

<p align="center">
  <a href="https://github.com/LimeGS/helena-framework/actions/workflows/audit.yml"><img src="https://github.com/LimeGS/helena-framework/actions/workflows/audit.yml/badge.svg" alt="audit"></a>
  <a href="https://github.com/LimeGS/helena-framework/actions/workflows/audit.yml"><img src="https://img.shields.io/badge/coverage-%E2%89%A560%25-brightgreen.svg" alt="coverage at least 60%"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT"></a>
</p>

Helena is the orchestration and evidence layer for the volume → surface → ink →
page pipeline of the [Vesuvius Challenge](https://scrollprize.org). It
re-implements none of it. **VC3D, m7, ScrollFiesta, the Volume Cartographer
flatteners and the ink models stay exactly what they are**, behind stable
adapters. Helena records what each was given and what it returned.

**It solves the problems that begin once the methods work.** A campaign produces
thousands of surfaces and detections, across several machines, over weeks. By
then the algorithms are not what costs you time. Reproducing a result a month
later is. So is putting another machine to work without babysitting it,
comparing two lanes fairly, knowing which volume and which checkpoint produced
an output, and telling a phase that exited zero from one that actually decided
something.

Treat those as first-class and three things follow.

**A tool becomes comparable.** Code, model, parameters, preprocessing and
dependencies are frozen against content. The base image resolves to
`repository@sha256` or the build refuses, checkpoints are pinned by SHA-256,
and profiles, plans and artifacts each carry their own digest. Hold two lanes
against each other, or replace one without invalidating what came before.

**Hardware becomes poolable.** Trusted teams put their servers and GPUs behind
one auditable queue. It routes work to hardware that can run it, recovers what
an interrupted host was holding, never hands the same row to two workers, and
publishes hash-verified artifacts to shared storage.

**Evidence stays separable.** Geometry certification, CT support, model
response and human review are four distinct judgements, recorded separately. A
surface can be CT-supported and geometrically rejected at once. They answer
different questions, and collapsing them into one number is how a campaign
convinces itself.

The result is a path from one breakthrough to something a second person can
reproduce, audit and build on.

## Quickstart

```bash
curl -fsSL https://raw.githubusercontent.com/LimeGS/helena-framework/main/install.sh | sh
```

The installer asks what this machine should be, the panel alone or the panel
plus CPU or GPU workers, and puts the panel on `https://localhost:8800`. Pass
`--panel`, `--cpu` or `--gpu`, or set `HELENA_INSTALL`, to answer ahead of
time. With no terminal to ask on it installs the panel only.

You need `git`, Docker with **Compose v2**, a free port 8800, and about 6 GB
where Docker stores images: `docker info --format '{{.DockerRootDir}}'`, which
is often not under `$HOME`. The user running it must reach the daemon, so
`sudo usermod -aG docker "$USER"` and a new login, or run it under `sudo`.

You do not need Node, Python, CUDA or a GPU. The frontend builds inside the
image and the panel runs on CPU.

Read the script before piping it to a shell; `curl -fsSLO` then `less` is the
better habit. It wraps `git clone`, `docker compose up` for the panel and
`containers/deploy-platform.sh` for workers. Nothing is pulled from a registry;
every image is built where it runs. Then [claim the first account](#deploying).

MIT licensed. `NOTICE.md` states what this project claims and what it does not.
Read it before quoting any number from here.

## The pipeline

Ten phases, defined once in `framework/contracts/pipeline_phases.json`. The
panel, the queue, the job schemas and the in-app documentation are generated
from that file, so a phase cannot mean one thing in the UI and another in the
worker.

| | | |
|---|---|---|
| **P0** | Volume intake | Freeze which CT volume, and at what physical scale |
| **P1** | Segmentation | Find sheet surfaces inside the volume |
| **P2** | Geometry certification | Is this a physically plausible lamina? |
| **P3** | Flattening | Unroll it, keeping the coordinate map |
| **P4** | Surface volume rendering | Sample the CT along the normal into a layer stack |
| **P5** | Ink detection | Run a detector, get a probability map |
| **P6** | Liveness | Does that map carry a decision at all? |
| **P7** | Screening and adjudication | Turn the map into a verdict about text-like structure |
| **P8** | Reconstruction | Stitch segments into one continuous sheet |
| **P9** | Rendering and reading | Compose the sheet into a readable page |

P2 and P6 are gates, not transformations.

Where a phase has more than one way of doing the work, the choice is a **lane**
the job records rather than a setting somebody remembers. P1 grows a surface or
fits a spiral through the whole scroll. P4 renders through tifxyz or bridges a
legacy PPM onto a newer rescan. P5 runs several detectors, each behind a profile
that pins its checkpoint by digest. A run names its lane, so two results are
comparable or visibly not.

<p align="center">
  <img src="docs/screenshots/p5-ink-detection.png"
       alt="P5 ink detection: the public ink control's probability map with its lane, state, liveness verdict and percentiles"
       width="900">
</p>

P5 after the public ink control, on a machine installed with one command. The
map names its job, lane and state; `ALIVE` is the liveness verdict, the column
that exists so a uniform map cannot pass as a result.

## How that is enforced

**`checkpoint_path` is an input, never provenance.** A worker handed a file
whose digest is not the profile's fails the job instead of using it.

**Liveness is asked explicitly.** A detector that produces a uniform map still
exits zero and writes a file. Liveness gives every map a verdict (ALIVE,
DEGENERATE, EMPTY), and a P5 job that finishes without one is recorded as
failed.

**Claims expire.** `FOR UPDATE SKIP LOCKED`, a lease with a deadline, a bounded
attempt count and a filter on the capability the job declares. A dead host
returns its work without an operator, and a CPU box is never handed GPU work.

**Publication is atomic and origin-tagged.** Surfaces are staged, then
promoted, never written in place. The totals keep what this fleet grew apart
from what was imported.

**Deploys verify themselves.** After bringing the stacks up, the deploy checks
every container against the image it just built, by image IDs rather than tag
strings, and exits non-zero on a mismatch.

Phases implemented elsewhere are registered in `framework/registries/` and
vendored under `framework/vendored/` with a `VENDOR.json` recording source,
commit and a per-file hash. Only technique is imported; findings stay at the
source.

## Deploying

The Quickstart line clones, builds and starts what you chose. It refuses a
machine that already runs Helena, or that still has its volumes from an earlier
install. `HELENA_ADOPT_VOLUMES=1` proceeds anyway.

Then claim the first account. The panel offers a form for it when first opened;
from a shell on that host:

```bash
curl -sk https://localhost:8800/api/session/bootstrap \
  -H 'Content-Type: application/json' \
  -d '{"username":"you","password":"at-least-ten-characters"}'
```

`-k` because the certificate is self-signed; the panel log prints its
fingerprint. The endpoint answers only on loopback and closes for good once an
account exists. Later accounts are made under **Users**. There are no roles, so
an account is the whole boundary.

Artifacts go to a volume on the panel host. Object storage is off by default,
and `HELENA_BACKUP_S3` is only for off-site copies.

### Workers

The panel runs no phase. Phases need workers, and P5 needs a CUDA device.

```bash
containers/deploy-platform.sh nogpu   # segmentation, flattening, reconstruction
containers/deploy-platform.sh gpu     # adds ink detection and surface QC
```

On first run the deploy writes `config/*.env` from the templates and never
overwrites what you put there. The files live in the checkout, git-ignored;
nothing goes in `/etc`. Object-storage credentials belong on the panel, so a
worker starts from a database URL alone.

Nothing external is needed. The deploy builds what it runs, including Volume
Cartographer **compiled from source** at the commit its lock pins and checked
against that commit's tree hash. Expect an hour or two, more for `gpu` —
unless the images have been published, in which case `HELENA_PUBLIC_REGISTRY`
pulls them instead on a fresh install, and a failed pull still falls back to
building.

A worker on another host needs a machine token: mint one under **Users →
Machine tokens** and set `HELENA_PANEL_TOKEN`. It reaches the artifact
endpoints and nothing else, and can be revoked on its own. `provision-host.sh`
prepares such a second machine to join an existing fleet.

<p align="center">
  <img src="docs/screenshots/p1-segments.png"
       alt="P1 segments: the surfaces grown by the public segmentation control, with origin, CT support, geometry verdict and human review as separate columns"
       width="900">
</p>

The surfaces the public segmentation control grew, with the four judgements
kept apart: origin, CT support, geometry verdict, human review. Every one is
`CERTIFIED` and two are `CT SUPPORTED`; the columns answer different questions.

## Two controls you can rerun

Two controls are kept as evidence because a stranger can reproduce them. Both
drive the API on a machine installed from this repository with one command, and
both leave a receipt naming the boundary each stage passed or failed at. Both
passed on a fresh machine; the receipts sit beside each document, and the two
screenshots on this page were taken there.

- **Segmentation, P0 to P3.** Intake, grow, geometry certification, CT-support
  screening and flattening on PHerc826. A grow is not deterministic, so the
  control passes on outcome within a bounded task budget and records surfaces
  by digest. [Reproduce it](docs/public-control/REPRODUCE-SEGMENTATION.md).
- **Ink, P4 to P7.** The whole ink chain on a volume read anonymously from the
  open-data bucket, with a non-gated checkpoint verified by digest.
  [Reproduce it](docs/public-control/REPRODUCE.md).

## Where to look next

Everything else is in the panel, under **Documentation**:

- **Handbook**: a walkthrough from a scan to a picture of letters, a page per
  phase with its traps, and every panel page with what its controls do and when
  to leave them alone.
- **Developer reference**: contracts, profiles, receipts, versioning, and how
  to put your own tool into a phase.
- **API reference**: the HTTP surface, from the routes themselves.

The two references are generated from the code they describe, so they cannot
drift from the deployment in front of you. That is why this file is short.

## Repository layout

```
framework/contracts/   phase definitions, profiles, receipt schemas
framework/stages/      one directory per stage
framework/vendored/    imported techniques, with provenance
panel/                 FastAPI control plane and React frontend
containers/            images, compose files, deploy scripts
tests/                 the suite; runs without a deployment
```

## Contributing

```bash
containers/run-tests.sh tests/ -q --ignore=tests/e2e   # as CI runs it
cd panel/web && npm ci && npx vitest run               # frontend
```

Use the script rather than `pytest` directly: without a database eighty-odd
tests skip silently. It builds the CI image, starts a throwaway postgres and
runs the suite inside it.

The suite is the specification. Tests are named after the failure they prevent,
and the docstring says what went wrong and why the check is shaped as it is. If
you change behaviour, change the test that describes it. `tests/e2e` needs a
running deployment and is skipped without one.

Versions follow Semantic Versioning and the current one is in `VERSION`, which
the compose defaults and the build script are checked against. Profiles, runs
and receipts carry their own immutable identities, which never move with it.

## TODO
- Update to the latest villa commit (Ongoing tonight).

## DONE
**Publish the images, for real.** The pull-before-build mechanism exists
  (`HELENA_PUBLIC_REGISTRY`, above) and so does the CI job that would push to
  it — `publish images to docker hub`, `when: manual` on purpose, gated on a
  Docker Hub token nobody has added yet. Until somebody clicks it, every
  install still builds Volume Cartographer from source.

