# Reproducing the public segmentation control

The other half of the public ink control. `REPRODUCE.md` drives P4 to P7 --
render, ink, screening -- on a volume read anonymously from the open-data
bucket. This drives P0 to P3: intake, grow, certify the geometry, screen for
physical support, flatten. Together they cover the pipeline from a scan to a
probability map, and both go **through Helena's API**: every step below is a
request, a worker claims the work, and the control reads back what the
deployment reports about itself. Nothing reads a path on a worker's disk.

Steps 1 to 3 are the ink control's steps 1 to 3 and are not repeated here: a
driver and a container runtime, `install.sh --gpu`, and the first account with
its cookie jar. Start from a deployment that has them. Step 4 there, the ink
checkpoint, is not needed: this control runs no ink. The surface-QC checkpoint
P2 needs is fetched by the GPU deploy itself, against the digest the weights
registry pins, before the QC stack starts.

## What it proves, and what it does not

It proves this platform can take a public scroll from its catalogue to a
flattened, CT-supported surface through its own queue and gates, within a
bounded budget of tasks, and that each boundary either passed or said why not.

It is a different kind of claim from the ink control, and the receipt says so.
The ink chain is deterministic modulo the GPU: every ink receipt in this
directory carries one set of statistics. A grow is not. Three runs of one
control -- same seed, same frozen profile, same deployment -- produced three
different surfaces, and that is measured and expected. So this control does
not pass on bytes matching. It passes on what the chain *produced*: at least
one surface that the platform certified and found CT-supported, within the
budget. It records which ones, with their digests, so a second run can be
compared for kind of outcome rather than identity.

It is not a reading, not an ink claim, and not the nine-boundary First Letters
campaign control. The evaluator refuses either substitution by schema, and the
receipt carries the non-claims itself.

## The six boundaries

    PUBLIC_SOURCE  the scroll is in the frozen eligible catalogue, and its CT
                   volume and m7 surface prediction answer anonymous HEADs on
                   .zgroup, 0/.zarray and .zattrs -- credentials_used: false
    INTAKE         the mission holds the scroll; POST .../artifacts/freeze-p0
                   returned a P0 artifact id; the selection names it
    GROW           POST /api/segmentation/runs was queued per tiling, and at
                   least one surface exists within the budget
    GEOMETRY       at least one surface is GEOMETRY_CERTIFIED
    PHYSICAL_QC    at least one is CT_SUPPORTED (or CT_SUPPORTED_REVIEW) under
                   the QC profile the deployment pins by hash; the receipt
                   names the profile and its digest
    FLATTEN        POST /api/flattening/run on one supported surface, and the
                   P3 job published a sheet by digest

The budget is `--max-tasks`, spent across `--grid-steps` in turn. One tiling
is not a fair test of a scroll: the planner hands out the cells its m7
prediction proposes seeds in, and the first run of this control shows why --
one 48-task tiling produced one certified surface, and the CT screen found it
insufficient. So GROW queues a tiling, waits for it to settle, waits for its
QC, and queues the next only while no surface is supported and budget remains.
Every batch is in the receipt.

## 5. Make a mission for the scroll

    curl -sk -b cookies -X POST https://127.0.0.1:8800/api/missions \
      -H 'Content-Type: application/json' \
      -d '{"mission_id":"segmentation-control","name":"Public segmentation control",
           "scrolls":["PHerc826"]}'

PHerc826 is the scroll the passing run used, and the one to start with. Any
scroll in the frozen eligible catalogue is accepted -- `workspace/catalog/
eligible_volumes.json` in the checkout lists thirteen -- and a name the
deployment cannot resolve to a volume is refused here, with the names it does
know, rather than at P1. Spell it as the catalogue does, `PHerc826` and not
`PHerc0826`, in the mission and in `--sample-id` both: the control checks that
the mission holds the name it was given. The runs below show what the other
scrolls did within the same budget: PHerc1203 and PHerc358 grew surfaces the CT
screen found insufficient, and on PHerc125 the planner found no seed at all.
PHerc0139, the ink control's scroll, is not in the catalogue and is not this
control's.

## 6. Run it

    docker run --rm --network host \
      -v <checkout>:/repo:ro -v <output dir>:/out \
      -e HELENA_PANEL_PASSWORD='<the password from step 3>' \
      -e HELENA_PANEL_TLS_INSECURE=1 \
      helena-panel:0.25.1 \
      python /repo/scripts/harness/run_public_segmentation_control.py \
        --panel https://127.0.0.1:8800 --user you --mission segmentation-control \
        --sample-id PHerc826 --max-tasks 144 --output /out

`helena-panel:0.25.1` is the panel image the installer built in step 2, tagged
with the version in `VERSION`; the GPU deploy builds it once more as
`helena-panel:local-<commit>`. `docker images | grep helena-panel` shows the
tags your install produced, and either runs the control, which needs nothing
beyond Python's standard library. `<checkout>` is the directory the installer
cloned into -- the control reads the frozen catalogue from it -- and
`<output dir>` any empty directory of yours.

The P0 freeze and selection are the control's own first requests; they are not
a separate step. `--user` is the account from step 3, and the password comes
from `HELENA_PANEL_PASSWORD` or `--password`; `--cookie-file` takes the jar
from step 3 instead of either (mount it into the container), because the
control signs in through the same `POST /api/session` as `curl` does.

`--max-tasks` (default 144) is the budget, spent across `--grid-steps`
(default `896,1024,768,1152,640,1280`) in turn. `--minutes` (default 90)
bounds each wait -- for a tiling to settle, for its QC, for the flattening job
-- and a slower card may need more of it.

Exit status is 0 on `CONTROL_PASS` and 3 otherwise. A tiling whose cells
mostly end `NO_SEED` settles in a few minutes on one 5090; the passing run's
GROW, 144 tasks with seventeen surfaces grown and screened, took fifty-two
minutes. `NO_SEED` means the planner found no candidate meeting the frozen
clearance policy in that cell, and it is a normal outcome per cell, not a
failure.

Run it once per scroll on a deployment. The queue identifies a task by volume,
grid and cell, not by mission, so a later mission on the same scroll is told
`nothing was queued: all N cells this run covers already have a task` and
grows nothing new. A second run on the same mission queues nothing and reads
what the first produced; its receipt says so with `queued_this_run`, and that
is what `segmentation-run-2026-09-02-f` is.

## What it writes

    PUBLIC_SEGMENTATION_CONTROL.json   the stage-survival receipt, content-addressed
    SURFACES.json                      every surface the mission holds, as the
                                       deployment reports it: states, area, digest

## Reading the receipt

`control_state` is `CONTROL_PASS` only when every boundary passed; the first
non-passing boundary owns the outcome and every row after it is normalised to
`NOT_RUN_PREREQUISITE`. The same rule as the ink control, applied to a
different schema -- `campaignx.public_segmentation_stage_survival.v1` -- so one
receipt cannot be read as the other.

* `stages[GROW].resource_identity.batches` lists every tiling queued: the grid
  step, how many tasks it inserted, the task states when it settled, and how
  many surfaces were supported by then. `inserted_total` is what the budget
  actually bought.
* `stages[PHYSICAL_QC].resource_identity.profiles` names the QC profile and
  its digest as the deployment ran it. A verdict under a different profile is
  a different measurement.
* `stages[PHYSICAL_QC].output_hashes` are the supported surfaces by digest, and
  `stages[FLATTEN].input_artifacts` names the one that was flattened.
* `stages[PUBLIC_SOURCE].resource_identity.credentials_used` is `false`.
* `stages[GROW].resource_identity.queued_this_run` and the same field under
  `FLATTEN` say whether this run queued the work or found it already in the
  mission. A first run on a mission queues; a second reads, and reports the
  mission's own task states as `mission_task_states`. Neither is hidden.
* `content_sha256` is the SHA-256 of the receipt itself, serialised with
  sorted keys and no whitespace before that field was added, as in the ink
  control.

## Runs

    2026-09-02  CONTROL_INCOMPLETE  PHYSICAL_QC  PHerc1203  48 tasks, grid 896    segmentation-run-2026-09-02-a
    2026-09-02  CONTROL_INCOMPLETE  PHYSICAL_QC  PHerc1203  144 tasks, six grids  segmentation-run-2026-09-02-b
    2026-09-02  CONTROL_INCOMPLETE  GROW         PHerc125   144 tasks, six grids  segmentation-run-2026-09-02-c
    2026-09-02  CONTROL_INCOMPLETE  PHYSICAL_QC  PHerc358   144 tasks, six grids  segmentation-run-2026-09-02-d
    2026-09-02  CONTROL_INCOMPLETE  FLATTEN      PHerc826   144 tasks, one tiling   segmentation-run-2026-09-02-e
    2026-09-02  CONTROL_PASS        -            PHerc826   the -e mission, read again  segmentation-run-2026-09-02-f

Runs `-e` and `-f` are one mission on one fresh machine. `-e` grew it -- 144
tasks, seventeen surfaces, four CT-supported, one flattened -- and then misread
the flattening job's result, a bug in the control and not in the platform.
`-f` is the same mission read again after that fix: it queued nothing, and its
receipt says so with `queued_this_run: 0`. The pass is `-f`'s, the work is
`-e`'s, and both receipts are kept.
