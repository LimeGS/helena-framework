# Reproducing the public segmentation control

The other half of the public ink control. `REPRODUCE.md` drives P4 to P7 --
render, ink, screening -- on a volume read anonymously from the open-data
bucket. This drives P0 to P3: intake, grow, certify the geometry, screen for
physical support, flatten. Together they cover the pipeline from a scan to a
probability map, and both go **through Helena's API**: every step below is a
request, a worker claims the work, and the control reads back what the
deployment reports about itself. Nothing reads a path on a worker's disk.

Steps 1 to 4 are the ink control's steps 1 to 4 and are not repeated here: a
driver and a container runtime, `install.sh`, the first account, and the QC
checkpoint the installer places. Start from a deployment that has them.

## What it proves, and what it does not

It proves this platform can take a public scroll from its catalogue to a
flattened, CT-supported surface through its own queue and gates, within a
bounded budget of tasks, and that each boundary either passed or said why not.

It is a different kind of claim from the ink control, and the receipt says so.
The ink chain is deterministic modulo the GPU: five runs, one set of
statistics. A grow is not. Three runs of one control -- same seed, same frozen
profile, same deployment -- produced three different surfaces, and that is
measured and expected. So this control does not pass on bytes matching. It
passes on what the chain *produced*: at least one surface that the platform
certified and found CT-supported, within the budget. It records which ones,
with their digests, so a second run can be compared for kind of outcome rather
than identity.

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
           "scrolls":["PHerc1203"]}'

Any scroll the catalogue lists works; a name the deployment cannot resolve to
a volume is refused here, with the names it does know, rather than at P1.

## 6. Run it

    docker run --rm --network host \
      -v <checkout>:/repo:ro -v <output dir>:/out \
      -e HELENA_PANEL_PASSWORD='<the password from step 3>' \
      -e HELENA_PANEL_TLS_INSECURE=1 \
      helena-panel:0.24.1 \
      python /repo/scripts/harness/run_public_segmentation_control.py \
        --panel https://127.0.0.1:8800 --user you --mission segmentation-control \
        --sample-id PHerc1203 --max-tasks 144 --output /out

The P0 freeze and selection are the control's own first requests; they are not
a separate step. `--cookie-file` takes a jar from a session already signed in
(`curl -c`) in place of `--user` and a password.

Exit status is 0 on `CONTROL_PASS` and 3 otherwise. A batch of 48 tasks
settles in a few minutes on one 5090; most tasks end `NO_SEED` -- the planner
found no candidate meeting the frozen clearance policy in that cell -- and
that is a normal outcome per cell, not a failure.

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

## Runs

    2026-09-02  CONTROL_INCOMPLETE  PHYSICAL_QC  PHerc1203  48 tasks, grid 896    segmentation-run-2026-09-02-a
    2026-09-02  CONTROL_INCOMPLETE  PHYSICAL_QC  PHerc1203  144 tasks, six grids  segmentation-run-2026-09-02-b
    2026-09-02  CONTROL_INCOMPLETE  GROW         PHerc125   144 tasks, six grids  segmentation-run-2026-09-02-c
    2026-09-02  CONTROL_INCOMPLETE  PHYSICAL_QC  PHerc358   144 tasks, six grids  segmentation-run-2026-09-02-d
    <PHERC826_FINAL_RUN_LINE>
