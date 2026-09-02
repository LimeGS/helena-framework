# The same scroll, with the budget spread across tilings

The run beside this one showed that one tiling is not a fair test. This one
gave the harness 144 tasks over six grid steps -- 896, 1024, 768, 1152, 640,
1280 -- on the same mission, and it is the run that found the harness bug.

    control_state: CONTROL_INCOMPLETE   first_nonpassing_boundary: PHYSICAL_QC
    content_sha256: d9fcd79648eb78b11ec071e9e2cd7a5b23336676dbae69ea1740242f388ee9d7

    PUBLIC_SOURCE  PASS                  PUBLIC_SOURCE_READ_ANONYMOUSLY     2.6s
    INTAKE         PASS                  P0_FROZEN_AND_SELECTED             0.1s
    GROW           PASS                  SURFACES_PRODUCED                124.1s
    GEOMETRY       PASS                  GEOMETRY_CERTIFIED                 0.0s
    PHYSICAL_QC    INCOMPLETE            NO_CT_SUPPORTED_SURFACE            0.0s
    FLATTEN        NOT_RUN_PREREQUISITE  PREREQUISITE_NOT_REACHED           0.0s

    through:        helena-queue
    mission:        segmentation-control
    batches:        1 -- grid 896, generated 144, inserted 134
    inserted_total: 134 of 144
    refused:        grid 1024, HTTP 409

## What it shows

The first tiling, grid 896, generated 144 tasks and inserted 134. They settled
in time, with the mission's tasks at NO_SEED 181 and QC_PENDING 1, and produced
no surface: `surfaces_so_far` is 1, and it is run a's surface, 8141e6d1…, again
GEOMETRY_CERTIFIED and again INK_SCREEN_INSUFFICIENT under the same pinned
profile -- `SURFACES.json` here is byte-identical to run a's. 0 of 1 supported,
so PHYSICAL_QC is where it stopped.

The second tiling, grid 1024, was answered HTTP 409: "nothing was queued: all
10 cells this run covers already have a task under grid ct-l0-v1 and policy
ink-blind-v1". The harness of that time recorded it under `refused` and
stopped, with four tilings untried and 10 tasks of budget unspent. That was a
harness bug, since fixed: a 409 is the queue saying this grid has nothing left
to grow, and the harness now records the tiling as already covered and moves
to the next one. The receipt is kept as it was written.

Everything went through the API, as `through: helena-queue` in GROW's resource
identity says; nothing read a path on the worker's disk.
