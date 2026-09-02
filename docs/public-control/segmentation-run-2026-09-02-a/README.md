# The segmentation control, one tiling, on a rented GPU

The first run of the other half of the public control: P0 to P3 through the
panel's API on a rented RTX 5090, with a budget of 48 tasks on one tiling.
The receipt predates the harness spending its budget across tilings, so its
`budget` names a single `grid_step` and GROW carries no `batches` list.

    control_state: CONTROL_INCOMPLETE   first_nonpassing_boundary: PHYSICAL_QC
    content_sha256: 5456143b08db54e03804133e3178d69c6ddfc8973f72ff37ba0f5a445a453646

    PUBLIC_SOURCE  PASS                  PUBLIC_SOURCE_READ_ANONYMOUSLY    32.7s
    INTAKE         PASS                  P0_FROZEN_AND_SELECTED             0.0s
    GROW           PASS                  SURFACES_PRODUCED                 61.9s
    GEOMETRY       PASS                  GEOMETRY_CERTIFIED                 0.0s
    PHYSICAL_QC    INCOMPLETE            NO_CT_SUPPORTED_SURFACE          330.5s
    FLATTEN        NOT_RUN_PREREQUISITE  PREREQUISITE_NOT_REACHED           0.0s

    through:  helena-queue
    mission:  segmentation-control
    p0:       p0:PHerc1203:1c757dcf0d6b
    surface:  8141e6d1-1ca0-5d8a-a760-8412b0bd0cdd  d3b53276…

## What it shows

Everything went through the API: the P0 freeze and selection, the grow queued
with `through: helena-queue` in GROW's resource identity, and the surfaces,
task counts and QC jobs read back as the deployment reports them. Both public
sources answered anonymous HEADs with 200 on all three keys; `credentials_used`
is false.

GROW asked for 48 tasks on grid 896; the planner generated 48 and inserted 48.
When they settled, the receipt's task counts read NO_SEED 220 and QC_PENDING
7. Those are the whole fleet's, not this mission's: `/api/fleet` accepted
`?mission=` and ignored it at the time, a bug found by this run and fixed since
(the harness now refuses a fleet-wide answer). The mission's own 48 tasks ended
47 NO_SEED and 1 QC_PENDING, and it held one surface, 8141e6d1…, 1.584 cm². GEOMETRY certified it, 1 of 1. PHYSICAL_QC ran
one QC job to COMPLETED under `surface-qc-gp-scroll1-ct-fiber-v3@1.0.0`
(e0e099ea…) and found it INK_SCREEN_INSUFFICIENT: 0 of 1 supported. The control
stopped there, and FLATTEN was not run.

One tiling is not a fair test of a scroll, and this run is why the harness now
spends its budget across six grid steps in turn.
