# A scroll on which the planner found no seed

The control on PHerc125, with the full budget of 144 tasks over six grid
steps. It is the first run to stop at GROW rather than at the CT screen.

    control_state: CONTROL_INCOMPLETE   first_nonpassing_boundary: GROW
    content_sha256: 974bef73ee0b6be7c5c239a92a5fc2f9b718994651b3bb35d63c4a493d19ffae

    PUBLIC_SOURCE  PASS                  PUBLIC_SOURCE_READ_ANONYMOUSLY     2.8s
    INTAKE         PASS                  P0_FROZEN_AND_SELECTED             0.0s
    GROW           INCOMPLETE            NO_SURFACE_WITHIN_BUDGET         152.1s
    GEOMETRY       NOT_RUN_PREREQUISITE  PREREQUISITE_NOT_REACHED           0.0s
    PHYSICAL_QC    NOT_RUN_PREREQUISITE  PREREQUISITE_NOT_REACHED           0.0s
    FLATTEN        NOT_RUN_PREREQUISITE  PREREQUISITE_NOT_REACHED           0.0s

    through:        helena-queue
    mission:        segmentation-control-125
    p0:             p0:PHerc125:2f3749abb8ea
    batches:        1 -- grid 896, generated 144, inserted 144
    inserted_total: 144 of 144

## What it shows

Both public sources for PHerc125 answered anonymous HEADs with 200 on all three
keys, and P0 froze and selected the scroll. GROW asked for 144 tasks on grid
896; the planner generated and inserted all 144, which spent the whole budget
on the first tiling, so no second grid was queued. They settled in time, and
every one of the 144 ended NO_SEED: the planner found no candidate meeting the
frozen clearance policy in any cell. No surface, so GROW is INCOMPLETE with
NO_SURFACE_WITHIN_BUDGET, and the three boundaries after it were not run. There
is no `SURFACES.json`, because it is written at PHYSICAL_QC and the control did
not get there.

Everything went through the API (`through: helena-queue` in GROW's resource
identity). A NO_SEED is a normal outcome per cell; 144 of 144 is the whole
budget answered by one grid, with the five other steps never asked.
