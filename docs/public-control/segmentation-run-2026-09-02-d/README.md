# Four surfaces, all certified, none supported

The control on PHerc358 with 144 tasks over six grid steps. It is the run that
produced the most surfaces, and every one of them failed the same screen.

    control_state: CONTROL_INCOMPLETE   first_nonpassing_boundary: PHYSICAL_QC
    content_sha256: f77b93ac47a05f3f428313a6f11faf084f01e1d32e831c01bad66fe7fcd8b02d

    PUBLIC_SOURCE  PASS                  PUBLIC_SOURCE_READ_ANONYMOUSLY     2.6s
    INTAKE         PASS                  P0_FROZEN_AND_SELECTED             0.0s
    GROW           PASS                  SURFACES_PRODUCED                302.4s
    GEOMETRY       PASS                  GEOMETRY_CERTIFIED                 0.0s
    PHYSICAL_QC    INCOMPLETE            NO_CT_SUPPORTED_SURFACE            0.0s
    FLATTEN        NOT_RUN_PREREQUISITE  PREREQUISITE_NOT_REACHED           0.0s

    through:        helena-queue
    mission:        segmentation-control-358
    p0:             p0:PHerc358:39b671c94eb1
    batches:        1 -- grid 896, generated 144, inserted 144
    inserted_total: 144 of 144
    surfaces:       0cb4fefd… fa263213…    a35deafe… d444cd60…
                    be7f4c0c… 65b7776a…    df1f6dca… 58a30d85…

## What it shows

GROW queued 144 tasks on grid 896 and the planner inserted all 144, spending
the budget on the first tiling. They settled in time with the mission's tasks
at NO_SEED 140 and QC_PENDING 4, and the mission held four surfaces, each
1.584 cm² as the deployment reports them. GEOMETRY certified all four, 4 of 4.
PHYSICAL_QC ran four QC jobs to COMPLETED under
`surface-qc-gp-scroll1-ct-fiber-v3@1.0.0` (e0e099ea…) and found every one
INK_SCREEN_INSUFFICIENT: 0 of 4 supported. The control stopped there, and
FLATTEN was not run.

Everything went through the API (`through: helena-queue` in GROW's resource
identity): the P0 freeze and selection, the queued grow, and the surfaces, task
counts and QC jobs read back as the deployment reports them. Four grows, four
certifications and one screen that refused all of them under one pinned
profile is the outcome this control is built to record. It is not a pass.
