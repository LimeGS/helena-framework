# The segmentation control, on a machine installed with one command

The passing run, and the second read of one mission. The machine is new --
`install.sh --gpu` from the published repository, nothing edited by hand -- and
the mission is the one `segmentation-run-2026-09-02-e` grew through the panel's
API against PHerc826 with a budget of 144 tasks. This run queued nothing: it
read that mission again after the control's FLATTEN bug was fixed, and its
receipt says so.

    control_state: CONTROL_PASS       first_nonpassing_boundary: null
    content_sha256: 378dbaa05f80876f5e93e0e2ffaee50b855df631cacffda6771b2773d829c94c

    PUBLIC_SOURCE  PASS                   PUBLIC_SOURCE_READ_ANONYMOUSLY          2.5s
    INTAKE         PASS                   P0_FROZEN_AND_SELECTED                  0.0s
    GROW           PASS                   SURFACES_HELD_BY_THE_MISSION            0.1s
    GEOMETRY       PASS                   GEOMETRY_CERTIFIED                      0.0s
    PHYSICAL_QC    PASS                   CT_SUPPORTED_SURFACE                    0.1s
    FLATTEN        PASS                   SHEET_PUBLISHED_BY_DIGEST               0.0s

    through:   helena-queue
    tasks:     {'NO_SEED': 127, 'QC_PENDING': 17} (the mission's, queued by the run in segmentation-run-2026-09-02-e)
    surfaces:  17 produced, 17 certified, 4 CT-supported
    qc:        surface-qc-gp-scroll1-ct-fiber-v3@1.0.0
    flattened: 836fc053-ac0c-5579-aa4a-4eda38c1f29f by p3-a884ab5e0e7749
    sheet:     70e92177054645471a1cac0567dd86b28d74da4e5820d1fa99a9a51b0f690873

This is the same mission as `segmentation-run-2026-09-02-e`, read correctly. That run
grew the surfaces, measured them and flattened one, and then misread the
flattening job's result; this one queued nothing (`queued_this_run: 0` in GROW
and `false` in FLATTEN), took the mission's own task and QC records as the
deployment reports them, and reports the sheet that job had already published,
by digest, with its source surface's digest checked against the surface. The
receipt says so in its own fields rather than leaving it to a reader.

Every step was a request; `SURFACES.json` beside this file is the deployment's
own report of the seventeen surfaces, verbatim.
