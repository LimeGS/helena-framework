# The first run on the fresh machine, stopped by the control's own bug

A new machine, `install.sh --gpu` from the published repository, nothing edited
by hand, the control driven through the API against PHerc826 with a budget
of 144 tasks. Five boundaries passed. The sixth reported the
flattening job as having published nothing -- while the job had succeeded and
the sheet was on the artifact volume with its digest beside it.

    control_state: CONTROL_INCOMPLETE first_nonpassing_boundary: FLATTEN
    content_sha256: d21cca31305207e4298088116e032186ed7894c263e6cddfd14e0e359d52f6dc

    PUBLIC_SOURCE  PASS                   PUBLIC_SOURCE_READ_ANONYMOUSLY          2.6s
    INTAKE         PASS                   P0_FROZEN_AND_SELECTED                  0.1s
    GROW           PASS                   SURFACES_PRODUCED                    3127.5s
    GEOMETRY       PASS                   GEOMETRY_CERTIFIED                      0.0s
    PHYSICAL_QC    PASS                   CT_SUPPORTED_SURFACE                    0.1s
    FLATTEN        INCOMPLETE             FLATTEN_DID_NOT_PUBLISH                15.1s

    through:   helena-queue
    batches:   grid 896: 144 tasks
    surfaces:  17 produced, 17 certified, 4 CT-supported
    qc:        surface-qc-gp-scroll1-ct-fiber-v3@1.0.0

The control read `result.artifact_sha256`, a field no P3 job has; a P3 job
reports what it flattened under `result.surfaces`, digest beside each entry.
The control reads that now, and the run in `segmentation-run-2026-09-02-f` is
the same mission read correctly: it queued nothing, measured nothing again,
and reports the sheet the job here had already published.

Two more things this run found on a fresh install, both fixed before it: the
QC runtime crash-looped on a database name wrapped twice, and its first job
died on a run directory Docker had created as root. Neither is a control
boundary; both are in the receipts of the platform's own tests.
