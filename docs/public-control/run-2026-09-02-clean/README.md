# The ink control, on the same fresh machine as the segmentation control

The same machine as `segmentation-run-2026-09-02-f`: installed with one command from
the published repository, the checkpoint placed through the models API against
its digest, the mission and its P0 selection made as requests, the job queued.

    control_state: CONTROL_PASS       first_nonpassing_boundary: null
    content_sha256: 532f0591c716aee277d486927b8ce71489b676ad46c9a8f22b27d8e1dcbc0d56

    PUBLIC_SOURCE  PASS                   PUBLIC_SOURCE_READ_ANONYMOUSLY          0.6s
    SCALE          PASS                   NATIVE_MODEL_SCALE                      0.0s
    CHECKPOINT     PASS                   CHECKPOINT_IS_THE_DECLARED_ONE          0.0s
    INK            PASS                   PROBABILITY_MAP_WRITTEN              1460.0s
    LIVENESS       PASS                   MAP_CARRIES_A_DECISION                  1.2s
    HUMAN_REVIEW   PASS                   ROUTED_TO_REVIEW_WITHOUT_A_CLAIM        0.0s

    through:  helena-queue
    job_id:   p5-6df06c3f10674e
    artifact: /artifacts/ink-maps-v1/surfaces/PHerc0139/ink-maps/p5-6df06c3f10674e
    liveness: p50 0.2784  p99 0.7765  std 0.1674

The same statistics as every run before it, to four decimals, on a different
set of bytes: the digest is a fair check only against a run on the same card.
