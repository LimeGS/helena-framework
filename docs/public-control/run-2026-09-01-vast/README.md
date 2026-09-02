# The control, driven through Helena, on a machine that was not ours

Everything before this ran the ink step as a local subprocess: the receipt said
`"through": "local-subprocess"`, and the script's own comment says that proves
the tooling and nothing about the queue. This one went through the panel's API
on a rented RTX 5090 installed from the public repository.

    control_state: CONTROL_PASS       first_nonpassing_boundary: null
    content_sha256: 2fedfe4935aa464bde17912d60e71f301cf86327cdad4b6549e31eedbb653eb4

    PUBLIC_SOURCE  PASS  PUBLIC_SOURCE_READ_ANONYMOUSLY        0.5s
    SCALE          PASS  NATIVE_MODEL_SCALE                    0.0s
    CHECKPOINT     PASS  CHECKPOINT_IS_THE_DECLARED_ONE        0.1s
    INK            PASS  PROBABILITY_MAP_WRITTEN            1655.3s
    LIVENESS       PASS  MAP_CARRIES_A_DECISION                1.1s
    HUMAN_REVIEW   PASS  ROUTED_TO_REVIEW_WITHOUT_A_CLAIM      0.0s

    through:  helena-queue
    job_id:   p5-1ada8a67cfe743
    artifact: /artifacts/ink-maps-v1/surfaces/PHerc0139/ink-maps/p5-1ada8a67cfe743

## What went through the API, and what did not

Through it: the account, the mission, the P0 artifact and its selection, the
queued job, and the map itself -- fetched back by the `artifact_uri` and
`artifact_sha256` the worker published, not read off its disk. That is the part
the earlier receipts could not show.

Not through it, and worth saying rather than leaving to be found:

* The driver, the container toolkit and the installer are host setup, and the
  installer is not the API either.

Two things that were not, when this run was made, are now. The checkpoint was
written into the `helena-models` volume with a `docker run`, because the models
API refused both a `.pth` and a subdirectory; it accepts a pickle against a
stated hash now, and the path, so the placement is a request like the rest. And
the `CHECKPOINT` boundary hashed that file off the volume, which is a claim
about the machine the control ran on rather than the one that ran the job; it
asks the deployment now, and the receipt says which way with
`established_by`. The receipt below predates both, and says `local-file`.

An earlier version of this page said nothing was done except through the API and
that no path on the worker's disk was read. Both were wrong.

## What differed, and what did not

The map's bytes are not the ones our own GPU produces -- `f4abc1b4…` here,
`fc9f91da…` on the two runs of ours. Its statistics are the same to four
decimals: p50 0.2784, p99 0.7765, std 0.1674. Floating-point accumulation
differs between cards; the result does not.

That is worth saying because the earlier page offered digest comparison as the
check. On one GPU it is a good one. Across two it is not, and a reviewer who
reproduced this and got a different hash would be right to distrust a page that
had promised otherwise.

## What it cost to get here

Twenty-five things had to be fixed between a clean machine and this receipt, and
every one of them worked on our fleet. The install script's `--gpu` could not
build what it needed; the GPU runtime's base had drifted from the one villa
compiles on; the env templates carried our absolute paths, our GPU count, our
control plane and two placeholder passwords; two lane virtualenvs did not
survive being relocated; and the control itself sent a parameter the panel owns
and omitted the two it requires. The commits between p5… and this file
name them one at a time.
