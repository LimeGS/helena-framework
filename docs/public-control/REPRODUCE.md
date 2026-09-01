# Reproducing the public ink control

Everything below runs from inputs anybody can download. No account, no
credential, no access to this deployment. That is the point: the reviewers of
the community-projects PR asked for "a run from a clean installation with
public input surfaces, a checkpoint others can obtain, downloadable outputs,
and an end-to-end test log without skipped pipeline stages", and a control that
reads a private bucket cannot answer any of it however well it works.

## What it proves, and what it does not

It proves this platform can drive the recommended ink-detection tooling end to
end on public data, and that each boundary either passed or said why not.

It is not a reading, not an ink claim, and not the nine-boundary First Letters
campaign control -- that is a different receipt with a different schema, and
the evaluator refuses the substitution. The receipt carries these as
`non_claims` rather than leaving them to a reader.

## Inputs, and where they come from

| Input | Source | Credential |
|---|---|---|
| Surface volume, 9.362 um isotropic | `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr` | none |

The https form, not the `s3://` one. This table gave the `s3://` URI and the
command below says "the URL above", so following this document literally
failed at the first boundary with `unknown url type: s3`: the control reads
the volume's own `.zattrs` over plain HTTP, deliberately anonymous, because a
control that proves "public" while holding credentials has proved nothing
about what a stranger can obtain. The lane reads the same form and not the
other one either. Same object, one address, both halves agree.
| Checkpoint `hybrid_3d2d-seed42/step-075000.pth` | `https://huggingface.co/scrollprize/ink_9um` | none, not gated |

The checkpoint's SHA-256 is `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`
and its size is 138360039 bytes. The control verifies both before running, and
records the digest it saw in the receipt.

## Building the runtime

    make -C containers/images build-ink-9um \
      VILLA_PYTHON_BASE_IMAGE=<minimal base, sha256-pinned> \
      VILLA_INK_SRC=<checkout of ScrollPrize/villa at 3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e> \
      UV_CONTEXT=<directory containing a uv binary>

The image installs upstream's own locked dependency set with `uv sync --frozen`
and verifies, at build time, that the files the adapters were written against
are byte-identical to the ones the source lock records. See
`containers/images/BUILD_STATE.md`.

## Running it

    docker run --rm --gpus all \
      -v <repo>:/repo:ro -v <checkpoint dir>:/models:ro -v <output dir>:/out \
      helena-ink-9um:local \
      python /repo/scripts/harness/run_public_ink_control.py \
        --surface-volume <the URL above> \
        --checkpoint /models/step-075000.pth \
        --expected-checkpoint-sha256 e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab \
        --output /out

Exit status is 0 on `CONTROL_PASS` and 3 otherwise. The run takes about
twenty-five minutes on two GTX 1660s and needs roughly 2.3 GB of VRAM.

## What it writes

    PUBLIC_INK_CONTROL.json   the stage-survival receipt, content-addressed
    INK_9UM_RECEIPT.json      the lane receipt: checkpoint, argv, liveness
    probability.npy           the map, raw
    ink.tif, ink_reverse.tif  upstream's own uint8 output, both directions

## Reading the receipt

`control_state` is `CONTROL_PASS` only when every boundary passed. The first
non-passing boundary owns the outcome and every row after it is normalised to
`NOT_RUN_PREREQUISITE`, so a receipt cannot show a later boundary passing over
an earlier failure.

Two fields are worth reading directly:

* `stages[PUBLIC_SOURCE].resource_identity.credentials_used` is `false`. The
  reader is anonymous by construction; a control that proved "public" while
  sending credentials would have proved nothing about what a stranger can
  obtain.
* `stages[LIVENESS].counts.p50` should sit near 0.25. That is this recipe's
  documented no-ink floor -- it trains with BCE label smoothing 0.5 -- so a map
  whose p50 is 0.000 is empty, not no-ink. The distinction is not cosmetic: a
  misconfigured batch size produced exactly that, exit 0 and all, and the
  liveness gate is what caught it.

## A run from 2026-08-23

    control_state: CONTROL_PASS      first_nonpassing_boundary: null
    content_sha256: 271487453312be12e4a95e6e5d33bd395ca8a4b163b28a4151f02ae312702eef

    PUBLIC_SOURCE  PASS  PUBLIC_SOURCE_READ_ANONYMOUSLY        1.4s
    SCALE          PASS  NATIVE_MODEL_SCALE                    0.0s
    CHECKPOINT     PASS  CHECKPOINT_IS_THE_DECLARED_ONE        0.3s
    INK            PASS  PROBABILITY_MAP_WRITTEN            1549.3s
    LIVENESS       PASS  MAP_CARRIES_A_DECISION                1.4s
    HUMAN_REVIEW   PASS  ROUTED_TO_REVIEW_WITHOUT_A_CLAIM      0.0s

    liveness: p50=0.278  p99=0.776  std=0.1674
