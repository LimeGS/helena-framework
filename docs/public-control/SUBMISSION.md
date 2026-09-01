# What was asked for, and where each piece is

The review closed with five things needed before an independent verification is
possible:

> a run from a clean installation with public input surfaces, a checkpoint
> others can obtain, downloadable outputs, and an end-to-end test log without
> skipped pipeline stages

This page answers them one at a time. Where something is not yet answered it
says so rather than pointing at the nearest thing that is.

The PHerc0826 golden run is **not** offered as the answer. It is an audit record
of a campaign that ran on older tooling, it reads surfaces out of a private
bucket, and no amount of presentation makes it reproducible by a stranger. It
stays in the tree as what it is: a dated record.

## 1. Public input surfaces

The surface volume is in the Vesuvius Challenge open-data bucket and is read
over plain HTTPS, anonymously, with no credential in the process:

    https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/
      segments/20260112000000-w043_2026011217/surface-volumes/
      9.362um-1.2m-113keV-volume-20250728140407.zarr

The receipt records this as a boundary of its own -- `PUBLIC_SOURCE` -- and
carries `stages[PUBLIC_SOURCE].resource_identity.credentials_used: false`. A
control that proved "public" while holding credentials would have proved
nothing about what a stranger can obtain, so the claim is checked rather than
asserted.

## 2. A checkpoint others can obtain

`hybrid_3d2d-seed42/step-075000.pth` from `https://huggingface.co/scrollprize/ink_9um`
-- a public, non-gated repository. 138360039 bytes, SHA-256

    e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab

The control verifies the digest before running and records what it saw. A
checkpoint whose digest is not the declared one fails the job instead of being
used.

## 3. A run from a clean installation

`REPRODUCE.md` in this directory is the whole procedure: the image build from
public sources, the single `docker run`, and what it writes. Nothing in it
touches this deployment, and no step needs an account.

The runtime is built from upstream's own locked dependency set, and the build
verifies that the files the adapters were written against are byte-identical to
the ones the source lock records.

Being exact about what this particular run proves: it reused an ink-9um image
built earlier on the same host from those same pinned sources, rather than
building one from nothing in the same sitting. The procedure in `REPRODUCE.md`
is the one a stranger runs and is what the image was built by; this run did not
re-execute its build step. Anyone re-running it from scratch should land on the
same digests, which is the point of the section below.

## 4. Downloadable outputs

Committed here, small enough to read in a browser:

In `run-2026-09-01/`, small enough to read in a browser:

| File | What it is |
|---|---|
| `PUBLIC_INK_CONTROL.json` | the stage-survival receipt, content-addressed |
| `INK_9UM_RECEIPT.json` | the lane receipt: checkpoint, argv, liveness |
| `OUTPUTS.sha256` | digests of the four maps below |

The maps are too large for a git tree and are attached to the release, so a
download can be checked against `OUTPUTS.sha256`: `probability.npy` and
`probability_reverse.npy` (198777728 bytes each), `ink.tif` (26603464),
`ink_reverse.tif` (23886713).

## The run

    control_state: CONTROL_PASS      first_nonpassing_boundary: null
    content_sha256: d8278ebdfe2341d6627bd72769efc8e7aedfb70e0ad54a33b1b382ea3aab3e41

    PUBLIC_SOURCE  PASS  PUBLIC_SOURCE_READ_ANONYMOUSLY        0.8s
    SCALE          PASS  NATIVE_MODEL_SCALE                    0.0s
    CHECKPOINT     PASS  CHECKPOINT_IS_THE_DECLARED_ONE        0.3s
    INK            PASS  PROBABILITY_MAP_WRITTEN            1475.9s
    LIVENESS       PASS  MAP_CARRIES_A_DECISION                2.3s
    HUMAN_REVIEW   PASS  ROUTED_TO_REVIEW_WITHOUT_A_CLAIM      0.0s

    liveness: p50=0.2784  p99=0.7765  std=0.1674

`p50` near 0.25 is this recipe's documented no-ink floor -- it trains with BCE
label smoothing 0.5 -- so a map whose p50 is 0.000 is empty, not no-ink. The
distinction is not cosmetic: a misconfigured batch size produced exactly that,
exit 0 and all, and the liveness gate is what caught it.

## Reproducibility, checked rather than claimed

`probability.npy`, `ink.tif` and `ink_reverse.tif` are **bit-identical** to a
run made nine days earlier on the same inputs -- same SHA-256, three for three.
The two runs were made from different working trees on different days. What
that demonstrates is narrow and worth stating precisely: the inference path is
deterministic given the same volume, checkpoint and scale, so a third party who
follows `REPRODUCE.md` can compare digests rather than eyeball two images.

## 5. An end-to-end test log without skipped pipeline stages

Both logs are in `e2e-2026-09-01/`, run against a deployment on 2026-09-01.

| Log | Result |
|---|---|
| `e2e-full.log` | 55 passed, 3 skipped, 94s |
| `e2e-heavy.log` | 2 passed, 56 deselected, 980s |

They are two jobs on purpose. The full suite runs everything that does not need
a GPU day; the heavy job runs the two that do, under `HELENA_E2E_HEAVY=1`, and
those two are two of the three skips in the first log. Between them 57 of the
58 tests ran.

**The remaining skip, stated plainly rather than buried.** One test is skipped
in both:

    tests/e2e/test_the_gates_hold_on_the_deployment.py:101
    no surface carries a human review on this control plane

It checks that a human verdict does not overwrite a geometry verdict, and it
skips because no surface on this deployment carries a human review — not
because a phase did not run. Every phase ran.

It could be made to pass by recording a review on a surface, and that was
deliberately not done: the record would be a human judgement attributed to an
account, created so that a test would not skip. A green log bought that way is
worth less than a log with one honest skip in it.

The suite refuses to pass by skipping everything: under `HELENA_E2E_NO_SKIP=1`
a run where nothing asserted is turned red, which is what the heavy job runs
under.
