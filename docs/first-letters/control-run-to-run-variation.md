# Two runs of the same control are not the same run

Measured 2026-08-06/07 on gpu-1, PHerc0139, control anchor cell `[156,234]`.

## What was observed

Three runs of the positive control, same anchor, same frozen profile, same
model, same deployment. Each produced a different surface:

| when | surface_id | area cm² | QC outcome |
|---|---|---|---|
| 16:50 | `b25d6d73…` | 1.5842569 | `CT_SUPPORTED_RETAINED_FOR_REVIEW` |
| 00:39 | `4c678035…` | 1.5843012 | `CT_SUPPORTED_RETAINED_FOR_REVIEW` |
| 15:04 | `d74b8cfe…` | 1.5842086 | `CT_SUPPORTED_NO_RETAINED_INK_SIGNAL` |

Different `surface_id`, different `artifact_sha256`, areas differing in the
sixth decimal. The ink screening saw it too: common valid pixels came out
2 213 406 / 2 238 363 / 2 250 980, about 1.7% apart.

## Why

The grow is a search, not a function. The same seed coordinate under the same
policy explores and settles slightly differently each time, so it yields a
different surface. Everything downstream inherits that: a different surface
flattens differently, renders a different layer stack, and screens to a
different probability map.

Nothing about the code, the data, the model or the profile changed between
those three runs. The variation is the pipeline being what it is.

## What did *not* vary

The liveness verdict. All three runs came back `ALIVE`:

```
spread p99 − p50:   0.391    0.391    0.319
```

What differed is whether anything crossed the **retention** threshold, which is
a separate gate from liveness. Two runs retained something for human review; the
third — lowest spread of the three — did not. It sits near the threshold and
falls either side depending on the surface.

For contrast, the four runs at the *previous* anchor `[182,170]` were all
`DEGENERATE`, spread 0.047 to 0.084. That is a different phenomenon entirely and
is recorded in the 1.1.0 profile's `anchor_revision_note`: a region of low CT
density where the model returns a constant map rather than a decision. Do not
confuse the two. Run-to-run variation moves the third decimal; the anchor
problem moved the verdict.

## What this means, and what it does not

**It does not affect whether the control passes.** Both outcomes are
`CT_SUPPORTED*`, and the frozen manifest admits both downstream:

```
terminal_physical_qc_compatible_with_downstream: [CT_SUPPORTED, CT_SUPPORTED_REVIEW]
```

`P5_LIVE_OUTPUT`, which is a pass requirement, asks the detector to *decide*. It
decided all three times.

**It does mean the control is not bit-reproducible.** Two receipts of the same
control will differ — different surface ids, different hashes, sometimes a
different QC outcome. The manifest never claimed otherwise, but the receipts
look identical enough that a difference invites the wrong conclusion.

So: **a differing `content_sha256` between two control receipts is not evidence
that anything changed.** Before reading a difference as a regression, check
whether the surface ids differ. If they do, you are looking at two grows, not at
a change in behaviour.

## The trap this exists to prevent

Narrowing `terminal_physical_qc_compatible_with_downstream` to only
`CT_SUPPORTED_REVIEW` would look harmless — every run that had been examined at
the time retained something — and would make the control fail intermittently,
for reasons that would look like a science problem and would not be one.
`tests/test_the_control_tolerates_its_own_variation.py` fails if either observed
outcome stops being admissible, or if a pass requirement starts keying on
retention.
