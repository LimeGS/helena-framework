# Terminal Surface-QC Liveness Refusal Design

## Problem

The surface-QC adapter launches `run_ink_timesformer.py` with its default
`--on-degenerate fail` policy. The runner correctly preserves its screening
receipt and exits `3` when the aggregate map is `DEGENERATE` or `EMPTY`.
`run_logged` then converts that declared scientific refusal into a generic
`RuntimeError`; the outer adapter exits `1`; and the QC worker classifies the
attempt as `RETRYABLE_QC_UNAVAILABLE`. The same immutable surface, model,
checkpoint and parameters therefore repeat indefinitely even though waiting
cannot change the verdict.

The production diagnostic for surface
`99fd9127-548b-52bd-991b-ad6e7277db0c` exposed
`RuntimeError: command failed with exit code 3: <path>`. Static call tracing and
local tests prove that the only repository-defined exit `3` in that adapter
path is the TimeSformer liveness refusal.

## Constraints

- Keep the liveness gate fail-closed. A non-`ALIVE` map must never reach
  stability analysis, the high-recall router, the CT/fiber gate or downstream
  flattening.
- Do not label a degenerate or empty map as CT-supported, no-ink, ink, text or
  letters.
- Preserve and publish a hash-bound evidence bundle before declaring the QC job
  terminal.
- Keep operational failures retryable and configuration failures blocked.
- Do not change queue order, retry timing, mission scope, authentication or API
  write surfaces.
- Do not use `allow_unvalidated` in the First Letters campaign.

## Considered Approaches

### 1. Keep retrying

Rejected. The inputs are immutable and the runner's exit `3` is a declared
measurement refusal, not transient infrastructure. Repetition only consumes GPU
time and hides the scientific state.

### 2. Propagate exit `3` as a terminal failed job

This is smaller, but the adapter stops before its normal evidence manifest and
publication boundary. The result would be terminal yet weaker than every other
surface-QC outcome and could be lost with a disposable worker.

### 3. Publish an explicit terminal insufficiency outcome

Selected. Invoke the runner with `--on-degenerate warn` only so it returns
control after writing the receipt and maps. The adapter then performs its own
mandatory fail-closed check of that receipt. A non-`ALIVE` verdict skips every
downstream screen, writes a terminal summary, publishes the ordinary immutable
evidence bundle and completes with an explicit non-admissible outcome.

`warn` is not a bypass in this design: the adapter is the immediate caller and
must reject any non-`ALIVE` receipt before another scientific stage can run.

## Outcome Contract

Add the canonical outcome:

```text
INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY
```

It maps to:

```text
surface.state = QC_INK_SCREEN_INSUFFICIENT
surface.physical_qc_state = INK_SCREEN_INSUFFICIENT
```

This state is terminal but not downstream-admissible. It means only that the
configured screen produced no usable decision for this surface. It does not
establish CT support or the presence or absence of ink.

The summary receipt must include the full structured liveness verdict and
reason from `INK_SCREENING_RECEIPT.json`, plus the receipt path and SHA-256. Raw
maps remain regenerable and excluded from durable publication; their hashes
remain bound through the screening receipt.

## Data Flow

1. Render and verify 65 ordered CT slices exactly as today.
2. Run the six-replica TimeSformer screen with `--on-degenerate warn`.
3. Require `INK_SCREENING_RECEIPT.json` and a structured liveness verdict.
4. If `ALIVE`, preserve the existing stability, high-recall and CT/fiber flow.
5. If `DEGENERATE` or `EMPTY`, write a terminal screen summary and do not call
   stability analysis or any high-recall component.
6. Build and publish the ordinary evidence manifest.
7. Complete the QC job with the new outcome and update the surface to the new
   non-admissible state.

Any missing or unknown liveness verdict remains an operational failure rather
than silently becoming an insufficiency outcome.

## Ledger and Verification

The surface-QC ledger treats the new completed outcome as
`SELECT_DIFFERENT_SURFACE_SCREEN_INSUFFICIENT`. Its verifier accepts the outcome
only when rendering, inference and the evidence manifest are complete and
hash-bound. Stability analysis and the CT/fiber gate must be absent for this
outcome; their presence is a contract violation because the fail-closed branch
should have stopped before them.

Existing three outcomes retain their existing stage requirements and meanings.

## Test Strategy

1. A RED adapter test supplies a `DEGENERATE` receipt and proves the current
   code raises/retries instead of producing a terminal summary.
2. GREEN proves the inference command records `--on-degenerate warn`, the
   adapter persists the exact liveness report, and analysis/high-recall are not
   called.
3. Store/worker contract tests prove the new outcome completes in SQLite and
   PostgreSQL, updates both surface state axes and is not claimable again.
4. Ledger tests prove the next action and outcome-specific evidence rules.
5. Regression tests prove `ALIVE`, no-common-valid, supported/no-retained,
   retained-for-review, exit `78`, and ordinary retryable failures are
   unchanged.
6. Run the focused suite and the unrestricted full test suite before review,
   push, staging deploy and natural retry observation.

## Production Acceptance

After exact-SHA staging convergence and smoke success, do not enqueue or
manually retry the existing QC job. Let its normal claim execute once. Accept
the repair only if the job reaches `COMPLETED`, records
`INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY`, publishes a verified evidence
manifest, updates the surface to `INK_SCREEN_INSUFFICIENT`, and does not return
to `PENDING`. Preserve the result as a measured terminal insufficiency and move
the campaign to the next bounded P1 wave.
