# Task 10 closeout: deployed, backfilled, control refused, campaign not launched

Date: 2026-08-05
Deployed revision: `7fca35cf622ea583f0b75251621f452f9fcbf9d3`
Host: gpu-1

**The First Letters campaign was not launched, and must not be until the
incident below is resolved.** Everything up to that gate was completed and is
recorded here with the commands that produced it.

## What was completed

| Task 10 step | State | Evidence |
|---|---|---|
| 1–2 offline verification | done | 3397 passed / 66 skipped / 0 failed with `HELENA_TEST_DSN`; 3298 / 165 / 0 without; `tsc` clean, 44 frontend tests, production build; 61 documentation-truth and audit tests; 39 profiles valid; secret scan clean |
| 3 staging pipeline | done | `staging` at `7fca35cf`, pipeline green through deploy |
| 4 deploy exact revision | done | gpu-1 running `7fca35cf` |
| 5 smoke every service | done | all eight services report label `7fca35cf` |
| 5b routing backfill | done | 293 STANDARD, 2 SMALL_SURFACE_DIAGNOSTIC, 21 unroutable; idempotent on a second pass |
| **6 positive control** | **REFUSED** | `CONTROL_INCOMPLETE`, first non-passing boundary `P0`, reason `CONTROL_INCOMPLETE_STALE` |
| 7 preflights and budgets | not started | blocked by 6 |
| 8 campaign waves | **not started, correctly** | blocked by 6 |

## The backfill, which is the part that changed the deployment

316 surfaces existed with no routing decision, because they predate Task 8.
Every gate Task 8 built fails closed on a missing receipt, so this had to run
with the deploy or pending work would have stopped.

```
by_route:    STANDARD_QC_PENDING 293, SMALL_SURFACE_DIAGNOSTIC 2
unroutable:  21
considered:  316
```

The two diagnostic surfaces measure `0.01983222455087575 cm2` and
`0.0034788851104890976 cm2`. The first is PHerc0268 by its frozen area — the
surface that on 2026-08-02 was GEOMETRY_CERTIFIED, entered physical QC, and came
back INK_SCREEN_INSUFFICIENT / EMPTY. Pending QC jobs went from 87 to 85 as
those two left the queue. **That is the whole point of Task 8, observed on real
data.**

The 21 unroutable are all `IMPORTED_COVERAGE` rows from PHerc0800 and PHerc1447.
They have no measured area, so they were reported rather than guessed — a repair
that invents a measurement is the failure the repair exists for. They never
entered physical QC or any downstream stage, so having no receipt blocks nothing.

## Incident: the positive control cannot run

`CONTROL_INCOMPLETE`, `first_nonpassing_boundary: P0`, `reason_code:
CONTROL_INCOMPLETE_STALE`, with empty `counts`, `input_artifacts` and
`output_hashes` at P0.

The cause is not a threshold, a lock, or a code defect. All six profile locks in
the frozen manifest hash-match the deployed files. It is simpler:

**PHerc0139 — the control scroll — is not registered as a source snapshot on
this control plane.** `segment_source_snapshots` holds the thirteen evaluation
scrolls (PHerc125, 191, 211, 257, 268, 358, 800, 813, 826, 1203, 1218, 1447,
1545) and nothing for PHerc0139.

So the control has no P0 to survive. `$CX_RUNS/control-w025` holds 88 MB of
comparison material for it — `mesh.tifxyz`, `community-ink.tif`, a side-by-side
against the community reading — but that is a comparison artifact set, not a
mission whose P0 the control plane has ingested.

The plain reading: **this control has never been executed.** Executing it is not
a verification script. It requires ingesting PHerc0139 at P0 and then running
DISCOVERY_CONTROL and PIPELINE_CONTROL — candidate availability, canonical grow,
and the unchanged downstream acceptance path through to blinded human review.
That is GPU work of hours and it queues real jobs the control plane records.

### What was deliberately not done

No threshold was changed, no gate bypassed, no `allow_unvalidated`, and the
campaign was not launched on a failing control. Task 10 forbids all four and
each was the shortest path from here.

The ephemeral account and empty mission created to attempt the run were deleted
afterwards; the generated password never left the host and was never read.

## Explicit non-claims

- **No First Letters claim is made.** No candidate was produced, no packet
  formed, no human review requested.
- **A refused control is not evidence about any scroll.** It says this pipeline's
  sensitivity is unproven on this revision, and nothing about whether any
  papyrus carries writing.
- **The backfill decided nothing new.** It ran the frozen router over rows that
  already existed. A surface routed SMALL_SURFACE_DIAGNOSTIC is too small for
  the standard acceptance path; that is a statement about size and about nothing
  else.
- **The control that was attempted is a development control.** PHerc0139
  participated in development, and the manifest states a pass by it is not
  independent validation. Even had it passed, the campaign would stand on weaker
  evidence than the program specified.

## What has to happen before a campaign

1. Ingest PHerc0139 at P0 on this control plane, or point the control at a
   deployment that already has it.
2. Execute DISCOVERY_CONTROL and PIPELINE_CONTROL to a terminal state and audit
   every stage receipt, manifest and artifact hash.
3. Only on `CONTROL_PASS`, and on this exact deployed revision, run the
   per-scroll preflights and derive budgets.
4. Then, and only then, the bounded waves under the Task 9 gates.

Freezing a genuinely independent control — on a scroll that took no part in
development — remains open and remains the stronger path.
