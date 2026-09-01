# Authorization to execute the frozen control on real data, as a development control

Status: **AUTHORIZED, NOT YET EXECUTED**
Manifest: `framework/profiles/01-segmentation/first-letters-control-policy-1.0.0.json`
Control: `PHerc0139-w025-public-positive-v1`

## What was authorized, and by whom

The frozen manifest declines to authorize its own execution on real data:

```
status:                                          FROZEN_DEVELOPMENT_CONTROL_NOT_INDEPENDENT_VALIDATION
control_cohort.role:                             DEVELOPMENT_ONLY_PUBLIC_POSITIVE_CONTROL
safety.real_data_execution_authorized_by_this_manifest:  false
safety.control_pass_is_independent_validation:   false
```

That is not an omission. The manifest defers the decision to a person, and the
project owner has now taken it: **execute it on real data as a development
control, and record that this is what it is.** This document is that record.

## What a pass will establish, and what it will not

A `CONTROL_PASS` will establish **sensitivity**: that the exact deployed
revision, against frozen source locks, profile locks and model locks, can carry
a case with publicly attributed writing through `DISCOVERY_CONTROL` (candidate
availability, content-blind) and `PIPELINE_CONTROL` (canonical grow and the
unchanged downstream acceptance path) to blinded human review.

It will **not** establish any of the following, and no downstream document may
say otherwise:

- **It is not independent validation.** PHerc0139 participated in development.
  The manifest states this as a non-claim and it survives this authorization
  unchanged.
- **A pass does not accept ink, text or letters.** It routes a known positive to
  blinded human review; the review is what adjudicates.
- **The public w25 attribution does not fix the frozen seed as a letter
  coordinate.** The scroll has recovered writing; that is not a claim about the
  specific coordinate this control grows from.
- **A miss is not evidence about PHerc0139.** A `CONTROL_FAILED` or
  `CONTROL_INCOMPLETE` is a failure of the frozen provider and gates, and says
  nothing about whether that scroll has writing.

## The consequence for the campaign gate

The program requires a *community-confirmed* positive control before a target
campaign. What is being run is a *development* control. Any First Letters
campaign launched on this gate is therefore standing on weaker evidence than the
program specified, and its evidence closeout must say so in those words.

Freezing a genuinely independent control — on a scroll that took no part in
development — remains open, and remains the stronger path. This authorization
does not close it.

## Preconditions not yet met

Execution requires a live deployment and authenticated APIs: the runner takes
`--panel` and `--deployed-revision` and cannot run offline. It is therefore
gated behind the ordered steps of plan Task 10, and one of them is currently
blocked by work in flight:

1. All Task 8 code complete and independently reviewed — **four agents in
   flight** (C3/C4/C7, C5/C6, C1/C2/I1, PostgreSQL parity).
2. Full offline verification: Python suite, frontend suite and build,
   documentation truth tests, secret scan.
3. Push, staging pipeline green.
4. Deploy the exact approved revision.
5. Smoke every service and prove each reports that exact revision.
6. Then, and only then, this control.

**The specific blocker.** `PIPELINE_CONTROL` exercises the canonical grow and
downstream acceptance path, which includes P3 flattening. C9 currently gates
`record_flattening` on an exact `STANDARD` route, and the PostgreSQL routing
receipt table has not yet been ported — so every PostgreSQL flattening refuses,
by design, until that port lands. Two tests carry an `xfail` naming it.

Running the control before the port would produce a `CONTROL_INCOMPLETE` caused
by our own unfinished work rather than by anything about the pipeline. The
manifest's `staleness_rule` would then invalidate that result the moment the
port changed the deployed revision, so it would have to be re-run regardless.

## Non-claims of this document

This document authorizes an execution. It reports no result, because none has
been produced. Nothing here may be cited as evidence that the control passed,
that the pipeline is sensitive, or that any scroll does or does not contain
writing.
