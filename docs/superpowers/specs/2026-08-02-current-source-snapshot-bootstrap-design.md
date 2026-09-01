# Current Source Snapshot Bootstrap Design

## Problem

`bootstrap_sources()` resolves one current catalog entry per requested sample and
returns its `source_snapshot_id`. `bootstrap_queue()` then ignores that resolved
set and calls `store.snapshots(samples)`, which returns every historical snapshot
for those samples.

On `PHerc826`, two historical snapshots existed. One API request therefore
created four tasks against each snapshot. The receipt stored counts in a map
keyed only by canonical `sample_id`, so the second four-task summary overwrote
the first. The API reported `generated=4, inserted=4` while eight tasks were
actually inserted.

This is both a scientific provenance defect and a control-plane accounting
defect. A run requested against the current catalog must not silently fan out to
historical sources.

## Decision

The normal bootstrap will process exactly the snapshots whose identifiers were
returned by `bootstrap_sources()` for the current eligible catalog.

After loading snapshot rows from the store, `bootstrap_queue()` will:

1. Index the loaded rows by `source_snapshot_id`.
2. Resolve each requested sample through the `sources` mapping returned by
   `bootstrap_sources()`.
3. Fail closed if a resolved identifier is missing from the store or if the row
   belongs to a different sample.
4. Generate tasks only for that ordered current set.

Historical snapshots remain immutable and queryable. Existing tasks and
attempts are not deleted or rewritten.

## Alternatives Rejected

### Process every historical snapshot and aggregate the receipt

This would make the count honest but preserve the scientific error: one run
would still mix historical sources without the caller requesting that scope.

### Delete or supersede older snapshots

This would destroy or obscure provenance used by historical tasks. Snapshot
history is evidence and must remain immutable.

### Pick the newest database row by timestamp

The current eligible catalog is already the authority and
`bootstrap_sources()` already resolves its content-addressed snapshot. Database
recency is weaker and can disagree with the catalog.

## Error Handling

Selection fails before task generation when:

- a current catalog sample has no resolved snapshot row;
- the resolved row's `sample_id` does not match the catalog sample; or
- duplicate rows expose an impossible identifier collision.

The error includes sample and snapshot identifiers, but never source URIs,
credentials, worker identities, or database connection material.

## Verification

A regression test will use a real temporary fleet store containing an old and a
current snapshot for one sample. It will prove that one bootstrap:

- creates tasks only for the current snapshot;
- leaves the historical snapshot untouched;
- reports counts equal to the rows created; and
- preserves the existing single-snapshot behavior.

Focused tests, the full local suite, independent review, staging pipeline,
deploy, and smoke must pass before another scientific mutation is allowed.

## Existing PHerc826 Evidence

The eight already-created attempts are retained as immutable negative evidence.
They all ended `NO_SEED` with `NO_M7_CANDIDATES`; no surface or downstream
artifact exists. The incident will be recorded as one request that fanned out to
two historical snapshots and under-reported its inserted count.
