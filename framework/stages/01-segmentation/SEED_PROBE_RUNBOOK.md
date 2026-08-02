# Seed probe v1: production runbook

## Decision

`seed-probe-v1` is a deterministic VC3D experiment beneath the existing
Cost-aware router. It does not replace Cost-aware routing and it does not
replace the Fusion panel.

The probe asks one narrow question before the full grow: which of the first
one to three ink-blind m7 candidates can produce a geometry-measurable
micro-surface under one frozen 10–20-generation recipe? It makes no claim
about the correct physical lamina, ink, text, or First Letters.

The supported production rollout is:

1. `off`: the existing path.
2. `shadow`: run and retain probes, then execute the existing planner path
   unchanged. This is the first production phase.
3. `select`: allow exactly one geometry-eligible probe to constrain the next
   full grow. This remains disabled until the scientific and source-lock gates
   below pass.

An inconclusive comparison abstains. It never turns a weighted score into a
winner. A one-candidate shadow preflight is explicitly
`INSUFFICIENT_CANDIDATES_FOR_COMPARISON`; it must not be counted as a shadow
win or used in the select benchmark.

## Architecture boundary

| Component | Responsibility |
|---|---|
| m7 candidate extractor | One bounded read; deterministic candidate and local geometry evidence |
| seed probe | Frozen micro-grows, TIFXYZ geometry evaluation, categorical decision |
| Cost-aware v2 | Outer planner/router and normal fallback policy |
| Fusion v2 | Expensive canary/fallback for the planner path; not called by a deterministic probe winner |
| finalizer and QC | Canonical full-grow publication, deduplication, geometry and physical QC |

Probe artifacts are noncanonical and live only under
`probes/<sample>/<run>/<trial>/<artifact-sha256>`. They never enter `surfaces`,
canonical `artifact_sets`, or downstream QC. A selected winner is materialized,
hash-checked, resumed into a normal full grow, and only that full grow can be
promoted.

## Admission controls

Workers must explicitly advertise the capability:

```bash
export HELENA_SEED_PROBE_SUPPORT=1
```

The supervisor then supplies `--seed-probe-support`. A queue request fails
closed when no recently reported capable worker exists.

`select` additionally requires all of:

- `HELENA_ENABLE_SEED_PROBE_SELECT=1`, set only after benchmark approval; and
- a `campaignx.source_content_lock.v1` receipt that binds lowercase SHA-256
  manifests to the exact immutable-version CT and m7 URIs the worker will read;
  and
- an explicit `--seed-probe-review-owner`, persisted on every production-select
  policy, so `PROBE_REVIEW_PENDING` has an accountable operator or team rather
  than an unowned terminal queue.

A digest next to a mutable URL is deliberately insufficient. The current
public eligible-volume catalogue has no such locks, so `select` is unavailable
there even if the rollout environment flag is set.

## Shadow rollout and benchmark gate

Shadow is an operational/calibration gate, not a causal yield experiment. Use
40–60 ambiguous cells across at least three scrolls. Pre-register the exact
cells before examining outcomes and compare the unchanged baseline path with
`shadow` using the same source snapshot, candidate extractor, worker tier, and
full-grow/QC policy.

Task identity includes `policy_version` but not probe mode. Use distinct arm IDs
(for example `seed-probe-baseline-arm-v1` and
`seed-probe-shadow-arm-v1`) and a frozen cell assignment. Reusing one policy
version makes the second insertion a no-op; allowing the first arm to change
coverage before generating the second destroys the match.

Shadow deliberately returns the original candidates to the planner. It can
establish replay, safety, abstention/rejection distribution, lower-ranked rescue
opportunities, and incremental probe cost. It cannot establish improved
canonical yield, because it never steers the canonical grow.

The causal rollout gate therefore requires a separate isolated,
nonproduction paired outcome experiment:

- compare `deterministic-v2` with probe `off` against `deterministic-v2` with
  probe `select`;
- use separate fresh control planes, artifact prefixes, and credentials that
  have no production catalogue import path;
- use verified immutable CT/m7 locks;
- freeze identical candidate bytes/history, VC3D binary, profiles, full-grow
  parameter envelope, worker tier, device type, and blinded review protocol;
- use a common explicit RNG identity across paired arms rather than the normal
  attempt-ID-derived RNG;
- record actual grow wall time/device identity and reviewer minutes;
- include every failure and abstention under intention-to-treat analysis; and
- emit an immutable benchmark decision receipt.

A matching envelope and RNG receipt does not establish that VC3D
`12 generations -> resume -> target 35` is numerically equivalent to a fresh
35-generation grow. Treat the whole closed-loop path as the experimental
treatment, charge every probe and resume operation to it, and validate the real
VC3D runtime behavior before rollout.

The repository skill at `.agents/skills/vc3d-select-seeds` validates this
receipt. It refuses approval when paired RNG/envelope/resource identities or
measurements are missing. Production
`HELENA_ENABLE_SEED_PROBE_SELECT` remains unset until that receipt says
`APPROVED_SELECT`. An isolated benchmark environment may enable select only
under its separate restricted credentials and namespace; that is not a
production rollout.

Queue each causal arm from its preregistration into a separate, empty control
plane. The command imports and source-verifies only the scrolls named in the
spec, imports their frozen catalogue, generates the exact cell cohort, and
inserts the whole arm in one transaction:

```bash
python3 framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py \
  seed-probe-benchmark-arm \
  --db /isolated/baseline/fleet.sqlite \
  --confirm-isolated-nonproduction \
  --benchmark-spec /isolated/spec.json \
  --arm baseline \
  --eligible /isolated/eligible.json \
  --catalog /isolated/GEOMETRY_SURFACE_CATALOG.sqlite \
  --grid-version paired-grid-v1
```

Run `closed_loop` against a different empty DB, run root, artifact prefix, and
credential set with the same spec and frozen generation options. An isolated
worker must load that same preregistration:

```bash
python3 framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py \
  worker run \
  --db /isolated/closed-loop/fleet.sqlite \
  --worker-id paired-closed-loop-01 \
  --run-root /isolated/closed-loop/runs \
  --artifact-root s3://NONPRODUCTION-BUCKET/paired-v1/closed-loop \
  --qc-profile-id geometry-qc@1.0.0 \
  --planner deterministic-v2 \
  --vc3d-binary /opt/vesuvius/bin/vc_grow_seg_from_seed \
  --isolated-benchmark-spec /isolated/spec.json \
  --confirm-isolated-nonproduction \
  --seed-probe-support
```

Omit `--seed-probe-support` on a baseline-only worker. Both commands refuse a
process with production select enabled; the arm command also refuses a control
plane with pre-existing tasks or surfaces. These are fail-closed guards, not
proof of isolation: `--confirm-isolated-nonproduction` is an operator
attestation. The CLI cannot prove that the DB endpoint, object-store
credentials, run root, or network access are nonproduction.

Approve production `select` only when all of these are true:

- no canonical-output, lease, replay, or budget invariant is violated;
- probe bytes and evaluations replay identically for identical inputs;
- incremental compute wall time stays within the registered budget;
- blinded reviewers see the pre-registered improvement in usable,
  nonduplicate, single-lamina area per compute wall-hour on a matched tier;
- an exact paired superiority test passes at the pre-registered one-sided
  alpha, rather than relying on an aggregate point estimate;
- reviewer minutes per usable square centimetre do not regress;
- no pair introduces a new incorrect-lamina harm and the exact one-sided
  zero-event upper bound stays within its registered margin;
- every scroll meets its minimum paired sample and neither usable yield nor
  reviewer rate regresses beyond its frozen margin;
- cells name distinct pre-registered spatial independence blocks;
- immutable source locks are generated and verified at intake; and
- an operator-owned review-resolution workflow exists before select traffic
  can produce `PROBE_REVIEW_PENDING`.

Do not use “micro-patch passed geometry” as the success metric. The business
metric is useful canonical surface yield after normal deduplication, geometry
QC, physical QC, and blinded lamina review, divided by measured compute and
reviewer cost.

At a 5% one-sided alpha, 40 zero-new-harm pairs only bound the new-harm rate
below about 7.22%. A 5% safety bound therefore needs at least 59 pairs with no
new harm. A smaller cohort remains diagnostic but cannot approve that margin.
Any approval is restricted to the receipt's `authorized_sample_ids`. Three
scrolls support a scoped canary on those sources; they do not establish
generalization to an untested scroll.

## Queueing

From the panel, choose **Shadow evidence**. From the CLI:

```bash
python3 framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py \
  bootstrap \
  --db postgres-env://CX_DB \
  --eligible workspace/catalog/eligible_volumes.json \
  --catalog workspace/catalog/geometry_surface_catalog_v2/GEOMETRY_SURFACE_CATALOG.sqlite \
  --policy-version seed-probe-shadow-arm-v1 \
  --planner cost-aware-v2 \
  --seed-probe-mode shadow \
  --seed-probe-top-k 2 \
  --seed-probe-generations 12
```

`top_k`, generations, attempts, and total probe generations are bounded in the
frozen policy. Retries reuse the same logical run and locked trial plan; they
do not reset the task budget.

Winner-artifact fetches retry at most five replacement attempts by default.
Transient finalization failures retry at most two replacement attempts, so a
persistent object-store or database outage cannot trigger unbounded duplicate
full grows. Operators may lower these caps with
`--probe-artifact-max-requeues` and `--finalization-max-requeues`; raising them
requires an explicit compute-budget decision. When either budget is exhausted,
the evidence stops in review/failure retention rather than looping.

## Monitoring

The Segmentation page reports aggregate probe run, trial, decision, and
promotion-record counts, successful canonical continuations, action counts,
and mode/action splits. It is a read-only operations summary. Full
trial/evaluation/artifact/attempt/promotion lineage is available from the raw
attempt API; it is not rendered as a select-review drill-down. Investigate:

- `HUMAN_REVIEW`: scientifically inconclusive, retained indefinitely;
- `PROMOTION_FAILED`: no cleanup; all evidence moves to review retention;
- `CONTINUATION_FAILED`: a unique select winner existed, but planning failed
  before an exact promotion contract could be created; retain and review;
- `REVIEW_PENDING` after winner materialization: retained bytes were missing,
  corrupt, outside the namespace, or exhausted their bounded fetch retries;
- `RETRYABLE_FINALIZATION_UNAVAILABLE`: transient storage/database outage;
  verify that the replacement-attempt counter remains within its configured
  cap;
- repeated `FAILED` trials: executor/source incident, not evidence against a
  seed;
- no capable worker: deployment/configuration error;
- shadow/select mixed in one aggregate: use the per-mode counts.

Lineage by parent attempt includes all seven probe ledger tables. Use it for
incident review rather than inferring state from artifact directories.

## Schema upgrade safety

Migration v8 binds every promotion to an exact continuation attempt, contract,
and locked-plan hash; v9 enforces those bindings for databases that may already
have recorded the earlier v8. A v6/v7/early-v8 database containing a legacy
promotion cannot be auto-bound honestly, so startup fails with
`seed-probe v8 cannot auto-bind legacy promotion rows`. Do not fill those
columns by hand. Export the task, decision, artifact, locked plan, and canonical
surface evidence; resolve each row under operator review before retrying the
migration. A database with no legacy promotion rows upgrades automatically.

## Retention and cleanup

Shadow and review evidence are retained indefinitely. Select losers and
promoted winners become eligible for deletion after 30 days. The immutable
ledger row remains as `EXPIRED`; only exact manifest-listed bytes are removed.

Always inspect a dry run first:

```bash
python3 framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py \
  probe-gc \
  --db postgres-env://CX_DB \
  --artifact-root s3://BUCKET/PREFIX \
  --receipt /tmp/seed-probe-gc-dry-run.json
```

Apply the exact reviewed set. The command re-queries the ledger and refuses to
delete if the set changed after the dry run:

```bash
GC_SET_SHA256="$(
  jq -r .candidate_set_sha256 /tmp/seed-probe-gc-dry-run.json
)"
python3 framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py \
  probe-gc \
  --db postgres-env://CX_DB \
  --artifact-root s3://BUCKET/PREFIX \
  --limit 100 \
  --apply \
  --expect-candidate-set-sha256 "$GC_SET_SHA256" \
  --receipt /tmp/seed-probe-gc-applied.json
```

Deletion fails closed if the URI is outside that artifact store's exact probe
namespace or does not end in the manifest content hash.

## Rollback

Set new queue requests to `off`, unset
`HELENA_ENABLE_SEED_PROBE_SELECT`, and set
`HELENA_DISABLE_SEED_PROBE_V1=1` on workers if execution must stop
immediately. Existing canonical surfaces are unaffected because probes never
write them. Keep the seven ledger tables and retained evidence for audit.
