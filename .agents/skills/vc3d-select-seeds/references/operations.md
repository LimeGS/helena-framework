# Operations

## Contents

1. Required runtime
2. Code readiness
3. Shadow run
4. Isolated causal benchmark
5. Select run
6. Monitoring
7. Failure handling
8. Rollback

## Required runtime

- Use Python 3.9 or newer and Git for readiness and release checks.
- Use PostgreSQL as the production control plane.
- Use S3 as the production artifact store.
- Require `psycopg2` to already be present in the readiness environment when
  checking live PostgreSQL worker freshness. If it is missing, report
  `NOT_READY`; do not install it automatically.
- Configure `VC_MCP_URL` and `VC_MCP_AUTH_TOKEN` for the Helena seed
  provider when using `--seed-provider mcp`. This is the pipeline's authenticated
  seed service, not an OpenCode or Codex MCP installation.
- Start stateless workers with `HELENA_SEED_PROBE_SUPPORT=1` and
  `--seed-probe-support`.
- Keep `HELENA_DISABLE_SEED_PROBE_V1` unset.
- Keep `HELENA_ENABLE_SEED_PROBE_SELECT` unset until a benchmark decision is
  approved.
- Use the exact project CLI below `CAMPAIGN_ROOT`:
  `framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py`.
- Keep secrets in environment-backed database aliases; do not place a raw DSN
  or cloud credential in prompts, receipts, commands, or artifacts.
- Inspect status and receipts freely. Bootstrap, worker execution, rollout
  changes, review resolution, and garbage collection require an explicit user
  request.
- Use the fleet executor for production; never call raw
  `vc_grow_seg_from_seed`.
- Never pass `--no-verify-sources`.

## Code readiness

Run:

```bash
python3 "$CAMPAIGN_ROOT/.agents/skills/vc3d-select-seeds/scripts/production_readiness.py" \
  --root "$CAMPAIGN_ROOT" \
  --mode shadow
```

The check verifies required code/contracts, stage declarations, profile
dependency hashes, catalogue readability, and skill resources. Add
`--production`, database, and artifact arguments to check a live deployment.

## Shadow run

Queue a pre-registered shadow arm:

```bash
python3 "$CAMPAIGN_ROOT/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py" \
  bootstrap \
  --db postgres-env://CX_DB \
  --eligible "$CAMPAIGN_ROOT/workspace/catalog/eligible_volumes.json" \
  --catalog "$CAMPAIGN_ROOT/workspace/catalog/geometry_surface_catalog_v2/GEOMETRY_SURFACE_CATALOG.sqlite" \
  --policy-version seed-probe-shadow-arm-v1 \
  --planner cost-aware-v2 \
  --seed-probe-mode shadow \
  --seed-probe-top-k 2 \
  --seed-probe-generations 12 \
  --reason "pre-registered seed-probe shadow cohort"
```

Use a distinct baseline policy version with `--seed-probe-mode off`. Freeze the
cell assignment before either arm observes results. Do not run the second arm
against coverage modified by the first; use a frozen assignment or isolated
control-plane copy.

After explicit authorization, execute one known queued task at a time:

```bash
python3 "$CAMPAIGN_ROOT/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py" \
  worker run \
  --db postgres-env://CX_DB \
  --worker-id "$WORKER_ID" \
  --run-root "$RUN_ROOT" \
  --artifact-root s3://BUCKET/PREFIX \
  --qc-profile-id "$QC_PROFILE_ID" \
  --seed-provider mcp \
  --planner cost-aware-v2 \
  --vc3d-binary "$VC3D_GROW_BINARY" \
  --seed-probe-support \
  --task-id "$TASK_ID" \
  --max-jobs 1 \
  --terminal-outcomes-exit-zero
```

Inspect the JSON result. The last flag makes expected terminal scientific
outcomes return process status zero; it does not make technical failures pass.
Do not add `--watch` implicitly.

## Isolated causal benchmark

Queue each arm into its own empty, nonproduction control plane. This mutates the
named benchmark database and therefore requires explicit user authorization.
The confirmation flag is an operator attestation; it cannot prove that
credentials or infrastructure are isolated.

```bash
python3 "$CAMPAIGN_ROOT/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py" \
  seed-probe-benchmark-arm \
  --db postgres-env://CX_BENCHMARK_BASELINE_DB \
  --confirm-isolated-nonproduction \
  --benchmark-spec /isolated/BENCHMARK_SPEC.json \
  --arm baseline \
  --eligible "$CAMPAIGN_ROOT/workspace/catalog/eligible_volumes.json" \
  --catalog "$CAMPAIGN_ROOT/workspace/catalog/geometry_surface_catalog_v2/GEOMETRY_SURFACE_CATALOG.sqlite" \
  --grid-version paired-grid-v1
```

Repeat with `--arm closed_loop` against a different empty database, run root,
artifact prefix, and restricted credential set. Workers for either arm must
load the identical preregistration:

```bash
python3 "$CAMPAIGN_ROOT/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py" \
  worker run \
  --db postgres-env://CX_BENCHMARK_CLOSED_DB \
  --worker-id "$WORKER_ID" \
  --run-root "$BENCHMARK_RUN_ROOT" \
  --artifact-root s3://NONPRODUCTION-BUCKET/paired-v1/closed-loop \
  --qc-profile-id "$QC_PROFILE_ID" \
  --seed-provider mcp \
  --planner deterministic-v2 \
  --vc3d-binary "$VC3D_GROW_BINARY" \
  --isolated-benchmark-spec /isolated/BENCHMARK_SPEC.json \
  --confirm-isolated-nonproduction \
  --seed-probe-support \
  --max-jobs 1
```

Omit `--seed-probe-support` on a baseline-only worker. Both paths refuse a
process configured for production select. The arm command also refuses a
control plane that already contains tasks or surfaces.

## Select run

Before queueing:

1. Run production readiness with `--mode select`.
2. Verify the output status is `READY`.
3. Verify the benchmark decision hash and source locks independently.
4. Set `HELENA_ENABLE_SEED_PROBE_SELECT=1` only in the authorized environment.
5. Pass the exact approved benchmark receipt to the bootstrap command.
6. Declare an accountable review owner.
7. Use `cost-aware-v2` or `deterministic-v2` and `top_k >= 2`.

Do not use the public catalogue for select while it contains no verified
immutable content locks.

For the panel, configure the same immutable receipt path as
`CX_SEED_PROBE_BENCHMARK_RECEIPT` and set
`CX_SEED_PROBE_REVIEW_OWNER`. The frontend displays the benchmark identity and
hash prefix, but never receives the local path or review-owner identity.

Queue only the approved production arm:

```bash
python3 "$CAMPAIGN_ROOT/framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py" \
  bootstrap \
  --db postgres-env://CX_DB \
  --eligible "$CAMPAIGN_ROOT/workspace/catalog/eligible_volumes.json" \
  --catalog "$CAMPAIGN_ROOT/workspace/catalog/geometry_surface_catalog_v2/GEOMETRY_SURFACE_CATALOG.sqlite" \
  --policy-version "$APPROVED_SELECT_POLICY_VERSION" \
  --planner deterministic-v2 \
  --seed-probe-mode select \
  --seed-probe-top-k 2 \
  --seed-probe-generations 12 \
  --seed-probe-benchmark-receipt "$BENCHMARK_DECISION" \
  --seed-probe-review-owner "$REVIEW_OWNER" \
  --reason "$REGISTERED_REASON"
```

The isolated causal benchmark uses a separate pre-registered authorization,
database, artifact prefix, and restricted credentials. That authorization may
create its `select` arm but cannot authorize production `select`.

## Monitoring

Inspect:

- task and ordinary attempt state;
- probe run/trial/attempt/artifact/evaluation state;
- categorical decision and winner identity;
- promotion and continuation binding;
- canonical full-grow geometry and physical QC;
- probe generations, full-grow compute wall-hours, device/tier identity, and
  reviewer minutes.

Investigate `HUMAN_REVIEW`, `PROBE_REVIEW_PENDING`, `CONTINUATION_FAILED`,
`PROMOTION_FAILED`, retry exhaustion, repeated failed trials, or an unavailable
capable worker. Do not reinterpret operational failure as evidence against a
seed.

`HUMAN_REVIEW` is an abstention, not an invitation to mutate the decision.
Inspect the attempt's complete probe lineage from the Segmentation page. If an
operator later supplies a seed, queue it as a new manual-seed task with its
author and reason; never rewrite the probe winner or promote probe bytes.

## Failure handling

- Retry transient object-store/database failures only within configured caps.
- Retain corrupt, missing, ambiguous, or authority-mismatched evidence for
  review.
- Never manually fill migration binding columns.
- Never infer state from artifact directories when the ledger is available.
- Never continue a winner whose candidate, decision, attempt, plan, or hash
  differs from the persisted authority.

## Rollback

Set new runs to `off`, unset `HELENA_ENABLE_SEED_PROBE_SELECT`, and set
`HELENA_DISABLE_SEED_PROBE_V1=1` on workers for an immediate stop. Preserve
ledger rows and retained evidence. Canonical surfaces already produced remain
ordinary versioned artifacts.
