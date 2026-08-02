---
name: vc3d-select-seeds
description: Select, validate, operate, or benchmark VC3D papyrus seed candidates with deterministic m7 evidence, bounded micro-growth, mesh geometry gates, provenance, and fail-closed human review. Use for VC3D seed search, GrowPatch preflights, seed-probe shadow/select rollout, seed quality comparisons, production readiness checks, or comparisons against deterministic-v2, Cost-Aware, Fusion, and manual seed workflows in Helena Framework.
---

# VC3D Seed Selection

Operate the repository's closed-loop seed pipeline. Keep numerical decisions in
deterministic code; use the agent to inspect evidence, invoke bounded tools,
explain failures, and coordinate review.

## Establish the boundary

- Keep Cost-Aware v2 as the outer router.
- Keep Fusion v2 as an expensive planner fallback/canary.
- Use `seed-probe-v1` beneath the router to compare one to three m7 candidates.
- Treat a geometry-certified micro-patch as evidence, not proof of the correct
  physical lamina.
- Never infer distances, normals, intersections, curvature, or sheet switching
  from a screenshot when a numerical receipt exists.
- Never bypass the fleet ledger, locked plans, content hashes, leases, retry
  budgets, artifact namespaces, finalizer, or downstream QC.
- Treat status, readiness, receipt inspection, and comparison as read-only.
  Require an explicit user request before bootstrap, worker execution, rollout
  changes, dependency installation, review resolution, or garbage collection.
- Never invoke raw `vc_grow_seg_from_seed` for production and never add
  `--no-verify-sources`.
- Default to `shadow`. Use `select` only after every production gate reports
  `PASS`; otherwise stop.

## Locate the project

Find the nearest ancestor containing both:

```text
framework/stages/01-segmentation/stage.json
framework/stages/01-segmentation/scripts/helena_segment_search_fleet.py
```

Call that directory `CAMPAIGN_ROOT`. Resolve every repository-relative path
against it. Do not assume the current directory is the root.

OpenCode discovers this project skill when it starts in `CAMPAIGN_ROOT` or one
of its descendants. The skill is repository-local and must not require a
Codex, OpenAI, or user-global installation. The shell snippets below use POSIX
syntax as examples; the workflow and receipts are agent-framework agnostic.
If a runtime dependency is missing, report `NOT_READY`; do not install it
automatically.

## Choose the workflow

1. **Explain or compare capabilities**
   - Read `references/capability-matrix.md`.
   - Separate implemented measurements from proposed normal-grid, visual, and
     stitching extensions.
2. **Assess readiness or prepare a run**
   - Read `references/operations.md`.
   - Run `scripts/production_readiness.py` before creating work.
3. **Evaluate a deterministic comparison**
   - Read `references/benchmark-contract.md`.
   - Run `scripts/compare_strategies.py` only on a frozen matched cohort.
4. **Investigate an attempt**
   - Read the task, probe run, trials, evaluations, decision, promotion,
     canonical surface, geometry QC, and physical QC receipts.
   - Prefer ledger/API evidence over directory inference.

## Run the readiness gate

Check repository/code readiness:

```bash
python3 "$CAMPAIGN_ROOT/.agents/skills/vc3d-select-seeds/scripts/production_readiness.py" \
  --root "$CAMPAIGN_ROOT" \
  --mode shadow
```

Check a live shadow deployment:

```bash
python3 "$CAMPAIGN_ROOT/.agents/skills/vc3d-select-seeds/scripts/production_readiness.py" \
  --root "$CAMPAIGN_ROOT" \
  --mode shadow \
  --production \
  --db postgres-env://CX_DB \
  --artifact-root s3://BUCKET/PREFIX \
  --output /tmp/vc3d-seed-shadow-readiness.json
```

For `select`, additionally pass the frozen benchmark approval receipt and
optionally restrict the source:

```bash
python3 "$CAMPAIGN_ROOT/.agents/skills/vc3d-select-seeds/scripts/production_readiness.py" \
  --root "$CAMPAIGN_ROOT" \
  --mode select \
  --production \
  --db postgres-env://CX_DB \
  --artifact-root s3://BUCKET/PREFIX \
  --benchmark-receipt /path/to/BENCHMARK_DECISION.json \
  --review-owner "$REVIEW_OWNER" \
  --sample PHerc123 \
  --output /tmp/vc3d-seed-select-readiness.json
```

Proceed only when the receipt says `READY`. Do not turn a failed check into a
warning.

## Operate shadow

Use distinct pre-registered policy versions for baseline and shadow. Queue the
same frozen cells, sources, candidate extractor, worker tier, full-grow profile,
QC policy, and reviewer protocol. Use the CLI documented in
`references/operations.md`.

In `shadow`:

- retain the complete probe ledger;
- leave the existing planner and canonical grow unchanged;
- report top-ranked eligibility, lower-ranked rescue, ambiguity, rejection,
  failure, cost, and replay behavior;
- make no causal canonical-yield claim.

## Operate select

Require all of the following:

- a successful matched benchmark receipt;
- verified immutable CT and m7 source locks;
- the explicit select rollout flag;
- a recent probe-capable PostgreSQL worker;
- an S3 artifact namespace;
- a review-resolution owner;
- no invariant violation.

Allow only Cost-Aware v2 or deterministic-v2. Require at least two candidates.
Continue only an exact unique `CONTINUE_WINNER`. Route ambiguity, missing
evidence, corrupt bytes, exhausted retries, or authority mismatches to review
or failure.

For the isolated causal experiment, use only the pre-registered benchmark
authorization bound into the task. It is not a production-select approval and
must not share a production database, artifact prefix, or credentials.

## Compare fairly

Use `scripts/compare_strategies.py` with a frozen spec and paired result file:

```bash
python3 "$CAMPAIGN_ROOT/.agents/skills/vc3d-select-seeds/scripts/compare_strategies.py" \
  --spec /path/to/BENCHMARK_SPEC.json \
  --results /path/to/BENCHMARK_RESULTS.json \
  --output /path/to/BENCHMARK_DECISION.json
```

Do not approve select from aggregate probe-win counts. Require usable,
nonduplicate, single-lamina canonical area after geometry and physical QC per
compute wall-hour on a matched device tier, reviewer effort, per-scroll rates,
an exact paired superiority test, zero new incorrect-lamina harms within the
frozen safety bound, and zero invariant violations.

## Report

Return:

- mode and exact policy identity;
- source snapshot/content-lock status;
- candidate XYZ and rank without inventing a calibrated probability;
- micro-growth configuration and bounded compute;
- geometry result and explicit non-claims;
- action and exact reason;
- artifact and ledger identities;
- readiness failures or benchmark deltas;
- recommended next action.

Never report normalized m7 intensity as a probability. Never call a probe
winner a validated physical sheet.
