# Benchmark contract

## Contents

1. Experimental boundary
2. Frozen specification
3. Result records
4. Decision metrics
5. Interpretation

## Experimental boundary

Run any steering comparison in an isolated, nonproduction control plane and
artifact namespace. `shadow` alone can measure replay, rescue opportunities,
ambiguity, and incremental probe cost, but it cannot cause a canonical-yield
improvement because it deliberately leaves the canonical seed unchanged.

Use a paired outcome experiment only after pre-registering:

- 40–60 ambiguous cells;
- at least three scrolls;
- exact cell assignments;
- baseline and closed-loop policy versions;
- identical source snapshots, worker tier, full-grow profile, QC, dedup, and
  blinded review;
- spatially distinct pre-registered independence blocks;
- numerical approval thresholds;
- an isolated execution scope.

Do not enable production select merely to collect benchmark evidence.

## Frozen specification

`compare_strategies.py` accepts:

```json
{
  "schema": "campaignx.seed_probe_benchmark_spec.v1",
  "benchmark_id": "seed-probe-2026q3-v1",
  "frozen_at_utc": "2026-08-01T00:00:00Z",
  "execution_scope": "ISOLATED_NONPRODUCTION",
  "baseline": {
    "policy_version": "deterministic-v2-baseline-v1",
    "planner": "deterministic-v2",
    "seed_probe_mode": "off"
  },
  "closed_loop": {
    "policy_version": "seed-probe-select-benchmark-v1",
    "planner": "deterministic-v2",
    "seed_probe_mode": "select"
  },
  "minimum_cells": 40,
  "maximum_cells": 60,
  "minimum_scrolls": 3,
  "minimum_relative_yield_improvement": 0.1,
  "maximum_relative_reviewer_rate_regression": 0.0,
  "maximum_incremental_compute_wall_hours_per_cell": 0.25,
  "maximum_new_incorrect_lamina_rate_upper_bound": 0.05,
  "paired_test_alpha": 0.05,
  "minimum_pairs_per_scroll": 5,
  "review_protocol_id": "blind-single-lamina-review-v1",
  "cells": [
    {
      "cell_id": "cell-001",
      "sample_id": "PHerc123",
      "independence_block_id": "PHerc123-block-001"
    }
  ]
}
```

List every paired cell and spatial independence block exactly once. Use
distinct policy versions. With a 5% new-harm upper bound, at least 59
zero-new-harm pairs are required; a 40-cell cohort can still diagnose the
system but cannot approve that safety margin.

## Result records

Provide one record per cell and arm. The abbreviated example below shows one
baseline row; supply its `closed_loop` pair and every other registered pair.

```json
{
  "schema": "campaignx.seed_probe_benchmark_results.v1",
  "benchmark_id": "seed-probe-2026q3-v1",
  "started_at_utc": "2026-08-02T00:00:00Z",
  "records": [
    {
      "cell_id": "cell-001",
      "sample_id": "PHerc123",
      "independence_block_id": "PHerc123-block-001",
      "arm": "baseline",
      "policy_version": "deterministic-v2-baseline-v1",
      "planner": "deterministic-v2",
      "seed_probe_mode": "off",
      "source_content_lock_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "candidate_set_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "vc3d_binary_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "full_grow_profile_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
      "full_grow_envelope_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
      "rng_seed": "matched-cell-001-v1",
      "worker_tier": "cpu-standard-v1",
      "compute_device": "cpu",
      "review_protocol_id": "blind-single-lamina-review-v1",
      "reviewer_blinded": true,
      "canonical_area_cm2": 1.2,
      "usable_nonduplicate_single_lamina_area_cm2": 0.8,
      "compute_wall_hours": 2.0,
      "reviewer_minutes": 8.0,
      "geometry_rejected": false,
      "incorrect_lamina": false,
      "abstained": false,
      "geometry_qc_passed": true,
      "physical_qc_passed": true,
      "dedup_passed": true,
      "single_lamina_confirmed": true,
      "canonical_output_invariant_ok": true,
      "lease_invariant_ok": true,
      "replay_invariant_ok": true,
      "budget_invariant_ok": true
    }
  ]
}
```

Every result must preserve the pre-registered identity. Do not omit a failed
arm; record zero canonical and usable area and its actual total compute cost.
`compute_wall_hours` includes probes, failed attempts, retries, and the full
grow attributable to that arm. An abstention or geometry rejection cannot
report canonical or usable area.

The runtime does not yet export this result document end to end. Build it as a
durable adjudication artifact from final grow, geometry QC, physical QC, dedup,
compute, and blinded-review receipts; then freeze and hash it before comparison.
The comparison script validates exact identities and contradictions, but it
does not prove that a manually supplied boolean came from the ledger.

## Decision metrics

Calculate:

- usable area per compute wall-hour from cohort totals;
- reviewer minutes per usable square centimetre;
- relative closed-loop change for both rates;
- incremental compute wall-hours per paired cell;
- per-scroll yield, rejection, and abstention;
- an exact one-sided paired sign test in which only cells exceeding the frozen
  yield margin count as wins;
- minimum sample plus yield and reviewer-rate non-regression on every scroll;
- new closed-loop-only incorrect-lamina harms and their exact one-sided upper
  bound, without netting them against baseline-only harms;
- all invariant violations.

Approve only when sample-size, scroll-count, identity, temporal ordering,
paired-evidence, per-scroll, safety, threshold, and invariant checks all pass.
The decision receipt authorizes only its explicit `authorized_sample_ids`.
Evidence from three scrolls is a scoped canary, not permission to steer an
untested scroll.

## Interpretation

An approved receipt authorizes a rollout decision; it does not prove a seed or
surface is physically correct. A failed or incomplete receipt must remain
`NOT_APPROVED`. Never replace missing measurements with assumptions.

Matched RNG and envelope identities also do not prove that VC3D
`12 generations -> resume -> target 35` is numerically equivalent to a fresh
35-generation grow. The paired benchmark measures the closed-loop system as a
treatment and must charge all probe/resume compute; validate that runtime
behavior separately before production rollout.
