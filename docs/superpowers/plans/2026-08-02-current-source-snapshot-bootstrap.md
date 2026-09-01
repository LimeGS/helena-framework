# Current Source Snapshot Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a P1 bootstrap create and report tasks only for the source snapshot resolved from the current eligible catalog.

**Architecture:** Keep immutable historical snapshots in the fleet store, but derive the runnable snapshot set from the `sources` mapping returned by `bootstrap_sources()`. Reconcile those identifiers against stored rows before task generation and fail closed on missing, duplicate, or sample-mismatched rows.

**Tech Stack:** Python 3.11, pytest, Helena segmentation fleet SQLite test store, PostgreSQL-compatible fleet interfaces.

## Global Constraints

- Preserve every historical snapshot, task, attempt, and receipt.
- Do not alter task identity or database schemas.
- One current snapshot per catalog sample may reach task generation.
- A receipt count must equal the rows the bootstrap attempted to insert.
- Errors must not include source URIs, credentials, worker identities, paths, or DSNs.
- Use TDD: regression must fail for the observed eight-versus-four behavior before production code changes.

---

### Task 1: Reproduce Historical Snapshot Fan-out

**Files:**
- Create: `tests/test_bootstrap_uses_current_snapshot.py`
- Read: `framework/stages/01-segmentation/fleet/generator.py`

**Interfaces:**
- Consumes: `fleet.generator.bootstrap_queue(store, eligible_path, catalog_path, ...) -> dict[str, Any]`
- Produces: a regression proving only the eligible catalog's resolved snapshot is runnable.

- [ ] **Step 1: Write the failing regression**

Create a real temporary `FleetStore`, register an old `PHerc826` snapshot, then provide an eligible catalog entry for a distinct current `PHerc826` snapshot. Call `bootstrap_queue(..., verify_sources=False, max_tasks_per_sample=4)` and assert:

```python
assert receipt["tasks"] == {
    "PHerc826": {"generated": 4, "inserted": 4}
}
assert task_snapshot_ids == {current_snapshot_id}
assert len(task_snapshot_ids_with_repetition) == 4
assert {row["source_snapshot_id"] for row in store.snapshots({"PHerc826"})} == {
    old_snapshot_id,
    current_snapshot_id,
}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest -q tests/test_bootstrap_uses_current_snapshot.py
```

Expected: FAIL because the store contains eight tasks across both historical and current snapshots while the receipt reports only the final four.

### Task 2: Select the Catalog-resolved Snapshot Set

**Files:**
- Modify: `framework/stages/01-segmentation/fleet/generator.py:880-1010`
- Test: `tests/test_bootstrap_uses_current_snapshot.py`

**Interfaces:**
- Consumes: `sources: dict[str, str]` from `bootstrap_sources()` and rows from `store.snapshots(samples)`.
- Produces: `_current_bootstrap_snapshots(rows, sources) -> list[dict[str, Any]]` in deterministic sample order.

- [ ] **Step 1: Add the minimal selector**

Implement a pure helper that indexes rows by `source_snapshot_id`, rejects duplicate identifiers, and resolves each sorted `(sample_id, source_snapshot_id)` pair from `sources`. Raise `ValueError` when the identifier is absent or the row sample differs. Return only the resolved rows.

- [ ] **Step 2: Route normal bootstrap through the selector**

Replace:

```python
snapshots = store.snapshots(samples)
```

with selection through `_current_bootstrap_snapshots(...)`. Do not change the isolated benchmark authorization checks after selection.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
python3 -m pytest -q tests/test_bootstrap_uses_current_snapshot.py
```

Expected: PASS with four stored tasks, four reported inserts, and both immutable snapshots still present.

### Task 3: Fail-closed Edge Coverage

**Files:**
- Modify: `tests/test_bootstrap_uses_current_snapshot.py`
- Modify only if required: `framework/stages/01-segmentation/fleet/generator.py`

**Interfaces:**
- Consumes: `_current_bootstrap_snapshots(rows, sources)`.
- Produces: exact error contracts for missing, duplicate, and sample-mismatched rows.

- [ ] **Step 1: Add parameterized tests**

Assert `ValueError` for:

```python
("missing", "current source snapshot .* is absent")
("duplicate", "source snapshot id .* appears more than once")
("sample mismatch", "belongs to .* not .*")
```

Assert error strings contain identifiers only and none of `ct_uri`, `m7_uri`, `postgresql`, `worker`, or `password`.

- [ ] **Step 2: Verify focused tests**

Run:

```bash
python3 -m pytest -q tests/test_bootstrap_uses_current_snapshot.py tests/test_seed_probe_benchmark_cli.py tests/test_segment_search_fleet.py
```

Expected: all pass.

### Task 4: Review, Full Verification, and Staging Deployment

**Files:**
- Modify if needed: production/test files from Tasks 1-3 only.

**Interfaces:**
- Consumes: reviewed patch and green focused suite.
- Produces: commit, staging pipeline, deployed revision, smoke evidence, and a read-only post-deploy check.

- [ ] **Step 1: Independent review**

Dispatch a read-only reviewer to compare the patch to the design, focusing on source authority, historical preservation, benchmark behavior, receipt truth, and secret-safe errors. Address every Critical or Important finding through a new RED/GREEN cycle.

- [ ] **Step 2: Run the full local suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
```

Record pass, skip, warning, duration, and commit SHA.

- [ ] **Step 3: Commit and push the feature branch**

Stage only the new spec, plan, tests, and generator change. Preserve the unrelated untracked First Letters campaign plan. Push `codex/fix-qc-review-stack-unpack`.

- [ ] **Step 4: Fast-forward staging and audit CI**

Push the same commit to `staging`. Require build, unit, frontend, panel build, gpu-1 deploy, and gpu-1 smoke success for the exact SHA.

- [ ] **Step 5: Verify runtime convergence**

Read-only verify all Helena services report the exact deployed revision and no transient build containers remain. Do not enqueue a replacement PHerc826 wave: the original eight terminal negatives remain the scientific evidence for this bounded campaign.
