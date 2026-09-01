# V19 Shadow Bridge Corrective TDD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the v19 first-letters shadow bridge account for exact retained history, enforce job-only execution, and fail closed on every current-control or persisted-graph drift in SQLite and PostgreSQL.

**Architecture:** History reconciliation will enumerate an exact retained execution projection and materialize non-executable historical reservation/import rows so cap accounting has one ledger. A shared validation flow in each backend will compare persisted JSON with all duplicated scalar columns and re-resolve current history, block, source, profile, cap, item, and mission authority at read/claim/revalidation boundaries. PostgreSQL incomplete reconciliation will return an outcome from the transaction and raise only after the connection context commits.

**Tech Stack:** Python 3, SQLite, PostgreSQL/psycopg2, pytest.

## Global Constraints

- Strict RED-GREEN per task: production cannot change until the named test fails for the expected missing behavior.
- No remote execution.
- Do not touch `docs/superpowers/plans/2026-08-02-first-letters-hybrid-campaign.md`.
- Do not commit until every scoped matrix test passes and the final diff is reviewed.
- Historical imports account for retained work but can never become executable jobs.

---

### Task 1: Exact retained history and cap accounting

**Files:**
- Modify: `tests/test_first_letters_discovery_shadow_bridge.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`

**Interfaces:**
- Consumes: retained v16 reservation/work and discovery execution rows already present in the SQLite schema.
- Produces: `reconcile_first_letters_discovery_history(mission_id=...)` with exact imports and cap-visible historical reservations.

- [ ] Add a test that inserts one complete retained v16 execution with literal source/profile/run/claim/evidence/file bytes, reconciles it, and asserts one `IMPORTED_HISTORICAL_EXACT` reservation, one `first_letters_discovery_historical_imports_v19` row, `fixed_units == 24`, and only 24 cap units remain.
- [ ] Parameterize independent REDs removing or adding one run/claim/evidence/file row and changing one retained byte; assert durable `CONTROL_INCOMPLETE` reconciliation plus compute block and zero new live reservations.
- [ ] Run each new node independently and record the expected failure: missing import row, cap total 0, or missing fail-close.
- [ ] Implement exact enumeration and atomic materialization, deriving identifiers only from retained rows and bytes.
- [ ] Re-run each node and the complete shadow-bridge module to GREEN.

### Task 2: Current history/watermark revalidation

**Files:**
- Modify: `tests/test_first_letters_discovery_shadow_bridge.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`

**Interfaces:**
- Consumes: a sealed v19 job claim and current retained graph/block state.
- Produces: claim/revalidation that rejects history drift before provider preparation.

- [ ] Add REDs for a new retained row after reservation and for a newly persisted block after reservation; execute through `FirstLettersDiscoveryController.run_job` and assert the raised control error, no run beyond `CLAIMED`, and provider prepare/execute counts both zero.
- [ ] Run the two tests separately and record that current code completes and calls provider once.
- [ ] Re-enumerate history and consult the block inside claim/revalidation instead of trusting the first stored reconciliation row.
- [ ] Re-run both nodes and the shadow-bridge module to GREEN.

### Task 3: PostgreSQL durable incomplete reconciliation

**Files:**
- Modify: `tests/test_first_letters_discovery_postgres.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`

**Interfaces:**
- Consumes: a mission with a nonempty or changed retained history cutover.
- Produces: committed reconciliation/block rows followed by `CONTROL_INCOMPLETE_COMPUTE_LEDGER` to the caller.

- [ ] Add a transaction-behavior RED using a real connection double that commits on normal `__exit__` and rolls back on exceptional `__exit__`; assert helper outcome exits normally, both rows are committed, then public reservation raises.
- [ ] Run the node and record rollback under the current in-transaction raise.
- [ ] Return an incomplete outcome from `_first_letters_empty_history_tx`, leave the connection context normally, and raise after commit in `_reserve_first_letters_shadow`.
- [ ] Re-run the node and PostgreSQL static tests to GREEN.

### Task 4: Job-only execution boundary

**Files:**
- Modify: `tests/test_first_letters_discovery_shadow_bridge.py`
- Modify: `tests/test_first_letters_discovery_controller.py`
- Modify: `framework/stages/01-segmentation/fleet/discovery_worker.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`

**Interfaces:**
- Consumes: only a v19 `job_id` at the executable public boundary.
- Produces: legacy `run_item`/public `begin` rejection before provider prepare for controlled and retained v16 reservations.

- [ ] Add separate REDs for Worker `run_item` against a controlled reservation and a retained v16 reservation, plus direct public `begin`; assert rejection and zero provider prepare/execute.
- [ ] Run each node and record controlled prepare=1 and retained v16 completion.
- [ ] Move the legacy-boundary rejection before provider preparation and make public no-job begin uniformly non-executable.
- [ ] Re-run the nodes and controller/evidence suites to GREEN.

### Task 5: Scalar, JSON, and current-authority drift matrix

**Files:**
- Modify: `tests/test_first_letters_discovery_shadow_bridge.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`

**Interfaces:**
- Consumes: persisted adapter/dispatch/job branches and current source/profile/cap/item/mission authorities.
- Produces: exact readback and pre-provider revalidation.

- [ ] Add parameterized REDs that independently drift every duplicated adapter, dispatch, and job scalar away from its JSON, and independently drift adapter/dispatch/job JSON hashes, source, cap, profile bytes, item, and mission.
- [ ] Assert `CONTROL_INCOMPLETE_DISCOVERY_DISPATCH` or claim-graph failure and provider prepare/execute both zero; run each parameter group independently and record accepted cases.
- [ ] Add complete scalar-to-JSON comparison and current-authority resolution to both backends' read/claim/revalidation paths.
- [ ] Re-run every parameter group and the SQLite/PostgreSQL scoped suites to GREEN.

### Task 6: Exact scientific dependency reconciliation

**Files:**
- Modify: `tests/test_first_letters_discovery_shadow_bridge.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`

**Interfaces:**
- Consumes: retained source/profile/item authority.
- Produces: retained projections bound to CT, M7, read sets, model, provider, resolution, level, transform, threshold, opportunity, and region.

- [ ] Add REDs changing each scientific dependency independently and omitting persisted opportunity/region; assert reconciliation fails before reservation count or cap total changes.
- [ ] Run dependency groups separately and record current acceptance or invented fallback values.
- [ ] Derive the projection only from exact persisted source/profile/item rows and reject absent or inconsistent fields.
- [ ] Re-run all dependency nodes and shadow-bridge tests to GREEN.

### Task 7: Honest PostgreSQL live and real OFF/shadow projections

**Files:**
- Modify: `tests/test_first_letters_discovery_postgres.py`
- Modify: `tests/test_first_letters_discovery_controller.py`

**Interfaces:**
- Consumes: real baseline/alternative producer adapters and controller/worker database effects.
- Produces: live tests that either run v19 behavior under `HELENA_TEST_DSN` or explicitly skip it, and OFF/shadow comparisons over real row projections.

- [ ] Replace live helper setup with server profile/arm resolvers and `reserve_first_letters_baseline_shadow` / `reserve_first_letters_alternative_shadow`; add static/no-DSN RED proving the helper no longer reaches generic reserve.
- [ ] Replace synthetic `off_shadow_mutation_projections` inputs with before/after projections queried from all affected controller/worker tables; run the comparison RED independently.
- [ ] Update fixtures minimally and run PostgreSQL tests without DSN, recording honest skips, then run controller/shadow tests to GREEN.

### Final matrix and diff gate

- [x] Run every new RED/GREEN node list, then all first-letters discovery tests.
- [x] Run the broader stage test matrix selected by the owner.
- [x] Inspect `git diff --check`, `git status --short`, and the full diff; confirm the hybrid plan remains untouched.
- [x] Only after all checks, create the single corrective commit requested by the owner.

## Verification ledger — 2026-08-03

- SQLite retained scientific authority: exact positive plus 21 independent dependency/omission cases passed.
- PostgreSQL retained scientific authority: exact positive plus 21 independent dependency/omission cases passed (`22 passed`).
- PostgreSQL retained projections: a psycopg-style timezone-aware `datetime` reproduced a JSON hashing `TypeError`; UTC timestamp canonicalization then passed with the retained/orphan/science set at `25 passed`.
- PostgreSQL current block/history pre-provider matrix: two expected REDs on the stale reconciliation lookup, then `2 passed` after advisory-lock/current-history revalidation.
- PostgreSQL orphan-root matrix: reservation/work cases were both expected RED (`COMPLETE`), then `2 passed`; the FK-valid live reservation-without-work gate was collected and skipped because no live DSN was configured.
- Production factory restart and evidence/bridge regression: `235 passed`.
- Compute cohort fixture regression: `27 passed`; production validation was unchanged.
- Scoped discovery matrix: `520 passed, 22 skipped`.
- Expanded First Letters, fleet backend, and M7/panel dependency matrix after all fixes: `995 passed, 25 skipped`.
- `git diff --check` and Python `compileall` completed with no errors.
- `HELENA_TEST_DSN` was absent. The newly added live PostgreSQL serialized-history race, orphan reservation, and baseline/alternative OFF-shadow projection gates were collected but **skipped, not passed** (four cases). No remote execution was performed.
