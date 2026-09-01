# First Letters Discovery Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Helena demonstrate that it can recover a known community-confirmed target, measure candidate availability before spending growth jobs, adapt or stop when the candidate source is unproductive, and keep exploratory results strictly separated from publishable acceptance evidence.

**Architecture:** The program adds three gates ahead of a new First Letters campaign: a source-locked positive-control gate, an ink-blind M7/CT/clearance coverage preflight, and a power-based campaign decision. Discovery runs in a noncanonical namespace using the existing seed-probe safety model; only a normal full grow that passes the unchanged geometry, physical-QC, flattening, rendering, ink, screening, and human-review gates can enter the acceptance path.

**Tech Stack:** Helena FastAPI control plane, PostgreSQL segmentation fleet, VC3D MCP candidate provider, CT material-support screen, seed-probe v1, geometry and physical QC, ABF flattening, TIFXYZ rendering, TimeSformer ink screening, React panel, pytest, Vitest, and the existing deployment/smoke workflow.

## Global Constraints

- This document is a review draft. Do not implement, deploy, enqueue, cancel, reorder, or delete work until the user approves it.
- Every real-data mutation must use Helena's authenticated API and a named mission. Direct SQL, direct queue edits, and manual artifact promotion are forbidden.
- A positive control must be community-confirmed, publicly attributable, source-locked, and disjoint from evaluation targets. A synthetic probability-map test is not sufficient.
- Discovery output is noncanonical and cannot satisfy P2/P3 admission. It may create a normal full-grow request only through an explicit, content-bound promotion contract.
- Acceptance mode retains the current geometry, physical-QC, P3, P4, P5, P7, and human-review gates. No `allow_unvalidated` path is permitted for campaign evidence.
- Candidate and search decisions remain ink-blind. Ink outputs must never influence P1 cell selection, seed ranking, task budgeting, or early stopping.
- A negative campaign outcome means the bounded method did not produce a reviewable candidate. It never means that a scroll contains no ink, text, or letters.
- Existing P0 artifacts, attempts, surfaces, QC jobs, receipts, and published artifacts remain immutable.
- Every new policy, grid, prediction source, model, threshold, and experimental arm receives a new versioned identity.
- Scientific terminal outcomes and platform failures remain separate denominators. Worker crashes, source outages, publication failures, and cancelled jobs do not count as negative scientific attempts.

## Program-level acceptance criteria

The recovery program is ready for a new search campaign only when all of these are true:

1. A community-confirmed control reaches its preregistered expected outcome through the exact deployed revision and production profiles.
2. Every target scroll has a content-bound preflight receipt reporting M7, CT, clearance, and spatial coverage counts.
3. Every target scroll has either a justified task budget with achieved detection probability or an explicit `DO_NOT_QUEUE_CURRENT_SOURCE` decision.
4. Candidate-starvation rules automatically pause new queue creation without cancelling already-running work.
5. Discovery artifacts cannot appear in canonical surface or downstream-job tables unless a normal full grow passes existing admission gates.
6. PHerc0358's 8/8/0 result is reproduced and explained without changing the global margin.
7. Surfaces below 0.10 cm2 are routed to a diagnostic path and cannot consume standard physical-QC or downstream acceptance capacity.
8. Focused, integration, end-to-end, full Python, frontend, deploy, and smoke tests pass on the exact revision.

---

### Task 1: Freeze the scientific contract and positive-control cohort

**Files:**
- Create: `docs/superpowers/specs/2026-08-02-first-letters-discovery-recovery-design.md`
- Create: `framework/profiles/01-segmentation/first-letters-control-policy-1.0.0.json`
- Create: `framework/profiles/01-segmentation/first-letters-campaign-decision-policy-1.0.0.json`
- Test: `tests/test_first_letters_discovery_policy.py`

**Interfaces:**
- Consumes: public community evidence, a P0 CT source, an M7 source, known CT-L0 coordinates or a source-locked community surface, and the current acceptance profiles.
- Produces: `campaignx.first_letters_control_manifest.v1` and `campaignx.first_letters_campaign_policy.v1`.

- [ ] **Step 1: Select and independently verify the control**

  Select one community-confirmed target with visible ink or letters and a stable public reference. Freeze the public reference, scroll ID, CT source identity, M7 source identity, coordinate frame, voxel size, known region, expected observation, and hashes where the source exposes bytes. Do not reuse any of the thirteen target regions as the evaluation control.

- [ ] **Step 2: Split control responsibilities explicitly**

  The control manifest must define two checks. `DISCOVERY_CONTROL` asks whether the frozen M7/provider/configuration offers at least one candidate inside the known region. `PIPELINE_CONTROL` starts from a provenance-marked manual seed or source-locked community surface and tests grow, geometry, physical QC, P3, P4, P5, P7, and human-review routing. One end-to-end control may satisfy both checks only if the community evidence genuinely fixes the expected seed region and surface.

- [ ] **Step 3: Freeze pass/fail semantics**

  A discovery pass requires at least one post-CT, post-clearance candidate inside the declared control tolerance. A pipeline pass requires a canonical full grow, geometry certification, a terminal physical-QC result compatible with downstream use, complete P3/P4 artifacts, live P5 output, and P7 routing consistent with the known target. Any missing stage is `CONTROL_INCOMPLETE`; a contradictory terminal result is `CONTROL_FAILED`.

- [ ] **Step 4: Test the policy loader fail closed**

  Add tests that reject missing source locks, missing expected outcomes, overlapping control/evaluation regions, non-finite coordinates, unversioned profiles, ink-informed discovery fields, and a control that only names the existing synthetic probability-map test.

- [ ] **Step 5: Commit the contract separately**

  Commit only the design, profiles, and contract tests. No execution path changes belong in this commit.

### Task 2: Build a stage-by-stage positive-control runner

**Files:**
- Create: `scripts/harness/run_first_letters_positive_control.py`
- Modify: `scripts/harness/panel_client.py`
- Modify: `framework/stages/01-segmentation/fleet/generator.py`
- Test: `tests/test_first_letters_positive_control.py`
- Test: `tests/e2e/test_first_letters_positive_control.py`

**Interfaces:**
- Consumes: `campaignx.first_letters_control_manifest.v1`, a human Helena session, and a named control mission.
- Produces: `campaignx.first_letters_stage_survival.v1` containing one row per P0/P1/P2/QC/P3/P4/P5/P7/human-review boundary plus content SHA-256.

- [ ] **Step 1: Write the failing orchestration tests**

  Cover a complete control, an M7 discovery miss, CT rejection, clearance rejection, grow failure, tiny surface, geometry rejection, physical-QC rejection, missing flattened artifact, incomplete render manifest, dead P5 map, negative P7 routing, and an ambiguous POST. Assert that the runner never retries an ambiguous mutation and never advances after a failed prerequisite.

- [ ] **Step 2: Reuse the existing manual-seed path for isolation**

  Use `bootstrap-manual` for `PIPELINE_CONTROL`, retaining `seed_origin=human`, submitter identity, exact coordinates, CT material-support screening, and a dedicated control policy version. This isolates downstream sensitivity from M7 candidate availability without disguising a manual seed as an autonomous discovery.

- [ ] **Step 3: Execute the discovery half without growth**

  Run the same provider, threshold, CT material gate, and clearance screen used by a normal task over the known control region. Record raw M7, post-CT, post-clearance, and packet-limited counts plus the distance from the closest surviving candidate to the known control coordinate.

- [ ] **Step 4: Emit a survival matrix**

  Each stage row records input artifact IDs/hashes, profile IDs, parameters, candidate or artifact counts, terminal state, elapsed time, resource identity, output hashes, and a non-claim. The overall control state is the first failed or incomplete stage, not a hand-authored summary.

- [ ] **Step 5: Gate campaigns on the deployed control**

  A control receipt is valid only for the exact deployed code revision, source locks, models, and profiles it names. Any relevant revision or input change makes it stale and requires a rerun before a new discovery campaign.

### Task 3: Extend M7 survey into a full candidate-coverage preflight

**Files:**
- Create: `framework/stages/01-segmentation/fleet/candidate_preflight.py`
- Modify: `framework/stages/01-segmentation/fleet/cli.py`
- Modify: `framework/stages/01-segmentation/fleet/ct_support.py`
- Modify: `panel/app.py`
- Modify: `panel/web/src/routes/Coverage.tsx`
- Test: `tests/test_candidate_coverage_preflight.py`
- Test: `tests/test_panel_serves_what_the_pages_read.py`
- Test: `panel/web/src/routes/Coverage.test.tsx`

**Interfaces:**
- Consumes: one frozen P0 source, grid and policy versions, provider configuration, M7 threshold, CT support policy, cell clearance, volume clearance, and selection strategy.
- Produces: `campaignx.segment_candidate_coverage_preflight.v1` without inserting, claiming, leasing, or completing a segmentation task.

- [x] **Step 1: Extract the existing survey primitive**

  Refactor `command_survey` so both pending-task survey and preflight call one pure operation: provider discovery, CT material-support filtering, clearance filtering, deterministic candidate ranking, and receipt construction. Preserve existing `campaignx.segment_fleet_m7_survey.v1` behavior.

- [x] **Step 2: Generate preflight cells in memory**

  Reuse `generate_tasks_for_snapshot` against a read-only surface view, but do not call `insert_tasks`. Enumerate the complete eligible grid for the chosen grid version. If the operator supplies a bounded sample, the receipt must label counts as estimates and report the deterministic sampling design; it must not present them as a census.

- [x] **Step 3: Record the complete candidate funnel**

  Report total grid cells, geometrically eligible cells, cells surveyed, cells with raw M7 candidates, raw candidate total, CT-retained candidates, clearance-retained candidates, packet-retained candidates, source errors, and no-candidate causes. Include minimum/median/p95 scores and clearances without using ink.

- [x] **Step 4: Record spatial distribution**

  Emit deterministic coarse XYZ bins containing total cells, surveyed cells, candidate-bearing cells, and usable-candidate cells. Include candidate coordinates only in the private receipt; a sanitized derivative uses bins and counts.

- [x] **Step 5: Bind preflight to its inputs**

  Hash the canonical JSON receipt and record P0 artifact ID/hash, source snapshot, CT and M7 locks, coordinate frame, voxel size, grid, threshold, all gates, code revision, and provider response hashes. The same inputs must reproduce the same cell order and aggregate counts.

- [x] **Step 6: Expose a bounded authenticated API**

  Add `POST /api/segmentation/preflight`. It must require mission write access, resolve the mission's P0 selection, accept a hard cell limit and parallelism cap, return the exact command with secrets redacted, and perform no fleet-state mutation. Return 409 for stale/missing P0, unsupported provider, source-lock mismatch, or an already-running identical preflight.

- [x] **Step 7: Extend the Coverage page**

  Show candidate availability separately from attempted-cell coverage. Display the raw -> CT -> clearance funnel, spatial bins, exact-versus-estimated label, planned sampling percentage, source errors, and the explicit statement that candidate scarcity is not surface or ink absence.

#### Task 3 implementation ledger

- **Implemented:** 2026-08-03 on `codex/fix-qc-review-stack-unpack`. The preflight now uses a database-enforced read-only fleet view, a bounded deterministic census/estimate design, the shared provider -> CT -> cell-clearance -> volume-clearance -> packet survey primitive, immutable private/sanitized receipt pairs, an authenticated mission/P0-bound API, and a Coverage UI that separates planned from achieved sampling and hides invalid evidence.
- **Scientific receipt:** the signed core binds the authoritative catalog bytes, P0 selection and artifact, source snapshot and content lock, exact 40-hex code revision, provider/policy/grid/threshold/gates, normalized request, sampling design, per-cell outcomes, manifest digests, aggregate funnel, statistics, spatial bins, and private retained coordinates. Volatile timestamps do not change the scientific digest.
- **Read-only and publication boundaries:** SQLite opens in `mode=ro` with `query_only`; PostgreSQL uses a read-only session; schema compatibility is verified without initialization or DDL. Cross-process mission locks prevent identical concurrent runs. Receipt pairs publish create-once with private-only repair and rollback if the sanitized half cannot be published.
- **Review loop 1:** adversarial tests initially proved that nine internally rehashed derivation drifts were accepted. Validation was consolidated to reject normalized-request drift, state mutation, coordinate order/bounds errors, unordered statistics, duplicate manifests, unordered spatial bins, impossible per-cell stage growth, and noncanonical sampling labels/rules.
- **Review loop 2:** independent review found that rehashed `grid_step`, a non-integer `parallelism`, and a relabelled sample were not independently derived. The receipt now binds exact request types and ranges, enforces `grid_step >= 2 * query_radius`, `volume_clearance >= query_radius`, and reconstructs the bounded grid to verify population, sampled indices, ordinal parameters, hash, and spatial totals.
- **Review loop 3:** independent review found that an internally self-consistent sample relabel could remain current. The loader now checks the canonical requested sample/evidence root against the signed core, request, binding, and registered source snapshot before returning counts; mismatch is `INVALID`, not `STALE`.
- **Review loop 4:** the analogous mission-root relabel was reproduced and closed. The requested mission/evidence root must match the signed normalized request and binding before freshness evaluation. Wrong-root evidence is `INVALID` with no funnel; legitimate same-root P0/source drift remains visible as `STALE`.
- **Independent verdict:** final read-only re-review reported no Critical or Important findings and `Ready: YES`. The reviewer independently ran 180 targeted Python tests, all 4 Coverage tests, and `git diff --check` without edits.
- **Final verification:** full Python suite outside the restricted socket/DNS sandbox: `2144 passed, 129 skipped, 0 failed` in 95.50 seconds. Full frontend suite: 11 files and 33 tests passed. TypeScript/Vite production build, Python `compileall`, and `git diff --check` passed. The focused candidate-control/API suite also passed 266 tests.
- **Operational status:** implementation is locally complete. No push or deploy is part of this Task 3 commit; production execution remains gated by the normal staging pipeline, deploy, smoke, and source-locked real-data preflight evidence.

### Task 4: Derive task budgets from measured search power

**Files:**
- Create: `framework/stages/01-segmentation/fleet/campaign_decision.py`
- Modify: `framework/profiles/01-segmentation/first-letters-campaign-decision-policy-1.0.0.json`
- Modify: `panel/app.py`
- Test: `tests/test_first_letters_campaign_decision.py`

**Interfaces:**
- Consumes: one or more preflight receipts and a frozen compute budget.
- Produces: `campaignx.first_letters_task_budget.v1` and an initial `CONTINUE`, `PAUSE_CANDIDATE_STARVATION`, or `DO_NOT_QUEUE_CURRENT_SOURCE` decision.

- [x] **Step 1: Implement finite-population budgeting**

  For a census with `K` usable candidate-bearing cells among `N` eligible cells, choose the smallest sample without replacement whose probability of selecting at least one usable cell is at least 0.95. When only a sample exists, use the preregistered conservative lower confidence bound for candidate-bearing prevalence. Report both the requested task count and the achieved probability after applying the compute cap.

- [x] **Step 2: Refuse false precision**

  If a complete census finds zero usable cells, return `DO_NOT_QUEUE_CURRENT_SOURCE`; more tasks under the same M7 source, grid, threshold, CT policy, and clearance policy cannot solve that result. If a sample has zero usable cells, report an upper confidence bound and require either a larger preflight or an alternate source; do not invent a task budget from zero.

- [x] **Step 3: Replace the fixed four-task cap**

  Queue requests consume the frozen budget receipt and cannot exceed it. The response records eligible population, planned sample count, planned percentage, target detection probability, achieved probability, and compute cap. A manual lower budget remains allowed only when the receipt clearly reports its lower achieved probability.

- [x] **Step 4: Test edge cases**

  Cover zero candidates, one candidate, all cells positive, budget above population, budget clipped by compute cap, estimated rather than census inputs, invalid or stale preflight hashes, and changed M7/clearance profiles.

#### Task 4 implementation ledger

- **Implemented:** 2026-08-03 on `codex/fix-qc-review-stack-unpack`. The fixed four-task cap is replaced by a content-bound task-budget receipt, exact current-population scan, deterministic outcome-independent probability prefix, frozen compute cap, authenticated panel endpoints, controlled-mission CLI admission, and transactional SQLite/PostgreSQL enforcement.
- **Census interpretation:** when the preflight is a complete census with `K > 0` usable candidate-bearing cells among `N` eligible cells, the requested task count is the smallest `n` for which the exact without-replacement hypergeometric probability `1 - C(N-K,n)/C(N,n)` reaches 0.95. The implementation locates the boundary in `O(N)` log space and verifies the final boundary with exact integer combinations, so floating-point rounding cannot add or remove a task.
- **Sampled-preflight interpretation:** a positive sampled result uses the one-sided 95% Clopper-Pearson lower prevalence bound and maps that conservative prevalence to a finite-population usable-cell count. This is explicitly labelled `MODEL_BASED_BINOMIAL_EXCHANGEABILITY_NOT_DESIGN_BASED`; it is not presented as a design-based guarantee. A zero-positive sample reports an upper bound and requires more preflight or a new source instead of inventing a queue budget.
- **Scope of the probability:** the target is the probability that the frozen prefix contains at least one cell already classified as usable by the exact M7, CT-support, clearance, and packet-retention policy. It is not the probability of producing a valid surface, ink, text, or a letter. A complete zero-usable census blocks only the unchanged current source/policy; it does not support an absence claim.
- **Compute interpretation:** the frozen cap truncates one nested probability order and the receipt reports requested tasks, planned tasks, planned population percentage, and achieved probability after clipping. A zero cap yields `NO_COMPUTE_AUTHORIZED`. A manually lower positive budget requires a reason and exposes its lower achieved probability.
- **Population and receipt integrity:** the population is scanned exactly with a hard 262,144-cell limit and a compact streaming accumulator. The order seed excludes candidate outcomes and the compute cap, so different caps are nested prefixes rather than outcome-selected samples. Private and sanitized receipts are immutable and content-addressed; the browser-safe receipt removes the cell prefix, and all scientific bindings expose only `m7_uri_sha256`, never a credential-bearing operational URI.
- **Admission boundary:** the mission contract freezes the campaign kind, policy profile/hash, P0 scope, source, preflight and budget identities. Bootstrap and resume validate the current mission/P0/preflight pair before writes; nonprefix creation paths and mission omission cannot bypass the controlled campaign. Every queued task must match the registered receipt, exact prefix/rank, grid-derived center/bounds, source snapshot, real prediction URI resolved from that snapshot, provider, planner, parameter envelope, gates, and discovery knobs.
- **Review loop 1:** adversarial mutation tests closed task-count and probability-boundary errors, cap-dependent reordering, incomplete execution bindings, stale P0/preflight acceptance, and uncontrolled bootstrap/resume/seed-recovery paths. The exact population proof and queue envelope are now derived and revalidated instead of trusted from caller JSON.
- **Review loop 2:** independent review showed that a task could alter both its payload and mutable `campaign_budget.execution_bindings` coherently. SQLite and PostgreSQL now use an immutable create-once admission registry keyed by mission, sample, and receipt hash, and validate the task against both that registry and the authoritative registered source snapshot within the task-creation transaction.
- **Review loop 3:** independent review found that raw `m7_uri` could enter public/private scientific bindings. The signed binding is now only its SHA-256; task execution resolves the operational URI from the authenticated immutable source snapshot. Generated-receipt tests use a credentialed URI and prove that neither private nor sanitized preflight/budget/browser evidence exposes it.
- **Review loop 4:** registration-first races initially allowed an unbudgeted first task before any controlled task row existed. Both stores now check mission-level admission authority inside SQLite `BEGIN IMMEDIATE` or the PostgreSQL mission advisory lock, so same-sample and cross-sample/nonprefix omissions fail before insertion.
- **Review loop 5:** the inverse task-first race initially allowed controlled authority to be registered over a pre-existing generic mission task. Registration now takes the same mission-level lock before any receipt lock, rejects pre-existing nonmatching mission tasks, and permits idempotent re-registration only when persisted tasks carry the exact admission and prefix rank. Both race orderings have explicit SQLite and PostgreSQL contract tests.
- **Independent verdict:** final read-only re-review reported `Ready: YES` with no Critical or Important findings. Fresh adversarial mutation/replay coverage was clean, including unregistered first writer, simultaneous task-plus-envelope forgery, same/cross-sample omission, both registration race orderings, exact geometry/source-query drift, concurrent idempotent registration, and credential-URI leakage.
- **Final verification:** full Python suite outside the restricted socket/DNS sandbox: `2241 passed, 129 skipped, 0 failed` in 100.44 seconds. The broad affected suite passed `385 tests` with `3 skipped`; Python `compileall` and `git diff --check` passed. Task 4 changed no frontend source, so the latest complete frontend evidence remains Task 3's `33/33` tests and successful production build.
- **Operational status:** implementation is locally complete. No push or deploy is part of this Task 4 commit; production use still requires the normal branch pipeline, exact-revision deploy/smoke, and real current-source evidence.

### Task 5: Add an automatic candidate-starvation pause

**Files:**
- Modify: `framework/stages/01-segmentation/fleet/campaign_decision.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`
- Modify: `panel/app.py`
- Modify: `panel/web/src/routes/Coverage.tsx`
- Test: `tests/test_first_letters_campaign_stop.py`
- Test: `tests/integration/test_first_letters_campaign_stop.py`

**Interfaces:**
- Consumes: mission-scoped terminal P1 attempts and the frozen campaign policy.
- Produces: an immutable `campaignx.first_letters_campaign_decision.v1` receipt that gates new queue insertion.

- [x] **Step 1: Freeze the first stopping rule**

  Evaluate after at least eight scientific-terminal P1 attempts. Pause when seven or more of the first eight are `NO_M7_CANDIDATES`, or when two consecutive scrolls complete their frozen budgets with zero raw M7 candidates. Re-evaluate after every additional block of eight scientific-terminal attempts.

- [x] **Step 2: Keep denominators honest**

  Exclude cancelled tasks, source failures, worker failures, lease exhaustion, publication failures, and configuration blocks. A mixed-cause `NO_SEED` counts as `NO_M7_CANDIDATES` only when its recorded raw M7 count is zero.

- [x] **Step 3: Pause future creation, not running work**

  The decision prevents new P1 queue insertion for that mission and policy. It never cancels or reprioritizes work already pending, leased, or running. A pause must remain visible even after all queues become empty.

- [x] **Step 4: Require a materially new strategy to resume**

  Resume requires a new policy version plus at least one changed causal input: M7 source, M7 threshold backed by calibration, grid, discovery provider, seed-probe mode with proper authorization, or clearance policy backed by the PHerc0358 review. Changing only the planner is refused for `NO_M7_CANDIDATES`, because the planner cannot choose a candidate the provider did not offer.

- [x] **Step 5: Make the decision visible**

  Show numerator, denominator, excluded platform failures, triggering attempts, receipt hash, and allowed next actions in the API and Coverage page.

#### Task 5 implementation ledger

- **Implemented:** 2026-08-03 on `codex/fix-qc-review-stack-unpack`. Candidate-starvation decisions are derived from immutable mission-scoped terminal evidence, persisted under the same mission transaction/lock used by queue creation, and enforced before any new controlled P1 insertion.
- **Eight-attempt rule:** each complete block of eight scientific-terminal attempts pauses at seven or more exact raw-M7-zero outcomes. Superseded attempts, malformed identities, unknown outcomes, fixtures, cancellations, configuration blocks, lease exhaustion, publication/source/worker failures, and evidence-incomplete `NO_SEED` diagnoses are excluded or fail closed instead of contaminating the denominator.
- **Cross-scroll rule:** two distinct consecutive registered scroll budgets with zero exact raw M7 candidates produce an immutable pause. Admission is mission-wide: a second scroll may join only when every pre-existing mission task belongs to a registered budget authority, so mixed or unbudgeted history cannot manufacture a cross-scroll decision.
- **Atomic enforcement:** SQLite uses `BEGIN IMMEDIATE`; PostgreSQL uses the shared mission advisory lock. Terminal writes refresh immutable decisions before commit, while queue creation refreshes and gates under the same lock. Existing pending, leased, or running work is never cancelled or reprioritized.
- **Resume authority:** a pause is mission-wide and monotonic. One content-addressed `campaignx.first_letters_campaign_resume_authorization.v1` may establish one successor policy in the persisted mission chain, for the same or another scroll; forks and unversioned policy switches fail closed. Source, grid, and provider evidence is recomputed from the exact registered admission plus private/sanitized preflight and budget hashes. Threshold, seed-probe, and clearance changes remain blocked until their calibration, paired-benchmark, and Task 7 evidence validators exist. Planner-only, caller-digest-only, and unproven changes are rejected.
- **Terminal integrity:** non-`NO_SEED` scientific terminals reject every platform `failure_class`; excluded terminals require their exact canonical state-to-class mapping. Missing, unknown, or mismatched classes produce `CONTROL_INCOMPLETE`. Exact hash-bound `NO_SEED` diagnosis remains authoritative.
- **Visibility:** `/api/coverage` returns immutable history plus an explicit server-derived `active_campaign_decision`. After authorized resume, the successor policy starts at `CONTINUE 0/8`; the old pause remains immutable history. The Coverage page renders only the server-selected active field and never infers activity from list ordering.
- **Independent review closure:** C5 closed the cross-sample/new-policy pause bypass. I2 replaced caller-supplied evidence digests with store-revalidatable evidence or fail-closed validators. I3 added the mission-wide successor chain and explicit active state. I4 made terminal state/failure-class mapping strict and added server-produced canonical failure classes.
- **Verification:** the broad affected Python suite passed `267 tests`; the complete Python suite passed `2275 tests` with `129 skipped` and zero failures. Frontend verification passed `35/35` tests and a production Vite build. Python `compileall` and `git diff --check` passed.
- **Review status:** implementation and review fixes are locally complete. No push or deploy is included in Task 5; production use still requires the normal integration review, branch pipeline, exact-revision deploy, and smoke evidence.

### Task 6: Separate discovery mode from acceptance mode

**Files:**
- Modify: `framework/stages/01-segmentation/SEED_PROBE_RUNBOOK.md`
- Modify: `framework/stages/01-segmentation/fleet/seed_probe.py`
- Modify: `framework/stages/01-segmentation/fleet/generator.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`
- Test: `tests/test_seed_probe.py`
- Test: `tests/test_seed_probe_benchmark_execution.py`
- Test: `tests/test_first_letters_discovery_isolation.py`

**Interfaces:**
- Consumes: a control PASS, preflight receipt, budget receipt, mission, discovery policy, and optional approved seed-probe benchmark receipt.
- Produces: bounded noncanonical probe evidence and, only when authorized, an exact normal full-grow continuation contract.

- [ ] **Step 1: Reuse the existing three-mode rollout**

  Keep `off` as baseline. Run `shadow` first with top-k 2 and 12 probe generations, retaining evidence while leaving canonical selection unchanged. Permit `select` only in an isolated paired benchmark until the existing benchmark decision says `APPROVED_SELECT` for the exact scroll cohort and source locks.

- [ ] **Step 2: Version alternative prediction sources as experimental arms**

  Each M7 model, resolution, threshold, or alternative prediction source gets a separate source snapshot and policy version. Compare arms on the same preregistered cells. Do not merge scores from different scales in v1; report candidate union, intersection, unique yield, and post-gate survival separately before considering an ensemble.

- [ ] **Step 3: Add bounded adaptive near-miss sampling**

  Allow one versioned retry around a cell only when its causal receipt recorded raw candidates that failed CT or clearance, or a probe produced measurable but noncanonical geometry. Generate a deterministic neighboring-cell set, cap it at eight cells per source attempt, and bind parent attempt, reason, radius, grid, source locks, and total additional compute.

- [ ] **Step 4: Preserve promotion isolation**

  Probe artifacts remain under `probes/...`, never enter canonical surface tables, and never trigger physical QC or P3 directly. Promotion creates a new normal full-grow attempt; only its canonical artifact may proceed through standard gates.

- [ ] **Step 5: Prove isolation in tests**

  Assert that shadow cannot steer, select cannot run without an approved receipt and enabled capability, an experimental source cannot overwrite the mission's accepted P0 source, adaptive retries cannot exceed one generation/eight cells, and no discovery artifact can satisfy downstream admission.

### Task 7: Reproduce and resolve PHerc0358's 8/8/0 clearance result

**Files:**
- Create: `scripts/harness/analyze_candidate_clearance.py`
- Create after execution: `docs/first-letters/pherc0358-clearance-review/README.md`
- Create after execution: `docs/first-letters/pherc0358-clearance-review/evidence.json`
- Test: `tests/test_candidate_clearance_analysis.py`

**Interfaces:**
- Consumes: the exact PHerc0358 attempt receipt, P0 source, raw candidate response, CT screen, task bounds, volume shape, coordinate frame, and voxel size.
- Produces: `campaignx.candidate_clearance_review.v1` without changing a margin or queue.

- [ ] **Step 1: Recompute every distance independently**

  For all eight candidates, recompute cell-face and volume-face distances in CT-L0 voxels and micrometres. Verify axis order, exclusive/inclusive high bounds, P0 volume shape, M7/CT alignment, and the exact policy thresholds. Compare recomputed causes with the terminal receipt.

- [ ] **Step 2: Classify the cause**

  The result must be exactly one of: `IMPLEMENTATION_OR_METADATA_DEFECT`, `TRUE_VOLUME_BOUNDARY`, `CELL_BOUNDARY_ONLY`, or `POLICY_MARGIN_UNCALIBRATED`. A defect gets a focused TDD fix before any experiment. A true volume boundary keeps the margin unchanged.

- [ ] **Step 3: Run a read-only sensitivity table**

  Re-evaluate retained counts at 100%, 75%, 50%, 25%, and 0% of the current volume margin, reporting physical distance and CT/M7 support at each level. This is diagnostic evidence only and cannot authorize a new production threshold.

- [ ] **Step 4: Define the only allowed follow-up**

  If the review finds a verified implementation/metadata defect, repair it and rerun the original policy. If it finds an uncalibrated but physically supported margin, create a new diagnostic-only policy and at most one eight-cell adaptive wave. Never edit the global margin in place.

### Task 8: Route tiny surfaces to a diagnostic path

**Files:**
- Create: `framework/profiles/01-segmentation/small-surface-routing-1.0.0.json`
- Create: `framework/stages/01-segmentation/fleet/surface_routing.py`
- Modify: `framework/stages/01-segmentation/fleet/finalizer.py`
- Modify: `framework/stages/01-segmentation/fleet/store.py`
- Modify: `framework/stages/01-segmentation/fleet/postgres_store.py`
- Modify: `panel/app.py`
- Test: `tests/test_small_surface_routing.py`
- Test: `tests/integration/test_small_surface_routing.py`

**Interfaces:**
- Consumes: a finalized canonical geometry artifact and measured area.
- Produces: `STANDARD_QC_PENDING`, `SMALL_SURFACE_DIAGNOSTIC`, or an existing geometry rejection without changing the scientific meaning of physical QC.

- [ ] **Step 1: Freeze the initial area floor**

  Use the existing effort-allocation floor of 0.10 cm2 as the v1 threshold. A surface below it is not declared bad or empty; it is declared too small for the standard acceptance/QC path.

- [ ] **Step 2: Route before expensive physical QC**

  Geometry-certified surfaces at or above 0.10 cm2 enqueue normal physical QC exactly as today. Smaller surfaces receive `SMALL_SURFACE_DIAGNOSTIC`, preserve their artifact and geometry certificate, and do not enter the normal physical-QC FIFO or P3.

- [ ] **Step 3: Define the diagnostic packet**

  Produce geometry dimensions, valid-coordinate fraction, triangle count, area, bounding box, local CT support, candidate/grow lineage, and a lightweight preview. Do not run P5/P7 and do not write a physical-QC verdict from the diagnostic.

- [ ] **Step 4: Define promotion by expansion only**

  A tiny surface may leave the diagnostic path only through a new versioned resume/grow attempt. The new canonical surface must independently reach at least 0.10 cm2 and pass all standard geometry and physical-QC gates. The original tiny surface remains diagnostic evidence.

- [ ] **Step 5: Regress PHerc0268**

  Use a fixture matching area 0.01983222455087575 cm2, 14x14 grid, and 132 triangles. Assert that it is preserved, marked diagnostic, omitted from normal physical-QC and downstream queues, and never described as evidence of no ink.

### Task 9: Integrate gates into the control plane and campaign workflow

**Files:**
- Modify: `panel/app.py`
- Modify: `panel/web/src/routes/Coverage.tsx`
- Modify: `scripts/harness/run_first_letters_positive_control.py`
- Create: `scripts/harness/run_first_letters_campaign.py`
- Test: `tests/test_first_letters_campaign_api.py`
- Test: `tests/e2e/test_first_letters_campaign_gates.py`

**Interfaces:**
- Consumes: control, preflight, budget, and campaign-decision receipts.
- Produces: guarded P1 waves and a single machine-readable campaign ledger.

- [x] **Step 1: Add a read-only readiness endpoint**

  `GET /api/missions/{mission_id}/first-letters-readiness` returns the deployed revision, control freshness, per-scroll preflight status, budgets, active pause, queue counts, small-surface diagnostics, and explicit blockers.

- [x] **Step 2: Guard P1 creation**

  A First Letters mission may queue P1 only when the exact deployed revision has a passing control, the selected scroll has a current preflight and budget, and the mission is not paused. Existing non-First-Letters workflows remain unchanged.

- [x] **Step 3: Make waves transactional at the decision level**

  Before each wave, capture readiness and attempt baseline; enqueue no more than the budgeted count; wait for all wave tasks to become terminal; recalculate the pause decision; and only then consider another wave. An ambiguous POST freezes the orchestrator until readback resolves it.

- [x] **Step 4: Surface explicit operator choices**

  The UI offers only evidence-backed actions: run/refresh control, run preflight, accept the computed budget, inspect pause causes, create a new versioned strategy, inspect small surfaces, or close the campaign. It must not offer a one-click acceptance-gate bypass.

- [x] **Step 5: Test the full state machine**

  Cover control stale/pass/fail, preflight missing/estimated/census, budget absent/clipped, six-of-eight NO_M7 continuing, seven-of-eight pausing, platform failures excluded, new strategy resuming, tiny surfaces isolated, a canonical candidate advancing, and a human-review packet stopping further expansion.

### Task 10: Validate offline, in staging, and with a controlled campaign

**Files:**
- Modify: `containers/run-smoke.sh`
- Modify: `tests/e2e/test_the_gates_hold_on_the_deployment.py`
- Create after execution: `docs/first-letters/discovery-recovery-validation/README.md`
- Create after execution: `docs/first-letters/discovery-recovery-validation/evidence.json`

**Interfaces:**
- Consumes: all previous implementation tasks.
- Produces: reviewed code, green CI, deployed revision, smoke evidence, control evidence, and a go/no-go decision for the next search mission.

- [ ] **Step 1: Run focused suites after every task**

  Run the named tests for each component, `git diff --check`, JSON/profile validation, and type/frontend checks where relevant. Commit each independently reviewable component separately.

- [ ] **Step 2: Run the complete offline verification**

  Run the full Python suite, frontend test suite, frontend production build, documentation truth tests, secret scan, and an independent code review. Resolve every Critical and Important finding before deploy.

- [ ] **Step 3: Deploy and smoke exact revision**

  Wait for the branch/staging pipeline, deploy using the existing script, require every Helena service to report the exact revision, and run the deployment gates. A documentation-only later commit must not be confused with the runtime revision.

- [ ] **Step 4: Execute the positive control before target work**

  Run the control mission through the API and audit every stage receipt and hash. A failure stops the program and opens a focused incident; it never triggers a target campaign.

- [ ] **Step 5: Preflight all thirteen scrolls**

  Generate one source-locked receipt per scroll. Do not queue scrolls whose current source has zero usable candidate-bearing cells. Rank eligible scrolls by achieved detection probability per measured compute cost, not by historical preference.

- [ ] **Step 6: Run a bounded v2 campaign**

  Create a new mission and policy versions; do not reuse `first-letters-hybrid-20260802`. Queue the smallest justified first wave, apply the eight-attempt starvation decision, and expand only while the decision is `CONTINUE`. Stop on the first credible human-review packet.

- [ ] **Step 7: Publish the closeout**

  Record controls, preflights, task budgets, attempted sampling percentage, every terminal cause, small-surface diagnostics, discovery/acceptance lineage, incidents, hashes, human review, and explicit non-claims. Verify final readback has zero unexplained active work before declaring operational completion.

## Recommended delivery sequence

1. Tasks 1-2: establish whether the exact pipeline can recover a known target.
2. Tasks 3-5: measure candidate supply and prevent another blind fixed-budget campaign.
3. Tasks 7-8: resolve the two concrete near-miss classes from the failed campaign.
4. Task 6: enable bounded discovery experiments without weakening acceptance.
5. Task 9: integrate gates and orchestration.
6. Task 10: validate, deploy, run controls, then decide whether a new campaign is justified.

The critical path is Tasks 1 -> 2 -> 3 -> 4 -> 5 -> 9 -> 10. Tasks 7 and 8 can be implemented after Task 3 and before Task 9. Task 6 may begin after Task 2, but production `select` remains blocked by its existing paired-benchmark approval contract.

## Explicit go/no-go decisions

- **No-go for implementation:** design or policy review is not approved.
- **No-go for target execution:** positive control is missing, stale, incomplete, or failed.
- **No-go for a scroll:** current-source census finds zero usable candidate-bearing cells.
- **Pause:** at least seven of the first eight scientific-terminal attempts are raw-M7 empty, or two consecutive scroll budgets are raw-M7 empty.
- **No-go for margin change:** PHerc0358 analysis does not prove a defect or a calibrated alternative.
- **No-go for seed-probe select:** existing paired benchmark receipt is absent, stale, failed, or outside the authorized cohort.
- **No-go for downstream:** surface is below 0.10 cm2 or fails any existing geometry/physical-QC prerequisite.
- **Go for human review:** a canonical candidate passes the existing automated chain and produces a hash-bound review packet.
- **Scientific success:** a blinded human confirms the preregistered target criterion. Automated outputs alone never satisfy this state.

## Plan self-review

- Every recommendation from the 2026-08-02 postmortem maps to at least one task.
- The plan reuses existing manual seeds, M7 survey, seed-probe modes, coverage reporting, and the 0.10 cm2 effort floor rather than creating parallel mechanisms.
- Discovery and acceptance remain separate in data model, namespace, admission, and language.
- No global threshold is weakened by the PHerc0358 diagnostic.
- No placeholder, silent override, direct database mutation, or unsupported absence claim is required.
