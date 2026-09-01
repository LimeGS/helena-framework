# Terminal Surface-QC Liveness Refusal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the TimeSformer `DEGENERATE`/`EMPTY` liveness refusal from an infinite surface-QC retry into a hash-bound terminal insufficiency outcome that remains non-admissible downstream.

**Architecture:** The ink runner keeps its existing liveness contract. The surface-QC adapter requests evidence-preserving return, immediately validates the receipt, and short-circuits before stability/high-recall work when the verdict is not `ALIVE`. The ordinary evidence publication and QC finalization boundaries carry a new explicit outcome through both stores and the ledger.

**Tech Stack:** Python 3.11, pytest, SQLite/PostgreSQL fleet stores, Helena surface-QC adapter, JSON receipts and SHA-256 evidence manifests.

## Global Constraints

- Never continue stability analysis, high-recall routing or the CT/fiber gate after `DEGENERATE` or `EMPTY`.
- Never map the new outcome to CT support, no-ink, ink, text or letters.
- `INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY` maps only to `QC_INK_SCREEN_INSUFFICIENT` / `INK_SCREEN_INSUFFICIENT`.
- Keep `ADMISSIBLE_PHYSICAL_QC_STATES` unchanged.
- Missing/unknown liveness remains an operational failure.
- Exit `78`, retry timing, queue ordering, mission/auth/API contracts and existing outcomes remain unchanged.
- Preserve the unrelated untracked campaign plan.
- Do not mutate the production queue; production acceptance uses the existing job's natural retry.

---

### Task 1: Add the terminal outcome to the fleet contract

**Files:**
- Modify: `framework/stages/01-segmentation/fleet/store.py:41-45`
- Test: `tests/test_segment_search_fleet.py`

**Interfaces:**
- Consumes: canonical outcome string from the design.
- Produces: `QC_OUTCOME_STATES["INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY"] == ("QC_INK_SCREEN_INSUFFICIENT", "INK_SCREEN_INSUFFICIENT")` for SQLite, PostgreSQL and the worker's shared validator.

- [ ] **Step 1: Write the failing SQLite worker test**

  Add a test beside `test_surface_qc_worker_completes_fixture_vertical_slice_without_leaking_lease` that runs `FixtureQcExecutor("INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY")` and asserts:

  ```python
  assert receipt["status"] == "COMPLETED"
  assert receipt["outcome"] == "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY"
  assert receipt["surface_state"] == "QC_INK_SCREEN_INSUFFICIENT"
  assert receipt["physical_qc_state"] == "INK_SCREEN_INSUFFICIENT"
  assert store.status()["qc_job_states"] == {"COMPLETED": 1}
  assert store.claim_qc("another-worker", 60) is None
  ```

- [ ] **Step 2: Run the RED test**

  Run:

  ```bash
  python3 -m pytest -q tests/test_segment_search_fleet.py::test_surface_qc_worker_completes_terminal_liveness_insufficiency
  ```

  Require failure because `FixtureQcExecutor` rejects the unknown outcome.

- [ ] **Step 3: Add the minimal shared mapping**

  Extend only `QC_OUTCOME_STATES`:

  ```python
  "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY": (
      "QC_INK_SCREEN_INSUFFICIENT",
      "INK_SCREEN_INSUFFICIENT",
  ),
  ```

  Do not add the physical state to `ADMISSIBLE_PHYSICAL_QC_STATES`.

- [ ] **Step 4: Run GREEN and mapping regressions**

  Run:

  ```bash
  python3 -m pytest -q \
    tests/test_segment_search_fleet.py::test_surface_qc_worker_completes_terminal_liveness_insufficiency \
    tests/test_segment_search_fleet.py::test_surface_qc_worker_completes_fixture_vertical_slice_without_leaking_lease \
    tests/test_segment_search_fleet.py::test_surface_qc_worker_requeues_operational_failure_without_scientific_result
  ```

- [ ] **Step 5: Commit Task 1**

  ```bash
  git add framework/stages/01-segmentation/fleet/store.py tests/test_segment_search_fleet.py
  git commit -m "feat(qc): model terminal liveness insufficiency"
  ```

---

### Task 2: Preserve and publish the non-live screening evidence

**Files:**
- Modify: `framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py:50-60,494-706,1215-1285`
- Test: `tests/test_surface_qc_adapter.py`

**Interfaces:**
- Consumes: `INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY` from Task 1 and the runner's existing `INK_SCREENING_RECEIPT.json` liveness object.
- Produces: a terminal screen summary and ordinary published QC result without calling analysis/high-recall for non-`ALIVE` maps.

- [ ] **Step 1: Write the RED adapter test**

  Add `test_non_alive_screen_is_terminal_and_never_reaches_analysis` using temporary renderer/checkpoint files and monkeypatches for metadata fetch, renderer and `run_logged`.

  The fake inference call must write:

  ```json
  {
    "liveness": {
      "verdict": "DEGENERATE",
      "reason": "std 0.0001 < 0.02",
      "metrics": {"std": 0.0001, "valid_pixels": 84}
    }
  }
  ```

  to the requested `--output` directory as `INK_SCREENING_RECEIPT.json`. The fake must raise if `analyze_ink_stability.py` is invoked. Assert the current code fails because it does not pass `--on-degenerate warn` and/or does not short-circuit.

- [ ] **Step 2: Run the RED test**

  Run:

  ```bash
  python3 -m pytest -q tests/test_surface_qc_adapter.py::test_non_alive_screen_is_terminal_and_never_reaches_analysis
  ```

  Require the expected assertion failure, not a fixture/setup error.

- [ ] **Step 3: Implement the fail-closed adapter branch**

  Add:

  ```python
  OUTCOME_INK_SCREEN_INSUFFICIENT = (
      "INK_SCREEN_INSUFFICIENT_DEGENERATE_OR_EMPTY"
  )
  ```

  Append `--on-degenerate`, `warn` to the inference command. Immediately after loading the screening receipt:

  ```python
  liveness = screening_payload.get("liveness")
  if not isinstance(liveness, dict):
      raise RuntimeError("six-replica inference receipt omitted liveness")
  verdict = str(liveness.get("verdict", ""))
  if verdict not in {"ALIVE", "DEGENERATE", "EMPTY"}:
      raise RuntimeError("six-replica inference receipt has an unknown liveness verdict")
  if verdict != "ALIVE":
      row = {
          "sample_id": sample_id,
          "seed_id": surface_id,
          "state": OUTCOME_INK_SCREEN_INSUFFICIENT,
          "screening_outcome": OUTCOME_INK_SCREEN_INSUFFICIENT,
          "manual_review_route": "NO_USABLE_INK_MAP",
          "tiff_count": REQUIRED_TIFF_COUNT,
          "slice_ordering": slice_ordering,
          "screening_receipt": str(screening_receipt),
          "screening_receipt_sha256": sha256(screening_receipt),
          "liveness": liveness,
      }
      summary = _write_screen_summary(
          output,
          row,
          renderer,
          checkpoint,
          inference,
          analysis,
          actual_checkpoint_sha256,
          input_sha256,
          qc_profile,
          qc_profile_sha256,
      )
      return summary, output / "FLEET_SURFACE_SCREEN_EXECUTION.json"
  ```

  Update `execute` to select the new outcome before the existing no-common-valid and supported branches. Leave `ink_used = True` because the model ran and produced the recorded measurement.

- [ ] **Step 4: Prove the branch and old paths**

  Run:

  ```bash
  python3 -m pytest -q \
    tests/test_surface_qc_adapter.py \
    tests/test_lane_liveness.py \
    tests/test_every_ink_lane_reports_liveness.py \
    tests/test_phase4_geometry_high_recall_manifest_paths.py
  ```

  Require the new test to prove analysis was not called and the liveness object/hash were persisted.

- [ ] **Step 5: Commit Task 2**

  ```bash
  git add framework/stages/04-validation/scripts/campaignx_surface_qc_adapter.py tests/test_surface_qc_adapter.py
  git commit -m "fix(qc): terminate unusable ink screens"
  ```

---

### Task 3: Make ledger semantics outcome-specific

**Files:**
- Modify: `framework/stages/04-validation/scripts/helena_build_surface_qc_ledger.py:99-145`
- Modify: `framework/stages/04-validation/scripts/helena_verify_surface_qc_ledger.py:100-160`
- Test: `tests/test_surface_qc_ledger.py`
- Test: `tests/test_surface_qc_ledger_verifier.py`

**Interfaces:**
- Consumes: completed new outcome and evidence manifest from Tasks 1-2.
- Produces: honest next-step routing and fail-closed stage verification for terminal liveness insufficiency.

- [ ] **Step 1: Write RED ledger tests**

  In `test_next_step_reports_each_pipeline_boundary`, assert a completed new outcome returns:

  ```text
  SELECT_DIFFERENT_SURFACE_SCREEN_INSUFFICIENT
  ```

  In the verifier tests, create a completed new-outcome row with rendering, inference and evidence `True`, stability and CT gate `False`. Assert verification succeeds. Then parameterize violations where stability or CT gate is `True` and assert failure reasons `STABILITY_COMPLETE_UNEXPECTED` and `CT_GATE_COMPLETE_UNEXPECTED`.

- [ ] **Step 2: Run the RED tests**

  Run:

  ```bash
  python3 -m pytest -q tests/test_surface_qc_ledger.py tests/test_surface_qc_ledger_verifier.py
  ```

  Require failures for unknown outcome/routing and old unconditional stage rules.

- [ ] **Step 3: Implement outcome-specific ledger rules**

  Add the new next-step branch before the generic completed branch. Add the outcome to the verifier allowlist. For the new outcome require exactly:

  ```python
  render_complete is True
  rendered_slice_count == 65
  inference_complete is True
  stability_complete is False
  ct_gate_complete is False
  evidence_manifest_complete is True
  evidence_manifest_matches_database is True
  ```

  Keep the existing required stage set unchanged for the original three outcomes.

- [ ] **Step 4: Run GREEN plus discovery regressions**

  Run:

  ```bash
  python3 -m pytest -q \
    tests/test_surface_qc_ledger.py \
    tests/test_surface_qc_ledger_verifier.py \
    tests/test_first_letters_review_queue.py
  ```

- [ ] **Step 5: Commit Task 3**

  ```bash
  git add \
    framework/stages/04-validation/scripts/helena_build_surface_qc_ledger.py \
    framework/stages/04-validation/scripts/helena_verify_surface_qc_ledger.py \
    tests/test_surface_qc_ledger.py \
    tests/test_surface_qc_ledger_verifier.py
  git commit -m "fix(qc): verify terminal liveness evidence"
  ```

---

### Task 4: Review, deploy and observe the natural retry

**Files:**
- Update: `/private/tmp/first-letters-hybrid-20260802/ledger.json`

**Interfaces:**
- Consumes: reviewed exact implementation SHA and existing pending QC job `849116b4-dfb2-5746-a449-c9767659a15a`.
- Produces: exact-SHA staging convergence, terminal hash-bound QC result, and authorization to resume the next bounded campaign wave.

- [ ] **Step 1: Run focused and full verification**

  ```bash
  python3 -m pytest -q \
    tests/test_surface_qc_adapter.py \
    tests/test_segment_search_fleet.py \
    tests/test_surface_qc_ledger.py \
    tests/test_surface_qc_ledger_verifier.py \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py
  python3 -m pytest -q
  git diff --check
  git status --short --branch
  ```

- [ ] **Step 2: Request independent final review**

  Review the exact diff for scientific semantics, evidence durability, fail-closed behavior, SQLite/PostgreSQL parity, secret/path redaction and regression coverage. Fix Critical/Important findings with RED-first commits and rerun affected tests plus the full suite.

- [ ] **Step 3: Push and deploy exact SHA**

  Push the feature branch, verify `origin/staging` is an ancestor, fast-forward staging without force, and require exact-SHA unit, image, deploy and smoke jobs. Verify all Helena services report the exact revision and both QC workers are active.

- [ ] **Step 4: Observe only the existing job**

  Poll the mission-scoped diagnostics API and surface endpoint. Do not claim, retry, reprioritize, cancel or re-enqueue. Require the existing job to reach `COMPLETED`, the new exact outcome, a verified evidence URI and manifest SHA-256, `physical_qc_state=INK_SCREEN_INSUFFICIENT`, and no subsequent claim-count increase.

- [ ] **Step 5: Update the incident ledger**

  Append the implementation SHA, test evidence, pipeline/deploy/smoke evidence, pre/post claim count, terminal outcome, surface state, evidence URI class, manifest digest and retrieval UTC. Preserve all earlier attempts. Validate with `jq empty` and record the ledger SHA-256.

- [ ] **Step 6: Resume the campaign**

  Treat this surface as a terminal measured insufficiency, not a negative ink result. Enqueue the next bounded P1 wave from the approved campaign plan through the Helena API only, then repeat the same geometry/physical-QC gate without `allow_unvalidated`.
