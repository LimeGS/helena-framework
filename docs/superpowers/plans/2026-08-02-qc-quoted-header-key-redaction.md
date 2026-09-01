# Quoted QC Header-Key Redaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final QC diagnostic leak by redacting quoted serialized `Authorization`, `Proxy-Authorization`, `Cookie`, and `Set-Cookie` keys at helper, worker, store, event, and API boundaries.

**Architecture:** Extend only the shared header-key recognizers in `framework/contracts/qc_diagnostics.py` so quoted and unquoted serialized header keys use the same conservative complete-value redaction. Reuse existing worker, SQLite, PostgreSQL, and API paths; add regressions at each already-established boundary without changing queue, scientific, routing, schema, or endpoint behavior.

**Tech Stack:** Python 3.11+, `re`, pytest, SQLite, PostgreSQL store test doubles, FastAPI.

## Global Constraints

- TDD is mandatory: the exact quoted-header cases must fail before production code changes.
- Persist and serialize only normalized, redacted messages of at most 500 Unicode code points.
- Never allow credentials, cookies, authorization material, DSNs, worker identities, artifact URIs, generic URIs, or private paths into results, receipts, store events, or API diagnostics.
- The shared `framework/contracts/qc_diagnostics.py` contract remains the sole redaction implementation; do not duplicate policy in worker, stores, panel, or tests.
- Preserve the approved attempt-local `scientific-executor.log` unchanged.
- Preserve exit `78` as terminal `BLOCKED_CONFIGURATION` and every other nonzero exit as `RETRYABLE_QC_UNAVAILABLE`.
- Do not change queue, retry timing, scheduling, validation, scientific outcome/state mapping, store schema, endpoint schema, whitelist, authentication, or mission scope.
- Do not push, deploy, call campaign APIs, or mutate queue state during implementation.
- Preserve and do not stage the unrelated untracked campaign plan.

## File Structure

- Modify `framework/contracts/qc_diagnostics.py`: accept optional matching quotes around the four serialized header names in header substitutions and diagnostic boundaries.
- Modify `tests/test_qc_job_diagnostics_api.py`: pin direct shared-helper and public panel-field behavior for quoted Authorization/Cookie mappings.
- Modify `tests/test_a_misconfigured_worker_stops_instead_of_spinning.py`: pin exit-78 worker receipt/store and direct SQLite/PostgreSQL durable boundaries for quoted header mappings.

---

### Task 1: Redact quoted serialized header keys everywhere

**Files:**
- Modify: `framework/contracts/qc_diagnostics.py`
- Modify: `tests/test_qc_job_diagnostics_api.py`
- Modify: `tests/test_a_misconfigured_worker_stops_instead_of_spinning.py`

**Interfaces:**
- Consumes: `sanitize_error(raw: object) -> tuple[str | None, str | None]`, `safe_message(raw: object, fallback: str) -> str`, and `receipt_with_safe_error(receipt: dict[str, Any], fallback: str) -> dict[str, Any]`.
- Produces: the same signatures and placeholders, with quoted/unquoted header-key parity.

- [ ] **Step 1: Add direct shared-helper and panel RED cases**

  Add exact cases for:

  ```python
  "RuntimeError: {'Authorization': 'Digest username=alice, realm=private-realm, response=digest-secret'}"
  "RuntimeError: {'Cookie': 'a=first-cookie, b=second-cookie'}"
  "RuntimeError: {\"Proxy-Authorization\": \"Token proxy-secret extra\"}"
  "RuntimeError: {\"Set-Cookie\": \"sid=first; admin=second\"}"
  ```

  Require `sanitize_error` and `panel.app._qc_diagnostic_fields` to retain the `RuntimeError` type, contain `<redacted>`, and contain none of `alice`, `private-realm`, `digest-secret`, `first-cookie`, `second-cookie`, `proxy-secret`, `sid=first`, or `admin=second`.

- [ ] **Step 2: Add worker and durable-store RED cases**

  Extend the real exit-78 subprocess regression so adapter JSON supplies an error mapping containing quoted `Authorization` and `Cookie` keys; assert the disk blocked receipt and store payload contain neither header value. Extend direct SQLite and PostgreSQL block/retry cases with the two exact serialized strings above; assert persisted result JSON/JSONB, emitted event payload, and blocked return contain only the safe redacted string and never mutate caller receipts.

- [ ] **Step 3: Run RED tests and verify the expected leak**

  Run the exact new helper/panel, worker, SQLite, and PostgreSQL cases. Expected: FAIL because the current header patterns accept only unquoted header keys, leaving the quoted mapping values intact. Reject import/setup/fixture failures as invalid RED evidence.

- [ ] **Step 4: Implement the minimal shared pattern correction**

  Introduce one reusable header-key fragment matching an optional same-class quote around:

  ```text
  authorization | proxy-authorization | cookie | set-cookie
  ```

  Use it in both complete header substitutions and in `_FIELD_BOUNDARY`, `_CREDENTIAL_BOUNDARY`, and `_PATH_BOUNDARY`. Keep the existing conservative value consumption, placeholder ordering, redaction-before-truncation, and public helper signatures unchanged. Do not special-case any caller.

- [ ] **Step 5: Run GREEN and regression gates**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py \
    tests/test_qc_job_diagnostics_api.py \
    tests/test_campaignx_run_vc3d_locked_plan.py
  python -m pytest -q
  git diff --check
  git status --short --branch
  ```

  Expected: all focused tests pass; full suite exits `0` when run without sandbox DNS/socket restrictions. The only untracked file remains `docs/superpowers/plans/2026-08-02-first-letters-hybrid-campaign.md`.

- [ ] **Step 6: Commit**

  ```bash
  git add \
    framework/contracts/qc_diagnostics.py \
    tests/test_qc_job_diagnostics_api.py \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py
  git commit -m "fix(qc): redact quoted diagnostic header keys"
  ```
