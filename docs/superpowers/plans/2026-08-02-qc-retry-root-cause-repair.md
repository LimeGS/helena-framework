# QC Retry Cause Propagation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a sanitized, actionable adapter exception in the persisted retry receipt so the authenticated QC diagnostics API can reveal the underlying scientific failure after the existing job retries naturally.

**Architecture:** The verified defect is an observability boundary failure, not a scientific verdict: `SubprocessQcExecutor` retains combined adapter output in an attempt log but replaces every non-configuration failure with only `scientific QC adapter failed with exit code N`, and `SurfaceQcWorker` persists that replacement. Add one shared pure sanitizer/extractor under `framework/contracts`, reuse it from the panel, and carry only the sanitized final Python exception line through a typed executor failure into the existing `RETRYABLE_QC_UNAVAILABLE` receipt. Do not change the adapter, retry timing, queue state transitions, scientific outcome mapping, or campaign state.

**Tech Stack:** Python 3.11+, `subprocess`, `re`, pytest, FastAPI, PostgreSQL-backed Helena staging deployment.

## Global Constraints

- TDD is mandatory: observe the focused propagation test fail for the missing detail before editing production code.
- The verified cause addressed here is loss of adapter exception detail between `SubprocessQcExecutor.execute` and `SurfaceQcWorker.run_one`; the underlying scientific cause remains unknown until a natural retry exposes it through the authorized API.
- Persist and serialize only a normalized, redacted message of at most 500 Unicode code points; never persist raw combined stdout, a traceback, credentials, cookies, authorization material, DSNs, worker identities, artifact URIs, or private paths in the retry result.
- Send every retryable error, including non-adapter exceptions and generic adapter-exit fallbacks, through the shared sanitizer before persistence. If sanitization leaves no safe message, persist only a fixed bounded generic wrapper.
- Preserve exit `78` as the existing terminal `BLOCKED_CONFIGURATION` contract and preserve every other nonzero adapter exit as `RETRYABLE_QC_UNAVAILABLE`.
- Do not add a manual retry, claim, cancellation, priority, ordering, validation, or queue-control operation.
- Do not cancel, duplicate, prioritize, reorder, or manually validate the existing pending QC job.
- Do not use direct SQL, SSH attempt-file inspection, raw QC receipts, or container logs for campaign diagnosis; staging validation is through the authenticated read-only HTTP API.
- Do not resume the campaign or enqueue a new wave while physical QC is `UNVALIDATED`.
- Treat `CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS`, `CT_SUPPORTED_NO_RETAINED_INK_SIGNAL`, and `CT_SUPPORTED_RETAINED_FOR_REVIEW` only as physical-QC outcomes; none is automatic evidence of ink or letters.

## File Structure

- Create `framework/contracts/qc_diagnostics.py`: one shared pure implementation for error normalization, redaction, bounding, and final Python-exception extraction.
- Modify `panel/app.py`: delegate the existing response-field sanitizer to the shared contract without changing the endpoint schema or whitelist.
- Modify `framework/stages/01-segmentation/fleet/qc_worker.py`: represent a non-configuration adapter exit as a typed failure carrying only sanitized diagnostic fields, then persist that safe detail through the existing retry path.
- Modify `tests/test_qc_job_diagnostics_api.py`: keep the endpoint security contract pinned while the sanitizer moves behind the shared helper.
- Modify `tests/test_a_misconfigured_worker_stops_instead_of_spinning.py`: add the exact subprocess-to-persisted-retry regression and retain exit-78/ordinary-retry behavior coverage.
- Update after authorized staging evidence: `/private/tmp/first-letters-hybrid-20260802/qc-diagnostic-response.json` and `/private/tmp/first-letters-hybrid-20260802/ledger.json`.

## TDD Sequencing Note

Tasks 1 and 2 intentionally leave
`test_python_adapter_failure_persists_only_safe_exception_detail`,
`test_native_adapter_failure_persists_only_generic_exit_wrapper`, and
`test_non_adapter_retry_error_is_sanitized_and_bounded_before_persistence`
red. Task 1 proves the end-to-end contract is missing; Task 2 adds and verifies
the independently reviewable shared sanitizer but does not yet change worker
propagation. The end-to-end regression is required to turn green only in Task
3, when the minimal worker change is introduced.

---

### Task 1: Pin the safe adapter-detail propagation contract

**Files:**
- Modify: `tests/test_a_misconfigured_worker_stops_instead_of_spinning.py`

**Interfaces:**
- Consumes: `SubprocessQcExecutor.execute(claim: dict[str, Any], attempt_dir: Path) -> dict[str, Any]` and `SurfaceQcWorker.run_one() -> dict[str, Any] | None`.
- Produces: three regressions covering sanitized Python detail, a generic native-output fallback, and sanitized non-adapter persistence.

- [ ] **Step 1: Extend the recording store to retain the retry receipt passed by the worker**

  Change `Recording.requeue_qc_unavailable` so the test double records its third positional argument without changing the result returned to existing tests:

  ```python
  class Recording:
      def __init__(self, claim):
          self._claim = claim
          self.calls: list[str] = []
          self.requeued_receipt: dict | None = None

      def requeue_qc_unavailable(self, *args, **kwargs):
          self.calls.append("requeue")
          self.requeued_receipt = args[2]
          return {"status": "RETRYABLE_QC_UNAVAILABLE"}
  ```

- [ ] **Step 2: Write the failing end-to-end worker test**

  Add a stub adapter that emits a normal Python traceback containing an actionable exception, a private path, and a token, then exits `1`. Drive it through the real `SubprocessQcExecutor` and `SurfaceQcWorker`:

  ```python
  def test_python_adapter_failure_persists_only_safe_exception_detail(
      tmp_path,
  ):
      from fleet.qc_worker import SubprocessQcExecutor

      stub = tmp_path / "adapter.py"
      stub.write_text(
          "raise FileNotFoundError("
          "'/srv/private/checkpoint.bin token=do-not-persist')\n"
      )
      store = Recording(_claim())
      worker = SurfaceQcWorker(
          store,
          "worker-1",
          SubprocessQcExecutor(stub),
          tmp_path / "runs",
      )

      result = worker.run_one()

      assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
      assert store.calls[-1] == "requeue"
      assert store.requeued_receipt is not None
      error = store.requeued_receipt["error"]
      assert error.startswith("FileNotFoundError: ")
      assert "<path>" in error
      assert "<redacted>" in error
      assert "/srv/private" not in error
      assert "do-not-persist" not in error
      assert len(error) <= 500
      assert store.requeued_receipt["no_scientific_conclusion"] is True
  ```

- [ ] **Step 3: Run the regression and verify the expected failure**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_python_adapter_failure_persists_only_safe_exception_detail
  ```

  Expected: FAIL because the current receipt contains only `RuntimeError: scientific QC adapter failed with exit code 1` and therefore lacks `FileNotFoundError`, `<path>`, and `<redacted>`.

- [ ] **Step 4: Write the failing real-subprocess native-output fallback test**

  Add a non-Python executable stub that writes sensitive native output and exits
  `1`. Drive it through the real `SubprocessQcExecutor` and
  `SurfaceQcWorker`, then prove only the fixed generic wrapper is persisted:

  ```python
  def test_native_adapter_failure_persists_only_generic_exit_wrapper(tmp_path):
      from fleet.qc_worker import SubprocessQcExecutor

      stub = tmp_path / "adapter.sh"
      stub.write_text(
          "#!/bin/sh\n"
          "echo 'native crash /srv/private/model.bin "
          "postgresql://alice:hunter2@private-db/campaignx "
          "AWS_SECRET_ACCESS_KEY=do-not-persist' >&2\n"
          "exit 1\n"
      )
      stub.chmod(0o700)
      store = Recording(_claim())
      worker = SurfaceQcWorker(
          store,
          "worker-1",
          SubprocessQcExecutor(stub),
          tmp_path / "runs",
      )

      result = worker.run_one()

      assert result["status"] == "RETRYABLE_QC_UNAVAILABLE"
      assert store.requeued_receipt is not None
      error = store.requeued_receipt["error"]
      assert error == (
          "QcAdapterExecutionError: scientific QC adapter failed with exit code 1"
      )
      assert len(error) <= 500
      encoded = json.dumps(store.requeued_receipt, sort_keys=True)
      for forbidden in (
          "/srv/private", "hunter2", "private-db", "do-not-persist",
      ):
          assert forbidden not in encoded
  ```

- [ ] **Step 5: Write the failing non-adapter fallback sanitization test**

  Drive a non-adapter exception containing every sensitive value class and
  oversized text through `SurfaceQcWorker`. Require redaction and bounding in
  the persisted retry receipt:

  ```python
  def test_non_adapter_retry_error_is_sanitized_and_bounded_before_persistence(
      tmp_path,
  ):
      raw = (
          "outage /srv/private/model.bin "
          "postgresql://alice:hunter2@private-db/campaignx "
          "token=raw-token AWS_SECRET_ACCESS_KEY=raw-aws-secret "
          + "x" * 700
      )
      store = Recording(_claim())

      _worker(store, RuntimeError(raw), tmp_path).run_one()

      assert store.requeued_receipt is not None
      error = store.requeued_receipt["error"]
      assert len(error) <= 500
      assert "<path>" in error
      assert "<dsn>" in error
      assert "<redacted>" in error
      for forbidden in (
          "/srv/private", "hunter2", "private-db", "raw-token",
          "raw-aws-secret",
      ):
          assert forbidden not in error
  ```

- [ ] **Step 6: Run all propagation regressions and verify the expected failures**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_python_adapter_failure_persists_only_safe_exception_detail \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_native_adapter_failure_persists_only_generic_exit_wrapper \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_non_adapter_retry_error_is_sanitized_and_bounded_before_persistence
  ```

  Expected: all three fail for the current raw/generic persistence behavior,
  not for test setup or subprocess execution errors.

- [ ] **Step 7: Commit only the red tests**

  ```bash
  git add tests/test_a_misconfigured_worker_stops_instead_of_spinning.py
  git commit -m "test(qc): expose lost adapter retry detail"
  ```

  Reviewer gate: the new end-to-end propagation tests remain RED by design; no
  production behavior changes in this task.

---

### Task 2: Share the existing diagnostics sanitizer

**Files:**
- Create: `framework/contracts/qc_diagnostics.py`
- Modify: `panel/app.py`
- Modify: `tests/test_qc_job_diagnostics_api.py`

**Interfaces:**
- Consumes: a raw object that may or may not be a string and combined adapter stdout.
- Produces: `sanitize_error(raw: object) -> tuple[str | None, str | None]` and `extract_last_python_exception(output: str) -> tuple[str | None, str | None]`.

- [ ] **Step 1: Add direct tests for the shared pure contract**

  Import `framework.contracts.qc_diagnostics` in `tests/test_qc_job_diagnostics_api.py` and add:

  ```python
  def test_shared_qc_error_contract_extracts_only_the_safe_final_exception():
      from framework.contracts import qc_diagnostics

      output = (
          "Traceback (most recent call last):\n"
          "  File '/srv/private/adapter.py', line 7, in <module>\n"
          "FileNotFoundError: /srv/private/model.bin token=raw-token\n"
      )
      error_type, error = qc_diagnostics.extract_last_python_exception(output)

      assert error_type == "FileNotFoundError"
      assert error == "FileNotFoundError: <path> <redacted>"
      assert "/srv/private" not in error
      assert "raw-token" not in error
      assert qc_diagnostics.extract_last_python_exception("exit 1\n") == (None, None)
  ```

  Keep the existing malicious/oversized `_qc_diagnostic_fields` tests unchanged; they are the compatibility gate for the public endpoint.

- [ ] **Step 2: Run the shared-contract test and verify it fails because the module is absent**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_qc_job_diagnostics_api.py::test_shared_qc_error_contract_extracts_only_the_safe_final_exception
  ```

  Expected: FAIL with an import error for `framework.contracts.qc_diagnostics`.

- [ ] **Step 3: Implement the pure shared contract**

  Create `framework/contracts/qc_diagnostics.py` with the same ordered redaction rules already approved for the API:

  ```python
  from __future__ import annotations

  import re

  _ERROR_PREFIX = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]{0,127}):")
  _SUBSTITUTIONS = (
      (r"(?i)\b(?:authorization\s*:\s*)?(?:bearer|basic)\s+[^\s,;]+", "<redacted>"),
      (r"(?i)\b(?:cookie|set-cookie)\s*:\s*[^\s,;]+(?:\s*;\s*[^\s,;]+)*", "<redacted>"),
      (r"(?i)\b(?:aws[_-](?:access[_-]key[_-]id|secret[_-]access[_-]key|session[_-]token|security[_-]token)|password|passwd|secret|token|access[_-]?key(?:_id)?)\s*[:=]\s*[^\s,;]+", "<redacted>"),
      (r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "<redacted>"),
      (r"(?i)\bpostgres(?:ql)?://[^\s]+", "<dsn>"),
      (r"(?i)\bhttps?://[^\s]+", "<url>"),
      (r"(?i)\bs3://[^\s]+", "<artifact-uri>"),
      (r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)*[^/\s]+", "<path>"),
  )


  def sanitize_error(raw: object) -> tuple[str | None, str | None]:
      if not isinstance(raw, str):
          return None, None
      message = " ".join(raw.split())
      matched = _ERROR_PREFIX.match(message)
      error_type = matched.group(1) if matched else None
      for pattern, replacement in _SUBSTITUTIONS:
          message = re.sub(pattern, replacement, message)
      if not message:
          return error_type, None
      if len(message) > 500:
          message = message[:499] + "…"
      return error_type, message


  def extract_last_python_exception(output: str) -> tuple[str | None, str | None]:
      for line in reversed(output.splitlines()):
          error_type, error = sanitize_error(line)
          if (
              error_type
              and error
              and error_type.rsplit(".", 1)[-1].endswith(("Error", "Exception"))
          ):
              return error_type, error
      return None, None
  ```

- [ ] **Step 4: Make the panel helper delegate without changing its output**

  Import the shared module beside the other framework contracts:

  ```python
  from framework.contracts import qc_diagnostics  # noqa: E402
  ```

  Replace only the normalization/redaction body of `_qc_diagnostic_fields`:

  ```python
  def _qc_diagnostic_fields(result: dict[str, Any] | None) -> dict[str, str | None]:
      receipt = result if isinstance(result, dict) else {}
      status = receipt.get("status")
      last_status = status if isinstance(status, str) else None
      error_type, error = qc_diagnostics.sanitize_error(receipt.get("error"))
      return {
          "last_status": last_status,
          "error_type": error_type,
          "error": error,
      }
  ```

  Remove `import re` from `panel/app.py` only if no other call site uses it; verify first with `rg -n '\bre\.' panel/app.py`.

- [ ] **Step 5: Run the shared and public-contract tests**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_qc_job_diagnostics_api.py::test_shared_qc_error_contract_extracts_only_the_safe_final_exception \
    tests/test_qc_job_diagnostics_api.py::test_qc_diagnostic_fields_exposes_only_safe_bounded_values \
    tests/test_qc_job_diagnostics_api.py::test_qc_diagnostic_fields_normalizes_missing_nonstring_and_long_errors
  ```

  Expected: `3 passed`.

- [ ] **Step 6: Commit the shared sanitizer refactor**

  ```bash
  git add framework/contracts/qc_diagnostics.py panel/app.py \
    tests/test_qc_job_diagnostics_api.py
  git commit -m "refactor(qc): share diagnostic error sanitizer"
  ```

  Reviewer gate: the shared sanitizer and unchanged panel contract are green,
  while Task 1's end-to-end propagation test remains RED until Task 3.

---

### Task 3: Carry sanitized adapter detail into the retry receipt

**Files:**
- Modify: `framework/stages/01-segmentation/fleet/qc_worker.py`
- Test: `tests/test_a_misconfigured_worker_stops_instead_of_spinning.py`

**Interfaces:**
- Consumes: `qc_diagnostics.extract_last_python_exception(completed.stdout)`.
- Produces: `QcAdapterExecutionError(exit_code: int, error_type: str | None, safe_error: str | None)` and `_retryable_error(error: BaseException) -> str`.

- [ ] **Step 1: Add a typed non-configuration executor failure**

  Import `qc_diagnostics` and define a failure that never exposes raw stdout through `str(error)`:

  ```python
  from framework.contracts import qc_diagnostics


  class QcAdapterExecutionError(RuntimeError):
      def __init__(
          self,
          exit_code: int,
          error_type: str | None,
          safe_error: str | None,
      ) -> None:
          super().__init__(
              f"scientific QC adapter failed with exit code {exit_code}"
          )
          self.exit_code = exit_code
          self.error_type = error_type
          self.safe_error = safe_error
  ```

- [ ] **Step 2: Extract only the sanitized final exception on ordinary adapter failure**

  Preserve the exit-78 branch byte-for-byte. Replace the generic nonzero raise with:

  ```python
  if completed.returncode:
      error_type, safe_error = qc_diagnostics.extract_last_python_exception(
          completed.stdout
      )
      raise QcAdapterExecutionError(
          completed.returncode,
          error_type,
          safe_error,
      )
  ```

  The attempt log remains unchanged and local to the worker. It must never be copied into a job result or API response.

- [ ] **Step 3: Persist the safe detail while keeping the existing retry state**

  Add:

  ```python
  def _retryable_error(error: BaseException) -> str:
      if isinstance(error, QcAdapterExecutionError):
          raw = error.safe_error or (
              "QcAdapterExecutionError: scientific QC adapter failed "
              f"with exit code {error.exit_code}"
          )
      else:
          raw = f"{type(error).__name__}: {error}"
      _error_type, safe_error = qc_diagnostics.sanitize_error(raw)
      return safe_error or "RuntimeError: retryable QC failure had no safe detail"
  ```

  This helper is the only value source for the persisted retry receipt. Even an
  already-sanitized adapter exception is passed through the shared sanitizer
  again at the persistence boundary. The fixed no-detail wrapper contains no
  exception text and is below the 500-code-point bound.

  Then change only the retry receipt's `error` value in `SurfaceQcWorker.run_one`:

  ```python
  "error": _retryable_error(error),
  ```

  Do not change `status`, `no_scientific_conclusion`, retry delay, store method, or exception routing.

- [ ] **Step 4: Run the red propagation tests and require them to turn green**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_python_adapter_failure_persists_only_safe_exception_detail \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_native_adapter_failure_persists_only_generic_exit_wrapper \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py::test_non_adapter_retry_error_is_sanitized_and_bounded_before_persistence
  ```

  Expected: all three Task 1 propagation regressions pass. The Python exception
  branch persists actionable safe detail, the native-output branch persists
  only the bounded generic exit wrapper, and the non-adapter branch redacts and
  bounds its fallback before persistence.

- [ ] **Step 5: Run the retry/configuration behavior slice**

  Run:

  ```bash
  python -m pytest -q \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py \
    tests/test_qc_job_diagnostics_api.py
  ```

  Expected: all tests pass. Specifically, exit `78` still blocks without a scientific verdict, exit `1` remains an ordinary retry, safe adapter detail is persisted, and the public response remains whitelisted and bounded.

- [ ] **Step 6: Commit the minimal propagation change**

  ```bash
  git add framework/stages/01-segmentation/fleet/qc_worker.py \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py
  git commit -m "fix(qc): preserve safe adapter retry detail"
  ```

---

### Task 4: Verify the exact repair tree

**Files:**
- No additional production files.

**Interfaces:**
- Consumes: the three Task 1-3 commits.
- Produces: a reviewable exact SHA with focused and full-suite evidence.

- [ ] **Step 1: Run the focused tests from a clean process**

  ```bash
  python -m pytest -q \
    tests/test_a_misconfigured_worker_stops_instead_of_spinning.py \
    tests/test_qc_job_diagnostics_api.py
  ```

  Expected: all focused tests pass.

- [ ] **Step 2: Run the repository full suite**

  ```bash
  python -m pytest -q
  ```

  Expected: exit `0`; record exact passed/skipped/warning counts and elapsed time. Do not dismiss a new failure as pre-existing without reproducing it on the parent revision.

- [ ] **Step 3: Inspect the exact diff and worktree**

  ```bash
  git diff --check
  git status --short --branch
  git log --oneline --decorate -5
  ```

  Require: no unstaged implementation file, no credential/runtime artifact, and only known pre-existing untracked files outside this repair.

- [ ] **Step 4: Request code review before deployment**

  Review against these exact gates: safe detail is actionable, redaction occurs before persistence, raw stdout never enters the result, exit-78 behavior is unchanged, ordinary failures still retry, no queue mutation surface was added, and the endpoint whitelist is unchanged.

- [ ] **Step 5: Commit any review correction separately and rerun the affected focused tests plus the full suite**

  Use a specific commit subject naming the correction. Do not squash away the observed red test before review.

---

### Task 5: Deploy and observe only the existing job's natural retry

**Files:**
- Update from authorized API evidence only: `/private/tmp/first-letters-hybrid-20260802/qc-diagnostic-response.json`
- Update from authorized API evidence only: `/private/tmp/first-letters-hybrid-20260802/ledger.json`

**Interfaces:**
- Consumes: a reviewed exact repair SHA, normal human Helena authentication held only in memory, and the existing pending QC job.
- Produces: exact-SHA staging convergence and one newly actionable sanitized retry result, or an admissible terminal physical-QC result reached without manual queue action.

- [ ] **Step 1: Publish without overwriting staging history**

  ```bash
  git push origin codex/fix-qc-review-stack-unpack
  git fetch origin staging
  git merge-base --is-ancestor origin/staging HEAD
  git push origin HEAD:staging
  ```

  Require the ancestry check to exit `0`. Never force-push.

- [ ] **Step 2: Require the pipeline for the full repair SHA**

  Wait for the exact revision's unit suite, `build the panel image`, `deploy to gpu-1`, and `smoke on gpu-1` jobs. Require every job to succeed; never accept a green pipeline or runtime from an older SHA.

- [ ] **Step 3: Verify exact-SHA service convergence through the Helena API**

  In one ephemeral authenticated client session, call `GET /api/session`, `GET /api/hosts`, and `GET /api/state`. Require every reported Helena service to carry the exact repair revision and require both QC services active. Do not save the session cookie.

- [ ] **Step 4: Wait for the existing job; never trigger it**

  Poll only:

  ```text
  GET /api/segmentation/qc-jobs?mission=first-letters-hybrid-20260802&sample=PHerc0268&surface=99fd9127-548b-52bd-991b-ad6e7277db0c&limit=10
  ```

  Stop when either `claim_count` increases beyond the pre-deploy baseline of `92`, the job reaches a terminal state, or an agreed bounded observation window expires. Do not call any claim, retry, cancellation, priority, ordering, finalize, validation, or campaign mutation endpoint.

- [ ] **Step 5: Bind only the authorized sanitized response**

  Require the response schema `campaignx.segment_qc_job_diagnostics.v1`, exact mission/sample/surface scope, exactly one job, and the documented field whitelist. Save the exact response bytes, compute SHA-256, and update the incident ledger with the deployed full SHA, response hash, retrieval UTC, state, claim count, last status, error type, and sanitized error. Run:

  ```bash
  jq empty /private/tmp/first-letters-hybrid-20260802/ledger.json
  ```

- [ ] **Step 6: Decide the next action from authorized evidence**

  - If the natural retry exposes a more specific non-terminal adapter error, reproduce that error locally, write a separate TDD scientific repair plan, and stop. Do not edit the scientific path in this propagation repair.
  - If `claim_count` advances but the same generic adapter-exit wrapper remains, record the cause as unresolved and stop. Keep the campaign paused and prohibit scientific-path edits because the authorized evidence still does not identify a scientific failure.
  - If the job reaches `BLOCKED_CONFIGURATION`, preserve it as no scientific conclusion and write a separately reviewed configuration repair plan; do not requeue it manually.
  - If the job reaches `COMPLETED`, require its physical outcome to map through `QC_OUTCOME_STATES` to one of `QC_CT_INSUFFICIENT` / `CT_INSUFFICIENT`, `QC_SCREENED` / `CT_SUPPORTED`, or `QC_REVIEW_PENDING` / `CT_SUPPORTED_REVIEW`. Preserve measured insufficiency or rejection as scientific evidence.
  - If the observation window expires with no new claim, report that state without scheduling or queue mutation.

- [ ] **Step 7: Apply the campaign acceptance gate**

  Call the read-only segments endpoint and the QC diagnostics endpoint once more. Continue the First Letters campaign only when physical QC is no longer `UNVALIDATED` and the terminal state is admissible under the approved campaign plan. Never use `allow_unvalidated`; never describe CT support or retained review as proof of ink or letters.
