# QC Job Diagnostics API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an authenticated, mission-isolated, read-only Helena API that exposes the sanitized persisted cause of a surface-QC retry.

**Architecture:** A pure serializer in `panel/app.py` converts the two permitted QC receipt fields into a bounded safe diagnostic. A FastAPI GET route queries only filtered QC/surface/event columns, reuses `read_scope`, and serializes an explicit response whitelist. Focused fake-PostgreSQL tests prove SQL scope, output shape, redaction, degradation, and lack of mutations before the staging pipeline deploys the exact revision.

**Tech Stack:** Python 3, FastAPI, psycopg 3, pytest, GitLab CI, Helena panel HTTP API.

## Global Constraints

- Implement the approved contract in `docs/superpowers/specs/2026-08-02-qc-job-diagnostics-api-design.md` without broadening it.
- The only new route is `GET /api/segmentation/qc-jobs` with a required `mission` query parameter and optional `sample`, `surface`, and `limit` filters.
- `mission` and `surface` accept only `1..128` characters; `limit` defaults to `100` and accepts only `1..500`.
- Reuse `read_scope(mission, sample)` and the existing P1/P8 surface-ownership model.
- Return only the documented field whitelist. Never expose raw `payload`, raw `result`, `worker_id`, lease fields, artifact URI, DSN, authorization material, cookies, private paths, or secrets.
- The endpoint is read-only: no queue claim, cancellation, retry, validation, DML, or manual state change.
- TDD is mandatory: observe each focused test fail for the missing behavior before adding its production code.
- Use staging mutations through Helena's HTTP API only. Deployment itself proceeds through the protected `staging` Git branch and GitLab pipeline.
- Do not cancel, duplicate, prioritize, reorder, or bypass the pending QC job for surface `99fd9127-548b-52bd-991b-ad6e7277db0c`.
- Do not enqueue another campaign wave until that surface has a terminal physical-QC verdict.

---

### Task 1: Pure QC diagnostic sanitizer

**Files:**
- Create: `tests/test_qc_job_diagnostics_api.py`
- Modify: `panel/app.py` near the other response-normalization helpers

**Interfaces:**
- Consumes: `result: dict[str, Any] | None` from `segment_qc_jobs.result`.
- Produces: `_qc_diagnostic_fields(result) -> dict[str, str | None]` with exactly `last_status`, `error_type`, and `error`.

- [ ] **Step 1: Read the test-quality rules before writing the test**

  Read `superpowers/test-driven-development/writing-good-tests.md` from the installed skill directory. Name the production helper whose absence makes each test fail, and keep each assertion on returned behavior rather than regex internals.

- [ ] **Step 2: Write failing sanitizer tests**

  Create `tests/test_qc_job_diagnostics_api.py` with imports and these focused cases:

  ```python
  from __future__ import annotations

  import json
  import sys
  from pathlib import Path
  from types import SimpleNamespace

  import pytest

  pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")

  ROOT = Path(__file__).resolve().parents[1]
  if str(ROOT) not in sys.path:
      sys.path.insert(0, str(ROOT))


  def test_qc_diagnostic_fields_exposes_only_safe_bounded_values():
      import panel.app as app

      raw = (
          "FileNotFoundError: /srv/helena/private/model.bin "
          "s3://private-bucket/secret-key "
          "postgresql://alice:hunter2@private-db/campaignx "
          "https://internal.example/object?token=raw-token "
          "Authorization: Bearer bearer-value "
          "Cookie: helena_session=cookie-value "
          "password=hunter2 secret=private token=raw-token "
          "AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP"
      )
      fields = app._qc_diagnostic_fields({
          "status": "RETRYABLE_QC_UNAVAILABLE",
          "error": raw,
          "artifact_uri": "s3://must-never-be-read/object",
      })

      assert set(fields) == {"last_status", "error_type", "error"}
      assert fields["last_status"] == "RETRYABLE_QC_UNAVAILABLE"
      assert fields["error_type"] == "FileNotFoundError"
      encoded = json.dumps(fields, sort_keys=True)
      for forbidden in (
          "/srv/helena", "private-bucket", "hunter2", "private-db",
          "internal.example", "bearer-value", "cookie-value", "raw-token",
          "AKIAABCDEFGHIJKLMNOP", "must-never-be-read",
      ):
          assert forbidden not in encoded
      assert "<path>" in fields["error"]
      assert "<artifact-uri>" in fields["error"]
      assert "<dsn>" in fields["error"]
      assert "<url>" in fields["error"]
      assert "<redacted>" in fields["error"]
      assert len(fields["error"]) <= 500


  def test_qc_diagnostic_fields_normalizes_missing_nonstring_and_long_errors():
      import panel.app as app

      assert app._qc_diagnostic_fields(None) == {
          "last_status": None, "error_type": None, "error": None,
      }
      assert app._qc_diagnostic_fields({"status": 7, "error": ["unsafe"]}) == {
          "last_status": None, "error_type": None, "error": None,
      }
      fields = app._qc_diagnostic_fields({
          "status": "RETRYABLE_QC_UNAVAILABLE",
          "error": "RuntimeError: " + "x" * 700,
      })
      assert fields["error_type"] == "RuntimeError"
      assert len(fields["error"]) == 500
      assert fields["error"].endswith("…")
  ```

- [ ] **Step 3: Run the tests and verify RED**

  Run:

  ```bash
  python3 -m pytest -q tests/test_qc_job_diagnostics_api.py
  ```

  Expected: both tests fail with `AttributeError: module 'panel.app' has no attribute '_qc_diagnostic_fields'`. A collection/import error is not the expected RED and must be corrected before continuing.

- [ ] **Step 4: Add the minimal sanitizer implementation**

  Add this helper to `panel/app.py`, using the existing module-level `re` import:

  ```python
  QC_DIAGNOSTIC_SCHEMA = "campaignx.segment_qc_job_diagnostics.v1"


  def _qc_diagnostic_fields(result: dict[str, Any] | None) -> dict[str, str | None]:
      receipt = result if isinstance(result, dict) else {}
      status = receipt.get("status")
      last_status = status if isinstance(status, str) else None
      raw = receipt.get("error")
      if not isinstance(raw, str):
          return {"last_status": last_status, "error_type": None, "error": None}

      message = " ".join(raw.split())
      matched = re.match(r"^([A-Za-z_][A-Za-z0-9_.]{0,127}):", message)
      error_type = matched.group(1) if matched else None
      substitutions = (
          (r"(?i)\bpostgres(?:ql)?://[^\s]+", "<dsn>"),
          (r"(?i)\bs3://[^\s]+", "<artifact-uri>"),
          (r"(?i)\bhttps?://[^\s]+", "<url>"),
          (r"(?i)\b(?:authorization\s*:\s*)?(?:bearer|basic)\s+[^\s,;]+", "<redacted>"),
          (r"(?i)\b(?:cookie|set-cookie)\s*:\s*[^\s,;]+", "<redacted>"),
          (r"(?i)\b(?:password|passwd|secret|token|access[_-]?key(?:_id)?)\s*[:=]\s*[^\s,;]+", "<redacted>"),
          (r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", "<redacted>"),
          (r"(?<![A-Za-z0-9_.-])/(?:[^/\s]+/)*[^/\s]+", "<path>"),
      )
      for pattern, replacement in substitutions:
          message = re.sub(pattern, replacement, message)
      if not message:
          safe = None
      elif len(message) > 500:
          safe = message[:499] + "…"
      else:
          safe = message
      return {"last_status": last_status, "error_type": error_type, "error": safe}
  ```

- [ ] **Step 5: Run focused tests and verify GREEN**

  Run:

  ```bash
  python3 -m pytest -q tests/test_qc_job_diagnostics_api.py
  ```

  Expected: `2 passed` with no warning or error output.

- [ ] **Step 6: Commit the independently testable sanitizer**

  ```bash
  git add panel/app.py tests/test_qc_job_diagnostics_api.py
  git commit -m "feat(panel): sanitize QC retry diagnostics"
  ```

---

### Task 2: Mission-isolated read-only QC jobs endpoint

**Files:**
- Modify: `tests/test_qc_job_diagnostics_api.py`
- Modify: `panel/app.py` immediately after `api_segments`

**Interfaces:**
- Consumes: `read_scope(mission, sample)`, PostgreSQL tables `segment_qc_jobs`, `segment_surfaces`, `segment_events`, and P1/P8 mission lineage tables.
- Produces: `api_segmentation_qc_jobs(sample, mission, surface, limit) -> JSONResponse` registered at `GET /api/segmentation/qc-jobs`.

- [ ] **Step 1: Add a reusable fake read-only PostgreSQL harness to the test file**

  Append this test utility:

  ```python
  class FakeCursor:
      def __init__(self, rows):
          self.rows = rows
          self.statements = []

      def __enter__(self):
          return self

      def __exit__(self, *_args):
          return False

      def execute(self, statement, parameters=()):
          normalized = " ".join(statement.split())
          self.statements.append((normalized, parameters))
          assert normalized.startswith("SELECT")
          for forbidden in (" INSERT ", " UPDATE ", " DELETE ", " FOR UPDATE"):
              assert forbidden not in f" {normalized.upper()} "

      def fetchall(self):
          return list(self.rows)


  class FakeConnection:
      def __init__(self, cursor):
          self._cursor = cursor

      def __enter__(self):
          return self

      def __exit__(self, *_args):
          return False

      def cursor(self):
          return self._cursor


  def install_fake_psycopg(monkeypatch, rows):
      cursor = FakeCursor(rows)
      connection = FakeConnection(cursor)
      monkeypatch.setitem(
          sys.modules, "psycopg",
          SimpleNamespace(connect=lambda *_args, **_kwargs: connection),
      )
      return cursor
  ```

- [ ] **Step 2: Write the failing mission/surface/serialization test**

  Append:

  ```python
  def test_qc_jobs_are_mission_scoped_whitelisted_and_read_only(monkeypatch):
      import panel.app as app
      from datetime import datetime, timezone

      now = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
      rows = [(
          "qc-1", "99fd9127-548b-52bd-991b-ad6e7277db0c", "PHerc268",
          "timesformer-material-support@1.0.0", "PENDING",
          now, now, now, 2,
          {"status": "RETRYABLE_QC_UNAVAILABLE",
           "error": "FileNotFoundError: /private/model.bin",
           "payload": {"password": "must-not-escape"}},
          "GEOMETRY_CERTIFIED", "UNVALIDATED",
      )]
      cursor = install_fake_psycopg(monkeypatch, rows)
      monkeypatch.setattr(app, "DSN", "postgresql://not-opened-directly")
      monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})

      response = app.api_segmentation_qc_jobs(
          sample="PHerc0268", mission="first-letters-hybrid-20260802",
          surface="99fd9127-548b-52bd-991b-ad6e7277db0c", limit=100,
      )
      body = json.loads(response.body)

      assert body["available"] is True
      assert body["schema"] == "campaignx.segment_qc_job_diagnostics.v1"
      assert body["scope"] == {
          "mission": "first-letters-hybrid-20260802",
          "samples": ["PHerc268"],
          "surface": "99fd9127-548b-52bd-991b-ad6e7277db0c",
      }
      assert body["summary"] == {"PENDING": 1}
      assert set(body["jobs"][0]) == {
          "qc_job_id", "surface_id", "sample_id", "profile_id", "state",
          "created_at", "updated_at", "retry_after", "claim_count",
          "last_status", "error_type", "error", "geometry_qc_state",
          "physical_qc_state",
      }
      encoded = json.dumps(body, sort_keys=True)
      assert "must-not-escape" not in encoded
      assert "/private/model.bin" not in encoded
      statement, parameters = cursor.statements[0]
      assert "t.mission_id = %s" in statement
      assert "j.mission_id = %s" in statement
      assert "s.sample_id = ANY(%s)" in statement
      assert "q.surface_id = %s" in statement
      assert parameters == (
          ["PHerc268"], ["PHerc268"],
          "first-letters-hybrid-20260802",
          "first-letters-hybrid-20260802",
          "99fd9127-548b-52bd-991b-ad6e7277db0c", 100,
      )
  ```

- [ ] **Step 3: Write the failing empty-scope isolation test**

  Append:

  ```python
  def test_qc_jobs_empty_scope_never_queries_globally(monkeypatch):
      import panel.app as app

      monkeypatch.setattr(app, "DSN", "configured")
      monkeypatch.setattr(app, "read_scope", lambda mission, sample: set())
      monkeypatch.setitem(
          sys.modules, "psycopg",
          SimpleNamespace(connect=lambda *_args, **_kwargs: pytest.fail(
              "an empty mission scope must not connect or query")),
      )
      body = json.loads(app.api_segmentation_qc_jobs(
          sample="PHerc0841", mission="first-letters-hybrid-20260802",
          surface="foreign-surface", limit=100,
      ).body)
      assert body == {
          "available": True,
          "schema": "campaignx.segment_qc_job_diagnostics.v1",
          "scope": {
              "mission": "first-letters-hybrid-20260802",
              "samples": [],
              "surface": "foreign-surface",
          },
          "summary": {},
          "jobs": [],
      }
  ```

- [ ] **Step 4: Write the remaining failing route-contract tests before implementation**

  Append the degraded, multi-state, validation, and authentication cases before the route exists:

  ```python
  def test_qc_jobs_serializes_multiple_states_and_summarizes_returned_rows(monkeypatch):
      import panel.app as app
      from datetime import datetime, timezone

      now = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
      rows = [
          ("qc-p", "surface-p", "PHerc268", "profile-a", "PENDING",
           now, now, now, 3, {"status": "RETRYABLE_QC_UNAVAILABLE"},
           "GEOMETRY_CERTIFIED", "UNVALIDATED"),
          ("qc-c", "surface-c", "PHerc268", "profile-b", "CLAIMED",
           now, now, None, 1, None,
           "GEOMETRY_CERTIFIED", "UNVALIDATED"),
          ("qc-d", "surface-d", "PHerc268", "profile-c", "COMPLETED",
           now, now, None, 1, {"status": "QC_RETAINED"},
           "GEOMETRY_CERTIFIED", "RETAINED"),
      ]
      install_fake_psycopg(monkeypatch, rows)
      monkeypatch.setattr(app, "DSN", "configured")
      monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})
      body = json.loads(app.api_segmentation_qc_jobs(
          sample="PHerc0268", mission="first-letters-hybrid-20260802",
          surface=None, limit=3).body)
      assert body["summary"] == {"PENDING": 1, "CLAIMED": 1, "COMPLETED": 1}
      assert [job["claim_count"] for job in body["jobs"]] == [3, 1, 1]
      assert body["jobs"][0]["retry_after"] == now.isoformat()
      assert body["jobs"][2]["physical_qc_state"] == "RETAINED"


  def test_qc_jobs_degrades_without_database(monkeypatch):
      import panel.app as app

      monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})
      monkeypatch.setattr(app, "DSN", "")
      body = json.loads(app.api_segmentation_qc_jobs(
          sample=None, mission="first-letters-hybrid-20260802",
          surface=None, limit=100).body)
      assert body == {
          "available": False,
          "schema": "campaignx.segment_qc_job_diagnostics.v1",
          "reason_code": "DATABASE_UNAVAILABLE",
      }


  def test_qc_jobs_degrades_without_postgres_driver(monkeypatch):
      import panel.app as app

      monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})
      monkeypatch.setattr(app, "DSN", "configured")
      monkeypatch.setitem(sys.modules, "psycopg", None)
      body = json.loads(app.api_segmentation_qc_jobs(
          sample=None, mission="first-letters-hybrid-20260802",
          surface=None, limit=100).body)
      assert body == {
          "available": False,
          "schema": "campaignx.segment_qc_job_diagnostics.v1",
          "reason_code": "DRIVER_UNAVAILABLE",
      }


  def test_qc_jobs_query_failure_never_echoes_exception_text(monkeypatch):
      import panel.app as app

      secret = "postgresql://alice:hunter2@private-db/campaignx"
      monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})
      monkeypatch.setattr(app, "DSN", "configured")
      monkeypatch.setitem(
          sys.modules, "psycopg",
          SimpleNamespace(connect=lambda *_args, **_kwargs: (_ for _ in ()).throw(
              RuntimeError(secret))),
      )
      response = app.api_segmentation_qc_jobs(
          sample=None, mission="first-letters-hybrid-20260802",
          surface=None, limit=100)
      encoded = response.body.decode()
      assert json.loads(encoded) == {
          "available": False,
          "schema": "campaignx.segment_qc_job_diagnostics.v1",
          "reason_code": "DIAGNOSTICS_QUERY_FAILED",
          "error_type": "RuntimeError",
      }
      assert "hunter2" not in encoded and "private-db" not in encoded


  @pytest.fixture
  def authenticated_client(tmp_path, monkeypatch):
      import panel.app as app
      from fastapi.testclient import TestClient
      from framework.contracts import auth

      app.AUTH_ROOT = tmp_path / "auth"
      app.AUDIT_ROOT = tmp_path / "audit"
      monkeypatch.setattr(app, "DSN", "")
      auth.create_user(app.AUTH_ROOT, "qc-reader", "qc-reader-password")
      token = auth.login(app.AUTH_ROOT, "qc-reader", "qc-reader-password")
      client = TestClient(app.app, raise_server_exceptions=False)
      client.cookies.set(auth.COOKIE, token)
      return client


  @pytest.mark.parametrize("query", [
      "mission=first-letters-hybrid-20260802&limit=0",
      "mission=first-letters-hybrid-20260802&limit=501",
      "mission=", "mission=" + "x" * 129,
      "mission=first-letters-hybrid-20260802&surface=",
      "mission=first-letters-hybrid-20260802&surface=" + "x" * 129,
  ])
  def test_qc_jobs_rejects_invalid_query_parameters(authenticated_client, query):
      assert authenticated_client.get(
          "/api/segmentation/qc-jobs?" + query).status_code == 422


  def test_qc_jobs_requires_a_human_session():
      import panel.app as app
      from fastapi.testclient import TestClient

      anonymous = TestClient(app.app, raise_server_exceptions=False)
      assert anonymous.get("/api/segmentation/qc-jobs").status_code == 401
  ```

  The unauthenticated assertion exercises pre-existing middleware and may already return 401 before the route exists. All other cases must fail because `api_segmentation_qc_jobs` is absent or the URL is not registered; they are the RED evidence for the new route contract.

- [ ] **Step 5: Run all new route tests and verify RED**

  Run:

  ```bash
  python3 -m pytest -q \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_are_mission_scoped_whitelisted_and_read_only \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_empty_scope_never_queries_globally \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_serializes_multiple_states_and_summarizes_returned_rows \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_degrades_without_database \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_degrades_without_postgres_driver \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_query_failure_never_echoes_exception_text \
    tests/test_qc_job_diagnostics_api.py::test_qc_jobs_rejects_invalid_query_parameters
  ```

  Expected: direct-call cases fail with `AttributeError` for the missing `api_segmentation_qc_jobs`; HTTP validation cases fail because the route is not registered. Fix test setup errors before production code.

- [ ] **Step 6: Implement the minimal route and filtered SELECT**

  Add the following structure after `api_segments` in `panel/app.py`. Keep the selected-column order aligned with the tuple unpacking and do not select `q.payload`, `q.worker_id`, lease columns, or `s.artifact_uri`:

  ```python
  @app.get("/api/segmentation/qc-jobs")
  def api_segmentation_qc_jobs(
      mission: str = Query(..., min_length=1, max_length=128),
      sample: str | None = Query(None),
      surface: str | None = Query(None, min_length=1, max_length=128),
      limit: int = Query(100, ge=1, le=500),
  ):
      resolved = read_scope(mission, sample)
      scope = None if resolved is None else sorted(resolved)
      response_scope = {"mission": mission, "samples": scope, "surface": surface}
      empty = {
          "available": True, "schema": QC_DIAGNOSTIC_SCHEMA,
          "scope": response_scope, "summary": {}, "jobs": [],
      }
      if resolved is not None and not resolved:
          return JSONResponse(empty)
      if not DSN:
          return JSONResponse({"available": False, "schema": QC_DIAGNOSTIC_SCHEMA,
                               "reason_code": "DATABASE_UNAVAILABLE"})
      try:
          import psycopg
      except ImportError:
          return JSONResponse({"available": False, "schema": QC_DIAGNOSTIC_SCHEMA,
                               "reason_code": "DRIVER_UNAVAILABLE"})

      filters = ["(%s::text[] IS NULL OR s.sample_id = ANY(%s))"]
      parameters: list[Any] = [scope, scope]
      if mission:
          filters.append("""(
            EXISTS (
              SELECT 1 FROM segment_artifact_sets art
              JOIN segment_attempts a ON a.attempt_id=art.attempt_id
              JOIN segment_tasks t ON t.task_id=a.task_id
              WHERE art.manifest->>'artifact_sha256'=s.artifact_sha256
                AND t.mission_id = %s)
            OR EXISTS (
              SELECT 1 FROM surface_derivations d
              JOIN ink_jobs j ON j.job_id=d.job_id
              WHERE d.child_surface_id=s.surface_id
                AND j.mission_id = %s)
          )""")
          parameters.extend([mission, mission])
      if surface:
          filters.append("q.surface_id = %s")
          parameters.append(surface)
      parameters.append(limit)

      try:
          with psycopg.connect(DSN, connect_timeout=5) as conn, conn.cursor() as cur:
              cur.execute(
                  """SELECT q.qc_job_id,q.surface_id,s.sample_id,q.profile_id,q.state,
                            q.created_at,q.updated_at,q.retry_after,
                            (SELECT count(*) FROM segment_events e
                              WHERE e.event_type='QC_CLAIMED'
                                AND e.payload->>'qc_job_id'=q.qc_job_id),
                            q.result,
                            COALESCE(to_jsonb(s)->>'geometry_qc_state',
                                     'GEOMETRY_UNMEASURED'),
                            s.physical_qc_state
                       FROM segment_qc_jobs q
                       JOIN segment_surfaces s ON s.surface_id=q.surface_id
                      WHERE """ + " AND ".join(filters) + """
                      ORDER BY q.updated_at DESC,q.created_at DESC,q.qc_job_id DESC
                      LIMIT %s""",
                  tuple(parameters),
              )
              rows = cur.fetchall()
      except Exception as exc:
          return JSONResponse({
              "available": False, "schema": QC_DIAGNOSTIC_SCHEMA,
              "reason_code": "DIAGNOSTICS_QUERY_FAILED",
              "error_type": type(exc).__name__,
          })

      jobs = []
      for row in rows:
          (qc_job_id, surface_id, sample_id, profile_id, state,
           created_at, updated_at, retry_after, claim_count, result,
           geometry_qc_state, physical_qc_state) = row
          jobs.append({
              "qc_job_id": qc_job_id, "surface_id": surface_id,
              "sample_id": sample_id, "profile_id": profile_id, "state": state,
              "created_at": created_at.isoformat() if created_at else None,
              "updated_at": updated_at.isoformat() if updated_at else None,
              "retry_after": retry_after.isoformat() if retry_after else None,
              "claim_count": int(claim_count),
              **_qc_diagnostic_fields(result),
              "geometry_qc_state": geometry_qc_state,
              "physical_qc_state": physical_qc_state,
          })
      summary: dict[str, int] = {}
      for job in jobs:
          summary[job["state"]] = summary.get(job["state"], 0) + 1
      return JSONResponse({**empty, "summary": summary, "jobs": jobs})
  ```

- [ ] **Step 7: Run the entire focused file and verify GREEN**

  ```bash
  python3 -m pytest -q tests/test_qc_job_diagnostics_api.py
  ```

  Expected: every focused sanitizer, scope, state, degraded, validation, and authentication test passes.

- [ ] **Step 8: Commit the route and its complete safety contract**

  ```bash
  git add panel/app.py tests/test_qc_job_diagnostics_api.py
  git commit -m "feat(panel): expose scoped QC job diagnostics"
  ```

---

### Task 3: Regression and specification-compliance gates

**Files:**
- Verify: `tests/test_qc_job_diagnostics_api.py`
- Verify: `tests/test_every_endpoint_answers.py`
- Verify: `tests/test_empty_mission_isolation.py`
- Verify: `tests/test_planner_is_per_task.py`

**Interfaces:**
- Consumes: the route and serializer from Tasks 1 and 2.
- Produces: full regression evidence and a requirement-by-requirement diff audit without adding unplanned behavior.

- [ ] **Step 1: Run the focused test file from a clean process**

  ```bash
  python3 -m pytest -q tests/test_qc_job_diagnostics_api.py
  ```

  Expected: all tests pass. If the HTTP authentication test inherits a prior module's cookie or auth root, repair the fixture isolation instead of weakening the assertion.

- [ ] **Step 2: Run relevant panel contract regressions**

  ```bash
  python3 -m pytest -q \
    tests/test_every_endpoint_answers.py \
    tests/test_empty_mission_isolation.py \
    tests/test_planner_is_per_task.py \
    tests/test_qc_job_diagnostics_api.py
  ```

  Expected: all tests pass with only pre-existing documented skips. Confirm `test_every_endpoint_answers.py` discovers the new route, accepts it with a session when `CX_DB` is absent, and rejects it anonymously.

- [ ] **Step 3: Run the repository's full Python regression suite**

  ```bash
  python3 -m pytest -q
  ```

  Expected: exit code 0. Record the exact passed/skipped counts and elapsed time. Do not claim full regression coverage if the command is interrupted, truncated before the summary, or fails outside the focused file.

- [ ] **Step 4: Audit the final diff against the approved spec**

  ```bash
  git diff --check HEAD~2..HEAD
  git diff --stat HEAD~2..HEAD
  rg -n "payload|worker_id|lease_token|lease_expires|artifact_uri|str\(exc\)" \
    panel/app.py tests/test_qc_job_diagnostics_api.py
  ```

  Inspect every match. Matches in assertions that prove forbidden fields are absent are expected; no forbidden field may be selected or serialized by the route.

- [ ] **Step 5: Record verification without changing production code**

  Preserve the exact focused and full-suite summaries in the task handoff. If the audit finds a missing requirement, return to Task 2 with a new failing test; do not patch production code inside this verification task.

---

### Task 4: Push, deploy to staging, and retrieve the persisted QC cause

**Files:**
- Update temporarily: `/private/tmp/first-letters-hybrid-20260802/ledger.json`
- Create temporarily: `/private/tmp/first-letters-hybrid-20260802/qc-diagnostic-response.json`
- Create after evidence: `docs/superpowers/plans/2026-08-02-qc-retry-root-cause-repair.md`

**Interfaces:**
- Consumes: a fully tested endpoint revision and the existing human Helena development account.
- Produces: a green `staging` deploy, one sanitized surface-bound diagnostic response, a ledger digest, and a separate evidence-based repair plan.

- [ ] **Step 1: Verify the branch is safe to publish**

  ```bash
  git status --short --branch
  git log --oneline --decorate -5
  git merge-base --is-ancestor origin/staging HEAD
  ```

  Require: only the pre-existing untracked campaign plan may remain; no implementation file is unstaged; `origin/staging` is an ancestor of `HEAD`.

- [ ] **Step 2: Push the feature branch and fast-forward staging**

  ```bash
  git push origin codex/fix-qc-review-stack-unpack
  git push origin HEAD:staging
  ```

  Do not force-push. If the second command is rejected, fetch and inspect the new staging commits before merging; do not overwrite them.

- [ ] **Step 3: Wait for the exact staging pipeline**

  In GitLab, require the pipeline for the full `HEAD` SHA to finish successfully. Specifically require `build panel image`, `deploy to gpu-1`, and `smoke on gpu-1` to pass. Do not accept a green pipeline for an older SHA.

- [ ] **Step 4: Verify deployment revision through the Helena API**

  Authenticate through `POST /api/session` using the supplied development account and an ephemeral client session. Call `GET /api/hosts`, `GET /api/state`, and `GET /api/session`. Require all services reported by the platform to carry the exact deployed revision and require gpu-1 plus both QC workers to be active. Never print the password, session cookie, or authorization material.

- [ ] **Step 5: Read the surface-bound diagnostic exactly once**

  Use `scripts/harness/panel_client.py` or an equivalent stdlib client within one process so the HttpOnly cookie remains in memory. Call:

  ```text
  GET /api/segmentation/qc-jobs?mission=first-letters-hybrid-20260802&sample=PHerc0268&surface=99fd9127-548b-52bd-991b-ad6e7277db0c&limit=10
  ```

  Require `available: true`, exactly one job for the named surface, mission scope `PHerc268`, and no forbidden fields. Save the sanitized JSON response to `/private/tmp/first-letters-hybrid-20260802/qc-diagnostic-response.json`. If the response is ambiguous, repeat only the idempotent GET; never mutate the queue.

- [ ] **Step 6: Bind the diagnostic to the campaign ledger**

  Compute SHA-256 over the exact response bytes. Update the incident in `ledger.json` with deployed commit, endpoint schema, QC job ID, state, claim count, last status, error type, sanitized error, response hash, and retrieval UTC time. Run `jq empty` afterward. Do not record credentials, cookies, worker IDs, private paths, DSNs, or raw receipts.

- [ ] **Step 7: State and verify one root-cause hypothesis**

  Trace the returned `error_type` and sanitized message through `framework/stages/01-segmentation/fleet/qc_worker.py`, the QC adapter, compose mounts/environment, and a neighboring successful QC path. Reproduce the failure locally or with read-only/API-visible staging evidence. Do not edit the QC path until the hypothesis explains both repeated `QC_REQUEUED_UNAVAILABLE` events and zero GPU utilization.

- [ ] **Step 8: Write the root-cause-specific repair plan**

  Create `docs/superpowers/plans/2026-08-02-qc-retry-root-cause-repair.md` only after the cause is known. Its title and body must name the verified cause, failing contract, exact regression test, minimal code/config change, full-suite verification, staging deployment, natural retry observation, and terminal physical-QC acceptance.

- [ ] **Step 9: Resume the First Letters campaign only after QC evidence**

  Deploy the separately reviewed TDD fix, let the existing pending job retry naturally, and poll the new read endpoint plus `/api/segmentation/segments`. Continue campaign Task 5 only if physical QC reaches an admissible terminal state. Preserve a measured rejection as scientific evidence and choose the next bounded P1 wave; never use `allow_unvalidated`.
