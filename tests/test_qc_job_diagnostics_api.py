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


def test_shared_qc_error_contract_redacts_all_required_serialized_forms():
    """A new sanitizer pattern must not leave a sibling sensitive form behind."""
    from framework.contracts import qc_diagnostics

    raw = (
        "RuntimeError: Authorization: Token auth-token Cookie=session-cookie "
        "Authorization Token delimiter-free-token "
        "redis://alice:redis-secret@private-redis:6379/0 "
        "mysql://bob:mysql-secret@private-mysql/qc "
        "gs://private-bucket/checkpoint.bin C:\\private\\checkpoint.bin "
        "worker_id=helper-underscore-worker worker-id=helper-hyphen-worker "
        "worker identity=helper-space-worker api_key=adapter-api-key "
        "client_secret=adapter-client-secret "
        "custom+artifact://private-authority/item "
        "x://single-letter-authority/item"
    )

    error_type, error = qc_diagnostics.sanitize_error(raw)

    assert error_type == "RuntimeError"
    assert error is not None
    for forbidden in (
        "auth-token", "session-cookie", "redis-secret", "private-redis",
        "mysql-secret", "private-mysql", "private-bucket", "C:\\private",
        "helper-underscore-worker", "helper-hyphen-worker",
        "helper-space-worker", "adapter-api-key", "adapter-client-secret",
        "private-authority", "single-letter-authority", "delimiter-free-token",
    ):
        assert forbidden not in error
    for replacement in ("<redacted>", "<dsn>", "<artifact-uri>", "<path>", "<uri>"):
        assert replacement in error
    assert qc_diagnostics.sanitize_error("RuntimeError: x://single-letter-authority/item") == (
        "RuntimeError", "RuntimeError: <uri>"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "RuntimeError: {'Authorization': 'Digest username=alice, realm=private-realm, response=digest-secret'}",
        "RuntimeError: {'Cookie': 'a=first-cookie, b=second-cookie'}",
        'RuntimeError: {"Proxy-Authorization": "Token proxy-secret extra"}',
        'RuntimeError: {"Set-Cookie": "sid=first; admin=second"}',
    ],
)
def test_shared_qc_error_contract_redacts_quoted_serialized_header_keys(raw):
    """Quoted mapping keys are as sensitive as unquoted header labels."""
    from framework.contracts import qc_diagnostics

    error_type, error = qc_diagnostics.sanitize_error(raw)

    assert error_type == "RuntimeError"
    assert error is not None
    assert "<redacted>" in error
    for forbidden in (
        "alice", "private-realm", "digest-secret", "first-cookie",
        "second-cookie", "proxy-secret", "sid=first", "admin=second",
    ):
        assert forbidden not in error


@pytest.mark.parametrize(
    ("raw", "forbidden", "replacement"),
    [
        (
            "RuntimeError: Authorization: Digest username=alice, "
            "realm=private-realm, response=digest-secret",
            ("alice", "private-realm", "digest-secret"),
            "<redacted>",
        ),
        (
            "RuntimeError: Proxy-Authorization: Digest username=bob, "
            "realm=proxy-realm, response=proxy-secret",
            ("bob", "proxy-realm", "proxy-secret"),
            "<redacted>",
        ),
        (
            "RuntimeError: Cookie: a=first-cookie, b=second-cookie",
            ("first-cookie", "second-cookie"),
            "<redacted>",
        ),
        (
            "RuntimeError: Set-Cookie: a=first-set-cookie, b=second-set-cookie",
            ("first-set-cookie", "second-set-cookie"),
            "<redacted>",
        ),
        (
            "RuntimeError: Authorization: Custom raw auth secret",
            ("Custom", "raw auth secret"),
            "<redacted>",
        ),
        (
            "RuntimeError: Cookie: raw cookie secret",
            ("raw cookie secret",),
            "<redacted>",
        ),
        (
            str({"token": "raw-secret", "worker_id": "node-17"}),
            ("raw-secret", "node-17"),
            "<redacted>",
        ),
        (
            "RuntimeError: 'password': 'two word secret', "
            '"worker_id": "private worker 17"',
            ("two word secret", "private worker 17"),
            "<redacted>",
        ),
        (
            "RuntimeError: token=raw unquoted secret phrase",
            ("unquoted secret phrase",),
            "<redacted>",
        ),
        (
            r"RuntimeError: C:\Users\Alice Smith\secret model.bin",
            ("Alice Smith", "secret model.bin"),
            "<path>",
        ),
        (
            r'RuntimeError: "C:\Users\Alice Smith\secret model.bin"',
            ("Alice Smith", "secret model.bin"),
            "<path>",
        ),
        (
            r"RuntimeError: \\private-server\secret share\hidden model.bin",
            ("private-server", "secret share", "hidden model.bin"),
            "<path>",
        ),
        (
            'RuntimeError: "/srv/private/Alice Smith/secret model.bin"',
            ("/srv/private", "Alice Smith", "secret model.bin"),
            "<path>",
        ),
        (
            "RuntimeError: /srv/private/Alice Smith/secret model.bin",
            ("/srv/private", "Alice Smith", "secret model.bin"),
            "<path>",
        ),
    ],
)
def test_shared_qc_error_contract_fully_redacts_sensitive_multitoken_forms(
    raw, forbidden, replacement,
):
    """Partial matches are leaks when the sensitive value spans tokens."""
    from framework.contracts import qc_diagnostics

    _error_type, error = qc_diagnostics.sanitize_error(raw)

    assert error is not None
    assert replacement in error
    assert len(error) <= 500
    for value in forbidden:
        assert value not in error


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


def test_qc_diagnostic_fields_redacts_every_semicolon_cookie_pair():
    import panel.app as app

    fields = app._qc_diagnostic_fields({
        "error": (
            "RuntimeError: Cookie: helena_session=first-cookie; "
            "csrf_token=second-cookie; preferences=third-cookie"
        ),
    })

    assert fields["error_type"] == "RuntimeError"
    assert fields["error"] == "RuntimeError: <redacted>"
    for forbidden in ("first-cookie", "second-cookie", "third-cookie"):
        assert forbidden not in fields["error"]


@pytest.mark.parametrize(
    "raw",
    [
        "RuntimeError: {'Authorization': 'Digest username=alice, realm=private-realm, response=digest-secret'}",
        "RuntimeError: {'Cookie': 'a=first-cookie, b=second-cookie'}",
        'RuntimeError: {"Proxy-Authorization": "Token proxy-secret extra"}',
        'RuntimeError: {"Set-Cookie": "sid=first; admin=second"}',
    ],
)
def test_qc_diagnostic_fields_redacts_quoted_serialized_header_keys(raw):
    """The panel must expose the same safe form as the shared helper."""
    import panel.app as app

    fields = app._qc_diagnostic_fields({"error": raw})

    assert fields["error_type"] == "RuntimeError"
    assert fields["error"] is not None
    assert "<redacted>" in fields["error"]
    for forbidden in (
        "alice", "private-realm", "digest-secret", "first-cookie",
        "second-cookie", "proxy-secret", "sid=first", "admin=second",
    ):
        assert forbidden not in fields["error"]


def test_qc_diagnostic_fields_redacts_common_aws_credential_assignments():
    import panel.app as app

    fields = app._qc_diagnostic_fields({
        "error": (
            "RuntimeError: AWS_ACCESS_KEY_ID=AKIAABCDEFGHIJKLMNOP "
            "AWS_SECRET_ACCESS_KEY=aws-secret-value "
            "AWS_SESSION_TOKEN=aws-session-value "
            "AWS_SECURITY_TOKEN=aws-security-value"
        ),
    })

    assert fields["error_type"] == "RuntimeError"
    for forbidden in (
        "AKIAABCDEFGHIJKLMNOP",
        "aws-secret-value",
        "aws-session-value",
        "aws-security-value",
    ):
        assert forbidden not in fields["error"]


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


def test_qc_jobs_selects_only_permitted_result_text_fields(monkeypatch):
    import panel.app as app

    cursor = install_fake_psycopg(monkeypatch, [])
    monkeypatch.setattr(app, "DSN", "configured")
    monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})

    app.api_segmentation_qc_jobs(
        sample="PHerc0268",
        mission="first-letters-hybrid-20260802",
        surface=None,
        limit=1,
    )

    statement, _parameters = cursor.statements[0]
    select_clause = statement.split(" FROM segment_qc_jobs q", 1)[0]
    assert "q.result->>'status'" in select_clause
    assert "q.result->>'error'" in select_clause
    permitted_removed = select_clause.replace(
        "q.result->>'status'", ""
    ).replace("q.result->>'error'", "")
    for field in ("schema", "surface_id", "outcome", "evidence_manifest_sha256", "ink_used"):
        permitted_removed = permitted_removed.replace(f"q.result->>'{field}'", "")
    permitted_removed = permitted_removed.replace("to_jsonb(q)->'result'->>'result_sha256'", "")
    assert "q.result" not in permitted_removed


def test_qc_jobs_returns_safe_hash_bound_causal_receipt_for_current_p2(monkeypatch):
    import panel.app as app
    from datetime import datetime, timezone

    now = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
    rows = [(
        "qc-current", "surface-control", "PHerc0139",
        "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0", "COMPLETED",
        now, now, None, 1, "QC_RETAINED", None, "4" * 64,
        "GEOMETRY_CERTIFIED", "CT_SUPPORTED_REVIEW", "3" * 64,
        "p2-current", "geometry-promotion:p2-current:qc-current",
        "campaignx.segment_qc_result.v1", "surface-control",
        "CT_SUPPORTED_RETAINED_FOR_REVIEW", "7" * 64, False,
    )]
    install_fake_psycopg(monkeypatch, rows)
    monkeypatch.setattr(app, "DSN", "configured")
    monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc0139"})
    body = json.loads(app.api_segmentation_qc_jobs(
        sample="PHerc0139", mission="control", surface="surface-control", limit=10).body)
    job = body["jobs"][0]
    assert job["unblocked_by_job_id"] == "p2-current"
    assert job["surface_artifact_sha256"] == "3" * 64
    assert job["promotion_event_id"] == "geometry-promotion:p2-current:qc-current"
    assert job["source_result_sha256"] == "4" * 64
    assert job["result_sha256"] == app._canonical_document_sha256(job["result"])
    assert job["profile_sha256"] == next(
        row["sha256"] for row in json.loads((ROOT / "framework/profiles/01-segmentation/first-letters-control-policy-1.1.0.json").read_text())["profile_locks"]
        if row["profile_id"] == job["profile_id"])
    assert "evidence_uri" not in job["result"]


def test_qc_jobs_are_mission_scoped_whitelisted_and_read_only(monkeypatch):
    import panel.app as app
    from datetime import datetime, timezone

    now = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
    rows = [(
        "qc-1", "99fd9127-548b-52bd-991b-ad6e7277db0c", "PHerc268",
        "timesformer-material-support@1.0.0", "PENDING",
        now, now, now, 2,
        "RETRYABLE_QC_UNAVAILABLE", "FileNotFoundError: /private/model.bin",
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
    assert "/private/model.bin" not in encoded
    statement, parameters = cursor.statements[0]
    assert "t.mission_id = %s" in statement
    assert "j.mission_id = %s" in statement
    # The third way a surface belongs to a mission: uploaded into it, with no
    # attempt and no derivation to join through.
    assert "s.payload ->> 'mission_id' = %s" in statement
    assert "s.sample_id = ANY(%s)" in statement
    assert "q.surface_id = %s" in statement
    assert "ORDER BY q.updated_at DESC,q.created_at DESC,q.qc_job_id DESC" in statement
    assert statement.endswith("LIMIT %s")
    assert parameters == (
        ["PHerc268"], ["PHerc268"],
        *(["first-letters-hybrid-20260802"] * app.SURFACE_MISSION_PARAMETERS),
        "99fd9127-548b-52bd-991b-ad6e7277db0c", 100,
    )


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


def test_qc_jobs_serializes_multiple_states_and_summarizes_returned_rows(monkeypatch):
    import panel.app as app
    from datetime import datetime, timezone

    now = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)
    rows = [
        ("qc-p", "surface-p", "PHerc268", "profile-a", "PENDING",
         now, now, None, 3, "RETRYABLE_QC_UNAVAILABLE", None,
         "GEOMETRY_CERTIFIED", "UNVALIDATED"),
        ("qc-delayed", "surface-delayed", "PHerc268", "profile-a", "PENDING",
         now, now, now, 4, "RETRYABLE_QC_UNAVAILABLE", None,
         "GEOMETRY_CERTIFIED", "UNVALIDATED"),
        ("qc-c", "surface-c", "PHerc268", "profile-b", "CLAIMED",
         now, now, None, 1, None, None,
         "GEOMETRY_CERTIFIED", "UNVALIDATED"),
        ("qc-d", "surface-d", "PHerc268", "profile-c", "COMPLETED",
         now, now, None, 1, "QC_RETAINED", None,
         "GEOMETRY_CERTIFIED", "RETAINED"),
    ]
    install_fake_psycopg(monkeypatch, rows)
    monkeypatch.setattr(app, "DSN", "configured")
    monkeypatch.setattr(app, "read_scope", lambda mission, sample: {"PHerc268"})
    body = json.loads(app.api_segmentation_qc_jobs(
        sample="PHerc0268", mission="first-letters-hybrid-20260802",
        surface=None, limit=4).body)
    assert body["summary"] == {"PENDING": 2, "CLAIMED": 1, "COMPLETED": 1}
    assert [job["claim_count"] for job in body["jobs"]] == [3, 4, 1, 1]
    assert [job["retry_after"] for job in body["jobs"]] == [
        None, now.isoformat(), None, None,
    ]
    assert body["jobs"][3]["physical_qc_state"] == "RETAINED"


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


def test_qc_jobs_requires_mission_scope(authenticated_client):
    assert authenticated_client.get("/api/segmentation/qc-jobs").status_code == 422


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
