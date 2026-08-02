"""Who changed what, and when.

Until this existed the platform could say a job was created "by panel" and
nothing else. Every mutation -- a queued render, a scroll removed from a
mission, an account added, a setting overridden -- was anonymous the moment the
browser tab closed, and the queue's own `created_by` column held the literal
string "panel" for all of them.

The trail is written in the middleware rather than in the routes, so what these
hold is that property: a route nobody thought about is audited because it is a
route. Two of them read the source directly, which is the only way to assert
"and the next one too".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    monkeypatch.setattr(module, "DSN", "")
    return module


@pytest.fixture
def anonymous(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


@pytest.fixture
def client(app_module, anonymous):
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert anonymous.post("/api/session",
                          json={"username": "tester",
                                "password": "a-long-enough-one"}).status_code == 200
    return anonymous


def trail(app_module) -> list[dict]:
    return app_module.read_audit(limit=100)


# --------------------------------------------------------------------------
# What is recorded, and what is not
# --------------------------------------------------------------------------

def test_a_mutation_is_recorded_with_the_four_things_asked_for(client, app_module):
    client.post("/api/users", json={"username": "colleague", "password": "another-long-one"})
    entry = next(e for e in trail(app_module) if e["path"] == "/api/users")
    assert entry["at"].endswith("Z") and entry["at"][4] == "-"   # horario
    assert len(entry["id"]) == 16                                 # id
    assert entry["user"] == "tester"                              # usuario
    assert entry["action"] == "POST /api/users"                   # accion


def test_reading_leaves_no_trace(client, app_module):
    for path in ("/api/users", "/api/config", "/api/scrolls", "/api/jobs"):
        client.get(path)
    assert [e for e in trail(app_module) if e["method"] == "GET"] == []


def test_a_refusal_is_recorded_too(client, app_module):
    """The half people go looking for. A trail of successes answers "what
    happened" and never "who tried"."""
    refused = client.post("/api/users", json={"username": "x", "password": "short"})
    assert refused.status_code >= 400
    entry = next(e for e in trail(app_module) if e["path"] == "/api/users")
    assert entry["status"] == refused.status_code


def test_a_stranger_is_recorded_as_one(anonymous, app_module):
    """The 401 is the interesting line. An unauthenticated attempt to queue GPU
    work is exactly what an audit log is opened to find."""
    assert anonymous.post("/api/jobs", json={"sample_id": "x", "phase": "P4"}).status_code == 401
    entry = next(e for e in trail(app_module) if e["path"] == "/api/jobs")
    assert entry["user"] == "anonymous" and entry["status"] == 401


def test_the_body_is_never_captured(anonymous, app_module):
    """One of these routes sets a password and another sets S3 credentials. A
    trail that recorded what was sent would be the most sensitive file on the
    machine."""
    anonymous.post("/api/session",
                   json={"username": "tester", "password": "hunter2-and-then-some"})
    written = (app_module.AUDIT_ROOT).glob("*.jsonl")
    assert "hunter2" not in "\n".join(f.read_text() for f in written)


def test_the_trail_survives_a_directory_that_cannot_be_written(client, app_module, monkeypatch):
    """An audit write that fails the request it describes is a reason to switch
    auditing off, which is worse than a gap in the trail."""
    monkeypatch.setattr(app_module, "AUDIT_ROOT", Path("/proc/nonexistent/audit"))
    assert client.post("/api/users", json={"username": "y", "password": "a-long-enough-one"}
                       ).status_code < 500


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------

def test_the_trail_is_newest_first_and_filterable(client, app_module):
    client.post("/api/users", json={"username": "one", "password": "a-long-enough-one"})
    client.post("/api/config/override", json={"name": "CX_NOTHING", "value": "1"})
    served = client.get("/api/audit?contains=/api/users").json()
    assert served["entries"] and all("/api/users" in e["action"] for e in served["entries"])
    assert served["count"] == len(served["entries"])

    everything = client.get("/api/audit").json()["entries"]
    assert everything == sorted(everything, key=lambda e: e["at"], reverse=True)
    assert client.get("/api/audit?user=nobody").json()["entries"] == []


def test_a_month_that_cannot_be_parsed_does_not_hide_the_rest(client, app_module):
    client.post("/api/users", json={"username": "two", "password": "a-long-enough-one"})
    month = next(app_module.AUDIT_ROOT.glob("*.jsonl"))
    month.write_text("{not json\n" + month.read_text())
    assert client.get("/api/audit").json()["entries"]


def test_the_trail_is_where_the_backup_already_looks(app_module, tmp_path):
    """It lives under the panel's state directory, so the S3 backup carries it
    without being told. A trail on a disk that dies with the host is a trail
    that answers questions until the moment somebody needs it."""
    import panel.app as module

    assert Path(module.AUDIT_ROOT).parent == Path(module.AUTH_ROOT).parent


# --------------------------------------------------------------------------
# The properties that outlive the routes written today
# --------------------------------------------------------------------------

def test_the_trail_is_written_once_for_every_route(app_module):
    """In the middleware, not in the handlers. A per-route decorator is a thing
    the next route forgets."""
    source = (ROOT / "panel/app.py").read_text()
    assert source.count("record_audit(") == 2   # the definition, and the one call site


def test_the_queue_records_the_person_rather_than_the_process(app_module):
    """`created_by="panel"` on every row told the control plane a job came
    through the panel, which it could already see from the row existing."""
    source = (ROOT / "panel/app.py").read_text()
    for call in ("store.enqueue(", "enqueue_fleet_phase("):
        assert "created_by=who_asked(http)" in source or "who_asked(http)" in source, call
    assert source.count("who_asked(http)") >= 3


def test_removing_a_scroll_refuses_rather_than_protecting_less(app_module):
    """The queue query decides which scrolls may not be disowned. It was wrapped
    in `except Exception: pass`, so an unreachable control plane produced a
    smaller protected set and the amendment went through."""
    source = (ROOT / "panel/app.py").read_text()
    body = source[source.index("def api_remove_scrolls("):]
    body = body[:body.index("\ndef ")]
    assert "pass" not in body.split("\n")[0:200] or "raise HTTPException(503" in body


# --------------------------------------------------------------------------
# Logs, which is the other half of the question
# --------------------------------------------------------------------------

def test_no_container_can_fill_the_disk_with_its_own_logs():
    """Docker's json-file driver is unbounded by default. This host has already
    lost PostgreSQL to a full disk once, with a worker retrying in a loop."""
    yaml = pytest.importorskip("yaml")
    for compose in sorted((ROOT / "containers/compose").glob("*.compose.yaml")):
        services = (yaml.safe_load(compose.read_text()) or {}).get("services") or {}
        for name, service in services.items():
            if not isinstance(service, dict) or "restart" not in service:
                continue   # one-shot runs die with their output
            options = (service.get("logging") or {}).get("options") or {}
            assert options.get("max-size"), f"{compose.name}:{name} logs without a ceiling"
            assert options.get("max-file"), f"{compose.name}:{name} keeps every rotation"


def test_nothing_broad_is_swallowed_in_silence():
    """Five places in the panel catch and continue, and all five are right to:
    a corrupt cache file is refetched, a mission that will not resolve
    contributes no names. What each of them catches is the specific failure it
    has a fallback for.

    `except Exception: pass` is the one that cannot be right, because it also
    catches the failure nobody predicted -- which is how a query that decides
    what may not be deleted came to decide it on partial data.
    """
    for module in ("panel/app.py", "framework/stages/03-ink/fleet/ink_worker.py",
                   "framework/stages/03-ink/fleet/job_store.py",
                   "framework/stages/01-segmentation/fleet/postgres_store.py"):
        source = (ROOT / module).read_text().splitlines()
        for index, line in enumerate(source):
            swallow = (index + 1 < len(source) and source[index + 1].strip() == "pass")
            if not swallow:
                continue
            caught = line.strip()
            assert "Exception" not in caught, f"{module}:{index + 1}: {caught}"
            assert caught != "except:", f"{module}:{index + 1}: bare"


def test_a_sign_in_records_the_name_that_was_claimed(anonymous, app_module):
    """Successful or not. "anonymous signed in" answers nothing, and a failed
    attempt against a real username is the line worth finding."""
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    anonymous.post("/api/session", json={"username": "tester", "password": "wrong-one-here"})
    anonymous.post("/api/session", json={"username": "tester", "password": "a-long-enough-one"})
    sessions = [e for e in trail(app_module) if e["path"] == "/api/session"]
    assert [(e["user"], e["status"]) for e in sessions] == [("tester", 200), ("tester", 401)]
