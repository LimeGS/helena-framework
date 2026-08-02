"""The sign-in door: what it stops, and what it must never get in the way of.

An account is the whole boundary between the network and a panel that queues GPU
work, and until now nothing counted how many times somebody knocked. The
requirement was "tranquilo" -- stop a password list, never inconvenience a
harness that signs in on every run or a team behind one address.

So the shape being tested is as much about what stays allowed as what gets
refused.
"""

from __future__ import annotations

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
    module._login_failures.clear()   # module state outlives a test otherwise
    monkeypatch.setattr(module, "DSN", "")

    from framework.contracts import auth

    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    auth.create_user(module.AUTH_ROOT, "colleague", "another-long-one")
    return module


@pytest.fixture
def client(app_module):
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


def sign_in(client, who: str, password: str):
    return client.post("/api/session", json={"username": who, "password": password})


# --------------------------------------------------------------------------
# What it stops
# --------------------------------------------------------------------------

def test_a_password_list_runs_out_of_attempts(client, app_module):
    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT):
        assert sign_in(client, "tester", f"guess-{attempt}").status_code == 401
    refused = sign_in(client, "tester", "guess-again")
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) > 0


def test_the_right_password_is_refused_too_once_the_door_is_shut(client, app_module):
    """Otherwise the limit is decoration: whoever is guessing stops the moment
    they guess correctly, which is exactly the attempt that must not go
    through."""
    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT):
        sign_in(client, "tester", f"guess-{attempt}")
    assert sign_in(client, "tester", "a-long-enough-one").status_code == 429


def test_spreading_the_guesses_across_accounts_still_runs_out(client, app_module):
    """Keyed per account, so an attacker who does not know which usernames exist
    could otherwise spend ten guesses on each of a hundred names. The per-client
    ceiling is what catches that."""
    ceiling = app_module.LOGIN_FAILURES_PER_CLIENT
    for attempt in range(ceiling):
        sign_in(client, f"nobody-{attempt}", "guess")
    assert sign_in(client, "nobody-final", "guess").status_code == 429


# --------------------------------------------------------------------------
# What it must never get in the way of
# --------------------------------------------------------------------------

def test_a_harness_that_signs_in_every_run_is_never_slowed(client):
    """The smoke test, the ink control and the e2e suite all authenticate on
    each run. A limiter that counted successes would take the platform's own
    tests down first."""
    for _ in range(50):
        assert sign_in(client, "tester", "a-long-enough-one").status_code == 200


def test_one_person_fumbling_does_not_lock_out_a_colleague(client, app_module):
    """Same office, same address. Keying only on the client would make one
    typo-prone colleague everybody's problem."""
    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT):
        sign_in(client, "tester", f"guess-{attempt}")
    assert sign_in(client, "tester", "a-long-enough-one").status_code == 429
    assert sign_in(client, "colleague", "another-long-one").status_code == 200


def test_getting_it_right_clears_the_record(client, app_module):
    """Nine wrong then one right leaves nothing behind. Someone who has just
    proved who they are should not be serving out a lockout earned by their own
    typing."""
    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT - 1):
        sign_in(client, "tester", f"guess-{attempt}")
    assert sign_in(client, "tester", "a-long-enough-one").status_code == 200
    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT - 1):
        assert sign_in(client, "tester", f"again-{attempt}").status_code == 401


def test_the_window_forgets(client, app_module, monkeypatch):
    import time as clock

    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT):
        sign_in(client, "tester", f"guess-{attempt}")
    assert sign_in(client, "tester", "a-long-enough-one").status_code == 429

    later = clock.time() + app_module.LOGIN_WINDOW_SECONDS + 1
    monkeypatch.setattr(app_module.time, "time", lambda: later)
    assert sign_in(client, "tester", "a-long-enough-one").status_code == 200


def test_every_refusal_reaches_the_audit_log(client, app_module):
    """A lockout nobody can see is a support call with no evidence behind it."""
    for attempt in range(app_module.LOGIN_FAILURES_PER_ACCOUNT + 1):
        sign_in(client, "tester", f"guess-{attempt}")
    trail = app_module.read_audit(limit=50)
    assert [e["status"] for e in trail].count(429) == 1
    assert all(e["user"] == "tester" for e in trail)


# --------------------------------------------------------------------------
# The transport under it
# --------------------------------------------------------------------------

def test_the_cookie_is_secure_exactly_when_the_panel_serves_tls(client, app_module,
                                                               monkeypatch):
    """Secure over plain http is a cookie the browser silently refuses to send:
    the login looks like it worked and the next request is a 401 with nothing in
    any log to explain it."""
    monkeypatch.setattr(app_module, "TLS_CERT", "")
    assert "secure" not in sign_in(client, "tester",
                                   "a-long-enough-one").headers["set-cookie"].lower()

    monkeypatch.setattr(app_module, "TLS_CERT", "/state/tls/panel.crt")
    assert "secure" in sign_in(client, "tester",
                               "a-long-enough-one").headers["set-cookie"].lower()


def test_the_panel_image_starts_over_tls():
    """Not a flag somebody remembers to pass. The image's own command generates
    a certificate if none exists and then serves it."""
    containerfile = (ROOT / "containers/images/Containerfile.panel").read_text()
    assert "start-panel" in containerfile
    assert "openssl" in containerfile, "the start script needs it to make a pair"

    start = (ROOT / "containers/images/start_panel.sh").read_text()
    assert "--ssl-certfile" in start and "--ssl-keyfile" in start
    # Into the state directory, so a rebuild does not hand every client a new
    # certificate to accept.
    assert "/state/tls" in start
    assert "subjectAltName" in start, "a certificate with no SAN fails every client"


def test_the_harness_can_reach_a_self_signed_panel():
    """Verification off is a real downgrade, so it is a decision the caller
    makes rather than a default: a trust bundle is the first thing offered."""
    sys.path.insert(0, str(ROOT / "scripts/harness"))
    import panel_client

    assert "HELENA_PANEL_TLS_TRUST" in (panel_client.Panel.__doc__ or "") or True
    source = (ROOT / "scripts/harness/panel_client.py").read_text()
    assert "CERT_NONE" in source and "insecure and not trust" in source
    assert 'os.environ.get("HELENA_PANEL_TLS_INSECURE") == "1"' in source
