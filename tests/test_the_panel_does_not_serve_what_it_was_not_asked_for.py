"""Two ways the panel handed out things it holds, found in an adversarial audit.

Neither needed a clever payload. One was a path with `..` in it and the other
was a plain GET, and both were reachable by any account that could log in --
which in this app is every account, because it has no roles.

**The SPA fallback served any file the process could read.** `/assets` is
mounted through Starlette's StaticFiles, which checks containment itself. Every
other path fell through to a route that did `DIST / full_path` and served it if
it was a file. `.is_file()` and `FileResponse` both resolve `..` at the OS
layer, so the traversal never had to survive routing -- it only had to reach
that line. Two levels up from `web/dist` is the auth directory, whose own
setting says it "holds password hashes, so it is written 0600": USERS.json and
SESSIONS.json, the scrypt hashes and the live session tokens.

**`/api/config` returned secrets in clear.** `secret: true` travelled beside the
value rather than instead of it, and the only thing between a reader and CX_DB
-- a Postgres DSN with its password in it -- was the Config page choosing an
input of type "password". Masking in the browser is not redaction.
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
def panel(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    monkeypatch.setattr(module, "DSN", "")
    # A real-looking secret, so the redaction is tested against a value rather
    # than against an empty default that would pass either way.
    for setting in module.SETTINGS:
        if setting.get("secret"):
            setting["value"] = "postgresql://helena:hunter2@10.0.0.9:5432/campaignx"
    client = TestClient(module.app)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return module, client


# -- the traversal ---------------------------------------------------------

def test_the_spa_route_refuses_to_leave_its_own_directory(panel):
    """Signed in, which is the only precondition: the app has no roles, so any
    account that exists reaches this."""
    module, client = panel
    if not module.DIST.exists():
        pytest.skip("the frontend is not built in this checkout")

    for escape in ("../../auth/USERS.json",
                   "../../auth/SESSIONS.json",
                   "assets/../../../../etc/passwd",
                   "..%2f..%2fauth%2fUSERS.json"):
        response = client.get(f"/{escape}")
        body = response.content
        # It answers -- the app shell is the fallback for every client-side
        # route -- but it must never answer with the file that was asked for.
        assert b"scrypt$" not in body, f"{escape} served a password hash"
        assert b"root:x:" not in body, f"{escape} served /etc/passwd"
        assert b"session" not in body.lower() or b"<!doctype" in body.lower(), (
            f"{escape} served something that is not the app shell")


def test_a_real_asset_is_still_served(panel):
    """The check has to keep the route doing its job, or it is not a fix."""
    module, client = panel
    if not (module.DIST / "index.html").is_file():
        pytest.skip("the frontend is not built in this checkout")

    assert client.get("/index.html").status_code == 200


def test_the_containment_check_resolves_before_comparing():
    """A prefix test on the unresolved path passes for `dist/../../auth`, which
    is exactly the string an attacker sends. Pinned as arithmetic so it cannot
    regress into a `startswith` on the raw join."""
    dist = Path("/app/panel/web/dist")

    assert not (dist / "../../auth/USERS.json").resolve().is_relative_to(dist)
    assert (dist / "index.html").resolve().is_relative_to(dist)


# -- the secrets -----------------------------------------------------------

def test_a_secret_setting_never_leaves_the_server(panel):
    module, client = panel

    body = client.get("/api/config").json()
    secrets = [row for row in body["environment"] if row.get("secret")]

    assert secrets, "no setting is marked secret; this test has stopped testing"
    for row in secrets:
        assert row["value"] == "", f"{row['name']} travelled with its value"
        assert "value_present" in row, (
            f"{row['name']} is redacted but the page cannot tell whether it is set")


def test_the_database_credential_is_not_in_the_response(panel):
    """CX_DB is a Postgres DSN with its password in it. The whole response is
    searched rather than one field, because a credential leaks through whatever
    field happens to carry it."""
    module, client = panel
    body = client.get("/api/config").json()
    row = next(r for r in body["environment"] if r["name"] == "CX_DB")

    assert row["value"] == "" and row["value_present"] is True
    # The default goes too: it is either a placeholder, which costs nothing to
    # hide, or a credential somebody set as one.
    assert row["default"] == ""
    assert "hunter2" not in json.dumps(body), "the password reached the client"
    # `example` survives on purpose -- it is documentation, written to be fake --
    # so the check is for the real value, not for the shape of a DSN.
    assert row["example"]


def test_the_settings_the_server_keeps_are_untouched(panel):
    """Redaction is at serialisation. The process still needs the real value --
    it is what it connects with -- so the in-memory table must not be scrubbed."""
    module, _ = panel

    assert any(row.get("secret") and row["value"] for row in module.SETTINGS), (
        "the redaction reached SETTINGS itself, which would break the panel's "
        "own use of the value it is hiding")


def test_a_surfaces_projection_that_moved_degrades_instead_of_500ing():
    """The four newest columns are read through to_jsonb so an older control
    plane still answers. That care was undone by where the unpacking sat: it
    named all eighteen columns *after* the except, so a control plane whose
    projection returned fewer raised ValueError out of the endpoint. A synthetic
    control's own stub drifted to fourteen and took the surfaces page with it.
    """
    import panel.app as panel_app

    short = [("surface-1", "PHerc0139", 1.0, "SUCCEEDED", "UNVALIDATED",
              "GEOMETRY_CERTIFIED", "s3://x", "a" * 64, None, None, None,
              "campaign-x", None, None)]
    with pytest.raises(ValueError):
        panel_app._segment_rows(short)

    full = [short[0] + (None, None, None, None)]
    rows = panel_app._segment_rows(full)
    assert rows[0]["lamina_qc_state"] == "LAMINA_UNMEASURED"
    assert rows[0]["seed_agreement_state"] == "SEED_UNPAIRED"
    assert rows[0]["seed_agreement_normal_um"] is None, (
        "no pair, no number -- an unpaired surface must not read as zero drift")


def test_the_dsn_never_reaches_an_argv():
    """`ps` is world-readable, and the DSN carries a password.

    The fleet CLI has taken `postgres-env://NAME` since it was written -- "the
    preferred production form because it keeps credentials out of process
    listings, shell history, logs and receipts" -- and the job queue already
    built its argv that way. The panel's own five call sites did not.
    """
    import inspect

    import panel.app as panel_app

    source = inspect.getsource(panel_app)
    assert '"--db", DSN,' not in source, (
        "a panel subprocess still puts the DSN on the command line")
    assert '"--db", DSN]' not in source
    assert panel_app.DSN_ARGUMENT == "postgres-env://CX_DB"
    # And the name only resolves if the value travels another way.
    assert panel_app.dsn_environment()["CX_DB"] == panel_app.DSN
