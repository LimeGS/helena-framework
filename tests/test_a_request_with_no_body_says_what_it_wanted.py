"""A POST with no body answered 500, which blamed the server for the caller.

Found by probing every declared POST route with four malformed bodies. 130 of
180 probes answered 422 naming the missing field -- the standard the rest of
the API already meets. Five answered `Internal Server Error` with nothing else,
across two routes and for two different reasons.

`POST /api/artifacts/{key}` read its body with `await http.json()` and no
guard. An absent or non-JSON body raises JSONDecodeError, which is not an
HTTPException, so it reached the error middleware as an unhandled exception.
The caller sent a bad request and was told the server broke.

`POST /api/strips` already refused an empty body with a 400 -- it never got
there. STRIPS defaulted to `workspace/strips` *inside the checkout*, and the
panel does not own the directory it runs from: mkdir raised PermissionError.
The default now sits beside the panel's other state, under CX_RUNS' parent,
which is a volume the panel writes to by design.
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
def panel(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    module.STRIPS = tmp_path / "strips"
    monkeypatch.setattr(module, "DSN", "")
    client = TestClient(module.app, raise_server_exceptions=False)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return module, client


BAD_BODIES = (
    ("no body at all", {}),
    ("an empty string", {"content": ""}),
    ("not json", {"content": "]["}),
    ("json that is not an object", {"json": [1, 2, 3]}),
)


@pytest.mark.parametrize("what,kwargs", BAD_BODIES, ids=[b[0] for b in BAD_BODIES])
def test_copying_an_artifact_without_a_body_is_a_bad_request(panel, what, kwargs):
    module, client = panel
    response = client.post("/api/artifacts/some-key", **kwargs)
    assert response.status_code < 500, (
        f"{what} answered {response.status_code}: {response.text[:200]}")
    assert 400 <= response.status_code < 500
    # The refusal names what the route wanted, so the caller can fix the call.
    assert "copy_from" in response.text


def test_strips_land_beside_the_panels_own_state(monkeypatch):
    """Wherever a deployment puts the panel's state, strips follow it there.

    The failing default pointed at the checkout, which in the container is the
    image's read-only app directory. Anchoring on the auth directory -- the one
    place the panel is guaranteed to write, because it keeps password hashes
    there -- moves strips wherever CX_AUTH already moved everything else.
    """
    import importlib

    import panel.app as module

    monkeypatch.setenv("CX_AUTH", "/state/auth")
    reloaded = importlib.reload(module)
    try:
        assert reloaded.STRIPS == Path("/state/strips"), (
            f"CX_STRIPS defaulted to {reloaded.STRIPS} while the panel keeps "
            "its state in /state; a directory the panel cannot write answers "
            "500 on upload")
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_uploading_an_empty_strip_is_a_bad_request(panel):
    module, client = panel
    response = client.post("/api/strips", content=b"")
    assert response.status_code < 500, (
        f"an empty upload answered {response.status_code}: {response.text[:200]}")
    assert response.status_code == 400
