"""A NUL byte in a URL answered 500 on a dozen routes.

Found by putting hostile values through every declared path and query
parameter. Most shapes were already refused cleanly -- 332 404s, 222 422s, a
handful of deliberate 400s. Two were not.

**%00 anywhere in a URL.** An id carrying one goes to PostgreSQL, which raises
DataError because a text field cannot hold NUL, or to Path(), which raises
"embedded null byte". Neither is an HTTPException, so twelve routes answered
`Internal Server Error`, and three others leaked the driver's own message in a
503. It is refused once now, before routing, because no route wants one: every
id this API takes is a name, a hash or a uuid. Fixing the twelve found would
have left the thirteenth for later.

**A 3000-character artifact key.** It passed the containment check -- it is
inside the root, it is just unnameable -- and reached open(), which raised
OSError ENAMETOOLONG. The length limit is per path component and belongs beside
the containment check, which is the other thing a key has to satisfy before it
is used as a path.
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
    module.ARTIFACTS = tmp_path / "artifacts"
    module.ARTIFACTS.mkdir()
    monkeypatch.setattr(module, "DSN", "")
    client = TestClient(module.app, raise_server_exceptions=False)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return module, client


# The routes the probe found, one per distinct way the byte reached a raiser:
# the database, a Path(), and an artifact key.
CARRIERS = (
    "/api/jobs/%00/events",
    "/api/segmentation/preflight/%00",
    "/api/segmentation/surface/%00/vc3d",
    "/api/artifacts/%00",
    "/api/ink/maps/%00",
)


@pytest.mark.parametrize("url", CARRIERS)
def test_a_nul_in_a_path_is_a_bad_request(panel, url):
    module, client = panel
    response = client.get(url)
    assert response.status_code == 400, (
        f"{url} answered {response.status_code}: {response.text[:200]}")
    assert "NUL" in response.text


def test_a_nul_in_a_query_string_is_a_bad_request(panel):
    module, client = panel
    response = client.get("/api/jobs?state=%00")
    assert response.status_code == 400, response.text[:200]
    assert "NUL" in response.text


def test_a_url_without_one_still_gets_through(panel):
    """The middleware refuses one byte, not traffic."""
    module, client = panel
    assert client.get("/api/session").status_code == 200
    # A missing job is still the route's own 404, not the middleware's 400.
    assert client.get("/api/jobs/no-such-job/events").status_code != 400


def test_a_key_too_long_to_be_a_filename_is_refused_before_open(panel):
    module, client = panel

    for url in (f"/api/artifacts/{'x' * 3000}",
                f"/api/artifacts/fine/{'x' * 300}/also-fine"):
        response = client.get(url)
        assert response.status_code == 400, (
            f"answered {response.status_code}: {response.text[:200]}")
        assert "255" in response.text

    # Nameable components can still make an unnameable path: the whole-path
    # limit is 1024 on macOS and 4096 on Linux, so this is asked of the
    # filesystem rather than assumed. Either way it is a 400, not an OSError.
    deep = "/".join(["x" * 200] * 30)
    response = client.get(f"/api/artifacts/{deep}")
    assert response.status_code == 400, (
        f"answered {response.status_code}: {response.text[:200]}")

    # And a key that fits is refused for the ordinary reason: it is not there.
    assert client.get("/api/artifacts/a/b/c").status_code == 404


def test_a_deployment_that_cannot_provision_says_so(panel, monkeypatch):
    """Registering a host with provisioning asked for answered 500.

    Found in the control run's own log: `POST /api/hosts` with
    {"provision": true} came back 500 saying the provisioning script was
    missing. The panel image does not copy containers/, so that is what every
    deployment answers -- a property of the install, not a fault in the
    request. It reads as 503 now, and says that the registration itself did
    happen, because it did.
    """
    module, client = panel
    monkeypatch.setattr(module, "REPO", Path("/nonexistent-checkout"))

    class _Registers:
        """Enough store to get past registration, which is not what is tested."""

        def register_host(self, *arguments, **named):
            return None

    monkeypatch.setattr(module, "job_store", lambda: _Registers())

    response = client.post("/api/hosts", json={
        "host_id": "ubuntu", "ssh_target": "root@127.0.0.1", "provision": True})
    assert response.status_code != 500, response.text[:200]
    assert response.status_code == 503, response.text[:200]
    # The caller has to learn that the host is registered and provisioning is
    # not: those are two different things to do next.
    assert "registered" in response.text
