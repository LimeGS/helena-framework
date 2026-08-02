"""Every endpoint is reachable, gated, and survives a missing control plane.

Three properties, swept across all of them rather than asserted one at a time,
because the failures this catches are failures of absence: a route that was
renamed and now 404s for a page still calling the old name, an endpoint that
forgot the session gate, and a handler that raises instead of degrading when the
database is not there.

That last one is the reason this suite exists. The panel is expected to render
without a control plane -- most handlers already say so, with `available: false`
and a reason -- and the ones that do not have no symptom until PostgreSQL is
down, which is exactly when nobody wants to discover them. Running the whole
surface with no CX_DB is the cheapest way to find them, and it needs no
deployment, no GPU and no data.

This is a contract sweep, not an end-to-end test. It proves an endpoint answers;
tests/e2e proves it answers correctly against a real deployment.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "panel/app.py").read_text()

# Pinned, not read. Reading OPEN_PATHS out of the source made this test agree with
# whatever the source said, so adding /api/fleet to the exemption list passed
# unchanged -- the one mutation that matters here, because that is how a session
# gate erodes: an exemption added for a debug route and never removed.
#
# Pinned means adding an exemption is a deliberate act that has to be argued for
# in a diff to this file.
EXPECTED_OPEN = {"/api/session", "/api/session/bootstrap", "/login"}
PUBLIC = set(EXPECTED_OPEN)
# The SPA itself. A browser has to be able to load the page it signs in from, so
# the catch-all and the root serve HTML to anyone; the API behind them does not.
PUBLIC |= {"/", "/{full_path:path}"}


def test_the_public_path_set_has_not_grown() -> None:
    """The exemption list itself, before anything else is asserted about it."""
    declared = set(re.search(r"OPEN_PATHS = frozenset\(\{([^}]+)\}", APP_SOURCE)
                   .group(1).replace('"', "").replace(" ", "").split(","))
    assert declared == EXPECTED_OPEN, (
        f"the set of paths outside the session gate changed: "
        f"added {sorted(declared - EXPECTED_OPEN)}, "
        f"removed {sorted(EXPECTED_OPEN - declared)}. "
        "If that is intended, say so here."
    )


def _routes() -> list[tuple[str, str]]:
    """Every declared method and path, read from the source.

    From the source rather than from app.routes so the list survives an import
    failure: if app.py cannot be imported at all, this file should fail with that
    error and not with an empty parametrisation that silently tests nothing.
    """
    found = re.findall(r'@app\.(get|post|put|delete)\("([^"]+)"', APP_SOURCE)
    assert len(found) > 50, f"only found {len(found)} routes; the regex has drifted"
    return [(method.upper(), path) for method, path in found]


def _fill(path: str) -> str:
    """A concrete URL for a templated one.

    The values are deliberately implausible. We are asserting that a handler
    refuses or degrades, never that it finds something, so a real id would make
    the test depend on the data that happens to be in the checkout.
    """
    return re.sub(r"\{[^}]+\}", "test-value-that-does-not-exist", path)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """The panel with no control plane and one account.

    CX_DB is cleared rather than pointed somewhere: "no database" is the state
    being tested. AUTH_ROOT and the state directories move to a temporary path so
    the sweep cannot read or write the developer's real accounts.
    """
    state = tmp_path_factory.mktemp("panel-state")
    for name in ("CX_DB", "CX_DSN", "FLEET_DB"):
        os.environ.pop(name, None)
    os.environ["CX_STATE"] = str(state)
    os.environ["CX_AUTH_ROOT"] = str(state / "auth")
    os.environ["CX_CACHE"] = str(state / "cache")

    sys.path.insert(0, str(ROOT / "panel"))
    import app  # noqa: PLC0415

    from fastapi.testclient import TestClient  # noqa: PLC0415

    app.AUTH_ROOT = state / "auth"
    app.AUTH_ROOT.mkdir(parents=True, exist_ok=True)
    app.DSN = ""

    from framework.contracts import auth as auth_contract  # noqa: PLC0415

    auth_contract.create_user(app.AUTH_ROOT, "sweep", "sweep-password-1234")
    token = auth_contract.login(app.AUTH_ROOT, "sweep", "sweep-password-1234")

    session = TestClient(app.app, raise_server_exceptions=False)
    session.cookies.set(auth_contract.COOKIE, token)
    return session


@pytest.mark.parametrize(("method", "path"), _routes(), ids=lambda v: str(v))
def test_the_endpoint_is_reachable_and_does_not_crash(client, method, path):
    """No 404 for a declared route, and no 500 without a database.

    404 here means the route table and this file disagree, which cannot happen
    while both read the same source -- so a 404 is a genuine routing bug, such as
    a path parameter the handler cannot parse.

    500 means the handler raised. Every one of these is expected either to answer,
    or to say it cannot and why: 409 for "there is no control plane", 404 for
    "no such surface", 400 for a body it will not take. An uncaught exception is
    the one answer that tells an operator nothing.
    """
    # The two SPA routes are registered inside `if DIST.exists()`, and
    # panel/web/dist is a build output that is not in the repository. Reading
    # the route list from the source -- which this file does deliberately, so an
    # import failure cannot empty it -- therefore names two routes that a clean
    # checkout genuinely does not serve. They are asserted wherever the frontend
    # has been built, which is every deployment and the image itself.
    if path in ("/", "/{full_path:path}") and not (ROOT / "panel/web/dist").exists():
        pytest.skip("the frontend is not built, so the SPA routes are not mounted")

    response = client.request(method, _fill(path))

    # 404 is only a bug for a path with no parameters. A templated one was filled
    # with an id that does not exist, and "no such mission" is the right answer.
    if "{" not in path:
        assert response.status_code != 404, f"{method} {path} is declared and not routed"

    # A 5xx the handler chose is fine and often correct: /api/jobs answers 503
    # with "CX_DB is not set: the command plane needs the fleet database", which
    # is exactly what an operator needs to read. What must not happen is an
    # uncaught exception, which FastAPI renders as the bare string "Internal
    # Server Error" with nothing to act on.
    if response.status_code >= 500:
        assert response.headers.get("content-type", "").startswith("application/json"), (
            f"{method} {path} raised rather than refusing: {response.status_code} "
            f"{response.text[:200]}"
        )
        assert "detail" in response.json(), (
            f"{method} {path} returned {response.status_code} with no detail"
        )


@pytest.mark.parametrize(("method", "path"), _routes(), ids=lambda v: str(v))
def test_the_endpoint_refuses_a_client_with_no_session(method, path, tmp_path):
    """The gate is middleware, so this is really one assertion swept wide.

    Worth sweeping anyway: middleware that exempts by path prefix is exactly the
    kind of thing that grows an exemption for a debug route and keeps it.
    """
    if path in PUBLIC:
        pytest.skip("deliberately public")

    sys.path.insert(0, str(ROOT / "panel"))
    import app  # noqa: PLC0415

    from fastapi.testclient import TestClient  # noqa: PLC0415

    anonymous = TestClient(app.app, raise_server_exceptions=False)
    response = anonymous.request(method, _fill(path))
    assert response.status_code == 401, (
        f"{method} {path} answered {response.status_code} to a client with no "
        "session"
    )
