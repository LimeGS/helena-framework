"""/api/missions took any string at all as a scroll.

"PHercInventado" and "PHerc9999" both created missions, and so did PHerc0139 on
a control plane that had never registered its source. Nothing said so until P1,
which refused every task with "resolves to no volume" -- so a mission looked
frozen and correct for as long as nobody tried to grow in it. The full-pipeline
mission on the rented deployment was exactly this: frozen on PHerc0139, and
every run against it came back with nothing to read.

The check is the predicate P1 already uses, so it does not impose an order on
the work. scroll_has_a_source passes anything in the frozen eligible catalog,
and a catalogued scroll can be named in a mission long before P0 has frozen
anything for it -- which is the normal way round, since P0 runs inside a
mission. What is refused is a name with no address in the catalog and none on
this control plane: the case no run will ever change.
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

CATALOGUED = ("PHerc826", "PHerc1203")


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
    # The catalogue and the control plane, stated rather than fetched: this is
    # about what the route does with the answer, not about how it is obtained.
    monkeypatch.setattr(module, "growable_scrolls", lambda: list(CATALOGUED))
    monkeypatch.setattr(module, "scroll_has_a_source",
                        lambda scroll: scroll in CATALOGUED)
    client = TestClient(module.app, raise_server_exceptions=False)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return module, client


def _mission(client, mission_id, scrolls):
    return client.post("/api/missions", json={
        "mission_id": mission_id, "name": mission_id,
        "description": "a mission created in a test",
        "scrolls": scrolls})


@pytest.mark.parametrize("scroll", ["PHercInventado", "PHerc9999", "PHerc0139"])
def test_a_name_with_no_volume_is_refused(panel, scroll):
    module, client = panel
    response = _mission(client, f"qa-{scroll.lower()}", [scroll])

    assert response.status_code == 400, (
        f"{scroll} was accepted: {response.text[:200]}")
    assert scroll in response.text
    # The refusal names what can be worked on, so the caller can fix the call
    # rather than guess.
    assert "PHerc826" in response.text


def test_a_catalogued_scroll_is_still_accepted(panel):
    """P0 runs inside a mission, so the mission has to come first."""
    module, client = panel
    assert _mission(client, "qa-good", ["PHerc826"]).status_code == 201


def test_one_bad_name_refuses_the_whole_mission(panel):
    """A mission is frozen as a set; half of one is not what was asked for."""
    module, client = panel
    response = _mission(client, "qa-mixed", ["PHerc826", "PHercInventado"])

    assert response.status_code == 400, response.text[:200]
    assert "PHercInventado" in response.text
    assert not (module.RUNS / "qa-mixed").exists(), (
        "the mission directory was created despite the refusal")


def test_an_amendment_cannot_widen_onto_nothing(panel):
    """The other way a scroll enters a mission, reached later and just as dead."""
    module, client = panel
    assert _mission(client, "qa-amend", ["PHerc826"]).status_code == 201

    response = client.post("/api/missions/qa-amend/amend", json={
        "add": ["PHercInventado"], "reason": "widening onto a name"})

    assert response.status_code == 400, response.text[:200]
    assert "PHercInventado" in response.text


def test_an_amendment_onto_a_catalogued_scroll_still_works(panel):
    module, client = panel
    assert _mission(client, "qa-amend-ok", ["PHerc826"]).status_code == 201

    response = client.post("/api/missions/qa-amend-ok/amend", json={
        "add": ["PHerc1203"], "reason": "a second scroll from the catalogue"})

    assert response.status_code != 400, response.text[:200]


def test_the_bucket_spelling_of_a_catalogued_scroll_is_accepted(panel,
                                                                monkeypatch):
    """PHerc0826 and PHerc826 are one scroll, and a mission is named the first
    way: that is what P0 read out of the bucket. Checking the raw name refused
    the very scroll the catalogue lists."""
    module, client = panel
    monkeypatch.setattr(module, "stored_scroll",
                        lambda name: "PHerc826" if name == "PHerc0826" else name)

    assert _mission(client, "qa-bucket", ["PHerc0826"]).status_code == 201


def test_no_catalogue_refuses_nothing(panel, monkeypatch):
    """A fresh deployment has no catalogue until the first refresh lands.

    Refusing every mission until then would leave a clean install unable to
    start any work at all -- and an absent catalogue is not evidence that a
    scroll has no volume.
    """
    module, client = panel
    monkeypatch.setattr(module, "growable_scrolls", list)
    monkeypatch.setattr(module, "scroll_has_a_source", lambda scroll: False)

    assert _mission(client, "qa-no-catalogue", ["PHercAnything"]).status_code == 201
