"""Asking for a deferral through the panel, not through the database.

The store has the lever; this is the only way to pull it that authenticates the
principal and writes down why. The alternative on the day was an UPDATE typed
into psql, which is the thing the whole programme forbids: no record, no
attribution, and nothing stopping the next person doing it to the evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def defer_qc_jobs(self, sample_id, *, until, reason, by):
        self.calls.append(("defer", sample_id, until, reason, by))
        return {"sample_id": sample_id, "deferred": 47, "until": until,
                "reason": reason, "by": by}

    def release_qc_jobs(self, sample_id, *, by):
        self.calls.append(("release", sample_id, by))
        return {"sample_id": sample_id, "released": 47, "by": by}


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import panel.app as panel_app
    from framework.contracts import auth

    store = _Store()
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(panel_app, "fleet_store", lambda: store)
    auth.create_user(panel_app.AUTH_ROOT, "tester", "a-long-enough-one")
    http = TestClient(panel_app.app)
    assert http.post("/api/session", json={
        "username": "tester", "password": "a-long-enough-one"}).status_code == 200
    http.store = store
    return http


def test_a_sample_can_be_deferred_with_a_reason(client) -> None:
    response = client.post("/api/segmentation/qc-jobs/defer", json={
        "sample_id": "PHerc826",
        "until": "2026-08-07T12:00:00Z",
        "reason": "the development control needs the GPUs first",
    })

    assert response.status_code == 200, response.text
    assert response.json()["deferred"] == 47
    action, sample, until, reason, by = client.store.calls[0]
    assert (action, sample) == ("defer", "PHerc826")
    assert by == "tester", "the deferral was not attributed to whoever asked"
    assert "control" in reason


def test_a_deferral_without_a_reason_is_refused(client) -> None:
    response = client.post("/api/segmentation/qc-jobs/defer", json={
        "sample_id": "PHerc826", "until": "2026-08-07T12:00:00Z", "reason": "",
    })

    assert response.status_code == 422, response.text
    assert client.store.calls == []


def test_a_deferral_without_an_end_is_refused(client) -> None:
    """An unbounded hold is a delete with better manners."""
    response = client.post("/api/segmentation/qc-jobs/defer", json={
        "sample_id": "PHerc826", "reason": "indefinitely",
    })

    assert response.status_code == 422, response.text
    assert client.store.calls == []


def test_a_sample_can_be_taken_back_up(client) -> None:
    response = client.post("/api/segmentation/qc-jobs/release", json={
        "sample_id": "PHerc826"})

    assert response.status_code == 200, response.text
    assert response.json()["released"] == 47
    assert client.store.calls[0] == ("release", "PHerc826", "tester")


def test_neither_route_answers_without_a_session(monkeypatch, tmp_path) -> None:
    from fastapi.testclient import TestClient

    import panel.app as panel_app

    store = _Store()
    monkeypatch.setattr(panel_app, "AUTH_ROOT", tmp_path / "auth")
    monkeypatch.setattr(panel_app, "AUDIT_ROOT", tmp_path / "audit")
    monkeypatch.setattr(panel_app, "fleet_store", lambda: store)
    anonymous = TestClient(panel_app.app)

    for path in ("/api/segmentation/qc-jobs/defer", "/api/segmentation/qc-jobs/release"):
        assert anonymous.post(path, json={
            "sample_id": "PHerc826", "until": "2026-08-07T12:00:00Z",
            "reason": "x"}).status_code in {401, 403}
    assert store.calls == []
