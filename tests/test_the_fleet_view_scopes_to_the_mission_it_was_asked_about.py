"""`GET /api/fleet?mission=` returned the whole fleet, whatever was asked.

fleet_status() has taken a sample set since the mission dashboard once drew
162 tasks from a different mission; the route never passed one. So the same
counts came back for a mission that does not exist, and the segmentation
control -- polling it to learn when its own tasks had settled -- was reading
another mission's 173 NO_SEED rows as its own.
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
    monkeypatch.setattr(module, "DSN", "")
    seen: list = []

    def fleet_status(samples=None):
        seen.append(samples)
        return {"available": True, "task_states": [], "surfaces": 0}

    monkeypatch.setattr(module, "fleet_status", fleet_status)
    monkeypatch.setattr(module, "mission_scrolls",
                        lambda mission: {"PHerc1203"} if mission == "m1" else set())
    monkeypatch.setattr(module, "job_store", lambda: (_ for _ in ()).throw(RuntimeError("no store")))
    client = TestClient(module.app, raise_server_exceptions=False)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return client, seen


def test_a_mission_narrows_the_counts_to_its_scrolls(panel):
    client, seen = panel
    answer = client.get("/api/fleet?mission=m1")
    assert answer.status_code == 200
    assert seen[-1] == {"PHerc1203"}
    assert answer.json()["scoped_to"] == "m1"


def test_no_mission_is_the_whole_fleet_and_says_so(panel):
    client, seen = panel
    answer = client.get("/api/fleet")
    assert seen[-1] is None
    assert answer.json()["scoped_to"] is None


def test_an_empty_mission_is_no_rows_not_the_whole_fleet(panel):
    """The distinction fleet_status draws, carried through the route: a new
    mission with nothing in it must not fall through to everybody's work."""
    client, seen = panel
    client.get("/api/fleet?mission=empty")
    assert seen[-1] == set()
