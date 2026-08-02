"""The panel's first upload endpoint, and the three things it must not do.

Accepting a file over HTTP is new here, and every property tested below fails
silently if it regresses: a traversal writes somewhere unintended and still
returns 201, a missing size cap fills a disk before anything parses, and an
.npz read with allow_pickle is arbitrary code execution that looks exactly like
a successful upload.

The scoring path has the same shape: it hands a caller-supplied path to a
subprocess that reads it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The panel's dependencies live in panel/.venv, not in the interpreter that runs
# the framework suite. Skipping keeps `pytest tests/` green everywhere and lets
# this file run where the panel actually runs.
pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REFERENCE_STRIPS = ROOT / "framework" / "vendored" / "reference-strips"


def make_strip(path: Path, *, scroll: str = "TEST", segment: str = "seg-01") -> Path:
    """Two concentric rings: the smallest thing that is a strip."""
    if str(REFERENCE_STRIPS) not in sys.path:
        sys.path.insert(0, str(REFERENCE_STRIPS))
    import strip_format

    wraps = {}
    for index, radius in enumerate((1000.0, 1150.0)):
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        wraps[index] = np.stack(
            [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1
        ).astype(np.float32)
    strip_format.save_strip(
        path, wraps=wraps,
        pitch_um={"median": 150.0, "p10": 148.0, "p90": 152.0},
        meta={"scroll": scroll, "segment_id": segment, "window": "0,64,0,2",
              "voxel_size_um": 1.0, "tier": "medium", "source_checksum": "synthetic"},
    )
    return path


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CX_STRIPS", str(tmp_path / "strips"))
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.STRIPS = tmp_path / "strips"
    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    return module


@pytest.fixture
def anonymous(app_module):
    """A client with no session, which is what the network is."""
    from fastapi.testclient import TestClient

    return TestClient(app_module.app)


@pytest.fixture
def client(app_module, anonymous):
    """Signed in, through the real login rather than around it.

    Bypassing the gate in tests would leave the gate itself unexercised, and it
    is the thing standing between the network and a panel that queues GPU work.
    """
    from framework.contracts import auth

    auth.create_user(app_module.AUTH_ROOT, "tester", "a-long-enough-one")
    response = anonymous.post("/api/session",
                              json={"username": "tester", "password": "a-long-enough-one"})
    assert response.status_code == 200, response.text
    return anonymous


def test_every_strip_route_is_closed_without_a_session(anonymous, tmp_path):
    """The gate covers these, and it covers them by default rather than by a
    list somebody has to remember to add to."""
    body = make_strip(tmp_path / "s.npz").read_bytes()
    assert anonymous.get("/api/strips").status_code == 401
    assert anonymous.post("/api/strips", content=body).status_code == 401
    assert anonymous.post("/api/strips/anything/qualify").status_code == 401
    assert anonymous.post("/api/strips/anything/score",
                          json={"pred_path": "x", "mode": "mesh"}).status_code == 401
    # Nothing was written on the way to being refused.
    assert not list((tmp_path / "strips").glob("*")) if (tmp_path / "strips").exists() else True


def test_a_strip_round_trips_and_is_identified_by_its_own_metadata(client, tmp_path):
    body = make_strip(tmp_path / "anything.npz", scroll="PHerc0139", segment="s-7").read_bytes()
    response = client.post("/api/strips", content=body)
    assert response.status_code == 201, response.text
    described = response.json()
    assert described["scroll"] == "PHerc0139"
    assert described["wraps"] == 2
    # The id comes from the metadata, never from what the client called the file.
    assert "anything" not in described["strip_id"]
    assert "PHerc0139" in described["strip_id"]
    assert described["qualified"] is False


@pytest.mark.parametrize("name", ["../escape.npz", "../../etc/passwd", "a/b.npz"])
def test_the_client_cannot_choose_where_the_file_lands(client, tmp_path, name):
    """The endpoint takes a raw body, so there is no client filename at all --
    but the id derived from metadata must not escape either."""
    body = make_strip(tmp_path / "s.npz", scroll=name, segment=name).read_bytes()
    response = client.post("/api/strips", content=body)
    assert response.status_code == 201, response.text
    written = Path(response.json()["path"]).resolve()
    assert written.parent == (tmp_path / "strips").resolve()


def test_a_body_that_is_not_a_strip_is_refused_and_leaves_nothing(client, tmp_path):
    response = client.post("/api/strips", content=b"this is not an npz at all")
    assert response.status_code == 400
    assert not list((tmp_path / "strips").glob("*.npz"))
    # No staging file survives the refusal either.
    assert not list((tmp_path / "strips").glob(".incoming-*"))


def test_an_empty_body_is_refused(client):
    assert client.post("/api/strips", content=b"").status_code == 400


def test_an_oversized_body_is_refused_before_it_is_parsed(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_STRIP_BYTES", 1024)
    response = client.post("/api/strips", content=b"x" * 4096)
    assert response.status_code == 413


def test_scoring_refuses_a_path_outside_the_runs_root(client, tmp_path):
    make_strip(tmp_path / "s.npz")
    uploaded = client.post("/api/strips", content=(tmp_path / "s.npz").read_bytes())
    strip_id = uploaded.json()["strip_id"]
    for outside in ("/etc/passwd", "../../../../etc/passwd", str(tmp_path / "strips")):
        response = client.post(f"/api/strips/{strip_id}/score",
                               json={"pred_path": outside, "mode": "mesh"})
        assert response.status_code in (400, 404), f"{outside} -> {response.status_code}"


def test_an_unknown_strip_id_is_a_404_not_a_traversal(client):
    """Never a 2xx, and never a file outside the strip directory.

    405 counts as passing: a dot segment is normalised before routing, so the
    request reaches a path that has no POST handler and the id never gets near
    the filesystem. Refusing earlier than the handler is refusing.
    """
    for bad in ("../secrets", "..", ".", "%2e%2e%2fsecrets"):
        response = client.post(f"/api/strips/{bad}/score",
                               json={"pred_path": "x", "mode": "mesh"})
        assert response.status_code in (400, 404, 405), f"{bad} -> {response.status_code}"


def test_strips_are_read_without_pickle():
    """A strip arrives over HTTP; np.load with pickle enabled would execute it.

    Checked in the vendored loader rather than mocked, because that is the
    function the endpoint calls and the property belongs to it.
    """
    source = (REFERENCE_STRIPS / "strip_format.py").read_text()
    assert "allow_pickle=False" in source
    assert "np.load(path, allow_pickle=False)" in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
