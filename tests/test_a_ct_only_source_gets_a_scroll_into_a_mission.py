"""PHerc1667 could not be named in a mission, because nothing could register it.

scroll_has_a_source's third path required `row.get("ct_uri") and
row.get("m7_uri")`, and the only thing that ever wrote a row there --
register_snapshot(), reachable solely from the P0 freeze flow -- required both
too. PHerc1667 has a community mesh, a published surface volume and a
published ink map, all at 2.399 um, and no m7 surface prediction anywhere:
growable_scrolls() correctly refuses it forever, and the registered-snapshot
path refused it just as hard, for a reason that has nothing to do with P1 --
P4 and P5 read the mesh and the ink map through their existing raw
`segmentation` parameter and need no m7 at all.

Two things had to change together. scroll_has_a_source now accepts a
registered CT alone, and POST /api/segmentation/sources is the way to
register one for a scroll the catalog and the control cohort do not name.
Neither makes the scroll growable: P1's own gate,
scroll_has_a_growable_source, still requires m7, exercised here as the
regression that a source with both fields keeps behaving exactly as before.
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

# A scroll already in the catalogue, so refuse_scrolls_with_no_volume actually
# runs its checks instead of taking the "no catalogue at all yet" early return.
CATALOGUED = ("PHerc826",)


class FakeSourceStore:
    """The segmentation control plane's source_snapshots table, in memory."""

    def __init__(self):
        self.rows: list[dict] = []

    def register_snapshot(self, payload):
        if not payload.get("sample_id") or not payload.get("ct_uri"):
            raise ValueError("register_snapshot requires sample_id and ct_uri")
        row = dict(payload)
        row["source_snapshot_id"] = f"snap-{len(self.rows)}"
        self.rows.append(row)
        return row["source_snapshot_id"]

    def snapshots(self, samples=None):
        return [row for row in self.rows
                if samples is None or row.get("sample_id") in samples]


@pytest.fixture
def panel(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    monkeypatch.setattr(module, "DSN", "postgresql://test.invalid/panel")
    monkeypatch.setattr(module, "growable_scrolls", lambda: list(CATALOGUED))

    store = FakeSourceStore()
    monkeypatch.setattr(module, "fleet_store", lambda: store)
    monkeypatch.setattr(module, "fleet_store_read_only", lambda: store)

    client = TestClient(module.app, raise_server_exceptions=False)

    from framework.contracts import auth
    auth.create_user(module.AUTH_ROOT, "tester", "a-long-enough-one")
    assert client.post("/api/session", json={"username": "tester",
                                             "password": "a-long-enough-one"}
                       ).status_code == 200
    return module, client, store


def _mission(client, mission_id, scrolls):
    return client.post("/api/missions", json={
        "mission_id": mission_id, "name": mission_id,
        "description": "a mission created in a test",
        "scrolls": scrolls})


def test_a_scroll_with_no_source_anywhere_is_still_refused(panel):
    """The gap being closed is PHerc1667's, not every uncatalogued name's."""
    module, client, store = panel
    response = _mission(client, "qa-nothing", ["PHerc1667"])
    assert response.status_code == 400, response.text
    assert "PHerc1667" in response.text
    assert not store.rows


def test_registering_a_ct_only_source_gets_the_scroll_into_a_mission(panel):
    module, client, store = panel
    response = client.post("/api/segmentation/sources", json={
        "sample_id": "PHerc1667",
        "ct_uri": "s3://vesuvius-challenge/PHerc1667/surface-volumes/"
                  "2.399um-0.22m-78keV-volume-20251217075048.zarr/",
    })
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["sample_id"] == "PHerc1667"
    assert body["source_snapshot_id"]
    assert body["growable"] is False

    assert module.scroll_has_a_source("PHerc1667")
    # Registered, but not growable: P1 has no m7 to seed on, and this call
    # never claimed one existed.
    assert not module.scroll_has_a_growable_source("PHerc1667")

    assert _mission(client, "qa-ct-only", ["PHerc1667"]).status_code == 201


def test_a_registered_m7_makes_the_scroll_growable_too(panel):
    """Regression: supplying m7 here still reaches the stricter P1 gate."""
    module, client, store = panel
    response = client.post("/api/segmentation/sources", json={
        "sample_id": "PHerc9001",
        "ct_uri": "s3://bucket/PHerc9001/ct.zarr",
        "m7_uri": "s3://bucket/PHerc9001/m7.zarr",
        "shape_xyz": [512, 512, 512],
        "voxel_size_um": 9.362,
    })
    assert response.status_code == 201, response.text
    assert response.json()["growable"] is True
    assert module.scroll_has_a_source("PHerc9001")
    assert module.scroll_has_a_growable_source("PHerc9001")


def test_a_full_snapshot_still_satisfies_both_checks(panel):
    """Regression: nothing that required ct_uri and m7_uri together broke for
    a scroll that has both, registered the way P0's freeze flow always did.

    A synthetic sample_id, not a real one: PHerc0268 is a real scroll with its
    own entry in the frozen catalogue, and stored_scroll() would normalize it
    to the catalogue's own spelling before this fake store ever saw the name,
    which is a different (true) thing this test does not exist to check.
    """
    module, client, store = panel
    store.register_snapshot({
        "sample_id": "PHercFullSnapshot",
        "ct_uri": "s3://bucket/PHercFullSnapshot/ct.zarr",
        "m7_uri": "s3://bucket/PHercFullSnapshot/m7.zarr",
    })
    assert module.scroll_has_a_source("PHercFullSnapshot")
    assert module.scroll_has_a_growable_source("PHercFullSnapshot")
    assert _mission(client, "qa-full-snapshot", ["PHercFullSnapshot"]).status_code == 201


def test_a_malformed_uri_is_refused(panel):
    module, client, store = panel
    response = client.post("/api/segmentation/sources", json={
        "sample_id": "PHerc1667", "ct_uri": "not-a-uri-at-all"})
    assert response.status_code == 400, response.text
    assert not store.rows


def test_an_empty_uri_is_refused(panel):
    module, client, store = panel
    response = client.post("/api/segmentation/sources", json={
        "sample_id": "PHerc1667", "ct_uri": ""})
    assert response.status_code in (400, 422), response.text
    assert not store.rows


def test_a_file_scheme_uri_is_refused(panel):
    """Not http(s) or s3: every reader in this fleet (probe_uri,
    open_prediction) would fail on it silently later, so it is refused here."""
    module, client, store = panel
    response = client.post("/api/segmentation/sources", json={
        "sample_id": "PHerc1667", "ct_uri": "file:///etc/passwd"})
    assert response.status_code == 400, response.text
    assert not store.rows


def test_an_unauthenticated_caller_is_refused(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    (tmp_path / "runs").mkdir()
    import panel.app as module

    module.RUNS = tmp_path / "runs"
    module.AUTH_ROOT = tmp_path / "auth"
    module.AUDIT_ROOT = tmp_path / "audit"
    monkeypatch.setattr(module, "DSN", "postgresql://test.invalid/panel")
    store = FakeSourceStore()
    monkeypatch.setattr(module, "fleet_store", lambda: store)

    # No /api/session call: this client never signs in.
    client = TestClient(module.app, raise_server_exceptions=False)
    response = client.post("/api/segmentation/sources", json={
        "sample_id": "PHerc1667", "ct_uri": "s3://bucket/PHerc1667/ct.zarr"})
    assert response.status_code == 401, response.text
    assert not store.rows
