"""A worker on another machine has to reach the panel over the network.

Object storage is optional. Where there is no bucket, the panel host keeps
everything in a volume of its own -- and a worker sharing that host writes into
it directly. This is the other case, and it used to fall through a gap:

    refuse_stranded_artifacts accepted an http:// artifact root.
    open_artifact_store did not implement one.

So the string went to LocalArtifactStore and became a directory literally named
`https:/panel.../surfaces` on the worker's own disk. Surfaces were written,
recorded in the control plane with that path, and invisible to every phase that
went looking. Nothing failed; it just quietly did not work.

These tests drive the real panel through a test client, with a real machine
token, and check the artifact arrives, comes back byte-identical, and that the
door is shut to everyone else.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

fastapi_testclient = pytest.importorskip("fastapi.testclient")


def _manifest(directory: Path) -> dict:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.name == "ARTIFACT_SET.json" or not path.is_file():
            continue
        files[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "size_bytes": path.stat().st_size}
    payload = json.dumps(files, sort_keys=True).encode()
    return {"files": files, "artifact_sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture()
def panel(tmp_path, monkeypatch):
    """The real panel, with its state somewhere disposable."""
    monkeypatch.setenv("CX_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("CX_AUTH", str(tmp_path / "auth"))
    monkeypatch.setenv("CX_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("CX_CACHE", str(tmp_path / "cache"))
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "runs").mkdir()
    import panel.app as app_module
    app_module = importlib.reload(app_module)
    from framework.contracts import auth

    token = auth.create_machine_token(app_module.AUTH_ROOT, "gpu-1-segment", by="test")
    # An account, so the panel is not in its bootstrap state.
    auth.create_user(app_module.AUTH_ROOT, "someone", "a-long-enough-password", by="test")
    client = fastapi_testclient.TestClient(app_module.app)
    return client, token, tmp_path / "artifacts"


@pytest.fixture()
def store(panel):
    from fleet.artifact_store import PanelArtifactStore

    client, token, _ = panel
    return PanelArtifactStore("https://panel.invalid/helena", token=token, session=client)


@pytest.fixture()
def surface(tmp_path):
    source = tmp_path / "grown"
    source.mkdir()
    (source / "surface.tif").write_bytes(b"not really a tif, but bytes are bytes")
    (source / "meta.json").write_text('{"winding": 3}')
    return source


def test_open_artifact_store_returns_the_panel_for_an_http_root(monkeypatch):
    """The gap itself. An http:// root used to become a local directory."""
    from fleet.artifact_store import LocalArtifactStore, PanelArtifactStore, open_artifact_store

    monkeypatch.setenv("HELENA_PANEL_TOKEN", "helena-machine-whatever")
    opened = open_artifact_store("https://panel.example/helena")
    assert isinstance(opened, PanelArtifactStore), (
        "an http artifact root still resolves to a local directory, which is "
        "how surfaces got written to a folder named 'https:/...' on a worker"
    )
    assert not isinstance(opened, LocalArtifactStore)


def test_publishing_without_a_token_refuses_rather_than_writing_somewhere(monkeypatch):
    """A worker with no credential must stop, not fall back to its own disk."""
    from fleet.artifact_store import open_artifact_store

    monkeypatch.delenv("HELENA_PANEL_TOKEN", raising=False)
    with pytest.raises(ValueError, match="machine token"):
        open_artifact_store("https://panel.example/helena")


def test_a_staged_surface_reaches_the_panel(store, surface, panel):
    _, _, artifacts = panel
    manifest = _manifest(surface)
    staged = store.stage(surface, "attempt-1", manifest)

    assert staged["backend"] == "panel"
    landed = artifacts / "helena/staging/attempt-1"
    assert landed.is_dir(), "the panel did not write the artifact anywhere"
    assert (landed / "surface.tif").read_bytes() == (surface / "surface.tif").read_bytes()
    # The manifest travels with the bytes, so a later phase can verify them.
    assert json.loads((landed / "ARTIFACT_SET.json").read_text())["files"] == manifest["files"]


def test_only_what_the_manifest_names_is_published(store, surface, panel):
    """A staging directory also holds logs and scratch. Publishing those makes
    the artifact's contents depend on what happened to be lying around."""
    _, _, artifacts = panel
    (surface / "worker.log").write_text("chatter that is not part of the surface")
    manifest = _manifest(surface)
    manifest["files"].pop("worker.log", None)

    store.stage(surface, "attempt-2", manifest)
    landed = artifacts / "helena/staging/attempt-2"
    assert not (landed / "worker.log").exists(), "scratch was published with the surface"


def test_promotion_copies_on_the_panel_rather_than_sending_it_again(store, surface, panel):
    """The bytes are already there. S3's backend does a server-side copy for the
    same reason: uploading twice makes publishing cost twice as much."""
    _, _, artifacts = panel
    manifest = _manifest(surface)
    staged = store.stage(surface, "attempt-3", manifest)
    promoted = store.promote(staged, "PHerc0826", "surface-7", manifest)

    final = artifacts / "helena/surfaces/PHerc0826/surface-7"
    assert final.is_dir(), "promotion did not produce a final surface"
    assert (final / "surface.tif").read_bytes() == (surface / "surface.tif").read_bytes()
    assert promoted["artifact_uri"].endswith("/helena/surfaces/PHerc0826/surface-7")


def test_promotion_can_be_retried(store, surface, panel):
    """Publication is retried after a lease expires. A retry that refuses is a
    surface that never gets its final name."""
    manifest = _manifest(surface)
    staged = store.stage(surface, "attempt-4", manifest)
    first = store.promote(staged, "PHerc0826", "surface-8", manifest)
    again = store.promote(staged, "PHerc0826", "surface-8", manifest)
    assert first["artifact_uri"] == again["artifact_uri"]


def test_a_probe_comes_back_byte_identical(store, surface, tmp_path):
    """Materialising verifies the manifest, so a truncated round trip is caught
    here rather than by VC3D reading a short file."""
    manifest = _manifest(surface)
    published = store.publish_probe(
        surface, "PHerc0826", "run-1", "trial-1", "set-1", manifest)

    back = store.materialize_probe(published["artifact_uri"], tmp_path / "back", manifest)
    assert (back / "surface.tif").read_bytes() == (surface / "surface.tif").read_bytes()


def test_a_probe_can_be_deleted_and_says_so(store, surface):
    manifest = _manifest(surface)
    published = store.publish_probe(
        surface, "PHerc0826", "run-2", "trial-2", "set-2", manifest)

    first = store.delete_probe(published["artifact_uri"], manifest)
    assert first["deleted"] is True
    second = store.delete_probe(published["artifact_uri"], manifest)
    assert second["already_absent"] is True


# ------------------------------------------------------------------ the door --

def test_without_a_token_the_artifact_endpoints_are_shut(panel, surface):
    client, _, _ = panel
    assert client.put("/api/artifacts/helena/staging/x", content=b"x").status_code == 401
    assert client.get("/api/artifacts/helena/staging/x").status_code == 401
    assert client.delete("/api/artifacts/helena/staging/x").status_code == 401


def test_a_revoked_token_stops_working(panel):
    client, token, _ = panel
    import panel.app as app_module
    from framework.contracts import auth

    assert client.head("/api/artifacts/helena/nothing",
                       headers={"Authorization": f"Bearer {token}"}).status_code == 404
    auth.revoke_machine_token(app_module.AUTH_ROOT, "gpu-1-segment")
    assert client.head("/api/artifacts/helena/nothing",
                       headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_a_machine_token_opens_the_artifact_endpoints_and_nothing_else(panel):
    """A token copied off a worker host should not be able to drive the panel.

    It exists so a worker can publish a surface. Queueing GPU work or reading
    somebody's missions is not part of that.
    """
    client, token, _ = panel
    headers = {"Authorization": f"Bearer {token}"}
    for path in ("/api/state", "/api/fleet", "/api/runs", "/api/audit"):
        response = client.get(path, headers=headers)
        assert response.status_code == 401, (
            f"a machine token reached {path}, which is not an artifact endpoint"
        )


def test_a_key_cannot_escape_the_artifact_root(panel):
    """Tested on the resolver, not through a URL.

    Sending `../../etc/passwd` in a path proves nothing: the HTTP client
    normalises it away before it leaves, and the request that arrives is for a
    different path entirely -- it came back 200 from the SPA catch-all, which
    looked like a traversal succeeding and was nothing of the sort.
    """
    import panel.app as app_module

    for escape in ("../../etc/passwd", "helena/../../outside", "/etc/passwd"):
        with pytest.raises(Exception) as raised:
            app_module._artifact_path(escape)
        assert "outside the artifact root" in str(raised.value), (
            f"{escape!r} was not refused; a worker token could write anywhere "
            "on the panel host"
        )


def test_a_symlink_cannot_lead_out_of_the_artifact_root(panel):
    """The check is on the resolved path for this reason.

    A key with no suspicious component at all still escapes if a directory
    along the way is a symlink -- and a worker that can write artifacts can
    create one.
    """
    import panel.app as app_module

    _, _, artifacts = panel
    (artifacts / "helena").mkdir(parents=True, exist_ok=True)
    (artifacts / "helena" / "escape").symlink_to("/etc")
    with pytest.raises(Exception, match="outside the artifact root"):
        app_module._artifact_path("helena/escape/passwd")


def test_an_encoded_traversal_is_refused_over_http(panel):
    """The one form a client does not normalise away."""
    client, token, _ = panel
    headers = {"Authorization": f"Bearer {token}"}
    response = client.put("/api/artifacts/helena/%2e%2e%2f%2e%2e%2fescaped",
                          content=b"x", headers=headers)
    assert response.status_code != 200, "an encoded traversal was accepted"
