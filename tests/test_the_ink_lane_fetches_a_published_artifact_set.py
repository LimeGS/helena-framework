"""The ink lane's fetch has the same two branches, and the same blind spot.

`fetch_artifact_set` is the P4/P5/P7 counterpart to the surface adapter's
`materialize_surface`: S3, or else treat the URI as a directory. It knows about
`file://` and stops there, so an `https://` layer stack becomes a relative path
and the lane goes looking for it under the worker's current directory.

Fixing only the surface reader would move the hang one hop down the line rather
than remove it: P3 publishes a flattened sheet, P4 fetches it back, and if the
artifact store is the panel -- which writes over HTTPS today, `artifact_store`
picks `PanelArtifactStore` for exactly those URIs -- the next job stalls where
this one did. The write side has spoken HTTP(S) all along; only the read side
never learned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

BASE = "https://panel.example.invalid/artifacts/layers/p4-1"


def make_stack(root: Path) -> dict[str, bytes]:
    """The bytes a published layer stack would serve, manifest included."""
    from fleet.common import artifact_manifest  # noqa: PLC0415

    root.mkdir(parents=True)
    names = tuple(f"{index:02d}.tif" for index in range(3))
    for index, name in enumerate(names):
        (root / name).write_bytes(f"slice-{index}".encode())
    manifest = {"files": artifact_manifest(root, names)}
    (root / "ARTIFACT_SET.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return {f"{BASE}/{name}": (root / name).read_bytes()
            for name in (*names, "ARTIFACT_SET.json")}


def serve(monkeypatch, wire: dict[str, bytes]) -> list[str]:
    import ink_worker  # noqa: PLC0415

    asked: list[str] = []

    def fake_get(url: str, **kwargs) -> bytes:
        asked.append(url)
        if url not in wire:
            raise FileNotFoundError(f"404: {url}")
        return wire[url]

    monkeypatch.setattr(ink_worker, "_http_get", fake_get)
    return asked


def test_a_published_artifact_set_is_fetched_over_the_network(
    tmp_path, monkeypatch
) -> None:
    import ink_worker

    wire = make_stack(tmp_path / "published")
    asked = serve(monkeypatch, wire)
    destination = tmp_path / "fetched"

    manifest = ink_worker.fetch_artifact_set(BASE, destination)

    assert set(manifest["files"]) == {"00.tif", "01.tif", "02.tif"}
    assert set(asked) == set(wire)
    for name in manifest["files"]:
        assert (destination / name).read_bytes() == wire[f"{BASE}/{name}"]


def test_an_https_stack_is_never_read_as_a_local_path(tmp_path, monkeypatch) -> None:
    import ink_worker

    def refuse(url: str, **kwargs) -> bytes:
        raise TimeoutError("the host did not answer")

    monkeypatch.setattr(ink_worker, "_http_get", refuse)

    with pytest.raises(TimeoutError):
        ink_worker.fetch_artifact_set(BASE, tmp_path / "fetched")


def test_a_scheme_the_lane_cannot_read_is_refused_by_name(tmp_path) -> None:
    import ink_worker

    with pytest.raises(ValueError, match="ftp"):
        ink_worker.fetch_artifact_set(
            "ftp://example.invalid/layers", tmp_path / "fetched")


def test_fetched_slices_are_verified_against_the_manifest(
    tmp_path, monkeypatch
) -> None:
    """The digest check that already guarded S3 and disk must guard the wire."""
    import ink_worker

    wire = make_stack(tmp_path / "published")
    wire[f"{BASE}/01.tif"] = b"forged!"  # same length as "slice-1"
    serve(monkeypatch, wire)

    with pytest.raises(RuntimeError, match="this is not that artifact"):
        ink_worker.fetch_artifact_set(BASE, tmp_path / "fetched")


def test_a_local_artifact_set_is_still_read_from_disk(tmp_path) -> None:
    import ink_worker

    source = tmp_path / "published"
    make_stack(source)

    manifest = ink_worker.fetch_artifact_set(str(source), tmp_path / "fetched")

    assert set(manifest["files"]) == {"00.tif", "01.tif", "02.tif"}


def test_the_fetch_is_bounded_by_a_timeout(monkeypatch) -> None:
    import ink_worker

    seen: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self) -> bytes:
            return b"payload"

    def fake_urlopen(request, timeout=None):
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(ink_worker.urllib.request, "urlopen", fake_urlopen)

    assert ink_worker._http_get(f"{BASE}/00.tif") == b"payload"
    assert isinstance(seen["timeout"], (int, float))
    assert 0 < float(seen["timeout"]) <= 600
