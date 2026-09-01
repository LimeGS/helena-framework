"""A surface published over HTTPS is fetched, not guessed at.

Every surface this lane has ever handled came from `s3://` or from a path on
disk, so `materialize_surface` grew two branches: S3, and *everything else is a
local directory*. An `https://` URI takes the second one. `Path()` collapses the
double slash, `.resolve()` anchors the remainder to the worker's current
directory, and the lane goes looking for `<cwd>/https:/host/key`. What it
reports, if it reports anything, is a missing file whose name is not the URI it
was given.

That matters beyond tidiness. Reading a surface somebody else published is the
whole of Villa's second requirement -- public input surfaces -- and it is how
the 600-scroll campaign will consume community segmentations it did not grow.
A scheme the lane cannot read must say so by name; silently degrading to a
relative path turns a URI the lane doesn't support into a filesystem accident
that depends on where the worker happened to be standing.

The module already knows how to do this: `fetch_zarr_metadata` speaks HTTPS with
a timeout and a User-Agent ten lines further down. The surface reader just never
used it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STAGE))

from tests.test_surface_qc_adapter import load_adapter, make_surface  # noqa: E402

BASE = "https://dl.example.invalid/community/w025.tifxyz"


def published(adapter, root: Path) -> tuple[dict[str, bytes], str]:
    """The bytes a public host would serve for one artifact set."""

    source, digest = make_surface(root)
    names = (*adapter.REQUIRED_SURFACE_FILES, "ARTIFACT_SET.json")
    return {f"{BASE}/{name}": (source / name).read_bytes() for name in names}, digest


def serve(adapter, monkeypatch, wire: dict[str, bytes]) -> list[str]:
    """Stand in for the network and record what was asked for."""

    asked: list[str] = []

    def fake_get(url: str, **kwargs) -> bytes:
        asked.append(url)
        if url not in wire:
            raise FileNotFoundError(f"404: {url}")
        return wire[url]

    monkeypatch.setattr(adapter, "_http_get", fake_get)
    return asked


def test_an_https_surface_is_fetched_over_the_network(tmp_path, monkeypatch) -> None:
    adapter = load_adapter()
    wire, digest = published(adapter, tmp_path / "source")
    asked = serve(adapter, monkeypatch, wire)
    destination = tmp_path / "materialized"

    manifest = adapter.materialize_surface(BASE, digest, destination)

    assert manifest["artifact_sha256"] == digest
    assert set(asked) == set(wire)
    for name in adapter.REQUIRED_SURFACE_FILES:
        assert (destination / name).read_bytes() == wire[f"{BASE}/{name}"]


def test_an_https_surface_is_never_read_as_a_local_path(tmp_path, monkeypatch) -> None:
    """The regression, stated as the symptom it produced.

    With the fetch stubbed to fail, the failure that reaches the caller must be
    the fetch's -- not a `FileNotFoundError` naming a mangled path under the
    worker's current directory.
    """
    adapter = load_adapter()

    def refuse(url: str, **kwargs) -> bytes:
        raise TimeoutError("the host did not answer")

    monkeypatch.setattr(adapter, "_http_get", refuse)

    with pytest.raises(TimeoutError):
        adapter.materialize_surface(BASE, "0" * 64, tmp_path / "materialized")


def test_a_scheme_the_lane_cannot_read_is_refused_by_name(tmp_path) -> None:
    """Not every URI is fetchable, and the unfetchable ones must say which."""
    adapter = load_adapter()

    with pytest.raises(ValueError, match="ftp"):
        adapter.materialize_surface(
            "ftp://example.invalid/w025.tifxyz", "0" * 64, tmp_path / "materialized")


def test_a_file_uri_reads_the_path_it_names(tmp_path) -> None:
    """`file://` was in the same silent bucket: it worked only because a path
    with a scheme on the front happened not to exist, which is not working."""
    adapter = load_adapter()
    source, digest = make_surface(tmp_path / "source")

    manifest = adapter.materialize_surface(
        source.as_uri(), digest, tmp_path / "materialized")

    assert manifest["artifact_sha256"] == digest


def test_a_plain_path_is_still_a_plain_path(tmp_path) -> None:
    """The branch that always worked keeps working."""
    adapter = load_adapter()
    source, digest = make_surface(tmp_path / "source")

    manifest = adapter.materialize_surface(
        str(source), digest, tmp_path / "materialized")

    assert manifest["artifact_sha256"] == digest


def test_fetched_bytes_are_verified_like_any_other_surface(tmp_path, monkeypatch) -> None:
    """A public host is not a trusted one. The artifact set is checked against
    the manifest, and the manifest against the catalogue digest, exactly as the
    S3 and local branches do."""
    adapter = load_adapter()
    wire, digest = published(adapter, tmp_path / "source")
    # Same length as the real thing, so this reaches the hash check rather than
    # stopping at the size one. Bytes swapped without a size change is the
    # substitution a size check cannot see.
    wire[f"{BASE}/x.tif"] = b"forged"
    serve(adapter, monkeypatch, wire)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        adapter.materialize_surface(BASE, digest, tmp_path / "materialized")


def test_a_surface_that_is_not_the_locked_one_is_refused(tmp_path, monkeypatch) -> None:
    adapter = load_adapter()
    wire, _digest = published(adapter, tmp_path / "source")
    serve(adapter, monkeypatch, wire)

    with pytest.raises(RuntimeError, match="differs from the catalogue"):
        adapter.materialize_surface(BASE, "0" * 64, tmp_path / "materialized")


def test_the_fetch_is_bounded_by_a_timeout(monkeypatch) -> None:
    """The defect this lane actually exhibited was a hang, not an error. A read
    with no deadline is how a job burns its whole lease and dies with nothing to
    say."""
    adapter = load_adapter()
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

    monkeypatch.setattr(adapter.urllib.request, "urlopen", fake_urlopen)

    assert adapter._http_get(f"{BASE}/x.tif") == b"payload"
    assert isinstance(seen["timeout"], (int, float))
    assert 0 < float(seen["timeout"]) <= 600
