"""Intake looking at the URIs it writes down.

P0 froze ct_uri and m7_uri out of the catalog and never read them. Seven tasks
were generated, claimed, sent to a worker and died there on HTTP 401 -- a week
after intake, one burned attempt each, and the failure named the worker rather
than the source. The check is a HEAD.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.generator import probe_uri  # noqa: E402


def test_a_zarr_is_probed_through_its_metadata(monkeypatch):
    """A HEAD on the directory itself answers 404 on a store that reads fine, so
    a naive probe would refuse every volume in the catalog."""
    seen: list[str] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(request, timeout=None):
        seen.append(request.full_url)
        return Response()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = probe_uri("https://example.invalid/scroll/volume.zarr")
    assert seen == ["https://example.invalid/scroll/volume.zarr/.zattrs"]
    assert result["reachable"] is True


def test_an_unauthorized_source_is_reported_rather_than_raised(monkeypatch):
    """Intake must still finish: the point is to record which sources cannot be
    read, not to make one bad row stop the catalog."""
    import urllib.error
    import urllib.request

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = probe_uri("https://example.invalid/scroll/m7.zarr")
    assert result["reachable"] is False
    assert result["status"] == 401


def test_what_the_worker_reads_itself_is_not_guessed_at():
    """s3:// and fixture:// are fetched by the worker's own client with its own
    credentials. An unauthenticated probe here would report a false negative and
    refuse to queue perfectly good work."""
    assert probe_uri("s3://bucket/prefix/volume.zarr")["checked"] is False
    assert probe_uri("fixture://ct/whatever")["checked"] is False
