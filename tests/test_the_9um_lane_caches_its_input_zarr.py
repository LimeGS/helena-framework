"""Fourteen checkpoints, one PHerc0139 volume, restreamed from S3 in full on
every one of them -- ~1.4 GB at ~7.5 MB/s, twice per checkpoint under
`direction: both`. This mirrors the store once, locally, and every job after
the first reads from disk instead.

Off unless HELENA_INK_9UM_ZARR_CACHE is set: a cache that cannot be reached
must fall back to the URL exactly as before, which is what most of these
tests hold. None of them touch a real network -- boto3's client is replaced
with a fake that records calls and returns canned responses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/scripts"))

import run_ink_9um as lane  # noqa: E402

URL = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
      "PHerc0139/segments/20260112000000-w043_2026011217/surface-volumes/"
      "9.362um-1.2m-113keV-volume-20250728140407.zarr")


def test_zarr_cache_root_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HELENA_INK_9UM_ZARR_CACHE", raising=False)
    assert lane.zarr_cache_root() is None


def test_zarr_cache_root_reads_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("HELENA_INK_9UM_ZARR_CACHE", str(tmp_path))
    assert lane.zarr_cache_root() == tmp_path


def test_parsing_the_virtual_hosted_style_url():
    bucket, prefix = lane.parse_s3_https_url(URL)
    assert bucket == "vesuvius-challenge-open-data"
    assert prefix == ("PHerc0139/segments/20260112000000-w043_2026011217/"
                      "surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr")


def test_a_local_path_or_unrelated_host_is_not_parsed():
    assert lane.parse_s3_https_url("/local/surface-volume.zarr") is None
    assert lane.parse_s3_https_url("https://example.com/not-s3/data.zarr") is None


def _fake_client(*, objects: list[str], etag: str | None = "abc123",
                 metadata_name: str = "zarr.json"):
    client = MagicMock()

    def head_object(Bucket, Key):  # noqa: N803 -- boto3's own casing
        if Key.endswith(metadata_name) and etag is not None:
            return {"ETag": f'"{etag}"'}
        raise client._ClientError("no such key")

    client.head_object.side_effect = head_object
    client._ClientError = type("NoSuchKey", (Exception,), {})

    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in objects]}]
    client.get_paginator.return_value = paginator

    written: dict[str, bytes] = {}

    def download_file(bucket, key, destination):
        Path(destination).write_bytes(b"chunk-bytes-for-" + key.encode())
        written[key] = Path(destination).read_bytes()

    client.download_file.side_effect = download_file
    client._written = written
    return client


def test_a_store_with_no_objects_under_its_prefix_is_not_cached(monkeypatch, tmp_path):
    client = _fake_client(objects=[])
    monkeypatch.setattr(lane, "_anonymous_s3_client", lambda: client)
    assert lane.mirror_zarr_to_local(URL, tmp_path) is None


def test_the_store_is_mirrored_and_marked_complete(monkeypatch, tmp_path):
    prefix = URL.split(".amazonaws.com/", 1)[1]
    objects = [f"{prefix}/zarr.json", f"{prefix}/0/0.0", f"{prefix}/0/0.1"]
    client = _fake_client(objects=objects)
    monkeypatch.setattr(lane, "_anonymous_s3_client", lambda: client)

    local = lane.mirror_zarr_to_local(URL, tmp_path)

    assert local is not None
    local_dir = Path(local)
    assert local_dir.is_relative_to(tmp_path)
    assert (local_dir / "zarr.json").is_file()
    assert (local_dir / "0" / "0.0").is_file()
    marker = json.loads((local_dir / "_HELENA_CACHE_COMPLETE").read_text())
    assert marker["objects"] == 3
    assert marker["revision"] == "abc123"
    # Nothing left behind under the name a racing job's temp dir would use.
    assert not any(p.name.startswith(".tmp-") for p in tmp_path.iterdir())


def test_a_second_call_reads_the_marker_instead_of_downloading_again(monkeypatch, tmp_path):
    prefix = URL.split(".amazonaws.com/", 1)[1]
    objects = [f"{prefix}/zarr.json", f"{prefix}/0/0.0"]
    client = _fake_client(objects=objects)
    monkeypatch.setattr(lane, "_anonymous_s3_client", lambda: client)

    first = lane.mirror_zarr_to_local(URL, tmp_path)
    client.download_file.reset_mock()
    second = lane.mirror_zarr_to_local(URL, tmp_path)

    assert first == second
    client.download_file.assert_not_called()


def test_a_republished_store_gets_a_different_cache_directory(monkeypatch, tmp_path):
    """The revision is part of the key, so a store whose content changed under
    the same URL is not silently served from a stale copy."""
    prefix = URL.split(".amazonaws.com/", 1)[1]
    objects = [f"{prefix}/zarr.json"]
    client_v1 = _fake_client(objects=objects, etag="v1")
    monkeypatch.setattr(lane, "_anonymous_s3_client", lambda: client_v1)
    first = lane.mirror_zarr_to_local(URL, tmp_path)

    client_v2 = _fake_client(objects=objects, etag="v2")
    monkeypatch.setattr(lane, "_anonymous_s3_client", lambda: client_v2)
    second = lane.mirror_zarr_to_local(URL, tmp_path)

    assert first != second


def test_any_failure_falls_back_to_none_rather_than_raising(monkeypatch, tmp_path):
    def broken():
        raise RuntimeError("no network here")
    monkeypatch.setattr(lane, "_anonymous_s3_client", broken)
    assert lane.mirror_zarr_to_local(URL, tmp_path) is None


def test_resolve_surface_volume_ignores_the_cache_when_disabled(monkeypatch, tmp_path):
    monkeypatch.delenv("HELENA_INK_9UM_ZARR_CACHE", raising=False)
    called = []
    monkeypatch.setattr(lane, "mirror_zarr_to_local",
                        lambda url, root: called.append(url) or "/cache/local-copy")
    result = lane.resolve_surface_volume(
        tiff_dir=None, surface_volume=URL, work_dir=tmp_path, source_voxel_um=None)
    # Unset means zarr_cache_root() returns None and mirror_zarr_to_local is
    # never consulted at all, even though the stub above would happily answer.
    assert result == URL
    assert called == []


def test_resolve_surface_volume_uses_the_cache_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HELENA_INK_9UM_ZARR_CACHE", str(tmp_path))
    monkeypatch.setattr(lane, "mirror_zarr_to_local", lambda url, root: "/cache/local-copy")
    result = lane.resolve_surface_volume(
        tiff_dir=None, surface_volume=URL, work_dir=tmp_path, source_voxel_um=None)
    assert result == "/cache/local-copy"


def test_resolve_surface_volume_falls_back_to_the_url_when_caching_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HELENA_INK_9UM_ZARR_CACHE", str(tmp_path))
    monkeypatch.setattr(lane, "mirror_zarr_to_local", lambda url, root: None)
    result = lane.resolve_surface_volume(
        tiff_dir=None, surface_volume=URL, work_dir=tmp_path, source_voxel_um=None)
    assert result == URL


def test_a_tiff_dir_input_never_touches_the_cache(monkeypatch, tmp_path):
    """The cache is for --surface-volume URLs; --tiff-dir already produces a
    local pooled copy and has nothing to cache."""
    monkeypatch.setenv("HELENA_INK_9UM_ZARR_CACHE", str(tmp_path))
    called = []
    monkeypatch.setattr(lane, "mirror_zarr_to_local",
                        lambda url, root: called.append(url) or None)
    monkeypatch.setattr(lane, "prepared_surface_volume",
                        lambda tiff_dir, work_dir, *, source_voxel_um,
                        resample_from_um=None: Path("/pooled.zarr"))
    lane.resolve_surface_volume(
        tiff_dir=Path("/layers"), surface_volume=None, work_dir=tmp_path,
        source_voxel_um=2.399)
    assert called == []
