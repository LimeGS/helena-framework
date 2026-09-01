"""The same gap, in the lane that reads through boto3 instead of aiohttp.

The segmentation preflight lost three runs to a dropped connection because its
retryable set was built from builtin exceptions and the real error was
`aiohttp.ServerDisconnectedError`, which never passes through OSError. The ink
lane reads its artifacts through boto3, and botocore repeats the trap exactly:

    EndpointConnectionError -> botocore's own ConnectionError -> BotoCoreError
    ConnectionClosedError   -> HTTPClientError -> BotoCoreError
    IncompleteReadError     -> BotoCoreError -> Exception

none of which is a builtin ConnectionError or an OSError.

What is *not* a gap here, checked rather than assumed: `download_file` retries
on its own. s3transfer carries S3_RETRYABLE_DOWNLOAD_ERRORS -- TimeoutError,
ConnectionError, ReadTimeoutError, IncompleteReadError, ResponseStreamingError
-- so the per-file transfer already survives a blip.

The manifest does not. `get_object(...)["Body"].read()` streams the body after
the API call has already returned, so botocore's request-layer retry is behind
it and s3transfer is not in front of it. A disconnect there fails the whole
fetch, and every file that would have been verified against that manifest goes
with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from fleet.retrying import RETRYABLE_NETWORK_ERRORS  # noqa: E402


def test_botocore_connection_errors_are_retryable() -> None:
    """They are not builtins, so naming them is the only way they are covered."""
    botocore = pytest.importorskip("botocore.exceptions")

    for name in ("EndpointConnectionError", "ConnectionClosedError",
                 "IncompleteReadError"):
        error = getattr(botocore, name)
        assert not issubclass(error, (OSError, ConnectionError, TimeoutError)), (
            f"{name} became a builtin; the explicit entry could go")
        assert issubclass(error, RETRYABLE_NETWORK_ERRORS), (
            f"{name} ends a job that a second attempt would have finished")


def test_an_s3_client_error_is_not_retried() -> None:
    """A 403 or a missing key is an answer, not a blip. Retrying it spends the
    budget and delays the report -- and `ClientError` is the parent of both."""
    botocore = pytest.importorskip("botocore.exceptions")

    assert not issubclass(botocore.ClientError, RETRYABLE_NETWORK_ERRORS)


class _Body:
    """A streamed response body that drops the connection the first time."""

    def __init__(self, payload: bytes, drops: list):
        self.payload = payload
        self.drops = drops

    def read(self):
        import botocore.exceptions as be

        if not self.drops:
            self.drops.append(1)
            raise be.ConnectionClosedError(endpoint_url="https://s3.invalid")
        return self.payload


def test_the_manifest_read_survives_a_dropped_connection(tmp_path, monkeypatch):
    """The gap itself. botocore's retry is behind this read and s3transfer is
    not in front of it, so nothing but this covers it."""
    pytest.importorskip("botocore")
    boto3 = pytest.importorskip("boto3")

    import ink_worker

    payload = tmp_path / "source"
    payload.mkdir()
    (payload / "layer.tif").write_bytes(b"a layer")
    import hashlib
    digest = hashlib.sha256(b"a layer").hexdigest()
    manifest = {"files": {"layer.tif": {"sha256": digest}}}

    drops: list = []

    class _Client:
        def get_object(self, Bucket, Key):  # noqa: N803 - boto3's own spelling
            return {"Body": _Body(json.dumps(manifest).encode(), drops)}

        def download_file(self, bucket, key, destination):
            Path(destination).write_bytes(b"a layer")

    monkeypatch.setattr(boto3, "client", lambda _service: _Client())
    monkeypatch.setattr(ink_worker.time, "sleep", lambda _s: None, raising=False)

    got = ink_worker.fetch_artifact_set("s3://bucket/base",
                                        tmp_path / "destination")

    assert drops, "the connection never dropped; the test proved nothing"
    assert got == manifest


def test_an_artifact_that_arrives_wrong_is_still_refused(tmp_path, monkeypatch):
    """Retrying a transport failure must not turn into retrying a verification
    failure. A digest mismatch means this is not that artifact, and no number of
    attempts changes that."""
    boto3 = pytest.importorskip("boto3")

    import ink_worker

    manifest = {"files": {"layer.tif": {"sha256": "0" * 64}}}
    attempts: list = []

    class _Client:
        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": _Body(json.dumps(manifest).encode(), [1])}

        def download_file(self, bucket, key, destination):
            attempts.append(1)
            Path(destination).write_bytes(b"something else entirely")

    monkeypatch.setattr(boto3, "client", lambda _service: _Client())

    with pytest.raises(RuntimeError, match="this is not that artifact"):
        ink_worker.fetch_artifact_set("s3://bucket/base", tmp_path / "destination")
