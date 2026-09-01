"""A network blip must not read as a scientific finding.

A preflight ran for an hour and died on

    CtSupportSourceUnavailable: could not sample frozen CT material support:
    ServerDisconnected

-- one dropped connection to S3, and seventy minutes of measurement gone. There
was no retry anywhere in the source reads: `grep -c 'retry|backoff'` over the
CT sampler and the preflight returned zero.

The fleet's own vocabulary already said this should not be terminal. The worker
classifies it as

    # The sources did not answer, or answered unusably. May recover.
    "PREFLIGHT_SOURCE_UNAVAILABLE"

and then nothing acted on "may recover". The classification existed; the
consequence did not.

Bounded on purpose. A retry that never gives up turns an outage into a hang, and
this program has already paid for one unbounded wait. Three attempts with a
short backoff covers a dropped connection and still surfaces a bucket that is
genuinely gone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.retrying import RETRYABLE_NETWORK_ERRORS, read_with_retry  # noqa: E402


def test_a_read_that_works_is_not_retried() -> None:
    calls = []

    def read():
        calls.append(1)
        return "payload"

    assert read_with_retry(read, sleep=lambda _s: None) == "payload"
    assert len(calls) == 1, "a working read was attempted more than once"


def test_a_dropped_connection_is_retried_and_succeeds() -> None:
    calls = []

    def read():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("ServerDisconnected")
        return "payload"

    assert read_with_retry(read, sleep=lambda _s: None) == "payload"
    assert len(calls) == 3


def test_it_gives_up_and_raises_the_last_failure() -> None:
    """An outage has to stay visible: the point is not to hide a bucket that is
    gone, it is to survive one that blinked."""
    calls = []

    def read():
        calls.append(1)
        raise ConnectionError("ServerDisconnected")

    with pytest.raises(ConnectionError):
        read_with_retry(read, attempts=3, sleep=lambda _s: None)
    assert len(calls) == 3, "the attempt budget was not honoured"


def test_it_waits_longer_between_attempts() -> None:
    """Hammering a service that just dropped you is how a blip becomes an
    outage."""
    waits: list[float] = []

    def read():
        raise ConnectionError("ServerDisconnected")

    with pytest.raises(ConnectionError):
        read_with_retry(read, attempts=4, sleep=waits.append)
    assert waits == sorted(waits) and len(set(waits)) > 1, waits


def test_a_failure_that_will_not_heal_is_not_retried() -> None:
    """Retrying a malformed document or a permission error just spends the
    budget before the real error is reported."""
    calls = []

    def read():
        calls.append(1)
        raise ValueError("this zarr array has the wrong dtype")

    with pytest.raises(ValueError):
        read_with_retry(read, sleep=lambda _s: None)
    assert len(calls) == 1


def test_the_retryable_set_names_what_actually_happened() -> None:
    assert ConnectionError in RETRYABLE_NETWORK_ERRORS
    assert TimeoutError in RETRYABLE_NETWORK_ERRORS
    assert ValueError not in RETRYABLE_NETWORK_ERRORS


def test_the_error_that_actually_killed_three_runs_is_retried() -> None:
    """The one that matters, named exactly.

        CtSupportSourceUnavailable: could not sample frozen CT material
        support: ServerDisconnectedError: Server disconnected

    is `aiohttp.ServerDisconnectedError`, and its MRO is

        ServerDisconnectedError -> ServerConnectionError ->
        ClientConnectionError -> ClientError -> Exception

    which never passes through OSError. A retry set built from the builtin
    connection errors looks like it covers a dropped connection and does not
    cover the dropped connection this program keeps losing hours to. s3fs reads
    through aiohttp, so this is the shape every source failure here arrives in.
    """
    aiohttp = pytest.importorskip("aiohttp")

    assert not issubclass(aiohttp.ServerDisconnectedError, OSError), (
        "if this ever becomes an OSError the builtin set would be enough")

    calls = []

    def read():
        calls.append(1)
        if len(calls) < 2:
            raise aiohttp.ServerDisconnectedError("Server disconnected")
        return "payload"

    assert read_with_retry(read, sleep=lambda _s: None) == "payload"
    assert len(calls) == 2


def test_an_http_status_is_not_retried() -> None:
    """The other side of reaching into aiohttp: a 403 or a 404 is an answer,
    not a blip, and it will be the same answer three times."""
    aiohttp = pytest.importorskip("aiohttp")

    calls = []

    def read():
        calls.append(1)
        raise aiohttp.ClientResponseError(None, (), status=403,
                                          message="Forbidden")

    with pytest.raises(aiohttp.ClientResponseError):
        read_with_retry(read, sleep=lambda _s: None)
    assert len(calls) == 1, "the budget was spent on a permission error"


@pytest.mark.parametrize("error", [
    FileNotFoundError("no such object"),
    PermissionError("access denied"),
])
def test_an_oserror_that_will_never_heal_is_not_retried(error) -> None:
    """OSError is wide enough to swallow these, and they are all subclasses of
    it. A missing object and a refused permission do not improve with waiting;
    retrying them spends the budget and delays the real report."""
    calls = []

    def read():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        read_with_retry(read, sleep=lambda _s: None)
    assert len(calls) == 1


def test_the_ct_sampler_survives_a_dropped_connection(monkeypatch) -> None:
    """The read that actually failed, at the boundary that actually failed."""
    from fleet import ct_support

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("ServerDisconnected")
        return {"ok": True}

    assert ct_support.read_with_retry(flaky, sleep=lambda _s: None) == {"ok": True}
    assert len(calls) == 2


def _write_ome_zarr(root, numpy):
    """A real OME-Zarr the sampler will actually open and read."""
    import zarr

    group = zarr.open_group(store=str(root), mode="w")
    array = group.create_array("0", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="uint8")
    array[:] = numpy.arange(8 * 8 * 8, dtype="uint8").reshape(8, 8, 8)
    group.attrs["multiscales"] = [{
        "datasets": [{"path": "0",
                      "coordinateTransformations": [
                          {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}],
    }]
    return root


def test_the_real_sampler_survives_a_connection_dropped_mid_read(tmp_path, monkeypatch):
    """End to end, at the boundary that actually failed.

    The unit tests above prove the helper; this proves it is *wired in*. It
    opens a real zarr through the real sampler and drops the connection on the
    first object fetch, which is exactly the shape of the failure that ended a
    seventy-minute preflight."""
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("zarr")
    import zarr.storage

    from fleet import ct_support

    root = _write_ome_zarr(tmp_path / "ct.zarr", numpy)

    dropped: list[int] = []
    real_local_store = zarr.storage.LocalStore

    # The exact exception the deployment raised, not a stand-in: three runs died
    # on this one, and a ConnectionError here would have passed while the real
    # thing went straight through the retry untouched.
    aiohttp = pytest.importorskip("aiohttp")
    outage = aiohttp.ServerDisconnectedError("Server disconnected")

    class DropsOnceLocalStore(real_local_store):
        async def get(self, *args, **kwargs):
            if not dropped:
                dropped.append(1)
                raise outage
            return await super().get(*args, **kwargs)

    # The sampler imports zarr.storage inside the call, so patching the module
    # attribute reaches the store it will actually construct.
    monkeypatch.setattr(zarr.storage, "LocalStore", DropsOnceLocalStore)

    sample = ct_support.OmeZarrCtSupportSampler().sample(
        str(root), {"x": 4, "y": 4, "z": 4}, level=0, radius_l0_voxels=2)

    assert dropped, "the connection was never dropped; the test proved nothing"
    assert sample["voxel_count"] > 0
    assert sample["source_read_set"]["objects"], "the read set lost its provenance"
