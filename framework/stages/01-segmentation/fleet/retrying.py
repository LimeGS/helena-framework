"""Survive a network blip without hiding an outage.

A preflight ran for an hour and died on one dropped connection to S3. There was
no retry anywhere in the source reads, while the fleet's own vocabulary already
said a source failure "may recover" -- the classification existed and nothing
acted on it.

Two things this deliberately is not:

* not unbounded. A retry that never gives up turns an outage into a hang, and an
  hour lost to a blip is still better than a run that never ends.
* not a way to make a real failure look like a success. Only the errors that
  describe a connection are retried; a malformed document or a refused
  permission is raised on the first attempt, because spending the budget on it
  only delays the report.

The retryable set is not just the builtins. The failure this exists for is
`aiohttp.ServerDisconnectedError`, which descends from Exception and never
passes through OSError -- so a set built from `ConnectionError` and friends
reads as though it covers a dropped connection and covers everything except the
one that keeps happening.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def _aiohttp_connection_errors() -> tuple[type[BaseException], ...]:
    """The transport the sources are actually read through.

    s3fs and fsspec read over aiohttp, whose exceptions descend from Exception
    and never pass through OSError:

        ServerDisconnectedError -> ServerConnectionError ->
        ClientConnectionError -> ClientError -> Exception

    So a retry set built from the builtin connection errors looks like it covers
    a dropped connection while missing the only one that has actually happened
    here. Named narrowly on purpose: `ClientError` would also swallow
    `ClientResponseError`, and a 403 is an answer, not a blip -- it will be the
    same answer three times.

    Optional: a control plane without aiohttp installed reads through something
    else, and an ImportError here would be a worker that cannot start.
    """
    try:
        import aiohttp
    except Exception:  # pragma: no cover - depends on the deployment
        return ()
    named = ("ClientConnectionError", "ClientPayloadError", "ServerTimeoutError")
    return tuple(error for error in
                 (getattr(aiohttp, name, None) for name in named)
                 if isinstance(error, type) and issubclass(error, BaseException))


def _botocore_connection_errors() -> tuple[type[BaseException], ...]:
    """The other transport, with the same trap.

    The ink lane reads its artifacts through boto3, and botocore's connection
    errors are no more builtin than aiohttp's:

        EndpointConnectionError -> botocore's own ConnectionError (not the
                                   builtin) -> BotoCoreError
        ConnectionClosedError   -> HTTPClientError -> BotoCoreError
        IncompleteReadError     -> BotoCoreError -> Exception

    Deliberately not `BotoCoreError` wholesale, and never `ClientError`: a 403
    or a missing key is an answer, and it will be the same answer three times.
    """
    try:
        from botocore import exceptions
    except Exception:  # pragma: no cover - depends on the deployment
        return ()
    named = ("EndpointConnectionError", "ConnectionClosedError",
             "IncompleteReadError", "ReadTimeoutError", "ConnectTimeoutError",
             "ResponseStreamingError")
    return tuple(error for error in
                 (getattr(exceptions, name, None) for name in named)
                 if isinstance(error, type) and issubclass(error, BaseException))


# What a blip looks like. urllib, httpx and fsspec all surface a dropped
# connection as one of these or a subclass; `http.client.RemoteDisconnected` and
# `ConnectionResetError` are both ConnectionError.
RETRYABLE_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    socket.timeout,
    socket.gaierror,
    OSError,
    # The builtins above are not enough on their own: the error that has ended
    # three runs of the control comes from aiohttp and never passes through
    # OSError, and botocore does the same thing in the ink lane. Each transport
    # has to be named. See the two helpers above.
    *_aiohttp_connection_errors(),
    *_botocore_connection_errors(),
)


# OSError is wide enough to swallow failures that will never heal, and they are
# all its subclasses: a missing object, a refused permission, a path that is a
# directory. Retrying those spends the budget and delays the real report.
PERMANENT_OS_ERRORS: tuple[type[BaseException], ...] = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    FileExistsError,
)

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0


def read_with_retry(
    read: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], Any] = time.sleep,
) -> T:
    """Call ``read`` until it works, up to ``attempts`` times.

    Waits longer between attempts: hammering a service that has just dropped you
    is how a blip becomes an outage. Raises the last failure when the budget is
    spent, so a bucket that is genuinely gone still reports as gone.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return read()
        except PERMANENT_OS_ERRORS:
            raise
        except RETRYABLE_NETWORK_ERRORS as error:
            last = error
            if attempt + 1 >= attempts:
                break
            sleep(backoff_seconds * (2 ** attempt))
    assert last is not None
    raise last


async def aread_with_retry(
    read: Callable[[], Awaitable[T]],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> T:
    """``read_with_retry`` for an awaitable read.

    The zarr stores fetch objects asynchronously, and that fetch is the boundary
    a dropped connection actually crosses. Retrying there costs one object;
    retrying the caller would cost the whole measurement.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await read()
        except PERMANENT_OS_ERRORS:
            raise
        except RETRYABLE_NETWORK_ERRORS as error:
            last = error
            if attempt + 1 >= attempts:
                break
            await sleep(backoff_seconds * (2 ** attempt))
    assert last is not None
    raise last
