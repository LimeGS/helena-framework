#!/usr/bin/env python3
"""One HTTP client for everything that drives this platform from outside.

The smoke test, the ink control and the end-to-end suite all talk to the same
panel the same way, and each of them had its own copy of this -- three cookie
jars, three ways of reporting a 400, three polling loops with different ideas of
when a job is finished.

stdlib only, so it runs anywhere the panel is reachable: inside the panel image,
on a laptop, in CI.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar

TERMINAL_JOB_STATES = ("succeeded", "failed", "cancelled")


class PanelError(RuntimeError):
    """The panel refused, with what it said."""

    def __init__(self, method: str, path: str, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"{method} {path} -> HTTP {status}: {body[:400]}")


class AmbiguousMutationError(RuntimeError):
    """A mutation may have committed, but its response could not be read.

    Callers must read platform state before deciding what happened.  Retrying
    here could create a second job or experiment under a new identity.
    """

    def __init__(self, method: str, path: str, detail: str):
        self.method = method
        self.path = path
        self.detail = detail
        super().__init__(
            f"{method} {path} has an ambiguous outcome ({detail}); the mutation "
            "must not be retried until platform state is read back"
        )


class Panel:
    """A session against one panel.

    `trust`: a CA bundle or the panel's own certificate. The panel generates a
    self-signed pair on first boot, so a client that verifies against the system
    store cannot reach it -- point this at /state/tls/panel.crt (any copy of it)
    and verification works normally, hostname check included.

    `insecure=True` skips verification entirely. It is off by default, and
    HELENA_PANEL_TLS_INSECURE=1 turns it on for a harness running against a
    deployment whose certificate it has no copy of. That is a real downgrade:
    it authenticates nothing, so it belongs on a trusted network and nowhere
    else.
    """

    def __init__(self, base: str, timeout: float = 3600, *,
                 trust: str | None = None, insecure: bool | None = None):
        self.base = base.rstrip("/")
        self.timeout = timeout
        handlers = [urllib.request.HTTPCookieProcessor(CookieJar())]
        trust = trust or os.environ.get("HELENA_PANEL_TLS_TRUST") or None
        if insecure is None:
            insecure = os.environ.get("HELENA_PANEL_TLS_INSECURE") == "1"
        if self.base.startswith("https://"):
            context = ssl.create_default_context(cafile=trust)
            if insecure and not trust:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            handlers.append(urllib.request.HTTPSHandler(context=context))
        self.http = urllib.request.build_opener(*handlers)

    def call(self, method: str, path: str, body: dict | None = None, *,
             timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with self.http.open(
                    request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as failure:
            body_text = failure.read().decode(errors="replace")
            if method.upper() in {"POST", "PUT", "PATCH", "DELETE"} \
                    and failure.code in {502, 504}:
                raise AmbiguousMutationError(
                    method.upper(), path, f"HTTP {failure.code}: {body_text}"
                ) from None
            raise PanelError(method, path, failure.code,
                             body_text) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                json.JSONDecodeError) as failure:
            if method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
                raise AmbiguousMutationError(
                    method.upper(), path, f"{type(failure).__name__}: {failure}"
                ) from None
            raise

    def fetch(self, path: str, *, timeout: float | None = None) -> bytes:
        """One response body, unparsed.

        `call` decodes JSON, which is right for every route but the artifact
        one: that hands back a directory as a gzipped tar, and a control that
        wants to see what a job published has to read it as bytes. Reading it
        off the worker's disk instead would be reaching into the machine under
        test rather than using the interface anybody else has.
        """
        request = urllib.request.Request(self.base + path, method="GET")
        try:
            with self.http.open(
                    request, timeout=self.timeout if timeout is None else timeout
            ) as response:
                return response.read()
        except urllib.error.HTTPError as failure:
            raise PanelError("GET", path, failure.code,
                             failure.read().decode(errors="replace")) from None

    def sign_in(self, username: str, password: str) -> str:
        """Through the real login. Nothing here has a way around it."""
        self.call("POST", "/api/session", {"username": username, "password": password})
        return str(self.call("GET", "/api/session").get("username") or "")

    def wait_for_job(self, job_id: str, *, minutes: float = 60,
                     tick: float = 15, on_tick=None) -> dict:
        """Poll one job to a terminal state, or say it did not reach one.

        Terminal means the queue is done with it -- succeeded, failed or
        cancelled. A caller that treats "failed" as an exception here would lose
        the result it needs to report.
        """
        deadline = time.monotonic() + minutes * 60
        while time.monotonic() < deadline:
            found = [job for job in self.call("GET", "/api/jobs?limit=50").get("jobs", [])
                     if job["job_id"] == job_id]
            if found and found[0]["state"] in TERMINAL_JOB_STATES:
                return found[0]
            if on_tick:
                on_tick()
            time.sleep(tick)
        raise TimeoutError(f"{job_id} did not finish within {minutes} minutes")

    def wait_until(self, predicate, *, minutes: float = 30, tick: float = 20,
                   on_tick=None):
        """Poll until a condition of the platform's own state holds."""
        deadline = time.monotonic() + minutes * 60
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            if on_tick:
                on_tick()
            time.sleep(tick)
        return None
