#!/usr/bin/env python3
"""The seed-candidate service, over the wire the fleet already speaks.

    vc-mcp --volume-root <cache> --token-file <path> [--host 127.0.0.1] [--port 18080]

Replaces the lost `vc_mcp_server`. The transport is whatever campaign_x.McpClient
sends and nothing more: JSON-RPC 2.0 over one POST endpoint, a bearer token, an
Mcp-Session-Id echoed back, `initialize` then `notifications/initialized`, then
`tools/call`. Reimplemented against that client rather than against a
specification, because the client is what will actually connect.

Loopback by default and a token from a file, both because the original was
documented that way and because this reads volumes and should not be a service
anyone on the network can aim at a bucket.

Dependencies are numpy, scipy and zarr, pinned in requirements.txt beside this
file and nothing else -- this said "standard library only" and was wrong about
all three, because every one of them is imported inside a function rather than
at the top. The module therefore imports cleanly on a host that has none of
them and fails at the first seed search, which is the worst place to find out.
Keep them declared: a bundle carries these sources and no dependency closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_candidates import SeedSearchError, find_candidates  # noqa: E402

PROTOCOL = "2025-06-18"

TOOLS = [{
    "name": "vc_find_seed_candidates",
    "description": "Points in a region of an m7 surface prediction that look "
                   "like sheet and are far enough apart to be different places.",
    "inputSchema": {
        "type": "object",
        "required": ["prediction_uri", "region"],
        "properties": {
            "prediction_uri": {"type": "string"},
            "prediction_space": {"type": "string", "default": "ct_l0_xyz"},
            "region": {"type": "object"},
            "max_candidates": {"type": "integer", "default": 8},
            "minimum_separation_voxels": {"type": "integer", "default": 16},
            "threshold": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Normalized threshold at which this exact m7 prediction "
                    "snapshot was published."
                ),
            },
        },
    },
}]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PredictionReadTracker:
    """Canonical byte inventory for the exact Zarr objects a query consumed.

    Cache behavior is intentionally receipt-stable: an identical logical
    object/range and identical returned bytes coalesces to one row whether the
    backing filesystem served it twice or from cache; contradictory bytes under
    that same identity fail the query.
    """

    def __init__(self):
        self.objects: dict[str, dict[str, Any]] = {}

    @staticmethod
    def canonical_byte_range(byte_range: object) -> tuple[str, dict[str, Any] | None]:
        if byte_range is None:
            return "", None
        if hasattr(byte_range, "start") and hasattr(byte_range, "end"):
            start, end = int(byte_range.start), int(byte_range.end)
            if start < 0 or end < start:
                raise SeedSearchError("prediction range is invalid")
            return f"#range={start}:{end}", {
                "kind": "range", "start": start, "end": end}
        if hasattr(byte_range, "offset"):
            offset = int(byte_range.offset)
            if offset < 0:
                raise SeedSearchError("prediction offset is invalid")
            return f"#offset={offset}", {"kind": "offset", "offset": offset}
        if hasattr(byte_range, "suffix"):
            suffix = int(byte_range.suffix)
            if suffix < 0:
                raise SeedSearchError("prediction suffix is invalid")
            return f"#suffix={suffix}", {"kind": "suffix", "suffix": suffix}
        raise SeedSearchError(
            f"unsupported prediction byte-range type {type(byte_range).__name__}")

    def record(self, key: object, value: Any, byte_range: object = None) -> None:
        if value is None:
            return
        raw = value.to_bytes()
        suffix, range_document = self.canonical_byte_range(byte_range)
        identity = f"{key}{suffix}"
        row = {"object_key": identity, "sha256": hashlib.sha256(raw).hexdigest(),
               "bytes": len(raw),
               **({"byte_range": range_document} if range_document else {})}
        previous = self.objects.get(identity)
        if previous is not None and previous != row:
            raise SeedSearchError(f"prediction object {identity!r} changed during the query")
        self.objects[identity] = row

    def receipt(self) -> dict[str, Any]:
        objects = [self.objects[key] for key in sorted(self.objects)]
        if not objects:
            raise SeedSearchError("prediction query recorded no source objects")
        return {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": objects,
            "canonical_manifest_sha256": _canonical_sha256(objects),
        }


def prediction_store_with_read_set(target: str | Path) -> tuple[Any, PredictionReadTracker]:
    """Build the same tracked LocalStore/FsspecStore used by production opens."""
    from urllib.parse import urlparse
    from zarr.storage import FsspecStore, LocalStore, WrapperStore

    class ReadTrackingStore(WrapperStore):
        def __init__(self, store, tracker):
            super().__init__(store)
            self.tracker = tracker

        async def get(self, key, prototype, byte_range=None):
            value = await self._store.get(key, prototype, byte_range)
            self.tracker.record(key, value, byte_range)
            return value

        async def get_partial_values(self, prototype, key_ranges):
            ranges = list(key_ranges)
            values = await self._store.get_partial_values(prototype, ranges)
            for (key, byte_range), value in zip(ranges, values, strict=True):
                self.tracker.record(key, value, byte_range)
            return values

    parsed = urlparse(str(target))
    backing = (LocalStore(Path(target), read_only=True)
               if parsed.scheme in {"", "file"}
               else FsspecStore.from_url(str(target), read_only=True))
    tracker = PredictionReadTracker()
    return ReadTrackingStore(backing, tracker), tracker


def open_prediction_with_read_set(
    uri: str, volume_root: Path | None, level: int = 0,
) -> tuple[Any, PredictionReadTracker]:
    """The m7 prediction as an array, not the group that contains it.

    A published prediction is OME-Zarr, so `zarr.open` returns a *group* whose
    scale levels are children. Slicing that group makes zarr treat the slice
    tuple as a key and raise `path=(slice(...)...) is not a string` -- which
    reached the fleet as BLOCKED_SOURCE_UNAVAILABLE and read like the bucket
    being down. The level has to be resolved first.

    Resolved the way ct_support.py resolves it, from the multiscales metadata
    rather than by assuming the array is at "0". The two now fail the same way on
    a source with no OME metadata, which matters because they read the same kind
    of volume.

    Level 0 by default: the region arrives in CT-L0 voxels, so a coarser level
    would return candidates whose coordinates mean something else.

    zarr is imported here rather than at module scope so the server starts, and
    reports a clear error, on a host where it is missing -- instead of failing at
    import and leaving systemd restarting something that will never work.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - depends on the image
        raise SeedSearchError(
            "zarr is not installed in this worker image, so no prediction can "
            "be read") from exc

    target = uri
    if volume_root:
        local = Path(volume_root) / uri.rstrip("/").split("/")[-1]
        if local.exists():
            target = str(local)
    # A local target must be under the root this server was given. `volume_root`
    # was a preference -- use the mirror if it happens to be there, otherwise
    # open whatever the request named -- and `prediction_uri` comes from the
    # request, so a plain path opened whatever the worker could read and a URL
    # made the worker fetch it. The remote fallback is the point of the option
    # and stays; naming a filesystem path outside the root is not a fallback.
    if volume_root and "://" not in target:
        root = Path(volume_root).resolve()
        resolved = Path(target).resolve()
        if resolved != root and root not in resolved.parents:
            raise SeedSearchError(
                f"prediction_uri names {target!r}, which is outside this "
                f"server's volume root. A local prediction is read from there "
                "or not at all.")

    backing, tracker = prediction_store_with_read_set(target)
    # The frozen CT/M7 root locks name v2 objects (`.zattrs`, `0/.zarray`).
    # Never auto-substitute v3 metadata or `.zmetadata`: either would make the
    # executed source receipt incomparable with those locks.
    opened = zarr.open(
        store=backing, mode="r",
        zarr_format=2, use_consolidated=False,
    )
    # An array already -- a plain zarr, or a level someone named directly. A
    # zarr Group has no shape, which is the whole distinction that matters here.
    if hasattr(opened, "shape"):
        return opened, tracker

    multiscales = opened.attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise SeedSearchError(
            f"prediction at {uri} is a group with no OME multiscales metadata, "
            "so there is no way to know which child is the volume")
    datasets = multiscales[0].get("datasets")
    if not isinstance(datasets, list):
        raise SeedSearchError(f"prediction at {uri} has no OME dataset list")
    dataset = next((row for row in datasets if str(row.get("path")) == str(level)), None)
    if dataset is None:
        available = [str(row.get("path")) for row in datasets]
        raise SeedSearchError(
            f"prediction at {uri} has no OME level {level}; it has {available}")
    return opened[str(dataset["path"])], tracker


def open_prediction(uri: str, volume_root: Path | None, level: int = 0) -> Any:
    """Compatibility helper for callers that only need the array."""
    return open_prediction_with_read_set(uri, volume_root, level)[0]


class Service:
    """The tool, separated from the transport so it can be tested without one."""

    def __init__(self, volume_root: Path | None):
        self.volume_root = volume_root

    def find_seed_candidates(self, arguments: dict[str, Any]) -> dict[str, Any]:
        uri = str(arguments["prediction_uri"])
        space = str(arguments.get("prediction_space", "ct_l0_xyz"))
        if space != "ct_l0_xyz":
            raise SeedSearchError(f"unsupported prediction_space {space!r}")
        volume, tracker = open_prediction_with_read_set(uri, self.volume_root)
        candidates = find_candidates(
            volume,
            dict(arguments["region"]),
            prediction_uri=uri,
            max_candidates=int(arguments.get("max_candidates", 8)),
            minimum_separation_voxels=int(arguments.get("minimum_separation_voxels", 16)),
            # Screened at what the caller says the prediction was published at.
            # DEFAULT_THRESHOLD stays as the fallback for a caller that does not
            # know -- and its own comment asked for exactly this.
            **({"threshold": float(arguments["threshold"])}
               if arguments.get("threshold") is not None else {}),
        )
        # `candidates` is the key the worker validates, and an empty list is a
        # real answer -- the prediction has nothing there. Only a failure to
        # look is an error, which is why SeedSearchError propagates.
        return {"candidates": candidates, "prediction_uri": uri,
                "candidate_count": len(candidates),
                "source_read_set": tracker.receipt()}


def handler_for(service: Service, token: str) -> type[BaseHTTPRequestHandler]:
    sessions: set[str] = set()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "helena-seed-mcp/1.0"

        def log_message(self, fmt, *args):  # noqa: A003 - quiet by default
            if os.environ.get("VC_MCP_VERBOSE"):
                sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        def reply(self, payload: dict | None, status: int = 200,
                  session: str | None = None) -> None:
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if session:
                self.send_header("Mcp-Session-Id", session)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def authorized(self) -> bool:
            supplied = (self.headers.get("Authorization") or "").removeprefix(
                "Bearer ").strip()
            # compare_digest, not ==, for the same reason the panel's login uses
            # it: the obvious comparison leaks how much of the token matched.
            return secrets.compare_digest(supplied, token)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            """One route, and it is authenticated for the reason it exists.

            The worker's entrypoint waits for this service before claiming any
            task, and it used to wait for `something accepts a connection on
            this port`. The container runs with the host's network, so that was
            satisfied by any process on the machine -- and the worker then sent
            every seed request to whatever it was.

            Answering this with the token is the proof that the thing on the
            port is the service that was started, which is what the wait was
            always trying to establish. Unauthenticated it would prove nothing:
            a squatter can return 200.
            """
            if self.path != "/healthz":
                self.reply({"error": "not found"}, status=404)
                return
            if not self.authorized():
                self.reply({"error": "unauthorized"}, status=401)
                return
            self.reply({"service": "helena-seed-mcp", "ready": True})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
            supplied = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            # compare_digest, not ==, for the same reason the panel's login uses
            # it: the obvious comparison leaks how much of the token matched.
            if not secrets.compare_digest(supplied, token):
                self.reply({"error": "unauthorized"}, status=401)
                return

            length = int(self.headers.get("Content-Length") or 0)
            try:
                message = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.reply({"jsonrpc": "2.0", "id": None,
                            "error": {"code": -32700, "message": "parse error"}})
                return

            method = message.get("method")
            identifier = message.get("id")

            # A notification takes an empty 202: the client returns {} for
            # anything without an id and would choke on a body.
            if identifier is None:
                self.reply(None, status=202)
                return

            if method == "initialize":
                session = secrets.token_urlsafe(16)
                with lock:
                    sessions.add(session)
                self.reply({
                    "jsonrpc": "2.0", "id": identifier,
                    "result": {"protocolVersion": PROTOCOL,
                               "capabilities": {"tools": {}},
                               "serverInfo": {"name": "helena-seed-mcp",
                                              "version": "1.0"}},
                }, session=session)
                return

            if method == "tools/list":
                self.reply({"jsonrpc": "2.0", "id": identifier,
                            "result": {"tools": TOOLS}})
                return

            if method != "tools/call":
                self.reply({"jsonrpc": "2.0", "id": identifier,
                            "error": {"code": -32601,
                                      "message": f"unknown method {method}"}})
                return

            params = message.get("params") or {}
            name = params.get("name")
            if name != "vc_find_seed_candidates":
                self.reply({"jsonrpc": "2.0", "id": identifier,
                            "error": {"code": -32602,
                                      "message": f"unknown tool {name}"}})
                return

            try:
                result = service.find_seed_candidates(params.get("arguments") or {})
            except (SeedSearchError, KeyError, ValueError, OSError) as exc:
                # A JSON-RPC error, not an empty result. The worker treats an
                # empty structured object as a source failure precisely because
                # a tool exception must never read as "no candidates here".
                self.reply({"jsonrpc": "2.0", "id": identifier,
                            "error": {"code": -32000,
                                      "message": f"{type(exc).__name__}: {exc}"}})
                return

            self.reply({"jsonrpc": "2.0", "id": identifier,
                        "result": {"structuredContent": result,
                                   "content": [{"type": "text",
                                                "text": json.dumps(result)}]}})

    return Handler


def read_token(path: Path) -> str:
    token = Path(path).read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise SystemExit("the MCP token must be at least 32 characters")
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="loopback by default: this reads volumes and is "
                             "not something to expose")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--volume-root", type=Path,
                        help="local cache of prediction volumes, if there is one")
    arguments = parser.parse_args()

    token = read_token(arguments.token_file)
    service = Service(arguments.volume_root)
    server = ThreadingHTTPServer((arguments.host, arguments.port),
                                 handler_for(service, token))
    ready = {"schema": "campaignx.vc3d_mcp_runtime.v1", "status": "READY",
             "pid": os.getpid(),
             "endpoint": f"http://{arguments.host}:{arguments.port}/mcp",
             "transport": "streamable-http"}
    print(json.dumps(ready), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
