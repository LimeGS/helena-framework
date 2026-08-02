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

Standard library only. A seed search that cannot start because a dependency
moved is a fleet that stops.
"""

from __future__ import annotations

import argparse
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


def open_prediction(uri: str, volume_root: Path | None, level: int = 0) -> Any:
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

    opened = zarr.open(target, mode="r")
    # An array already -- a plain zarr, or a level someone named directly. A
    # zarr Group has no shape, which is the whole distinction that matters here.
    if hasattr(opened, "shape"):
        return opened

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
    return opened[str(dataset["path"])]


class Service:
    """The tool, separated from the transport so it can be tested without one."""

    def __init__(self, volume_root: Path | None):
        self.volume_root = volume_root

    def find_seed_candidates(self, arguments: dict[str, Any]) -> dict[str, Any]:
        uri = str(arguments["prediction_uri"])
        space = str(arguments.get("prediction_space", "ct_l0_xyz"))
        if space != "ct_l0_xyz":
            raise SeedSearchError(f"unsupported prediction_space {space!r}")
        volume = open_prediction(uri, self.volume_root)
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
                "candidate_count": len(candidates)}


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
