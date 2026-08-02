"""The seed service, exercised through the client the fleet actually uses.

Written against campaign_x.McpClient rather than against a specification of
MCP, because the client is what will connect and its expectations are the ones
that matter -- the session header round-trip, the empty 202 for notifications,
and an error that arrives as JSON-RPC rather than as an empty result.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework" / "stages" / "01-segmentation" / "mcp"))

import server as mcp_server  # noqa: E402
from campaign_x import McpClient, structured  # noqa: E402

TOKEN = "a" * 40
URI = "s3://bucket/PHerc0139/surface-m7-L0-th0.2.zarr"


class FakeService(mcp_server.Service):
    """The transport under test, with the volume read stubbed out."""

    def __init__(self, volume):
        super().__init__(None)
        self.volume = volume

    def find_seed_candidates(self, arguments):
        from seed_candidates import find_candidates
        return {"candidates": find_candidates(
            self.volume, dict(arguments["region"]),
            prediction_uri=str(arguments["prediction_uri"]),
            max_candidates=int(arguments.get("max_candidates", 8)),
            minimum_separation_voxels=int(arguments.get("minimum_separation_voxels", 16)),
        )}


@pytest.fixture
def endpoint():
    volume = np.zeros((64, 64, 64), dtype=np.uint8)
    volume[10, 20, 30] = 220
    handler = mcp_server.handler_for(FakeService(volume), TOKEN)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/mcp"
    httpd.shutdown()


def region(size=64):
    return {f"{a}_{b}": (0 if b == "min" else size) for a in "xyz" for b in ("min", "max")}


def test_the_fleet_client_can_complete_a_search(endpoint):
    """The whole handshake, through the real client."""
    client = McpClient(endpoint, TOKEN)
    client.initialize()
    response = client.call("vc_find_seed_candidates",
                           {"prediction_uri": URI, "region": region()}, "req-1")
    result = structured(response)
    assert isinstance(result["candidates"], list)
    assert result["candidates"][0]["ct_l0_coordinate"] == {"x": 30, "y": 20, "z": 10}


def test_a_wrong_token_is_refused(endpoint):
    client = McpClient(endpoint, "b" * 40)
    with pytest.raises(Exception):
        client.initialize()


def test_a_failure_arrives_as_an_error_not_an_empty_result(endpoint):
    """The worker treats an empty structured object as a source failure exactly
    because a tool exception must never read as "no candidates here"."""
    client = McpClient(endpoint, TOKEN)
    client.initialize()
    bad = region()
    bad["x_max"] = bad["x_min"]          # no extent
    response = client.call("vc_find_seed_candidates",
                           {"prediction_uri": URI, "region": bad}, "req-2")
    assert "error" in response
    with pytest.raises(RuntimeError):
        structured(response)


def test_an_empty_region_is_an_answer_not_an_error(endpoint):
    """NO_SEED is a finding. It must not look like a broken source."""
    client = McpClient(endpoint, TOKEN)
    client.initialize()
    away = {f"{a}_{b}": (40 if b == "min" else 60) for a in "xyz" for b in ("min", "max")}
    result = structured(client.call("vc_find_seed_candidates",
                                    {"prediction_uri": URI, "region": away}, "req-3"))
    assert result["candidates"] == []


def test_an_unknown_tool_is_refused(endpoint):
    client = McpClient(endpoint, TOKEN)
    client.initialize()
    response = client.call("vc_delete_everything", {}, "req-4")
    assert "error" in response


def test_a_short_token_is_refused_at_startup(tmp_path):
    """A token nobody can guess is the only thing between this and the network."""
    weak = tmp_path / "token"
    weak.write_text("short")
    with pytest.raises(SystemExit):
        mcp_server.read_token(weak)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
