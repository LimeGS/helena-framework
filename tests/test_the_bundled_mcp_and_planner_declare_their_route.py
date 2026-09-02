"""Two contracts the seed path relies on, pinned where they live.

These were the tail of a file that otherwise tested an agent skill directory
shipped with the repository. That directory was internal tooling and is gone;
these two never depended on it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = ROOT / "framework/stages/01-segmentation"
if str(STAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MCP_SERVER = _load("vc3d_mcp_server", STAGE_ROOT / "mcp/server.py")
RECOVERY = _load("geometry_recovery",
                 STAGE_ROOT / "scripts/build_geometry_recovery_v1.py")


def test_bundled_mcp_threshold_is_declared_and_recovery_reads_its_xyz() -> None:
    properties = MCP_SERVER.TOOLS[0]["inputSchema"]["properties"]
    assert properties["threshold"]["minimum"] == 0.0
    assert properties["threshold"]["maximum"] == 1.0
    expected = {"x": 11, "y": 22, "z": 33}
    assert RECOVERY.candidate_coordinate({"ct_l0_coordinate": expected}) == expected
    assert RECOVERY.candidate_coordinate(expected) == expected
    assert RECOVERY.candidate_coordinate({"coordinate": expected}) == expected


def test_active_cost_aware_profile_declares_probe_winner_route() -> None:
    profile = json.loads(
        (ROOT / "framework/profiles/01-segmentation/"
         "segmentation-planner-cost-aware-v2-1.0.0.json").read_text(encoding="utf-8"))
    assert profile["decision_order"][0] == "DETERMINISTIC_PROBE_WINNER"
