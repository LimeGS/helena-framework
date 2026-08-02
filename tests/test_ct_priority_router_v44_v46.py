from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework" / "stages" / "04-validation" / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v46 = load("helena_develop_ct_priority_router_v46")
v7 = load("helena_execute_ct_priority_transfer_v7_once")
v47 = load("helena_develop_ct_priority_router_v47")
v8 = load("helena_execute_ct_priority_transfer_v8_once")


def test_v46_preserve_all_never_drops_evidence() -> None:
    row = {
        "scroll_id": "PHerc0139",
        "rank_features": {"r3_p90": 0.0},
    }
    route = v7.route_row(
        row,
        {"PHerc0139": {"mode": "B1_PRESERVE_ALL"}},
    )
    assert route["tier"] == "B1"
    assert route["score"] is None


def test_v46_unknown_scroll_fails_safe() -> None:
    route = v7.route_row(
        {"scroll_id": "PHercUnknown", "rank_features": {}},
        {},
    )
    assert route["router_mode"] == "B1_PRESERVE_ALL"
    assert route["tier"] == "B1"


def test_v46_calibrated_route_is_strictly_greater_than_threshold() -> None:
    router = {
        "PHerc1667": {
            "mode": "CALIBRATED_B1_B2",
            "feature_name": "r41_maximum",
            "threshold": 0.947,
        }
    }
    equal = v7.route_row(
        {"scroll_id": "PHerc1667", "rank_features": {"r41_maximum": 0.947}},
        router,
    )
    above = v7.route_row(
        {"scroll_id": "PHerc1667", "rank_features": {"r41_maximum": 0.948}},
        router,
    )
    assert equal["tier"] == "B2"
    assert above["tier"] == "B1"


def test_v7_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "claimed"
    output.mkdir()
    (output / "EXECUTION_CLAIM.json").write_text("{}")
    with pytest.raises(RuntimeError, match="refusing to rerun"):
        v7.execute(
            tmp_path / "development.json",
            tmp_path / "features.json",
            tmp_path / "controls.json",
            output,
        )


def test_v7_summary_applies_efficiency_only_to_calibrated_scrolls() -> None:
    routed = [
        {
            "scroll_id": "A",
            "expected_class": "POSITIVE",
            "tier": "B1",
            "router_mode": "B1_PRESERVE_ALL",
        },
        {
            "scroll_id": "A",
            "expected_class": "CONFOUND",
            "tier": "B1",
            "router_mode": "B1_PRESERVE_ALL",
        },
        {
            "scroll_id": "B",
            "expected_class": "POSITIVE",
            "tier": "B1",
            "router_mode": "CALIBRATED_B1_B2",
        },
        {
            "scroll_id": "B",
            "expected_class": "CONFOUND",
            "tier": "B2",
            "router_mode": "CALIBRATED_B1_B2",
        },
    ]
    result = v7.summarize(routed)
    assert result["A"]["confound_b2_rate"] == 0.0
    assert result["A"]["router_mode"] == "B1_PRESERVE_ALL"
    assert result["B"]["confound_b2_rate"] == 1.0


@needs_campaign_evidence
def test_v46_development_receipt_matches_expected_profile() -> None:
    receipt = json.loads(
        (
            ROOT
            / "workspace"
            / "campaigns"
            / "campaign-x-2026"
            / "findings"
            / "ct-priority-router-v46"
            / "development"
            / "DEVELOPMENT_RECEIPT.json"
        ).read_text()
    )
    assert receipt["profile_id"] == v46.PROFILE_ID
    assert receipt["router"]["unknown_surface_mode"] == "B1_PRESERVE_ALL"


def test_v47_keeps_only_prospectively_passing_calibrators() -> None:
    development = {
        "profile_id": v46.PROFILE_ID,
        "router": {
            "by_scroll": {
                "failed": {
                    "mode": "CALIBRATED_B1_B2",
                    "feature_name": "r3_p90",
                    "threshold": 0.5,
                },
                "passed": {
                    "mode": "CALIBRATED_B1_B2",
                    "feature_name": "r3_p90",
                    "threshold": 0.5,
                },
            }
        },
    }
    result = {
        "benchmark_id": "SURFACE_CALIBRATION_TRANSFER_V7",
        "status": "FAILED",
        "metrics": {
            "by_scroll": {
                "failed": {
                    "positive_b1_recall": 0.48,
                    "confound_b2_rate": 0.88,
                },
                "passed": {
                    "positive_b1_recall": 0.98,
                    "confound_b2_rate": 0.90,
                },
            }
        },
    }
    frozen = v47.build(
        development,
        result,
        development_sha256="a" * 64,
        v7_result_sha256="b" * 64,
    )
    assert frozen["router"]["by_scroll"]["failed"]["mode"] == "B1_PRESERVE_ALL"
    assert frozen["router"]["by_scroll"]["passed"]["mode"] == "CALIBRATED_B1_B2"


def test_v8_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "claimed"
    output.mkdir()
    (output / "EXECUTION_CLAIM.json").write_text("{}")
    with pytest.raises(RuntimeError, match="refusing to rerun"):
        v8.execute(
            tmp_path / "development.json",
            tmp_path / "features.json",
            tmp_path / "controls.json",
            output,
        )
