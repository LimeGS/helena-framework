from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework"
    / "stages"
    / "04-validation"
    / "scripts"
    / "helena_instrument_u0_threshold_chain.py"
)
SPEC = importlib.util.spec_from_file_location("u0_threshold_chain", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_u0_reports_destructive_and_non_destructive_retention(tmp_path: Path) -> None:
    analysis = tmp_path / "robust_lane" / "analysis" / "INK_STABILITY_ANALYSIS.json"
    write(
        analysis,
        {
            "text_like_screening": {
                "glyph_like_candidate_count": 9,
                "rows_with_at_least_four_candidates": 1,
                "screening_outcome": "INSUFFICIENT_TEXT_LIKE_SUPPORT",
            }
        },
    )
    gate_dir = tmp_path / "high-recall" / "ct_application" / "gate"
    decisions = [
        {
            "group_id": "g1",
            "candidate_id": "c1",
            "retained": False,
            "failed_features": ["depth_profile_peak_count"],
            "checks": [
                {
                    "feature": "depth_profile_peak_count",
                    "value": 9,
                    "operator": "<=",
                    "threshold": 8,
                    "passed": False,
                }
            ],
        }
    ]
    decisions_path = gate_dir / "CT_FIBER_GATE_DECISIONS.json"
    write(decisions_path, decisions)
    write(
        gate_dir / "CT_FIBER_GATE_EVALUATION.json",
        {
            "artifacts": {
                "decisions": decisions_path.name,
                "decisions_sha256": MODULE.sha256(decisions_path),
            }
        },
    )
    shadow_dir = (
        tmp_path / "high-recall" / "ct_application" / "shadow_router_v4"
    )
    routes = [
        {
            "group_id": "g1",
            "candidate_id": "c1",
            "shadow_tier": "TIER_B_SHADOW_REVIEW",
            "required_action": "PRESERVE_AND_REVIEW_IN_SHADOW_QUEUE",
            "routing_reason": "localization failure is non-destructive",
            "not_discarded": True,
        }
    ]
    routes_path = shadow_dir / "CT_GATE_V4_SHADOW_ROUTES.json"
    write(routes_path, routes)
    write(
        shadow_dir / "CT_GATE_V4_SHADOW_RECEIPT.json",
        {
            "artifacts": {
                "routes": routes_path.name,
                "routes_sha256": MODULE.sha256(routes_path),
            }
        },
    )

    result = MODULE.inspect_u0(tmp_path)

    assert result["strict_screen"]["retained"] is False
    assert result["aggregate"]["v3_retained_count"] == 0
    assert result["aggregate"]["strict_destructive_chain_retained_count"] == 0
    assert result["aggregate"]["v4_1_preserved_count"] == 1
    assert result["aggregate"]["v3_failed_feature_counts"] == {
        "depth_profile_peak_count": 1
    }
    assert result["components"][0]["decision_points"]["ct_fiber_gate_v3"] == [
        {
            "decision_id": "v3.depth_profile_peak_count",
            "operator": "<=",
            "passed": False,
            "threshold": 8,
            "value": 9,
        }
    ]
