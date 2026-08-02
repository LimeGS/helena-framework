from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/04-validation/scripts/helena_build_ct_gate_near_miss_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("helena_near_miss", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(candidate_id: str, *, value: float, failed: bool = True):
    return {
        "candidate_id": candidate_id,
        "group_id": "surface-1",
        "retained": False,
        "decision": "CT_DEPTH_DIFFUSE_DOWNRANK_LIKELY_FIBER_LAMINAR_CONFOUND",
        "failed_features": ["depth_profile_top3_fraction"] if failed else [],
        "checks": [
            {
                "feature": "depth_profile_top3_fraction",
                "value": value,
                "threshold": 0.12,
                "operator": ">=",
                "passed": not failed,
            }
        ],
    }


def test_near_misses_are_ranked_by_normalized_gate_margin():
    module = load_module()
    rows = [
        decision("far", value=0.06),
        decision("closest", value=0.116),
        decision("middle", value=0.09),
    ]
    selected = module.select_near_misses(rows, max_failed_checks=1, limit=2)

    assert [row["candidate_id"] for row in selected] == ["closest", "middle"]
    assert round(selected[0]["normalized_gate_violation"], 6) == 0.033333
    assert all(row["retained"] is False for row in selected)


def test_retained_and_non_failing_rows_never_enter_near_miss_audit():
    module = load_module()
    retained = decision("retained", value=0.13, failed=False)
    retained["retained"] = True
    rows = [retained, decision("no-failure", value=0.13, failed=False)]

    assert module.select_near_misses(rows, max_failed_checks=1, limit=5) == []


def test_stage_registry_resolves_near_miss_review():
    from scripts.harness.stage_script_registry import resolve_stage_script

    assert resolve_stage_script(ROOT, SCRIPT.name).resolve() == SCRIPT.resolve()
