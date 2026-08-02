from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "framework/stages/06-discovery/scripts/helena_prioritize_first_letters_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("helena_review_priority", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path: Path):
    module = load_module()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "EVIDENCE_MANIFEST.json").write_text("{}\n")
    decisions = [
        {
            "candidate_id": "C-HIGH",
            "retained": True,
            "checks": [
                {"feature": "depth_profile_top3_fraction", "value": 0.25}
            ],
        },
        {
            "candidate_id": "C-STANDARD",
            "retained": True,
            "checks": [
                {"feature": "depth_profile_top3_fraction", "value": 0.14}
            ],
        },
    ]
    (evidence / "CT_FIBER_GATE_DECISIONS.json").write_text(
        json.dumps(decisions) + "\n"
    )
    queue = {
        "schema": module.QUEUE_SCHEMA,
        "queue_id": "queue-1",
        "candidates": [
            {
                "candidate_id": "surface-1:qc-1",
                "sample_id": "PHercTEST",
                "surface_id": "surface-1",
                "qc_job_id": "qc-1",
                "local_manifest": "evidence/EVIDENCE_MANIFEST.json",
                "retained_component_count": 2,
                "retained_candidate_ids": ["C-HIGH", "C-STANDARD"],
                "review_image_paths": [
                    "evidence/C-HIGH-model-context.png",
                    "evidence/C-STANDARD-model-context.png",
                    "evidence/comparison_layers/ct.png",
                ],
            }
        ],
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue) + "\n")
    profile = {
        "kind": module.PROFILE_KIND,
        "requirements": [
            {
                "feature": "depth_profile_top3_fraction",
                "operator": ">=",
                "threshold": 0.18,
            }
        ],
        "decision": {
            "all_requirements_pass": "HIGH",
            "any_requirement_fails": "STANDARD",
        },
        "policy": {"failed_priority_is_still_reviewable": True},
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile) + "\n")
    return module, queue_path, profile_path


def test_priority_preserves_every_retained_component(tmp_path: Path):
    module, queue_path, profile_path = fixture(tmp_path)

    result = module.prioritize(queue_path, profile_path)

    assert result["candidate_count"] == 2
    assert result["high_priority_count"] == 1
    assert result["standard_review_count"] == 1
    assert [row["candidate_id"] for row in result["candidates"]] == [
        "C-HIGH",
        "C-STANDARD",
    ]
    assert result["candidates"][0]["high_priority"] is True
    assert result["candidates"][1]["high_priority"] is False
    assert result["candidates"][1]["review_state"] == "UNREVIEWED"
    assert any(
        path.endswith("C-STANDARD-model-context.png")
        for path in result["candidates"][1]["review_image_paths"]
    )


def test_missing_retained_decision_fails_closed(tmp_path: Path):
    module, queue_path, profile_path = fixture(tmp_path)
    decisions_path = tmp_path / "evidence/CT_FIBER_GATE_DECISIONS.json"
    decisions = json.loads(decisions_path.read_text())
    decisions[1]["retained"] = False
    decisions_path.write_text(json.dumps(decisions) + "\n")

    try:
        module.prioritize(queue_path, profile_path)
    except RuntimeError as error:
        assert "retained decision missing" in str(error)
    else:
        raise AssertionError("missing retained decision was accepted")


def test_stage_registry_resolves_priority_router():
    from scripts.harness.stage_script_registry import resolve_stage_script

    assert resolve_stage_script(ROOT, SCRIPT.name).resolve() == SCRIPT.resolve()
