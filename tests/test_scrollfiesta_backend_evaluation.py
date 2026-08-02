from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "framework/stages/04-validation/scripts/helena_compare_segmentation_backends.py"
)
FIXTURE = ROOT / "tests/fixtures/hybrid_backend_evaluation/pass_manifest.json"
sys.path.insert(0, str(ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location("helena_compare_backends", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()

from framework.contracts.hybrid_surface_contracts import (  # noqa: E402
    HybridContractValidationError,
    validate_hybrid_contract,
)


def manifest() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def run(tmp_path: Path, value: dict, name: str = "result") -> tuple[dict, Path]:
    source = tmp_path / f"{name}.manifest.json"
    source.write_text(json.dumps(value), encoding="utf-8")
    output = (tmp_path / name).resolve()
    evaluation_path, viewer = MODULE.evaluate_manifest(source, output)
    return json.loads(evaluation_path.read_text(encoding="utf-8")), viewer


def requirement(evaluation: dict, gate: str, metric: str) -> dict:
    return next(
        row
        for row in evaluation["gates"][gate]["requirements"]
        if row["metric"] == metric
    )


def test_all_measured_pass_emits_both_contracts_and_like_for_like_html(
    tmp_path: Path,
) -> None:
    evaluation, viewer = run(tmp_path, manifest())
    assert evaluation["overall_status"] == "PASS"
    assert evaluation["recommendation"] == "PILOT_PASS"
    assert {name: gate["status"] for name, gate in evaluation["gates"].items()} == {
        "G2": "PASS",
        "G3": "PASS",
        "G4": "PASS",
        "G5": "PASS",
    }
    assert evaluation["physical_mesh_fusion_performed"] is False
    assert evaluation["ink_used"] is False
    assert evaluation["comparison_contract"] is not None
    comparison_path = viewer.parent / "SURFACE_BACKEND_COMPARISON.json"
    comparison = json.loads(comparison_path.read_text())
    validate_hybrid_contract(
        comparison, expected_contract="campaignx.surface_backend_comparison.v1"
    )
    assert comparison["physical_mesh_fusion_performed"] is False
    assert [row["classification"] for row in evaluation["regions"]] == [
        "CONSENSUS",
        "SCROLLFIESTA_ONLY",
    ]
    page = viewer.read_text(encoding="utf-8")
    assert "VC3D ↔ ScrollFiesta" in page
    assert "No physical fusion" in page
    assert "s3://fixture/vc3d/ct.png" in page
    assert "s3://fixture/scrollfiesta/ct.png" in page


@pytest.mark.parametrize(
    ("group", "field", "gate"),
    [
        ("topology", "self_intersections", "G2"),
        ("ct", "ct_legible_area_fraction", "G3"),
        ("flattening", "stretch_p95", "G4"),
    ],
)
def test_missing_critical_metric_is_unmeasured_and_cannot_pass(
    tmp_path: Path, group: str, field: str, gate: str
) -> None:
    value = manifest()
    value["measurements"][group]["scrollfiesta"].pop(field)
    evaluation, _ = run(tmp_path, value, f"missing-{gate.lower()}")
    assert evaluation["gates"][gate]["status"] == "UNMEASURED"
    assert evaluation["overall_status"] == "UNMEASURED"
    assert evaluation["recommendation"] == "BLOCKED_UNMEASURED"
    assert any(
        row["status"] == "UNMEASURED"
        for row in evaluation["gates"][gate]["requirements"]
    )


@pytest.mark.parametrize(
    ("group", "field", "bad_value", "gate"),
    [
        ("topology", "self_intersections", 1, "G2"),
        ("ct", "confirmed_bridge_count", 1, "G3"),
        ("flattening", "uv_flipped_triangle_count", 1, "G4"),
    ],
)
def test_measured_hard_defect_fails_closed(
    tmp_path: Path, group: str, field: str, bad_value: int, gate: str
) -> None:
    value = manifest()
    value["measurements"][group]["scrollfiesta"][field] = bad_value
    evaluation, _ = run(tmp_path, value, f"fail-{gate.lower()}")
    assert evaluation["gates"][gate]["status"] == "FAIL"
    assert evaluation["overall_status"] == "FAIL"
    assert evaluation["recommendation"] == "REJECT_FOR_TARGET"


def test_valid_geometry_but_insufficient_gain_stays_experimental(
    tmp_path: Path,
) -> None:
    value = manifest()
    utility = value["measurements"]["utility"]["scrollfiesta"]
    utility["usable_area_cm2"] = 1.05
    utility["component_count"] = 9
    evaluation, _ = run(tmp_path, value, "g5-fail")
    assert evaluation["gates"]["G2"]["status"] == "PASS"
    assert evaluation["gates"]["G3"]["status"] == "PASS"
    assert evaluation["gates"]["G4"]["status"] == "PASS"
    assert evaluation["gates"]["G5"]["status"] == "FAIL"
    assert evaluation["recommendation"] == "KEEP_EXPERIMENTAL"


def test_g5_or_branch_remains_unmeasured_when_false_area_branch_has_unknown_alternative(
    tmp_path: Path,
) -> None:
    value = manifest()
    utility = value["measurements"]["utility"]["scrollfiesta"]
    utility["usable_area_cm2"] = 1.05
    utility.pop("component_count")
    evaluation, _ = run(tmp_path, value, "g5-unmeasured")
    row = requirement(evaluation, "G5", "scrollfiesta.useful_gain_branch")
    assert row["status"] == "UNMEASURED"
    assert evaluation["gates"]["G5"]["status"] == "UNMEASURED"


def test_missing_overlap_metrics_prevents_comparison_contract(tmp_path: Path) -> None:
    value = manifest()
    value["regions"][0]["distance_p95_voxels"] = None
    value["regions"][0]["same_sheet"] = None
    evaluation, _ = run(tmp_path, value, "no-comparison")
    assert evaluation["comparison_contract"] is None
    assert evaluation["regions"][0]["classification"] == "DISAGREEMENT"
    assert evaluation["regions"][0]["measurement_status"] == "UNMEASURED"
    assert not (tmp_path / "no-comparison/SURFACE_BACKEND_COMPARISON.json").exists()


def test_region_conflict_is_disagreement_not_mesh_fusion(tmp_path: Path) -> None:
    value = manifest()
    value["regions"][0]["distance_p95_voxels"] = 3.0
    evaluation, _ = run(tmp_path, value, "region-conflict")
    region = evaluation["regions"][0]
    assert region["classification"] == "DISAGREEMENT"
    assert region["measurement_status"] == "FAIL"
    assert region["disposition"] == "AUDIT_REQUIRED"
    assert evaluation["physical_mesh_fusion_performed"] is False


def test_threshold_changes_and_output_overwrite_are_rejected(tmp_path: Path) -> None:
    changed = manifest()
    changed["thresholds"]["minimum_ct_legible_fraction"] = 0.90
    source = tmp_path / "changed.json"
    source.write_text(json.dumps(changed))
    with pytest.raises(MODULE.EvaluationInputError, match="changed from frozen"):
        MODULE.evaluate_manifest(source, (tmp_path / "changed").resolve())

    good = manifest()
    evaluation, _ = run(tmp_path, good, "immutable")
    assert evaluation["overall_status"] == "PASS"
    source = tmp_path / "second.json"
    source.write_text(json.dumps(good))
    with pytest.raises(MODULE.EvaluationInputError, match="already exists"):
        MODULE.evaluate_manifest(source, (tmp_path / "immutable").resolve())


def test_gate_contract_semantics_reject_status_laundering(tmp_path: Path) -> None:
    evaluation, _ = run(tmp_path, manifest(), "semantic")
    invalid = copy.deepcopy(evaluation)
    invalid["gates"]["G2"]["requirements"][0]["status"] = "FAIL"
    with pytest.raises(HybridContractValidationError, match="requirements imply"):
        validate_hybrid_contract(
            invalid, expected_contract="campaignx.surface_backend_gate_evaluation.v1"
        )
