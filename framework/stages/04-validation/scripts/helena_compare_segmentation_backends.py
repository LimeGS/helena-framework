#!/usr/bin/env python3
"""Evaluate frozen VC3D/ScrollFiesta evidence through gates G2-G5.

The input manifest contains measurements produced by dedicated topology, CT,
flattening, and accounting tools.  This evaluator never infers a missing
measurement, never treats missing as zero, and never fuses physical meshes.
It emits a validated gate contract plus a like-for-like HTML evidence viewer.
When all aggregate comparison values exist it also emits the existing
``campaignx.surface_backend_comparison.v1`` contract.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from framework.contracts.hybrid_surface_contracts import (  # noqa: E402
    HybridContractValidationError,
    validate_hybrid_contract,
)


INPUT_SCHEMA = "campaignx.surface_backend_gate_input.v1"
STATUS_ORDER = {"PASS": 0, "UNMEASURED": 1, "FAIL": 2}
BACKENDS = ("vc3d", "scrollfiesta")
TOPOLOGY_FIELDS = {
    "non_manifold_edges",
    "self_intersections",
    "invalid_coordinates",
    "inconsistent_winding_edges",
    "uv_flipped_triangles",
    "degenerate_triangle_fraction",
    "orientable",
    "hole_repair_count",
    "hole_repair_max_span_voxels",
}
CT_FIELDS = {
    "ct_legible_area_fraction",
    "ct_within_tolerance_fraction",
    "confirmed_bridge_count",
    "confirmed_sheet_change_count",
    "surface_ct_distance_p95_voxels",
}
FLATTEN_FIELDS = {
    "uv_flipped_triangle_count",
    "uv_degenerate_triangle_fraction",
    "stretch_p95",
    "baseline_stretch_p95",
    "confirmed_ct_seam_jump_count",
}
UTILITY_FIELDS = {
    "usable_area_cm2",
    "component_count",
    "downstream_usable_tiles",
    "cost_per_usable_cm2",
    "area_equivalent_for_component_reduction",
    "continuity_gain_equivalent",
}
THRESHOLD_FIELDS = {
    "local_sheet_separation_voxels",
    "maximum_agreement_distance_voxels",
    "minimum_normal_dot",
    "minimum_ct_legible_fraction",
    "minimum_ct_within_tolerance_fraction",
    "maximum_degenerate_fraction",
    "maximum_relative_stretch_regression",
    "minimum_area_gain_fraction",
    "minimum_component_reduction_fraction",
    "maximum_downstream_tile_loss_fraction",
    "maximum_relative_cost_per_cm2",
}
INPUT_FIELDS = {
    "schema",
    "evaluation_id",
    "comparison_id",
    "sample_id",
    "roi_level0_zyx",
    "surfaces",
    "thresholds",
    "measurements",
    "regions",
    "evidence",
}


class EvaluationInputError(ValueError):
    """Raised when a comparison manifest is structurally ambiguous."""


def _strict_keys(
    value: Any, *, allowed: set[str], context: str, required: bool
) -> dict:
    if not isinstance(value, dict):
        raise EvaluationInputError(f"{context} must be an object")
    unknown = set(value) - allowed
    missing = allowed - set(value) if required else set()
    if unknown or missing:
        raise EvaluationInputError(
            f"{context} keys mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _number(
    value: Any,
    *,
    context: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise EvaluationInputError(f"{context} must be finite numeric or null")
    result = float(value)
    if minimum is not None and result < minimum:
        raise EvaluationInputError(f"{context} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise EvaluationInputError(f"{context} must be <= {maximum}")
    return result


def _count(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationInputError(f"{context} must be a non-negative integer or null")
    return value


def _optional_bool(value: Any, *, context: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise EvaluationInputError(f"{context} must be boolean or null")


def _validate_artifact(value: Any, *, context: str) -> None:
    artifact = _strict_keys(
        value, allowed={"uri", "sha256", "bytes"}, context=context, required=True
    )
    uri = artifact["uri"]
    if not isinstance(uri, str) or not urlparse(uri).scheme:
        raise EvaluationInputError(f"{context}/uri must be an absolute URI")
    digest = artifact["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise EvaluationInputError(f"{context}/sha256 must be lowercase 64-hex")
    if (
        isinstance(artifact["bytes"], bool)
        or not isinstance(artifact["bytes"], int)
        or artifact["bytes"] < 1
    ):
        raise EvaluationInputError(f"{context}/bytes must be a positive integer")


def _validate_measurement_group(
    measurements: dict, group: str, fields: set[str]
) -> dict[str, dict[str, Any]]:
    group_value = _strict_keys(
        measurements.get(group),
        allowed=set(BACKENDS),
        context=f"measurements/{group}",
        required=True,
    )
    result = {}
    for backend in BACKENDS:
        result[backend] = _strict_keys(
            group_value[backend],
            allowed=fields,
            context=f"measurements/{group}/{backend}",
            required=False,
        )
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"cannot load manifest {path}: {exc}") from exc
    _strict_keys(manifest, allowed=INPUT_FIELDS, context="manifest", required=True)
    if manifest["schema"] != INPUT_SCHEMA:
        raise EvaluationInputError(f"schema must be {INPUT_SCHEMA}")
    roi = manifest["roi_level0_zyx"]
    if (
        not isinstance(roi, list)
        or len(roi) != 6
        or any(not isinstance(item, int) or item < 0 for item in roi)
        or any(roi[index] >= roi[index + 3] for index in range(3))
    ):
        raise EvaluationInputError("roi_level0_zyx must have positive Z/Y/X extents")

    surfaces = _strict_keys(
        manifest["surfaces"], allowed=set(BACKENDS), context="surfaces", required=True
    )
    surface_fields = {
        "surface_id",
        "backend_profile",
        "artifact_uri",
        "artifact_sha256",
        "receipt_sha256",
    }
    for backend in BACKENDS:
        _strict_keys(
            surfaces[backend],
            allowed=surface_fields,
            context=f"surfaces/{backend}",
            required=True,
        )
    if not str(surfaces["vc3d"]["backend_profile"]).startswith("segmentation.vc3d"):
        raise EvaluationInputError("surfaces/vc3d backend_profile is not VC3D")
    if not str(surfaces["scrollfiesta"]["backend_profile"]).startswith(
        "segmentation.scrollfiesta"
    ):
        raise EvaluationInputError(
            "surfaces/scrollfiesta backend_profile is not ScrollFiesta"
        )

    thresholds = _strict_keys(
        manifest["thresholds"],
        allowed=THRESHOLD_FIELDS,
        context="thresholds",
        required=True,
    )
    for field in THRESHOLD_FIELDS:
        _number(thresholds[field], context=f"thresholds/{field}", minimum=0)
    if float(thresholds["local_sheet_separation_voxels"]) <= 0:
        raise EvaluationInputError("local_sheet_separation_voxels must be positive")
    frozen = {
        "minimum_normal_dot": 0.866,
        "minimum_ct_legible_fraction": 0.95,
        "minimum_ct_within_tolerance_fraction": 0.98,
        "maximum_degenerate_fraction": 0.0001,
        "maximum_relative_stretch_regression": 0.05,
        "minimum_area_gain_fraction": 0.15,
        "minimum_component_reduction_fraction": 0.30,
        "maximum_downstream_tile_loss_fraction": 0.05,
        "maximum_relative_cost_per_cm2": 2.0,
    }
    for field, expected in frozen.items():
        if not math.isclose(
            float(thresholds[field]), expected, rel_tol=0, abs_tol=1e-12
        ):
            raise EvaluationInputError(
                f"thresholds/{field} changed from frozen value {expected}"
            )
    expected_distance = min(
        2.0, 0.5 * float(thresholds["local_sheet_separation_voxels"])
    )
    if not math.isclose(
        float(thresholds["maximum_agreement_distance_voxels"]),
        expected_distance,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise EvaluationInputError(
            "maximum_agreement_distance_voxels must equal min(2, 0.5 * local separation)"
        )

    measurements = _strict_keys(
        manifest["measurements"],
        allowed={"topology", "ct", "flattening", "utility", "continuity_supported"},
        context="measurements",
        required=True,
    )
    _validate_measurement_group(measurements, "topology", TOPOLOGY_FIELDS)
    _validate_measurement_group(measurements, "ct", CT_FIELDS)
    _validate_measurement_group(measurements, "flattening", FLATTEN_FIELDS)
    _validate_measurement_group(measurements, "utility", UTILITY_FIELDS)
    _optional_bool(
        measurements["continuity_supported"],
        context="measurements/continuity_supported",
    )
    if not isinstance(manifest["regions"], list) or not manifest["regions"]:
        raise EvaluationInputError("regions must be a non-empty array")
    if not isinstance(manifest["evidence"], list) or not manifest["evidence"]:
        raise EvaluationInputError("evidence must be a non-empty array")
    for index, pair in enumerate(manifest["evidence"]):
        pair = _strict_keys(
            pair,
            allowed={"panel", "vc3d", "scrollfiesta"},
            context=f"evidence/{index}",
            required=True,
        )
        if pair["panel"] not in {"CT", "MESH", "NORMALS", "SEAMS", "FLATTENING"}:
            raise EvaluationInputError(f"evidence/{index}/panel is invalid")
        if pair["vc3d"] is None and pair["scrollfiesta"] is None:
            raise EvaluationInputError(f"evidence/{index} has no backend artifact")
        for backend in BACKENDS:
            if pair[backend] is not None:
                _validate_artifact(pair[backend], context=f"evidence/{index}/{backend}")
    return manifest


def _measurement(manifest: dict, group: str, backend: str, field: str) -> Any:
    return manifest["measurements"][group][backend].get(field)


def _requirement(
    metric: str,
    observed: Any,
    operator: str,
    threshold: Any,
    predicate: Callable[[Any], bool],
    *,
    pass_reason: str,
    fail_reason: str,
) -> dict[str, Any]:
    if observed is None:
        status, reason = "UNMEASURED", f"{metric} was not measured"
    else:
        status = "PASS" if predicate(observed) else "FAIL"
        reason = pass_reason if status == "PASS" else fail_reason
    return {
        "metric": metric,
        "status": status,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "reason": reason,
    }


def _gate(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    status = max((row["status"] for row in requirements), key=STATUS_ORDER.__getitem__)
    return {"status": status, "requirements": requirements}


def evaluate_g2(manifest: dict) -> dict:
    sf = manifest["measurements"]["topology"]["scrollfiesta"]
    vc = manifest["measurements"]["topology"]["vc3d"]
    limit = manifest["thresholds"]["maximum_degenerate_fraction"]
    requirements = []
    for field in (
        "non_manifold_edges",
        "self_intersections",
        "invalid_coordinates",
        "inconsistent_winding_edges",
        "uv_flipped_triangles",
    ):
        value = sf.get(field)
        _count(value, context=f"measurements/topology/scrollfiesta/{field}")
        requirements.append(
            _requirement(
                f"scrollfiesta.{field}",
                value,
                "EQ",
                0,
                lambda item: item == 0,
                pass_reason=f"{field}=0",
                fail_reason=f"{field} must be zero",
            )
        )
    degenerate = sf.get("degenerate_triangle_fraction")
    vc_degenerate = vc.get("degenerate_triangle_fraction")
    _number(
        degenerate, context="scrollfiesta degenerate fraction", minimum=0, maximum=1
    )
    _number(vc_degenerate, context="vc3d degenerate fraction", minimum=0, maximum=1)
    requirements.append(
        _requirement(
            "scrollfiesta.degenerate_triangle_fraction",
            degenerate,
            "LE",
            limit,
            lambda value: value <= limit,
            pass_reason="degenerate fraction is within the absolute gate",
            fail_reason="degenerate fraction exceeds 0.01%",
        )
    )
    comparison_observed = (
        None
        if degenerate is None or vc_degenerate is None
        else degenerate - vc_degenerate
    )
    requirements.append(
        _requirement(
            "scrollfiesta.degenerate_fraction_minus_vc3d",
            comparison_observed,
            "LE",
            0.0,
            lambda value: value <= 0,
            pass_reason="ScrollFiesta degeneracy does not exceed VC3D",
            fail_reason="ScrollFiesta degeneracy exceeds VC3D",
        )
    )
    orientable = _optional_bool(sf.get("orientable"), context="scrollfiesta orientable")
    requirements.append(
        _requirement(
            "scrollfiesta.orientable",
            orientable,
            "EQ",
            True,
            lambda value: value is True,
            pass_reason="all promoted components are orientable",
            fail_reason="one or more components are non-orientable",
        )
    )
    repair_count = sf.get("hole_repair_count")
    repair_span = sf.get("hole_repair_max_span_voxels")
    _count(repair_count, context="scrollfiesta hole_repair_count")
    _number(repair_span, context="scrollfiesta hole_repair_max_span_voxels", minimum=0)
    maximum_span = 0.5 * manifest["thresholds"]["local_sheet_separation_voxels"]
    if repair_count is None:
        repair_observed, repair_predicate = None, lambda value: False
    elif repair_count == 0:
        repair_observed, repair_predicate = 0.0, lambda value: True
    else:
        repair_observed, repair_predicate = (
            repair_span,
            lambda value: value <= maximum_span,
        )
    requirements.append(
        _requirement(
            "scrollfiesta.hole_repair_max_span_voxels",
            repair_observed,
            "LE",
            maximum_span,
            repair_predicate,
            pass_reason="hole repairs do not bridge more than half a local sheet gap",
            fail_reason="a hole repair bridges too much of the local sheet gap",
        )
    )
    return _gate(requirements)


def evaluate_g3(manifest: dict) -> dict:
    sf = manifest["measurements"]["ct"]["scrollfiesta"]
    thresholds = manifest["thresholds"]
    specs = [
        (
            "ct_legible_area_fraction",
            "GE",
            thresholds["minimum_ct_legible_fraction"],
            lambda v, t: v >= t,
        ),
        (
            "ct_within_tolerance_fraction",
            "GE",
            thresholds["minimum_ct_within_tolerance_fraction"],
            lambda v, t: v >= t,
        ),
        ("confirmed_bridge_count", "EQ", 0, lambda v, t: v == 0),
        ("confirmed_sheet_change_count", "EQ", 0, lambda v, t: v == 0),
        (
            "surface_ct_distance_p95_voxels",
            "LE",
            thresholds["maximum_agreement_distance_voxels"],
            lambda v, t: v <= t,
        ),
    ]
    rows = []
    for field, operator, threshold, predicate in specs:
        value = sf.get(field)
        if field.endswith("_fraction"):
            _number(
                value,
                context=f"measurements/ct/scrollfiesta/{field}",
                minimum=0,
                maximum=1,
            )
        elif field.endswith("_count"):
            _count(value, context=f"measurements/ct/scrollfiesta/{field}")
        else:
            _number(value, context=f"measurements/ct/scrollfiesta/{field}", minimum=0)
        rows.append(
            _requirement(
                f"scrollfiesta.{field}",
                value,
                operator,
                threshold,
                lambda observed, threshold=threshold, predicate=predicate: predicate(
                    observed, threshold
                ),
                pass_reason=f"{field} satisfies the frozen CT gate",
                fail_reason=f"{field} violates the frozen CT gate",
            )
        )
    return _gate(rows)


def evaluate_g4(manifest: dict) -> dict:
    sf = manifest["measurements"]["flattening"]["scrollfiesta"]
    thresholds = manifest["thresholds"]
    rows = []
    for field in ("uv_flipped_triangle_count", "confirmed_ct_seam_jump_count"):
        value = sf.get(field)
        _count(value, context=f"measurements/flattening/scrollfiesta/{field}")
        rows.append(
            _requirement(
                f"scrollfiesta.{field}",
                value,
                "EQ",
                0,
                lambda observed: observed == 0,
                pass_reason=f"{field}=0",
                fail_reason=f"{field} must be zero",
            )
        )
    degenerate = sf.get("uv_degenerate_triangle_fraction")
    _number(
        degenerate,
        context="scrollfiesta UV degenerate fraction",
        minimum=0,
        maximum=1,
    )
    rows.append(
        _requirement(
            "scrollfiesta.uv_degenerate_triangle_fraction",
            degenerate,
            "LE",
            thresholds["maximum_degenerate_fraction"],
            lambda observed: observed <= thresholds["maximum_degenerate_fraction"],
            pass_reason="UV degeneracy is within 0.01%",
            fail_reason="UV degeneracy exceeds 0.01%",
        )
    )
    stretch = sf.get("stretch_p95")
    baseline = sf.get("baseline_stretch_p95")
    _number(stretch, context="scrollfiesta stretch_p95", minimum=0)
    _number(baseline, context="scrollfiesta baseline_stretch_p95", minimum=0)
    stretch_ratio = (
        None
        if stretch is None or baseline is None or baseline <= 0
        else stretch / baseline
    )
    maximum_ratio = 1.0 + thresholds["maximum_relative_stretch_regression"]
    rows.append(
        _requirement(
            "scrollfiesta.stretch_p95_ratio_to_valid_baseline",
            stretch_ratio,
            "LE",
            maximum_ratio,
            lambda observed: observed <= maximum_ratio,
            pass_reason="stretch p95 regresses no more than 5%",
            fail_reason="stretch p95 regresses more than 5%",
        )
    )
    return _gate(rows)


def _hard_defect_total(manifest: dict, backend: str) -> float | None:
    values = [
        _measurement(manifest, "topology", backend, field)
        for field in (
            "non_manifold_edges",
            "self_intersections",
            "invalid_coordinates",
            "inconsistent_winding_edges",
            "uv_flipped_triangles",
        )
    ]
    if any(value is None for value in values):
        return None
    for index, value in enumerate(values):
        _count(value, context=f"{backend} hard defect metric {index}")
    return float(sum(values))


def _confirmed_issue_total(manifest: dict, backend: str) -> float | None:
    values = [
        _measurement(manifest, "ct", backend, "confirmed_bridge_count"),
        _measurement(manifest, "ct", backend, "confirmed_sheet_change_count"),
    ]
    if any(value is None for value in values):
        return None
    for index, value in enumerate(values):
        _count(value, context=f"{backend} confirmed issue metric {index}")
    return float(sum(values))


def evaluate_g5(manifest: dict, prereqs: dict[str, dict]) -> dict:
    utility = manifest["measurements"]["utility"]
    sf, vc = utility["scrollfiesta"], utility["vc3d"]
    thresholds = manifest["thresholds"]
    rows = []
    prerequisite_statuses = [prereqs[name]["status"] for name in ("G2", "G3", "G4")]
    prereq = (
        "FAIL"
        if "FAIL" in prerequisite_statuses
        else ("UNMEASURED" if "UNMEASURED" in prerequisite_statuses else "PASS")
    )
    rows.append(
        {
            "metric": "scrollfiesta.g2_g3_g4_prerequisite",
            "status": prereq,
            "observed": prereq,
            "operator": "EQ",
            "threshold": "PASS",
            "reason": "G5 counts only area that passes G2-G4",
        }
    )

    sf_area = _number(
        sf.get("usable_area_cm2"), context="scrollfiesta usable area", minimum=0
    )
    vc_area = _number(vc.get("usable_area_cm2"), context="vc3d usable area", minimum=0)
    area_gain = (
        None
        if sf_area is None or vc_area is None or vc_area <= 0
        else sf_area / vc_area - 1.0
    )
    sf_components = _count(
        sf.get("component_count"), context="scrollfiesta component count"
    )
    vc_components = _count(vc.get("component_count"), context="vc3d component count")
    component_reduction = (
        None
        if sf_components is None or vc_components is None or vc_components <= 0
        else 1.0 - sf_components / vc_components
    )
    equivalent = _optional_bool(
        sf.get("area_equivalent_for_component_reduction"),
        context="scrollfiesta area_equivalent_for_component_reduction",
    )
    area_pass = (
        area_gain is not None and area_gain >= thresholds["minimum_area_gain_fraction"]
    )
    component_pass = (
        component_reduction is not None
        and equivalent is True
        and component_reduction >= thresholds["minimum_component_reduction_fraction"]
    )
    area_branch_known_false = area_gain is not None and not area_pass
    component_branch_known_false = equivalent is False or (
        component_reduction is not None
        and component_reduction < thresholds["minimum_component_reduction_fraction"]
    )
    if area_pass:
        gain_status, gain_branch = "PASS", "AREA_GAIN"
    elif component_pass:
        gain_status, gain_branch = "PASS", "COMPONENT_REDUCTION"
    elif area_branch_known_false and component_branch_known_false:
        gain_status, gain_branch = "FAIL", "NONE"
    else:
        gain_status, gain_branch = "UNMEASURED", None
    rows.append(
        {
            "metric": "scrollfiesta.useful_gain_branch",
            "status": gain_status,
            "observed": gain_branch,
            "operator": "OR",
            "threshold": "area_gain>=0.15 OR equivalent_component_reduction>=0.30",
            "reason": f"area_gain={area_gain!r}; component_reduction={component_reduction!r}; area_equivalent={equivalent!r}",
        }
    )

    sf_tiles = _count(sf.get("downstream_usable_tiles"), context="scrollfiesta tiles")
    vc_tiles = _count(vc.get("downstream_usable_tiles"), context="vc3d tiles")
    tile_ratio = (
        None
        if sf_tiles is None or vc_tiles is None or vc_tiles <= 0
        else sf_tiles / vc_tiles
    )
    minimum_tile_ratio = 1.0 - thresholds["maximum_downstream_tile_loss_fraction"]
    rows.append(
        _requirement(
            "scrollfiesta.downstream_usable_tile_ratio",
            tile_ratio,
            "GE",
            minimum_tile_ratio,
            lambda observed: observed >= minimum_tile_ratio,
            pass_reason="downstream usable tile loss is no more than 5%",
            fail_reason="downstream usable tile loss exceeds 5%",
        )
    )

    sf_defects, vc_defects = _hard_defect_total(
        manifest, "scrollfiesta"
    ), _hard_defect_total(manifest, "vc3d")
    defect_delta = (
        None if sf_defects is None or vc_defects is None else sf_defects - vc_defects
    )
    rows.append(
        _requirement(
            "scrollfiesta.hard_defect_delta_vs_vc3d",
            defect_delta,
            "NO_INCREASE",
            0,
            lambda observed: observed <= 0,
            pass_reason="hard defects do not increase",
            fail_reason="hard defects increase",
        )
    )
    sf_issues, vc_issues = _confirmed_issue_total(
        manifest, "scrollfiesta"
    ), _confirmed_issue_total(manifest, "vc3d")
    issue_delta = (
        None if sf_issues is None or vc_issues is None else sf_issues - vc_issues
    )
    rows.append(
        _requirement(
            "scrollfiesta.confirmed_bridge_sheet_change_delta",
            issue_delta,
            "NO_INCREASE",
            0,
            lambda observed: observed <= 0,
            pass_reason="confirmed bridges/sheet changes do not increase",
            fail_reason="confirmed bridges/sheet changes increase",
        )
    )

    sf_cost = _number(
        sf.get("cost_per_usable_cm2"), context="scrollfiesta cost", minimum=0
    )
    vc_cost = _number(vc.get("cost_per_usable_cm2"), context="vc3d cost", minimum=0)
    equivalent_continuity = _optional_bool(
        sf.get("continuity_gain_equivalent"),
        context="scrollfiesta continuity_gain_equivalent",
    )
    cost_ratio = (
        None
        if sf_cost is None or vc_cost is None or vc_cost <= 0
        else sf_cost / vc_cost
    )
    if cost_ratio is None:
        cost_status = "UNMEASURED"
    elif (
        cost_ratio <= thresholds["maximum_relative_cost_per_cm2"]
        or equivalent_continuity is True
    ):
        cost_status = "PASS"
    elif (
        equivalent_continuity is None
        and cost_ratio > thresholds["maximum_relative_cost_per_cm2"]
    ):
        cost_status = "UNMEASURED"
    else:
        cost_status = "FAIL"
    rows.append(
        {
            "metric": "scrollfiesta.cost_ratio_or_equivalent_continuity",
            "status": cost_status,
            "observed": cost_ratio,
            "operator": "OR",
            "threshold": thresholds["maximum_relative_cost_per_cm2"],
            "reason": f"cost_ratio={cost_ratio!r}; continuity_gain_equivalent={equivalent_continuity!r}",
        }
    )
    return _gate(rows)


def classify_region(region: dict, thresholds: dict) -> dict:
    allowed = {
        "region_id",
        "area_cm2",
        "vc3d_component_ids",
        "scrollfiesta_component_ids",
        "distance_p95_voxels",
        "normal_dot_p05",
        "same_sheet",
        "ct_support",
        "backend_region_validity",
        "notes",
    }
    _strict_keys(
        region,
        allowed=allowed,
        context=f"region/{region.get('region_id')}",
        required=True,
    )
    vc_ids, sf_ids = region["vc3d_component_ids"], region["scrollfiesta_component_ids"]
    if (
        not isinstance(vc_ids, list)
        or not isinstance(sf_ids, list)
        or not (vc_ids or sf_ids)
    ):
        raise EvaluationInputError("each region needs at least one backend component")
    area = _number(region["area_cm2"], context="region area", minimum=0)
    if area is None or area <= 0:
        raise EvaluationInputError("region area_cm2 must be positive")
    distance = _number(
        region["distance_p95_voxels"], context="region distance", minimum=0
    )
    normal = _number(
        region["normal_dot_p05"], context="region normal", minimum=-1, maximum=1
    )
    same_sheet = _optional_bool(region["same_sheet"], context="region same_sheet")
    support = _strict_keys(
        region["ct_support"],
        allowed=set(BACKENDS),
        context="region ct_support",
        required=True,
    )
    validity = _strict_keys(
        region["backend_region_validity"],
        allowed=set(BACKENDS),
        context="region backend_region_validity",
        required=True,
    )
    for value in (*support.values(),):
        if value not in {"SUPPORTED", "UNVERIFIED", "CONTRADICTED"}:
            raise EvaluationInputError("region ct_support value is invalid")
    for value in validity.values():
        if value not in {"PASS", "FAIL", "UNMEASURED"}:
            raise EvaluationInputError(
                "region backend_region_validity value is invalid"
            )
    notes = region["notes"]
    if (
        not isinstance(notes, list)
        or not notes
        or any(not isinstance(item, str) or not item for item in notes)
    ):
        raise EvaluationInputError("region notes must be a non-empty string array")

    reasons = list(notes)
    if vc_ids and sf_ids:
        missing = (
            distance is None
            or normal is None
            or same_sheet is None
            or "UNMEASURED" in validity.values()
            or "UNVERIFIED" in support.values()
        )
        passes = (
            not missing
            and validity["vc3d"] == validity["scrollfiesta"] == "PASS"
            and support["vc3d"] == support["scrollfiesta"] == "SUPPORTED"
            and same_sheet is True
            and distance <= thresholds["maximum_agreement_distance_voxels"]
            and normal >= thresholds["minimum_normal_dot"]
        )
        if passes:
            classification, status = "CONSENSUS", "PASS"
            disposition, ct = "ELIGIBLE_FOR_PROMOTION", "SUPPORTED"
            reasons.append(
                "both backends agree in distance, normal, CT support, and sheet identity"
            )
        else:
            classification = "DISAGREEMENT"
            status = "UNMEASURED" if missing else "FAIL"
            disposition = "AUDIT_REQUIRED"
            ct = "CONTRADICTED" if "CONTRADICTED" in support.values() else "UNVERIFIED"
            reasons.append(
                "overlapping backend regions do not satisfy every consensus condition"
            )
    else:
        backend = "vc3d" if vc_ids else "scrollfiesta"
        other = "scrollfiesta" if vc_ids else "vc3d"
        valid = validity[backend]
        supported = support[backend]
        if valid == "PASS" and supported == "SUPPORTED":
            classification = "VC3D_ONLY" if backend == "vc3d" else "SCROLLFIESTA_ONLY"
            status, disposition, ct = "PASS", "KEEP_PROVISIONAL", "SUPPORTED"
            reasons.append(f"only {backend} covers this measured, CT-supported region")
        else:
            classification = "DISAGREEMENT"
            status = (
                "UNMEASURED"
                if valid == "UNMEASURED" or supported == "UNVERIFIED"
                else "FAIL"
            )
            disposition = "AUDIT_REQUIRED"
            ct = "CONTRADICTED" if supported == "CONTRADICTED" else "UNVERIFIED"
            reasons.append(
                f"{backend}-only region is not eligible; {other} has no component"
            )
    return {
        "region_id": region["region_id"],
        "classification": classification,
        "measurement_status": status,
        "area_cm2": area,
        "vc3d_component_ids": vc_ids,
        "scrollfiesta_component_ids": sf_ids,
        "distance_p95_voxels": distance,
        "normal_dot_p05": normal,
        "ct_support": ct,
        "reasons": reasons,
        "disposition": disposition,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _comparison_contract(
    manifest: dict, regions: list[dict], generated_at: str
) -> dict | None:
    paired = [
        row
        for row in regions
        if row["vc3d_component_ids"] and row["scrollfiesta_component_ids"]
    ]
    if (
        not paired
        or any(
            row["distance_p95_voxels"] is None or row["normal_dot_p05"] is None
            for row in paired
        )
        or manifest["measurements"]["continuity_supported"] is None
    ):
        return None
    distances = [row["distance_p95_voxels"] for row in paired]
    normals = [row["normal_dot_p05"] for row in paired]
    classes = ("CONSENSUS", "SCROLLFIESTA_ONLY", "VC3D_ONLY", "DISAGREEMENT")
    areas = {
        name: sum(row["area_cm2"] for row in regions if row["classification"] == name)
        for name in classes
    }
    counts = {
        name: sum(row["classification"] == name for row in regions) for name in classes
    }
    total_area = sum(areas.values())
    supported_area = sum(
        row["area_cm2"] for row in regions if row["ct_support"] == "SUPPORTED"
    )
    evidence_roles = {
        "CT": "CT_OVERLAY",
        "MESH": "TOPOLOGY_REPORT",
        "NORMALS": "NORMAL_AGREEMENT",
        "SEAMS": "DISTANCE_MAP",
        "FLATTENING": "COMPARISON_VIEWER",
    }
    evidence = []
    for pair in manifest["evidence"]:
        for backend in BACKENDS:
            artifact = pair.get(backend)
            if artifact is not None:
                evidence.append({"role": evidence_roles[pair["panel"]], **artifact})
    if not evidence:
        return None
    if areas["DISAGREEMENT"] > 0:
        decision = "DISAGREEMENT_REQUIRES_AUDIT"
    elif areas["CONSENSUS"] > 0:
        decision = "CONSENSUS_AVAILABLE"
    else:
        decision = "PROVISIONAL_ONLY"
    contract = {
        "schema": "campaignx.surface_backend_comparison.v1",
        "comparison_id": manifest["comparison_id"],
        "generated_at_utc": generated_at,
        "sample_id": manifest["sample_id"],
        "roi_level0_zyx": manifest["roi_level0_zyx"],
        "surfaces": manifest["surfaces"],
        "thresholds": {
            "local_sheet_separation_voxels": manifest["thresholds"][
                "local_sheet_separation_voxels"
            ],
            "maximum_agreement_distance_voxels": manifest["thresholds"][
                "maximum_agreement_distance_voxels"
            ],
            "minimum_normal_dot": manifest["thresholds"]["minimum_normal_dot"],
            "ct_support_required": True,
        },
        "aggregate_metrics": {
            "common_area_cm2": areas["CONSENSUS"],
            "scrollfiesta_only_area_cm2": areas["SCROLLFIESTA_ONLY"],
            "vc3d_only_area_cm2": areas["VC3D_ONLY"],
            "disagreement_area_cm2": areas["DISAGREEMENT"],
            "bidirectional_distance_voxels": {
                "p50": _percentile(distances, 0.50),
                "p95": _percentile(distances, 0.95),
                "p99": _percentile(distances, 0.99),
            },
            "normal_dot_p50": _percentile(normals, 0.50),
            "normal_dot_p05": _percentile(normals, 0.05),
            "ct_supported_fraction": supported_area / total_area,
            "continuity_supported": manifest["measurements"]["continuity_supported"],
        },
        "regions": [
            {key: value for key, value in row.items() if key != "measurement_status"}
            for row in regions
        ],
        "summary": {"area_by_class_cm2": areas, "region_count_by_class": counts},
        "decision": decision,
        "physical_mesh_fusion_performed": False,
        "evidence": evidence,
        "ink_used": False,
    }
    validate_hybrid_contract(
        contract, expected_contract="campaignx.surface_backend_comparison.v1"
    )
    return contract


def _artifact(path: Path) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "uri": path.resolve().as_uri(),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def _evidence_img(artifact: dict | None, alt: str) -> str:
    if artifact is None:
        return '<div class="missing">Evidencia no disponible</div>'
    uri = html.escape(artifact["uri"], quote=True)
    return f'<img src="{uri}" alt="{html.escape(alt, quote=True)}" loading="lazy">'


def build_html(evaluation: dict, destination: Path) -> None:
    gate_cards = "".join(
        f'<section class="gate {gate["status"].lower()}"><h2>{name} · {gate["status"]}</h2>'
        + "".join(
            f'<p><b>{html.escape(row["metric"])}</b>: {html.escape(str(row["observed"]))} '
            f'<span>{html.escape(row["reason"])}</span></p>'
            for row in gate["requirements"]
        )
        + "</section>"
        for name, gate in evaluation["gates"].items()
    )
    panels = "".join(
        f'<section class="panel"><h2>{html.escape(pair["panel"])}</h2><div class="pair">'
        f'<article><h3>VC3D</h3>{_evidence_img(pair["vc3d"], pair["panel"] + " VC3D")}</article>'
        f'<article><h3>ScrollFiesta</h3>{_evidence_img(pair["scrollfiesta"], pair["panel"] + " ScrollFiesta")}</article>'
        "</div></section>"
        for pair in evaluation["evidence"]
    )
    region_rows = "".join(
        f'<tr><td>{html.escape(row["region_id"])}</td><td>{row["classification"]}</td>'
        f'<td>{row["measurement_status"]}</td><td>{row["area_cm2"]:.4f}</td>'
        f'<td>{html.escape("; ".join(row["reasons"]))}</td></tr>'
        for row in evaluation["regions"]
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Helena Framework - VC3D / ScrollFiesta comparison</title>
<style>
:root{{--bg:#08101d;--card:#111c2d;--line:#2d405d;--text:#eaf1fb;--muted:#aebbd0;--pass:#38d996;--fail:#ff6b72;--unknown:#f5bf57}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px system-ui,sans-serif}}
header,main{{max-width:1800px;margin:auto;padding:18px}} header{{position:sticky;top:0;background:#08101df2;border-bottom:1px solid var(--line);z-index:2}}
h1{{margin:0 0 5px}} .notice{{padding:12px;border:1px solid var(--unknown);border-radius:10px;color:#ffe2a0}}
.gates{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0}} .gate,.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}
.gate h2{{margin:0 0 8px}} .gate p{{font-size:13px;margin:5px 0}} .gate span{{color:var(--muted)}} .gate.pass{{border-color:var(--pass)}} .gate.fail{{border-color:var(--fail)}} .gate.unmeasured{{border-color:var(--unknown)}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} article{{min-width:0}} article h3{{margin:0;padding:7px;background:#17243a}}
img,.missing{{display:block;width:100%;height:min(38vh,520px);object-fit:contain;background:#000}} .missing{{display:grid;place-items:center;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;margin-top:14px}} th,td{{border:1px solid var(--line);padding:8px;text-align:left}} th{{background:#17243a}}
@media(max-width:900px){{.gates{{grid-template-columns:1fr 1fr}}.pair{{grid-template-columns:1fr}}}} @media(max-width:550px){{.gates{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>VC3D ↔ ScrollFiesta · {html.escape(evaluation['sample_id'])}</h1>
<div>Result: <b>{evaluation['overall_status']}</b> - recommendation: <b>{evaluation['recommendation']}</b></div></header>
<main><p class="notice"><b>No physical fusion.</b> UNMEASURED never passes a gate. This viewer compares evidence; it certifies neither ink nor text.</p>
<div class="gates">{gate_cards}</div>{panels}<table><thead><tr><th>Region</th><th>Class</th><th>Measurement</th><th>Area cm2</th><th>Reasones</th></tr></thead><tbody>{region_rows}</tbody></table></main></body></html>"""
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(document)


def evaluate_manifest(manifest_path: Path, output_dir: Path) -> tuple[Path, Path]:
    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        raise EvaluationInputError("output_dir must be absolute")
    if output_dir.exists():
        raise EvaluationInputError(f"immutable output_dir already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    generated_at = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    g2, g3, g4 = evaluate_g2(manifest), evaluate_g3(manifest), evaluate_g4(manifest)
    gates = {"G2": g2, "G3": g3, "G4": g4}
    gates["G5"] = evaluate_g5(manifest, gates)
    overall = max(
        (gate["status"] for gate in gates.values()), key=STATUS_ORDER.__getitem__
    )
    geometry = [gates[name]["status"] for name in ("G2", "G3", "G4")]
    if all(status == "PASS" for status in geometry):
        recommendation = (
            "PILOT_PASS"
            if gates["G5"]["status"] == "PASS"
            else (
                "KEEP_EXPERIMENTAL"
                if gates["G5"]["status"] == "FAIL"
                else "BLOCKED_UNMEASURED"
            )
        )
    elif "FAIL" in geometry:
        recommendation = "REJECT_FOR_TARGET"
    else:
        recommendation = "BLOCKED_UNMEASURED"
    regions = [
        classify_region(row, manifest["thresholds"]) for row in manifest["regions"]
    ]
    comparison = _comparison_contract(manifest, regions, generated_at)
    comparison_artifact = None
    if comparison is not None:
        comparison_path = output_dir / "SURFACE_BACKEND_COMPARISON.json"
        _write_json(comparison_path, comparison)
        comparison_artifact = _artifact(comparison_path)
    evaluation = {
        "schema": "campaignx.surface_backend_gate_evaluation.v1",
        "evaluation_id": manifest["evaluation_id"],
        "comparison_id": manifest["comparison_id"],
        "generated_at_utc": generated_at,
        "sample_id": manifest["sample_id"],
        "roi_level0_zyx": manifest["roi_level0_zyx"],
        "surfaces": manifest["surfaces"],
        "thresholds": manifest["thresholds"],
        "gates": gates,
        "overall_status": overall,
        "recommendation": recommendation,
        "regions": regions,
        "evidence": manifest["evidence"],
        "comparison_contract": comparison_artifact,
        "physical_mesh_fusion_performed": False,
        "ink_used": False,
    }
    validate_hybrid_contract(
        evaluation, expected_contract="campaignx.surface_backend_gate_evaluation.v1"
    )
    evaluation_path = output_dir / "SURFACE_BACKEND_GATE_EVALUATION.json"
    _write_json(evaluation_path, evaluation)
    viewer_path = output_dir / "SURFACE_BACKEND_COMPARISON.html"
    build_html(evaluation, viewer_path)
    return evaluation_path, viewer_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluation, viewer = evaluate_manifest(args.manifest, args.output_dir)
    except (EvaluationInputError, HybridContractValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(evaluation)
    print(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
