"""Fail-closed validation for the hybrid segmentation interchange contracts.

JSON Schema owns structural validation.  The checks in this module cover the
small set of cross-field invariants that JSON Schema cannot express clearly.
Unknown contract names, unknown fields, unresolved schemas, and semantic
inconsistencies are errors; callers must never treat them as warnings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_DIRECTORY = Path(__file__).resolve().parent / "schemas"
SCHEMA_FILES = {
    "campaignx.surface_artifact.v2": "surface_artifact.v2.schema.json",
    "campaignx.segmentation_backend_run.v1": "segmentation_backend_run.v1.schema.json",
    "campaignx.surface_backend_comparison.v1": "surface_backend_comparison.v1.schema.json",
    "campaignx.surface_backend_gate_evaluation.v1": "surface_backend_gate_evaluation.v1.schema.json",
}
_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:AUTH|CREDENTIAL|KEY|PASSWORD|SECRET|SESSION|TOKEN)", re.IGNORECASE
)


class HybridContractValidationError(ValueError):
    """Raised when a hybrid segmentation document is not safe to consume."""


def _load_schema(contract: str) -> dict[str, Any]:
    try:
        path = SCHEMA_DIRECTORY / SCHEMA_FILES[contract]
    except KeyError as exc:
        raise HybridContractValidationError(f"unknown contract: {contract!r}") from exc
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HybridContractValidationError(
            f"cannot load schema for {contract!r}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise HybridContractValidationError(f"schema for {contract!r} is not an object")
    try:
        Draft202012Validator.check_schema(value)
    except Exception as exc:  # jsonschema exposes validator-specific subclasses.
        raise HybridContractValidationError(
            f"invalid schema for {contract!r}: {exc}"
        ) from exc
    return value


def _schema_errors(contract: str, document: Mapping[str, Any]) -> list[str]:
    validator = Draft202012Validator(
        _load_schema(contract), format_checker=FormatChecker()
    )
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    rendered: list[str] = []
    for error in errors:
        path = "/".join(str(part) for part in error.absolute_path) or "<root>"
        rendered.append(f"{path}: {error.message}")
    return rendered


def _positive_roi(roi: list[int]) -> bool:
    return all(roi[index] < roi[index + 3] for index in range(3))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _surface_semantics(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    roi = document["roi_level0_zyx"]
    if not _positive_roi(roi):
        errors.append("roi_level0_zyx: each lower Z/Y/X bound must be below its upper bound")

    artifacts = document["artifacts"]
    mesh_keys = ("source_mesh_obj", "canonical_mesh_obj")
    present_meshes = [artifacts[key] is not None for key in mesh_keys]
    if any(present_meshes) and not all(present_meshes):
        errors.append(
            "artifacts: source_mesh_obj and canonical_mesh_obj must both be present or both be null"
        )
    if document["backend_family"] == "scrollfiesta" and not all(present_meshes):
        errors.append("artifacts: ScrollFiesta artifacts require both source and canonical OBJ meshes")
    family = document["backend_family"]
    if not document["backend_profile"].startswith(f"segmentation.{family}-"):
        errors.append("backend_profile: profile family must match backend_family")
    transform = document["coordinate_transform"]
    if family == "scrollfiesta" and (
        transform["source_order"] != "ZYX" or not transform["winding_flipped"]
    ):
        errors.append(
            "coordinate_transform: ScrollFiesta requires source_order ZYX and one winding flip"
        )
    if document["validation_state"] == "GEOMETRY_VALIDATED":
        metrics = document["metrics"]
        zero_fields = (
            "invalid_coordinate_count",
            "non_manifold_edge_count",
            "self_intersection_count",
            "inconsistent_winding_edge_count",
        )
        # ``UNMEASURED`` is a string, so it is neither zero nor a defect count.
        # Reject it explicitly instead of letting ``!= 0`` decide, so the reason
        # a promotion was refused stays legible in the error.
        unmeasured = [
            field for field in zero_fields if not isinstance(metrics[field], int)
        ]
        if unmeasured:
            errors.append(
                "metrics: GEOMETRY_VALIDATED requires a measured value for "
                + ", ".join(sorted(unmeasured))
            )
        if any(
            isinstance(metrics[field], int) and metrics[field] != 0
            for field in zero_fields
        ):
            errors.append(
                "metrics: GEOMETRY_VALIDATED requires zero hard topology defects"
            )
        if not metrics["orientable"] or metrics["degenerate_triangle_fraction"] > 0.0001:
            errors.append(
                "metrics: GEOMETRY_VALIDATED requires orientable geometry and "
                "degenerate_triangle_fraction <= 0.0001"
            )
        # The profile declares hard gates on UV flips and stretch.  A surface
        # that never measured them cannot be promoted on the strength of the
        # topology metrics alone.
        distortion_fields = (
            "uv_flipped_triangle_count",
            "uv_degenerate_triangle_count",
            "stretch_p50",
            "stretch_p95",
            "stretch_max",
        )
        missing_distortion = [field for field in distortion_fields if field not in metrics]
        if missing_distortion:
            errors.append(
                "metrics: GEOMETRY_VALIDATED requires post-flattening distortion "
                "measurements: " + ", ".join(sorted(missing_distortion))
            )
        elif metrics["uv_flipped_triangle_count"] or metrics[
            "uv_degenerate_triangle_count"
        ]:
            errors.append(
                "metrics: GEOMETRY_VALIDATED requires zero flipped and zero "
                "degenerate UV triangles"
            )
        elif not (
            metrics["stretch_p50"] <= metrics["stretch_p95"] <= metrics["stretch_max"]
        ):
            errors.append(
                "metrics: require stretch_p50 <= stretch_p95 <= stretch_max"
            )
    return errors


def _run_semantics(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _positive_roi(document["inputs"]["roi_level0_zyx"]):
        errors.append(
            "inputs/roi_level0_zyx: each lower Z/Y/X bound must be below its upper bound"
        )
    family = document["backend"]["family"]
    if not document["backend"]["profile"].startswith(f"segmentation.{family}-"):
        errors.append("backend/profile: profile family must match backend/family")

    for variable in document["environment_non_sensitive"]:
        if _SENSITIVE_ENVIRONMENT_NAME.search(variable["name"]):
            errors.append(
                f"environment_non_sensitive: sensitive variable name is forbidden: {variable['name']}"
            )
    environment_names = [row["name"] for row in document["environment_non_sensitive"]]
    if len(environment_names) != len(set(environment_names)):
        errors.append("environment_non_sensitive: variable names must be unique")

    command_hash = _sha256_text(_canonical_json(document["command"]["argv"]))
    if document["command"]["argv_sha256"] != command_hash:
        errors.append("command/argv_sha256: does not hash canonical argv JSON")
    for parameter in document["parameters"]:
        try:
            parsed = json.loads(parameter["value_json"])
        except json.JSONDecodeError:
            errors.append(f"parameters/{parameter['name']}: value_json is not JSON")
            continue
        canonical = _canonical_json(parsed)
        if parameter["value_json"] != canonical:
            errors.append(f"parameters/{parameter['name']}: value_json is not canonical JSON")
        if parameter["value_sha256"] != _sha256_text(parameter["value_json"]):
            errors.append(f"parameters/{parameter['name']}: value_sha256 mismatch")
    parameter_names = [row["name"] for row in document["parameters"]]
    if len(parameter_names) != len(set(parameter_names)):
        errors.append("parameters: parameter names must be unique")

    start = datetime.fromisoformat(document["started_at_utc"].replace("Z", "+00:00"))
    finish = datetime.fromisoformat(document["finished_at_utc"].replace("Z", "+00:00"))
    if finish < start:
        errors.append("finished_at_utc: cannot precede started_at_utc")

    status = document["status"]
    output = document["surface_artifact"]
    error = document["error"]
    if status == "SUCCEEDED":
        if output is None:
            errors.append("surface_artifact: required when status is SUCCEEDED")
        if error is not None:
            errors.append("error: must be null when status is SUCCEEDED")
    else:
        if output is not None:
            errors.append("surface_artifact: must be null unless status is SUCCEEDED")
        if error is None:
            errors.append("error: required when status is not SUCCEEDED")
    return errors


def _comparison_semantics(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _positive_roi(document["roi_level0_zyx"]):
        errors.append("roi_level0_zyx: each lower Z/Y/X bound must be below its upper bound")

    left = document["surfaces"]["vc3d"]
    right = document["surfaces"]["scrollfiesta"]
    if left["surface_id"] == right["surface_id"]:
        errors.append("surfaces: VC3D and ScrollFiesta surface_id values must differ")
    if left["artifact_sha256"] == right["artifact_sha256"]:
        errors.append("surfaces: backend artifact_sha256 values must differ")
    if not left["backend_profile"].startswith("segmentation.vc3d"):
        errors.append("surfaces/vc3d/backend_profile: must identify the VC3D family")
    if not right["backend_profile"].startswith("segmentation.scrollfiesta"):
        errors.append(
            "surfaces/scrollfiesta/backend_profile: must identify the ScrollFiesta family"
        )

    thresholds = document["thresholds"]
    local_gap = thresholds["local_sheet_separation_voxels"]
    expected_distance = min(2.0, 0.5 * local_gap)
    if not math.isclose(
        thresholds["maximum_agreement_distance_voxels"],
        expected_distance,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        errors.append(
            "thresholds/maximum_agreement_distance_voxels: must equal "
            "min(2, 0.5 * local_sheet_separation_voxels)"
        )

    observed = {name: 0.0 for name in document["summary"]["area_by_class_cm2"]}
    observed_counts = {
        name: 0 for name in document["summary"]["region_count_by_class"]
    }
    for region in document["regions"]:
        observed[region["classification"]] += region["area_cm2"]
        observed_counts[region["classification"]] += 1
        classification = region["classification"]
        vc3d_ids = region["vc3d_component_ids"]
        scroll_ids = region["scrollfiesta_component_ids"]
        if classification == "CONSENSUS" and (not vc3d_ids or not scroll_ids):
            errors.append(
                f"regions/{region['region_id']}: CONSENSUS requires components from both backends"
            )
        if classification == "SCROLLFIESTA_ONLY" and (vc3d_ids or not scroll_ids):
            errors.append(
                f"regions/{region['region_id']}: SCROLLFIESTA_ONLY requires only ScrollFiesta components"
            )
        if classification == "VC3D_ONLY" and (not vc3d_ids or scroll_ids):
            errors.append(
                f"regions/{region['region_id']}: VC3D_ONLY requires only VC3D components"
            )
    for classification, declared in document["summary"]["area_by_class_cm2"].items():
        if not math.isclose(declared, observed[classification], rel_tol=0.0, abs_tol=1e-8):
            errors.append(
                f"summary/area_by_class_cm2/{classification}: declared {declared} "
                f"but regions sum to {observed[classification]}"
            )
    for classification, declared in document["summary"]["region_count_by_class"].items():
        if declared != observed_counts[classification]:
            errors.append(
                f"summary/region_count_by_class/{classification}: declared {declared} "
                f"but regions contain {observed_counts[classification]}"
            )
    distances = document["aggregate_metrics"]["bidirectional_distance_voxels"]
    if not distances["p50"] <= distances["p95"] <= distances["p99"]:
        errors.append("aggregate_metrics/bidirectional_distance_voxels: require p50 <= p95 <= p99")
    aggregate_to_class = {
        "common_area_cm2": "CONSENSUS",
        "scrollfiesta_only_area_cm2": "SCROLLFIESTA_ONLY",
        "vc3d_only_area_cm2": "VC3D_ONLY",
        "disagreement_area_cm2": "DISAGREEMENT",
    }
    for metric, classification in aggregate_to_class.items():
        declared = document["aggregate_metrics"][metric]
        if not math.isclose(declared, observed[classification], rel_tol=0.0, abs_tol=1e-8):
            errors.append(
                f"aggregate_metrics/{metric}: declared {declared} but regions sum to "
                f"{observed[classification]}"
            )
    return errors


def _summarize_requirement_statuses(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if "UNMEASURED" in statuses:
        return "UNMEASURED"
    return "PASS"


def _gate_evaluation_semantics(document: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _positive_roi(document["roi_level0_zyx"]):
        errors.append("roi_level0_zyx: each lower Z/Y/X bound must be below its upper bound")
    gate_statuses: dict[str, str] = {}
    for gate_name, gate in document["gates"].items():
        expected = _summarize_requirement_statuses(
            [requirement["status"] for requirement in gate["requirements"]]
        )
        if gate["status"] != expected:
            errors.append(
                f"gates/{gate_name}/status: declared {gate['status']} but requirements imply {expected}"
            )
        gate_statuses[gate_name] = gate["status"]
    expected_overall = _summarize_requirement_statuses(list(gate_statuses.values()))
    if document["overall_status"] != expected_overall:
        errors.append(
            f"overall_status: declared {document['overall_status']} but gates imply {expected_overall}"
        )

    geometry_statuses = [gate_statuses[name] for name in ("G2", "G3", "G4")]
    if all(status == "PASS" for status in geometry_statuses):
        if gate_statuses["G5"] == "PASS":
            expected_recommendation = "PILOT_PASS"
        elif gate_statuses["G5"] == "FAIL":
            expected_recommendation = "KEEP_EXPERIMENTAL"
        else:
            expected_recommendation = "BLOCKED_UNMEASURED"
    elif "FAIL" in geometry_statuses:
        expected_recommendation = "REJECT_FOR_TARGET"
    else:
        expected_recommendation = "BLOCKED_UNMEASURED"
    if document["recommendation"] != expected_recommendation:
        errors.append(
            f"recommendation: declared {document['recommendation']} but gates imply "
            f"{expected_recommendation}"
        )
    return errors


_SEMANTIC_VALIDATORS = {
    "campaignx.surface_artifact.v2": _surface_semantics,
    "campaignx.segmentation_backend_run.v1": _run_semantics,
    "campaignx.surface_backend_comparison.v1": _comparison_semantics,
    "campaignx.surface_backend_gate_evaluation.v1": _gate_evaluation_semantics,
}


def validate_hybrid_contract(
    document: Mapping[str, Any], *, expected_contract: str | None = None
) -> None:
    """Validate a document or raise :class:`HybridContractValidationError`.

    ``expected_contract`` prevents a structurally valid document of the wrong
    kind from crossing a stage boundary.  When omitted, the document's exact
    ``schema`` discriminator selects the contract.
    """

    if not isinstance(document, Mapping):
        raise HybridContractValidationError("document must be a JSON object")
    contract = document.get("schema")
    if not isinstance(contract, str):
        raise HybridContractValidationError("schema: required string discriminator")
    if expected_contract is not None and contract != expected_contract:
        raise HybridContractValidationError(
            f"schema: expected {expected_contract!r}, received {contract!r}"
        )
    if contract not in SCHEMA_FILES:
        raise HybridContractValidationError(f"unknown contract: {contract!r}")

    errors = _schema_errors(contract, document)
    if not errors:
        errors.extend(_SEMANTIC_VALIDATORS[contract](document))
    if errors:
        raise HybridContractValidationError("contract validation failed:\n- " + "\n- ".join(errors))


def validate_hybrid_contract_file(
    path: Path, *, expected_contract: str | None = None
) -> dict[str, Any]:
    """Load and validate one UTF-8 JSON contract file, failing closed."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HybridContractValidationError(f"cannot load contract {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise HybridContractValidationError("document must be a JSON object")
    validate_hybrid_contract(document, expected_contract=expected_contract)
    return document
