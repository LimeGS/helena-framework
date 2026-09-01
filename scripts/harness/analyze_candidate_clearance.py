#!/usr/bin/env python3
"""Reconstruct a candidate-clearance terminal without changing policy or state."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from itertools import product
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse


SCHEMA = "campaignx.candidate_clearance_review.v1"
INPUT_SCHEMA = "campaignx.candidate_clearance_review_input.v1"
CAUSE_CELL = "INSUFFICIENT_CELL_INTERIOR_CLEARANCE"
CAUSE_VOLUME = "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE"
SENSITIVITY_PERCENTAGES = (100, 75, 50, 25, 0)
NON_CLAIM = "This diagnostic does not establish the absence of ink, text, or letters."
OUTPUT_FIELDS = {
    "schema", "sample_id", "task_id", "attempt_id", "source_snapshot_id",
    "coordinate_frame", "shape_xyz", "voxel_size_um", "input_sha256",
    "trusted_manifest_sha256", "input_evidence_sha256", "clearance_policy",
    "candidates", "counts",
    "classification", "classification_basis", "defects", "sensitivity",
    "policy_change_authorized", "state_mutation", "non_claim", "review_sha256",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _coordinate(candidate: dict[str, Any]) -> tuple[float, float, float]:
    value = candidate.get("ct_l0_coordinate")
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        raise ValueError("candidate coordinate must contain exactly x, y, z")
    if any(isinstance(value[axis], bool) or type(value[axis]) is not int for axis in "xyz"):
        raise ValueError("candidate coordinate must be integral CT-L0 voxels")
    return tuple(float(value[axis]) for axis in "xyz")


def _axis_distances(
    coordinate: tuple[float, float, float],
    low: tuple[float, float, float],
    high: tuple[float, float, float],
    *,
    inclusive_high: bool,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, axis in enumerate("xyz"):
        result[f"{axis}_low"] = coordinate[index] - low[index]
        high_face = high[index] - coordinate[index]
        result[f"{axis}_high"] = high_face - 1.0 if inclusive_high else high_face
    return result


def _validate_triplet(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain three values in xyz order")
    if any(isinstance(row, bool) or type(row) is not int for row in value):
        raise ValueError(f"{label} must contain exact integer voxels")
    return tuple(float(row) for row in value)


def _require_fields(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - actual)} "
            f"extra={sorted(actual - fields)}"
        )
    return value


def _require_allowed_fields(
    value: Any, required: set[str], optional: set[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if not required <= actual or not actual <= required | optional:
        raise ValueError(
            f"{label} fields differ: missing={sorted(required - actual)} "
            f"extra={sorted(actual - required - optional)}"
        )
    return value


def _validate_read_set(value: Any, label: str) -> None:
    row = _require_fields(
        value, {"schema", "objects", "canonical_manifest_sha256"}, label
    )
    if row["schema"] != "campaignx.first_letters_source_read_set.v1":
        raise ValueError(f"{label} schema is unsupported")
    if not isinstance(row["objects"], list):
        raise ValueError(f"{label}.objects must be an array")
    for index, item in enumerate(row["objects"]):
        _require_fields(item, {"object_key", "sha256", "bytes"},
                        f"{label}.objects[{index}]")
        if (not isinstance(item["object_key"], str)
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in item["sha256"])
                or type(item["bytes"]) is not int or item["bytes"] < 0):
            raise ValueError(f"{label} contains an invalid object identity")
    if row["canonical_manifest_sha256"] != _sha256(row["objects"]):
        raise ValueError(f"{label} manifest hash differs")


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_integer_range(value: Any, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")
    return value


def _validate_xyz_mapping(
    value: Any, label: str, *, low: int | None = None,
    high: int | None = None,
) -> dict[str, int]:
    row = _require_fields(value, {"x", "y", "z"}, label)
    if any(type(row[axis]) is not int for axis in "xyz"):
        raise ValueError(f"{label} must contain exact integer xyz values")
    if (low is not None and any(row[axis] < low for axis in "xyz")):
        raise ValueError(f"{label} is below its valid range")
    if (high is not None and any(row[axis] > high for axis in "xyz")):
        raise ValueError(f"{label} is above its valid range")
    return row


def _validate_region(
    value: Any, label: str, *, shape: tuple[float, float, float],
    radius_high: int = 192,
) -> dict[str, Any]:
    row = _require_fields(value, {"center", "radius"}, label)
    center = _validate_xyz_mapping(row["center"], f"{label} center")
    radius = _validate_xyz_mapping(
        row["radius"], f"{label} radius", low=1, high=radius_high)
    if any(center[axis] < 0 or center[axis] >= shape[index]
           for index, axis in enumerate("xyz")):
        raise ValueError(f"{label} center is outside the P0 volume")
    return row


def _validate_source_reachability(source: dict[str, Any]) -> list[str]:
    value = source.get("source_reachable")
    if value is None:
        return []
    reachable = _require_fields(
        value, {"checked_at_utc", "ct", "m7"}, "source reachability")
    timestamp = reachable["checked_at_utc"]
    if (not isinstance(timestamp, str)
            or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
                r"(?:\.\d+)?(?:Z|\+00:00)",
                timestamp,
            ) is None):
        raise ValueError("source reachability timestamp is not RFC3339 UTC")
    try:
        instant = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "source reachability timestamp is not RFC3339 UTC") from error
    if instant.utcoffset() != dt.timedelta(0):
        raise ValueError("source reachability timestamp is not RFC3339 UTC")

    defects: list[str] = []
    for role in ("ct", "m7"):
        probe = reachable[role]
        if not isinstance(probe, dict):
            raise ValueError(f"source reachability {role} must be an object")
        uri = _require_nonempty_string(
            probe.get("uri"), f"source reachability {role} URI")
        checked = probe.get("checked")
        if type(checked) is not bool:
            raise ValueError(f"source reachability {role} checked flag differs")
        is_http = urlparse(uri).scheme.lower() in {"http", "https"}
        if checked is False:
            _require_fields(
                probe, {"uri", "checked", "reason"},
                f"source reachability {role}")
            if is_http or probe["reason"] != "not an http(s) URI":
                raise ValueError(
                    f"source reachability {role} unchecked state differs")
        else:
            if not is_http or type(probe.get("reachable")) is not bool:
                raise ValueError(
                    f"source reachability {role} checked state differs")
            status = probe.get("status")
            error = probe.get("error")
            if probe["reachable"] is True:
                _require_fields(
                    probe, {"uri", "checked", "reachable", "status"},
                    f"source reachability {role}")
                if type(status) is not int or not 200 <= status <= 399:
                    raise ValueError(
                        f"source reachability {role} reachable status differs")
            elif "status" in probe:
                _require_fields(
                    probe,
                    {"uri", "checked", "reachable", "status", "error"},
                    f"source reachability {role}")
                if (type(status) is not int or not 400 <= status <= 599
                        or not isinstance(error, str) or not error):
                    raise ValueError(
                        f"source reachability {role} HTTP failure differs")
            else:
                _require_fields(
                    probe, {"uri", "checked", "reachable", "error"},
                    f"source reachability {role}")
                if not isinstance(error, str) or not error:
                    raise ValueError(
                        f"source reachability {role} transport failure differs")
        if uri != source[f"{role}_uri"]:
            defects.append(f"source reachability {role} URI differs")
    return defects


CAMPAIGN_EXECUTION_FIELDS = {
    "sample_id", "mission_id", "source_snapshot_id",
    "p0_artifact_id", "p0_artifact_sha256",
    "p0_selection_version", "p0_selection_sha256",
    "catalog_snapshot_sha256", "source_content_lock_sha256",
    "ct_sha256", "m7_sha256", "m7_uri_sha256", "coordinate_frame",
    "voxel_size_um", "shape_xyz", "grid_version", "policy_version",
    "provider", "m7_threshold", "grid_step", "query_radius",
    "parallelism", "maximum_cells", "selection_strategy",
    "candidate_selection_policy", "seed_region_policy", "code_revision",
    "gates", "queue_execution",
}
CAMPAIGN_GATE_FIELDS = {
    "cell_clearance", "volume_clearance", "candidate_interior_clearance",
    "packet_candidate_limit", "ct_material_support_gate",
}
CAMPAIGN_CT_GATE_FIELDS = {
    "policy", "level", "radius_l0_voxels", "minimum_nonzero_voxels",
}
CAMPAIGN_QUEUE_FIELDS = {
    "parameter_envelope", "planner", "planner_model", "prediction_space",
    "minimum_separation_voxels", "recenter_probe_max_candidates",
    "recenter_radius_xyz", "seed_probe_mode", "seed_probe_top_k",
    "seed_probe_generations", "verify_sources",
}
SEED_REGION_POLICIES = {
    "fixed-v1", "m7-recenter-z-v1", "m7-recenter-xyz-v1",
    "m7-recenter-z-chunk-safe-v1", "m7-chunk-safe-merge-interior-v2",
}


def _expected_campaign_cell_geometry(
    cell_id: Any, execution: dict[str, Any], gates: dict[str, Any],
) -> tuple[dict[str, int], list[list[int]]]:
    match = re.fullmatch(r"r(\d{5})c(\d{5})a(\d{5})", str(cell_id))
    if match is None:
        raise ValueError("campaign execution cell ID is malformed")
    shape = execution["shape_xyz"]
    step = execution["grid_step"]
    radius = execution["query_radius"]
    margin = max(radius, gates["volume_clearance"])
    axes: list[list[int]] = []
    for size in shape:
        centers = list(range(margin, size - margin, step))
        if centers:
            last = size - margin - 1
            if last - centers[-1] >= step // 2:
                centers.append(last)
        axes.append(centers)
    indices = [int(row) for row in match.groups()]
    if any(index >= len(axes[axis])
           for axis, index in enumerate(indices)):
        raise ValueError("campaign execution cell is outside the frozen grid")
    coordinates = [axes[axis][index] for axis, index in enumerate(indices)]
    center = dict(zip("xyz", coordinates, strict=True))
    bounds = [
        [coordinate - radius for coordinate in coordinates],
        [coordinate + radius for coordinate in coordinates],
    ]
    return center, bounds


def _validate_campaign_execution_bindings(
    execution_value: Any, *, task: dict[str, Any], source: dict[str, Any],
    discovery: dict[str, Any], defects: list[str],
) -> None:
    execution = _require_fields(
        execution_value, CAMPAIGN_EXECUTION_FIELDS,
        "campaign execution bindings")
    gates = _require_fields(
        execution["gates"], CAMPAIGN_GATE_FIELDS,
        "campaign execution gates")
    ct_gate = _require_fields(
        gates["ct_material_support_gate"], CAMPAIGN_CT_GATE_FIELDS,
        "campaign execution CT gate")
    queue = _require_fields(
        execution["queue_execution"], CAMPAIGN_QUEUE_FIELDS,
        "campaign queue execution")
    recenter_radius = _validate_xyz_mapping(
        queue["recenter_radius_xyz"], "campaign queue recenter radius",
        low=1, high=192)

    for field in (
        "p0_artifact_sha256", "p0_selection_sha256",
        "catalog_snapshot_sha256", "source_content_lock_sha256",
        "ct_sha256", "m7_sha256", "m7_uri_sha256",
    ):
        _require_sha256(execution[field], f"campaign execution {field}")
    if (not isinstance(execution["code_revision"], str)
            or re.fullmatch(r"[0-9a-f]{40}", execution["code_revision"]) is None):
        raise ValueError(
            "campaign execution code_revision must be lowercase 40-hex")
    for field in (
        "sample_id", "mission_id", "source_snapshot_id", "p0_artifact_id",
        "p0_selection_version", "grid_version", "policy_version", "provider",
        "selection_strategy", "candidate_selection_policy",
        "seed_region_policy",
    ):
        _require_nonempty_string(
            execution[field], f"campaign execution {field}")
    if execution["coordinate_frame"] != "ct_l0_xyz":
        raise ValueError("campaign execution coordinate frame differs")
    if execution["provider"] != "vc3d-mcp":
        raise ValueError("campaign execution provider is unsupported")
    if execution["seed_region_policy"] not in SEED_REGION_POLICIES:
        raise ValueError("campaign execution seed-region policy is unsupported")
    voxel_size_um = _finite_number(
        execution["voxel_size_um"], "campaign execution voxel size")
    threshold = _finite_number(
        execution["m7_threshold"], "campaign execution M7 threshold")
    if voxel_size_um <= 0 or not 0 <= threshold <= 1:
        raise ValueError("campaign execution source metrics are outside range")
    shape = execution["shape_xyz"]
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(row) is not int or row <= 0 for row in shape)):
        raise ValueError("campaign execution shape must be positive xyz integers")
    grid_step = _require_integer_range(
        execution["grid_step"], 2, 16384, "campaign execution grid_step")
    query_radius = _require_integer_range(
        execution["query_radius"], 1, 4096,
        "campaign execution query_radius")
    if grid_step < 2 * query_radius:
        raise ValueError("campaign execution grid step is smaller than diameter")
    _require_integer_range(
        execution["parallelism"], 1, 8, "campaign execution parallelism")
    _require_integer_range(
        execution["maximum_cells"], 1, 4096,
        "campaign execution maximum_cells")

    cell_clearance = _finite_number(
        gates["cell_clearance"], "campaign execution cell clearance")
    if cell_clearance < 0:
        raise ValueError("campaign execution cell clearance is negative")
    volume_clearance = _require_integer_range(
        gates["volume_clearance"], 1, 8192,
        "campaign execution volume clearance")
    if volume_clearance < query_radius:
        raise ValueError(
            "campaign execution volume clearance is smaller than query radius")
    _require_integer_range(
        gates["candidate_interior_clearance"], 0, 4096,
        "campaign execution candidate clearance")
    packet_limit = _require_integer_range(
        gates["packet_candidate_limit"], 1, 100,
        "campaign execution packet candidate limit")
    if ct_gate["policy"] != "ome-zarr-nearby-material-v1":
        raise ValueError("campaign execution CT policy is unsupported")
    _require_integer_range(
        ct_gate["level"], 0, 2**31 - 1,
        "campaign execution CT level")
    _require_integer_range(
        ct_gate["radius_l0_voxels"], 1, 2**31 - 1,
        "campaign execution CT radius")
    _require_integer_range(
        ct_gate["minimum_nonzero_voxels"], 1, 2**31 - 1,
        "campaign execution CT minimum nonzero")

    if not isinstance(queue["parameter_envelope"], dict):
        raise ValueError("campaign queue parameter envelope must be an object")
    _require_nonempty_string(queue["planner"], "campaign queue planner")
    if (queue["planner_model"] is not None
            and (not isinstance(queue["planner_model"], str)
                 or not queue["planner_model"])):
        raise ValueError("campaign queue planner model differs")
    if queue["prediction_space"] != "ct_l0_xyz":
        raise ValueError("campaign queue prediction space differs")
    _require_integer_range(
        queue["minimum_separation_voxels"], 1, 4096,
        "campaign queue minimum separation")
    _require_integer_range(
        queue["recenter_probe_max_candidates"], 1, 100,
        "campaign queue recenter candidate limit")
    _require_integer_range(
        queue["seed_probe_top_k"], 1, 100,
        "campaign queue seed probe top_k")
    _require_integer_range(
        queue["seed_probe_generations"], 1, 10000,
        "campaign queue seed probe generations")
    if type(queue["verify_sources"]) is not bool:
        raise ValueError("campaign queue source verification must be boolean")
    if queue["seed_probe_mode"] not in {"off", "shadow", "select"}:
        raise ValueError("campaign queue seed probe mode is unsupported")

    source_lock = source.get("source_content_lock")
    if not isinstance(source_lock, dict):
        raise ValueError(
            "campaign execution requires a canonical P0 source content lock")
    m7_uri = _require_nonempty_string(
        source.get("m7_uri"), "campaign execution P0 m7_uri")
    expected_source = {
        "sample_id": task.get("sample_id"),
        "mission_id": task.get("mission_id"),
        "source_snapshot_id": source.get("source_snapshot_id"),
        "ct_sha256": source.get("ct_sha256"),
        "m7_sha256": source.get("m7_sha256"),
        "m7_uri_sha256": hashlib.sha256(
            m7_uri.encode("utf-8")).hexdigest(),
        "source_content_lock_sha256": _sha256(source_lock),
        "coordinate_frame": source.get("coordinate_frame"),
        "voxel_size_um": source.get("voxel_size_um"),
        "shape_xyz": source.get("shape_xyz"),
        "m7_threshold": source.get("m7_threshold"),
    }
    if any(execution.get(field) != expected
           for field, expected in expected_source.items()):
        defects.append("campaign execution source binding differs")

    expected_p0 = {
        "p0_artifact_id": task.get("p0_artifact_id"),
        "p0_artifact_sha256": task.get("p0_artifact_sha256"),
        "p0_selection_version": task.get("p0_selection_version"),
        "p0_selection_sha256": task.get("p0_selection_sha256"),
        "catalog_snapshot_sha256": task.get("catalog_snapshot_sha256"),
    }
    if any(execution.get(field) != expected
           for field, expected in expected_p0.items()):
        defects.append("campaign execution P0/catalog binding differs")

    expected_center, expected_bounds = _expected_campaign_cell_geometry(
        task.get("cell_id"), execution, gates)
    nominal_radius = {axis: query_radius for axis in "xyz"}
    region = discovery.get("region")
    if (execution["grid_version"] != task.get("grid_version")
            or task.get("center_xyz") != expected_center
            or task.get("bounds_xyz") != expected_bounds
            or not isinstance(region, dict)
            or region.get("center") != expected_center
            or region.get("radius") != nominal_radius):
        defects.append("campaign execution geometry/grid binding differs")

    expected_gate = discovery.get("ct_material_support_gate")
    if (execution["policy_version"] != task.get("policy_version")
            or execution["provider"] != discovery.get("provider")
            or execution["m7_threshold"] != discovery.get("m7_threshold")
            or execution["candidate_selection_policy"] !=
                task.get("candidate_selection_policy")
            or execution["seed_region_policy"] !=
                discovery.get("seed_region_policy")
            or discovery.get("prediction_uri") != source.get("m7_uri")
            or discovery.get("minimum_cell_interior_clearance_voxels") !=
                gates["candidate_interior_clearance"]
            or discovery.get("minimum_volume_interior_clearance_voxels") !=
                gates["volume_clearance"]
            or expected_gate != ct_gate
            or discovery.get("max_candidates") != packet_limit
            or task.get("parameter_envelope", {}).get(
                "maximum_candidate_count") != packet_limit):
        defects.append("campaign execution policy/provider/gates binding differs")

    if (queue["parameter_envelope"] != task.get("parameter_envelope")
            or queue["planner"] != task.get("planner")
            or queue["planner_model"] != task.get("planner_model")
            or queue["prediction_space"] != discovery.get("prediction_space")
            or queue["minimum_separation_voxels"] !=
                discovery.get("minimum_separation_voxels")
            or queue["recenter_probe_max_candidates"] !=
                discovery.get("recenter_probe_max_candidates")
            or recenter_radius != discovery.get("recenter_radius_xyz")):
        defects.append("campaign execution queue binding differs")

    probe = task.get("seed_probe")
    expected_mode = queue["seed_probe_mode"]
    resource_requirements = task.get("resource_requirements")
    if not isinstance(resource_requirements, dict):
        raise ValueError("campaign execution task resources are malformed")
    if expected_mode == "off":
        probe_differs = (
            probe is not None
            or resource_requirements.get("seed_probe_required") is not False)
    else:
        probe_differs = (
            not isinstance(probe, dict)
            or probe.get("mode") != expected_mode
            or not isinstance(probe.get("probe_parameters"), dict)
            or probe["probe_parameters"].get("top_k") !=
                queue["seed_probe_top_k"]
            or probe["probe_parameters"].get("generations") !=
                queue["seed_probe_generations"]
            or resource_requirements.get("seed_probe_required") is not True)
    guaranteed = task.get("guaranteed_cell_clearance_voxels")
    if (probe_differs or not isinstance(guaranteed, (int, float))
            or isinstance(guaranteed, bool) or not math.isfinite(float(guaranteed))
            or float(guaranteed) < cell_clearance):
        defects.append("campaign execution queue binding differs")


def _validate_campaign_budget(
    budget_value: Any, *, task: dict[str, Any], source: dict[str, Any],
    discovery: dict[str, Any], defects: list[str],
) -> None:
    budget = _require_fields(budget_value, {
        "schema", "mission_id", "sample_id", "receipt_sha256",
        "preflight_receipt_sha256", "preflight_sanitized_receipt_sha256",
        "approved_task_count", "order_seed_sha256", "population_order_sha256",
        "prefix_sha256", "prefix_cell_ids", "execution_bindings",
        "admission_sha256", "selection_rank",
    }, "campaign budget")
    common = {key: row for key, row in budget.items()
              if key not in {"admission_sha256", "selection_rank"}}
    digest_fields = (
        "receipt_sha256", "preflight_receipt_sha256",
        "preflight_sanitized_receipt_sha256", "order_seed_sha256",
        "population_order_sha256", "prefix_sha256", "admission_sha256",
    )
    if any(not isinstance(budget[field], str) or len(budget[field]) != 64
           or any(character not in "0123456789abcdef"
                  for character in budget[field])
           for field in digest_fields):
        raise ValueError("campaign budget digests are malformed")
    prefix = budget["prefix_cell_ids"]
    outer_differs = (
        budget["schema"] != "campaignx.first_letters_task_budget_admission.v1"
        or budget["admission_sha256"] != _sha256(common)
        or budget["mission_id"] != task["mission_id"]
        or budget["sample_id"] != task["sample_id"]
        or type(budget["approved_task_count"]) is not int
        or budget["approved_task_count"] < 1
        or not isinstance(prefix, list)
        or len(prefix) != budget["approved_task_count"]
        or any(not isinstance(row, str) or not row for row in prefix)
        or len(prefix) != len(set(prefix))
        or budget["prefix_sha256"] != _sha256(prefix)
        or type(budget["selection_rank"]) is not int
        or not 0 <= budget["selection_rank"] < len(prefix)
        or prefix[budget["selection_rank"]] != task["cell_id"]
    )
    if outer_differs:
        defects.append("campaign budget binding differs from claimed task")
    _validate_campaign_execution_bindings(
        budget["execution_bindings"], task=task, source=source,
        discovery=discovery, defects=defects)


def _validate_request_matches_discovery(
    request: dict[str, Any], discovery: dict[str, Any], source: dict[str, Any],
    *, label: str,
) -> None:
    expected = {
        "prediction_uri": discovery["prediction_uri"],
        "prediction_space": discovery["prediction_space"],
        "region": discovery["region"],
        "max_candidates": discovery["max_candidates"],
        "minimum_separation_voxels": discovery["minimum_separation_voxels"],
    }
    if "m7_threshold" in discovery:
        expected["threshold"] = discovery["m7_threshold"]
    if request != expected or request["prediction_uri"] != source["m7_uri"]:
        raise ValueError(f"{label} differs from task discovery and P0 source")


def _validate_initial_probe(
    *, response: dict[str, Any], request: dict[str, Any],
    discovery: dict[str, Any], source: dict[str, Any],
    shape: tuple[float, float, float],
) -> tuple[dict[str, Any], tuple[float, float, float],
           tuple[float, float, float]]:
    provider = response["provider_exchange"]
    call_count = provider["call_count"]
    if type(call_count) is not int or call_count < 0:
        raise ValueError("initial probe provider call count differs")
    seed_policy = discovery["seed_region_policy"]
    if seed_policy not in SEED_REGION_POLICIES:
        raise ValueError("initial probe policy is unsupported")
    initial_probe = response["initial_probe"]
    effective = response["effective_candidate_region"]
    nominal = request["region"]
    if seed_policy == "fixed-v1":
        _validate_region(
            nominal, "fixed request region", shape=shape, radius_high=4096)
        if initial_probe is not None:
            raise ValueError("fixed-v1 response must have a null initial probe")
        if effective != nominal or call_count != 1:
            raise ValueError(
                "fixed-v1 effective region or provider call count differs")
    else:
        _validate_region(nominal, "initial probe nominal region", shape=shape)
        effective = _validate_region(
            effective, "initial probe effective region", shape=shape)
        probe_limit = _require_integer_range(
            discovery["recenter_probe_max_candidates"], 1, 100,
            "initial probe candidate limit")
        recenter_radius = _validate_xyz_mapping(
            discovery["recenter_radius_xyz"], "initial probe recenter radius",
            low=1, high=192)
        if (seed_policy in {
                "m7-recenter-z-chunk-safe-v1",
                "m7-chunk-safe-merge-interior-v2",
            } and recenter_radius != {"x": 64, "y": 64, "z": 64}):
            raise ValueError(
                "initial probe chunk-safe recenter radius must be exactly "
                "64x64x64")
        raw_candidates = response["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("initial probe final candidates must be an array")
        for index, candidate in enumerate(raw_candidates):
            coordinate = _coordinate(candidate)
            if any(coordinate[axis] < 0 or coordinate[axis] >= shape[axis]
                   for axis in range(3)):
                raise ValueError(
                    f"initial probe candidate {index} is outside the P0 volume")
        if seed_policy in {"m7-recenter-z-v1", "m7-recenter-xyz-v1"}:
            probe = _require_allowed_fields(initial_probe, {
                "policy", "request", "candidate_count", "recentered",
            }, {"median_z_ct_l0", "median_coordinate_ct_l0"},
                "initial probe")
            if probe["policy"] != seed_policy:
                raise ValueError("initial probe policy differs")
            candidate_count = probe["candidate_count"]
            if type(candidate_count) is not int or candidate_count < 0:
                raise ValueError("initial probe candidate count differs")
            probe_request = _require_allowed_fields(probe["request"], {
                "prediction_uri", "prediction_space", "region",
                "max_candidates", "minimum_separation_voxels",
            }, {"threshold"}, "initial probe request")
            expected_probe_request = {**request, "max_candidates": probe_limit}
            if probe_request != expected_probe_request:
                raise ValueError("initial probe request differs")
            _validate_region(
                probe_request["region"], "initial probe request region",
                shape=shape)
            if candidate_count == 0:
                if (probe["recentered"] is not False
                        or set(probe) != {
                            "policy", "request", "candidate_count", "recentered"
                        }
                        or effective != nominal or call_count != 1
                        or raw_candidates):
                    raise ValueError("initial probe empty branch differs")
            else:
                if not 1 <= candidate_count <= probe_limit:
                    raise ValueError("initial probe candidate count differs")
                expected_fields = {
                    "policy", "request", "candidate_count", "recentered",
                    "median_z_ct_l0", "median_coordinate_ct_l0",
                }
                _require_fields(probe, expected_fields, "initial probe")
                median_z = probe["median_z_ct_l0"]
                if (type(median_z) is not int
                        or not 0 <= median_z < shape[2]
                        or probe["recentered"] is not True
                        or effective["radius"] != recenter_radius
                        or call_count != 2):
                    raise ValueError("initial probe recentered branch differs")
                if seed_policy == "m7-recenter-z-v1":
                    if (probe["median_coordinate_ct_l0"] is not None
                            or effective["center"]["x"] != nominal["center"]["x"]
                            or effective["center"]["y"] != nominal["center"]["y"]
                            or effective["center"]["z"] != median_z):
                        raise ValueError("initial probe Z recenter binding differs")
                else:
                    median = _validate_xyz_mapping(
                        probe["median_coordinate_ct_l0"],
                        "initial probe median coordinate")
                    if (any(median[axis] < 0 or median[axis] >= shape[index]
                            for index, axis in enumerate("xyz"))
                            or median["z"] != median_z
                            or effective["center"] != median):
                        raise ValueError(
                            "initial probe XYZ recenter binding differs")
        else:
            probe = _require_allowed_fields(initial_probe, {
                "policy", "subquery_count", "subqueries", "candidate_count",
                "recentered",
            }, {
                "median_z_ct_l0", "merged_candidate_count",
                "merge_used_as_final_candidate_set",
            }, "initial probe")
            if probe["policy"] != seed_policy:
                raise ValueError("initial probe policy differs")
            if (probe["subquery_count"] != 8
                    or not isinstance(probe["subqueries"], list)
                    or len(probe["subqueries"]) != 8):
                raise ValueError("initial probe subquery cardinality differs")
            subquery_total = 0
            for index, (subquery, offsets) in enumerate(zip(
                    probe["subqueries"], product((-64, 64), repeat=3),
                    strict=True), start=1):
                row = _require_fields(
                    subquery, {"region", "candidate_count", "response_sha256"},
                    f"initial probe subquery {index}")
                region = _validate_region(
                    row["region"], f"initial probe subquery {index} region",
                    shape=shape)
                expected_center = {
                    axis: nominal["center"][axis] + offset
                    for axis, offset in zip("xyz", offsets, strict=True)
                }
                if (region["center"] != expected_center
                        or region["radius"] != {"x": 64, "y": 64, "z": 64}):
                    raise ValueError(
                        f"initial probe subquery {index} geometry differs")
                count = _require_integer_range(
                    row["candidate_count"], 0, probe_limit,
                    f"initial probe subquery {index} candidate count")
                _require_sha256(
                    row["response_sha256"],
                    f"initial probe subquery {index} response hash")
                subquery_total += count
            candidate_count = probe["candidate_count"]
            if (type(candidate_count) is not int or candidate_count < 0
                    or candidate_count > subquery_total):
                raise ValueError("initial probe combined candidate count differs")
            if seed_policy == "m7-chunk-safe-merge-interior-v2":
                _require_fields(probe, {
                    "policy", "subquery_count", "subqueries", "candidate_count",
                    "merged_candidate_count", "recentered",
                    "merge_used_as_final_candidate_set",
                }, "initial probe")
                if (probe["merged_candidate_count"] != candidate_count
                        or candidate_count != len(raw_candidates)
                        or probe["recentered"] is not False
                        or probe["merge_used_as_final_candidate_set"] is not True
                        or effective != nominal or call_count != 8):
                    raise ValueError("initial probe merged final set differs")
            elif candidate_count == 0:
                _require_fields(probe, {
                    "policy", "subquery_count", "subqueries", "candidate_count",
                    "recentered",
                }, "initial probe")
                if (probe["recentered"] is not False or raw_candidates
                        or effective != nominal or call_count != 8):
                    raise ValueError("initial probe empty chunk-safe branch differs")
            else:
                _require_fields(probe, {
                    "policy", "subquery_count", "subqueries", "candidate_count",
                    "median_z_ct_l0", "recentered",
                }, "initial probe")
                median_z = probe["median_z_ct_l0"]
                if (type(median_z) is not int
                        or not 0 <= median_z < shape[2]
                        or probe["recentered"] is not True
                        or effective["center"] != {
                            "x": nominal["center"]["x"],
                            "y": nominal["center"]["y"],
                            "z": median_z,
                        }
                        or effective["radius"] != {
                            "x": 64, "y": 64, "z": 64,
                        }
                        or call_count != 9):
                    raise ValueError(
                        "initial probe chunk-safe recenter binding differs")

    center = effective["center"]
    radius = effective["radius"]
    low = tuple(float(center[axis] - radius[axis]) for axis in "xyz")
    high = tuple(float(center[axis] + radius[axis]) for axis in "xyz")
    return effective, low, high


def query_cube_volume_gate_geometry(
    *,
    bounds_xyz: Any,
    shape_xyz: Any,
    minimum_volume_interior_clearance_voxels: Any,
) -> dict[str, Any]:
    """Measure how much of a task's query cube its own volume gate can admit.

    The volume gate is an absolute distance from a candidate to a scanned-volume
    face, while the grid only guarantees the margin at a cell *centre*.  A
    candidate may sit up to `query_radius` voxels outward of that centre, so a
    cell on the rim offers the provider ground the task's own gate can never
    accept.  This measures that overlap exactly, in voxels.

    It reads nothing, writes nothing, and authorizes no threshold change.  It
    is a property of two frozen numbers and a shape, not evidence about ink.
    """

    if (not isinstance(bounds_xyz, list) or len(bounds_xyz) != 2
            or any(not isinstance(row, list) or len(row) != 3 for row in bounds_xyz)):
        raise ValueError("bounds_xyz must be [[x,y,z],[x,y,z]]")
    low = [_finite_number(value, "bounds_xyz low") for value in bounds_xyz[0]]
    high = [_finite_number(value, "bounds_xyz high") for value in bounds_xyz[1]]
    if any(low[axis] > high[axis] for axis in range(3)):
        raise ValueError("bounds_xyz low exceeds high")
    if (not isinstance(shape_xyz, list) or len(shape_xyz) != 3
            or any(isinstance(size, bool) or type(size) is not int or size < 1
                   for size in shape_xyz)):
        raise ValueError("shape_xyz must be three positive integer extents")
    minimum = minimum_volume_interior_clearance_voxels
    if isinstance(minimum, bool) or type(minimum) is not int or minimum < 0:
        raise ValueError(
            "minimum_volume_interior_clearance_voxels must be a non-negative integer")

    cube_counts: list[int] = []
    admissible_low: list[int] = []
    admissible_high: list[int] = []
    admissible_counts: list[int] = []
    inadmissible_faces: list[str] = []
    for index, axis in enumerate("xyz"):
        size = int(shape_xyz[index])
        cube_low = max(math.ceil(low[index]), 0)
        cube_high = min(math.floor(high[index]), size - 1)
        cube_counts.append(max(0, cube_high - cube_low + 1))
        gate_low = minimum
        gate_high = size - 1 - minimum
        if cube_low < gate_low:
            inadmissible_faces.append(f"{axis}_low")
        if cube_high > gate_high:
            inadmissible_faces.append(f"{axis}_high")
        open_low = max(cube_low, gate_low)
        open_high = min(cube_high, gate_high)
        admissible_low.append(open_low)
        admissible_high.append(open_high)
        admissible_counts.append(max(0, open_high - open_low + 1))
    if any(count == 0 for count in cube_counts):
        raise ValueError("query cube does not intersect the scanned volume")

    cube_voxels = cube_counts[0] * cube_counts[1] * cube_counts[2]
    admissible_voxels = (
        admissible_counts[0] * admissible_counts[1] * admissible_counts[2])
    return {
        "query_cube_voxel_count": cube_voxels,
        "admissible_voxel_count": admissible_voxels,
        "admissible_voxel_fraction": admissible_voxels / cube_voxels,
        "admissible_bounds_xyz": (
            [list(admissible_low), list(admissible_high)]
            if admissible_voxels else None),
        "inadmissible_faces": sorted(inadmissible_faces),
        "gate_admits_whole_cube": admissible_voxels == cube_voxels,
        "gate_admits_no_voxel": admissible_voxels == 0,
        "minimum_volume_interior_clearance_voxels": minimum,
        "diagnostic_only": True,
        "non_claim": NON_CLAIM,
    }


def policy_margin_verdict_reachability(
    *,
    minimum_volume_interior_clearance_voxels: Any,
    query_radius_voxels: Any,
) -> dict[str, Any]:
    """Say whether `POLICY_MARGIN_UNCALIBRATED` is reachable for a configuration.

    This review calls a volume rejection *physically safe* when the candidate's
    own query cube still fits inside the scanned volume, that is when its face
    distance is at least `query_radius`.  `POLICY_MARGIN_UNCALIBRATED` requires
    at least one such rejection.  When the margin does not exceed the query
    radius, every rejected candidate is unsafe by construction and the verdict
    cannot be produced -- so a `TRUE_VOLUME_BOUNDARY` result under that
    configuration is uninformative about whether the margin is calibrated.
    """

    minimum = minimum_volume_interior_clearance_voxels
    if isinstance(minimum, bool) or type(minimum) is not int or minimum < 0:
        raise ValueError(
            "minimum_volume_interior_clearance_voxels must be a non-negative integer")
    if (isinstance(query_radius_voxels, bool)
            or type(query_radius_voxels) is not int or query_radius_voxels < 1):
        raise ValueError("query_radius_voxels must be a positive integer")
    reachable = minimum > query_radius_voxels
    return {
        "minimum_volume_interior_clearance_voxels": minimum,
        "query_radius_voxels": query_radius_voxels,
        "policy_margin_uncalibrated_reachable": reachable,
        "reason": (
            "MARGIN_EXCEEDS_QUERY_RADIUS_A_PHYSICALLY_SAFE_REJECTION_IS_POSSIBLE"
            if reachable else
            "MARGIN_WITHIN_QUERY_RADIUS_EVERY_REJECTION_IS_PHYSICALLY_UNSAFE"),
        "diagnostic_only": True,
        "non_claim": NON_CLAIM,
    }


def analyze_candidate_clearance(
    value: dict[str, Any], *, trusted_manifest_sha256: str
) -> dict[str, Any]:
    """Return immutable diagnostic evidence for one frozen clearance attempt."""
    source_input = copy.deepcopy(value)
    defects: list[str] = []
    _require_fields(value, {
        "schema", "attempt_receipt", "p0_source", "raw_candidate_response",
        "ct_screen", "task", "alignment_authority", "evidence_manifest",
    }, "clearance review input")
    if value["schema"] != INPUT_SCHEMA:
        raise ValueError("input schema is not candidate clearance review v1")
    attempt = _require_fields(value["attempt_receipt"], {
        "status", "raw_candidate_count", "post_ct_candidate_count",
        "usable_candidate_count", "no_seed_cause_counts", "primary_causes",
        "no_seed_causal_diagnosis", "no_seed_causal_diagnosis_sha256",
        "clearance_policy", "reason", "generated_at_utc", "ink_used", "non_claim",
    }, "TERMINAL_RECEIPT.json")
    source = _require_allowed_fields(value["p0_source"], {
        "schema", "sample_id", "source_snapshot_id", "ct_uri", "ct_sha256", "m7_uri",
        "m7_sha256", "shape_xyz", "coordinate_frame", "voxel_size_um",
        "m7_threshold", "eligible_manifest_sha256", "source_status",
    }, {"source_content_lock", "source_reachable"}, "P0_SOURCE.json")
    response = _require_fields(value["raw_candidate_response"], {
        "candidates", "request", "effective_candidate_region", "initial_probe",
        "source_read_set", "provider_exchange", "ink_used",
    }, "SEED_CANDIDATES.json")
    ct_screen = _require_fields(value["ct_screen"], {
        "schema", "status", "policy", "config", "source_snapshot_id",
        "input_candidate_count", "retained_candidate_count",
        "rejected_candidate_count", "assessments", "filtered_response_sha256",
        "generated_at_utc", "ink_used", "non_claim",
    }, "CT_MATERIAL_SUPPORT_SCREEN.json")
    task = _require_allowed_fields(value["task"], {
        "schema", "task_id", "attempt_id", "sample_id", "source_snapshot_id",
        "cell_id", "grid_version", "policy_version", "bounds_xyz", "center_xyz",
        "priority", "distance_to_known_aabb_voxels",
        "guaranteed_cell_clearance_voxels", "parameter_envelope",
        "catalog_snapshot_sha256", "candidate_discovery",
        "candidate_selection_policy", "planner_contract_version",
        "resource_requirements", "mission_id", "ink_used", "source",
    }, {
        "state", "attempt_number", "worker_id", "lease_expires_at", "retry_after",
        "planner", "planner_model", "queued_reason", "created_by", "seed_probe",
        "benchmark_execution", "p0_selection_version", "p0_selection_sha256",
        "p0_artifact_id", "p0_artifact_sha256", "p0_resolved_by",
        "campaign_budget",
    }, "CLAIMED_TASK.json")
    alignment = _require_fields(value["alignment_authority"], {
        "schema", "source_snapshot_id", "m7_level", "m7_to_ct_l0_transform",
        "coordinate_frame", "authority_sha256",
    }, "M7_CT_ALIGNMENT.json")
    manifest = _require_fields(value["evidence_manifest"], {
        "schema", "task_id", "attempt_id", "source_snapshot_id", "files",
    }, "evidence manifest")
    if (attempt["status"] != "NO_SEED" or attempt["ink_used"] is not False
            or response["ink_used"] is not False
            or ct_screen["schema"] != "campaignx.ct_material_support_screen.v1"
            or ct_screen["status"] != "COMPLETED_INK_BLIND"
            or ct_screen["ink_used"] is not False):
        raise ValueError("review requires an ink-blind NO_SEED clearance terminal")
    if source["coordinate_frame"] != "ct_l0_xyz":
        defects.append("coordinate frame is not ct_l0_xyz")
    if source["schema"] != "campaignx.source_snapshot.v1":
        raise ValueError("P0 source schema is unsupported")
    eligible_digest = source["eligible_manifest_sha256"]
    if (not isinstance(eligible_digest, str) or len(eligible_digest) != 64
            or any(character not in "0123456789abcdef" for character in eligible_digest)):
        raise ValueError("P0 eligible manifest must be lowercase SHA-256")
    source_status = source["source_status"]
    if source_status not in {
        "URI_LOCKED_HASH_UNAVAILABLE", "HASH_LOCKED_NOT_IMMUTABLE",
        "IMMUTABLE_CONTENT_LOCKED",
    }:
        raise ValueError("P0 source status is unsupported")
    source_hashes = [source["ct_sha256"], source["m7_sha256"]]
    if source_status == "URI_LOCKED_HASH_UNAVAILABLE":
        if all(row is not None for row in source_hashes):
            raise ValueError("URI-locked source status contradicts available hashes")
        defects.append("P0 source lacks one or more immutable content hashes")
    elif any(row is None for row in source_hashes):
        raise ValueError("hash-locked source is missing a content hash")
    if source_status == "IMMUTABLE_CONTENT_LOCKED" and "source_content_lock" not in source:
        raise ValueError("immutable source is missing its content lock")
    defects.extend(_validate_source_reachability(source))
    if task["sample_id"] != source["sample_id"]:
        defects.append("task sample differs from P0 source")
    if task["source_snapshot_id"] != source["source_snapshot_id"]:
        defects.append("task source snapshot differs from P0 source")
    if task["source"] != source:
        defects.append("task source projection differs from P0 source")
    if (ct_screen["source_snapshot_id"] != source["source_snapshot_id"]):
        defects.append("CT screen source snapshot differs from P0 source")

    if (manifest["schema"] !=
            "campaignx.candidate_clearance_review_input_manifest.v1"):
        raise ValueError("evidence manifest schema is unsupported")
    if (not isinstance(trusted_manifest_sha256, str)
            or len(trusted_manifest_sha256) != 64
            or trusted_manifest_sha256 != _sha256(manifest)):
        raise ValueError("trusted manifest SHA-256 differs from evidence manifest")
    if (manifest["task_id"] != task["task_id"]
            or manifest["attempt_id"] != task["attempt_id"]
            or manifest["source_snapshot_id"] != task["source_snapshot_id"]):
        defects.append("evidence manifest identity differs from task identity")
    documents = {
        "TERMINAL_RECEIPT.json": attempt,
        "SEED_CANDIDATES.json": response,
        "CT_MATERIAL_SUPPORT_SCREEN.json": ct_screen,
        "CLAIMED_TASK.json": task,
        "P0_SOURCE.json": source,
        "M7_CT_ALIGNMENT.json": alignment,
    }
    if not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(documents):
        raise ValueError("evidence manifest file roles are not exact")
    for name, document in documents.items():
        entry = _require_fields(manifest["files"][name], {"sha256"},
                                f"evidence manifest {name}")
        if entry["sha256"] != _sha256(document):
            defects.append(f"evidence manifest hash differs for {name}")

    diagnosis = _require_fields(attempt["no_seed_causal_diagnosis"], {
        "schema", "status", "task_id", "attempt_id", "m7_raw_candidate_count",
        "ct_support_input_candidate_count", "ct_support_retained_candidate_count",
        "ct_support_rejected_candidate_count", "post_ct_candidate_count",
        "eligible_after_clearance_count", "cause_counts", "primary_causes",
        "clearance_policy", "clearance_rejection_examples_first_32", "ink_used",
        "non_claim", "generated_at_utc", "diagnosis_sha256",
    }, "NO_SEED_CAUSAL_DIAGNOSIS.json")
    diagnosis_core = {key: row for key, row in diagnosis.items()
                      if key not in {"generated_at_utc", "diagnosis_sha256"}}
    if (diagnosis["schema"] != "campaignx.no_seed_causal_diagnosis.v1"
            or diagnosis["status"] != "NO_SEED" or diagnosis["ink_used"] is not False
            or diagnosis["diagnosis_sha256"] != _sha256(diagnosis_core)
            or attempt["no_seed_causal_diagnosis_sha256"] != diagnosis["diagnosis_sha256"]):
        defects.append("NO_SEED causal diagnosis hash or identity differs")
    if (diagnosis["task_id"] != task["task_id"]
            or diagnosis["attempt_id"] != task["attempt_id"]):
        defects.append("NO_SEED causal diagnosis belongs to another task or attempt")

    discovery = _require_allowed_fields(task["candidate_discovery"], {
        "provider", "prediction_uri", "prediction_space", "region",
        "max_candidates", "minimum_separation_voxels",
        "minimum_cell_interior_clearance_voxels",
        "minimum_volume_interior_clearance_voxels", "seed_region_policy",
        "recenter_probe_max_candidates", "recenter_radius_xyz",
        "ct_material_support_gate",
    }, {"m7_threshold"}, "task candidate_discovery")
    if "campaign_budget" in task:
        _validate_campaign_budget(
            task["campaign_budget"], task=task, source=source,
            discovery=discovery, defects=defects)
    policy = _require_fields(attempt["clearance_policy"], {
        "minimum_cell_interior_clearance_voxels",
        "minimum_cell_interior_clearance_um",
        "minimum_volume_interior_clearance_voxels",
        "minimum_volume_interior_clearance_um", "voxel_size_um",
    }, "terminal clearance_policy")
    if policy != diagnosis["clearance_policy"]:
        defects.append("terminal and diagnosis clearance policies differ")
    if any(discovery[key] != policy[key] for key in (
        "minimum_cell_interior_clearance_voxels",
        "minimum_volume_interior_clearance_voxels",
    )):
        defects.append("task and terminal clearance policies differ")
    gate = _require_fields(discovery["ct_material_support_gate"], {
        "policy", "level", "radius_l0_voxels", "minimum_nonzero_voxels",
    }, "task CT material gate")
    config = _require_fields(ct_screen["config"], {
        "level", "radius_l0_voxels", "minimum_nonzero_voxels",
    }, "CT screen config")
    if gate != {**config, "policy": ct_screen["policy"]}:
        defects.append("task CT material gate differs from retained CT screen")
    request = _require_allowed_fields(response["request"], {
        "prediction_uri", "prediction_space", "region", "max_candidates",
        "minimum_separation_voxels",
    }, {"threshold"}, "M7 request")
    provider_exchange = _require_fields(response["provider_exchange"], {
        "encoding", "call_count", "request_sha256", "request_bytes",
        "response_sha256", "response_bytes",
    }, "provider exchange")
    if (provider_exchange["encoding"] != "canonical-json-utf8"
            or any(type(provider_exchange[field]) is not int
                   or provider_exchange[field] < 0
                   for field in ("request_bytes", "response_bytes"))):
        raise ValueError("provider exchange encoding or byte counts differ")
    for field in ("request_sha256", "response_sha256"):
        _require_sha256(provider_exchange[field], f"provider exchange {field}")
    _validate_request_matches_discovery(
        request, discovery, source, label="M7 request")
    shape = _validate_triplet(source["shape_xyz"], "shape_xyz")
    effective, clearance_low, clearance_high = _validate_initial_probe(
        response=response, request=request, discovery=discovery,
        source=source, shape=shape)
    alignment_core = {key: row for key, row in alignment.items()
                      if key != "authority_sha256"}
    if (alignment["schema"] != "campaignx.m7_ct_alignment_authority.v1"
            or alignment["authority_sha256"] != _sha256(alignment_core)
            or alignment["source_snapshot_id"] != source["source_snapshot_id"]
            or alignment["coordinate_frame"] != source["coordinate_frame"]):
        defects.append("M7/CT alignment authority hash or source identity differs")
    transform = alignment["m7_to_ct_l0_transform"]
    if (not isinstance(transform, list) or len(transform) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in transform)
            or any(type(number) not in {int, float} or isinstance(number, bool)
                   or not math.isfinite(float(number)) for row in transform for number in row)):
        raise ValueError("M7 to CT-L0 transform must be a finite 4x4 matrix")
    identity_transform = [[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]]
    if (request["prediction_uri"] != source["m7_uri"]
            or discovery["prediction_uri"] != source["m7_uri"]
            or request["prediction_space"] != "ct_l0_xyz"
            or discovery["prediction_space"] != "ct_l0_xyz"
            or transform != identity_transform):
        defects.append("M7 source level, URI, coordinate space, or transform differs")
    for field in ("ct_sha256", "m7_sha256"):
        digest = source[field]
        if digest is None:
            continue
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"P0 {field} must be lowercase SHA-256")

    if (not isinstance(task["bounds_xyz"], list) or len(task["bounds_xyz"]) != 2):
        raise ValueError("task bounds must be [low_xyz, high_xyz]")
    nominal_low = _validate_triplet(
        task["bounds_xyz"][0], "task.bounds_xyz[0]")
    nominal_high = _validate_triplet(
        task["bounds_xyz"][1], "task.bounds_xyz[1]")
    task_center = _validate_xyz_mapping(task["center_xyz"], "task center")
    request_region = request["region"]
    expected_nominal_low = [
        request_region["center"][axis] - request_region["radius"][axis]
        for axis in "xyz"
    ]
    expected_nominal_high = [
        request_region["center"][axis] + request_region["radius"][axis]
        for axis in "xyz"
    ]
    if (task_center != request_region["center"]
            or expected_nominal_low != [int(row) for row in nominal_low]
            or expected_nominal_high != [int(row) for row in nominal_high]):
        defects.append("nominal M7 request region differs from task geometry")
    radii = [effective["radius"][axis] for axis in "xyz"]
    if any(radius < 1 for radius in radii):
        raise ValueError("candidate region must carry positive query radii")
    voxel_um = _finite_number(source["voxel_size_um"], "voxel_size_um")
    minimum_cell = _finite_number(
        policy["minimum_cell_interior_clearance_voxels"], "minimum cell clearance")
    minimum_volume = _finite_number(
        policy["minimum_volume_interior_clearance_voxels"], "minimum volume clearance")
    query_radius = float(max(radii))
    if (policy["minimum_cell_interior_clearance_um"] != minimum_cell * voxel_um
            or policy["minimum_volume_interior_clearance_um"] != minimum_volume * voxel_um
            or policy["voxel_size_um"] != voxel_um):
        defects.append("terminal clearance physical units differ from P0 voxel size")
    if any(size <= 0 for size in shape) or voxel_um <= 0:
        defects.append("volume shape and voxel size must be positive")
    if minimum_cell < 0 or minimum_volume < 0 or query_radius <= 0:
        defects.append("clearance thresholds are outside their valid range")
    if minimum_volume < query_radius:
        defects.append("volume clearance is smaller than the query radius")
    if any(nominal_low[index] < 0 or nominal_high[index] > shape[index]
           or nominal_low[index] >= nominal_high[index] for index in range(3)):
        defects.append("task bounds are outside the P0 volume or empty")

    raw = response["candidates"]
    screened = ct_screen["assessments"]
    if not isinstance(raw, list) or not isinstance(screened, list):
        raise ValueError("candidate response and CT screen must contain arrays")
    _validate_read_set(response["source_read_set"], "M7 source read set")
    ct_by_id: dict[str, str] = {}
    ct_coordinates: dict[str, dict[str, int]] = {}
    for row in screened:
        _require_fields(row, {"candidate_id", "coordinate_xyz_l0", "status", "sample"},
                        "CT assessment")
        if not isinstance(row["candidate_id"], str):
            raise ValueError("CT assessment candidate_id must be a string")
        candidate_id = row["candidate_id"]
        if candidate_id in ct_by_id:
            defects.append(f"duplicate CT result for {candidate_id}")
        if (not isinstance(row["coordinate_xyz_l0"], dict)
                or set(row["coordinate_xyz_l0"]) != {"x", "y", "z"}
                or any(type(row["coordinate_xyz_l0"][axis]) is not int for axis in "xyz")):
            raise ValueError("CT assessment coordinate must be exact xyz integers")
        sample = _require_fields(row["sample"], {
            "level", "scale_zyx", "center_zyx", "radius_zyx", "shape_zyx",
            "voxel_count", "nonzero_voxel_count", "nonzero_fraction", "mean",
            "standard_deviation", "maximum", "source_read_set",
        }, "CT sample")
        for field in ("scale_zyx", "center_zyx", "radius_zyx", "shape_zyx"):
            vector = sample[field]
            if (not isinstance(vector, list) or len(vector) != 3
                    or any(type(number) not in {int, float}
                           or isinstance(number, bool)
                           or not math.isfinite(float(number)) for number in vector)):
                raise ValueError(f"CT sample {field} must be a finite zyx vector")
        if any(float(number) <= 0 for number in sample["scale_zyx"]):
            raise ValueError("CT sample scale_zyx must be positive")
        if (type(sample["level"]) is not int or sample["level"] != config["level"]
                or any(type(number) is not int or number < 1
                       for number in sample["shape_zyx"])
                or type(sample["voxel_count"]) is not int
                or sample["voxel_count"] != math.prod(sample["shape_zyx"])
                or type(sample["nonzero_voxel_count"]) is not int
                or not 0 <= sample["nonzero_voxel_count"] <= sample["voxel_count"]):
            raise ValueError("CT sample dimensions or counts differ")
        expected_fraction = (
            sample["nonzero_voxel_count"] / sample["voxel_count"]
        )
        numeric_metrics = ("nonzero_fraction", "mean", "standard_deviation", "maximum")
        metrics = {field: _finite_number(sample[field], f"CT sample {field}")
                   for field in numeric_metrics}
        if (not math.isclose(metrics["nonzero_fraction"], expected_fraction,
                             rel_tol=0.0, abs_tol=1e-15)
                or metrics["standard_deviation"] < 0):
            raise ValueError("CT sample statistics differ from counts")
        expected_status = (
            "CT_MATERIAL_NEARBY"
            if sample["nonzero_voxel_count"] >= config["minimum_nonzero_voxels"]
            else "NO_NEARBY_CT_MATERIAL"
        )
        if row["status"] != expected_status:
            defects.append(f"{candidate_id}: CT status differs from retained sample")
        coordinate_zyx = [row["coordinate_xyz_l0"][axis] for axis in "zyx"]
        expected_center = [
            int(math.floor(number / float(scale)))
            for number, scale in zip(coordinate_zyx, sample["scale_zyx"], strict=True)
        ]
        expected_radius = [
            max(1, int(math.ceil(config["radius_l0_voxels"] / float(scale))))
            for scale in sample["scale_zyx"]
        ]
        if sample["center_zyx"] != expected_center or sample["radius_zyx"] != expected_radius:
            defects.append(f"{candidate_id}: CT sample geometry differs from candidate")
        _validate_read_set(sample["source_read_set"], f"CT read set {candidate_id}")
        ct_by_id[candidate_id] = row["status"]
        ct_coordinates[candidate_id] = row["coordinate_xyz_l0"]

    rows: list[dict[str, Any]] = []
    computed_causes: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for index, candidate in enumerate(raw, start=1):
        _require_fields(candidate, {"candidate_id", "ct_l0_coordinate", "score"},
                        f"M7 candidate {index}")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            defects.append(f"candidate {index} has no stable ID")
            continue
        if candidate_id in seen_ids:
            defects.append(f"duplicate candidate ID {candidate_id}")
            continue
        seen_ids.add(candidate_id)
        coordinate = _coordinate(candidate)
        score = _finite_number(candidate["score"], f"{candidate_id}.score")
        if any(coordinate[axis] < 0 or coordinate[axis] >= shape[axis]
               for axis in range(3)):
            defects.append(f"{candidate_id}: coordinate is outside the P0 volume")
        if any(coordinate[axis] < clearance_low[axis]
               or coordinate[axis] >= clearance_high[axis]
               for axis in range(3)):
            defects.append(
                f"{candidate_id}: coordinate is outside the effective candidate bounds")

        cell_faces = _axis_distances(
            coordinate, clearance_low, clearance_high, inclusive_high=False
        )
        volume_faces = _axis_distances(
            coordinate, (0.0, 0.0, 0.0), shape, inclusive_high=True
        )
        cell_distance = min(cell_faces.values())
        volume_distance = min(volume_faces.values())
        ct_status = ct_by_id.get(candidate_id)
        if ct_status not in {"CT_MATERIAL_NEARBY", "NO_NEARBY_CT_MATERIAL"}:
            defects.append(f"{candidate_id}: CT result is missing or invalid")
        if ct_coordinates.get(candidate_id) != candidate["ct_l0_coordinate"]:
            defects.append(f"{candidate_id}: M7 and CT coordinates differ")
        causes: list[str] = []
        if ct_status == "CT_MATERIAL_NEARBY":
            if cell_distance < minimum_cell:
                causes.append(CAUSE_CELL)
            if volume_distance < minimum_volume:
                causes.append(CAUSE_VOLUME)
        computed_causes[candidate_id] = causes
        rows.append({
            "candidate_id": candidate_id,
            "coordinate_xyz": [int(number) for number in coordinate],
            "m7_score": score,
            "ct_status": ct_status,
            "cell_face_distances_voxels": cell_faces,
            "volume_face_distances_voxels": volume_faces,
            "minimum_cell_face_distance_voxels": cell_distance,
            "minimum_cell_face_distance_um": cell_distance * voxel_um,
            "minimum_volume_face_distance_voxels": volume_distance,
            "minimum_volume_face_distance_um": volume_distance * voxel_um,
            "computed_causes": causes,
            "physically_safe_for_query": volume_distance >= query_radius,
        })

    if set(ct_by_id) != seen_ids:
        defects.append("CT screen membership differs from raw candidate membership")
    retained_response = {
        **response,
        "candidates": [
            {**candidate, "ct_material_support": screened[index]}
            for index, candidate in enumerate(raw)
            if ct_by_id.get(candidate["candidate_id"]) == "CT_MATERIAL_NEARBY"
        ],
    }
    if ct_screen["filtered_response_sha256"] != _sha256(retained_response):
        defects.append("CT filtered response hash differs from exact retained response")
    examples = diagnosis["clearance_rejection_examples_first_32"]
    if not isinstance(examples, list):
        raise ValueError("clearance examples must be an array")
    reported: dict[str, list[str]] = {}
    for example in examples:
        _require_fields(example, {"candidate_id", "causes"}, "clearance example")
        if example["candidate_id"] in reported or not isinstance(example["causes"], list):
            raise ValueError("clearance examples contain duplicates or invalid causes")
        reported[example["candidate_id"]] = example["causes"]
    if reported != computed_causes:
        defects.append("terminal reported causes differ from independent recomputation")

    raw_count = len(raw)
    ct_count = sum(row["ct_status"] == "CT_MATERIAL_NEARBY" for row in rows)
    retained_count = sum(
        row["ct_status"] == "CT_MATERIAL_NEARBY" and not row["computed_causes"]
        for row in rows
    )
    expected_counts = {
        "raw_candidate_count": raw_count,
        "post_ct_candidate_count": ct_count,
        "usable_candidate_count": retained_count,
    }
    for key, expected in expected_counts.items():
        if attempt.get(key) != expected:
            defects.append(f"attempt {key} differs from independent recomputation")

    if (diagnosis["m7_raw_candidate_count"] != raw_count
            or diagnosis["post_ct_candidate_count"] != ct_count
            or diagnosis["eligible_after_clearance_count"] != retained_count):
        defects.append("NO_SEED diagnosis counts differ from independent recomputation")
    if (ct_screen["input_candidate_count"] != raw_count
            or ct_screen["retained_candidate_count"] != ct_count
            or ct_screen["rejected_candidate_count"] != raw_count - ct_count):
        defects.append("CT screen counts differ from independent recomputation")
    ct_rows = [row for row in rows if row["ct_status"] == "CT_MATERIAL_NEARBY"]
    cell_rejected = [row for row in ct_rows if CAUSE_CELL in row["computed_causes"]]
    volume_rejected = [row for row in ct_rows if CAUSE_VOLUME in row["computed_causes"]]
    safe_policy_rejected = [
        row for row in volume_rejected if row["physically_safe_for_query"]
    ]
    unsafe_volume = [
        row for row in volume_rejected if not row["physically_safe_for_query"]
    ]
    if defects:
        classification = "IMPLEMENTATION_OR_METADATA_DEFECT"
    elif ct_rows and len(cell_rejected) == len(ct_rows) and not volume_rejected:
        classification = "CELL_BOUNDARY_ONLY"
    elif safe_policy_rejected and not unsafe_volume and not cell_rejected:
        classification = "POLICY_MARGIN_UNCALIBRATED"
    else:
        classification = "TRUE_VOLUME_BOUNDARY"

    sensitivity: list[dict[str, Any]] = []
    for percentage in SENSITIVITY_PERCENTAGES:
        threshold = minimum_volume * percentage / 100.0
        retained = [
            row for row in ct_rows
            if row["minimum_cell_face_distance_voxels"] >= minimum_cell
            and row["minimum_volume_face_distance_voxels"] >= threshold
        ]
        sensitivity.append({
            "margin_percentage": percentage,
            "minimum_volume_clearance_voxels": threshold,
            "minimum_volume_clearance_um": threshold * voxel_um,
            "m7_candidate_count": raw_count,
            "ct_supported_count": ct_count,
            "retained_count": len(retained),
            "physically_safe_retained_count": sum(
                row["physically_safe_for_query"] for row in retained
            ),
            "diagnostic_only": True,
        })

    core = {
        "schema": SCHEMA,
        "sample_id": source["sample_id"],
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "source_snapshot_id": source["source_snapshot_id"],
        "coordinate_frame": source["coordinate_frame"],
        "shape_xyz": [int(row) for row in shape],
        "voxel_size_um": voxel_um,
        "input_sha256": _sha256(source_input),
        "trusted_manifest_sha256": trusted_manifest_sha256,
        "input_evidence_sha256": {
            "attempt_receipt": _sha256(attempt),
            "p0_source": _sha256(source),
            "raw_candidate_response": _sha256(response),
            "ct_screen": _sha256(ct_screen),
            "task": _sha256(task),
            "alignment_authority": _sha256(alignment),
            "evidence_manifest": _sha256(manifest),
            "clearance_policy": _sha256(policy),
        },
        "clearance_policy": copy.deepcopy(policy),
        "candidates": rows,
        "counts": expected_counts,
        "classification": classification,
        "classification_basis": {
            "ct_supported_candidates": len(ct_rows),
            "cell_rejected_candidates": len(cell_rejected),
            "volume_rejected_candidates": len(volume_rejected),
            "physically_safe_but_policy_rejected": len(safe_policy_rejected),
            "physically_unsafe_volume_candidates": len(unsafe_volume),
        },
        "defects": sorted(set(defects)),
        "sensitivity": sensitivity,
        "policy_change_authorized": False,
        "state_mutation": "NONE_READ_ONLY",
        "non_claim": NON_CLAIM,
    }
    return {**core, "review_sha256": _sha256(core)}


def validate_candidate_clearance_review(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the closed top-level review and its content hash."""
    _require_fields(value, OUTPUT_FIELDS, "candidate clearance review")
    if value["schema"] != SCHEMA:
        raise ValueError("candidate clearance review schema differs")
    core = {key: row for key, row in value.items() if key != "review_sha256"}
    if value["review_sha256"] != _sha256(core):
        raise ValueError("candidate clearance review hash differs")
    def require_sha(digest: Any, label: str) -> None:
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"{label} must be lowercase SHA-256")

    require_sha(value["review_sha256"], "review hash")
    require_sha(value["input_sha256"], "input hash")
    require_sha(value["trusted_manifest_sha256"], "trusted manifest hash")
    for field in ("sample_id", "task_id", "attempt_id", "source_snapshot_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"review identity {field} differs")
    if (value["coordinate_frame"] != "ct_l0_xyz"
            or not isinstance(value["shape_xyz"], list)
            or len(value["shape_xyz"]) != 3
            or any(type(row) is not int or row <= 0 for row in value["shape_xyz"])
            or type(value["voxel_size_um"]) not in {int, float}
            or isinstance(value["voxel_size_um"], bool)
            or not math.isfinite(float(value["voxel_size_um"]))
            or value["voxel_size_um"] <= 0):
        raise ValueError("review source geometry differs")
    if (value["policy_change_authorized"] is not False
            or value["state_mutation"] != "NONE_READ_ONLY"
            or value["non_claim"] != NON_CLAIM):
        raise ValueError("candidate clearance review safety fields differ")
    _require_fields(value["input_evidence_sha256"], {
        "attempt_receipt", "p0_source", "raw_candidate_response", "ct_screen",
        "task", "alignment_authority", "evidence_manifest", "clearance_policy",
    }, "review input hashes")
    _require_fields(value["clearance_policy"], {
        "minimum_cell_interior_clearance_voxels",
        "minimum_cell_interior_clearance_um",
        "minimum_volume_interior_clearance_voxels",
        "minimum_volume_interior_clearance_um", "voxel_size_um",
    }, "review clearance policy")
    for field, row in value["clearance_policy"].items():
        if (type(row) not in {int, float} or isinstance(row, bool)
                or not math.isfinite(float(row)) or row < 0):
            raise ValueError(f"review clearance policy {field} differs")
    output_policy = value["clearance_policy"]
    if (output_policy["voxel_size_um"] <= 0
            or output_policy["voxel_size_um"] != value["voxel_size_um"]
            or output_policy["minimum_cell_interior_clearance_um"] !=
            output_policy["minimum_cell_interior_clearance_voxels"] *
            value["voxel_size_um"]
            or output_policy["minimum_volume_interior_clearance_um"] !=
            output_policy["minimum_volume_interior_clearance_voxels"] *
            value["voxel_size_um"]):
        raise ValueError("review clearance policy physical units differ")
    _require_fields(value["counts"], {
        "raw_candidate_count", "post_ct_candidate_count", "usable_candidate_count",
    }, "review counts")
    _require_fields(value["classification_basis"], {
        "ct_supported_candidates", "cell_rejected_candidates",
        "volume_rejected_candidates", "physically_safe_but_policy_rejected",
        "physically_unsafe_volume_candidates",
    }, "review classification basis")
    if value["classification"] not in {
        "IMPLEMENTATION_OR_METADATA_DEFECT", "TRUE_VOLUME_BOUNDARY",
        "CELL_BOUNDARY_ONLY", "POLICY_MARGIN_UNCALIBRATED",
    }:
        raise ValueError("candidate clearance review classification differs")
    if (not isinstance(value["defects"], list)
            or any(not isinstance(row, str) for row in value["defects"])):
        raise ValueError("review defects must be strings")
    for label, digest in value["input_evidence_sha256"].items():
        require_sha(digest, f"review input hash {label}")
    for label, count in value["counts"].items():
        if type(count) is not int or count < 0:
            raise ValueError(f"review count {label} must be non-negative integer")
    for label, count in value["classification_basis"].items():
        if type(count) is not int or count < 0:
            raise ValueError(f"review classification count {label} differs")
    candidate_fields = {
        "candidate_id", "coordinate_xyz", "m7_score", "ct_status",
        "cell_face_distances_voxels", "volume_face_distances_voxels",
        "minimum_cell_face_distance_voxels", "minimum_cell_face_distance_um",
        "minimum_volume_face_distance_voxels", "minimum_volume_face_distance_um",
        "computed_causes", "physically_safe_for_query",
    }
    face_fields = {f"{axis}_{side}" for axis in "xyz" for side in ("low", "high")}
    if not isinstance(value["candidates"], list):
        raise ValueError("review candidates must be an array")
    for index, candidate in enumerate(value["candidates"]):
        _require_fields(candidate, candidate_fields, f"review candidate {index}")
        _require_fields(candidate["cell_face_distances_voxels"], face_fields,
                        f"review candidate {index} cell faces")
        _require_fields(candidate["volume_face_distances_voxels"], face_fields,
                        f"review candidate {index} volume faces")
        if (not isinstance(candidate["coordinate_xyz"], list)
                or len(candidate["coordinate_xyz"]) != 3
                or any(type(row) is not int for row in candidate["coordinate_xyz"])
                or not isinstance(candidate["computed_causes"], list)
                or type(candidate["physically_safe_for_query"]) is not bool):
            raise ValueError(f"review candidate {index} values differ")
        if (not isinstance(candidate["candidate_id"], str)
                or candidate["ct_status"] not in {
                    "CT_MATERIAL_NEARBY", "NO_NEARBY_CT_MATERIAL"
                }
                or any(cause not in {CAUSE_CELL, CAUSE_VOLUME}
                       for cause in candidate["computed_causes"])):
            raise ValueError(f"review candidate {index} identity differs")
        numeric = [candidate["m7_score"],
                   candidate["minimum_cell_face_distance_voxels"],
                   candidate["minimum_cell_face_distance_um"],
                   candidate["minimum_volume_face_distance_voxels"],
                   candidate["minimum_volume_face_distance_um"],
                   *candidate["cell_face_distances_voxels"].values(),
                   *candidate["volume_face_distances_voxels"].values()]
        if any(type(row) not in {int, float} or isinstance(row, bool)
               or not math.isfinite(float(row)) for row in numeric):
            raise ValueError(f"review candidate {index} metrics differ")
    sensitivity_fields = {
        "margin_percentage", "minimum_volume_clearance_voxels",
        "minimum_volume_clearance_um", "m7_candidate_count", "ct_supported_count",
        "retained_count", "physically_safe_retained_count", "diagnostic_only",
    }
    if not isinstance(value["sensitivity"], list) or len(value["sensitivity"]) != 5:
        raise ValueError("review sensitivity must contain five rows")
    for index, row in enumerate(value["sensitivity"]):
        _require_fields(row, sensitivity_fields, f"review sensitivity {index}")
        if row["diagnostic_only"] is not True:
            raise ValueError("review sensitivity must remain diagnostic")
        for field in ("margin_percentage", "m7_candidate_count",
                      "ct_supported_count", "retained_count",
                      "physically_safe_retained_count"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"review sensitivity {index} counts differ")
        for field in ("minimum_volume_clearance_voxels",
                      "minimum_volume_clearance_um"):
            if type(row[field]) not in {int, float} or not math.isfinite(float(row[field])):
                raise ValueError(f"review sensitivity {index} metrics differ")
    if [row["margin_percentage"] for row in value["sensitivity"]] != list(
        SENSITIVITY_PERCENTAGES
    ):
        raise ValueError("review sensitivity percentages differ")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute a frozen candidate-clearance result read-only."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trusted-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.input.resolve() == arguments.output.resolve():
        raise ValueError("input and output paths must be distinct")
    value = json.loads(arguments.input.read_text(encoding="utf-8"))
    evidence = validate_candidate_clearance_review(analyze_candidate_clearance(
        value, trusted_manifest_sha256=arguments.trusted_manifest_sha256
    ))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{arguments.output.name}.", suffix=".tmp",
        dir=arguments.output.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(evidence) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, arguments.output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
