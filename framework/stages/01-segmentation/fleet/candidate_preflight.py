"""Read-only candidate coverage primitive shared by controls and later censuses."""

from __future__ import annotations

import hashlib
import fcntl
import io
import json
import math
import copy
import os
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .common import content_sha256, utc_now
from .ct_support import OmeZarrCtSupportSampler, apply_ct_material_support_gate
from .generator import bounded_preflight_grid_design, generate_tasks_for_snapshot
from .planner import candidate_rank_key, diagnose_candidate_rejections, normalize_candidates, screen_candidates
from .seed_probe import normalize_source_content_lock
from .worker import McpSeedProvider


_VOLATILE_EVIDENCE_KEYS = {
    "generated_at", "generated_at_utc", "started_at", "started_at_utc",
    "updated_at", "updated_at_utc", "completed_at", "completed_at_utc",
    "checked_at", "checked_at_utc", "elapsed_seconds",
}


def _without_volatile_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_volatile_evidence(row) for key, row in value.items()
                if key not in _VOLATILE_EVIDENCE_KEYS}
    if isinstance(value, list):
        return [_without_volatile_evidence(row) for row in value]
    return value


def _coordinate(candidate: object) -> tuple[int, int, int] | None:
    if not isinstance(candidate, dict):
        return None
    value = (candidate.get("ct_l0_coordinate") or candidate.get("coordinate")
             or candidate.get("selected_seed") or candidate)
    try:
        if not isinstance(value, dict):
            return None
        result = tuple(int(value[axis]) for axis in "xyz")
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return result


def survey_candidate_task(task: dict[str, Any], provider,
                          *, ct_sampler=None) -> dict[str, Any]:
    """Run the read-only candidate funnel for one cell.

    This is deliberately usable by both the pending-task survey and the full
    preflight. It has no store argument and therefore no way to claim or alter
    fleet work.
    """
    identity = {
        key: task.get(key) for key in ("task_id", "sample_id", "cell_id", "priority")
    }
    try:
        response = provider.discover(task)
        raw = response.get("candidates")
        if not isinstance(raw, list):
            raise ValueError("provider response has no candidate array")
        ct_filtered, ct_receipt = apply_ct_material_support_gate(
            response, task, sampler=ct_sampler)
        post_ct = ct_filtered.get("candidates") or []
        cell_task = copy.deepcopy(task)
        cell_task.setdefault("candidate_discovery", {})[
            "minimum_volume_interior_clearance_voxels"] = 0
        post_cell = normalize_candidates(ct_filtered, cell_task)
        post_volume = normalize_candidates(ct_filtered, task)
        diagnostics = diagnose_candidate_rejections(ct_filtered, task)
        maximum = int(task.get("parameter_envelope", {}).get(
            "maximum_candidate_count", len(post_volume)))
        if maximum < 1:
            raise ValueError("maximum_candidate_count must be positive")
        packet = sorted(post_volume, key=candidate_rank_key)[:maximum]
        return {
            **identity,
            "counts": {
                "raw_m7": len(raw),
                "post_ct": len(post_ct),
                "post_cell_clearance": len(post_cell),
                "post_volume_clearance": len(post_volume),
                "packet_retained": len(packet),
            },
            "raw_candidates": raw,
            "ct_candidates": post_ct,
            "cell_clearance_candidates": post_cell,
            "volume_clearance_candidates": post_volume,
            "clearance_candidates": post_volume,
            "packet_candidates": packet,
            "best_candidate": packet[0] if packet else None,
            "rejection_counts": diagnostics.get("rejection_counts") or {},
            "ct_material_support": ct_receipt,
            "provider_response_sha256": content_sha256(
                _without_volatile_evidence(response)),
            "provider_response_raw_sha256": content_sha256(response),
            "response": response,
            "provider_exchange": response.get("provider_exchange"),
            "m7_read_set": response.get("source_read_set"),
            "source_error": None,
            "ink_used": False,
        }
    except BaseException as error:
        return {
            **identity,
            "counts": {"raw_m7": 0, "post_ct": 0, "post_cell_clearance": 0,
                       "post_volume_clearance": 0, "packet_retained": 0},
            "raw_candidates": [], "ct_candidates": [],
            "cell_clearance_candidates": [], "volume_clearance_candidates": [],
            "clearance_candidates": [], "packet_candidates": [],
            "best_candidate": None, "rejection_counts": {},
            "provider_response_sha256": None, "provider_response_raw_sha256": None,
            "provider_exchange": None,
            "response": None,
            "m7_read_set": None,
            "source_error": f"{type(error).__name__}: {error}",
            "ink_used": False,
        }


def aggregate_candidate_survival(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Deduplicate after each gate, retaining survival from any covering cell."""
    stages = {
        "raw": "raw_candidates",
        "post_ct": "ct_candidates",
        "post_cell_clearance": "cell_clearance_candidates",
        "post_volume_clearance": "volume_clearance_candidates",
        "packet_retained": "packet_candidates",
    }
    unique: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {
        stage: {} for stage in stages
    }
    raw_observations = 0
    for row in rows:
        for stage, field in stages.items():
            for candidate in row.get(field) or []:
                key = _coordinate(candidate)
                if key is None:
                    continue
                if stage == "raw":
                    raw_observations += 1
                existing = unique[stage].get(key)
                # A candidate may appear in overlapping cells with different
                # measured clearances. Keep the strongest surviving observation;
                # never let the first cell's rejection erase a later survival.
                strength = (
                    float(candidate.get("score", 0.0)),
                    float(candidate.get("cell_interior_clearance_voxels", float("-inf"))),
                    float(candidate.get("volume_interior_clearance_voxels", float("-inf"))),
                )
                old_strength = (
                    float(existing.get("score", 0.0)),
                    float(existing.get("cell_interior_clearance_voxels", float("-inf"))),
                    float(existing.get("volume_interior_clearance_voxels", float("-inf"))),
                ) if existing else None
                if existing is None or strength > old_strength:
                    unique[stage][key] = dict(candidate)
    return {
        "unique_raw": len(unique["raw"]),
        "unique_post_ct": len(unique["post_ct"]),
        "unique_post_cell_clearance": len(unique["post_cell_clearance"]),
        "unique_post_volume_clearance": len(unique["post_volume_clearance"]),
        "unique_post_clearance": len(unique["post_volume_clearance"]),
        "unique_packet_retained": len(unique["packet_retained"]),
        "duplicate_observations": max(0, raw_observations - len(unique["raw"])),
        "candidates": unique,
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p95": None}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {"minimum": ordered[0], "median": float(median(ordered)),
            "p95": ordered[p95_index]}


def _bin_for(center: dict[str, Any], shape: list[int], sides: int = 4) -> tuple[int, int, int]:
    return tuple(min(sides - 1, max(0, int(float(center[axis]) * sides / shape[index])))
                 for index, axis in enumerate("xyz"))


SCIENTIFIC_DUPLICATE_FIELDS = (
    "status", "measurement_kind", "sampling_design",
    "planned_sampling_percentage", "achieved_successful_sampling_percentage",
    "sample_id", "source_snapshot_id", "bindings", "gates", "funnel",
    "no_candidate_causes", "score_statistics", "cell_clearance_statistics",
    "volume_clearance_statistics", "spatial_bins", "candidate_coordinates_xyz",
    "state_mutation", "growth_allowed", "ink_used", "non_claim",
    "normalized_request",
)
SCIENTIFIC_CORE_SCHEMA = (
    "campaignx.segment_candidate_coverage_preflight.scientific_core.v1")
SCIENTIFIC_CORE_FIELDS = frozenset({
    "schema", *SCIENTIFIC_DUPLICATE_FIELDS, "cell_outcomes",
    "m7_read_set_manifest_sha256", "ct_read_set_manifest_sha256",
})


def candidate_receipt_scientific_duplicates_match(private: dict[str, Any]) -> bool:
    """Duplicated readable fields must be exact projections of the signed core."""
    core = private.get("scientific_core")
    return isinstance(core, dict) and all(
        private.get(key) == core.get(key) for key in SCIENTIFIC_DUPLICATE_FIELDS
    )


def validate_candidate_scientific_core(core: object) -> dict[str, Any]:
    """Accept only the complete V1 core whose hash carries scientific meaning."""
    def finite_number(value: object) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)))

    def nonnegative_integer(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    def lowercase_digest(value: object, length: int = 64) -> bool:
        return (isinstance(value, str) and len(value) == length
                and all(character in "0123456789abcdef" for character in value))

    def expected_golden_stride(population: int) -> int:
        golden_numerator = 6_180_339_887_498_949
        golden_denominator = 10_000_000_000_000_000
        target = max(1, (
            population * golden_numerator + golden_denominator // 2
        ) // golden_denominator)
        for distance in range(population + 1):
            for candidate in (target - distance, target + distance):
                if (0 < candidate < population
                        and math.gcd(candidate, population) == 1):
                    return candidate
        raise ValueError("candidate preflight estimate stride cannot be derived")

    if not isinstance(core, dict) or core.get("schema") != SCIENTIFIC_CORE_SCHEMA:
        raise ValueError("candidate preflight scientific core schema is unsupported")
    if set(core) != SCIENTIFIC_CORE_FIELDS:
        raise ValueError("candidate preflight scientific core is incomplete")
    dict_fields = {
        "sampling_design", "bindings", "gates", "funnel", "no_candidate_causes",
        "score_statistics", "cell_clearance_statistics",
        "volume_clearance_statistics", "normalized_request",
    }
    list_fields = {
        "spatial_bins", "candidate_coordinates_xyz", "cell_outcomes",
        "m7_read_set_manifest_sha256", "ct_read_set_manifest_sha256",
    }
    if any(not isinstance(core[field], dict) for field in dict_fields):
        raise ValueError("candidate preflight scientific core has invalid mappings")
    if any(not isinstance(core[field], list) for field in list_fields):
        raise ValueError("candidate preflight scientific core has invalid lists")
    if core["measurement_kind"] not in {
            "CENSUS", "ESTIMATE", "INCOMPLETE_CENSUS", "INCOMPLETE_ESTIMATE"}:
        raise ValueError("candidate preflight measurement kind is invalid")
    if core["status"] not in {"COMPLETE", "COMPLETE_WITH_SOURCE_ERRORS"}:
        raise ValueError("candidate preflight status is invalid")
    if (not isinstance(core["sample_id"], str)
            or not isinstance(core["source_snapshot_id"], str)
            or not isinstance(core["non_claim"], str)
            or not isinstance(core["state_mutation"], str)
            or not isinstance(core["growth_allowed"], bool)
            or not isinstance(core["ink_used"], bool)):
        raise ValueError("candidate preflight scientific scalar types are invalid")
    if any(not finite_number(core[field]) or not 0 <= float(core[field]) <= 100
           for field in (
               "planned_sampling_percentage", "achieved_successful_sampling_percentage")):
        raise ValueError("candidate preflight sampling percentages are invalid")
    required_bindings = {
        "sample_id", "mission_id", "p0_artifact_id", "p0_artifact_sha256",
        "p0_selection_version", "p0_selection_sha256", "catalog_snapshot_sha256",
        "source_snapshot_id", "source_content_lock_sha256", "ct_sha256", "m7_sha256",
        "m7_uri_sha256",
        "coordinate_frame", "voxel_size_um", "m7_threshold", "grid_version",
        "policy_version", "provider", "candidate_selection_policy",
        "seed_region_policy", "selection_strategy", "maximum_cells", "shape_xyz",
        "grid_step", "query_radius", "parallelism", "source_snapshot_sha256",
        "normalized_request_sha256", "code_revision",
    }
    if set(core["bindings"]) != required_bindings:
        raise ValueError("candidate preflight source bindings are incomplete")
    bindings = core["bindings"]
    digest_fields = {
        "p0_artifact_sha256", "p0_selection_sha256", "catalog_snapshot_sha256",
        "source_content_lock_sha256", "ct_sha256", "m7_sha256",
        "m7_uri_sha256", "source_snapshot_sha256", "normalized_request_sha256",
    }
    if any(not lowercase_digest(bindings[field]) for field in digest_fields):
        raise ValueError("candidate preflight binding digests are invalid")
    revision = bindings["code_revision"]
    if not lowercase_digest(revision, 40):
        raise ValueError("candidate preflight code revision is invalid")
    if (not isinstance(bindings["shape_xyz"], list)
            or len(bindings["shape_xyz"]) != 3
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 1
                   for value in bindings["shape_xyz"])):
        raise ValueError("candidate preflight bound shape is invalid")
    string_bindings = required_bindings - digest_fields - {
        "code_revision", "shape_xyz", "maximum_cells", "grid_step", "query_radius",
        "parallelism", "voxel_size_um", "m7_threshold"}
    if any(not isinstance(bindings[field], str) or not bindings[field]
           for field in string_bindings):
        raise ValueError("candidate preflight string bindings are invalid")
    if (not isinstance(bindings["maximum_cells"], int)
            or isinstance(bindings["maximum_cells"], bool)
            or not 1 <= bindings["maximum_cells"] <= 4096
            or not isinstance(bindings["grid_step"], int)
            or isinstance(bindings["grid_step"], bool)
            or not 2 <= bindings["grid_step"] <= 16384
            or not isinstance(bindings["query_radius"], int)
            or isinstance(bindings["query_radius"], bool)
            or not 1 <= bindings["query_radius"] <= 4096
            or bindings["grid_step"] < 2 * bindings["query_radius"]
            or not isinstance(bindings["parallelism"], int)
            or isinstance(bindings["parallelism"], bool)
            or not 1 <= bindings["parallelism"] <= 8
            or not finite_number(bindings["voxel_size_um"])
            or float(bindings["voxel_size_um"]) <= 0
            or not finite_number(bindings["m7_threshold"])
            or not 0 <= float(bindings["m7_threshold"]) <= 1):
        raise ValueError("candidate preflight numeric bindings are invalid")
    normalized_request = core["normalized_request"]
    required_request = {
        "sample_id", "mission_id", "provider", "catalog_snapshot_sha256", "grid_version",
        "policy_version", "grid_step", "query_radius", "cell_clearance",
        "volume_clearance", "candidate_interior_clearance", "selection_strategy",
        "candidate_selection_policy", "seed_region_policy", "m7_threshold",
        "packet_candidate_limit", "maximum_cells", "parallelism",
        "ct_material_support_gate", "p0_artifact_id", "p0_artifact_sha256",
        "p0_selection_version", "p0_selection_sha256",
        "source_content_lock_sha256",
    }
    if (set(normalized_request) != required_request
            or content_sha256(normalized_request) !=
                bindings["normalized_request_sha256"]):
        raise ValueError("candidate preflight normalized request binding is invalid")
    required_funnel = {
        "total_grid_cells", "grid_cells_in_design_sample",
        "geometrically_eligible_cells", "geometrically_eligible_cells_estimate",
        "geometrically_eligible_sampled_cells", "cells_attempted",
        "cells_surveyed_successfully", "cells_failed_source",
        "cells_with_raw_m7_candidates", "raw_m7_candidates",
        "post_ct_candidates", "post_cell_clearance_candidates",
        "post_volume_clearance_candidates", "packet_retained_candidates",
    }
    exact_funnel = required_funnel | {
        "cells_surveyed", "raw_m7_candidate_observations",
        "duplicate_candidate_observations", "source_errors",
    }
    if set(core["funnel"]) != exact_funnel:
        raise ValueError("candidate preflight funnel is incomplete")
    for key in exact_funnel:
        value = core["funnel"][key]
        if (key == "geometrically_eligible_cells" and value is None):
            continue
        if not nonnegative_integer(value):
            raise ValueError("candidate preflight funnel counts are invalid")
    funnel = core["funnel"]
    if (funnel["grid_cells_in_design_sample"] > funnel["total_grid_cells"]
            or funnel["geometrically_eligible_sampled_cells"] >
                funnel["grid_cells_in_design_sample"]
            or funnel["cells_attempted"] > funnel["geometrically_eligible_sampled_cells"]
            or funnel["cells_attempted"] !=
                funnel["cells_surveyed_successfully"] + funnel["cells_failed_source"]
            or funnel["cells_surveyed"] != funnel["cells_surveyed_successfully"]
            or funnel["source_errors"] != funnel["cells_failed_source"]):
        raise ValueError("candidate preflight funnel relationships are invalid")
    required_design = {
        "name", "measurement_kind", "population_grid_cells", "sampled_grid_cells",
        "inclusion_fraction", "ordinal_rule", "ordinal_stride", "ordinal_offset",
        "sampled_index_sha256", "cell_order_sha256",
    }
    if set(core["sampling_design"]) != required_design:
        raise ValueError("candidate preflight sampling design is incomplete")
    design = core["sampling_design"]
    if (not isinstance(design["name"], str) or not design["name"]
            or design["measurement_kind"] not in {"CENSUS", "ESTIMATE"}
            or not nonnegative_integer(design["population_grid_cells"])
            or not nonnegative_integer(design["sampled_grid_cells"])
            or design["sampled_grid_cells"] > design["population_grid_cells"]
            or not finite_number(design["inclusion_fraction"])
            or not 0 <= float(design["inclusion_fraction"]) <= 1
            or not isinstance(design["ordinal_rule"], str) or not design["ordinal_rule"]
            or not isinstance(design["ordinal_stride"], int)
            or isinstance(design["ordinal_stride"], bool) or design["ordinal_stride"] < 1
            or not nonnegative_integer(design["ordinal_offset"])
            or not lowercase_digest(design["sampled_index_sha256"])
            or not lowercase_digest(design["cell_order_sha256"])):
        raise ValueError("candidate preflight sampling design values are invalid")
    census_rule = "lexicographic mixed-radix enumeration"
    estimate_rule = (
        "unflatten((floor(stride/2)+i*stride) mod N), "
        "stride nearest round(N/phi) with gcd(stride,N)=1"
    )
    population = design["population_grid_cells"]
    sampled = design["sampled_grid_cells"]
    if design["measurement_kind"] == "CENSUS":
        if (design["name"] != "complete-eligible-grid-census-v1"
                or design["ordinal_rule"] != census_rule
                or design["ordinal_stride"] != 1
                or design["ordinal_offset"] != 0):
            raise ValueError("candidate preflight census design is noncanonical")
    elif (population < 2
            or design["name"] !=
                "deterministic-golden-coprime-rank1-grid-sample-v1"
            or design["ordinal_rule"] != estimate_rule
            or design["ordinal_stride"] != expected_golden_stride(population)
            or design["ordinal_offset"] != design["ordinal_stride"] // 2):
        raise ValueError("candidate preflight estimate design is noncanonical")
    if (not all(isinstance(row, dict) for row in core["spatial_bins"])
            or not all(isinstance(row, dict) for row in core["candidate_coordinates_xyz"])
            or not all(isinstance(row, dict) for row in core["cell_outcomes"])):
        raise ValueError("candidate preflight scientific rows are invalid")
    bin_fields = {
        "bin_xyz", "total_cells", "surveyed_cells", "candidate_bearing_cells",
        "usable_candidate_cells", "sampled_eligible_cells",
    }
    spatial_coordinates: list[tuple[int, int, int]] = []
    for row in core["spatial_bins"]:
        coordinates = row.get("bin_xyz")
        if (set(row) != bin_fields or not isinstance(coordinates, list)
                or len(coordinates) != 3
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or not 0 <= value < 4 for value in coordinates)
                or any(not nonnegative_integer(row[field])
                       for field in bin_fields - {"bin_xyz"})
                or row["sampled_eligible_cells"] > row["total_cells"]
                or row["surveyed_cells"] > row["sampled_eligible_cells"]
                or row["candidate_bearing_cells"] > row["surveyed_cells"]
                or row["usable_candidate_cells"] > row["surveyed_cells"]):
            raise ValueError("candidate preflight spatial bin is invalid")
        spatial_coordinates.append(tuple(coordinates))
    if (spatial_coordinates != sorted(spatial_coordinates)
            or len(spatial_coordinates) != len(set(spatial_coordinates))):
        raise ValueError("candidate preflight spatial bins are noncanonical")
    for row in core["candidate_coordinates_xyz"]:
        if (set(row) != {"x", "y", "z"}
                or any(not nonnegative_integer(row[axis]) for axis in "xyz")):
            raise ValueError("candidate preflight private coordinate is invalid")
    if (any(not isinstance(key, str) or not key or not nonnegative_integer(value)
            for key, value in core["no_candidate_causes"].items())):
        raise ValueError("candidate preflight no-candidate causes are invalid")
    for field in ("score_statistics", "cell_clearance_statistics",
                  "volume_clearance_statistics"):
        statistics = core[field]
        if (set(statistics) != {"minimum", "median", "p95"}
                or any(value is not None and not finite_number(value)
                       for value in statistics.values())):
            raise ValueError("candidate preflight statistics are invalid")
    gates = core["gates"]
    if set(gates) != {"cell_clearance", "volume_clearance",
                      "candidate_interior_clearance", "ct_material_support_gate",
                      "packet_candidate_limit"}:
        raise ValueError("candidate preflight gates are incomplete")
    ct_gate = gates["ct_material_support_gate"]
    if (not finite_number(gates["cell_clearance"])
            or float(gates["cell_clearance"]) < 0
            or any(not nonnegative_integer(gates[field]) for field in (
                "volume_clearance", "candidate_interior_clearance",
                "packet_candidate_limit"))
            or gates["packet_candidate_limit"] < 1
            or not isinstance(ct_gate, dict)
            or set(ct_gate) != {"policy", "level", "radius_l0_voxels",
                                "minimum_nonzero_voxels"}
            or ct_gate["policy"] != "ome-zarr-nearby-material-v1"
            or any(not nonnegative_integer(ct_gate[field]) for field in (
                "level", "radius_l0_voxels", "minimum_nonzero_voxels"))
            or ct_gate["radius_l0_voxels"] < 1
            or ct_gate["minimum_nonzero_voxels"] < 1):
        raise ValueError("candidate preflight gate values are invalid")
    for field in ("m7_read_set_manifest_sha256", "ct_read_set_manifest_sha256"):
        values = core[field]
        if (any(not lowercase_digest(value) for value in values)
                or values != sorted(set(values))):
            raise ValueError("candidate preflight read-set manifest digests are invalid")

    # Cross-field meaning is part of the signed contract too.  Without these
    # relations a perfectly typed sample can be relabelled as an exact census.
    base_kind = design["measurement_kind"]
    failed = funnel["cells_failed_source"]
    expected_measurement = f"INCOMPLETE_{base_kind}" if failed else base_kind
    expected_status = "COMPLETE_WITH_SOURCE_ERRORS" if failed else "COMPLETE"
    expected_fraction = sampled / population if population else 1.0
    expected_planned = 100.0 * expected_fraction
    expected_achieved = (100.0 * funnel["cells_surveyed_successfully"] / population
                         if population else 100.0)
    if (core["measurement_kind"] != expected_measurement
            or core["status"] != expected_status
            or funnel["source_errors"] != failed
            or funnel["cells_attempted"] !=
                funnel["geometrically_eligible_sampled_cells"]
            or core["no_candidate_causes"].get("SOURCE_ERROR", 0) != failed
            or design["population_grid_cells"] != funnel["total_grid_cells"]
            or design["sampled_grid_cells"] != funnel["grid_cells_in_design_sample"]
            or not math.isclose(float(design["inclusion_fraction"]), expected_fraction,
                                rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(float(core["planned_sampling_percentage"]),
                                expected_planned, rel_tol=1e-12, abs_tol=1e-9)
            or not math.isclose(float(core["achieved_successful_sampling_percentage"]),
                                expected_achieved, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError("candidate preflight measurement semantics are contradictory")
    request_to_binding = {
        "sample_id": "sample_id", "mission_id": "mission_id", "provider": "provider",
        "catalog_snapshot_sha256": "catalog_snapshot_sha256",
        "grid_version": "grid_version", "policy_version": "policy_version",
        "selection_strategy": "selection_strategy",
        "candidate_selection_policy": "candidate_selection_policy",
        "seed_region_policy": "seed_region_policy", "m7_threshold": "m7_threshold",
        "maximum_cells": "maximum_cells", "grid_step": "grid_step",
        "query_radius": "query_radius", "parallelism": "parallelism",
        "p0_artifact_id": "p0_artifact_id",
        "p0_artifact_sha256": "p0_artifact_sha256",
        "p0_selection_version": "p0_selection_version",
        "p0_selection_sha256": "p0_selection_sha256",
        "source_content_lock_sha256": "source_content_lock_sha256",
    }
    request_to_gate = {
        "cell_clearance": "cell_clearance",
        "volume_clearance": "volume_clearance",
        "candidate_interior_clearance": "candidate_interior_clearance",
        "packet_candidate_limit": "packet_candidate_limit",
        "ct_material_support_gate": "ct_material_support_gate",
    }
    if (any(normalized_request[request_key] != bindings[binding_key]
            for request_key, binding_key in request_to_binding.items())
            or any(normalized_request[request_key] != gates[gate_key]
                   for request_key, gate_key in request_to_gate.items())
            or core["sample_id"] != bindings["sample_id"]
            or core["source_snapshot_id"] != bindings["source_snapshot_id"]
            or gates["volume_clearance"] < bindings["query_radius"]
            or core["state_mutation"] != "NONE"
            or core["growth_allowed"] is not False
            or core["ink_used"] is not False):
        raise ValueError("candidate preflight execution bindings are contradictory")
    derived_design = bounded_preflight_grid_design(
        bindings["shape_xyz"], query_radius=bindings["query_radius"],
        grid_step=bindings["grid_step"],
        volume_edge_margin=gates["volume_clearance"],
        hard_cell_limit=bindings["maximum_cells"],
    )
    expected_grid_bins = {
        tuple(key): int(value)
        for key, value in derived_design["grid_bin_counts"].items()
    }
    actual_grid_bins = {
        tuple(row["bin_xyz"]): row["total_cells"] for row in core["spatial_bins"]
    }
    if (design["measurement_kind"] != derived_design["measurement_kind"]
            or population != derived_design["total_grid_cells"]
            or sampled != len(derived_design["indices"])
            or design["ordinal_rule"] != derived_design["ordinal_rule"]
            or design["ordinal_stride"] != derived_design["ordinal_stride"]
            or design["ordinal_offset"] != derived_design["ordinal_offset"]
            or design["sampled_index_sha256"] !=
                content_sha256(derived_design["indices"])
            or actual_grid_bins != expected_grid_bins):
        raise ValueError("candidate preflight request does not derive its design")
    stage_counts = (
        funnel["raw_m7_candidates"], funnel["post_ct_candidates"],
        funnel["post_cell_clearance_candidates"],
        funnel["post_volume_clearance_candidates"],
        funnel["packet_retained_candidates"],
    )
    coordinate_tuples = [tuple(row[axis] for axis in "xyz")
                         for row in core["candidate_coordinates_xyz"]]
    if (any(left < right for left, right in zip(stage_counts, stage_counts[1:]))
            or funnel["raw_m7_candidate_observations"] < funnel["raw_m7_candidates"]
            or funnel["duplicate_candidate_observations"] !=
                funnel["raw_m7_candidate_observations"] - funnel["raw_m7_candidates"]
            or len(core["candidate_coordinates_xyz"]) !=
                funnel["packet_retained_candidates"]
            or len(set(coordinate_tuples)) != len(coordinate_tuples)
            or coordinate_tuples != sorted(coordinate_tuples)
            or any(any(coordinate[axis] >= bindings["shape_xyz"][axis]
                       for axis in range(3)) for coordinate in coordinate_tuples)
            or funnel["cells_with_raw_m7_candidates"] >
                funnel["cells_surveyed_successfully"]):
        raise ValueError("candidate preflight filter funnel is contradictory")
    statistics_fields = (
        "score_statistics", "cell_clearance_statistics",
        "volume_clearance_statistics",
    )
    if funnel["packet_retained_candidates"] == 0:
        if any(value is not None for field in statistics_fields
               for value in core[field].values()):
            raise ValueError("candidate preflight empty-packet statistics are invalid")
    else:
        for field in statistics_fields:
            statistics = core[field]
            if (any(value is None for value in statistics.values())
                    or not (statistics["minimum"] <= statistics["median"] <=
                            statistics["p95"])):
                raise ValueError("candidate preflight ordered statistics are invalid")
    if base_kind == "CENSUS":
        if (sampled != population
                or funnel["geometrically_eligible_cells"] is None
                or funnel["geometrically_eligible_cells"] !=
                    funnel["geometrically_eligible_sampled_cells"]
                or funnel["geometrically_eligible_cells_estimate"] !=
                    funnel["geometrically_eligible_cells"]):
            raise ValueError("candidate preflight census semantics are contradictory")
    else:
        expected_eligible_estimate = (
            round(population * funnel["geometrically_eligible_sampled_cells"] / sampled)
            if sampled else 0)
        if (sampled >= population
                or funnel["geometrically_eligible_cells"] is not None
                or funnel["geometrically_eligible_cells_estimate"] !=
                    expected_eligible_estimate
                or not 0 <= funnel["geometrically_eligible_cells_estimate"] <= population):
            raise ValueError("candidate preflight estimate semantics are contradictory")
    if (sum(row["total_cells"] for row in core["spatial_bins"]) != population
            or sum(row["sampled_eligible_cells"] for row in core["spatial_bins"]) !=
                funnel["geometrically_eligible_sampled_cells"]
            or sum(row["surveyed_cells"] for row in core["spatial_bins"]) !=
                funnel["cells_surveyed_successfully"]
            or sum(row["candidate_bearing_cells"] for row in core["spatial_bins"]) !=
                funnel["cells_with_raw_m7_candidates"]):
        raise ValueError("candidate preflight spatial totals contradict its funnel")
    causes = core["no_candidate_causes"]
    supported_causes = {
        "SOURCE_ERROR", "NO_M7_CANDIDATES", "NO_CT_SUPPORTED_CANDIDATES",
        "NO_CELL_CLEARANCE_ELIGIBLE_CANDIDATES",
        "NO_CLEARANCE_ELIGIBLE_CANDIDATES", "NO_PACKET_RETAINED_CANDIDATES",
    }
    usable_cells = sum(row["usable_candidate_cells"] for row in core["spatial_bins"])
    outcome_count_fields = {
        "raw_m7", "post_ct", "post_cell_clearance",
        "post_volume_clearance", "packet_retained",
    }
    outcome_fields = {
        "cell_id", "counts", "rejection_counts",
        "provider_response_sha256", "source_error",
    }
    derived_causes: dict[str, int] = {}
    outcome_cell_ids: list[str] = []
    outcome_totals = {field: 0 for field in outcome_count_fields}
    outcome_successes = 0
    outcome_failures = 0
    outcome_bearing = 0
    outcome_usable = 0
    for outcome in core["cell_outcomes"]:
        counts = outcome.get("counts")
        rejections = outcome.get("rejection_counts")
        if (set(outcome) != outcome_fields
                or not isinstance(outcome.get("cell_id"), str)
                or not outcome["cell_id"]
                or not isinstance(counts, dict)
                or set(counts) != outcome_count_fields
                or any(not nonnegative_integer(counts[field])
                       for field in outcome_count_fields)
                or not isinstance(rejections, dict)
                or any(not isinstance(key, str) or not key
                       or not nonnegative_integer(value)
                       for key, value in rejections.items())):
            raise ValueError("candidate preflight cell outcome is invalid")
        per_cell_stages = (
            counts["raw_m7"], counts["post_ct"], counts["post_cell_clearance"],
            counts["post_volume_clearance"], counts["packet_retained"],
        )
        if any(left < right for left, right in zip(per_cell_stages,
                                                   per_cell_stages[1:])):
            raise ValueError("candidate preflight cell outcome stages are contradictory")
        source_error = outcome.get("source_error")
        provider_digest = outcome.get("provider_response_sha256")
        if source_error is None:
            if not lowercase_digest(provider_digest):
                raise ValueError("candidate preflight successful cell lacks provenance")
            outcome_successes += 1
        else:
            if (not isinstance(source_error, str) or not source_error
                    or provider_digest is not None or any(per_cell_stages)):
                raise ValueError("candidate preflight failed cell is contradictory")
            outcome_failures += 1
        cause = None
        if source_error is not None:
            cause = "SOURCE_ERROR"
        elif counts["raw_m7"] == 0:
            cause = "NO_M7_CANDIDATES"
        elif counts["post_ct"] == 0:
            cause = "NO_CT_SUPPORTED_CANDIDATES"
        elif counts["post_cell_clearance"] == 0:
            cause = "NO_CELL_CLEARANCE_ELIGIBLE_CANDIDATES"
        elif counts["post_volume_clearance"] == 0:
            cause = "NO_CLEARANCE_ELIGIBLE_CANDIDATES"
        elif counts["packet_retained"] == 0:
            cause = "NO_PACKET_RETAINED_CANDIDATES"
        if cause is not None:
            derived_causes[cause] = derived_causes.get(cause, 0) + 1
        outcome_cell_ids.append(outcome["cell_id"])
        outcome_bearing += int(counts["raw_m7"] > 0)
        outcome_usable += int(counts["packet_retained"] > 0)
        for field in outcome_count_fields:
            outcome_totals[field] += counts[field]
    if (len(core["cell_outcomes"]) != funnel["cells_attempted"]
            or len(outcome_cell_ids) != len(set(outcome_cell_ids))
            or outcome_successes != funnel["cells_surveyed_successfully"]
            or outcome_failures != funnel["cells_failed_source"]
            or outcome_bearing != funnel["cells_with_raw_m7_candidates"]
            or outcome_usable != usable_cells
            or outcome_totals["raw_m7"] !=
                funnel["raw_m7_candidate_observations"]
            or any(outcome_totals[field] < funnel[aggregate_field]
                   for field, aggregate_field in (
                       ("raw_m7", "raw_m7_candidates"),
                       ("post_ct", "post_ct_candidates"),
                       ("post_cell_clearance", "post_cell_clearance_candidates"),
                       ("post_volume_clearance", "post_volume_clearance_candidates"),
                       ("packet_retained", "packet_retained_candidates"),
                   ))
            or dict(sorted(derived_causes.items())) != causes
            or (outcome_successes > 0 and
                not core["m7_read_set_manifest_sha256"])
            or (funnel["raw_m7_candidate_observations"] > 0 and
                not core["ct_read_set_manifest_sha256"])):
        raise ValueError("candidate preflight cell-outcome derivation is contradictory")
    stage_failure_cells = sum(causes.get(name, 0) for name in (
        "NO_CT_SUPPORTED_CANDIDATES",
        "NO_CELL_CLEARANCE_ELIGIBLE_CANDIDATES",
        "NO_CLEARANCE_ELIGIBLE_CANDIDATES", "NO_PACKET_RETAINED_CANDIDATES",
    ))
    if (not set(causes) <= supported_causes
            or (funnel["raw_m7_candidates"] == 0) !=
                (funnel["cells_with_raw_m7_candidates"] == 0)
            or (funnel["packet_retained_candidates"] == 0) != (usable_cells == 0)
            or sum(causes.values()) + usable_cells != funnel["cells_attempted"]
            or causes.get("NO_M7_CANDIDATES", 0) +
                funnel["cells_with_raw_m7_candidates"] !=
                funnel["cells_surveyed_successfully"]
            or stage_failure_cells + usable_cells !=
                funnel["cells_with_raw_m7_candidates"]):
        raise ValueError("candidate preflight cell outcomes contradict its causes")
    return core


def validate_candidate_preflight_receipt_pair(
    private: object, public: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the complete private/public pair against the signed V1 core."""
    if not isinstance(private, dict) or not isinstance(public, dict):
        raise ValueError("candidate preflight receipt pair must contain JSON objects")
    if private.get("schema") != "campaignx.segment_candidate_coverage_preflight.v1":
        raise ValueError("candidate preflight private schema is unsupported")
    core = validate_candidate_scientific_core(private.get("scientific_core"))
    digest = content_sha256(core)
    if private.get("receipt_sha256") != digest:
        raise ValueError("candidate preflight scientific core hash is invalid")
    if not candidate_receipt_scientific_duplicates_match(private):
        raise ValueError("candidate preflight top-level fields differ from its signed core")
    if public.get("schema") != "campaignx.segment_candidate_coverage_preflight.sanitized.v1":
        raise ValueError("candidate preflight sanitized schema is unsupported")
    if public.get("private_receipt_sha256") != digest:
        raise ValueError("candidate preflight sanitized receipt binds another private core")
    expected_public_hash = content_sha256({
        key: value for key, value in public.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })
    if public.get("receipt_sha256") != expected_public_hash:
        raise ValueError("candidate preflight sanitized hash is invalid")
    if sanitize_candidate_coverage_receipt(private) != public:
        raise ValueError("candidate preflight sanitized projection is invalid")
    return private, public


def sanitize_candidate_coverage_receipt(private: dict[str, Any]) -> dict[str, Any]:
    """Return the browser-safe aggregate, never raw cells or candidate XYZ."""
    hidden = {"candidate_coordinates_xyz", "cells", "scientific_core",
              "provider_response_hashes",
              "ct_read_sets", "m7_read_sets"}
    public = {key: value for key, value in private.items()
              if key not in hidden and key not in {"receipt_sha256", "generated_at_utc"}}
    core = private.get("scientific_core")
    if isinstance(core, dict):
        for key in SCIENTIFIC_DUPLICATE_FIELDS:
            if key in core and key not in hidden:
                public[key] = core[key]
    public.update({
        "schema": "campaignx.segment_candidate_coverage_preflight.sanitized.v1",
        "private_receipt_sha256": private.get("receipt_sha256"),
        "generated_at_utc": private.get("generated_at_utc"),
    })
    public["receipt_sha256"] = content_sha256({
        key: value for key, value in public.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })
    return public


def _receipt_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n").encode("utf-8")


def _read_receipt_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"candidate preflight receipt is not an object: {path.name}")
    return value


def _stage_receipt(path: Path, value: dict[str, Any], mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(_receipt_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
        staged = Path(handle.name)
    staged.chmod(mode)
    return staged


def persist_candidate_preflight_receipt_pair(
    private_path: Path, public_path: Path,
    private: dict[str, Any], public: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Publish or reuse one immutable receipt pair under a cross-process lock.

    The scientific hash intentionally excludes observation time, so an exact
    rerun may have a new timestamp but the same paths.  A complete existing
    pair is therefore validated and reused byte-for-byte.  A private-only
    interrupted publication is repaired from its signed core.  New pairs are
    staged first and linked create-once; failure of the second link rolls back
    the first, so this helper never creates an unrecoverable half-pair.
    """
    private_path, public_path = Path(private_path), Path(public_path)
    if private_path == public_path:
        raise ValueError("private and sanitized receipt paths must differ")
    validate_candidate_preflight_receipt_pair(private, public)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = private_path.with_name(private_path.name + ".pair.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        private_exists, public_exists = private_path.exists(), public_path.exists()
        if private_exists and public_exists:
            existing_private = _read_receipt_object(private_path)
            existing_public = _read_receipt_object(public_path)
            validate_candidate_preflight_receipt_pair(existing_private, existing_public)
            if existing_private["scientific_core"] != private["scientific_core"]:
                raise ValueError("hash-named preflight receipt pair collision")
            return {"private_receipt": existing_private,
                    "sanitized_receipt": existing_public}
        if private_exists:
            existing_private = _read_receipt_object(private_path)
            existing_public = sanitize_candidate_coverage_receipt(existing_private)
            validate_candidate_preflight_receipt_pair(existing_private, existing_public)
            if existing_private["scientific_core"] != private["scientific_core"]:
                raise ValueError("hash-named private receipt collision")
            staged_public = _stage_receipt(public_path, existing_public, 0o644)
            try:
                os.link(staged_public, public_path)
            finally:
                staged_public.unlink(missing_ok=True)
            return {"private_receipt": existing_private,
                    "sanitized_receipt": existing_public}
        if public_exists:
            existing_public = _read_receipt_object(public_path)
            if existing_public != public:
                raise ValueError("orphan sanitized receipt cannot bind this private core")
            validate_candidate_preflight_receipt_pair(private, existing_public)
            staged_private = _stage_receipt(private_path, private, 0o600)
            try:
                os.link(staged_private, private_path)
            finally:
                staged_private.unlink(missing_ok=True)
            return {"private_receipt": private,
                    "sanitized_receipt": existing_public}

        staged_private = _stage_receipt(private_path, private, 0o600)
        staged_public = _stage_receipt(public_path, public, 0o644)
        private_published = False
        try:
            os.link(staged_private, private_path)
            private_published = True
            os.link(staged_public, public_path)
        except BaseException:
            if private_published:
                private_path.unlink(missing_ok=True)
            raise
        finally:
            staged_private.unlink(missing_ok=True)
            staged_public.unlink(missing_ok=True)
        return {"private_receipt": private, "sanitized_receipt": public}


def run_candidate_coverage_preflight(
    snapshot: dict[str, Any], request: dict[str, Any], *, surface_view,
    provider=None, ct_sampler=None, code_revision: str,
) -> dict[str, Any]:
    """Survey a deterministic, source-locked in-memory grid without queue writes."""
    integer_request_bounds = {
        "maximum_cells": (1, 4096), "parallelism": (1, 8),
        "grid_step": (2, 16384), "query_radius": (1, 4096),
        "volume_clearance": (1, 8192), "candidate_interior_clearance": (0, 4096),
        "packet_candidate_limit": (1, 100),
    }
    for field, (minimum, maximum) in integer_request_bounds.items():
        value = request.get(field)
        if (not isinstance(value, int) or isinstance(value, bool)
                or not minimum <= value <= maximum):
            raise ValueError(f"{field} must be an integer between {minimum} and {maximum}")
    maximum_cells = request["maximum_cells"]
    parallelism = request["parallelism"]
    if request["grid_step"] < 2 * request["query_radius"]:
        raise ValueError("grid_step must be at least twice query_radius")
    if request["volume_clearance"] < request["query_radius"]:
        raise ValueError("volume_clearance must be at least query_radius")
    if not isinstance(code_revision, str) or not __import__("re").fullmatch(
            r"[0-9a-f]{40}", code_revision):
        raise ValueError("code_revision must be an exact lowercase 40-hex commit")
    if not isinstance(request.get("mission_id"), str) or not request["mission_id"]:
        raise ValueError("candidate preflight requires a named mission binding")
    if request.get("provider") != "vc3d-mcp":
        raise ValueError("unsupported candidate preflight provider")
    source_lock = snapshot.get("source_content_lock")
    expected_lock_sha = request.get("source_content_lock_sha256")
    if not isinstance(source_lock, dict) or content_sha256(source_lock) != expected_lock_sha:
        raise ValueError("source content lock does not match the frozen snapshot")
    normalize_source_content_lock(snapshot)
    if snapshot.get("coordinate_frame") != "ct_l0_xyz":
        raise ValueError("candidate preflight supports only ct_l0_xyz")
    if float(request.get("m7_threshold", snapshot.get("m7_threshold", -1))) != float(
            snapshot.get("m7_threshold", -2)):
        raise ValueError("m7 threshold differs from the frozen source snapshot")

    generation = {
        "catalog_snapshot_sha256": request["catalog_snapshot_sha256"],
        "grid_step": int(request["grid_step"]),
        "query_radius": int(request["query_radius"]),
        "clearance": float(request["cell_clearance"]),
        "volume_edge_margin": int(request["volume_clearance"]),
        "candidate_interior_clearance": int(request["candidate_interior_clearance"]),
        "selection_strategy": str(request["selection_strategy"]),
        "grid_version": str(request["grid_version"]),
        "policy_version": str(request["policy_version"]),
        "parameter_envelope": {"maximum_candidate_count": int(request["packet_candidate_limit"])},
        "candidate_selection_policy": str(request["candidate_selection_policy"]),
        "seed_region_policy": str(request["seed_region_policy"]),
        "ct_material_support_gate": dict(request["ct_material_support_gate"]),
    }
    shape = [int(value) for value in snapshot["shape_xyz"]]
    design = bounded_preflight_grid_design(
        shape, query_radius=generation["query_radius"],
        grid_step=generation["grid_step"],
        volume_edge_margin=generation["volume_edge_margin"],
        hard_cell_limit=maximum_cells,
    )
    population: dict[str, int] = {}
    eligible_bins: dict[tuple[int, int, int], int] = {}
    tasks = generate_tasks_for_snapshot(
        surface_view, snapshot, max_tasks=maximum_cells,
        population_count_out=population, population_bins_out=eligible_bins,
        bounded_preflight_indices=design["indices"], **generation)
    total_grid = int(design["total_grid_cells"])
    sampled_grid = len(design["indices"])
    sampled_eligible = int(population["geometrically_eligible_cells"])
    census = design["measurement_kind"] == "CENSUS"
    eligible_estimate = (sampled_eligible if census else
                         round(total_grid * sampled_eligible / sampled_grid)
                         if sampled_grid else 0)
    for task in tasks:
        task["source"] = snapshot
        task["task_id"] = "preflight-" + content_sha256({
            "snapshot": snapshot["source_snapshot_id"], "grid": task["grid_version"],
            "policy": task["policy_version"], "cell": task["cell_id"],
        })[:20]
        task["attempt_id"] = "read-only-preflight"

    active_provider = provider or McpSeedProvider()
    active_sampler = ct_sampler or OmeZarrCtSupportSampler()
    rows: list[dict[str, Any] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(survey_candidate_task, task, active_provider,
                               ct_sampler=active_sampler): index
                   for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    surveys = [row for row in rows if row is not None]
    successful_surveys = [row for row in surveys if not row.get("source_error")]
    aggregate = aggregate_candidate_survival(surveys)

    no_causes: dict[str, int] = {}
    for row in surveys:
        counts = row["counts"]
        cause = None
        if row.get("source_error"):
            cause = "SOURCE_ERROR"
        elif counts["raw_m7"] == 0:
            cause = "NO_M7_CANDIDATES"
        elif counts["post_ct"] == 0:
            cause = "NO_CT_SUPPORTED_CANDIDATES"
        elif counts["post_cell_clearance"] == 0:
            cause = "NO_CELL_CLEARANCE_ELIGIBLE_CANDIDATES"
        elif counts["post_volume_clearance"] == 0:
            cause = "NO_CLEARANCE_ELIGIBLE_CANDIDATES"
        elif counts["packet_retained"] == 0:
            cause = "NO_PACKET_RETAINED_CANDIDATES"
        if cause:
            no_causes[cause] = no_causes.get(cause, 0) + 1

    survey_by_cell = {row["cell_id"]: row for row in surveys}
    bins: dict[tuple[int, int, int], dict[str, Any]] = {}
    for key, total in design["grid_bin_counts"].items():
        bins[key] = {"bin_xyz": list(key), "total_cells": int(total),
                                    "surveyed_cells": 0, "candidate_bearing_cells": 0,
                                    "usable_candidate_cells": 0,
                                    "sampled_eligible_cells": int(eligible_bins.get(key, 0))}
    for task in tasks:
        key = _bin_for(task["center_xyz"], shape)
        row = bins[key]
        survey = survey_by_cell.get(task["cell_id"])
        if survey and not survey.get("source_error"):
            row["surveyed_cells"] += 1
            row["candidate_bearing_cells"] += int(survey["counts"]["raw_m7"] > 0)
            row["usable_candidate_cells"] += int(survey["counts"]["packet_retained"] > 0)

    retained = list(aggregate["candidates"]["packet_retained"].values())
    scores = [float(row.get("score", 0.0)) for row in retained]
    cell_clearances = [float(row["cell_interior_clearance_voxels"]) for row in retained]
    volume_clearances = [float(row["volume_interior_clearance_voxels"]) for row in retained]
    design_measurement = design["measurement_kind"]
    measurement = (f"INCOMPLETE_{design_measurement}"
                   if len(successful_surveys) != len(surveys) else design_measurement)
    normalized_request = _without_volatile_evidence(copy.deepcopy(request))
    normalized_request.setdefault("sample_id", snapshot.get("sample_id"))
    normalized_request.setdefault("m7_threshold", snapshot.get("m7_threshold"))
    private: dict[str, Any] = {
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "status": "COMPLETE_WITH_SOURCE_ERRORS" if any(row.get("source_error") for row in surveys) else "COMPLETE",
        "measurement_kind": measurement,
        "sampling_design": {
            "name": ("complete-eligible-grid-census-v1" if design_measurement == "CENSUS"
                     else "deterministic-golden-coprime-rank1-grid-sample-v1"),
            "measurement_kind": design_measurement,
            "population_grid_cells": total_grid,
            "sampled_grid_cells": sampled_grid,
            "inclusion_fraction": (sampled_grid / total_grid if total_grid else 1.0),
            "ordinal_rule": design["ordinal_rule"],
            "ordinal_stride": design["ordinal_stride"],
            "ordinal_offset": design["ordinal_offset"],
            "sampled_index_sha256": content_sha256(design["indices"]),
            "cell_order_sha256": content_sha256([task["cell_id"] for task in tasks]),
        },
        "planned_sampling_percentage": (100.0 * sampled_grid / total_grid if total_grid else 100.0),
        "achieved_successful_sampling_percentage": (
            100.0 * len(successful_surveys) / total_grid if total_grid else 100.0),
        "sample_id": snapshot["sample_id"],
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "bindings": {
            **({"mission_id": request["mission_id"]} if request.get("mission_id") else {}),
            "sample_id": snapshot["sample_id"],
            "p0_artifact_id": request["p0_artifact_id"],
            "p0_artifact_sha256": request["p0_artifact_sha256"],
            "p0_selection_version": request["p0_selection_version"],
            "p0_selection_sha256": request["p0_selection_sha256"],
            "catalog_snapshot_sha256": request["catalog_snapshot_sha256"],
            "source_snapshot_id": snapshot["source_snapshot_id"],
            "source_content_lock_sha256": expected_lock_sha,
            "ct_sha256": snapshot.get("ct_sha256"), "m7_sha256": snapshot.get("m7_sha256"),
            "m7_uri_sha256": hashlib.sha256(
                str(snapshot.get("m7_uri") or "").encode("utf-8")
            ).hexdigest(),
            "coordinate_frame": snapshot.get("coordinate_frame"),
            "voxel_size_um": snapshot.get("voxel_size_um"),
            "m7_threshold": snapshot.get("m7_threshold"),
            "grid_version": request["grid_version"], "policy_version": request["policy_version"],
            "provider": request["provider"],
            "candidate_selection_policy": request["candidate_selection_policy"],
            "seed_region_policy": request["seed_region_policy"],
            "selection_strategy": request["selection_strategy"],
            "maximum_cells": maximum_cells,
            "grid_step": generation["grid_step"],
            "query_radius": generation["query_radius"],
            "parallelism": parallelism,
            "shape_xyz": shape,
            "source_snapshot_sha256": content_sha256(snapshot),
            "normalized_request_sha256": content_sha256(normalized_request),
            "code_revision": code_revision,
        },
        "gates": {
            "cell_clearance": request["cell_clearance"],
            "volume_clearance": request["volume_clearance"],
            "candidate_interior_clearance": request["candidate_interior_clearance"],
            "ct_material_support_gate": request["ct_material_support_gate"],
            "packet_candidate_limit": request["packet_candidate_limit"],
        },
        "funnel": {
            "total_grid_cells": total_grid,
            "grid_cells_in_design_sample": sampled_grid,
            "geometrically_eligible_cells": sampled_eligible if census else None,
            "geometrically_eligible_cells_estimate": eligible_estimate,
            "geometrically_eligible_sampled_cells": sampled_eligible,
            "cells_attempted": len(surveys),
            "cells_surveyed": len(successful_surveys),
            "cells_surveyed_successfully": len(successful_surveys),
            "cells_failed_source": len(surveys) - len(successful_surveys),
            "cells_with_raw_m7_candidates": sum(
                row["counts"]["raw_m7"] > 0 for row in successful_surveys),
            "raw_m7_candidates": aggregate["unique_raw"],
            "raw_m7_candidate_observations": sum(row["counts"]["raw_m7"] for row in surveys),
            "post_ct_candidates": aggregate["unique_post_ct"],
            "post_cell_clearance_candidates": aggregate["unique_post_cell_clearance"],
            "post_volume_clearance_candidates": aggregate["unique_post_volume_clearance"],
            "packet_retained_candidates": aggregate["unique_packet_retained"],
            "duplicate_candidate_observations": aggregate["duplicate_observations"],
            "source_errors": sum(bool(row.get("source_error")) for row in surveys),
        },
        "no_candidate_causes": dict(sorted(no_causes.items())),
        "normalized_request": normalized_request,
        "score_statistics": _percentiles(scores),
        "cell_clearance_statistics": _percentiles(cell_clearances),
        "volume_clearance_statistics": _percentiles(volume_clearances),
        "spatial_bins": [bins[key] for key in sorted(bins)],
        "candidate_coordinates_xyz": [dict(zip("xyz", key, strict=True))
                                      for key in sorted(aggregate["candidates"]["packet_retained"])],
        "cells": [{"cell_id": task["cell_id"], "center_xyz": task["center_xyz"],
                   "survey": survey_by_cell.get(task["cell_id"])} for task in tasks],
        "provider_response_hashes": [row.get("provider_response_sha256") for row in surveys],
        "m7_read_sets": [row.get("m7_read_set") for row in surveys if row.get("m7_read_set")],
        "ct_read_sets": [assessment.get("sample", {}).get("source_read_set")
                         for row in surveys for assessment in
                         (row.get("ct_material_support") or {}).get("assessments", [])
                         if assessment.get("sample", {}).get("source_read_set")],
        "state_mutation": "NONE", "growth_allowed": False, "ink_used": False,
        "non_claim": "Candidate scarcity is not evidence of surface, ink, text, or letter absence.",
        "generated_at_utc": utc_now(),
    }
    private["scientific_core"] = {
        "schema": "campaignx.segment_candidate_coverage_preflight.scientific_core.v1",
        **{key: private[key] for key in (
            "status", "measurement_kind", "sampling_design",
            "planned_sampling_percentage", "achieved_successful_sampling_percentage",
            "sample_id", "source_snapshot_id",
            "bindings", "gates", "funnel", "no_candidate_causes",
            "score_statistics", "cell_clearance_statistics",
            "volume_clearance_statistics", "spatial_bins",
            "candidate_coordinates_xyz", "state_mutation", "growth_allowed",
            "ink_used", "non_claim",
        )},
        "normalized_request": private["normalized_request"],
        "cell_outcomes": [{
            "cell_id": row["cell_id"], "counts": row["counts"],
            "rejection_counts": row.get("rejection_counts") or {},
            "provider_response_sha256": row.get("provider_response_sha256"),
            "source_error": row.get("source_error"),
        } for row in surveys],
        "m7_read_set_manifest_sha256": sorted({
            str(row.get("canonical_manifest_sha256")) for row in private["m7_read_sets"]
        }),
        "ct_read_set_manifest_sha256": sorted({
            str(row.get("canonical_manifest_sha256")) for row in private["ct_read_sets"]
        }),
    }
    private["receipt_sha256"] = content_sha256(private["scientific_core"])
    return {"private_receipt": private,
            "sanitized_receipt": sanitize_candidate_coverage_receipt(private)}


def _valid_read_set(value: object) -> bool:
    if not isinstance(value, dict) or value.get("schema") != "campaignx.first_letters_source_read_set.v1":
        return False
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects:
        return False
    keys = []
    for item in objects:
        if not isinstance(item, dict):
            return False
        key, digest, size = item.get("object_key"), item.get("sha256"), item.get("bytes")
        if not isinstance(key, str) or not key or not isinstance(digest, str) or len(digest) != 64:
            return False
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            return False
        keys.append(key)
    return (keys == sorted(keys) and len(keys) == len(set(keys))
            and value.get("canonical_manifest_sha256") == content_sha256(objects))


def _merge_read_sets(values: list[object]) -> dict[str, Any] | None:
    merged: dict[str, dict[str, Any]] = {}
    for value in values:
        if not _valid_read_set(value):
            return None
        for item in value["objects"]:
            key = item["object_key"]
            if key in merged and merged[key] != item:
                return None
            merged[key] = dict(item)
    if not merged:
        return None
    objects = [merged[key] for key in sorted(merged)]
    return {
        "schema": "campaignx.first_letters_source_read_set.v1",
        "objects": objects,
        "canonical_manifest_sha256": content_sha256(objects),
    }


def _load_locked_tifxyz(spec: dict[str, Any]) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """Fetch and byte-verify the four files defining the frozen TIFXYZ surface."""
    expected = {str(row["path"]): row for row in spec.get("artifacts") or []}
    if set(expected) != {"meta.json", "x.tif", "y.tif", "z.tif"}:
        raise ValueError("the control surface must lock meta.json and x/y/z.tif")
    base = str(spec.get("uri") or "").rstrip("/")
    payloads: dict[str, bytes] = {}
    objects = []
    for name in ("meta.json", "x.tif", "y.tif", "z.tif"):
        with urllib.request.urlopen(f"{base}/{name}", timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected[name].get("sha256"):
            raise ValueError(f"control surface byte hash drifted at {name}")
        payloads[name] = payload
        objects.append({"object_key": name, "sha256": digest, "bytes": len(payload)})
    json.loads(payloads["meta.json"])
    arrays = tuple(np.asarray(tifffile.imread(io.BytesIO(payloads[name])), dtype=np.float64)
                   for name in ("x.tif", "y.tif", "z.tif"))
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("the frozen x/y/z surface arrays disagree on shape")
    expected_shape = tuple(int(value) for value in spec.get("grid_shape_yx") or [])
    if expected_shape and arrays[0].shape != expected_shape:
        raise ValueError("the frozen surface grid shape drifted")
    valid = np.logical_and.reduce([np.isfinite(array) & (array >= 0) for array in arrays])
    expected_bbox = spec.get("bbox_ct_l0_xyz") or {}
    if expected_bbox:
        observed_min = [float(array[valid].min()) for array in arrays]
        observed_max = [float(array[valid].max()) for array in arrays]
        for observed, expected in zip(observed_min, expected_bbox.get("minimum") or [], strict=True):
            if not math.isclose(observed, float(expected), abs_tol=1e-6):
                raise ValueError("the frozen surface minimum bbox drifted")
        for observed, expected in zip(observed_max, expected_bbox.get("maximum") or [], strict=True):
            if not math.isclose(observed, float(expected), abs_tol=1e-6):
                raise ValueError("the frozen surface maximum bbox drifted")
    return arrays, {
        "schema": "campaignx.first_letters_source_read_set.v1",
        "objects": objects,
        "canonical_manifest_sha256": content_sha256(objects),
    }


def _read_frozen_root_objects(base_uri: object, rows: object, *,
                              reader=None) -> dict[str, Any] | None:
    """Fetch and byte-verify the objects the manifest froze as a source's identity.

    `.zattrs` and `0/.zarray` are not data. They are what says this is the volume
    the control was frozen to, and the panel refuses a receipt whose read sets do
    not contain them -- rightly, because a run that never touched them cannot
    show which volume it read. The measurement works at a downsampled level and
    so never opened them on its own; a real run recorded level 5 and stopped at
    that check with the measurement otherwise complete.

    Byte-verified like the locked surface above, and for the same reason: a
    recorded hash that nobody checked is a decoration, not a binding.
    """
    frozen = [row for row in (rows or []) if isinstance(row, dict) and row.get("path")]
    if not frozen:
        return None
    base = str(base_uri or "").rstrip("/")
    if not base:
        raise ValueError("frozen root objects were locked against no source URI")
    objects = []
    for row in sorted(frozen, key=lambda item: str(item["path"])):
        path = str(row["path"])
        with (reader or urllib.request.urlopen)(f"{base}/{path}", timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != str(row.get("sha256")):
            raise ValueError(f"frozen root object byte hash drifted at {path}")
        objects.append({"object_key": path, "sha256": digest, "bytes": len(payload)})
    return {
        "schema": "campaignx.first_letters_source_read_set.v1",
        "objects": objects,
        "canonical_manifest_sha256": content_sha256(objects),
    }


def _valid_surface_mask(arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.logical_and.reduce([np.isfinite(array) & (array >= 0) for array in arrays])


def _surface_regions(arrays: tuple[np.ndarray, np.ndarray, np.ndarray], *,
                     step: int = 128) -> list[dict[str, float]]:
    """Cover every valid bilinear quad with MCP-safe overlapping local probes."""
    valid = _valid_surface_mask(arrays)
    quad = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    vertices = np.zeros_like(valid)
    vertices[:-1, :-1] |= quad
    vertices[1:, :-1] |= quad
    vertices[:-1, 1:] |= quad
    vertices[1:, 1:] |= quad
    coordinates = np.column_stack([array[vertices] for array in arrays])
    if not len(coordinates):
        raise ValueError("the frozen surface has no finite bilinear quad")
    buckets = np.floor(coordinates / float(step)).astype(np.int64)
    unique = sorted({tuple(int(value) for value in row) for row in buckets})
    return [dict(zip("xyz", ((value + 0.5) * step for value in bucket), strict=True))
            for bucket in unique]


def _bilinear_distance(point: dict[str, float],
                       arrays: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """Nearest distance to valid bilinear quads, with projected Gauss-Newton."""
    p = np.asarray([point[axis] for axis in "xyz"], dtype=np.float64)
    valid = _valid_surface_mask(arrays)
    quad_mask = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    ys, xs = np.nonzero(quad_mask)
    vertices = np.column_stack([array[valid] for array in arrays])
    best = float(np.min(np.linalg.norm(vertices - p, axis=1)))
    for y, x in zip(ys, xs, strict=True):
        q00 = np.asarray([array[y, x] for array in arrays])
        q10 = np.asarray([array[y, x + 1] for array in arrays])
        q01 = np.asarray([array[y + 1, x] for array in arrays])
        q11 = np.asarray([array[y + 1, x + 1] for array in arrays])
        corners = np.stack((q00, q10, q01, q11))
        if np.any(p < corners.min(axis=0) - best) \
                or np.any(p > corners.max(axis=0) + best):
            continue
        a, b, c = q10 - q00, q01 - q00, q11 - q10 - q01 + q00
        for initial in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)):
            uv = np.asarray(initial, dtype=np.float64)
            for _ in range(12):
                u, v = uv
                surface = q00 + a * u + b * v + c * u * v
                jacobian = np.column_stack((a + c * v, b + c * u))
                try:
                    delta = np.linalg.lstsq(jacobian, p - surface, rcond=None)[0]
                except np.linalg.LinAlgError:
                    break
                updated = np.clip(uv + delta, 0.0, 1.0)
                if np.linalg.norm(updated - uv) < 1e-9:
                    uv = updated
                    break
                uv = updated
            u, v = uv
            distance = float(np.linalg.norm(p - (q00 + a * u + b * v + c * u * v)))
            best = min(best, distance)
    return best


def run_control_region_preflight(
    snapshot: dict[str, Any],
    request: dict[str, Any],
    *,
    provider=None,
    ct_sampler=None,
    surface_loader=None,
    frozen_root_objects=None,
    root_object_reader=None,
) -> dict[str, Any]:
    """Probe the locked surface through provider, CT, and clearance without growth."""
    center = {axis: float(request["region_center_xyz"][axis]) for axis in "xyz"}
    radius = {axis: float(request["region_radius_xyz"][axis]) for axis in "xyz"}
    known = {axis: float(request["known_coordinate_xyz"][axis]) for axis in "xyz"}
    surface_spec = request.get("control_surface")
    surface_arrays = None
    surface_read_set = None
    centers = [center]
    if isinstance(surface_spec, dict):
        surface_arrays, surface_read_set = (surface_loader or _load_locked_tifxyz)(surface_spec)
        centers = _surface_regions(surface_arrays)
        centers.sort(key=lambda item: (
            sum((float(item[axis]) - known[axis]) ** 2 for axis in "xyz"),
            *(float(item[axis]) for axis in "xyz"),
        ))

    def task_for(region_center):
        local_radius = {axis: 128.0 for axis in "xyz"} if surface_arrays is not None else radius
        return {
        "task_id": "control-region-preflight",
        "attempt_id": "read-only-preflight",
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "sample_id": snapshot["sample_id"],
        "source": snapshot,
        "bounds_xyz": [
            [region_center[axis] - local_radius[axis] for axis in "xyz"],
            [region_center[axis] + local_radius[axis] for axis in "xyz"],
        ],
        "center_xyz": region_center,
        "candidate_discovery": {
            "provider": "vc3d-mcp",
            "prediction_uri": snapshot["m7_uri"],
            "prediction_space": "ct_l0_xyz",
            "region": {"center": region_center, "radius": local_radius},
            "max_candidates": int(request["max_candidates"]),
            "minimum_separation_voxels": int(request["minimum_separation_voxels"]),
            "minimum_cell_interior_clearance_voxels": float(request["minimum_cell_clearance_voxels"]),
            "minimum_volume_interior_clearance_voxels": float(request["minimum_volume_clearance_voxels"]),
            "seed_region_policy": str(request["seed_region_policy"]),
            "m7_threshold": float(request["m7_threshold"]),
            "ct_material_support_gate": dict(request["ct_material_support_gate"]),
        },
        "parameter_envelope": {"maximum_candidate_count": int(request["packet_candidate_limit"])},
        "ink_used": False,
        }

    provider_instance = provider or McpSeedProvider()
    sampler = ct_sampler or OmeZarrCtSupportSampler()
    planned_region_count = len(centers)
    visit_limit = min(planned_region_count, int(request.get("maximum_surface_probe_regions", 4096)))
    tasks, raw_responses, ct_receipts, screenings, survivors = [], [], [], [], []
    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    observed_ids: dict[tuple[Any, ...], set[str]] = {}
    raw_candidate_count = 0
    tolerance = float(request["tolerance_ct_l0_voxels"])
    found_inside = False
    # One row per region, kept because a mean hides the region that cost the
    # hour among eleven that cost seconds. A run took 4184 seconds over twelve
    # regions and the parts profiled in isolation accounted for fifty; the rest
    # could only be attributed by guessing, which is what this exists to stop.
    region_timings: list[dict[str, Any]] = []
    walk_started = time.monotonic()
    for task_index, region_center in enumerate(centers[:visit_limit]):
        timing = {"region_index": task_index, "discover_seconds": 0.0,
                  "ct_gate_seconds": 0.0, "screen_seconds": 0.0}
        region_timings.append(timing)
        task = task_for(region_center)
        tasks.append(task)
        step = time.monotonic()
        response = provider_instance.discover(task)
        timing["discover_seconds"] = round(time.monotonic() - step, 3)
        raw_responses.append(response)
        new_candidates = []
        for ordinal, candidate in enumerate(response.get("candidates") or []):
            raw_candidate_count += 1
            coordinate = candidate.get("ct_l0_coordinate") or candidate.get("coordinate") or candidate
            if isinstance(coordinate, dict) and all(axis in coordinate for axis in "xyz"):
                values = tuple(float(coordinate[axis]) for axis in "xyz")
                key = values
            else:
                key = ("malformed", task_index, ordinal, content_sha256(candidate))
            candidate_id = str(candidate.get("candidate_id") or candidate.get("id") or "")
            observed_ids.setdefault(key, set()).add(candidate_id)
            if key in chosen:
                continue
            chosen[key] = dict(candidate)
            new_candidates.append(candidate)
        if not new_candidates:
            continue
        step = time.monotonic()
        gated, ct_receipt = apply_ct_material_support_gate(
            {"candidates": new_candidates, "ink_used": False}, task, sampler=sampler)
        timing["ct_gate_seconds"] = round(time.monotonic() - step, 3)
        step = time.monotonic()
        screened = screen_candidates(gated, task)
        timing["screen_seconds"] = round(time.monotonic() - step, 3)
        ct_receipts.append(ct_receipt)
        screenings.append(screened)
        survivors.extend(screened["usable_candidates"])
        if surface_arrays is not None and any(
                _bilinear_distance(candidate, surface_arrays) <= tolerance
                for candidate in screened["usable_candidates"]):
            found_inside = True
            break
    raw_response = raw_responses[0] if len(raw_responses) == 1 else {"candidates": [
        candidate for response in raw_responses for candidate in response.get("candidates") or []]}
    ct_receipt = ct_receipts[0] if len(ct_receipts) == 1 else {
        "assessments": [row for receipt in ct_receipts for row in receipt.get("assessments") or []],
        "retained_candidate_count": sum(int(receipt.get("retained_candidate_count", 0)) for receipt in ct_receipts),
    }
    screened = screenings[0] if len(screenings) == 1 else {
        "eligible_candidate_count": sum(int(row.get("eligible_candidate_count", 0)) for row in screenings),
        "usable_candidate_count": sum(int(row.get("usable_candidate_count", 0)) for row in screenings),
    }
    distances = [
        (_bilinear_distance(candidate, surface_arrays)
         if surface_arrays is not None else
         math.sqrt(sum((float(candidate[axis]) - known[axis]) ** 2 for axis in "xyz")))
        for candidate in survivors
    ]
    exchanges = [response.get("provider_exchange") or {} for response in raw_responses]
    exchange = exchanges[0] if len(exchanges) == 1 else {
        "request_sha256": content_sha256([row.get("request_sha256") for row in exchanges]),
        "request_bytes": sum(int(row.get("request_bytes", 0)) for row in exchanges),
        "response_sha256": content_sha256([row.get("response_sha256") for row in exchanges]),
        "response_bytes": sum(int(row.get("response_bytes", 0)) for row in exchanges),
    }
    # Added to what the run read, not substituted for it: the roots say which
    # volume, the rest says what was measured in it.
    locked = frozen_root_objects or {}
    ct_roots = _read_frozen_root_objects(
        snapshot.get("ct_uri"), locked.get("ct"), reader=root_object_reader)
    m7_roots = _read_frozen_root_objects(
        snapshot.get("m7_uri"), locked.get("m7"), reader=root_object_reader)
    # `_merge_read_sets` fails closed on anything that is not a read set, so a
    # caller that locked no roots must contribute nothing rather than a None.
    m7_read_set = _merge_read_sets(
        [response.get("source_read_set") for response in raw_responses]
        + ([m7_roots] if m7_roots else []))
    ct_read_sets = [
        assessment.get("sample", {}).get("source_read_set")
        for assessment in ct_receipt.get("assessments", [])
        if assessment.get("sample")
    ]
    ct_read_set = _merge_read_sets(ct_read_sets + ([ct_roots] if ct_roots else []))
    exchange_complete = (
        isinstance(exchange, dict)
        and isinstance(exchange.get("request_bytes"), int)
        and isinstance(exchange.get("response_bytes"), int)
        and all(isinstance(exchange.get(field), str) and len(exchange[field]) == 64
                for field in ("request_sha256", "response_sha256"))
    )
    source_binding_complete = (
        exchange_complete and _valid_read_set(m7_read_set)
        and _valid_read_set(ct_read_set)
        and (surface_arrays is None or _valid_read_set(surface_read_set))
    )
    visited_region_count = len(raw_responses)
    coverage_complete = found_inside or visited_region_count == planned_region_count
    status = ("INCOMPLETE_REGION_COVERAGE" if source_binding_complete and not coverage_complete
              else "COMPLETE" if source_binding_complete else "INCOMPLETE_SOURCE_BINDING")
    return {
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "scope": "CONTROL_REGION",
        "status": status,
        "sample_id": snapshot["sample_id"],
        "source_snapshot_id": snapshot["source_snapshot_id"],
        "state_mutation": "NONE",
        "growth_allowed": False,
        "ink_used": False,
        "parameters": request,
        "counts": {
            "raw_m7": len(chosen),
            "post_ct": int(ct_receipt.get("retained_candidate_count", 0)),
            "post_clearance": int(screened["eligible_candidate_count"]),
            "packet_limited": int(screened["usable_candidate_count"]),
            "duplicates_coalesced": raw_candidate_count - len(chosen),
        },
        "duplicate_candidate_count": raw_candidate_count - len(chosen),
        "candidate_identity_aliases": [
            {"coordinate_xyz": list(key), "observed_candidate_ids": sorted(identifier for identifier in identifiers if identifier)}
            for key, identifiers in sorted(observed_ids.items(), key=lambda item: tuple(str(value) for value in item[0]))
            if len({identifier for identifier in identifiers if identifier}) > 1
            and all(isinstance(value, float) for value in key)
        ],
        "closest_survivor_distance_ct_l0_voxels": min(distances) if distances else None,
        "within_tolerance": bool(distances) and min(distances) <= float(request["tolerance_ct_l0_voxels"]),
        "ct_read_set": ct_read_set,
        "m7_read_set": m7_read_set,
        "provider_exchange": exchange,
        "surface_read_set": surface_read_set,
        "planned_region_count": planned_region_count,
        "region_timings": region_timings,
        "elapsed_seconds": round(time.monotonic() - walk_started, 3),
        "visited_region_count": visited_region_count,
        "coverage_fraction": (visited_region_count / planned_region_count
                              if planned_region_count else 0.0),
        "coverage_complete": coverage_complete,
        "provider_response_sha256": content_sha256(raw_response),
        "ct_material_support": ct_receipt,
        "clearance_screen": screened,
        "generated_at_utc": utc_now(),
        "non_claim": "Candidate availability in this bounded ink-blind region makes no ink, text, letter, or absence claim.",
    }
