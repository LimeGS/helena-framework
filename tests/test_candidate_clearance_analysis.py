"""Independent, read-only reconstruction of the PHerc0358 clearance result."""

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
from itertools import product
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/harness/analyze_candidate_clearance.py"
SPEC = importlib.util.spec_from_file_location("candidate_clearance_analysis", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

CAUSE_CELL = "INSUFFICIENT_CELL_INTERIOR_CLEARANCE"
CAUSE_VOLUME = "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE"


def _sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def _input(*, coordinates=None, reported_causes=None):
    coordinates = coordinates or [
        {"x": value, "y": 50, "z": 50} for value in range(12, 20)
    ]
    candidates = [
        {"candidate_id": f"m7-{index}", "ct_l0_coordinate": coordinate,
         "score": 0.9 - index / 100}
        for index, coordinate in enumerate(coordinates, start=1)
    ]
    if reported_causes is None:
        reported_causes = {
            candidate["candidate_id"]: ["INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE"]
            for candidate in candidates
        }
    source_read_objects = [{"object_key": "m7/chunk-1", "sha256": "a" * 64,
                            "bytes": 128}]
    raw_response = {
        "candidates": candidates,
        "request": {
            "prediction_uri": "s3://fixture/m7",
            "prediction_space": "ct_l0_xyz",
            "region": {"center": {"x": 10, "y": 50, "z": 50},
                       "radius": {"x": 10, "y": 10, "z": 10}},
            "max_candidates": 8,
            "minimum_separation_voxels": 16,
        },
        "effective_candidate_region": {
            "center": {"x": 10, "y": 50, "z": 50},
            "radius": {"x": 10, "y": 10, "z": 10},
        },
        "source_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": source_read_objects,
            "canonical_manifest_sha256": _sha(source_read_objects),
        },
        "initial_probe": None,
        "provider_exchange": {
            "encoding": "canonical-json-utf8", "call_count": 1,
            "request_sha256": "e" * 64, "request_bytes": 256,
            "response_sha256": "f" * 64, "response_bytes": 512,
        },
        "ink_used": False,
    }
    ct_assessments = [
        {
            "candidate_id": candidate["candidate_id"],
            "coordinate_xyz_l0": dict(candidate["ct_l0_coordinate"]),
            "status": "CT_MATERIAL_NEARBY",
            "sample": {
                "level": 1,
                "scale_zyx": [2.0, 2.0, 2.0],
                "center_zyx": [
                    candidate["ct_l0_coordinate"]["z"] // 2,
                    candidate["ct_l0_coordinate"]["y"] // 2,
                    candidate["ct_l0_coordinate"]["x"] // 2,
                ],
                "radius_zyx": [4, 4, 4],
                "shape_zyx": [9, 9, 9],
                "voxel_count": 729,
                "nonzero_voxel_count": 2,
                "nonzero_fraction": 2 / 729,
                "mean": 0.1,
                "standard_deviation": 0.2,
                "maximum": 1.0,
                "source_read_set": {
                    "schema": "campaignx.first_letters_source_read_set.v1",
                    "objects": [{
                        "object_key": f"ct/{candidate['candidate_id']}",
                        "sha256": "b" * 64,
                        "bytes": 64,
                    }],
                },
            },
        }
        for candidate in candidates
    ]
    for assessment in ct_assessments:
        objects = assessment["sample"]["source_read_set"]["objects"]
        assessment["sample"]["source_read_set"]["canonical_manifest_sha256"] = _sha(objects)
    ct_screen = {
        "schema": "campaignx.ct_material_support_screen.v1",
        "status": "COMPLETED_INK_BLIND",
        "generated_at_utc": "2026-08-02T00:00:00Z",
        "policy": "ome-zarr-nearby-material-v1",
        "config": {"level": 1, "radius_l0_voxels": 8,
                   "minimum_nonzero_voxels": 1},
        "source_snapshot_id": "snapshot-pherc0358-1",
        "input_candidate_count": 8,
        "retained_candidate_count": 8,
        "rejected_candidate_count": 0,
        "assessments": ct_assessments,
        "filtered_response_sha256": _sha(raw_response),
        "ink_used": False,
        "non_claim": "Nearby CT material is not ink evidence.",
    }
    policy = {
        "minimum_cell_interior_clearance_voxels": 0,
        "minimum_cell_interior_clearance_um": 0.0,
        "minimum_volume_interior_clearance_voxels": 20,
        "minimum_volume_interior_clearance_um": 200.0,
        "voxel_size_um": 10.0,
    }
    diagnosis_core = {
        "schema": "campaignx.no_seed_causal_diagnosis.v1",
        "status": "NO_SEED",
        "task_id": "task-pherc0358-1",
        "attempt_id": "attempt-pherc0358-1",
        "m7_raw_candidate_count": 8,
        "ct_support_input_candidate_count": 8,
        "ct_support_retained_candidate_count": 8,
        "ct_support_rejected_candidate_count": 0,
        "post_ct_candidate_count": 8,
        "eligible_after_clearance_count": 0,
        "cause_counts": {
            "NO_M7_CANDIDATES": 0,
            "CT_MATERIAL_SUPPORT_REJECTED": 0,
            "MALFORMED_COORDINATE_OR_SCORE": 0,
            "INSUFFICIENT_CELL_INTERIOR_CLEARANCE": sum(
                CAUSE_CELL in causes for causes in reported_causes.values()),
            "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE": sum(
                CAUSE_VOLUME in causes for causes in reported_causes.values()),
        },
        "primary_causes": sorted({cause for causes in reported_causes.values()
                                  for cause in causes}),
        "clearance_policy": policy,
        "clearance_rejection_examples_first_32": [
            {"candidate_id": candidate_id, "causes": causes}
            for candidate_id, causes in reported_causes.items()
        ],
        "ink_used": False,
        "non_claim": "NO_SEED does not establish absence of a surface.",
    }
    diagnosis = {**diagnosis_core, "diagnosis_sha256": _sha(diagnosis_core)}
    diagnosis["generated_at_utc"] = "2026-08-02T00:00:00Z"
    terminal = {
        "status": "NO_SEED",
        "raw_candidate_count": 8,
        "post_ct_candidate_count": 8,
        "usable_candidate_count": 0,
        "no_seed_cause_counts": diagnosis["cause_counts"],
        "primary_causes": diagnosis["primary_causes"],
        "no_seed_causal_diagnosis": diagnosis,
        "no_seed_causal_diagnosis_sha256": diagnosis["diagnosis_sha256"],
        "clearance_policy": policy,
        "reason": "No MCP candidate met the frozen policy.",
        "generated_at_utc": "2026-08-02T00:00:00Z",
        "ink_used": False,
        "non_claim": "This terminal does not establish that no surface exists.",
    }
    p0_source = {
        "schema": "campaignx.source_snapshot.v1",
        "sample_id": "PHerc0358",
        "source_snapshot_id": "snapshot-pherc0358-1",
        "ct_uri": "s3://fixture/ct",
        "ct_sha256": "c" * 64,
        "m7_uri": "s3://fixture/m7",
        "m7_sha256": "d" * 64,
        "shape_xyz": [100, 100, 100],
        "coordinate_frame": "ct_l0_xyz",
        "voxel_size_um": 10.0,
        "m7_threshold": None,
        "eligible_manifest_sha256": "6" * 64,
        "source_status": "HASH_LOCKED_NOT_IMMUTABLE",
    }
    alignment_authority_core = {
        "schema": "campaignx.m7_ct_alignment_authority.v1",
        "source_snapshot_id": "snapshot-pherc0358-1",
        "m7_level": 0,
        "m7_to_ct_l0_transform": [[1, 0, 0, 0], [0, 1, 0, 0],
                                  [0, 0, 1, 0], [0, 0, 0, 1]],
        "coordinate_frame": "ct_l0_xyz",
    }
    alignment_authority = {
        **alignment_authority_core,
        "authority_sha256": _sha(alignment_authority_core),
    }
    task = {
        "schema": "campaignx.segmentation_task.v1",
        "task_id": "task-pherc0358-1",
        "attempt_id": "attempt-pherc0358-1",
        "sample_id": "PHerc0358",
        "source_snapshot_id": "snapshot-pherc0358-1",
        "cell_id": "r00000c00000a00000",
        "grid_version": "first-letters-grid@1.0.0",
        "policy_version": "first-letters@1.0.0",
        "bounds_xyz": [[0, 40, 40], [20, 60, 60]],
        "center_xyz": {"x": 10, "y": 50, "z": 50},
        "priority": 1.0,
        "distance_to_known_aabb_voxels": 100.0,
        "guaranteed_cell_clearance_voxels": 0.0,
        "parameter_envelope": {"maximum_candidate_count": 8},
        "catalog_snapshot_sha256": "7" * 64,
        "candidate_discovery": {
            "provider": "vc3d-mcp",
            "prediction_uri": p0_source["m7_uri"],
            "prediction_space": "ct_l0_xyz",
            "region": raw_response["effective_candidate_region"],
            "max_candidates": 8,
            "minimum_separation_voxels": 16,
            "minimum_cell_interior_clearance_voxels": 0,
            "minimum_volume_interior_clearance_voxels": 20,
            "seed_region_policy": "fixed-v1",
            "recenter_probe_max_candidates": 100,
            "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
            "ct_material_support_gate": ct_screen["config"] | {"policy": ct_screen["policy"]},
        },
        "candidate_selection_policy": "score-cell-volume-clearance-v1",
        "planner_contract_version": "v1",
        "resource_requirements": {"gpu_required": False, "minimum_vram_gb": 0.0,
                                  "seed_probe_required": False},
        "mission_id": "first-letters",
        "ink_used": False,
        "source": p0_source,
    }
    retained = []
    for candidate, assessment in zip(candidates, ct_assessments, strict=True):
        retained.append({**candidate, "ct_material_support": assessment})
    raw_response["filtered_response_sha256"] = _sha({**raw_response, "candidates": retained})
    ct_screen["filtered_response_sha256"] = raw_response.pop("filtered_response_sha256")
    documents = {
        "TERMINAL_RECEIPT.json": terminal,
        "SEED_CANDIDATES.json": raw_response,
        "CT_MATERIAL_SUPPORT_SCREEN.json": ct_screen,
        "CLAIMED_TASK.json": task,
        "P0_SOURCE.json": p0_source,
        "M7_CT_ALIGNMENT.json": alignment_authority,
    }
    manifest = {
        "schema": "campaignx.candidate_clearance_review_input_manifest.v1",
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "source_snapshot_id": task["source_snapshot_id"],
        "files": {name: {"sha256": _sha(document)}
                  for name, document in documents.items()},
    }
    return {
        "schema": "campaignx.candidate_clearance_review_input.v1",
        "attempt_receipt": terminal,
        "p0_source": p0_source,
        "alignment_authority": alignment_authority,
        "raw_candidate_response": raw_response,
        "ct_screen": ct_screen,
        "task": task,
        "evidence_manifest": manifest,
    }


def _analyze(value, *, trusted_manifest_sha256=None):
    return analysis.analyze_candidate_clearance(
        value,
        trusted_manifest_sha256=(trusted_manifest_sha256
                                 or _sha(value["evidence_manifest"])),
    )


def _rebind(value):
    authority = value["alignment_authority"]
    authority_core = {key: row for key, row in authority.items()
                      if key != "authority_sha256"}
    authority["authority_sha256"] = _sha(authority_core)
    diagnosis = value["attempt_receipt"]["no_seed_causal_diagnosis"]
    diagnosis_core = {key: row for key, row in diagnosis.items()
                      if key not in {"generated_at_utc", "diagnosis_sha256"}}
    diagnosis["diagnosis_sha256"] = _sha(diagnosis_core)
    value["attempt_receipt"]["no_seed_causal_diagnosis_sha256"] = diagnosis[
        "diagnosis_sha256"]
    value["task"]["source"] = json.loads(json.dumps(value["p0_source"]))
    assessments = {row["candidate_id"]: row for row in value["ct_screen"]["assessments"]}
    retained = [
        {**candidate, "ct_material_support": assessments[candidate["candidate_id"]]}
        for candidate in value["raw_candidate_response"]["candidates"]
        if assessments[candidate["candidate_id"]]["status"] == "CT_MATERIAL_NEARBY"
    ]
    value["ct_screen"]["filtered_response_sha256"] = _sha({
        **value["raw_candidate_response"], "candidates": retained,
    })
    documents = {
        "TERMINAL_RECEIPT.json": value["attempt_receipt"],
        "SEED_CANDIDATES.json": value["raw_candidate_response"],
        "CT_MATERIAL_SUPPORT_SCREEN.json": value["ct_screen"],
        "CLAIMED_TASK.json": value["task"],
        "P0_SOURCE.json": value["p0_source"],
        "M7_CT_ALIGNMENT.json": value["alignment_authority"],
    }
    value["evidence_manifest"]["files"] = {
        name: {"sha256": _sha(document)} for name, document in documents.items()
    }
    return value


def _rebind_campaign_budget(value):
    budget = value["task"]["campaign_budget"]
    common = {key: row for key, row in budget.items()
              if key not in {"admission_sha256", "selection_rank"}}
    budget["admission_sha256"] = _sha(common)
    return _rebind(value)


def _campaign_input():
    value = _input(coordinates=[
        {"x": x, "y": 40, "z": 40} for x in range(12, 20)
    ])
    source = value["p0_source"]
    source["m7_threshold"] = 0.2
    source["source_status"] = "IMMUTABLE_CONTENT_LOCKED"
    source["source_content_lock"] = {
        "schema": "campaignx.source_content_lock.v1",
        "source_snapshot_id": source["source_snapshot_id"],
    }
    task = value["task"]
    task.update({
        "cell_id": "r00000c00001a00001",
        "center_xyz": {"x": 20, "y": 40, "z": 40},
        "bounds_xyz": [[10, 30, 30], [30, 50, 50]],
        "planner": "cost-aware-v2",
        "planner_model": None,
        "p0_artifact_id": "p0-pherc0358-1",
        "p0_artifact_sha256": "8" * 64,
        "p0_selection_version": "first-letters-p0-selection@1.0.0",
        "p0_selection_sha256": "9" * 64,
    })
    nominal = {
        "center": {"x": 20, "y": 40, "z": 40},
        "radius": {"x": 10, "y": 10, "z": 10},
    }
    response = value["raw_candidate_response"]
    response["request"]["region"] = copy.deepcopy(nominal)
    response["request"]["threshold"] = 0.2
    response["effective_candidate_region"] = copy.deepcopy(nominal)
    discovery = task["candidate_discovery"]
    discovery["region"] = copy.deepcopy(nominal)
    discovery["m7_threshold"] = 0.2
    gates = {
        "cell_clearance": 0.0,
        "volume_clearance": 20,
        "candidate_interior_clearance": 0,
        "packet_candidate_limit": 8,
        "ct_material_support_gate": copy.deepcopy(
            discovery["ct_material_support_gate"]),
    }
    queue = {
        "parameter_envelope": copy.deepcopy(task["parameter_envelope"]),
        "planner": task["planner"],
        "planner_model": task["planner_model"],
        "prediction_space": discovery["prediction_space"],
        "minimum_separation_voxels": discovery[
            "minimum_separation_voxels"],
        "recenter_probe_max_candidates": discovery[
            "recenter_probe_max_candidates"],
        "recenter_radius_xyz": copy.deepcopy(discovery[
            "recenter_radius_xyz"]),
        "seed_probe_mode": "off",
        "seed_probe_top_k": 2,
        "seed_probe_generations": 12,
        "verify_sources": True,
    }
    execution = {
        "sample_id": task["sample_id"],
        "mission_id": task["mission_id"],
        "source_snapshot_id": source["source_snapshot_id"],
        "p0_artifact_id": task["p0_artifact_id"],
        "p0_artifact_sha256": task["p0_artifact_sha256"],
        "p0_selection_version": task["p0_selection_version"],
        "p0_selection_sha256": task["p0_selection_sha256"],
        "catalog_snapshot_sha256": task["catalog_snapshot_sha256"],
        "source_content_lock_sha256": _sha(source["source_content_lock"]),
        "ct_sha256": source["ct_sha256"],
        "m7_sha256": source["m7_sha256"],
        "m7_uri_sha256": hashlib.sha256(
            source["m7_uri"].encode("utf-8")).hexdigest(),
        "coordinate_frame": source["coordinate_frame"],
        "voxel_size_um": source["voxel_size_um"],
        "shape_xyz": copy.deepcopy(source["shape_xyz"]),
        "grid_version": task["grid_version"],
        "policy_version": task["policy_version"],
        "provider": discovery["provider"],
        "m7_threshold": source["m7_threshold"],
        "grid_step": 20,
        "query_radius": 10,
        "parallelism": 1,
        "maximum_cells": 64,
        "selection_strategy": "stratified-clearance-v1",
        "candidate_selection_policy": task["candidate_selection_policy"],
        "seed_region_policy": discovery["seed_region_policy"],
        "code_revision": "a" * 40,
        "gates": gates,
        "queue_execution": queue,
    }
    prefix = [task["cell_id"]]
    common = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": task["mission_id"],
        "sample_id": task["sample_id"],
        "receipt_sha256": "1" * 64,
        "preflight_receipt_sha256": "2" * 64,
        "preflight_sanitized_receipt_sha256": "3" * 64,
        "approved_task_count": 1,
        "order_seed_sha256": "4" * 64,
        "population_order_sha256": "5" * 64,
        "prefix_sha256": _sha(prefix),
        "prefix_cell_ids": prefix,
        "execution_bindings": execution,
    }
    task["campaign_budget"] = {
        **common,
        "admission_sha256": _sha(common),
        "selection_rank": 0,
    }
    return _rebind(value)


def _set_clearance_policy(value, *, cell, volume):
    discovery = value["task"]["candidate_discovery"]
    discovery["minimum_cell_interior_clearance_voxels"] = cell
    discovery["minimum_volume_interior_clearance_voxels"] = volume
    terminal_policy = value["attempt_receipt"]["clearance_policy"]
    terminal_policy["minimum_cell_interior_clearance_voxels"] = cell
    terminal_policy["minimum_cell_interior_clearance_um"] = cell * 10.0
    terminal_policy["minimum_volume_interior_clearance_voxels"] = volume
    terminal_policy["minimum_volume_interior_clearance_um"] = volume * 10.0
    diagnosis = value["attempt_receipt"]["no_seed_causal_diagnosis"]
    diagnosis["clearance_policy"] = copy.deepcopy(terminal_policy)


def _recenter_z_input(*, empty=False):
    coordinates = [] if empty else [
        {"x": x, "y": 50, "z": 40} for x in range(12, 20)
    ]
    value = _input(coordinates=coordinates or None)
    response = value["raw_candidate_response"]
    discovery = value["task"]["candidate_discovery"]
    discovery["seed_region_policy"] = "m7-recenter-z-v1"
    discovery["recenter_radius_xyz"] = {"x": 10, "y": 10, "z": 10}
    probe_request = copy.deepcopy(response["request"])
    probe_request["max_candidates"] = discovery["recenter_probe_max_candidates"]
    if empty:
        response["candidates"] = []
        value["ct_screen"]["assessments"] = []
        for field in ("input_candidate_count", "retained_candidate_count",
                      "rejected_candidate_count"):
            value["ct_screen"][field] = 0
        attempt = value["attempt_receipt"]
        attempt["raw_candidate_count"] = 0
        attempt["post_ct_candidate_count"] = 0
        attempt["usable_candidate_count"] = 0
        diagnosis = attempt["no_seed_causal_diagnosis"]
        diagnosis["m7_raw_candidate_count"] = 0
        diagnosis["ct_support_input_candidate_count"] = 0
        diagnosis["ct_support_retained_candidate_count"] = 0
        diagnosis["ct_support_rejected_candidate_count"] = 0
        diagnosis["post_ct_candidate_count"] = 0
        diagnosis["eligible_after_clearance_count"] = 0
        diagnosis["cause_counts"] = {
            key: 0 for key in diagnosis["cause_counts"]
        }
        diagnosis["primary_causes"] = []
        diagnosis["clearance_rejection_examples_first_32"] = []
        attempt["no_seed_cause_counts"] = copy.deepcopy(diagnosis["cause_counts"])
        attempt["primary_causes"] = []
        response["initial_probe"] = {
            "policy": "m7-recenter-z-v1",
            "request": probe_request,
            "candidate_count": 0,
            "recentered": False,
        }
        response["provider_exchange"]["call_count"] = 1
    else:
        effective = {
            "center": {"x": 10, "y": 50, "z": 40},
            "radius": {"x": 10, "y": 10, "z": 10},
        }
        response["effective_candidate_region"] = effective
        response["initial_probe"] = {
            "policy": "m7-recenter-z-v1",
            "request": probe_request,
            "candidate_count": 8,
            "median_z_ct_l0": 40,
            "median_coordinate_ct_l0": None,
            "recentered": True,
        }
        response["provider_exchange"]["call_count"] = 2
    return _rebind(value)


def _recenter_xyz_input():
    causes = {f"m7-{index}": [CAUSE_CELL] for index in range(1, 9)}
    value = _input(
        coordinates=[{"x": x, "y": 50, "z": 50} for x in range(22, 30)],
        reported_causes=causes,
    )
    _set_clearance_policy(value, cell=10, volume=20)
    response = value["raw_candidate_response"]
    discovery = value["task"]["candidate_discovery"]
    discovery["seed_region_policy"] = "m7-recenter-xyz-v1"
    discovery["recenter_radius_xyz"] = {"x": 10, "y": 10, "z": 10}
    effective = {
        "center": {"x": 30, "y": 50, "z": 50},
        "radius": {"x": 10, "y": 10, "z": 10},
    }
    probe_request = copy.deepcopy(response["request"])
    probe_request["max_candidates"] = discovery["recenter_probe_max_candidates"]
    response["effective_candidate_region"] = copy.deepcopy(effective)
    response["initial_probe"] = {
        "policy": "m7-recenter-xyz-v1",
        "request": probe_request,
        "candidate_count": 8,
        "median_z_ct_l0": 50,
        "median_coordinate_ct_l0": copy.deepcopy(effective["center"]),
        "recentered": True,
    }
    response["provider_exchange"]["call_count"] = 2
    return _rebind(value)


def _chunk_input(policy="m7-recenter-z-chunk-safe-v1"):
    causes = {f"m7-{index}": [CAUSE_CELL] for index in range(1, 9)}
    recentered = policy == "m7-recenter-z-chunk-safe-v1"
    z = 200 if recentered else 192
    value = _input(
        coordinates=[{"x": x, "y": 192, "z": z} for x in range(136, 144)],
        reported_causes=causes,
    )
    source = value["p0_source"]
    source["shape_xyz"] = [384, 384, 384]
    task = value["task"]
    task["center_xyz"] = {"x": 192, "y": 192, "z": 192}
    task["bounds_xyz"] = [[128, 128, 128], [256, 256, 256]]
    _set_clearance_policy(value, cell=65, volume=64)
    nominal = {
        "center": {"x": 192, "y": 192, "z": 192},
        "radius": {"x": 64, "y": 64, "z": 64},
    }
    response = value["raw_candidate_response"]
    response["request"]["region"] = copy.deepcopy(nominal)
    discovery = task["candidate_discovery"]
    discovery["region"] = copy.deepcopy(nominal)
    discovery["seed_region_policy"] = policy
    discovery["recenter_radius_xyz"] = {"x": 64, "y": 64, "z": 64}
    subqueries = []
    for index, offsets in enumerate(product((-64, 64), repeat=3), start=1):
        subqueries.append({
            "region": {
                "center": {
                    axis: nominal["center"][axis] + offset
                    for axis, offset in zip("xyz", offsets, strict=True)
                },
                "radius": {"x": 64, "y": 64, "z": 64},
            },
            "candidate_count": 1,
            "response_sha256": f"{index:064x}",
        })
    if recentered:
        response["effective_candidate_region"] = {
            "center": {"x": 192, "y": 192, "z": 200},
            "radius": {"x": 64, "y": 64, "z": 64},
        }
        response["initial_probe"] = {
            "policy": policy,
            "subquery_count": 8,
            "subqueries": subqueries,
            "candidate_count": 8,
            "median_z_ct_l0": 200,
            "recentered": True,
        }
        response["provider_exchange"]["call_count"] = 9
    else:
        response["effective_candidate_region"] = copy.deepcopy(nominal)
        response["initial_probe"] = {
            "policy": policy,
            "subquery_count": 8,
            "subqueries": subqueries,
            "candidate_count": 8,
            "merged_candidate_count": 8,
            "recentered": False,
            "merge_used_as_final_candidate_set": True,
        }
        response["provider_exchange"]["call_count"] = 8
    return _rebind(value)


def _reachability_input(role_state, *, http=True, checked_at_utc="2026-08-02T00:00:00Z"):
    value = _input()
    source = value["p0_source"]
    if http:
        source["ct_uri"] = "https://fixture.invalid/ct"
        source["m7_uri"] = "https://fixture.invalid/m7"
        value["raw_candidate_response"]["request"]["prediction_uri"] = source[
            "m7_uri"]
        value["task"]["candidate_discovery"]["prediction_uri"] = source["m7_uri"]
    source["source_reachable"] = {
        "checked_at_utc": checked_at_utc,
        "ct": {"uri": source["ct_uri"], **copy.deepcopy(role_state)},
        "m7": {"uri": source["m7_uri"], **copy.deepcopy(role_state)},
    }
    return _rebind(value)


def test_recomputes_exclusive_cell_and_inclusive_volume_face_distances():
    evidence = _analyze(_input())

    assert evidence["review_sha256"] == (
        "0250403299a568bf8eb46f64a916305bf7e885cc4609e6b43456f5400dbf4ff5"
    )
    first = evidence["candidates"][0]
    assert first["coordinate_xyz"] == [12, 50, 50]
    assert first["cell_face_distances_voxels"] == {
        "x_low": 12.0, "x_high": 8.0,
        "y_low": 10.0, "y_high": 10.0,
        "z_low": 10.0, "z_high": 10.0,
    }
    assert first["volume_face_distances_voxels"] == {
        "x_low": 12.0, "x_high": 87.0,
        "y_low": 50.0, "y_high": 49.0,
        "z_low": 50.0, "z_high": 49.0,
    }
    assert first["minimum_volume_face_distance_um"] == 120.0


def test_claimed_task_and_snapshot_use_producer_fields_with_separate_alignment_authority():
    value = _input()
    assert "m7_level" not in value["p0_source"]
    assert "m7_to_ct_l0_transform" not in value["p0_source"]
    assert "query_radius_voxels" not in value["task"]["candidate_discovery"]
    assert value["task"]["schema"] == "campaignx.segmentation_task.v1"

    assert _analyze(value)["classification"] == "POLICY_MARGIN_UNCALIBRATED"


def test_canonical_reachability_fields_survive_claimed_source_readback():
    value = _input()
    value["p0_source"]["source_reachable"] = {
        "checked_at_utc": "2026-08-02T00:00:00Z",
        "ct": {"uri": value["p0_source"]["ct_uri"], "checked": False,
               "reason": "not an http(s) URI"},
        "m7": {"uri": value["p0_source"]["m7_uri"], "checked": False,
                "reason": "not an http(s) URI"},
    }
    _rebind(value)

    assert _analyze(value)["classification"] == "POLICY_MARGIN_UNCALIBRATED"


def test_uri_locked_source_without_catalog_hashes_is_analyzable_but_defective():
    value = _input()
    value["p0_source"]["ct_sha256"] = None
    value["p0_source"]["m7_sha256"] = None
    value["p0_source"]["source_status"] = "URI_LOCKED_HASH_UNAVAILABLE"
    _rebind(value)

    evidence = _analyze(value)
    assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
    assert any("content hashes" in defect for defect in evidence["defects"])


def test_campaign_budget_envelope_is_semantically_validated_not_only_self_hashed():
    value = _campaign_input()
    assert _analyze(value)["classification"] == "POLICY_MARGIN_UNCALIBRATED"

    invalid = _campaign_input()
    invalid["task"]["campaign_budget"]["receipt_sha256"] = {
        "unexpected_nested": True,
    }
    _rebind_campaign_budget(invalid)
    with pytest.raises(ValueError, match="digests"):
        _analyze(invalid)


def test_campaign_budget_execution_bindings_require_exact_closed_schema():
    assert _analyze(_campaign_input())["defects"] == []

    mutations = [
        lambda execution: execution.clear(),
        lambda execution: execution.pop("source_snapshot_id"),
        lambda execution: execution.update({"unrelated": {}}),
        lambda execution: execution["gates"].pop("volume_clearance"),
        lambda execution: execution["gates"].update({"unrelated": 1}),
        lambda execution: execution["gates"]["ct_material_support_gate"].pop(
            "level"),
        lambda execution: execution["gates"]["ct_material_support_gate"].update(
            {"unrelated": 1}),
        lambda execution: execution["queue_execution"].pop("planner"),
        lambda execution: execution["queue_execution"].update({"unrelated": 1}),
        lambda execution: execution["queue_execution"]["recenter_radius_xyz"].pop(
            "z"),
    ]
    for mutate in mutations:
        value = _campaign_input()
        mutate(value["task"]["campaign_budget"]["execution_bindings"])
        _rebind_campaign_budget(value)
        with pytest.raises(ValueError, match="campaign"):
            _analyze(value)


def test_campaign_budget_execution_bindings_reject_malformed_hashes_ranges_and_booleans():
    mutations = [
        lambda row: row.update({"p0_artifact_sha256": "z" * 64}),
        lambda row: row.update({"source_content_lock_sha256": "1"}),
        lambda row: row.update({"catalog_snapshot_sha256": True}),
        lambda row: row.update({"m7_uri_sha256": "F" * 64}),
        lambda row: row.update({"code_revision": "a" * 39}),
        lambda row: row.update({"shape_xyz": [True, 100, 100]}),
        lambda row: row.update({"voxel_size_um": 0}),
        lambda row: row.update({"m7_threshold": 1.1}),
        lambda row: row.update({"grid_step": True}),
        lambda row: row.update({"query_radius": 0}),
        lambda row: row.update({"parallelism": 9}),
        lambda row: row.update({"maximum_cells": 4097}),
        lambda row: row["gates"].update({"packet_candidate_limit": 0}),
        lambda row: row["gates"].update({"volume_clearance": True}),
        lambda row: row["gates"]["ct_material_support_gate"].update(
            {"level": True}),
        lambda row: row["gates"]["ct_material_support_gate"].update(
            {"radius_l0_voxels": 0}),
        lambda row: row["queue_execution"].update(
            {"recenter_probe_max_candidates": 0}),
        lambda row: row["queue_execution"]["recenter_radius_xyz"].update(
            {"x": 193}),
        lambda row: row["queue_execution"].update({"verify_sources": 1}),
        lambda row: row["queue_execution"].update({"seed_probe_top_k": True}),
    ]
    for mutate in mutations:
        value = _campaign_input()
        mutate(value["task"]["campaign_budget"]["execution_bindings"])
        _rebind_campaign_budget(value)
        with pytest.raises(ValueError, match="campaign"):
            _analyze(value)


def test_campaign_budget_execution_bindings_cross_bind_source_p0_geometry_grid_policy_provider_gates_and_queue():
    cases = [
        (lambda row: row.update({"ct_sha256": "e" * 64}),
         "campaign execution source binding differs"),
        (lambda row: row.update({"p0_artifact_id": "p0-other"}),
         "campaign execution P0/catalog binding differs"),
        (lambda row: row.update({"grid_version": "other-grid@1.0.0"}),
         "campaign execution geometry/grid binding differs"),
        (lambda row: row["gates"].update(
            {"candidate_interior_clearance": 1}),
         "campaign execution policy/provider/gates binding differs"),
        (lambda row: row["queue_execution"].update({"planner": "other-v2"}),
         "campaign execution queue binding differs"),
    ]
    for mutate, expected_defect in cases:
        value = _campaign_input()
        mutate(value["task"]["campaign_budget"]["execution_bindings"])
        _rebind_campaign_budget(value)
        evidence = _analyze(value)
        assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
        assert expected_defect in evidence["defects"]
        assert evidence["policy_change_authorized"] is False


def test_campaign_budget_geometry_is_independently_derived_from_cell_id_and_frozen_grid():
    value = _campaign_input()
    forged = {
        "center": {"x": 40, "y": 40, "z": 40},
        "radius": {"x": 10, "y": 10, "z": 10},
    }
    value["task"]["center_xyz"] = copy.deepcopy(forged["center"])
    value["task"]["bounds_xyz"] = [[30, 30, 30], [50, 50, 50]]
    value["task"]["candidate_discovery"]["region"] = copy.deepcopy(forged)
    value["raw_candidate_response"]["request"]["region"] = copy.deepcopy(forged)
    value["raw_candidate_response"]["effective_candidate_region"] = copy.deepcopy(
        forged)
    _rebind_campaign_budget(value)

    evidence = _analyze(value)
    assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
    assert "campaign execution geometry/grid binding differs" in evidence["defects"]


def test_campaign_p0_m7_uri_must_be_a_string_before_utf8_hash_binding():
    value = _campaign_input()
    source = value["p0_source"]
    source["m7_uri"] = 123
    value["raw_candidate_response"]["request"]["prediction_uri"] = 123
    value["task"]["candidate_discovery"]["prediction_uri"] = 123
    execution = value["task"]["campaign_budget"]["execution_bindings"]
    execution["m7_uri_sha256"] = hashlib.sha256(b"123").hexdigest()
    _rebind_campaign_budget(value)

    with pytest.raises(ValueError, match="m7_uri|URI|string"):
        _analyze(value)


def test_reachability_accepts_only_the_four_closed_checked_status_variants():
    cases = [
        ({"checked": False, "reason": "not an http(s) URI"}, False),
        ({"checked": True, "reachable": True, "status": 204}, True),
        ({"checked": True, "reachable": False, "status": 503,
          "error": "service unavailable"}, True),
        ({"checked": True, "reachable": False,
          "error": "connection refused"}, True),
    ]
    for state, http in cases:
        evidence = _analyze(_reachability_input(state, http=http))
        assert evidence["classification"] == "POLICY_MARGIN_UNCALIBRATED"
        assert not any("reachability" in row for row in evidence["defects"])


def test_reachability_rejects_checked_reachable_status_error_contradictions():
    cases = [
        ({"checked": False, "reason": "not an http(s) URI",
          "reachable": False}, True, "2026-08-02T00:00:00Z"),
        ({"checked": False, "reason": "not an http(s) URI", "status": 200},
         True, "2026-08-02T00:00:00Z"),
        ({"checked": True, "status": 200}, True, "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": True}, True,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": True, "status": 200,
          "error": "contradiction"}, True, "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": False}, True,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": False, "status": 503}, True,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": False, "error": ""}, True,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": True, "status": True}, True,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": True, "status": 404}, True,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": False, "status": 399,
          "error": "bad branch"}, True, "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": True, "status": 200}, False,
         "2026-08-02T00:00:00Z"),
        ({"checked": True, "reachable": True, "status": 200}, True,
         "2026-08-02T00:00:00"),
        ({"checked": True, "reachable": True, "status": 200}, True,
         "2026-08-02T01:00:00+01:00"),
    ]
    for state, http, timestamp in cases:
        with pytest.raises(ValueError, match="reachability"):
            _analyze(_reachability_input(
                state, http=http, checked_at_utc=timestamp))


@pytest.mark.parametrize(
    "timestamp",
    ["20260802T000000Z", "2026-08-02T00:00Z"],
)
def test_reachability_timestamp_requires_full_rfc3339_date_and_seconds(timestamp):
    with pytest.raises(ValueError, match="timestamp|RFC3339"):
        _analyze(_reachability_input(
            {"checked": True, "reachable": True, "status": 204},
            http=True,
            checked_at_utc=timestamp,
        ))


@pytest.mark.parametrize(
    "timestamp",
    ["2026-08-02T00:00:00.123Z", "2026-08-02T00:00:00+00:00"],
)
def test_reachability_timestamp_accepts_rfc3339_utc_variants(timestamp):
    evidence = _analyze(_reachability_input(
        {"checked": True, "reachable": True, "status": 204},
        http=True,
        checked_at_utc=timestamp,
    ))

    assert evidence["classification"] == "POLICY_MARGIN_UNCALIBRATED"
    assert not any("reachability" in row for row in evidence["defects"])


def test_recenter_z_probe_binds_request_counts_median_effective_region_and_call_count():
    assert _analyze(_recenter_z_input(empty=True))["defects"] == []
    assert _analyze(_recenter_z_input())["defects"] == []

    mutations = [
        lambda value: value["raw_candidate_response"]["initial_probe"].update(
            {"candidate_count": -1}),
        lambda value: value["raw_candidate_response"]["initial_probe"][
            "request"].update({"max_candidates": 99}),
        lambda value: value["raw_candidate_response"]["initial_probe"].update(
            {"median_z_ct_l0": 100}),
        lambda value: value["raw_candidate_response"][
            "effective_candidate_region"]["center"].update({"z": 41}),
        lambda value: value["raw_candidate_response"]["provider_exchange"].update(
            {"call_count": 1}),
        lambda value: value["raw_candidate_response"]["initial_probe"].update(
            {"median_coordinate_ct_l0": {"x": 10, "y": 50, "z": 40}}),
    ]
    for mutate in mutations:
        value = _recenter_z_input()
        mutate(value)
        _rebind(value)
        with pytest.raises(ValueError, match="initial probe"):
            _analyze(value)


def test_recenter_xyz_probe_binds_median_coordinate_to_effective_center_and_volume():
    evidence = _analyze(_recenter_xyz_input())
    assert evidence["classification"] == "CELL_BOUNDARY_ONLY"
    assert evidence["defects"] == []

    mutations = [
        lambda probe: probe.pop("median_coordinate_ct_l0"),
        lambda probe: probe.update({"median_coordinate_ct_l0": {"x": 30, "y": 50}}),
        lambda probe: probe.update({"median_coordinate_ct_l0": {
            "x": 100, "y": 50, "z": 50}}),
        lambda probe: probe["median_coordinate_ct_l0"].update({"z": 49}),
    ]
    for mutate in mutations:
        value = _recenter_xyz_input()
        mutate(value["raw_candidate_response"]["initial_probe"])
        _rebind(value)
        with pytest.raises(ValueError, match="initial probe"):
            _analyze(value)


def test_chunk_safe_probe_requires_eight_ordered_subqueries_with_exact_centers_radii_counts_and_hashes():
    assert _analyze(_chunk_input())["defects"] == []

    mutations = [
        lambda response: response["initial_probe"].update({"subquery_count": 7}),
        lambda response: response["initial_probe"]["subqueries"].pop(),
        lambda response: response["initial_probe"]["subqueries"].append(
            copy.deepcopy(response["initial_probe"]["subqueries"][-1])),
        lambda response: response["initial_probe"]["subqueries"].reverse(),
        lambda response: response["initial_probe"]["subqueries"].__setitem__(
            1, copy.deepcopy(response["initial_probe"]["subqueries"][0])),
        lambda response: response["initial_probe"]["subqueries"][0]["region"][
            "center"].update({"x": 129}),
        lambda response: response["initial_probe"]["subqueries"][0]["region"][
            "radius"].update({"x": 63}),
        lambda response: response["initial_probe"]["subqueries"][0].update(
            {"candidate_count": -1}),
        lambda response: response["initial_probe"]["subqueries"][0].update(
            {"candidate_count": 101}),
        lambda response: response["initial_probe"]["subqueries"][0].update(
            {"response_sha256": "z" * 64}),
        lambda response: response["provider_exchange"].update({"call_count": 8}),
        lambda response: response["initial_probe"].update({"candidate_count": 9}),
    ]
    for mutate in mutations:
        value = _chunk_input()
        mutate(value["raw_candidate_response"])
        _rebind(value)
        with pytest.raises(ValueError, match="initial probe"):
            _analyze(value)


@pytest.mark.parametrize(
    "policy",
    ["m7-recenter-z-chunk-safe-v1", "m7-chunk-safe-merge-interior-v2"],
)
def test_chunk_safe_policies_require_exact_producer_recenter_radius(policy):
    value = _chunk_input(policy)
    value["task"]["candidate_discovery"]["recenter_radius_xyz"] = {
        "x": 10,
        "y": 10,
        "z": 10,
    }
    _rebind(value)

    with pytest.raises(ValueError, match="initial probe|recenter"):
        _analyze(value)


def test_chunk_safe_merge_probe_binds_merged_count_raw_membership_effective_region_and_final_set_flag():
    assert _analyze(_chunk_input(
        "m7-chunk-safe-merge-interior-v2"))["defects"] == []

    mutations = [
        lambda response: response["initial_probe"].update({"candidate_count": 7}),
        lambda response: response["initial_probe"].update(
            {"merged_candidate_count": 7}),
        lambda response: response["initial_probe"].update(
            {"merge_used_as_final_candidate_set": False}),
        lambda response: response["effective_candidate_region"]["center"].update(
            {"z": 193}),
        lambda response: response["provider_exchange"].update({"call_count": 9}),
    ]
    for mutate in mutations:
        value = _chunk_input("m7-chunk-safe-merge-interior-v2")
        mutate(value["raw_candidate_response"])
        _rebind(value)
        with pytest.raises(ValueError, match="initial probe"):
            _analyze(value)

    raw_mismatch = _chunk_input("m7-chunk-safe-merge-interior-v2")
    raw_mismatch["raw_candidate_response"]["candidates"].pop()
    raw_mismatch["ct_screen"]["assessments"].pop()
    raw_mismatch["ct_screen"].update({
        "input_candidate_count": 7,
        "retained_candidate_count": 7,
        "rejected_candidate_count": 0,
    })
    attempt = raw_mismatch["attempt_receipt"]
    attempt.update({
        "raw_candidate_count": 7,
        "post_ct_candidate_count": 7,
        "usable_candidate_count": 0,
    })
    diagnosis = attempt["no_seed_causal_diagnosis"]
    diagnosis.update({
        "m7_raw_candidate_count": 7,
        "ct_support_input_candidate_count": 7,
        "ct_support_retained_candidate_count": 7,
        "post_ct_candidate_count": 7,
    })
    diagnosis["cause_counts"][CAUSE_CELL] = 7
    diagnosis["clearance_rejection_examples_first_32"].pop()
    attempt["no_seed_cause_counts"] = copy.deepcopy(diagnosis["cause_counts"])
    _rebind(raw_mismatch)
    with pytest.raises(ValueError, match="initial probe"):
        _analyze(raw_mismatch)


def test_nonfixed_candidate_clearance_uses_validated_effective_region_not_nominal_task_bounds():
    evidence = _analyze(_recenter_xyz_input())

    assert evidence["classification"] == "CELL_BOUNDARY_ONLY"
    assert evidence["defects"] == []
    assert evidence["candidates"][0]["cell_face_distances_voxels"] == {
        "x_low": 2.0, "x_high": 18.0,
        "y_low": 10.0, "y_high": 10.0,
        "z_low": 10.0, "z_high": 10.0,
    }


def test_classifies_physically_supported_rejections_as_uncalibrated_policy():
    evidence = _analyze(_input())

    assert evidence["classification"] == "POLICY_MARGIN_UNCALIBRATED"
    assert evidence["classification_basis"]["physically_safe_but_policy_rejected"] == 8
    assert evidence["non_claim"] == (
        "This diagnostic does not establish the absence of ink, text, or letters."
    )


def test_sensitivity_uses_frozen_margin_percentages_without_authorizing_change():
    evidence = _analyze(_input())

    assert [row["margin_percentage"] for row in evidence["sensitivity"]] == [
        100, 75, 50, 25, 0
    ]
    assert [row["retained_count"] for row in evidence["sensitivity"]] == [
        0, 5, 8, 8, 8
    ]
    assert all(row["diagnostic_only"] is True for row in evidence["sensitivity"])
    assert evidence["policy_change_authorized"] is False


def test_classifies_candidates_inside_query_radius_as_true_volume_boundary():
    coordinates = [
        {"x": value, "y": 50, "z": 50} for value in [0, 1, 2, 3, 4, 5, 6, 7]
    ]
    evidence = _analyze(_input(coordinates=coordinates))

    assert evidence["classification"] == "TRUE_VOLUME_BOUNDARY"
    assert evidence["classification_basis"]["physically_unsafe_volume_candidates"] == 8


def test_classifies_cell_only_rejection_without_weakening_volume_policy():
    coordinates = [{"x": 50, "y": 50, "z": 50} for _ in range(8)]
    causes = {f"m7-{index}": ["INSUFFICIENT_CELL_INTERIOR_CLEARANCE"]
              for index in range(1, 9)}
    value = _input(coordinates=coordinates, reported_causes=causes)
    value["task"]["bounds_xyz"] = [[45, 45, 45], [55, 55, 55]]
    region = {"center": {"x": 50, "y": 50, "z": 50},
              "radius": {"x": 5, "y": 5, "z": 5}}
    value["raw_candidate_response"]["request"]["region"] = region
    value["raw_candidate_response"]["effective_candidate_region"] = region
    value["task"]["center_xyz"] = {"x": 50, "y": 50, "z": 50}
    value["task"]["candidate_discovery"]["region"] = region
    value["task"]["candidate_discovery"][
        "minimum_cell_interior_clearance_voxels"] = 6
    value["attempt_receipt"]["clearance_policy"][
        "minimum_cell_interior_clearance_voxels"] = 6
    value["attempt_receipt"]["clearance_policy"][
        "minimum_cell_interior_clearance_um"] = 60.0
    value["attempt_receipt"]["no_seed_causal_diagnosis"]["clearance_policy"][
        "minimum_cell_interior_clearance_voxels"] = 6
    _rebind(value)

    evidence = _analyze(value)

    assert evidence["classification"] == "CELL_BOUNDARY_ONLY"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["p0_source"].update({"coordinate_frame": "zyx"}),
        lambda value: value["attempt_receipt"].update({"post_ct_candidate_count": 7}),
        lambda value: value["attempt_receipt"]["no_seed_causal_diagnosis"][
            "clearance_rejection_examples_first_32"][0].update(
                {"causes": ["INSUFFICIENT_CELL_INTERIOR_CLEARANCE"]}),
        lambda value: value["task"].update({"bounds_xyz": [[0, 0, 0], [101, 100, 100]]}),
    ],
)
def test_metadata_or_terminal_disagreement_is_a_defect_not_a_margin_signal(mutate):
    value = _input()
    mutate(value)

    evidence = _analyze(value)

    assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
    assert evidence["defects"]
    assert evidence["policy_change_authorized"] is False


def test_mixed_safe_and_unsafe_volume_rejections_fail_closed_as_true_boundary():
    coordinates = [{"x": 2, "y": 50, "z": 50}] + [
        {"x": value, "y": 50, "z": 50} for value in range(12, 19)
    ]
    evidence = _analyze(_input(coordinates=coordinates))

    assert evidence["classification"] == "TRUE_VOLUME_BOUNDARY"
    assert evidence["classification_basis"]["physically_unsafe_volume_candidates"] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unexpected": True}),
        lambda value: value["p0_source"].update({"shape_xyz": [100.5, 100, 100]}),
        lambda value: value["task"].update({"bounds_xyz": [[0, 0, 0], [99.5, 100, 100]]}),
        lambda value: value["raw_candidate_response"]["candidates"][0][
            "ct_l0_coordinate"].update({"x": 12.0}),
    ],
)
def test_review_input_is_closed_and_integer_exact(mutate):
    value = _input()
    mutate(value)

    with pytest.raises(ValueError):
        _analyze(value)


def test_manifest_and_m7_ct_coordinate_alignment_are_recomputed():
    value = _input()
    value["ct_screen"]["assessments"][0]["coordinate_xyz_l0"]["x"] = 13
    _rebind(value)

    evidence = _analyze(value)

    assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
    assert any("coordinate" in defect for defect in evidence["defects"])


def test_trusted_manifest_root_prevents_coordinated_bundle_rehash():
    value = _input()
    trusted = _sha(value["evidence_manifest"])
    value["p0_source"]["m7_sha256"] = "9" * 64
    _rebind(value)

    with pytest.raises(ValueError, match="trusted manifest"):
        _analyze(value, trusted_manifest_sha256=trusted)


def test_m7_ct_transform_and_filtered_response_are_validated():
    transform = _input()
    transform["alignment_authority"]["m7_to_ct_l0_transform"] = "not-a-transform"
    _rebind(transform)
    with pytest.raises(ValueError, match="transform"):
        _analyze(transform)

    filtered = _input()
    filtered["ct_screen"]["filtered_response_sha256"] = "0" * 64
    filtered["evidence_manifest"]["files"]["CT_MATERIAL_SUPPORT_SCREEN.json"] = {
        "sha256": _sha(filtered["ct_screen"])
    }
    evidence = _analyze(filtered)
    assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
    assert any("filtered response" in defect for defect in evidence["defects"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sample: sample.update({"level": 2}),
        lambda sample: sample.update({"scale_zyx": "not-a-vector"}),
        lambda sample: sample.update({"voxel_count": "not-an-int"}),
        lambda sample: sample.update({"nonzero_fraction": "not-a-number"}),
        lambda sample: sample.update({"nonzero_voxel_count": 1000}),
    ],
)
def test_ct_support_is_recomputed_from_closed_sample_metrics(mutate):
    value = _input()
    mutate(value["ct_screen"]["assessments"][0]["sample"])
    _rebind(value)

    with pytest.raises(ValueError, match="CT sample"):
        _analyze(value)


def test_ct_sample_center_is_bound_to_candidate_and_level_scale():
    value = _input()
    value["ct_screen"]["assessments"][0]["sample"]["center_zyx"] = [1, 1, 1]
    _rebind(value)

    evidence = _analyze(value)
    assert evidence["classification"] == "IMPLEMENTATION_OR_METADATA_DEFECT"
    assert any("sample geometry" in defect for defect in evidence["defects"])


def test_read_set_requires_lowercase_hex_sha256_and_initial_probe_is_closed():
    read_set = _input()
    read_set["raw_candidate_response"]["source_read_set"]["objects"][0][
        "sha256"] = "z" * 64
    _rebind(read_set)
    with pytest.raises(ValueError, match="object identity"):
        _analyze(read_set)

    probe = _input()
    probe["raw_candidate_response"]["initial_probe"] = {"unknown_nested": True}
    _rebind(probe)
    with pytest.raises(ValueError, match="fixed-v1"):
        _analyze(probe)

    nonfixed = _input()
    nonfixed["task"]["candidate_discovery"]["seed_region_policy"] = "m7-recenter-z-v1"
    nonfixed["raw_candidate_response"]["initial_probe"] = {
        "policy": "m7-recenter-z-v1", "candidate_count": 8,
        "recentered": True, "request": {"unexpected_nested": True},
    }
    _rebind(nonfixed)
    with pytest.raises(ValueError, match="initial probe request"):
        _analyze(nonfixed)

    scalar = _input()
    scalar["task"]["candidate_discovery"]["seed_region_policy"] = "m7-recenter-z-v1"
    scalar["raw_candidate_response"]["initial_probe"] = {
        "policy": "m7-recenter-z-v1", "candidate_count": 8,
        "recentered": True, "subquery_count": {"unexpected_nested": True},
    }
    _rebind(scalar)
    with pytest.raises(ValueError, match="subquery_count"):
        _analyze(scalar)


def test_output_hash_and_closed_schema_reject_tampering():
    evidence = _analyze(_input())
    assert analysis.validate_candidate_clearance_review(evidence) == evidence

    tampered = json.loads(json.dumps(evidence))
    tampered["classification"] = "TRUE_VOLUME_BOUNDARY"
    with pytest.raises(ValueError, match="hash"):
        analysis.validate_candidate_clearance_review(tampered)

    extended = json.loads(json.dumps(evidence))
    extended["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        analysis.validate_candidate_clearance_review(extended)

    nested = json.loads(json.dumps(evidence))
    nested["candidates"][0]["unexpected_nested"] = True
    nested["review_sha256"] = _sha({key: row for key, row in nested.items()
                                     if key != "review_sha256"})
    with pytest.raises(ValueError, match="fields"):
        analysis.validate_candidate_clearance_review(nested)

    bad_defects = json.loads(json.dumps(evidence))
    bad_defects["defects"] = [{"unexpected_nested": True}]
    bad_defects["review_sha256"] = _sha({key: row for key, row in bad_defects.items()
                                         if key != "review_sha256"})
    with pytest.raises(ValueError, match="defects"):
        analysis.validate_candidate_clearance_review(bad_defects)

    bad_shape = json.loads(json.dumps(evidence))
    bad_shape["shape_xyz"] = {"unexpected_nested": True}
    bad_shape["review_sha256"] = _sha({key: row for key, row in bad_shape.items()
                                       if key != "review_sha256"})
    with pytest.raises(ValueError, match="source geometry"):
        analysis.validate_candidate_clearance_review(bad_shape)

    bad_policy = json.loads(json.dumps(evidence))
    bad_policy["clearance_policy"]["voxel_size_um"] = 0
    bad_policy["review_sha256"] = _sha({key: row for key, row in bad_policy.items()
                                        if key != "review_sha256"})
    with pytest.raises(ValueError, match="physical units"):
        analysis.validate_candidate_clearance_review(bad_policy)


def test_cli_rejects_same_input_output_and_leaves_input_untouched(tmp_path):
    source = tmp_path / "input.json"
    original = _input()
    source.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct"):
        analysis.main([
            "--input", str(source), "--output", str(source),
            "--trusted-manifest-sha256", _sha(original["evidence_manifest"]),
        ])

    assert json.loads(source.read_text(encoding="utf-8")) == original


def test_cli_writes_canonical_evidence_without_mutating_the_input(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "evidence.json"
    original = _input()
    source.write_text(json.dumps(original), encoding="utf-8")

    assert analysis.main([
        "--input", str(source), "--output", str(output),
        "--trusted-manifest-sha256", _sha(original["evidence_manifest"]),
    ]) == 0

    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["schema"] == "campaignx.candidate_clearance_review.v1"
    assert len(evidence["review_sha256"]) == 64
    assert json.loads(source.read_text(encoding="utf-8")) == original
