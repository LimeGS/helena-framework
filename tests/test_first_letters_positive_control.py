"""Behavioral contract for the First Letters positive-control runner.

The fake is the HTTP boundary.  Everything inside the runner -- stage order,
classification, receipt hashing, and prerequisite stopping -- is real.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from control_manifest import IN_FORCE_ID, IN_FORCE_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from panel_client import AmbiguousMutationError  # noqa: E402
from run_first_letters_positive_control import (  # noqa: E402
    canonical_sha256,
    evaluate_survival_matrix,
    run_positive_control,
)
from fleet.generator import generate_manual_tasks  # noqa: E402
from fleet.common import stable_id  # noqa: E402
from job_store import validate_parameters  # noqa: E402

MANIFEST_PATH = IN_FORCE_PATH
REVISION = "1" * 40
MISSION = "first-letters-control-20260802"


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def runtime(document: dict) -> dict:
    profile_locks = []
    for row in document["profile_locks"]:
        profile_document = json.loads((ROOT / row["path"]).read_text(encoding="utf-8"))
        profile_locks.append({
            **copy.deepcopy(row),
            "declared_sha256_semantics": "RAW_FILE_BYTES_SHA256",
            "actual_sha256": row["sha256"],
            "actual_file_sha256": row["sha256"],
            "actual_canonical_document_sha256": canonical_sha256(profile_document),
            "actual_profile_id": row["profile_id"],
            "verified": True,
        })
    return {
        "deployed_revision": REVISION,
        "control_profile_id": document["profile_id"],
        "control_profile_sha256": canonical_sha256(document),
        "source_locks_sha256": canonical_sha256(document["source_locks"]),
        "profile_locks": profile_locks,
        "profile_locks_verified": True,
        "models": [
            {**row, "installed": True, "installed_at": "/models/timesformer.ckpt"}
            for row in document["model_locks"]
        ],
        "resource_identity": {"panel": "helena-test", "host": "control-1"},
    }


def discovery(**changes) -> dict:
    locks = manifest()["source_locks"]
    ct_objects = [
        {"object_key": row["path"], "sha256": row["sha256"], "bytes": 11}
        for row in locks["ct"]["metadata"]
    ] + [{"object_key": "0/1/2/3", "sha256": "b" * 64, "bytes": 21}]
    m7_objects = [
        {"object_key": row["path"], "sha256": row["sha256"], "bytes": 13}
        for row in locks["m7"]["metadata"]
    ] + [{"object_key": "0/1/2/3", "sha256": "d" * 64, "bytes": 17}]
    ct_objects.sort(key=lambda row: row["object_key"])
    m7_objects.sort(key=lambda row: row["object_key"])
    surface_objects = [{"object_key": "meta.json", "sha256": "a" * 64, "bytes": 11}]
    answer = {
        # The four the worker's finishing gate checks by name before it will
        # persist anything. `run_control_region_preflight` returns all of them;
        # a stand-in without them is refused as PREFLIGHT_RECEIPT_NOT_INK_BLIND,
        # which reads as a science failure and is a gap in the fixture.
        "schema": "campaignx.segment_candidate_coverage_preflight.v1",
        "ink_used": False,
        "growth_allowed": False,
        "state_mutation": "NONE",
        "state": "SUCCEEDED",
        "status": "COMPLETE",
        "counts": {
            "raw_m7": 4,
            "post_ct": 3,
            "post_clearance": 2,
            "packet_limited": 2,
        },
        "closest_survivor_distance_ct_l0_voxels": 1.25,
        "planned_region_count": 12,
        "visited_region_count": 4,
        "coverage_fraction": 1 / 3,
        "coverage_complete": True,
        "parameters": {
            "server_normalized": "frozen-control-policy",
            "maximum_surface_probe_regions": 4096,
        },
        "ct_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": ct_objects,
            "canonical_manifest_sha256": canonical_sha256(ct_objects),
        },
        "m7_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": m7_objects,
            "canonical_manifest_sha256": canonical_sha256(m7_objects),
        },
        "surface_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": surface_objects,
            "canonical_manifest_sha256": canonical_sha256(surface_objects),
        },
        "provider_exchange": {
            "request_sha256": "f" * 64,
            "request_bytes": 99,
            "response_sha256": "0" * 64,
            "response_bytes": 123,
        },
        "output_hashes": {"discovery_receipt_sha256": "2" * 64},
        "resource_identity": {"provider": "vc3d-mcp@1.0.0", "host": "control-1"},
    }
    answer.update(changes)
    return answer


def pipeline(**changes) -> dict:
    answer = {
        # What the fleet finalizes a completed grow with. It was `SUCCEEDED`
        # here, which no attempt reaches -- the fixture agreed with the runner
        # and neither agreed with the fleet, which is why P1 could never pass.
        "state": "QC_PENDING",
        "surface_id": "surface-control",
        "attempt_id": "attempt-control",
        "artifact_id": "surface-control",
        "artifact_sha256": "3" * 64,
        "area_cm2": 0.24,
        "canonical_full_grow": True,
        "seed_origin": "human",
        "submitted_by": "helena-scientist",
        "coordinate_ct_l0_xyz": [3079.062744140625, 3961.3037109375, 4441.35595703125],
        "resource_identity": {"worker": "segment-1"},
    }
    answer.update(changes)
    return answer


def p2_result(**changes) -> dict:
    geometry_result = {
        "schema": "campaignx.tifxyz_geometry_certification.v1",
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
        "measurement_complete": True,
        "status": "PASS",
    }
    answer = {
        "state": "SUCCEEDED",
        "surface_id": "surface-control",
        "surface_artifact_sha256": "3" * 64,
        "geometry_certified_by_job_id": "p2-job",
        "profile_id": "tifxyz-geometry-certification@1.0.0",
        "profile_sha256": "e2b0dd1c3608edc5a190b676dcddd1a5bbf4cb904f11c74e89b447af60476a6e",
        "result": geometry_result,
        "result_sha256": canonical_sha256(geometry_result),
        "geometry_qc_state": "GEOMETRY_CERTIFIED",
        "physical_qc_state": "CT_SUPPORTED_REVIEW",
        "receipt_sha256": "4" * 64,
        "resource_identity": {"worker": "qc-1"},
    }
    answer.update(changes)
    return answer


def p3_result(**changes) -> dict:
    objects = [
        {"object_key": name, "sha256": str(index) * 64, "bytes": 42 + index}
        for index, name in enumerate(("meta.json", "x.tif", "y.tif", "z.tif"), start=1)
    ]
    files = {row["object_key"]: {"sha256": row["sha256"], "size_bytes": row["bytes"]}
             for row in objects}
    artifact_sha256 = canonical_sha256(files)
    receipt = {
        "schema": "campaignx.surface_flattening_receipt.v1",
        "requested_by_job_id": "p3-job",
        "surface_id": "surface-control",
        "source_artifact_sha256": "3" * 64,
        "profile_id": "flatten-abf-v1@1.0.0",
        "profile_file_sha256": "5771b2dc6465c8e59e530fa6410f5cdffafcda4b429ea3c5279b0e98834117a8",
        "artifact_uri": "s3://control/flat",
        "artifact_sha256": artifact_sha256,
        "files": len(objects),
        "objects": objects,
    }
    answer = {
        "state": "FLATTENED",
        "surface_id": "surface-control",
        "source_artifact_sha256": "3" * 64,
        "requested_by_job_id": "p3-job",
        "artifact_id": "flat-control",
        "artifact_uri": "s3://control/flat",
        "artifact_sha256": artifact_sha256,
        "files": len(objects),
        "objects": objects,
        "profile_id": "flatten-abf-v1@1.0.0",
        "profile_file_sha256": "5771b2dc6465c8e59e530fa6410f5cdffafcda4b429ea3c5279b0e98834117a8",
        "receipt": receipt,
        "receipt_sha256": canonical_sha256(receipt),
        "resource_identity": {"worker": "flatten-1"},
    }
    answer.update(changes)
    return answer


def orientation_result(document: dict, **changes) -> dict:
    """Synthetic scientific proof used only to exercise runner plumbing."""
    p3 = p3_result()
    orientation = document["checks"]["PIPELINE_CONTROL"]["orientation_parity"]
    absolute = {
        "verified": True,
        "evidence_receipt_sha256": "6" * 64,
        "same_winding_flip_normals": False,
    }
    receipt = {
        "schema": "campaignx.first_letters_orientation_parity.v1",
        "profile_id": orientation["policy"]["profile_id"],
        "profile_sha256": canonical_sha256(orientation["policy"]),
        "lineage": {
            "reference": {
                "uri": document["source_locks"]["community_surface"]["uri"],
                "objects": document["source_locks"]["community_surface"]["artifacts"],
                "artifact_manifest_sha256": canonical_sha256(
                    document["source_locks"]["community_surface"]["artifacts"]),
            },
            "grown_mesh_artifact": {
                "artifact_id": "surface-control", "sha256": "3" * 64,
            },
            "flattened_artifact": {
                "artifact_id": "flat-control", "sha256": p3["artifact_sha256"],
            },
            "p3": {
                "job_id": "p3-job", "profile_id": p3["profile_id"],
                "profile_sha256": p3["profile_file_sha256"],
                "receipt_sha256": p3["receipt_sha256"],
            },
            "absolute_orientation": absolute,
        },
        "policy": copy.deepcopy(orientation["policy"]),
        "status": "PROVEN",
        "reason_code": "ORIENTATION_PROVEN",
        "parity_state": "PROVEN_SAME_WINDING",
        "selected_flip_normals": False,
    }
    receipt.update(changes)
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"})
    return receipt


def p4_result(**changes) -> dict:
    objects = [
        {"object_key": f"{index}.tif", "sha256": "8" * 64, "bytes": 42}
        for index in range(33)]
    files = {row["object_key"]: {"sha256": row["sha256"], "size_bytes": row["bytes"]}
             for row in objects}
    artifact_sha256 = canonical_sha256(files)
    layer_manifest_document = {"schema": "campaignx.layer_stack_artifact_set.v1",
                               "job_id": "p4-job", "files": files,
                               "artifact_sha256": artifact_sha256}
    layer_manifest_sha256 = canonical_sha256(layer_manifest_document)
    metric_policy = manifest()["checks"]["PIPELINE_CONTROL"]["p3_p4_lateral_metric"]["policy"]
    metric = {
        "schema": "campaignx.first_letters_p3_p4_lateral_metric.v1",
        "profile_id": metric_policy["profile_id"],
        "profile_sha256": canonical_sha256(metric_policy),
        "status": "PROVEN",
        "lineage": {
            "flattened_artifact_id": "flat-control",
            "flattened_artifact_sha256": p3_result()["artifact_sha256"],
            "p3_job_id": "p3-job",
            "p4_job_id": "p4-job",
            "p4_layer_artifact_sha256": artifact_sha256,
            "p4_layer_manifest_sha256": layer_manifest_sha256,
            "p4_layer_objects": objects,
        },
        "source_voxel_um": 9.362,
        "lateral_pixel_um": 8.5,
        "minimum_valid_triangle_fraction": 0.95,
        "valid_triangle_fraction": 1.0,
        "maximum_uv_to_3d_distortion_ratio": 1.25,
        "observed_uv_to_3d_distortion_ratio": 1.0,
        "raster_transform": {"scale_xy": [1.0, 1.0], "offset_xy": [0.0, 0.0]},
    }
    metric["receipt_sha256"] = canonical_sha256(metric)
    answer = {
        "layers": {"slices": 33, "shape": [364, 340], "dtype": "uint16"},
        "layer_stack": {"artifact_uri": "s3://control/layers",
                        "artifact_sha256": artifact_sha256,
                        "manifest_sha256": layer_manifest_sha256,
                        "files": 33, "objects": objects,
                        "slice_indices": list(range(33)),
                        "slice_ordering": "NUMERIC_STEM_CONTIGUOUS_ASCENDING"},
        "lateral_metric": metric,
        "resource_identity": {"worker": "render-1"},
    }
    answer.update(changes)
    return answer


def p5_result(**changes) -> dict:
    objects = [
        {"object_key": "mean_probability.npy", "sha256": "9" * 64, "bytes": 42},
        {"object_key": "INK_SCREENING_RECEIPT.json", "sha256": "0" * 64, "bytes": 84},
    ]
    files = {row["object_key"]: {"sha256": row["sha256"], "size_bytes": row["bytes"]}
             for row in objects}
    p4 = p4_result()
    normalization = {
        "schema": "campaignx.first_letters_p5_normalization.v1",
        "p4_job_id": "p4-job",
        "p4_layer_artifact_sha256": p4["layer_stack"]["artifact_sha256"],
        "p4_layer_manifest_sha256": p4["layer_stack"]["manifest_sha256"],
        "source_layer_objects": p4["layer_stack"]["objects"],
        "lateral_metric_receipt_sha256": p4["lateral_metric"]["receipt_sha256"],
        "source_pixel_um": 8.5,
        "source_slice_um": 9.362,
        "training_pixel_um": 7.91,
        "checkpoint_sha256": "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488",
        "profile_id": "timesformer-gp-scroll1-screening@1.0.0",
        "profile_sha256": "189bbdf8dc32de8b552c09c1137dd9daea96807624cfcf671a824ebbf20dd6db",
    }
    normalization["receipt_sha256"] = canonical_sha256(normalization)
    map_manifest = {"schema": "campaignx.ink_probability_map.v1",
                    "job_id": "p5-job", "files": files,
                    "artifact_sha256": canonical_sha256(files)}
    answer = {
        "probability_map": {"artifact_uri": "s3://control/map",
                            "artifact_sha256": canonical_sha256(files),
                            "manifest_sha256": canonical_sha256(map_manifest),
                            "files": 2, "objects": objects},
        "liveness": {"verdict": "ALIVE"},
        "checkpoint_sha256": "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488",
        "physical_normalization": normalization,
        "map_shape_yx": [364, 340],
        "resource_identity": {"worker": "ink-1"},
    }
    answer.update(changes)
    return answer


def roi_result(document: dict, **changes) -> dict:
    screening = p5_result()
    lock = {
        "verified": True,
        "provenance_artifact_uri": "locked://first-letters/letterform-roi.json",
        "provenance_artifact_sha256": "6" * 64,
        "source_coordinate_system": "flat-control-uv",
        "source_bbox_xyxy": [20, 30, 120, 130],
        "p5_transform_receipt_sha256": "7" * 64,
        "transformed_bbox_xyxy": [20, 30, 120, 130],
        "verified_training_pixel_um": 7.91,
    }
    receipt = {
        "schema": "campaignx.first_letters_positive_control_roi.v1",
        "status": "PROVEN",
        "reason_code": "POSITIVE_CONTROL_ROI_PROVEN",
        "lock": lock,
        "lineage": {
            "surface_id": "surface-control",
            "p5_job_id": "p5-job",
            "probability_map_sha256": screening["probability_map"]["artifact_sha256"],
            "probability_map_manifest_sha256": screening[
                "probability_map"]["manifest_sha256"],
            "normalization_receipt_sha256": screening[
                "physical_normalization"]["receipt_sha256"],
            "checkpoint_sha256": screening["checkpoint_sha256"],
            "profile_id": screening["physical_normalization"]["profile_id"],
            "profile_sha256": screening["physical_normalization"]["profile_sha256"],
        },
        "map_shape_yx": screening["map_shape_yx"],
        "transformed_bbox_xyxy": lock["transformed_bbox_xyxy"],
        "verified_training_pixel_um": lock["verified_training_pixel_um"],
    }
    receipt.update(changes)
    receipt["receipt_sha256"] = canonical_sha256({
        key: value for key, value in receipt.items() if key != "receipt_sha256"})
    return receipt


def p7_result(**changes) -> dict:
    probability_map = p5_result()["probability_map"]
    answer = {
        "adjudication": {"verdict": "PASS", "overall": {"pass": True},
                         "verdict_sha256": "c" * 64, "card_sha256": "d" * 64,
                         "config_hash": "e" * 64},
        "probability_map_input": {
            "screened_by": "p5-job",
            "artifact_sha256": probability_map["artifact_sha256"],
            "manifest_sha256": probability_map["manifest_sha256"],
        },
        "resource_identity": {"worker": "validation-1"},
    }
    answer.update(changes)
    return answer


def human_result(**changes) -> dict:
    answer = {
        "schema": "campaignx.human_review_event.v1",
        "review_event_id": "human-review-control",
        "intent": "INSPECT",
        "note": "First Letters positive-control packet routed for human inspection",
        "mission_id": MISSION,
        "sample_id": "PHerc0139",
        "surface_id": "surface-control",
        "vetting_packet_sha256": "b" * 64,
        "p7_job_id": "p7-job",
        "p5_job_id": "p5-job",
        "p4_job_id": "p4-job",
        "p3_job_id": "p3-job",
        "flattened_artifact_id": "flat-control",
        "flattened_artifact_sha256": p3_result()["artifact_sha256"],
        "p4_layer_artifact_sha256": p4_result()["layer_stack"]["artifact_sha256"],
        "p7_artifact_id": "p7-control",
        "verdict_sha256": "c" * 64,
        "card_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "roi_receipt_sha256": roi_result(manifest())["receipt_sha256"],
        "by": "helena-scientist",
        "at": "2026-08-02T00:00:00+00:00",
    }
    answer.update(changes)
    answer["event_sha256"] = canonical_sha256({
        key: value for key, value in answer.items() if key != "event_sha256"})
    return answer


class ScriptedPanel:
    """Strict HTTP-boundary fake using Helena's real routes and queue schemas."""

    def __init__(self, document: dict, *, overrides: dict[str, object] | None = None):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.responses: dict[str, object] = {
            "runtime": runtime(document),
            "discovery": discovery(),
            "manual": {
                "inserted": 1,
                "submitted_by": "helena-scientist",
                "receipt": {"seed_origin": "human", "policy_version": document["profile_id"]},
            },
            "pipeline": pipeline(),
            "p2": p2_result(),
            "p3": p3_result(),
            "orientation": orientation_result(document),
            "p4": p4_result(),
            "p5": p5_result(),
            "roi": roi_result(document),
            "p7": p7_result(),
            "qc_jobs": [{
                "qc_job_id": "qc-control", "surface_id": "surface-control",
                "profile_id": "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0",
                "profile_sha256": "e0e099eac347f4c590f4428dc361be8bdb5432ccb5b791c2801f131d0eaa4793",
                "state": "COMPLETED", "surface_artifact_sha256": "3" * 64,
                "source_attempt_id": "attempt-control",
                "created_geometry_certified": True,
                "result": {
                    "schema": "campaignx.segment_surface_qc_result.v1",
                    "surface_id": "surface-control",
                    "status": "QC_RETAINED",
                    "evidence_manifest_sha256": "7" * 64,
                },
            }],
            "human": human_result(),
            "initial_reviews": [],
        }
        self.responses.update(overrides or {})
        for job in self.responses["qc_jobs"]:
            if isinstance(job.get("result"), dict):
                job.setdefault("result_sha256", canonical_sha256(job["result"]))
        self.manual_queued = False
        self.reviewed = False
        self.job_parameters: dict[str, dict] = {}
        self.progress_events: list[dict] = []

    def call(self, method: str, path: str, body: dict | None = None, *,
             timeout: float | None = None) -> dict:
        self.calls.append((method, path, body))
        if method == "GET" and path.startswith("/api/state?"):
            return {"first_letters_control_runtime": copy.deepcopy(self.responses["runtime"])}
        elif method == "GET" and path.startswith(f"/api/missions/{MISSION}/artifacts?"):
            if "phase=P0" in path:
                return {"artifacts": [{"artifact_id": "p0-pherc0139",
                                        "content_sha256": "a" * 64, "selected": True,
                                        "phase": "P0", "sample_id": "PHerc0139"}]}
            if "phase=P7" in path:
                return {"artifacts": [{"artifact_id": "p7-control",
                                        "content_sha256": "b" * 64,
                                        "produced_by": "job:p7-job",
                                        "phase": "P7", "sample_id": "PHerc0139"}]}
            raise AssertionError(path)
        elif method == "POST" and path == "/api/segmentation/preflight":
            # The preflight is queued now: the POST hands back a handle and the
            # measurement arrives from the status route. A scripted exception
            # still belongs here, because that is where an ambiguous mutation
            # happens.
            value = self.responses["discovery"]
            if isinstance(value, BaseException):
                raise value
            return {"schema": "campaignx.segment_control_region_preflight_handle.v1",
                    "preflight_job_id": "pf-control", "state": "PENDING",
                    "created": True}
        elif method == "GET" and path.startswith("/api/segmentation/preflight/"):
            value = self.responses["discovery"]
            if isinstance(value, BaseException):
                raise value
            return {**copy.deepcopy(value), "job_state": "COMPLETED",
                    "preflight_job_id": "pf-control", "resource_identity": {
                        **copy.deepcopy(value).get("resource_identity", {}),
                        "p0_artifact_id": "p0-pherc0139",
                        "p0_artifact_sha256": "a" * 64,
                        "source_snapshot_id": "control-source",
                    }}
        elif method == "GET" and path.startswith("/api/segmentation/runs?"):
            return {"runs": [copy.deepcopy(self.responses["pipeline"])] if self.manual_queued else []}
        elif path == "/api/segmentation/manual-seeds":
            value = self.responses["manual"]
            if isinstance(value, BaseException):
                raise value
            self.manual_queued = True
            self.manual_body = copy.deepcopy(body)
            answer = copy.deepcopy(value)
            answer.setdefault("resource_identity", {
                "p0_artifact_id": "p0-pherc0139", "p0_artifact_sha256": "a" * 64,
                "source_snapshot_id": "control-source",
            })
            return answer
        elif method == "GET" and path.startswith("/api/segmentation/attempt/"):
            candidate = pipeline()
            return {"payload": {
                "source_snapshot_id": "control-source",
                "grid_version": self.manual_body["grid_version"],
                "policy_version": self.manual_body["policy_version"],
                "manual_candidates": [{
                "seed_origin": candidate["seed_origin"],
                "submitted_by": candidate["submitted_by"],
                "ct_l0_coordinate": dict(zip("xyz", candidate["coordinate_ct_l0_xyz"], strict=True)),
            }]}}
        elif path == "/api/geometry/certify":
            key, job_id = "p2", "p2-job"
        elif path == "/api/flattening/run":
            key, job_id = "p3", "p3-job"
        elif method == "POST" and path == "/api/jobs":
            key = str((body or {}).get("phase", "")).lower()
            if key == "p4":
                assert (body or {})["parameters"]["remote_url"] == (
                    manifest()["source_locks"]["ct"]["uri"])
                (body or {})["parameters"]["volume"] = "/control-cache/PHerc0139"
            validate_parameters(dict((body or {})["parameters"]), str((body or {})["phase"]))
            job_id = f"{key}-job"
        elif method == "GET" and path.startswith("/api/jobs/") and path.endswith("/events"):
            job_id = path.split("/")[3]
            key = job_id.split("-", 1)[0]
            payload = {
                "p2": copy.deepcopy(self.responses["p2"]),
                "p3": copy.deepcopy(self.responses["p3"]),
                "p4": {
                    "surface_id": "surface-control",
                    "flattening_id": "flat-control",
                    "p3_job_id": "p3-job",
                    "flattened_artifact_sha256": self.responses["p3"][
                        "artifact_sha256"],
                    "profile_id": "flatten-abf-v1@1.0.0",
                    "orientation_receipt_sha256": self.job_parameters.get(
                        "p4-job", {}).get("orientation_receipt_sha256"),
                    "flip_normals": self.job_parameters.get(
                        "p4-job", {}).get("flip_normals"),
                },
                "p5": {"rendered_by": "p4-job"},
                "p7": {"screened_by": "p5-job"},
            }.get(key, {})
            if key == "p5":
                normalization = self.responses["p5"].get("physical_normalization") or {}
                payload.update({
                    "layer_stack_artifact_sha256": normalization.get(
                        "p4_layer_artifact_sha256"),
                    "layer_stack_manifest_sha256": normalization.get(
                        "p4_layer_manifest_sha256"),
                    "lateral_metric_receipt_sha256": normalization.get(
                        "lateral_metric_receipt_sha256"),
                    "source_pixel_um": normalization.get("source_pixel_um"),
                    "source_slice_um": normalization.get("source_slice_um"),
                })
            if key == "p7":
                payload.update({
                    "roi_receipt_sha256": self.job_parameters.get(
                        "p7-job", {}).get("roi_receipt_sha256"),
                    "bbox": self.job_parameters.get("p7-job", {}).get("bbox"),
                    "px_um": self.job_parameters.get("p7-job", {}).get("px_um"),
                    "surface_id": self.job_parameters.get("p7-job", {}).get("surface_id"),
                    "probability_map_artifact_sha256": self.job_parameters.get(
                        "p7-job", {}).get("probability_map_artifact_sha256"),
                    "probability_map_manifest_sha256": self.job_parameters.get(
                        "p7-job", {}).get("probability_map_manifest_sha256"),
                })
            return {"events": [{
                "event_type": "succeeded" if key in {"p2", "p3"} else "rendered_from",
                "payload": payload,
            }]}
        elif method == "GET" and path.startswith("/api/segmentation/segments?"):
            p2 = self.responses["p2"]
            row = {
                "surface_id": "surface-control", "sample_id": "PHerc0139",
                "artifact_sha256": "3" * 64,
                "geometry_qc_state": p2.get("geometry_qc_state"),
                "physical_qc_state": p2.get("physical_qc_state"),
                "geometry_certification": {
                    "schema": "campaignx.tifxyz_geometry_certification.v1",
                    "surface_artifact_sha256": "3" * 64,
                    "source_attempt_id": p2.get("source_attempt_id", "attempt-control"),
                    "profile_id": p2.get("profile_id"),
                    "profile_sha256": p2.get("profile_sha256"),
                    "result": p2.get("result"),
                    "result_sha256": p2.get("result_sha256"),
                },
            }
            if self.reviewed:
                row["human_review"] = copy.deepcopy(self.responses["human"])
            return {"segments": [row]}
        elif method == "GET" and path.startswith("/api/segmentation/qc-jobs?"):
            return {"jobs": copy.deepcopy(self.responses["qc_jobs"])}
        elif method == "GET" and path.startswith("/api/flattening?"):
            value = copy.deepcopy(self.responses["p3"])
            return {"rows": [value]}
        elif method == "GET" and path.startswith("/api/geometry/orientation-proof?"):
            return copy.deepcopy(self.responses["orientation"])
        elif method == "GET" and path.startswith("/api/validation/positive-control-roi?"):
            return copy.deepcopy(self.responses["roi"])
        elif method == "GET" and path == "/api/jobs/p7-job/review":
            rows = copy.deepcopy(self.responses["initial_reviews"])
            if self.reviewed:
                rows.append(copy.deepcopy(self.responses["human"]))
            return {"human_reviews": rows}
        elif method == "POST" and path == "/api/jobs/p7-job/review":
            assert body == {
                "verdict": "INSPECT",
                "note": "First Letters positive-control packet routed for human inspection",
            }
            self.reviewed = True
            return copy.deepcopy(self.responses["human"])
        elif method == "POST" and path == f"/api/missions/{MISSION}/first-letters-control/progress":
            # The real route accepts this and appends a line; a fake that only
            # ever exercised the swallowed-failure path would never prove the
            # happy path posts anything at all.
            event = dict((body or {}).get("event") or {})
            self.progress_events.append(event)
            return {**event, "received_at_utc": "2026-01-01T00:00:00+00:00"}
        else:
            raise AssertionError(f"unexpected panel call: {method} {path}")
        value = self.responses[key]
        if isinstance(value, BaseException):
            raise value
        if key in {"p2", "p3"}:
            self.job_parameters[job_id] = {
                field: copy.deepcopy((body or {})[field])
                for field in ("surface_id", "limit", "dry_run", "allow_unvalidated")
                if field in (body or {})
            }
        else:
            self.job_parameters[job_id] = copy.deepcopy((body or {}).get("parameters", {}))
        return {"job_id": job_id}

    def wait_for_job(self, job_id: str, *, minutes: float = 60, tick: float = 15,
                     on_tick=None) -> dict:
        # Resolves on the first poll rather than looping, but still ticks once --
        # the real Panel never resolves without at least checking, and a caller
        # wiring a heartbeat through on_tick needs that one call to be real here
        # too, not skipped because the fake is fast.
        if on_tick:
            on_tick()
        key = job_id.split("-", 1)[0]
        result = copy.deepcopy(self.responses[key])
        return {"job_id": job_id, "state": "succeeded", "worker_id": f"{key}-worker",
                "parameters": copy.deepcopy(self.job_parameters[job_id]), "result": result}

    def wait_until(self, predicate, *, minutes: float, tick: float, on_tick=None):
        if on_tick:
            on_tick()
        return predicate()


def run(panel: ScriptedPanel) -> dict:
    return run_positive_control(
        panel,
        manifest(),
        mission_id=MISSION,
        deployed_revision=REVISION,
        submitted_by="helena-scientist",
    )


def assert_stopped_before(panel: ScriptedPanel, path: str) -> None:
    assert path not in [called_path for _, called_path, _ in panel.calls]


def test_pure_matrix_uses_first_nonpass_and_erases_impossible_later_passes():
    """Catches a hand-authored PASS or later row overriding a failed prerequisite."""
    stages = [
        {"boundary": boundary, "terminal_state": "PASS", "reason_code": "IMPOSSIBLE"}
        for boundary in ("P0", "P1", "P2", "QC", "P3", "P4", "P5", "P7", "HUMAN_REVIEW")
    ]
    stages[0]["reason_code"] = "OK"
    stages[1].update(terminal_state="FAILED", reason_code="ZERO_RAW_M7_CANDIDATES")
    receipt = evaluate_survival_matrix({"schema": "campaignx.first_letters_stage_survival.v1", "stages": stages})
    assert receipt["control_state"] == "CONTROL_FAILED"
    assert receipt["first_nonpassing_boundary"] == "P1"
    assert receipt["stages"][2]["terminal_state"] == "NOT_RUN_PREREQUISITE"
    assert receipt["stages"][2]["reason_code"] == "PREREQUISITE_NOT_REACHED"
    assert receipt["content_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )


def test_pure_matrix_rejects_a_missing_boundary_instead_of_hashing_partial_evidence():
    """Catches a partial matrix becoming a content-bound CONTROL_PASS."""
    with pytest.raises(ValueError, match="one row per required boundary"):
        evaluate_survival_matrix({
            "schema": "campaignx.first_letters_stage_survival.v1",
            "stages": [{"boundary": "P0", "terminal_state": "PASS"}],
        })


def test_complete_control_emits_hash_bound_survival_matrix_without_an_acceptance_claim():
    """Synthetic receipts catch dropping a boundary or deriving an unbound PASS."""
    panel = ScriptedPanel(manifest())
    receipt = run(panel)
    assert receipt["schema"] == "campaignx.first_letters_stage_survival.v1"
    assert receipt["control_state"] == "CONTROL_PASS"
    assert receipt["stages"][2]["reason_code"] == \
        "PASS_ALREADY_CERTIFIED_AT_FINALIZATION"
    assert not any(path == "/api/geometry/certify" for _, path, _ in panel.calls)
    assert [row["boundary"] for row in receipt["stages"]] == [
        "P0", "P1", "P2", "QC", "P3", "P4", "P5", "P7", "HUMAN_REVIEW"
    ]
    assert all(row["terminal_state"] == "PASS" for row in receipt["stages"])
    assert receipt["bindings"]["deployed_revision"] == REVISION
    assert receipt["bindings"]["source_locks_sha256"] == canonical_sha256(manifest()["source_locks"])
    expected_discovery = discovery()
    expected_discovery["resource_identity"] = {
        **expected_discovery["resource_identity"],
        "p0_artifact_id": "p0-pherc0139",
        "p0_artifact_sha256": "a" * 64,
        "source_snapshot_id": "control-source",
    }
    assert receipt["stages"][1]["output_hashes"] == {
        "ct_read_set_manifest_sha256": expected_discovery["ct_read_set"]["canonical_manifest_sha256"],
        "m7_read_set_manifest_sha256": expected_discovery["m7_read_set"]["canonical_manifest_sha256"],
        "m7_provider_request_sha256": "f" * 64,
        "m7_provider_response_sha256": "0" * 64,
        "surface_read_set_manifest_sha256": canonical_sha256(
            [{"object_key": "meta.json", "sha256": "a" * 64, "bytes": 11}]),
        "surface_sha256": "3" * 64,
        "discovery_receipt_sha256": "2" * 64,
        "preflight_receipt_sha256": canonical_sha256(expected_discovery),
    }
    assert receipt["stages"][1]["parameters"] == expected_discovery["parameters"]
    assert receipt["stages"][1]["counts"] == {
        **expected_discovery["counts"],
        "closest_survivor_distance_ct_l0_voxels": 1.25,
        "planned_region_count": 12,
        "visited_region_count": 4,
        "maximum_surface_probe_regions": 4096,
        "coverage_fraction": 1 / 3,
        "coverage_complete": True,
        "surface_count": 1,
    }
    assert receipt["allow_unvalidated"] is False
    assert receipt["control_pass_is_independent_validation"] is False
    assert receipt["automatic_letter_acceptance"] is False
    unhashed = {k: v for k, v in receipt.items() if k != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(unhashed)
    assert all(row["elapsed_seconds"] >= 0 for row in receipt["stages"])
    assert receipt["stages"][5]["output_hashes"]["orientation_receipt_sha256"] == \
        orientation_result(manifest())["receipt_sha256"]


def test_unproven_orientation_stops_before_p4_is_enqueued():
    document = manifest()
    unproven = orientation_result(
        document, status="UNPROVEN",
        reason_code="ABSOLUTE_ORIENTATION_EVIDENCE_MISSING",
        parity_state="NOT_RUN_ABSOLUTE_LOCK_MISSING",
        selected_flip_normals=None,
        lineage={
            **orientation_result(document)["lineage"],
            "absolute_orientation": copy.deepcopy(
                document["checks"]["PIPELINE_CONTROL"]["orientation_parity"][
                    "absolute_orientation"]),
        },
    )
    assert unproven["lineage"]["absolute_orientation"]["verified"] is False
    panel = ScriptedPanel(document, overrides={"orientation": unproven})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "P4"
    assert receipt["stages"][5]["reason_code"] == "ORIENTATION_UNPROVEN"
    assert not any(body and body.get("phase") == "P4" for _, _, body in panel.calls)


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"counts": {"raw_m7": 0, "post_ct": 0, "post_clearance": 0, "packet_limited": 0}}, "ZERO_RAW_M7_CANDIDATES"),
        ({"counts": {"raw_m7": 3, "post_ct": 0, "post_clearance": 0, "packet_limited": 0}}, "ZERO_POST_CT_CANDIDATES"),
        ({"counts": {"raw_m7": 3, "post_ct": 2, "post_clearance": 0, "packet_limited": 0}}, "ZERO_POST_CLEARANCE_CANDIDATES"),
    ],
)
def test_discovery_rejection_stops_before_the_manual_pipeline(changed, reason):
    """Catches growth beginning after M7, CT, or clearance rejected discovery."""
    panel = ScriptedPanel(manifest(), overrides={"discovery": discovery(**changed)})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_FAILED"
    assert receipt["first_nonpassing_boundary"] == "P1"
    assert receipt["stages"][1]["reason_code"] == reason
    assert_stopped_before(panel, "/api/segmentation/manual-seeds")


def test_discovery_without_exact_frozen_root_objects_stops_before_growth():
    incomplete = discovery()
    incomplete["m7_read_set"] = {
        "schema": "campaignx.first_letters_source_read_set.v1",
        "objects": [{"object_key": "0/1/2/3", "sha256": "d" * 64, "bytes": 17}],
        "canonical_manifest_sha256": canonical_sha256(
            [{"object_key": "0/1/2/3", "sha256": "d" * 64, "bytes": 17}]
        ),
    }
    panel = ScriptedPanel(manifest(), overrides={"discovery": incomplete})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "P1"
    assert receipt["stages"][1]["reason_code"] == "MISSING_CONTENT_BOUND_DISCOVERY_EVIDENCE"
    assert_stopped_before(panel, "/api/segmentation/manual-seeds")


# -- a refusal is a boundary result, not a traceback ---------------------------
#
# P3 through HUMAN_REVIEW caught AmbiguousMutationError and TimeoutError and
# nothing else, so any 4xx from the platform escaped as an exception: the runner
# died, wrote no receipt, and published nothing. It happened on the deployment
# the first time a control ever reached P3 --
#
#   POST /api/flattening/run -> 400
#   "PHerc0139 is not a scroll the frozen catalog registers"
#
# -- and the only account of four crossed boundaries was a traceback in a
# container's log. A control that cannot say where it stopped is not evidence.

@pytest.mark.parametrize(
    ("key", "row", "path"),
    [
        ("p3", 4, "/api/flattening/run"),
        ("p4", 5, "/api/jobs"),
        ("p5", 6, "/api/jobs"),
        ("p7", 7, "/api/jobs"),
    ],
)
def test_a_refused_phase_is_recorded_rather_than_raised(key, row, path):
    from panel_client import PanelError

    refused = PanelError("POST", path, 400,
                         '{"detail":"PHerc0139 is not a scroll the frozen '
                         'catalog registers"}')
    panel = ScriptedPanel(manifest(), overrides={key: refused})

    receipt = run(panel)

    assert receipt["control_state"] in {"CONTROL_INCOMPLETE", "CONTROL_FAILED"}
    assert receipt["stages"][row]["reason_code"], (
        f"{key} was refused and the matrix says nothing about it")
    assert receipt["content_sha256"], "a refused run still owes a receipt"


def test_the_refusal_reason_reaches_the_matrix():
    """An operator reading the matrix should not have to open a container log."""
    from panel_client import PanelError

    refused = PanelError("POST", "/api/flattening/run", 400,
                         '{"detail":"PHerc0139 is not a scroll the frozen '
                         'catalog registers"}')
    receipt = run(ScriptedPanel(manifest(), overrides={"p3": refused}))

    row = receipt["stages"][4]
    recorded = json.dumps(row)
    assert "400" in recorded or "REFUSED" in row["reason_code"], row


@pytest.mark.parametrize(
    ("pipeline_change", "reason"),
    [
        # GROW_FAILED, not FAILED: the fleet writes the first and never the
        # second, and asserting against a state that cannot occur is how a real
        # failed grow came to hang until the timeout instead of being reported.
        ({"state": "GROW_FAILED", "canonical_full_grow": False}, "GROW_REJECTED"),
        ({"area_cm2": 0.09}, "TINY_SURFACE_DIAGNOSTIC_ONLY"),
    ],
)
def test_pipeline_growth_failure_never_advances_to_geometry(pipeline_change, reason):
    """Catches geometry queueing after a failed or sub-0.10 cm2 grow."""
    doc = manifest()
    panel = ScriptedPanel(doc, overrides={"pipeline": pipeline(**pipeline_change)})
    receipt = run(panel)
    assert receipt["stages"][1]["reason_code"] == reason
    assert_stopped_before(panel, "/api/geometry/certify")


def test_geometry_rejection_never_advances_to_flattening():
    """Catches P3 queueing for a geometry-rejected surface."""
    panel = ScriptedPanel(manifest(), overrides={
        "p2": p2_result(geometry_qc_state="GEOMETRY_REJECTED_COVERAGE")
    })
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P2"
    assert_stopped_before(panel, "/api/flattening/run")


def test_physical_qc_rejection_never_advances_to_flattening():
    """Catches treating terminal incompatible physical QC as downstream-ready."""
    panel = ScriptedPanel(manifest(), overrides={
        "p2": p2_result(physical_qc_state="REJECTED")
    })
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "QC"
    assert receipt["stages"][3]["reason_code"] == "PHYSICAL_QC_TERMINAL_INCOMPATIBLE"
    assert_stopped_before(panel, "/api/flattening/run")


def test_ct_supported_is_the_real_downstream_compatible_physical_state():
    panel = ScriptedPanel(manifest(), overrides={
        "p2": p2_result(physical_qc_state="CT_SUPPORTED")
    })
    receipt = run(panel)
    assert receipt["stages"][3]["terminal_state"] == "PASS"
    assert receipt["control_state"] == "CONTROL_PASS"


def test_pending_physical_qc_times_out_incomplete_and_never_flattens():
    panel = ScriptedPanel(manifest(), overrides={
        "p2": p2_result(physical_qc_state="PENDING"),
        "qc_jobs": [{"qc_job_id": "qc-control", "surface_id": "surface-control",
                     "state": "RUNNING", "result_sha256": "4" * 64}],
    })
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "QC"
    assert receipt["stages"][3]["reason_code"] == "PHYSICAL_QC_NONTERMINAL"
    assert_stopped_before(panel, "/api/flattening/run")


def test_finalizer_geometry_evidence_must_belong_to_the_exact_source_attempt():
    stale = p2_result(source_attempt_id="older-attempt")
    panel = ScriptedPanel(manifest(), overrides={"p2": stale})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "P2"
    assert receipt["stages"][2]["reason_code"] == "P2_CURRENT_JOB_EVIDENCE_MISSING"
    assert_stopped_before(panel, "/api/flattening/run")


def test_physical_qc_must_be_causally_unblocked_by_the_exact_p2_job():
    stale_qc = copy.deepcopy(ScriptedPanel(manifest()).responses["qc_jobs"])
    stale_qc[0]["unblocked_by_job_id"] = "older-p2-job"
    panel = ScriptedPanel(manifest(), overrides={"qc_jobs": stale_qc})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "QC"
    assert receipt["stages"][3]["reason_code"] == "PHYSICAL_QC_CURRENT_JOB_EVIDENCE_MISSING"
    assert_stopped_before(panel, "/api/flattening/run")


def test_missing_flattened_artifact_is_incomplete_and_never_renders():
    """Catches equating a successful P3 process with a materialized artifact."""
    panel = ScriptedPanel(manifest(), overrides={
        "p3": p3_result(artifact_sha256=None)
    })
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][4]["reason_code"] == "MISSING_HASH_BOUND_FLATTENED_ARTIFACT"
    assert not any(body and body.get("phase") == "P4" for _, _, body in panel.calls)


def test_p3_success_from_an_older_job_cannot_satisfy_the_current_flatten_request():
    stale = p3_result(requested_by_job_id="older-p3-job")
    stale["receipt"] = {**stale["receipt"], "requested_by_job_id": "older-p3-job"}
    stale["receipt_sha256"] = canonical_sha256(stale["receipt"])
    panel = ScriptedPanel(manifest(), overrides={"p3": stale})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["first_nonpassing_boundary"] == "P3"
    assert receipt["stages"][4]["reason_code"] == "P3_CURRENT_JOB_EVIDENCE_MISSING"
    assert not any(body and body.get("phase") == "P4" for _, _, body in panel.calls)


def test_incomplete_render_manifest_never_runs_p5():
    """Catches P5 consuming a layer stack whose byte manifest is incomplete."""
    panel = ScriptedPanel(manifest(), overrides={
        "p4": p4_result(layer_stack={"files": 0})
    })
    receipt = run(panel)
    assert receipt["stages"][5]["reason_code"] == "INCOMPLETE_RENDER_MANIFEST"
    assert not any(body and body.get("phase") == "P5" for _, _, body in panel.calls)


@pytest.mark.parametrize("keys", [
    [f"{index}.tif" for index in range(32)],
    [f"{index}.tif" for index in range(1, 34)],
    [*(f"{index}.tif" for index in range(33) if index != 17), "33.tif"],
    [*(f"{index}.tif" for index in range(32)), "slice.tif"],
    [*(f"{index}.tif" for index in range(32)), "00.tif"],
])
def test_p4_inventory_must_bind_exact_numeric_indices_zero_through_32(keys):
    forged = p4_result()
    forged["layer_stack"]["objects"] = [
        {"object_key": key, "sha256": "8" * 64, "bytes": 42} for key in keys]
    forged["layer_stack"]["files"] = len(keys)
    forged["layer_stack"]["artifact_sha256"] = canonical_sha256({
        row["object_key"]: {"sha256": row["sha256"], "size_bytes": row["bytes"]}
        for row in forged["layer_stack"]["objects"]})
    forged["layers"]["slices"] = len(keys)
    panel = ScriptedPanel(manifest(), overrides={"p4": forged})
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P4"
    assert receipt["stages"][5]["reason_code"] == "INCOMPLETE_RENDER_MANIFEST"
    assert not any(body and body.get("phase") == "P5" for _, _, body in panel.calls)


def test_render_inventory_digest_is_recomputed_before_p5():
    forged = p4_result()
    forged["layer_stack"]["artifact_sha256"] = "7" * 64
    panel = ScriptedPanel(manifest(), overrides={"p4": forged})
    receipt = run(panel)
    assert receipt["stages"][5]["reason_code"] == "INCOMPLETE_RENDER_MANIFEST"
    assert not any(body and body.get("phase") == "P5" for _, _, body in panel.calls)


def test_unproven_p3_to_p4_lateral_metric_never_runs_p5():
    render = p4_result()
    render["lateral_metric"] = {
        **render["lateral_metric"], "status": "UNPROVEN",
        "reason_code": "MISSING_RASTER_TRANSFORM",
    }
    render["lateral_metric"]["receipt_sha256"] = canonical_sha256({
        key: value for key, value in render["lateral_metric"].items()
        if key != "receipt_sha256"})
    panel = ScriptedPanel(manifest(), overrides={"p4": render})
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P5"
    assert receipt["stages"][6]["reason_code"] == "P5_LATERAL_METRIC_UNPROVEN"
    assert not any(body and body.get("phase") == "P5" for _, _, body in panel.calls)


@pytest.mark.parametrize(("field", "value"), [
    ("source_slice_um", 18.724),
    ("source_pixel_um", 9.362),
    ("p4_job_id", "older-p4-job"),
    ("p4_layer_artifact_sha256", "7" * 64),
    ("training_pixel_um", 9.362),
])
def test_p5_normalization_must_bind_metric_depth_model_and_exact_p4(field, value):
    screening = p5_result()
    screening["physical_normalization"][field] = value
    screening["physical_normalization"]["receipt_sha256"] = canonical_sha256({
        key: item for key, item in screening["physical_normalization"].items()
        if key != "receipt_sha256"})
    panel = ScriptedPanel(manifest(), overrides={"p5": screening})
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P5"
    assert receipt["stages"][6]["reason_code"] == "P5_INPUT_OR_MODEL_LINEAGE_MISMATCH"
    assert not any(body and body.get("phase") == "P7" for _, _, body in panel.calls)


def test_p5_normalization_requires_the_complete_33_file_source_manifest():
    screening = p5_result()
    screening["physical_normalization"]["source_layer_objects"] = \
        screening["physical_normalization"]["source_layer_objects"][:-1]
    screening["physical_normalization"]["receipt_sha256"] = canonical_sha256({
        key: item for key, item in screening["physical_normalization"].items()
        if key != "receipt_sha256"})
    panel = ScriptedPanel(manifest(), overrides={"p5": screening})
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P5"
    assert receipt["stages"][6]["reason_code"] == "P5_INPUT_OR_MODEL_LINEAGE_MISMATCH"
    assert not any(body and body.get("phase") == "P7" for _, _, body in panel.calls)


def test_dead_p5_map_never_runs_p7():
    """Catches execution success being mistaken for an ALIVE probability map."""
    panel = ScriptedPanel(manifest(), overrides={
        "p5": p5_result(liveness={"verdict": "DEAD"})
    })
    receipt = run(panel)
    assert receipt["stages"][6]["reason_code"] == "P5_DEGENERATE_OR_EMPTY"
    assert not any(body and body.get("phase") == "P7" for _, _, body in panel.calls)


def test_probability_inventory_digest_is_recomputed_before_p7():
    forged = p5_result()
    forged["probability_map"]["artifact_sha256"] = "7" * 64
    panel = ScriptedPanel(manifest(), overrides={"p5": forged})
    receipt = run(panel)
    assert receipt["stages"][6]["reason_code"] == "P5_DEGENERATE_OR_EMPTY"
    assert not any(body and body.get("phase") == "P7" for _, _, body in panel.calls)


def test_real_unverified_roi_lock_stops_before_p7_is_enqueued():
    document = manifest()
    unproven = roi_result(document, status="UNPROVEN",
                          reason_code="POSITIVE_CONTROL_ROI_EVIDENCE_MISSING",
                          transformed_bbox_xyxy=None,
                          verified_training_pixel_um=None,
                          lock=copy.deepcopy(document["checks"]["PIPELINE_CONTROL"][
                              "positive_control_roi"]))
    assert unproven["lock"]["verified"] is False
    panel = ScriptedPanel(document, overrides={"roi": unproven})
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P7"
    assert receipt["stages"][7]["reason_code"] == "P7_ROI_UNPROVEN"
    assert not any(body and body.get("phase") == "P7" for _, _, body in panel.calls)


@pytest.mark.parametrize("change", [
    {"transformed_bbox_xyxy": [0, 0, 340, 364]},
    {"transformed_bbox_xyxy": [20, 30, 121, 130]},
    {"verified_training_pixel_um": 9.362},
    {"lock": {"verified": True}},
])
def test_p7_rejects_full_map_cropped_synthetic_or_unverified_roi(change):
    proof = roi_result(manifest(), **change)
    panel = ScriptedPanel(manifest(), overrides={"roi": proof})
    receipt = run(panel)
    assert receipt["first_nonpassing_boundary"] == "P7"
    assert receipt["stages"][7]["reason_code"] == "P7_ROI_UNPROVEN"
    assert not any(body and body.get("phase") == "P7" for _, _, body in panel.calls)


def test_negative_p7_routing_never_routes_human_review():
    """Catches a contradictory P7 route passing because its job exited zero."""
    panel = ScriptedPanel(manifest(), overrides={
        "p7": p7_result(adjudication={"verdict": "FAIL"})
    })
    receipt = run(panel)
    assert receipt["stages"][7]["reason_code"] == "P7_ROUTE_CONTRADICTS_KNOWN_POSITIVE"
    assert_stopped_before(panel, "/api/jobs/p7-job/review")


def test_human_review_ignores_older_lookalike_and_posts_only_intent_and_note():
    old = human_result(review_event_id="old-review", p7_job_id="older-p7-job",
                       event_sha256="f" * 64)
    old["event_sha256"] = canonical_sha256({
        key: value for key, value in old.items() if key != "event_sha256"})
    panel = ScriptedPanel(manifest(), overrides={"initial_reviews": [old]})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_PASS"
    review_posts = [body for method, path, body in panel.calls
                    if method == "POST" and path == "/api/jobs/p7-job/review"]
    assert review_posts == [{
        "verdict": "INSPECT",
        "note": "First Letters positive-control packet routed for human inspection",
    }]
    assert receipt["stages"][8]["output_hashes"][
        "human_review_record_sha256"] == human_result()["event_sha256"]


def test_legacy_mutable_surface_review_cannot_satisfy_exact_job_control():
    class LegacyOnlyPanel(ScriptedPanel):
        def call(self, method: str, path: str, body: dict | None = None):
            if method == "GET" and path == "/api/jobs/p7-job/review":
                self.calls.append((method, path, copy.deepcopy(body)))
                return {"human_reviews": []}
            if method == "POST" and path == "/api/jobs/p7-job/review":
                self.calls.append((method, path, copy.deepcopy(body)))
                return {
                    "verdict": "INSPECT", "surface_id": "surface-control",
                    "vetting_packet_sha256": "b" * 64,
                    "p7_job_id": "p7-job",
                }
            return super().call(method, path, body)

    panel = LegacyOnlyPanel(manifest())
    panel.reviewed = True
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][8]["reason_code"] == "HUMAN_REVIEW_ROUTING_MISSING"


def test_ambiguous_post_is_not_retried_and_no_later_stage_runs():
    """Catches automatic mutation retry after the server may have committed it."""
    failure = AmbiguousMutationError("POST", "/api/segmentation/manual-seeds", "connection closed")
    panel = ScriptedPanel(manifest(), overrides={"manual": failure})
    receipt = run(panel)
    manual_calls = [call for call in panel.calls if call[1] == "/api/segmentation/manual-seeds"]
    assert len(manual_calls) == 1
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][1]["reason_code"] == "AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK"
    assert_stopped_before(panel, "/api/geometry/certify")


def test_post_preflight_p0_drift_stops_before_accepting_manual_growth():
    panel = ScriptedPanel(manifest(), overrides={"manual": {
        "inserted": 1, "submitted_by": "helena-scientist",
        "receipt": {"seed_origin": "human"},
        "resource_identity": {
            "p0_artifact_id": "replacement", "p0_artifact_sha256": "f" * 64,
            "source_snapshot_id": "replacement-source",
        },
    }})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][1]["reason_code"] == "MANUAL_SEED_PROVENANCE_MISSING"
    assert_stopped_before(panel, "/api/geometry/certify")


def test_discovery_request_uses_the_locked_surface_not_the_manual_anchor_as_region():
    panel = ScriptedPanel(manifest(), overrides={
        "discovery": discovery(counts={"raw_m7": 0, "post_ct": 0,
                                       "post_clearance": 0, "packet_limited": 0})})
    run(panel)
    body = next(body for method, path, body in panel.calls
                if method == "POST" and path == "/api/segmentation/preflight")
    assert body["control_surface"]["uri"] == manifest()["source_locks"]["community_surface"]["uri"]
    assert body["control_surface"]["artifacts"] == manifest()["source_locks"]["community_surface"]["artifacts"]


def test_runtime_binding_drift_marks_the_receipt_stale_without_mutation():
    """Catches reuse of a control receipt across deployed revisions or inputs."""
    doc = manifest()
    drifted = runtime(doc)
    drifted["deployed_revision"] = "2" * 40
    panel = ScriptedPanel(doc, overrides={"runtime": drifted})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][0]["reason_code"] == "CONTROL_INCOMPLETE_STALE"
    assert_stopped_before(panel, "/api/segmentation/preflight")


def test_control_manifest_hash_drift_marks_the_receipt_stale_without_mutation():
    """Catches accepting a different control profile merely because its hash is shaped correctly."""
    doc = manifest()
    drifted = runtime(doc)
    drifted["control_profile_sha256"] = "f" * 64
    panel = ScriptedPanel(doc, overrides={"runtime": drifted})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][0]["reason_code"] == "CONTROL_INCOMPLETE_STALE"
    assert_stopped_before(panel, "/api/segmentation/preflight")


def test_unverified_actual_profile_bytes_make_runtime_stale_without_mutation():
    doc = manifest()
    drifted = runtime(doc)
    drifted["profile_locks"][0] = {
        **drifted["profile_locks"][0],
        "actual_sha256": "f" * 64,
        "actual_file_sha256": "f" * 64,
        "verified": False,
    }
    drifted["profile_locks_verified"] = False
    panel = ScriptedPanel(doc, overrides={"runtime": drifted})
    receipt = run(panel)
    assert receipt["control_state"] == "CONTROL_INCOMPLETE"
    assert receipt["stages"][0]["reason_code"] == "CONTROL_INCOMPLETE_STALE"
    assert_stopped_before(panel, "/api/segmentation/preflight")


def test_manual_control_seed_preserves_fractional_coordinate_and_control_non_bypass():
    """Catches rounding the preregistered seed or losing control-only safety metadata."""
    class NoSurfaces:
        def surfaces_for_snapshot(self, _identifier):
            return []

    point = {"x": 3079.062744140625, "y": 3961.3037109375, "z": 4441.35595703125}
    task = generate_manual_tasks(
        NoSurfaces(),
        {"source_snapshot_id": "src", "sample_id": "PHerc0139", "shape_xyz": [6621, 6621, 20974], "m7_uri": "m7"},
        [point],
        catalog_snapshot_sha256="d" * 64,
        grid_step=2048,
        query_radius=64,
        volume_edge_margin=64,
        grid_version="first-letters-control-manual-v1",
        policy_version=IN_FORCE_ID,
        submitted_by="helena-scientist",
    )[0]
    assert task["manual_candidates"][0]["ct_l0_coordinate"] == point
    assert task["manual_candidates"][0]["seed_origin"] == "human"
    assert task["first_letters_control"] == {
        "check": "PIPELINE_CONTROL",
        "profile_id": IN_FORCE_ID,
        "seed_origin": "human",
        "allow_unvalidated": False,
    }
    assert task["ink_used"] is False


def test_integer_manual_seed_keeps_its_historical_candidate_identity():
    """Catches exact-coordinate support duplicating every previously integer-valued seed."""
    class NoSurfaces:
        def surfaces_for_snapshot(self, _identifier):
            return []

    task = generate_manual_tasks(
        NoSurfaces(),
        {"source_snapshot_id": "src", "sample_id": "PHerc0139", "shape_xyz": [8000, 8000, 20000], "m7_uri": "m7"},
        [{"x": 4000, "y": 4000, "z": 10000}],
        catalog_snapshot_sha256="d" * 64,
        grid_step=2048, query_radius=64, volume_edge_margin=64,
        grid_version="ct-l0-manual-v1", policy_version="ink-blind-v1",
        submitted_by="helena-scientist",
    )[0]
    historical = "manual-" + stable_id(
        "manual-seed",
        {"source_snapshot_id": "src", "x": 4000, "y": 4000, "z": 10000},
    )[:16]
    assert task["manual_candidates"][0]["candidate_id"] == historical
