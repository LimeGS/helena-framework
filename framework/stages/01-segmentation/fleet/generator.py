from __future__ import annotations

import copy
import hashlib
import heapq
import itertools
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .common import content_sha256, file_sha256, read_json, stable_id, utc_now
from .seed_probe import (
    BENCHMARK_AUTHORIZATION_SCHEMA,
    BENCHMARK_FULL_GROW_ENVELOPE_SHA256,
    build_seed_probe_benchmark_execution,
    normalize_seed_probe_benchmark_execution_authorization,
    normalize_seed_probe_policy,
    normalize_source_content_lock,
)
from .store import FleetStore


DEFAULT_ENVELOPE = {
    "profile_ids": ["vc3d-m7-growth-v1"],
    "parameters": {
        "generations": {"type": "integer", "minimum": 20, "maximum": 45, "default": 35},
        "step_size": {"type": "integer", "minimum": 12, "maximum": 24, "default": 20},
        "min_area_cm": {"type": "number", "minimum": 0.0, "maximum": 0.0, "default": 0.0},
        "use_cuda": {"type": "boolean", "const": False, "default": False},
    },
    "maximum_candidate_count": 8,
    "ink_used": False,
}

# Resume-compatible full growth for a seed-probe winner.  The probe fixes every
# topology-affecting parameter; Cost-Aware remains the outer router and may
# still choose the absolute final generation target inside this envelope.
SEED_PROBE_CONTINUATION_ENVELOPE = {
    "profile_ids": ["vc3d-m7-growth-v2"],
    "parameters": {
        "generations": {
            "type": "integer",
            "minimum": 20,
            "maximum": 45,
            "default": 35,
        },
        "step_size": {
            "type": "integer",
            "const": 12,
            "default": 12,
        },
        "min_area_cm": {
            "type": "number",
            "const": 0.0,
            "default": 0.0,
        },
        "use_cuda": {
            "type": "boolean",
            "const": False,
            "default": False,
        },
        "inpaint": {
            "type": "boolean",
            "const": False,
            "default": False,
        },
        "skip_overlap_check": {
            "type": "boolean",
            "const": False,
            "default": False,
        },
    },
    "maximum_candidate_count": 8,
    "ink_used": False,
}


def probe_uri(uri: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """Whether the thing a snapshot names can actually be read.

    P0 froze ct_uri and m7_uri out of the catalog and never looked at them, so
    seven tasks were claimed, sent to a worker and died on HTTP 401 a week after
    intake -- one attempt burned each. A HEAD costs a second and moves that from
    fleet time to intake time, where it is one message instead of seven receipts.

    A zarr URI names a directory, so the probe reads its .zattrs: a HEAD on the
    directory itself answers 404 on a store that is perfectly readable.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    target = uri.rstrip("/") + "/.zattrs" if uri.rstrip("/").endswith(".zarr") else uri
    if not target.startswith(("http://", "https://")):
        # s3:// and fixture:// are read by the worker's own client, and an
        # unauthenticated probe here would report a false negative.
        return {"uri": uri, "checked": False, "reason": "not an http(s) URI"}
    request = urllib.request.Request(target, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"uri": uri, "checked": True, "reachable": True,
                    "status": int(response.status)}
    except urllib.error.HTTPError as failure:
        return {"uri": uri, "checked": True, "reachable": False,
                "status": int(failure.code), "error": str(failure)}
    except Exception as failure:  # noqa: BLE001 -- DNS, TLS, timeouts
        return {"uri": uri, "checked": True, "reachable": False,
                "error": f"{type(failure).__name__}: {failure}"}


def bootstrap_sources(store: FleetStore, eligible_path: Path,
                      samples: set[str] | None = None,
                      verify: bool = True) -> dict[str, str]:
    eligible = read_json(eligible_path)
    entries = eligible["entries"] if isinstance(eligible, dict) else eligible
    manifest_sha = file_sha256(eligible_path)
    result: dict[str, str] = {}
    for entry in sorted(entries, key=lambda row: str(row["sample_id"])):
        sample_id = str(entry["sample_id"])
        if samples and sample_id not in samples:
            continue
        shape_zyx = [int(value) for value in entry["shape_zyx"]]
        payload = {
            "schema": "campaignx.source_snapshot.v1",
            "sample_id": sample_id,
            "ct_uri": entry["ct_uri"],
            "ct_sha256": entry.get("ct_sha256"),
            "m7_uri": entry["surface_prediction_uri"],
            "m7_sha256": entry.get("surface_prediction_sha256"),
            "shape_xyz": [shape_zyx[2], shape_zyx[1], shape_zyx[0]],
            "voxel_size_um": float(entry["voxel_size_um"]),
            "coordinate_frame": "ct_l0_xyz",
            # The threshold the prediction was published at, which decides which
            # m7 voxels count as sheet and therefore which seeds get proposed. It
            # sat in the catalogue and reached nothing: seed_candidates carried
            # 0.2 as a constant with a comment saying reading it would be better.
            # A catalogue that published 0.3 would have been grown at 0.2, in
            # silence, under the same task identity.
            "m7_threshold": (float(entry["surface_prediction_threshold"])
                             if entry.get("surface_prediction_threshold") is not None
                             else None),
            "eligible_manifest_sha256": manifest_sha,
            "source_status": (
                "URI_LOCKED_HASH_UNAVAILABLE"
                if not entry.get("ct_sha256")
                or not entry.get("surface_prediction_sha256")
                else "HASH_LOCKED_NOT_IMMUTABLE"
            ),
        }
        if entry.get("source_content_lock") is not None:
            payload["source_content_lock"] = entry["source_content_lock"]
            payload["source_content_lock"] = normalize_source_content_lock(payload)
            payload["source_status"] = "IMMUTABLE_CONTENT_LOCKED"
        if verify:
            # On the snapshot, so a task queued a month later can be traced to
            # what the source looked like when it was frozen.
            payload["source_reachable"] = {
                "checked_at_utc": utc_now(),
                "ct": probe_uri(str(entry["ct_uri"])),
                "m7": probe_uri(str(entry["surface_prediction_uri"])),
            }
        # A snapshot is immutable -- register_snapshot does nothing on conflict --
        # so a catalogue that changes the published threshold would leave the old
        # value in place and grow at it while the catalogue said otherwise. That is
        # the same silence the threshold already had, moved one layer down.
        #
        # Not by putting the threshold into the snapshot's identity: that identity
        # is hashed into source_snapshot_id, which is part of task identity, so
        # adding a field would give every existing source a new id and re-queue the
        # whole backlog as new work. Refusing says the same thing and touches
        # nothing.
        existing = [row for row in store.snapshots({sample_id})
                    if row.get("ct_uri") == payload["ct_uri"]
                    and row.get("m7_uri") == payload["m7_uri"]]
        for row in existing:
            was = row.get("m7_threshold")
            now = payload["m7_threshold"]
            if was is not None and now is not None and float(was) != float(now):
                raise ValueError(
                    f"{sample_id}: the catalogue now publishes m7 at {now} and the "
                    f"registered source was frozen at {was}. A different threshold "
                    "is a different set of seeds, so this needs a new source rather "
                    "than a silent change under the old one.")
        result[sample_id] = store.register_snapshot(payload)
    if samples and set(result) != samples:
        raise ValueError(f"unknown samples requested: {sorted(samples - set(result))}")
    return result


def _catalog_rows(catalog: Path) -> Iterable[dict[str, Any]]:
    if catalog.suffix == ".sqlite":
        connection = sqlite3.connect(catalog)
        connection.row_factory = sqlite3.Row
        try:
            for row in connection.execute("SELECT row_json FROM campaign_surfaces ORDER BY surface_id"):
                yield json.loads(row["row_json"])
            for row in connection.execute("SELECT row_json FROM public_surfaces ORDER BY public_surface_id"):
                value = json.loads(row["row_json"])
                value.setdefault("owner", "public")
                value.setdefault("surface_id", value.get("public_surface_id"))
                value.setdefault("sample_id", value.get("campaign_sample_id"))
                value.setdefault("bbox_l0_xyz", value.get("bbox_xyz"))
                yield value
        finally:
            connection.close()
        return
    value = read_json(catalog)
    for row in value.get("campaign_surfaces", []):
        yield row
    for row in value.get("public_surfaces", []):
        yield row


def import_catalog(store: FleetStore, catalog: Path, sources_by_sample: dict[str, str]) -> dict[str, int]:
    counts = {"imported": 0, "skipped_unknown_source": 0, "missing_bbox": 0}
    for row in _catalog_rows(catalog):
        sample_id = str(row.get("sample_id") or row.get("campaign_sample_id") or "")
        if sample_id not in sources_by_sample:
            counts["skipped_unknown_source"] += 1
            continue
        bbox = row.get("bbox_l0_xyz") or row.get("bbox_xyz")
        if not (isinstance(bbox, list) and len(bbox) == 2):
            counts["missing_bbox"] += 1
            continue
        artifact_hash = row.get("tifxyz_sha256") or row.get("artifact_sha256")
        payload = {
            "surface_id": str(row.get("surface_id") or row.get("public_surface_id")),
            "source_snapshot_id": sources_by_sample[sample_id],
            "sample_id": sample_id,
            "owner": row.get("owner", "imported"),
            "artifact_sha256": artifact_hash,
            "artifact_uri": row.get("artifact_uri") or row.get("archive_relative"),
            "bbox_xyz": bbox,
            "area_cm2": row.get("area_cm2"),
            "state": row.get("state", "IMPORTED_COVERAGE"),
            "physical_qc_state": row.get("physical_qc_state", "UNVALIDATED"),
            "source_catalog": str(catalog),
        }
        store.import_surface(payload)
        counts["imported"] += 1
    return counts


def point_bbox_gap(point: list[float], bbox: list[list[float]]) -> float:
    return math.sqrt(sum(max(float(low) - value, 0.0, value - float(high)) ** 2 for value, low, high in zip(point, bbox[0], bbox[1], strict=True)))


def _axis_centers(size: int, radius: int, step: int, volume_edge_margin: int) -> list[int]:
    """Return deterministic cell centres while reserving a source-volume rim.

    `query_radius` protects the MCP query cube itself.  The optional larger
    `volume_edge_margin` is a quality policy: early autonomous grows should not
    spend a VC3D attempt on an m7 response sitting against the scanned-volume
    boundary, where a short surface is much more likely to be truncated.
    """
    margin = max(radius, volume_edge_margin)
    if size <= margin * 2:
        return []
    centers = list(range(margin, size - margin, step))
    last = size - margin - 1
    if centers and last - centers[-1] >= step // 2:
        centers.append(last)
    return centers


def generate_tasks_for_snapshot(
    store: FleetStore,
    snapshot: dict[str, Any],
    *,
    catalog_snapshot_sha256: str,
    grid_step: int,
    query_radius: int,
    clearance: float,
    volume_edge_margin: int,
    candidate_interior_clearance: int,
    selection_strategy: str,
    max_tasks: int,
    grid_version: str,
    policy_version: str,
    parameter_envelope: dict[str, Any] | None = None,
    candidate_selection_policy: str = "score-cell-volume-clearance-v1",
    seed_region_policy: str = "fixed-v1",
    recenter_probe_max_candidates: int = 100,
    recenter_radius_xyz: dict[str, int] | None = None,
    ct_material_support_gate: dict[str, Any] | None = None,
    planner: str | None = None,
    planner_model: str | None = None,
    queued_reason: str | None = None,
    created_by: str | None = None,
    mission_id: str | None = None,
    p0_selection_version: str | None = None,
    p0_artifact_id: str | None = None,
    p0_artifact_sha256: str | None = None,
    p0_resolved_by: str | None = None,
    seed_probe: dict[str, Any] | None = None,
    benchmark_execution_authorization: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if grid_step < query_radius * 2:
        raise ValueError("grid_step must be at least twice query_radius")
    if max_tasks < 1:
        return []
    shape = [int(value) for value in snapshot["shape_xyz"]]
    if volume_edge_margin < query_radius:
        raise ValueError("volume_edge_margin must be at least query_radius")
    if candidate_interior_clearance < 0:
        raise ValueError("candidate_interior_clearance must be non-negative")
    if candidate_selection_policy not in {
        "score-cell-volume-clearance-v1",
        "adaptive-geometry-history-v2",
    }:
        raise ValueError("unsupported candidate_selection_policy")
    if seed_region_policy not in {
        "fixed-v1",
        "m7-recenter-z-v1",
        "m7-recenter-xyz-v1",
        "m7-recenter-z-chunk-safe-v1",
        "m7-chunk-safe-merge-interior-v2",
    }:
        raise ValueError("unsupported seed_region_policy")
    if recenter_probe_max_candidates < 1 or recenter_probe_max_candidates > 100:
        raise ValueError("recenter_probe_max_candidates must be 1..100")
    normalized_probe = (
        normalize_seed_probe_policy(seed_probe) if seed_probe is not None else None
    )
    if (
        normalized_probe is not None
        and normalized_probe["mode"] == "select"
        and (
            production_authorization := normalized_probe.get(
                "benchmark_authorization"
            )
        )
        is not None
        and production_authorization.get("schema")
        == BENCHMARK_AUTHORIZATION_SCHEMA
        and str(snapshot.get("sample_id") or "")
        not in production_authorization["authorized_sample_ids"]
    ):
        raise ValueError(
            "snapshot sample is outside the production-approved select cohort"
        )
    benchmark_authorization = (
        normalize_seed_probe_benchmark_execution_authorization(
            benchmark_execution_authorization
        )
        if benchmark_execution_authorization is not None
        else None
    )
    benchmark_arm: str | None = None
    if benchmark_authorization is not None:
        if planner != "deterministic-v2":
            raise ValueError(
                "isolated benchmark execution requires deterministic-v2"
            )
        if normalized_probe is None:
            benchmark_arm = "baseline"
            expected_policy = benchmark_authorization[
                "baseline_policy_version"
            ]
        elif normalized_probe["mode"] == "select":
            benchmark_arm = "closed_loop"
            expected_policy = benchmark_authorization["policy_version"]
            if (
                normalized_probe.get("benchmark_authorization")
                != benchmark_authorization
            ):
                raise ValueError(
                    "closed-loop benchmark policy lost its preregistered "
                    "authorization"
                )
        else:
            raise ValueError(
                "isolated causal benchmark supports only off and select arms"
            )
        if policy_version != expected_policy:
            raise ValueError(
                "benchmark task policy_version differs from its frozen arm"
            )
        if (
            parameter_envelope is not None
            and content_sha256(parameter_envelope)
            != BENCHMARK_FULL_GROW_ENVELOPE_SHA256
        ):
            raise ValueError(
                "benchmark arms must use the common resume-compatible envelope"
            )
    if normalized_probe is not None and normalized_probe["mode"] == "select":
        if planner not in {"cost-aware-v2", "deterministic-v2"}:
            raise ValueError(
                "seed probe select mode requires cost-aware-v2 or deterministic-v2"
            )
        normalize_source_content_lock(snapshot)
    recenter_radius = recenter_radius_xyz or {"x": 64, "y": 64, "z": 64}
    if any(axis not in recenter_radius or int(recenter_radius[axis]) < 1 or int(recenter_radius[axis]) > 192 for axis in "xyz"):
        raise ValueError("recenter_radius_xyz must define x/y/z values from 1..192")
    if ct_material_support_gate is not None:
        expected_gate_keys = {
            "policy",
            "level",
            "radius_l0_voxels",
            "minimum_nonzero_voxels",
        }
        if not isinstance(ct_material_support_gate, dict) or set(ct_material_support_gate) != expected_gate_keys:
            raise ValueError("ct_material_support_gate must contain the exact frozen gate keys")
        if ct_material_support_gate["policy"] != "ome-zarr-nearby-material-v1":
            raise ValueError("unsupported ct_material_support_gate policy")
        if int(ct_material_support_gate["level"]) < 0:
            raise ValueError("ct_material_support_gate level must be non-negative")
        if int(ct_material_support_gate["radius_l0_voxels"]) < 1:
            raise ValueError("ct_material_support_gate radius must be positive")
        if int(ct_material_support_gate["minimum_nonzero_voxels"]) < 1:
            raise ValueError("ct_material_support_gate minimum_nonzero_voxels must be positive")
    axes = [_axis_centers(size, query_radius, grid_step, volume_edge_margin) for size in shape]
    if any(not axis for axis in axes):
        if benchmark_authorization is not None:
            raise ValueError(
                "preregistered benchmark cohort does not fit the frozen grid"
            )
        return []
    benchmark_cell_order: list[str] | None = None
    benchmark_cell_ids: set[str] | None = None
    if benchmark_authorization is not None:
        benchmark_cell_order = [
            cell["cell_id"]
            for cell in benchmark_authorization["cells"]
            if cell["sample_id"] == str(snapshot["sample_id"])
        ]
        if not benchmark_cell_order:
            raise ValueError(
                "snapshot sample is outside the preregistered benchmark cohort"
            )
        if len(benchmark_cell_order) > max_tasks:
            raise ValueError(
                "max_tasks is smaller than the preregistered benchmark cohort "
                "for this sample"
            )
        benchmark_cell_ids = set(benchmark_cell_order)
    known = store.surfaces_for_snapshot(snapshot["source_snapshot_id"])
    bboxes = [surface["bbox_xyz"] for surface in known]
    if selection_strategy not in {"max-clearance-v1", "stratified-clearance-v1"}:
        raise ValueError(f"unsupported selection_strategy: {selection_strategy}")
    # Keep only the best max_tasks entries without materialising the whole 3-D grid.
    heap: list[tuple[float, tuple[int, int, int], list[float]]] = []
    buckets: dict[tuple[int, int, int], tuple[float, tuple[int, int, int], list[float]]] = {}
    benchmark_selected: dict[
        str, tuple[float, tuple[int, int, int], list[float]]
    ] = {}
    bucket_side = max(1, math.ceil(max_tasks ** (1.0 / 3.0)))
    for indices in itertools.product(*(range(len(axis)) for axis in axes)):
        center = [float(axes[axis][indices[axis]]) for axis in range(3)]
        gap = min((point_bbox_gap(center, bbox) for bbox in bboxes), default=float("inf"))
        # A candidate can lie query_radius voxels away from the cell centre.
        # Subtract that maximum displacement so every point in the claimed
        # query cube, rather than only its centre, satisfies the clearance.
        guaranteed_gap = gap - query_radius
        if guaranteed_gap < clearance:
            continue
        finite_gap = guaranteed_gap if math.isfinite(guaranteed_gap) else float(max(shape))
        # Lower lexical indices win exact score ties, regardless of traversal/runtime.
        rank_key = (finite_gap, tuple(-index for index in indices), center)
        if benchmark_cell_ids is not None:
            cell_id = "r%05dc%05da%05d" % indices
            if cell_id in benchmark_cell_ids:
                benchmark_selected[cell_id] = rank_key
            continue
        if selection_strategy == "max-clearance-v1":
            if len(heap) < max_tasks:
                heapq.heappush(heap, rank_key)
            elif rank_key > heap[0]:
                heapq.heapreplace(heap, rank_key)
        else:
            bucket = tuple((indices[axis] * bucket_side) // len(axes[axis]) for axis in range(3))
            existing = buckets.get(bucket)
            if existing is None or rank_key > existing:
                buckets[bucket] = rank_key
    if benchmark_cell_order is not None:
        missing = [
            cell_id
            for cell_id in benchmark_cell_order
            if cell_id not in benchmark_selected
        ]
        if missing:
            raise ValueError(
                "preregistered benchmark cells are not eligible under the "
                f"frozen grid/source constraints: {missing}"
            )
        selected = [
            benchmark_selected[cell_id] for cell_id in benchmark_cell_order
        ]
    elif selection_strategy == "max-clearance-v1":
        selected = sorted(heap, key=lambda item: (-item[0], tuple(-value for value in item[1])))
    else:
        # One top clearance cell per coarse spatial bucket, then deterministic
        # truncation. This is a geometry-only exploration policy, not a score
        # learned from ink or past outcomes.
        selected = sorted(buckets.values(), key=lambda item: (-item[0], tuple(-value for value in item[1])))[:max_tasks]
    tasks: list[dict[str, Any]] = []
    for gap, negative_indices, center in selected:
        indices = tuple(-value for value in negative_indices)
        bounds = [[center[axis] - query_radius for axis in range(3)], [center[axis] + query_radius for axis in range(3)]]
        cell_id = "r%05dc%05da%05d" % indices
        task_envelope = (
            copy.deepcopy(SEED_PROBE_CONTINUATION_ENVELOPE)
            if benchmark_authorization is not None
            else (
                parameter_envelope
                or (
                    SEED_PROBE_CONTINUATION_ENVELOPE
                    if normalized_probe is not None
                    and normalized_probe["mode"] == "select"
                    else DEFAULT_ENVELOPE
                )
            )
        )
        benchmark_execution = (
            build_seed_probe_benchmark_execution(
                benchmark_authorization,
                arm=str(benchmark_arm),
                sample_id=str(snapshot["sample_id"]),
                cell_id=cell_id,
                parameter_envelope=task_envelope,
            )
            if benchmark_authorization is not None
            else None
        )
        tasks.append({
            "schema": "campaignx.segmentation_task.v1",
            "source_snapshot_id": snapshot["source_snapshot_id"],
            "sample_id": snapshot["sample_id"],
            "cell_id": cell_id,
            "grid_version": grid_version,
            "policy_version": policy_version,
            "bounds_xyz": bounds,
            "center_xyz": dict(zip("xyz", [int(value) for value in center], strict=True)),
            "priority": float(gap),
            "distance_to_known_aabb_voxels": float(gap + query_radius),
            "guaranteed_cell_clearance_voxels": float(gap),
            "parameter_envelope": task_envelope,
            "catalog_snapshot_sha256": catalog_snapshot_sha256,
            "candidate_discovery": {
                "provider": "vc3d-mcp",
                "prediction_uri": snapshot["m7_uri"],
                "prediction_space": "ct_l0_xyz",
                # Screened at the threshold this prediction was published at, not
                # at a constant compiled into the seed provider. It decides which
                # voxels count as sheet, so it decides which seeds exist.
                **({"m7_threshold": snapshot["m7_threshold"]}
                   if snapshot.get("m7_threshold") is not None else {}),
                "region": {
                    "center": dict(zip("xyz", [int(value) for value in center], strict=True)),
                    "radius": {axis: query_radius for axis in "xyz"},
                },
                "max_candidates": 8,
                "minimum_separation_voxels": 16,
                "minimum_cell_interior_clearance_voxels": candidate_interior_clearance,
                "minimum_volume_interior_clearance_voxels": volume_edge_margin,
                "seed_region_policy": seed_region_policy,
                "recenter_probe_max_candidates": recenter_probe_max_candidates,
                "recenter_radius_xyz": {axis: int(recenter_radius[axis]) for axis in "xyz"},
                **({"ct_material_support_gate": ct_material_support_gate} if ct_material_support_gate else {}),
            },
            "candidate_selection_policy": candidate_selection_policy,
            "planner_contract_version": (
                "v2"
                if candidate_selection_policy == "adaptive-geometry-history-v2"
                or (
                    normalized_probe is not None
                    and normalized_probe["mode"] == "select"
                )
                or str(planner or "").endswith("-v2")
                else "v1"
            ),
            **({"seed_probe": normalized_probe} if normalized_probe else {}),
            **(
                {"benchmark_execution": benchmark_execution}
                if benchmark_execution is not None
                else {}
            ),
            "resource_requirements": {
                "gpu_required": bool(
                    normalized_probe
                    and normalized_probe["probe_parameters"]["use_cuda"]
                ),
                "minimum_vram_gb": 0.0,
                "seed_probe_required": normalized_probe is not None,
            },
            **({"planner": planner} if planner else {}),
            **({"planner_model": planner_model} if planner_model else {}),
            # Why somebody asked for this run, carried on the task itself.
            # The launcher says the reason is "kept with the run" and it was kept
            # nowhere: returned in the HTTP reply and dropped. Request bodies are
            # deliberately absent from the panel's audit trail -- one of those
            # routes sets S3 credentials -- so the trail was not the place
            # either. It rides here, where the run itself can be asked.
            **({"queued_reason": queued_reason} if queued_reason else {}),
            # And who asked, on the task itself rather than only in the panel's
            # audit trail. Without it the fleet could report what was running
            # and never whose, so nobody could see their own runs and no share
            # of the machines could be worked out at all.
            **({"created_by": created_by} if created_by else {}),
            # Ownership, not input scope. Two missions can intentionally run
            # the same scroll and must still receive different task identities.
            "mission_id": mission_id or "unfiled",
            # Which P0 selection this task reads. The snapshot already carries
            # the CT and m7 URIs, the frame, the scale and the manifest hash;
            # this says which selection chose them.
            **({"p0_selection_version": p0_selection_version}
               if p0_selection_version else {}),
            **({"p0_artifact_id": p0_artifact_id} if p0_artifact_id else {}),
            **({"p0_artifact_sha256": p0_artifact_sha256}
               if p0_artifact_sha256 else {}),
            **({"p0_resolved_by": p0_resolved_by} if p0_resolved_by else {}),
            "ink_used": False,
        })
    return tasks


def generate_seed_probe_benchmark_arm_tasks(
    store: FleetStore,
    snapshots: Iterable[dict[str, Any]],
    *,
    benchmark_execution_authorization: dict[str, Any],
    arm: str,
    generation_options: dict[str, Any],
    seed_probe: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Materialize exactly one preregistered causal arm, without queue writes.

    The returned list is suitable for one atomic ``store.create_tasks`` call.
    It contains every authorized (sample, cell) pair exactly once and no other
    pair.  This is the programmatic isolated-benchmark entry point; ordinary
    fleet bootstrap does not acquire benchmark authority implicitly.
    """

    authorization = normalize_seed_probe_benchmark_execution_authorization(
        benchmark_execution_authorization
    )
    if arm not in {"baseline", "closed_loop"}:
        raise ValueError("benchmark arm must be baseline or closed_loop")
    if arm == "baseline" and seed_probe is not None:
        raise ValueError("baseline benchmark arm must run with seed probe off")
    if arm == "closed_loop" and seed_probe is None:
        raise ValueError("closed_loop benchmark arm requires a select policy")
    reserved = {
        "benchmark_execution_authorization",
        "policy_version",
        "planner",
        "seed_probe",
    }
    overlap = reserved.intersection(generation_options)
    if overlap:
        raise ValueError(
            "benchmark generation_options may not override frozen fields: "
            f"{sorted(overlap)}"
        )

    expected = {
        (cell["sample_id"], cell["cell_id"])
        for cell in authorization["cells"]
    }
    expected_samples = {sample_id for sample_id, _ in expected}
    snapshots_by_sample: dict[str, list[dict[str, Any]]] = {
        sample_id: [] for sample_id in expected_samples
    }
    for snapshot in snapshots:
        sample_id = str(snapshot.get("sample_id") or "")
        if sample_id in snapshots_by_sample:
            snapshots_by_sample[sample_id].append(snapshot)
    invalid_snapshot_counts = {
        sample_id: len(rows)
        for sample_id, rows in snapshots_by_sample.items()
        if len(rows) != 1
    }
    if invalid_snapshot_counts:
        raise ValueError(
            "benchmark requires exactly one frozen source snapshot per "
            f"authorized sample: {invalid_snapshot_counts}"
        )
    for rows in snapshots_by_sample.values():
        # The off arm must be just as source-frozen as select.  Otherwise the
        # paired contract would share a cell/RNG label while reading mutable or
        # differently versioned bytes.
        normalize_source_content_lock(rows[0])

    policy_version = (
        authorization["baseline_policy_version"]
        if arm == "baseline"
        else authorization["policy_version"]
    )
    tasks: list[dict[str, Any]] = []
    for sample_id in sorted(expected_samples):
        tasks.extend(
            generate_tasks_for_snapshot(
                store,
                snapshots_by_sample[sample_id][0],
                policy_version=policy_version,
                planner="deterministic-v2",
                seed_probe=seed_probe,
                benchmark_execution_authorization=authorization,
                **copy.deepcopy(generation_options),
            )
        )
    actual = [(str(task["sample_id"]), str(task["cell_id"])) for task in tasks]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(set(actual).difference(expected))
        raise ValueError(
            "generated benchmark arm differs from preregistered cohort: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return tasks


def generate_manual_tasks(
    store: FleetStore,
    snapshot: dict[str, Any],
    points: list[dict[str, Any]],
    *,
    catalog_snapshot_sha256: str,
    grid_step: int,
    query_radius: int,
    volume_edge_margin: int,
    grid_version: str,
    policy_version: str,
    submitted_by: str,
    ct_material_support_gate: dict[str, Any] | None = None,
    planner: str | None = None,
    planner_model: str | None = None,
    queued_reason: str | None = None,
    created_by: str | None = None,
    mission_id: str | None = None,
    p0_selection_version: str | None = None,
    p0_artifact_id: str | None = None,
    p0_artifact_sha256: str | None = None,
    p0_resolved_by: str | None = None,
) -> list[dict[str, Any]]:
    """One task per point a person supplied, instead of per uncovered cell.

    Separate from generate_tasks_for_snapshot because the two answer different
    questions. That one sweeps: it ranks cells by how far they are from ground
    already segmented and takes the best few. This one is told where to look, so
    ranking and coverage have no opinion to offer -- and folding points into the
    sweep would mean the coverage policy deciding what to do with an instruction
    it did not choose.

    Clearance is measured and recorded but never used to reject. A point close to
    an existing surface is exactly what someone might mean to place -- extending a
    partial lamina, or testing whether a join is real -- and refusing it would be
    the grid's policy overruling the person who overrode the grid.

    What is refused is a point that cannot be grown from at all: outside the
    volume, or inside the edge margin, where a surface grows into the crop and
    the result is an artefact of where the scan stops rather than of papyrus.

    The task carries `manual_candidates`, which ManualSeedProvider reads instead
    of probing the prediction. Everything after discovery is unchanged: the CT
    gate still asks the raw scan whether there is material there, and for a
    manual seed that is the only screen left, because the prediction was skipped.
    """
    shape = [int(value) for value in snapshot["shape_xyz"]]
    known = store.surfaces_for_snapshot(snapshot["source_snapshot_id"])
    bboxes = [surface["bbox_xyz"] for surface in known]
    tasks: list[dict[str, Any]] = []
    for index, point in enumerate(points, start=1):
        try:
            centre = {axis: int(round(float(point[axis]))) for axis in "xyz"}
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"point {index} needs numeric x, y and z: {point!r}") from None
        for position, axis in enumerate("xyz"):
            low, high = volume_edge_margin, shape[position] - volume_edge_margin
            if not low <= centre[axis] < high:
                raise ValueError(
                    f"point {index} has {axis}={centre[axis]}, outside the usable "
                    f"range {low}..{high} for a volume of {shape[position]} on that "
                    "axis -- a seed inside the edge margin grows a surface into the "
                    "crop")
        ordered = [float(centre[axis]) for axis in "xyz"]
        gap = min((point_bbox_gap(ordered, bbox) for bbox in bboxes), default=float("inf"))
        finite_gap = gap if math.isfinite(gap) else float(max(shape))
        # Stable across resubmissions of the same point, so uploading a file
        # twice does not queue the same growth twice.
        candidate_id = "manual-" + stable_id(
            "manual-seed",
            {"source_snapshot_id": snapshot["source_snapshot_id"], **centre},
        )[:16]
        # The point is the cell, not the grid square it lands in.
        #
        # A task's identity is (snapshot, grid_version, cell_id, policy_version),
        # which is right for a sweep where one grid square is one task. Two
        # supplied points 100 voxels apart share a 2048-voxel square, so the
        # second was deduplicated away -- silently, which is the worst way to lose
        # an instruction somebody typed. Deriving the cell from the coordinate
        # keeps the deduplication that matters (the same point twice is the same
        # request) and drops the one that does not.
        cell_id = candidate_id
        tasks.append({
            "schema": "campaignx.segmentation_task.v1",
            "source_snapshot_id": snapshot["source_snapshot_id"],
            "sample_id": snapshot["sample_id"],
            "cell_id": cell_id,
            "grid_version": grid_version,
            "policy_version": policy_version,
            "bounds_xyz": [[centre[axis] - query_radius for axis in "xyz"],
                           [centre[axis] + query_radius for axis in "xyz"]],
            "center_xyz": dict(centre),
            "priority": float(finite_gap),
            "distance_to_known_aabb_voxels": float(finite_gap + query_radius),
            "guaranteed_cell_clearance_voxels": float(finite_gap),
            "parameter_envelope": DEFAULT_ENVELOPE,
            "catalog_snapshot_sha256": catalog_snapshot_sha256,
            "manual_candidates": [{
                "candidate_id": candidate_id,
                "ct_l0_coordinate": dict(centre),
                **centre,
                # No score and no clearance, and the keys are absent rather than
                # present-and-null. The screen resolves a score through
                # `candidate.get("score", candidate.get(..., 0.0))`, and a key
                # that exists with None defeats every default in that chain:
                # float(None) raises and the candidate is counted
                # MALFORMED_COORDINATE_OR_SCORE. Every manual seed was rejected
                # that way -- after the CT gate had accepted it -- so manual
                # seeding never produced a surface.
                #
                # The clearances are omitted for a different reason: the screen
                # measures them itself from the coordinate and the volume shape,
                # so a copy here could only disagree.
                "score_origin": "human_selection",
                "seed_origin": "human",
                "submitted_by": submitted_by,
                "note": str(point.get("note") or "")[:200],
            }],
            "candidate_discovery": {
                "provider": "manual",
                "prediction_uri": snapshot["m7_uri"],
                "prediction_space": "ct_l0_xyz",
                "region": {"center": dict(centre),
                           "radius": {axis: query_radius for axis in "xyz"}},
                "max_candidates": 1,
                "minimum_separation_voxels": 0,
                "minimum_cell_interior_clearance_voxels": 0,
                "minimum_volume_interior_clearance_voxels": volume_edge_margin,
                "seed_region_policy": "fixed-v1",
                "seed_origin": "human",
                "submitted_by": submitted_by,
                **({"ct_material_support_gate": ct_material_support_gate}
                   if ct_material_support_gate else {}),
            },
            "candidate_selection_policy": "score-cell-volume-clearance-v1",
            "planner_contract_version": "v1",
            **({"planner": planner} if planner else {}),
            **({"planner_model": planner_model} if planner_model else {}),
            # Why somebody asked for this run, carried on the task itself.
            # The launcher says the reason is "kept with the run" and it was kept
            # nowhere: returned in the HTTP reply and dropped. Request bodies are
            # deliberately absent from the panel's audit trail -- one of those
            # routes sets S3 credentials -- so the trail was not the place
            # either. It rides here, where the run itself can be asked.
            **({"queued_reason": queued_reason} if queued_reason else {}),
            # And who asked, on the task itself rather than only in the panel's
            # audit trail. Without it the fleet could report what was running
            # and never whose, so nobody could see their own runs and no share
            # of the machines could be worked out at all.
            **({"created_by": created_by} if created_by else {}),
            "mission_id": mission_id or "unfiled",
            # Which P0 selection this task reads. The snapshot already carries
            # the CT and m7 URIs, the frame, the scale and the manifest hash;
            # this says which selection chose them.
            **({"p0_selection_version": p0_selection_version}
               if p0_selection_version else {}),
            **({"p0_artifact_id": p0_artifact_id} if p0_artifact_id else {}),
            **({"p0_artifact_sha256": p0_artifact_sha256}
               if p0_artifact_sha256 else {}),
            **({"p0_resolved_by": p0_resolved_by} if p0_resolved_by else {}),
            "ink_used": False,
        })
    return tasks


def bootstrap_queue(
    store: FleetStore,
    eligible_path: Path,
    catalog_path: Path,
    *,
    samples: set[str] | None,
    grid_step: int,
    query_radius: int,
    clearance: float,
    volume_edge_margin: int,
    candidate_interior_clearance: int,
    selection_strategy: str,
    max_tasks_per_sample: int,
    grid_version: str,
    policy_version: str,
    candidate_selection_policy: str = "score-cell-volume-clearance-v1",
    seed_region_policy: str = "fixed-v1",
    recenter_probe_max_candidates: int = 100,
    recenter_radius_xyz: dict[str, int] | None = None,
    ct_material_support_gate: dict[str, Any] | None = None,
    planner: str | None = None,
    planner_model: str | None = None,
    queued_reason: str | None = None,
    created_by: str | None = None,
    mission_id: str | None = None,
    p0_selection_version: str | None = None,
    p0_artifact_id: str | None = None,
    p0_artifact_sha256: str | None = None,
    p0_resolved_by: str | None = None,
    verify_sources: bool = True,
    seed_probe: dict[str, Any] | None = None,
    benchmark_execution_authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store.initialize()
    sources = bootstrap_sources(store, eligible_path, samples, verify=verify_sources)
    catalog_counts = import_catalog(store, catalog_path, sources)
    catalog_sha = file_sha256(catalog_path)
    generated: dict[str, dict[str, int]] = {}
    unreachable: dict[str, Any] = {}
    snapshots = store.snapshots(samples)
    if benchmark_execution_authorization is not None:
        benchmark_execution_authorization = (
            normalize_seed_probe_benchmark_execution_authorization(
                benchmark_execution_authorization
            )
        )
        authorized_samples = {
            cell["sample_id"]
            for cell in benchmark_execution_authorization["cells"]
        }
        snapshot_counts = {
            sample_id: sum(
                str(snapshot.get("sample_id") or "") == sample_id
                for snapshot in snapshots
            )
            for sample_id in authorized_samples
        }
        invalid_counts = {
            sample_id: count
            for sample_id, count in snapshot_counts.items()
            if count != 1
        }
        if invalid_counts:
            raise ValueError(
                "isolated benchmark bootstrap requires exactly one frozen "
                f"snapshot per authorized sample: {invalid_counts}"
            )
        snapshots = [
            snapshot
            for snapshot in snapshots
            if str(snapshot.get("sample_id") or "") in authorized_samples
        ]
        unreachable_authorized = {
            str(snapshot["sample_id"]): probe
            for snapshot in snapshots
            if (
                (probe := (snapshot.get("source_reachable") or {}).get("m7") or {})
                .get("checked")
                and not probe.get("reachable")
            )
        }
        if unreachable_authorized:
            raise ValueError(
                "isolated benchmark source is unreachable; no arm tasks were "
                f"queued: {sorted(unreachable_authorized)}"
            )
    for snapshot in snapshots:
        # A cell whose prediction volume cannot be read is a task that will be
        # claimed, sent to a worker and fail there. The snapshot stays -- it is
        # a record of what the catalog said -- and no work is queued against it.
        probe = (snapshot.get("source_reachable") or {}).get("m7") or {}
        if probe.get("checked") and not probe.get("reachable"):
            unreachable[str(snapshot["sample_id"])] = probe
            continue
        tasks = generate_tasks_for_snapshot(
            store,
            snapshot,
            catalog_snapshot_sha256=catalog_sha,
            grid_step=grid_step,
            query_radius=query_radius,
            clearance=clearance,
            volume_edge_margin=volume_edge_margin,
            candidate_interior_clearance=candidate_interior_clearance,
            selection_strategy=selection_strategy,
            max_tasks=max_tasks_per_sample,
            grid_version=grid_version,
            policy_version=policy_version,
            candidate_selection_policy=candidate_selection_policy,
            planner=planner,
            planner_model=planner_model,
            seed_region_policy=seed_region_policy,
            recenter_probe_max_candidates=recenter_probe_max_candidates,
            recenter_radius_xyz=recenter_radius_xyz,
            ct_material_support_gate=ct_material_support_gate,
            queued_reason=queued_reason,
            created_by=created_by,
            mission_id=mission_id,
            p0_selection_version=p0_selection_version,
            p0_artifact_id=p0_artifact_id,
            p0_artifact_sha256=p0_artifact_sha256,
            p0_resolved_by=p0_resolved_by,
            seed_probe=seed_probe,
            benchmark_execution_authorization=(
                benchmark_execution_authorization
            ),
        )
        inserted, seen = store.create_tasks(tasks)
        generated[snapshot["sample_id"]] = {"generated": seen, "inserted": inserted}
    return {
        "schema": "campaignx.segment_fleet_bootstrap_receipt.v1",
        "unreachable_sources": unreachable,
        "eligible_sha256": file_sha256(eligible_path),
        "catalog_sha256": catalog_sha,
        "sources": sources,
        "catalog": catalog_counts,
        "tasks": generated,
        "grid": {
            "step": grid_step,
            "query_radius": query_radius,
            "clearance": clearance,
            "volume_edge_margin": volume_edge_margin,
            "candidate_interior_clearance": candidate_interior_clearance,
            "selection_strategy": selection_strategy,
            "grid_version": grid_version,
            "seed_region_policy": seed_region_policy,
            "recenter_probe_max_candidates": recenter_probe_max_candidates,
            "recenter_radius_xyz": recenter_radius_xyz or {"x": 64, "y": 64, "z": 64},
            "ct_material_support_gate": ct_material_support_gate,
            "planner": planner,
            "planner_model": planner_model,
            "candidate_selection_policy": candidate_selection_policy,
            "planner_contract_version": (
                "v2"
                if candidate_selection_policy == "adaptive-geometry-history-v2"
                else "v1"
            ),
        },
        "policy_version": policy_version,
        "ink_used": False,
        "status": store.status(),
    }


def replan_no_seed_cells(
    store: FleetStore,
    *,
    grid_version: str,
    policy_version: str,
    planner: str | None = None,
    planner_model: str | None = None,
    sample_id: str | None = None,
    causes: list[str] | None = None,
    limit: int = 50,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ask the cells that gave nothing again, differently.

    169 of 241 tasks on this control plane ended NO_SEED, the worker recorded why
    for each, and nothing ever acted on it: the cell was terminal, the diagnosis
    sat in a jsonb column, and the next bootstrap picked cells by distance from
    known surfaces as though the failure had never happened. The fleet explored
    and did not learn.

    A replan is a new task over the same ground under an explicitly different
    policy version -- task identity is (snapshot, grid, cell, policy), so without
    a new one this is a silent no-op -- and usually a different planner, since
    the planner is what decides the seed.

    This is not a retry. The original task and its diagnosis stay exactly as they
    are; what comes out is a new question about the same cell, and comparing the
    two is the point.

    Non-claims
    ----------
    * Re-asking is not evidence that a lamina is there. A cell where the m7
      provider offers nothing may simply be empty scroll.
    * Nothing here changes the prediction volume or its threshold. A cell that
      failed for NO_M7_CANDIDATES will fail the same way under a new planner
      unless what is offered changes.
    """
    if not grid_version:
        raise ValueError("a replan needs its own grid_version")
    if not policy_version:
        raise ValueError("a replan needs its own policy_version")
    store.initialize()
    cells = store.no_seed_cells(sample_id=sample_id, causes=causes, limit=limit)
    tasks: list[dict[str, Any]] = []
    for cell in cells:
        if cell["grid_version"] == grid_version and cell["policy_version"] == policy_version:
            # The same identity as the task that already failed: inserting it
            # would be an ON CONFLICT DO NOTHING that looks like work.
            continue
        payload = dict(cell["payload"] or {})
        for key in ("task_id", "state", "active_attempt_id", "created_at",
                    "updated_at", "lease_expires_at", "worker_id"):
            payload.pop(key, None)
        payload.update({
            "grid_version": grid_version,
            "policy_version": policy_version,
            "cell_id": cell["cell_id"],
            "source_snapshot_id": cell["source_snapshot_id"],
            "sample_id": cell["sample_id"],
            "replan_of": cell["task_id"],
            "replan_reason": cell["causes"],
        })
        if planner:
            payload["planner"] = planner
        if planner_model:
            payload["planner_model"] = planner_model
        tasks.append(payload)
    if dry_run:
        return {"schema": "campaignx.segment_replan_receipt.v1", "dry_run": True,
                "considered": len(cells), "would_queue": len(tasks),
                "cells": [c["cell_id"] for c in tasks],
                "generated_at_utc": utc_now()}
    inserted, seen = store.create_tasks(tasks) if tasks else (0, 0)
    return {"schema": "campaignx.segment_replan_receipt.v1",
            "considered": len(cells), "queued": inserted, "seen": seen,
            "grid_version": grid_version, "policy_version": policy_version,
            "planner": planner,
            "causes": causes or ["any"],
            "generated_at_utc": utc_now(),
            "non_claims": [
                "re-asking a cell is not evidence that a lamina is there",
                "the prediction volume and its threshold are unchanged, so a cell "
                "that failed for NO_M7_CANDIDATES fails the same way unless what "
                "the provider offers changes",
            ]}


def bootstrap_seed_positive_recovery_queue(
    source_store: FleetStore,
    output_store: FleetStore,
    source_attempt_root: Path,
    *,
    grid_version: str,
    policy_version: str,
    seed_region_policy: str = "m7-chunk-safe-merge-interior-v2",
    candidate_selection_policy: str = "score-cell-volume-clearance-v1",
    ct_material_support_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a novel recovery queue from geometry-only m7-positive failures.

    The source run may have returned m7 candidates but rejected all of them at
    its frozen interior-clearance gate.  This builder copies only the source
    and surface catalogue, then creates new immutable tasks for those cells
    under an explicitly different policy version.  It never copies attempts,
    outcomes, model proposals, or ink-derived evidence into the new control
    plane.
    """

    if source_store.path.resolve() == output_store.path.resolve():
        raise ValueError("recovery output database must differ from its source")
    if output_store.path.exists():
        raise FileExistsError(f"recovery output database already exists: {output_store.path}")
    if seed_region_policy != "m7-chunk-safe-merge-interior-v2":
        raise ValueError("seed-positive recovery requires the merged-interior v2 policy")
    if candidate_selection_policy not in {
        "score-cell-volume-clearance-v1",
        "adaptive-geometry-history-v2",
    }:
        raise ValueError("unsupported candidate_selection_policy")
    source_attempt_root = source_attempt_root.resolve()
    if not source_attempt_root.is_dir():
        raise FileNotFoundError(source_attempt_root)

    output_store.initialize()
    snapshots = source_store.snapshots()
    if not snapshots:
        raise RuntimeError("source recovery database has no snapshots")
    surface_rows: list[dict[str, Any]] = []
    with source_store.connect() as connection:
        for row in connection.execute("SELECT * FROM surfaces ORDER BY surface_id"):
            value = json.loads(row["payload_json"])
            value.update({
                "surface_id": row["surface_id"],
                "source_snapshot_id": row["source_snapshot_id"],
                "sample_id": row["sample_id"],
                "owner": row["owner"],
                "artifact_sha256": row["artifact_sha256"],
                "artifact_uri": row["artifact_uri"],
                "bbox_xyz": json.loads(row["bbox_xyz_json"]),
                "sample_points": (
                    json.loads(row["sample_points_json"])
                    if row["sample_points_json"] is not None
                    else None
                ),
                "area_cm2": row["area_cm2"],
                "state": row["state"],
                "physical_qc_state": row["physical_qc_state"],
            })
            surface_rows.append(value)
        terminal_rows = connection.execute(
            """SELECT task_id,active_attempt_id FROM tasks
               WHERE state='NO_SEED' ORDER BY task_id"""
        ).fetchall()

    for snapshot in snapshots:
        registered = output_store.register_snapshot(snapshot)
        if registered != snapshot["source_snapshot_id"]:
            raise RuntimeError("source snapshot identity changed during recovery bootstrap")
    for surface in surface_rows:
        output_store.import_surface(surface)

    catalogue_contract = {
        "snapshots": snapshots,
        "surfaces": surface_rows,
    }
    catalogue_sha256 = content_sha256(catalogue_contract)
    tasks: list[dict[str, Any]] = []
    source_task_ids: list[str] = []
    raw_candidate_count = 0
    skipped_missing_screen = 0
    skipped_zero_candidate = 0
    skipped_prior_eligible = 0
    for row in terminal_rows:
        attempt_id = str(row["active_attempt_id"] or "")
        screen_path = source_attempt_root / str(row["task_id"]) / attempt_id / "SEED_SCREEN.json"
        if not attempt_id or not screen_path.is_file():
            skipped_missing_screen += 1
            continue
        screen = read_json(screen_path)
        raw_count = int(screen.get("raw_candidate_count", 0))
        eligible_count = int(screen.get("eligible_candidate_count", 0))
        if raw_count <= 0:
            skipped_zero_candidate += 1
            continue
        if eligible_count > 0:
            skipped_prior_eligible += 1
            continue
        original = source_store.task_packet(str(row["task_id"]))
        discovery = dict(original["candidate_discovery"])
        discovery["seed_region_policy"] = seed_region_policy
        discovery["recenter_radius_xyz"] = {"x": 64, "y": 64, "z": 64}
        if ct_material_support_gate is None:
            discovery.pop("ct_material_support_gate", None)
        else:
            discovery["ct_material_support_gate"] = ct_material_support_gate
        task = {
            key: value
            for key, value in original.items()
            if key not in {
                "task_id",
                "state",
                "source",
                "attempt_id",
                "attempt_number",
                "worker_id",
                "lease_token",
            }
        }
        task.update({
            "grid_version": grid_version,
            "policy_version": policy_version,
            "catalog_snapshot_sha256": catalogue_sha256,
            "candidate_discovery": discovery,
            "candidate_selection_policy": candidate_selection_policy,
            "planner_contract_version": (
                "v2"
                if candidate_selection_policy == "adaptive-geometry-history-v2"
                else "v1"
            ),
            "recovery_source_task_id": str(row["task_id"]),
            "recovery_source_attempt_id": attempt_id,
            "recovery_basis": "M7_RAW_POSITIVE_INTERIOR_REJECTED",
            "ink_used": False,
        })
        tasks.append(task)
        source_task_ids.append(str(row["task_id"]))
        raw_candidate_count += raw_count

    inserted, seen = output_store.create_tasks(tasks)
    if inserted != seen:
        raise RuntimeError("new recovery database did not insert every selected task")
    return {
        "schema": "campaignx.seed_positive_recovery_bootstrap_receipt.v1",
        "source_database": str(source_store.path),
        "source_attempt_root": str(source_attempt_root),
        "output_database": str(output_store.path),
        "catalogue_contract_sha256": catalogue_sha256,
        "copied_snapshot_count": len(snapshots),
        "copied_surface_count": len(surface_rows),
        "source_no_seed_task_count": len(terminal_rows),
        "selected_task_count": seen,
        "inserted_task_count": inserted,
        "selected_raw_candidate_count": raw_candidate_count,
        "candidate_selection_policy": candidate_selection_policy,
        "skipped_missing_screen_count": skipped_missing_screen,
        "skipped_zero_candidate_count": skipped_zero_candidate,
        "skipped_prior_eligible_count": skipped_prior_eligible,
        "source_task_ids": source_task_ids,
        "grid_version": grid_version,
        "policy_version": policy_version,
        "seed_region_policy": seed_region_policy,
        "ct_material_support_gate": ct_material_support_gate,
        "ink_used": False,
        "non_claim": "Selection uses only m7 candidate availability and prior geometry-gate rejection; it makes no ink, text, or surface-validity claim.",
        "status": output_store.status(),
    }


def bootstrap_adaptive_retry_task(
    store: Any,
    source_task_id: str,
    *,
    grid_version: str,
    policy_version: str,
    seed_region_policy: str = "m7-chunk-safe-merge-interior-v2",
    minimum_cell_interior_clearance_voxels: int = 0,
) -> dict[str, Any]:
    """Create one novel planner-v2 task from a terminal geometry failure."""

    store.initialize()
    original = store.task_packet(source_task_id)
    if original.get("state") not in {
        "NO_SEED",
        "GROW_FAILED",
        "POLICY_REJECTED",
        "BLOCKED_SOURCE_UNAVAILABLE",
    }:
        raise ValueError("adaptive retry source must be a terminal geometry failure")
    if original.get("ink_used") is not False:
        raise ValueError("adaptive retry source must be ink-blind")
    if not grid_version or grid_version == original.get("grid_version"):
        raise ValueError("adaptive retry requires a new grid_version")
    if not policy_version or policy_version == original.get("policy_version"):
        raise ValueError("adaptive retry requires a new policy_version")
    if seed_region_policy != "m7-chunk-safe-merge-interior-v2":
        raise ValueError("adaptive retry requires merged-interior v2 seed discovery")
    if minimum_cell_interior_clearance_voxels < 0:
        raise ValueError("minimum cell interior clearance must be non-negative")

    task = {
        key: value
        for key, value in original.items()
        if key
        not in {
            "task_id",
            "state",
            "source",
            "attempt_id",
            "attempt_number",
            "worker_id",
            "lease_token",
            "lease_expires_at",
            "retry_after",
        }
    }
    discovery = dict(task.get("candidate_discovery") or {})
    discovery.update(
        {
            "seed_region_policy": seed_region_policy,
            "minimum_cell_interior_clearance_voxels": int(
                minimum_cell_interior_clearance_voxels
            ),
            "recenter_radius_xyz": {"x": 64, "y": 64, "z": 64},
        }
    )
    task.update(
        {
            "grid_version": grid_version,
            "policy_version": policy_version,
            "candidate_discovery": discovery,
            "candidate_selection_policy": "adaptive-geometry-history-v2",
            "planner_contract_version": "v2",
            "adaptive_source_task_id": source_task_id,
            "adaptive_retry_basis": str(original["state"]),
            "ink_used": False,
        }
    )
    inserted, seen = store.create_tasks([task])
    if (inserted, seen) != (1, 1):
        raise RuntimeError("adaptive retry task identity already exists")
    created = next(
        row
        for row in store.pending_tasks(10000)
        if row.get("grid_version") == grid_version
        and row.get("policy_version") == policy_version
        and row.get("cell_id") == original.get("cell_id")
    )
    return {
        "schema": "campaignx.segmentation_adaptive_retry_bootstrap_receipt.v1",
        "source_task_id": source_task_id,
        "task_id": created["task_id"],
        "sample_id": created["sample_id"],
        "cell_id": created["cell_id"],
        "grid_version": grid_version,
        "policy_version": policy_version,
        "candidate_selection_policy": "adaptive-geometry-history-v2",
        "planner_contract_version": "v2",
        "seed_region_policy": seed_region_policy,
        "minimum_cell_interior_clearance_voxels": int(
            minimum_cell_interior_clearance_voxels
        ),
        "ink_used": False,
    }
