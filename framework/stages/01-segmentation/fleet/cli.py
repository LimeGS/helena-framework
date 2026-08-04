from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .artifact_store import open_artifact_store
from .certifier import certify_pending, load_qc_adapter
from .common import content_sha256, file_sha256, utc_now, write_json_atomic
from .executor import FixtureGrowExecutor, VC3DGrowExecutor
from .generator import (
    DEFAULT_ENVELOPE,
    bootstrap_adaptive_retry_task,
    generate_manual_tasks,
    generate_seed_probe_benchmark_arm_tasks,
    bootstrap_queue,
    bootstrap_seed_positive_recovery_queue,
    bootstrap_sources,
    import_catalog,
)
from .planner import (
    DEFAULT_OPENCODE_MODEL,
    CostAwareSegmentationPlanner,
    DeterministicPlanner,
    OpenCodePlanner,
    OpenRouterFusionPlanner,
    Planner,
    screen_candidates,
)
from .qc_worker import FixtureQcExecutor, SubprocessQcExecutor, SurfaceQcWorker
from .seed_probe import (
    default_seed_probe_policy,
    load_seed_probe_benchmark_spec,
    load_seed_probe_benchmark_receipt,
)
from .store import FleetStore, QC_OUTCOME_STATES
from .store_factory import open_fleet_store, store_identity
from .worker import (
    ManualSeedProvider,
    TaskRoutedSeedProvider,
    McpSeedProvider,
    RecordedSeedProvider,
    SegmentWorker,
)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def command_init(args: argparse.Namespace) -> int:
    store = open_fleet_store(args.db)
    store.initialize()
    print_json(store.status())
    return 0


def command_probe_gc(args: argparse.Namespace) -> int:
    """Dry-run or delete only ledger-approved expired probe evidence."""

    store = open_fleet_store(args.db)
    store.initialize()
    candidates = store.probe_artifacts_due_for_gc(limit=args.limit)
    public_candidates = [
        {
            key: row[key]
            for key in (
                "probe_artifact_set_id",
                "artifact_uri",
                "state",
                "retain_until",
                "probe_run_id",
                "task_id",
            )
        }
        for row in candidates
    ]
    candidate_set_sha256 = content_sha256(public_candidates)
    deleted: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if args.apply:
        expected = str(
            getattr(args, "expect_candidate_set_sha256", "") or ""
        )
        if expected != candidate_set_sha256:
            raise RuntimeError(
                "probe GC apply requires the exact "
                "--expect-candidate-set-sha256 printed by a current dry run"
            )
        artifact_store = open_artifact_store(args.artifact_root)
        for row in candidates:
            try:
                deletion = artifact_store.delete_probe(
                    row["artifact_uri"], row["manifest"]
                )
                store.mark_probe_artifact_expired(
                    row["probe_artifact_set_id"], row["artifact_uri"]
                )
                deleted.append(
                    {
                        "probe_artifact_set_id": row[
                            "probe_artifact_set_id"
                        ],
                        **deletion,
                    }
                )
            except Exception as error:  # independent, content-addressed rows
                errors.append(
                    {
                        "probe_artifact_set_id": row[
                            "probe_artifact_set_id"
                        ],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    receipt = {
        "schema": "campaignx.seed_probe_gc.v1",
        "generated_at_utc": utc_now(),
        "database": store_identity(store),
        "artifact_root": str(args.artifact_root),
        "mode": "APPLY" if args.apply else "DRY_RUN",
        "candidate_count": len(candidates),
        "candidate_set_sha256": candidate_set_sha256,
        "candidates": public_candidates,
        "deleted_count": len(deleted),
        "deleted": deleted,
        "error_count": len(errors),
        "errors": errors,
    }
    if args.receipt is not None:
        write_json_atomic(args.receipt, receipt)
    print_json(receipt)
    return 0 if not errors else 2


def command_bootstrap(args: argparse.Namespace) -> int:
    benchmark_receipt_path = getattr(
        args, "seed_probe_benchmark_receipt", None
    )
    review_owner = str(
        getattr(args, "seed_probe_review_owner", "") or ""
    ).strip()
    benchmark_authorization = None
    if args.seed_probe_mode == "select":
        if (
            os.environ.get("HELENA_ENABLE_SEED_PROBE_SELECT", "")
            .strip()
            .lower()
            not in {"1", "true", "yes", "on"}
        ):
            raise RuntimeError(
                "seed-probe select is rollout-gated; set "
                "HELENA_ENABLE_SEED_PROBE_SELECT=1 only after causal approval"
            )
        if benchmark_receipt_path is None:
            raise RuntimeError(
                "seed-probe select requires an explicit "
                "--seed-probe-benchmark-receipt; an environment flag alone "
                "does not authorize steering"
            )
        try:
            benchmark_authorization = load_seed_probe_benchmark_receipt(
                benchmark_receipt_path
            )
        except ValueError as error:
            raise RuntimeError(
                f"seed-probe select benchmark receipt is invalid: {error}"
            ) from error
        if not review_owner:
            raise RuntimeError(
                "seed-probe select requires --seed-probe-review-owner; a "
                "production review queue must have an accountable owner"
            )
    else:
        if benchmark_receipt_path is not None:
            raise RuntimeError(
                "--seed-probe-benchmark-receipt is only valid with "
                "--seed-probe-mode select"
            )
        if str(getattr(args, "seed_probe_review_owner", "") or "").strip():
            raise RuntimeError(
                "--seed-probe-review-owner is only valid with "
                "--seed-probe-mode select"
            )
    store = open_fleet_store(args.db)
    receipt = bootstrap_queue(
        store,
        args.eligible,
        args.catalog,
        samples=set(args.sample) or None,
        grid_step=args.grid_step,
        query_radius=args.query_radius,
        clearance=args.clearance,
        volume_edge_margin=args.volume_edge_margin,
        candidate_interior_clearance=args.candidate_interior_clearance,
        selection_strategy=args.selection_strategy,
        max_tasks_per_sample=args.max_tasks_per_sample,
        grid_version=args.grid_version,
        policy_version=args.policy_version,
        candidate_selection_policy=args.candidate_selection_policy,
        planner=args.planner,
        planner_model=args.planner_model,
        # Rides in the task payload, which is the dict the worker claims, so
        # task.get("candidate_rank") reaches the planner it builds.
        candidate_rank=getattr(args, "candidate_rank", 1),
        reconsider_covered=getattr(args, "reconsider_covered", False),
        queued_reason=args.reason,
        created_by=args.queued_by,
        mission_id=args.mission_id,
        p0_selection_version=args.p0_selection_version,
        p0_artifact_id=args.p0_artifact_id,
        p0_artifact_sha256=args.p0_artifact_sha256,
        p0_resolved_by=args.p0_resolved_by,
        seed_region_policy=args.seed_region_policy,
        verify_sources=not args.no_verify_sources,
        recenter_probe_max_candidates=args.recenter_probe_max_candidates,
        recenter_radius_xyz={"x": args.recenter_radius_x, "y": args.recenter_radius_y, "z": args.recenter_radius_z},
        ct_material_support_gate=(
            {
                "policy": "ome-zarr-nearby-material-v1",
                "level": args.ct_support_level,
                "radius_l0_voxels": args.ct_support_radius_l0,
                "minimum_nonzero_voxels": args.ct_support_minimum_nonzero_voxels,
            }
            if args.ct_material_support_gate
            else None
        ),
        seed_probe=(
            default_seed_probe_policy(
                mode=args.seed_probe_mode,
                top_k=args.seed_probe_top_k,
                generations=args.seed_probe_generations,
                benchmark_authorization=benchmark_authorization,
                review_owner=(
                    review_owner
                    if args.seed_probe_mode == "select"
                    else None
                ),
            )
            if args.seed_probe_mode != "off"
            else None
        ),
    )
    output = args.receipt or Path("SEGMENT_FLEET_BOOTSTRAP_RECEIPT.json")
    write_json_atomic(output, receipt)
    # "queued" is per sample and says generated vs inserted. Without it the
    # only thing printed is the queue's totals, and a bootstrap that inserted
    # nothing because every cell already had a task looked exactly like one
    # that filled the queue.
    print_json({"receipt": str(output), "queued": receipt["tasks"], **receipt["status"]})
    return 0


def command_seed_probe_benchmark_arm(args: argparse.Namespace) -> int:
    """Queue exactly one preregistered, isolated causal benchmark arm.

    This deliberately has no production approval receipt or rollout switch.
    The input is a *pre-result* benchmark spec and the resulting tasks are
    tagged ``ISOLATED_NONPRODUCTION`` by the generator.  Keeping it separate
    from ``bootstrap`` means an operator cannot accidentally turn an approved
    production-select receipt into benchmark authority (or vice versa).
    """

    if not bool(getattr(args, "confirm_isolated_nonproduction", False)):
        raise RuntimeError(
            "isolated benchmark execution requires "
            "--confirm-isolated-nonproduction"
        )
    if os.environ.get("HELENA_ENABLE_SEED_PROBE_SELECT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError(
            "isolated benchmark execution refuses a process configured for "
            "production seed-probe select; use a separate process and control "
            "plane"
        )
    try:
        authorization = load_seed_probe_benchmark_spec(args.benchmark_spec)
    except ValueError as error:
        raise RuntimeError(
            f"seed-probe benchmark spec is invalid: {error}"
        ) from error

    store = open_fleet_store(args.db)
    store.initialize()
    existing = store.status()
    existing_task_count = sum(
        int(count) for count in (existing.get("tasks") or {}).values()
    )
    existing_surface_count = int(existing.get("surfaces") or 0)
    if existing_task_count or existing_surface_count:
        raise RuntimeError(
            "isolated benchmark control plane must have no pre-existing tasks "
            f"or surfaces (tasks={existing_task_count}, "
            f"surfaces={existing_surface_count})"
        )
    authorized_samples = {
        str(cell["sample_id"]) for cell in authorization["cells"]
    }
    # This command starts from a fresh, dedicated control plane.  It imports
    # only the preregistered scrolls, verifies their frozen sources, and never
    # calls normal bootstrap (which could materialize unrelated coverage work).
    sources = bootstrap_sources(
        store, args.eligible, authorized_samples, verify=True
    )
    if set(sources) != authorized_samples:
        raise RuntimeError(
            "benchmark source import did not resolve the exact preregistered "
            "scroll cohort"
        )
    catalog = import_catalog(store, args.catalog, sources)
    snapshots = store.snapshots(authorized_samples)
    max_tasks_per_sample = max(
        sum(
            cell["sample_id"] == sample_id
            for cell in authorization["cells"]
        )
        for sample_id in authorized_samples
    )
    seed_probe = (
        None
        if args.arm == "baseline"
        else default_seed_probe_policy(
            mode="select", benchmark_authorization=authorization
        )
    )
    generation_options = {
        "catalog_snapshot_sha256": file_sha256(args.catalog),
        "grid_step": args.grid_step,
        "query_radius": args.query_radius,
        "clearance": args.clearance,
        "volume_edge_margin": args.volume_edge_margin,
        "candidate_interior_clearance": args.candidate_interior_clearance,
        "selection_strategy": args.selection_strategy,
        # The exact cohort wins over an operator-provided queue size.
        "max_tasks": max_tasks_per_sample,
        "grid_version": args.grid_version,
        "candidate_selection_policy": args.candidate_selection_policy,
        "seed_region_policy": args.seed_region_policy,
        "recenter_probe_max_candidates": args.recenter_probe_max_candidates,
        "recenter_radius_xyz": {
            "x": args.recenter_radius_x,
            "y": args.recenter_radius_y,
            "z": args.recenter_radius_z,
        },
        "ct_material_support_gate": (
            {
                "policy": "ome-zarr-nearby-material-v1",
                "level": args.ct_support_level,
                "radius_l0_voxels": args.ct_support_radius_l0,
                "minimum_nonzero_voxels": args.ct_support_minimum_nonzero_voxels,
            }
            if args.ct_material_support_gate
            else None
        ),
        "queued_reason": "isolated-seed-probe-causal-benchmark-v1",
    }
    tasks = generate_seed_probe_benchmark_arm_tasks(
        store,
        snapshots,
        benchmark_execution_authorization=authorization,
        arm=args.arm,
        generation_options=generation_options,
        seed_probe=seed_probe,
    )
    # create_tasks is transactional on both supported control planes.  Do not
    # loop here: a partial arm is neither a valid benchmark cohort nor a retry.
    inserted, seen = store.create_tasks(tasks)
    receipt = {
        "schema": "campaignx.seed_probe_benchmark_arm_receipt.v1",
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "isolation_attestation": (
            "OPERATOR_ASSERTION_ONLY: separate DB, run/artifact roots, and "
            "credentials are required but not technically proven by this CLI"
        ),
        "benchmark_id": authorization["benchmark_id"],
        "benchmark_spec_sha256": authorization["benchmark_spec_sha256"],
        "arm": args.arm,
        "authorized_samples": sorted(authorized_samples),
        "authorized_cell_count": len(authorization["cells"]),
        "generated_task_count": len(tasks),
        "inserted_task_count": inserted,
        "seen_task_count": seen,
        "sources": sources,
        "catalog": catalog,
        "database": store_identity(store),
    }
    if args.receipt is not None:
        write_json_atomic(args.receipt, receipt)
    print_json(receipt)
    return 0


def command_bootstrap_resume(args: argparse.Namespace) -> int:
    """Queue one grow that continues an existing surface under corrections.

    A drifted surface was terminal: the attempt succeeded, the geometry was
    wrong, and nothing could ask again. The executor has always known how to
    resume -- optional_grow_flags reads resume_from and corrections off the locked
    plan -- and no producer ever wrote them, so this path existed and could not
    be reached.

    The result is a new surface, never an edit. Task identity is
    (source_snapshot, grid_version, cell, policy_version), so the resume rides a
    policy version of its own and the original stays exactly as it was measured.
    Overwriting it would destroy the record of what the fleet actually produced.
    """
    store = open_fleet_store(args.db)
    store.initialize()

    surface = store.surface(args.surface) if hasattr(store, "surface") else None
    if surface is None:
        rows = [s for s in store.surfaces() if s.get("surface_id") == args.surface]
        if not rows:
            raise SystemExit(f"no surface {args.surface!r} on this control plane")
        surface = rows[0]
    if not surface.get("artifact_uri"):
        raise SystemExit(f"{args.surface} has no artifact to resume from")

    snapshots = store.snapshots({str(surface["sample_id"])})
    if not snapshots:
        raise SystemExit(f"no source snapshot for {surface['sample_id']!r}")

    bbox = surface.get("bbox_xyz") or [[0, 0, 0], [0, 0, 0]]
    centre = [int((float(bbox[0][axis]) + float(bbox[1][axis])) / 2) for axis in range(3)]
    task = {
        "schema": "campaignx.segmentation_task.v1",
        "source_snapshot_id": snapshots[0]["source_snapshot_id"],
        "sample_id": surface["sample_id"],
        # The cell names the surface it continues, so two resumes of the same
        # surface under one policy version are the same task rather than two.
        "cell_id": f"resume-{str(args.surface)[:24]}",
        "grid_version": args.grid_version,
        "policy_version": args.policy_version,
        "bounds_xyz": bbox,
        "center_xyz": dict(zip("xyz", centre, strict=True)),
        "priority": 1000.0,
        "parameter_envelope": DEFAULT_ENVELOPE,
        "catalog_snapshot_sha256": surface.get("catalog_snapshot_sha256") or "",
        "resume_from": surface["artifact_uri"],
        "corrections": str(args.corrections),
        "submitted_by": args.submitted_by,
        "resumes_surface": args.surface,
        **({"resume_generations": args.resume_generations}
           if args.resume_generations else {}),
        **({"rewind_gen": args.rewind_gen} if args.rewind_gen is not None else {}),
        "ink_used": False,
    }
    inserted, seen = store.create_tasks([task])
    print(json.dumps({
        "schema": "campaignx.segment_fleet_resume_receipt.v1",
        "resumes_surface": args.surface,
        "corrections": str(args.corrections),
        "policy_version": args.policy_version,
        "generated": seen, "inserted": inserted,
        "note": ("already queued under this policy version"
                 if inserted == 0 else "queued"),
    }, indent=1))
    return 0


def command_bootstrap_manual(args: argparse.Namespace) -> int:
    """Queue growth from points a person supplied.

    Reads a JSON list of {x, y, z} in CT-L0 voxels, or the same as CSV. One task
    per point, and no coverage ranking: the caller has already said where.

    Deliberately a separate command from `bootstrap`. That one decides where to
    look; this one is told. Sharing an entry point would mean the flag that
    supplies points sitting beside twenty that rank cells nobody asked it to
    rank.
    """
    store = open_fleet_store(args.db)
    store.initialize()
    points = read_seed_points(args.points)
    if not points:
        raise SystemExit(f"{args.points} contains no points")
    snapshots = store.snapshots({args.sample})
    if not snapshots:
        raise SystemExit(
            f"no source snapshot for {args.sample!r}; run bootstrap for it first, "
            "because a manual seed still needs the volume it refers to")
    generated: dict[str, dict[str, int]] = {}
    for snapshot in snapshots:
        tasks = generate_manual_tasks(
            store,
            snapshot,
            points,
            catalog_snapshot_sha256=file_sha256(args.catalog),
            grid_step=args.grid_step,
            query_radius=args.query_radius,
            volume_edge_margin=args.volume_edge_margin,
            grid_version=args.grid_version,
            policy_version=args.policy_version,
            submitted_by=args.submitted_by,
            mission_id=args.mission_id,
            planner=args.planner,
            planner_model=args.planner_model,
            ct_material_support_gate=(
                {
                    "policy": "ome-zarr-nearby-material-v1",
                    "level": args.ct_support_level,
                    "radius_l0_voxels": args.ct_support_radius_l0,
                    "minimum_nonzero_voxels": args.ct_support_minimum_nonzero_voxels,
                }
                if args.ct_material_support_gate
                else None
            ),
        )
        inserted, seen = store.create_tasks(tasks)
        generated[snapshot["sample_id"]] = {"generated": seen, "inserted": inserted}
    receipt = {
        "schema": "campaignx.segment_fleet_manual_seed_receipt.v1",
        "sample_id": args.sample,
        "points": len(points),
        "tasks": generated,
        "grid_version": args.grid_version,
        "policy_version": args.policy_version,
        "submitted_by": args.submitted_by,
        "seed_origin": "human",
        "non_claim": "A surface grown from a supplied point is not evidence the "
                     "fleet found it. The seed origin is recorded on every task "
                     "so the two populations stay separable.",
        "generated_at_utc": utc_now(),
    }
    output = args.receipt or Path("SEGMENT_FLEET_MANUAL_SEED_RECEIPT.json")
    write_json_atomic(output, receipt)
    print_json({"receipt": str(output), **receipt})
    return 0


def read_seed_points(path: Path) -> list[dict[str, Any]]:
    """JSON or CSV, because a person with coordinates has them in one of the two.

    CSV needs a header naming x, y and z; guessing column order from three
    numbers is how a y ends up in a z.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith(("[", "{")):
        loaded = json.loads(text)
        rows = loaded.get("points", []) if isinstance(loaded, dict) else loaded
        if not isinstance(rows, list):
            raise SystemExit("JSON seed points must be a list, or {\"points\": [...]}")
        return [dict(row) for row in rows if isinstance(row, dict)]
    import csv
    import io

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"x", "y", "z"} <= {
            name.strip().lower() for name in reader.fieldnames}:
        raise SystemExit(
            "CSV seed points need a header row naming x, y and z; column order "
            "cannot be guessed from three numbers")
    return [{key.strip().lower(): value for key, value in row.items() if key}
            for row in reader]


def command_bootstrap_seed_recovery(args: argparse.Namespace) -> int:
    receipt = bootstrap_seed_positive_recovery_queue(
        FleetStore(args.source_db),
        FleetStore(args.db),
        args.source_attempt_root,
        grid_version=args.grid_version,
        policy_version=args.policy_version,
        seed_region_policy=args.seed_region_policy,
        candidate_selection_policy=args.candidate_selection_policy,
        ct_material_support_gate=(
            {
                "policy": "ome-zarr-nearby-material-v1",
                "level": args.ct_support_level,
                "radius_l0_voxels": args.ct_support_radius_l0,
                "minimum_nonzero_voxels": args.ct_support_minimum_nonzero_voxels,
            }
            if args.ct_material_support_gate
            else None
        ),
    )
    write_json_atomic(args.receipt, receipt)
    print_json({"receipt": str(args.receipt), **receipt["status"]})
    return 0


def command_bootstrap_adaptive_retry(args: argparse.Namespace) -> int:
    receipt = bootstrap_adaptive_retry_task(
        open_fleet_store(args.db),
        args.source_task_id,
        grid_version=args.grid_version,
        policy_version=args.policy_version,
        seed_region_policy=args.seed_region_policy,
        minimum_cell_interior_clearance_voxels=args.candidate_interior_clearance,
    )
    write_json_atomic(args.receipt, receipt)
    print_json({"receipt": str(args.receipt), **receipt})
    return 0


def seed_positive_count(results: list[dict[str, Any]]) -> int:
    return sum(
        isinstance(row.get("candidate_count"), int) and int(row["candidate_count"]) > 0
        for row in results
    )


def command_survey(args: argparse.Namespace) -> int:
    """Probe m7 candidates for pending tasks without taking a task lease."""
    store = open_fleet_store(args.db)
    store.initialize()
    provider = McpSeedProvider()
    if args.task_id:
        task = store.task_packet(args.task_id)
        if task.get("state") != "PENDING":
            raise RuntimeError("task-specific survey requires a pending task and never claims it")
        tasks = [task]
    else:
        tasks = store.pending_tasks(args.limit)

    def survey_one(task: dict[str, Any]) -> dict[str, Any]:
        try:
            response = provider.discover(task)
            screened = screen_candidates(response, task)
            return {
                "task_id": task["task_id"],
                "sample_id": task["sample_id"],
                "cell_id": task["cell_id"],
                "priority": task["priority"],
                "candidate_count": screened["raw_candidate_count"],
                "usable_candidate_count": screened["usable_candidate_count"],
                "best_candidate": screened["best_candidate"],
                "response_sha256": content_sha256(response),
                "response": response,
                "ink_used": False,
            }
        except BaseException as error:
            return {
                "task_id": task["task_id"],
                "sample_id": task["sample_id"],
                "cell_id": task["cell_id"],
                "priority": task["priority"],
                "candidate_count": None,
                "error": f"{type(error).__name__}: {error}",
                "ink_used": False,
            }
    if args.parallelism < 1:
        raise ValueError("parallelism must be at least one")
    if args.progress:
        write_json_atomic(args.progress, {
            "schema": "campaignx.segment_fleet_m7_survey_progress.v1",
            "status": "RUNNING",
            "started_at_utc": utc_now(),
            "requested_count": len(tasks),
            "completed_count": 0,
            "seed_positive_count": 0,
            "source_error_count": 0,
            "state_mutation": "NONE",
        })
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = {pool.submit(survey_one, task): index for index, task in enumerate(tasks)}
        results: list[dict[str, Any] | None] = [None] * len(tasks)
        for completed_count, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if args.progress:
                completed = [row for row in results if row is not None]
                write_json_atomic(args.progress, {
                    "schema": "campaignx.segment_fleet_m7_survey_progress.v1",
                    "status": "RUNNING",
                    "updated_at_utc": utc_now(),
                    "requested_count": len(tasks),
                    "completed_count": completed_count,
                    "seed_positive_count": seed_positive_count(completed),
                    "source_error_count": sum("error" in row for row in completed),
                    "state_mutation": "NONE",
                })
    results = [row for row in results if row is not None]
    eligible = [row for row in results if int(row.get("usable_candidate_count") or 0) > 0]
    eligible.sort(key=lambda row: (
        -float((row.get("best_candidate") or {}).get("score", 0.0)),
        -float((row.get("best_candidate") or {}).get("cell_interior_clearance_voxels", 0.0)),
        -float((row.get("best_candidate") or {}).get("volume_interior_clearance_voxels", 0.0)),
        str(row["task_id"]),
    ))
    receipt = {
        "schema": "campaignx.segment_fleet_m7_survey.v1",
        "generated_at_utc": utc_now(),
        "database": store_identity(store),
        "requested_limit": args.limit,
        "requested_task_id": args.task_id,
        "parallelism": args.parallelism,
        "surveyed_count": len(results),
        "seed_positive_count": seed_positive_count(results),
        "growth_eligible_count": len(eligible),
        "recommended_task_ids": [row["task_id"] for row in eligible],
        "state_mutation": "NONE",
        "ink_used": False,
        "results": results,
    }
    write_json_atomic(args.output, receipt)
    if args.progress:
        write_json_atomic(args.progress, {
            "schema": "campaignx.segment_fleet_m7_survey_progress.v1",
            "status": "COMPLETE",
            "completed_at_utc": receipt["generated_at_utc"],
            "requested_count": len(tasks),
            "completed_count": len(results),
            "seed_positive_count": receipt["seed_positive_count"],
            "source_error_count": sum("error" in row for row in results),
            "output": str(args.output),
            "state_mutation": "NONE",
        })
    print_json({"output": str(args.output), "surveyed_count": receipt["surveyed_count"], "seed_positive_count": receipt["seed_positive_count"]})
    return 0


PLANNER_CHOICES = ("deterministic", "deterministic-v2", "cost-aware-v2",
                   "fusion-v2", "opencode", "opencode-v2")


def build_planner(name: str, args: argparse.Namespace,
                  model: str | None = None) -> Planner:
    """The planner that goes by this name, built with this host's settings.

    `model` is the task's own choice when it made one -- otherwise picking a
    model when queueing a run would set a field nothing reads.
    """
    if name in {"deterministic", "deterministic-v2"}:
        return DeterministicPlanner(
            contract_version="v2" if name == "deterministic-v2" else "v1"
        )
    if name == "cost-aware-v2":
        return CostAwareSegmentationPlanner(
            api_key_env=args.openrouter_key_env,
            timeout_seconds=args.planner_timeout,
        )
    if name == "fusion-v2":
        return OpenRouterFusionPlanner(
            api_key_env=args.openrouter_key_env,
            timeout_seconds=args.planner_timeout,
        )
    if name in {"opencode", "opencode-v2"}:
        executable = args.opencode or shutil.which("opencode")
        if not executable:
            raise RuntimeError("opencode executable was not found")
        return OpenCodePlanner(
            executable,
            args.repo_root,
            model=model or args.model,
            timeout_seconds=args.planner_timeout,
            contract_version="v2" if name == "opencode-v2" else "v1",
        )
    raise RuntimeError(f"unknown planner: {name}")


def adopt_fleet_secrets(store: Any) -> list[str]:
    """Put the fleet's credentials into this process's environment.

    A worker is ephemeral: it should be able to start from a database URL and
    nothing else. Before this, reaching object storage meant a file placed by
    hand on each machine, on tmpfs on the one host that had it, so a reboot took
    surface QC and the ink worker down until somebody remembered.

    The environment already set wins. An operator who exported a key for one run
    means it, and a value from the control plane silently overriding it is a
    debugging session about which credential is actually in use.
    """
    try:
        held = store.secrets()
    except Exception as failure:  # noqa: BLE001
        # An older control plane has no table for these, and a worker that can
        # segment without object storage should still start.
        print(f"fleet credentials unavailable: {type(failure).__name__}: {failure}",
              file=sys.stderr, flush=True)
        return []
    adopted = []
    for name, value in sorted(held.items()):
        if os.environ.get(name):
            continue
        os.environ[name] = value
        adopted.append(name)
    if adopted:
        print(f"credentials from the control plane: {', '.join(adopted)}", flush=True)
    return adopted


def refuse_stranded_artifacts(artifact_root: str, allowed: bool) -> None:
    """A worker is ephemeral; what it produces must not be.

    A surface published to a directory on the worker carries that path as its
    artifact_uri, and every phase downstream resolves it against its own
    filesystem. Surface QC on another host requeues it every five minutes and
    reads as a scientific failure; P3 cannot fetch it; and when the worker goes
    away so does the only copy. Four surfaces were grown that way before this
    check existed, and had to be copied host-to-host by hand to be measured.

    State belongs in the control plane and artifacts belong in object storage.
    Local publication stays available for a single-machine run, but it has to be
    asked for, because the failure it causes appears somewhere else entirely.
    """
    if str(artifact_root).startswith(("s3://", "http://", "https://")):
        return
    if allowed:
        return
    raise RuntimeError(
        f"--artifact-root {artifact_root!r} is a local path. A worker is "
        "ephemeral: a surface published there is invisible to QC, to flattening "
        "and to rendering on any other host, and is lost with the machine. Give "
        "an s3:// prefix, or pass --allow-local-artifacts for a single-machine "
        "run.")


def _worker(args: argparse.Namespace) -> SegmentWorker:
    refuse_stranded_artifacts(args.artifact_root,
                              getattr(args, "allow_local_artifacts", False))
    benchmark_authorization = None
    benchmark_spec_path = getattr(args, "isolated_benchmark_spec", None)
    if benchmark_spec_path is not None:
        if not bool(getattr(args, "confirm_isolated_nonproduction", False)):
            raise RuntimeError(
                "isolated benchmark worker requires "
                "--confirm-isolated-nonproduction"
            )
        if os.environ.get("HELENA_ENABLE_SEED_PROBE_SELECT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError(
                "isolated benchmark worker refuses a process configured for "
                "production seed-probe select; use separate DB, run/artifact "
                "roots, and credentials"
            )
        try:
            benchmark_authorization = load_seed_probe_benchmark_spec(
                benchmark_spec_path
            )
        except ValueError as error:
            raise RuntimeError(
                f"isolated benchmark spec is invalid: {error}"
            ) from error
    store = open_fleet_store(args.db)
    store.initialize()
    adopt_fleet_secrets(store)
    if args.seed_provider == "recorded":
        seed_provider = RecordedSeedProvider()
    elif args.seed_provider == "manual":
        seed_provider = ManualSeedProvider()
    else:
        # Wrapped, so a task that names the manual provider is honoured whatever
        # this worker was started with. A host should not need restarting to grow
        # a point somebody uploaded.
        seed_provider = TaskRoutedSeedProvider(McpSeedProvider())
    planner = build_planner(args.planner, args)
    if args.fixture_grow:
        if not args.allow_fixture:
            raise RuntimeError("fixture grow requires --allow-fixture and cannot produce scientific geometry")
        executor = FixtureGrowExecutor()
    else:
        if args.vc3d_binary is None:
            raise RuntimeError("scientific worker requires --vc3d-binary")
        executor = VC3DGrowExecutor(args.vc3d_binary, timeout_seconds=args.grow_timeout, minimum_free_gib=args.minimum_free_gib)
    seed_probe_support = bool(args.seed_probe_support) and os.environ.get(
        "HELENA_DISABLE_SEED_PROBE_V1", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}
    worker_capabilities = {
        "cuda_available": args.cuda_available,
        "gpu_model": args.gpu_model,
        "gpu_vram_gb": args.gpu_vram_gb,
        "cuda_device_index": args.cuda_device_index,
        "seed_probe_v1": seed_probe_support,
        **(
            {
                "benchmark_spec_sha256": benchmark_authorization[
                    "benchmark_spec_sha256"
                ]
            }
            if benchmark_authorization is not None
            else {}
        ),
    }
    return SegmentWorker(
        store,
        args.worker_id,
        seed_provider,
        planner,
        executor,
        args.run_root,
        args.artifact_root,
        args.qc_profile_id,
        lease_seconds=args.lease_seconds,
        provider_retry_delay_seconds=args.provider_retry_delay_seconds,
        source_retry_delay_seconds=args.source_retry_delay_seconds,
        finalization_retry_delay_seconds=(
            args.finalization_retry_delay_seconds
        ),
        probe_artifact_max_requeues=args.probe_artifact_max_requeues,
        finalization_max_requeues=args.finalization_max_requeues,
        task_id=args.task_id,
        worker_capabilities=worker_capabilities,
        # So a task that names a planner is grown by that planner, whatever this
        # host was started with. Without it the choice made when a run is queued
        # is a label on a row and the host default does all the work.
        planner_factory=lambda name, model=None: build_planner(name, args, model),
        seed_probe_support=seed_probe_support,
    )


def command_worker_run(args: argparse.Namespace) -> int:
    worker = _worker(args)
    results = worker.run(max_jobs=args.max_jobs or None, idle_exit=not args.watch, poll_seconds=args.poll_seconds)
    print_json({"schema": "campaignx.segment_fleet_worker_run.v1", "worker_id": args.worker_id, "worker_capabilities": worker.worker_capabilities, "completed": len(results), "results": results, "status": worker.store.status()})
    completed_states = {
        "QC_PENDING",
        "FIXTURE_ONLY",
        "DUPLICATE_SURFACE",
        "RETRY_ON_LARGER_GPU",
    }
    if args.terminal_outcomes_exit_zero:
        # A bounded fleet supervisor must distinguish a valid scientific
        # terminal outcome from an infrastructure failure.  These outcomes
        # consume the claimed task safely but intentionally produce no new
        # surface; transient provider/source failures and execution failures
        # remain non-zero so the supervisor can fail closed.
        completed_states.update(
            {
                "NO_SEED",
                "POLICY_REJECTED",
                "BLOCKED_SOURCE_UNAVAILABLE",
                "PROBE_REVIEW_PENDING",
                "PROBE_REJECTED_ALL",
            }
        )
    return 0 if all(result.get("status") in completed_states for result in results) else 2 if results else 0


def command_status(args: argparse.Namespace) -> int:
    print_json(open_fleet_store(args.db).status())
    return 0


def command_certify(args: argparse.Namespace) -> int:
    """P2 over the corpus, not just over surfaces on their way out."""
    store = open_fleet_store(args.db)
    store.initialize()
    receipt = certify_pending(
        store,
        workspace=args.workspace,
        limit=args.limit,
        sample_id=args.sample,
        dry_run=args.dry_run,
    )
    if args.receipt:
        write_json_atomic(args.receipt, receipt)
    print_json(receipt)
    # A run that measured nothing is not a failure: it is an empty backlog.
    return 0


def flatten_batch_exit_code(states: dict[str, int]) -> int:
    """Translate P3's per-surface outcomes into the worker process status.

    A measured area rejection is a valid scientific outcome.  An operational
    failure is not: returning zero for it makes the queue report a successful
    job even though no consumable sheet exists.
    """
    return 2 if states.get("FLATTENING_FAILED", 0) else 0


def command_flatten(args: argparse.Namespace) -> int:
    """P3: unroll certified surfaces into flat sheets."""
    import shutil
    import sys as _sys
    import tempfile

    _sys.path.insert(0, str(args.repo_root / "framework/stages/02-flattening"))
    from flatten import (  # noqa: PLC0415
        FlatteningFailed,
        SurfaceNotAdmissible,
        SurfaceNotCertified,
        flatten_surface,
        load_profile,
    )

    store = open_fleet_store(args.db)
    store.initialize()
    profile = load_profile(args.profile)
    pending = store.surfaces_awaiting_flattening(
        profile["profile_id"], limit=args.limit, sample_id=args.sample,
        require_physical_qc=not args.allow_unvalidated,
        surface_id=args.surface_id)
    if args.dry_run:
        print_json({"schema": "campaignx.surface_flattening_run.v1", "dry_run": True,
                    "profile_id": profile["profile_id"], "considered": len(pending),
                    "surfaces": [s["surface_id"] for s in pending]})
        return 0

    adapter = load_qc_adapter()
    artifacts = open_artifact_store(args.artifact_store) if args.artifact_store else None
    root = Path(args.workspace) if args.workspace else Path(
        tempfile.mkdtemp(prefix="p3-flatten-"))
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for surface in pending:
        surface_id = str(surface["surface_id"])
        work = root / surface_id.replace("/", "_").replace(":", "_")
        source, flat = work / "surface", work / "flat"
        try:
            adapter.materialize_surface(
                str(surface["artifact_uri"]), str(surface["artifact_sha256"] or ""),
                source)
            receipt = flatten_surface(
                source, flat,
                binary=args.binary, profile=profile,
                voxel_size_um=float(surface["voxel_size_um"]),
                geometry_qc_state=str(surface["geometry_qc_state"]),
                physical_qc_state=surface.get("physical_qc_state"),
                require_physical_qc=not args.allow_unvalidated,
                timeout_seconds=args.timeout)
            # Publish only what P4 may consume. A sheet rejected on area is
            # still a measurement worth recording, and publishing it would put
            # it in the queue P4 reads.
            if artifacts is not None and receipt.get("status") == "FLATTENED":
                # The same manifest shape the finalizer publishes a grown
                # surface with: per-file size and digest, plus one digest over
                # the set. The S3 backend puts artifact_sha256 in the object
                # metadata of ARTIFACT_SET.json, so a manifest without it
                # publishes nothing and reports KeyError.
                files = {name: {"sha256": digest,
                                "size_bytes": (flat / name).stat().st_size}
                         for name, digest in receipt["output"].items()}
                manifest = {
                    "schema": "campaignx.flattened_artifact_set.v1",
                    "surface_id": surface_id,
                    "profile_id": profile["profile_id"],
                    "files": files,
                    "artifact_sha256": content_sha256(files),
                }
                receipt["artifact_sha256"] = manifest["artifact_sha256"]
                staged = artifacts.stage(flat, f"flat-{surface_id}", manifest)
                promoted = artifacts.promote(
                    staged, str(surface["sample_id"]), f"flat/{surface_id}", manifest)
                receipt["artifact_uri"] = promoted["artifact_uri"]
        # Every failure, not just the two named ones: one surface that cannot
        # be flattened must not end a batch, and the reason is recorded either
        # way. SurfaceNotCertified and FlatteningFailed are imported because
        # they are what this raises deliberately.
        except Exception as failure:  # noqa: BLE001
            receipt = {
                "schema": "campaignx.surface_flattening_receipt.v1",
                "state": "FLATTENING_FAILED",
                "profile_id": profile["profile_id"],
                "error": f"{type(failure).__name__}: {failure}",
                "generated_at_utc": utc_now(),
                "ink_used": False,
                "non_claims": ["a failed flattening says nothing about the surface"],
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)
        receipt.setdefault("state", receipt.get("status", "FLATTENED"))
        receipt["surface_id"] = surface_id
        results.append({**store.record_flattening(receipt),
                        "surface_id": surface_id,
                        "area_ratio": receipt.get("area_ratio"),
                        "error": receipt.get("error")})
    states: dict[str, int] = {}
    for result in results:
        states[str(result["state"])] = states.get(str(result["state"]), 0) + 1
    receipt = {"schema": "campaignx.surface_flattening_run.v1",
               "profile_id": profile["profile_id"], "considered": len(pending),
               "flattened": states, "surfaces": results,
               "generated_at_utc": utc_now()}
    if args.receipt:
        write_json_atomic(args.receipt, receipt)
    print_json(receipt)
    return flatten_batch_exit_code(states)


def command_republish(args: argparse.Namespace) -> int:
    """Move surfaces off a worker's disk and into object storage.

    A surface published to a local path exists on one machine. Every other host
    reads its artifact_uri and finds nothing, and the phase that consumes it
    fails with FileNotFoundError while the files sit where they were written.

    Run this where the files are. The digest is not recomputed from the copy --
    it is verified against the one the surface was recorded with, because a
    republish that quietly changed the digest would be a different artifact
    wearing the verdicts of the original.
    """
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    store = open_fleet_store(args.db)
    store.initialize()
    stranded = store.surfaces_on_local_artifacts(limit=args.limit, sample_id=args.sample)
    if args.dry_run:
        print_json({"schema": "campaignx.surface_republish_receipt.v1", "dry_run": True,
                    "considered": len(stranded),
                    "surfaces": [{"surface_id": s["surface_id"],
                                  "artifact_uri": s["artifact_uri"]} for s in stranded]})
        return 0

    adapter = load_qc_adapter()
    artifacts = open_artifact_store(args.artifact_store)
    root = Path(tempfile.mkdtemp(prefix="republish-"))
    moved: list[dict[str, Any]] = []
    for surface in stranded:
        surface_id = str(surface["surface_id"])
        staging = root / surface_id.replace("/", "_").replace(":", "_")
        try:
            # The same fetch-and-verify P2 and P3 use. A local path that no
            # longer holds the artifact fails here, which is the honest outcome:
            # those bytes are gone and no republish can recover them.
            adapter.materialize_surface(str(surface["artifact_uri"]),
                                        str(surface["artifact_sha256"] or ""), staging)
            files = {name: {"sha256": file_sha256(staging / name),
                            "size_bytes": (staging / name).stat().st_size}
                     for name in ("x.tif", "y.tif", "z.tif", "meta.json")}
            manifest = {"schema": "campaignx.segmentation_artifact_set.v1",
                        "surface_id": surface_id, "files": files,
                        "artifact_sha256": str(surface["artifact_sha256"] or "")
                        or content_sha256(files)}
            staged = artifacts.stage(staging, f"republish-{surface_id}", manifest)
            promoted = artifacts.promote(staged, str(surface["sample_id"]),
                                         surface_id, manifest)
            store.repoint_surface_artifact(surface_id, promoted["artifact_uri"],
                                           manifest["artifact_sha256"])
            moved.append({"surface_id": surface_id, "state": "REPUBLISHED",
                          "from": surface["artifact_uri"],
                          "artifact_uri": promoted["artifact_uri"]})
        except Exception as failure:  # noqa: BLE001 -- one lost surface is not the batch
            moved.append({"surface_id": surface_id, "state": "REPUBLISH_FAILED",
                          "from": surface["artifact_uri"],
                          "error": f"{type(failure).__name__}: {failure}"})
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    states: dict[str, int] = {}
    for entry in moved:
        states[entry["state"]] = states.get(entry["state"], 0) + 1
    print_json({"schema": "campaignx.surface_republish_receipt.v1",
                "considered": len(stranded), "states": states, "surfaces": moved,
                "generated_at_utc": utc_now()})
    return 0


def command_replan(args: argparse.Namespace) -> int:
    """Ask the cells that gave no seed again, under a different policy."""
    from .generator import replan_no_seed_cells  # noqa: PLC0415

    store = open_fleet_store(args.db)
    print_json(replan_no_seed_cells(
        store, grid_version=args.grid_version, policy_version=args.policy_version,
        planner=args.planner, planner_model=args.planner_model,
        sample_id=args.sample, causes=args.cause or None,
        limit=args.limit, dry_run=args.dry_run))
    return 0


def command_coverage(args: argparse.Namespace) -> int:
    """How much of a scroll has been looked at, and with what result."""
    store = open_fleet_store(args.db)
    store.initialize()
    print_json(store.coverage(args.sample))
    return 0


def command_qc_run(args: argparse.Namespace) -> int:
    store = open_fleet_store(args.db)
    store.initialize()
    if args.fixture_qc:
        if not args.allow_fixture:
            raise RuntimeError("fixture QC requires --allow-fixture and has no scientific validity")
        executor = FixtureQcExecutor(args.fixture_outcome)
    else:
        if args.qc_executable is None:
            raise RuntimeError("scientific QC requires --qc-executable")
        executor = SubprocessQcExecutor(
            args.qc_executable, timeout_seconds=args.qc_timeout
        )
    worker = SurfaceQcWorker(
        store,
        args.worker_id,
        executor,
        args.run_root,
        lease_seconds=args.lease_seconds,
        retry_delay_seconds=args.retry_delay_seconds,
        profile_id=args.profile_id,
    )
    results = worker.run(
        max_jobs=args.max_jobs or None,
        idle_exit=not args.watch,
        poll_seconds=args.poll_seconds,
    )
    print_json(
        {
            "schema": "campaignx.segment_qc_worker_run.v1",
            "worker_id": args.worker_id,
            "completed": len(results),
            "results": results,
            "status": store.status(),
        }
    )
    return 0 if all(result.get("status") == "COMPLETED" for result in results) else 2 if results else 0


def command_qc_enqueue_backfill(args: argparse.Namespace) -> int:
    """Idempotently turn a verified TIFXYZ manifest into QC work."""

    store = open_fleet_store(args.db)
    store.initialize()
    selected_samples = set(args.sample) or None
    selected_surface_ids = set(args.surface_id) or None
    sources = bootstrap_sources(store, args.eligible, selected_samples)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "campaignx.surface_qc_backfill_manifest.v1":
        raise RuntimeError("unsupported surface-QC backfill manifest")
    manifest_sha256 = content_sha256(manifest)
    rows = manifest.get("surfaces")
    if not isinstance(rows, list):
        raise RuntimeError("surface-QC backfill manifest has no surface list")
    results: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("surface-QC backfill row is not an object")
        sample_id = str(row.get("sample_id", ""))
        if selected_samples and sample_id not in selected_samples:
            continue
        surface_id = str(row.get("surface_id", ""))
        if selected_surface_ids and surface_id not in selected_surface_ids:
            continue
        if sample_id not in sources:
            raise RuntimeError(f"backfill surface has no source snapshot: {sample_id}")
        payload = {
            "schema": "campaignx.segment_fleet_surface_backfill.v1",
            "surface_id": surface_id,
            "source_snapshot_id": sources[sample_id],
            "sample_id": sample_id,
            "owner": row.get("owner", "campaign-x"),
            "artifact_sha256": str(row["artifact_sha256"]),
            "artifact_uri": str(row["artifact_uri"]),
            "bbox_xyz": row["bbox_xyz"],
            "area_cm2": row.get("area_cm2"),
            "state": "QC_PENDING",
            "physical_qc_state": "UNVALIDATED",
            "backfill_manifest_sha256": manifest_sha256,
            "legacy_tifxyz_sha256": row.get("legacy_tifxyz_sha256"),
            "artifact_manifest_sha256": row.get("artifact_manifest_sha256"),
        }
        results.append(
            store.enqueue_imported_surface_qc(
                payload,
                profile_id=args.profile_id,
                job_payload={
                    "kind": "campaignx.surface_qc_backfill_job.v1",
                    "backfill_manifest_sha256": manifest_sha256,
                    "artifact_manifest_sha256": row.get("artifact_manifest_sha256"),
                    "no_automatic_acceptance": True,
                },
            )
        )
    receipt = {
        "schema": "campaignx.surface_qc_backfill_enqueue_receipt.v1",
        "generated_at_utc": utc_now(),
        "database": store_identity(store),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "eligible_sha256": file_sha256(args.eligible),
        "profile_id": args.profile_id,
        "selected_samples": sorted(selected_samples) if selected_samples else None,
        "selected_surface_ids": (
            sorted(selected_surface_ids) if selected_surface_ids else None
        ),
        "counts": {
            "seen": len(results),
            "enqueued": sum(row["status"] == "ENQUEUED" for row in results),
            "already_enqueued": sum(
                row["status"] == "ALREADY_ENQUEUED" for row in results
            ),
            "inserted": sum(
                row.get("reconciliation") == "INSERTED" for row in results
            ),
            "reconciled_unvalidated": sum(
                row.get("reconciliation") == "RECONCILED_UNVALIDATED"
                for row in results
            ),
        },
        "results": results,
        "status": store.status(),
        "ink_used": False,
        "no_automatic_acceptance": True,
    }
    write_json_atomic(args.receipt, receipt)
    print_json({"receipt": str(args.receipt), **receipt["counts"], "status": receipt["status"]})
    return 0


def command_worker_recover_provider(args: argparse.Namespace) -> int:
    store = open_fleet_store(args.db)
    store.initialize()
    result = store.recover_terminal_provider_outage(
        args.task_id,
        args.attempt_id,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print_json(result)
    return 0


def command_worker_recover_mcp_auth(args: argparse.Namespace) -> int:
    store = open_fleet_store(args.db)
    store.initialize()
    result = store.recover_terminal_mcp_auth_outage(
        args.task_id,
        args.attempt_id,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print_json(result)
    return 0


def command_worker_recover_finalizer_dependency(args: argparse.Namespace) -> int:
    store = open_fleet_store(args.db)
    store.initialize()
    result = store.recover_terminal_finalizer_dependency(
        args.task_id,
        args.attempt_id,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print_json(result)
    return 0


def command_demo(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"demo root must be new or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    store = FleetStore(root / "fleet.sqlite")
    store.initialize()
    source_id = store.register_snapshot({
        "schema": "campaignx.source_snapshot.v1",
        "sample_id": "PHercDEMO",
        "ct_uri": "fixture://ct",
        "ct_sha256": "0" * 64,
        "m7_uri": "fixture://m7",
        "m7_sha256": "1" * 64,
        "shape_xyz": [512, 512, 512],
        "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz",
        "fixture_only": True,
    })
    task = {
        "schema": "campaignx.segmentation_task.v1",
        "source_snapshot_id": source_id,
        "sample_id": "PHercDEMO",
        "cell_id": "demo-r00000c00000a00000",
        "grid_version": "demo-grid-v1",
        "policy_version": "demo-policy-v1",
        "candidate_selection_policy": (
            "adaptive-geometry-history-v2"
            if args.planner in {"opencode-v2", "fusion-v2", "cost-aware-v2", "deterministic-v2"}
            else "score-cell-volume-clearance-v1"
        ),
        "planner_contract_version": (
            "v2"
            if args.planner in {"opencode-v2", "fusion-v2", "cost-aware-v2", "deterministic-v2"}
            else "v1"
        ),
        "bounds_xyz": [[192, 192, 192], [320, 320, 320]],
        "center_xyz": {"x": 256, "y": 256, "z": 256},
        "priority": 1.0,
        "distance_to_known_aabb_voxels": 512.0,
        "parameter_envelope": DEFAULT_ENVELOPE,
        "catalog_snapshot_sha256": "2" * 64,
        "recorded_candidates": [
            {"candidate_id": "c01", "coordinate": {"x": 256, "y": 256, "z": 256}, "score": 0.9},
            {"candidate_id": "c02", "coordinate": {"x": 224, "y": 240, "z": 256}, "score": 0.5},
        ],
        "fixture_only": True,
        "ink_used": False,
    }
    store.create_tasks([task])
    if args.planner in {"opencode", "opencode-v2"}:
        executable = args.opencode or shutil.which("opencode")
        if not executable:
            raise RuntimeError("opencode executable was not found")
        demo_planner = OpenCodePlanner(
            executable,
            args.repo_root,
            model=args.model,
            timeout_seconds=args.planner_timeout,
            contract_version="v2" if args.planner == "opencode-v2" else "v1",
        )
    elif args.planner == "cost-aware-v2":
        demo_planner = CostAwareSegmentationPlanner(
            api_key_env=args.openrouter_key_env,
            timeout_seconds=args.planner_timeout,
        )
    elif args.planner == "fusion-v2":
        demo_planner = OpenRouterFusionPlanner(
            api_key_env=args.openrouter_key_env,
            timeout_seconds=args.planner_timeout,
        )
    else:
        demo_planner = DeterministicPlanner(
            contract_version="v2" if args.planner == "deterministic-v2" else "v1"
        )
    worker = SegmentWorker(
        store,
        "demo-worker",
        RecordedSeedProvider(),
        demo_planner,
        FixtureGrowExecutor(),
        root / "runs",
        root / "surfaces",
        "fixture-surface-qc@1.0.0",
        lease_seconds=60,
    )
    result = worker.run_one()
    receipt = {
        "schema": "campaignx.segment_fleet_demo.v1",
        "planner": args.planner,
        "planner_model": args.model if args.planner in {"opencode", "opencode-v2"} else None,
        "result": result,
        "status": store.status(),
        "fixture_only": True,
    }
    write_json_atomic(root / "DEMO_RECEIPT.json", receipt)
    print_json({"demo_root": str(root), **receipt})
    return 0 if result and result.get("status") == "FIXTURE_ONLY" else 2


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="helena-segment-fleet", description="Autonomous ink-blind Stage 01 segment search worker.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    init = subcommands.add_parser("init", help="Initialize SQLite or PostgreSQL control plane.")
    init.add_argument("--db", required=True, help="SQLite path or postgres-env://ENV_NAME (preferred for PostgreSQL).")
    init.set_defaults(handler=command_init)

    probe_gc = subcommands.add_parser(
        "probe-gc",
        help=(
            "List expired noncanonical seed-probe bytes, or delete their exact "
            "content-addressed namespaces with --apply."
        ),
    )
    probe_gc.add_argument("--db", required=True)
    probe_gc.add_argument(
        "--artifact-root",
        required=True,
        help="the same local path or s3://bucket/prefix used by probe workers",
    )
    probe_gc.add_argument(
        "--limit", type=int, default=100, choices=range(1, 1001)
    )
    probe_gc.add_argument(
        "--apply",
        action="store_true",
        help="perform deletion; without this flag the command is a dry run",
    )
    probe_gc.add_argument(
        "--expect-candidate-set-sha256",
        help=(
            "required with --apply; must equal the candidate_set_sha256 from "
            "a current dry run so the reviewed and deleted sets are identical"
        ),
    )
    probe_gc.add_argument("--receipt", type=Path)
    probe_gc.set_defaults(handler=command_probe_gc)

    bootstrap = subcommands.add_parser("bootstrap", help="Import sources/catalogue and generate uncovered spatial tasks deterministically.")
    bootstrap.add_argument(
        "--no-verify-sources", action="store_true",
        help="skip the HEAD on each frozen URI. The check costs a second and "
             "catches at intake what otherwise surfaces as a worker dying on "
             "HTTP 401 a week later, one burned attempt at a time")
    bootstrap.add_argument("--db", required=True)
    bootstrap.add_argument("--eligible", type=Path, required=True)
    bootstrap.add_argument("--catalog", type=Path, required=True)
    bootstrap.add_argument("--receipt", type=Path)
    bootstrap.add_argument("--sample", action="append", default=[])
    bootstrap.add_argument("--grid-step", type=int, default=2048)
    bootstrap.add_argument("--query-radius", type=int, default=64)
    bootstrap.add_argument("--clearance", type=float, default=256.0)
    bootstrap.add_argument("--volume-edge-margin", type=int, default=64, help="Keep query cells this far from every CT volume boundary.")
    bootstrap.add_argument("--candidate-interior-clearance", type=int, default=0, help="Require this candidate clearance from its claimed query-cell faces.")
    bootstrap.add_argument("--selection-strategy", choices=("max-clearance-v1", "stratified-clearance-v1"), default="max-clearance-v1")
    bootstrap.add_argument("--max-tasks-per-sample", type=int, default=24)
    bootstrap.add_argument("--grid-version", default="ct-l0-grid-2048-v1")
    bootstrap.add_argument("--policy-version", default="ink-blind-v1")
    bootstrap.add_argument(
        "--candidate-selection-policy",
        choices=("score-cell-volume-clearance-v1", "adaptive-geometry-history-v2"),
        default="adaptive-geometry-history-v2",
        help="Use adaptive-geometry-history-v2 only with new versioned v2 queues and planner cost-aware-v2, fusion-v2, or deterministic-v2.",
    )
    bootstrap.add_argument(
        "--seed-region-policy",
        choices=(
            "fixed-v1",
            "m7-recenter-z-v1",
            "m7-recenter-xyz-v1",
            "m7-recenter-z-chunk-safe-v1",
            "m7-chunk-safe-merge-interior-v2",
        ),
        default="fixed-v1",
    )
    bootstrap.add_argument("--recenter-probe-max-candidates", type=int, default=100)
    bootstrap.add_argument("--recenter-radius-x", type=int, default=64)
    bootstrap.add_argument("--recenter-radius-y", type=int, default=64)
    bootstrap.add_argument("--recenter-radius-z", type=int, default=64)
    bootstrap.add_argument(
        "--ct-material-support-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject m7 seeds with no nearby material in the frozen CT OME-Zarr. "
             "On, because growing from a point the scan has nothing at costs "
             "hours and cannot succeed; --no- turns it off deliberately.",
    )
    bootstrap.add_argument(
        "--planner", choices=PLANNER_CHOICES, default="cost-aware-v2",
        help="which planner grows these tasks; cost-aware-v2 is the stage's declared default")
    bootstrap.add_argument(
        "--planner-model", default=None,
        help="provider/model for an agent planner; the worker's default when unset")
    bootstrap.add_argument(
        "--p0-selection-version", default=None,
        help="the P0 selection version these tasks read, so a mission that "
             "reselected between two runs does not produce two tasks that look "
             "identical and read different inputs")
    bootstrap.add_argument(
        "--p0-artifact-id", default=None,
        help="the P0 artifact chosen for this scroll under that selection")
    bootstrap.add_argument(
        "--p0-artifact-sha256", default=None,
        help="that artifact's content hash, so a task can show the artifact "
             "changed underneath rather than only which one it named")
    bootstrap.add_argument(
        "--p0-resolved-by", default=None,
        help="whether the artifact was selected or was simply the newest "
             "registered one; those are different claims")
    bootstrap.add_argument(
        "--queued-by", default=None,
        help="the person who asked for this queue, written onto every task. "
             "The panel passes its signed-in user; a hand-run bootstrap should "
             "pass whoever is running it, because 'unattributed' is what the "
             "row says otherwise.")
    bootstrap.add_argument(
        "--mission-id", default=None,
        help="mission that owns every generated task; defaults to unfiled")
    bootstrap.add_argument(
        "--reason", default=None,
        help="why this queue was asked for, written onto every task it creates. "
             "A comparison run needs its justification on the record, and the "
             "panel's audit trail deliberately holds no request bodies.")
    bootstrap.add_argument("--ct-support-level", type=int, default=5)
    bootstrap.add_argument("--ct-support-radius-l0", type=int, default=192)
    bootstrap.add_argument("--ct-support-minimum-nonzero-voxels", type=int, default=1)
    bootstrap.add_argument(
        "--seed-probe-mode",
        choices=("off", "shadow", "select"),
        default="off",
        help=(
            "Run deterministic short VC3D trials before planning. shadow records "
            "evidence without steering; select may constrain a v2 planner to one "
            "unambiguous probe winner and otherwise stops for review."
        ),
    )
    bootstrap.add_argument(
        "--reconsider-covered",
        action="store_true",
        # Clearance skips cells near a surface somebody already grew, which is
        # what keeps coverage spreading -- and also what puts m7's alternatives
        # out of reach, since a cell only has alternatives where it produced
        # something. A run that means to revisit says so here.
        help="offer cells clearance would skip, to grow m7's other candidates",
    )
    bootstrap.add_argument(
        "--candidate-rank",
        type=int,
        default=1,
        # Which rung of m7's frozen ordering to grow. 1 is the best candidate and
        # what every run does; a higher rank grows an alternative the planner
        # would otherwise only record as rejected. Stamped on the task so the
        # worker can hand it to the planner it builds -- a rank accepted at the
        # API and dropped before the host would attribute a surface to a
        # configuration that never ran.
        help="which of m7's ranked candidates to grow (1 = the best)",
    )
    bootstrap.add_argument(
        "--seed-probe-top-k",
        type=int,
        # The cap lives here and in the panel's request model, and raising one
        # without the other leaves the API accepting a value this refuses.
        # Three made the probe a tie-break between m7's best candidates; the
        # first shadow run measured 8 of 8 ELIGIBLE at that depth, so it paid
        # for nothing. Whether the ordering holds at rank 20 is the question.
        choices=range(1, 21),
        default=2,
        metavar="{1..20}",
    )
    bootstrap.add_argument(
        "--seed-probe-generations",
        type=int,
        choices=range(10, 21),
        default=12,
        metavar="{10..20}",
    )
    bootstrap.add_argument(
        "--seed-probe-benchmark-receipt",
        type=Path,
        help=(
            "required for select: canonical "
            "campaignx.seed_probe_benchmark_decision.v1 approval from the "
            "isolated paired deterministic-v2 benchmark; the task stores its "
            "hash binding, never this local path"
        ),
    )
    bootstrap.add_argument(
        "--seed-probe-review-owner",
        help=(
            "required with production --seed-probe-mode select; accountable "
            "owner for any HUMAN_REVIEW probe outcome"
        ),
    )
    bootstrap.set_defaults(handler=command_bootstrap)

    benchmark_arm = subcommands.add_parser(
        "seed-probe-benchmark-arm",
        help=(
            "Queue one preregistered isolated seed-probe causal arm; never "
            "uses production select authorization."
        ),
    )
    benchmark_arm.add_argument("--db", required=True)
    benchmark_arm.add_argument(
        "--confirm-isolated-nonproduction",
        action="store_true",
        required=True,
        help=(
            "operator assertion that this has a separate nonproduction DB, "
            "run/artifact roots, and credentials; the CLI cannot prove it"
        ),
    )
    benchmark_arm.add_argument(
        "--benchmark-spec",
        type=Path,
        required=True,
        help=(
            "frozen campaignx.seed_probe_benchmark_spec.v1 preregistration; "
            "not a production decision receipt"
        ),
    )
    benchmark_arm.add_argument(
        "--arm",
        choices=("baseline", "closed_loop"),
        required=True,
        help="baseline is seed-probe off; closed_loop is preregistered select",
    )
    benchmark_arm.add_argument(
        "--eligible",
        type=Path,
        required=True,
        help=(
            "frozen eligible-source manifest; only scrolls named by the "
            "preregistered cohort are imported and source-verified"
        ),
    )
    benchmark_arm.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="frozen coverage catalogue; its SHA-256 is bound into every task",
    )
    benchmark_arm.add_argument("--receipt", type=Path)
    benchmark_arm.add_argument("--grid-step", type=int, default=2048)
    benchmark_arm.add_argument("--query-radius", type=int, default=64)
    benchmark_arm.add_argument("--clearance", type=float, default=256.0)
    benchmark_arm.add_argument("--volume-edge-margin", type=int, default=64)
    benchmark_arm.add_argument(
        "--candidate-interior-clearance", type=int, default=0
    )
    benchmark_arm.add_argument(
        "--selection-strategy",
        choices=("max-clearance-v1", "stratified-clearance-v1"),
        default="max-clearance-v1",
    )
    benchmark_arm.add_argument("--grid-version", required=True)
    benchmark_arm.add_argument(
        "--candidate-selection-policy",
        choices=("score-cell-volume-clearance-v1", "adaptive-geometry-history-v2"),
        default="score-cell-volume-clearance-v1",
    )
    benchmark_arm.add_argument(
        "--seed-region-policy",
        choices=(
            "fixed-v1",
            "m7-recenter-z-v1",
            "m7-recenter-xyz-v1",
            "m7-recenter-z-chunk-safe-v1",
            "m7-chunk-safe-merge-interior-v2",
        ),
        default="fixed-v1",
    )
    benchmark_arm.add_argument(
        "--recenter-probe-max-candidates", type=int, default=100
    )
    benchmark_arm.add_argument("--recenter-radius-x", type=int, default=64)
    benchmark_arm.add_argument("--recenter-radius-y", type=int, default=64)
    benchmark_arm.add_argument("--recenter-radius-z", type=int, default=64)
    benchmark_arm.add_argument(
        "--ct-material-support-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    benchmark_arm.add_argument("--ct-support-level", type=int, default=5)
    benchmark_arm.add_argument("--ct-support-radius-l0", type=int, default=192)
    benchmark_arm.add_argument(
        "--ct-support-minimum-nonzero-voxels", type=int, default=1
    )
    benchmark_arm.set_defaults(handler=command_seed_probe_benchmark_arm)

    manual = subcommands.add_parser(
        "bootstrap-manual",
        help="Queue growth from points a person supplied, one task per point.")
    manual.add_argument("--db", required=True)
    manual.add_argument("--catalog", type=Path, required=True)
    manual.add_argument("--sample", required=True,
                        help="the scroll these points are in; it needs a source "
                             "snapshot already, because a point refers to a volume")
    manual.add_argument("--points", type=Path, required=True,
                        help="JSON list of {x,y,z} in CT-L0 voxels, or CSV with an "
                             "x,y,z header")
    manual.add_argument("--submitted-by", required=True,
                        help="recorded on every task; a human seed without an "
                             "author is not auditable")
    manual.add_argument(
        "--mission-id",
        default=None,
        help="mission that owns these tasks; omitted tasks are filed as unfiled",
    )
    manual.add_argument("--grid-step", type=int, default=2048)
    manual.add_argument("--query-radius", type=int, default=64)
    manual.add_argument("--volume-edge-margin", type=int, default=64)
    manual.add_argument("--grid-version", default="ct-l0-manual-v1")
    manual.add_argument("--policy-version", default="ink-blind-v1")
    manual.add_argument("--ct-material-support-gate",
                        action=argparse.BooleanOptionalAction, default=True)
    manual.add_argument("--planner", choices=PLANNER_CHOICES, default=None)
    manual.add_argument("--planner-model", default=None)
    manual.add_argument("--ct-support-level", type=int, default=5)
    manual.add_argument("--ct-support-radius-l0", type=int, default=192)
    manual.add_argument("--ct-support-minimum-nonzero-voxels", type=int, default=1)
    manual.add_argument("--receipt", type=Path, default=None)
    manual.set_defaults(handler=command_bootstrap_manual)

    resume = subcommands.add_parser(
        "bootstrap-resume",
        help="Continue an existing surface under correction points, as a new task.")
    resume.add_argument("--db", required=True)
    resume.add_argument("--surface", required=True,
                        help="the surface to resume from; it must have an artifact")
    resume.add_argument("--corrections", required=True, type=Path,
                        help="a VC3D point collection, passed to --correct")
    resume.add_argument("--submitted-by", default="anonymous")
    resume.add_argument("--resume-generations", type=int, default=None)
    resume.add_argument("--rewind-gen", type=int, default=None)
    resume.add_argument("--grid-version", default="resume-v1")
    # Its own policy version, so the resumed grow is a new task and the surface
    # it continues is left exactly as measured.
    resume.add_argument("--policy-version", default="resume-corrections-v1")
    resume.set_defaults(handler=command_bootstrap_resume)

    recovery = subcommands.add_parser(
        "bootstrap-seed-recovery",
        help="Create a new immutable queue from prior m7-positive cells rejected only by geometry gates.",
    )
    recovery.add_argument("--source-db", type=Path, required=True)
    recovery.add_argument("--source-attempt-root", type=Path, required=True)
    recovery.add_argument("--db", type=Path, required=True)
    recovery.add_argument("--receipt", type=Path, required=True)
    recovery.add_argument("--grid-version", required=True)
    recovery.add_argument("--policy-version", required=True)
    recovery.add_argument(
        "--candidate-selection-policy",
        choices=("score-cell-volume-clearance-v1", "adaptive-geometry-history-v2"),
        default="score-cell-volume-clearance-v1",
    )
    recovery.add_argument(
        "--seed-region-policy",
        choices=("m7-chunk-safe-merge-interior-v2",),
        default="m7-chunk-safe-merge-interior-v2",
    )
    recovery.add_argument("--ct-material-support-gate",
                          action=argparse.BooleanOptionalAction, default=True)
    recovery.add_argument("--ct-support-level", type=int, default=5)
    recovery.add_argument("--ct-support-radius-l0", type=int, default=192)
    recovery.add_argument("--ct-support-minimum-nonzero-voxels", type=int, default=1)
    recovery.set_defaults(handler=command_bootstrap_seed_recovery)

    adaptive_retry = subcommands.add_parser(
        "bootstrap-adaptive-retry",
        help="Create one new planner-v2 task from a terminal geometry-only task.",
    )
    adaptive_retry.add_argument("--db", required=True)
    adaptive_retry.add_argument("--source-task-id", required=True)
    adaptive_retry.add_argument("--receipt", type=Path, required=True)
    adaptive_retry.add_argument("--grid-version", required=True)
    adaptive_retry.add_argument("--policy-version", required=True)
    adaptive_retry.add_argument(
        "--seed-region-policy",
        choices=("m7-chunk-safe-merge-interior-v2",),
        default="m7-chunk-safe-merge-interior-v2",
    )
    adaptive_retry.add_argument("--candidate-interior-clearance", type=int, default=0)
    adaptive_retry.set_defaults(handler=command_bootstrap_adaptive_retry)

    worker = subcommands.add_parser("worker", help="Worker operations.")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    run = worker_commands.add_parser("run", help="Claim and execute queued tasks autonomously.")
    run.add_argument("--db", required=True)
    run.add_argument("--worker-id", required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument(
        "--artifact-root",
        required=True,
        help="s3://bucket/prefix for immutable TIFXYZ artifacts. A local "
             "directory needs --allow-local-artifacts and keeps the surface on "
             "one machine.",
    )
    run.add_argument(
        "--allow-local-artifacts",
        action="store_true",
        help="Publish surfaces to a directory on this host. For a single-machine "
             "run only: a fleet worker is ephemeral, and a surface on its disk is "
             "invisible to every other phase and gone when the host is.",
    )
    run.add_argument(
        "--qc-profile-id",
        required=True,
        help="Versioned semantic downstream QC profile bound to each promoted surface.",
    )
    run.add_argument("--repo-root", type=Path, default=repo_root)
    # "manual" reads points a person uploaded off the task, instead of probing
    # the prediction for them. Everything after discovery is identical.
    run.add_argument("--seed-provider", choices=("mcp", "recorded", "manual"),
                     default="mcp")
    run.add_argument(
        "--planner",
        choices=("cost-aware-v2", "fusion-v2", "deterministic-v2", "deterministic", "opencode-v2", "opencode"),
        default="cost-aware-v2",
        help="cost-aware-v2 is the active bounded adaptive planner; fusion-v2 reproduces the expensive maximum-reasoning canary.",
    )
    run.add_argument("--opencode")
    run.add_argument("--model", default=DEFAULT_OPENCODE_MODEL)
    run.add_argument("--openrouter-key-env", default="OPENROUTER_API_KEY")
    run.add_argument("--planner-timeout", type=int, default=600)
    run.add_argument("--vc3d-binary", type=Path)
    run.add_argument("--fixture-grow", action="store_true")
    run.add_argument("--allow-fixture", action="store_true")
    run.add_argument("--grow-timeout", type=int, default=1800)
    run.add_argument("--minimum-free-gib", type=float, default=4.0)
    run.add_argument(
        "--cuda-available",
        action="store_true",
        help="Advertise this worker as CUDA-capable for resource-aware claims.",
    )
    run.add_argument("--gpu-model", help="Non-secret GPU model recorded with claims, for example GTX 1660.")
    run.add_argument("--gpu-vram-gb", type=float, default=0.0, help="Usable VRAM on this worker's selected GPU.")
    run.add_argument("--cuda-device-index", type=int, help="CUDA device index assigned by the supervisor.")
    run.add_argument(
        "--seed-probe-support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Advertise and execute seed-probe-v1 tasks. Off by default so an "
            "older/unvalidated host cannot silently claim a probe-required task."
        ),
    )
    run.add_argument(
        "--isolated-benchmark-spec",
        type=Path,
        help=(
            "validated preregistered benchmark spec this worker may claim. "
            "It binds the worker to its exact spec SHA and refuses a process "
            "configured for production select. Use separate DB/run/artifact "
            "roots and credentials; add --seed-probe-support for closed_loop "
            "tasks."
        ),
    )
    run.add_argument(
        "--confirm-isolated-nonproduction",
        action="store_true",
        help=(
            "required together with --isolated-benchmark-spec; operator "
            "assertion of separate nonproduction DB, run/artifact roots, and "
            "credentials"
        ),
    )
    run.add_argument("--lease-seconds", type=int, default=900)
    run.add_argument("--provider-retry-delay-seconds", type=int, default=300, help="Defer a task after a transient OpenCode provider outage.")
    run.add_argument("--source-retry-delay-seconds", type=int, default=300, help="Defer a task after a transient VC3D/MCP source outage.")
    run.add_argument("--finalization-retry-delay-seconds", type=int, default=300, help="Defer a task after a transient artifact-store or database finalization outage.")
    run.add_argument("--probe-artifact-max-requeues", type=int, default=5, help="Bound retries that fetch and verify a selected noncanonical probe winner.")
    run.add_argument("--finalization-max-requeues", type=int, default=2, help="Bound replacement full grows after persistent finalization infrastructure outages.")
    # 0 means unlimited, which is what a long-running service needs: the
    # loop is bounded by `len(completed) < max_jobs`, so 0 used to mean
    # "exit before claiming anything" -- a worker that could only be told
    # to stop early or to run exactly N times.
    run.add_argument("--max-jobs", type=int, default=1,
                     help="how many tasks to run before exiting; 0 for no limit")
    run.add_argument(
        "--terminal-outcomes-exit-zero",
        action="store_true",
        help=(
            "Treat expected terminal no-surface outcomes as successful process "
            "completion for bounded supervisors; technical and transient "
            "failures remain non-zero."
        ),
    )
    run.add_argument("--task-id", help="Claim only this previously surveyed pending task ID.")
    run.add_argument("--watch", action="store_true")
    run.add_argument("--poll-seconds", type=float, default=10.0)
    run.set_defaults(handler=command_worker_run)

    recover_provider = worker_commands.add_parser(
        "recover-provider",
        help="Recover only a historically misclassified PlannerProviderUnavailable terminal attempt.",
    )
    recover_provider.add_argument("--db", required=True)
    recover_provider.add_argument("--task-id", required=True)
    recover_provider.add_argument("--attempt-id", required=True)
    recover_provider.add_argument("--retry-delay-seconds", type=int, default=300)
    recover_provider.set_defaults(handler=command_worker_recover_provider)

    recover_mcp_auth = worker_commands.add_parser(
        "recover-mcp-auth",
        help="Recover only an exact terminal loopback-MCP HTTP 401 attempt.",
    )
    recover_mcp_auth.add_argument("--db", required=True)
    recover_mcp_auth.add_argument("--task-id", required=True)
    recover_mcp_auth.add_argument("--attempt-id", required=True)
    recover_mcp_auth.add_argument("--retry-delay-seconds", type=int, default=1)
    recover_mcp_auth.set_defaults(handler=command_worker_recover_mcp_auth)

    recover_finalizer = worker_commands.add_parser(
        "recover-finalizer-dependency",
        help="Recover only an exact terminal missing numpy/tifffile finalizer dependency.",
    )
    recover_finalizer.add_argument("--db", required=True)
    recover_finalizer.add_argument("--task-id", required=True)
    recover_finalizer.add_argument("--attempt-id", required=True)
    recover_finalizer.add_argument("--retry-delay-seconds", type=int, default=1)
    recover_finalizer.set_defaults(handler=command_worker_recover_finalizer_dependency)

    certify = subcommands.add_parser(
        "certify",
        help="Give a geometry verdict to surfaces that carry none (P2).")
    certify.add_argument("--db", required=True)
    certify.add_argument("--limit", type=int, default=25,
                         help="how many surfaces to measure in one pass; each is "
                              "fetched and meshed, so this bounds the run")
    certify.add_argument("--sample", default=None, help="one scroll only")
    certify.add_argument("--workspace", type=Path, default=None,
                         help="where surfaces are staged; a temporary directory "
                              "by default, removed per surface either way")
    certify.add_argument("--receipt", type=Path, default=None)
    certify.add_argument("--dry-run", action="store_true",
                         help="list what has no verdict without fetching anything")
    certify.set_defaults(handler=command_certify)

    flatten = subcommands.add_parser(
        "flatten",
        help="Unroll certified surfaces into flat sheets (P3).")
    flatten.add_argument("--db", required=True)
    flatten.add_argument("--binary", required=True,
                         help="the vc_flatten executable; recorded by hash on "
                              "every receipt")
    flatten.add_argument("--profile", type=Path, required=True,
                         help="a frozen flattening profile, because a flag typed "
                              "at a prompt is a setting no receipt records")
    flatten.add_argument(
        "--allow-unvalidated", action="store_true",
        help="flatten surfaces whose CT support was never measured. The default "
             "refuses them: geometry says the shape is a plausible lamina, the "
             "physical axis says there is papyrus there, and P3 read only the "
             "first for as long as it existed")
    flatten.add_argument("--artifact-store", default=None,
                         help="where flattened sheets are published; local path "
                              "or s3://bucket/prefix. Omitted keeps them nowhere, "
                              "which is only useful for a trial run.")
    flatten.add_argument("--limit", type=int, default=5)
    flatten.add_argument("--sample", default=None)
    flatten.add_argument(
        "--surface-id", default=None,
        help="flatten only this immutable surface id; preserves an explicit "
             "merge-to-flatten continuation instead of taking another backlog item")
    flatten.add_argument("--workspace", type=Path, default=None)
    flatten.add_argument("--timeout", type=int, default=7200)
    flatten.add_argument("--receipt", type=Path, default=None)
    flatten.add_argument("--dry-run", action="store_true")
    flatten.set_defaults(handler=command_flatten, repo_root=repo_root)

    qc = subcommands.add_parser(
        "qc", help="Lease and process preserved surfaces through downstream QC."
    )
    qc_commands = qc.add_subparsers(dest="qc_command", required=True)
    qc_run = qc_commands.add_parser(
        "run", help="Claim surface-QC jobs and invoke a frozen scientific adapter."
    )
    qc_run.add_argument("--db", required=True)
    qc_run.add_argument("--worker-id", required=True)
    qc_run.add_argument("--run-root", type=Path, required=True)
    qc_run.add_argument("--qc-executable", type=Path)
    qc_run.add_argument("--qc-timeout", type=int, default=7200)
    qc_run.add_argument("--lease-seconds", type=int, default=900)
    qc_run.add_argument("--retry-delay-seconds", type=int, default=300)
    qc_run.add_argument("--profile-id", required=True)
    qc_run.add_argument("--max-jobs", type=int, default=1)
    qc_run.add_argument("--watch", action="store_true")
    qc_run.add_argument("--poll-seconds", type=float, default=10.0)
    qc_run.add_argument("--fixture-qc", action="store_true")
    qc_run.add_argument("--allow-fixture", action="store_true")
    qc_run.add_argument(
        "--fixture-outcome",
        choices=tuple(sorted(QC_OUTCOME_STATES)),
        default="CT_SUPPORTED_NO_RETAINED_INK_SIGNAL",
    )
    qc_run.set_defaults(handler=command_qc_run)

    qc_backfill = qc_commands.add_parser(
        "enqueue-backfill",
        help="Idempotently import verified historical TIFXYZ and queue scientific QC.",
    )
    qc_backfill.add_argument("--db", required=True)
    qc_backfill.add_argument("--eligible", type=Path, required=True)
    qc_backfill.add_argument("--manifest", type=Path, required=True)
    qc_backfill.add_argument("--receipt", type=Path, required=True)
    qc_backfill.add_argument("--sample", action="append", default=[])
    qc_backfill.add_argument("--surface-id", action="append", default=[])
    qc_backfill.add_argument("--profile-id", required=True)
    qc_backfill.set_defaults(handler=command_qc_enqueue_backfill)

    survey = subcommands.add_parser("survey", help="MCP-probe pending m7 cells without claiming tasks.")
    survey.add_argument("--db", required=True)
    survey.add_argument("--output", type=Path, required=True)
    survey.add_argument("--limit", type=int, default=32)
    survey.add_argument("--parallelism", type=int, default=1, help="Concurrent independent MCP probes; result ordering remains deterministic.")
    survey.add_argument("--task-id", help="Probe only this pending task without leasing, growing, or mutating it.")
    survey.add_argument("--progress", type=Path, help="Optional atomically-written read-only progress receipt.")
    survey.set_defaults(handler=command_survey)

    republish = subcommands.add_parser(
        "republish",
        help="Move surfaces published to a worker's local path into object storage.")
    republish.add_argument("--db", required=True)
    republish.add_argument("--artifact-store", required=True,
                           help="s3://bucket/prefix, or a path for a local mirror")
    republish.add_argument("--sample", default=None)
    republish.add_argument("--limit", type=int, default=100)
    republish.add_argument("--dry-run", action="store_true")
    republish.set_defaults(handler=command_republish)

    replan = subcommands.add_parser(
        "replan",
        help="Re-ask cells that ended NO_SEED, under a new policy and planner.")
    replan.add_argument("--db", required=True)
    replan.add_argument("--grid-version", required=True,
                        help="a new coverage universe; the same one is a no-op")
    replan.add_argument("--policy-version", required=True)
    replan.add_argument("--planner", default=None,
                        help="the planner is what decides the seed, so re-asking "
                             "with the same one usually gets the same answer")
    replan.add_argument("--planner-model", default=None)
    replan.add_argument("--sample", default=None)
    replan.add_argument("--cause", action="append", default=[],
                        help="only cells whose diagnosis names this cause, e.g. "
                             "NO_M7_CANDIDATES or MALFORMED_COORDINATE_OR_SCORE")
    replan.add_argument("--limit", type=int, default=50)
    replan.add_argument("--dry-run", action="store_true")
    replan.set_defaults(handler=command_replan)

    coverage = subcommands.add_parser(
        "coverage",
        help="How much of a scroll has been explored, per grid version.")
    coverage.add_argument("--db", required=True)
    coverage.add_argument("--sample", default=None)
    coverage.set_defaults(handler=command_coverage)

    status = subcommands.add_parser("status", help="Show queue/catalogue counts without changing state.")
    status.add_argument("--db", required=True)
    status.set_defaults(handler=command_status)

    demo = subcommands.add_parser("demo", help="Run the full vertical slice with an explicitly non-scientific fixture grow.")
    demo.add_argument("--root", type=Path, required=True)
    demo.add_argument("--repo-root", type=Path, default=repo_root)
    demo.add_argument(
        "--planner",
        choices=("deterministic", "deterministic-v2", "cost-aware-v2", "fusion-v2", "opencode", "opencode-v2"),
        default="deterministic",
    )
    demo.add_argument("--opencode")
    demo.add_argument("--model", default=DEFAULT_OPENCODE_MODEL)
    demo.add_argument("--openrouter-key-env", default="OPENROUTER_API_KEY")
    demo.add_argument("--planner-timeout", type=int, default=600)
    demo.set_defaults(handler=command_demo)
    return parser


def main(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    root = repo_root or Path(__file__).resolve().parents[4]
    args = build_parser(root).parse_args(argv)
    return int(args.handler(args))
