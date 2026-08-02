#!/usr/bin/env python3
"""Render and coarsely route a shard of COVERAGE_AND_SURFACE_V2.

This is a resumable prioritization pass.  It keeps the centre CT render,
probability map and receipts, then removes the other temporary rendered
slices.  A retained activation is never called ink or a letter here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import (  # noqa: E402
    NUMERIC_STEM_INDEX,
    ordered_tiff_files,
    resolve_tiff_slice,
)


PLAN_KIND = "campaign_x_phase4_coverage_and_surface_v2_plan_v1"
EXPECTED_PLAN_SHA256 = (
    "ddfbe2f3324d6a72c07ecafda4dd0f5f43c581bb4a75d0069b07952a218204b4"
)
EXPECTED_RENDERER_SHA256 = (
    "735089de3365f11ad371fab2f585e16142a6bfa672883768518d055fd41452d6"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "490a98f9491e1180274ed3a0c0a9c611d73a0109c0e0c0fbba1097562a972488"
)
MODEL_FAMILY = "timesformer_GP_scroll1"
RENDERED_SLICE_COUNT = 65
CENTRAL_SLICE_INDEX = 32


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_tiffs(directory: Path) -> list[Path]:
    """Order a render stack under the shared slice-order contract.

    The previous implementation pushed any non-numeric stem into a silent
    ``1_000_000`` reserve bucket at the end of the stack instead of failing.
    """

    return ordered_tiff_files(directory, require_numeric=True, allow_empty=True)[0]


def source_hashes(directory: Path) -> dict[str, str] | None:
    paths = [directory / name for name in ("x.tif", "y.tif", "z.tif", "meta.json")]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return None
    return {path.name: sha256_file(path) for path in paths}


def fetch_zarr_metadata(ct_uri: str, cache: Path) -> None:
    for suffix in (".zgroup", ".zattrs", "0/.zarray"):
        destination = cache / suffix
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{ct_uri.rstrip('/')}/{suffix}",
            headers={"User-Agent": "Campaign-X-coverage-surface-v2/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            destination.write_bytes(response.read())


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def flatten_tasks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = [
        surface
        for target in plan["targets"]
        for surface in target["surfaces"]
    ]
    tasks.sort(key=lambda row: int(row["coverage_order"]))
    if len(tasks) != 148:
        raise RuntimeError("frozen plan does not contain 148 surfaces")
    return tasks


def task_output(runtime_root: Path, task: dict[str, Any]) -> Path:
    return runtime_root / str(task["sample_id"]) / str(task["seed_id"])


def remote_source(
    runtime_root: Path,
    task: dict[str, Any],
) -> Path:
    return (
        runtime_root
        / "source_jobs"
        / str(task["source_job_id"])
        / "surface"
    )


def completed_state(
    output: Path,
    task: dict[str, Any],
    checkpoint: Path,
) -> dict[str, Any] | None:
    receipt_path = output / "coarse_gp_scroll1_v1/INK_SCREENING_RECEIPT.json"
    try:
        central = resolve_tiff_slice(output / "tiffs", CENTRAL_SLICE_INDEX)
    except RuntimeError:
        central = None
    if not receipt_path.is_file() or central is None:
        return None
    receipt = read_json(receipt_path)
    frozen = receipt.get("checkpoint", {})
    if (
        frozen.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or frozen.get("model_family") != MODEL_FAMILY
    ):
        raise RuntimeError(
            f"existing screen belongs to another checkpoint: {task['seed_id']}"
        )
    runs = receipt.get("inference", {}).get("runs", [])
    if len(runs) != 1:
        raise RuntimeError(
            f"coarse screen must have one inference run: {task['seed_id']}"
        )
    return {
        "status": "COARSE_COMPLETED",
        "output": str(output),
        "coarse_receipt": str(receipt_path),
        "coarse_receipt_sha256": sha256_file(receipt_path),
        "central_tiff": str(central),
        "central_tiff_sha256": sha256_file(central),
        "slice_ordering": NUMERIC_STEM_INDEX,
        "probability_mean": float(runs[0]["probability_mean"]),
        "probability_p99": float(runs[0]["probability_p99"]),
        "render_stack_pruned_after_screen": len(exact_tiffs(output / "tiffs")) == 1,
        "reused": True,
    }


def prune_render_stack(tiffs: Path) -> Path:
    paths = exact_tiffs(tiffs)
    if len(paths) != RENDERED_SLICE_COUNT:
        raise RuntimeError(
            f"cannot prune a non-exact render stack: {len(paths)} TIFFs"
        )
    central = resolve_tiff_slice(tiffs, CENTRAL_SLICE_INDEX)
    for path in paths:
        if path != central:
            path.unlink()
    return central


def checkpoint_batch(
    path: Path,
    *,
    plan_path: Path,
    renderer: Path,
    checkpoint: Path,
    shard_count: int,
    shard_index: int,
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> None:
    completed = sum(row["status"] == "COARSE_COMPLETED" for row in states.values())
    failed = sum(row["status"] == "FAILED" for row in states.values())
    write_json(
        path,
        {
            "kind": "campaign_x_phase4_coverage_surface_v2_coarse_shard_v1",
            "status": (
                "COMPLETED_PRIORITIZATION_ONLY"
                if completed == len(tasks) and failed == 0
                else "RUNNING_OR_PARTIAL"
            ),
            "updated_at_utc": utc_now(),
            "plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "renderer": {
                "path": str(renderer),
                "sha256": sha256_file(renderer),
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "model_family": MODEL_FAMILY,
            },
            "shard_count": shard_count,
            "shard_index": shard_index,
            "task_count": len(tasks),
            "completed_count": completed,
            "failed_count": failed,
            "tasks": [
                {
                    "coverage_order": task["coverage_order"],
                    "sample_id": task["sample_id"],
                    "seed_id": task["seed_id"],
                    "initial_area_cm2": task["initial_area_cm2"],
                    **states[str(task["seed_id"])],
                }
                for task in tasks
            ],
            "interpretation": [
                "coarse probability only routes later compute",
                "no task is accepted as ink or letters",
                "retained signals require replica stability and orthogonal CT",
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference-script", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cache-gb", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Process only the first N tasks in this shard (smoke/resume aid).",
    )
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    runtime_root = args.runtime_root.resolve()
    renderer = args.renderer.resolve()
    checkpoint = args.checkpoint.resolve()
    inference_script = args.inference_script.resolve()
    if not 1 <= args.shard_count <= 8:
        raise ValueError("shard-count must be between 1 and 8")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index is outside shard-count")
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("COVERAGE_AND_SURFACE_V2 plan hash mismatch")
    if sha256_file(renderer) != EXPECTED_RENDERER_SHA256:
        raise RuntimeError("renderer hash mismatch")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash mismatch")
    if not inference_script.is_file():
        raise RuntimeError("inference script is missing")

    plan = read_json(plan_path)
    if plan.get("kind") != PLAN_KIND or plan.get("status") != "FROZEN_READY":
        raise RuntimeError("unexpected or unfrozen COVERAGE_AND_SURFACE_V2 plan")
    tasks = [
        task
        for index, task in enumerate(flatten_tasks(plan))
        if index % args.shard_count == args.shard_index
    ]
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise ValueError("max-tasks must be positive")
        tasks = tasks[: args.max_tasks]
    receipt_path = (
        runtime_root / f"COARSE_BATCH_SHARD_{args.shard_index:02d}.json"
    )
    states: dict[str, dict[str, Any]] = {
        str(task["seed_id"]): {
            "status": "PENDING",
            "output": str(task_output(runtime_root, task)),
        }
        for task in tasks
    }

    for position, task in enumerate(tasks, start=1):
        seed_id = str(task["seed_id"])
        output = task_output(runtime_root, task)
        source = remote_source(runtime_root, task)
        tiffs = output / "tiffs"
        coarse = output / "coarse_gp_scroll1_v1"
        cache = runtime_root / "ct_metadata_cache" / f"{task['sample_id']}.zarr"
        started = time.monotonic()
        try:
            prior = completed_state(output, task, checkpoint)
            if prior is not None:
                states[seed_id] = prior
                print(
                    f"REUSE:{position}/{len(tasks)}:{seed_id}",
                    flush=True,
                )
                checkpoint_batch(
                    receipt_path,
                    plan_path=plan_path,
                    renderer=renderer,
                    checkpoint=checkpoint,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                    tasks=tasks,
                    states=states,
                )
                continue

            actual_hashes = source_hashes(source)
            if actual_hashes != task["source_surface_hashes"]:
                raise RuntimeError("transferred source TIFXYZ hash mismatch")
            print(
                f"START:{position}/{len(tasks)}:{seed_id}",
                flush=True,
            )
            fetch_zarr_metadata(str(task["ct_uri"]), cache)
            paths = exact_tiffs(tiffs)
            if len(paths) != RENDERED_SLICE_COUNT:
                if paths:
                    raise RuntimeError(
                        f"partial pre-existing render stack: {len(paths)} TIFFs"
                    )
                run_logged(
                    [
                        str(renderer),
                        "--segmentation",
                        str(source),
                        "--volume",
                        str(cache),
                        "--remote-url",
                        str(task["ct_uri"]),
                        "--prefetch-remote",
                        "--scale",
                        "1",
                        "--group-idx",
                        "0",
                        "--auto-crop",
                        "--flatten",
                        "--flatten-iterations",
                        "10",
                        "--num-slices",
                        str(RENDERED_SLICE_COUNT),
                        "--slice-step",
                        "1",
                        "--cache-gb",
                        str(args.cache_gb),
                        "--timeout",
                        "90",
                        "--voxel-size",
                        str(task["voxel_size_um"]),
                        "--voxel-unit",
                        "micrometer",
                        "--tif-output",
                        str(tiffs),
                        "--log-path",
                        str(output / "render.log"),
                    ],
                    output / "render.stdout.log",
                )
            if len(exact_tiffs(tiffs)) != RENDERED_SLICE_COUNT:
                raise RuntimeError("renderer did not produce exactly 65 TIFFs")

            receipt = coarse / "INK_SCREENING_RECEIPT.json"
            if not receipt.is_file():
                run_logged(
                    [
                        "python3",
                        str(inference_script),
                        "--sample-id",
                        str(task["sample_id"]),
                        "--tiff-dir",
                        str(tiffs),
                        "--checkpoint",
                        str(checkpoint),
                        "--model-family",
                        MODEL_FAMILY,
                        "--output",
                        str(coarse),
                        "--depth-centers",
                        str(CENTRAL_SLICE_INDEX),
                        "--tiling-offsets",
                        "0",
                        "--frames",
                        "26",
                        "--source-pixel-um",
                        str(task["voxel_size_um"]),
                        "--source-slice-um",
                        str(task["voxel_size_um"]),
                        # FIX-09: the training scale is resolved by the runner
                        # from the ink lane profile, never restated here.
                        "--tile-size",
                        "64",
                        "--stride",
                        "32",
                        "--batch-size",
                        str(args.batch_size),
                        "--min-valid-ratio",
                        "0.60",
                        "--device",
                        "cuda",
                    ],
                    output / "coarse.stdout.log",
                )
            central = prune_render_stack(tiffs)
            current = completed_state(output, task, checkpoint)
            if current is None:
                raise RuntimeError("completed artifacts failed postcondition")
            current["reused"] = False
            current["duration_seconds"] = round(time.monotonic() - started, 6)
            current["central_tiff"] = str(central)
            states[seed_id] = current
            print(
                f"DONE:{position}/{len(tasks)}:{seed_id}:"
                f"{current['duration_seconds']}s:"
                f"p99={current['probability_p99']:.6f}",
                flush=True,
            )
        except Exception as error:
            states[seed_id] = {
                "status": "FAILED",
                "output": str(output),
                "error": f"{type(error).__name__}: {error}",
                "duration_seconds": round(time.monotonic() - started, 6),
            }
            print(
                f"FAILED:{position}/{len(tasks)}:{seed_id}:{error}",
                flush=True,
            )
        checkpoint_batch(
            receipt_path,
            plan_path=plan_path,
            renderer=renderer,
            checkpoint=checkpoint,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            tasks=tasks,
            states=states,
        )

    failures = [
        seed_id for seed_id, state in states.items() if state["status"] == "FAILED"
    ]
    if failures:
        raise RuntimeError(f"coarse shard failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
