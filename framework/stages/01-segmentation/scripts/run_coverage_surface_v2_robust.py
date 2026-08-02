#!/usr/bin/env python3
"""Run six-replica screening on the frozen COVERAGE_AND_SURFACE_V2 queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CONTRACT_ROOT = Path(__file__).resolve().parents[4]
if str(_CONTRACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_ROOT))

from framework.contracts.slice_order import (  # noqa: E402
    ordered_tiff_files,
    resolve_tiff_slice,
)


RANKING_KIND = "campaign_x_phase4_coverage_surface_v2_coarse_ranking_v1"
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
# FIX-09: the training scale is declared by the ink lane profile; a literal here
# silently disagrees with the profile the checkpoint was actually trained under.
INK_PROFILE_PATH = (
    Path(__file__).resolve().parents[3]
    / "profiles/03-ink/timesformer-gp-scroll1-screening-1.0.0.json"
)


def training_pixel_um() -> str:
    profile = json.loads(INK_PROFILE_PATH.read_text(encoding="utf-8"))
    return str(float(profile["input_contract"]["training_pixel_um"]))


def training_slice_um() -> str:
    profile = json.loads(INK_PROFILE_PATH.read_text(encoding="utf-8"))
    return str(float(profile["input_contract"]["training_slice_um"]))


POSITIVE_OUTCOMES = {
    "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW",
    "TEXT_LIKE_SUPPORT_OBSERVED",
}
NO_COMMON_VALID_OUTCOME = "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS"
NO_COMMON_VALID_ERROR = "RuntimeError: screening maps have no common valid pixels"
TERMINAL_TASK_STATUSES = {
    "ROBUST_COMPLETED",
    "ROBUST_NO_COMMON_VALID_PIXELS",
}


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


CENTRAL_SLICE_INDEX = 32


def exact_tiffs(directory: Path) -> list[Path]:
    """Order a rendered stack through the one shared contract.

    FIX-02: the previous key pushed any non-numeric stem into a silent reserve
    bucket (``else 1_000_000``) instead of failing, so one stray file shifted
    every subsequent index by one.  That is the same defect class that made the
    adjudication panel render slice 38 while its receipt claimed 32.
    """

    files, _ = ordered_tiff_files(directory)
    return files


def central_tiff(directory: Path) -> Path:
    return resolve_tiff_slice(directory, CENTRAL_SLICE_INDEX)


def prune_to_central(directory: Path) -> Path:
    central = central_tiff(directory)
    for path in exact_tiffs(directory):
        if path != central:
            path.unlink()
    return central


def flatten_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        str(surface["seed_id"]): surface
        for target in plan["targets"]
        for surface in target["surfaces"]
    }
    if len(rows) != 148:
        raise RuntimeError("plan must bind 148 unique surfaces")
    return rows


def validate_inputs(
    ranking_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("plan hash mismatch")
    plan = read_json(plan_path)
    surfaces = flatten_plan(plan)
    ranking = read_json(ranking_path)
    if (
        ranking.get("kind") != RANKING_KIND
        or ranking.get("status") != "COMPLETED_PRIORITIZATION_ONLY"
    ):
        raise RuntimeError("coarse ranking is incomplete or unexpected")
    if ranking.get("plan", {}).get("sha256") != EXPECTED_PLAN_SHA256:
        raise RuntimeError("coarse ranking uses another plan")
    if int(ranking.get("surface_count", -1)) != 148:
        raise RuntimeError("coarse ranking does not cover 148 surfaces")
    queue = ranking.get("robust_queue")
    if not isinstance(queue, list) or not queue:
        raise RuntimeError("robust queue is empty")
    ids = [str(row["seed_id"]) for row in queue]
    if len(ids) != len(set(ids)) or not set(ids) <= set(surfaces):
        raise RuntimeError("robust queue contains duplicate or unknown surfaces")
    counts: dict[str, int] = {}
    for row in queue:
        sample_id = str(row["sample_id"])
        counts[sample_id] = counts.get(sample_id, 0) + 1
    if set(counts) != {str(target["sample_id"]) for target in plan["targets"]}:
        raise RuntimeError("robust queue does not represent all 13 scrolls")
    if min(counts.values()) < 2:
        raise RuntimeError("robust queue has fewer than two rows for a scroll")
    return ranking, queue, surfaces


def summary(analysis: dict[str, Any]) -> dict[str, Any]:
    screening = analysis["text_like_screening"]
    outcome = str(screening["screening_outcome"])
    return {
        "screening_outcome": outcome,
        "glyph_like_candidate_count": int(
            screening["glyph_like_candidate_count"]
        ),
        "row_band_count": int(screening["row_band_count"]),
        "rows_with_at_least_four_candidates": int(
            screening["rows_with_at_least_four_candidates"]
        ),
        "route": (
            "RAW_CT_REVIEW_REQUIRED"
            if outcome in POSITIVE_OUTCOMES
            else "HIGH_RECALL_CT_ROUTER_PENDING"
        ),
    }


def completed(output: Path) -> dict[str, Any] | None:
    analysis_path = (
        output
        / "robust_gp_scroll1_v1/analysis/INK_STABILITY_ANALYSIS.json"
    )
    receipt_path = output / "robust_gp_scroll1_v1/INK_SCREENING_RECEIPT.json"
    if not analysis_path.is_file() or not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    checkpoint = receipt.get("checkpoint", {})
    if (
        checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("model_family") != MODEL_FAMILY
    ):
        raise RuntimeError("existing robust output uses another checkpoint")
    if len(receipt.get("inference", {}).get("runs", [])) != 6:
        raise RuntimeError("existing robust output is not six-replica")
    return {
        "status": "ROBUST_COMPLETED",
        "analysis": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "screening_receipt": str(receipt_path),
        "screening_receipt_sha256": sha256_file(receipt_path),
        **summary(read_json(analysis_path)),
        "reused": True,
    }


def no_common_valid_pixels(output: Path) -> dict[str, Any] | None:
    """Classify a proven six-replica no-coverage result as terminal.

    The stability analyzer correctly refuses to manufacture morphology when
    all six maps have an empty valid-pixel intersection.  This is not a worker
    failure and repeating it cannot recover CT coverage.  The classifier is
    deliberately exact: it requires the frozen six-replica receipt and the
    analyzer's precise recorded exception.
    """

    receipt_path = output / "robust_gp_scroll1_v1/INK_SCREENING_RECEIPT.json"
    log_path = output / "robust-analysis.stdout.log"
    analysis_path = output / "robust_gp_scroll1_v1/analysis/INK_STABILITY_ANALYSIS.json"
    if analysis_path.exists() or not receipt_path.is_file() or not log_path.is_file():
        return None
    receipt = read_json(receipt_path)
    checkpoint = receipt.get("checkpoint", {})
    if (
        checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.get("model_family") != MODEL_FAMILY
        or len(receipt.get("inference", {}).get("runs", [])) != 6
    ):
        raise RuntimeError("existing no-coverage output uses another inference grid")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if NO_COMMON_VALID_ERROR not in log:
        return None
    rendered_tiff_count = len(exact_tiffs(output / "tiffs"))
    if rendered_tiff_count != 65:
        raise RuntimeError(
            "no-coverage result does not preserve the required 65-slice CT stack"
        )
    return {
        "status": "ROBUST_NO_COMMON_VALID_PIXELS",
        "screening_outcome": NO_COMMON_VALID_OUTCOME,
        "glyph_like_candidate_count": 0,
        "row_band_count": 0,
        "rows_with_at_least_four_candidates": 0,
        "route": "NO_CROSS_REPLICA_CT_SUPPORT",
        "screening_receipt": str(receipt_path),
        "screening_receipt_sha256": sha256_file(receipt_path),
        "analysis_log": str(log_path),
        "analysis_log_sha256": sha256_file(log_path),
        "rendered_tiff_count": rendered_tiff_count,
        "reused": True,
    }


def checkpoint_batch(
    path: Path,
    *,
    ranking_path: Path,
    plan_path: Path,
    renderer: Path,
    checkpoint: Path,
    shard_count: int,
    shard_index: int,
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> None:
    completed_count = sum(row["status"] in TERMINAL_TASK_STATUSES for row in states.values())
    failed_count = sum(row["status"] == "FAILED" for row in states.values())
    ct_count = sum(
        row.get("route") == "RAW_CT_REVIEW_REQUIRED"
        for row in states.values()
    )
    write_json(
        path,
        {
            "kind": "campaign_x_phase4_coverage_surface_v2_robust_shard_v1",
            "status": (
                "COMPLETED_WITH_CT_QUEUE"
                if completed_count == len(tasks)
                and failed_count == 0
                and ct_count > 0
                else (
                    "COMPLETED_DIAGNOSTIC_ONLY"
                if completed_count == len(tasks) and failed_count == 0
                    else "RUNNING_OR_PARTIAL"
                )
            ),
            "updated_at_utc": utc_now(),
            "ranking": {
                "path": str(ranking_path),
                "sha256": sha256_file(ranking_path),
            },
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
            "completed_count": completed_count,
            "failed_count": failed_count,
            "strict_ct_queue_count": ct_count,
            "tasks": [
                {
                    "seed_id": task["seed_id"],
                    "sample_id": task["sample_id"],
                    "global_localized_rank": task["global_localized_rank"],
                    "robust_queue_reasons": task["robust_queue_reasons"],
                    **states[str(task["seed_id"])],
                }
                for task in tasks
            ],
            "policy": [
                "three depths and two offsets are mandatory",
                "strict positives route directly to orthogonal CT",
                "strict negatives remain eligible for the frozen high-recall CT router",
                "no result is accepted as ink or letters",
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference-script", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cache-gb", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()

    ranking_path = args.ranking.resolve()
    plan_path = args.plan.resolve()
    runtime_root = args.runtime_root.resolve()
    renderer = args.renderer.resolve()
    checkpoint = args.checkpoint.resolve()
    inference_script = args.inference_script.resolve()
    analysis_script = args.analysis_script.resolve()
    if sha256_file(renderer) != EXPECTED_RENDERER_SHA256:
        raise RuntimeError("renderer hash mismatch")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash mismatch")
    if not inference_script.is_file() or not analysis_script.is_file():
        raise RuntimeError("inference or analysis script is missing")
    if not 1 <= args.shard_count <= 8:
        raise ValueError("shard-count must be between 1 and 8")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index is outside shard-count")

    _, queue, surfaces = validate_inputs(ranking_path, plan_path)
    tasks = [
        task
        for index, task in enumerate(queue)
        if index % args.shard_count == args.shard_index
    ]
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise ValueError("max-tasks must be positive")
        tasks = tasks[: args.max_tasks]
    receipt_path = (
        runtime_root / f"ROBUST_BATCH_SHARD_{args.shard_index:02d}.json"
    )
    states = {
        str(task["seed_id"]): {"status": "PENDING"} for task in tasks
    }

    for position, task in enumerate(tasks, start=1):
        seed_id = str(task["seed_id"])
        surface = surfaces[seed_id]
        output = runtime_root / str(task["sample_id"]) / seed_id
        tiffs = output / "tiffs"
        screening = output / "robust_gp_scroll1_v1"
        analysis_dir = screening / "analysis"
        started = time.monotonic()
        try:
            prior = completed(output)
            if prior is None:
                prior = no_common_valid_pixels(output)
            if prior is not None:
                states[seed_id] = prior
                print(
                    f"REUSE:{position}/{len(tasks)}:{seed_id}:{prior['status']}",
                    flush=True,
                )
                checkpoint_batch(
                    receipt_path,
                    ranking_path=ranking_path,
                    plan_path=plan_path,
                    renderer=renderer,
                    checkpoint=checkpoint,
                    shard_count=args.shard_count,
                    shard_index=args.shard_index,
                    tasks=tasks,
                    states=states,
                )
                continue
            print(f"START:{position}/{len(tasks)}:{seed_id}", flush=True)
            current_tiffs = exact_tiffs(tiffs)
            # A coarse central TIFF is reused when this runner shares its
            # runtime with coarse screening. A separately isolated robust
            # runtime legitimately starts with no TIFFs and must render the
            # same full 65-slice stack from scratch.
            if len(current_tiffs) not in {0, 1, 65}:
                raise RuntimeError(
                    f"unexpected render count before robust run: {len(current_tiffs)}"
                )
            if len(current_tiffs) in {0, 1}:
                run_logged(
                    [
                        str(renderer),
                        "--segmentation",
                        str(
                            runtime_root
                            / "source_jobs"
                            / str(surface["source_job_id"])
                            / "surface"
                        ),
                        "--volume",
                        str(
                            runtime_root
                            / "ct_metadata_cache"
                            / f"{task['sample_id']}.zarr"
                        ),
                        "--remote-url",
                        str(surface["ct_uri"]),
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
                        "65",
                        "--slice-step",
                        "1",
                        "--cache-gb",
                        str(args.cache_gb),
                        "--timeout",
                        "90",
                        "--voxel-size",
                        str(surface["voxel_size_um"]),
                        "--voxel-unit",
                        "micrometer",
                        "--tif-output",
                        str(tiffs),
                        "--log-path",
                        str(output / "robust-render.log"),
                    ],
                    output / "robust-render.stdout.log",
                )
            if len(exact_tiffs(tiffs)) != 65:
                raise RuntimeError("robust renderer did not produce 65 TIFFs")
            if not (screening / "INK_SCREENING_RECEIPT.json").is_file():
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
                        str(screening),
                        "--depth-centers",
                        "25,32,39",
                        "--tiling-offsets",
                        "0,8",
                        "--frames",
                        "26",
                        "--source-pixel-um",
                        str(surface["voxel_size_um"]),
                        "--training-pixel-um",
                        training_pixel_um(),
                        "--source-slice-um",
                        str(surface["voxel_size_um"]),
                        "--training-slice-um",
                        training_slice_um(),
                        "--tile-size",
                        "64",
                        "--stride",
                        "16",
                        "--batch-size",
                        str(args.batch_size),
                        "--min-valid-ratio",
                        "0.60",
                        "--device",
                        "cuda",
                    ],
                    output / "robust-inference.stdout.log",
                )
            analysis_path = analysis_dir / "INK_STABILITY_ANALYSIS.json"
            if not analysis_path.is_file():
                run_logged(
                    [
                        "python3",
                        str(analysis_script),
                        "--sample-id",
                        str(task["sample_id"]),
                        "--screening-dir",
                        str(screening),
                        "--tiff-dir",
                        str(tiffs),
                        "--output",
                        str(analysis_dir),
                        "--source-center",
                        "32",
                        "--source-pixel-um",
                        str(surface["voxel_size_um"]),
                        "--training-pixel-um",
                        training_pixel_um(),
                        "--glyph-threshold",
                        "0.5",
                        "--hotspots",
                        "12",
                        "--crop-size",
                        "384",
                    ],
                    output / "robust-analysis.stdout.log",
                )
            current = completed(output)
            if current is None:
                raise RuntimeError("robust output failed postcondition")
            current["reused"] = False
            current["duration_seconds"] = round(time.monotonic() - started, 6)
            current["rendered_tiff_count"] = len(exact_tiffs(tiffs))
            if current["rendered_tiff_count"] != 65:
                raise RuntimeError(
                    "robust CT stack must remain intact for orthogonal review"
                )
            states[seed_id] = current
            print(
                f"DONE:{position}/{len(tasks)}:{seed_id}:"
                f"{current['screening_outcome']}:"
                f"forms={current['glyph_like_candidate_count']}:"
                f"rows4={current['rows_with_at_least_four_candidates']}:"
                f"{current['duration_seconds']}s",
                flush=True,
            )
        except Exception as error:
            states[seed_id] = {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "duration_seconds": round(time.monotonic() - started, 6),
            }
            print(
                f"FAILED:{position}/{len(tasks)}:{seed_id}:{error}",
                flush=True,
            )
        checkpoint_batch(
            receipt_path,
            ranking_path=ranking_path,
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
        raise RuntimeError(f"robust shard failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
