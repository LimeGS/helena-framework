#!/usr/bin/env python3
"""Run the frozen six-replica screen on globally ranked expanded windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(
            command,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def robust_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    screening = analysis["text_like_screening"]
    positive_outcomes = {
        "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW",
        # Retained only so older frozen analyses remain readable.
        "TEXT_LIKE_SUPPORT_OBSERVED",
    }
    return {
        "screening_outcome": screening["screening_outcome"],
        "glyph_like_candidate_count": screening["glyph_like_candidate_count"],
        "row_band_count": screening["row_band_count"],
        "rows_with_at_least_four_candidates": screening[
            "rows_with_at_least_four_candidates"
        ],
        "route": (
            "RAW_CT_REVIEW_REQUIRED"
            if screening["screening_outcome"] in positive_outcomes
            else "NOT_QUEUED_TEXT_LIKE_GATE_FAILED"
        ),
    }


def select_ranked_windows(
    ranking: dict[str, Any],
    *,
    start_rank: int,
    limit: int,
) -> list[dict[str, Any]]:
    if start_rank < 1:
        raise ValueError("start_rank must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    selected = [
        window
        for window in ranking["global_priority"]
        if int(window["global_rank"]) >= start_rank
    ][:limit]
    if not selected:
        raise RuntimeError(f"no ranked windows available from rank {start_rank}")
    actual_ranks = [int(window["global_rank"]) for window in selected]
    expected_ranks = list(range(start_rank, start_rank + len(selected)))
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"global ranking is not contiguous: {actual_ranks} != {expected_ranks}"
        )
    return selected


def crop_requires_smaller_fit(
    window: dict[str, Any],
    *,
    pixel_um: float,
    requested_size_mm: float = 20.0,
) -> bool:
    """Validate the frozen crop metadata and identify a source-limited square."""

    crop = [int(value) for value in window["source_crop_xyxy"]]
    if len(crop) != 4:
        raise ValueError("source_crop_xyxy must contain four coordinates")
    x0, y0, x1, y1 = crop
    width = x1 - x0
    height = y1 - y0
    if width < 1 or height < 1 or width != height:
        raise ValueError("frozen source crop must be a positive square")
    declared_side = window.get("source_side_pixels")
    if declared_side is not None and int(declared_side) != width:
        raise ValueError("source_side_pixels disagrees with source_crop_xyxy")
    expected = math.floor(requested_size_mm * 1000.0 / pixel_um)
    if width > expected:
        raise ValueError("frozen source crop exceeds the requested physical size")
    return width < expected


def checkpoint_receipt(
    path: Path,
    *,
    ranking_path: Path,
    checkpoint: Path,
    model_family: str,
    screening_name: str,
    start_rank: int,
    limit: int,
    results: list[dict[str, Any]],
    stopped_on_text_like: bool,
    continue_after_text_like: bool,
) -> None:
    positive_count = sum(
        result.get("route") == "RAW_CT_REVIEW_REQUIRED"
        for result in results
    )
    actually_stopped = stopped_on_text_like and not continue_after_text_like
    write_json(
        path,
        {
            "kind": "campaign_x_phase4_expanded_window_robust_batch_v1",
            "status": (
                "PAUSED_FOR_RAW_CT_REVIEW"
                if actually_stopped
                else (
                    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE"
                    if positive_count
                    else "COMPLETED_DIAGNOSTIC_ONLY"
                )
            ),
            "updated_at_utc": utc_now(),
            "global_ranking": str(ranking_path),
            "global_ranking_sha256": sha256_file(ranking_path),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "model_family": model_family,
            },
            "screening_name": screening_name,
            "requested_start_rank": start_rank,
            "requested_limit": limit,
            "selected_global_ranks": [
                int(result["global_rank"]) for result in results
            ],
            "completed_count": len(results),
            "text_like_positive_count": positive_count,
            "stopped_on_text_like": actually_stopped,
            "continue_after_text_like": continue_after_text_like,
            "results": results,
            "policy": [
                "three depths and two tiling offsets are frozen",
                "support must persist at every sampled depth for both offsets",
                "text-like support requires at least ten bounded forms",
                "text-like support requires two rows with at least four forms",
                (
                    "text-like support is recorded without stopping the frozen "
                    "comparison batch"
                    if continue_after_text_like
                    else "text-like support pauses compute for raw-CT fiber review"
                ),
            ],
            "explicit_non_claims": [
                "not automatic ink acceptance",
                "not automatic letter acceptance",
                "not a First Letters submission claim",
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument(
        "--ranking-path",
        type=Path,
        help="Optional frozen ranking JSON; defaults to the canonical top list.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        default="timesformer_scroll5_july_retreat",
    )
    parser.add_argument(
        "--screening-name",
        default="ink_screening_v1",
        help="Subdirectory used beneath each robust window.",
    )
    parser.add_argument(
        "--batch-receipt-name",
        default="ROBUST_WINDOW_BATCH_RECEIPT.json",
    )
    parser.add_argument(
        "--robust-root-name",
        default="robust_windows_v1",
    )
    parser.add_argument(
        "--continue-after-text-like",
        action="store_true",
        help="Complete the frozen comparison batch even when a window passes.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="GPU batching only; does not alter tiles or probabilities.",
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--start-rank",
        type=int,
        default=1,
        help="First frozen global rank to process (1-based, inclusive).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    batch_root = (
        args.batch_root.resolve()
        if args.batch_root
        else root / "phase4" / "expanded_candidate_surface_screen_v1"
    )
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for value, label in (
        (args.screening_name, "screening name"),
        (args.batch_receipt_name, "batch receipt name"),
        (args.robust_root_name, "robust root name"),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"unsafe {label}: {value!r}")
    ranking_path = (
        args.ranking_path.resolve()
        if args.ranking_path
        else batch_root / "GLOBAL_COARSE_WINDOW_RANKING.json"
    )
    ranking = read_json(ranking_path)
    if ranking.get("status") not in {
        "COMPLETED_PRIORITIZATION_ONLY",
        "COMPLETED_REMAINDER_PRIORITIZATION_ONLY",
    }:
        raise RuntimeError("global coarse ranking is not complete")

    output_root = batch_root / args.robust_root_name
    receipt_path = output_root / args.batch_receipt_name
    results: list[dict[str, Any]] = []
    stopped = False
    selected = select_ranked_windows(
        ranking,
        start_rank=args.start_rank,
        limit=args.limit,
    )

    for index, window in enumerate(selected, start=1):
        sample_id = str(window["sample_id"])
        surface_id = str(window["surface_id"])
        rank = int(window["global_rank"])
        voxel_um = float(
            read_json(root / "phase4" / "targets" / sample_id / "TARGET_LOCK.json")[
                "voxel_size_um"
            ]
        )
        output = output_root / f"rank-{rank:02d}-{sample_id}-{surface_id}"
        tiffs = output / "tiffs"
        screening = output / args.screening_name
        analysis_dir = screening / "analysis"
        analysis_path = analysis_dir / "INK_STABILITY_ANALYSIS.json"
        print(
            f"START:{index}/{len(selected)}:rank-{rank:02d}:{sample_id}:{surface_id}",
            flush=True,
        )

        if not (tiffs / "PHYSICAL_CROP_RECEIPT.json").is_file():
            crop = ",".join(str(int(value)) for value in window["source_crop_xyxy"])
            crop_command = [
                "python3",
                str(root / "scripts" / "crop_render_window.py"),
                "--sample-id",
                sample_id,
                "--input-dir",
                str(batch_root / sample_id / surface_id / "tiffs"),
                "--output-dir",
                str(tiffs),
                "--pixel-um",
                str(voxel_um),
                "--size-mm",
                "20",
                "--crop-xyxy",
                crop,
            ]
            if crop_requires_smaller_fit(window, pixel_um=voxel_um):
                crop_command.append("--allow-smaller-fit")
            run_logged(
                crop_command,
                output / "crop.stdout.log",
            )
        if not (screening / "INK_SCREENING_RECEIPT.json").is_file():
            run_logged(
                [
                    "python3",
                    str(root / "scripts" / "run_ink_timesformer.py"),
                    "--sample-id",
                    sample_id,
                    "--tiff-dir",
                    str(tiffs),
                    "--checkpoint",
                    str(checkpoint),
                    "--model-family",
                    args.model_family,
                    "--output",
                    str(screening),
                    "--depth-centers",
                    "25,32,39",
                    "--tiling-offsets",
                    "0,8",
                    "--frames",
                    "26",
                    "--source-pixel-um",
                    str(voxel_um),
                    "--source-slice-um",
                    str(voxel_um),
                    # FIX-09: the training scale is resolved by the runner from
                    # the ink lane profile, never restated here.
                    "--tile-size",
                    "64",
                    "--stride",
                    "16",
                    "--batch-size",
                    str(args.inference_batch_size),
                    "--min-valid-ratio",
                    "0.60",
                    "--device",
                    "cuda",
                ],
                output / f"{args.screening_name}.inference.stdout.log",
            )
        if not analysis_path.is_file():
            run_logged(
                [
                    "python3",
                    str(root / "scripts" / "analyze_ink_stability.py"),
                    "--sample-id",
                    sample_id,
                    "--screening-dir",
                    str(screening),
                    "--tiff-dir",
                    str(tiffs),
                    "--output",
                    str(analysis_dir),
                    "--source-center",
                    "32",
                    "--source-pixel-um",
                    str(voxel_um),
                    # FIX-09: the analysis resolves the training scale from the
                    # ink lane profile, never restated here.
                    "--glyph-threshold",
                    "0.5",
                    "--hotspots",
                    "12",
                    "--crop-size",
                    "384",
                ],
                output / f"{args.screening_name}.analysis.stdout.log",
            )

        summary = robust_summary(read_json(analysis_path))
        result = {
            "global_rank": rank,
            "sample_id": sample_id,
            "surface_id": surface_id,
            "source_crop_xyxy": window["source_crop_xyxy"],
            "coarse_score": window["score"],
            "analysis": str(analysis_path),
            "analysis_sha256": sha256_file(analysis_path),
            "model_family": args.model_family,
            "checkpoint_sha256": sha256_file(checkpoint),
            **summary,
        }
        results.append(result)
        print(
            f"DONE:{index}/{len(selected)}:rank-{rank:02d}:"
            f"{summary['screening_outcome']}:"
            f"forms={summary['glyph_like_candidate_count']}:"
            f"rows4={summary['rows_with_at_least_four_candidates']}",
            flush=True,
        )
        stopped = summary["route"] == "RAW_CT_REVIEW_REQUIRED"
        checkpoint_receipt(
            receipt_path,
            ranking_path=ranking_path,
            checkpoint=checkpoint,
            model_family=args.model_family,
            screening_name=args.screening_name,
            start_rank=args.start_rank,
            limit=args.limit,
            results=results,
            stopped_on_text_like=stopped,
            continue_after_text_like=args.continue_after_text_like,
        )
        if stopped and not args.continue_after_text_like:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
