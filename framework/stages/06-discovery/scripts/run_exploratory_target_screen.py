#!/usr/bin/env python3
"""Run one resumable failed-A/B exploratory target screen end to end.

This never promotes a failed A/B arm. It reconstructs the selected frozen arm,
guards incorrect-sign endpoints, exports the best measured window, streams only
the CT chunks required by the official renderer, crops below 4 cm², and applies
the frozen six-replica ink screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
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


def quality_flag(candidate_arm: str) -> str:
    if candidate_arm == "baseline":
        return "--allow-failed-ab-baseline-rescue"
    if candidate_arm == "r61":
        return "--allow-failed-ab-experimental-rescue"
    raise ValueError("candidate arm must be baseline or r61")


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(
            command,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def fetch_zarr_metadata(ct_url: str, cache: Path) -> None:
    for suffix in (".zgroup", ".zattrs", "0/.zarray"):
        destination = cache / suffix
        if destination.is_file() and destination.stat().st_size:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{ct_url.rstrip('/')}/{suffix}", timeout=60) as r:
            destination.write_bytes(r.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--candidate-arm", choices=("baseline", "r61"), required=True)
    parser.add_argument("--spiral-root", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-subdir")
    parser.add_argument("--output-subdir", default="exploratory_window_v1")
    parser.add_argument(
        "--allow-weak-holdout-local-screen",
        action="store_true",
        help="permit the separately marked 10-11 evaluable-holdout screen",
    )
    parser.add_argument("--cache-gb", type=int, default=24)
    args = parser.parse_args()

    root = args.root.resolve()
    target = root / "phase4" / "targets" / args.sample_id
    lock_path = target / "TARGET_LOCK.json"
    lock = read_json(lock_path)
    if lock.get("sample_id") != args.sample_id:
        raise RuntimeError("target lock sample binding mismatch")
    ct_url = str(lock["ct_uri"])
    voxel_um = float(lock["voxel_size_um"])
    quality_subdir = args.quality_subdir or (
        "baseline_sign_rescue_quality_map"
        if args.candidate_arm == "baseline"
        else "r61_exploratory_quality_map"
    )
    fit_ab = target / "fit_ab"
    quality = fit_ab / quality_subdir
    output = fit_ab / args.output_subdir
    pipeline_receipt = output / "EXPLORATORY_SCREEN_PIPELINE.json"

    quality_receipt = quality / "QUALITY_MAP_RECEIPT.json"
    if not quality_receipt.is_file():
        quality_command = [
                "python3",
                str(root / "scripts" / "build_target_quality_map.py"),
                "--root",
                str(root),
                "--sample-id",
                args.sample_id,
                "--spiral-root",
                str(args.spiral_root),
                "--fit-subdir",
                "fit_ab",
                "--candidate-arm",
                args.candidate_arm,
                "--output-subdir",
                quality_subdir,
                quality_flag(args.candidate_arm),
            ]
        if args.allow_weak_holdout_local_screen:
            quality_command.append("--allow-weak-holdout-local-screen")
        run_logged(
            quality_command,
            fit_ab / f"{quality_subdir}.stdout.log",
        )

    window_receipt = output / "EXPLORATORY_WINDOW_RECEIPT.json"
    if not window_receipt.is_file():
        run_logged(
            [
                "python3",
                str(root / "scripts" / "extract_exploratory_window.py"),
                "--root",
                str(root),
                "--sample-id",
                args.sample_id,
                "--fit-subdir",
                "fit_ab",
                "--quality-subdir",
                quality_subdir,
                "--candidate-arm",
                args.candidate_arm,
                "--output-subdir",
                args.output_subdir,
            ],
            fit_ab / f"{args.output_subdir}.stdout.log",
        )

    cache = target / "ct_remote_cache.zarr"
    fetch_zarr_metadata(ct_url, cache)

    render = output / "render_v1"
    tiffs = render / "tiffs"
    if len(list(tiffs.glob("*.tif"))) != 65:
        run_logged(
            [
                str(args.renderer),
                "--volume",
                str(cache),
                "--remote-url",
                ct_url,
                "--segmentation",
                str(output / "segmentation"),
                "--scale",
                "1",
                "--group-idx",
                "0",
                "--cache-gb",
                str(args.cache_gb),
                "--prefetch-remote",
                "--timeout",
                "90",
                "--num-slices",
                "65",
                "--slice-step",
                "1",
                "--auto-crop",
                "--zarr-output",
                str(render / "normal_stack.zarr"),
                "--zarr-compressor",
                "zstd",
                "--zarr-compression-level",
                "3",
                "--tif-output",
                str(tiffs),
                "--flatten",
                "--flatten-iterations",
                "10",
                "--voxel-size",
                str(voxel_um),
                "--voxel-unit",
                "micrometer",
                "--log-path",
                str(render / "render.log"),
            ],
            render / "launcher.log",
        )

    exact_tiffs = render / "exact_4cm2_tiffs"
    crop_receipt = exact_tiffs / "PHYSICAL_CROP_RECEIPT.json"
    if not crop_receipt.is_file():
        run_logged(
            [
                "python3",
                str(root / "scripts" / "crop_render_window.py"),
                "--sample-id",
                args.sample_id,
                "--input-dir",
                str(tiffs),
                "--output-dir",
                str(exact_tiffs),
                "--pixel-um",
                str(voxel_um),
                "--size-mm",
                "20",
                "--allow-smaller-fit",
            ],
            render / "crop.stdout.log",
        )

    screening = output / "ink_screening_v1"
    screening_receipt = screening / "INK_SCREENING_RECEIPT.json"
    if not screening_receipt.is_file():
        run_logged(
            [
                "python3",
                str(root / "scripts" / "run_ink_timesformer.py"),
                "--sample-id",
                args.sample_id,
                "--tiff-dir",
                str(exact_tiffs),
                "--checkpoint",
                str(args.checkpoint),
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
                # FIX-09: the training scale is resolved by the runner from the
                # ink lane profile, never restated here.
                "--tile-size",
                "64",
                "--stride",
                "16",
                "--batch-size",
                "64",
                "--min-valid-ratio",
                "0.60",
                "--device",
                "cuda",
            ],
            screening / "inference.log",
        )

    analysis = screening / "analysis" / "INK_STABILITY_ANALYSIS.json"
    if not analysis.is_file():
        run_logged(
            [
                "python3",
                str(root / "scripts" / "analyze_ink_stability.py"),
                "--sample-id",
                args.sample_id,
                "--screening-dir",
                str(screening),
                "--tiff-dir",
                str(exact_tiffs),
                "--output",
                str(screening / "analysis"),
                "--source-center",
                "32",
                "--source-pixel-um",
                str(voxel_um),
                # FIX-09: the analysis resolves the training scale from the ink
                # lane profile, never restated here.
                "--glyph-threshold",
                "0.5",
                "--hotspots",
                "12",
                "--crop-size",
                "384",
            ],
            screening / "analysis.stdout.log",
        )

    quality_data = read_json(quality_receipt)
    crop_data = read_json(crop_receipt)
    analysis_data = read_json(analysis)
    text = analysis_data["text_like_screening"]
    write_json(
        pipeline_receipt,
        {
            "kind": "campaign_x_phase4_exploratory_screen_pipeline_v1",
            "generated_at_utc": utc_now(),
            "status": "COMPLETED",
            "sample_id": args.sample_id,
            "candidate_arm": args.candidate_arm,
            "weak_holdout_local_screen": args.allow_weak_holdout_local_screen,
            "ct_url": ct_url,
            "voxel_um": voxel_um,
            "geometry_qualified": False,
            "quality_classification": quality_data["classification"],
            "crop": crop_data["crop"],
            "text_like_screening": {
                "glyph_like_candidate_count": text["glyph_like_candidate_count"],
                "row_band_count": text["row_band_count"],
                "rows_with_at_least_four_candidates": text[
                    "rows_with_at_least_four_candidates"
                ],
                "screening_outcome": text["screening_outcome"],
                "manual_review_routing": analysis_data["manual_review_routing"],
            },
            "bindings": {
                "target_lock_sha256": sha256_file(lock_path),
                "quality_receipt_sha256": sha256_file(quality_receipt),
                "window_receipt_sha256": sha256_file(window_receipt),
                "crop_receipt_sha256": sha256_file(crop_receipt),
                "screening_receipt_sha256": sha256_file(screening_receipt),
                "analysis_receipt_sha256": sha256_file(analysis),
                "checkpoint_sha256": sha256_file(args.checkpoint),
            },
            "explicit_non_claims": [
                "not a passed A/B",
                "not a geometry-qualified ROI",
                "not automatic ink or letter acceptance",
                "not a First Letters submission claim",
                *(
                    ["not supported by twelve evaluable holdout relations"]
                    if args.allow_weak_holdout_local_screen
                    else []
                ),
            ],
        },
    )
    print(pipeline_receipt.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
