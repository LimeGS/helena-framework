#!/usr/bin/env python3
"""Memory-bounded coarse ink scan for a large official surface stack.

The model is loaded once.  Exact <=4 cm² source windows are then read from
disk one at a time, avoiding a whole-surface depth stack in RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import sys


# The repository root, so the shared device resolver is importable from a
# script launched by path. `auto` has to mean the same thing in every runner or
# it is six defaults again, wearing one word.
_ROOT = Path(__file__).resolve()
while _ROOT != _ROOT.parent and not (_ROOT / "framework").is_dir():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from framework.contracts.host_probe import resolve_device  # noqa: E402
_STAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()) / "framework/stages"
for _stage_scripts in _STAGE_ROOT.glob("*/scripts"):
    _stage_scripts_text = str(_stage_scripts)
    if _stage_scripts_text not in sys.path:
        sys.path.insert(0, _stage_scripts_text)
ROOT = _STAGE_ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

import numpy as np
from PIL import Image

from framework.contracts.slice_order import ordered_tiff_files
from rank_coarse_ink_windows import (
    physical_side_pixels,
    sha256_file,
    starts,
    utc_now,
    window_score,
    write_json,
)
from run_ink_timesformer import (
    depth_positions,
    infer_map,
    interpolate_depth,
    load_model,
    resize_stack,
    save_probability_png,
)


def load_crop_stack(
    files: list[Path],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for path in files:
        with Image.open(path) as image:
            arrays.append(np.asarray(image.crop(box), dtype=np.uint8))
    shapes = {value.shape for value in arrays}
    if len(shapes) != 1:
        raise RuntimeError("cropped TIFF slices do not share one shape")
    return np.stack(arrays)


def enumerate_boxes(
    width: int,
    height: int,
    *,
    side: int,
    step: int,
) -> list[tuple[int, int, int, int]]:
    return [
        (x0, y0, x0 + side, y0 + side)
        for y0 in starts(height, side, step)
        for x0 in starts(width, side, step)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--surface-id", required=True)
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-pixel-um", type=float, required=True)
    parser.add_argument("--training-pixel-um", type=float, required=True)
    parser.add_argument("--source-slice-um", type=float, required=True)
    parser.add_argument("--training-slice-um", type=float, required=True)
    parser.add_argument("--size-mm", type=float, default=20.0)
    parser.add_argument("--step-fraction", type=float, default=1.0)
    parser.add_argument("--depth-center", type=int, default=15)
    parser.add_argument("--frames", type=int, default=26)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-valid-ratio", type=float, default=0.60)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--device", default="auto",
                        help="auto (the card if this host has one), cpu, or cuda[:N] to require one")
    args = parser.parse_args()

    # Resolve the device before anything expensive: an explicit `cuda` on a host
    # with no card is refused here rather than several minutes into a load, and
    # `auto` becomes the word the receipt can stand behind.
    _device = resolve_device(args.device)
    args.device = _device["device"]

    started = time.monotonic()
    tiff_dir = args.tiff_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files, slice_ordering = ordered_tiff_files(tiff_dir)
    with Image.open(files[len(files) // 2]) as image:
        width, height = image.size
        source_mode = image.mode
    side = physical_side_pixels(args.size_mm, args.source_pixel_um)
    step = max(1, round(side * args.step_fraction))
    boxes = enumerate_boxes(width, height, side=side, step=step)
    positions = depth_positions(
        args.depth_center,
        args.frames,
        source_slice_um=args.source_slice_um,
        training_slice_um=args.training_slice_um,
    )
    model = load_model(args.checkpoint.resolve(), args.frames, args.device)
    target_side = round(side * args.source_pixel_um / args.training_pixel_um)
    records: list[dict[str, Any]] = []
    for index, box in enumerate(boxes, start=1):
        window_started = time.monotonic()
        stack = load_crop_stack(files, box)
        normalized = interpolate_depth(stack, positions)
        normalized = resize_stack(
            normalized,
            target_height=target_side,
            target_width=target_side,
        )
        probability, valid, counts = infer_map(
            normalized,
            model,
            device=args.device,
            tile_size=args.tile_size,
            stride=args.stride,
            tiling_offset=0,
            batch_size=args.batch_size,
            min_valid_ratio=args.min_valid_ratio,
        )
        values = probability[valid]
        statistics = (
            window_score(values)
            if values.size
            else {
                "score": 0.0,
                "p99": 0.0,
                "p99_9": 0.0,
                "top_1pct_mean": 0.0,
                "active_fraction_ge_0_5": 0.0,
            }
        )
        png_path = output / f"window-{index:03d}.png"
        save_probability_png(png_path, probability, valid)
        records.append(
            {
                "window_id": f"{args.surface_id}-w{index:03d}",
                "source_crop_xyxy": list(box),
                "source_area_cm2": (
                    side * args.source_pixel_um / 10000.0
                )
                ** 2,
                "valid_prediction_ratio": float(valid.mean()),
                **counts,
                **statistics,
                "probability_preview": png_path.name,
                "probability_preview_sha256": sha256_file(png_path),
                "duration_seconds": time.monotonic() - window_started,
            }
        )
        del stack, normalized, probability, valid, values
        gc.collect()

    ranked = sorted(
        records,
        key=lambda row: (-row["score"], row["source_crop_xyxy"]),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    receipt = {
        "kind": "campaign_x_phase4_large_surface_coarse_window_scan_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_PRIORITIZATION_ONLY",
        "sample_id": args.sample_id,
        "surface_id": args.surface_id,
        "input": {
            "tiff_dir": str(tiff_dir),
            "slice_count": len(files),
            "slice_ordering": slice_ordering,
            "mode": source_mode,
            "shape_y_x": [height, width],
            "representative_hashes": {
                files[0].name: sha256_file(files[0]),
                files[len(files) // 2].name: sha256_file(
                    files[len(files) // 2]
                ),
                files[-1].name: sha256_file(files[-1]),
            },
        },
        "physical_window": {
            "requested_size_mm": args.size_mm,
            "source_pixel_um": args.source_pixel_um,
            "side_pixels": side,
            "area_cm2": (side * args.source_pixel_um / 10000.0) ** 2,
            "step_pixels": step,
            "step_fraction": args.step_fraction,
        },
        "inference": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "depth_center": args.depth_center,
            "frames": args.frames,
            "source_slice_um": args.source_slice_um,
            "training_slice_um": args.training_slice_um,
            "training_pixel_um": args.training_pixel_um,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "min_valid_ratio": args.min_valid_ratio,
        },
        "window_count": len(records),
        "ranked_windows": ranked,
        "top_windows": ranked[: args.top_n],
        "duration_seconds": time.monotonic() - started,
        "policy": [
            "one depth and one tiling offset only prioritize full screening",
            "the model is loaded once and source crops are streamed one at a time",
            "every source crop is at or below 4 cm2",
            "no coarse result is accepted as ink or letters",
        ],
    }
    write_json(output / "LARGE_SURFACE_COARSE_SCAN.json", receipt)
    print(
        json.dumps(
            {
                "surface_id": args.surface_id,
                "window_count": len(records),
                "best": ranked[0] if ranked else None,
                "duration_seconds": receipt["duration_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
