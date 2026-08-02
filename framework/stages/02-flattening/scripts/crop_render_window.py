#!/usr/bin/env python3
"""Crop a rendered CT stack to one deterministic physical screening window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from flattening_gates import (  # noqa: E402
    MINIMUM_VALID_RASTER_FRACTION,
    MINIMUM_WINDOW_VALID_FRACTION,
    evaluate_raster_gate,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def crop_geometry(
    width: int,
    height: int,
    *,
    pixel_um: float,
    requested_size_mm: float,
    allow_smaller_fit: bool = False,
) -> tuple[int, int, int, int]:
    if width < 1 or height < 1:
        raise ValueError("input dimensions must be positive")
    if pixel_um <= 0 or requested_size_mm <= 0:
        raise ValueError("physical dimensions must be positive")
    pixels = math.floor(requested_size_mm * 1000.0 / pixel_um)
    if pixels < 1:
        raise ValueError("requested window is smaller than one pixel")
    if pixels > width or pixels > height:
        if allow_smaller_fit:
            pixels = min(pixels, width, height)
        else:
            raise ValueError(
                f"requested {pixels}x{pixels} crop exceeds {width}x{height} input"
            )
    if pixels < 1:
        raise ValueError(
            "input is too small to produce a positive fitted crop"
        )
    left = (width - pixels) // 2
    top = (height - pixels) // 2
    return left, top, left + pixels, top + pixels


def parse_box(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(item.strip()) for item in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1")
    return parts


def validate_crop_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    expected_pixels: int,
    allow_smaller_fit: bool = False,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError("crop is outside the input image")
    crop_width = x1 - x0
    crop_height = y1 - y0
    if crop_width < 1 or crop_height < 1:
        raise ValueError("crop dimensions must be positive")
    if crop_width != crop_height:
        raise ValueError("crop must be square")
    if crop_width == expected_pixels:
        return box
    largest_source_limited_side = min(expected_pixels, width, height)
    if not (
        allow_smaller_fit
        and expected_pixels > min(width, height)
        and crop_width == largest_source_limited_side
    ):
        raise ValueError(
            f"crop must be exactly {expected_pixels}x{expected_pixels} pixels"
            " unless it is explicitly allowed and equals the largest square "
            f"that fits the source ({largest_source_limited_side}x"
            f"{largest_source_limited_side})"
        )
    return box


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-um", type=float, required=True)
    parser.add_argument("--size-mm", type=float, default=20.0)
    parser.add_argument(
        "--allow-smaller-fit",
        action="store_true",
        help=(
            "when the render is smaller than the requested square, use the largest "
            "centred square that fits and record its achieved area"
        ),
    )
    parser.add_argument(
        "--crop-xyxy",
        type=parse_box,
        help="optional exact x0,y0,x1,y1 source crop; defaults to centre",
    )
    parser.add_argument(
        "--render-log",
        type=Path,
        required=True,
        help=(
            "renderer log carrying 'Rasterized N / M points (X%%)'. The "
            "unrasterized remainder becomes zero fill, which is the documented "
            "ink false-positive confuser, so the fraction is gated here rather "
            "than left to a downstream reviewer"
        ),
    )
    parser.add_argument(
        "--minimum-valid-raster-fraction",
        type=float,
        default=MINIMUM_VALID_RASTER_FRACTION,
    )
    parser.add_argument(
        "--window-receipt",
        type=Path,
        help=(
            "optional EXPLORATORY_WINDOW_RECEIPT.json / CONNECTED_ROI_RECEIPT.json "
            "whose validity gate is re-asserted before this crop is written"
        ),
    )
    args = parser.parse_args()

    # Fail closed before any pixel is written: a crop of an under-rasterized
    # render is an under-rasterized crop, and the centre of the frame is not
    # necessarily the valid part of the surface.
    raster_gate = evaluate_raster_gate(
        args.render_log.resolve().read_text(encoding="utf-8", errors="replace"),
        minimum_valid_raster_fraction=float(args.minimum_valid_raster_fraction),
    )
    window_gate: dict[str, Any] | None = None
    if args.window_receipt is not None:
        window_receipt = json.loads(
            args.window_receipt.resolve().read_text(encoding="utf-8")
        )
        window_gate = window_receipt.get("window_validity_gate")
        if window_gate is None:
            raise RuntimeError(
                f"{args.window_receipt} carries no window_validity_gate; the "
                "window was exported before the Stage 02 validity gate existed "
                "and its sentinel coverage is UNMEASURED"
            )
        if not window_gate.get("passed"):
            raise RuntimeError(
                f"{args.window_receipt} records a failed window validity gate"
            )
        if float(window_gate["valid_fraction"]) < MINIMUM_WINDOW_VALID_FRACTION:
            raise RuntimeError(
                f"window valid_fraction {window_gate['valid_fraction']:.4f} is "
                f"below {MINIMUM_WINDOW_VALID_FRACTION:.2f}"
            )

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    files, slice_ordering = ordered_tiff_files(input_dir)

    with Image.open(files[0]) as first:
        source_mode = first.mode
        source_size = first.size
    pixels = math.floor(args.size_mm * 1000.0 / args.pixel_um)
    box = (
        validate_crop_box(
            args.crop_xyxy,
            width=source_size[0],
            height=source_size[1],
            expected_pixels=pixels,
            allow_smaller_fit=args.allow_smaller_fit,
        )
        if args.crop_xyxy is not None
        else crop_geometry(
            *source_size,
            pixel_um=args.pixel_um,
            requested_size_mm=args.size_mm,
            allow_smaller_fit=args.allow_smaller_fit,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for source in files:
        with Image.open(source) as image:
            if image.mode != source_mode or image.size != source_size:
                raise RuntimeError(f"inconsistent TIFF slice: {source}")
            cropped = image.crop(box)
            destination = output_dir / source.name
            cropped.save(destination, compression="tiff_lzw")
        artifacts.append(
            {
                "name": destination.name,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    pixels = box[2] - box[0]
    achieved_mm = pixels * args.pixel_um / 1000.0
    receipt = {
        "kind": "campaign_x_phase4_physical_render_crop_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED",
        "sample_id": args.sample_id,
        "input": {
            "directory": str(input_dir),
            "slice_count": len(files),
            "slice_ordering": slice_ordering,
            "mode": source_mode,
            "shape_y_x": [source_size[1], source_size[0]],
        },
        "render_quality_gate": {
            **raster_gate,
            "render_log": str(args.render_log.resolve()),
            "render_log_sha256": sha256_file(args.render_log.resolve()),
        },
        "window_validity_gate": window_gate,
        "crop": {
            "box_left_top_right_bottom": list(box),
            "shape_y_x": [pixels, pixels],
            "pixel_um": args.pixel_um,
            "requested_size_mm": args.size_mm,
            "requested_area_cm2": (args.size_mm / 10.0) ** 2,
            "achieved_size_mm": achieved_mm,
            "achieved_area_cm2": (achieved_mm / 10.0) ** 2,
            "surface_limited": pixels < math.floor(
                args.size_mm * 1000.0 / args.pixel_um
            ),
            "rounding_policy": "floor pixels so the physical area never exceeds the request",
            "selection_policy": (
                (
                    "explicit largest source-limited square"
                    if pixels
                    < math.floor(args.size_mm * 1000.0 / args.pixel_um)
                    else "explicit source coordinates"
                )
                if args.crop_xyxy is not None
                else (
                    "centred largest square not exceeding request"
                    if pixels < math.floor(args.size_mm * 1000.0 / args.pixel_um)
                    else "centred crop"
                )
            ),
        },
        "artifacts": artifacts,
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not a First Letters submission claim",
        ],
    }
    write_json(output_dir / "PHYSICAL_CROP_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
