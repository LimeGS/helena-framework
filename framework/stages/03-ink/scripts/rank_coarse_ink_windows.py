#!/usr/bin/env python3
"""Rank exact physical windows from coarse Phase 4 ink maps.

This is a throughput stage.  It uses one depth/tiling pass to decide where to
spend the full six-replica screen; it never accepts ink or letters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


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


def physical_side_pixels(size_mm: float, pixel_um: float) -> int:
    if size_mm <= 0 or pixel_um <= 0:
        raise ValueError("physical values must be positive")
    return math.floor(size_mm * 1000.0 / pixel_um)


def fitted_source_side_pixels(
    source_shape_y_x: tuple[int, int],
    requested_side: int,
    *,
    allow_smaller_fit: bool,
) -> int:
    if requested_side < 1 or min(source_shape_y_x) < 1:
        raise ValueError("pixel dimensions must be positive")
    available = min(source_shape_y_x)
    if available < requested_side and not allow_smaller_fit:
        raise ValueError("source cannot contain the requested physical square")
    return min(requested_side, available)


def starts(length: int, side: int, step: int) -> list[int]:
    if length < side or side < 1 or step < 1:
        raise ValueError("invalid window geometry")
    values = list(range(0, length - side + 1, step))
    if values[-1] != length - side:
        values.append(length - side)
    return values


def window_score(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("window has no valid values")
    p99, p999 = np.percentile(values, [99.0, 99.9])
    top = values[values >= p99]
    active_fraction = float((values >= 0.5).mean())
    score = (
        float(p99)
        + float(p999)
        + float(top.mean())
        + 5.0 * min(active_fraction, 0.05)
    )
    return {
        "score": score,
        "p99": float(p99),
        "p99_9": float(p999),
        "top_1pct_mean": float(top.mean()),
        "active_fraction_ge_0_5": active_fraction,
    }


def separated(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    minimum_center_distance: float,
) -> bool:
    for other in selected:
        if other["surface_id"] != candidate["surface_id"]:
            continue
        dy = candidate["center_y_x"][0] - other["center_y_x"][0]
        dx = candidate["center_y_x"][1] - other["center_y_x"][1]
        if math.hypot(dy, dx) < minimum_center_distance:
            return False
    return True


def draw_preview(
    tiff_path: Path,
    windows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = Image.open(tiff_path).convert("RGB")
    scale = min(1.0, 1400.0 / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.BILINEAR,
        )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row in windows:
        x0, y0, x1, y1 = row["source_crop_xyxy"]
        box = tuple(round(value * scale) for value in (x0, y0, x1, y1))
        draw.rectangle(box, outline="#ffcc33", width=3)
        label = f"#{row['rank']} {row['score']:.3f}"
        draw.rectangle(
            (box[0], box[1], box[0] + 92, box[1] + 17),
            fill="#07101d",
        )
        draw.text((box[0] + 3, box[1] + 2), label, fill="#ffcc33", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--screening-name",
        default="coarse_screen_v1",
    )
    parser.add_argument("--source-pixel-um", type=float, required=True)
    parser.add_argument("--normalized-pixel-um", type=float, required=True)
    parser.add_argument("--size-mm", type=float, default=20.0)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--step-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.70)
    parser.add_argument("--top-n", type=int, default=16)
    parser.add_argument("--max-per-surface", type=int, default=4)
    parser.add_argument(
        "--allow-smaller-fit",
        action="store_true",
        help=(
            "rank the largest square that fits when a surface cannot contain "
            "20x20 mm; every achieved area remains at or below the request"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized_side = physical_side_pixels(
        args.size_mm,
        args.normalized_pixel_um,
    )
    source_side = physical_side_pixels(args.size_mm, args.source_pixel_um)

    candidates: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for surface_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        surface_id = surface_dir.name
        screening_dir = surface_dir / args.screening_name
        map_path = screening_dir / "mean_probability.npy"
        receipt_path = screening_dir / "INK_SCREENING_RECEIPT.json"
        tiff_dir = surface_dir / "tiffs"
        tiff_files, slice_ordering = ordered_tiff_files(tiff_dir, allow_empty=True)
        if not map_path.exists() or not receipt_path.exists() or not tiff_files:
            continue
        # ``len // 2`` is the central slice of the *ordered* stack.  Before the
        # shared contract this list was lexicographic, so on an unpadded
        # 0.tif..64.tif render the "central" slice was 38.tif, not 32.tif.
        source_shape = Image.open(tiff_files[len(tiff_files) // 2]).size
        source_shape_y_x = (source_shape[1], source_shape[0])
        try:
            surface_source_side = fitted_source_side_pixels(
                source_shape_y_x,
                source_side,
                allow_smaller_fit=args.allow_smaller_fit,
            )
        except ValueError:
            source_records.append(
                {
                    "surface_id": surface_id,
                    "coarse_receipt": str(receipt_path),
                    "coarse_receipt_sha256": sha256_file(receipt_path),
                    "map": str(map_path),
                    "map_sha256": sha256_file(map_path),
                    "source_tiff_count": len(tiff_files),
                    "slice_ordering": slice_ordering,
                    "source_shape_y_x": list(source_shape_y_x),
                    "eligible_window_count": 0,
                    "skip_reason": "source cannot contain requested physical square",
                }
            )
            continue
        surface_normalized_side = max(
            1,
            math.floor(
                surface_source_side
                * args.source_pixel_um
                / args.normalized_pixel_um
            ),
        )
        probability = np.load(map_path).astype(np.float32)
        reduced = probability[:: args.downsample, :: args.downsample]
        surface_downsampled_side = min(
            max(1, surface_normalized_side // args.downsample),
            reduced.shape[0],
            reduced.shape[1],
        )
        valid = reduced > 0
        normalized_scale = args.normalized_pixel_um / args.source_pixel_um
        surface_step = max(
            1,
            round(surface_downsampled_side * args.step_fraction),
        )
        surface_candidates = 0
        for y0 in starts(reduced.shape[0], surface_downsampled_side, surface_step):
            for x0 in starts(
                reduced.shape[1],
                surface_downsampled_side,
                surface_step,
            ):
                local_valid = valid[
                    y0 : y0 + surface_downsampled_side,
                    x0 : x0 + surface_downsampled_side,
                ]
                valid_ratio = float(local_valid.mean())
                if valid_ratio < args.minimum_valid_ratio:
                    continue
                local = reduced[
                    y0 : y0 + surface_downsampled_side,
                    x0 : x0 + surface_downsampled_side,
                ][local_valid]
                statistics = window_score(local)
                center_y = (
                    y0 + surface_downsampled_side / 2
                ) * args.downsample
                center_x = (
                    x0 + surface_downsampled_side / 2
                ) * args.downsample
                source_center_y = round(center_y * normalized_scale)
                source_center_x = round(center_x * normalized_scale)
                source_x0 = min(
                    max(0, source_center_x - surface_source_side // 2),
                    source_shape[0] - surface_source_side,
                )
                source_y0 = min(
                    max(0, source_center_y - surface_source_side // 2),
                    source_shape[1] - surface_source_side,
                )
                candidates.append(
                    {
                        "surface_id": surface_id,
                        "center_y_x": [center_y, center_x],
                        "normalized_window_yxyx": [
                            y0 * args.downsample,
                            x0 * args.downsample,
                            (y0 + surface_downsampled_side) * args.downsample,
                            (x0 + surface_downsampled_side) * args.downsample,
                        ],
                        "source_crop_xyxy": [
                            source_x0,
                            source_y0,
                            source_x0 + surface_source_side,
                            source_y0 + surface_source_side,
                        ],
                        "source_side_pixels": surface_source_side,
                        "normalized_side_pixels": (
                            surface_downsampled_side * args.downsample
                        ),
                        "achieved_area_cm2": (
                            surface_source_side
                            * args.source_pixel_um
                            / 10000.0
                        )
                        ** 2,
                        "valid_ratio": valid_ratio,
                        **statistics,
                    }
                )
                surface_candidates += 1
        source_records.append(
            {
                "surface_id": surface_id,
                "coarse_receipt": str(receipt_path),
                "coarse_receipt_sha256": sha256_file(receipt_path),
                "map": str(map_path),
                "map_sha256": sha256_file(map_path),
                "source_tiff_count": len(tiff_files),
                "slice_ordering": slice_ordering,
                "source_shape_y_x": [source_shape[1], source_shape[0]],
                "source_side_pixels": surface_source_side,
                "achieved_area_cm2": (
                    surface_source_side * args.source_pixel_um / 10000.0
                )
                ** 2,
                "eligible_window_count": surface_candidates,
            }
        )

    candidates.sort(
        key=lambda row: (
            -row["score"],
            row["surface_id"],
            row["source_crop_xyxy"],
        )
    )
    selected: list[dict[str, Any]] = []
    per_surface: dict[str, int] = {}
    for candidate in candidates:
        surface_id = candidate["surface_id"]
        if per_surface.get(surface_id, 0) >= args.max_per_surface:
            continue
        if not separated(
            candidate,
            selected,
            minimum_center_distance=float(candidate["normalized_side_pixels"]) * 0.5,
        ):
            continue
        selected.append(candidate)
        per_surface[surface_id] = per_surface.get(surface_id, 0) + 1
        if len(selected) == args.top_n:
            break
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank

    previews: list[dict[str, str]] = []
    for source in source_records:
        surface_id = source["surface_id"]
        rows = [row for row in selected if row["surface_id"] == surface_id]
        if not rows:
            continue
        tiffs, _ = ordered_tiff_files(root / surface_id / "tiffs")
        preview_path = output / f"{surface_id}-ranked-windows.jpg"
        draw_preview(tiffs[len(tiffs) // 2], rows, preview_path)
        previews.append(
            {
                "surface_id": surface_id,
                "path": str(preview_path),
                "sha256": sha256_file(preview_path),
            }
        )

    receipt = {
        "kind": "campaign_x_phase4_coarse_ink_window_ranking_v1",
        "generated_at_utc": utc_now(),
        "status": "COMPLETED_PRIORITIZATION_ONLY",
        "sample_id": args.sample_id,
        "physical_window": {
            "requested_size_mm": args.size_mm,
            "source_pixel_um": args.source_pixel_um,
            "source_side_pixels": source_side,
            "source_area_cm2": (
                source_side * args.source_pixel_um / 10000.0
            )
            ** 2,
            "normalized_pixel_um": args.normalized_pixel_um,
            "normalized_side_pixels": normalized_side,
        },
        "search": {
            "screening_name": args.screening_name,
            "downsample": args.downsample,
            "step_fraction": args.step_fraction,
            "minimum_valid_ratio": args.minimum_valid_ratio,
            "top_n": args.top_n,
            "max_per_surface": args.max_per_surface,
            "eligible_window_count": len(candidates),
        },
        "sources": source_records,
        "ranked_windows": selected,
        "previews": previews,
        "policy": [
            "coarse one-pass model output only prioritizes compute",
            "each selected source crop is at or below 4 cm2",
            "a smaller fitted square is used only when explicitly enabled",
            "selected windows require the frozen six-replica screen",
            "no coarse window is accepted as ink or letters",
        ],
    }
    write_json(output / "COARSE_INK_WINDOW_RANKING.json", receipt)
    print(
        json.dumps(
            {
                "eligible_windows": len(candidates),
                "selected_windows": len(selected),
                "best": selected[0] if selected else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
