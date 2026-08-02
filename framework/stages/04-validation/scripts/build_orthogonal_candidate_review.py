#!/usr/bin/env python3
"""Build raw-CT XY/XZ/YZ evidence for every positive screen component."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import (  # noqa: E402
    NUMERIC_STEM_INDEX,
    ordered_tiff_files,
    ordered_tiff_stack_position,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_stack(directory: Path) -> tuple[np.ndarray, list[Path], str]:
    files, ordering = ordered_tiff_files(directory)
    arrays = [np.asarray(Image.open(path), dtype=np.uint8) for path in files]
    if len({array.shape for array in arrays}) != 1:
        raise RuntimeError("TIFF slices do not share one shape")
    return np.stack(arrays), files, ordering


def crop_with_padding(
    value: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    half_size: int,
) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError("crop source must be two-dimensional")
    size = half_size * 2
    output = np.zeros((size, size), dtype=value.dtype)
    source_y0 = max(0, center_y - half_size)
    source_y1 = min(value.shape[0], center_y + half_size)
    source_x0 = max(0, center_x - half_size)
    source_x1 = min(value.shape[1], center_x + half_size)
    target_y0 = source_y0 - (center_y - half_size)
    target_x0 = source_x0 - (center_x - half_size)
    output[
        target_y0 : target_y0 + source_y1 - source_y0,
        target_x0 : target_x0 + source_x1 - source_x0,
    ] = value[source_y0:source_y1, source_x0:source_x1]
    return output


def map_analysis_point_to_source(
    *,
    analysis_y: int,
    analysis_x: int,
    analysis_shape_y_x: tuple[int, int],
    source_shape_y_x: tuple[int, int],
) -> tuple[int, int]:
    """Map a point from physically normalized model space into source CT space."""
    analysis_height, analysis_width = analysis_shape_y_x
    source_height, source_width = source_shape_y_x
    if min(analysis_height, analysis_width, source_height, source_width) < 1:
        raise ValueError("analysis and source shapes must be positive")
    source_y = round(analysis_y * source_height / analysis_height)
    source_x = round(analysis_x * source_width / analysis_width)
    return (
        min(max(source_y, 0), source_height - 1),
        min(max(source_x, 0), source_width - 1),
    )


def map_analysis_half_size_to_source(
    *,
    analysis_half_size: int,
    analysis_shape_y_x: tuple[int, int],
    source_shape_y_x: tuple[int, int],
) -> int:
    """Preserve the requested physical field of view after model resampling."""
    if analysis_half_size < 1:
        raise ValueError("analysis half-size must be positive")
    analysis_height, analysis_width = analysis_shape_y_x
    source_height, source_width = source_shape_y_x
    scale_y = source_height / analysis_height
    scale_x = source_width / analysis_width
    if not np.isclose(scale_y, scale_x, rtol=0.01):
        raise ValueError("anisotropic XY mapping is not supported")
    return max(1, round(analysis_half_size * (scale_y + scale_x) / 2))


def normalize_ct(value: np.ndarray, upper: float) -> Image.Image:
    scaled = np.rint(np.clip(value.astype(np.float32), 0, upper) / upper * 255)
    return Image.fromarray(scaled.astype(np.uint8), mode="L")


def orthogonal_views(
    stack: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    central_position: int,
    half_size: int,
    average_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ``central_position`` is an offset along axis 0 of the ordered stack, not
    # a physical slice number.  The caller resolves the physical slice through
    # ``ordered_tiff_stack_position`` so padding and lexicographic accidents
    # cannot silently shift the rendered depth.
    if not 0 <= central_position < stack.shape[0]:
        raise ValueError("central slice outside stack")
    xy = crop_with_padding(
        stack[central_position],
        center_y=center_y,
        center_x=center_x,
        half_size=half_size,
    )
    y0 = max(0, center_y - average_width)
    y1 = min(stack.shape[1], center_y + average_width + 1)
    x0 = max(0, center_x - average_width)
    x1 = min(stack.shape[2], center_x + average_width + 1)
    x_span0 = max(0, center_x - half_size)
    x_span1 = min(stack.shape[2], center_x + half_size)
    y_span0 = max(0, center_y - half_size)
    y_span1 = min(stack.shape[1], center_y + half_size)
    xz_raw = stack[:, y0:y1, x_span0:x_span1].mean(axis=1)
    yz_raw = stack[:, y_span0:y_span1, x0:x1].mean(axis=2)
    xz = np.zeros((stack.shape[0], half_size * 2), dtype=np.float32)
    yz = np.zeros((stack.shape[0], half_size * 2), dtype=np.float32)
    x_target = max(0, x_span0 - (center_x - half_size))
    y_target = max(0, y_span0 - (center_y - half_size))
    xz[:, x_target : x_target + xz_raw.shape[1]] = xz_raw
    yz[:, y_target : y_target + yz_raw.shape[1]] = yz_raw
    return xy, xz, yz


def labeled_panel(
    xy: np.ndarray,
    xz: np.ndarray,
    yz: np.ndarray,
    *,
    candidate_id: str,
    analysis_center_y: int,
    analysis_center_x: int,
    source_center_y: int,
    source_center_x: int,
    upper: float,
    depth_scale: int,
) -> Image.Image:
    size = xy.shape[0]
    view_height = max(size, xz.shape[0] * depth_scale)
    panel = Image.new("RGB", (size * 3, view_height + 28), "#08101c")
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 7),
        (
            f"{candidate_id} · modelo y={analysis_center_y}, x={analysis_center_x}"
            f" → CT y={source_center_y}, x={source_center_x}"
        ),
        fill="#eef6ff",
    )
    for index, (label, array) in enumerate((("XY", xy), ("XZ", xz), ("YZ", yz))):
        image = normalize_ct(array, upper).resize(
            (size, view_height),
            Image.Resampling.NEAREST if label != "XY" else Image.Resampling.BILINEAR,
        )
        panel.paste(Image.merge("RGB", (image, image, image)), (index * size, 28))
        draw.rectangle(
            (index * size, 28, (index + 1) * size - 1, view_height + 27),
            outline="#36516f",
            width=1,
        )
        draw.text((index * size + 5, 33), label, fill="#ffd66b")
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--central-slice", type=int, default=32)
    parser.add_argument("--half-size", type=int, default=128)
    parser.add_argument("--average-width", type=int, default=2)
    parser.add_argument("--depth-scale", type=int, default=4)
    parser.add_argument("--display-upper", type=float, default=200)
    parser.add_argument("--voxel-um", type=float, required=True)
    args = parser.parse_args()

    if min(args.half_size, args.average_width, args.depth_scale) < 1:
        raise ValueError("display parameters must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stack, source_files, slice_ordering = load_stack(args.tiff_dir.resolve())
    central_position = ordered_tiff_stack_position(
        source_files,
        int(args.central_slice),
    )
    central_file = source_files[central_position]
    analysis = read_json(args.analysis.resolve())
    analysis_shape_y_x = tuple(map(int, analysis["input"]["shape_y_x"]))
    if len(analysis_shape_y_x) != 2:
        raise RuntimeError("analysis input shape_y_x must have two dimensions")
    source_shape_y_x = (int(stack.shape[1]), int(stack.shape[2]))
    source_half_size = map_analysis_half_size_to_source(
        analysis_half_size=args.half_size,
        analysis_shape_y_x=analysis_shape_y_x,
        source_shape_y_x=source_shape_y_x,
    )
    screening = analysis["text_like_screening"]
    candidates = screening["candidates"]
    if not candidates:
        raise RuntimeError("analysis has no positive components")

    artifacts: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    for candidate in candidates:
        x0, y0, x1, y1 = map(int, candidate["bbox_xyxy"])
        analysis_center_y = (y0 + y1) // 2
        analysis_center_x = (x0 + x1) // 2
        source_center_y, source_center_x = map_analysis_point_to_source(
            analysis_y=analysis_center_y,
            analysis_x=analysis_center_x,
            analysis_shape_y_x=analysis_shape_y_x,
            source_shape_y_x=source_shape_y_x,
        )
        xy, xz, yz = orthogonal_views(
            stack,
            center_y=source_center_y,
            center_x=source_center_x,
            central_position=central_position,
            half_size=source_half_size,
            average_width=args.average_width,
        )
        panel = labeled_panel(
            xy,
            xz,
            yz,
            candidate_id=str(candidate["candidate_id"]),
            analysis_center_y=analysis_center_y,
            analysis_center_x=analysis_center_x,
            source_center_y=source_center_y,
            source_center_x=source_center_x,
            upper=args.display_upper,
            depth_scale=args.depth_scale,
        )
        path = output / f"{candidate['candidate_id']}-orthogonal-ct.png"
        panel.save(path)
        panels.append(panel)
        artifacts.append(
            {
                "candidate_id": candidate["candidate_id"],
                "bbox_xyxy": candidate["bbox_xyxy"],
                "analysis_center_y_x": [
                    analysis_center_y,
                    analysis_center_x,
                ],
                "source_ct_center_y_x": [source_center_y, source_center_x],
                "png": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    contact = Image.new(
        "RGB",
        (panels[0].width * columns, panels[0].height * rows),
        "#050a12",
    )
    for index, panel in enumerate(panels):
        contact.paste(
            panel,
            (
                (index % columns) * panel.width,
                (index // columns) * panel.height,
            ),
        )
    contact_path = output / "ORTHOGONAL_CT_CONTACT_SHEET.png"
    contact.save(contact_path)
    receipt = {
        "kind": "campaign_x_phase4_positive_component_orthogonal_ct_review_v1",
        "status": "ORTHOGONAL_EVIDENCE_READY",
        "generated_at_utc": utc_now(),
        "sample_id": analysis["sample_id"],
        "source": {
            "tiff_directory": str(args.tiff_dir.resolve()),
            "slice_count": len(source_files),
            "slice_ordering": slice_ordering,
            "central_slice_resolution": NUMERIC_STEM_INDEX,
            "shape_depth_y_x": list(stack.shape),
            "first_slice_sha256": sha256_file(source_files[0]),
            "central_slice_name": central_file.name,
            "central_slice_stack_position": central_position,
            "central_slice_sha256": sha256_file(central_file),
            "last_slice_sha256": sha256_file(source_files[-1]),
            "analysis": str(args.analysis.resolve()),
            "analysis_sha256": sha256_file(args.analysis.resolve()),
        },
        "physical": {
            "voxel_um": args.voxel_um,
            "xy_crop_width_um": source_half_size * 2 * args.voxel_um,
            "depth_span_um": len(source_files) * args.voxel_um,
        },
        "display": {
            "central_slice": args.central_slice,
            "analysis_half_size": args.half_size,
            "source_ct_half_size": source_half_size,
            "analysis_shape_y_x": list(analysis_shape_y_x),
            "source_ct_shape_y_x": list(source_shape_y_x),
            "average_width_each_side": args.average_width,
            "depth_display_scale": args.depth_scale,
            "fixed_intensity_range": [0, args.display_upper],
            "warning": "XZ and YZ depth axes are enlarged for visibility",
        },
        "candidate_count": len(artifacts),
        "candidates": artifacts,
        "contact_sheet": {
            "path": contact_path.name,
            "sha256": sha256_file(contact_path),
            "size_bytes": contact_path.stat().st_size,
        },
        "interpretation": [
            "a broad structure continuous through depth supports a fiber or laminar confound",
            "a thin response localized near one surface depth is more compatible with surface ink",
            "orthogonal appearance is supporting evidence, not automatic ink acceptance",
        ],
    }
    write_json(output / "ORTHOGONAL_CT_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
