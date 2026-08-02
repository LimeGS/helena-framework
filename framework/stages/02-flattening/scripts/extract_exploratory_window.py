#!/usr/bin/env python3
"""Export the best exact-window location as an explicitly exploratory TIFXYZ.

Unlike ``extract_connected_roi.py``, this tool does not require every
quad to pass the geometry gate.  It exists to screen promising ink locations
quickly while preserving the measured valid/safe/guard fractions and refusing
to promote the result as a geometry-qualified ROI.
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
import tifffile
from PIL import Image, ImageDraw

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from flattening_gates import (  # noqa: E402
    MINIMUM_WINDOW_VALID_FRACTION,
    SENTINEL_AUDIT_RADIUS_QUADS,
    evaluate_window_validity,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rectangular_window_mask(
    shape: tuple[int, int],
    top_left: tuple[int, int],
    side_cells: float,
) -> tuple[np.ndarray, int]:
    if side_cells <= 0:
        raise ValueError("side_cells must be positive")
    side = int(math.ceil(side_cells))
    row, column = top_left
    if row < 0 or column < 0 or row + side > shape[0] or column + side > shape[1]:
        raise ValueError("window falls outside the quality map")
    result = np.zeros(shape, dtype=bool)
    result[row : row + side, column : column + side] = True
    return result, side


def vertex_mask(quad_mask: np.ndarray) -> np.ndarray:
    vertices = np.zeros((quad_mask.shape[0] + 1, quad_mask.shape[1] + 1), dtype=bool)
    ii, jj = np.where(quad_mask)
    for di in (0, 1):
        for dj in (0, 1):
            vertices[ii + di, jj + dj] = True
    return vertices


def measured_fractions(
    roi: np.ndarray,
    *,
    valid: np.ndarray,
    safe: np.ndarray,
    guard: np.ndarray,
    satisfied: np.ndarray,
) -> dict[str, float | int]:
    count = int(roi.sum())
    if count == 0:
        raise ValueError("ROI is empty")
    return {
        "quad_count": count,
        "valid_fraction": float((roi & valid).sum() / count),
        "safe_satisfied_fraction": float((roi & safe).sum() / count),
        "guard_fraction": float((roi & guard).sum() / count),
        "satisfied_fraction": float((roi & satisfied).sum() / count),
    }


def select_primary_window(receipt: dict) -> tuple[dict, str]:
    primary = next(
        (
            row
            for row in receipt.get("patches", [])
            if row.get("patch_id") == "surface-00-primary"
        ),
        None,
    )
    if not isinstance(primary, dict):
        raise RuntimeError("quality receipt has no primary surface patch")

    global_best = receipt.get("exact_window", {}).get("best")
    if isinstance(global_best, dict) and global_best.get("patch_id") == "surface-00-primary":
        return global_best, "GLOBAL_BEST_IS_PRIMARY"

    primary_best = primary.get("best_4cm2_window")
    if not isinstance(primary_best, dict):
        raise RuntimeError("quality receipt has no primary exact-window candidate")
    return (
        {**primary_best, "patch_id": "surface-00-primary"},
        "PRIMARY_PATCH_FALLBACK_WHEN_GLOBAL_BEST_IS_NEIGHBOR",
    )


def render_preview(
    path: Path,
    *,
    roi: np.ndarray,
    valid: np.ndarray,
    safe: np.ndarray,
    guard: np.ndarray,
) -> None:
    rgb = np.zeros((*roi.shape, 3), dtype=np.uint8)
    rgb[valid] = [55, 65, 80]
    rgb[safe] = [25, 100, 65]
    rgb[guard] = [145, 40, 145]
    rgb[roi & valid] = np.maximum(rgb[roi & valid], [80, 110, 120])
    rgb[roi & safe] = [40, 215, 120]
    image = Image.fromarray(rgb).resize(
        (roi.shape[1] * 8, roi.shape[0] * 8), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    rows, columns = np.where(roi)
    draw.rectangle(
        (
            int(columns.min() * 8),
            int(rows.min() * 8),
            int((columns.max() + 1) * 8 - 1),
            int((rows.max() + 1) * 8 - 1),
        ),
        outline="#ffcc33",
        width=4,
    )
    image.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--fit-subdir", default="fit_ab")
    parser.add_argument(
        "--quality-subdir", default="baseline_sign_rescue_quality_map"
    )
    parser.add_argument("--candidate-arm", default="baseline")
    parser.add_argument("--output-subdir", default="exploratory_window_v1")
    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=MINIMUM_WINDOW_VALID_FRACTION,
        help=(
            "refuse to export a requested window whose valid quad fraction is "
            "below this value; the invalid remainder becomes the -1.0 TIFXYZ "
            "sentinel and then zero fill in the render"
        ),
    )
    parser.add_argument(
        "--sentinel-audit-radius-quads",
        type=int,
        default=SENTINEL_AUDIT_RADIUS_QUADS,
        help="refuse to export quads this close to an invalid (sentinel) quad",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    target = root / "phase4" / "targets" / args.sample_id
    fit = target / args.fit_subdir
    quality_root = fit / args.quality_subdir
    receipt_path = quality_root / "QUALITY_MAP_RECEIPT.json"
    receipt = read_json(receipt_path)
    if receipt.get("sample_id") != args.sample_id:
        raise RuntimeError("quality receipt sample binding mismatch")
    if receipt.get("candidate_arm") != args.candidate_arm:
        raise RuntimeError("quality receipt arm binding mismatch")
    if bool(receipt.get("ink_used")):
        raise RuntimeError("quality map used ink")
    if receipt.get("status") != "NO_QUALIFIED_ROI":
        raise RuntimeError("exploratory export is only for an explicitly failed ROI gate")

    best, selection_policy = select_primary_window(receipt)
    primary = next(
        row for row in receipt["patches"] if row["patch_id"] == "surface-00-primary"
    )
    quality_path = quality_root / primary["npz_path"]
    quality = np.load(quality_path)
    requested, selected_side_cells = rectangular_window_mask(
        quality["valid"].shape,
        tuple(int(value) for value in best["top_left_quad_ij"]),
        float(best["side_cells"]),
    )
    valid_mask = quality["valid"].astype(bool)
    requested_fractions = measured_fractions(
        requested,
        valid=valid_mask,
        safe=quality["safe_satisfied"].astype(bool),
        guard=quality["guard"].astype(bool),
        satisfied=quality["satisfied"].astype(bool),
    )
    # The requested rectangle is a pure rectangle: it was never intersected
    # with the validity map, so every quad outside the surface was exported as
    # the -1.0 sentinel and became zero fill downstream.  Gate first, then
    # export only the intersection.
    roi, validity_gate = evaluate_window_validity(
        requested,
        valid_mask,
        minimum_valid_fraction=float(args.minimum_valid_fraction),
        sentinel_audit_radius_quads=int(args.sentinel_audit_radius_quads),
    )
    fractions = measured_fractions(
        roi,
        valid=valid_mask,
        safe=quality["safe_satisfied"].astype(bool),
        guard=quality["guard"].astype(bool),
        satisfied=quality["satisfied"].astype(bool),
    )
    cell_area_cm2 = float(primary["cell_area_cm2"])
    selected_area_cm2 = float(roi.sum() * cell_area_cm2)

    lock = read_json(target / "TARGET_LOCK.json")
    source_surface = (
        target / "candidate_surfaces" / str(lock["best_candidate"]) / "expanded"
    )
    output = fit / args.output_subdir
    segmentation = output / "segmentation"
    segmentation.mkdir(parents=True, exist_ok=True)
    vertices = vertex_mask(roi)
    source_hashes: dict[str, str] = {}
    for name in ("x.tif", "y.tif", "z.tif"):
        source = source_surface / name
        source_hashes[name] = sha256_file(source)
        array = tifffile.imread(source).astype(np.float32)
        if array.shape != vertices.shape:
            raise RuntimeError(f"{name} shape does not match quality-map vertices")
        masked = array.copy()
        masked[~vertices] = -1.0
        tifffile.imwrite(segmentation / name, masked)
    generations_source = source_surface / "generations.tif"
    source_hashes["generations.tif"] = sha256_file(generations_source)
    generations = tifffile.imread(generations_source)
    generations_masked = generations.copy()
    generations_masked[~vertices] = 0
    tifffile.imwrite(segmentation / "generations.tif", generations_masked)

    meta_source = source_surface / "meta.json"
    source_hashes["meta.json"] = sha256_file(meta_source)
    meta = read_json(meta_source)
    xyz = np.stack(
        [
            tifffile.imread(segmentation / f"{axis}.tif")
            for axis in ("x", "y", "z")
        ],
        axis=-1,
    )
    values = xyz[vertices]
    meta.update(
        {
            "uuid": "exploratory-window",
            "source": "campaign_x_phase4_exploratory_exact_window_v1",
            "source_surface": source_surface.as_posix(),
            "bbox": [
                [float(values[:, axis].min()) for axis in range(3)],
                [float(values[:, axis].max()) for axis in range(3)],
            ],
            "requested_window_side_mm": float(best["side_mm"]),
            "requested_window_area_cm2": float(best["area_cm2"]),
            "mesh_window_area_cm2": selected_area_cm2,
            "mesh_window_side_cells": selected_side_cells,
            "crop_render_to_exact_20mm_required": True,
            "candidate_arm": args.candidate_arm,
            "geometry_qualified": False,
            # Carried so the render crop can re-assert the validity gate
            # instead of cropping a geometric centre it cannot interpret.
            "window_validity": validity_gate,
            "roi_quad_mask": "../roi_quad_mask.tif",
            "roi_vertex_mask": "../roi_vertex_mask.tif",
        }
    )
    write_json(segmentation / "meta.json", meta)
    tifffile.imwrite(output / "roi_quad_mask.tif", roi.astype(np.uint8))
    tifffile.imwrite(output / "roi_vertex_mask.tif", vertices.astype(np.uint8))
    preview_path = output / "exploratory_window_preview.png"
    render_preview(
        preview_path,
        roi=roi,
        valid=quality["valid"].astype(bool),
        safe=quality["safe_satisfied"].astype(bool),
        guard=quality["guard"].astype(bool),
    )

    artifacts = [
        output / "roi_quad_mask.tif",
        output / "roi_vertex_mask.tif",
        preview_path,
        *[
            segmentation / name
            for name in ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json")
        ],
    ]
    result = {
        "kind": "campaign_x_phase4_exploratory_exact_window_v1",
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_PRIVATE_INK_SCREENING_NOT_GEOMETRY_QUALIFIED",
        "gate": "P4_11_EXPLORATORY_INK_ROI_PREPARATION",
        "sample_id": args.sample_id,
        "candidate_arm": args.candidate_arm,
        "source_quality_receipt": receipt_path.as_posix(),
        "source_quality_receipt_sha256": sha256_file(receipt_path),
        "source_quality_npz_sha256": sha256_file(quality_path),
        "source_surface": source_surface.as_posix(),
        "source_surface_hashes": source_hashes,
        "requested_window": {
            "selection_policy": selection_policy,
            "side_mm": float(best["side_mm"]),
            "area_cm2": float(best["area_cm2"]),
            "top_left_quad_ij": best["top_left_quad_ij"],
            "fractional_side_cells": float(best["side_cells"]),
        },
        "requested_rectangle": {
            "side_cells": selected_side_cells,
            **requested_fractions,
        },
        "window_validity_gate": validity_gate,
        "exported_mesh_window": {
            "side_cells": selected_side_cells,
            "area_cm2": selected_area_cm2,
            **fractions,
            "mask_intersected_with_valid": True,
            "crop_render_to_exact_20mm_required": True,
        },
        "geometry_qualified": False,
        "first_letters_claim_allowed": False,
        "scope": "PRIVATE_EXPLORATORY_INK_SCREENING_ONLY",
        "ink_used_for_selection": False,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
        "artifacts": {
            path.relative_to(output).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        },
    }
    write_json(output / "EXPLORATORY_WINDOW_RECEIPT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
