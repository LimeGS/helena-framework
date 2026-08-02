#!/usr/bin/env python3
"""Extract a deterministic <=4 cm² connected safe ROI from a Phase 4 map."""

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
from scipy import ndimage

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from flattening_gates import (  # noqa: E402
    MAXIMUM_ROI_WINDING_STEP,
    label_winding_aware_components,
    winding_step_report,
)


CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


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
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def largest_component(
    mask: np.ndarray,
    winding: np.ndarray | None = None,
    *,
    maximum_winding_step: int = MAXIMUM_ROI_WINDING_STEP,
) -> tuple[np.ndarray, int, int]:
    """Largest safe component, split wherever the winding index jumps.

    Four-neighbour labelling alone lets one component span a wrap of the
    spiral: the two quads are adjacent in the flattened map and on different
    physical sheets in the scroll.  ``target_winding_index`` is already in the
    NPZ; it decides where the component is cut.
    """

    labels, count = label_winding_aware_components(
        mask, winding, maximum_winding_step=maximum_winding_step
    )
    if count == 0:
        raise RuntimeError("safe mask has no connected component")
    sizes = np.bincount(labels.ravel())
    component_id = int(np.argmax(sizes[1:]) + 1)
    plain_count = int(ndimage.label(mask, structure=CONNECTIVITY)[1])
    return labels == component_id, count, plain_count


def trim_connected(component: np.ndarray, target_cells: int) -> np.ndarray:
    """Remove distant boundary cells while preserving four-neighbour connectivity."""

    result = component.copy()
    remove_count = int(result.sum()) - target_cells
    if remove_count < 0:
        raise RuntimeError("largest safe component is smaller than requested target")
    if remove_count == 0:
        return result
    coordinates = np.argwhere(result)
    centroid = coordinates.mean(axis=0)
    for _ in range(remove_count):
        boundary = result & ~ndimage.binary_erosion(result, structure=CONNECTIVITY)
        candidates = np.argwhere(boundary)
        candidates = sorted(
            (tuple(int(value) for value in cell) for cell in candidates),
            key=lambda cell: (
                -float((cell[0] - centroid[0]) ** 2 + (cell[1] - centroid[1]) ** 2),
                cell,
            ),
        )
        removed = False
        for cell in candidates:
            result[cell] = False
            _, count = ndimage.label(result, structure=CONNECTIVITY)
            if count == 1:
                removed = True
                break
            result[cell] = True
        if not removed:
            raise RuntimeError("cannot trim component without disconnecting it")
    return result


def vertex_mask(quad_mask: np.ndarray) -> np.ndarray:
    vertices = np.zeros((quad_mask.shape[0] + 1, quad_mask.shape[1] + 1), dtype=bool)
    ii, jj = np.where(quad_mask)
    for di in (0, 1):
        for dj in (0, 1):
            vertices[ii + di, jj + dj] = True
    return vertices


def render_mask(path: Path, roi: np.ndarray, safe: np.ndarray, guard: np.ndarray) -> None:
    rgb = np.zeros((*roi.shape, 3), dtype=np.uint8)
    rgb[safe] = [25, 85, 60]
    rgb[guard] = [145, 40, 145]
    rgb[roi] = [40, 215, 120]
    image = Image.fromarray(rgb).resize(
        (roi.shape[1] * 8, roi.shape[0] * 8),
        Image.Resampling.NEAREST,
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="#e6b83f", width=3)
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--fit-subdir", default="fit_rescue_v1")
    parser.add_argument("--quality-subdir", default="quality_map")
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--target-area-cm2", type=float, default=4.0)
    parser.add_argument(
        "--maximum-winding-step",
        type=int,
        default=MAXIMUM_ROI_WINDING_STEP,
        help=(
            "split connected components between adjacent quads whose "
            "target_winding_index differs by at least this many turns"
        ),
    )
    parser.add_argument(
        "--allow-unmeasured-winding",
        action="store_true",
        help=(
            "permit a quality map with no target_winding_index; the receipt "
            "then records the sheet-jump gate as UNMEASURED"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    target = root / "phase4" / "targets" / args.sample_id
    fit = target / args.fit_subdir
    quality_root = fit / args.quality_subdir
    receipt_path = quality_root / "QUALITY_MAP_RECEIPT.json"
    receipt = read_json(receipt_path)
    primary = next(
        row for row in receipt["patches"] if row["patch_id"] == "surface-00-primary"
    )
    quality_npz_path = quality_root / primary["npz_path"]
    quality = np.load(quality_npz_path)
    safe = quality["safe_satisfied"].astype(bool)
    guard = quality["guard"].astype(bool)
    winding = (
        quality["target_winding_index"]
        if "target_winding_index" in quality.files
        else None
    )
    if winding is None and not args.allow_unmeasured_winding:
        raise RuntimeError(
            "quality map carries no target_winding_index; the ROI cannot be "
            "shown to stay on one sheet. Rebuild the quality map or pass "
            "--allow-unmeasured-winding to record the ROI as UNMEASURED."
        )
    component, winding_component_count, plain_component_count = largest_component(
        safe, winding, maximum_winding_step=args.maximum_winding_step
    )
    cell_area_cm2 = float(primary["cell_area_cm2"])
    component_area_cm2 = float(component.sum() * cell_area_cm2)
    if component_area_cm2 < args.target_area_cm2:
        raise RuntimeError(
            f"largest safe component is {component_area_cm2:.9f} cm2, below "
            f"{args.target_area_cm2:.9f} cm2"
            + (
                " after splitting components at winding steps"
                if winding is not None
                else ""
            )
        )
    target_cells = math.floor(args.target_area_cm2 / cell_area_cm2)
    roi = trim_connected(component, target_cells)
    labels, count = ndimage.label(roi, structure=CONNECTIVITY)
    if count != 1 or int(roi.sum()) != target_cells:
        raise RuntimeError("trimmed ROI failed connectivity or area accounting")
    if bool((roi & ~safe).any()) or bool((roi & guard).any()):
        raise RuntimeError("trimmed ROI contains unsafe or guarded cells")
    # Trimming only removes cells, but it removes them by distance from the
    # centroid, so re-assert on the trimmed result rather than inheriting the
    # component's guarantee.
    _, winding_aware_count = label_winding_aware_components(
        roi, winding, maximum_winding_step=args.maximum_winding_step
    )
    if winding_aware_count != 1:
        raise RuntimeError(
            f"trimmed ROI splits into {winding_aware_count} pieces once "
            "adjacent quads with a winding step are separated; it crosses a "
            "sheet jump"
        )
    winding_gate = winding_step_report(
        roi, winding, maximum_winding_step=args.maximum_winding_step
    )
    if winding_gate["status"] == "FAIL":
        raise RuntimeError(
            f"ROI contains {winding_gate['adjacent_pair_crossing_count']} "
            "adjacent quad pairs across a winding discontinuity"
        )

    lock = read_json(target / "TARGET_LOCK.json")
    source_surface = (
        target
        / "candidate_surfaces"
        / str(lock["best_candidate"])
        / "expanded"
    )
    output = fit / "connected_roi"
    segmentation = output / "segmentation"
    segmentation.mkdir(parents=True, exist_ok=True)
    vertices = vertex_mask(roi)
    source_hashes: dict[str, str] = {}
    for name in ("x.tif", "y.tif", "z.tif"):
        source = source_surface / name
        source_hashes[name] = sha256_file(source)
        array = tifffile.imread(source).astype(np.float32)
        masked = array.copy()
        masked[~vertices] = -1.0
        tifffile.imwrite(segmentation / name, masked)
    generations_source = source_surface / "generations.tif"
    source_hashes["generations.tif"] = sha256_file(generations_source)
    generations = tifffile.imread(generations_source)
    generations_masked = generations.copy()
    generations_masked[~vertices] = 0
    tifffile.imwrite(segmentation / "generations.tif", generations_masked)
    meta = read_json(source_surface / "meta.json")
    source_hashes["meta.json"] = sha256_file(source_surface / "meta.json")
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
            "uuid": "connected-roi",
            "source": "campaign_x_phase4_connected_safe_roi_v1",
            "source_surface": str(source_surface),
            "area_cm2": float(roi.sum() * cell_area_cm2),
            "bbox": [
                [float(values[:, axis].min()) for axis in range(3)],
                [float(values[:, axis].max()) for axis in range(3)],
            ],
            "roi_quad_count": int(roi.sum()),
            "roi_mask": "../roi_quad_mask.tif",
            "candidate_arm": args.candidate_arm,
            "winding_continuity": winding_gate,
        }
    )
    write_json(segmentation / "meta.json", meta)
    tifffile.imwrite(output / "roi_quad_mask.tif", roi.astype(np.uint8))
    tifffile.imwrite(output / "roi_vertex_mask.tif", vertices.astype(np.uint8))
    preview_path = output / "connected_roi_preview.png"
    render_mask(preview_path, roi, safe, guard)

    files = [
        output / "roi_quad_mask.tif",
        output / "roi_vertex_mask.tif",
        preview_path,
        *[segmentation / name for name in ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json")],
    ]
    roi_receipt = {
        "kind": "campaign_x_phase4_connected_safe_roi_v1",
        "generated_at_utc": utc_now(),
        "status": "PASSED_CONNECTED_ROI_LOCAL_FUNCTIONAL",
        "gate": "P4_6B_CONNECTED_COMPONENT_ROI",
        "sample_id": args.sample_id,
        "candidate_arm": args.candidate_arm,
        "source_quality_receipt": str(receipt_path),
        "source_quality_receipt_sha256": sha256_file(receipt_path),
        "source_quality_npz_sha256": sha256_file(quality_npz_path),
        "source_surface": str(source_surface),
        "source_surface_hashes": source_hashes,
        "largest_safe_component": {
            "quad_count": int(component.sum()),
            "area_cm2": component_area_cm2,
            "labelling": "FOUR_NEIGHBOUR_CUT_AT_WINDING_STEP_V1",
            "component_count_four_neighbour_only": plain_component_count,
            "component_count_after_winding_split": winding_component_count,
            "components_added_by_winding_split": int(
                winding_component_count - plain_component_count
            ),
        },
        "winding_continuity_gate": winding_gate,
        "selected_roi": {
            "quad_count": int(roi.sum()),
            "cell_area_cm2": cell_area_cm2,
            "area_cm2": float(roi.sum() * cell_area_cm2),
            "target_max_area_cm2": args.target_area_cm2,
            "connected_four_neighbour": True,
            "guard_quad_count": int((roi & guard).sum()),
            "unsafe_quad_count": int((roi & ~safe).sum()),
            "bbox_quad_ij": [
                int(np.where(roi)[0].min()),
                int(np.where(roi)[1].min()),
                int(np.where(roi)[0].max() + 1),
                int(np.where(roi)[1].max() + 1),
            ],
        },
        "artifacts": {
            str(path.relative_to(output)): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        },
        "scope": "PRIVATE_LOCAL_FUNCTIONAL_GEOMETRY_ONLY",
        "ink_used": False,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
    }
    write_json(output / "CONNECTED_ROI_RECEIPT.json", roi_receipt)
    print(json.dumps(roi_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
