#!/usr/bin/env python3
"""Prepare label-blind CT evidence for MULTISCROLL_TRANSFER_V1.

The official ink maps are used only as location proposals.  This program does
not assign POSITIVE/CONFOUND labels and never executes the v3/v4 gates.  For
each frozen surface it:

* selects one 2.4 um prediction/render pair deterministically;
* downloads the downsampled locator, registered surface-volume metadata and
  transformed TIFXYZ coordinates;
* samples high- and mid-response locations without interpreting them;
* downloads only the registered Zarr chunks touched by those locations;
* writes compact CT/orthogonal review images and an auditable proposal file.

The registered surface volumes currently use uncompressed Zarr v2 chunks.  A
different codec or dimension layout fails closed instead of being guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import requests
import tifffile
import zarr
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from scipy import ndimage


PUBLIC_ROOT = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
PREFERRED_RESOLUTIONS = ("2.399um", "2.4um")
ZARR_LEVEL = 3
ZARR_CHUNK_EDGE = 128
ZARR_DEPTH = 109
LOCATOR_MAX_DIMENSION = 6000
Image.MAX_IMAGE_PIXELS = 400_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def public_url(path: str) -> str:
    return f"{PUBLIC_ROOT}/{quote(path, safe='/._-')}"


def fetch(
    session: requests.Session,
    url: str,
    output: Path,
    *,
    expected_bytes: int | None = None,
    retries: int = 5,
) -> Path:
    if output.exists() and (expected_bytes is None or output.stat().st_size == expected_bytes):
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            stream.write(block)
            if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
                raise RuntimeError(
                    f"size mismatch for {url}: {temporary.stat().st_size} != "
                    f"{expected_bytes}"
                )
            temporary.replace(output)
            return output
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def origins_for_type(segment: dict[str, Any], data_type: str) -> list[str]:
    paths: list[str] = []
    for item in segment.get("data", []):
        if item.get("type") != data_type:
            continue
        paths.extend(
            str(origin["path"])
            for origin in item.get("origins", [])
            if origin.get("path")
        )
    return sorted(set(paths))


def resolution_token(path: str) -> str | None:
    for token in PREFERRED_RESOLUTIONS:
        if token in path:
            return token
    return None


def choose_asset(paths: list[str], *, token: str | None = None) -> str:
    eligible = [
        path
        for path in paths
        if (token is None or token in path)
        and any(preferred in path for preferred in PREFERRED_RESOLUTIONS)
    ]
    if not eligible:
        raise RuntimeError(f"no eligible 2.4 um asset among {paths}")
    return sorted(eligible)[0]


def volume_id_from_path(path: str) -> str:
    match = re.search(r"volume-(\d+)", path)
    if not match:
        match = re.search(r"-on-(\d+)-", path)
    if not match:
        raise RuntimeError(f"cannot infer volume id from {path}")
    return match.group(1)


def transformed_tifxyz(segment: dict[str, Any], token: str, volume_id: str) -> str:
    paths = origins_for_type(segment, "tifxyz-transformed")
    eligible = [
        path
        for path in paths
        if token in path and f"-on-{volume_id}-" in path
    ]
    if not eligible:
        raise RuntimeError(
            f"no transformed TIFXYZ for volume={volume_id} token={token}"
        )
    return sorted(eligible)[0]


@dataclass(frozen=True)
class SurfaceAssets:
    preview_path: str
    zarr_path: str
    tifxyz_path: str
    volume_id: str
    voxel_size_um: float
    scanner_domain: str
    z_direction_is_top_to_bottom: bool
    left_handed_coordinates: bool


def resolve_surface_assets(
    metadata: dict[str, Any], scroll_id: str, segment_id: str
) -> tuple[dict[str, Any], SurfaceAssets]:
    scroll = metadata["samples"][scroll_id]
    segment = scroll["segments"][segment_id]
    previews = origins_for_type(segment, "ink-detection-downsampled")
    preview = choose_asset(previews)
    token = resolution_token(preview)
    if token is None:
        raise RuntimeError(f"preview resolution is unsupported: {preview}")
    zarr = choose_asset(origins_for_type(segment, "layers-zarr"), token=token)
    volume_id = volume_id_from_path(zarr)
    tifxyz = transformed_tifxyz(segment, token, volume_id)
    volume = scroll["volumes"][volume_id]
    properties = volume["properties"]
    voxel = float(properties["pixel_size_um"])
    scanner_domain = (
        f"{scroll_id}:{volume_id}:{voxel:g}um:"
        f"{float(properties.get('energy_keV') or 0):g}keV"
    )
    return segment, SurfaceAssets(
        preview_path=preview,
        zarr_path=zarr,
        tifxyz_path=tifxyz,
        volume_id=volume_id,
        voxel_size_um=voxel,
        scanner_domain=scanner_domain,
        z_direction_is_top_to_bottom=bool(
            properties.get("z_direction_is_top_to_bottom", False)
        ),
        left_handed_coordinates=bool(
            properties.get("left_handed_coordinates", False)
        ),
    )


def greedy_spaced(
    candidates: list[tuple[float, int, int]],
    *,
    limit: int,
    minimum_distance: float,
) -> list[tuple[float, int, int]]:
    selected: list[tuple[float, int, int]] = []
    minimum_squared = minimum_distance * minimum_distance
    for score, y, x in candidates:
        if all((y - sy) ** 2 + (x - sx) ** 2 >= minimum_squared for _, sy, sx in selected):
            selected.append((score, y, x))
            if len(selected) == limit:
                break
    return selected


def sample_proposals(
    image: np.ndarray,
    *,
    task_id: str,
    high_count: int,
    mid_count: int,
    registered_shape_yx: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    if image.ndim != 2:
        raise ValueError("locator preview must be grayscale")
    nonzero = image[image > 0]
    if not len(nonzero):
        raise RuntimeError("locator preview contains no non-zero response")
    registered_height, registered_width = registered_shape_yx or image.shape
    scale_y = registered_height / image.shape[0]
    scale_x = registered_width / image.shape[1]
    margin = max(
        1,
        int(np.ceil((ZARR_CHUNK_EDGE // 2) / min(scale_y, scale_x))),
    )
    valid = np.zeros(image.shape, dtype=bool)
    valid[margin : image.shape[0] - margin, margin : image.shape[1] - margin] = True
    maxima = image == ndimage.maximum_filter(image, size=9, mode="nearest")
    high_threshold = max(float(np.quantile(nonzero, 0.985)), 1.0)
    ys, xs = np.nonzero(maxima & valid & (image >= high_threshold))
    high = sorted(
        ((float(image[y, x]), int(y), int(x)) for y, x in zip(ys, xs)),
        key=lambda row: (-row[0], row[1], row[2]),
    )
    minimum_distance = max(8.0, 36.0 / max(scale_y, scale_x))
    high_selected = greedy_spaced(
        high, limit=high_count, minimum_distance=minimum_distance
    )

    q55, q90 = np.quantile(nonzero, [0.55, 0.90])
    grid = max(12, int(round(48 / max(scale_y, scale_x))))
    mid: list[tuple[float, int, int]] = []
    for y0 in range(margin, image.shape[0] - margin, grid):
        for x0 in range(margin, image.shape[1] - margin, grid):
            block = image[y0 : y0 + grid, x0 : x0 + grid]
            mask = (block >= q55) & (block <= q90)
            if not np.any(mask):
                continue
            by, bx = np.unravel_index(np.argmax(np.where(mask, block, -1)), block.shape)
            y, x = y0 + int(by), x0 + int(bx)
            tie_break = int(
                hashlib.sha256(f"{task_id}:{y}:{x}".encode()).hexdigest()[:12], 16
            )
            mid.append((float(image[y, x]) + tie_break / 10**18, y, x))
    mid.sort(key=lambda row: (-row[0], row[1], row[2]))
    mid_selected = greedy_spaced(
        mid,
        limit=mid_count,
        minimum_distance=minimum_distance,
    )

    proposals: list[dict[str, Any]] = []
    for stratum, rows in (
        ("HIGH_RESPONSE_LOCATOR", high_selected),
        ("MID_RESPONSE_LOCATOR", mid_selected),
    ):
        for score, y, x in rows:
            registered_y = min(
                registered_height - 1,
                max(0, int(round((y + 0.5) * scale_y - 0.5))),
            )
            registered_x = min(
                registered_width - 1,
                max(0, int(round((x + 0.5) * scale_x - 0.5))),
            )
            proposal_id = hashlib.sha256(
                f"{task_id}:{stratum}:{registered_y}:{registered_x}".encode()
            ).hexdigest()[:20]
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "proposal_stratum": stratum,
                    "locator_score_uint8": int(round(min(score, 255))),
                    "locator_preview_y_x": [y, x],
                    "preview_y_x": [registered_y, registered_x],
                    "surface_y_x_level0": [registered_y * 8, registered_x * 8],
                    "zarr_level": ZARR_LEVEL,
                    "zarr_chunk_zyx": [
                        0,
                        registered_y // ZARR_CHUNK_EDGE,
                        registered_x // ZARR_CHUNK_EDGE,
                    ],
                    "chunk_local_y_x": [
                        registered_y % ZARR_CHUNK_EDGE,
                        registered_x % ZARR_CHUNK_EDGE,
                    ],
                    "expected_class": None,
                    "label_authority": None,
                    "adjudication_status": "UNREVIEWED",
                }
            )
    return proposals


def normalize_uint8(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = array[np.isfinite(array)]
    if not len(values):
        return np.zeros(array.shape, dtype=np.uint8)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def resized_gray(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(normalize_uint8(array), mode="L").resize(
        size, Image.Resampling.NEAREST
    )


def build_review_image(
    stack: np.ndarray,
    preview: np.ndarray,
    proposal: dict[str, Any],
    output: Path,
) -> None:
    local_y, local_x = proposal["chunk_local_y_x"]
    patch_radius = 24
    y0, y1 = max(0, local_y - patch_radius), min(stack.shape[1], local_y + patch_radius + 1)
    x0, x1 = max(0, local_x - patch_radius), min(stack.shape[2], local_x + patch_radius + 1)
    patch = stack[:, y0:y1, x0:x1].astype(np.float32)
    gradient = np.abs(np.diff(patch, axis=0))
    profile = gradient.mean(axis=(1, 2))
    peak = int(np.argmax(profile))
    surface_slice = min(stack.shape[0] - 1, peak + 1)

    py, px = proposal["locator_preview_y_x"]
    preview_crop = preview[
        max(0, py - 64) : min(preview.shape[0], py + 65),
        max(0, px - 64) : min(preview.shape[1], px + 65),
    ]
    preview_image = Image.fromarray(preview_crop, mode="L").resize(
        (360, 360), Image.Resampling.NEAREST
    )
    preview_draw = ImageDraw.Draw(preview_image)
    preview_draw.line((180, 150, 180, 210), fill=255, width=2)
    preview_draw.line((150, 180, 210, 180), fill=255, width=2)

    panels = [
        ("locator (not GT)", preview_image),
        (f"CT slice {surface_slice}", resized_gray(patch[surface_slice], (360, 360))),
        ("max depth gradient", resized_gray(gradient.max(axis=0), (360, 360))),
        ("orthogonal XZ", resized_gray(patch[:, :, patch.shape[2] // 2], (360, 360))),
        ("orthogonal YZ", resized_gray(patch[:, patch.shape[1] // 2, :], (360, 360))),
        (
            "depth montage",
            Image.fromarray(
                np.concatenate(
                    [
                        normalize_uint8(patch[index])
                        for index in np.linspace(
                            max(0, surface_slice - 12),
                            min(stack.shape[0] - 1, surface_slice + 12),
                            9,
                            dtype=int,
                        )
                    ],
                    axis=1,
                ),
                mode="L",
            ).resize((360, 360), Image.Resampling.NEAREST),
        ),
    ]
    canvas = Image.new("RGB", (1110, 780), "#0a0f18")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (14, 8),
        f"{proposal['proposal_id']} · {proposal['proposal_stratum']} · "
        f"score {proposal['locator_score_uint8']} · depth peak {surface_slice}",
        fill="#eef4ff",
    )
    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, 3)
        x, y = 10 + column * 370, 36 + row * 370
        canvas.paste(panel.convert("RGB"), (x, y))
        draw.rectangle((x, y, x + 360, y + 360), outline="#33445d")
        draw.text((x + 8, y + 8), label, fill="#ffdf76", stroke_width=2, stroke_fill="#000000")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def load_tifxyz_coordinates(
    tifxyz_dir: Path,
    proposal: dict[str, Any],
    *,
    canvas_size_xy: tuple[int, int],
) -> list[float]:
    surface_y, surface_x = proposal["surface_y_x_level0"]
    canvas_x, canvas_y = canvas_size_xy
    coordinates: list[float] = []
    coordinate_pixel_y_x: list[float] | None = None
    for axis in ("x", "y", "z"):
        path = tifxyz_dir / f"{axis}.tif"
        with tifffile.TiffFile(path) as tiff:
            store = tiff.aszarr()
            try:
                array = zarr.open(store, mode="r")
                if not (
                    0 <= surface_y < canvas_y
                    and 0 <= surface_x < canvas_x
                    and array.ndim == 2
                ):
                    raise RuntimeError(
                        f"surface coordinate {(surface_y, surface_x)} outside "
                        f"canvas {(canvas_y, canvas_x)}"
                    )
                # Public transformed TIFXYZ may be stored on a coarser grid than
                # the registered surface-volume canvas (currently 20x in the
                # first PHerc1667 controls).  Map by declared canvas extent and
                # bilinearly interpolate four coordinate pixels.  Zarr-backed
                # access decodes only the touched TIFF tiles.
                pixel_y = surface_y * (array.shape[0] - 1) / max(canvas_y - 1, 1)
                pixel_x = surface_x * (array.shape[1] - 1) / max(canvas_x - 1, 1)
                y0, x0 = int(np.floor(pixel_y)), int(np.floor(pixel_x))
                y1, x1 = min(y0 + 1, array.shape[0] - 1), min(
                    x0 + 1, array.shape[1] - 1
                )
                dy, dx = pixel_y - y0, pixel_x - x0
                values = np.asarray(array[y0 : y1 + 1, x0 : x1 + 1], dtype=np.float64)
                top = values[0, 0] * (1.0 - dx) + values[0, -1] * dx
                bottom = values[-1, 0] * (1.0 - dx) + values[-1, -1] * dx
                coordinates.append(float(top * (1.0 - dy) + bottom * dy))
                if coordinate_pixel_y_x is None:
                    coordinate_pixel_y_x = [pixel_y, pixel_x]
            finally:
                store.close()
    proposal["tifxyz_coordinate_pixel_y_x"] = coordinate_pixel_y_x
    proposal["tifxyz_canvas_size_xy"] = [canvas_x, canvas_y]
    return coordinates


def run(
    *,
    queue_path: Path,
    metadata_path: Path,
    output_root: Path,
    high_per_surface: int,
    mid_per_surface: int,
) -> dict[str, Any]:
    queue = json.loads(queue_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if sha256(metadata_path) != queue["source_metadata_sha256"]:
        raise RuntimeError("metadata SHA-256 does not match frozen queue")
    if queue.get("benchmark_id") != "MULTISCROLL_TRANSFER_V1":
        raise RuntimeError("wrong benchmark queue")

    session = requests.Session()
    session.headers["User-Agent"] = "CampaignX-MULTISCROLL_TRANSFER_V1/1.0"
    completed_tasks: list[dict[str, Any]] = []
    all_proposals: list[dict[str, Any]] = []

    for task in queue["tasks"]:
        scroll_id, segment_id = task["scroll_id"], task["segment_id"]
        segment, assets = resolve_surface_assets(metadata, scroll_id, segment_id)
        task_root = output_root / "assets" / scroll_id / segment_id
        preview_file = fetch(
            session, public_url(assets.preview_path), task_root / "locator-ds8.jpg"
        )
        with Image.open(preview_file) as source:
            registered_preview_shape = (source.height, source.width)
            locator = source.convert("L")
            if max(locator.size) > LOCATOR_MAX_DIMENSION:
                scale = LOCATOR_MAX_DIMENSION / max(locator.size)
                locator = locator.resize(
                    (
                        max(1, int(round(locator.width * scale))),
                        max(1, int(round(locator.height * scale))),
                    ),
                    Image.Resampling.BOX,
                )
            preview = np.asarray(locator)

        zarray_file = fetch(
            session,
            public_url(f"{assets.zarr_path}{ZARR_LEVEL}/.zarray"),
            task_root / f"zarr-level-{ZARR_LEVEL}.zarray.json",
        )
        zattrs_file = fetch(
            session,
            public_url(f"{assets.zarr_path}.zattrs"),
            task_root / "zarr.zattrs.json",
        )
        zarray = json.loads(zarray_file.read_text())
        zattrs = json.loads(zattrs_file.read_text())
        if zarray.get("compressor") is not None:
            raise RuntimeError("compressed Zarr is unsupported by frozen reader")
        if zarray.get("dtype") != "|u1" or zarray.get("order") != "C":
            raise RuntimeError(f"unsupported Zarr layout: {zarray}")
        if list(zarray["chunks"]) != [ZARR_DEPTH, ZARR_CHUNK_EDGE, ZARR_CHUNK_EDGE]:
            raise RuntimeError(f"unexpected Zarr chunks: {zarray['chunks']}")
        zshape = list(map(int, zarray["shape"]))
        canvas_size = tuple(map(int, zattrs.get("canvas_size", [])))
        expected_canvas_yx = [
            int(zshape[1] * 2**ZARR_LEVEL),
            int(zshape[2] * 2**ZARR_LEVEL),
        ]
        if len(canvas_size) != 2 or any(
            abs(actual - expected) >= 2**ZARR_LEVEL
            for actual, expected in zip(canvas_size[::-1], expected_canvas_yx)
        ):
            raise RuntimeError(
                f"invalid or inconsistent canvas_size={canvas_size}, zarr={zshape}"
            )
        if (
            abs(zshape[1] - registered_preview_shape[0]) > 1
            or abs(zshape[2] - registered_preview_shape[1]) > 1
        ):
            raise RuntimeError(
                "preview/Zarr mismatch: "
                f"preview={registered_preview_shape}, zarr={zshape}"
            )

        tifxyz_dir = task_root / "tifxyz"
        for axis in ("x", "y", "z"):
            fetch(
                session,
                public_url(f"{assets.tifxyz_path}{axis}.tif"),
                tifxyz_dir / f"{axis}.tif",
            )

        proposals = sample_proposals(
            preview,
            task_id=task["task_id"],
            high_count=high_per_surface,
            mid_count=mid_per_surface,
            registered_shape_yx=registered_preview_shape,
        )
        chunk_cache: dict[tuple[int, int, int], Path] = {}
        for proposal in proposals:
            chunk = tuple(proposal["zarr_chunk_zyx"])
            if chunk not in chunk_cache:
                relative = "/".join(map(str, chunk))
                chunk_cache[chunk] = fetch(
                    session,
                    public_url(f"{assets.zarr_path}{ZARR_LEVEL}/{relative}"),
                    task_root / "chunks" / f"{relative.replace('/', '_')}.bin",
                    expected_bytes=ZARR_DEPTH * ZARR_CHUNK_EDGE * ZARR_CHUNK_EDGE,
                )
            stack = np.fromfile(chunk_cache[chunk], dtype=np.uint8).reshape(
                ZARR_DEPTH, ZARR_CHUNK_EDGE, ZARR_CHUNK_EDGE
            )
            proposal["ct_coordinate_xyz"] = load_tifxyz_coordinates(
                tifxyz_dir,
                proposal,
                canvas_size_xy=canvas_size,
            )
            proposal["voxel_size_um"] = [assets.voxel_size_um] * 3
            proposal["slice_order"] = (
                "TOP_TO_BOTTOM"
                if assets.z_direction_is_top_to_bottom
                else "BOTTOM_TO_TOP"
            )
            proposal["scanner_domain"] = assets.scanner_domain
            proposal["left_handed_coordinates"] = assets.left_handed_coordinates
            proposal["surface_group_id"] = task["surface_group_id"]
            proposal["scroll_id"] = scroll_id
            proposal["segment_id"] = segment_id
            proposal["task_id"] = task["task_id"]
            proposal["ct_chunk_sha256"] = sha256(chunk_cache[chunk])
            review_file = (
                task_root / "review" / f"{proposal['proposal_id']}.png"
            )
            build_review_image(stack, preview, proposal, review_file)
            proposal["review_image"] = str(review_file.relative_to(output_root))
            proposal["review_image_sha256"] = sha256(review_file)
            all_proposals.append(proposal)

        task_receipt = {
            "task_id": task["task_id"],
            "scroll_id": scroll_id,
            "segment_id": segment_id,
            "surface_group_id": task["surface_group_id"],
            "status": "CT_EVIDENCE_READY_LABELS_UNREVIEWED",
            "proposal_count": len(proposals),
            "assets": {
                "preview": {
                    "path": assets.preview_path,
                    "sha256": sha256(preview_file),
                    "shape_y_x": list(map(int, preview.shape)),
                },
                "surface_zarr": {
                    "path": assets.zarr_path,
                    "zarray_sha256": sha256(zarray_file),
                    "zattrs_sha256": sha256(zattrs_file),
                    "level": ZARR_LEVEL,
                    "shape_zyx": zshape,
                },
                "tifxyz": {
                    "path": assets.tifxyz_path,
                    "x_sha256": sha256(tifxyz_dir / "x.tif"),
                    "y_sha256": sha256(tifxyz_dir / "y.tif"),
                    "z_sha256": sha256(tifxyz_dir / "z.tif"),
                },
            },
            "source": {
                "volume_id": assets.volume_id,
                "voxel_size_um": assets.voxel_size_um,
                "scanner_domain": assets.scanner_domain,
            },
        }
        task_receipt_path = task_root / "ACQUISITION_RECEIPT.json"
        task_receipt_path.write_text(
            json.dumps(task_receipt, indent=2, sort_keys=True) + "\n"
        )
        task_receipt["receipt_sha256"] = sha256(task_receipt_path)
        completed_tasks.append(task_receipt)

    proposals_file = output_root / "LABEL_BLIND_CT_PROPOSALS.json"
    proposals_file.write_text(
        json.dumps(all_proposals, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "schema": "campaignx.multiscroll_ct_adjudication_acquisition.v1",
        "benchmark_id": "MULTISCROLL_TRANSFER_V1",
        "status": "CT_EVIDENCE_READY_LABELS_UNREVIEWED",
        "generated_at_utc": utc_now(),
        "queue": {"path": str(queue_path), "sha256": sha256(queue_path)},
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256(metadata_path),
        },
        "policy": {
            "prediction_role": "CANDIDATE_LOCATOR_ONLY",
            "high_response_proposals_per_surface": high_per_surface,
            "mid_response_proposals_per_surface": mid_per_surface,
            "zarr_level": ZARR_LEVEL,
            "gate_outputs_visible": False,
        },
        "task_count": len(completed_tasks),
        "proposal_count": len(all_proposals),
        "proposals_file": {
            "path": str(proposals_file.relative_to(output_root)),
            "sha256": sha256(proposals_file),
        },
        "tasks": completed_tasks,
        "non_claims": [
            "no proposal is labeled positive or confound",
            "official predictions are not ground truth",
            "v3 and v4 have not been executed",
            "no ink, text, letters, or First Letters are accepted",
        ],
    }
    receipt_path = output_root / "ACQUISITION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--high-per-surface", type=int, default=20)
    parser.add_argument("--mid-per-surface", type=int, default=20)
    args = parser.parse_args()
    if min(args.high_per_surface, args.mid_per_surface) < 1:
        raise ValueError("both proposal strata must be non-empty")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "ACQUISITION_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError("refusing to overwrite completed acquisition receipt")
    receipt = run(
        queue_path=args.queue.resolve(),
        metadata_path=args.metadata.resolve(),
        output_root=output_root,
        high_per_surface=args.high_per_surface,
        mid_per_surface=args.mid_per_surface,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "task_count": receipt["task_count"],
                "proposal_count": receipt["proposal_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
