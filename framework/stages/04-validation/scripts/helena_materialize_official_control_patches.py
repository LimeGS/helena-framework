#!/usr/bin/env python3
"""Materialize frozen level-0 CT patches for a multiscroll benchmark.

The input controls were selected from public curated labels without reading
v3/v4.  This stage downloads only the level-0 surface-CT chunks intersecting
the fixed 200 um audit window, verifies the label at every center, reads the
corresponding public TIFXYZ coordinate, and writes one immutable patch tensor.

It does not import or execute either gate.  Its output status is therefore
``OFFICIAL_CONTROLS_FROZEN_BEFORE_V3_V4``.  A separate one-shot executor must
consume the tensor and create an execution claim before exposing decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import zarr
from skimage.transform import resize


PATCH_RADIUS_UM = 200.0
VOXEL_PROVENANCE = {
    "PHerc1667": {
        "voxel_size_um": [2.399, 2.399, 2.399],
        "volume_id": "20251217075048",
        "volume_long_id": "20251217075048-2.399um-0.2m-78keV-masked.zarr",
    },
    "PHercParis4": {
        "voxel_size_um": [2.4, 2.4, 2.4],
        "volume_id": "20260411134726",
        "volume_long_id": "20260411134726-2.400um-0.2m-78keV-masked.zarr",
    },
    "PHerc0814": {
        "voxel_size_um": [2.399, 2.399, 2.399],
        "volume_id": "20260309142202",
        "volume_long_id": "20260309142202-2.399um-0.2m-78keV-masked.zarr",
    },
}


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


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _copy(uri: str, destination: Path) -> Path:
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["hf", "buckets", "cp", uri, str(destination), "-q"],
            check=True,
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"incomplete public asset after copy: {uri}")
    return destination


def _sync_ct_chunks(
    uri: str,
    destination: Path,
    chunk_names: set[str],
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "hf",
        "buckets",
        "sync",
        f"{uri}/0",
        str(destination),
        "--include",
        ".zarray",
        "--include",
        ".zattrs",
    ]
    for name in sorted(chunk_names):
        command.extend(["--include", name])
    command.append("-q")
    subprocess.run(command, check=True)
    if not (destination / ".zarray").is_file():
        raise RuntimeError(f"level-0 CT metadata missing after sync: {uri}")
    missing = [
        name
        for name in sorted(chunk_names)
        if not (destination / name).is_file()
        or (destination / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"level-0 CT chunks missing after sync for {uri}: {missing[:5]}"
        )
    return destination


def _zarr_metadata(uri: str, destination: Path) -> dict[str, Any]:
    path = _copy(f"{uri}/0/.zarray", destination)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("shape", [])) != 3 or len(payload.get("chunks", [])) != 3:
        raise RuntimeError(f"unexpected surface CT Zarr metadata: {uri}")
    if int(payload["shape"][0]) != 65 or int(payload["chunks"][0]) <= 0:
        raise RuntimeError(f"surface CT must contain exactly 65 planes: {uri}")
    return payload


def _chunks_for_controls(
    controls: list[dict[str, Any]],
    *,
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
    radius: int,
) -> set[str]:
    names: set[str] = set()
    for control in controls:
        y, x = map(int, control["surface_y_x_level0"])
        if not (0 <= y < shape[1] and 0 <= x < shape[2]):
            raise RuntimeError(
                f"control outside surface CT: {control['component_id']}"
            )
        y0, y1 = max(0, y - radius), min(shape[1], y + radius + 1)
        x0, x1 = max(0, x - radius), min(shape[2], x + radius + 1)
        for chunk_z in range(math.ceil(shape[0] / chunks[0])):
            for chunk_y in range(y0 // chunks[1], (y1 - 1) // chunks[1] + 1):
                for chunk_x in range(
                    x0 // chunks[2],
                    (x1 - 1) // chunks[2] + 1,
                ):
                    names.add(f"{chunk_z}.{chunk_y}.{chunk_x}")
    return names


def _crop_with_padding(
    array: Any,
    *,
    center_y: int,
    center_x: int,
    radius: int,
) -> np.ndarray:
    size = 2 * radius + 1
    output = np.zeros((65, size, size), dtype=np.uint8)
    y0 = max(0, center_y - radius)
    y1 = min(int(array.shape[1]), center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(int(array.shape[2]), center_x + radius + 1)
    output_y = y0 - (center_y - radius)
    output_x = x0 - (center_x - radius)
    output[
        :,
        output_y : output_y + y1 - y0,
        output_x : output_x + x1 - x0,
    ] = np.asarray(array[:, y0:y1, x0:x1], dtype=np.uint8)
    return output


def _read_points(
    path: Path,
    controls: list[dict[str, Any]],
    *,
    surface_shape_y_x: tuple[int, int],
    tifxyz_scale_y_x: tuple[float, float],
) -> tuple[list[float], list[list[int]]]:
    array = tifffile.imread(path)
    try:
        if array.ndim != 2:
            raise RuntimeError(f"TIFXYZ map must be 2D: {path}")
        expected_shapes = [
            tuple(
                length * scale
                for length, scale in zip(
                    surface_shape_y_x,
                    candidate_scale,
                    strict=True,
                )
            )
            for candidate_scale in (
                tifxyz_scale_y_x,
                tuple(reversed(tifxyz_scale_y_x)),
            )
        ]
        compatible = any(
            all(
                abs(actual - expected) <= max(4.0, expected * 0.002)
                for actual, expected in zip(
                    array.shape,
                    expected_shape,
                    strict=True,
                )
            )
            for expected_shape in expected_shapes
        )
        if not compatible:
            raise RuntimeError(
                "TIFXYZ shape does not match the declared meta.json scale: "
                f"{path} expected_one_of={expected_shapes} actual={array.shape}"
            )
        values: list[float] = []
        sampled_y_x: list[list[int]] = []
        for control in controls:
            y, x = map(int, control["surface_y_x_level0"])
            # Some official surfaces round the low-resolution TIFXYZ extent
            # differently on each axis (for example expected 2125x4000 but
            # published 2124x4001).  The declared scale validates the domain;
            # normalized extent mapping avoids an off-by-one drift at the far
            # edge while remaining deterministic.
            tif_y = min(
                int(math.floor(y * array.shape[0] / surface_shape_y_x[0])),
                array.shape[0] - 1,
            )
            tif_x = min(
                int(math.floor(x * array.shape[1] / surface_shape_y_x[1])),
                array.shape[1] - 1,
            )
            if not (0 <= tif_y < array.shape[0] and 0 <= tif_x < array.shape[1]):
                raise RuntimeError(
                    f"TIFXYZ point outside map: {control['component_id']}"
                )
            value = float(array[tif_y, tif_x])
            sample_y, sample_x = tif_y, tif_x
            if not math.isfinite(value) or value < 0:
                alternatives: list[tuple[int, int, int, float]] = []
                for delta_y in range(-2, 3):
                    for delta_x in range(-2, 3):
                        sample_y = tif_y + delta_y
                        sample_x = tif_x + delta_x
                        if not (
                            0 <= sample_y < array.shape[0]
                            and 0 <= sample_x < array.shape[1]
                        ):
                            continue
                        sample = float(array[sample_y, sample_x])
                        if math.isfinite(sample) and sample >= 0:
                            alternatives.append(
                                (
                                    delta_y * delta_y + delta_x * delta_x,
                                    delta_y,
                                    delta_x,
                                    sample,
                                )
                            )
                if not alternatives:
                    raise RuntimeError(
                        "invalid TIFXYZ coordinate sentinel with no valid "
                        f"coarse-grid neighbor: {control['component_id']}"
                    )
                _, delta_y, delta_x, value = min(alternatives)
                sample_y = tif_y + delta_y
                sample_x = tif_x + delta_x
            values.append(value)
            sampled_y_x.append([sample_y, sample_x])
        return values, sampled_y_x
    finally:
        del array


def materialize(
    candidates: dict[str, Any],
    source_plan: dict[str, Any],
    *,
    mirror_root: Path,
    output_root: Path,
    official_metadata_sha256: str,
    benchmark_id: str = "MULTISCROLL_TRANSFER_V1",
    retain_source_cache: bool = False,
    voxel_provenance: dict[str, dict[str, Any]] | None = None,
    canonical_xy_spacing_um: float | None = None,
) -> dict[str, Any]:
    if (
        candidates.get("status")
        != "OFFICIAL_CONTROL_CANDIDATES_READY_FOR_LEVEL0_MATERIALIZATION"
    ):
        raise RuntimeError("official control candidate set is not ready")
    if source_plan.get("status") != "OFFICIAL_LABEL_SOURCES_READY":
        raise RuntimeError("official source plan is not ready")
    controls = list(candidates["controls"])
    if not controls:
        raise RuntimeError("cannot materialize an empty control set")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to overwrite materialized controls")
    output_root.mkdir(parents=True, exist_ok=True)
    mirror_root.mkdir(parents=True, exist_ok=True)

    surfaces = {
        (str(row["scroll_id"]), str(row["official_surface_id"])): row
        for row in source_plan["surfaces"]
        if row["status"] == "READY"
    }
    by_surface: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for control in controls:
        key = (str(control["scroll_id"]), str(control["official_surface_id"]))
        if key not in surfaces:
            raise RuntimeError(f"control references unknown official surface: {key}")
        by_surface[key].append(control)

    selected_voxel_provenance = voxel_provenance or VOXEL_PROVENANCE
    unknown_scrolls = sorted(
        {str(row["scroll_id"]) for row in controls}
        - set(selected_voxel_provenance)
    )
    if unknown_scrolls:
        raise RuntimeError(
            f"missing voxel provenance for scrolls: {unknown_scrolls}"
        )
    radius_by_scroll = {
        scroll: max(2, round(PATCH_RADIUS_UM / payload["voxel_size_um"][0]))
        for scroll, payload in selected_voxel_provenance.items()
    }
    if canonical_xy_spacing_um is None:
        patch_sizes = {
            2 * radius_by_scroll[str(row["scroll_id"])] + 1
            for row in controls
        }
        if len(patch_sizes) != 1:
            raise RuntimeError(
                "patch sizes differ across scrolls; provide "
                "--canonical-xy-spacing-um to resample physically"
            )
        patch_size = patch_sizes.pop()
        output_xy_spacing_um_by_scroll = {
            scroll: float(payload["voxel_size_um"][0])
            for scroll, payload in selected_voxel_provenance.items()
        }
    else:
        if not math.isfinite(canonical_xy_spacing_um) or canonical_xy_spacing_um <= 0:
            raise RuntimeError("canonical XY spacing must be finite and positive")
        canonical_radius = max(2, round(PATCH_RADIUS_UM / canonical_xy_spacing_um))
        patch_size = 2 * canonical_radius + 1
        output_xy_spacing_um_by_scroll = {
            scroll: float(canonical_xy_spacing_um)
            for scroll in selected_voxel_provenance
        }
    tensor_path = output_root / "CONTROL_CT_PATCHES.npy"
    tensor = np.lib.format.open_memmap(
        tensor_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(controls), 65, patch_size, patch_size),
    )
    index_by_component = {
        str(control["component_id"]): index
        for index, control in enumerate(controls)
    }
    if len(index_by_component) != len(controls):
        raise RuntimeError("duplicate component_id in official controls")

    source_receipts: list[dict[str, Any]] = []
    frozen_rows: list[dict[str, Any]] = []
    for key in sorted(by_surface):
        scroll_id, surface_id = key
        surface = surfaces[key]
        selected = sorted(
            by_surface[key],
            key=lambda row: str(row["component_id"]),
        )
        local = mirror_root / scroll_id / surface_id
        ct_uri = str(surface["assets"]["surface_ct"]["uri"])
        metadata = _zarr_metadata(ct_uri, local / "ct-level0.zarray.json")
        shape = tuple(map(int, metadata["shape"]))
        chunks = tuple(map(int, metadata["chunks"]))
        radius = radius_by_scroll[scroll_id]
        chunk_names = _chunks_for_controls(
            selected,
            shape=shape,
            chunks=chunks,
            radius=radius,
        )
        ct_root = _sync_ct_chunks(
            ct_uri,
            local / "surface-ct.zarr" / "0",
            chunk_names,
        )
        ct_array = zarr.open(str(ct_root), mode="r")

        coordinate_values: dict[str, list[float]] = {}
        coordinate_sample_indices: dict[str, list[list[int]]] = {}
        coordinate_assets: list[dict[str, Any]] = []
        metadata_asset = surface["assets"]["metadata"]
        metadata_path = _copy(
            str(metadata_asset["uri"]),
            local / "official-meta.json",
        )
        official_surface_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        tifxyz_scale = tuple(
            map(float, official_surface_metadata.get("scale", []))
        )
        if len(tifxyz_scale) != 2 or any(
            not math.isfinite(value) or value <= 0 or value > 1
            for value in tifxyz_scale
        ):
            raise RuntimeError(f"invalid TIFXYZ scale in metadata for {key}")
        for axis in ("x", "y", "z"):
            asset = surface["assets"][f"tifxyz_{axis}"]
            path = _copy(str(asset["uri"]), local / f"{axis}.tif")
            (
                coordinate_values[axis],
                coordinate_sample_indices[axis],
            ) = _read_points(
                path,
                selected,
                surface_shape_y_x=(shape[1], shape[2]),
                tifxyz_scale_y_x=tifxyz_scale,
            )
            coordinate_assets.append(
                {
                    "axis": axis,
                    "uri": asset["uri"],
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
            if not retain_source_cache:
                path.unlink()
        if not (
            coordinate_sample_indices["x"]
            == coordinate_sample_indices["y"]
            == coordinate_sample_indices["z"]
        ):
            raise RuntimeError(
                f"TIFXYZ axes selected different coarse-grid samples for {key}"
            )

        label_path = mirror_root / "selection" / scroll_id / surface_id / "label.tif"
        supervision_path = (
            mirror_root / "selection" / scroll_id / surface_id / "supervision.tif"
        )
        if not label_path.is_file() or not supervision_path.is_file():
            raise RuntimeError(
                f"selection masks were not provided in mirror for {key}"
            )
        label = np.asarray(tifffile.imread(label_path)) > 0
        supervision = np.asarray(tifffile.imread(supervision_path)) > 0
        output_xy_spacing_um = output_xy_spacing_um_by_scroll[scroll_id]
        bbox_half_width = max(
            2,
            round(
                40.0
                / output_xy_spacing_um
            ),
        )
        try:
            for local_index, control in enumerate(selected):
                component_id = str(control["component_id"])
                y, x = map(int, control["surface_y_x_level0"])
                is_label = bool(label[y, x])
                is_supervised = bool(supervision[y, x])
                expected_label = str(control["expected_class"])
                y0, y1 = max(0, y - bbox_half_width), min(
                    label.shape[0], y + bbox_half_width + 1
                )
                x0, x1 = max(0, x - bbox_half_width), min(
                    label.shape[1], x + bbox_half_width + 1
                )
                if not is_supervised:
                    raise RuntimeError(
                        f"control center is outside supervision: {component_id}"
                    )
                if not bool(np.all(supervision[y0:y1, x0:x1])):
                    raise RuntimeError(
                        "control audit bbox leaves official supervision: "
                        f"{component_id}"
                    )
                if expected_label == "POSITIVE" and not is_label:
                    raise RuntimeError(
                        f"positive center is not official ink: {component_id}"
                    )
                if expected_label == "CONFOUND" and bool(
                    np.any(label[y0:y1, x0:x1])
                ):
                    raise RuntimeError(
                        f"confound audit bbox overlaps official ink: {component_id}"
                    )
                tensor_index = index_by_component[component_id]
                raw_patch = _crop_with_padding(
                    ct_array,
                    center_y=y,
                    center_x=x,
                    radius=radius,
                )
                if raw_patch.shape[1:] != (patch_size, patch_size):
                    raw_patch = resize(
                        raw_patch,
                        (65, patch_size, patch_size),
                        order=1,
                        mode="edge",
                        anti_aliasing=True,
                        preserve_range=True,
                    ).astype(np.uint8)
                tensor[tensor_index] = raw_patch
                center = patch_size // 2
                frozen_rows.append(
                    {
                        **control,
                        "ct_coordinate_xyz": [
                            coordinate_values[axis][local_index]
                            for axis in ("x", "y", "z")
                        ],
                        "voxel_size_um": selected_voxel_provenance[scroll_id][
                            "voxel_size_um"
                        ],
                        "patch_xy_spacing_um": output_xy_spacing_um,
                        "source_patch_radius_pixels": radius,
                        "slice_order": "SURFACE_CT_DEPTH_ASCENDING_0_TO_64",
                        "scanner_domain": selected_voxel_provenance[scroll_id][
                            "volume_long_id"
                        ],
                        "volume_id": selected_voxel_provenance[scroll_id][
                            "volume_id"
                        ],
                        "patch_tensor_index": tensor_index,
                        "patch_center_y_x": [center, center],
                        "analysis_bbox_xyxy": [
                            center - bbox_half_width,
                            center - bbox_half_width,
                            center + bbox_half_width + 1,
                            center + bbox_half_width + 1,
                        ],
                        "label_center_verified": True,
                        "label_frozen_before_v3_v4": True,
                    }
                )
        finally:
            del label
            del supervision
            del ct_array
        ct_tree_digest = tree_sha256(ct_root)
        source_receipts.append(
            {
                "scroll_id": scroll_id,
                "official_surface_id": surface_id,
                "surface_ct_uri": ct_uri,
                "surface_ct_level0_shape": list(shape),
                "surface_ct_level0_chunks": list(chunks),
                "downloaded_chunk_count": len(chunk_names),
                "downloaded_chunk_names": sorted(chunk_names),
                "downloaded_ct_tree_sha256": ct_tree_digest,
                "tifxyz": coordinate_assets,
                "tifxyz_mapping": {
                    "metadata_uri": metadata_asset["uri"],
                    "metadata_sha256": sha256(metadata_path),
                    "scale_y_x": list(tifxyz_scale),
                    "index_transform": (
                        "floor(surface_index * tifxyz_extent / surface_extent); "
                        "meta.json scale (XY or YX order) validates extent "
                        "within 0.2% or 4 pixels"
                    ),
                    "invalid_sentinel_policy": (
                        "deterministic nearest finite nonnegative coarse-grid "
                        "sample within radius 2; fail closed if absent"
                    ),
                },
            }
        )
        if not retain_source_cache:
            shutil.rmtree(ct_root.parent)
    tensor.flush()
    del tensor

    frozen_rows.sort(
        key=lambda row: (
            str(row["scroll_id"]),
            str(row["expected_class"]),
            str(row["component_id"]),
        )
    )
    frozen_controls_path = output_root / "FROZEN_CONTROLS.json"
    frozen_controls_path.write_text(
        json.dumps(frozen_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "campaignx.multiscroll_official_control_materialization.v1",
        "benchmark_id": benchmark_id,
        "status": "OFFICIAL_CONTROLS_FROZEN_BEFORE_V3_V4",
        "generated_at_utc": utc_now(),
        "candidate_set_sha256": candidates["content_sha256"],
        "source_plan_sha256": source_plan["content_sha256"],
        "official_metadata_sha256": official_metadata_sha256,
        "control_count": len(frozen_rows),
        "patch_radius_um": PATCH_RADIUS_UM,
        "patch_shape_n_z_y_x": [len(frozen_rows), 65, patch_size, patch_size],
        "label_center_verification_count": sum(
            bool(row["label_center_verified"]) for row in frozen_rows
        ),
        "voxel_provenance": selected_voxel_provenance,
        "canonical_xy_spacing_um": canonical_xy_spacing_um,
        "source_receipts": source_receipts,
        "artifacts": {
            "patch_tensor": tensor_path.name,
            "patch_tensor_sha256": sha256(tensor_path),
            "patch_tensor_size_bytes": tensor_path.stat().st_size,
            "frozen_controls": frozen_controls_path.name,
            "frozen_controls_sha256": sha256(frozen_controls_path),
        },
        "gate_visibility": {
            "v3_executed": False,
            "v4_executed": False,
            "v3_v4_outputs_read_during_label_creation": False,
        },
        "source_cache_retained": retain_source_cache,
        "non_claims": [
            "Public curated ink labels define controls; model predictions do not.",
            "No filter result influenced label selection.",
            "No ink, text, letters, or First Letters are accepted automatically.",
        ],
    }
    receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    (output_root / "MATERIALIZATION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--selection-mirror-root", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-metadata", type=Path, required=True)
    parser.add_argument(
        "--benchmark-id",
        choices=[
            "MULTISCROLL_TRANSFER_V1",
            "MULTISCROLL_TRANSFER_V2",
            "MULTISCROLL_TRANSFER_V3",
            "CT_PRIORITY_ROUTER_V43_DEVELOPMENT",
            "MULTISCROLL_TRANSFER_V4",
            "MULTISCROLL_TRANSFER_V5",
        ],
        default="MULTISCROLL_TRANSFER_V1",
    )
    parser.add_argument(
        "--retain-source-cache",
        action="store_true",
        help="Keep downloaded level-0 CT chunks and TIFXYZ maps after freezing.",
    )
    parser.add_argument(
        "--voxel-provenance",
        type=Path,
        help="Optional JSON mapping scroll_id to voxel and scanner provenance.",
    )
    parser.add_argument(
        "--canonical-xy-spacing-um",
        type=float,
        help="Resample every physical audit patch to this XY spacing.",
    )
    args = parser.parse_args()
    candidates_path = args.candidates.resolve()
    source_plan_path = args.source_plan.resolve()
    metadata_path = args.official_metadata.resolve()
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates["content_sha256"] = sha256(candidates_path)
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    # Keep the label/supervision mirror separate from the level-0 execution
    # mirror.  This prevents gate materialization from mutating label inputs.
    execution_mirror = args.mirror_root.resolve()
    selection_mirror = args.selection_mirror_root.resolve()
    if execution_mirror == selection_mirror:
        raise RuntimeError("selection and execution mirrors must be separate")
    # Link-free path mapping: the materializer only reads the frozen selection
    # masks through a stable subdirectory in its own mirror.
    execution_mirror.mkdir(parents=True, exist_ok=True)
    selection_mount = execution_mirror / "selection"
    if selection_mount.is_symlink():
        if selection_mount.resolve(strict=False) != selection_mirror:
            raise RuntimeError(
                "selection mirror symlink points at a different frozen source"
            )
    elif not selection_mount.exists():
        selection_mount.symlink_to(selection_mirror, target_is_directory=True)
    elif selection_mount.resolve() != selection_mirror:
        raise RuntimeError(
            "selection mirror mount exists but is not the requested source"
        )
    receipt = materialize(
        candidates,
        source_plan,
        mirror_root=execution_mirror,
        output_root=args.output_root.resolve(),
        official_metadata_sha256=sha256(metadata_path),
        benchmark_id=args.benchmark_id,
        retain_source_cache=args.retain_source_cache,
        voxel_provenance=(
            json.loads(
                args.voxel_provenance.resolve().read_text(encoding="utf-8")
            )
            if args.voxel_provenance
            else None
        ),
        canonical_xy_spacing_um=args.canonical_xy_spacing_um,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
