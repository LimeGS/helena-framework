#!/usr/bin/env python3
"""Select prospective positive and hard-confound controls from official labels.

This is the second stage of the public-label path for
``MULTISCROLL_TRANSFER_V1``.  It downloads only level 5 of the official
ink-label, supervision-mask and surface-CT Zarr pyramids.  The reduced arrays
are used to select controls; exact level-0 CT/TIFXYZ evidence is materialized
by the following stage.

Positive controls are spatially separated, label-supported windows in the
curated ink mask.  A single connected mask may contain many letters or strokes
and is therefore not treated as one benchmark observation.
Confounds are high-CT-gradient locations explicitly covered by the curated
supervision mask and labeled non-ink.  The gradient is a label-blind
difficulty locator, not ground truth.  Neither v3 nor v4 is imported or run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import tifffile
import zarr
from scipy import ndimage
from skimage.measure import block_reduce


PYRAMID_LEVEL = 5
LEVEL_SCALE = 2**PYRAMID_LEVEL
TARGET_PER_CLASS_PER_SCROLL = 50
TARGET_GROUPS_PER_SCROLL = 5
SUPERVISION_MARGIN_LEVEL0_PIXELS = 20
DEFAULT_EXCLUSION_RADIUS_LEVEL0_PIXELS = 176.0
CONFOUND_WITHIN_BENCHMARK_SPACING_LEVEL5_PIXELS = 1.0


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


class SurfaceLoader(Protocol):
    def load(self, surface: dict[str, Any]) -> dict[str, Any]: ...


class HfLevelLoader:
    def __init__(self, mirror_root: Path):
        self.mirror_root = mirror_root.resolve()

    @staticmethod
    def _zarr_uri_from_tif(uri: str) -> str:
        if not uri.endswith(".tif"):
            raise ValueError(f"expected TIFF asset, got {uri}")
        return uri[:-4] + ".zarr"

    def _sync(self, uri: str, destination: Path) -> Path:
        level_uri = f"{uri}/{PYRAMID_LEVEL}"
        if not (destination / ".zarray").is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["hf", "buckets", "sync", level_uri, str(destination), "-q"],
                check=True,
            )
        if not (destination / ".zarray").is_file():
            raise RuntimeError(
                f"incomplete Hugging Face Zarr level after sync: {level_uri}"
            )
        return destination

    def _copy(self, uri: str, destination: Path) -> Path:
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["hf", "buckets", "cp", uri, str(destination)],
                check=True,
            )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"incomplete Hugging Face file after copy: {uri}")
        return destination

    @staticmethod
    def _downsample_masks(
        label: np.ndarray,
        supervision: np.ndarray,
        output_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Conservatively align level-0 masks to one reduced CT plane.

        Any ink in a reduced cell excludes that cell from the confound pool.
        A cell is supervised only when all source pixels in the corresponding
        neighborhood are supervised.
        """

        if label.shape != supervision.shape:
            raise RuntimeError("official label/supervision TIFF shapes differ")
        if label.ndim != 2:
            raise RuntimeError("expected 2D official label/supervision TIFFs")
        output_y, output_x = output_shape
        reduced_label = block_reduce(
            label,
            block_size=(LEVEL_SCALE, LEVEL_SCALE),
            func=np.max,
            cval=0,
        )
        reduced_supervision = block_reduce(
            supervision,
            block_size=(LEVEL_SCALE, LEVEL_SCALE),
            func=np.min,
            cval=0,
        )
        reduced_shape = tuple(map(int, reduced_label.shape))
        target_shape = (int(output_y), int(output_x))
        pad_y = target_shape[0] - reduced_shape[0]
        pad_x = target_shape[1] - reduced_shape[1]
        if pad_y < 0 or pad_x < 0 or pad_y > 4 or pad_x > 4:
            raise RuntimeError(
                "official mask pyramid shape does not match surface CT: "
                f"{reduced_shape} != {target_shape}; only <=4-cell "
                "bottom/right CT chunk padding is supported"
            )
        if pad_y or pad_x:
            padding = ((0, pad_y), (0, pad_x))
            reduced_label = np.pad(
                reduced_label,
                padding,
                mode="constant",
                constant_values=False,
            )
            reduced_supervision = np.pad(
                reduced_supervision,
                padding,
                mode="constant",
                constant_values=False,
            )
        alignment = {
            "kind": (
                "TOP_LEFT_ORIGIN_BOTTOM_RIGHT_ZERO_PADDING"
                if pad_y or pad_x
                else "EXACT_PYRAMID_SHAPE"
            ),
            "reduced_mask_shape_before_padding": list(reduced_shape),
            "surface_ct_plane_shape": list(target_shape),
            "bottom_padding_cells": int(pad_y),
            "right_padding_cells": int(pad_x),
            "label_interpolation": "NONE_BLOCK_MAX",
            "supervision_interpolation": "NONE_BLOCK_MIN",
            "padded_cells_are_selectable": False,
        }
        return reduced_label, reduced_supervision, alignment

    def load(self, surface: dict[str, Any]) -> dict[str, Any]:
        surface_id = str(surface["official_surface_id"])
        local = self.mirror_root / str(surface["scroll_id"]) / surface_id
        label_uri = str(surface["assets"]["ink_label"]["uri"])
        supervision_uri = str(surface["assets"]["supervision_mask"]["uri"])
        ct_uri = str(surface["assets"]["surface_ct"]["uri"])
        paths = {
            "label": self._copy(label_uri, local / "label.tif"),
            "supervision": self._copy(
                supervision_uri,
                local / "supervision.tif",
            ),
            "ct": self._sync(ct_uri, local / "surface-ct.zarr" / str(PYRAMID_LEVEL)),
        }
        label = np.asarray(tifffile.imread(paths["label"])) > 0
        supervision = np.asarray(tifffile.imread(paths["supervision"])) > 0
        ct_array = zarr.open(str(paths["ct"]), mode="r")
        ct_shape = tuple(ct_array.shape)
        if len(ct_shape) != 3:
            raise RuntimeError(f"expected depth,y,x arrays for {surface_id}")
        if label.shape != supervision.shape:
            raise RuntimeError(
                f"unaligned official label masks for {surface_id}: "
                f"{label.shape} != {supervision.shape}"
            )
        central_slice = int(ct_shape[0]) // 2
        ct = np.asarray(ct_array[central_slice], dtype=np.float32)
        reduced_label, reduced_supervision, mask_alignment = self._downsample_masks(
            label,
            supervision,
            tuple(ct.shape),
        )
        payload = {
            "label_level0": label,
            "supervision_level0": supervision,
            "label_reduced": reduced_label,
            "supervision_reduced": reduced_supervision,
            "ct": ct,
            "central_slice": central_slice,
            "mask_alignment": mask_alignment,
            "shapes": {
                "label_yx": list(label.shape),
                "supervision_yx": list(supervision.shape),
                "ct_zyx": list(ct_shape),
            },
            "assets": {
                name: {
                    "uri": (
                        label_uri
                        if name == "label"
                        else supervision_uri
                        if name == "supervision"
                        else ct_uri
                    )
                    + (f"/{PYRAMID_LEVEL}/" if name == "ct" else ""),
                    "sha256": (
                        tree_sha256(path) if name == "ct" else sha256(path)
                    ),
                    "role": (
                        "INK_LABEL"
                        if name == "label"
                        else "SUPERVISION_MASK"
                        if name == "supervision"
                        else "SURFACE_CT"
                    ),
                }
                for name, path in paths.items()
            },
        }
        return payload


def _spaced(
    candidates: list[tuple[float, int, int]],
    *,
    limit: int,
    minimum_distance: float,
) -> list[tuple[float, int, int]]:
    selected: list[tuple[float, int, int]] = []
    minimum_squared = minimum_distance * minimum_distance
    for score, y, x in candidates:
        if all(
            (y - other_y) ** 2 + (x - other_x) ** 2 >= minimum_squared
            for _, other_y, other_x in selected
        ):
            selected.append((score, y, x))
            if len(selected) == limit:
                break
    return selected


def _outside_frozen_control_exclusions(
    candidates: list[tuple[float, int, int]],
    *,
    selection_level: int,
    excluded_level0_y_x: list[tuple[int, int]],
    minimum_distance_level0: float,
) -> list[tuple[float, int, int]]:
    """Keep prospective centers disjoint from a previously frozen benchmark.

    The distance is evaluated in the official surface's level-0 canvas.  The
    default radius is larger than the diameter of one 200 um audit patch at
    2.4 um/voxel, so V2 cannot silently reuse V1 CT evidence.
    """

    if not excluded_level0_y_x:
        return candidates
    scale = 2**selection_level
    minimum_squared = minimum_distance_level0 * minimum_distance_level0
    kept: list[tuple[float, int, int]] = []
    for score, y, x in candidates:
        full_y = int(y * scale + scale // 2)
        full_x = int(x * scale + scale // 2)
        if all(
            (full_y - old_y) ** 2 + (full_x - old_x) ** 2
            >= minimum_squared
            for old_y, old_x in excluded_level0_y_x
        ):
            kept.append((score, y, x))
    return kept


def _positive_candidates(label: np.ndarray) -> list[tuple[float, int, int]]:
    if label.ndim != 2:
        raise ValueError("positive label must be 2D")
    rows: set[tuple[float, int, int]] = set()
    step = 64
    for offset_y, offset_x in (
        (step // 2, step // 2),
        (0, 0),
        (0, step // 2),
        (step // 2, 0),
    ):
        ys_grid = np.arange(offset_y, label.shape[0], step, dtype=np.int64)
        xs_grid = np.arange(offset_x, label.shape[1], step, dtype=np.int64)
        ys_grid = ys_grid[(ys_grid >= 1) & (ys_grid < label.shape[0] - 1)]
        xs_grid = xs_grid[(xs_grid >= 1) & (xs_grid < label.shape[1] - 1)]
        if not len(ys_grid) or not len(xs_grid):
            continue
        interior = np.ones((len(ys_grid), len(xs_grid)), dtype=bool)
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                interior &= label[
                    np.ix_(ys_grid + delta_y, xs_grid + delta_x)
                ]
        ys, xs = np.nonzero(interior)
        for y_index, x_index in zip(ys, xs, strict=True):
            full_y = int(ys_grid[y_index])
            full_x = int(xs_grid[x_index])
            rows.add((1.0, full_y, full_x))
    return sorted(rows, key=lambda row: (row[1], row[2]))


def _inside_supervision_margin(
    candidates: list[tuple[float, int, int]],
    supervision: np.ndarray,
    *,
    margin: int,
) -> list[tuple[float, int, int]]:
    """Filter sparse candidate centers without a full-size distance transform."""

    kept: list[tuple[float, int, int]] = []
    for score, y, x in candidates:
        y0, y1 = y - margin, y + margin + 1
        x0, x1 = x - margin, x + margin + 1
        if (
            y0 >= 0
            and x0 >= 0
            and y1 <= supervision.shape[0]
            and x1 <= supervision.shape[1]
            and bool(np.all(supervision[y0:y1, x0:x1]))
        ):
            kept.append((score, y, x))
    return kept


def _confound_candidates(
    label: np.ndarray,
    supervision: np.ndarray,
    ct: np.ndarray,
) -> list[tuple[float, int, int]]:
    supported_non_ink = supervision & ~label
    supported_non_ink &= ndimage.distance_transform_edt(~label) >= 2.0
    supported_non_ink &= ndimage.distance_transform_edt(supervision) >= 2.0
    gradient = np.hypot(ndimage.sobel(ct, axis=0), ndimage.sobel(ct, axis=1))
    # Rank every supervised non-ink location.  Requiring a 3x3 local maximum
    # created a domain-specific sampling hole: in PHerc0814 all eligible
    # maxima fell into only two of five frozen regions after V1/V2 exclusions.
    # Spatial independence is enforced below by _spaced and, across benchmark
    # versions, by the much larger level-0 exclusion radius.
    ys, xs = np.nonzero(supported_non_ink)
    return sorted(
        (
            (float(gradient[y, x]), int(y), int(x))
            for y, x in zip(ys, xs, strict=True)
        ),
        key=lambda row: (-row[0], row[1], row[2]),
    )


def _region_partition(
    mask: np.ndarray,
    partitions: int,
) -> tuple[bool, int, int, int]:
    if partitions == 1:
        return True, 0, 1, 1
    occupied_y = np.flatnonzero(np.any(mask, axis=1))
    occupied_x = np.flatnonzero(np.any(mask, axis=0))
    if not len(occupied_y):
        raise RuntimeError("cannot partition an empty supervision mask")
    use_y = int(occupied_y[-1] - occupied_y[0]) >= int(
        occupied_x[-1] - occupied_x[0]
    )
    lower = int(occupied_y[0]) if use_y else int(occupied_x[0])
    upper = (
        int(occupied_y[-1]) + 1 if use_y else int(occupied_x[-1]) + 1
    )
    return use_y, lower, upper, partitions


def _region_index(
    y: int,
    x: int,
    partition: tuple[bool, int, int, int],
) -> int:
    use_y, lower, upper, partitions = partition
    if partitions == 1:
        return 0
    coordinate = y if use_y else x
    width = max(1, upper - lower)
    return min(partitions - 1, max(0, (coordinate - lower) * partitions // width))


def select_controls(
    plan: dict[str, Any],
    loader: SurfaceLoader,
    *,
    target_per_class: int = TARGET_PER_CLASS_PER_SCROLL,
    excluded_controls: list[dict[str, Any]] | None = None,
    exclusion_radius_level0: float = DEFAULT_EXCLUSION_RADIUS_LEVEL0_PIXELS,
    benchmark_id: str | None = None,
) -> dict[str, Any]:
    if plan.get("status") not in {
        "OFFICIAL_LABEL_SOURCES_READY",
        "OFFICIAL_LABEL_AND_SEMANTIC_SOURCES_READY",
    }:
        raise RuntimeError("official source plan is not ready")
    ready_by_scroll: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for surface in plan["surfaces"]:
        if surface["status"] == "READY":
            ready_by_scroll[str(surface["scroll_id"])].append(surface)
    exclusions_by_surface: dict[tuple[str, str], list[tuple[int, int]]] = (
        defaultdict(list)
    )
    for row in excluded_controls or []:
        key = (str(row["scroll_id"]), str(row["official_surface_id"]))
        y, x = map(int, row["surface_y_x_level0"])
        exclusions_by_surface[key].append((y, x))

    selected_benchmark_id = benchmark_id or str(plan["benchmark_id"])
    component_prefix = (
        ""
        if selected_benchmark_id == "MULTISCROLL_TRANSFER_V1"
        else f"{selected_benchmark_id}:"
    )
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    by_scroll: dict[str, dict[str, Any]] = {}
    for scroll_id in sorted(ready_by_scroll):
        surfaces = sorted(
            ready_by_scroll[scroll_id],
            key=lambda row: (
                bool(
                    exclusions_by_surface[
                        (
                            str(row["scroll_id"]),
                            str(row["official_surface_id"]),
                        )
                    ]
                ),
                sum(
                    int(row["assets"][axis].get("size_bytes", 0))
                    for axis in ("tifxyz_x", "tifxyz_y", "tifxyz_z")
                ),
                str(row["official_surface_id"]),
            ),
        )
        selected_surfaces: list[dict[str, Any]] = []
        groups = 0
        for surface in surfaces:
            selected_surfaces.append(surface)
            groups += int(surface["region_partition_count"])
            if groups >= TARGET_GROUPS_PER_SCROLL:
                break

        candidates_by_class_group: dict[
            str, dict[str, list[tuple[float, int, int, dict[str, Any], dict[str, Any]]]]
        ] = {
            "POSITIVE": defaultdict(list),
            "CONFOUND": defaultdict(list),
        }
        for surface in selected_surfaces:
            loaded = loader.load(surface)
            label_level0 = loaded["label_level0"]
            supervision_level0 = loaded["supervision_level0"]
            label_reduced = loaded["label_reduced"]
            supervision_reduced = loaded["supervision_reduced"]
            if (
                label_level0.shape != supervision_level0.shape
                or label_reduced.shape != supervision_reduced.shape
                or label_reduced.shape != loaded["ct"].shape
            ):
                raise RuntimeError(
                    f"unaligned level arrays for {surface['official_surface_id']}"
                )
            partitions = int(surface["region_partition_count"])
            for expected_class, candidates, selection_level, region_mask in (
                (
                    "POSITIVE",
                    _outside_frozen_control_exclusions(
                        _inside_supervision_margin(
                            _positive_candidates(
                                label_level0 & supervision_level0
                            ),
                            supervision_level0,
                            margin=SUPERVISION_MARGIN_LEVEL0_PIXELS,
                        ),
                        selection_level=0,
                        excluded_level0_y_x=exclusions_by_surface[
                            (
                                str(surface["scroll_id"]),
                                str(surface["official_surface_id"]),
                            )
                        ],
                        minimum_distance_level0=exclusion_radius_level0,
                    ),
                    0,
                    supervision_level0,
                ),
                (
                    "CONFOUND",
                    _outside_frozen_control_exclusions(
                        _confound_candidates(
                            label_reduced,
                            supervision_reduced,
                            loaded["ct"],
                        ),
                        selection_level=PYRAMID_LEVEL,
                        excluded_level0_y_x=exclusions_by_surface[
                            (
                                str(surface["scroll_id"]),
                                str(surface["official_surface_id"]),
                            )
                        ],
                        minimum_distance_level0=exclusion_radius_level0,
                    ),
                    PYRAMID_LEVEL,
                    supervision_reduced,
                ),
            ):
                partition = _region_partition(region_mask, partitions)
                for score, y, x in candidates:
                    region = _region_index(y, x, partition)
                    group_id = (
                        f"{scroll_id}:{surface['official_surface_id']}:region-{region}"
                    )
                    candidates_by_class_group[expected_class][group_id].append(
                        (score, y, x, surface, loaded, selection_level)
                    )
            receipts.append(
                {
                    "scroll_id": scroll_id,
                    "official_surface_id": surface["official_surface_id"],
                    "shapes": loaded["shapes"],
                    "mask_alignment": loaded["mask_alignment"],
                    "central_slice": loaded["central_slice"],
                    "assets": loaded["assets"],
                }
            )

        selected_by_class: dict[str, list[dict[str, Any]]] = {}
        for expected_class in ("POSITIVE", "CONFOUND"):
            group_map = candidates_by_class_group[expected_class]
            group_ids = sorted(group_map)
            per_group = int(math.ceil(target_per_class / max(1, len(group_ids))))
            selected: list[
                tuple[
                    float,
                    int,
                    int,
                    dict[str, Any],
                    dict[str, Any],
                    int,
                    str,
                ]
            ] = []
            for group_id in group_ids:
                raw = group_map[group_id]
                selection_level = int(raw[0][5])
                spaced = _spaced(
                    [(score, y, x) for score, y, x, _, _, _ in raw],
                    limit=per_group,
                    minimum_distance=(
                        32.0
                        if selection_level == 0
                        else CONFOUND_WITHIN_BENCHMARK_SPACING_LEVEL5_PIXELS
                    ),
                )
                lookup = {
                    (score, y, x): (surface, loaded, level)
                    for score, y, x, surface, loaded, level in raw
                }
                selected.extend(
                    (*candidate, *lookup[candidate], group_id) for candidate in spaced
                )
            # A sparse region must not lower the scroll-wide quota when other
            # prospectively frozen regions contain eligible controls.  Fill
            # the shortfall deterministically while retaining the same
            # within-surface spacing and the V1 exclusion already applied
            # above.  This changes allocation only; it never reads gate output.
            if len(selected) < target_per_class:
                selected_keys = {
                    (
                        str(row[3]["official_surface_id"]),
                        int(row[1]),
                        int(row[2]),
                    )
                    for row in selected
                }
                remaining = sorted(
                    (
                        (score, y, x, surface, loaded, int(level), group_id)
                        for group_id in group_ids
                        for score, y, x, surface, loaded, level in group_map[group_id]
                        if (
                            str(surface["official_surface_id"]),
                            int(y),
                            int(x),
                        )
                        not in selected_keys
                    ),
                    key=lambda row: (
                        row[6],
                        -row[0],
                        row[1],
                        row[2],
                    ),
                )
                for candidate in remaining:
                    _, y, x, surface, _, level, _ = candidate
                    minimum_distance = (
                        32.0
                        if level == 0
                        else CONFOUND_WITHIN_BENCHMARK_SPACING_LEVEL5_PIXELS
                    )
                    minimum_squared = minimum_distance * minimum_distance
                    if all(
                        str(other[3]["official_surface_id"])
                        != str(surface["official_surface_id"])
                        or (y - other[1]) ** 2 + (x - other[2]) ** 2
                        >= minimum_squared
                        for other in selected
                    ):
                        selected.append(candidate)
                        if len(selected) == target_per_class:
                            break
            selected.sort(
                key=lambda row: (
                    row[5],
                    -row[0],
                    row[1],
                    row[2],
                )
            )
            selected = selected[:target_per_class]
            payloads: list[dict[str, Any]] = []
            for index, (
                _,
                y,
                x,
                surface,
                loaded,
                selection_level,
                group_id,
            ) in enumerate(selected):
                scale = 2**selection_level
                full_y = int(y * scale + scale // 2)
                full_x = int(x * scale + scale // 2)
                label_assets = [
                    {
                        "uri": asset["uri"],
                        "sha256": asset["sha256"],
                        "role": asset["role"],
                    }
                    for asset in loaded["assets"].values()
                ]
                payload = {
                    "scroll_id": scroll_id,
                    "surface_group_id": group_id,
                    "component_id": (
                        f"{component_prefix}{scroll_id}-"
                        f"{expected_class.lower()}-{index:03d}"
                    ),
                    "expected_class": expected_class,
                    "confound_subtype": (
                        "OTHER_ADJUDICATED_NON_INK"
                        if expected_class == "CONFOUND"
                        else None
                    ),
                    "official_surface_id": surface["official_surface_id"],
                    "surface_y_x_level0": [full_y, full_x],
                    "selection_level": selection_level,
                    "selection_y_x": [y, x],
                    "label_source": {
                        "label_authority": "PUBLIC_CURATED_SURFACE_LABEL",
                        "coordinate_frame_id": (
                            f"{scroll_id}:{surface['official_surface_id']}:surface-canvas"
                        ),
                        "prediction_used_as_ground_truth": False,
                        "assets": label_assets,
                    },
                    "selection_policy": (
                        "SPATIALLY_SEPARATED_CURATED_INK_WINDOW"
                        if expected_class == "POSITIVE"
                        else "HIGH_CT_GRADIENT_CURATED_NON_INK"
                    ),
                    "v3_v4_outputs_read_during_selection": False,
                }
                if payload["confound_subtype"] is None:
                    payload.pop("confound_subtype")
                payloads.append(payload)
            selected_by_class[expected_class] = payloads
            rows.extend(payloads)

        counts = {
            label: len(selected_by_class[label])
            for label in ("POSITIVE", "CONFOUND")
        }
        used_groups = {
            label: len(
                {row["surface_group_id"] for row in selected_by_class[label]}
            )
            for label in ("POSITIVE", "CONFOUND")
        }
        by_scroll[scroll_id] = {
            "counts": counts,
            "surface_groups": used_groups,
            "ready": (
                all(counts[label] >= target_per_class for label in counts)
                and all(
                    used_groups[label] >= TARGET_GROUPS_PER_SCROLL
                    for label in used_groups
                )
            ),
        }

    status = (
        "OFFICIAL_CONTROL_CANDIDATES_READY_FOR_LEVEL0_MATERIALIZATION"
        if all(payload["ready"] for payload in by_scroll.values())
        else "BLOCKED_INSUFFICIENT_OFFICIAL_CONTROLS"
    )
    return {
        "schema": "campaignx.multiscroll_official_control_candidates.v1",
        "benchmark_id": selected_benchmark_id,
        "status": status,
        "generated_at_utc": utc_now(),
        "source_plan_sha256": plan["content_sha256"],
        "selection_policy": {
            "pyramid_level": PYRAMID_LEVEL,
            "surface_scale": LEVEL_SCALE,
            "target_per_class_per_scroll": target_per_class,
            "minimum_groups_per_scroll": TARGET_GROUPS_PER_SCROLL,
            "positive_supervision_margin_level0_pixels": (
                SUPERVISION_MARGIN_LEVEL0_PIXELS
            ),
            "confound_within_benchmark_spacing_level5_pixels": (
                CONFOUND_WITHIN_BENCHMARK_SPACING_LEVEL5_PIXELS
            ),
            "v3_v4_outputs_visible": False,
            "predictions_used_as_ground_truth": False,
            "excluded_frozen_control_count": len(excluded_controls or []),
            "exclusion_radius_level0_pixels": (
                exclusion_radius_level0 if excluded_controls else None
            ),
            "audit_patch_non_overlap_required": bool(excluded_controls),
        },
        "by_scroll": by_scroll,
        "source_receipts": receipts,
        "controls": sorted(
            rows,
            key=lambda row: (
                row["scroll_id"],
                row["expected_class"],
                row["component_id"],
            ),
        ),
        "non_claims": [
            "Controls are not frozen until level-0 CT and TIFXYZ provenance pass.",
            "The CT gradient only locates hard negatives.",
            "No gate result influenced selection.",
            "No ink, text, letters, or First Letters are accepted automatically.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-per-class", type=int, default=50)
    parser.add_argument(
        "--benchmark-id",
        choices=[
            "MULTISCROLL_TRANSFER_V1",
            "MULTISCROLL_TRANSFER_V2",
            "MULTISCROLL_TRANSFER_V3",
            "CT_PRIORITY_ROUTER_V43_DEVELOPMENT",
            "MULTISCROLL_TRANSFER_V4",
            "MULTISCROLL_TRANSFER_V5",
            "MULTISCROLL_TRANSFER_V6",
            "SURFACE_CALIBRATION_TRANSFER_V7",
            "SURFACE_CALIBRATION_TRANSFER_V8",
        ],
        help="Override the benchmark identity while preserving source-plan provenance.",
    )
    parser.add_argument(
        "--exclude-controls",
        type=Path,
        action="append",
        help="Previously frozen controls whose level-0 audit patches must not overlap.",
    )
    parser.add_argument(
        "--exclusion-radius-level0",
        type=float,
        default=DEFAULT_EXCLUSION_RADIUS_LEVEL0_PIXELS,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite official control candidates")
    plan = json.loads(args.source_plan.resolve().read_text(encoding="utf-8"))
    excluded_controls: list[dict[str, Any]] = []
    for exclusion_path in args.exclude_controls or []:
        exclusion_payload = json.loads(
            exclusion_path.resolve().read_text(encoding="utf-8")
        )
        excluded_controls.extend(
            exclusion_payload.get("controls", [])
            if isinstance(exclusion_payload, dict)
            else exclusion_payload
        )
    result = select_controls(
        plan,
        HfLevelLoader(args.mirror_root),
        target_per_class=args.target_per_class,
        excluded_controls=excluded_controls,
        exclusion_radius_level0=args.exclusion_radius_level0,
        benchmark_id=args.benchmark_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["by_scroll"], indent=2, sort_keys=True))
    print(result["status"])
    return (
        0
        if result["status"]
        == "OFFICIAL_CONTROL_CANDIDATES_READY_FOR_LEVEL0_MATERIALIZATION"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
