from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from .common import content_sha256, utc_now


class CtSupportSourceUnavailable(RuntimeError):
    """The frozen CT source could not be sampled for an ink-blind gate."""


class CtSupportSampler(Protocol):
    def sample(
        self,
        ct_uri: str,
        coordinate_xyz: dict[str, int],
        *,
        level: int,
        radius_l0_voxels: int,
    ) -> dict[str, Any]: ...


@dataclass
class OmeZarrCtSupportSampler:
    """Sample a small, downscaled OME-Zarr cube near an m7 seed.

    The gate only asks whether scanned material exists near a proposed seed.
    It never reads an ink prediction, OCR output or downstream result.  A
    coarse OME level keeps this check much cheaper than an otherwise wasted
    VC3D grow and 65-slice render.
    """

    def sample(
        self,
        ct_uri: str,
        coordinate_xyz: dict[str, int],
        *,
        level: int,
        radius_l0_voxels: int,
    ) -> dict[str, Any]:
        try:
            import numpy as np
            import zarr

            group = zarr.open_group(ct_uri, mode="r")
            multiscales = group.attrs.get("multiscales")
            if not isinstance(multiscales, list) or not multiscales:
                raise ValueError("CT source has no OME multiscales metadata")
            datasets = multiscales[0].get("datasets")
            if not isinstance(datasets, list):
                raise ValueError("CT source has no OME dataset list")
            dataset = next((row for row in datasets if str(row.get("path")) == str(level)), None)
            if dataset is None:
                raise ValueError(f"CT source has no OME level {level}")
            transforms = dataset.get("coordinateTransformations")
            scale_row = next(
                (row for row in transforms or [] if row.get("type") == "scale"),
                None,
            )
            scale_zyx = scale_row.get("scale") if isinstance(scale_row, dict) else None
            if not (
                isinstance(scale_zyx, list)
                and len(scale_zyx) == 3
                and all(float(value) > 0 for value in scale_zyx)
            ):
                raise ValueError("CT OME level has no valid z/y/x scale")
            array = group[str(level)]
            coordinate_zyx = [int(coordinate_xyz[axis]) for axis in "zyx"]
            center_zyx = [
                int(math.floor(value / float(scale)))
                for value, scale in zip(coordinate_zyx, scale_zyx, strict=True)
            ]
            radius_zyx = [
                max(1, int(math.ceil(radius_l0_voxels / float(scale))))
                for scale in scale_zyx
            ]
            slices = tuple(
                slice(max(0, center - radius), min(int(size), center + radius + 1))
                for center, radius, size in zip(center_zyx, radius_zyx, array.shape, strict=True)
            )
            values = np.asarray(array[slices])
            if values.size == 0:
                raise ValueError("CT support sample is empty")
            nonzero = int(np.count_nonzero(values))
            return {
                "level": int(level),
                "scale_zyx": [float(value) for value in scale_zyx],
                "center_zyx": center_zyx,
                "radius_zyx": radius_zyx,
                "shape_zyx": [int(value) for value in values.shape],
                "voxel_count": int(values.size),
                "nonzero_voxel_count": nonzero,
                "nonzero_fraction": float(nonzero / values.size),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std()),
                "maximum": float(values.max()),
            }
        except CtSupportSourceUnavailable:
            raise
        except BaseException as error:
            raise CtSupportSourceUnavailable(
                f"could not sample frozen CT material support: {type(error).__name__}: {error}"
            ) from error


def apply_ct_material_support_gate(
    response: dict[str, Any],
    task: dict[str, Any],
    sampler: CtSupportSampler | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply an optional, frozen CT-material gate to final m7 candidates."""

    discovery = task.get("candidate_discovery", {})
    config = discovery.get("ct_material_support_gate")
    raw = response.get("candidates")
    if not isinstance(raw, list):
        raise CtSupportSourceUnavailable("m7 response has no candidate array for CT support")
    if config is None:
        return response, {
            "schema": "campaignx.ct_material_support_screen.v1",
            "status": "NOT_CONFIGURED",
            "input_candidate_count": len(raw),
            "retained_candidate_count": len(raw),
            "ink_used": False,
        }
    if not isinstance(config, dict):
        raise ValueError("ct_material_support_gate must be an object")
    policy = str(config.get("policy", ""))
    if policy != "ome-zarr-nearby-material-v1":
        raise ValueError(f"unsupported CT material support policy: {policy}")
    level = int(config.get("level", 5))
    radius = int(config.get("radius_l0_voxels", 192))
    minimum_nonzero = int(config.get("minimum_nonzero_voxels", 1))
    if level < 0 or radius < 1 or minimum_nonzero < 1:
        raise ValueError("invalid CT material support gate parameters")
    ct_uri = str(task.get("source", {}).get("ct_uri") or "")
    if not ct_uri:
        raise ValueError("task source has no CT URI")
    active_sampler = sampler or OmeZarrCtSupportSampler()
    retained: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw, start=1):
        coordinate = (
            candidate.get("ct_l0_coordinate")
            or candidate.get("coordinate")
            or candidate.get("selected_seed")
            or candidate
        )
        if not isinstance(coordinate, dict) or any(axis not in coordinate for axis in "xyz"):
            assessments.append({
                "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or f"c{index:02d}"),
                "status": "REJECTED_MALFORMED_COORDINATE",
            })
            continue
        coordinate_xyz = {axis: int(float(coordinate[axis])) for axis in "xyz"}
        sample = active_sampler.sample(
            ct_uri,
            coordinate_xyz,
            level=level,
            radius_l0_voxels=radius,
        )
        accepted = int(sample["nonzero_voxel_count"]) >= minimum_nonzero
        assessment = {
            "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or f"c{index:02d}"),
            "coordinate_xyz_l0": coordinate_xyz,
            "status": "CT_MATERIAL_NEARBY" if accepted else "NO_NEARBY_CT_MATERIAL",
            "sample": sample,
        }
        assessments.append(assessment)
        if accepted:
            retained.append({**candidate, "ct_material_support": assessment})
    filtered = {**response, "candidates": retained}
    receipt = {
        "schema": "campaignx.ct_material_support_screen.v1",
        "status": "COMPLETED_INK_BLIND",
        "generated_at_utc": utc_now(),
        "policy": policy,
        "config": {
            "level": level,
            "radius_l0_voxels": radius,
            "minimum_nonzero_voxels": minimum_nonzero,
        },
        "source_snapshot_id": task.get("source_snapshot_id"),
        "input_candidate_count": len(raw),
        "retained_candidate_count": len(retained),
        "rejected_candidate_count": len(raw) - len(retained),
        "assessments": assessments,
        "filtered_response_sha256": content_sha256(filtered),
        "ink_used": False,
        "non_claim": "Nearby CT material is a grow-efficiency gate, not geometry acceptance or ink evidence.",
    }
    return filtered, receipt
