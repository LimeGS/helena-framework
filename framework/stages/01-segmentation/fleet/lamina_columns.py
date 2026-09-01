"""Reading the CT along a surface's own normal, one column per sampled cell.

`lamina.py` is the arithmetic and knows nothing about volumes. This is the half
that touches bytes: it takes a TIFXYZ -- x, y and z of CT coordinates per grid
cell -- works out where the sheet's normal points, and reads a short column of
the volume through each sampled cell.

Two things it refuses to guess.

A cell whose column runs off the edge of what the volume returned is marked
missing rather than padded: a hole reads as air, and air is exactly what a
thickness measurement looks for. And a cell whose neighbours carry no
coordinates has no normal to speak of, so it is not sampled at all rather than
sampled along some default direction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .lamina import (
    assess_lamina, column_material_depth, histogram_bimodality, interface_level,
)


def _unit(vector: list[float]) -> list[float] | None:
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or length <= 0:
        return None
    return [component / length for component in vector]


def cell_normal(x, y, z, row: int, column: int) -> list[float] | None:
    """The normal at one grid cell, from its neighbours' coordinates.

    Central differences along the two grid directions and their cross product.
    A cell on the border, or one whose neighbours hold no coordinates, returns
    None: a surface with a hole beside it has no local plane, and sampling it
    along a made-up direction would produce a profile that measures nothing.
    """
    height, width = x.shape
    if not (0 < row < height - 1 and 0 < column < width - 1):
        return None
    def point(r: int, c: int) -> list[float] | None:
        coordinates = [float(x[r, c]), float(y[r, c]), float(z[r, c])]
        if any(not math.isfinite(value) for value in coordinates):
            return None
        # (0,0,0) is how a TIFXYZ says "no coordinate here", not the origin of
        # the scroll: the volume starts at 0 and no lamina passes through it.
        if all(value == 0.0 for value in coordinates):
            return None
        return coordinates

    left, right = point(row, column - 1), point(row, column + 1)
    up, down = point(row - 1, column), point(row + 1, column)
    if None in (left, right, up, down):
        return None
    du = [right[i] - left[i] for i in range(3)]
    dv = [down[i] - up[i] for i in range(3)]
    cross = [du[1] * dv[2] - du[2] * dv[1],
             du[2] * dv[0] - du[0] * dv[2],
             du[0] * dv[1] - du[1] * dv[0]]
    return _unit(cross)


def _trilinear(values, origin_zyx: list[int], point_zyx: list[float]):
    """One sample inside a cube, or None when it falls outside it."""
    local = [point_zyx[axis] - origin_zyx[axis] for axis in range(3)]
    floors = [math.floor(value) for value in local]
    if any(index < 0 or index + 1 >= values.shape[axis]
           for axis, index in enumerate(floors)):
        return None
    weights = [local[axis] - floors[axis] for axis in range(3)]
    total = 0.0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                weight = ((weights[0] if dz else 1 - weights[0])
                          * (weights[1] if dy else 1 - weights[1])
                          * (weights[2] if dx else 1 - weights[2]))
                if weight == 0.0:
                    continue
                total += weight * float(
                    values[floors[0] + dz, floors[1] + dy, floors[2] + dx])
    return total


def sample_columns(
    surface_directory: Path,
    *,
    ct_uri: str,
    voxel_size_um: float,
    profile: dict[str, Any],
    sampler,
    level: int = 0,
) -> dict[str, Any]:
    """Read the volume along the normal at a sample of this surface's cells.

    Returns what `assess_lamina` needs plus the read set of every object the
    volume answered with, so a verdict can name the bytes it was measured from.
    """
    import numpy as np
    import tifffile

    sampling = profile["sampling"]
    wanted = int(sampling["columns"])
    per_column = int(sampling["samples_per_column"])
    step_um = float(sampling["sample_step_um"])
    voxel_um = float(voxel_size_um)
    if voxel_um <= 0:
        raise ValueError("a lamina column needs the volume's voxel size in microns")

    x = tifffile.imread(surface_directory / "x.tif")
    y = tifffile.imread(surface_directory / "y.tif")
    z = tifffile.imread(surface_directory / "z.tif")
    if not (x.shape == y.shape == z.shape) or x.ndim != 2:
        raise ValueError("a TIFXYZ is three matching 2-D coordinate planes")

    height, width = x.shape
    # A regular lattice rather than the first N cells: a surface is a sheet and
    # its first rows are one edge of it, which is where the coordinates are
    # worst. The step is chosen so the lattice covers the whole patch.
    stride = max(1, int(math.sqrt((height * width) / max(1, wanted))))
    span_voxels = (per_column * step_um) / voxel_um
    radius = int(math.ceil(span_voxels / 2.0)) + 2

    profiles: list[tuple[list[float], list[bool]]] = []
    intensities: list[float] = []
    read_objects: dict[str, dict[str, Any]] = {}
    skipped_without_normal = 0

    for row in range(stride, height - 1, stride):
        for column in range(stride, width - 1, stride):
            if len(profiles) >= wanted:
                break
            normal = cell_normal(x, y, z, row, column)
            if normal is None:
                skipped_without_normal += 1
                continue
            centre = [float(z[row, column]), float(y[row, column]),
                      float(x[row, column])]
            cube = sampler.read_cube(
                ct_uri,
                {"x": int(x[row, column]), "y": int(y[row, column]),
                 "z": int(z[row, column])},
                level=level, radius_l0_voxels=radius)
            values = cube["values"]
            origin = cube["origin_zyx"]
            scale = [float(value) for value in cube["scale_zyx"]]
            for entry in cube["source_read_set"]["objects"]:
                read_objects[entry["object_key"]] = entry

            profile_values: list[float] = []
            missing: list[bool] = []
            for index in range(per_column):
                offset_um = (index - (per_column - 1) / 2.0) * step_um
                offset_voxels = offset_um / voxel_um
                point_zyx = [
                    (centre[0] + normal[2] * offset_voxels) / scale[0],
                    (centre[1] + normal[1] * offset_voxels) / scale[1],
                    (centre[2] + normal[0] * offset_voxels) / scale[2],
                ]
                sample = _trilinear(values, origin, point_zyx)
                missing.append(sample is None)
                profile_values.append(0.0 if sample is None else sample)
            intensities.extend(
                value for value, absent in zip(profile_values, missing)
                if not absent)
            profiles.append((profile_values, missing))

    # The level is measured over the whole surface and the columns are read
    # against it. Per column it cannot be: a column that lies entirely inside
    # the material has no two levels to be halfway between, and that column is
    # the fused case this gate exists to catch.
    crossing_level = interface_level(intensities)
    columns = [column_material_depth(values, sample_step_um=step_um,
                                     missing=missing, level=crossing_level)
               for values, missing in profiles]

    objects = [read_objects[key] for key in sorted(read_objects)]
    from .common import content_sha256  # noqa: PLC0415

    outcome = assess_lamina(
        columns, profile=profile,
        bimodality=histogram_bimodality(intensities),
        interface_level=crossing_level,
        level_was_measured=crossing_level is not None)
    return {
        **outcome,
        "cells_without_a_normal": skipped_without_normal,
        "sampling": {"columns": len(profiles), "samples_per_column": per_column,
                     "sample_step_um": step_um, "level": int(level),
                     "voxel_size_um": voxel_um},
        "source_read_set": {
            "schema": "campaignx.first_letters_source_read_set.v1",
            "objects": objects,
            "canonical_manifest_sha256": content_sha256(objects),
        },
    }
