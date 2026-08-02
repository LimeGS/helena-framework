"""FIX-10.5 — analytic cm2 area through both surviving formulas.

WHAT THIS COVERS
----------------
The repository computes usable surface area in cm2 in two independent places
with two differently-shaped formulas, and neither was ever checked against a
mesh of known analytic area:

* ``framework/stages/01-segmentation/backends/scrollfiesta/topology.py:67``
  ``area_cm2 = triangle_areas_um2.sum() * 1e-8``
  (vertices are scaled by a **per-axis** ``voxel_size_um_xyz`` first)
* ``framework/stages/04-validation/scripts/helena_measure_tifxyz_agreement.py:72``
  ``area_cm2 = areas.sum() * voxel_um * voxel_um / 1e8``
  (triangle areas stay in voxel^2 and are scaled by a **single scalar**
  ``voxel_um`` squared)

The reference object is a flat 3x3 TIFXYZ raster == 2x2 quads == 8 triangles at
10 um voxels, whose analytic area is 4 * (10 um)^2 = 400 um^2 = 4e-6 cm^2.
Tilted and larger meshes are checked too, so the test constrains the geometry
and not just the constant.

``tests/test_helena_measure_tifxyz_agreement.py:66`` asserts
``triangle_count_in_roi == 8`` on the same fixture but never looks at
``usable_area_cm2_in_roi``; that value is asserted here.

WHAT THIS DOES *NOT* COVER
--------------------------
Area *deduplication* across overlapping surfaces (audit 3.5, T-5) is a separate
gap: ``campaign_gross_area_cm2`` remains a direct sum.  Nothing here measures
that.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "framework/stages/01-segmentation/backends"
MEASURE_SCRIPT = (
    ROOT / "framework/stages/04-validation/scripts/helena_measure_tifxyz_agreement.py"
)

for _path in (str(ROOT), str(BACKENDS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scrollfiesta.coordinate_transform import ObjMesh  # noqa: E402
from scrollfiesta.topology import topology_metrics  # noqa: E402

# Generous ROI: every synthetic vertex lies strictly inside it, so no test
# here is ever measuring an ROI clip instead of the area formula.
ROI = [-1_000_000, -1_000_000, -1_000_000, 1_000_000, 1_000_000, 1_000_000]
UM2_PER_CM2 = 1e8


def _load_measure():
    spec = importlib.util.spec_from_file_location(
        "helena_measure_tifxyz_agreement_analytic_area", MEASURE_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MEASURE = _load_measure()


def grid(rows: int, columns: int, height: "callable") -> tuple[np.ndarray, ...]:
    yy, xx = np.mgrid[0:rows, 0:columns]
    zz = height(xx.astype(np.float64), yy.astype(np.float64))
    return xx.astype(np.float64), yy.astype(np.float64), np.asarray(zz, dtype=np.float64)


def quad_mesh(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> ObjMesh:
    """Two triangles per quad, matching the TIFXYZ rasteriser's split."""

    rows, columns = xx.shape
    vertices = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    faces: list[list[int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            top_left = row * columns + column
            top_right = top_left + 1
            bottom_left = (row + 1) * columns + column
            bottom_right = bottom_left + 1
            faces.append([top_left, bottom_left, top_right])
            faces.append([bottom_right, top_right, bottom_left])
    return ObjMesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        vertex_trailing_fields=(),
        passthrough_lines=(),
        source_triangle_count=len(faces),
        dropped_degenerate_triangle_count=0,
    )


def topology_area_cm2(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    voxel_um_xyz: tuple[float, float, float],
) -> float:
    return topology_metrics(
        quad_mesh(xx, yy, zz), voxel_size_um_xyz=voxel_um_xyz
    )["area_cm2"]


def tifxyz_summary(
    directory: Path,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    voxel_um: float,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    for axis, values in (("x", xx), ("y", yy), ("z", zz)):
        tifffile.imwrite(directory / f"{axis}.tif", values.astype(np.float32))
    (directory / "meta.json").write_text(
        json.dumps({"area_cm2": None}), encoding="utf-8"
    )
    return MEASURE.summarize_surface(directory, ROI, voxel_um)["summary"]


# ---------------------------------------------------------------------------
# The specified reference object
# ---------------------------------------------------------------------------


def test_flat_three_by_three_raster_matches_the_analytic_area(tmp_path: Path):
    """3x3 raster -> 2x2 quads -> 4 * 100 um^2 = 4e-6 cm^2, both formulas."""

    xx, yy, zz = grid(3, 3, lambda x, y: np.zeros_like(x))
    analytic_cm2 = 4 * (10.0 * 10.0) / UM2_PER_CM2
    assert analytic_cm2 == 4e-6

    topology = topology_area_cm2(xx, yy, zz, (10.0, 10.0, 10.0))
    summary = tifxyz_summary(tmp_path / "flat", xx, yy, zz, 10.0)

    assert summary["triangle_count_in_roi"] == 8
    assert summary["usable_area_cm2_in_roi"] == pytest.approx(analytic_cm2, rel=1e-12)
    assert topology == pytest.approx(analytic_cm2, rel=1e-12)
    assert topology == pytest.approx(
        summary["usable_area_cm2_in_roi"], rel=1e-12
    )


@pytest.mark.parametrize("voxel_um", [1.0, 8.64, 9.362, 10.0, 53.0])
def test_both_formulas_agree_at_every_campaign_voxel_size(
    tmp_path: Path, voxel_um: float
):
    xx, yy, zz = grid(5, 7, lambda x, y: np.zeros_like(x))
    quads = (5 - 1) * (7 - 1)
    analytic_cm2 = quads * voxel_um * voxel_um / UM2_PER_CM2

    topology = topology_area_cm2(xx, yy, zz, (voxel_um, voxel_um, voxel_um))
    summary = tifxyz_summary(
        tmp_path / f"flat-{voxel_um}", xx, yy, zz, voxel_um
    )

    assert summary["triangle_count_in_roi"] == 2 * quads
    assert topology == pytest.approx(analytic_cm2, rel=1e-12)
    assert summary["usable_area_cm2_in_roi"] == pytest.approx(
        analytic_cm2, rel=1e-12
    )


def test_a_forty_five_degree_tilt_scales_both_formulas_by_sqrt_two(tmp_path: Path):
    """A plane tilted 45 deg has sqrt(2) the projected area: not a constant test."""

    xx, yy, zz = grid(3, 3, lambda x, y: x)
    analytic_cm2 = 4 * (10.0 * 10.0) * math.sqrt(2.0) / UM2_PER_CM2

    topology = topology_area_cm2(xx, yy, zz, (10.0, 10.0, 10.0))
    summary = tifxyz_summary(tmp_path / "tilt", xx, yy, zz, 10.0)

    assert topology == pytest.approx(analytic_cm2, rel=1e-12)
    assert summary["usable_area_cm2_in_roi"] == pytest.approx(
        analytic_cm2, rel=1e-12
    )


def test_a_one_square_centimetre_sheet_reports_one_square_centimetre(tmp_path: Path):
    """End-to-end sanity on the physical unit itself, at Helena Framework scale.

    1 cm = 10000 um = 1157.407... voxels at 8.64 um.  A 101x101 raster of
    100x100 quads with that pitch is exactly 1 cm on a side.
    """

    voxel_um = 8.64
    pitch = (10000.0 / voxel_um) / 100.0
    xx, yy, zz = grid(101, 101, lambda x, y: np.zeros_like(x))
    xx, yy = xx * pitch, yy * pitch

    topology = topology_area_cm2(xx, yy, zz, (voxel_um, voxel_um, voxel_um))
    summary = tifxyz_summary(tmp_path / "one-cm2", xx, yy, zz, voxel_um)

    assert topology == pytest.approx(1.0, rel=1e-6)
    # float32 TIFXYZ storage costs a little precision; 1e-5 relative is ample.
    assert summary["usable_area_cm2_in_roi"] == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Where the two formulas stop being equivalent
# ---------------------------------------------------------------------------


def test_anisotropic_voxels_are_a_real_divergence_between_the_two_formulas(
    tmp_path: Path,
):
    """FINDING: only ``topology.py`` can express a non-cubic voxel.

    ``topology_metrics`` scales each vertex axis independently, so a
    (10, 20, 10) um voxel doubles the area of a y-extended quad -- the correct
    answer, 8e-6 cm^2.  ``helena_measure_tifxyz_agreement`` accepts a single
    scalar ``--voxel-size-um`` and squares it, so it reports 4e-6 cm^2: a
    factor-2 understatement with no way to express the anisotropy.

    Helena Framework scans are isotropic (8.64 / 9.362 um), so this does not affect
    any number currently reported.  It is pinned here because the two formulas
    are *not* interchangeable in general, and a future anisotropic source would
    silently halve the measured area on the validation side.
    """

    xx, yy, zz = grid(3, 3, lambda x, y: np.zeros_like(x))

    anisotropic_topology = topology_area_cm2(xx, yy, zz, (10.0, 20.0, 10.0))
    scalar_measure = tifxyz_summary(tmp_path / "aniso", xx, yy, zz, 10.0)[
        "usable_area_cm2_in_roi"
    ]

    assert anisotropic_topology == pytest.approx(8e-6, rel=1e-12)
    assert scalar_measure == pytest.approx(4e-6, rel=1e-12)
    assert anisotropic_topology == pytest.approx(2.0 * scalar_measure, rel=1e-12)


def test_topology_rejects_a_non_positive_voxel_size():
    xx, yy, zz = grid(3, 3, lambda x, y: np.zeros_like(x))
    with pytest.raises(ValueError, match="three finite positive values"):
        topology_area_cm2(xx, yy, zz, (10.0, 0.0, 10.0))


def test_degenerate_triangles_contribute_no_area(tmp_path: Path):
    """A raster collapsed onto a line has zero area under both formulas."""

    xx, yy, zz = grid(3, 3, lambda x, y: np.zeros_like(x))
    flattened = np.zeros_like(yy)

    assert topology_area_cm2(xx, flattened, zz, (10.0, 10.0, 10.0)) == 0.0
    summary = tifxyz_summary(tmp_path / "degenerate", xx, flattened, zz, 10.0)
    assert summary["usable_area_cm2_in_roi"] == 0.0
    assert summary["triangle_count_in_roi"] == 0
