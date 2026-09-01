"""The gate's thresholds were one corpus's numbers, applied as if universal.

`step_discontinuity_factor: 8.0` is applied to the mesh's global median edge,
and the source says why: "TIFXYZ grids are arc-length uniform by construction".
That is true of a seeded grow and false of a global fit. A grown surface's
longest edge is 7.1x its median -- against a factor of 8.0, almost no headroom,
which is the sign that the constant measures that corpus rather than a defect.
A fitted spiral winding's is 13x to 26x, and its local step ranges from 21 to 65
voxels across one strip, so hundreds of edges read as lamina steps.

`band_cells: 4` and `resolution_limit_voxels: 16.0` are the same thing in
different units: lengths written as counts. Four cells is 480 um of sheet on the
corpus this was calibrated on and 731 um on a fitted winding, so the same number
asks each of them a different question; sixteen voxels is the 150 um floor of an
inter-lamina range the gate never reads from the scroll it is measuring.

What this pins is that the rules became measurements without becoming knobs:

  * the grown corpus does not move, with or without a scale;
  * a supplied scale is used and an absent one is declared, so a verdict made
    without knowing the scroll is identifiable as such;
  * deriving is not loosening -- on a fitted winding the derived band is
    stricter than the frozen one, and the tests say so out loud.

The detectors themselves are untouched. Fold-back, near-coincident overlap,
interpenetration and exact self-intersection keep their thresholds, because a
policy that let a mesh past those would be switching the gate off exactly where
it is right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/04-validation/scripts"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

import helena_tifxyz_geometry_gate as gate  # noqa: E402

GROWN = ROOT / "vendor/villa/volume-cartographer/core/test/data/segments"
GROWN_VOXEL_UM = 9.362


def ramp(height: int = 40, width: int = 40, step: float = 10.0):
    """A flat patch sampled at a constant step, in CT-L0 coordinates."""
    rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    points = np.stack([
        100.0 + step * columns, 200.0 + step * rows,
        np.full((height, width), 300.0),
    ], axis=-1)
    return points, np.ones((height, width), dtype=bool)


# -- the step rule ---------------------------------------------------------

def test_a_uniform_grid_has_no_discontinuity_either_way():
    points, valid = ramp()
    for window in (0, 9):
        assert gate._step_discontinuities(points, valid, 10.0, 8.0,
                                          window_cells=window) == 0


def test_one_long_edge_in_a_fine_region_is_still_a_discontinuity():
    """The rule's whole point. A local median must not excuse a real jump."""
    points, valid = ramp()
    points[20, 20:, 0] += 400.0            # a 400-voxel step across one row

    assert gate._step_discontinuities(points, valid, 10.0, 8.0, window_cells=9) > 0


def test_a_coarsely_sampled_region_is_not_a_row_of_discontinuities():
    """The failure being corrected: half a mesh sampled four times more coarsely
    is a parametrisation, and against one global median it read as a jump at
    every step."""
    points, valid = ramp()
    points[:, 20:, 0] = 100.0 + 10.0 * 20 + 90.0 * (np.arange(20) + 1)[None, :]
    median = 10.0

    globally = gate._step_discontinuities(points, valid, median, 8.0, window_cells=0)
    locally = gate._step_discontinuities(points, valid, median, 8.0, window_cells=9)

    assert globally > 0, "the fixture does not reproduce the old finding"
    assert locally == 0, "a coarse region is still being read as a jump"


def test_a_window_of_one_is_the_published_global_rule():
    """A caller with no scale to derive a window from gets what shipped."""
    points, valid = ramp()
    points[:, 20:, 0] = 100.0 + 10.0 * 20 + 90.0 * (np.arange(20) + 1)[None, :]
    for window in (0, 1):
        assert (gate._step_discontinuities(points, valid, 10.0, 8.0, window_cells=window)
                == gate._step_discontinuities(points, valid, 10.0, 8.0))


# -- the derived lengths ---------------------------------------------------

def test_without_a_scale_the_frozen_counts_stand_and_the_receipt_says_so():
    step = {"median_edge_voxels": 12.94}
    derived, notes = gate.scale_derived_policy(
        gate.DEFAULT_POLICY, step, voxel_um=None, inter_lamina_um=None)

    assert derived["band_cells"] == gate.DEFAULT_POLICY["band_cells"]
    assert derived["resolution_limit_voxels"] == gate.DEFAULT_POLICY["resolution_limit_voxels"]
    assert "no voxel size" in notes["reason"]
    assert notes["derived"] == []


def test_the_grown_corpus_derives_back_to_its_own_frozen_numbers():
    """480 um is where `band_cells: 4` came from, so on that corpus the derived
    value has to be 4 again -- otherwise the length was mis-read, not derived."""
    derived, notes = gate.scale_derived_policy(
        gate.DEFAULT_POLICY, {"median_edge_voxels": 12.94},
        voxel_um=GROWN_VOXEL_UM, inter_lamina_um=None)

    assert derived["band_cells"] == 4
    assert {entry["name"] for entry in notes["derived"]} == {
        "band_cells", "step_window_cells", "resolution_limit_voxels"}


def test_deriving_the_band_on_a_finer_grid_is_stricter_not_looser():
    """Said out loud because it is the thing a reader will assume backwards: a
    fitted winding samples finer in microns, so the same 480 um of sheet is
    fewer cells, and fewer cells excludes less."""
    fit, _ = gate.scale_derived_policy(
        gate.DEFAULT_POLICY, {"median_edge_voxels": 23.12},
        voxel_um=7.91, inter_lamina_um=371.0)

    assert fit["band_cells"] == 3 < gate.DEFAULT_POLICY["band_cells"]


def test_a_measured_spacing_replaces_the_frozen_floor():
    """PHerc0826's own inter-lamina spacing is about 371 um; the frozen limit is
    the 150 um bottom of a range, and a 183 um step is limited against one and
    not the other."""
    frozen, frozen_notes = gate.scale_derived_policy(
        gate.DEFAULT_POLICY, {"median_edge_voxels": 23.12},
        voxel_um=7.91, inter_lamina_um=None)
    measured, notes = gate.scale_derived_policy(
        gate.DEFAULT_POLICY, {"median_edge_voxels": 23.12},
        voxel_um=7.91, inter_lamina_um=371.0)

    assert 23.12 >= frozen["resolution_limit_voxels"]      # limited
    assert 23.12 < measured["resolution_limit_voxels"]     # not limited
    # And a verdict reached without the spacing says which of the two it was.
    assert "was not supplied" in frozen_notes["reason"]
    assert next(entry for entry in frozen_notes["derived"]
                if entry["name"] == "resolution_limit_voxels")["measured"] is False
    assert next(entry for entry in notes["derived"]
                if entry["name"] == "resolution_limit_voxels")["measured"] is True


# -- the corpus the gate was written for -----------------------------------

@pytest.mark.parametrize("segment", sorted(
    (path.name for path in GROWN.glob("*") if (path / "x.tif").is_file())
    if GROWN.is_dir() else []))
def test_the_grown_corpus_certifies_the_same_with_and_without_a_scale(segment):
    """Every one of these changes is a no-op where the premise held. If that
    stops being true, the policy stopped being a measurement."""
    pytest.importorskip("tifffile")

    frozen = gate.certify(GROWN / segment)
    scaled = gate.certify(GROWN / segment, voxel_um=GROWN_VOXEL_UM,
                          inter_lamina_um=200.0)

    assert frozen["geometry_qc_state"] == "GEOMETRY_CERTIFIED"
    assert scaled["geometry_qc_state"] == "GEOMETRY_CERTIFIED"
    assert frozen["seam"] == scaled["seam"]
    assert frozen["resolution_limited"] is False
    assert frozen["resolution_limit_measured_here"] is False
    assert scaled["resolution_limit_measured_here"] is True


# -- and the candidate search ---------------------------------------------

def test_the_candidate_search_offers_every_pair_that_could_be_close():
    """The published bound was global, so a triangle whose partner sat in the
    top 0.1% searched with a radius below the one that pair needed and was
    recovered only when index order happened to cooperate. One such pair on the
    grown corpus. A fail-closed gate must not drop candidates quietly."""
    pytest.importorskip("tifffile")
    # The corpus is vendored and not in the repository, so it is absent
    # wherever the checkout is a clone -- CI included. The parametrised test
    # above collapses to no cases there; this one asserted its way into a
    # StopIteration instead, and has been failing every pipeline since it was
    # written. A red suite stops the deploy, which is how a broken worker image
    # went on running.
    segment = next((path for path in sorted(GROWN.glob("*"))
                    if (path / "x.tif").is_file()), None)
    if segment is None:
        pytest.skip(f"the grown corpus is not vendored here: {GROWN}")
    points, valid = gate.load_tifxyz(segment)
    mesh = gate.build_mesh(points, valid)
    left, right = gate._far_candidate_pairs(mesh, 4, 3.0, 20_000_000)
    offered = set(zip(left.tolist(), right.tolist()))

    centroid, radius = mesh["centroid"], mesh["circumradius"]
    rows, cols = mesh["triangle_row"], mesh["triangle_col"]
    # Every pair that could be within the gap, with no bound taken on trust.
    from scipy.spatial import cKDTree
    tree = cKDTree(centroid)
    missed = 0
    for here, candidates in enumerate(
            tree.query_ball_point(centroid, r=3.0 + radius + float(radius.max()))):
        other = np.asarray(candidates, dtype=np.int64)
        other = other[other > here]
        if other.size == 0:
            continue
        reach = 3.0 + radius[here] + radius[other]
        other = other[np.linalg.norm(centroid[other] - centroid[here], axis=1) <= reach]
        separation = np.maximum(np.abs(rows[other] - rows[here]),
                                np.abs(cols[other] - cols[here]))
        for partner in other[separation > 4].tolist():
            if (here, partner) not in offered:
                missed += 1
    assert missed == 0, f"{missed} pairs never reached the detectors"
