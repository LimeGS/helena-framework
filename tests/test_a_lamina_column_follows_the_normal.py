"""Reading the CT along a surface's own normal.

The measurement in `lamina.py` is arithmetic over a profile. This is the half
that decides which profile it gets: where the sheet's normal points, how far
along it to read, and what to do when the volume cannot answer.

Built against a synthetic volume rather than a scroll: a slab of known thickness
at a known place, so a wrong normal or an off-by-one in the sampling shows up as
a thickness that is wrong by a factor rather than as a plausible number nobody
can check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from fleet.lamina_columns import cell_normal, sample_columns  # noqa: E402

PROFILE = {
    "profile_id": "test-bands@1.0.0",
    "sampling": {"columns": 16, "samples_per_column": 61, "sample_step_um": 2.0},
    "thickness_um": {"sheet_low": 15.0, "sheet_high": 70.0},
    "minimum_clean_fraction": 0.90,
    "bimodality_ceiling": 1.0,
}


class SlabVolume:
    """A volume holding one horizontal sheet, `thickness` voxels thick, at z=64.

    Answers like the fleet's OME-Zarr reader: a cube around a coordinate, its
    origin, the level's scale, and the read set.
    """

    def __init__(self, thickness_voxels: float, *, size: int = 128):
        self.thickness = thickness_voxels
        self.size = size
        self.reads = 0

    def read_cube(self, ct_uri, coordinate_xyz, *, level, radius_l0_voxels):
        self.reads += 1
        centre = [int(coordinate_xyz["z"]), int(coordinate_xyz["y"]),
                  int(coordinate_xyz["x"])]
        origin = [max(0, value - radius_l0_voxels) for value in centre]
        stop = [min(self.size, value + radius_l0_voxels + 1) for value in centre]
        shape = [stop[axis] - origin[axis] for axis in range(3)]
        cube = np.full(shape, 10.0, dtype=np.float32)
        for index in range(shape[0]):
            z = origin[0] + index
            if abs(z - 64) <= self.thickness / 2.0:
                cube[index, :, :] = 200.0
        return {"values": cube, "origin_zyx": origin, "scale_zyx": [1.0, 1.0, 1.0],
                "level": level, "center_zyx": centre,
                "source_read_set": {"objects": [
                    {"object_key": f"{centre}", "sha256": "0" * 64, "bytes": 1}]}}


def flat_surface(directory: Path, *, size: int = 24, z: int = 64) -> Path:
    """A patch of the plane z = 64, whose normal is the z axis."""
    directory.mkdir(parents=True, exist_ok=True)
    rows, columns = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    tifffile.imwrite(directory / "x.tif", (40 + columns).astype(np.float32))
    tifffile.imwrite(directory / "y.tif", (40 + rows).astype(np.float32))
    tifffile.imwrite(directory / "z.tif", np.full((size, size), float(z), np.float32))
    return directory


def test_a_flat_patch_has_the_normal_of_its_plane(tmp_path) -> None:
    flat_surface(tmp_path / "s")
    x = tifffile.imread(tmp_path / "s/x.tif")
    y = tifffile.imread(tmp_path / "s/y.tif")
    z = tifffile.imread(tmp_path / "s/z.tif")

    normal = cell_normal(x, y, z, 5, 5)

    assert abs(abs(normal[2]) - 1.0) < 1e-9      # along z
    assert abs(normal[0]) < 1e-9 and abs(normal[1]) < 1e-9


def test_a_cell_beside_a_hole_has_no_normal(tmp_path) -> None:
    """A TIFXYZ says "no coordinate here" with zeros, and that is not the
    origin of the scroll -- so a cell whose neighbour is empty is not sampled
    along some default direction, it is skipped."""
    directory = flat_surface(tmp_path / "s")
    z = tifffile.imread(directory / "z.tif")
    x = tifffile.imread(directory / "x.tif")
    y = tifffile.imread(directory / "y.tif")
    x[5, 6] = y[5, 6] = z[5, 6] = 0.0

    assert cell_normal(x, y, z, 5, 5) is None
    assert cell_normal(x, y, z, 10, 10) is not None


def test_a_sheet_reads_as_its_own_thickness(tmp_path) -> None:
    """Eight voxels of slab, sampled every 2 um at 4 um per voxel, is 32 um."""
    directory = flat_surface(tmp_path / "s")
    volume = SlabVolume(thickness_voxels=8)

    outcome = sample_columns(directory, ct_uri="memory://ct", voxel_size_um=4.0,
                             profile=PROFILE, sampler=volume)

    assert outcome["state"] == "LAMINA_SINGLE_SHEET"
    # Nine voxels read as material (|z-64| <= 4 inclusive) at 4 um each.
    assert 34.0 <= outcome["median_thickness_um"] <= 38.0
    assert outcome["clean_fraction"] == 1.0
    assert volume.reads == outcome["sampling"]["columns"]


def test_two_fused_laminae_read_as_a_slab(tmp_path) -> None:
    """Thick enough to measure inside the window: 22 voxels is 88 um."""
    directory = flat_surface(tmp_path / "s")

    outcome = sample_columns(directory, ct_uri="memory://ct", voxel_size_um=4.0,
                             profile=PROFILE, sampler=SlabVolume(thickness_voxels=22))

    assert outcome["state"] == "LAMINA_FUSED"
    assert outcome["median_thickness_um"] > 70.0


def test_a_window_with_no_air_in_it_says_so(tmp_path) -> None:
    """Material end to end, everywhere: there is no interface to measure.

    Not "too few columns" -- the columns are fine and the two populations are
    not there. Filing this under a hole in the sampling would hide a featureless
    volume behind a word about coverage.
    """
    directory = flat_surface(tmp_path / "s")

    outcome = sample_columns(directory, ct_uri="memory://ct", voxel_size_um=4.0,
                             profile=PROFILE, sampler=SlabVolume(thickness_voxels=60))

    assert outcome["state"] == "LAMINA_UNRESOLVED"
    assert outcome["interface_level"] is None
    assert "one population" in outcome["reason"]


def test_the_verdict_names_the_bytes_it_was_measured_from(tmp_path) -> None:
    directory = flat_surface(tmp_path / "s")

    outcome = sample_columns(directory, ct_uri="memory://ct", voxel_size_um=4.0,
                             profile=PROFILE, sampler=SlabVolume(thickness_voxels=8))

    read_set = outcome["source_read_set"]
    assert read_set["objects"] and read_set["canonical_manifest_sha256"]
    # Deduplicated and ordered, so the same read set hashes the same twice.
    keys = [entry["object_key"] for entry in read_set["objects"]]
    assert keys == sorted(set(keys))
