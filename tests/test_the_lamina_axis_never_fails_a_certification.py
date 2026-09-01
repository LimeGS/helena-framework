"""The third axis is a verdict beside the other two, never a way to lose one.

P2's job is the geometry verdict. The lamina gate rides along with it -- same
staged copy, same pass -- because fetching a surface twice to ask two questions
about it is the expensive half done twice. What that must not buy is a
certification lost to a volume that would not answer: an absent measurement is
an absence, and it says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.certifier import assess_surface_lamina, load_lamina_profile  # noqa: E402

SURFACE = {"surface_id": "s-1", "sample_id": "PHerc0826",
           "artifact_uri": "s3://bucket/s-1", "artifact_sha256": "a" * 64}


def write_flat_surface(directory: Path, *, size: int = 24, z: float = 200.0) -> None:
    """A patch with real coordinates, so cells have a normal and get sampled.

    A constant grid has none -- the cross product of two zero vectors -- and a
    surface nobody samples cannot exercise what the sampler does when the volume
    refuses.
    """
    import numpy as np
    import tifffile

    rows, columns = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    tifffile.imwrite(directory / "x.tif", (200 + columns).astype(np.float32))
    tifffile.imwrite(directory / "y.tif", (200 + rows).astype(np.float32))
    tifffile.imwrite(directory / "z.tif", np.full((size, size), z, np.float32))


class Store:
    def __init__(self, snapshots):
        self._snapshots = snapshots
        self.recorded: list[tuple] = []

    def snapshots(self, samples=None):
        return self._snapshots

    def record_lamina_assessment(self, surface_id, state, receipt, **lineage):
        self.recorded.append((surface_id, state, receipt, lineage))
        return {"surface_id": surface_id, "lamina_qc_state": state}


def test_a_scroll_with_no_registered_volume_is_unmeasured(tmp_path) -> None:
    store = Store([])

    outcome = assess_surface_lamina(store, SURFACE, tmp_path,
                                    requested_by_job_id="p2-1")

    assert outcome["state"] == "LAMINA_UNMEASURED"
    assert "no registered source" in outcome["reason"]
    assert store.recorded == []


def test_a_source_with_no_voxel_size_is_unmeasured(tmp_path) -> None:
    """Every micron figure downstream hangs off that number, and a thickness
    measured against a guessed one is a number with no unit."""
    store = Store([{"ct_uri": "s3://bucket/ct.zarr", "voxel_size_um": None}])

    outcome = assess_surface_lamina(store, SURFACE, tmp_path,
                                    requested_by_job_id="p2-1")

    assert outcome["state"] == "LAMINA_UNMEASURED"
    assert "voxel size" in outcome["reason"]


def test_a_volume_that_will_not_answer_is_unmeasured_not_fatal(tmp_path) -> None:
    """The failure this exists to prevent: a geometry certification that
    succeeded, lost to a network error on a second measurement."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("tifffile")
    write_flat_surface(tmp_path)
    store = Store([{"ct_uri": "s3://bucket/ct.zarr", "voxel_size_um": 9.362}])

    class Refuses:
        def read_cube(self, *args, **kwargs):
            raise RuntimeError("no route to host")

    outcome = assess_surface_lamina(store, SURFACE, tmp_path,
                                    requested_by_job_id="p2-1",
                                    ct_sampler=Refuses())

    assert outcome["state"] == "LAMINA_UNMEASURED"
    assert "no route to host" in outcome["reason"]
    assert store.recorded == []


def test_a_measured_verdict_is_recorded_with_its_calibration(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("tifffile")
    write_flat_surface(tmp_path)

    class Slab:
        def read_cube(self, ct_uri, coordinate_xyz, *, level, radius_l0_voxels):
            centre = [int(coordinate_xyz[axis]) for axis in "zyx"]
            origin = [value - radius_l0_voxels for value in centre]
            span = 2 * radius_l0_voxels + 1
            cube = np.full((span, span, span), 10.0, dtype=np.float32)
            for index in range(span):
                if abs(origin[0] + index - 200) <= 2:
                    cube[index, :, :] = 200.0
            return {"values": cube, "origin_zyx": origin,
                    "scale_zyx": [1.0, 1.0, 1.0], "level": level,
                    "source_read_set": {"objects": []}}

    store = Store([{"ct_uri": "s3://bucket/ct.zarr", "voxel_size_um": 9.362}])
    outcome = assess_surface_lamina(store, SURFACE, tmp_path,
                                    requested_by_job_id="p2-1",
                                    ct_sampler=Slab())

    assert outcome["lamina_qc_state"] == "LAMINA_SINGLE_SHEET"
    surface_id, state, receipt, lineage = store.recorded[0]
    assert surface_id == "s-1" and state == "LAMINA_SINGLE_SHEET"
    # The verdict names the calibration it was read against, by id and by hash.
    assert lineage["profile_id"] == load_lamina_profile()["profile_id"]
    assert len(lineage["profile_sha256"]) == 64
    assert receipt["median_thickness_um"] > 0
