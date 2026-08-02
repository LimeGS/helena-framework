import unittest

import numpy as np

from make_strip import (
    InsufficientRevolutionsError,
    TIER_MEDIUM_MAX_UM,
    TIER_ULTRA_MAX_UM,
    UnreliableAxisEstimateError,
    assign_tier,
    build_strip_from_grids,
)
from synthetic import spiral_grid

PITCH = 30.0
REVOLUTIONS = 2.6


class TierBoundaryTest(unittest.TestCase):
    """Both sides of the PROVISIONAL 120 / 250 um tier thresholds."""

    def test_just_below_and_at_120(self):
        self.assertEqual(assign_tier(TIER_ULTRA_MAX_UM - 0.1), "ultra")
        self.assertEqual(assign_tier(TIER_ULTRA_MAX_UM), "medium")

    def test_just_below_and_at_250(self):
        self.assertEqual(assign_tier(TIER_MEDIUM_MAX_UM - 0.1), "medium")
        self.assertEqual(assign_tier(TIER_MEDIUM_MAX_UM), "easy")


class BuildStripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grids = spiral_grid(rows=24, cols=5000, revolutions=REVOLUTIONS,
                                r0=400.0, pitch=PITCH)
        x, y, z, valid = cls.grids
        cls.built = build_strip_from_grids(
            x, y, z, valid, (0, x.shape[0], 0, x.shape[1]), voxel_size_um=1.0
        )

    def test_recovers_expected_wrap_count(self):
        # 2.6 revolutions -> 3 phase bins (2 full + 1 partial with plenty
        # of points)
        self.assertEqual(sorted(self.built.wraps.keys()), [0, 1, 2])

    def test_winding_axis_detected_as_columns(self):
        self.assertEqual(self.built.diagnostics["winding_axis"], "col")
        self.assertAlmostEqual(
            self.built.diagnostics["measured_revolutions"], REVOLUTIONS,
            delta=0.05,
        )

    def test_pitch_within_2_percent_of_analytic(self):
        measured = self.built.pitch_um["median"]
        self.assertLess(abs(measured - PITCH) / PITCH, 0.02,
                        f"pitch {measured} vs analytic {PITCH}")

    def test_tier_from_pipeline_pitch(self):
        # native pitch 30 * voxel 1.0 um = 30 um -> ultra
        self.assertEqual(self.built.tier, "ultra")

    def test_edge_flags_present_and_sparse(self):
        for k, flags in self.built.edges.items():
            self.assertEqual(flags.shape[0], self.built.wraps[k].shape[0])
            frac = flags.mean()
            self.assertGreater(frac, 0.0, f"wrap {k} has no edge flags")
            self.assertLess(frac, 0.10, f"wrap {k} is {frac:.0%} edge-flagged")

    def test_normals_unit_length(self):
        for k, nrm in self.built.normals.items():
            lengths = np.linalg.norm(nrm, axis=1)
            np.testing.assert_allclose(lengths, 1.0, atol=1e-4)

    def test_voxel_size_scales_tier(self):
        x, y, z, valid = self.grids
        built = build_strip_from_grids(
            x, y, z, valid, (0, x.shape[0], 0, x.shape[1]),
            voxel_size_um=10.0,  # native pitch 30 -> 300 um -> easy
        )
        self.assertEqual(built.tier, "easy")
        built2 = build_strip_from_grids(
            x, y, z, valid, (0, x.shape[0], 0, x.shape[1]),
            voxel_size_um=6.0,  # -> 180 um -> medium
        )
        self.assertEqual(built2.tier, "medium")


class RowAxisWindingTest(unittest.TestCase):
    def test_transposed_grid_detects_row_axis(self):
        x, y, z, valid = spiral_grid(rows=24, cols=3000,
                                     revolutions=2.2, r0=400.0, pitch=PITCH)
        built = build_strip_from_grids(
            x.T.copy(), y.T.copy(), z.T.copy(), valid.T.copy(),
            (0, x.shape[1], 0, x.shape[0]), voxel_size_um=1.0
        )
        self.assertEqual(built.diagnostics["winding_axis"], "row")
        self.assertEqual(len(built.wraps), 3)


class FailurePathsTest(unittest.TestCase):
    def test_sub_revolution_window_raises(self):
        x, y, z, valid = spiral_grid(rows=24, cols=5000,
                                     revolutions=REVOLUTIONS, r0=400.0,
                                     pitch=PITCH)
        # a column window spanning ~0.4 revolutions
        cols = int(0.4 / REVOLUTIONS * x.shape[1])
        with self.assertRaises(InsufficientRevolutionsError):
            build_strip_from_grids(x, y, z, valid, (0, x.shape[0], 0, cols))

    def test_small_planar_patch_raises_unreliable_axis(self):
        # A near-planar local patch (like the real seed_segment/): the
        # centroid-default axis measures the patch's own shape, which the
        # radius-span sanity check must refuse.
        u = np.linspace(0, 100, 50)
        v = np.linspace(0, 100, 40)
        uu, vv = np.meshgrid(u, v)
        x = 2000.0 + uu
        y = 1500.0 + vv
        z = 7000.0 + 0.05 * uu  # nearly flat
        valid = np.ones_like(x, dtype=bool)
        with self.assertRaises(UnreliableAxisEstimateError):
            build_strip_from_grids(x, y, z, valid, (0, 40, 0, 50))

    def test_explicit_axis_bypasses_span_check_but_still_needs_revolutions(self):
        u = np.linspace(0, 100, 50)
        v = np.linspace(0, 100, 40)
        uu, vv = np.meshgrid(u, v)
        x = 2000.0 + uu
        y = 1500.0 + vv
        z = 7000.0 + 0.05 * uu
        valid = np.ones_like(x, dtype=bool)
        axis = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        with self.assertRaises(InsufficientRevolutionsError):
            build_strip_from_grids(x, y, z, valid, (0, 40, 0, 50), axis=axis)


if __name__ == "__main__":
    unittest.main()
