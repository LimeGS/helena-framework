import unittest

import numpy as np

from scoring_core import build_trees, score_points, summarize
from synthetic import spiral_strip, wrap_surface_points

R0, PITCH = 300.0, 30.0


class ScorePointsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strip = spiral_strip(r0=R0, pitch=PITCH)
        cls.trees = build_trees(cls.strip)
        # interior thetas, well away from the 2*pi cut
        cls.thetas = np.linspace(0.5, 5.8, 400)
        cls.z_mid = 20.0

    def test_perfect_predictions_zero_wrong_hop(self):
        # exact points of the target wrap's own sampling (interior only)
        interior = ~self.strip.edges[2]
        pred = self.strip.wraps[2][interior][:500].astype(np.float64)
        s = summarize(score_points(self.strip, self.trees, pred,
                                   from_wrap=1, direction="front"))
        self.assertEqual(s["wrong_hop_pct"], 0.0)
        self.assertEqual(s["correct_pct"], 100.0)
        self.assertEqual(s["excluded"], 0)

    def test_half_pitch_offset_is_100_percent_wrong(self):
        # halfway between wrap 1 and wrap 2: exactly at the half-gap
        # threshold (>= threshold counts as a miss, matching the source
        # audit's `d_exp >= 0.5 * gap` rule)
        pred = wrap_surface_points(1, self.thetas, self.z_mid,
                                   r0=R0, pitch=PITCH)
        radial = pred[:, :2] / np.linalg.norm(pred[:, :2], axis=1,
                                              keepdims=True)
        pred = pred.copy()
        pred[:, :2] += 0.5 * PITCH * radial
        s = summarize(score_points(self.strip, self.trees, pred,
                                   from_wrap=1, direction="front"))
        self.assertEqual(s["wrong_hop_pct"], 100.0)

    def test_points_near_wrong_wrap_flagged_as_wrong_wrap(self):
        # tracer asked for wrap 2 (front of 1) but landed on wrap 0
        pred = wrap_surface_points(0, self.thetas, self.z_mid,
                                   r0=R0, pitch=PITCH)
        result = score_points(self.strip, self.trees, pred,
                              from_wrap=1, direction="front")
        s = summarize(result)
        self.assertEqual(s["wrong_wrap_pct"], 100.0)
        self.assertEqual(s["wrong_hop_pct"], 100.0)
        self.assertEqual(s["distance_miss_pct"], 0.0)

    def test_back_direction_targets_lower_wrap(self):
        pred = wrap_surface_points(0, self.thetas, self.z_mid,
                                   r0=R0, pitch=PITCH)
        s = summarize(score_points(self.strip, self.trees, pred,
                                   from_wrap=1, direction="back"))
        self.assertEqual(s["wrong_hop_pct"], 0.0)

    def test_points_outside_coverage_are_excluded_not_scored(self):
        pred = np.full((50, 3), 5000.0)
        result = score_points(self.strip, self.trees, pred,
                              from_wrap=1, direction="front")
        s = summarize(result)
        self.assertEqual(s["included"], 0)
        self.assertEqual(s["excluded"], 50)
        self.assertEqual(s["excluded_no_coverage"], 50)
        # excluded points contribute to no numerator
        self.assertEqual(s["correct"], 0)
        self.assertEqual(s["wrong_hop"], 0)

    def test_predictions_near_reference_edge_are_excluded(self):
        # predictions right at the 2*pi cut of the target wrap: their
        # nearest reference point is edge-flagged -> excluded (ported
        # band-row-edge rule)
        edge_pts = self.strip.wraps[2][self.strip.edges[2]][:20].astype(
            np.float64
        )
        result = score_points(self.strip, self.trees, edge_pts,
                              from_wrap=1, direction="front")
        s = summarize(result)
        self.assertEqual(s["excluded"], 20)
        self.assertEqual(s["excluded_edge"], 20)

    def test_missing_target_wrap_raises(self):
        with self.assertRaises(ValueError):
            score_points(self.strip, self.trees,
                         np.zeros((3, 3)), from_wrap=3, direction="front")
        with self.assertRaises(ValueError):
            score_points(self.strip, self.trees,
                         np.zeros((3, 3)), from_wrap=0, direction="back")


if __name__ == "__main__":
    unittest.main()


class HalfGapBoundaryTest(unittest.TestCase):
    """Pin the exact d == 0.5*gap boundary convention (correct iff
    d < threshold, distance_miss iff d >= threshold), matching the source
    audit's `d_exp < 0.5*gap` / `>= 0.5*gap` split. Uses flat parallel
    wraps so distances are exact rather than KD-tree-discretized."""

    def _flat_strip(self):
        from strip_format import Strip
        grid = np.array([[i, j] for i in range(6) for j in range(6)], float)
        wraps = {k: np.column_stack([np.full(len(grid), 10.0 * k), grid])
                 for k in range(3)}  # planes x=0,10,20; gap 10
        return Strip(wraps=wraps, meta={})

    def test_exact_half_gap_is_distance_miss_not_correct(self):
        strip = self._flat_strip()
        trees = build_trees(strip)
        # from wrap1 (x=10) front -> target wrap2 (x=20); local gap = 10,
        # threshold = 5. x=25 is exactly 5 from wrap2 (== threshold).
        pred = np.array([[25.0, 2.0, 2.0]])
        r = score_points(strip, trees, pred, from_wrap=1, direction="front")
        self.assertAlmostEqual(float(r["gap"][0]), 10.0)
        self.assertEqual(int(r["status"][0]), 2)  # distance_miss, not correct

    def test_just_inside_half_gap_is_correct(self):
        strip = self._flat_strip()
        trees = build_trees(strip)
        pred = np.array([[24.0, 2.0, 2.0]])  # 4 from wrap2, < threshold 5
        r = score_points(strip, trees, pred, from_wrap=1, direction="front")
        self.assertEqual(int(r["status"][0]), 0)  # correct
