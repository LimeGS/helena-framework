import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from strip_format import (
    MIN_POINTS_PER_WRAP,
    Strip,
    edge_flags_from_phase,
    is_qualified,
    load_strip,
    save_strip,
    validate_strip,
)
from synthetic import spiral_strip


class SaveLoadRoundtripTest(unittest.TestCase):
    def test_roundtrip_preserves_wraps_normals_edges_pitch_meta(self):
        strip = spiral_strip(n_wraps=3, pts_per_row=60, rows=4)
        normals = {
            k: np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                       (v.shape[0], 1))
            for k, v in strip.wraps.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strip.npz"
            save_strip(path, strip.wraps, normals=normals,
                       pitch_um=strip.pitch_um, meta=strip.meta,
                       edges=strip.edges)
            loaded = load_strip(path)

        self.assertEqual(loaded.wrap_indices, [0, 1, 2])
        for k in loaded.wrap_indices:
            np.testing.assert_array_equal(loaded.wraps[k], strip.wraps[k])
            np.testing.assert_array_equal(loaded.edges[k], strip.edges[k])
            np.testing.assert_array_equal(loaded.normals[k], normals[k])
        self.assertEqual(loaded.pitch_um["median"], strip.pitch_um["median"])
        self.assertEqual(loaded.meta["scroll"], "synthetic")
        self.assertEqual(loaded.meta["schema_version"], "strip-v0")

    def test_noncontiguous_wrap_keys_are_reindexed_with_mapping(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        wraps = {5: strip.wraps[0], 9: strip.wraps[1]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strip.npz"
            save_strip(path, wraps, meta=strip.meta)
            loaded = load_strip(path)
        self.assertEqual(loaded.wrap_indices, [0, 1])
        self.assertEqual(loaded.meta["wrap_reindex"], {"5": 0, "9": 1})


class ValidationTest(unittest.TestCase):
    def _meta(self):
        return dict(spiral_strip(n_wraps=2, pts_per_row=60, rows=4).meta)

    def test_valid_strip_has_no_problems(self):
        strip = spiral_strip(n_wraps=3, pts_per_row=60, rows=4)
        self.assertEqual(validate_strip(strip), [])

    def test_single_wrap_is_caught(self):
        full = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip = Strip(wraps={0: full.wraps[0]}, meta=full.meta)
        problems = validate_strip(strip)
        self.assertTrue(any("need >= 2" in p for p in problems))

    def test_nans_are_caught(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip.wraps[1] = strip.wraps[1].copy()
        strip.wraps[1][3, 1] = np.nan
        problems = validate_strip(strip)
        self.assertTrue(any("NaN" in p for p in problems))

    def test_empty_wrap_is_caught(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip.wraps[1] = np.zeros((0, 3), dtype=np.float32)
        strip.edges.pop(1, None)
        problems = validate_strip(strip)
        self.assertTrue(
            any(f"need >= {MIN_POINTS_PER_WRAP}" in p for p in problems)
        )

    def test_too_few_points_is_caught(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip.wraps[1] = strip.wraps[1][: MIN_POINTS_PER_WRAP - 1]
        strip.edges.pop(1, None)
        problems = validate_strip(strip)
        self.assertTrue(any("point(s), need >=" in p for p in problems))

    def test_missing_meta_keys_are_caught(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip.meta = {"scroll": "x"}
        problems = validate_strip(strip)
        self.assertTrue(any("meta missing required keys" in p for p in problems))

    def test_mismatched_edge_shape_is_caught(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip.edges[0] = strip.edges[0][:-5]
        problems = validate_strip(strip)
        self.assertTrue(any("edges_00" in p for p in problems))

    def test_wrong_wrap_shape_is_caught(self):
        strip = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip.wraps[0] = strip.wraps[0][:, :2]
        strip.edges.pop(0, None)
        problems = validate_strip(strip)
        self.assertTrue(any("expected shape (N, 3)" in p for p in problems))


class EdgeFlagHelperTest(unittest.TestCase):
    def test_flags_points_near_phase_extremes_only(self):
        phase = np.array([0.0, 0.05, 1.0, 3.0, 6.0, 6.23])
        flags = edge_flags_from_phase(phase, margin=0.1)
        np.testing.assert_array_equal(
            flags, [True, True, False, False, False, True]
        )

    def test_empty_input(self):
        self.assertEqual(edge_flags_from_phase(np.zeros(0), 0.1).shape, (0,))


class IsQualifiedTest(unittest.TestCase):
    def test_reads_sibling_qualification_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            strip_path = Path(tmp) / "foo.npz"
            strip_path.write_bytes(b"placeholder")
            self.assertFalse(is_qualified(strip_path))
            qual = strip_path.with_name("foo.qualification.json")
            qual.write_text(json.dumps({"overall_pass": False}))
            self.assertFalse(is_qualified(strip_path))
            qual.write_text(json.dumps({"overall_pass": True}))
            self.assertTrue(is_qualified(strip_path))


if __name__ == "__main__":
    unittest.main()


class PitchValidationTest(unittest.TestCase):
    def test_nonpositive_pitch_median_is_caught(self):
        strip = spiral_strip()
        strip.pitch_um = {"median": -5.0, "p10": 1.0, "p90": 9.0}
        problems = validate_strip(strip)
        self.assertTrue(any("pitch" in p.lower() for p in problems),
                        f"expected a pitch problem, got {problems}")

    def test_positive_pitch_median_is_fine(self):
        strip = spiral_strip()
        strip.pitch_um = {"median": 30.0, "p10": 25.0, "p90": 35.0}
        self.assertEqual(
            [p for p in validate_strip(strip) if "pitch" in p.lower()], [])
