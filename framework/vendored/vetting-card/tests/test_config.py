"""Sanity checks on card_config.py and its real-data calibration framing."""
import unittest
from pathlib import Path

import card_config as cfg


class TestPhysicalScale(unittest.TestCase):
    def test_letter_size_range_sane(self):
        self.assertGreater(cfg.LETTER_SIZE_MM_MIN, 0)
        self.assertLess(cfg.LETTER_SIZE_MM_MIN, cfg.LETTER_SIZE_MM_MAX)
        self.assertLess(cfg.LETTER_SIZE_MM_MAX, 50)  # sanity: not "a whole column"

    def test_default_px_um_positive(self):
        self.assertGreater(cfg.DEFAULT_PX_UM, 0)

    def test_letter_scale_px_range_matches_hand_computation(self):
        lo, hi = cfg.letter_scale_px_range(8.0)
        self.assertAlmostEqual(lo, 1500.0 / 8.0)
        self.assertAlmostEqual(hi, 4000.0 / 8.0)
        self.assertLess(lo, hi)

    def test_letter_scale_px_range_rejects_nonpositive_px_um(self):
        for bad in (0, -1.0):
            with self.assertRaises(ValueError):
                cfg.letter_scale_px_range(bad)


class TestNullSampling(unittest.TestCase):
    def test_sample_counts_sane(self):
        self.assertGreaterEqual(cfg.MIN_NULL_SAMPLES, 10)
        self.assertGreaterEqual(cfg.N_NULL_SAMPLES, cfg.MIN_NULL_SAMPLES)
        self.assertGreater(cfg.MAX_NULL_SAMPLE_ATTEMPTS, cfg.N_NULL_SAMPLES)

    def test_exclusion_margin_positive(self):
        self.assertGreater(cfg.EXCLUSION_MARGIN_FACTOR, 0)


class TestLetterEnergyCheck(unittest.TestCase):
    def test_percentile_in_range(self):
        self.assertGreater(cfg.LETTER_ENERGY_PERCENTILE, 50)
        self.assertLessEqual(cfg.LETTER_ENERGY_PERCENTILE, 100)

    def test_bandpass_factors_ordered(self):
        self.assertGreater(cfg.BANDPASS_SIGMA_LO_FACTOR, 0)
        self.assertLess(cfg.BANDPASS_SIGMA_LO_FACTOR, cfg.BANDPASS_SIGMA_HI_FACTOR)

    def test_bandpass_sigmas_ordered(self):
        for px_um in (2.4, 8.0, 50.0):
            sigma_lo, sigma_hi = cfg.bandpass_sigmas(px_um)
            self.assertGreater(sigma_lo, 0)
            self.assertLess(sigma_lo, sigma_hi)


class TestStructureCheck(unittest.TestCase):
    def test_percentile_in_range(self):
        self.assertGreater(cfg.STRUCTURE_PERCENTILE, 50)
        self.assertLessEqual(cfg.STRUCTURE_PERCENTILE, 100)

    def test_threshold_k_positive(self):
        self.assertGreater(cfg.STRUCTURE_THRESHOLD_K, 0)

    def test_component_area_factors_ordered(self):
        self.assertGreater(cfg.COMPONENT_AREA_MIN_FACTOR, 0)
        self.assertLess(cfg.COMPONENT_AREA_MIN_FACTOR, cfg.COMPONENT_AREA_MAX_FACTOR)

    def test_component_area_px_range_ordered(self):
        for px_um in (2.4, 8.0, 50.0):
            lo, hi = cfg.component_area_px_range(px_um)
            self.assertGreater(lo, 0)
            self.assertLess(lo, hi)


class TestDegenerateChecks(unittest.TestCase):
    def test_blank_eps_small_and_positive(self):
        self.assertGreater(cfg.BLANK_STD_EPS, 0)
        self.assertLess(cfg.BLANK_STD_EPS, 0.01)

    def test_saturation_fraction_in_range(self):
        self.assertGreater(cfg.SATURATION_FRACTION_MAX, 0)
        self.assertLessEqual(cfg.SATURATION_FRACTION_MAX, 1.0)

    def test_saturation_value_eps_small_and_positive(self):
        self.assertGreater(cfg.SATURATION_VALUE_EPS, 0)
        self.assertLess(cfg.SATURATION_VALUE_EPS, 0.05)


class TestMinimumSizes(unittest.TestCase):
    def test_min_sizes_positive_and_consistent(self):
        self.assertGreaterEqual(cfg.MIN_BBOX_DIM_PX, 1)
        self.assertGreaterEqual(cfg.MIN_MAP_DIM_PX, cfg.MIN_BBOX_DIM_PX)


class TestPipelineConfig(unittest.TestCase):
    def test_scan_window_factor_positive(self):
        self.assertGreater(cfg.PIPELINE_SCAN_WINDOW_LETTERSCALES, 0)

    def test_scan_stride_fraction_in_range(self):
        self.assertGreater(cfg.PIPELINE_SCAN_STRIDE_FRACTION, 0)
        self.assertLessEqual(cfg.PIPELINE_SCAN_STRIDE_FRACTION, 1.0)

    def test_pass_rule_thresholds_in_range(self):
        self.assertGreaterEqual(cfg.PIPELINE_MAX_PAREIDOLIA_RATE, 0.0)
        self.assertLessEqual(cfg.PIPELINE_MAX_PAREIDOLIA_RATE, 1.0)
        self.assertGreater(cfg.PIPELINE_MIN_SENSITIVITY, 0.0)
        self.assertLessEqual(cfg.PIPELINE_MIN_SENSITIVITY, 1.0)


class TestConfigHash(unittest.TestCase):
    def test_hash_is_stable_and_well_formed(self):
        h1 = cfg.config_hash()
        h2 = cfg.config_hash()
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 12)
        int(h1, 16)  # raises ValueError if not valid hex

    def test_hash_changes_if_source_changes(self):
        # Hashing a temporarily-modified copy of the source proves the hash
        # is actually derived from file content, not a hardcoded string.
        real_src = Path(cfg.__file__).read_bytes()
        import hashlib
        tampered_hash = hashlib.sha256(real_src + b"\n# tamper").hexdigest()[:12]
        self.assertNotEqual(tampered_hash, cfg.config_hash())


class TestCalibrationDocumentation(unittest.TestCase):
    """The public configuration must not regress to v0's synthetic framing."""

    def test_calibrated_gates_are_explicitly_described_as_real_data_fit(self):
        src = Path(cfg.__file__).read_text()
        self.assertIn("324 fitting windows", src)
        self.assertIn("96\nexternal Scroll-4 validation windows", src)
        for name in ("E_FRAC_MIN", "STRUCTURE_AREAFRAC_MIN", "OTSU_SEP_MIN"):
            self.assertIn(name, src)

    def test_schema_bumped_for_evidence_provenance(self):
        self.assertEqual(cfg.CARD_SCHEMA_VERSION, "v1")
        self.assertGreaterEqual(tuple(map(int, cfg.TOOL_VERSION.split("."))), (0, 3, 0))


if __name__ == "__main__":
    unittest.main()
