"""Optional real-data test for the UC-01 converter.

Skipped by default so the suite stays fast and needs no data files. To run
it, point UC01_BAND_NPZ at a local copy of band_r1145_200_xyz.npz (see
strips/UC-01/README.md for the recipe):

    UC01_BAND_NPZ=/path/to/band_r1145_200_xyz.npz \
        python -m unittest tests.test_uc01_converter

Offline (reads only the local file) and deterministic, but slow (~1 min:
full-band unwrap + KD trees over 5M points).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

BAND = os.environ.get("UC01_BAND_NPZ", "")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strips" / "UC-01"))


@unittest.skipUnless(BAND and Path(BAND).exists(),
                     "set UC01_BAND_NPZ to a local band_r1145_200_xyz.npz")
class Uc01ConverterTest(unittest.TestCase):
    def test_conversion_reproduces_source_audit_populations(self):
        from convert_ntaudit_band import convert
        from strip_format import load_strip, validate_strip

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "UC-01.npz"
            convert(Path(BAND), out)
            strip = load_strip(out)

        self.assertEqual(validate_strip(strip), [])
        self.assertEqual(strip.wrap_indices, [0, 1, 2, 3])
        # the source audit's published class populations (BENCHMARK.md §2)
        expected = {0: 114509, 1: 1654338, 2: 1675193, 3: 1631237}
        for wid, n in expected.items():
            self.assertEqual(strip.wraps[wid].shape[0], n)
        self.assertEqual(
            strip.meta["source_winding_class_to_wrap"],
            {"-2": 0, "-1": 1, "0": 2, "1": 3},
        )


if __name__ == "__main__":
    unittest.main()
