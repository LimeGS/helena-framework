"""Fast checks for the committed calibration registry and source manifest."""
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SET_PATH = ROOT / "calibration" / "calibration_set.json"
SOURCES_PATH = ROOT / "calibration" / "source_manifest.json"
sys.path.insert(0, str(ROOT / "calibration"))
import replay_results


class TestCalibrationRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.set_bytes = SET_PATH.read_bytes()
        cls.registry = json.loads(cls.set_bytes)
        cls.rows = cls.registry["windows"]

    def test_roles_are_complete_and_disjoint(self):
        counts = Counter(row["role"] for row in self.rows)
        self.assertEqual(counts, {
            "fit": 324,
            "external_validation": 96,
            "context_only": 87,
            "report_only": 9,
        })
        self.assertTrue(all(row["split"] in {"train", "test"} for row in self.rows if row["role"] == "fit"))
        self.assertTrue(all(row["split"] == "external_validation" for row in self.rows if row["role"] == "external_validation"))

    def test_source_manifest_pins_every_registry_map(self):
        manifest = json.loads(SOURCES_PATH.read_text())
        self.assertEqual(
            manifest["calibration_set_sha256"],
            hashlib.sha256(self.set_bytes).hexdigest(),
        )
        sources = {item["path"] for item in manifest["sources"]}
        self.assertEqual(sources, {row["map"] for row in self.rows})
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["sources"]))

    def test_frozen_real_data_replay_reports_the_published_counts(self):
        self.assertEqual(replay_results.summary(), {
            "context_only/context_only/negative": {"passed": 55, "total": 87},
            "external_validation/external_validation/negative": {"passed": 0, "total": 60},
            "external_validation/external_validation/positive": {"passed": 27, "total": 36},
            "fit/test/negative": {"passed": 1, "total": 45},
            "fit/test/positive": {"passed": 55, "total": 56},
            "fit/train/negative": {"passed": 5, "total": 105},
            "fit/train/positive": {"passed": 116, "total": 118},
            "report_only/report_only/ambiguous": {"passed": 7, "total": 9},
        })


if __name__ == "__main__":
    unittest.main()
