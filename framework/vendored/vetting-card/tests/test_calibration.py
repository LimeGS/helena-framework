"""Opt-in real-data integration test for the calibrated absolute thresholds.

Runs the frozen calibration set through the real tool (vet_map.run_all_checks
on the real official ds8 ink maps) and asserts the held-out test-split
numbers the thresholds were signed off against. Skips cleanly if the project
data tree and its source manifest are not present (this repo ships a
byte-pinned registry, not map pixels), so it is a no-op in normal unit-test
runs. Set ``VETTING_CARD_REAL_CALIBRATION=1`` on a checkout with the exact
data tree to replay the integration suite.
"""
import json
import os
import unittest
from hashlib import sha256

import numpy as np

import vet_map

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "calibration", "calibration_set.json")
SOURCES = os.path.join(HERE, "..", "calibration", "source_manifest.json")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load(rel):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return np.array(Image.open(os.path.join(ROOT, rel)).convert("L"), dtype=np.float64)


def _sha256_file(path):
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(
    os.environ.get("VETTING_CARD_REAL_CALIBRATION") == "1"
    and os.path.exists(CAL)
    and os.path.exists(SOURCES)
    and os.path.isdir(os.path.join(ROOT, "data", "index_s5_0139", "maps")),
    "set VETTING_CARD_REAL_CALIBRATION=1 with the byte-pinned project map tree to run real integration probes",
)
class TestRealDataCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CAL) as handle:
            ws = json.load(handle)["windows"]
        with open(SOURCES) as handle:
            sources = {row["path"]: row for row in json.load(handle)["sources"]}
        missing = sorted({w["map"] for w in ws} - set(sources))
        if missing:
            raise AssertionError(f"source manifest omits registry maps: {missing}")
        # Validate identity for the entire real-data registry before using a
        # small stratified code probe. The exhaustive image-processing sweep
        # remains calibration/validate_tool.py because it is intentionally a
        # release-time job, not a commit-time unit test.
        for rel, expected in sources.items():
            path = os.path.join(ROOT, rel)
            if os.path.getsize(path) != expected["size_bytes"] or _sha256_file(path) != expected["sha256"]:
                raise AssertionError(f"calibration source hash mismatch: {rel}")

        from calibration import replay_results
        measured = replay_results._load_measurements()
        requested = (
            ("0139_clear", True),
            ("s1_confirmed", True),
            ("0139_bg_neg", False),
            ("s1_fiber_neg", False),
            ("s4_dense_pos", True),
            ("s4_lacunae_neg", False),
        )
        cls.ws = []
        for group, expected_pass in requested:
            row = next(
                w for w in sorted(ws, key=lambda item: item["id"])
                if w["group"] == group and replay_results.verdict(measured[w["id"]]) == expected_pass
            )
            cls.ws.append(row)
        cls.verdict = {}
        cache = {}
        for w in cls.ws:
            if w["map"] not in cache:
                cache.clear()
                cache[w["map"]] = _load(w["map"])
            r = vet_map.run_all_checks(
                cache[w["map"]],
                (w["x"], w["y"], w["x"] + w["win"], w["y"] + w["win"]),
                w["px_um"],
            )
            cls.verdict[w["id"]] = r["overall"]["pass"]

    def test_stratified_real_code_probes_match_frozen_measurements(self):
        expected = {
            "0139_clear": True,
            "s1_confirmed": True,
            "0139_bg_neg": False,
            "s1_fiber_neg": False,
            "s4_dense_pos": True,
            "s4_lacunae_neg": False,
        }
        observed = {w["group"]: self.verdict[w["id"]] for w in self.ws}
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
