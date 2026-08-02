import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qualify_strip import qualify
from strip_format import Strip, is_qualified, save_strip
from synthetic import shuffled_strip, spiral_strip

REPO_ROOT = Path(__file__).resolve().parent.parent


class QualifyCleanStripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strip = spiral_strip()
        cls.report = qualify(cls.strip)

    def test_overall_pass(self):
        self.assertTrue(self.report["overall_pass"])

    def test_self_test_passes_at_zero_distance(self):
        c = self.report["checks"]["self_test"]
        self.assertTrue(c["pass"])
        self.assertLessEqual(c["max_dist_any_wrap"], 1e-6)
        self.assertEqual(c["min_pct_nearest_own_wrap"], 100.0)

    def test_wrong_side_passes(self):
        c = self.report["checks"]["wrong_side"]
        self.assertTrue(c["pass"])
        self.assertGreaterEqual(c["min_fail_pct_over_pairs"], 95.0)

    def test_null_baseline_passes(self):
        c = self.report["checks"]["null_baseline"]
        self.assertTrue(c["pass"])
        self.assertGreaterEqual(c["min_wrong_hop_pct_over_pairs"], 99.0)

    def test_ct_check_not_run_offline(self):
        c = self.report["checks"]["ct_check"]
        self.assertFalse(c["run"])


class QualifyCorruptedStripTest(unittest.TestCase):
    """The anti-tautology guard: a strip whose wrap labels were shuffled
    between wraps MUST fail the suite. If this test ever starts passing
    qualification, the checks have regressed into measuring nothing."""

    @classmethod
    def setUpClass(cls):
        cls.strip = shuffled_strip(spiral_strip(), fraction=0.4, seed=42)
        cls.report = qualify(cls.strip)

    def test_overall_fails(self):
        self.assertFalse(self.report["overall_pass"])

    def test_wrong_side_catches_the_shuffle(self):
        c = self.report["checks"]["wrong_side"]
        self.assertFalse(c["pass"])
        # the shuffled pairs collapse far below the bar, not marginally
        self.assertLess(c["min_fail_pct_over_pairs"], 50.0)


class QualifyStructuralFailureTest(unittest.TestCase):
    def test_invalid_strip_skips_geometric_checks(self):
        full = spiral_strip(n_wraps=2, pts_per_row=60, rows=4)
        strip = Strip(wraps={0: full.wraps[0]}, meta=full.meta)
        report = qualify(strip)
        self.assertFalse(report["overall_pass"])
        self.assertFalse(report["validation"]["pass"])
        self.assertIn("skipped_reason", report["checks"])


class QualifyCliRoundtripTest(unittest.TestCase):
    """End-to-end: save -> qualify CLI writes reports + exit 0 ->
    is_qualified() sees it -> score CLI drops the UNQUALIFIED warning."""

    def test_cli_writes_reports_and_gates_scorer_warning(self):
        strip = spiral_strip(pts_per_row=120, rows=6)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            strip_path = tmp / "synth.npz"
            save_strip(strip_path, strip.wraps, pitch_um=strip.pitch_um,
                       meta=strip.meta, edges=strip.edges)
            self.assertFalse(is_qualified(strip_path))

            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "qualify_strip.py"),
                 str(strip_path)],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            qual_json = tmp / "synth.qualification.json"
            self.assertTrue(qual_json.exists())
            self.assertTrue((tmp / "synth.qualification.md").exists())
            report = json.loads(qual_json.read_text())
            self.assertTrue(report["overall_pass"])
            self.assertTrue(is_qualified(strip_path))

            # scorer: perfect predictions, no UNQUALIFIED warning
            import numpy as np
            interior = ~strip.edges[2]
            pred_path = tmp / "pred.npz"
            np.savez(pred_path, points=strip.wraps[2][interior][:200])
            proc2 = subprocess.run(
                [sys.executable, str(REPO_ROOT / "score_strip.py"),
                 "--strip", str(strip_path), "--mode", "points",
                 "--pred", str(pred_path),
                 "--from-wrap", "1", "--direction", "front"],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertEqual(proc2.returncode, 0, proc2.stderr)
            self.assertNotIn("UNQUALIFIED", proc2.stderr)
            card = json.loads((tmp / "pred.scorecard.json").read_text())
            self.assertEqual(
                card["percentages_of_included"]["wrong_hop_pct"], 0.0
            )

    def test_cli_exit_2_and_warning_for_corrupted_strip(self):
        strip = shuffled_strip(spiral_strip(pts_per_row=120, rows=6),
                               fraction=0.4, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            strip_path = tmp / "bad.npz"
            save_strip(strip_path, strip.wraps, pitch_um=strip.pitch_um,
                       meta=strip.meta, edges=strip.edges)
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "qualify_strip.py"),
                 str(strip_path)],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertFalse(is_qualified(strip_path))

            import numpy as np
            pred_path = tmp / "pred.npz"
            np.savez(pred_path, points=strip.wraps[2][:50])
            proc2 = subprocess.run(
                [sys.executable, str(REPO_ROOT / "score_strip.py"),
                 "--strip", str(strip_path), "--mode", "points",
                 "--pred", str(pred_path),
                 "--from-wrap", "1", "--direction", "front"],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            self.assertEqual(proc2.returncode, 0, proc2.stderr)
            self.assertIn("UNQUALIFIED", proc2.stderr)


if __name__ == "__main__":
    unittest.main()
