"""make_card.py tests: end-to-end on synthetic verdicts, checking the
rendered card contains the expected pass/fail lines (task requirement).
"""
import json
import tempfile
import unittest
from pathlib import Path

import card_config
import make_card


def sample_map_verdict(overall_pass: bool) -> dict:
    pct = 99.5 if overall_pass else 42.0
    return {
        "schema_version": "v0", "tool": "vet_map.py", "tool_version": card_config.TOOL_VERSION,
        "config_hash": card_config.config_hash(),
        "evaluator": {"source_sha256": "e" * 64},
        "input": {"map": "claim_map.npy", "map_sha256": "a" * 64, "map_size_bytes": 123, "map_shape": [512, 512], "map_dtype": "float64", "bbox": [10, 10, 110, 110], "px_um": 8.0, "seed": 0},
        "status": "ok",
        "checks": {
            "degenerate": {"pass": True, "map_constant": False, "blank": False, "saturated": False, "message": "ok"},
            "letter_energy": {
                "pass": overall_pass, "value": 0.01234,
                "null_stats": {"n": 100, "mean": 0.001, "std": 0.0005, "min": 0.0, "max": 0.005},
                "percentile": pct, "threshold_percentile": 99.0,
            },
            "structure": {
                "pass": overall_pass, "value": 4,
                "null_stats": {"n": 100, "mean": 1.0, "std": 0.5, "min": 0, "max": 3},
                "percentile": pct, "threshold_percentile": 99.0, "component_area_px_range": [10, 1000],
            },
        },
        "overall": {"pass": overall_pass},
    }


def sample_pipeline_verdict(overall_pass: bool) -> dict:
    return {
        "schema_version": "v0", "tool": "vet_pipeline.py", "tool_version": card_config.TOOL_VERSION,
        "config_hash": card_config.config_hash(),
        "evaluator": {"source_sha256": "p" * 64, "vet_map_source_sha256": "e" * 64},
        "input": {"manifest": "fixtures/manifest.json", "manifest_sha256": "b" * 64, "outputs": "claimant_outputs/", "px_um": 8.0},
        "status": "ok",
        "pareidolia_rate": 0.0 if overall_pass else 1.0,
        "sensitivity": 1.0,
        "thresholds": {"max_pareidolia_rate": 0.0, "min_sensitivity": 0.5},
        "n_blank_evaluated": 4, "n_positive_evaluated": 2, "missing_ids": [],
        "fixtures": [
            {"id": "blank_1", "kind": "blank", "status": "ok", "fired": not overall_pass, "best_percentile": 10.0, "output_sha256": "c" * 64, "cells_evaluated": 9, "cells_total": 9},
            {"id": "positive_1", "kind": "positive", "status": "ok", "fired": True, "best_percentile": 99.9, "output_sha256": "d" * 64, "cells_evaluated": 1, "cells_total": 9},
        ],
        "overall": {"pass": overall_pass},
    }


def error_map_verdict() -> dict:
    return {
        "schema_version": "v0", "tool": "vet_map.py", "tool_version": card_config.TOOL_VERSION,
        "config_hash": card_config.config_hash(),
        "input": {"map": "x.npy", "bbox": None, "px_um": 8.0, "seed": 0},
        "status": "error", "error": "bbox exceeds map bounds", "overall": {"pass": False},
    }


class TestRenderCard(unittest.TestCase):
    def test_both_pass_renders_pass_everywhere(self):
        card = make_card.render_card(sample_map_verdict(True), sample_pipeline_verdict(True))
        self.assertIn("Overall (claim check): PASS", card)
        self.assertIn("Overall (pipeline check): PASS", card)
        self.assertIn("## Overall verdict: PASS", card)
        self.assertIn(make_card.FOOTER, card)
        self.assertIn(card_config.config_hash(), card)
        self.assertIn(card_config.TOOL_VERSION, card)
        self.assertIn("Map SHA-256", card)
        self.assertIn("Output SHA-256", card)

    def test_one_failing_check_fails_overall(self):
        card = make_card.render_card(sample_map_verdict(False), sample_pipeline_verdict(True))
        self.assertIn("Overall (claim check): FAIL", card)
        self.assertIn("Overall (pipeline check): PASS", card)
        self.assertIn("## Overall verdict: FAIL", card)

    def test_mismatched_evaluator_hashes_are_flagged(self):
        map_verdict = sample_map_verdict(True)
        pipeline_verdict = sample_pipeline_verdict(True)
        pipeline_verdict["evaluator"]["vet_map_source_sha256"] = "f" * 64
        card = make_card.render_card(map_verdict, pipeline_verdict)
        self.assertIn("calibrated check-engine", card)

    def test_map_only(self):
        card = make_card.render_card(sample_map_verdict(True), None)
        self.assertIn("Claim check", card)
        self.assertNotIn("Pipeline check", card)
        self.assertIn("## Overall verdict: PASS", card)

    def test_pipeline_only(self):
        card = make_card.render_card(None, sample_pipeline_verdict(False))
        self.assertNotIn("Claim check", card)
        self.assertIn("Pipeline check", card)
        self.assertIn("## Overall verdict: FAIL", card)

    def test_error_status_map_verdict_renders_fail(self):
        card = make_card.render_card(error_map_verdict(), None)
        self.assertIn("ERROR", card)
        self.assertIn("Overall (claim check): FAIL", card)
        self.assertIn("## Overall verdict: FAIL", card)


class TestCLIIntegration(unittest.TestCase):
    def test_end_to_end_writes_expected_lines(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            map_path, pipe_path, out_path = d / "map_verdict.json", d / "pipe_verdict.json", d / "VETTING_CARD.md"
            map_path.write_text(json.dumps(sample_map_verdict(True)))
            pipe_path.write_text(json.dumps(sample_pipeline_verdict(True)))

            rc = make_card.main([
                "--map-verdict", str(map_path), "--pipeline-verdict", str(pipe_path), "--out", str(out_path),
            ])
            self.assertEqual(rc, 0)
            text = out_path.read_text()
            self.assertIn("## Overall verdict: PASS", text)
            self.assertIn("letter_energy", text)
            self.assertIn("pareidolia_rate", text)
            self.assertIn(make_card.FOOTER, text)

    def test_requires_at_least_one_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "VETTING_CARD.md"
            rc = make_card.main(["--out", str(out_path)])
            self.assertEqual(rc, 2)

    def test_map_verdict_only_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            map_path, out_path = d / "map_verdict.json", d / "VETTING_CARD.md"
            map_path.write_text(json.dumps(sample_map_verdict(False)))
            rc = make_card.main(["--map-verdict", str(map_path), "--out", str(out_path)])
            self.assertEqual(rc, 0)
            text = out_path.read_text()
            self.assertIn("Claim check", text)
            self.assertNotIn("Pipeline check", text)
            self.assertIn("## Overall verdict: FAIL", text)


if __name__ == "__main__":
    unittest.main()


class TestCardFlagOnVetMap(unittest.TestCase):
    """--card on vet_map.py must render the same card make_card.py would,
    in one command (the common claimant flow)."""

    def test_card_flag_writes_card(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        import vet_map
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(0)
            np.save(d / "m.npy", rng.random((300, 300)))
            rc = vet_map.main([
                "--map", str(d / "m.npy"), "--bbox", "0,0,120,120",
                "--px-um", "50.0", "--out", str(d / "v.json"),
                "--card", str(d / "card.md"),
            ])
            self.assertIn(rc, (0, 1))  # verdict may fail on noise; tool must not error
            card = (d / "card.md").read_text()
            self.assertIn("Vetting Card", card)
            self.assertIn("letter_energy", card)
