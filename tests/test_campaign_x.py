import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "campaign_x.py"
SPEC = importlib.util.spec_from_file_location("campaign_x", MODULE)
cx = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(cx)


class CampaignXTest(unittest.TestCase):
    def test_all_structural_slots_are_bounded(self):
        shape = [24297, 8343, 8343]
        for axial in range(8):
            for structural in ("outer", "middle", "inner"):
                region = cx.planned_region(shape, axial, structural)
                self.assertEqual(region["coordinate_space"], "ct_l0_xyz")
                self.assertGreaterEqual(min(region["radius"].values()), 32)
                self.assertLessEqual(max(region["radius"].values()), 64)

    def test_pherc1203_firewall(self):
        original = cx.PHASE0
        with tempfile.TemporaryDirectory() as tmp:
            cx.PHASE0 = Path(tmp)
            (cx.PHASE0 / "target_contamination_ledger.json").write_text(json.dumps({"entries": [{"higher_resolution_sibling_uri": cx.FORBIDDEN_1203}]}))
            with self.assertRaises(ValueError):
                cx.forbid_uri(cx.FORBIDDEN_1203)
        cx.PHASE0 = original

    def test_job_metrics_uses_rfc3339_timestamps(self):
        metrics = cx.job_metrics({"state": "succeeded", "started_at": "2026-07-15T00:00:00Z", "finished_at": "2026-07-15T00:01:02Z"})
        self.assertEqual(metrics["runtime_seconds"], 62.0)

    def test_build_plan_is_idempotent_after_freeze(self):
        original_phase1, original_load_eligible = cx.PHASE1, cx.load_eligible
        entries = [{
            "sample_id": sample,
            "surface_prediction_uri": f"https://example.invalid/{sample}.zarr",
            "voxel_size_um": 1.0,
            "shape_zyx": [8192, 4096, 4096],
        } for sample in cx.SAMPLES]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                cx.PHASE1 = Path(tmp)
                cx.load_eligible = lambda: entries
                cx.build_plan()
                first = (cx.PHASE1 / "seed_plan.json").read_bytes()
                cx.build_plan()
                self.assertEqual((cx.PHASE1 / "seed_plan.json").read_bytes(), first)
            finally:
                cx.PHASE1, cx.load_eligible = original_phase1, original_load_eligible

    def test_build_plan_refuses_to_overwrite_different_frozen_plan(self):
        original_phase1, original_load_eligible = cx.PHASE1, cx.load_eligible
        entries = [{
            "sample_id": sample,
            "surface_prediction_uri": f"https://example.invalid/{sample}.zarr",
            "voxel_size_um": 1.0,
            "shape_zyx": [8192, 4096, 4096],
        } for sample in cx.SAMPLES]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                cx.PHASE1 = Path(tmp)
                cx.load_eligible = lambda: entries
                cx.build_plan()
                path = cx.PHASE1 / "seed_plan.json"
                frozen = json.loads(path.read_text())
                frozen["slots"][0]["seed_id"] = "tampered"
                path.write_text(json.dumps(frozen))
                with self.assertRaises(ValueError):
                    cx.build_plan()
            finally:
                cx.PHASE1, cx.load_eligible = original_phase1, original_load_eligible


    def test_structured_raises_for_tool_error_instead_of_returning_empty_object(self):
        with self.assertRaisesRegex(RuntimeError, "candidate region touches"):
            cx.structured({
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "candidate region touches more than 8 chunks"}],
            }
        })


    def test_structured_rejects_missing_structured_content(self):
        with self.assertRaisesRegex(RuntimeError, "no structured object"):
            cx.structured({"result": {}})


if __name__ == "__main__":
    unittest.main()
