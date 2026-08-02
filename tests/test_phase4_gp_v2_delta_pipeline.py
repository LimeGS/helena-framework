import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


prepare = load_module(
    "prepare_gp_v2_delta_test",
    ROOT
    / "framework"
    / "stages"
    / "06-discovery"
    / "scripts"
    / "prepare_gp_v2_delta.py",
)
merge = load_module(
    "merge_gp_v2_manifest_test",
    ROOT
    / "framework"
    / "stages"
    / "05-reconstruction"
    / "scripts"
    / "merge_gp_v2_manifest.py",
)
completion = load_module(
    "validate_gp_v1_completion_test",
    ROOT
    / "framework"
    / "stages"
    / "06-discovery"
    / "scripts"
    / "validate_gp_v1_completion.py",
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Fixture:
    def __init__(self, root: Path):
        self.root = root
        self.coarse = root / "coarse.json"
        self.v1 = root / "v1.json"
        self.v2 = root / "v2.json"
        self.robust_v1 = root / "robust-v1.json"
        self.checkpoint = root / "model.safetensors"
        self.gate = root / "gate-freeze.json"
        self.delta = root / "delta.json"
        self.freeze = root / "freeze.json"
        self.v1_gate = root / "v1-ct.json"
        self.delta_robust = root / "delta-robust.json"
        self.delta_gate = root / "delta-ct.json"
        self.viewer = root / "viewer-manifest.json"
        self.log = root / "orchestrator.log"
        self.checkpoint.write_bytes(b"checkpoint")
        self.gate.write_bytes(b"gate")

        write_json(
            self.coarse,
            {
                "kind": "coarse",
                "status": "COMPLETED_PRIORITIZATION_ONLY",
                "task_count": 3,
                "completed_count": 3,
                "failed_count": 0,
            },
        )
        self.v1_rows = [
            self.window("PHercA", "surface-a", [0, 0, 20, 20], 1),
            self.window("PHercA", "surface-b", [20, 0, 40, 20], 2),
            self.window("PHercB", "surface-c", [0, 20, 20, 40], 3),
        ]
        write_json(
            self.v1,
            {
                "kind": prepare.V1_RANKING_KIND,
                "status": "COMPLETED_PRIORITIZATION_ONLY",
                "source_batch_receipt_sha256": prepare.sha256_file(self.coarse),
                "global_priority": self.v1_rows,
            },
        )
        self.v2_rows = [
            self.v2_window(self.v1_rows[1], 1, 0.95),
            self.v2_window(
                self.window("PHercC", "surface-new", [5, 5, 25, 25], 1),
                2,
                0.90,
            ),
            self.v2_window(self.v1_rows[0], 3, 0.85),
        ]
        write_json(
            self.v2,
            {
                "kind": prepare.V2_RANKING_KIND,
                "status": "COMPLETED_PRIORITIZATION_ONLY",
                "source_batch_receipt_sha256": prepare.sha256_file(self.coarse),
                "frozen_v1_audit_binding": {
                    "sha256": prepare.sha256_file(self.v1)
                },
                "global_priority_v2": self.v2_rows,
            },
        )

        robust_results = []
        for row in self.v1_rows:
            rank = row["global_rank"]
            analysis = root / f"v1-analysis-{rank}.json"
            write_json(analysis, {"rank": rank})
            robust_results.append(
                {
                    "global_rank": rank,
                    "sample_id": row["sample_id"],
                    "surface_id": row["surface_id"],
                    "source_crop_xyxy": row["source_crop_xyxy"],
                    "analysis": str(analysis),
                    "analysis_sha256": prepare.sha256_file(analysis),
                    "screening_outcome": "NO_TEXT_LIKE_SUPPORT_OBSERVED",
                    "route": "NOT_QUEUED_TEXT_LIKE_GATE_FAILED",
                    "glyph_like_candidate_count": 1,
                    "row_band_count": 0,
                }
            )
        write_json(
            self.robust_v1,
            {
                "kind": prepare.ROBUST_KIND,
                "status": "COMPLETED_DIAGNOSTIC_ONLY",
                "global_ranking_sha256": prepare.sha256_file(self.v1),
                "checkpoint": {
                    "sha256": prepare.sha256_file(self.checkpoint)
                },
                "selected_global_ranks": [1, 2, 3],
                "completed_count": 3,
                "results": robust_results,
            },
        )
        self.make_gate(self.v1_gate, "v1-features.csv")
        write_json(self.viewer, {"status": "complete"})
        self.log.write_text("COMPLETE: GP-Scroll1 shortlist pipeline\n")

    @staticmethod
    def window(sample, surface, crop, rank):
        return {
            "sample_id": sample,
            "surface_id": surface,
            "source_crop_xyxy": crop,
            "global_rank": rank,
            "score": 0.5,
        }

    @staticmethod
    def v2_window(row, rank, score):
        value = {
            key: item
            for key, item in row.items()
            if key not in {"global_rank", "score"}
        }
        value.update(
            {
                "global_rank_v2": rank,
                "global_score_v2": score,
                "global_selection_lane_v2": "V2_PRIMARY",
            }
        )
        return value

    def make_gate(self, path: Path, feature_name: str):
        features = self.root / feature_name
        features.write_text("group_id,value\nx,1\n")
        write_json(
            path,
            {
                "kind": merge.GATE_KIND,
                "status": "COMPLETED",
                "rule_sha256": prepare.sha256_file(self.gate),
                "features": str(features),
                "features_sha256": prepare.sha256_file(features),
                "retained_count": 0,
                "downranked_count": 1,
            },
        )

    def prepare(self):
        delta, freeze = prepare.prepare_artifacts(
            coarse_receipt_path=self.coarse,
            v1_ranking_path=self.v1,
            v2_ranking_path=self.v2,
            v1_robust_receipt_path=self.robust_v1,
            checkpoint_path=self.checkpoint,
            gate_freeze_path=self.gate,
            delta_ranking_path=self.delta,
            freeze_path=self.freeze,
            expected_selected_count=3,
            script_root=ROOT,
        )
        prepare.write_exact(self.delta, delta, dry_run=False)
        prepare.write_exact(self.freeze, freeze, dry_run=False)
        return delta, freeze

    def make_delta_results(self):
        delta = json.loads(self.delta.read_text())
        row = delta["global_priority"][0]
        analysis = self.root / "delta-analysis-1.json"
        write_json(analysis, {"rank": 1})
        write_json(
            self.delta_robust,
            {
                "kind": merge.ROBUST_KIND,
                "status": "COMPLETED_DIAGNOSTIC_ONLY",
                "global_ranking_sha256": prepare.sha256_file(self.delta),
                "checkpoint": {
                    "sha256": prepare.sha256_file(self.checkpoint)
                },
                "selected_global_ranks": [1],
                "completed_count": 1,
                "results": [
                    {
                        "global_rank": 1,
                        "sample_id": row["sample_id"],
                        "surface_id": row["surface_id"],
                        "source_crop_xyxy": row["source_crop_xyxy"],
                        "analysis": str(analysis),
                        "analysis_sha256": prepare.sha256_file(analysis),
                        "screening_outcome": (
                            "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"
                        ),
                        "route": "RAW_CT_REVIEW_REQUIRED",
                        "glyph_like_candidate_count": 10,
                        "row_band_count": 2,
                    }
                ],
            },
        )
        self.make_gate(self.delta_gate, "delta-features.csv")


class Phase4GpV2DeltaPipelineTest(unittest.TestCase):
    def test_exact_overlap_is_reused_and_only_delta_is_adapted(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            delta, freeze = fixture.prepare()
            self.assertEqual(len(delta["global_priority"]), 1)
            adapted = delta["global_priority"][0]
            self.assertEqual(adapted["global_rank"], 1)
            self.assertEqual(adapted["v2_global_rank"], 2)
            self.assertEqual(adapted["score"], 0.90)
            self.assertEqual(freeze["selection"]["reused_v1_exact_count"], 2)
            self.assertEqual(freeze["selection"]["delta_compute_count"], 1)
            self.assertEqual(
                [row["provenance"] for row in freeze["selection"]["entries"]],
                [
                    "REUSED_V1_EXACT_WINDOW",
                    "COMPUTE_V2_EXACT_SET_DIFFERENCE",
                    "REUSED_V1_EXACT_WINDOW",
                ],
            )

    def test_maximum_mode_allows_different_v1_and_v2_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fourth = fixture.v2_window(
                fixture.window(
                    "PHercD",
                    "surface-four",
                    [40, 40, 60, 60],
                    1,
                ),
                4,
                0.80,
            )
            value = json.loads(fixture.v2.read_text())
            value["global_priority_v2"].append(fourth)
            write_json(fixture.v2, value)
            delta, freeze = prepare.prepare_artifacts(
                coarse_receipt_path=fixture.coarse,
                v1_ranking_path=fixture.v1,
                v2_ranking_path=fixture.v2,
                v1_robust_receipt_path=fixture.robust_v1,
                checkpoint_path=fixture.checkpoint,
                gate_freeze_path=fixture.gate,
                delta_ranking_path=fixture.delta,
                freeze_path=fixture.freeze,
                expected_selected_count=None,
                maximum_selected_count=4,
                script_root=ROOT,
            )
            self.assertEqual(freeze["v1_ranking"]["selected_count"], 3)
            self.assertEqual(freeze["v2_ranking"]["selected_count"], 4)
            self.assertEqual(freeze["selection"]["expected_v2_count"], 4)
            self.assertEqual(delta["selection"]["delta_compute_count"], 2)

    def test_v2_must_bind_exact_v1_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            value = json.loads(fixture.v2.read_text())
            value["frozen_v1_audit_binding"]["sha256"] = "0" * 64
            write_json(fixture.v2, value)
            with self.assertRaisesRegex(RuntimeError, "exact v1 ranking"):
                fixture.prepare()

    def test_merge_has_one_provenance_for_every_v2_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.prepare()
            fixture.make_delta_results()
            manifest = merge.build_manifest(
                freeze_path=fixture.freeze,
                v1_robust_receipt_path=fixture.robust_v1,
                v1_ct_evaluation_path=fixture.v1_gate,
                delta_robust_receipt_path=fixture.delta_robust,
                delta_ct_evaluation_path=fixture.delta_gate,
            )
            self.assertEqual(
                manifest["status"], "COMPLETED_MERGED_DIAGNOSTIC_ONLY"
            )
            self.assertEqual(manifest["summary"]["merged_entry_count"], 3)
            self.assertEqual(
                [row["v2_global_rank"] for row in manifest["entries"]],
                [1, 2, 3],
            )
            self.assertEqual(
                sum(
                    row["provenance"]
                    == "COMPUTE_V2_EXACT_SET_DIFFERENCE"
                    for row in manifest["entries"]
                ),
                1,
            )

    def test_merge_fails_closed_without_required_delta_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            fixture.prepare()
            with self.assertRaisesRegex(RuntimeError, "required"):
                merge.build_manifest(
                    freeze_path=fixture.freeze,
                    v1_robust_receipt_path=fixture.robust_v1,
                    v1_ct_evaluation_path=fixture.v1_gate,
                    delta_robust_receipt_path=None,
                    delta_ct_evaluation_path=None,
                )

    def test_v1_completion_validator_distinguishes_ready_from_waiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = completion.validate(
                coarse_path=fixture.coarse,
                v1_ranking_path=fixture.v1,
                robust_path=fixture.robust_v1,
                ct_evaluation_path=fixture.v1_gate,
                viewer_manifest_path=fixture.viewer,
                checkpoint_path=fixture.checkpoint,
                gate_freeze_path=fixture.gate,
                orchestrator_log_path=fixture.log,
                expected_tasks=3,
                expected_windows=3,
            )
            self.assertEqual(result["status"], "READY_FOR_V2_DELTA")
            fixture.viewer.unlink()
            with self.assertRaises(completion.StillRunning):
                completion.validate(
                    coarse_path=fixture.coarse,
                    v1_ranking_path=fixture.v1,
                    robust_path=fixture.robust_v1,
                    ct_evaluation_path=fixture.v1_gate,
                    viewer_manifest_path=fixture.viewer,
                    checkpoint_path=fixture.checkpoint,
                    gate_freeze_path=fixture.gate,
                    orchestrator_log_path=fixture.log,
                    expected_tasks=3,
                    expected_windows=3,
                )

    def test_v1_completion_accepts_actual_count_below_frozen_maximum(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Fixture(Path(temporary))
            result = completion.validate(
                coarse_path=fixture.coarse,
                v1_ranking_path=fixture.v1,
                robust_path=fixture.robust_v1,
                ct_evaluation_path=fixture.v1_gate,
                viewer_manifest_path=fixture.viewer,
                checkpoint_path=fixture.checkpoint,
                gate_freeze_path=fixture.gate,
                orchestrator_log_path=fixture.log,
                expected_tasks=3,
                expected_windows=None,
                maximum_windows=4,
            )
            self.assertEqual(result["v1_selected_count"], 3)
            self.assertEqual(result["frozen_maximum_windows"], 4)

    def test_watcher_dry_run_never_requires_runtime_artifacts(self):
        watcher = (
            ROOT
            / "framework"
            / "stages"
            / "03-ink"
            / "scripts"
            / "run_gp_v2_delta_after_v1.sh"
        )
        syntax = subprocess.run(
            ["bash", "-n", str(watcher)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        result = subprocess.run(
            ["bash", str(watcher), "--dry-run", str(ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY_RUN_ONLY", result.stdout)
        self.assertIn("no files written and no jobs launched", result.stdout)
        self.assertFalse(
            (
                ROOT
                / "phase4/expanded_candidate_surface_screen_v1/"
                "GP_SCROLL1_ALL_SHORTLIST_V2_DELTA_FREEZE.json"
            ).exists()
        )


if __name__ == "__main__":
    unittest.main()
