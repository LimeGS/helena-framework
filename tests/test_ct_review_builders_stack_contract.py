from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "framework/stages/04-validation/scripts"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_fixture(root: Path) -> tuple[Path, Path]:
    tiffs = root / "tiffs"
    tiffs.mkdir()
    for index in range(65):
        Image.fromarray(
            np.full((16, 16), index + 1, dtype=np.uint8)
        ).save(tiffs / f"{index:02d}.tif")

    screening = root / "screening"
    screening.mkdir()
    np.save(
        screening / "mean_probability.npy",
        np.linspace(0.1, 0.9, 16 * 16, dtype=np.float32).reshape(16, 16),
    )

    spec = root / "spec.json"
    _write_json(
        spec,
        {
            "groups": [
                {
                    "group_id": "surface-1",
                    "tiff_directory": "tiffs",
                    "central_slice": 32,
                }
            ]
        },
    )
    analysis = root / "analysis.json"
    _write_json(
        analysis,
        {
            "sample_id": "PHerc826",
            "input": {
                "shape_y_x": [16, 16],
                "screening_directory": "screening",
            },
            "text_like_screening": {
                "candidates": [
                    {
                        "candidate_id": "retained",
                        "bbox_xyxy": [2, 2, 6, 6],
                        "high_recall_routing_score": 0.9,
                    },
                    {
                        "candidate_id": "near-miss",
                        "bbox_xyxy": [8, 8, 12, 12],
                        "high_recall_routing_score": 0.8,
                    },
                ]
            },
        },
    )
    adapter = root / "adapter.json"
    _write_json(
        adapter,
        {
            "status": "READY_FOR_FROZEN_CT_FEATURE_EXTRACTION",
            "spec": {"path": "spec.json", "sha256": _sha256(spec)},
            "analyses": [
                {
                    "window_id": "surface-1",
                    "path": "analysis.json",
                    "sha256": _sha256(analysis),
                }
            ],
        },
    )
    decisions = root / "gate" / "decisions.json"
    _write_json(
        decisions,
        [
            {
                "group_id": "surface-1",
                "candidate_id": "retained",
                "retained": True,
                "checks": [],
            },
            {
                "group_id": "surface-1",
                "candidate_id": "near-miss",
                "retained": False,
                "decision": "CT_DEPTH_DIFFUSE_DOWNRANK_LIKELY_FIBER_LAMINAR_CONFOUND",
                "failed_features": ["depth_profile_top3_fraction"],
                "checks": [
                    {
                        "feature": "depth_profile_top3_fraction",
                        "value": 0.11,
                        "threshold": 0.12,
                        "operator": ">=",
                        "passed": False,
                    }
                ],
            },
        ],
    )
    gate = root / "gate" / "evaluation.json"
    _write_json(
        gate,
        {
            "status": "COMPLETED",
            "row_count": 2,
            "retained_count": 1,
            "artifacts": {
                "decisions": decisions.name,
                "decisions_sha256": _sha256(decisions),
            },
        },
    )
    return adapter, gate


def test_both_ct_review_builders_accept_the_ordered_stack_contract(tmp_path: Path):
    adapter, gate = _review_fixture(tmp_path)
    cases = [
        (
            "build_high_recall_retained_review.py",
            "retained-review",
            "HIGH_RECALL_RETAINED_REVIEW_RECEIPT.json",
            "REVIEW_EVIDENCE_READY_NO_AUTOMATIC_ACCEPTANCE",
            [],
        ),
        (
            "helena_build_ct_gate_near_miss_review.py",
            "near-miss-review",
            "CT_GATE_NEAR_MISS_AUDIT.json",
            "AUDIT_READY_GATE_DECISIONS_UNCHANGED",
            ["--max-failed-checks", "1", "--limit", "1"],
        ),
    ]

    for script_name, output_name, receipt_name, status, extra in cases:
        output = tmp_path / output_name
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / script_name),
                "--root",
                str(tmp_path),
                "--adapter-receipt",
                str(adapter),
                "--gate-evaluation",
                str(gate),
                "--output",
                str(output),
                "--half-size",
                "2",
                *extra,
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        assert completed.returncode == 0, completed.stdout
        receipt = json.loads((output / receipt_name).read_text(encoding="utf-8"))
        assert receipt["status"] == status
        assert receipt["candidate_count"] == 1
        assert receipt["candidates"][0]["candidate_id"] in {
            "retained",
            "near-miss",
        }
