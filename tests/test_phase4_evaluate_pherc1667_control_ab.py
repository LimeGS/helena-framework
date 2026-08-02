from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "framework" / "stages" / "06-discovery" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_pherc1667_control_ab import (  # noqa: E402
    evaluate_negative,
    evaluate_positive,
)


def test_positive_metrics_prefer_labeled_pixels() -> None:
    prediction = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    coverage = np.ones((2, 2), dtype=bool)
    label = np.array([[False, True], [False, True]])
    metrics = evaluate_positive(prediction, coverage, label)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["labeled_median_probability"] > metrics[
        "unlabeled_median_probability"
    ]


def test_negative_metrics_use_covered_pixels_only() -> None:
    prediction = np.array([[0.1, 1.0], [0.2, 0.3]], dtype=np.float32)
    coverage = np.array([[True, False], [True, True]])
    metrics = evaluate_negative(prediction, coverage)
    assert metrics["maximum_probability"] == np.float32(0.3)
    assert metrics["covered_fraction"] == 0.75
