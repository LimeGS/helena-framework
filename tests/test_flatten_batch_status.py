"""P3 worker status must reflect operational failures inside its batch."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.cli import flatten_batch_exit_code  # noqa: E402


def test_operational_flattening_failure_fails_the_worker_job() -> None:
    assert flatten_batch_exit_code({"FLATTENING_FAILED": 1}) == 2


def test_measured_area_rejection_is_not_an_operational_failure() -> None:
    assert flatten_batch_exit_code({"FLATTENING_REJECTED_AREA": 1}) == 0


def test_empty_and_successful_batches_are_successful() -> None:
    assert flatten_batch_exit_code({}) == 0
    assert flatten_batch_exit_code({"FLATTENED": 2}) == 0
