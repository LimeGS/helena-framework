"""Acting on the diagnosis the planner already writes.

169 of 241 tasks on this control plane ended NO_SEED. The worker records for
each one how many candidates the provider offered and which screen removed them,
the panel displays it, and nothing ever used it to decide anything: the cell was
terminal, the next bootstrap picked cells by distance from known surfaces, and
the fleet explored without learning.

A replan is a new task over the same cell under a different policy. What is
tested here is that it cannot quietly do nothing, which is the failure mode
task identity makes easy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.generator import replan_no_seed_cells  # noqa: E402

CELL = {
    "task_id": "old-task", "cell_id": "r00002c00007a00002",
    "grid_version": "ct-l0-grid-1024-edge512-stratified-v1.0.0",
    "policy_version": "pherc826-m7-recenter-mcp27-v1.0.0",
    "source_snapshot_id": "snap-1", "sample_id": "PHerc826",
    "payload": {"cell_id": "r00002c00007a00002", "state": "NO_SEED",
                "task_id": "old-task", "center_xyz": {"x": 1, "y": 2, "z": 3},
                "planner": "deterministic"},
    "causes": ["NO_M7_CANDIDATES"], "raw_candidate_count": 0,
}


class Store:
    def __init__(self, cells):
        self.cells = cells
        self.created: list[dict] = []

    def initialize(self):
        return None

    def no_seed_cells(self, *, sample_id=None, causes=None, limit=50):
        if causes:
            return [c for c in self.cells if set(causes) & set(c["causes"])]
        return list(self.cells)

    def create_tasks(self, tasks):
        self.created = list(tasks)
        return len(self.created), len(self.created)


def test_a_replan_under_the_same_identity_is_refused():
    """Task identity is (snapshot, grid, cell, policy) behind an ON CONFLICT DO
    NOTHING, so re-queueing under the same policy inserts nothing and reports
    success -- the exact shape of a button that looks like it worked."""
    store = Store([CELL])
    receipt = replan_no_seed_cells(
        store, grid_version=CELL["grid_version"],
        policy_version=CELL["policy_version"], planner="deterministic-v2")
    assert receipt["queued"] == 0
    assert receipt["considered"] == 1
    assert store.created == []


def test_a_replan_carries_the_new_policy_and_what_it_replans():
    store = Store([CELL])
    receipt = replan_no_seed_cells(
        store, grid_version="replan-r01", policy_version="replan-2026-07-28",
        planner="laguna-v2", sample_id="PHerc826")
    assert receipt["queued"] == 1
    queued = store.created[0]
    assert queued["grid_version"] == "replan-r01"
    assert queued["policy_version"] == "replan-2026-07-28"
    assert queued["planner"] == "laguna-v2"
    assert queued["cell_id"] == CELL["cell_id"]
    # Traceable back to the attempt that found nothing, and to why.
    assert queued["replan_of"] == "old-task"
    assert queued["replan_reason"] == ["NO_M7_CANDIDATES"]
    # The old task's own identity must not travel with it.
    assert "task_id" not in queued
    assert "state" not in queued


def test_a_cause_filter_selects_what_to_re_ask():
    """A cell where the provider offered nothing and a cell where eight
    candidates were rejected on clearance are different problems."""
    store = Store([CELL])
    receipt = replan_no_seed_cells(
        store, grid_version="replan-r01", policy_version="replan-2026-07-28",
        causes=["MALFORMED_COORDINATE_OR_SCORE"])
    assert receipt["queued"] == 0


def test_a_replan_needs_its_own_versions():
    for missing in ({"grid_version": "", "policy_version": "p"},
                    {"grid_version": "g", "policy_version": ""}):
        with pytest.raises(ValueError):
            replan_no_seed_cells(Store([CELL]), **missing)
