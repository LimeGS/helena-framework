"""Seeds a person supplied, and what has to stay true about them.

Manual ground truth enters where a model proposal would, so everything after
discovery screens it identically -- the CT-material gate especially, because a
manual seed skips the m7 prediction and that gate is then the only thing between
somebody's guess and hours of growing.

The separable-population property is the one worth a test: "the fleet found forty
surfaces" is only true if a surface grown from a supplied point can be told from
one the fleet found alone, afterwards, from the record.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet.generator import generate_manual_tasks  # noqa: E402
from fleet.worker import ManualSeedProvider, TaskRoutedSeedProvider  # noqa: E402

SNAPSHOT = {"source_snapshot_id": "src-1", "sample_id": "PHerc826",
            "shape_xyz": [8000, 8000, 20000],
            "m7_uri": "https://example.invalid/m7.zarr"}


class NoSurfaces:
    def surfaces_for_snapshot(self, _identifier):
        return []


def build(points, **overrides):
    arguments = dict(catalog_snapshot_sha256="deadbeef", grid_step=2048,
                     query_radius=64, volume_edge_margin=64,
                     grid_version="ct-l0-manual-v1", policy_version="ink-blind-v1",
                     submitted_by="limegs")
    arguments.update(overrides)
    return generate_manual_tasks(NoSurfaces(), SNAPSHOT, points, **arguments)


def test_one_task_per_point_and_the_point_is_the_centre():
    tasks = build([{"x": 4000, "y": 4000, "z": 10000},
                   {"x": 5000, "y": 4000, "z": 10000}])
    assert len(tasks) == 2
    assert tasks[0]["center_xyz"] == {"x": 4000, "y": 4000, "z": 10000}


def test_the_seed_travels_on_the_task_in_the_shape_the_ct_gate_reads():
    """ct_support reads ct_l0_coordinate and candidate_id off each candidate."""
    candidate = build([{"x": 4000, "y": 4000, "z": 10000}])[0]["manual_candidates"][0]
    assert candidate["ct_l0_coordinate"] == {"x": 4000, "y": 4000, "z": 10000}
    assert candidate["candidate_id"].startswith("manual-")


def test_the_origin_and_the_author_survive_on_the_task():
    """Without these two, a human seed is indistinguishable from a found one."""
    task = build([{"x": 4000, "y": 4000, "z": 10000}])[0]
    assert task["manual_candidates"][0]["seed_origin"] == "human"
    assert task["manual_candidates"][0]["submitted_by"] == "limegs"
    assert task["candidate_discovery"]["seed_origin"] == "human"
    assert task["candidate_discovery"]["provider"] == "manual"


def test_no_score_is_invented_for_a_point_nobody_scored():
    """Absent, not present-and-null.

    This test asserted `surface_score is None` and passed, which is exactly what
    broke the phase: the screen resolves a score through a chain of dict.get
    defaults, a key holding None returns that None rather than the default, and
    float(None) made every manual candidate MALFORMED_COORDINATE_OR_SCORE.
    Inventing a number is still wrong; the fix is to say nothing.
    """
    candidate = build([{"x": 4000, "y": 4000, "z": 10000}])[0]["manual_candidates"][0]
    for key in ("surface_score", "combined_score", "confidence", "score"):
        assert key not in candidate
    assert candidate["score_origin"] == "human_selection"


def test_the_same_point_twice_is_the_same_candidate():
    """So uploading a file again does not queue the same growth twice."""
    first = build([{"x": 4000, "y": 4000, "z": 10000}])[0]
    again = build([{"x": 4000, "y": 4000, "z": 10000}])[0]
    assert first["manual_candidates"][0]["candidate_id"] == \
        again["manual_candidates"][0]["candidate_id"]


@pytest.mark.parametrize("point", [
    {"x": 10, "y": 4000, "z": 10000},          # inside the edge margin
    {"x": 999999, "y": 4000, "z": 10000},      # outside the volume
    {"x": 4000, "y": 4000, "z": 19999},        # past the far z margin
])
def test_a_point_that_cannot_be_grown_from_is_refused(point):
    with pytest.raises(ValueError, match="outside the usable range"):
        build([point])


def test_proximity_to_a_known_surface_is_recorded_and_not_refused():
    """Placing a seed next to existing geometry is a thing people mean to do.

    Extending a partial lamina, or testing whether a join is real. Refusing it
    would be the grid's coverage policy overruling the person who overrode it.
    """
    class OneSurface:
        def surfaces_for_snapshot(self, _identifier):
            return [{"bbox_xyz": [[3990, 3990, 9990], [4010, 4010, 10010]]}]

    tasks = generate_manual_tasks(
        OneSurface(), SNAPSHOT, [{"x": 4000, "y": 4000, "z": 10000}],
        catalog_snapshot_sha256="deadbeef", grid_step=2048, query_radius=64,
        volume_edge_margin=64, grid_version="v", policy_version="p",
        submitted_by="limegs")
    assert len(tasks) == 1
    assert tasks[0]["guaranteed_cell_clearance_voxels"] == 0.0


def test_an_empty_manual_list_is_a_mistake_not_a_finding():
    with pytest.raises(RuntimeError, match="at least one"):
        ManualSeedProvider().discover({"manual_candidates": []})


def test_a_task_is_routed_by_what_it_declares():
    """So a host never needs restarting to grow an uploaded point."""
    class Default:
        def discover(self, _task):
            return {"schema": "default"}

    router = TaskRoutedSeedProvider(Default())
    manual = router.discover({"candidate_discovery": {"provider": "manual"},
                              "manual_candidates": [{"candidate_id": "c1"}]})
    assert manual["schema"] == "campaignx.manual_seed_candidates.v1"
    assert manual["fixture"] is False
    assert router.discover({"candidate_discovery": {"provider": "vc3d-mcp"}})["schema"] \
        == "default"
    # A provider a newer generator invented must not strand on an older worker.
    assert router.discover({"candidate_discovery": {"provider": "not-yet"}})["schema"] \
        == "default"


def test_two_points_in_one_grid_square_are_two_tasks():
    """A task's identity includes cell_id, and a 2048-voxel square holds many points.

    Both of these land in the same grid square, so identifying a manual task by
    that square deduplicated the second one away -- silently, which is the worst
    way to lose an instruction somebody typed. The point is the unit here.
    """
    tasks = build([{"x": 4200, "y": 4100, "z": 8200},
                   {"x": 4300, "y": 4100, "z": 8200}])
    assert len({task["cell_id"] for task in tasks}) == 2


def test_the_same_point_twice_is_still_one_task():
    """The deduplication that matters survives: re-uploading a file is not a
    second request to grow the same point."""
    first = build([{"x": 4200, "y": 4100, "z": 8200}])[0]
    again = build([{"x": 4200, "y": 4100, "z": 8200}])[0]
    assert first["cell_id"] == again["cell_id"]


def test_a_manual_candidate_survives_the_screen(tmp_path):
    """It did not, and the receipt blamed the coordinate.

    The screen resolves a score through a chain of dict.get defaults. A key
    present with None returns that None rather than the default, float(None)
    raises, and the candidate is counted MALFORMED_COORDINATE_OR_SCORE -- after
    the CT gate had already accepted the point. Every manual seed died there, so
    manual seeding never produced a surface.
    """
    from fleet.planner import screen_candidates

    task = build([{"x": 4000, "y": 4000, "z": 10000}])[0]
    screened = screen_candidates({"candidates": task["manual_candidates"]},
                                 {**task, "source": SNAPSHOT})
    assert screened["usable_candidate_count"] == 1, (
        screened["rejection_diagnostics"]["cause_counts"]
        if "cause_counts" in screened["rejection_diagnostics"]
        else screened["rejection_diagnostics"])

