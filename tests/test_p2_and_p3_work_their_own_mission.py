"""A certify or flatten run works the mission that asked for it.

Both were queued from inside a mission and worked the whole control plane:
`--sample` and `--limit` were the only filters the queue passed. Running the
pipeline by hand caught it -- a P3 started in `golden-run` flattened a surface
whose grow task belongs to `unfiled`, and the sheet it published is invisible in
the mission that asked for it, because every view in the panel is mission-scoped.

That is the rule this platform holds to: nothing exists outside a mission. The
backlog now asks the same three-way question the panel asks -- grown by one of
this mission's tasks, derived by one of its ink jobs, or uploaded into it.

Omitting the mission still means the whole control plane, which is what a
fleet-wide sweep from the CLI wants and what a run started inside a mission
must not get.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from job_store import command_for  # noqa: E402


def _job(mission_id: str | None, phase: str = "P3") -> dict:
    parameters = {"limit": 1, "sample": "PHerc826"}
    if phase == "P3":
        parameters["artifact_store"] = "s3://bucket/flattened-v1"
    return {"job_id": f"{phase.lower()}-abc", "phase": phase,
            "sample_id": "PHerc826", "mission_id": mission_id,
            "profile_id": None, "parameters": parameters}


@pytest.mark.parametrize("phase", ["P2", "P3"])
def test_the_command_carries_the_mission_the_job_belongs_to(phase: str) -> None:
    argv = command_for(_job("golden-run", phase), runner="python3", output_dir="/out")

    assert "--mission-id" in argv
    assert argv[argv.index("--mission-id") + 1] == "golden-run"


@pytest.mark.parametrize("phase", ["P2", "P3"])
def test_a_job_with_no_mission_asks_for_no_scope(phase: str) -> None:
    """`unfiled` history predates missions and a CLI sweep has no mission at
    all; neither should be handed an empty filter that matches nothing."""
    argv = command_for(_job(None, phase), runner="python3", output_dir="/out")

    assert "--mission-id" not in argv


def test_the_mission_is_not_a_parameter_a_request_can_set() -> None:
    """It comes off the job row. A mission that could be named in `parameters`
    would let a request work another mission's surfaces while its own row said
    otherwise -- which is the failure this closes, with a signature."""
    job = _job("golden-run")
    job["parameters"]["mission_id"] = "somebody-elses"

    argv = command_for(job, runner="python3", output_dir="/out")

    assert argv[argv.index("--mission-id") + 1] == "golden-run"


def test_both_backlogs_take_a_mission() -> None:
    import inspect

    for method in (PostgresFleetStore.surfaces_without_geometry_verdict,
                   PostgresFleetStore.surfaces_awaiting_flattening):
        assert "mission_id" in inspect.signature(method).parameters, method.__name__


def test_the_predicate_asks_all_three_ways() -> None:
    predicate = PostgresFleetStore.MISSION_SURFACE_PREDICATE

    assert "segment_tasks t" in predicate          # grown by this mission
    assert "ink_jobs j" in predicate               # derived by one of its jobs
    assert "payload ->> 'mission_id'" in predicate  # uploaded into it
    # Three branches, three placeholders, and the callers pass three ids.
    assert predicate.count("%s") == 3


def test_the_sqlite_mirror_asks_what_it_can_see() -> None:
    """No ink_jobs in this database, so no derived surfaces in it either.

    Stated rather than silently dropped: the mirror answers the same question
    about the surfaces it can hold, and the parity tests compare those.
    """
    from fleet.store import FleetStore

    predicate = FleetStore.MISSION_SURFACE_PREDICATE
    assert "tasks t" in predicate
    assert "mission_id" in predicate
    assert predicate.count("?") == 2
    assert "ink_jobs" not in predicate
