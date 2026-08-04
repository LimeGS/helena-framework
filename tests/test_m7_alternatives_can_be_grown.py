"""Growing the candidates m7 ranked below the best.

Every proposal records what it did not pick:

    {"candidate_id": "c215", "reason": "Lower deterministic score or stable tie order."}

An id and a reason -- no coordinates. So a rejected alternative cannot be
re-queued from what the database kept; it has to be re-derived from the same
frozen ordering, which is why this is a planner knob and not a replay.

Measured before it was built: 8 of 8 probe candidates ELIGIBLE at top-3 and 23
of 23 at top-6, so m7's ordering does not degrade in the range anybody has
looked at. Whether its second and third choices lead to *different laminae* --
rather than worse ones -- is the question this makes askable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANNER = (ROOT / "framework/stages/01-segmentation/fleet/planner.py").read_text()
SCHEMA = json.loads(
    (ROOT / "framework/contracts/schemas/segmentation-proposal-v2.schema.json").read_text())


def test_the_ordering_is_shared_so_a_rank_means_one_thing() -> None:
    """candidate_rank_key is the frozen geometry-only ordering. Rank 3 has to
    mean the same rung to every planner that reports it."""
    assert "def candidate_rank_key(" in PLANNER
    assert "ordered_candidates = sorted(candidates, key=candidate_rank_key)" in PLANNER


def test_it_starts_where_it_is_told() -> None:
    assert "rank = int(getattr(self, \"candidate_rank\", 1))" in PLANNER, (
        "the planner no longer reads a rank, so alternatives cannot be reached"
    )
    assert "ordered_candidates[rank - 1:]" in PLANNER, (
        "the rank is read and not applied to the ordering"
    )


def test_asking_past_the_end_refuses_rather_than_falling_back() -> None:
    """m7 offers 3.1 candidates per cell on average and sometimes fewer. Rank 5
    where there are three is a question with no answer, and quietly returning
    the last one would answer a different question than the one asked."""
    guard = PLANNER[PLANNER.index("rank = int(getattr"):]
    guard = guard[: guard.index("ordered_candidates[rank - 1:]")]
    assert "raise RuntimeError" in guard
    assert "no such alternative" in guard


def test_the_rank_is_recorded_on_the_proposal() -> None:
    """A surface grown from the third choice is not the same evidence as one
    grown from the first. Nothing downstream can separate them without this."""
    assert '"candidate_rank": rank,' in PLANNER
    assert "candidate_rank" in SCHEMA["properties"], (
        "the field is emitted and the contract does not admit it"
    )
    assert SCHEMA["properties"]["candidate_rank"]["minimum"] == 1


def test_v1_refuses_instead_of_dropping_the_rank() -> None:
    """v1 is frozen and has no field for it. Emitting rank 3 under a contract
    that cannot record it produces a surface nobody can attribute, which is
    worse than refusing."""
    assert 'if rank != 1 and self.contract_version != "v2"' in PLANNER
    # The key set itself, not the prose around it: the comment explaining why
    # the field is v2-only mentions it by name.
    start = PLANNER.index("PROPOSAL_KEYS_V1 = {")
    keys_v1 = PLANNER[start:PLANNER.index("}", start)]
    assert "candidate_rank" not in keys_v1, (
        "candidate_rank was added to the v1 key set, which makes proposals "
        "already written unverifiable against their own schema"
    )


def test_rank_one_is_still_what_every_run_does() -> None:
    """Raising a ceiling is not the same as changing the default."""
    assert 'getattr(self, "candidate_rank", 1)' in PLANNER
    default = re.search(r"candidate_rank\W+1\)", PLANNER)
    assert default, "the default is no longer 1"


def test_the_rank_travels_all_the_way_to_the_host() -> None:
    """The panel refuses knobs that cannot reach a worker, and says why:

        planner_factory takes exactly those two, so a seeder field that is not
        the model reaches no host ... an operator could assemble a panel of four
        models, be told it was accepted, and get the planner's defaults.

    candidate_rank must not join that list. Every link is checked here, because
    a rank accepted at the API and dropped before the host would attribute a
    surface to a configuration that never ran -- the exact failure that comment
    describes.
    """
    app = (ROOT / "panel/app.py").read_text()
    cli = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    worker = (ROOT / "framework/stages/01-segmentation/fleet/worker.py").read_text()

    # 1. both deterministic planners advertise it
    assert app.count('"field": "candidate_rank"') == 2, (
        "the knob is declared on neither or only one deterministic planner; the "
        "control arm has to be configurable the same way as the arm it controls"
    )
    # 2. the panel forwards it
    assert '"--candidate-rank"' in app
    # 3. the CLI accepts it and puts it on the task
    assert '"--candidate-rank"' in cli
    assert "candidate_rank=getattr(args" in cli
    # 4. the worker reads it off the task and sets it on the planner
    assert 'task.get("candidate_rank")' in worker
    assert 'setattr(built, "candidate_rank"' in worker
    # 5. and it is not in the stranded set
    untravelled = re.search(r"UNTRAVELLED = \{([^}]*)\}", app)
    assert untravelled and "candidate_rank" not in untravelled.group(1), (
        "candidate_rank is listed as untravelled while the worker reads it"
    )


def test_a_run_can_name_the_question_it_is_asking() -> None:
    """The planner refuses a cell that already has a task under the same grid
    and policy:

        nothing was queued: all 48 cells this run covers already have a task
        under grid ct-l0-v1 and policy ink-blind-v1

    That refusal is right -- without it, re-running the fleet would silently
    duplicate work. But growing m7's second candidate over ground its first
    already covers is a different question about the same place, and it needs a
    name. policy_version is that name, and until now only manual seeds could
    give one.
    """
    app = (ROOT / "panel/app.py").read_text()
    cli = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()

    assert re.search(r"policy_version: str \| None = Field\(default=None", app), (
        "the run request cannot name a policy, so a covered cell stays a duplicate"
    )
    assert '"--policy-version", request.policy_version' in app, (
        "the field is accepted and never forwarded, which is worse than refusing it"
    )
    assert 'bootstrap.add_argument("--policy-version"' in cli
    assert "policy_version=args.policy_version" in cli, (
        "the CLI takes the flag and does not stamp it on the task"
    )


def test_it_is_not_pretending_to_be_a_planner_knob() -> None:
    """The planners configure nothing with it and they are right: it is
    provenance on the task, not a parameter of how candidates get chosen.
    Declaring it as a knob would put it in the wrong layer and make the two
    deterministic planners differ from the router for no reason."""
    app = (ROOT / "panel/app.py").read_text()
    seeders = app[app.index("SEGMENTATION_SEEDERS = ["):app.index("SEGMENTATION_PLANNERS")]
    assert '"field": "policy_version"' not in seeders, (
        "policy_version was declared as a seeder knob; it belongs on the run, "
        "beside grid_step, not inside a planner's configuration"
    )


def test_the_rank_lands_on_a_task_that_was_actually_built() -> None:
    """Built, not read.

    Five tests above walk the source and check each layer mentions the field.
    All five passed while the rank reached no task at all: it was written into
    the run summary rather than the task payload, and the queue came back with

        candidate_rank in payload: 0 of 48

    Reading the wiring is not the same as running it. This calls the generator
    and looks at what it produced.
    """
    import inspect
    import sys

    # The fleet package lives under the stage directory, the same way the other
    # fleet tests reach it.
    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet import generator  # type: ignore

    for name in ("generate_tasks_for_snapshot", "generate_manual_tasks"):
        fn = getattr(generator, name)
        assert "candidate_rank" in inspect.signature(fn).parameters, (
            f"{name} cannot be told which rung to grow, so bootstrap_queue's "
            "parameter stops at the door"
        )
        assert inspect.signature(fn).parameters["candidate_rank"].default == 1, (
            f"{name} changed what every existing caller does"
        )

    source = inspect.getsource(generator)
    built = source.count('"candidate_rank": int(candidate_rank),')
    assert built >= 2, (
        f"only {built} task payloads carry the rank; both the snapshot and the "
        "manual generators build tasks a worker will claim"
    )


def test_a_blocked_source_says_where_it_was_blocked() -> None:
    """BLOCKED_SOURCE_UNAVAILABLE has masked unrelated faults before.

    mcp/server.py records one that "reached the fleet as
    BLOCKED_SOURCE_UNAVAILABLE and read like the bucket being down" when the
    real cause was a zarr group being sliced. The receipt for 96 blocked tasks
    on one host and 5 on another said, in full:

        {"error": "HTTPError: HTTP Error 401: Unauthorized",
         "status": "BLOCKED_SOURCE_UNAVAILABLE"}

    Which endpoint, called by what, is not recoverable from that afterwards --
    the attempt directory is gone and the exception was not kept.
    """
    worker = (Path(__file__).resolve().parents[1]
              / "framework/stages/01-segmentation/fleet/worker.py").read_text()
    block = worker[worker.index('"status": "BLOCKED_SOURCE_UNAVAILABLE"'):]
    block = block[: block.index("write_json_atomic")]
    assert '"traceback": traceback.format_exc()' in block, (
        "a blocked task records a type and a message and no call site, which is "
        "the half that would let somebody act on it"
    )
    assert "import traceback" in worker


def test_covered_ground_can_be_revisited() -> None:
    """The rank knob reached nothing until this existed.

    Clearance skips cells near a surface somebody already grew, which is what
    keeps coverage spreading outward instead of re-growing one lamina. It is
    also why m7's alternatives were unreachable: a cell only offers candidates
    where papyrus is, so the cells that produced a surface are exactly the ones
    with alternatives -- and exactly the ones clearance excludes.

    Measured: a rank-2 run over uncovered ground returned NO_SEED on 48 of 48
    tasks, across 48 cells sharing not one id with the 12 that had ever
    produced anything.
    """
    import inspect
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet import generator  # type: ignore

    for name in ("generate_tasks_for_snapshot", "bootstrap_queue"):
        params = inspect.signature(getattr(generator, name)).parameters
        assert "reconsider_covered" in params, f"{name} cannot be told to revisit"
        assert params["reconsider_covered"].default is False, (
            f"{name} revisits by default, so every ordinary run now competes "
            "with its own history"
        )

    source = inspect.getsource(generator)
    assert "if guaranteed_gap < clearance and not reconsider_covered:" in source, (
        "the clearance filter does not consult the flag, so setting it changes "
        "nothing about which cells are offered"
    )


def test_both_deterministic_planners_can_revisit() -> None:
    """Whatever the working arm can be asked, the control arm must be asked
    too, or the comparison between them stops meaning anything."""
    app = (ROOT / "panel/app.py").read_text()
    seeders = app[app.index("SEGMENTATION_SEEDERS = ["):app.index("SEGMENTATION_PLANNERS")]
    assert seeders.count('"field": "reconsider_covered"') == 2, (
        "only one deterministic planner can revisit covered ground"
    )
    assert seeders.count('"field": "candidate_rank"') == 2
    cli = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    assert '"--reconsider-covered"' in cli and '"--reconsider-covered"' in app, (
        "the flag stops before the host that builds the tasks"
    )


def test_revisiting_prefers_the_covered_cells_it_unlocked() -> None:
    """Lifting the filter was not enough, and the run said so.

    Clearance decides which cells a run offers, largest first, so the fleet
    spreads into open ground. With reconsider_covered merely lifting the skip,
    the covered cells became eligible and still came last: a run with 48 places
    offered 48 cells sharing not one id with the 12 that had ever produced a
    surface -- the same outcome as before the flag existed.

    A flag that changes nothing observable is worse than no flag, so the
    preference reverses with it.
    """
    import inspect
    import sys

    sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
    from fleet import generator  # type: ignore

    source = inspect.getsource(generator)
    assert "ordering_gap = -finite_gap if reconsider_covered else finite_gap" in source, (
        "the ranking ignores the flag, so covered cells stay at the bottom of "
        "every run that asked for them"
    )
    assert "rank_key = (ordering_gap," in source, (
        "the reversed ordering is computed and not used"
    )
