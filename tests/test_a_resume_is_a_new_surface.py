"""Correcting a surface produces another one; it never edits the first.

The original is the record of what the fleet actually grew and what QC measured
on it. If a resumed grow overwrote it, the catalogue would say a surface passed
certification that no longer exists in the shape it was certified in.

Task identity is (source_snapshot, grid_version, cell, policy_version), so the
resume rides its own policy version and the two cannot collide.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
PLANNER = (ROOT / "framework/stages/01-segmentation/fleet/planner.py").read_text()
EXECUTOR = (ROOT / "framework/stages/01-segmentation/fleet/executor.py").read_text()
APP = (ROOT / "panel/app.py").read_text()


def _resume_command() -> str:
    start = CLI.index("def command_bootstrap_resume")
    return CLI[start:][: CLI[start:].index("\ndef ")]


def test_the_resume_writes_no_surface() -> None:
    command = _resume_command()
    assert "create_tasks" in command, "the resume does not queue a task"
    for forbidden in ("UPDATE segment_surfaces", "record_surface", "import_surface",
                      "finalize"):
        assert forbidden not in command, (
            f"the resume touches {forbidden}, so the surface it continues is not "
            "left as it was measured"
        )


def test_the_resume_has_a_policy_version_of_its_own() -> None:
    """Otherwise it would be the same task as the grow it is correcting."""
    command = _resume_command()
    assert '"policy_version": args.policy_version' in command
    assert 'default="resume-corrections-v1"' in CLI, (
        "the resume shares the default policy version with a fresh grow, which "
        "makes it the same task identity and a silent no-op"
    )
    # And it says which surface it continues, so the pair is traceable.
    assert '"resumes_surface": args.surface' in command


def test_the_chain_the_executor_reads_is_actually_written() -> None:
    """optional_grow_flags read these long before anything produced them."""
    assert 'locked_plan.get("resume_from")' in EXECUTOR
    assert 'locked_plan.get("corrections")' in EXECUTOR
    # task -> packet
    assert '"resume_from": task["resume_from"]' in PLANNER, (
        "the packet drops resume_from, so --resume never reaches VC3D"
    )
    # packet -> locked plan
    assert '"resume_from": packet["resume_from"]' in PLANNER
    assert '"corrections": packet["corrections"]' in PLANNER


def test_the_planner_cannot_choose_what_to_resume() -> None:
    """Which artifact gets continued is an assertion, not a model's suggestion.

    One assertion, on where the value comes from. An earlier version also tried
    to prove the packet block mentions no proposal, which needed a text slice
    that kept swallowing the code after it -- a check too brittle to trust is
    worse than no check.
    """
    assert 'task["resume_from"]' in PLANNER, "resume_from does not come from the task"
    assert 'proposal["resume_from"]' not in PLANNER, (
        "resume_from is read off a proposal, so a planner could name the surface "
        "to continue"
    )
    assert 'proposal.get("resume_from")' not in PLANNER


def test_an_import_is_never_recorded_as_grown_here() -> None:
    """The origin split the panel draws everywhere depends on this."""
    start = APP.index('@app.post("/api/segmentation/import")')
    handler = APP[start:][: APP[start:].index("\n@app.", 1)]
    assert "IMPORTED_COVERAGE" in handler
    assert "attempt_id" not in handler, (
        "an imported surface is being given an attempt, which is exactly what "
        "makes the aggregations count it as fleet output"
    )
    # And it must carry a hash: an unhashed patch is a claim, not a record.
    assert "len(digest) != 64" in handler
