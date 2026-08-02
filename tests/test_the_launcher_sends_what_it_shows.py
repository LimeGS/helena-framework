"""The launcher's controls have to reach the run.

An independent audit found several that did not: a grid step that went into a
dict the API skipped, a backend validated and echoed and never dispatched, a
reason returned in the HTTP reply and written nowhere. Each looked wired from the
browser and each left the run unchanged, which is worse than a missing control --
a missing one is visible.

There was no test over this boundary at all, which is why they survived. This is
that test: it reads the request the browser builds and the command the API builds
from it, and checks that a value chosen in one arrives in the other.

Static reading rather than a live run, deliberately. The defects were not timing
or environment; they were a field name that did not match on either side of a
JSON body, and a parameter with no consumer anywhere. Both are visible in the
source, and a test that needs PostgreSQL and a VC3D binary to catch a typo in a
key is a test nobody runs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()
LAUNCHER = (ROOT / "panel/web/src/routes/Segmentation.tsx").read_text()


def _runs_endpoint() -> str:
    """The body of the handler behind POST /api/segmentation/runs."""
    start = APP.index('@app.post("/api/segmentation/runs")')
    rest = APP[start:]
    # To the next top-level decorator, which is the next endpoint.
    end = rest.index("\n@app.", 1)
    return rest[:end]


def test_the_grid_step_the_form_sends_is_the_one_the_command_gets() -> None:
    """The form puts it in `options`; the command must read it from there.

    It was declared twice -- a top-level field with a default of 2048 and an
    entry in SEGMENTATION_OPTIONS with the same flag -- so the options loop
    skipped it to avoid passing --grid-step twice, and the top-level default was
    what ran. Every grid step typed into the browser was discarded.
    """
    # The form's own option list is where grid_step lives, so the browser sends
    # it inside options rather than beside them.
    assert '"field": "grid_step"' in APP

    handler = _runs_endpoint()
    assert 'request.options.get("grid_step")' in handler, (
        "the handler never reads the grid step out of options, which is where "
        "the form puts it"
    )
    # And the flag is built from that resolution, not from the raw field.
    assert '"--grid-step", str(grid_step)' in handler
    assert '"--grid-step", str(request.grid_step)' not in handler


def test_the_grid_step_is_validated_on_the_path_that_is_used() -> None:
    """Bounds lived on the top-level field, which is not the path in use."""
    handler = _runs_endpoint()
    assert "256 <= grid_step <= 8192" in handler, (
        "the options path takes an unchecked integer straight to a command line"
    )


def test_a_backend_with_no_executor_is_refused_not_relabelled() -> None:
    """The fleet grows on VC3D and only VC3D.

    _worker instantiates VC3DGrowExecutor unconditionally, no task carries a
    backend, and the bootstrap script has no --backend argument. A request naming
    ScrollFiesta therefore grew on VC3D while the reply said ScrollFiesta. The
    queue has to refuse it: a wrong method on a scientific record cannot be
    detected after the fact.
    """
    executors = (ROOT / "framework/stages/01-segmentation/fleet/executor.py").read_text()
    available = set(re.findall(r"^class (\w+GrowExecutor)", executors, re.MULTILINE))
    # Fixtures are excluded from scientific QC and are not a scientific backend.
    assert available == {"VC3DGrowExecutor", "FixtureGrowExecutor"}, (
        f"executors changed: {sorted(available)}. If a comparison backend now has "
        "one, dispatch the task's backend to it and rewrite this test."
    )

    handler = _runs_endpoint()
    assert 'backend["id"] != "vc3d"' in handler, (
        "any backend is accepted, and every one of them runs VC3D"
    )
    assert "501" in handler


def test_the_reason_is_not_advertised_as_kept_unless_it_is() -> None:
    """The field says the reason is kept with the run.

    Request bodies are deliberately absent from the audit log, so if the reason
    reaches neither the task nor a receipt then the sentence under the box is
    false. Either it travels or the form stops claiming it does; this pins the
    two together so they cannot drift apart again.
    """
    promises_kept = "kept with the run" in LAUNCHER
    handler = _runs_endpoint()
    travels = "--reason" in handler or "reason" in handler.split("argv = [")[1].split("]")[0]
    assert promises_kept <= travels, (
        "the form says the reason is kept with the run and the command does not "
        "carry it anywhere"
    )


def test_the_mission_reaches_each_segmentation_task() -> None:
    """A scroll is input, not ownership: two missions may run the same scroll."""
    handler = _runs_endpoint()
    assert '"--mission-id", request.mission_id' in handler
    cli = (ROOT / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    generator = (ROOT / "framework/stages/01-segmentation/fleet/generator.py").read_text()
    postgres = (ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py").read_text()
    assert '"--mission-id"' in cli
    assert '"mission_id": mission_id' in generator
    assert "mission_id,source_snapshot_id,grid_version,cell_id,policy_version" in postgres


def test_the_seed_probe_controls_reach_the_bootstrap() -> None:
    """Mode, fan-out and budget are one experiment and must travel together."""
    for field in (
        "seed_probe_mode",
        "seed_probe_top_k",
        "seed_probe_generations",
    ):
        assert field in LAUNCHER, f"the launcher does not send {field}"
        assert field in APP, f"the API does not accept {field}"

    handler = _runs_endpoint()
    for flag in (
        "--seed-probe-mode",
        "--seed-probe-top-k",
        "--seed-probe-generations",
    ):
        assert flag in handler, f"{flag} is accepted by the panel and sent nowhere"
    assert "--seed-probe-benchmark-receipt" in handler, (
        "the panel can mark select ready but does not bind its approval receipt "
        "to the bootstrap command"
    )
    assert '"--catalog", str(GEOMETRY_CATALOG)' in handler, (
        "the launcher sends the eligible-volume JSON where the fleet requires "
        "the geometry catalog SQLite database"
    )
    assert "--seed-probe-review-owner" in handler, (
        "the panel can mark select ready without binding its review owner"
    )


def test_probe_select_cannot_steer_an_unapproved_lane() -> None:
    """Shadow is observational; select changes the seed and is deliberately narrower."""
    handler = _runs_endpoint()
    assert 'request.seed_probe_mode == "select"' in handler
    assert '{"cost-aware-v2", "deterministic-v2"}' in handler
    assert "Shadow mode is available with every lane" in handler


def test_probe_status_checks_the_additive_migration_before_reading_it() -> None:
    """A rolling panel deploy must still render against the preceding database."""
    assert "to_regclass(%s)" in APP
    for table in (
        "segment_probe_runs",
        "segment_probe_trials",
        "segment_probe_attempts",
        "segment_probe_artifact_sets",
        "segment_probe_evaluations",
        "segment_probe_decisions",
        "segment_probe_promotions",
    ):
        assert table in APP
    assert "the seed-probe migration is not installed" in APP
