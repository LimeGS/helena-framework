"""A mission is the project that contains a run. Work does not exist outside one.

The queue accepted `mission_id=None` and the panel's write guard skipped every
one of its checks when the mission was absent -- `if not mission_id: return`.
Five P5 jobs were queued that way in one afternoon, ran on real data, published
real artifacts, and could not be seen from the panel at all: every view there is
scoped to a mission, and MissionGate deliberately renders nothing without one.

So the work went in through a door the interface does not have. That is worse
than an unscoped job. A job nobody can find is a job nobody can review, cancel
or attribute, and the receipts it produces belong to no declared scope.

`unfiled` is not the escape hatch either. It is a read-only view assembled from
the receipts of runs that predate missions -- a description of what happened,
not a scope anybody chose -- which is why amending it is already refused.

The column stays nullable on purpose: those five jobs and everything before
missions existed are history, and a NOT NULL that cannot be satisfied by the
rows already in the table is a migration that does not run. The refusal belongs
at the door, where new work arrives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import InkJobStore, JobRejected  # noqa: E402

P5 = {"checkpoint": "/models/m/model.ckpt", "tiff_dir": "/layers",
      "source_pixel_um": 9.362}


def enqueue(**overrides):
    store = InkJobStore("postgresql://unused")
    body = {"sample_id": "PHerc0332", "phase": "P5",
            "profile_id": "ink-9um-hybrid-3d2d-screening@1.0.0",
            "parameters": dict(P5)}
    body.update(overrides)
    return store.enqueue(**body)


def test_a_job_without_a_mission_is_refused() -> None:
    with pytest.raises(JobRejected) as refused:
        enqueue()

    assert "mission" in str(refused.value).lower()


def test_an_empty_mission_is_the_same_as_none() -> None:
    for absent in ("", "   ", None):
        with pytest.raises(JobRejected):
            enqueue(mission_id=absent)


def test_unfiled_is_not_a_mission_to_queue_into() -> None:
    """It is a view of history, not a scope. Amending it is already refused
    elsewhere for the same reason."""
    with pytest.raises(JobRejected) as refused:
        enqueue(mission_id="unfiled")

    assert "unfiled" in str(refused.value)


def test_the_refusal_is_at_the_queue_not_only_at_the_panel() -> None:
    """The panel is one caller. A harness, a script or a worker reaching the
    store directly must meet the same rule, or the rule is a suggestion."""
    import inspect

    assert "mission" in inspect.getsource(InkJobStore.enqueue).lower()


# -- and the panel refuses earlier, with a reason a person can act on --------


def test_the_panel_write_guard_no_longer_waves_a_missing_mission_through() -> None:
    pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
    sys.path.insert(0, str(ROOT))
    from fastapi import HTTPException

    from panel.app import require_write_sample

    with pytest.raises(HTTPException) as refused:
        require_write_sample(None, "PHerc0332", "P5 job")

    assert refused.value.status_code == 409
    assert "mission" in str(refused.value.detail).lower()


def test_the_job_request_asks_for_one() -> None:
    """A 422 naming the field beats a 409 from three layers down."""
    pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
    sys.path.insert(0, str(ROOT))
    from panel.app import JobRequest

    assert JobRequest.model_fields["mission_id"].is_required(), (
        "mission_id is still optional on the queue's front door")


def test_history_is_left_alone() -> None:
    """Jobs queued before this rule are real and their receipts are evidence.
    A constraint the existing rows cannot satisfy would refuse to migrate."""
    migrations = ROOT / "framework/stages/03-ink/fleet/migrations"
    applied = "\n".join(p.read_text() for p in sorted(migrations.glob("*.sql")))

    assert "ALTER TABLE ink_jobs ALTER COLUMN mission_id SET NOT NULL" not in applied


# Every panel route that creates work. `/api/jobs` is the obvious one and was
# the only one this test knew about at first, which is why the deployment's own
# e2e suite went red on `/api/segmentation/runs` and `/api/segmentation/replan`
# after the rule landed: they create work too, and nothing here was looking.
WORK_CREATING = (
    "/api/jobs",
    "/api/segmentation/runs",
    "/api/segmentation/replan",
    "/api/segmentation/preflight",
    "/api/segmentation/manual-seeds",
    "/api/flattening/run",
    "/api/geometry/certify",
)


def test_nothing_that_queues_work_forgets_to_name_a_mission() -> None:
    """The rule is enforced at the queue, so a caller that forgets gets a
    refusal rather than a surprise -- but a refusal an hour into a control run,
    or in a pipeline's own smoke stage, is a poor way to learn it.

    Read from the source rather than by running these: they need a panel, a
    password, and in some cases hours of GPU. Scanned across the harness scripts
    and the deployment suite both, because the second is where this was missed:
    the first version of this test looked only at /api/jobs in scripts/harness,
    and the e2e suite went red on /api/segmentation/runs and .../replan, which
    create work too.

    Only calls whose body is written out at the call site are judged. Where the
    body arrives as a variable this cannot follow it, and guessing produces
    false alarms -- it accused the positive control twice, whose bodies do name
    a mission, twenty lines further up. A check that cries wolf teaches people
    to ignore it, so it stays inside what it can actually read.
    """
    offenders = []
    for directory in ((ROOT / "scripts" / "harness"), (ROOT / "tests" / "e2e")):
        for script in sorted(directory.glob("*.py")):
            lines = script.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                for route in WORK_CREATING:
                    # The literal-dict form, which is the one a reader -- and
                    # this test -- can check.
                    if f'"POST", "{route}", {{' not in line.replace(" {", " {"):
                        continue
                    # A window either side: the body follows the call, and a
                    # stand-in panel is named just above it.
                    call = "\n".join(lines[max(0, index - 12):index + 12])
                    if "panel.invalid" in call:
                        # Not a deployment. This one drives a fake opener to
                        # check that an ambiguous mutation is not retried; it
                        # never reaches a route, and a mission_id added to make
                        # this test quiet would be saying something untrue
                        # about what it is testing.
                        continue
                    if "mission_id" not in call:
                        offenders.append(
                            f"{script.name}:{index + 1} posts {route} "
                            "without naming a mission")

    assert not offenders, "\n".join(offenders)


def test_a_backend_with_no_executor_is_refused_before_any_mission_policy() -> None:
    """Whether a request can be honoured at all is decided before whether this
    mission may run work.

    The queue runs VC3D only: a task naming ScrollFiesta would grow on VC3D and
    be recorded under the wrong method, which is an attribution nobody could
    detect afterwards. That 501 sat below the campaign gate, and was reachable
    only because a run without a mission skipped the gate entirely -- so the
    deployment's own test for it passed on the hole this rule closes, and went
    red the moment the request carried a real campaign-bound mission.

    Ordering it first is also the better error. Telling somebody their mission
    lacks positive-control evidence, when the request they sent could never have
    been honoured anyway, sends them to fix the wrong thing.
    """
    pytest.importorskip("fastapi", reason="panel dependencies are in panel/.venv")
    sys.path.insert(0, str(ROOT))
    import inspect

    from panel.app import api_queue_segmentation

    body = inspect.getsource(api_queue_segmentation)
    no_executor = body.index('backend["id"] != "vc3d"')
    # require_write_sample is where this route first consults the mission at
    # all; the campaign gate it leads to lives in a helper, so this is the
    # anchor that is both precise and in this function.
    first_mission_policy = body.index("require_write_sample")

    assert no_executor < first_mission_policy, (
        "the comparison-backend refusal sits behind mission policy, so a "
        "campaign-bound mission answers first and hides it")
