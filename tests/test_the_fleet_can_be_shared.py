"""What several people sharing these machines needs from the queues.

An audit on 2026-07-31 found the queues themselves correct under contention --
30 jobs, 8 concurrent workers, no duplicates, no losses, evenly spread -- and
three things missing around them. This pins the three.

Fair-share scheduling and restricting priority are deliberately not here. They
are policy, they were deferred, and a test asserting a policy nobody has chosen
would be a test asserting my guess.

  * a segmentation task records who asked for it. Without it the platform can
    report what is running and never whose, so nobody sees their own runs and no
    share of the machines can be worked out at all.
  * the ink queue reads the capability columns it has always carried. A CPU-only
    host claiming a GPU job fails it, burns an attempt, and leaves the queue
    reporting a real-looking verdict on work that never ran.
  * the ink worker can run more than once per host. One instance served five
    phases in series, so a second person's screening waited behind the first
    even with a card idle.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

SEGMENT = ROOT / "framework/stages/01-segmentation/fleet"
MIGRATION = (SEGMENT / "migrations/001_postgresql.sql").read_text()
GENERATOR = (SEGMENT / "generator.py").read_text()
PG_STORE = (SEGMENT / "postgres_store.py").read_text()
CLI = (SEGMENT / "cli.py").read_text()
PANEL = (ROOT / "panel/app.py").read_text()


# --------------------------------------------------------------------------
# Whose work is this
# --------------------------------------------------------------------------

def test_a_segmentation_task_has_somewhere_to_record_its_owner() -> None:
    """ink_jobs has carried created_by since it was written. segment_tasks --
    the larger consumer of GPU time -- carried nothing."""
    assert re.search(
        r"ALTER TABLE segment_tasks ADD COLUMN IF NOT EXISTS created_by text",
        MIGRATION), "there is no owner column on segmentation tasks"
    # Additive and defaulted, so a rolling deploy against the previous database
    # does not fail and the existing backlog stays insertable.
    owner = MIGRATION[MIGRATION.index("created_by text"):][:200]
    assert "NOT NULL DEFAULT" in owner
    assert "'unattributed'" in owner, (
        "the backlog would be credited to whoever asks next"
    )


def test_the_owner_reaches_the_row() -> None:
    """A column nothing writes is worse than no column: it reads as answered."""
    insert = PG_STORE[PG_STORE.index("INSERT INTO segment_tasks"):]
    insert = insert[: insert.index("if cursor.rowcount")]
    assert "created_by" in insert.split("VALUES")[0], (
        "the insert does not name the column"
    )
    assert 'task.get("created_by")' in insert, "nothing supplies a value"


def test_every_task_builder_carries_it() -> None:
    """There are two task constructors and a wrapper over them. One that took
    the owner and one that dropped it would attribute half the fleet."""
    assert GENERATOR.count('"created_by": created_by') == 2, (
        "a task constructor builds rows with no owner"
    )
    assert GENERATOR.count("created_by: str | None = None,") == 3


def test_the_panel_passes_the_person_who_asked() -> None:
    """The audit trail already records the caller. The trail is not the queue:
    a row that cannot name its owner cannot be shown to that person."""
    assert '"--queued-by", default=None,' in CLI, "the bootstrap takes no owner"
    assert "created_by=args.queued_by" in CLI, "the flag reaches no task"

    handler = PANEL[PANEL.index("def api_queue_segmentation("):]
    handler = handler[: handler.index("\n@app.")]
    assert '"--queued-by", who_asked(http)' in handler, (
        "the panel queues work without saying who asked for it"
    )
    assert "http: Request" in PANEL[
        PANEL.index("def api_queue_segmentation("):][:120], (
        "the handler cannot see the caller it is meant to name"
    )


# --------------------------------------------------------------------------
# A worker takes only what it can run
# --------------------------------------------------------------------------

def test_the_ink_queue_reads_the_capability_columns_it_carries() -> None:
    """gpu_required and minimum_vram_gb existed on ink_jobs from the start and
    the claim read neither, while the segmentation claim has always filtered on
    exactly these. The asymmetry is what gave it away."""
    store = (ROOT / "framework/stages/03-ink/fleet/job_store.py").read_text()
    claim = store[store.index("    def claim(self, *, worker_id"):]
    claim = claim[: claim.index("\n    def ")]
    assert "gpu_required=false OR" in claim
    assert "minimum_vram_gb <=" in claim

    segmentation = PG_STORE[PG_STORE.index("    def claim("):][:3000]
    assert "gpu_required=false OR" in segmentation, (
        "the segmentation claim stopped filtering; the two must not diverge again"
    )


def test_the_ink_worker_measures_what_it_has() -> None:
    """Binding a constant would be the same bug with more steps.

    has_gpu used to read the once-a-minute host_state() reading (`cards`).
    helena-ink-0 losing its GPU passthrough silently showed that reading was
    not fresh enough: eligibility now reads worker_gpu_visible(), asked fresh
    on every poll, and gpu_visible is the same value -- not a second guess
    that could disagree with the first.
    """
    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    assert "has_gpu=bool(gpu_visible)" in worker
    assert "gpu_vram_gb=max(" in worker
    # Probed before the first claim, or a fresh worker refuses every GPU job for
    # its first minute while the heartbeat catches up.
    assert worker.index("last_probe = host_state(") < worker.index(
        "has_gpu=bool(gpu_visible)")
    # And re-probed inside the loop, not only here before it starts -- the
    # incident this guards against was an answer computed once and trusted
    # for the life of the process.
    assert worker.index("while True:") < worker.index("gpu_visible = worker_gpu_visible()")


# --------------------------------------------------------------------------
# More than one at a time
# --------------------------------------------------------------------------

def test_the_ink_worker_can_run_more_than_once_on_a_host() -> None:
    """One container served P4, P5, P7, P8 and P9 in series."""
    compose = yaml.safe_load(
        (ROOT / "containers/compose/ink.compose.yaml").read_text())
    service = compose["services"]["ink"]
    assert "${HELENA_INK_SLOT" in service["container_name"], (
        "a second instance would collide on the container name"
    )
    command = service["command"]
    assert "--worker-id" in command, (
        "two instances would share an identity, or take one from their pid and "
        "change it on every restart"
    )
    identity = command[command.index("--worker-id") + 1]
    assert "${HELENA_INK_SLOT" in identity, "the identity is not per instance"


def test_the_file_says_how_to_run_a_second_one() -> None:
    """The QC compose earned this rule: -p is required, and a reader who does
    not know that makes the second instance a recreation of the first."""
    text = (ROOT / "containers/compose/ink.compose.yaml").read_text()
    assert "-p helena-ink-1" in text
    assert "HELENA_INK_SLOT=1" in text


def test_two_instances_do_not_write_the_same_output() -> None:
    """The QC workers needed a run root each because they staged surfaces at a
    shared path. This one does not, and the reason is worth pinning: the output
    directory carries the job id."""
    worker = (ROOT / "framework/stages/03-ink/fleet/ink_worker.py").read_text()
    assert re.search(r"output = runs_root / f\"\{job\['sample_id'\]\.lower\(\)\}-\{job_id\}\"",
                     worker), (
        "the output directory no longer includes the job id, so two workers "
        "sharing a runs root can write the same path"
    )
