"""Two bounds on the same run must not contradict each other in silence.

The preflight now survives a source outage by going back to the queue, bounded
by `PREFLIGHT_MAXIMUM_REQUEUES` and delayed by `PREFLIGHT_OUTAGE_RETRY_SECONDS`.
The control harness independently bounds how long it will wait for that same
preflight, with `PREFLIGHT_WAIT_SECONDS`.

Nothing connects them. Set the budget to three and the wait to two hours, and a
job is entitled to four attempts of about seventy minutes inside a window that
fits fewer than two -- so the requeues past the first can never be used, and the
run reports PREFLIGHT_DID_NOT_FINISH_WITHIN_THE_WAIT instead of the outage that
actually caused it. The bound that looks generous is the one that is dead.

This does not pick the numbers. It asserts they mean the same thing: the wait
has to cover every attempt the budget permits, at the slowest preflight this
program has actually measured. Change either constant and this fails until the
other one agrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))
sys.path.insert(0, str(ROOT / "scripts/harness"))

from fleet.preflight_worker import (  # noqa: E402
    PREFLIGHT_MAXIMUM_REQUEUES,
    PREFLIGHT_OUTAGE_RETRY_SECONDS,
)
import run_first_letters_positive_control as harness  # noqa: E402

# The slowest complete preflight this deployment has produced, from
# three runs of the same control, days apart. Not a guess and not a
# limit -- a measurement, which is the only thing that makes the arithmetic
# below mean anything.
SLOWEST_MEASURED_PREFLIGHT_SECONDS = 4184


def test_the_wait_covers_every_attempt_the_budget_allows() -> None:
    attempts = PREFLIGHT_MAXIMUM_REQUEUES + 1
    needed = (attempts * SLOWEST_MEASURED_PREFLIGHT_SECONDS
              + PREFLIGHT_MAXIMUM_REQUEUES * PREFLIGHT_OUTAGE_RETRY_SECONDS)

    assert harness.PREFLIGHT_WAIT_SECONDS >= needed, (
        f"the budget allows {attempts} attempts needing {needed}s, but the "
        f"harness stops waiting at {harness.PREFLIGHT_WAIT_SECONDS}s -- so the "
        "requeues past the first can never be used and the run reports a "
        "timeout instead of the outage that caused it"
    )


def test_the_wait_is_not_open_ended() -> None:
    """The other direction, kept from the constant's own reasoning: a wait that
    cannot end is worse than one that ends early, because the early one reports
    a timeout somebody can read."""
    attempts = PREFLIGHT_MAXIMUM_REQUEUES + 1
    generous = 2 * (attempts * SLOWEST_MEASURED_PREFLIGHT_SECONDS
                    + PREFLIGHT_MAXIMUM_REQUEUES * PREFLIGHT_OUTAGE_RETRY_SECONDS)

    assert harness.PREFLIGHT_WAIT_SECONDS <= generous, (
        "the wait is more than twice what the budget can consume; room over the "
        "measurement there should be, and no more"
    )


def test_at_least_one_requeue_is_actually_usable() -> None:
    """A budget of zero would make the requeue lane decorative: the outage that
    ended three runs would still end the fourth."""
    assert PREFLIGHT_MAXIMUM_REQUEUES >= 1


def test_the_delay_is_long_enough_to_let_an_outage_lift() -> None:
    """Requeuing instantly just re-reads a source that is still down, and the
    per-read retry in `fleet.retrying` has already covered the instant case."""
    assert PREFLIGHT_OUTAGE_RETRY_SECONDS >= 30
