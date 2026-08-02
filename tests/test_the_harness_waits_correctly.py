"""The client every script outside this platform drives it with.

Three scripts had their own cookie jar, their own way of reporting a 400 and
their own idea of when a job is finished -- and one of those ideas was wrong in
a way that made a phase look instant: the smoke test polled a key no response
has ever carried, so its wait loop broke on the first pass and P1 appeared to
grow four surfaces in under a second.

No network here. What is tested is the waiting: that a terminal state ends it,
that a failure is a result rather than an exception, and that a timeout says so
instead of returning something that looks like an answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/harness"))

from panel_client import Panel, PanelError, TERMINAL_JOB_STATES  # noqa: E402


class Fake(Panel):
    """A panel whose answers are scripted, so the loop is what is under test."""

    def __init__(self, answers):
        super().__init__("http://example.invalid", timeout=1)
        self.answers = list(answers)
        self.asked = 0

    def call(self, method, path, body=None):  # noqa: ARG002
        self.asked += 1
        return self.answers.pop(0) if self.answers else {"jobs": []}


def jobs(state: str, job_id: str = "p4-1") -> dict:
    return {"jobs": [{"job_id": job_id, "state": state, "result": {"exit_code": 0}}]}


def test_every_terminal_state_ends_the_wait():
    """Terminal means the queue is done with it. A caller that treated `failed`
    as an exception would lose the result it needs to report."""
    for state in TERMINAL_JOB_STATES:
        panel = Fake([jobs(state)])
        outcome = panel.wait_for_job("p4-1", minutes=1, tick=0)
        assert outcome["state"] == state


def test_it_keeps_asking_while_the_job_is_running():
    panel = Fake([jobs("pending"), jobs("running"), jobs("succeeded")])
    assert panel.wait_for_job("p4-1", minutes=1, tick=0)["state"] == "succeeded"
    assert panel.asked == 3


def test_a_job_that_never_finishes_says_so():
    """Rather than returning the last thing it saw, which a caller would read as
    an answer."""
    panel = Fake([jobs("running")] * 3)
    with pytest.raises(TimeoutError):
        panel.wait_for_job("p4-1", minutes=0, tick=0)


def test_another_job_finishing_is_not_this_one_finishing():
    panel = Fake([jobs("succeeded", job_id="p4-other"), jobs("succeeded")])
    assert panel.wait_for_job("p4-1", minutes=1, tick=0)["job_id"] == "p4-1"


def test_waiting_on_a_condition_returns_what_it_saw():
    seen = iter([0, 0, 7])
    panel = Fake([])
    assert panel.wait_until(lambda: next(seen), minutes=1, tick=0) == 7


def test_a_condition_that_never_holds_returns_nothing_rather_than_lying():
    panel = Fake([])
    assert panel.wait_until(lambda: False, minutes=0, tick=0) is None


def test_a_refusal_carries_the_status_and_the_body():
    """A smoke test that dies with a traceback tells you less than one that
    prints what the panel said."""
    failure = PanelError("POST", "/api/jobs", 400, '{"detail":"unknown parameters"}')
    assert failure.status == 400
    assert "unknown parameters" in str(failure)
    assert "/api/jobs" in str(failure)


def test_the_smoke_harness_does_not_call_itself():
    """`panel.call = call` where `call` invokes `panel.call` is a wrapper that
    wraps itself. The smoke test died at sign-in with a RecursionError from the
    day the shared client was extracted until 2026-07-28, which is a long time
    for a test nobody could run."""
    source = (ROOT / "scripts/harness/smoke_p0_p4.py").read_text()
    body = source[source.index("def call(method"):source.index("panel.call = call")]
    assert "panel.call(" not in body, "the wrapper calls the attribute it replaces"
    assert "over_http(" in body
