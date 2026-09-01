"""A gate answers "may this be certified", not "may this run".

This is an exploration framework with a dynamic workflow. Wanting to run one
phase, today, against something arbitrary -- mid-campaign, with an upload that
is bound to nothing -- is an ordinary thing to want, and the platform refusing
it outright is the platform deciding what questions may be asked. What such a
run does not get is a receipt: no certification, no place in the chain of trust.

So there are two lanes. In CERTIFIED, every precondition holds exactly as it
does today; nothing here relaxes one. In EXPLORATORY, an unmet chain-of-trust
precondition no longer stops the work -- it is recorded, and the output says, in
the document itself, that it certifies nothing and why.

The permissive half is the easy half. The half that has to be right is the
exclusion: an exploratory output must be structurally unable to be read as
evidence by a certified run. Without that, this is an acceptance-gate bypass
wearing a friendlier name.

Two kinds of refusal are NOT modes and never become one:

* integrity -- a tampered binding, a drifted identity, a hash that does not
  match. Running anyway means acting on evidence known to be corrupt, and the
  answer would be wrong rather than merely uncertified.
* authorization -- who may act, and on what. A lane is not a login.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.contracts import execution_mode  # noqa: E402


# -- the two lanes -------------------------------------------------------------

def test_certified_is_what_you_get_when_you_ask_for_nothing() -> None:
    """Every existing caller passes no mode, and must behave exactly as before."""
    trust = execution_mode.Trust()
    assert trust.mode == execution_mode.CERTIFIED
    assert trust.blocks("no selected P0") is True


def test_an_exploratory_run_is_not_stopped_by_an_unmet_precondition() -> None:
    trust = execution_mode.Trust(execution_mode.EXPLORATORY)
    assert trust.blocks("no selected P0") is False
    assert trust.blocks("the mission has no scrolls") is False
    assert trust.unmet == ["no selected P0", "the mission has no scrolls"]


def test_an_exploratory_run_that_met_everything_still_certifies_nothing() -> None:
    """The lane is the caller's declaration, not a score it earns on the way.

    Otherwise the same request would certify or not depending on the state of
    the system when it happened to run, which is the opposite of a chain of
    trust.
    """
    trust = execution_mode.Trust(execution_mode.EXPLORATORY)
    assert trust.unmet == []
    assert trust.certified is False


def test_a_certified_run_certifies_only_if_nothing_was_recorded() -> None:
    trust = execution_mode.Trust()
    assert trust.certified is True


def test_an_unknown_mode_is_refused_rather_than_treated_as_exploratory() -> None:
    """A typo must not silently open the permissive lane."""
    with pytest.raises(execution_mode.ExecutionModeError):
        execution_mode.Trust("PERMISSIVE")
    assert execution_mode.parse(None) == execution_mode.CERTIFIED
    assert execution_mode.parse("exploratory") == execution_mode.EXPLORATORY


# -- what the output says about itself -----------------------------------------

def test_an_exploratory_document_says_so_in_itself() -> None:
    """Not in a log, not in the request: in the document that outlives both."""
    trust = execution_mode.Trust(execution_mode.EXPLORATORY)
    trust.blocks("no selected P0")

    stamped = trust.stamp({"schema": "campaignx.something.v1", "value": 3})

    assert stamped["value"] == 3, "the measurement is untouched"
    assert stamped["certified"] is False
    assert stamped["execution_mode"] == execution_mode.EXPLORATORY
    assert stamped["uncertified_because"] == ["no selected P0"]
    assert "certifies nothing" in stamped["non_claim"]


def test_a_certified_document_says_that_too() -> None:
    stamped = execution_mode.Trust().stamp({"schema": "campaignx.something.v1"})
    assert stamped["certified"] is True
    assert stamped["execution_mode"] == execution_mode.CERTIFIED
    assert "uncertified_because" not in stamped


def test_stamping_never_overwrites_a_field_the_document_owns() -> None:
    """A receipt with a `certified` of its own means something by it.

    The preflight envelope already did exactly this once: merging a field over a
    document that had its own turned the measurement's answer into the queue's.
    """
    with pytest.raises(execution_mode.ExecutionModeError, match="certified"):
        execution_mode.Trust().stamp({"schema": "x", "certified": "yes"})


# -- the exclusion, which is the part that has to be right ----------------------

def test_a_certified_run_refuses_to_read_an_uncertified_document() -> None:
    exploratory = execution_mode.Trust(execution_mode.EXPLORATORY).stamp({"schema": "x"})

    with pytest.raises(execution_mode.ExecutionModeError, match="not certified"):
        execution_mode.require_certified_input(exploratory, what="P0 artifact")


def test_an_exploratory_run_may_read_anything() -> None:
    """Exploration on top of exploration is the point of the lane."""
    exploratory = execution_mode.Trust(execution_mode.EXPLORATORY).stamp({"schema": "x"})
    downstream = execution_mode.Trust(execution_mode.EXPLORATORY)

    assert downstream.blocks_uncertified_input(exploratory, what="P0 artifact") is False
    assert downstream.unmet == ["P0 artifact is not certified"]


def test_a_document_with_no_stamp_at_all_is_not_certified() -> None:
    """Fails closed. Everything written before this existed carries no stamp, and
    absence of a claim is not a claim."""
    with pytest.raises(execution_mode.ExecutionModeError):
        execution_mode.require_certified_input({"schema": "x"}, what="P0 artifact")
    assert execution_mode.is_certified({"schema": "x"}) is False


def test_a_forged_stamp_is_not_a_certification() -> None:
    """`certified: true` in a document that also says why it is not.

    Reading the flag alone would let an exploratory receipt be laundered by
    adding one key.
    """
    laundered = execution_mode.Trust(execution_mode.EXPLORATORY).stamp({"schema": "x"})
    laundered["certified"] = True

    assert execution_mode.is_certified(laundered) is False
    with pytest.raises(execution_mode.ExecutionModeError):
        execution_mode.require_certified_input(laundered, what="P0 artifact")


def test_the_exclusion_reads_nested_inputs_too() -> None:
    """A certified document that merely quotes an uncertified one is not
    certified either -- the chain is only as good as what it stands on."""
    exploratory = execution_mode.Trust(execution_mode.EXPLORATORY).stamp({"schema": "x"})
    quoting = execution_mode.Trust().stamp({
        "schema": "y", "bindings": {"p0": exploratory}})

    assert execution_mode.is_certified(quoting) is False
