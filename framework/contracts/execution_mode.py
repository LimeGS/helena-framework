"""Two lanes, one platform: certified and exploratory.

A gate answers "may this be certified", not "may this run". This is an
exploration framework with a dynamic workflow: running one phase against
something arbitrary, mid-campaign, with an upload bound to nothing, is an
ordinary thing to want. Refusing it outright is the platform deciding which
questions may be asked. What such a run does not get is a receipt.

In CERTIFIED -- the default, and what every existing caller gets by passing
nothing -- every precondition holds exactly as it does today. Nothing here
relaxes one. In EXPLORATORY an unmet chain-of-trust precondition does not stop
the work: it is recorded, and the output says in itself that it certifies
nothing, and why.

The permissive half is the easy half. The half that has to be right is the
exclusion below: an exploratory output must be structurally unable to be read as
evidence by a certified run. Without that this is an acceptance-gate bypass
wearing a friendlier name.

Two kinds of refusal are not modes and never become one. Integrity -- a tampered
binding, a drifted identity, a hash that does not match -- because running anyway
means acting on evidence known to be corrupt, and the answer would be wrong
rather than merely uncertified. And authorization: a lane is not a login.
"""

from __future__ import annotations

from typing import Any

CERTIFIED = "CERTIFIED"
EXPLORATORY = "EXPLORATORY"
MODES = frozenset({CERTIFIED, EXPLORATORY})

NON_CLAIM = (
    "This document certifies nothing. It was produced in the exploratory lane "
    "with chain-of-trust preconditions unmet, and no certified run may read it "
    "as evidence."
)

# Written by `stamp`, read by `is_certified`. A document may not arrive carrying
# them: the stamp is this module's statement about the run, not the run's about
# itself.
STAMP_FIELDS = ("certified", "execution_mode", "uncertified_because")


class ExecutionModeError(ValueError):
    """A lane was misused: an unknown mode, a forged stamp, or a certified run
    reaching for something uncertified."""


def parse(value: object) -> str:
    """The mode a request asked for, defaulting to the strict one.

    Absent means CERTIFIED. A value that is not a mode is refused rather than
    treated as EXPLORATORY: a typo must never open the permissive lane.
    """
    if value is None or value == "":
        return CERTIFIED
    if not isinstance(value, str):
        raise ExecutionModeError(f"execution mode must be a string, not {type(value).__name__}")
    mode = value.strip().upper()
    if mode not in MODES:
        raise ExecutionModeError(
            f"unknown execution mode {value!r}; expected one of {sorted(MODES)}")
    return mode


def is_certified(document: object) -> bool:
    """Whether this document may be read as evidence by a certified run.

    Fails closed three ways. A document with no stamp is not certified -- and
    everything written before this module existed carries no stamp, so absence
    of a claim must not read as a claim. A stamp that says CERTIFIED while also
    recording why it is not is a forgery, and adding one key must not launder an
    exploratory receipt. And a certified document that quotes an uncertified one
    anywhere inside it is not certified either: the chain is only as good as
    what it stands on.
    """
    if not isinstance(document, dict):
        return False
    if document.get("execution_mode") != CERTIFIED:
        return False
    if document.get("certified") is not True:
        return False
    if document.get("uncertified_because"):
        return False
    return not _quotes_uncertified(document)


def _quotes_uncertified(value: object, *, depth: int = 0) -> bool:
    if depth > 24:
        # Deeper than this cannot be read, and unreadable is not certified.
        return True
    if isinstance(value, dict):
        if value.get("execution_mode") == EXPLORATORY or value.get("certified") is False:
            return True
        return any(_quotes_uncertified(item, depth=depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_quotes_uncertified(item, depth=depth + 1) for item in value)
    return False


def declares_uncertified(document: object) -> bool:
    """Whether this document says, itself, that it certifies nothing.

    The complement is not `is_certified`. That one fails closed on a missing
    stamp, which is right for a reader deciding what to lean on: absence of a
    claim is not a claim. It is wrong for a gate that stands in front of
    everything produced before stamps existed. Every screening on the fleet
    from before this module reached P5 carries no stamp at all -- and every one
    of them came out of a pinned lane under the integrity rules that have not
    changed, which is to say they were certified by construction and simply
    never said so. A gate that read their silence as a refusal would block the
    adjudication of every existing map the day it deployed.

    So this asks the narrower question: did the run declare itself
    exploratory, or quote something that did? Only the experimental lane
    produces such a document, and it always stamps. A forged `certified: true`
    still does not help anyone: `is_certified` is what a reader uses, and this
    is only what a gate uses to refuse.
    """
    if not isinstance(document, dict):
        return False
    if document.get("execution_mode") == EXPLORATORY:
        return True
    if document.get("certified") is False:
        return True
    if document.get("uncertified_because"):
        return True
    return _quotes_uncertified(document)


def require_certified_input(document: object, *, what: str) -> None:
    """Refuse, in a certified run, to read something that certifies nothing."""
    if not is_certified(document):
        raise ExecutionModeError(
            f"{what} is not certified, so a certified run may not read it as "
            "evidence; run in the exploratory lane to use it anyway")


class Trust:
    """One run's lane, and the record of what it could not satisfy.

    Callers keep their own refusals. This only answers whether the refusal
    applies::

        if trust.blocks("control scope requires a selected P0 artifact"):
            raise HTTPException(409, "control scope requires a selected P0 artifact")

    which leaves every existing message where it is and makes the default path
    byte-for-byte what it was.
    """

    def __init__(self, mode: object = CERTIFIED) -> None:
        self.mode = parse(mode)
        self.unmet: list[str] = []

    @property
    def certified(self) -> bool:
        """An exploratory run certifies nothing even when it met everything.

        The lane is the caller's declaration, not a score earned along the way:
        otherwise one request would certify or not depending on the state of the
        system when it happened to run, which is the opposite of a chain of
        trust.
        """
        return self.mode == CERTIFIED and not self.unmet

    def blocks(self, reason: str) -> bool:
        """Whether this unmet precondition stops the operation.

        Always in CERTIFIED. Never in EXPLORATORY, where it is recorded instead
        and ends up in the document.
        """
        if self.mode == CERTIFIED:
            return True
        if reason not in self.unmet:
            self.unmet.append(reason)
        return False

    def blocks_uncertified_input(self, document: object, *, what: str) -> bool:
        """The exclusion, as a gate: reading uncertified evidence."""
        if is_certified(document):
            return False
        return self.blocks(f"{what} is not certified")

    def stamp(self, document: dict[str, Any]) -> dict[str, Any]:
        """Say, in the document, which lane produced it.

        The document may not already carry these fields. A receipt with a
        `certified` of its own means something by it, and merging over it would
        turn its answer into this one -- the same substitution that once made a
        preflight receipt's `state` become the queue's.
        """
        if not isinstance(document, dict):
            raise ExecutionModeError("only a document can be stamped")
        clashes = [field for field in STAMP_FIELDS if field in document]
        if clashes:
            raise ExecutionModeError(
                f"the document already carries {clashes}; stamping would replace "
                "a field it owns")
        stamped = dict(document)
        stamped["execution_mode"] = self.mode
        stamped["certified"] = self.certified
        if not self.certified:
            stamped["uncertified_because"] = list(self.unmet)
            stamped["non_claim"] = NON_CLAIM
        return stamped
