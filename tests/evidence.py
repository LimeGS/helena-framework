"""Where the campaign evidence is, for the tests that check numbers against it.

Several profiles in this framework do not merely declare a threshold, they bind
the receipt the threshold was derived from -- path plus sha256 -- and the tests
recompute the declared result from that file. It is the strongest form of
calibration check there is: a constant cannot drift from its own evidence
without a test failing.

That evidence is a data release, not source code. It is tens of thousands of
files and gigabytes of images, and it does not belong in a repository people
clone to read the code. So it lives outside, and these tests skip when it is
absent rather than failing a clean checkout.

Point HELENA_EVIDENCE_ROOT at an unpacked release to run them:

    HELENA_EVIDENCE_ROOT=/data/helena-campaign-x-2026 python3 -m pytest tests/

The layout under that root is the one the profiles record, so an unpacked
release drops straight in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get("HELENA_EVIDENCE_ROOT", ROOT))

WHERE = ("set HELENA_EVIDENCE_ROOT to an unpacked campaign evidence release; "
         "see tests/evidence.py")


def at(*relative: str) -> Path:
    """One evidence file, under whichever root is in force."""
    return EVIDENCE.joinpath(*relative)


def needs(*relative: str):
    """Skip the module unless every one of these is present.

    Named paths rather than a blanket "is the directory there", because a
    partial release should skip the tests it cannot serve and run the rest.
    """
    missing = [r for r in relative if not at(r).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"campaign evidence not present ({missing[0]}): {WHERE}")


# One mark for the tests that recompute a declared number from a bound receipt.
# Applied per test rather than per module: a file that holds twenty checks of
# the framework and eight of a campaign should still run the twenty.
needs_campaign_evidence = pytest.mark.skipif(
    not at("workspace/campaigns/campaign-x-2026").is_dir(),
    reason=f"campaign evidence not present: {WHERE}")
