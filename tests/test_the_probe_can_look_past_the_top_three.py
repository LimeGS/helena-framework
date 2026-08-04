"""top_k caps how far down m7's ordering the probe looks.

At 3 it is a tie-break between the best candidates, and the first shadow run
ever executed measured 8 of 8 ELIGIBLE -- a 0% rejection rate against a 34%
break-even, so probing cost 34% per candidate and saved none.

That is not evidence the probe is useless. It is evidence that m7's top three
are all good. Whether the ordering holds at rank 20 is a different question,
and one a cap of 3 makes unaskable.
"""
from __future__ import annotations

import re
from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / "panel/app.py").read_text()


def test_the_cap_allows_asking_about_the_tail() -> None:
    field = re.search(r"seed_probe_top_k: int = Field\(\s*(\d+),\s*ge=(\d+),\s*le=(\d+)\)", APP)
    assert field, "seed_probe_top_k is no longer a bounded field"
    default, low, high = (int(g) for g in field.groups())
    assert high >= 10, (
        f"top_k caps at {high}; m7's ordering past rank 3 cannot be measured, "
        "which is the open question about whether a cheap filter pays"
    )
    assert low == 1 and default <= 3, (
        "the default should stay where it was: raising the ceiling is not the "
        "same as changing what every run does"
    )


def test_the_advertised_range_matches_the_field() -> None:
    """The panel publishes the bounds so the UI can offer them. Two copies."""
    advertised = re.search(r'"top_k": \{"minimum": (\d+), "maximum": (\d+)', APP)
    assert advertised, "the panel no longer advertises the top_k range"
    field = re.search(r"seed_probe_top_k: int = Field\(\s*\d+,\s*ge=(\d+),\s*le=(\d+)\)", APP)
    assert (advertised.group(1), advertised.group(2)) == field.groups(), (
        "the advertised range and the validated one disagree, so the UI offers "
        "a value the API rejects"
    )


def test_the_cli_cap_matches_the_api_cap() -> None:
    """Two caps, one question. The panel calls this CLI, so an API that accepts
    20 and a CLI that refuses it is a run queued and then rejected downstream --
    the worst place to find out."""
    cli = (Path(__file__).resolve().parents[1]
           / "framework/stages/01-segmentation/fleet/cli.py").read_text()
    choices = re.search(r'"--seed-probe-top-k".*?choices=range\((\d+), (\d+)\)', cli, re.S)
    assert choices, "the CLI flag no longer declares a range"
    low, stop = int(choices.group(1)), int(choices.group(2))
    field = re.search(r"seed_probe_top_k: int = Field\(\s*\d+,\s*ge=(\d+),\s*le=(\d+)\)", APP)
    assert (low, stop - 1) == tuple(int(g) for g in field.groups()), (
        f"the CLI accepts {low}..{stop - 1} and the API accepts "
        f"{field.group(1)}..{field.group(2)}"
    )


def test_all_four_copies_of_the_bound_agree() -> None:
    """This number lived in four places and none of them knew about the others.

    The panel's request model, the fleet CLI's argparse choices, the launcher
    contract the panel serves to the UI, and a runtime check in seed_probe. I
    raised them one at a time and each next run failed further in: the API
    accepted 20, then the CLI refused it, then the contract disagreed, then the
    bootstrap raised. Four rounds to move one number.
    """
    root = Path(__file__).resolve().parents[1]
    probe = (root / "framework/stages/01-segmentation/fleet/seed_probe.py").read_text()
    cli = (root / "framework/stages/01-segmentation/fleet/cli.py").read_text()

    constant = re.search(r"^TOP_K_MAXIMUM = (\d+)", probe, re.M)
    assert constant, "seed_probe no longer names the bound, so there is nothing to compare against"
    ceiling = int(constant.group(1))

    assert f"1 <= top_k <= TOP_K_MAXIMUM" in probe, (
        "the runtime check hardcodes a bound again instead of reading the constant"
    )

    field = re.search(r"seed_probe_top_k: int = Field\(\s*\d+,\s*ge=1,\s*le=(\d+)\)", APP)
    contract = re.search(r'"top_k": \{"minimum": 1, "maximum": (\d+)', APP)
    choices = re.search(r'"--seed-probe-top-k".*?choices=range\(1, (\d+)\)', cli, re.S)
    assert field and contract and choices, "one of the four declarations has moved"

    seen = {
        "seed_probe.TOP_K_MAXIMUM": ceiling,
        "panel request model": int(field.group(1)),
        "launcher contract": int(contract.group(1)),
        "fleet CLI": int(choices.group(1)) - 1,
    }
    assert len(set(seen.values())) == 1, f"the four bounds disagree: {seen}"


def test_the_retry_bound_was_not_dragged_along() -> None:
    """maximum_attempts is also 3 and is a different question -- how many times
    to retry one candidate, not how far down the ordering to look. Raising the
    wrong one would multiply cost without asking anything new."""
    probe = (Path(__file__).resolve().parents[1]
             / "framework/stages/01-segmentation/fleet/seed_probe.py").read_text()
    assert "maximum_attempts <= 3" in probe, (
        "the per-candidate retry cap moved with top_k; they are not the same bound"
    )
