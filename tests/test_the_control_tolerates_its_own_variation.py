"""The control must not depend on the one thing that varies between its runs.

Three runs of the control on anchor cell [156,234] -- same profile, same model,
same deployment -- produced three different surfaces, because the grow is a
search and not a function. Two of them screened
CT_SUPPORTED_RETAINED_FOR_REVIEW and the third CT_SUPPORTED_NO_RETAINED_INK_
SIGNAL. All three were ALIVE; what moved was whether anything crossed the
retention threshold, which sits near the boundary.

Both outcomes are admissible downstream today, so the control passes either way.
The trap is that narrowing the admissible set to the outcome that happened to be
seen most often would look harmless and would make the control fail
intermittently -- for a reason that would read as a science problem and would
not be one.

docs/first-letters/control-run-to-run-variation.md has the measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROFILES = Path(__file__).resolve().parents[1] / "framework/profiles/01-segmentation"
CONTROLS = sorted(PROFILES.glob("first-letters-control-policy-*.json"))

# Both were produced by the same anchor, on the same deployment, days apart.
OBSERVED_QC_OUTCOMES = {
    "CT_SUPPORTED_RETAINED_FOR_REVIEW": "CT_SUPPORTED_REVIEW",
    "CT_SUPPORTED_NO_RETAINED_INK_SIGNAL": "CT_SUPPORTED",
}


def _pipeline_control(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    return (document.get("checks") or {}).get("PIPELINE_CONTROL") or {}


def test_there_is_a_control_profile_to_check() -> None:
    assert CONTROLS


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_every_outcome_this_anchor_produces_is_admissible(path: Path) -> None:
    admissible = set(_pipeline_control(path).get(
        "terminal_physical_qc_compatible_with_downstream") or [])
    missing = set(OBSERVED_QC_OUTCOMES.values()) - admissible
    assert not missing, (
        f"{sorted(missing)} is a state this control's own anchor has produced, "
        "and dropping it makes the control fail on some runs and not others"
    )


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_no_pass_requirement_depends_on_retention(path: Path) -> None:
    """`P5_LIVE_OUTPUT` asks the detector to decide, which it did every time.
    Asking it to *retain* would be asking for the thing that varies."""
    requirements = _pipeline_control(path).get("pass_requirements") or []
    assert requirements, "the pass requirements are not where this test looks"
    for requirement in requirements:
        assert "RETAIN" not in str(requirement).upper(), (
            f"{requirement} keys on retention, which two runs of the same anchor "
            "disagreed about while both stayed ALIVE"
        )


@pytest.mark.parametrize("path", CONTROLS, ids=lambda p: p.stem)
def test_the_control_still_asks_for_a_live_detector(path: Path) -> None:
    """The other half: tolerating variation must not become tolerating a dead
    output. A constant map is what the old anchor produced, four times."""
    requirements = {str(row) for row in
                    (_pipeline_control(path).get("pass_requirements") or [])}
    assert "P5_LIVE_OUTPUT" in requirements
