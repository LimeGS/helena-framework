"""The user guide has to keep covering the panel it documents.

Prose rots quietly. Somebody adds a button, the guide still reads as complete,
and the gap is found by the one person who needed that button explained. This
turns that into a failing test.

It lives here rather than in vitest because it reads the route files from disk,
and pulling node's fs into the browser build to do it costs more than it saves.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "panel/web/src/routes"
CONTROLS = (ROUTES / "guide-controls.ts").read_text()
APP = (ROOT / "panel/web/src/App.tsx").read_text()

# Pages that document rather than being documented.
DOCUMENTATION = {"UserGuide", "Tutorial", "ApiReference", "DeveloperReference",
                 "Docs", "Handbook"}

# Pages with controls that no route and no import reaches: dead surfaces, not
# undocumented ones. Proved orphaned below rather than merely asserted, so this
# cannot quietly become somewhere to hide a page nobody got round to.
UNREACHABLE = {"Scrolls"}


def _documented() -> set[str]:
    return set(re.findall(r'^    page: "([A-Za-z]+)"', CONTROLS, re.MULTILINE))


def _pages_with_controls() -> set[str]:
    found = set()
    for path in ROUTES.glob("*.tsx"):
        if ".test." in path.name:
            continue
        if re.search(r"<(button|select|input)[\s>]", path.read_text()):
            found.add(path.stem)
    return found


def test_every_page_a_person_drives_is_documented() -> None:
    """A page with something to press and no guide entry is a page somebody has
    to work out by clicking."""
    missing = _pages_with_controls() - _documented() - DOCUMENTATION - UNREACHABLE
    assert not missing, (
        f"these pages have controls and no entry in guide-controls.ts: "
        f"{', '.join(sorted(missing))}"
    )


def test_the_guide_names_only_pages_that_exist() -> None:
    """An entry for a deleted page is worse than none: it describes controls
    nobody can find."""
    for page in _documented():
        assert (ROUTES / f"{page}.tsx").is_file(), (
            f"the guide documents {page}, which has no route file"
        )


def test_the_unreachable_list_is_honest() -> None:
    """The escape hatch has to prove itself.

    If somebody routes one of these again it becomes a page users can open, and
    this fails until it is documented.
    """
    everything = "\n".join(
        p.read_text() for p in ROUTES.glob("*.tsx") if ".test." not in p.name
    ) + APP
    for page in UNREACHABLE:
        reachable = re.search(rf'import\([^)]*/{page}["\'`]\)|from ["\'][^"\']*/{page}["\']',
                              everything)
        assert not reachable, (
            f"{page} is reachable again and has to be documented, or the entry "
            "here removed"
        )


def test_the_tutorial_and_the_guide_are_different_things() -> None:
    """The split is the point.

    One page tried to be both and was, in practice, the walkthrough with the
    reference missing: it told you to leave controls on their defaults without
    ever saying what those defaults were for.
    """
    tutorial = (ROUTES / "tutorial-content.ts").read_text()
    # The tutorial walks phases; the guide indexes pages. Neither should have
    # grown the other's shape.
    assert '  id: "P0"' in tutorial and "page:" not in tutorial.split("export const STOPS")[1]
    assert "STOPS" not in CONTROLS, "the guide has grown a walkthrough"


def test_both_are_reachable_from_the_documentation_page() -> None:
    docs = (ROUTES / "Docs.tsx").read_text()
    for tab in ("Handbook", "Tutorial", "UserGuide", "DeveloperReference",
                "ApiReference"):
        assert tab in docs, f"{tab} is not on the documentation page"
    # It opened on the walkthrough, for the reason that the person who has never
    # run this cannot tell which tab they need. The handbook is that reason
    # carried further: it has the walkthrough inside it as a section, and a
    # contents list beside it, so the first screen answers "what is here" as
    # well as "where do I start".
    assert '"handbook"' in docs, (
        "the documentation page does not open on the handbook"
    )


def test_the_tutorial_describes_the_phases_the_contract_defines() -> None:
    """The walkthrough shipped with ink detection at P4, screening at P5 and
    coverage at P6. All three were wrong.

    The contract puts rendering at P4, ink at P5, liveness at P6, screening at
    P7. Documentation that renumbers the pipeline is worse than none: somebody
    following it queues the wrong phase and reads the result as a failure of the
    science.

    Checked by keyword rather than by prose, so the wording stays free while the
    subject cannot drift.
    """
    import json

    contract = json.loads((ROOT / "framework/contracts/pipeline_phases.json").read_text())
    phases = contract if isinstance(contract, list) else contract["phases"]
    by_id = {p["id"]: p for p in phases}

    tutorial = (ROOT / "panel/web/src/routes/tutorial-content.ts").read_text()
    stops = dict(re.findall(r'id: "(P\d)",\n    goal: "([^"]+)"', tutorial))

    assert set(stops) == set(by_id), "the tutorial and the contract disagree on which phases exist"

    # One word each that the phase is unmistakably about. If a stop stops being
    # about its phase, this is what notices.
    subject = {
        "P0": ("volume", "scale"), "P1": ("surface", "seed", "segment"),
        "P2": ("lamina", "plausible", "certif"), "P3": ("unroll", "flat"),
        "P4": ("layer stack", "normal", "sample"), "P5": ("detector", "probability"),
        "P6": ("decision", "liveness", "carries"), "P7": ("verdict", "text-like"),
        "P8": ("stitch", "continuous", "sheet"), "P9": ("readable", "page", "compose"),
    }
    for phase, words in subject.items():
        goal = stops[phase].lower()
        assert any(w in goal for w in words), (
            f"the tutorial's {phase} is not about {by_id[phase]['name']!r}: {stops[phase]!r}"
        )
