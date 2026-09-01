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
DOCUMENTATION = {"ApiReference", "DeveloperReference", "Docs", "Handbook"}

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


def test_every_tab_is_reachable_from_the_documentation_page() -> None:
    """Three tabs now, where there were five.

    The Tutorial and the User guide were separate pages, and the handbook has
    both inside it -- the walkthrough as a section, every panel page with its
    controls as another. Keeping the old tabs beside it meant three answers to
    the same question, two of them going stale unwatched.

    What remains beside the handbook is generated from the code it describes
    rather than written, which is why it is not prose that can rot.
    """
    docs = (ROUTES / "Docs.tsx").read_text()
    for tab in ("Handbook", "DeveloperReference", "ApiReference"):
        assert f"<{tab} />" in docs, f"{tab} is not on the documentation page"
    for gone in ("<Tutorial />", "<UserGuide />"):
        assert gone not in docs, (
            f"{gone} is back beside the handbook, which has that material inside it"
        )
    assert '"handbook"' in docs, (
        "the documentation page does not open on the handbook"
    )


def test_the_handbook_describes_the_phases_the_contract_defines() -> None:
    """The walkthrough shipped with ink detection at P4, screening at P5 and
    coverage at P6. All three were wrong.

    The contract puts rendering at P4, ink at P5, liveness at P6, screening at
    P7. Documentation that renumbers the pipeline is worse than none: somebody
    following it queues the wrong phase and reads the result as a failure of the
    science.

    This used to match keywords against the tutorial's prose, because the prose
    was the only place the phase appeared. The handbook titles each phase
    section with the contract's own name, so the check is now an equality rather
    than a guess at which words mean which phase.
    """
    import json

    contract = json.loads((ROOT / "framework/contracts/pipeline_phases.json").read_text())
    phases = contract if isinstance(contract, list) else contract["phases"]
    by_id = {p["id"]: p["name"] for p in phases}

    handbook = (ROUTES / "handbook-content.ts").read_text()
    titled = dict(re.findall(r'"title":\s*"(P\d) \u2014 ([^"]+)"', handbook))

    assert titled == by_id, (
        "the handbook and the contract disagree about the phases: "
        f"handbook has {titled}, contract has {by_id}"
    )
