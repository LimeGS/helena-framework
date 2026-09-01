"""A surface uploaded into a mission is visible in that mission.

The browser upload answered `{"inserted": 1, "origin": "IMPORTED"}` and the
segments table under it kept reporting zero. Both were true: the bytes were on
the artifact volume and the row was in `segment_surfaces`, and every
mission-scoped query reached the mission by joining
`segment_artifact_sets -> segment_attempts -> segment_tasks`. An import has no
attempt and no task -- nothing ran -- so an inner join through them can never
match one, and the panel has no view that is not mission-scoped.

The mission an import belongs to is the one it was uploaded into, which
`api_import` records in the surface payload. `surface_mission_predicate` is the
one place that knows all three ways a surface reaches a mission.

Read statically, like `test_grown_surfaces_are_not_imported_ones`: what went
wrong is which rows a SQL predicate selects, and a test that needs PostgreSQL,
an attempt and an upload to notice a missing branch is a test nobody runs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()


def test_the_predicate_knows_the_three_ways_a_surface_arrives() -> None:
    start = APP.index("def surface_mission_predicate(")
    body = APP[start:APP.index("\n\n\n", start)]
    # Grown here, derived by an ink job, and uploaded. The third is the one
    # that was missing.
    assert "segment_tasks t" in body
    assert "ink_jobs j" in body
    assert "payload ->> 'mission_id'" in body


def test_no_query_scopes_surfaces_to_a_mission_on_its_own() -> None:
    """Every mission-scoped surface query goes through the predicate.

    A query that reaches `mission_id` from `segment_surfaces` by itself has
    written a fourth definition of belonging, and it will be the join-only one
    again: that is the shape the imported case keeps falling out of.
    """
    offenders = [
        query.strip()[:160]
        for query in re.findall(r'"""(.*?)"""', APP, re.DOTALL)
        if "FROM segment_surfaces" in query and "mission_id" in query
    ]
    assert offenders == [], offenders


def test_the_predicate_and_its_arity_stay_together() -> None:
    """Three placeholders, and callers that pass three arguments.

    psycopg raises on a mismatch, but only for the code path that runs; the
    count is asserted here so adding a branch to the predicate without telling
    its callers fails at once.
    """
    start = APP.index("def surface_mission_predicate(")
    body = APP[start:APP.index("\n\n\n", start)]
    assert body.count("{placeholder}") == 3
    assert "SURFACE_MISSION_PARAMETERS = 3" in APP
