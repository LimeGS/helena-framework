"""The fleet is not credited with surfaces it did not grow.

Imported catalogue surfaces live in `segment_surfaces` beside the ones the fleet
grew, and three separate aggregations summed the whole table while the page
called the result "surfaces we grew". Ten surfaces on this control plane came
from a catalogue with no attempt anywhere, so the number was wrong in the
direction that flatters us.

The origin rule is not new: /api/segmentation/segments already decides it the
same way -- a surface with an attempt was grown here, one without was imported --
and it reports grown_here and imported separately. The aggregations simply did
not use it.

Read statically, because what went wrong is which rows a SQL predicate selects,
and that is legible in the query. A test needing PostgreSQL and a grown surface
to notice a missing FILTER clause is a test that does not get run.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "panel/app.py").read_text()


def _body_of(function: str) -> str:
    """One top-level function, to the start of the next one."""
    start = APP.index(f"\ndef {function}(")
    rest = APP[start + 1:]
    end = rest.index("\ndef ")
    return rest[:end]


def test_the_totals_behind_the_tile_name_an_origin() -> None:
    """Only the aggregations that feed a "we grew" claim, and all of them.

    Not every count over that table: the certification and QC-state views count
    every row on purpose, because an imported surface that was certified is
    certified and its origin is beside the point there. These two functions are
    the ones the tile reads, and they were both summing the whole table.
    """
    for function in ("fleet_status", "scoped_queue"):
        body = _body_of(function)
        # The SQL itself, not the function around it: a window cut the FILTER
        # clauses in half and the whole body passed with one predicate removed,
        # because `attempt_id` still appeared somewhere else in it.
        queries = re.findall(r'"""(.*?)"""', body, re.DOTALL)
        counting = [q for q in queries
                    # Mission scoping first constructs an
                    # ``eligible_surfaces`` CTE and then aggregates that name.
                    # The origin FILTERs remain in the aggregation; requiring
                    # the base table's spelling made this test reject the
                    # correctly scoped query without examining them.
                    if ("segment_surfaces" in q or "eligible_surfaces" in q)
                    and ("count(*)" in q or "sum(area_cm2)" in q)]
        assert counting, (
            f"{function} has no triple-quoted aggregation over segment_surfaces; "
            "either it moved or the quoting changed and this test cannot see it"
        )
        for query in counting:
            aggregates = re.findall(
                r"(?:count\(\*\)|sum\(area_cm2\))(?:\s*FILTER\s*\([^)]*\))?", query)

            # Two honest shapes, and the distinction is what makes this test
            # bite. A query returning several figures at once has to name the
            # origin on each of them, or one of those figures is a total over
            # everything sitting in a row of split ones. A query returning a
            # single number may scope once instead -- including through an
            # interpolated WHERE this cannot read, which is why the whole-query
            # check is only allowed when there is one aggregate to scope.
            if len(aggregates) == 1:
                assert "attempt_id" in query, (
                    f"{function} totals segment_surfaces with no origin predicate "
                    f"anywhere:\n{query[:300]}"
                )
                continue
            # Every one of them qualified by something. Origin for the grown and
            # imported figures; a QC state for the certified and CT-supported
            # ones, which are deliberately origin-blind -- an imported surface
            # that passed certification is certified, and that is a different
            # question from who grew it. What cannot be there is a bare aggregate
            # sitting in a row of qualified ones: that is the whole-table total
            # the tile used to call "surfaces we grew".
            assert any("attempt_id" in aggregate for aggregate in aggregates), (
                f"{function} splits its figures and none of them is by origin"
            )
            for aggregate in aggregates:
                assert "FILTER" in aggregate, (
                    f"{function} returns several figures and this one is unqualified, "
                    f"so it counts every row: {' '.join(aggregate.split())[:120]}"
                )


def test_the_tile_reports_imported_apart_from_grown() -> None:
    launcher = (ROOT / "panel/web/src/routes/Segmentation.tsx").read_text()
    assert "imported, not counted here" in launcher, (
        "the tile does not say what it is leaving out, so a reader cannot tell "
        "whether the catalogue surfaces are in the number"
    )
    # And the area keeps its honest name: the scientific handoff is explicit that
    # naive VC3D area double-counts identity and can include radial spokes.
    assert "an upper bound" in launcher
    assert "certified area is P2's number, not this" not in launcher


def test_certified_and_ct_supported_are_reported_too() -> None:
    """Gross area is not yield. The downstream gates have their own figures."""
    assert '"certified_area_cm2"' in APP
    assert '"ct_supported_area_cm2"' in APP
    # The vocabulary has to match the fleet's, not a guess at it.
    store = (ROOT / "framework/stages/01-segmentation/fleet/store.py").read_text()
    assert 'ADMISSIBLE_PHYSICAL_QC_STATES = ("CT_SUPPORTED", "CT_SUPPORTED_REVIEW")' in store
    assert "'CT_SUPPORTED', 'CT_SUPPORTED_REVIEW'" in APP
    assert "'GEOMETRY_CERTIFIED'" in APP
