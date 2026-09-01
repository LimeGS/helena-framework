"""Stage 3 must decode a surface row by column name, not by position.

`_lineage_from_surface_row` unpacked seven positional fields. That is a promise
that the SELECT list will never change and that its order will never change --
a promise nobody can keep, because the routing work widens exactly this row to
carry the route beside the surface.

Two failures, and the quiet one is the dangerous one. A row with more columns
than the unpack expects raises, which is at least loud. A row with the *same*
number of columns in a different order unpacks cleanly and puts the state where
the artifact digest belongs: the lineage is then wrong in every field, the guard
downstream still says yes, and nothing anywhere raises.

So the decode reads the cursor's own description. A query that grows a column
keeps working, a query that reorders keeps working, and neither can silently
hand a downstream stage another surface's facts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))

from job_store import InkJobStore  # noqa: E402


SURFACE = {
    "surface_id": "surface-a",
    "source_snapshot_id": "snap-a",
    "sample_id": "PHerc0268",
    "artifact_sha256": "1" * 64,
    "artifact_uri": "s3://helena/surfaces/surface-a",
    "state": "QC_PENDING",
    "payload": {"owner": "campaign-x", "sample_id": "PHerc0268"},
}

# What the surface row looks like once routing travels with it: the seven fields
# the lineage needs, the rest of segment_surfaces, and the routing receipt.
WIDE = {
    **SURFACE,
    "owner": "campaign-x",
    "bbox_xyz": [[0, 0, 0], [1, 1, 1]],
    "sample_points": None,
    "area_cm2": 0.0198322,
    "physical_qc_state": "UNVALIDATED",
    "geometry_qc_state": "GEOMETRY_CERTIFIED",
    "created_at": "2026-08-02T00:00:00Z",
    "route": "SMALL_SURFACE_DIAGNOSTIC",
    "measured_area_cm2": 0.0198322,
    "minimum_area_cm2": 0.10,
    "receipt_sha256": "2" * 64,
}


class Cursor:
    """A cursor that answers with the columns it says it is answering with."""

    def __init__(self, row: dict[str, object]):
        self.row = row
        self.description = tuple((name,) for name in row)

    def execute(self, sql: str, args: tuple = ()) -> None:
        self.sql = sql

    def fetchall(self) -> list[tuple]:
        return [tuple(self.row.values())]

    def fetchone(self) -> tuple | None:
        return tuple(self.row.values())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Connection:
    def __init__(self, cursor: Cursor):
        self._cursor = cursor

    def cursor(self) -> Cursor:
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _lineage(row: dict[str, object]) -> dict[str, object]:
    store = InkJobStore("postgresql://unused")
    store._connect = lambda: Connection(Cursor(row))  # noqa: SLF001
    return store.require_surface_lineage(
        surface_id="surface-a", mission_id=None,
        boundary="P5_EXECUTION_RESOLUTION",
    )


def _assert_surface_facts(lineage: dict[str, object]) -> None:
    assert lineage["surface_id"] == "surface-a"
    assert lineage["source_snapshot_id"] == "snap-a"
    assert lineage["artifact_sha256"] == "1" * 64
    assert lineage["artifact_uri"] == "s3://helena/surfaces/surface-a"
    assert lineage["surface_state"] == "QC_PENDING"
    assert lineage["namespace"] == "CANONICAL_SURFACE"


def test_the_seven_column_row_still_decodes():
    """The shape shipping today, so the fix is not paid for by a regression."""
    _assert_surface_facts(_lineage(dict(SURFACE)))


def test_a_wider_row_decodes_the_same_surface_facts():
    """The routing port widens this row. Position stops meaning anything."""
    _assert_surface_facts(_lineage(dict(WIDE)))


def test_a_reordered_row_decodes_the_same_surface_facts():
    """The quiet one: seven columns, different order, no exception anywhere."""
    reordered = {key: SURFACE[key] for key in (
        "payload", "state", "artifact_uri", "artifact_sha256",
        "sample_id", "source_snapshot_id", "surface_id",
    )}
    _assert_surface_facts(_lineage(reordered))
