"""Stage 3 rechecks the route where the work happens, not where it was queued.

`require_job_canonical_lineage` already re-resolves lineage at execution, which
is the right shape and the right moment. The size question was never asked at
all: a surface below the 0.10 cm2 effort floor was admissible to P4, P5 and P7
because nothing between the queue and the renderer looked.

That is how a 0.01983222455087575 cm2 patch reached the ink screen and came back
EMPTY, filed beside an EMPTY over five square centimetres.

The gate is asked immediately before the runner starts, from the row as it is
then. What it cannot do yet is read a stored receipt on PostgreSQL, because the
receipts table lives only in the SQLite schema; the last test here names that
gap and turns red the moment it closes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/03-ink/fleet"))
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing  # noqa: E402
from job_store import InkJobStore, RoutingRefused  # noqa: E402

PHERC0268_AREA = 0.01983222455087575
POLICY = surface_routing.load_policy()

SURFACE_COLUMNS = ("surface_id", "area_cm2", "artifact_sha256", "routing_receipt")


def _receipt(surface_id: str, area: float) -> dict:
    return surface_routing.build_receipt(
        surface_id=surface_id, area_cm2=area, policy=POLICY,
        measurement={}, read_set={},
    )


class Cursor:
    def __init__(self, row: tuple | None, error: Exception | None = None):
        self.row = row
        self.error = error
        self.description = tuple((name,) for name in SURFACE_COLUMNS)
        self.statements: list[str] = []

    def execute(self, sql, args=()):
        self.statements.append(sql)
        if self.error is not None:
            raise self.error

    def fetchall(self):
        return [] if self.row is None else [self.row]

    def fetchone(self):
        return self.row

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Connection:
    def __init__(self, cursor):
        self.value = cursor

    def cursor(self):
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _store(row, error=None) -> InkJobStore:
    store = InkJobStore("postgresql://unused")
    store._connect = lambda: Connection(Cursor(row, error))  # noqa: SLF001
    return store


def _job(surface_id: str = "surface-a") -> dict:
    return {"job_id": "p5-1", "phase": "P5", "sample_id": "PHerc0268",
            "mission_id": "mission-a", "lease_token": "lease",
            "parameters": {"surface_id": surface_id}}


def _refusal(row, error=None) -> RoutingRefused:
    with pytest.raises(RoutingRefused) as caught:
        _store(row, error).require_job_standard_route(_job())
    return caught.value


# -- what the gate refuses --------------------------------------------------

def test_a_diagnostic_receipt_stops_the_phase():
    refusal = _refusal(
        ("surface-a", PHERC0268_AREA, "d" * 64,
         _receipt("surface-a", PHERC0268_AREA)))
    assert refusal.reason_code == "SMALL_SURFACE_DIAGNOSTIC_NOT_CANONICAL_WORK"


def test_a_forged_receipt_fails_closed():
    forged = {**_receipt("surface-a", PHERC0268_AREA),
              "route": surface_routing.STANDARD}
    refusal = _refusal(("surface-a", PHERC0268_AREA, "d" * 64, forged))
    assert refusal.reason_code == "ROUTING_RECEIPT_UNVERIFIED"


def test_no_receipt_falls_back_to_the_area_on_the_row():
    """Only imports write receipts today, so a grown surface has none."""
    refusal = _refusal(("surface-a", PHERC0268_AREA, "d" * 64, None))
    assert refusal.reason_code == "SMALL_SURFACE_DIAGNOSTIC_NOT_CANONICAL_WORK"


def test_an_unmeasured_surface_is_refused():
    refusal = _refusal(("surface-a", None, "d" * 64, None))
    assert refusal.reason_code == "ROUTING_UNDECIDABLE_NO_MEASURED_AREA"


def test_a_surface_that_is_not_registered_is_refused():
    assert _refusal(None).reason_code == "SURFACE_NOT_REGISTERED"


def test_a_control_plane_without_the_table_is_refused_by_name():
    """PostgreSQL has no routing receipts yet, and silence is not a yes."""
    class UndefinedTable(Exception):
        sqlstate = "42P01"

    refusal = _refusal(("surface-a", 0.5, "d" * 64, None),
                       error=UndefinedTable("relation does not exist"))
    assert refusal.reason_code == "ROUTING_UNAVAILABLE_ON_THIS_CONTROL_PLANE"


# -- and what it lets through -----------------------------------------------

def test_a_standard_receipt_passes():
    _store(("surface-a", 0.5, "d" * 64, _receipt("surface-a", 0.5))
           ).require_job_standard_route(_job())


def test_a_job_that_names_no_surface_is_not_this_gate_s_business():
    store = _store(None)
    store.require_job_standard_route(
        {"phase": "P5", "parameters": {}, "mission_id": None})


# -- the worker asks before it starts anything ------------------------------

def test_run_job_refuses_before_it_makes_a_directory_or_a_process(tmp_path):
    import ink_worker

    started: list = []

    class Store:
        def require_job_canonical_lineage(self, job, *, execution):
            assert execution is True

        def require_job_standard_route(self, job):
            raise RoutingRefused(
                "SMALL_SURFACE_DIAGNOSTIC_NOT_CANONICAL_WORK", "0.02 cm2")

        def mark_running(self, *a, **k):
            started.append("running")

        def note(self, *a, **k):
            started.append("note")

        def finish(self, *a, **k):
            started.append("finish")

    runs = tmp_path / "runs"
    with pytest.raises(RoutingRefused):
        ink_worker.run_job(Store(), _job(), runs_root=runs, timeout=10)

    assert started == []
    assert not runs.exists(), "the phase made itself an output directory anyway"


# -- and the gap it cannot close on its own ---------------------------------

def test_the_postgresql_schema_carries_the_routing_receipts():
    ddl = (ROOT / "framework/stages/01-segmentation/fleet/migrations"
           / "001_postgresql.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS segment_surface_routing_receipts" in ddl
