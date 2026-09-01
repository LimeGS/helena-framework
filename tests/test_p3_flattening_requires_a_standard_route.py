"""P3 refuses to flatten a surface that is not routed onto the standard path.

PHerc0268 was 0.0198 cm2 -- about two square millimetres -- GEOMETRY_CERTIFIED,
and it walked into physical QC because one upstream gate is one bug away from
being no gate. Routing is decided once, in the transaction that creates the
surface, and every stage after it has to ask for itself rather than assume
somebody upstream did.

`record_flattening` is the write that makes a surface a P3 result and puts it in
the queue P4 reads, so that is where P3 asks. The question is asked through
`surface_routing.enters_canonical_downstream`, never by reading the route column
and comparing a string: the helper also verifies the receipt digest, so a
tampered receipt fails exactly like a missing one instead of being believed.

Three ways to have no standard route, one answer to all three: no receipt at
all, a `SMALL_SURFACE_DIAGNOSTIC` receipt, and a receipt whose digest no longer
matches what it says. None of them may produce a `surface_flattenings` row.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing as routing  # noqa: E402
from fleet.common import content_sha256  # noqa: E402
from fleet.postgres_store import PostgresFleetStore  # noqa: E402
from fleet.store import FleetStore  # noqa: E402

ROUTING_TABLE = "segment_surface_routing_receipts"

SURFACE = "surface-under-test"
STANDARD_AREA = 0.5
TINY_AREA = 0.01983222455087575


def receipt_for(area: float, *, surface_id: str = SURFACE) -> dict:
    return routing.build_receipt(
        surface_id=surface_id, area_cm2=area, policy=routing.load_policy(),
        measurement={"bbox_xyz": [[0, 0, 0], [14, 14, 3]],
                     "sample_point_count": 84},
        read_set={"source_snapshot_id": "snapshot-1", "sample_id": "PHerc0268",
                  "artifact_sha256": "d" * 64,
                  "geometry_qc_state": "GEOMETRY_CERTIFIED"})


def flattening(**changed) -> dict:
    value = {
        "schema": "campaignx.surface_flattening_receipt.v1",
        "surface_id": SURFACE, "profile_id": "flatten-abf-v1@1.0.0",
        "state": "FLATTENED", "requested_by_job_id": "p3-job",
        "source_artifact_sha256": "1" * 64, "profile_file_sha256": "2" * 64,
        "artifact_uri": "s3://bucket/flat", "artifact_sha256": "3" * 64,
    }
    value.update(changed)
    value["receipt_sha256"] = content_sha256(value)
    return value


class ScriptedCursor:
    """One cursor that answers record_flattening's reads by statement."""

    def __init__(self, *, surface_payload: dict, routing_row: dict | None):
        self.surface_payload = surface_payload
        self.routing_row = routing_row
        self.executed: list[str] = []
        self.rowcount = 1
        self._pending = None

    def __enter__(self): return self
    def __exit__(self, *_args): return False

    def execute(self, statement, parameters=()):
        self.executed.append(statement)
        if "FROM segment_surfaces" in statement:
            self._pending = {"payload": self.surface_payload}
        elif ROUTING_TABLE in statement:
            self._pending = self.routing_row
        elif "FROM surface_flattenings" in statement:
            self._pending = {"payload": {"state": "FLATTENED"}}
        else:
            self._pending = None

    def fetchone(self):
        return self._pending

    @property
    def wrote_a_flattening(self) -> bool:
        return any("INSERT INTO surface_flattenings" in statement
                   for statement in self.executed)


class ScriptedConnection:
    def __init__(self, cursor): self.value = cursor
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def cursor(self): return self.value


def scripted_store(routing_row: dict | None) -> tuple[PostgresFleetStore,
                                                      ScriptedCursor]:
    cursor = ScriptedCursor(surface_payload={"surface_id": SURFACE},
                            routing_row=routing_row)
    store = PostgresFleetStore("postgresql://scripted")
    store.connect = lambda: ScriptedConnection(cursor)
    return store, cursor


# --- the gate ---------------------------------------------------------------


def test_a_surface_with_no_routing_receipt_cannot_be_flattened():
    """Fail closed. An unrouted surface was never measured against the floor."""
    store, cursor = scripted_store(None)
    with pytest.raises(RuntimeError, match="routing"):
        store.record_flattening(flattening())
    assert not cursor.wrote_a_flattening


def test_a_diagnostic_surface_cannot_be_flattened():
    """The PHerc0268 case: certified geometry, two square millimetres, refused."""
    store, cursor = scripted_store({"receipt": receipt_for(TINY_AREA)})
    with pytest.raises(RuntimeError, match="SMALL_SURFACE_DIAGNOSTIC"):
        store.record_flattening(flattening())
    assert not cursor.wrote_a_flattening


def test_a_routing_receipt_whose_digest_no_longer_verifies_fails_closed():
    """A forged receipt is not a route. Same refusal as having none at all."""
    forged = {**receipt_for(TINY_AREA), "route": routing.STANDARD}
    assert routing.verify_receipt(forged) is False
    store, cursor = scripted_store({"receipt": forged})
    with pytest.raises(RuntimeError, match="routing"):
        store.record_flattening(flattening())
    assert not cursor.wrote_a_flattening


def test_a_receipt_that_is_not_a_document_at_all_fails_closed():
    for row in ({"receipt": None}, {"receipt": []}, {"receipt": "STANDARD_QC_PENDING"},
                {}, {"receipt": {"route": routing.STANDARD}}):
        store, cursor = scripted_store(row)
        with pytest.raises(RuntimeError, match="routing"):
            store.record_flattening(flattening())
        assert not cursor.wrote_a_flattening


def test_a_failed_flattening_is_refused_on_the_same_terms():
    """A failure row is what makes the queue retry the surface, so it is a write
    onto the standard path too and a diagnostic surface may not have one."""
    store, cursor = scripted_store({"receipt": receipt_for(TINY_AREA)})
    with pytest.raises(RuntimeError, match="SMALL_SURFACE_DIAGNOSTIC"):
        store.record_flattening(
            flattening(state="FLATTENING_FAILED", error="gone",
                       artifact_uri=None, artifact_sha256=None))
    assert not cursor.wrote_a_flattening


def test_a_standard_surface_is_flattened_exactly_as_before():
    store, cursor = scripted_store({"receipt": receipt_for(STANDARD_AREA)})
    result = store.record_flattening(flattening())
    assert cursor.wrote_a_flattening
    assert result["inserted"] is True


def test_the_route_is_read_before_the_flattening_is_written():
    """The order is the gate. Reading after the insert is a second opinion."""
    store, cursor = scripted_store({"receipt": receipt_for(STANDARD_AREA)})
    store.record_flattening(flattening())
    read = next(index for index, statement in enumerate(cursor.executed)
                if ROUTING_TABLE in statement)
    write = next(index for index, statement in enumerate(cursor.executed)
                 if "INSERT INTO surface_flattenings" in statement)
    assert read < write


def test_the_gate_asks_the_shared_helper_and_not_a_column_comparison():
    """Gating on `route == "STANDARD_QC_PENDING"` skips the digest check, which
    is what makes a stored receipt worth more than a stored boolean."""
    source = (ROOT / "framework/stages/01-segmentation/fleet/postgres_store.py"
              ).read_text(encoding="utf-8")
    gate = source[source.index("def _require_standard_route("):
                  source.index("def record_flattening(")]
    writer = source[source.index("def record_flattening("):
                    source.index("def flattenings(")]
    assert "enters_canonical_downstream" in gate
    assert "_require_standard_route" in writer
    # The literal appears only as surface_routing.STANDARD in the refusal text.
    assert f'"{routing.STANDARD}"' not in gate
    assert f"'{routing.STANDARD}'" not in gate


# --- the reader -------------------------------------------------------------


def test_routing_receipt_returns_the_document_the_router_signed():
    signed = receipt_for(STANDARD_AREA)
    store, _cursor = scripted_store({"receipt": signed})
    read = store.routing_receipt(SURFACE)
    assert read == signed
    assert routing.verify_receipt(read) is True


def test_routing_receipt_accepts_a_document_delivered_as_text():
    signed = receipt_for(STANDARD_AREA)
    store, _cursor = scripted_store({"receipt": json.dumps(signed)})
    assert store.routing_receipt(SURFACE) == signed


def test_routing_receipt_is_none_when_the_surface_predates_routing():
    store, _cursor = scripted_store(None)
    assert store.routing_receipt(SURFACE) is None


def test_both_control_planes_return_the_identical_routing_document(tmp_path):
    """The parity that matters is the document, not the column it came out of."""
    sqlite_store = FleetStore(tmp_path / "fleet.sqlite")
    sqlite_store.initialize()
    snapshot = sqlite_store.register_snapshot({
        "sample_id": "PHerc0268",
        "ct_uri": "https://example.invalid/ct.zarr",
        "m7_uri": "https://example.invalid/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    surface = sqlite_store.import_surface({
        "surface_id": SURFACE, "source_snapshot_id": snapshot,
        "sample_id": "PHerc0268", "artifact_sha256": "d" * 64,
        "artifact_uri": "s3://bucket/tiny",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": TINY_AREA,
        "sample_points": [], "state": "QC_SCREENED",
        "physical_qc_state": "UNVALIDATED"})
    from_sqlite = sqlite_store.routing_receipt(surface)
    assert from_sqlite["route"] == routing.DIAGNOSTIC

    store, _cursor = scripted_store({"receipt": from_sqlite})
    assert store.routing_receipt(surface) == from_sqlite


# --- against a real PostgreSQL ---------------------------------------------

DSN = os.environ.get("HELENA_TEST_DSN")


@pytest.mark.skipif(not DSN, reason="set HELENA_TEST_DSN to a throwaway PostgreSQL")
def test_the_gate_holds_against_a_real_postgresql():
    """Everything above scripts the cursor; this one needs the table to exist.

    Migration v20 created it, so this is the check that the gate holds against
    a real server rather than against a fake that agrees with it.
    """
    import uuid

    store = PostgresFleetStore(DSN)
    store.initialize()
    scroll = f"TEST{uuid.uuid4().hex[:10]}"
    snapshot = store.register_snapshot({
        "sample_id": scroll, "ct_uri": f"https://example.invalid/{scroll}/ct.zarr",
        "m7_uri": f"https://example.invalid/{scroll}/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    tiny = store.import_surface({
        "surface_id": f"{scroll}-tiny", "source_snapshot_id": snapshot,
        "sample_id": scroll, "artifact_sha256": "d" * 64,
        "artifact_uri": f"s3://bucket/{scroll}-tiny",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": TINY_AREA,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})

    stored = store.routing_receipt(tiny)
    assert stored is not None and stored["route"] == routing.DIAGNOSTIC
    assert routing.verify_receipt(stored) is True

    with pytest.raises(RuntimeError, match="SMALL_SURFACE_DIAGNOSTIC"):
        store.record_flattening(flattening(surface_id=tiny))
