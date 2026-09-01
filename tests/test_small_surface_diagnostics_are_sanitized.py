"""What a small-surface diagnostic may say outside, and what it may not.

The classification and the measured area are the finding, and publishing them is
the point: PHerc0268's 0.0198 cm2 is why the routing exists. The read-set is not.
It names the source snapshot the decision was made against, the artifact digest,
and the geometry verdict -- internal identities the campaign's own public
evidence lists under `redacted`, next to source-snapshot identifiers and task,
attempt and QC identifiers.

`sanitize_receipt` is an allow-list, so the interesting failure is not "a field
escaped the deny-list" -- there is no deny-list -- it is the allow-list growing.
So the checks below are written from both ends: nothing outside
`policy["public_fields"]` survives sanitizing, and no field of a real receipt is
in `public_fields` unless it is genuinely a finding about size.

Then the same question of every place a diagnostic actually reaches an operator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "framework/stages/01-segmentation"
for extra in (STAGE, ROOT, ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fleet import surface_routing as routing  # noqa: E402

# The one controlled-mission fixture in this repository, reused rather than
# rebuilt: a second copy of that wiring is a second thing to keep in step.
pytest.importorskip("fastapi", reason="the panel's dependencies are not here")
from test_first_letters_campaign_api import (  # noqa: E402,F401
    STORED_SCROLL,
    FakeSurfaceRows,
    controlled,
    ready,
)

PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_ARTIFACT_SHA256 = (
    "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")

# Values that exist only inside the receipt. If any of these strings appears in
# something published, it came from the read-set or the measurement.
PRIVATE_MARKERS = (
    "snapshot-private-0268",
    "sample-private-0268",
    PHERC0268_ARTIFACT_SHA256,
    "GEOMETRY_CERTIFIED",
    "private-measurement-marker",
)


@pytest.fixture
def policy() -> dict:
    return routing.load_policy()


def _receipt(policy) -> dict:
    return routing.build_receipt(
        surface_id="PHerc0268-tiny", area_cm2=PHERC0268_AREA_CM2, policy=policy,
        measurement={"grid_xy": [14, 14], "triangles": 132,
                     "finite_coordinates": 84,
                     "note": "private-measurement-marker"},
        read_set={"source_snapshot_id": "snapshot-private-0268",
                  "sample_id": "sample-private-0268",
                  "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
                  "geometry_qc_state": "GEOMETRY_CERTIFIED"})


# ---------------------------------------------------------------------------
# The sanitizer itself, from both ends
# ---------------------------------------------------------------------------

def test_sanitizing_drops_the_read_set_and_the_measurement(policy) -> None:
    public = routing.sanitize_receipt(_receipt(policy))
    assert "read_set" not in public
    assert "measurement" not in public


def test_nothing_outside_the_public_field_list_survives(policy) -> None:
    """The allow-list is the contract, so it is checked as one.

    A field added to `build_receipt` is private by construction. This fails the
    moment that stops being true.
    """
    receipt = _receipt(policy)
    public = routing.sanitize_receipt(receipt)
    allowed = set(policy["public_fields"])
    assert set(public) <= allowed
    assert set(public) == allowed & set(receipt), (
        "sanitizing dropped a field the policy allows, or invented one it does not")


def test_no_private_value_survives_sanitizing_anywhere_in_the_document(
    policy,
) -> None:
    """Not "the key is gone" -- the *value* is gone, at any depth.

    A future receipt that copies the artifact digest into a public field would
    pass a key-name check and leak exactly as much.
    """
    published = json.dumps(routing.sanitize_receipt(_receipt(policy)))
    for marker in PRIVATE_MARKERS:
        assert marker not in published, f"{marker} reached a public diagnostic"


def test_the_public_fields_are_the_finding_and_not_the_lineage(policy) -> None:
    """What the allow-list is allowed to contain, named rather than counted."""
    assert set(policy["public_fields"]) == {
        "schema", "surface_id", "route", "measured_area_cm2",
        "minimum_area_cm2", "policy_version", "profile_id", "preserved",
        "is_absence_evidence", "ink_claim", "why", "receipt_sha256"}
    for field in policy["public_fields"]:
        assert not field.startswith(("source_", "artifact_", "geometry_")), (
            f"{field} is lineage, not a finding about size")


def test_the_public_half_still_carries_the_finding(policy) -> None:
    """Sanitizing must not sanitize away the reason the receipt exists."""
    public = routing.sanitize_receipt(_receipt(policy))
    assert public["route"] == routing.DIAGNOSTIC
    assert public["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert public["minimum_area_cm2"] == 0.10
    assert public["is_absence_evidence"] is False
    assert public["ink_claim"] == "NONE_MADE"
    assert public["receipt_sha256"] == _receipt(policy)["receipt_sha256"]


def test_sanitizing_does_not_touch_the_receipt_it_was_given(policy) -> None:
    receipt = _receipt(policy)
    before = json.dumps(receipt, sort_keys=True)
    routing.sanitize_receipt(receipt)
    assert json.dumps(receipt, sort_keys=True) == before
    assert routing.verify_receipt(receipt) is True


def test_the_sanitized_half_is_not_mistakable_for_the_whole(policy) -> None:
    """A public receipt must not verify: it is a derivative, not the evidence.

    If it verified, a sanitized copy could be presented as the stored decision
    while missing the read-set the decision was actually made against.
    """
    assert routing.verify_receipt(routing.sanitize_receipt(_receipt(policy))) is False


# ---------------------------------------------------------------------------
# Every place a diagnostic actually reaches an operator
# ---------------------------------------------------------------------------

def test_the_readiness_endpoint_publishes_no_private_receipt_field(
    controlled, monkeypatch,
) -> None:
    """`GET /api/missions/{id}/first-letters-readiness` -> `small_surfaces`."""
    ready(controlled)
    app = controlled.app
    monkeypatch.setattr(app, "DSN", "postgresql://fixture.invalid/fleet")
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})
    monkeypatch.setitem(sys.modules, "psycopg", FakeSurfaceRows([
        ("PHerc0268-tiny", STORED_SCROLL, PHERC0268_AREA_CM2),
        ("surface-canonical", STORED_SCROLL, 1.25),
    ]))
    readiness = controlled.readiness()
    published = json.dumps(readiness)
    for marker in PRIVATE_MARKERS:
        assert marker not in published, (
            f"{marker} reached the readiness endpoint")
    diagnostics = readiness["small_surfaces"]
    for row in diagnostics["surfaces"]:
        assert "read_set" not in row
        assert "measurement" not in row
        assert set(row) <= {
            # What a row derived here can say: the page still answers for a
            # surface with no stored receipt, and says in `why` that it did.
            "surface_id", "sample_id", "measured_area_cm2", "route", "why",
            "ink_claim", "is_absence_evidence",
            # Plus the rest of `sanitize_receipt`'s output, because a row backed
            # by a stored receipt now *is* that receipt sanitized -- which is
            # what makes the published finding checkable rather than merely
            # plausible. Spelled out rather than read from
            # `policy["public_fields"]` on purpose: derived from the policy,
            # this pin would follow an allow-list that grew and never say so.
            "schema", "minimum_area_cm2", "policy_version", "profile_id",
            "preserved", "receipt_sha256"}, (
            f"the readiness page publishes {sorted(row)} per surface")


def test_the_readiness_endpoint_never_publishes_a_connection_string(
    controlled, monkeypatch,
) -> None:
    """The one credential-bearing string in scope for this page."""
    ready(controlled)
    app = controlled.app
    dsn = "postgresql://helena:super-secret-password@private-host:5432/fleet"
    monkeypatch.setattr(app, "DSN", dsn)
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})
    monkeypatch.setitem(sys.modules, "psycopg", FakeSurfaceRows([
        ("PHerc0268-tiny", STORED_SCROLL, PHERC0268_AREA_CM2)]))
    published = json.dumps(controlled.readiness())
    for forbidden in ("super-secret-password", "private-host", dsn):
        assert forbidden not in published


def test_a_failed_surface_read_publishes_no_connection_string(
    controlled, monkeypatch,
) -> None:
    """The error path, which is where a DSN usually escapes.

    `surfaces_reason` is built from the exception text, and a psycopg failure
    routinely quotes the connection it could not open.
    """
    ready(controlled)
    app = controlled.app
    dsn = "postgresql://helena:super-secret-password@private-host:5432/fleet"
    monkeypatch.setattr(app, "DSN", dsn)
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})

    class _Exploding:
        def connect(self, *_args, **_kwargs):
            raise RuntimeError(f"could not connect to {dsn}")

    monkeypatch.setitem(sys.modules, "psycopg", _Exploding())
    readiness = controlled.readiness()
    assert readiness["small_surfaces"]["surfaces_available"] is False
    published = json.dumps(readiness)
    for forbidden in ("super-secret-password", "private-host"):
        assert forbidden not in published, (
            "the small-surface read failed and quoted its connection string: "
            f"{readiness['small_surfaces']['surfaces_reason']!r}")


def test_a_published_diagnostic_never_reads_as_an_absence_of_ink(
    controlled, monkeypatch,
) -> None:
    """The sentence this whole class exists to prevent, checked where it would appear."""
    ready(controlled)
    app = controlled.app
    monkeypatch.setattr(app, "DSN", "postgresql://fixture.invalid/fleet")
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})
    monkeypatch.setitem(sys.modules, "psycopg", FakeSurfaceRows([
        ("PHerc0268-tiny", STORED_SCROLL, PHERC0268_AREA_CM2)]))
    said = json.dumps(controlled.readiness()).lower()
    for forbidden in ("contains no ink", "has no ink", "there is no ink",
                      "absence of ink", "no ink was found", "ink absent",
                      "holds no text", "contains no letters", "empty scroll"):
        assert forbidden not in said
    tiny = controlled.readiness()["small_surfaces"]["surfaces"][0]
    assert tiny["is_absence_evidence"] is False
    assert tiny["ink_claim"] == "NONE_MADE"


# ---------------------------------------------------------------------------
# Reading the stored decision, and serving it
# ---------------------------------------------------------------------------

@pytest.fixture
def routed_surface(tmp_path):
    """One real store holding one real receipt for PHerc0268's 0.0198 cm2.

    A real ``FleetStore`` rather than a stub because the thing under test is
    whether the panel reads what the fleet actually wrote: a stub would agree
    with whatever the panel expected, which is the failure being prevented.
    """
    from fleet.store import FleetStore

    store = FleetStore(tmp_path / "fleet.sqlite")
    store.initialize()
    snapshot = store.register_snapshot({
        "sample_id": STORED_SCROLL,
        "ct_uri": "https://example.invalid/ct.zarr",
        "m7_uri": "https://example.invalid/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    surface = store.import_surface({
        "surface_id": "PHerc0268-tiny", "source_snapshot_id": snapshot,
        "sample_id": STORED_SCROLL,
        "artifact_sha256": PHERC0268_ARTIFACT_SHA256,
        "artifact_uri": "s3://bucket/tiny",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]],
        "area_cm2": PHERC0268_AREA_CM2,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})
    return store, surface


def test_the_endpoint_serves_the_stored_receipt_sanitized(
    controlled, monkeypatch, routed_surface,
) -> None:
    """Not "a route exists" -- the body, against the receipt the fleet wrote."""
    import json as json_module

    store, surface = routed_surface
    app = controlled.app
    monkeypatch.setattr(app, "fleet_store_read_only", lambda: store)
    body = json_module.loads(
        bytes(app.api_surface_routing_receipt(surface).body))

    stored = store.routing_receipt(surface)
    assert stored["route"] == routing.DIAGNOSTIC
    assert "read_set" in stored, "the stored receipt is the one carrying lineage"
    assert body == routing.sanitize_receipt(stored), (
        "the endpoint serves something other than the sanitized stored receipt")
    assert body["receipt_sha256"] == stored["receipt_sha256"]
    assert body["measured_area_cm2"] == PHERC0268_AREA_CM2
    published = json_module.dumps(body)
    for marker in PRIVATE_MARKERS[:3]:
        assert marker not in published, f"{marker} reached the API"


def test_the_endpoint_refuses_rather_than_inventing_a_receipt(
    controlled, monkeypatch, routed_surface,
) -> None:
    """A surface nobody routed has no answer here, and 200 is not one."""
    from fastapi import HTTPException

    store, _ = routed_surface
    app = controlled.app
    monkeypatch.setattr(app, "fleet_store_read_only", lambda: store)
    with pytest.raises(HTTPException) as refusal:
        app.api_surface_routing_receipt("a-surface-nobody-routed")
    assert refusal.value.status_code == 404


def test_a_receipt_that_does_not_verify_is_not_served(
    controlled, monkeypatch, routed_surface,
) -> None:
    """A digest that does not check out is not a decision, so it is not served.

    Serving it would be worse than serving nothing: it carries a route, a policy
    version and an area, and looks exactly like the real thing.
    """
    from fastapi import HTTPException

    store, surface = routed_surface
    app = controlled.app
    tampered = {**store.routing_receipt(surface), "measured_area_cm2": 5.0}
    monkeypatch.setattr(app, "fleet_store_read_only", lambda: type(
        "Tampered", (), {"routing_receipt": lambda _self, _id: tampered})())
    with pytest.raises(HTTPException) as refusal:
        app.api_surface_routing_receipt(surface)
    assert refusal.value.status_code == 409


def test_the_readiness_row_is_the_stored_receipt_and_not_a_recomputation(
    controlled, monkeypatch, routed_surface,
) -> None:
    """The whole point of reading rather than deriving, checked as a value.

    The area column is deliberately wrong here. A page that recomputes reports
    5.0 cm2 and STANDARD; a page that reads the stored decision reports what was
    decided, which is the only thing a downstream gate will honour.
    """
    store, surface = routed_surface
    app = controlled.app
    ready(controlled)
    monkeypatch.setattr(app, "DSN", "postgresql://fixture.invalid/fleet")
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})
    monkeypatch.setattr(app, "fleet_store_read_only", lambda: store)
    monkeypatch.setitem(sys.modules, "psycopg", FakeSurfaceRows([
        (surface, STORED_SCROLL, 5.0)]))

    row = controlled.readiness()["small_surfaces"]["surfaces"][0]
    assert row["route"] == routing.DIAGNOSTIC, (
        "the readiness page believed the area column over the stored decision")
    assert row["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert row["receipt_sha256"] == store.routing_receipt(surface)["receipt_sha256"]
    assert "read_set" not in row and "measurement" not in row


def test_a_surface_with_no_stored_receipt_says_so_rather_than_going_quiet(
    controlled, monkeypatch,
) -> None:
    """The page still answers, and never lets the two answers look alike.

    Reporting nothing would hide a surface; reporting a derived route as though
    it were the stored one is what put an unrouted surface on the ink screen. So
    it answers, and the answer says which kind of answer it is.
    """
    app = controlled.app
    ready(controlled)
    monkeypatch.setattr(app, "DSN", "postgresql://fixture.invalid/fleet")
    monkeypatch.setattr(app, "mission_scrolls", lambda _m: {STORED_SCROLL})
    monkeypatch.setitem(sys.modules, "psycopg", FakeSurfaceRows([
        ("PHerc0268-tiny", STORED_SCROLL, PHERC0268_AREA_CM2)]))

    diagnostics = controlled.readiness()["small_surfaces"]
    row = diagnostics["surfaces"][0]
    assert row["route"] == routing.DIAGNOSTIC
    assert "receipt_sha256" not in row, (
        "a derived route was published as though a receipt stood behind it")
    assert "no verified routing receipt is stored" in row["why"]
    assert diagnostics["receipts_reason"], (
        "the page fell back to deriving and did not say why")


def test_the_published_diagnostic_is_the_stored_receipt_sanitized() -> None:
    import inspect
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    source = inspect.getsource(panel_app._first_letters_small_surface_readiness)
    assert "sanitize_receipt" in source, (
        "the endpoint publishes a freshly computed route rather than the stored "
        "decision, so nothing published is bound to a receipt digest")


def test_a_routing_receipt_is_retrievable_through_the_api() -> None:
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    routes = {getattr(route, "path", "") for route in panel_app.app.routes}
    assert any("routing-receipt" in path for path in routes), (
        f"no route serves a routing receipt; found {sorted(routes)[:5]}...")


def test_no_endpoint_leaks_a_receipt_field_it_never_serves() -> None:
    """The honest statement of today's exposure, so a regression is visible.

    Nothing publishes `read_set` or `measurement` right now for the simple
    reason that nothing publishes a receipt. That is a weaker guarantee than
    sanitizing, and it stops being true the moment a receipt is served -- so
    this pins the current surface rather than claiming it is safe by design.
    """
    import inspect
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    published = inspect.getsource(
        panel_app._first_letters_small_surface_readiness)
    assert "read_set" not in published and "measurement" not in published, (
        "the small-surface diagnostic now names a private receipt field; it "
        "must go through surface_routing.sanitize_receipt")
    assert "routing_receipt" not in published or "sanitize_receipt" in published, (
        "the small-surface diagnostic now reads a stored routing receipt "
        "without sanitizing it")
