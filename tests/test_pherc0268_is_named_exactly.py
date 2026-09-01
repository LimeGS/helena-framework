"""PHerc0268 by its frozen identity, and by the spelling each layer uses.

Two failure modes, and they compound.

The first is drift: this surface is quoted by area, triangle count and digest in
prose across the repository, and a number retyped from a paragraph is a number
that can be wrong. The single frozen truth is
`docs/first-letters/first-letters-hybrid-20260802/evidence.json`, so every
identifier below is read out of that file rather than restated here, and the
restatements that already exist in other tests are checked against it.

The second is spelling. The bucket, the mission manifest and that evidence file
all call it `PHerc0268`; the frozen catalog registers it as `PHerc268`, and the
catalog name is what the control plane stores and what every frozen plan hashes.
Filtering by the raw request name matched no row, so pages reported zero while
the fleet held the surface -- which for a small-surface diagnostic would read as
"there is no such surface" rather than "the query asked for the wrong name". So
the endpoint that surfaces this diagnostic has to translate, and the check is
that it does, not that it happens to work on a fixture where both spell alike.
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
from fleet.store import FleetStore  # noqa: E402

pytest.importorskip("fastapi", reason="the panel's dependencies are not here")
from test_first_letters_campaign_api import (  # noqa: E402,F401
    controlled,
    ready,
)

EVIDENCE = (ROOT / "docs/first-letters/first-letters-hybrid-20260802"
            / "evidence.json")

# The bucket / mission / evidence spelling, and the control-plane spelling.
REQUESTED = "PHerc0268"
STORED = "PHerc268"


@pytest.fixture(scope="module")
def frozen() -> dict:
    """The one authority. Read, never restated."""
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))["surface_path"]


@pytest.fixture
def policy() -> dict:
    return routing.load_policy()


# ---------------------------------------------------------------------------
# The frozen identity
# ---------------------------------------------------------------------------

def test_the_frozen_evidence_still_says_what_this_class_was_built_from(
    frozen,
) -> None:
    """A guard on the authority itself, so drift is caught at the source.

    If this file ever changes, every constant derived from it below changes
    silently and correctly -- but the change should be deliberate, so it is
    stated once, here, and nowhere else.
    """
    assert frozen["scroll"] == REQUESTED
    assert frozen["area_cm2"] == 0.01983222455087575
    assert frozen["grid_shape_y_x"] == [14, 14]
    assert frozen["valid_triangle_count"] == 132
    assert frozen["finite_coordinate_count"] == 84
    assert frozen["artifact_sha256"] == (
        "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6")
    assert frozen["artifact_manifest_sha256"] == (
        "5d45a67c7fa45e5cb82c738a3be52d14273f47472045bb2101c3db6a13ceb09a")


def test_the_artifact_and_its_manifest_are_two_different_digests(frozen) -> None:
    """They name different bytes, and swapping them would still look like a hash."""
    assert frozen["artifact_sha256"] != frozen["artifact_manifest_sha256"]
    for digest in (frozen["artifact_sha256"], frozen["artifact_manifest_sha256"]):
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_every_restatement_of_these_identifiers_matches_the_frozen_file(
    frozen,
) -> None:
    """The constants other test files hardcode, checked against the authority.

    Each of these is a number typed a second time. This is what makes the second
    copy safe rather than a place for the two to disagree.
    """
    restatements = {
        "tests/test_small_surface_is_routed_at_finalization.py": (
            frozen["area_cm2"], frozen["artifact_sha256"],
            frozen["valid_triangle_count"], frozen["finite_coordinate_count"]),
        "tests/test_small_surface_routing.py": (
            frozen["area_cm2"], frozen["valid_triangle_count"]),
        "tests/test_small_surface_routing_mutations.py": (
            frozen["area_cm2"], frozen["artifact_sha256"]),
        "tests/test_small_surface_diagnostics_are_sanitized.py": (
            frozen["area_cm2"], frozen["artifact_sha256"]),
        "tests/test_small_surface_routing_postgres_parity.py": (
            frozen["area_cm2"], frozen["artifact_sha256"]),
        "docs/superpowers/plans/2026-08-02-first-letters-discovery-recovery.md": (
            frozen["area_cm2"], frozen["valid_triangle_count"]),
    }
    for relative, values in restatements.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for value in values:
            assert str(value) in text, (
                f"{relative} quotes PHerc0268 but not with {value!r} from "
                f"{EVIDENCE.relative_to(ROOT)}")


def test_the_frozen_area_routes_diagnostic_and_by_a_wide_margin(
    frozen, policy,
) -> None:
    """Not a borderline call: 0.0198 cm2 is a fifth of the floor."""
    route, evidence = routing.route(frozen["area_cm2"], policy=policy)
    assert route == routing.DIAGNOSTIC
    assert frozen["area_cm2"] < policy["minimum_area_cm2"] / 4
    assert evidence["is_absence_evidence"] is False
    assert evidence["ink_claim"] == "NONE_MADE"


def test_the_receipt_binds_the_frozen_artifact_and_not_its_manifest(
    frozen, policy,
) -> None:
    """The read-set names the surface bytes; the manifest digest is a different
    document and must not stand in for them."""
    receipt = routing.build_receipt(
        surface_id=f"{STORED}-surface", area_cm2=frozen["area_cm2"],
        policy=policy,
        measurement={"grid_shape_y_x": frozen["grid_shape_y_x"],
                     "valid_triangle_count": frozen["valid_triangle_count"],
                     "finite_coordinate_count": frozen["finite_coordinate_count"]},
        read_set={"artifact_sha256": frozen["artifact_sha256"],
                  "artifact_manifest_sha256": frozen["artifact_manifest_sha256"],
                  "geometry_qc_state": frozen["geometry_state"]})
    assert receipt["read_set"]["artifact_sha256"] == frozen["artifact_sha256"]
    assert routing.verify_receipt(receipt) is True
    swapped = {**receipt, "read_set": {
        **receipt["read_set"],
        "artifact_sha256": frozen["artifact_manifest_sha256"]}}
    assert routing.verify_receipt(swapped) is False


def test_the_frozen_terminal_states_are_not_a_finding_about_the_scroll(
    frozen,
) -> None:
    """What the campaign actually recorded, and what it explicitly did not.

    EMPTY and INK_SCREEN_INSUFFICIENT are terminal states of a screening path
    over two square millimetres. The evidence file says so in its own words;
    this asserts the words are still there, because the routing exists to stop
    that distinction from being lost.
    """
    assert frozen["liveness_verdict"] == "EMPTY"
    assert frozen["physical_qc_state"] == "INK_SCREEN_INSUFFICIENT"
    assert frozen["downstream_admissible"] is False
    non_claim = frozen["non_claim"].lower()
    assert "not an ink-absence finding" in non_claim
    assert "for this surface path" in non_claim


# ---------------------------------------------------------------------------
# The spelling, and the endpoint that has to translate it
# ---------------------------------------------------------------------------

def test_the_control_plane_and_the_evidence_spell_it_differently(frozen) -> None:
    """The premise. If these ever converge, the tests below stop testing anything."""
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    assert frozen["scroll"] == REQUESTED
    assert panel_app.catalog_sample_id(REQUESTED) == STORED
    assert REQUESTED != STORED, (
        "the two spellings converged; every normalisation test below is now "
        "vacuous and needs a new example")


def test_normalisation_is_idempotent_and_accepts_every_spelling() -> None:
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    for spelling in (REQUESTED, STORED, "pherc-0268", "PHERC_0268", "pherc268"):
        assert panel_app.catalog_sample_id(spelling) == STORED
    assert panel_app.catalog_sample_id(STORED) == STORED
    assert panel_app.stored_scroll(REQUESTED) == STORED
    assert panel_app.stored_scroll(None) is None


def test_the_catalog_registers_the_stored_spelling_and_not_the_other() -> None:
    """Read from the frozen catalog, because that is what makes it the truth."""
    sys.path.insert(0, str(ROOT / "panel"))
    import app as panel_app

    rows = json.loads(panel_app.CATALOG.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("entries") or []
    registered = {str(row.get("sample_id")) for row in rows}
    assert STORED in registered
    assert REQUESTED not in registered


def test_the_small_surface_endpoint_queries_the_stored_spelling(
    controlled, monkeypatch,
) -> None:
    """The check that would have caught the page of zeros.

    The mission names `PHerc0268`. The control plane stores `PHerc268`. What
    matters is the value that reaches the SQL parameter, so it is captured
    there rather than inferred from the answer.
    """
    from framework.contracts import mission as mission_contract

    ready(controlled)
    app = controlled.app
    monkeypatch.setattr(app, "DSN", "postgresql://fixture.invalid/fleet")
    # A real mission manifest naming the scroll the way a mission does.
    monkeypatch.setattr(
        mission_contract, "resolve",
        lambda _runs, _mission: (controlled.directory,
                                 {"scrolls": [REQUESTED]}))
    recorder = _RecordingRows([(f"{STORED}-surface", STORED,
                                0.01983222455087575)])
    monkeypatch.setitem(sys.modules, "psycopg", recorder)

    diagnostics = app._first_letters_small_surface_readiness("first-letters")
    assert diagnostics["surfaces_available"] is True
    assert recorder.parameters, "the endpoint issued no query"
    scope = recorder.parameters[0][0]
    assert scope == [STORED], (
        f"the endpoint asked the control plane for {scope}; it stores {STORED!r}")
    assert REQUESTED not in scope


def test_the_untranslated_spelling_would_have_found_nothing(
    controlled, monkeypatch,
) -> None:
    """Proof the test above is not passing by coincidence.

    A store holding the surface under the catalog spelling returns nothing for
    the mission's spelling, which is exactly the empty page this normalisation
    exists to prevent.
    """
    store = FleetStore(controlled.directory / "spelling.sqlite")
    store.initialize()
    snapshot = store.register_snapshot({
        "sample_id": STORED,
        "ct_uri": "https://example.invalid/ct.zarr",
        "m7_uri": "https://example.invalid/m7.zarr",
        "shape_xyz": [1024, 1024, 2048], "voxel_size_um": 9.362,
        "coordinate_frame": "ct_l0_xyz"})
    store.import_surface({
        "surface_id": f"{STORED}-surface", "source_snapshot_id": snapshot,
        "sample_id": STORED,
        "artifact_sha256":
            "d6791503f3d5e1418ba92b9b3dd2b73051558c89e0fefdc280ffbb3b2a5752c6",
        "artifact_uri": "s3://bucket/0268",
        "bbox_xyz": [[0, 0, 0], [14, 14, 3]], "area_cm2": 0.01983222455087575,
        "state": "QC_SCREENED", "physical_qc_state": "UNVALIDATED"})

    with store.connect() as connection:
        raw = connection.execute(
            "SELECT count(*) AS n FROM surfaces WHERE sample_id=?",
            (REQUESTED,)).fetchone()["n"]
        translated = connection.execute(
            "SELECT count(*) AS n FROM surfaces WHERE sample_id=?",
            (STORED,)).fetchone()["n"]
    assert raw == 0, "the two spellings now match; this test proves nothing"
    assert translated == 1
    receipt = store.routing_receipt(f"{STORED}-surface")
    assert receipt["route"] == routing.DIAGNOSTIC


class _RecordingRows:
    """A psycopg stand-in that keeps the parameters it was asked with."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.queries: list[str] = []
        self.parameters: list = []

    def connect(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def cursor(self):
        return self

    def execute(self, query, parameters=None):
        self.queries.append(query)
        if parameters is not None:
            self.parameters.append(parameters)
        return self

    def fetchall(self):
        return list(self.rows)

