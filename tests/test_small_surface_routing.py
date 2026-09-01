"""A surface too small for the acceptance path is evidence, not a verdict.

The effort allocator already refuses to spend on anything under 0.10 cm2. What
was missing is what happens to the surface that falls under it: it was reaching
physical QC, where a bounded negative on 0.02 cm2 of papyrus reads exactly like
a negative on 0.5 cm2 and means nothing like it.

So the routing decides one thing and refuses to decide the other. Below the
floor is `SMALL_SURFACE_DIAGNOSTIC`: too small for the standard path, and
explicitly not a claim that there is no ink there. The distinction is the whole
point of the class, so it is asserted rather than assumed.

The router is pure. Given an area and a frozen policy it returns the same verdict
and the same receipt bytes forever, which is what makes the receipt worth storing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/stages/01-segmentation"))

from fleet import surface_routing as routing  # noqa: E402

# PHerc0268, the surface this class exists for: measured, preserved, and never
# once described as absence of ink.
PHERC0268_AREA_CM2 = 0.01983222455087575
PHERC0268_GRID = [14, 14]
PHERC0268_TRIANGLES = 132


@pytest.fixture
def policy() -> dict:
    return routing.load_policy()


# -- the frozen policy -------------------------------------------------------

def test_the_floor_is_the_one_effort_allocation_already_uses(policy) -> None:
    assert policy["minimum_area_cm2"] == 0.10
    assert policy["profile_id"] == routing.PROFILE_ID
    assert policy["policy_version"] == "1.0.0"


def test_the_policy_on_disk_is_the_policy_in_the_module(policy) -> None:
    """A second copy of a threshold is a second threshold."""
    on_disk = json.loads(
        (ROOT / "framework/profiles/01-segmentation"
              / "small-surface-routing-1.0.0.json").read_text()
    )
    assert on_disk == policy


# -- the routing decision ----------------------------------------------------

def test_at_or_above_the_floor_takes_the_standard_path(policy) -> None:
    for area in (0.10, 0.1000001, 0.5, 12.0):
        assert routing.route(area, policy=policy)[0] == routing.STANDARD, (
            f"{area} cm2 is at or above the floor and must route as it does today"
        )


def test_below_the_floor_is_diagnostic(policy) -> None:
    for area in (0.0999999, PHERC0268_AREA_CM2, 1e-9):
        assert routing.route(area, policy=policy)[0] == routing.DIAGNOSTIC


def test_diagnostic_is_not_a_statement_about_ink(policy) -> None:
    """The reason this class exists, so it is in the receipt and not the prose."""
    _, evidence = routing.route(PHERC0268_AREA_CM2, policy=policy)
    assert evidence["ink_claim"] == "NONE_MADE"
    assert evidence["is_absence_evidence"] is False
    why = evidence["why"].lower()
    assert "ink" not in why or "no ink" not in why


def test_an_unusable_area_fails_closed(policy) -> None:
    """Absent, negative, or not a number: refuse rather than pick a path.

    Defaulting either way is wrong. STANDARD would push an unmeasured surface
    into physical QC; DIAGNOSTIC would quarantine a good one on a measurement
    bug. Both are decisions the data does not support.
    """
    for bad in (None, -1.0, float("nan"), float("inf"), "0.5", True, False):
        with pytest.raises(ValueError):
            routing.route(bad, policy=policy)


# -- the receipt -------------------------------------------------------------

def _receipt(policy, area=PHERC0268_AREA_CM2):
    return routing.build_receipt(
        surface_id="PHerc0268-tiny",
        area_cm2=area,
        policy=policy,
        measurement={"grid_xy": PHERC0268_GRID, "triangles": PHERC0268_TRIANGLES,
                     "valid_coordinate_fraction": 0.87,
                     "bbox_xyz": [[0, 0, 0], [14, 14, 3]]},
        read_set={"source_snapshot_id": "snap-1", "artifact_sha256": "a" * 64,
                  "geometry_qc_state": "GEOMETRY_CERTIFIED"},
    )


def test_the_receipt_carries_what_the_decision_was_made_from(policy) -> None:
    receipt = _receipt(policy)
    assert receipt["schema"] == routing.RECEIPT_SCHEMA
    assert receipt["route"] == routing.DIAGNOSTIC
    assert receipt["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert receipt["minimum_area_cm2"] == 0.10
    assert receipt["policy_version"] == "1.0.0"
    assert receipt["profile_id"] == routing.PROFILE_ID
    # The read-set is what makes the decision reproducible against the same
    # inputs rather than merely repeatable in the same process.
    assert receipt["read_set"]["artifact_sha256"] == "a" * 64
    assert receipt["measurement"]["triangles"] == PHERC0268_TRIANGLES


def test_the_receipt_is_deterministic(policy) -> None:
    assert _receipt(policy) == _receipt(policy)
    assert _receipt(policy)["receipt_sha256"] == _receipt(policy)["receipt_sha256"]


def test_tampering_with_the_receipt_is_detectable(policy) -> None:
    receipt = _receipt(policy)
    assert routing.verify_receipt(receipt) is True

    for field, value in (("route", routing.STANDARD),
                         ("measured_area_cm2", 5.0),
                         ("minimum_area_cm2", 0.001),
                         ("policy_version", "9.9.9")):
        forged = {**receipt, field: value}
        assert routing.verify_receipt(forged) is False, (
            f"changing {field} left the receipt verifying, so the digest does "
            "not cover the field the decision turns on"
        )


def test_a_receipt_without_its_digest_does_not_verify(policy) -> None:
    receipt = {k: v for k, v in _receipt(policy).items() if k != "receipt_sha256"}
    assert routing.verify_receipt(receipt) is False


# -- what may be shown in public --------------------------------------------

def test_public_diagnostics_are_sanitized(policy) -> None:
    """The classification is public; the internal read-set is not."""
    public = routing.sanitize_receipt(_receipt(policy))
    assert public["route"] == routing.DIAGNOSTIC
    assert public["measured_area_cm2"] == PHERC0268_AREA_CM2
    assert public["is_absence_evidence"] is False
    for private in ("read_set", "measurement"):
        assert private not in public, f"{private} reached a public diagnostic"


def test_sanitizing_does_not_mutate_the_receipt(policy) -> None:
    receipt = _receipt(policy)
    routing.sanitize_receipt(receipt)
    assert routing.verify_receipt(receipt) is True


# -- PHerc0268, the case in the plan ----------------------------------------

def test_pherc0268_is_preserved_and_not_called_absence_of_ink(policy) -> None:
    route, evidence = routing.route(PHERC0268_AREA_CM2, policy=policy)
    receipt = _receipt(policy)
    assert route == routing.DIAGNOSTIC
    assert receipt["preserved"] is True, (
        "the surface is evidence; nothing about being small discards it"
    )
    assert evidence["is_absence_evidence"] is False
    assert routing.enters_standard_qc(receipt) is False
    assert routing.enters_canonical_downstream(receipt) is False


def test_a_normal_surface_still_goes_where_it_went(policy) -> None:
    receipt = _receipt(policy, area=0.5)
    assert receipt["route"] == routing.STANDARD
    assert routing.enters_standard_qc(receipt) is True
    assert routing.enters_canonical_downstream(receipt) is True
