"""Where a finalized surface goes, decided once and written down.

A surface below the effort floor was reaching physical QC, and a bounded
negative on 0.02 cm2 of papyrus reads exactly like a negative on 0.5 cm2 while
meaning nothing like it. This routes the small ones somewhere they stay
evidence: preserved, geometry certificate intact, out of the physical-QC FIFO
and out of every canonical downstream stage.

Two things it deliberately does not do. It does not say there is no ink there --
nothing here measures ink, and a surface too small to ask cannot answer. And it
does not reject the geometry: the shape was certified and stays certified.

The routing is pure. Given an area and the frozen policy it returns the same
verdict and the same receipt bytes forever, which is what makes a stored receipt
worth more than a stored boolean.

Leaving the diagnostic path is by expansion only: a new versioned grow or resume
attempt whose *new* surface independently clears the floor and every standard
gate. The original stays diagnostic permanently. Promoting it in place would be
claiming the measurement was wrong, and the measurement is the evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .common import content_sha256

PROFILE_ID = "small-surface-routing@1.0.0"
POLICY_SCHEMA = "campaignx.small_surface_routing_policy.v1"
RECEIPT_SCHEMA = "campaignx.small_surface_routing_receipt.v1"

STANDARD = "STANDARD_QC_PENDING"
DIAGNOSTIC = "SMALL_SURFACE_DIAGNOSTIC"
ROUTES = (STANDARD, DIAGNOSTIC)

_PROFILE_PATH = (
    Path(__file__).resolve().parents[3]
    / "profiles/01-segmentation/small-surface-routing-1.0.0.json"
)

_DIGEST_FIELDS = (
    "surface_id", "route", "measured_area_cm2", "minimum_area_cm2",
    "policy_version", "profile_id", "measurement", "read_set",
    "preserved", "is_absence_evidence",
)


def load_policy(path: Path | None = None) -> dict[str, Any]:
    """The frozen policy, read from disk rather than restated here.

    A threshold written twice is two thresholds, and the one that drifts is
    always the copy nobody is looking at.
    """
    policy = json.loads((path or _PROFILE_PATH).read_text())
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"not a small-surface routing policy: {policy.get('schema')}")
    if policy.get("profile_id") != PROFILE_ID:
        raise ValueError(f"policy names {policy.get('profile_id')}, not {PROFILE_ID}")
    floor = policy.get("minimum_area_cm2")
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or floor <= 0:
        raise ValueError("policy has no usable minimum_area_cm2")
    return policy


def _measured_area(value: Any) -> float:
    """The area, or a refusal.

    Defaulting either way is a decision the data does not support: STANDARD
    pushes an unmeasured surface into physical QC, DIAGNOSTIC quarantines a good
    one on a measurement bug. `True` is rejected before the numeric check
    because bool is an int and would otherwise route as 1.0 cm2.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"measured area is not a number: {value!r}")
    area = float(value)
    if not math.isfinite(area) or area < 0.0:
        raise ValueError(f"measured area is not usable: {area!r}")
    return area


def route(area_cm2: Any, *, policy: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Which path this surface takes, and the evidence for saying so."""
    area = _measured_area(area_cm2)
    floor = float(policy["minimum_area_cm2"])
    if area >= floor:
        return STANDARD, {
            "ink_claim": "NONE_MADE",
            "is_absence_evidence": False,
            "why": f"{area:.6g} cm2 is at or above the {floor:.6g} cm2 floor",
        }
    return DIAGNOSTIC, {
        "ink_claim": "NONE_MADE",
        "is_absence_evidence": False,
        # Worded as what it is -- a statement about size -- because the whole
        # failure this class prevents is a size result read as a content result.
        "why": (f"{area:.6g} cm2 is below the {floor:.6g} cm2 floor: too small "
                "for the standard acceptance path, and no claim either way "
                "about what is written on it"),
    }


def build_receipt(*, surface_id: str, area_cm2: Any, policy: dict[str, Any],
                  measurement: dict[str, Any],
                  read_set: dict[str, Any]) -> dict[str, Any]:
    """An immutable record of one routing decision and what produced it.

    `read_set` is what the decision was made against -- snapshot, artifact
    digest, geometry verdict -- so the decision is reproducible against the same
    inputs rather than merely repeatable in the same process.
    """
    if not isinstance(surface_id, str) or not surface_id.strip():
        raise ValueError("a routing receipt needs the surface it routes")
    for name, value in (("measurement", measurement), ("read_set", read_set)):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must be a mapping")

    decision, evidence = route(area_cm2, policy=policy)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "surface_id": surface_id,
        "route": decision,
        "measured_area_cm2": _measured_area(area_cm2),
        "minimum_area_cm2": float(policy["minimum_area_cm2"]),
        "policy_version": policy["policy_version"],
        "profile_id": policy["profile_id"],
        "measurement": measurement,
        "read_set": read_set,
        # Being small is a reason to look harder, never a reason to drop it.
        "preserved": True,
        **evidence,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def receipt_for_surface(surface: dict[str, Any],
                        *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """The receipt one surface earns, built identically wherever it is written.

    Both stores call this rather than assembling the measurement and read-set
    themselves. Two assemblies of the same fields are two receipt formats, and
    the one that drifts is the one whose store nobody ran the test against --
    which for this repository has historically been PostgreSQL, the one the
    deployment actually uses.
    """
    return build_receipt(
        surface_id=str(surface["surface_id"]),
        area_cm2=surface.get("area_cm2"),
        policy=policy if policy is not None else load_policy(),
        measurement={
            "bbox_xyz": surface.get("bbox_xyz"),
            "sample_point_count": len(surface.get("sample_points") or []),
        },
        read_set={
            "source_snapshot_id": surface.get("source_snapshot_id"),
            "sample_id": surface.get("sample_id"),
            "artifact_sha256": surface.get("artifact_sha256"),
            "geometry_qc_state": surface.get("geometry_qc_state"),
            # Which diagnostic surface this one continues, if any. Inside the
            # digest, so the claim that a surface is a second attempt at a tiny
            # one is as immutable as the measurement that made the first one
            # diagnostic. `None` for the ordinary case, which is most of them.
            "expands_surface_id": surface.get("resumes_surface"),
        },
    )


def agrees_with_measurement(receipt: Any, area_cm2: Any,
                            *, policy: dict[str, Any]) -> bool:
    """Whether a stored decision is still the decision this area produces.

    A receipt is immutable, and the area behind it is not: the QC backfill path
    may replace an unvalidated surface's `area_cm2`. A surface imported at half
    a square centimetre and reconciled to two square millimetres would otherwise
    keep a STANDARD receipt describing an area it no longer has, which is the
    original failure wearing a valid signature.

    Re-deciding rather than trusting also means a policy version bump cannot
    leave old receipts silently authoritative.
    """
    if not verify_receipt(receipt):
        return False
    try:
        decision, _ = route(area_cm2, policy=policy)
    except ValueError:
        return False
    return (receipt.get("route") == decision
            and receipt.get("measured_area_cm2") == _measured_area(area_cm2)
            and receipt.get("minimum_area_cm2")
                == float(policy["minimum_area_cm2"])
            and receipt.get("policy_version") == policy["policy_version"]
            and receipt.get("profile_id") == policy["profile_id"])


def _digest(receipt: dict[str, Any]) -> str:
    return content_sha256({field: receipt.get(field) for field in _DIGEST_FIELDS})


def verify_receipt(receipt: Any) -> bool:
    """Whether the receipt still says what it was signed saying."""
    if not isinstance(receipt, dict):
        return False
    stored = receipt.get("receipt_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        return False
    return stored == _digest(receipt)


def sanitize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """The part of a routing decision that may be shown outside.

    The classification and the measured area are the finding. The read-set names
    internal snapshots and artifact digests and is not.
    """
    policy = load_policy()
    public = {field: receipt[field] for field in policy["public_fields"]
              if field in receipt}
    return public


def enters_standard_qc(receipt: dict[str, Any]) -> bool:
    """Physical QC admits exactly one route, and never a forged receipt."""
    return verify_receipt(receipt) and receipt.get("route") == STANDARD


def enters_canonical_downstream(receipt: dict[str, Any]) -> bool:
    """P3 and everything after it. Same answer, named separately because the
    two questions are asked in different stages and would drift if shared."""
    return verify_receipt(receipt) and receipt.get("route") == STANDARD
