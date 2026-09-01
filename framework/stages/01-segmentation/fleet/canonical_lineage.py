"""One fail-closed canonical-lineage guard shared by every surface boundary."""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

from .common import content_sha256


class CanonicalLineageBoundary(str, Enum):
    P1_FINALIZATION_INSERT = "P1_FINALIZATION_INSERT"
    DIRECT_SURFACE_IMPORT = "DIRECT_SURFACE_IMPORT"
    P2_QUEUE_ADMISSION = "P2_QUEUE_ADMISSION"
    P2_EXECUTION_RESOLUTION = "P2_EXECUTION_RESOLUTION"
    PHYSICAL_QC_DIRECT_ENQUEUE = "PHYSICAL_QC_DIRECT_ENQUEUE"
    PHYSICAL_QC_CLAIM_RESOLUTION = "PHYSICAL_QC_CLAIM_RESOLUTION"
    P3_QUEUE_ADMISSION = "P3_QUEUE_ADMISSION"
    P3_EXECUTION_RESOLUTION = "P3_EXECUTION_RESOLUTION"
    P4_QUEUE_ADMISSION = "P4_QUEUE_ADMISSION"
    P4_EXECUTION_RESOLUTION = "P4_EXECUTION_RESOLUTION"
    P5_QUEUE_ADMISSION = "P5_QUEUE_ADMISSION"
    P5_EXECUTION_RESOLUTION = "P5_EXECUTION_RESOLUTION"
    P7_QUEUE_ADMISSION = "P7_QUEUE_ADMISSION"
    P7_EXECUTION_RESOLUTION = "P7_EXECUTION_RESOLUTION"
    P8_QUEUE_ADMISSION = "P8_QUEUE_ADMISSION"
    P8_PARENT_MATERIALIZATION = "P8_PARENT_MATERIALIZATION"
    P8_DERIVED_SURFACE_REGISTRATION = "P8_DERIVED_SURFACE_REGISTRATION"


FROZEN_REASON_CODES = frozenset({
    "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED",
    "CONTROLLED_MISSION_EXTERNAL_ADMISSION_REQUIRED",
    "CANONICAL_LINEAGE_MISSING",
    "CANONICAL_LINEAGE_AMBIGUOUS",
    "CANONICAL_LINEAGE_HASH_CONFLICT",
    "CANONICAL_SOURCE_BINDING_MISMATCH",
    "CANONICAL_SURFACE_STATE_INVALID",
    "ALLOW_UNVALIDATED_PROHIBITED",
    "P1_FINALIZATION_LINEAGE_INCOMPLETE",
    "SURFACE_IMPORT_LINEAGE_INCOMPLETE",
    "P2_LINEAGE_INCOMPLETE",
    "PHYSICAL_QC_LINEAGE_INCOMPLETE",
    "P3_LINEAGE_INCOMPLETE",
    "P4_LINEAGE_INCOMPLETE",
    "P5_LINEAGE_INCOMPLETE",
    "P7_LINEAGE_INCOMPLETE",
    "P8_INPUT_LINEAGE_INCOMPLETE",
    "P8_OUTPUT_LINEAGE_INCOMPLETE",
    "CONTROL_INCOMPLETE_NO_RETRY_UNTIL_READBACK",
})

INCOMPLETE_REASON_BY_BOUNDARY = {
    CanonicalLineageBoundary.P1_FINALIZATION_INSERT.value:
        "P1_FINALIZATION_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.DIRECT_SURFACE_IMPORT.value:
        "SURFACE_IMPORT_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P2_QUEUE_ADMISSION.value: "P2_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P2_EXECUTION_RESOLUTION.value:
        "P2_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.PHYSICAL_QC_DIRECT_ENQUEUE.value:
        "PHYSICAL_QC_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.PHYSICAL_QC_CLAIM_RESOLUTION.value:
        "PHYSICAL_QC_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P3_QUEUE_ADMISSION.value: "P3_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P3_EXECUTION_RESOLUTION.value:
        "P3_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P4_QUEUE_ADMISSION.value: "P4_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P4_EXECUTION_RESOLUTION.value:
        "P4_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P5_QUEUE_ADMISSION.value: "P5_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P5_EXECUTION_RESOLUTION.value:
        "P5_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P7_QUEUE_ADMISSION.value: "P7_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P7_EXECUTION_RESOLUTION.value:
        "P7_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P8_QUEUE_ADMISSION.value:
        "P8_INPUT_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P8_PARENT_MATERIALIZATION.value:
        "P8_INPUT_LINEAGE_INCOMPLETE",
    CanonicalLineageBoundary.P8_DERIVED_SURFACE_REGISTRATION.value:
        "P8_OUTPUT_LINEAGE_INCOMPLETE",
}

_CANONICAL_STATES = frozenset({
    "IMPORTED", "QC_PENDING", "QC_SCREENED", "QC_REVIEW_PENDING",
    "ARCHIVED", "PHYSICAL_QC_COMPLETE", "GEOMETRY_CERTIFIED", "MERGED",
})


class CanonicalLineageRejected(ValueError):
    def __init__(self, decision: dict[str, Any]):
        super().__init__(str(decision["reason_code"]))
        self.decision = copy.deepcopy(decision)
        self.reason_code = str(decision["reason_code"])


def _boundary_value(boundary: str | CanonicalLineageBoundary) -> str:
    try:
        return CanonicalLineageBoundary(boundary).value
    except ValueError as error:
        raise ValueError("unknown canonical-lineage boundary") from error


def _is_discovery(lineage: dict[str, Any]) -> bool:
    namespace = str(lineage.get("namespace") or "")
    artifact_identity = str(lineage.get("artifact_identity") or "")
    uri = str(lineage.get("artifact_uri") or "")
    promotion_kind = lineage.get("promotion_lineage_kind")
    return (
        namespace == "NONCANONICAL_DISCOVERY"
        or artifact_identity.startswith(("probe_artifact_sets:", "discovery:"))
        or "/probes/" in uri or uri.startswith("probe://")
        or promotion_kind == "DISCOVERY_PARENT"
        or (
            lineage.get("promotion_lineage_sha256") is not None
            and promotion_kind != "FRESH_ORDINARY_CHILD"
        )
    )


def canonical_lineage_decision(
    *, boundary: str | CanonicalLineageBoundary, controlled_mission: bool,
    authoritative_lineage: dict[str, Any] | None, allow_unvalidated: Any,
) -> dict[str, Any]:
    """Return a frozen, evidence-bearing admission decision without mutation."""

    boundary_value = _boundary_value(boundary)
    lineage = (
        copy.deepcopy(authoritative_lineage)
        if isinstance(authoritative_lineage, dict) else {}
    )
    reason = "CANONICAL_LINEAGE_ACCEPTED"
    allowed = True
    if controlled_mission and allow_unvalidated is not False:
        allowed = False
        reason = "ALLOW_UNVALIDATED_PROHIBITED"
    elif _is_discovery(lineage):
        allowed = False
        reason = "NONCANONICAL_DISCOVERY_LINEAGE_PROHIBITED"
    elif not controlled_mission:
        allowed = True
        reason = "GENERIC_LINEAGE_UNCHANGED"
    elif not lineage or lineage.get("schema") != (
        "campaignx.authoritative_surface_lineage.v1"
    ):
        allowed = False
        reason = INCOMPLETE_REASON_BY_BOUNDARY[boundary_value]
    elif lineage.get("ambiguous") is True:
        allowed = False
        reason = "CANONICAL_LINEAGE_AMBIGUOUS"
    elif lineage.get("hash_conflict") is True:
        allowed = False
        reason = "CANONICAL_LINEAGE_HASH_CONFLICT"
    elif lineage.get("source_binding_mismatch") is True:
        allowed = False
        reason = "CANONICAL_SOURCE_BINDING_MISMATCH"
    elif lineage.get("external") is True and (
        lineage.get("canonical") is not True
        or not isinstance(lineage.get("external_admission_sha256"), str)
    ):
        allowed = False
        reason = "CONTROLLED_MISSION_EXTERNAL_ADMISSION_REQUIRED"
    elif lineage.get("canonical") is not True:
        allowed = False
        reason = "CANONICAL_LINEAGE_MISSING"
    elif lineage.get("surface_state") not in _CANONICAL_STATES:
        allowed = False
        reason = "CANONICAL_SURFACE_STATE_INVALID"
    decision_core = {
        "schema": "campaignx.canonical_lineage_decision.v1",
        "boundary": boundary_value,
        "allowed": allowed,
        "reason_code": reason,
        "mission_id": lineage.get("mission_id"),
        "surface_id": lineage.get("surface_id"),
        "artifact_identity": lineage.get("artifact_identity"),
        "artifact_sha256": lineage.get("artifact_sha256"),
        "source_snapshot_id": lineage.get("source_snapshot_id"),
        "source_binding_sha256": lineage.get("source_binding_sha256"),
        "promotion_lineage_sha256": lineage.get("promotion_lineage_sha256"),
        "route_sha256": lineage.get("route_sha256"),
        "controlled_mission": bool(controlled_mission),
        "allow_unvalidated": False if controlled_mission else allow_unvalidated,
    }
    return {
        **decision_core,
        "decision_sha256": content_sha256(decision_core),
    }


# The three keys a surface may not arrive carrying.
#
# `import_surface` persists the payload whole, and
# `resolve_canonical_surface_lineage` reads `authoritative_lineage` back out of
# that stored row and hands it to every downstream gate as authoritative. A
# caller that wrote one wrote its own provenance: the resolver's promise to go
# through a store-owned row rather than caller nesting was true about the row
# and false about how the row got there.
#
# `controlled_first_letters` is the same shape from the other side -- False
# takes the decision straight to GENERIC_LINEAGE_UNCHANGED -- and
# `allow_unvalidated` is the flag the controlled path exists to refuse.
#
# Nothing legitimate sets them here. The finalization path that does build a
# lineage server-side writes its surface with its own INSERT and never reaches
# this method.
CALLER_OWNED_LINEAGE_KEYS = ("authoritative_lineage", "controlled_first_letters",
                             "allow_unvalidated")


# The one namespace a payload may name for itself. When no lineage document is
# retained the resolver derives one, and derives `canonical` as "namespace is
# not the discovery one" -- so naming any other value is a claim of canonicity,
# while naming this one is a surface limiting itself. Narrowing is allowed;
# widening is the thing being refused.
SELF_LIMITING_NAMESPACE = "NONCANONICAL_DISCOVERY"


def refuse_asserted_lineage(payload: dict[str, Any]) -> None:
    """Refuse a surface that arrives describing its own place in the chain."""
    asserted = sorted(set(CALLER_OWNED_LINEAGE_KEYS).intersection(payload))
    if asserted:
        raise ValueError(
            "a surface being imported does not get to state its own lineage: "
            f"{asserted}. That document is resolved from what this store "
            "recorded, and a payload carrying one would be read back as the "
            "answer to whether the surface is canonical.")
    namespace = payload.get("namespace")
    if namespace is not None and namespace != SELF_LIMITING_NAMESPACE:
        raise ValueError(
            f"a surface being imported may name its namespace only as "
            f"{SELF_LIMITING_NAMESPACE!r}, which limits it. {namespace!r} is a "
            "claim that it is canonical, and the resolver would return it as "
            "one.")


def require_canonical_lineage(
    *, boundary: str | CanonicalLineageBoundary, controlled_mission: bool,
    authoritative_lineage: dict[str, Any] | None, allow_unvalidated: Any,
) -> dict[str, Any]:
    decision = canonical_lineage_decision(
        boundary=boundary, controlled_mission=controlled_mission,
        authoritative_lineage=authoritative_lineage,
        allow_unvalidated=allow_unvalidated,
    )
    if not decision["allowed"]:
        raise CanonicalLineageRejected(decision)
    return copy.deepcopy(authoritative_lineage or {})


def resolve_authoritative_surface_lineage(
    store: Any, *, surface_id: str, mission_id: str,
) -> dict[str, Any]:
    """Resolve lineage through a store-owned row resolver, never caller nesting."""

    resolver = getattr(store, "resolve_canonical_surface_lineage", None)
    if not callable(resolver):
        raise CanonicalLineageRejected(canonical_lineage_decision(
            boundary=CanonicalLineageBoundary.P2_EXECUTION_RESOLUTION,
            controlled_mission=True, authoritative_lineage=None,
            allow_unvalidated=False,
        ))
    lineage = resolver(surface_id=surface_id, mission_id=mission_id)
    if (not isinstance(lineage, dict)
            or lineage.get("surface_id") != surface_id
            or lineage.get("mission_id") != mission_id):
        raise CanonicalLineageRejected(canonical_lineage_decision(
            boundary=CanonicalLineageBoundary.P2_EXECUTION_RESOLUTION,
            controlled_mission=True, authoritative_lineage=None,
            allow_unvalidated=False,
        ))
    return copy.deepcopy(lineage)
