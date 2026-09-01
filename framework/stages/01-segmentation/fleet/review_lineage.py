"""Which surface a job is about, decided by walking rows rather than by asking.

A client may name a job. It may not say what that job is downstream of. That
distinction is the whole of this module, and it exists because on 2026-08-02 a
0.0198 cm2 surface of PHerc0268 -- two square millimetres -- was
GEOMETRY_CERTIFIED and reached the ink screen. Every gate it walked past had
the same shape: something outside the server supplied an identity, and the
server checked the identity against itself.

So the resolution starts at a job the caller named and walks P7 -> P5 -> P4 ->
P3 through persisted rows only. The surface falls out of the walk at P3, where
it was actually decided. A caller may still *assert* a surface; the assertion
is compared to the walk and rejected when it disagrees, never substituted for
it.

Everything fails closed. A missing hop, a hop that did not succeed, a hop in
another mission or sample, a P3 that cannot name its surface, two flattenings
where there must be one, a P4 that flattened a different surface than its own
P3 produced -- each is a refusal with a reason code, not a best guess. Silently
preferring one branch of a forked chain is how a decoy becomes provenance.

The review event carries a lock over the resolved chain, the frozen intent, the
adjudication hashes and the routing decision. The store re-derives that lock and
re-reads the routing receipt from its own immutable table before it will write
a row, so a hand-built event that never went through the resolver has nowhere
to enter.
"""

from __future__ import annotations

import re
from typing import Any

from . import surface_routing
from .common import content_sha256

# One intent, server-owned. A person's opinion of a surface has its own,
# separate vocabulary; this enum is only about where a reviewed surface is
# routed, and a route is not an opinion.
REVIEW_INTENTS = ("INSPECT",)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CHAIN_FIELDS = (
    "mission_id", "sample_id", "surface_id",
    "p3_job_id", "p4_job_id", "p5_job_id", "p7_job_id",
    "flattened_artifact_id", "flattened_artifact_sha256",
    "p4_layer_artifact_sha256",
)

# What the lock covers: the whole resolved chain, the route it was allowed to
# take, and the adjudication it was allowed to claim. Anything a later caller
# could swap to point the same review at a different surface belongs here.
LOCKED_FIELDS = (
    "intent", "mission_id", "sample_id", "surface_id",
    "p3_job_id", "p4_job_id", "p5_job_id", "p7_job_id",
    "flattened_artifact_id", "flattened_artifact_sha256",
    "p4_layer_artifact_sha256", "chain_sha256",
    "route", "routing_receipt_sha256",
    "verdict_sha256", "card_sha256", "config_sha256",
    "vetting_packet_sha256",
)


class SurfaceOriginRejected(ValueError):
    """A refusal that says which rule refused, so a caller cannot retry blind."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{detail} [{reason_code}]")
        self.reason_code = reason_code
        self.detail = detail


def _reject(reason_code: str, detail: str) -> None:
    raise SurfaceOriginRejected(reason_code, detail)


def _text(value: Any) -> str | None:
    """A usable identifier, or nothing. Whitespace is not an identifier."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_job(jobs: Any, *, job_id: Any, phase: str,
                 mission_id: str, sample_id: str) -> dict[str, Any]:
    identifier = _text(job_id)
    if identifier is None:
        _reject("LINEAGE_EDGE_MISSING", f"{phase} lineage names no job")
    row = jobs.job(identifier)
    if not isinstance(row, dict):
        _reject("JOB_NOT_FOUND", f"no {phase} job {identifier}")
    if row.get("phase") != phase:
        _reject("JOB_PHASE_MISMATCH",
                f"job {identifier} is {row.get('phase')}, not {phase}")
    if row.get("state") != "succeeded":
        _reject("JOB_NOT_SUCCEEDED",
                f"{phase} job {identifier} is {row.get('state')}")
    if (row.get("mission_id") != mission_id
            or row.get("sample_id") != sample_id):
        _reject("JOB_SCOPE_MISMATCH",
                f"{phase} job {identifier} belongs to another mission or sample")
    return row


def _p3_origin(jobs: Any, *, p3_job_id: Any,
               mission_id: str, sample_id: str) -> dict[str, Any]:
    """The surface is whatever P3 was asked to flatten, and only that."""
    job = _require_job(jobs, job_id=p3_job_id, phase="P3",
                       mission_id=mission_id, sample_id=sample_id)
    identifier = _text(p3_job_id)
    surface_id = _text((job.get("parameters") or {}).get("surface_id"))
    if surface_id is None:
        # Deliberately no fallback to a lone row in the result. A result row is
        # what the job produced; the parameter is what it was asked for, and
        # only the second one is an origin.
        _reject("P3_SURFACE_UNRESOLVABLE",
                f"P3 job {identifier} does not name the surface it flattened")
    rows = [row for row in (job.get("result") or {}).get("surfaces") or []
            if isinstance(row, dict)
            and row.get("surface_id") == surface_id
            and row.get("requested_by_job_id") == identifier]
    if len(rows) != 1:
        _reject("P3_FLATTENED_LINEAGE_AMBIGUOUS",
                f"P3 job {identifier} has {len(rows)} flattenings of "
                f"{surface_id}, not one")
    flattened = rows[0]
    return {
        "mission_id": mission_id, "sample_id": sample_id,
        "surface_id": surface_id, "p3_job_id": identifier,
        "p4_job_id": None, "p5_job_id": None, "p7_job_id": None,
        "flattened_artifact_id": _text(flattened.get("artifact_id")
                                       or flattened.get("flattening_id")),
        "flattened_artifact_sha256": flattened.get("artifact_sha256"),
        "p4_layer_artifact_sha256": None,
    }


def _p4_origin(jobs: Any, *, p4_job_id: Any,
               mission_id: str, sample_id: str) -> dict[str, Any]:
    job = _require_job(jobs, job_id=p4_job_id, phase="P4",
                       mission_id=mission_id, sample_id=sample_id)
    parameters = job.get("parameters") or {}
    chain = _p3_origin(jobs, p3_job_id=parameters.get("p3_job_id"),
                       mission_id=mission_id, sample_id=sample_id)
    flattened_surface = _text(parameters.get("flattened_surface"))
    if flattened_surface != chain["surface_id"]:
        _reject("LINEAGE_SURFACE_CONFLICT",
                f"P4 flattened {flattened_surface} but its P3 produced "
                f"{chain['surface_id']}")
    for field, expected in (
        ("flattening_id", chain["flattened_artifact_id"]),
        ("flattened_artifact_sha256", chain["flattened_artifact_sha256"]),
    ):
        if parameters.get(field) is not None and parameters[field] != expected:
            _reject("LINEAGE_SURFACE_CONFLICT",
                    f"P4 {field} disagrees with its own P3 flattening")
    layer_sha = ((job.get("result") or {}).get("layer_stack")
                 or {}).get("artifact_sha256")
    return {**chain, "p4_job_id": _text(p4_job_id),
            "p4_layer_artifact_sha256": layer_sha}


def _p5_origin(jobs: Any, *, p5_job_id: Any,
               mission_id: str, sample_id: str) -> dict[str, Any]:
    job = _require_job(jobs, job_id=p5_job_id, phase="P5",
                       mission_id=mission_id, sample_id=sample_id)
    parameters = job.get("parameters") or {}
    normalization = (job.get("result") or {}).get("physical_normalization") or {}
    consumed = _text(parameters.get("layer_stack"))
    if consumed is None:
        _reject("LINEAGE_EDGE_MISSING", "P5 names no layer stack")
    declared = normalization.get("p4_job_id")
    if declared is not None and declared != consumed:
        # Two answers to one question. Preferring either is the failure.
        _reject("LINEAGE_EDGE_AMBIGUOUS",
                "P5 normalization and parameters name different P4 jobs")
    chain = _p4_origin(jobs, p4_job_id=consumed,
                       mission_id=mission_id, sample_id=sample_id)
    layer_sha = normalization.get("p4_layer_artifact_sha256")
    if layer_sha is not None and layer_sha != chain["p4_layer_artifact_sha256"]:
        _reject("LINEAGE_EDGE_AMBIGUOUS",
                "P5 normalization does not match the layer stack it consumed")
    return {**chain, "p5_job_id": _text(p5_job_id)}


def _p7_origin(jobs: Any, *, p7_job_id: Any,
               mission_id: str, sample_id: str) -> dict[str, Any]:
    job = _require_job(jobs, job_id=p7_job_id, phase="P7",
                       mission_id=mission_id, sample_id=sample_id)
    parameters = job.get("parameters") or {}
    chain = _p5_origin(jobs, p5_job_id=parameters.get("screening_of"),
                       mission_id=mission_id, sample_id=sample_id)
    screened = _text(parameters.get("surface_id"))
    if screened is None:
        _reject("LINEAGE_EDGE_MISSING", "P7 names no surface")
    if screened != chain["surface_id"]:
        _reject("LINEAGE_SURFACE_CONFLICT",
                f"P7 screened {screened} but its chain produced "
                f"{chain['surface_id']}")
    return {**chain, "p7_job_id": _text(p7_job_id)}


_WALKERS = {"P3": _p3_origin, "P4": _p4_origin,
            "P5": _p5_origin, "P7": _p7_origin}
_ARGUMENT = {"P3": "p3_job_id", "P4": "p4_job_id",
             "P5": "p5_job_id", "P7": "p7_job_id"}


def resolve_surface_origin(
    jobs: Any, *, phase: str, job_id: Any, mission_id: str, sample_id: str,
    asserted_surface_id: Any = None,
) -> dict[str, Any]:
    """Walk persisted rows down to the surface, and hash what was walked.

    `asserted_surface_id` is the only place a caller's opinion appears, and it
    is compared, never consulted: naming the right surface is allowed, naming
    the surface is not.
    """
    if phase not in _WALKERS:
        raise ValueError(f"not a lineage phase: {phase!r}")
    if not _text(mission_id) or not _text(sample_id):
        _reject("JOB_SCOPE_MISMATCH", "lineage resolution needs mission and sample")
    chain = _WALKERS[phase](jobs, **{_ARGUMENT[phase]: job_id},
                            mission_id=mission_id, sample_id=sample_id)
    asserted = _text(asserted_surface_id)
    if asserted is not None and asserted != chain["surface_id"]:
        _reject("CLIENT_SURFACE_ASSERTION_REJECTED",
                f"caller named {asserted}; the persisted chain resolves to "
                f"{chain['surface_id']}")
    ordered = {field: chain[field] for field in _CHAIN_FIELDS}
    return {**ordered, "chain_sha256": content_sha256(ordered)}


def require_review_intent(intent: Any) -> str:
    """The routing intent, from a frozen list the client cannot extend."""
    if intent not in REVIEW_INTENTS:
        _reject("REVIEW_INTENT_NOT_ALLOWED",
                f"review intent {intent!r} is not one of {list(REVIEW_INTENTS)}")
    return str(intent)


def require_passing_adjudication(adjudication: Any) -> dict[str, Any]:
    """Only a P7 that passed, and said so in hashes, may be routed onward."""
    row = adjudication if isinstance(adjudication, dict) else {}
    if (row.get("verdict") != "PASS"
            or (row.get("overall") or {}).get("pass") is not True):
        _reject("ADJUDICATION_NOT_PASSING",
                "only a P7 PASS may be routed for inspection")
    for key in ("verdict_sha256", "card_sha256", "config_hash"):
        if not isinstance(row.get(key), str) or not _SHA256.match(row[key]):
            _reject("ADJUDICATION_NOT_PASSING",
                    f"P7 adjudication lacks {key}")
    return dict(row)


def require_standard_route(receipt: Any) -> dict[str, Any]:
    """Exactly the standard route, proved by a receipt that still verifies.

    A missing receipt is a refusal too. "We never classified it" is not
    evidence that it belongs downstream, and the surface this whole module
    exists for was one nobody had classified.
    """
    if not isinstance(receipt, dict) or not surface_routing.enters_canonical_downstream(
            receipt):
        route = receipt.get("route") if isinstance(receipt, dict) else None
        _reject("SURFACE_ROUTE_NOT_STANDARD",
                f"surface route is {route!r}, not {surface_routing.STANDARD}")
    return dict(receipt)


def review_lineage_sha256(event: dict[str, Any]) -> str:
    return content_sha256({field: event.get(field) for field in LOCKED_FIELDS})


def build_review_event(
    *, origin: dict[str, Any], intent: str, note: str | None,
    routing_receipt: dict[str, Any], adjudication: dict[str, Any],
    vetting_packet_sha256: str, author: str, at: str, review_event_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the one event shape the store will accept, and lock it.

    `event_sha256` is deliberately left to the caller: the lock covers what the
    server resolved, the event hash covers everything the caller then added.
    """
    receipt = routing_receipt if isinstance(routing_receipt, dict) else {}
    event = {
        "schema": "campaignx.human_review_event.v1",
        "review_event_id": review_event_id,
        "intent": intent,
        "note": note,
        **{field: origin.get(field) for field in _CHAIN_FIELDS},
        "chain_sha256": origin.get("chain_sha256"),
        "route": receipt.get("route"),
        "routing_receipt_sha256": receipt.get("receipt_sha256"),
        "verdict_sha256": adjudication.get("verdict_sha256"),
        "card_sha256": adjudication.get("card_sha256"),
        "config_sha256": adjudication.get("config_hash"),
        "vetting_packet_sha256": vetting_packet_sha256,
        **dict(extra or {}),
        "by": author,
        "at": at,
    }
    event["review_lineage_sha256"] = review_lineage_sha256(event)
    return event


def verify_review_lineage_lock(event: Any) -> dict[str, Any]:
    """Re-derive the lock the resolver wrote, and refuse anything else.

    This is what makes a direct `insert_human_review` fail closed. The lock
    alone is not a secret -- the store pairs it with a routing receipt read from
    its own immutable table, which a caller cannot supply.
    """
    row = event if isinstance(event, dict) else {}
    stored = row.get("review_lineage_sha256")
    if not isinstance(stored, str) or not _SHA256.match(stored):
        _reject("REVIEW_LINEAGE_LOCK_MISSING",
                "review lineage lock is missing; this event did not come from "
                "the server resolver")
    if stored != review_lineage_sha256(row):
        _reject("REVIEW_LINEAGE_LOCK_INVALID",
                "review lineage lock does not cover this event")
    return dict(row)


def require_reviewable_event(event: Any, routing_receipt: Any) -> dict[str, Any]:
    """The three refusals every store applies before a review becomes a row."""
    row = event if isinstance(event, dict) else {}
    require_review_intent(row.get("intent"))
    verify_review_lineage_lock(row)
    receipt = require_standard_route(routing_receipt)
    if receipt.get("surface_id") != row.get("surface_id") \
            or receipt.get("receipt_sha256") != row.get("routing_receipt_sha256"):
        _reject("SURFACE_ROUTE_NOT_STANDARD",
                "the stored route is not the route this review was locked to")
    return row
