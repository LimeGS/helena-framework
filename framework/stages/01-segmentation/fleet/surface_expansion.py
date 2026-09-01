"""The only way out of the diagnostic path, written down so it can be checked.

`small-surface-routing-1.0.0.json` says it in four words -- promotion in place is
PROHIBITED -- and adds that the original surface REMAINS_DIAGNOSTIC_PERMANENTLY.
A tiny surface leaves only through a new versioned grow or resume whose *new*
surface independently clears the floor and passes every standard gate.

That sentence needs something durable behind it, because otherwise the only
record that a surface is a second attempt at a diagnostic one is a string in a
payload nobody validates. This module is that record: which diagnostic surface
is being continued, under which routing decision, from which policy version to
which, digest-bound so a later edit is visible.

It is deliberately pure. The stores resolve it against a locked catalogue inside
the transaction that creates the successor and persist it there; nothing here
reads or writes a database, so the same bytes are produced by SQLite and by
PostgreSQL.

Two things it does not do. It does not decide whether the successor is big
enough -- that is the router's question, asked of the successor's own
measurement, which is what "independently" means. And it never touches the
predecessor: an authority is permission to make a new surface, not permission to
edit an old one.
"""

from __future__ import annotations

from typing import Any

from .common import content_sha256

SCHEMA = "campaignx.small_surface_expansion_authority.v1"

IN_PLACE = "PROHIBITED"
ORIGINAL_SURFACE = "REMAINS_DIAGNOSTIC_PERMANENTLY"

_DIGEST_FIELDS = (
    "schema", "expands_surface_id", "successor_surface_id",
    "predecessor_route", "predecessor_receipt_sha256",
    "prior_policy_version", "new_policy_version", "resume_from",
    "in_place", "original_surface",
)


def resume_shape(source: Any) -> dict[str, Any] | None:
    """What a payload says about the surface it continues, or nothing.

    `None` means this is not an expansion, which is the ordinary case and must
    stay cheap. A malformed shape raises rather than being read as `None`: a
    `resumes_surface` of `""` or `7` is a caller that meant to continue
    something, and treating it as an ordinary surface would silently drop the
    binding the whole contract rests on.
    """
    if not isinstance(source, dict) or "resumes_surface" not in source:
        return None
    expands = source.get("resumes_surface")
    if not isinstance(expands, str) or not expands.strip():
        raise ValueError(
            f"resumes_surface is not a surface id: {expands!r}")
    resume_from = source.get("resume_from")
    if resume_from is not None and (
        not isinstance(resume_from, str) or not resume_from.strip()
    ):
        raise ValueError(f"resume_from is not a location: {resume_from!r}")
    new_policy_version = source.get("policy_version")
    if new_policy_version is not None and (
        not isinstance(new_policy_version, str)
        or not new_policy_version.strip()
    ):
        raise ValueError(
            f"a resume policy version is not a version: {new_policy_version!r}")
    return {
        "expands_surface_id": expands.strip(),
        "new_policy_version": new_policy_version,
        "resume_from": resume_from,
    }


def build_authority(*, expands_surface_id: str, successor_surface_id: str,
                    predecessor_route: str, predecessor_receipt_sha256: str,
                    prior_policy_version: str | None,
                    new_policy_version: str | None,
                    resume_from: str | None) -> dict[str, Any]:
    """One expansion's permission, and the evidence it was permitted."""
    for name, value in (("expands_surface_id", expands_surface_id),
                        ("successor_surface_id", successor_surface_id),
                        ("predecessor_route", predecessor_route)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"an expansion authority needs {name}")
    if expands_surface_id == successor_surface_id:
        # The whole point. Promoting a surface in place would be claiming the
        # measurement was wrong, and the measurement is the evidence.
        raise ValueError(
            "a surface cannot expand itself: promotion in place is prohibited")
    if (not isinstance(predecessor_receipt_sha256, str)
            or len(predecessor_receipt_sha256) != 64):
        raise ValueError(
            "an expansion authority needs the predecessor's routing receipt")
    # A resume rides a policy version of its own, or it is the same task as the
    # grow it is correcting and the catalogue cannot tell them apart. Where
    # neither side has a version -- a direct catalogue import has no task, so no
    # task identity -- there is nothing to compare and the successor being a
    # separate surface with its own measurement is what carries the contract.
    if (prior_policy_version is not None
            and new_policy_version is not None
            and prior_policy_version == new_policy_version):
        raise ValueError(
            "an expansion must be a new versioned attempt, not a repeat of "
            f"policy version {new_policy_version!r}")
    authority = {
        "schema": SCHEMA,
        "expands_surface_id": expands_surface_id,
        "successor_surface_id": successor_surface_id,
        "predecessor_route": predecessor_route,
        "predecessor_receipt_sha256": predecessor_receipt_sha256,
        "prior_policy_version": prior_policy_version,
        "new_policy_version": new_policy_version,
        "resume_from": resume_from,
        "in_place": IN_PLACE,
        "original_surface": ORIGINAL_SURFACE,
    }
    authority["authority_sha256"] = _digest(authority)
    return authority


def _digest(authority: dict[str, Any]) -> str:
    return content_sha256(
        {field: authority.get(field) for field in _DIGEST_FIELDS})


def verify_authority(authority: Any) -> bool:
    """Whether the authority still says what it was signed saying."""
    if not isinstance(authority, dict):
        return False
    stored = authority.get("authority_sha256")
    if not isinstance(stored, str) or len(stored) != 64:
        return False
    return stored == _digest(authority)


def agrees_with_stamp(resolved: Any, stamped: Any) -> bool:
    """Whether a re-resolution matches what was stamped when work was queued.

    Every field but the successor's identity, because a resume task is stamped
    before its surface exists and the surface id is the one thing that could not
    have been known then. Everything the catalogue could have changed underneath
    -- which surface is continued, its route, its receipt digest, the versions --
    has to be identical, and both documents have to verify.
    """
    if not (verify_authority(resolved) and verify_authority(stamped)):
        return False
    return all(resolved.get(field) == stamped.get(field)
               for field in _DIGEST_FIELDS if field != "successor_surface_id")


def leaves_diagnostic(authority: Any, successor_receipt: Any) -> bool:
    """Whether this expansion actually got a surface off the diagnostic path.

    Both halves, or neither. The predecessor has to have been diagnostic -- an
    expansion of an already-standard surface is a correction, not an escape --
    and the successor has to clear the floor on its own measurement, which is
    the router's answer about the successor and nothing else.
    """
    from . import surface_routing  # noqa: PLC0415

    return (verify_authority(authority)
            and authority.get("predecessor_route") == surface_routing.DIAGNOSTIC
            and surface_routing.enters_standard_qc(successor_receipt)
            and successor_receipt.get("surface_id")
                == authority.get("successor_surface_id"))
