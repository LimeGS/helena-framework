#!/usr/bin/env python3
"""Run the frozen First Letters control one prerequisite at a time.

The module is deliberately import-safe.  Importing it performs no argument
parsing, authentication, HTTP request, or filesystem write.  Actual execution
requires an explicit CLI invocation or a call to :func:`run_positive_control`.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, urlencode

from panel_client import AmbiguousMutationError, Panel, PanelError

SCHEMA = "campaignx.first_letters_stage_survival.v1"
BOUNDARIES = ("P0", "P1", "P2", "QC", "P3", "P4", "P5", "P7", "HUMAN_REVIEW")

# The second control: the same evaluation rule over the post's own recommended
# tooling, on inputs anybody can obtain. It exists because the nine-boundary
# control cannot answer what was asked of it -- it reads surfaces from a
# private bucket, so "public input surfaces" and "a run from a clean
# installation" are out of its reach however well it is repaired.
#
# Two schemas rather than one with a variable boundary list. That list being
# fixed per schema is a safety property: a receipt cannot declare its own shape
# and pass trivially, and a six-row receipt cannot be published as a
# nine-boundary control.
PUBLIC_SCHEMA = "campaignx.public_ink_stage_survival.v1"
PUBLIC_BOUNDARIES = (
    "PUBLIC_SOURCE", "SCALE", "CHECKPOINT", "INK", "LIVENESS", "HUMAN_REVIEW",
)
# The segmentation half, P0 to P3: intake, grow, certify geometry, measure
# physical support, flatten. Its own schema for the same reason the ink control
# has one -- a receipt cannot declare its own shape -- and because the two are
# different kinds of claim. The ink chain is deterministic modulo the GPU; a
# grow is not: three runs of one control, same seed, same profile, gave three
# surfaces. So this control's rows pass on what the chain *produced* -- at least
# one surface certified and CT-supported within a bounded number of tasks --
# and record which, rather than on bytes matching.
SEGMENTATION_SCHEMA = "campaignx.public_segmentation_stage_survival.v1"
SEGMENTATION_BOUNDARIES = (
    "PUBLIC_SOURCE", "INTAKE", "GROW", "GEOMETRY", "PHYSICAL_QC", "FLATTEN",
)
CONTROL_SHAPES: dict[str, tuple[str, ...]] = {
    SCHEMA: BOUNDARIES,
    PUBLIC_SCHEMA: PUBLIC_BOUNDARIES,
    SEGMENTATION_SCHEMA: SEGMENTATION_BOUNDARIES,
}
PASS = "PASS"
FAILED = "FAILED"
INCOMPLETE = "INCOMPLETE"
NOT_RUN = "NOT_RUN_PREREQUISITE"
SHA256_LENGTH = 64
MINIMUM_STANDARD_SURFACE_AREA_CM2 = 0.10

NON_CLAIM = (
    "This row tests the frozen method at one boundary. It does not establish "
    "independent validation or automatically accept ink, text, letters, or a reading."
)


PROGRESS_SCHEMA = "campaignx.first_letters_control_progress_event.v1"
# Deliberately short, and deliberately not the harness's own `Panel.timeout`
# (3600s, sized for the real work a boundary does). A heartbeat fires from
# inside that boundary's own wait loop, before its next deadline check -- so
# without a short timeout of its own, a merely slow panel response blocks the
# loop long enough to consume the boundary's entire wait budget by itself,
# turning a heartbeat meant to prove the run is alive into the reason it times
# out.
PROGRESS_POST_TIMEOUT_SECONDS = 10.0

_BOUNDARY_ALTERNATION = "|".join(re.escape(name) for name in BOUNDARIES)
_ANNOUNCE_BOUNDARY_STARTED = re.compile(rf"^({_BOUNDARY_ALTERNATION}) starting$")
_ANNOUNCE_BOUNDARY_FINISHED = re.compile(
    rf"^({_BOUNDARY_ALTERNATION}) -> (\S+) \((.*)\) after")
_ANNOUNCE_HEARTBEAT = re.compile(rf"^({_BOUNDARY_ALTERNATION})\b.*\bwait")
_ANNOUNCE_RUN_FINISHED = re.compile(
    r"^control run finished: (\S+) \(first non-passing boundary: (.*)\)$")

# Set for the span of one control run so _announce can reach the panel without
# every one of its ~70 call sites carrying a reporter through by hand -- the
# same reasoning `clock` is threaded explicitly for, applied to something that
# cannot be threaded that way without rewriting all of them.
_REPORTER: contextvars.ContextVar["_ProgressReporter | None"] = \
    contextvars.ContextVar("_control_progress_reporter", default=None)


class _ProgressReporter:
    """Posts what `_announce` already prints, through the run's own session.

    Best-effort by construction: a control can be hours from finishing, and a
    heartbeat that cannot reach the panel must cost this run nothing more than
    the one line failing to arrive.
    """

    def __init__(self, panel, mission_id: str, run_id: str):
        self.panel = panel
        self.mission_id = mission_id
        self.run_id = run_id

    def post(self, message: str, **fields) -> None:
        event = {
            "schema": PROGRESS_SCHEMA,
            "run_id": self.run_id,
            "mission_id": self.mission_id,
            "message": message,
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields,
        }
        try:
            self.panel.call(
                "POST",
                f"/api/missions/{quote(self.mission_id, safe='')}"
                "/first-letters-control/progress",
                {"event": event}, timeout=PROGRESS_POST_TIMEOUT_SECONDS)
        except Exception as failure:  # noqa: BLE001 - deliberate; see the docstring
            print(f"    (progress not recorded: {type(failure).__name__}: {failure})",
                 flush=True)


@contextlib.contextmanager
def _reporting_progress(panel, mission_id: str, run_id: str):
    token = _REPORTER.set(_ProgressReporter(panel, mission_id, run_id))
    try:
        yield
    finally:
        _REPORTER.reset(token)


def _classify_announcement(message: str) -> dict[str, Any]:
    """What each shape `_announce` writes actually means, for the call sites
    that describe their own event only through the printed sentence.

    `_set_result` is not one of them: its `reason` can be a `reason_code` the
    panel sent in a refusal's HTTP body (`_refusal_reason`), not text this
    module authored, so it hands `event`/`boundary`/`state`/`reason` to
    `_announce` directly instead of asking this function to recover them by
    re-matching a sentence built from that same untrusted text. Every other
    call site's message is this module's own prose, safe to classify by
    shape; one that still fails to match falls through to the generic "note"
    event below rather than misparsing.
    """
    if match := _ANNOUNCE_BOUNDARY_FINISHED.match(message):
        boundary, state, reason = match.groups()
        return {"event": "boundary_finished", "boundary": boundary,
                "state": state, "reason": reason}
    if match := _ANNOUNCE_BOUNDARY_STARTED.match(message):
        return {"event": "boundary_started", "boundary": match.group(1)}
    if match := _ANNOUNCE_HEARTBEAT.match(message):
        return {"event": "heartbeat", "boundary": match.group(1)}
    if message.startswith("control run starting"):
        return {"event": "run_started"}
    if match := _ANNOUNCE_RUN_FINISHED.match(message):
        state, boundary = match.groups()
        return {"event": "run_finished", "control_state": state,
                "first_nonpassing_boundary": None if boundary == "None" else boundary}
    return {"event": "note"}


def _announce(message: str, **fields: Any) -> None:
    """One line of progress, printed as it happens and, once a run has
    attached a reporter, posted to the panel so it outlives the terminal.

    A boundary here can run over an hour -- P1's discovery-and-grow measured at
    66 minutes, P2's finalization at 60, in the run that went looking for this.
    Nothing was printed or recorded anywhere until the run returned, so there
    was no way to tell "slow" from "hung" for as long as either boundary ran;
    the only evidence was whether the process was still alive.

    `fields`, when given, are posted as-is instead of being re-derived from
    `message` by `_classify_announcement` -- see that function's docstring
    for why `_set_result` must supply them rather than let the message alone
    carry a refusal's own, potentially adversarial, reason text.

    Nothing in here may propagate. `_set_result` calls this as its own last,
    unguarded statement -- a print failure or a broken heartbeat used to reach
    `run_positive_control`'s outer refusal handler, which blames the *first
    undecided row*. On a boundary whose own result had already been written
    correctly, that is the *next* boundary, never attempted: a narration
    failure on P4's line was misattributed to P5. The whole point of this
    function is that its side effects cost at most themselves.
    """
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    try:
        print(f"[{stamp}] {message}", flush=True)
    except Exception:  # noqa: BLE001 - deliberate; see the docstring
        pass
    reporter = _REPORTER.get()
    if reporter is None:
        return
    try:
        reporter.post(message, **(fields or _classify_announcement(message)))
    except Exception as failure:  # noqa: BLE001 - deliberate; see the docstring
        try:
            print(f"    (progress not recorded: {type(failure).__name__}: {failure})",
                 flush=True)
        except Exception:  # noqa: BLE001
            pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _profiles(manifest: Mapping[str, Any], *needles: str) -> list[str]:
    rows = manifest.get("profile_locks") or []
    selected = [
        str(row.get("profile_id"))
        for row in rows
        if any(needle in str(row.get("profile_id", "")).lower() for needle in needles)
    ]
    return selected


def _row(boundary: str, *, profiles: list[str] | None = None,
         parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "boundary": boundary,
        "input_artifacts": [],
        "profile_ids": list(profiles or []),
        "parameters": dict(parameters or {}),
        "counts": {},
        "terminal_state": NOT_RUN,
        "reason_code": "PREREQUISITE_NOT_REACHED",
        "elapsed_seconds": 0.0,
        "resource_identity": {},
        "output_hashes": {},
        "non_claim": NON_CLAIM,
    }


def _set_result(row: dict[str, Any], *, state: str, reason: str,
                started: float, clock, observation: Mapping[str, Any] | None = None,
                input_artifacts: list[dict[str, Any]] | None = None,
                output_hashes: Mapping[str, Any] | None = None,
                counts: Mapping[str, Any] | None = None) -> None:
    observation = observation or {}
    row.update({
        "terminal_state": state,
        "reason_code": reason,
        "elapsed_seconds": max(0.0, float(clock()) - started),
        "resource_identity": dict(observation.get("resource_identity") or {}),
        "input_artifacts": list(input_artifacts or []),
        "output_hashes": dict(output_hashes or observation.get("output_hashes") or {}),
        "counts": dict(counts or observation.get("counts") or {}),
    })
    # Only when there is one, so designed boundaries keep the shape they had.
    # For a boundary that ended the way somebody planned, the reason code
    # carries the meaning; for one that was refused unexpectedly the detail is
    # the whole information, and dropping it leaves a receipt that records a
    # death without its cause -- which is what sent the last diagnosis back to
    # the panel's logs to learn what the receipt had been written for.
    if observation.get("detail"):
        row["detail"] = str(observation["detail"])[:600]
    _announce(f"{row['boundary']} -> {state} ({reason}) "
             f"after {row['elapsed_seconds']:.0f}s",
             event="boundary_finished", boundary=row["boundary"],
             state=state, reason=reason)


def _read_set_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("schema") != "campaignx.first_letters_source_read_set.v1":
        return False
    if not _is_sha256(value.get("canonical_manifest_sha256")):
        return False
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects:
        return False
    keys: list[str] = []
    for item in objects:
        if not isinstance(item, Mapping):
            return False
        key = item.get("object_key")
        byte_count = item.get("bytes")
        if not isinstance(key, str) or not key or not _is_sha256(item.get("sha256")):
            return False
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            return False
        keys.append(key)
    return (keys == sorted(keys) and len(keys) == len(set(keys))
            and value.get("canonical_manifest_sha256") == canonical_sha256(objects))


def _read_set_contains_locked_metadata(
    value: object, required: object,
) -> bool:
    if not _read_set_complete(value) or not isinstance(required, list):
        return False
    observed = {
        str(row.get("object_key")): str(row.get("sha256"))
        for row in value["objects"] if isinstance(row, Mapping)
    }
    return all(
        isinstance(row, Mapping)
        and observed.get(str(row.get("path"))) == row.get("sha256")
        for row in required
    )


def _traversal_evidence_complete(value: Mapping[str, Any]) -> bool:
    planned = value.get("planned_region_count")
    visited = value.get("visited_region_count")
    fraction = value.get("coverage_fraction")
    parameters = value.get("parameters")
    cap = parameters.get("maximum_surface_probe_regions") if isinstance(parameters, Mapping) else None
    return (
        isinstance(planned, int) and not isinstance(planned, bool) and planned > 0
        and isinstance(visited, int) and not isinstance(visited, bool) and 0 < visited <= planned
        and isinstance(cap, int) and not isinstance(cap, bool) and cap > 0
        and visited <= cap
        and isinstance(fraction, (int, float)) and not isinstance(fraction, bool)
        and math.isfinite(float(fraction))
        and abs(float(fraction) - (visited / planned)) <= 1e-12
        and value.get("coverage_complete") is True
    )


def _artifact_inventory_complete(value: object) -> bool:
    if not isinstance(value, Mapping) or not _is_sha256(value.get("artifact_sha256")):
        return False
    objects = value.get("objects")
    if not isinstance(objects, list) or not objects or len(objects) != value.get("files"):
        return False
    files: dict[str, dict[str, Any]] = {}
    for item in objects:
        if not isinstance(item, Mapping):
            return False
        key, digest, byte_count = item.get("object_key"), item.get("sha256"), item.get("bytes")
        if not isinstance(key, str) or not key or key in files or not _is_sha256(digest):
            return False
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            return False
        files[key] = {"sha256": digest, "size_bytes": byte_count}
    return value["artifact_sha256"] == canonical_sha256(files)


def _exact_numeric_tiff_inventory(value: object, expected_slices: int) -> bool:
    """Require actual TIFF names to denote each index exactly once."""
    if not isinstance(value, Mapping) or value.get("files") != expected_slices:
        return False
    objects = value.get("objects")
    if not isinstance(objects, list) or len(objects) != expected_slices:
        return False
    indices: list[int] = []
    for row in objects:
        key = row.get("object_key") if isinstance(row, Mapping) else None
        if not isinstance(key, str) or "/" in key or not key.endswith(".tif"):
            return False
        stem = key[:-4]
        if not stem or any(character not in "0123456789" for character in stem):
            return False
        indices.append(int(stem))
    return (
        sorted(indices) == list(range(expected_slices))
        and len(set(indices)) == expected_slices
        and value.get("slice_indices") == list(range(expected_slices))
        and value.get("slice_ordering") == "NUMERIC_STEM_CONTIGUOUS_ASCENDING"
    )


def _provider_exchange_complete(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_sha256(value.get("request_sha256"))
        and _is_sha256(value.get("response_sha256"))
        and isinstance(value.get("request_bytes"), int)
        and not isinstance(value.get("request_bytes"), bool)
        and int(value["request_bytes"]) >= 0
        and isinstance(value.get("response_bytes"), int)
        and not isinstance(value.get("response_bytes"), bool)
        and int(value["response_bytes"]) >= 0
    )


def _artifact(observation: Mapping[str, Any], *, default_id: str | None = None) -> dict[str, Any]:
    identifier = observation.get("artifact_id") or default_id
    digest = observation.get("artifact_sha256")
    return {
        **({"artifact_id": str(identifier)} if identifier else {}),
        **({"sha256": str(digest)} if digest else {}),
    }


def _job_result(panel, response: Mapping[str, Any], *, minutes: float,
                boundary: str | None = None) -> Mapping[str, Any]:
    result = response.get("result")
    if isinstance(result, Mapping):
        return result
    job_id = response.get("job_id")
    if not job_id:
        return response
    waited_since = time.monotonic()
    on_tick = (lambda: _announce(
        f"{boundary} still waiting on job {job_id} "
        f"({int(time.monotonic() - waited_since)}s)")) if boundary else None
    job = panel.wait_for_job(str(job_id), minutes=minutes, on_tick=on_tick)
    events = panel.call("GET", f"/api/jobs/{job_id}/events").get("events", [])
    return {
        **dict(job.get("result") or {"state": job.get("state")}),
        "job_id": str(job_id),
        "job_state": job.get("state"),
        "job_parameters": dict(job.get("parameters") or {}),
        "job_events": list(events),
        "resource_identity": {"job_id": str(job_id), "worker": job.get("worker_id")},
    }


def _mission_artifact(panel, *, mission_id: str, sample_id: str,
                      phase: str, produced_by: str) -> dict[str, Any]:
    response = panel.call(
        "GET",
        f"/api/missions/{mission_id}/artifacts?" + urlencode({
            "phase": phase, "sample": sample_id,
        }),
    )
    row = next(
        (item for item in response.get("artifacts", [])
         if item.get("produced_by") == produced_by),
        None,
    )
    return {
        "artifact_id": (row or {}).get("artifact_id"),
        "sha256": (row or {}).get("content_sha256"),
    }


def _runtime_matches(runtime: Mapping[str, Any], manifest: Mapping[str, Any],
                     deployed_revision: str) -> bool:
    if runtime.get("deployed_revision") != deployed_revision:
        return False
    if runtime.get("control_profile_id") != manifest.get("profile_id"):
        return False
    if runtime.get("control_profile_sha256") != canonical_sha256(manifest):
        return False
    if runtime.get("source_locks_sha256") != canonical_sha256(manifest.get("source_locks")):
        return False
    expected_profiles = {
        (str(row.get("profile_id")), str(row.get("sha256")))
        for row in manifest.get("profile_locks") or []
    }
    actual_profiles = {
        (str(row.get("profile_id")), str(row.get("sha256")))
        for row in runtime.get("profile_locks") or []
    }
    if (runtime.get("profile_locks_verified") is not True
            or actual_profiles != expected_profiles
            or any(row.get("verified") is not True
                   or row.get("declared_sha256_semantics") != "RAW_FILE_BYTES_SHA256"
                   or row.get("actual_file_sha256") != row.get("sha256")
                   or not _is_sha256(row.get("actual_canonical_document_sha256"))
                   for row in runtime.get("profile_locks") or [])):
        return False
    expected_models = {
        (str(row.get("profile_id")), str(row.get("checkpoint_sha256")))
        for row in manifest.get("model_locks") or []
    }
    actual_models = {
        (str(row.get("profile_id")), str(row.get("checkpoint_sha256")))
        for row in runtime.get("models") or []
        if row.get("installed") is True and row.get("installed_at")
    }
    return bool(expected_models) and expected_models == actual_models


def _stage_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    discovery_inputs = ((manifest.get("checks") or {}).get("DISCOVERY_CONTROL") or {}).get("inputs") or {}
    return [
        _row("P0", profiles=[str(manifest.get("profile_id") or "")]),
        _row("P1", profiles=_profiles(manifest, "vc3d", "segmentation"), parameters=discovery_inputs),
        _row("P2", profiles=_profiles(manifest, "surface-qc")),
        _row("QC", profiles=_profiles(manifest, "surface-qc")),
        _row("P3", profiles=_profiles(manifest, "flatten"), parameters={"allow_unvalidated": False}),
        _row("P4", profiles=_profiles(manifest, "flatten")),
        _row("P5", profiles=_profiles(manifest, "timesformer")),
        _row("P7", profiles=_profiles(manifest, "strict-text")),
        _row("HUMAN_REVIEW"),
    ]


def evaluate_survival_matrix(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Purely derive overall state from ordered stage evidence.

    The first non-pass owns the outcome.  Rows after it cannot resurrect the
    control even if a caller accidentally supplies PASS evidence for them; they
    are normalized to prerequisite-not-reached.  No I/O, clock, or environment
    access occurs here, so receipt semantics can be tested independently of the
    authenticated orchestration adapter.
    """
    evaluated = copy.deepcopy(dict(receipt))
    evaluated.pop("content_sha256", None)
    # Which shape this receipt claims to be decides which boundaries it owes.
    # Dispatching on the declared schema rather than on the rows themselves is
    # what stops one control's receipt being read as the other's.
    schema = evaluated.get("schema")
    if not schema:
        raise ValueError(
            "a stage-survival receipt must declare its schema: the boundaries "
            "it owes are a property of which control it is")
    expected = CONTROL_SHAPES.get(str(schema))
    if expected is None:
        raise ValueError(
            f"{schema} is not a stage-survival schema this evaluator knows; "
            f"it knows {sorted(CONTROL_SHAPES)}")
    stages = evaluated.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("a stage-survival receipt needs ordered stage rows")
    if [row.get("boundary") for row in stages if isinstance(row, Mapping)] != list(expected):
        raise ValueError(
            f"a {schema} matrix needs one row per required boundary: {list(expected)}")
    first = next(
        (row for row in stages if row.get("terminal_state") != PASS),
        None,
    )
    if first is None:
        evaluated["control_state"] = "CONTROL_PASS"
        evaluated["first_nonpassing_boundary"] = None
    else:
        terminal = first.get("terminal_state")
        evaluated["control_state"] = (
            "CONTROL_FAILED" if terminal == FAILED else "CONTROL_INCOMPLETE"
        )
        evaluated["first_nonpassing_boundary"] = first.get("boundary")
        first_index = stages.index(first)
        for row in stages[first_index + 1:]:
            row["terminal_state"] = NOT_RUN
            row["reason_code"] = "PREREQUISITE_NOT_REACHED"
            row["elapsed_seconds"] = 0.0
            row["resource_identity"] = {}
            row["output_hashes"] = {}
            row["counts"] = {}
    evaluated["content_sha256"] = canonical_sha256(evaluated)
    return evaluated


def _finish(receipt: dict[str, Any]) -> dict[str, Any]:
    return evaluate_survival_matrix(receipt)


# How long a queued preflight may take before this run reports rather than waits.
# Generous, because the measurement reads a remote store; bounded, because a
# control that never reports is worse than one that reports INCOMPLETE.
# What the status route adds around a receipt: how the job was fetched, not what
# it measured. Hashing these would make the receipt's identity depend on the way
# it was retrieved, so a measurement fetched twice would have two identities.
PREFLIGHT_ENVELOPE_FIELDS = ("job_state", "preflight_job_id", "attempts",
                             "mission_id", "sample_id")


# The fleet's own vocabulary, copied because this runner is stdlib-only: it
# speaks HTTP to a panel and has to stay importable from a bare checkout, so it
# cannot import the store's constants. A test reads them out of the store and
# fails when this copy drifts.
#
# Every state an attempt can end in. Waiting for a state that is missing here is
# not a wrong answer, it is no answer -- the wait runs out and the receipt
# reports a timeout for work that finished.
ATTEMPT_TERMINAL_STATES = frozenset({
    "ARCHIVED", "FIXTURE_ONLY", "QC_PENDING", "NO_SEED", "GROW_FAILED",
    "DUPLICATE_SURFACE", "FINALIZATION_FAILED", "POLICY_REJECTED",
    "BLOCKED_SOURCE_UNAVAILABLE", "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE",
    "PROBE_TECHNICAL_FAILURE", "PROBE_REVIEW_PENDING", "PROBE_REJECTED_ALL",
    "DISCOVERY_PROMOTED", "DISCOVERY_REJECTED_CANDIDATES",
    "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER",
})

# What the fleet finalizes a completed grow with. It is terminal, and QC does
# not change it: finalizing QC updates the surface and leaves the attempt alone.
# P1 previously required `SUCCEEDED`, which no attempt ever reaches, so three
# surfaces grew to 1.58 cm² and the control reported GROW_NONTERMINAL_TIMEOUT.
GROWN_STATE = "QC_PENDING"

# Every way a QC job stops. `BLOCKED_CONFIGURATION` was missing and `CANCELLED`,
# which no job reaches, was being waited for in its place.
QC_JOB_TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "BLOCKED_CONFIGURATION"})

SEED_ALREADY_PLACED = "already have a task"


def _grow_finished(run: Mapping[str, Any]) -> bool:
    """Whether the fleet is done with this attempt, however it ended."""
    return str(run.get("state") or "").upper() in ATTEMPT_TERMINAL_STATES


def _grow_succeeded(run: Mapping[str, Any]) -> bool:
    """Whether this attempt is the grow the control asked for.

    The state alone is not the evidence. `DUPLICATE_SURFACE` is terminal and
    carries a surface somebody else grew, and a receipt without a hash-bound
    artifact is not a grow whatever it is called -- so the artifact is required
    too, and this fails closed without one.
    """
    return (str(run.get("state") or "").upper() == GROWN_STATE
            and bool(run.get("surface_id"))
            and bool(run.get("artifact_sha256")))


class _PlacedSeed:
    """Whether this run placed the control's seed, or a previous one did."""

    def __init__(self, created: dict | None) -> None:
        self.created = created
        self.resumed = created is None

    def excluded_attempts(self, before: set[str]) -> set[str]:
        """Which attempts the wait must ignore.

        Normally the ones that existed before the seed was placed, so the wait
        cannot mistake an older run for this one. On a resume the attempt being
        waited for is precisely one of those, and excluding them waits forever.
        """
        return set() if self.resumed else set(before)


def _place_control_seed(panel, body: Mapping[str, Any]) -> _PlacedSeed:
    """Place the control's seed, or find that a previous run already did.

    A point under a policy is one task forever -- the platform says so and is
    right. The control's policy version encodes the deployed revision, so
    re-running the same revision after an interrupted run cannot re-place the
    seed, and the interrupted run may well have finished the growing.

    Only "this already exists" is a resume. A 409 about a drifted binding or a
    rejected policy is the platform refusing, and it keeps refusing.
    """
    try:
        return _PlacedSeed(panel.call("POST", "/api/segmentation/manual-seeds", dict(body)))
    except PanelError as refused:
        if refused.status == 409 and SEED_ALREADY_PLACED in str(refused):
            return _PlacedSeed(None)
        raise


def _binding_matches(body: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Whether a task's persisted control binding, if it has one, is this run's.

    A P1 seed task has none: the panel attaches the control binding for P4, P5
    and P7 only. This was written the other way round -- requiring the binding on
    a resume, on the belief that it stood in for the creation response -- and so
    refused every resumed control with MANUAL_SEED_PROVENANCE_MISSING, for want
    of evidence that phase does not produce. The unit tests agreed because their
    fixtures supplied the field.

    Absent is therefore the ordinary case and not a failure. What ties a resumed
    task to this run is checked either way, from the attempt the fleet ran: the
    policy version (which encodes the deployed revision and the locks), the grid
    version, the source snapshot -- which is how the P0 is bound -- the seed
    origin, the author and the coordinates.

    Present and disagreeing is still refused, and a half-written binding is
    refused too, exactly as the platform's own `control_job_binding` does.
    """
    identity = payload.get("control_p0_artifact_id")
    digest = payload.get("control_p0_artifact_sha256")
    if identity is None and digest is None:
        return True
    return (identity == body.get("expected_p0_artifact_id")
            and digest == body.get("expected_p0_artifact_sha256")
            and bool(identity))


def _preflight_measurement(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The receipt proper, with the transport envelope removed."""
    return {key: value for key, value in dict(answer).items()
            if key not in PREFLIGHT_ENVELOPE_FIELDS}


# Two hours. 1800 was chosen before any real preflight had finished; the first
# one that did took 4184 seconds -- an hour and nine minutes, COMPLETE, 4200 raw
# M7 candidates -- so the runner gave up on work that was still going and every
# first run on a fresh revision failed at P1 on the clock. Three runs in a row
# went that way.
#
# It is every first run, not a rare path: the deployed revision is part of the
# preflight's request identity, so new code re-measures by design.
#
# Room over the one measurement there is, and no more. A wait that cannot end is
# worse than one that ends early, because the early one reports a timeout
# somebody can read.
#
# It covers two attempts now, not one: a source outage sends the preflight back
# to the queue instead of ending it (`PREFLIGHT_MAXIMUM_REQUEUES`), and a wait
# that stopped before the requeued attempt could finish would turn every
# survived outage into a timeout -- reporting the wait as the problem when the
# dropped connection was. Three runs of this control have died on that
# connection, all of them past minute twenty.
#
# 2 x 4184s (the slowest preflight measured) + one 60s requeue delay = 8428s.
# `tests/test_the_requeue_budget_fits_the_wait_that_bounds_it.py` keeps this and
# the worker's budget to the same arithmetic.
PREFLIGHT_WAIT_SECONDS = 9000
PREFLIGHT_POLL_SECONDS = 5


def _await_preflight(panel, handle: Mapping[str, Any], *, clock,
                     wait_seconds: int = PREFLIGHT_WAIT_SECONDS,
                     poll_seconds: int = PREFLIGHT_POLL_SECONDS,
                     sleep=time.sleep) -> tuple[dict[str, Any] | None, str]:
    """Poll one queued preflight to a terminal state.

    Returns the receipt and an empty reason, or None and the reason code to
    record. Every ending is a reason a reader can act on: a job that failed
    reaching its source says so, one whose handle vanished says that, and one
    that simply took too long says which bound it hit rather than looking like a
    failure of the pipeline.
    """
    job_id = str((handle or {}).get("preflight_job_id") or "")
    if not job_id:
        return None, "PREFLIGHT_HANDLE_MISSING"
    path = f"/api/segmentation/preflight/{quote(job_id, safe='')}"
    deadline = clock() + wait_seconds
    while True:
        try:
            answer = panel.call("GET", path)
        except PanelError as refused:
            if refused.status == 404:
                return None, "PREFLIGHT_JOB_NOT_FOUND"
            if refused.status == 503:
                # The panel puts the reason in the body: a job that failed at its
                # source, a completed one with no receipt, or one whose read sets
                # do not carry the frozen root objects. Each is a different
                # finding and none of them is "still running".
                return None, _refusal_reason(refused, "PREFLIGHT_SOURCE_UNAVAILABLE")
            raise
        if str(answer.get("job_state") or "").upper() == "COMPLETED":
            return dict(answer), ""
        if clock() >= deadline:
            return None, "PREFLIGHT_DID_NOT_FINISH_WITHIN_THE_WAIT"
        sleep(poll_seconds)


def _refusal_reason(refused: PanelError, fallback: str) -> str:
    """The reason code the panel sent, or the fallback if it sent none."""
    body = getattr(refused, "body", "")
    try:
        detail = json.loads(body).get("detail")
    except (TypeError, ValueError):
        return fallback
    if isinstance(detail, Mapping):
        return str(detail.get("reason_code") or fallback)
    return fallback


def _record_unexpected_refusal(rows, refusal) -> None:
    """Put an escaped failure on the boundary the run had reached.

    On the first row not yet decided, never on the first row overall: a receipt
    that blames P0 for a refusal at P4 is worse than no receipt.
    """
    for row in rows:
        # A row starts as NOT_RUN_PREREQUISITE, not blank -- so "undecided" is
        # that marker, and testing for emptiness silently matches nothing.
        if row.get("terminal_state") in (None, "", NOT_RUN):
            _set_result(
                row, state=INCOMPLETE,
                reason=(_refusal_reason(refusal, "UNEXPECTED_REFUSAL")
                        if isinstance(refusal, PanelError)
                        else "UNEXPECTED_FAILURE"),
                started=0.0, clock=lambda: 0.0,
                observation={
                    "detail": f"{type(refusal).__name__}: {str(refusal)[:400]}",
                    "status": getattr(refusal, "status", None),
                })
            return


def run_positive_control(panel, manifest: Mapping[str, Any], *, mission_id: str,
                         deployed_revision: str, submitted_by: str,
                         wait_minutes: float = 60, clock=time.monotonic) -> dict[str, Any]:
    """Run both controls and return a content-bound survival matrix.

    The caller supplies an authenticated human ``panel`` session.  Mutations are
    deliberately issued once.  An ambiguous response stops the matrix at that
    boundary and requires readback by an operator; this function never retries.
    """
    if manifest.get("schema") != "campaignx.first_letters_control_manifest.v1":
        raise ValueError("wrong First Letters control manifest schema")
    if not mission_id.strip() or not submitted_by.strip() or not deployed_revision.strip():
        raise ValueError("mission, deployed revision, and human submitter are required")
    safety = manifest.get("safety") or {}
    if safety.get("allow_unvalidated") is not False:
        raise ValueError("the control manifest must keep allow_unvalidated=false")
    if safety.get("discovery_inputs_are_content_blind") is not True:
        raise ValueError("the control discovery must remain content blind")

    rows = _stage_rows(manifest)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "control_id": (manifest.get("control_cohort") or {}).get("control_id"),
        "mission_id": mission_id,
        "bindings": {
            "deployed_revision": deployed_revision,
            "control_profile_id": manifest.get("profile_id"),
            "source_locks_sha256": canonical_sha256(manifest.get("source_locks")),
            "profile_locks": list(manifest.get("profile_locks") or []),
        },
        "allow_unvalidated": False,
        "ink_blind_discovery": True,
        "control_pass_is_independent_validation": False,
        "automatic_letter_acceptance": False,
        "stages": rows,
    }

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    with _reporting_progress(panel, mission_id, run_id):
        _announce(f"control run starting: mission={mission_id} "
                 f"revision={deployed_revision[:12]} run_id={run_id}")
        try:
            outcome = _run_boundaries(
                panel, manifest, rows, receipt, mission_id=mission_id,
                deployed_revision=deployed_revision, submitted_by=submitted_by,
                wait_minutes=wait_minutes, clock=clock)
        except Exception as escaped:  # noqa: BLE001 - deliberate; see the docstring
            _record_unexpected_refusal(rows, escaped)
            outcome = _finish(receipt)
        _announce(f"control run finished: {outcome.get('control_state')} "
                 f"(first non-passing boundary: "
                 f"{outcome.get('first_nonpassing_boundary')})")
    return outcome


def _run_boundaries(panel, manifest: Mapping[str, Any], rows: list,
                    receipt: dict[str, Any], *, mission_id: str,
                    deployed_revision: str, submitted_by: str,
                    wait_minutes: float, clock) -> dict[str, Any]:
    """Every boundary in order, for a caller that records whatever escapes."""

    # P0: verify that the panel names the exact deployed code and frozen inputs.
    started = clock()
    _announce("P0 starting")
    state = panel.call("GET", "/api/state?" + urlencode({"mission": mission_id}))
    runtime = dict(state.get("first_letters_control_runtime") or {})
    artifacts = panel.call(
        "GET",
        f"/api/missions/{mission_id}/artifacts?" + urlencode({
            "phase": "P0", "sample": (manifest.get("control_cohort") or {}).get("scroll_id")
        }),
    )
    selected_p0 = next((row for row in artifacts.get("artifacts", []) if row.get("selected")), None)
    p0_artifact = {
        "artifact_id": (selected_p0 or {}).get("artifact_id"),
        "sha256": (selected_p0 or {}).get("content_sha256") or (selected_p0 or {}).get("sha256"),
    }
    if not _runtime_matches(runtime, manifest, deployed_revision) or not (
        p0_artifact.get("artifact_id") and _is_sha256(p0_artifact.get("sha256"))
    ):
        _set_result(rows[0], state=INCOMPLETE, reason="CONTROL_INCOMPLETE_STALE",
                    started=started, clock=clock, observation=runtime)
        return _finish(receipt)
    receipt["bindings"].update({
        "control_profile_sha256": runtime.get("control_profile_sha256"),
        "models": list(runtime.get("models") or []),
        "p0_artifact": p0_artifact,
    })
    _set_result(rows[0], state=PASS, reason="EXACT_RUNTIME_BINDING",
                started=started, clock=clock, observation=runtime,
                output_hashes={"p0_artifact_sha256": p0_artifact["sha256"]},
                counts={"input_artifact_count": 1})

    # P1 discovery: same frozen provider and gates, explicitly without growth.
    started = clock()
    _announce("P1 starting")
    discovery_check = (manifest.get("checks") or {}).get("DISCOVERY_CONTROL") or {}
    known_region = manifest.get("known_region") or {}
    bbox = known_region.get("surface_bbox_ct_l0_xyz") or {}
    bbox_min, bbox_max = bbox.get("minimum") or [], bbox.get("maximum") or []
    region_center = {
        axis: (float(bbox_min[index]) + float(bbox_max[index])) / 2.0
        for index, axis in enumerate("xyz")
    }
    region_radius = {
        axis: (float(bbox_max[index]) - float(bbox_min[index])) / 2.0
        for index, axis in enumerate("xyz")
    }
    discovery_inputs = discovery_check.get("inputs") or {}
    discovery_body = {
        "mission_id": mission_id,
        "sample_id": (manifest.get("control_cohort") or {}).get("scroll_id"),
        "region_center_xyz": region_center,
        "region_radius_xyz": region_radius,
        "known_coordinate_xyz": dict(zip("xyz", known_region.get("anchor_ct_l0_xyz") or [], strict=True)),
        "tolerance_ct_l0_voxels": known_region.get("control_tolerance_ct_l0_voxels"),
        "m7_threshold": discovery_inputs.get("m7_level_set_iso_value"),
        "max_candidates": 100,
        "packet_candidate_limit": 8,
        "minimum_separation_voxels": 16,
        "minimum_cell_clearance_voxels": 0,
        "minimum_volume_clearance_voxels": 64,
        "seed_region_policy": discovery_inputs.get("seed_region_policy"),
        "control_surface": {
            "uri": ((manifest.get("source_locks") or {}).get("community_surface") or {}).get("uri"),
            "grid_shape_yx": (manifest.get("known_region") or {}).get("surface_grid_shape_yx"),
            "bbox_ct_l0_xyz": (manifest.get("known_region") or {}).get("surface_bbox_ct_l0_xyz"),
            "artifacts": ((manifest.get("source_locks") or {}).get("community_surface") or {}).get("artifacts"),
        },
        "maximum_surface_probe_regions": 4096,
        "ct_material_support_gate": {
            "policy": discovery_inputs.get("ct_material_support_policy"),
            "level": 5, "radius_l0_voxels": 192, "minimum_nonzero_voxels": 1,
        },
    }
    # The preflight is queued now: it measures through the MCP seed service,
    # which lives where workers live. So this enqueues and waits rather than
    # receiving the answer in the response.
    try:
        handle = panel.call("POST", "/api/segmentation/preflight", discovery_body)
    except (AmbiguousMutationError, TimeoutError):
        # Enqueue is idempotent on the digest of the request, so the safe move
        # is to read state -- but this runner has no handle to read yet, and
        # inventing one would be the retry the client just refused to make.
        _set_result(rows[1], state=INCOMPLETE,
                    reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, input_artifacts=[p0_artifact])
        return _finish(receipt)

    discovery, waited = _await_preflight(panel, handle, clock=clock)
    if discovery is None:
        # A wait that never ends is a control that never reports, so the bound
        # is part of the contract and the reason names which bound was hit.
        _set_result(rows[1], state=INCOMPLETE, reason=waited,
                    started=started, clock=clock,
                    observation={"preflight_job": handle},
                    input_artifacts=[p0_artifact])
        return _finish(receipt)
    discovery_identity = discovery.get("resource_identity") or {}
    source_locks = manifest.get("source_locks") or {}
    if (discovery.get("status") != "COMPLETE"
            or discovery_identity.get("p0_artifact_id") != p0_artifact["artifact_id"]
            or discovery_identity.get("p0_artifact_sha256") != p0_artifact["sha256"]
            or not (_read_set_complete(discovery.get("ct_read_set"))
            and _read_set_complete(discovery.get("m7_read_set"))
            and _read_set_complete(discovery.get("surface_read_set"))
            and _provider_exchange_complete(discovery.get("provider_exchange"))
            and _read_set_contains_locked_metadata(
                discovery.get("ct_read_set"), (source_locks.get("ct") or {}).get("metadata"))
            and _read_set_contains_locked_metadata(
                discovery.get("m7_read_set"), (source_locks.get("m7") or {}).get("metadata"))
            and _traversal_evidence_complete(discovery))):
        _set_result(rows[1], state=INCOMPLETE,
                    reason="MISSING_CONTENT_BOUND_DISCOVERY_EVIDENCE",
                    started=started, clock=clock, observation=discovery,
                    input_artifacts=[p0_artifact])
        return _finish(receipt)
    counts = dict(discovery.get("counts") or {})
    for field, reason in (
        ("raw_m7", "ZERO_RAW_M7_CANDIDATES"),
        ("post_ct", "ZERO_POST_CT_CANDIDATES"),
        ("post_clearance", "ZERO_POST_CLEARANCE_CANDIDATES"),
        ("packet_limited", "ZERO_PACKET_LIMITED_CANDIDATES"),
    ):
        if int(counts.get(field) or 0) <= 0:
            _set_result(rows[1], state=FAILED, reason=reason, started=started,
                        clock=clock, observation=discovery,
                        input_artifacts=[p0_artifact], counts=counts)
            return _finish(receipt)
    distance = discovery.get("closest_survivor_distance_ct_l0_voxels")
    tolerance = (manifest.get("known_region") or {}).get("control_tolerance_ct_l0_voxels")
    if not isinstance(distance, (int, float)) or not math.isfinite(float(distance)) \
            or float(distance) > float(tolerance):
        _set_result(rows[1], state=FAILED,
                    reason="NO_SURVIVOR_INSIDE_DECLARED_TOLERANCE",
                    started=started, clock=clock, observation=discovery,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)

    # P1 pipeline: bootstrap-manual keeps the seed human and leaves the CT gate on.
    seed = ((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {}).get("seed_provenance") or {}
    coordinates = list(seed.get("coordinate_ct_l0_xyz") or [])
    execution_id = canonical_sha256({
        "deployed_revision": deployed_revision,
        "control_profile_sha256": runtime["control_profile_sha256"],
        "source_locks_sha256": runtime["source_locks_sha256"],
        "profile_locks": runtime["profile_locks"],
        "models": runtime["models"],
    })[:16]
    manual_body = {
        "sample_id": (manifest.get("control_cohort") or {}).get("scroll_id"),
        "points": [dict(zip("xyz", coordinates, strict=True))],
        "note": "First Letters PIPELINE_CONTROL; public surface seed, not a letter coordinate",
        "policy_version": f"first-letters-control@1.0.0-{execution_id}",
        "grid_version": "first-letters-control-manual-v1",
        "mission_id": mission_id,
        "expected_p0_artifact_id": p0_artifact["artifact_id"],
        "expected_p0_artifact_sha256": p0_artifact["sha256"],
        "expected_source_snapshot_id": discovery_identity.get("source_snapshot_id"),
        "options": {},
    }
    runs_path = "/api/segmentation/runs?" + urlencode({
        "sample": manual_body["sample_id"], "mission": mission_id, "limit": 200
    })
    before_attempts = {
        str(row.get("attempt_id")) for row in panel.call("GET", runs_path).get("runs", [])
        if row.get("attempt_id")
    }
    try:
        placed = _place_control_seed(panel, manual_body)
    except (AmbiguousMutationError, TimeoutError):
        _set_result(rows[1], state=INCOMPLETE,
                    reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, observation=discovery,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)
    # On a resume there is no creation response to read provenance from. Nothing
    # is skipped: every field below is verified again from the attempt the fleet
    # actually ran, and the P0 binding persisted on the task is checked there
    # too -- which is what the fleet grew against, rather than what a request
    # said it would.
    before_attempts = placed.excluded_attempts(before_attempts)
    manual = placed.created
    manual_identity = (manual or {}).get("resource_identity") or {}
    if manual is not None and (
            manual.get("submitted_by") != submitted_by
            or manual_identity.get("p0_artifact_id") != p0_artifact["artifact_id"]
            or manual_identity.get("p0_artifact_sha256") != p0_artifact["sha256"]
            or manual_identity.get("source_snapshot_id") != discovery_identity.get("source_snapshot_id")
            or (manual.get("receipt") or {}).get("seed_origin") != "human"):
        _set_result(rows[1], state=INCOMPLETE, reason="MANUAL_SEED_PROVENANCE_MISSING",
                    started=started, clock=clock,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)
    def finished_control_attempt():
        for run in panel.call("GET", runs_path).get("runs", []):
            if str(run.get("attempt_id")) in before_attempts:
                continue
            if _grow_finished(run):
                return run
        return None

    pipeline_run = panel.wait_until(
        finished_control_attempt, minutes=wait_minutes, tick=5,
        on_tick=lambda: _announce(
            f"P1 still waiting for the grow to finish ({clock() - started:.0f}s)"))
    if pipeline_run is None:
        _set_result(rows[1], state=INCOMPLETE, reason="GROW_NONTERMINAL_TIMEOUT",
                    started=started, clock=clock, observation=discovery,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)
    attempt = panel.call("GET", f"/api/segmentation/attempt/{pipeline_run['attempt_id']}")
    task_payload = attempt.get("payload") or {}
    actual_candidate = next(iter(task_payload.get("manual_candidates") or []), {})
    pipeline = {
        **pipeline_run,
        "canonical_full_grow": bool(pipeline_run.get("surface_id") and pipeline_run.get("artifact_sha256")),
        "artifact_id": pipeline_run.get("surface_id"),
        "seed_origin": actual_candidate.get("seed_origin"),
        "submitted_by": actual_candidate.get("submitted_by"),
        "coordinate_ct_l0_xyz": [
            (actual_candidate.get("ct_l0_coordinate") or {}).get(axis) for axis in "xyz"
        ],
        "resource_identity": {"attempt_id": pipeline_run.get("attempt_id"),
                              "worker": pipeline_run.get("worker_id")},
    }
    if not _grow_succeeded(pipeline_run) or pipeline.get("canonical_full_grow") is not True:
        _set_result(rows[1], state=FAILED, reason="GROW_REJECTED", started=started,
                    clock=clock, observation=pipeline, input_artifacts=[p0_artifact],
                    counts=counts)
        return _finish(receipt)
    if (task_payload.get("source_snapshot_id") != discovery_identity.get("source_snapshot_id")
            or task_payload.get("grid_version") != manual_body["grid_version"]
            or task_payload.get("policy_version") != manual_body["policy_version"]
            or (placed.resumed and not _binding_matches(manual_body, task_payload))
            or pipeline.get("seed_origin") != "human" or pipeline.get("submitted_by") != submitted_by) \
            or list(pipeline.get("coordinate_ct_l0_xyz") or []) != coordinates:
        _set_result(rows[1], state=INCOMPLETE, reason="MANUAL_SEED_PROVENANCE_MISSING",
                    started=started, clock=clock, observation=pipeline,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)
    if float(pipeline.get("area_cm2") or 0.0) < MINIMUM_STANDARD_SURFACE_AREA_CM2:
        _set_result(rows[1], state=FAILED, reason="TINY_SURFACE_DIAGNOSTIC_ONLY",
                    started=started, clock=clock, observation=pipeline,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)
    surface_artifact = _artifact(pipeline, default_id=str(pipeline.get("surface_id") or ""))
    if not surface_artifact.get("artifact_id") or not _is_sha256(surface_artifact.get("sha256")):
        _set_result(rows[1], state=INCOMPLETE, reason="MISSING_CANONICAL_GROW_ARTIFACT",
                    started=started, clock=clock, observation=pipeline,
                    input_artifacts=[p0_artifact], counts=counts)
        return _finish(receipt)
    _set_result(rows[1], state=PASS, reason="DISCOVERY_AND_MANUAL_GROW_SURVIVED",
                started=started, clock=clock, observation={
                    **pipeline,
                    "resource_identity": {
                        "discovery": dict(discovery.get("resource_identity") or {}),
                        "pipeline": dict(pipeline.get("resource_identity") or {}),
                    },
                },
                input_artifacts=[p0_artifact], counts={
                    **counts,
                    "closest_survivor_distance_ct_l0_voxels": float(distance),
                    "planned_region_count": discovery["planned_region_count"],
                    "visited_region_count": discovery["visited_region_count"],
                    "maximum_surface_probe_regions": discovery["parameters"]["maximum_surface_probe_regions"],
                    "coverage_fraction": discovery["coverage_fraction"],
                    "coverage_complete": discovery["coverage_complete"],
                    "surface_count": 1,
                }, output_hashes={
                    "ct_read_set_manifest_sha256": discovery["ct_read_set"]["canonical_manifest_sha256"],
                    "m7_read_set_manifest_sha256": discovery["m7_read_set"]["canonical_manifest_sha256"],
                    "surface_read_set_manifest_sha256": discovery["surface_read_set"]["canonical_manifest_sha256"],
                    "m7_provider_request_sha256": discovery["provider_exchange"]["request_sha256"],
                    "m7_provider_response_sha256": discovery["provider_exchange"]["response_sha256"],
                    "surface_sha256": surface_artifact["sha256"],
                    "preflight_receipt_sha256": canonical_sha256(
                        _preflight_measurement(discovery)),
                    **dict(discovery.get("output_hashes") or {}),
                })
    rows[1]["parameters"] = dict(discovery["parameters"])

    # A newly finalized surface has already crossed the geometry gate. P2 is
    # retained for backlog/recertification, but inventing a redundant P2 job
    # here cannot certify anything: P2 deliberately selects only UNMEASURED rows.
    started = clock()
    _announce("P2 starting")
    p2: dict[str, Any] = {}
    readback: dict[str, Any] = {}
    def terminal_qc_readback():
        segments = panel.call("GET", "/api/segmentation/segments?" + urlencode({
            "sample": manual_body["sample_id"], "mission": mission_id, "limit": 2000
        }))
        surface = next(
            (row for row in segments.get("segments", [])
             if row.get("surface_id") == pipeline.get("surface_id")), None
        )
        qc_jobs = panel.call("GET", "/api/segmentation/qc-jobs?" + urlencode({
            "mission": mission_id, "sample": manual_body["sample_id"],
            "surface": pipeline.get("surface_id"), "limit": 100,
        }))
        readback.update(surface=surface, qc_jobs=qc_jobs.get("jobs", []))
        if surface is None:
            return None
        geometry = str(surface.get("geometry_qc_state") or "")
        if geometry.startswith("GEOMETRY_REJECTED"):
            return readback
        physical = str(surface.get("physical_qc_state") or "")
        terminal_job = any(
            str(job.get("state") or "").upper() in QC_JOB_TERMINAL_STATES
            and _is_sha256(job.get("result_sha256"))
            for job in readback["qc_jobs"]
        )
        if geometry == "GEOMETRY_CERTIFIED" and terminal_job \
                and physical not in {"", "UNVALIDATED", "PENDING", "RUNNING"}:
            return readback
        return None

    try:
        panel.wait_until(
            terminal_qc_readback, minutes=wait_minutes, tick=5,
            on_tick=lambda: _announce(
                f"P2 still waiting for a terminal QC readback "
                f"({clock() - started:.0f}s)"))
    except TimeoutError:
        pass
    certified_surface = readback.get("surface")
    if certified_surface is None:
        _set_result(rows[2], state=INCOMPLETE, reason="P2_SURFACE_LINEAGE_MISSING",
                    started=started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    current_p2_surface = next(
        (row for row in p2.get("surfaces") or []
         if row.get("surface_id") == pipeline.get("surface_id")),
        None,
    )
    if current_p2_surface is not None:
        p2 = {**p2, **current_p2_surface}
    p2 = {
        **p2,
        "surface_id": certified_surface["surface_id"],
        "geometry_qc_state": certified_surface.get("geometry_qc_state"),
        "physical_qc_state": certified_surface.get("physical_qc_state"),
        "qc_jobs": readback.get("qc_jobs", []),
    }
    finalizer_geometry = certified_surface.get("geometry_certification") or {}
    if isinstance(finalizer_geometry, Mapping):
        p2.update({
            "surface_artifact_sha256": finalizer_geometry.get(
                "surface_artifact_sha256"),
            "profile_id": finalizer_geometry.get("profile_id"),
            "profile_sha256": finalizer_geometry.get("profile_sha256"),
            "result": finalizer_geometry.get("result"),
            "result_sha256": finalizer_geometry.get("result_sha256"),
            "source_attempt_id": finalizer_geometry.get("source_attempt_id"),
            "created_geometry_certified": True,
        })
    p2_job_id = str(p2.get("job_id") or "")
    geometry_profile = (((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {})
                        .get("geometry_certification_profile") or {})
    p2_result_document = p2.get("result")
    p2_event = None
    p2_event_lineage = None
    for event in p2.get("job_events") or []:
        payload = event.get("payload")
        if event.get("event_type") != "succeeded" or not isinstance(payload, Mapping):
            continue
        candidates = [payload, *(payload.get("surfaces") or [])]
        lineage = next(
            (row for row in candidates if isinstance(row, Mapping)
             and row.get("geometry_certified_by_job_id") == p2_job_id
             and row.get("surface_id") == surface_artifact["artifact_id"]),
            None,
        )
        if lineage is not None:
            p2_event, p2_event_lineage = event, lineage
            break
    finalizer_current_evidence = (
        p2.get("geometry_qc_state") == "GEOMETRY_CERTIFIED"
        and p2.get("surface_id") == surface_artifact["artifact_id"]
        and p2.get("surface_artifact_sha256") == surface_artifact["sha256"]
        and certified_surface.get("artifact_sha256") == surface_artifact["sha256"]
        and p2.get("source_attempt_id") == pipeline.get("attempt_id")
        and p2.get("profile_id") == geometry_profile.get("profile_id")
        and p2.get("profile_sha256") == geometry_profile.get(
            "canonical_policy_sha256")
        and isinstance(p2_result_document, Mapping)
        and p2.get("result_sha256") == canonical_sha256(p2_result_document)
    )
    p2_current_evidence = finalizer_current_evidence or (
        p2.get("job_state") == "succeeded"
        and p2_job_id
        and p2.get("geometry_certified_by_job_id") == p2_job_id
        and (p2.get("job_parameters") or {}).get("surface_id") == surface_artifact["artifact_id"]
        and p2.get("surface_id") == surface_artifact["artifact_id"]
        and p2.get("surface_artifact_sha256") == surface_artifact["sha256"]
        and certified_surface.get("artifact_sha256") == surface_artifact["sha256"]
        and p2.get("profile_id") == geometry_profile.get("profile_id")
        and p2.get("profile_sha256") == geometry_profile.get("canonical_policy_sha256")
        and isinstance(p2_result_document, Mapping)
        and p2.get("result_sha256") == canonical_sha256(p2_result_document)
        and p2_event is not None
        and p2_event_lineage.get("surface_artifact_sha256") == surface_artifact["sha256"]
        and p2_event_lineage.get("profile_id") == p2.get("profile_id")
        and p2_event_lineage.get("profile_sha256") == p2.get("profile_sha256")
        and p2_event_lineage.get("result_sha256") == p2.get("result_sha256")
    )
    physical_receipt = next(
        (job for job in p2["qc_jobs"]
         if job.get("surface_id") == pipeline.get("surface_id")
         and (not finalizer_current_evidence or (
             job.get("created_geometry_certified") is True
             and job.get("source_attempt_id") == pipeline.get("attempt_id")
             and job.get("surface_artifact_sha256") == surface_artifact["sha256"]))
         and str(job.get("state") or "").upper() == str(
             ((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {}).get(
                 "qc_job_terminal_state") or "COMPLETED").upper()
         and _is_sha256(job.get("result_sha256"))),
        None,
    )
    p2["geometry_receipt_sha256"] = canonical_sha256({
        "job_id": p2.get("job_id"),
        "surface_id": certified_surface.get("surface_id"),
        "artifact_sha256": certified_surface.get("artifact_sha256"),
        "geometry_qc_state": certified_surface.get("geometry_qc_state"),
    })
    if not p2_current_evidence:
        _set_result(rows[2], state=INCOMPLETE,
                    reason="P2_CURRENT_JOB_EVIDENCE_MISSING",
                    started=started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    if p2.get("geometry_qc_state") != "GEOMETRY_CERTIFIED":
        state = FAILED if str(p2.get("geometry_qc_state", "")).startswith("GEOMETRY_REJECTED") else INCOMPLETE
        _set_result(rows[2], state=state,
                    reason="GEOMETRY_REJECTED" if state == FAILED else "GEOMETRY_NONTERMINAL",
                    started=started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    _set_result(rows[2], state=PASS,
                reason=("PASS_ALREADY_CERTIFIED_AT_FINALIZATION"
                        if finalizer_current_evidence else "GEOMETRY_CERTIFIED"), started=started,
                clock=clock, observation=p2, input_artifacts=[surface_artifact],
                output_hashes={"geometry_receipt_sha256": p2["geometry_receipt_sha256"],
                               "geometry_result_sha256": p2["result_sha256"]},
                counts={"certified_surface_count": 1})

    qc_started = clock()
    _announce("QC starting")
    admissible = set(((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {}).get(
        "terminal_physical_qc_compatible_with_downstream") or [])
    physical = p2.get("physical_qc_state")
    if physical in {None, "", "UNVALIDATED", "PENDING", "RUNNING"}:
        _set_result(rows[3], state=INCOMPLETE,
                    reason="PHYSICAL_QC_NONTERMINAL",
                    started=qc_started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    if physical_receipt is None:
        _set_result(rows[3], state=INCOMPLETE,
                    reason="PHYSICAL_QC_HASH_BOUND_RECEIPT_MISSING",
                    started=qc_started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    qc_profile = next(
        (row for row in manifest.get("profile_locks") or []
         if row.get("profile_id") == "surface-qc-gp-scroll1-ct-fiber-v3@1.0.0"),
        None,
    )
    qc_result = physical_receipt.get("result")
    if not (
        qc_profile
        and physical_receipt.get("profile_id") == qc_profile.get("profile_id")
        and physical_receipt.get("profile_sha256") == qc_profile.get("sha256")
        and physical_receipt.get("surface_artifact_sha256") == surface_artifact["sha256"]
        and ((finalizer_current_evidence
              and physical_receipt.get("created_geometry_certified") is True
              and physical_receipt.get("source_attempt_id") == pipeline.get("attempt_id")
              and physical_receipt.get("unblocked_by_job_id") in (None, ""))
             or (not finalizer_current_evidence
                 and physical_receipt.get("unblocked_by_job_id") == p2_job_id
                 and isinstance(physical_receipt.get("promotion_event_id"), str)
                 and physical_receipt.get("promotion_event_id")))
        and isinstance(qc_result, Mapping)
        and physical_receipt.get("result_sha256") == canonical_sha256(qc_result)
    ):
        _set_result(rows[3], state=INCOMPLETE,
                    reason="PHYSICAL_QC_CURRENT_JOB_EVIDENCE_MISSING",
                    started=qc_started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    if physical not in admissible:
        _set_result(rows[3], state=FAILED,
                    reason="PHYSICAL_QC_TERMINAL_INCOMPATIBLE",
                    started=qc_started, clock=clock, observation=p2,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    _set_result(rows[3], state=PASS, reason="PHYSICAL_QC_DOWNSTREAM_COMPATIBLE",
                started=qc_started, clock=clock, observation=p2,
                input_artifacts=[surface_artifact],
                output_hashes={"physical_qc_receipt_sha256": physical_receipt["result_sha256"]},
                counts={"downstream_compatible_surface_count": 1})

    # P3: allow_unvalidated is explicit false and cannot be overridden.
    started = clock()
    _announce("P3 starting")
    try:
        p3_response = panel.call("POST", "/api/flattening/run", {
            "sample_id": manual_body["sample_id"], "mission_id": mission_id,
            "surface_id": pipeline.get("surface_id"), "limit": 1,
            "dry_run": False, "allow_unvalidated": False,
        })
        p3 = _job_result(panel, p3_response, minutes=wait_minutes,
                          boundary="P3")
    except (AmbiguousMutationError, TimeoutError):
        _set_result(rows[4], state=INCOMPLETE,
                    reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, input_artifacts=[surface_artifact])
        return _finish(receipt)
    except PanelError as refused:
        # A refusal is a boundary result, not an exception. Catching only the
        # ambiguous cases let every 4xx escape: the run died, wrote no receipt,
        # and the boundaries it had already crossed went unrecorded. That is how
        # the first control ever to reach P3 left nothing but a traceback.
        _set_result(rows[4], state=INCOMPLETE,
                    reason=_refusal_reason(refused, "P3_REFUSED_BY_THE_PLATFORM"),
                    started=started, clock=clock,
                    observation={"status": refused.status, "detail": str(refused)[:400]},
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    flattening = panel.call("GET", "/api/flattening?" + urlencode({
        "sample": manual_body["sample_id"], "mission": mission_id
    }))
    flat_row = next(
        (row for row in flattening.get("rows", [])
         if row.get("surface_id") == pipeline.get("surface_id")
         and row.get("profile_id") == "flatten-abf-v1@1.0.0"), None
    )
    if flat_row is not None:
        p3 = {**p3, **flat_row}
    flat_artifact = {
        "artifact_id": ((flat_row or {}).get("artifact_id")
                        or (flat_row or {}).get("flattening_id")),
        "sha256": (flat_row or {}).get("artifact_sha256"),
    }
    p3_job_id = str(p3.get("job_id") or "")
    p3_profile = next(
        (row for row in manifest.get("profile_locks") or []
         if row.get("profile_id") == "flatten-abf-v1@1.0.0"), None)
    p3_receipt = p3.get("receipt")
    p3_event_lineage = None
    for event in p3.get("job_events") or []:
        payload = event.get("payload")
        if event.get("event_type") != "succeeded" or not isinstance(payload, Mapping):
            continue
        candidates = [payload, *(payload.get("surfaces") or [])]
        p3_event_lineage = next(
            (row for row in candidates if isinstance(row, Mapping)
             and row.get("requested_by_job_id") == p3_job_id
             and row.get("surface_id") == surface_artifact["artifact_id"]),
            None,
        )
        if p3_event_lineage is not None:
            break
    p3_current_evidence = (
        p3.get("job_state") == "succeeded"
        and p3_job_id
        and (p3.get("job_parameters") or {}).get("surface_id") == surface_artifact["artifact_id"]
        and (p3.get("job_parameters") or {}).get("allow_unvalidated") is not True
        and p3.get("requested_by_job_id") == p3_job_id
        and p3.get("surface_id") == surface_artifact["artifact_id"]
        and p3.get("source_artifact_sha256") == surface_artifact["sha256"]
        and p3_profile is not None
        and p3.get("profile_id") == p3_profile.get("profile_id")
        and p3.get("profile_file_sha256") == p3_profile.get("sha256")
        and isinstance(p3.get("artifact_uri"), str) and p3.get("artifact_uri")
        and _artifact_inventory_complete(p3)
        and isinstance(p3_receipt, Mapping)
        and p3.get("receipt_sha256") == canonical_sha256(p3_receipt)
        and p3_receipt.get("requested_by_job_id") == p3_job_id
        and p3_receipt.get("surface_id") == surface_artifact["artifact_id"]
        and p3_receipt.get("source_artifact_sha256") == surface_artifact["sha256"]
        and p3_receipt.get("profile_id") == p3_profile.get("profile_id")
        and p3_receipt.get("profile_file_sha256") == p3_profile.get("sha256")
        and p3_receipt.get("artifact_uri") == p3.get("artifact_uri")
        and p3_receipt.get("artifact_sha256") == p3.get("artifact_sha256")
        and p3_receipt.get("objects") == p3.get("objects")
        and p3_event_lineage is not None
        and p3_event_lineage.get("receipt_sha256") == p3.get("receipt_sha256")
        and p3_event_lineage.get("artifact_sha256") == p3.get("artifact_sha256")
    )
    if not p3_current_evidence:
        reason = ("MISSING_HASH_BOUND_FLATTENED_ARTIFACT"
                  if not _is_sha256(p3.get("artifact_sha256")) else
                  "P3_CURRENT_JOB_EVIDENCE_MISSING")
        _set_result(rows[4], state=INCOMPLETE,
                    reason=reason,
                    started=started, clock=clock, observation=p3,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    if p3.get("state") != "FLATTENED" or not flat_artifact.get("artifact_id") \
            or not _is_sha256(flat_artifact.get("sha256")):
        _set_result(rows[4], state=INCOMPLETE,
                    reason="MISSING_HASH_BOUND_FLATTENED_ARTIFACT",
                    started=started, clock=clock, observation=p3,
                    input_artifacts=[surface_artifact])
        return _finish(receipt)
    _set_result(rows[4], state=PASS, reason="HASH_BOUND_FLATTENED_ARTIFACT",
                started=started, clock=clock, observation=p3,
                input_artifacts=[surface_artifact],
                output_hashes={"flattened_artifact_sha256": flat_artifact["sha256"],
                               "flattening_receipt_sha256": p3["receipt_sha256"]},
                counts={"flattened_artifact_count": 1})

    # Orientation is server-derived from the exact P3 lineage. Parity alone is
    # insufficient: only a content-bound proof with verified absolute evidence
    # is allowed to choose the renderer's normal direction.
    orientation_policy = (((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {})
                          .get("orientation_parity") or {}).get("policy") or {}
    reference = (manifest.get("source_locks") or {}).get("community_surface") or {}
    orientation = panel.call("GET", "/api/geometry/orientation-proof?" + urlencode({
        "mission": mission_id,
        "sample": manual_body["sample_id"],
        "surface": surface_artifact["artifact_id"],
        "p3_job": p3_job_id,
    }))
    orientation_lineage = orientation.get("lineage") or {}
    orientation_absolute = orientation_lineage.get("absolute_orientation") or {}
    parity_state = orientation.get("parity_state")
    selected_flip = orientation.get("selected_flip_normals")
    expected_flip = (
        orientation_absolute.get("same_winding_flip_normals")
        if parity_state == "PROVEN_SAME_WINDING"
        else not orientation_absolute.get("same_winding_flip_normals")
        if parity_state == "PROVEN_OPPOSITE_WINDING"
        and isinstance(orientation_absolute.get("same_winding_flip_normals"), bool)
        else None
    )
    orientation_receipt_sha = orientation.get("receipt_sha256")
    orientation_evidence_complete = (
        orientation.get("schema") == "campaignx.first_letters_orientation_parity.v1"
        and orientation.get("status") == "PROVEN"
        and orientation.get("profile_id") == orientation_policy.get("profile_id")
        and orientation.get("profile_sha256") == canonical_sha256(orientation_policy)
        and orientation.get("policy") == orientation_policy
        and _is_sha256(orientation_receipt_sha)
        and orientation_receipt_sha == canonical_sha256({
            key: value for key, value in orientation.items()
            if key != "receipt_sha256"
        })
        and orientation_lineage.get("reference") == {
            "uri": reference.get("uri"),
            "objects": reference.get("artifacts") or [],
            "artifact_manifest_sha256": canonical_sha256(reference.get("artifacts") or []),
        }
        and orientation_lineage.get("grown_mesh_artifact") == {
            "artifact_id": surface_artifact["artifact_id"],
            "sha256": surface_artifact["sha256"],
        }
        and orientation_lineage.get("flattened_artifact") == {
            "artifact_id": flat_artifact["artifact_id"],
            "sha256": flat_artifact["sha256"],
        }
        and orientation_lineage.get("p3") == {
            "job_id": p3_job_id,
            "profile_id": p3.get("profile_id"),
            "profile_sha256": p3.get("profile_file_sha256"),
            "receipt_sha256": p3.get("receipt_sha256"),
        }
        and orientation_absolute.get("verified") is True
        and _is_sha256(orientation_absolute.get("evidence_receipt_sha256"))
        and isinstance(orientation_absolute.get("same_winding_flip_normals"), bool)
        and isinstance(selected_flip, bool)
        and selected_flip == expected_flip
    )
    if not orientation_evidence_complete:
        _set_result(rows[5], state=INCOMPLETE, reason="ORIENTATION_UNPROVEN",
                    started=clock(), clock=clock, observation=orientation,
                    input_artifacts=[flat_artifact],
                    output_hashes={
                        **({"orientation_receipt_sha256": orientation_receipt_sha}
                           if _is_sha256(orientation_receipt_sha) else {}),
                    })
        return _finish(receipt)

    # P4/P5/P7 use the normal job API and consume only the preceding bound artifact.
    execution = ((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {}).get(
        "execution_parameters") or {}
    p4_parameters = execution.get("P4") or {}
    p5_parameters = execution.get("P5") or {}
    started = clock()
    _announce("P4 starting")
    try:
        p4_response = panel.call("POST", "/api/jobs", {
            "sample_id": manual_body["sample_id"], "mission_id": mission_id,
            "phase": "P4", "max_attempts": int(execution.get("maximum_attempts", 1)),
            "parameters": {
                "lane": "vc-render-tifxyz",
                "flattened_surface": surface_artifact["artifact_id"],
                "flattening_profile": p3.get("profile_id"),
                "flattening_id": flat_artifact["artifact_id"],
                "p3_job_id": p3_job_id,
                "flattened_artifact_sha256": flat_artifact["sha256"],
                "remote_url": (manifest.get("source_locks") or {}).get("ct", {}).get("uri"),
                "source_voxel_um": (manifest.get("source_locks") or {}).get("ct", {}).get(
                    "voxel_size_um"),
                "scale": p4_parameters.get("scale"),
                "group_idx": p4_parameters.get("group_idx"),
                "num_slices": p4_parameters.get("num_slices"),
                "slice_step": p4_parameters.get("slice_step"),
                "flip_normals": selected_flip,
                "orientation_receipt_sha256": orientation_receipt_sha,
            },
        })
        p4 = _job_result(panel, p4_response, minutes=wait_minutes,
                          boundary="P4")
    except (AmbiguousMutationError, TimeoutError):
        _set_result(rows[5], state=INCOMPLETE, reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, input_artifacts=[flat_artifact])
        return _finish(receipt)
    except PanelError as refused:
        _set_result(rows[5], state=INCOMPLETE,
                    reason=_refusal_reason(refused, "P4_REFUSED_BY_THE_PLATFORM"),
                    started=started, clock=clock,
                    observation={"status": refused.status, "detail": str(refused)[:400]},
                    input_artifacts=[flat_artifact])
        return _finish(receipt)
    stored_p4 = p4.get("job_parameters", {})
    p4_edge = next((event.get("payload") for event in p4.get("job_events", [])
                    if event.get("event_type") == "rendered_from"), {})
    if (stored_p4.get("flattened_surface") != surface_artifact["artifact_id"]
            or stored_p4.get("flattening_profile") != p3.get("profile_id")
            or stored_p4.get("flattening_id") != flat_artifact["artifact_id"]
            or stored_p4.get("p3_job_id") != p3_job_id
            or stored_p4.get("flattened_artifact_sha256") != flat_artifact["sha256"]
            or stored_p4.get("remote_url") != (manifest.get("source_locks") or {}).get("ct", {}).get("uri")
            or stored_p4.get("source_voxel_um") != (manifest.get("source_locks") or {}).get(
                "ct", {}).get("voxel_size_um")
            or not stored_p4.get("volume")
            or p4_edge.get("surface_id") != surface_artifact["artifact_id"]
            or p4_edge.get("flattening_id") != flat_artifact["artifact_id"]
            or p4_edge.get("p3_job_id") != p3_job_id
            or p4_edge.get("flattened_artifact_sha256") != flat_artifact["sha256"]
            or p4_edge.get("profile_id") != p3.get("profile_id")
            or stored_p4.get("orientation_receipt_sha256") != orientation_receipt_sha
            or stored_p4.get("flip_normals") is not selected_flip
            or p4_edge.get("orientation_receipt_sha256") != orientation_receipt_sha
            or p4_edge.get("flip_normals") is not selected_flip
            or any(stored_p4.get(field) != p4_parameters.get(field)
                   for field in ("scale", "group_idx", "num_slices", "slice_step"))):
        _set_result(rows[5], state=INCOMPLETE, reason="P4_INPUT_LINEAGE_MISMATCH",
                    started=started, clock=clock, observation=p4,
                    input_artifacts=[flat_artifact])
        return _finish(receipt)
    layer_manifest = p4.get("layer_stack") or {}
    layer_measurement = p4.get("layers") or {}
    layer_objects = layer_manifest.get("objects") or []
    render_artifact = {"artifact_id": p4.get("job_id"),
                       "sha256": layer_manifest.get("artifact_sha256")}
    manifest_complete = (
        p4.get("job_state") == "succeeded"
        and _artifact_inventory_complete(layer_manifest)
        and _exact_numeric_tiff_inventory(layer_manifest, 33)
        and isinstance(layer_manifest.get("files"), int)
        and layer_manifest.get("files") == 33
        and len(layer_objects) == layer_manifest.get("files")
        and _is_sha256(layer_manifest.get("manifest_sha256"))
        and all(isinstance(item, Mapping) and item.get("object_key")
                and _is_sha256(item.get("sha256"))
                and isinstance(item.get("bytes"), int) for item in layer_objects)
        and isinstance(layer_measurement.get("slices"), int)
        and layer_measurement.get("slices") == layer_manifest.get("files")
        and isinstance(layer_measurement.get("shape"), list)
        and len(layer_measurement.get("shape")) == 2
        and render_artifact.get("artifact_id") and _is_sha256(render_artifact.get("sha256"))
    )
    if not manifest_complete:
        _set_result(rows[5], state=INCOMPLETE, reason="INCOMPLETE_RENDER_MANIFEST",
                    started=started, clock=clock, observation=p4,
                    input_artifacts=[flat_artifact])
        return _finish(receipt)
    _set_result(rows[5], state=PASS, reason="COMPLETE_HASH_BOUND_LAYER_MANIFEST",
                started=started, clock=clock, observation=p4,
                input_artifacts=[flat_artifact], counts={"layer_file_count": layer_manifest["files"]},
                output_hashes={"render_artifact_sha256": render_artifact["sha256"],
                               "layer_manifest_sha256": layer_manifest["manifest_sha256"],
                               "orientation_receipt_sha256": orientation_receipt_sha})

    started = clock()
    _announce("P5 starting")
    metric_contract = (((manifest.get("checks") or {}).get("PIPELINE_CONTROL") or {})
                       .get("p3_p4_lateral_metric") or {})
    metric_policy = metric_contract.get("policy") or {}
    lateral_metric = p4.get("lateral_metric") or {}
    metric_receipt_sha = lateral_metric.get("receipt_sha256")
    ct_voxel_um = (manifest.get("source_locks") or {}).get("ct", {}).get("voxel_size_um")
    try:
        lateral_pixel_um = float(lateral_metric.get("lateral_pixel_um"))
        source_voxel_um = float(lateral_metric.get("source_voxel_um"))
        valid_fraction = float(lateral_metric.get("valid_triangle_fraction"))
        distortion = float(lateral_metric.get("observed_uv_to_3d_distortion_ratio"))
        expected_source_slice_um = (
            float(ct_voxel_um) * float(stored_p4.get("scale"))
            * float(stored_p4.get("slice_step")))
    except (TypeError, ValueError):
        lateral_pixel_um = source_voxel_um = valid_fraction = distortion = math.nan
        expected_source_slice_um = math.nan
    metric_complete = (
        lateral_metric.get("schema") == "campaignx.first_letters_p3_p4_lateral_metric.v1"
        and lateral_metric.get("status") == "PROVEN"
        and lateral_metric.get("profile_id") == metric_policy.get("profile_id")
        and lateral_metric.get("profile_sha256") == canonical_sha256(metric_policy)
        and _is_sha256(metric_receipt_sha)
        and metric_receipt_sha == canonical_sha256({
            key: value for key, value in lateral_metric.items()
            if key != "receipt_sha256"
        })
        and lateral_metric.get("lineage") == {
            "flattened_artifact_id": flat_artifact["artifact_id"],
            "flattened_artifact_sha256": flat_artifact["sha256"],
            "p3_job_id": p3_job_id,
            "p4_job_id": p4.get("job_id"),
            "p4_layer_artifact_sha256": render_artifact["sha256"],
            "p4_layer_manifest_sha256": layer_manifest.get("manifest_sha256"),
            "p4_layer_objects": layer_manifest.get("objects"),
        }
        and all(math.isfinite(value) and value > 0 for value in (
            lateral_pixel_um, source_voxel_um, valid_fraction, distortion,
            expected_source_slice_um))
        and source_voxel_um == float(ct_voxel_um)
        and valid_fraction >= float(metric_policy.get("minimum_valid_triangle_fraction") or math.inf)
        and distortion <= float(metric_policy.get("maximum_uv_to_3d_distortion_ratio") or 0)
        and isinstance(lateral_metric.get("raster_transform"), Mapping)
    )
    if not metric_complete:
        _set_result(rows[6], state=INCOMPLETE, reason="P5_LATERAL_METRIC_UNPROVEN",
                    started=started, clock=clock, observation=p4,
                    input_artifacts=[render_artifact],
                    output_hashes={
                        **({"lateral_metric_receipt_sha256": metric_receipt_sha}
                           if _is_sha256(metric_receipt_sha) else {}),
                    })
        return _finish(receipt)

    p5_profile = next(iter(_profiles(manifest, "timesformer")), None)
    model = next(
        (row for row in runtime.get("models", []) if row.get("profile_id") == p5_profile),
        None,
    )
    if not model or not model.get("installed_at"):
        _set_result(rows[6], state=INCOMPLETE, reason="P5_MODEL_NOT_INSTALLED",
                    started=started, clock=clock, input_artifacts=[render_artifact])
        return _finish(receipt)
    try:
        p5_response = panel.call("POST", "/api/jobs", {
            "sample_id": manual_body["sample_id"], "mission_id": mission_id,
            "phase": "P5", "profile_id": p5_profile,
            "max_attempts": int(execution.get("maximum_attempts", 1)),
            "parameters": {
                "layer_stack": p4.get("job_id"),
                "checkpoint": model["installed_at"],
                "source_pixel_um": lateral_pixel_um,
                "source_slice_um": expected_source_slice_um,
                "on_degenerate": p5_parameters.get("on_degenerate"),
                "batch_size": p5_parameters.get("batch_size"),
                "device": p5_parameters.get("device"),
            },
        })
        p5 = _job_result(panel, p5_response, minutes=wait_minutes,
                          boundary="P5")
    except (AmbiguousMutationError, TimeoutError):
        _set_result(rows[6], state=INCOMPLETE, reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, input_artifacts=[render_artifact])
        return _finish(receipt)
    except PanelError as refused:
        _set_result(rows[6], state=INCOMPLETE,
                    reason=_refusal_reason(refused, "P5_REFUSED_BY_THE_PLATFORM"),
                    started=started, clock=clock,
                    observation={"status": refused.status, "detail": str(refused)[:400]},
                    input_artifacts=[render_artifact])
        return _finish(receipt)
    stored_p5 = p5.get("job_parameters", {})
    p5_edge = next((event.get("payload") for event in p5.get("job_events", [])
                    if event.get("event_type") == "rendered_from"), {})
    normalization = p5.get("physical_normalization") or {}
    normalization_receipt_sha = normalization.get("receipt_sha256")
    expected_layer_manifest_sha = layer_manifest.get("manifest_sha256")
    training_pixel_um = p5_parameters.get("verified_training_pixel_um")
    normalization_complete = (
        normalization.get("schema") == "campaignx.first_letters_p5_normalization.v1"
        and _is_sha256(normalization_receipt_sha)
        and normalization_receipt_sha == canonical_sha256({
            key: value for key, value in normalization.items()
            if key != "receipt_sha256"
        })
        and normalization.get("p4_job_id") == p4.get("job_id")
        and normalization.get("p4_layer_artifact_sha256") == render_artifact["sha256"]
        and normalization.get("p4_layer_manifest_sha256") == expected_layer_manifest_sha
        and normalization.get("source_layer_objects") == layer_manifest["objects"]
        and len(normalization.get("source_layer_objects") or []) == 33
        and normalization.get("lateral_metric_receipt_sha256") == metric_receipt_sha
        and normalization.get("source_pixel_um") == lateral_pixel_um
        and normalization.get("source_slice_um") == expected_source_slice_um
        and normalization.get("training_pixel_um") == training_pixel_um
        and normalization.get("checkpoint_sha256") == model.get("checkpoint_sha256")
        and normalization.get("profile_id") == p5_parameters.get("training_profile_id")
        and normalization.get("profile_sha256") == p5_parameters.get("training_profile_sha256")
    )
    if (stored_p5.get("layer_stack") != p4.get("job_id")
            or p5_edge.get("rendered_by") != p4.get("job_id")
            or p5_edge.get("layer_stack_artifact_sha256") != render_artifact["sha256"]
            or p5_edge.get("layer_stack_manifest_sha256") != expected_layer_manifest_sha
            or p5_edge.get("lateral_metric_receipt_sha256") != metric_receipt_sha
            or p5_edge.get("source_pixel_um") != lateral_pixel_um
            or p5_edge.get("source_slice_um") != expected_source_slice_um
            or stored_p5.get("checkpoint") != model.get("installed_at")
            or stored_p5.get("source_pixel_um") != lateral_pixel_um
            or stored_p5.get("source_slice_um") != expected_source_slice_um
            or any(stored_p5.get(field) != p5_parameters.get(field)
                   for field in ("on_degenerate", "batch_size", "device"))
            or p5.get("checkpoint_sha256") != model.get("checkpoint_sha256")
            or not normalization_complete):
        _set_result(rows[6], state=INCOMPLETE, reason="P5_INPUT_OR_MODEL_LINEAGE_MISMATCH",
                    started=started, clock=clock, observation=p5,
                    input_artifacts=[render_artifact])
        return _finish(receipt)
    probability_map = p5.get("probability_map") or {}
    map_objects = probability_map.get("objects") or []
    map_artifact = {"artifact_id": p5.get("job_id"),
                    "sha256": probability_map.get("artifact_sha256")}
    if p5.get("job_state") != "succeeded" or (p5.get("liveness") or {}).get("verdict") != "ALIVE" \
            or not map_artifact.get("artifact_id") or not _artifact_inventory_complete(probability_map):
        _set_result(rows[6], state=FAILED, reason="P5_DEGENERATE_OR_EMPTY",
                    started=started, clock=clock, observation=p5,
                    input_artifacts=[render_artifact])
        return _finish(receipt)
    _set_result(rows[6], state=PASS, reason="ALIVE_HASH_BOUND_PROBABILITY_MAP",
                started=started, clock=clock, observation=p5,
                input_artifacts=[render_artifact],
                output_hashes={"probability_map_sha256": map_artifact["sha256"],
                               "probability_map_manifest_sha256": probability_map.get(
                                   "manifest_sha256"),
                               "checkpoint_sha256": model["checkpoint_sha256"],
                               "normalization_receipt_sha256": normalization_receipt_sha,
                               "source_layer_manifest_sha256": expected_layer_manifest_sha,
                               "lateral_metric_receipt_sha256": metric_receipt_sha},
                counts={"alive_probability_map_count": 1})

    started = clock()
    _announce("P7 starting")
    shape_yx = p5.get("map_shape_yx") or layer_measurement.get("shape")
    if not isinstance(shape_yx, list) or len(shape_yx) != 2:
        _set_result(rows[7], state=INCOMPLETE, reason="P7_BBOX_INPUT_MISSING",
                    started=started, clock=clock, input_artifacts=[map_artifact])
        return _finish(receipt)
    roi = panel.call("GET", "/api/validation/positive-control-roi?" + urlencode({
        "mission": mission_id,
        "sample": manual_body["sample_id"],
        "surface": surface_artifact["artifact_id"],
        "p5_job": p5.get("job_id"),
    }))
    roi_lineage = roi.get("lineage") or {}
    roi_lock = roi.get("lock") or {}
    roi_bbox = roi.get("transformed_bbox_xyxy")
    roi_receipt_sha = roi.get("receipt_sha256")
    training_pixel_um = normalization.get("training_pixel_um")
    bbox_complete = (
        isinstance(roi_bbox, list) and len(roi_bbox) == 4
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in roi_bbox)
        and 0 <= roi_bbox[0] < roi_bbox[2] <= int(shape_yx[1])
        and 0 <= roi_bbox[1] < roi_bbox[3] <= int(shape_yx[0])
        and roi_bbox != [0, 0, int(shape_yx[1]), int(shape_yx[0])]
    )
    roi_complete = (
        roi.get("schema") == "campaignx.first_letters_positive_control_roi.v1"
        and roi.get("status") == "PROVEN"
        and _is_sha256(roi_receipt_sha)
        and roi_receipt_sha == canonical_sha256({
            key: value for key, value in roi.items() if key != "receipt_sha256"
        })
        and roi_lock.get("verified") is True
        and isinstance(roi_lock.get("provenance_artifact_uri"), str)
        and bool(roi_lock.get("provenance_artifact_uri"))
        and _is_sha256(roi_lock.get("provenance_artifact_sha256"))
        and isinstance(roi_lock.get("source_coordinate_system"), str)
        and bool(roi_lock.get("source_coordinate_system"))
        and isinstance(roi_lock.get("source_bbox_xyxy"), list)
        and _is_sha256(roi_lock.get("p5_transform_receipt_sha256"))
        and roi_lock.get("transformed_bbox_xyxy") == roi_bbox
        and roi_lock.get("verified_training_pixel_um") == training_pixel_um
        and roi.get("verified_training_pixel_um") == training_pixel_um
        and roi.get("map_shape_yx") == shape_yx
        and bbox_complete
        and roi_lineage == {
            "surface_id": surface_artifact["artifact_id"],
            "p5_job_id": p5.get("job_id"),
            "probability_map_sha256": map_artifact["sha256"],
            "probability_map_manifest_sha256": probability_map.get(
                "manifest_sha256"),
            "normalization_receipt_sha256": normalization_receipt_sha,
            "checkpoint_sha256": model.get("checkpoint_sha256"),
            "profile_id": normalization.get("profile_id"),
            "profile_sha256": normalization.get("profile_sha256"),
        }
    )
    if not roi_complete:
        _set_result(rows[7], state=INCOMPLETE, reason="P7_ROI_UNPROVEN",
                    started=started, clock=clock, observation=roi,
                    input_artifacts=[map_artifact],
                    output_hashes={
                        **({"roi_receipt_sha256": roi_receipt_sha}
                           if _is_sha256(roi_receipt_sha) else {}),
                    })
        return _finish(receipt)
    bbox_parameter = ",".join(str(value) for value in roi_bbox)
    if not isinstance(training_pixel_um, (int, float)) or float(training_pixel_um) <= 0:
        _set_result(rows[7], state=INCOMPLETE, reason="P7_PHYSICAL_SCALE_MISSING",
                    started=started, clock=clock, input_artifacts=[map_artifact])
        return _finish(receipt)
    try:
        p7_response = panel.call("POST", "/api/jobs", {
            "sample_id": manual_body["sample_id"], "mission_id": mission_id,
            "phase": "P7",
            "max_attempts": int(execution.get("maximum_attempts", 1)),
            "parameters": {
                "screening_of": p5.get("job_id"),
                "surface_id": surface_artifact["artifact_id"],
                "bbox": bbox_parameter,
                "px_um": float(training_pixel_um),
                "roi_receipt_sha256": roi_receipt_sha,
                "probability_map_artifact_sha256": map_artifact["sha256"],
                "probability_map_manifest_sha256": probability_map.get(
                    "manifest_sha256"),
            },
        })
        p7 = _job_result(panel, p7_response, minutes=wait_minutes,
                          boundary="P7")
    except (AmbiguousMutationError, TimeoutError):
        _set_result(rows[7], state=INCOMPLETE, reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, input_artifacts=[map_artifact])
        return _finish(receipt)
    except PanelError as refused:
        _set_result(rows[7], state=INCOMPLETE,
                    reason=_refusal_reason(refused, "P7_REFUSED_BY_THE_PLATFORM"),
                    started=started, clock=clock,
                    observation={"status": refused.status, "detail": str(refused)[:400]},
                    input_artifacts=[map_artifact])
        return _finish(receipt)
    p7_edge = next((event.get("payload") for event in p7.get("job_events", [])
                    if event.get("event_type") == "rendered_from"), {})
    p7_map_input = p7.get("probability_map_input") or {}
    if (p7.get("job_parameters", {}).get("screening_of") != p5.get("job_id")
            or p7.get("job_parameters", {}).get("surface_id") != surface_artifact["artifact_id"]
            or p7.get("job_parameters", {}).get(
                "probability_map_artifact_sha256") != map_artifact["sha256"]
            or p7.get("job_parameters", {}).get(
                "probability_map_manifest_sha256") != probability_map.get(
                    "manifest_sha256")
            or p7_edge.get("screened_by") != p5.get("job_id")
            or p7_edge.get("surface_id") != surface_artifact["artifact_id"]
            or p7_edge.get("probability_map_artifact_sha256") != map_artifact["sha256"]
            or p7_edge.get("probability_map_manifest_sha256") != probability_map.get(
                "manifest_sha256")
            or p7_map_input != {
                "screened_by": p5.get("job_id"),
                "artifact_sha256": map_artifact["sha256"],
                "manifest_sha256": probability_map.get("manifest_sha256"),
            }
            or p7.get("job_parameters", {}).get("bbox") != bbox_parameter
            or p7.get("job_parameters", {}).get("roi_receipt_sha256") != roi_receipt_sha
            or p7_edge.get("bbox") != bbox_parameter
            or p7_edge.get("roi_receipt_sha256") != roi_receipt_sha
            or float(p7_edge.get("px_um") or 0) != float(training_pixel_um)
            or float(p7.get("job_parameters", {}).get("px_um") or 0) != float(training_pixel_um)):
        _set_result(rows[7], state=INCOMPLETE, reason="P7_INPUT_LINEAGE_MISMATCH",
                    started=started, clock=clock, observation=p7,
                    input_artifacts=[map_artifact])
        return _finish(receipt)
    adjudication = p7.get("adjudication") or {}
    packet_artifact = _mission_artifact(
        panel, mission_id=mission_id, sample_id=manual_body["sample_id"],
        phase="P7", produced_by=f"job:{p7.get('job_id')}",
    )
    if (p7.get("job_state") != "succeeded" or adjudication.get("verdict") != "PASS"
            or not _is_sha256(adjudication.get("verdict_sha256"))
            or not _is_sha256(adjudication.get("card_sha256"))
            or not _is_sha256(adjudication.get("config_hash"))):
        _set_result(rows[7], state=FAILED,
                    reason="P7_ROUTE_CONTRADICTS_KNOWN_POSITIVE",
                    started=started, clock=clock, observation=p7,
                    input_artifacts=[map_artifact])
        return _finish(receipt)
    if not packet_artifact.get("artifact_id") or not _is_sha256(packet_artifact.get("sha256")):
        _set_result(rows[7], state=INCOMPLETE, reason="MISSING_HASH_BOUND_P7_PACKET",
                    started=started, clock=clock, observation=p7,
                    input_artifacts=[map_artifact])
        return _finish(receipt)
    _set_result(rows[7], state=PASS, reason="KNOWN_POSITIVE_ROUTED_TO_REVIEW",
                started=started, clock=clock, observation=p7,
                input_artifacts=[map_artifact],
                output_hashes={"p7_artifact_sha256": packet_artifact["sha256"],
                               "vetting_packet_sha256": packet_artifact["sha256"],
                               "verdict_sha256": adjudication["verdict_sha256"],
                               "vetting_card_sha256": adjudication["card_sha256"],
                               "strict_screen_config_sha256": adjudication["config_hash"],
                               "roi_receipt_sha256": roi_receipt_sha},
                counts={"priority_review_route_count": 1})

    # Human boundary means a hash-bound packet is routed, never automated acceptance.
    started = clock()
    _announce("HUMAN_REVIEW starting")
    review_path = f"/api/jobs/{p7.get('job_id')}/review"
    before_reviews = panel.call("GET", review_path).get("human_reviews", [])
    before_hashes = {
        row.get("event_sha256") for row in before_reviews
        if isinstance(row, Mapping) and _is_sha256(row.get("event_sha256"))
    }
    try:
        posted_review = panel.call("POST", review_path, {
            "verdict": "INSPECT",
            "note": "First Letters positive-control packet routed for human inspection",
        })
    except (AmbiguousMutationError, TimeoutError):
        _set_result(rows[8], state=INCOMPLETE,
                    reason="AMBIGUOUS_MUTATION_NO_RETRY_UNTIL_READBACK",
                    started=started, clock=clock, input_artifacts=[packet_artifact])
        return _finish(receipt)
    except PanelError as refused:
        _set_result(rows[8], state=INCOMPLETE,
                    reason=_refusal_reason(refused, "HUMAN_REVIEW_REFUSED_BY_THE_PLATFORM"),
                    started=started, clock=clock,
                    observation={"status": refused.status, "detail": str(refused)[:400]},
                    input_artifacts=[packet_artifact])
        return _finish(receipt)

    def new_review():
        current = panel.call("GET", review_path).get("human_reviews", [])
        candidates = [
            row for row in current if isinstance(row, Mapping)
            and row.get("event_sha256") not in before_hashes
            and row.get("p7_job_id") == p7.get("job_id")
            and row.get("intent") == "INSPECT"
        ]
        return candidates[0] if len(candidates) == 1 else None

    human = dict(panel.wait_until(
        new_review, minutes=wait_minutes, tick=5,
        on_tick=lambda: _announce(
            f"HUMAN_REVIEW still waiting for a reviewer "
            f"({clock() - started:.0f}s)")) or {})
    human_sha = human.get("event_sha256")
    if (not human
            or human != posted_review
            or human.get("schema") != "campaignx.human_review_event.v1"
            or human.get("mission_id") != mission_id
            or human.get("sample_id") != manual_body["sample_id"]
            or human.get("surface_id") != pipeline.get("surface_id")
            or human.get("p5_job_id") != p5.get("job_id")
            or human.get("p4_job_id") != p4.get("job_id")
            or human.get("p3_job_id") != p3_job_id
            or human.get("flattened_artifact_id") != flat_artifact["artifact_id"]
            or human.get("flattened_artifact_sha256") != flat_artifact["sha256"]
            or human.get("p4_layer_artifact_sha256") != render_artifact["sha256"]
            or human.get("vetting_packet_sha256") != packet_artifact["sha256"]
            or human.get("p7_job_id") != p7.get("job_id")
            or human.get("p7_artifact_id") != packet_artifact["artifact_id"]
            or human.get("verdict_sha256") != adjudication["verdict_sha256"]
            or human.get("card_sha256") != adjudication["card_sha256"]
            or human.get("config_sha256") != adjudication["config_hash"]
            or human.get("roi_receipt_sha256") != roi_receipt_sha
            or human.get("intent") != "INSPECT"
            or not _is_sha256(human_sha)
            or human_sha != canonical_sha256({
                key: value for key, value in human.items() if key != "event_sha256"
            })):
        _set_result(rows[8], state=INCOMPLETE, reason="HUMAN_REVIEW_ROUTING_MISSING",
                    started=started, clock=clock, observation=human,
                    input_artifacts=[packet_artifact])
        return _finish(receipt)
    _set_result(rows[8], state=PASS, reason="HUMAN_REVIEW_PACKET_ROUTED",
                started=started, clock=clock, observation=human,
                input_artifacts=[packet_artifact],
                output_hashes={
                    "vetting_packet_sha256": human["vetting_packet_sha256"],
                    "human_review_record_sha256": human_sha,
                },
                counts={"routed_packet_count": 1})
    return _finish(receipt)


def publish_positive_control(panel, receipt: Mapping[str, Any], *,
                             mode: str | None = None) -> dict[str, Any]:
    """Hand the matrix to the control plane that will gate on it.

    The receipt used to exist only as a local file, so the panel had no way to
    answer "is the control current and passing on this exact revision?" and the
    campaign gate had to take an operator's word for it. Publishing is not
    self-certification: the server re-derives the matrix from the stage rows
    with :func:`evaluate_survival_matrix` and refuses anything it cannot
    reproduce, including a receipt whose ``control_state`` was edited.

    A failing or incomplete control is published too. "The control failed" is
    evidence the gate must see; leaving it unpublished would be indistinguishable
    from never having run one.
    """
    mission_id = quote(str(receipt.get("mission_id") or ""), safe="")
    return panel.call(
        "POST", f"/api/missions/{mission_id}/first-letters-control",
        {"receipt": dict(receipt),
         **({"execution_mode": mode} if mode else {})})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--deployed-revision", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default=os.environ.get("HELENA_PANEL_PASSWORD"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--no-publish", action="store_true",
        help="write the receipt locally without publishing it to the panel; "
             "the campaign readiness gate will keep reporting no control")
    parser.add_argument(
        "--exploratory", action="store_true",
        help="publish as a development run: the mission carries no campaign "
             "binding, so the record certifies nothing and no campaign's "
             "readiness gate can read it")
    parser.add_argument("--trust")
    parser.add_argument("--insecure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not arguments.password:
        raise SystemExit("pass --password or set HELENA_PANEL_PASSWORD")
    document = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    panel = Panel(arguments.panel, trust=arguments.trust, insecure=arguments.insecure)
    signed_in = panel.sign_in(arguments.user, arguments.password)
    if signed_in != arguments.user:
        raise SystemExit("the authenticated Helena session did not retain its user")
    result = run_positive_control(
        panel, document, mission_id=arguments.mission,
        deployed_revision=arguments.deployed_revision, submitted_by=signed_in,
    )
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    published = "NOT_REQUESTED"
    if not arguments.no_publish:
        try:
            publish_positive_control(
                panel, result,
                mode="EXPLORATORY" if arguments.exploratory else None)
        except AmbiguousMutationError as ambiguous:
            # Content-addressed on the server, so a replay would be harmless --
            # but the rule in this program is that an unread mutation is read
            # back by a person, not guessed at by a retry loop.
            print(json.dumps({"receipt": str(arguments.receipt),
                              "control_state": result["control_state"],
                              "content_sha256": result["content_sha256"],
                              "published": "AMBIGUOUS_NOT_RETRIED",
                              "detail": str(ambiguous)}, sort_keys=True))
            return 2
        published = "PUBLISHED"
    print(json.dumps({"receipt": str(arguments.receipt),
                      "control_state": result["control_state"],
                      "content_sha256": result["content_sha256"],
                      "published": published}, sort_keys=True))
    return 0 if result["control_state"] == "CONTROL_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
