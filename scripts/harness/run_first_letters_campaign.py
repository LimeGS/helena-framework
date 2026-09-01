#!/usr/bin/env python3
"""Run one First Letters campaign as a sequence of transactional waves.

The failed campaign queued a fixed budget of tasks, watched a counter, and
queued more. Nothing between waves re-read whether the positive control still
matched the deployed revision, whether the preflight still described the
current source, or whether the starvation rule had already fired -- so the
decision to expand was made from a number rather than from evidence, and the
campaign ran to eight hundred attempts past the point where its own policy said
to stop.

A wave here is transactional at the decision level. Before each one the
orchestrator captures readiness and an attempt baseline; it enqueues no more
than the budget the control plane derived; it waits for every task that wave
created to reach a terminal state; it recomputes the pause; and only then does
it consider another wave.

An ambiguous POST -- a gateway timeout, a dropped connection, any response the
client could not read -- FREEZES the orchestrator. The mutation may or may not
have committed, and retrying it would create a second wave under a new
identity. The run stops with an authoritative readback attached and a human
decides what happened. There is no retry path in this file.

The module is import-safe: importing it parses no arguments, authenticates
nothing, opens no socket, and writes no file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from panel_client import AmbiguousMutationError, Panel

SCHEMA = "campaignx.first_letters_campaign_ledger.v1"
VETTING_PACKET_KIND = "vetting-packet"
SHA256_LENGTH = 64

# Stop reasons. Every one of them is a statement about evidence or about the
# orchestrator's own limits, never about what a scroll contains.
BLOCKED = "READINESS_BLOCKED"
FROZEN = "FROZEN_AMBIGUOUS_WAVE_POST"
HUMAN_REVIEW = "HUMAN_REVIEW_PACKET_ROUTED"
NOT_CONTROLLED = "MISSION_IS_NOT_A_CONTROLLED_CAMPAIGN"
PAUSED = "PAUSE_CANDIDATE_STARVATION"
SCROLL_BLOCKED = "SCROLL_NOT_ADMITTED"
QUEUE_UNREADABLE = "QUEUE_STATE_UNREADABLE"
WAVE_LIMIT = "WAVE_LIMIT_REACHED"
WAVE_NOT_TERMINAL = "WAVE_DID_NOT_REACH_TERMINAL"

NON_CLAIMS = (
    "A completed wave is a statement about attempted cells. It is not evidence "
    "that a scroll holds no surface, no ink, and no text.",
    "This ledger records what was queued and what became terminal. It accepts "
    "no letter, no reading, and no ink claim.",
    "Scientific success requires a blinded human to confirm the preregistered "
    "criterion. Reaching a human-review packet is the start of that, not the "
    "end of it.",
)


class CampaignFrozen(RuntimeError):
    """A wave POST had an unreadable outcome, so the campaign stopped.

    Carries the authoritative readback taken after the ambiguity and the ledger
    up to that point. The wave must not be reissued until a human has read the
    control plane and decided whether it committed.
    """

    def __init__(self, wave_index: int, detail: str, readback: dict[str, Any],
                 ledger: dict[str, Any]):
        self.wave_index = wave_index
        self.detail = detail
        self.readback = readback
        self.ledger = ledger
        super().__init__(
            f"wave {wave_index} has an ambiguous outcome ({detail}); it must "
            "not be retried until an operator reads the control plane back and "
            "decides whether it committed"
        )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == SHA256_LENGTH
            and all(character in "0123456789abcdef" for character in value))


def read_readiness(panel, mission_id: str) -> dict[str, Any]:
    """The one read every wave decision is made from."""
    return panel.call(
        "GET", f"/api/missions/{quote(mission_id, safe='')}/first-letters-readiness")


def read_review_packets(panel, mission_id: str) -> list[str]:
    """Hash-bound P7 vetting packets, which stop further expansion."""
    body = panel.call(
        "GET",
        f"/api/missions/{quote(mission_id, safe='')}/artifacts?"
        + urlencode({"phase": "P7"}))
    return sorted({
        str(row.get("content_sha256"))
        for row in body.get("artifacts") or []
        if isinstance(row, dict)
        and row.get("kind") == VETTING_PACKET_KIND
        and _is_sha256(row.get("content_sha256"))
    })


def scroll_state(readiness: dict[str, Any], sample_id: str) -> dict[str, Any] | None:
    for row in readiness.get("scrolls") or []:
        if not isinstance(row, dict):
            continue
        if sample_id in {row.get("sample_id"), row.get("requested_sample_id")}:
            return row
    return None


def _queue(readiness: dict[str, Any]) -> dict[str, Any]:
    queue = readiness.get("queue")
    return queue if isinstance(queue, dict) else {}


def _active_task_ids(readiness: dict[str, Any]) -> set[str]:
    return {str(row) for row in _queue(readiness).get("active_task_ids") or []}


def _stop(ledger: dict[str, Any], because: str, *, detail: str | None = None,
          readiness: dict[str, Any] | None = None) -> dict[str, Any]:
    ledger["stopped_because"] = because
    ledger["stopped_detail"] = detail
    if readiness is not None:
        ledger["closing_readiness_sha256"] = readiness.get("readiness_sha256")
        ledger["blockers"] = list(readiness.get("blockers") or [])
    return _sealed(ledger)


def _sealed(ledger: dict[str, Any]) -> dict[str, Any]:
    ledger.pop("content_sha256", None)
    ledger["content_sha256"] = canonical_sha256(ledger)
    return ledger


def _wait_for_wave(panel, mission_id: str, *, baseline_active: set[str],
                   wait_minutes: float, poll_seconds: float, clock, sleep,
                   ) -> tuple[bool, dict[str, Any]]:
    """Poll until every task this wave added is terminal, or say it did not.

    Only tasks outside the baseline are waited on. Work that was already
    running belongs to somebody else's decision, and blocking on it would let
    an unrelated queue hold this campaign open forever.
    """
    deadline = clock() + wait_minutes * 60
    readiness = read_readiness(panel, mission_id)
    while True:
        outstanding = _active_task_ids(readiness) - baseline_active
        if not outstanding:
            return True, readiness
        if clock() >= deadline:
            return False, readiness
        sleep(poll_seconds)
        readiness = read_readiness(panel, mission_id)


def run_campaign(panel, *, mission_id: str, sample_id: str,
                 maximum_waves: int = 4, wait_minutes: float = 180,
                 poll_seconds: float = 20, clock=time.monotonic,
                 sleep=time.sleep) -> dict[str, Any]:
    """Run bounded waves while the control plane keeps saying it is ready.

    Returns a machine-readable ledger. Raises :class:`CampaignFrozen` -- and
    only that -- when a wave POST could not be read back.
    """
    if maximum_waves < 1:
        raise ValueError("a campaign needs at least one wave")
    opening = read_readiness(panel, mission_id)
    ledger: dict[str, Any] = {
        "schema": SCHEMA,
        "mission_id": mission_id,
        "sample_id": sample_id,
        "deployed_revision": opening.get("deployed_revision"),
        "opening_readiness_sha256": opening.get("readiness_sha256"),
        "closing_readiness_sha256": opening.get("readiness_sha256"),
        "maximum_waves": maximum_waves,
        "waves": [],
        "blockers": list(opening.get("blockers") or []),
        "review_packet_sha256s": [],
        "stopped_because": None,
        "stopped_detail": None,
        "non_claims": list(NON_CLAIMS),
    }
    if opening.get("controlled") is not True:
        return _stop(ledger, NOT_CONTROLLED, readiness=opening,
                     detail="the mission carries no First Letters campaign binding")

    opening_packets = read_review_packets(panel, mission_id)
    ledger["opening_review_packet_sha256s"] = list(opening_packets)
    ledger["review_packet_sha256s"] = list(opening_packets)
    if opening_packets:
        return _stop(ledger, HUMAN_REVIEW, readiness=opening,
                     detail="a hash-bound review packet is already awaiting a person")

    readiness = opening
    for index in range(1, maximum_waves + 1):
        if readiness.get("blockers"):
            first = readiness["blockers"][0]
            return _stop(ledger, BLOCKED, readiness=readiness,
                         detail=f"{first.get('code')}: {first.get('detail')}")
        pause = readiness.get("pause") or {}
        if pause.get("active") is True:
            return _stop(ledger, PAUSED, readiness=readiness,
                         detail=f"the campaign decision is {pause.get('decision')}")
        scroll = scroll_state(readiness, sample_id)
        if scroll is None:
            return _stop(ledger, SCROLL_BLOCKED, readiness=readiness,
                         detail=f"{sample_id} is not a scroll of this campaign")
        if scroll.get("queue_admitted") is not True:
            blockers = scroll.get("blockers") or [{}]
            return _stop(ledger, SCROLL_BLOCKED, readiness=readiness,
                         detail=(f"{blockers[0].get('code')}: "
                                 f"{blockers[0].get('detail')}"))
        budget = scroll.get("budget") or {}
        planned = budget.get("planned_task_count")
        if not isinstance(planned, int) or isinstance(planned, bool) or planned < 1:
            return _stop(ledger, SCROLL_BLOCKED, readiness=readiness,
                         detail="the current budget authorizes no tasks")
        queue = _queue(readiness)
        if queue.get("available") is not True:
            return _stop(ledger, QUEUE_UNREADABLE, readiness=readiness,
                         detail=str(queue.get("reason")))

        baseline_active = _active_task_ids(readiness)
        baseline_attempts = queue.get("attempt_count")
        wave: dict[str, Any] = {
            "index": index,
            "readiness_sha256": readiness.get("readiness_sha256"),
            "budget_receipt_sha256": budget.get("receipt_sha256"),
            "budgeted_task_count": planned,
            "requested_task_count": planned,
            "baseline_attempt_count": baseline_attempts,
            "baseline_active_task_ids": sorted(baseline_active),
        }
        body = {
            "mission_id": mission_id,
            "sample_id": sample_id,
            # Exactly the frozen budget. The control plane refuses anything
            # else, and asking for less would silently reduce the detection
            # probability the budget was derived to reach.
            "max_tasks": planned,
            "task_budget_receipt_sha256": budget.get("receipt_sha256"),
        }
        try:
            created = panel.call("POST", "/api/segmentation/runs", body)
        except AmbiguousMutationError as ambiguous:
            # Read back before saying anything, and never post again. A second
            # POST here is how one wave becomes two under different identities.
            wave["enqueued_task_count"] = None
            wave["outcome"] = "AMBIGUOUS_NOT_RETRIED"
            ledger["waves"].append(wave)
            readback = read_readiness(panel, mission_id)
            ledger["closing_readiness_sha256"] = readback.get("readiness_sha256")
            _stop(ledger, FROZEN, readiness=readback, detail=str(ambiguous))
            raise CampaignFrozen(
                index, str(ambiguous), readback, ledger) from ambiguous

        inserted = created.get("inserted")
        wave["enqueued_task_count"] = inserted
        wave["generated_task_count"] = created.get("generated")
        if not isinstance(inserted, int) or isinstance(inserted, bool) \
                or inserted > planned:
            wave["outcome"] = "OVER_BUDGET_INSERT"
            ledger["waves"].append(wave)
            return _stop(ledger, SCROLL_BLOCKED, readiness=readiness,
                         detail=("the control plane reported more inserted tasks "
                                 "than the frozen budget allows"))

        reached, readiness = _wait_for_wave(
            panel, mission_id, baseline_active=baseline_active,
            wait_minutes=wait_minutes, poll_seconds=poll_seconds,
            clock=clock, sleep=sleep)
        wave["reached_terminal"] = reached
        wave["active_task_ids_at_close"] = sorted(
            _active_task_ids(readiness) - baseline_active)
        wave["closing_attempt_count"] = _queue(readiness).get("attempt_count")
        wave["pause_after_wave"] = readiness.get("pause")
        wave["outcome"] = "TERMINAL" if reached else "NOT_TERMINAL"
        ledger["waves"].append(wave)
        ledger["closing_readiness_sha256"] = readiness.get("readiness_sha256")
        if not reached:
            return _stop(ledger, WAVE_NOT_TERMINAL, readiness=readiness,
                         detail=(f"wave {index} still had work in flight after "
                                 f"{wait_minutes} minutes"))

        packets = read_review_packets(panel, mission_id)
        ledger["review_packet_sha256s"] = list(packets)
        if set(packets) - set(opening_packets):
            return _stop(ledger, HUMAN_REVIEW, readiness=readiness,
                         detail=("a canonical candidate produced a hash-bound "
                                 "review packet; a person decides next"))
        pause = readiness.get("pause") or {}
        if pause.get("active") is True:
            return _stop(ledger, PAUSED, readiness=readiness,
                         detail=f"the campaign decision is {pause.get('decision')}")

    return _stop(ledger, WAVE_LIMIT, readiness=readiness,
                 detail=f"the requested {maximum_waves}-wave bound was reached")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--mission", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default=os.environ.get("HELENA_PANEL_PASSWORD"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--maximum-waves", type=int, default=4)
    parser.add_argument("--wait-minutes", type=float, default=180)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--trust")
    parser.add_argument("--insecure", action="store_true")
    return parser


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if not arguments.password:
        raise SystemExit("pass --password or set HELENA_PANEL_PASSWORD")
    panel = Panel(arguments.panel, trust=arguments.trust,
                  insecure=arguments.insecure)
    if panel.sign_in(arguments.user, arguments.password) != arguments.user:
        raise SystemExit("the authenticated Helena session did not retain its user")
    try:
        ledger = run_campaign(
            panel, mission_id=arguments.mission, sample_id=arguments.sample,
            maximum_waves=arguments.maximum_waves,
            wait_minutes=arguments.wait_minutes,
            poll_seconds=arguments.poll_seconds)
    except CampaignFrozen as frozen:
        _write_ledger(arguments.ledger, frozen.ledger)
        print(json.dumps({"ledger": str(arguments.ledger),
                          "stopped_because": FROZEN,
                          "wave_index": frozen.wave_index,
                          "detail": frozen.detail,
                          "readiness_sha256": frozen.readback.get(
                              "readiness_sha256")}, sort_keys=True))
        # Distinct from an ordinary refusal on purpose: this exit code means
        # the platform must be read by a person before anything else runs.
        return 2
    _write_ledger(arguments.ledger, ledger)
    print(json.dumps({"ledger": str(arguments.ledger),
                      "stopped_because": ledger["stopped_because"],
                      "waves": len(ledger["waves"]),
                      "content_sha256": ledger["content_sha256"]}, sort_keys=True))
    return 0 if ledger["stopped_because"] in {
        WAVE_LIMIT, HUMAN_REVIEW, PAUSED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
