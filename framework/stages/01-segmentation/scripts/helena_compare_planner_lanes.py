#!/usr/bin/env python3
"""Measure whether the LLM planner lane decides anything the deterministic one does not.

The original recommendation was to switch the fleet default to
``--planner deterministic`` and measure unique area added per GPU-hour against
the LLM lane on the same cells.  That comparison was never run; a
``cost-aware-v2`` router was built instead, which sends obvious packets to the
deterministic planner at zero cost.  That serves the intent partially but never
answers the question.

This harness answers the *decision* half of the question, which needs neither
CT nor VC3D: given a frozen set of real ``PLANNER_PACKET.json`` files, run the
deterministic planner over each packet and compare its proposal against the
proposal the model lane actually produced for that same packet, recorded in the
run directory.  It reports, per packet and in aggregate:

* whether the two lanes selected the same seed;
* whether they selected the same profile and the same complete parameter map;
* how large the packet's legal decision space actually was, so that a match
  that the fail-closed validator *forced* is never counted as a model and a
  deterministic planner independently agreeing;
* the recorded provider cost of the model lane, where a cost receipt exists.

Explicitly out of scope: unique area added per GPU-hour.  That requires a
matched CT + VC3D grow run on both lanes and cannot be recovered from planner
receipts.  This harness measures the planning decision only.

The harness contacts no provider.  Model-lane proposals are read from
immutable historical receipts; only the deterministic planner is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from helena_fleet_backlog_support import (  # noqa: E402  (sibling stage module)
    canonical_hash,
    framework_version,
    load_json,
    repository_root,
    utc_now,
    write_json_atomic,
)
from fleet.planner import (  # noqa: E402  (resolved by helena_fleet_backlog_support)
    DeterministicPlanner,
    PlannerScientificViolation,
    validate_and_lock,
)


PACKET_NAME = "PLANNER_PACKET.json"
PROPOSAL_NAMES = ("SEGMENTATION_PROPOSAL.json", "PROPOSAL.json")

# A model lane is only credited when the run directory carries an artefact that
# a provider call produced.  Everything else is UNATTRIBUTED and is reported
# separately rather than being silently counted as agreement.
MODEL_LANE_EVIDENCE = {
    "OPENCODE": ("opencode.stdout.log", "OPENCODE_PARSE_RECEIPT.json"),
    "OPENROUTER_FUSION": ("fusion-call", "FUSION_COST_RECEIPT.json", "FUSION_RESPONSE.json"),
    "OPENROUTER_DIRECT": ("DIRECT_COST_RECEIPT.json", "DIRECT_RESPONSE.json"),
}
DETERMINISTIC_LANE_EVIDENCE = ("deterministic-fallback", "DETERMINISTIC_FALLBACK_RECEIPT.json")

COST_RECEIPTS = (
    ("fusion-call/FUSION_COST_RECEIPT.json", "total_cost_usd"),
    ("FUSION_COST_RECEIPT.json", "total_cost_usd"),
    ("DIRECT_COST_RECEIPT.json", "total_cost_usd"),
)
RECONCILIATION_RECEIPT = "COST_RECONCILIATION.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_packets(roots: list[Path]) -> list[Path]:
    """Return every real planner packet below ``roots``, in a stable order.

    The OpenCode sandbox copy is a byte-identical duplicate of its parent
    packet made so the agent could read it without repository access.  Counting
    it would double one cell and inflate the denominator.
    """

    found: dict[Path, None] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(PACKET_NAME)):
            if path.parent.name == "opencode-planner-sandbox":
                continue
            found[path.resolve()] = None
    return list(found)


def parameter_space_size(envelope: dict[str, Any]) -> int | None:
    """Count the legal parameter tuples, or ``None`` when unbounded."""

    total = 1
    for rule in envelope["parameters"].values():
        if "const" in rule:
            size = 1
        elif "enum" in rule:
            size = len(rule["enum"])
        elif rule.get("type") == "integer":
            size = int(rule["maximum"]) - int(rule["minimum"]) + 1
        elif rule.get("type") == "boolean":
            size = 2
        elif rule.get("type") == "number":
            minimum = rule.get("minimum")
            maximum = rule.get("maximum")
            if minimum is None or maximum is None:
                return None
            size = 1 if float(minimum) == float(maximum) else None
            if size is None:
                return None
        else:
            return None
        if size < 1:
            raise RuntimeError("parameter envelope declares an empty value set")
        total *= size
    return total * max(len(envelope["profile_ids"]), 1)


def allowed_seed_count(packet: dict[str, Any]) -> int:
    """How many seeds the fail-closed validator would actually accept."""

    if packet.get("candidate_selection_policy") == "score-cell-volume-clearance-v1":
        # The validator recomputes the frozen argmax and rejects anything else,
        # so the model lane has exactly one legal seed regardless of its prose.
        return 1
    return len(packet["candidate_seeds"])


def classify_lane(run_dir: Path) -> dict[str, Any]:
    lanes = [
        name
        for name, markers in MODEL_LANE_EVIDENCE.items()
        if any((run_dir / marker).exists() for marker in markers)
    ]
    deterministic_fallback = any(
        (run_dir / marker).exists() for marker in DETERMINISTIC_LANE_EVIDENCE
    )
    if deterministic_fallback:
        return {"lane": "DETERMINISTIC_FALLBACK", "evidence": lanes, "comparable": False}
    if lanes:
        return {"lane": "+".join(sorted(lanes)), "evidence": lanes, "comparable": True}
    return {"lane": "UNATTRIBUTED", "evidence": [], "comparable": False}


def recorded_proposal(run_dir: Path, packet: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, str | None]:
    for name in PROPOSAL_NAMES:
        path = run_dir / name
        if not path.is_file():
            continue
        proposal = load_json(path)
        if proposal.get("task_id") != packet.get("task_id") or proposal.get(
            "attempt_id"
        ) != packet.get("attempt_id"):
            return None, name, "IDENTITY_MISMATCH"
        return proposal, name, None
    return None, None, "NO_RECORDED_PROPOSAL"


def recorded_cost(run_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for relative, key in COST_RECEIPTS:
        path = run_dir / relative
        if not path.is_file():
            continue
        receipt = load_json(path)
        entries.append(
            {
                "receipt": relative,
                "schema": receipt.get("schema"),
                "provider_reported_cost_usd": receipt.get(key),
                "cost_scope": receipt.get("cost_scope"),
                "cost_complete": receipt.get("cost_complete"),
            }
        )
    reconciliation = run_dir / RECONCILIATION_RECEIPT
    budget: float | None = None
    if reconciliation.is_file():
        receipt = load_json(reconciliation)
        value = receipt.get("cost_to_use_for_budgeting_usd")
        budget = float(value) if isinstance(value, (int, float)) else None
        entries.append(
            {
                "receipt": RECONCILIATION_RECEIPT,
                "schema": receipt.get("schema"),
                "provider_reported_cost_usd": receipt.get(
                    "provider_parent_response_cost_usd"
                ),
                "cost_scope": receipt.get("status"),
                "cost_complete": False,
            }
        )
    return {"receipts": entries, "operator_budget_cost_usd": budget}


def deterministic_proposal(packet: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    contract = "v2" if str(packet.get("schema", "")).endswith(".v2") else "v1"
    with tempfile.TemporaryDirectory() as scratch:
        try:
            return (
                DeterministicPlanner(contract_version=contract).propose(
                    packet, Path(scratch)
                ),
                None,
            )
        except (PlannerScientificViolation, RuntimeError, ValueError, KeyError) as error:
            return None, f"{type(error).__name__}: {error}"


def validation_status(packet: dict[str, Any], proposal: dict[str, Any] | None) -> str | None:
    if proposal is None:
        return None
    try:
        validate_and_lock(packet, proposal)
    except Exception as error:  # noqa: BLE001 - the status string is the product
        return f"{type(error).__name__}: {str(error)[:200]}"
    return "VALIDATED"


def compare_packet(path: Path, root: Path) -> dict[str, Any]:
    packet = load_json(path)
    run_dir = path.parent
    envelope = packet["parameter_envelope"]
    determinist, deterministic_error = deterministic_proposal(packet)
    model, proposal_name, proposal_error = recorded_proposal(run_dir, packet)
    lane = classify_lane(run_dir)
    seeds_allowed = allowed_seed_count(packet)
    parameter_space = parameter_space_size(envelope)
    decision_space = None if parameter_space is None else seeds_allowed * parameter_space

    same_seed: bool | None = None
    same_parameters: bool | None = None
    same_profile: bool | None = None
    if determinist is not None and model is not None:
        same_seed = determinist["selected_seed"] == model.get("selected_seed")
        same_profile = determinist["profile_id"] == model.get("profile_id")
        same_parameters = determinist["parameters"] == model.get("parameters")

    comparable = bool(lane["comparable"] and determinist is not None and model is not None)
    return {
        "packet_path": str(path.relative_to(root)),
        "packet_sha256": file_sha256(path),
        "packet_schema": packet.get("schema"),
        "sample_id": packet.get("sample_id"),
        "cell_id": packet.get("cell", {}).get("cell_id"),
        "task_id": packet.get("task_id"),
        "attempt_id": packet.get("attempt_id"),
        "candidate_selection_policy": packet.get("candidate_selection_policy"),
        "candidate_seed_count": len(packet["candidate_seeds"]),
        "regional_history_count": len(
            packet.get("regional_attempt_history", {}).get("attempts", [])
        ),
        "decision_space": {
            "validator_allowed_seed_count": seeds_allowed,
            "seed_choice_is_free": seeds_allowed > 1,
            "parameter_and_profile_tuple_count": parameter_space,
            "total_legal_proposal_count": decision_space,
            "seed_forced_by": (
                "CANDIDATE_SELECTION_POLICY"
                if packet.get("candidate_selection_policy")
                == "score-cell-volume-clearance-v1"
                else "SINGLE_LISTED_CANDIDATE"
                if len(packet["candidate_seeds"]) == 1
                else None
            ),
        },
        "deterministic": {
            "error": deterministic_error,
            "selected_seed": None if determinist is None else determinist["selected_seed"],
            "profile_id": None if determinist is None else determinist["profile_id"],
            "parameters": None if determinist is None else determinist["parameters"],
            "validation": validation_status(packet, determinist),
            "provider_calls": 0,
        },
        "model_lane": {
            "lane": lane["lane"],
            "lane_evidence": lane["evidence"],
            "recorded_proposal_file": proposal_name,
            "error": proposal_error,
            "selected_seed": None if model is None else model.get("selected_seed"),
            "profile_id": None if model is None else model.get("profile_id"),
            "parameters": None if model is None else model.get("parameters"),
            "validation": validation_status(packet, model),
            "cost": recorded_cost(run_dir),
        },
        "comparable": comparable,
        "same_seed": same_seed,
        "same_profile": same_profile,
        "same_parameters": same_parameters,
        "same_decision": (
            None
            if not comparable
            else bool(same_seed and same_profile and same_parameters)
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [row for row in rows if row["comparable"]]
    free_seed = [row for row in comparable if row["decision_space"]["seed_choice_is_free"]]
    identical = [row for row in comparable if row["same_decision"]]
    seed_matches = [row for row in comparable if row["same_seed"]]
    parameter_matches = [
        row for row in comparable if row["same_parameters"] and row["same_profile"]
    ]
    costs = [
        row["model_lane"]["cost"]["operator_budget_cost_usd"]
        for row in comparable
        if row["model_lane"]["cost"]["operator_budget_cost_usd"] is not None
    ]
    return {
        "packet_count": len(rows),
        "comparable_packet_count": len(comparable),
        "excluded_packet_count": len(rows) - len(comparable),
        "identical_seed_and_parameters_count": len(identical),
        "identical_seed_and_parameters_fraction": (
            len(identical) / len(comparable) if comparable else None
        ),
        "identical_seed_count": len(seed_matches),
        "identical_profile_and_parameters_count": len(parameter_matches),
        "identical_profile_and_parameters_fraction": (
            len(parameter_matches) / len(comparable) if comparable else None
        ),
        "free_seed_choice_packet_count": len(free_seed),
        "identical_seed_where_choice_was_free_count": sum(
            1 for row in free_seed if row["same_seed"]
        ),
        "minimum_parameter_tuple_space": min(
            (
                row["decision_space"]["parameter_and_profile_tuple_count"]
                for row in comparable
                if row["decision_space"]["parameter_and_profile_tuple_count"] is not None
            ),
            default=None,
        ),
        "deterministic_provider_call_count": 0,
        "model_lane_reconciled_cost_usd": sum(costs) if costs else None,
        "model_lane_reconciled_cost_packet_count": len(costs),
    }


def build_report(roots: list[Path], root: Path) -> dict[str, Any]:
    packets = discover_packets(roots)
    if not packets:
        raise RuntimeError("no PLANNER_PACKET.json found under the requested roots")
    rows = [compare_packet(path, root) for path in packets]
    rows.sort(key=lambda row: row["packet_path"])
    totals = summarize(rows)
    packet_set = [
        {"packet_path": row["packet_path"], "packet_sha256": row["packet_sha256"]}
        for row in rows
    ]
    return {
        "schema": "campaignx.planner_lane_comparison.v1",
        "generated_at_utc": utc_now(),
        "framework_version": framework_version(root),
        "question": (
            "On real planner packets, do the deterministic planner and the LLM "
            "lane choose the same seed and the same parameters?"
        ),
        "method": [
            "the frozen packet set is every non-sandbox PLANNER_PACKET.json under the roots",
            "the deterministic planner is executed locally on each packet",
            "the model-lane proposal is read from the immutable run receipt, never re-requested",
            "a packet counts as comparable only when a provider artefact proves a model lane ran",
            "decision_space records how much freedom the fail-closed validator actually left",
        ],
        "packet_set": {
            "packet_count": len(packet_set),
            "packet_set_sha256": canonical_hash(packet_set),
            "packets": packet_set,
        },
        "totals": totals,
        "rows": rows,
        "ink_used": False,
        "out_of_scope": [
            "unique area added per GPU-hour: needs a matched CT + VC3D grow run on both "
            "lanes and cannot be recovered from planner receipts",
            "surface quality, duplication, or downstream QC outcome of either lane",
        ],
        "non_claims": [
            "planner agreement is not evidence about ink, text, or First Letters",
            "a matching decision is not proof that either lane chose well",
            "agreement forced by the fail-closed validator is not independent agreement; "
            "read decision_space before reading the fraction",
            "the measured cost covers only decisions that left a provider cost receipt "
            "and is not a reconciled campaign total",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    root = repository_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet-root",
        type=Path,
        action="append",
        default=None,
        help="directory searched recursively for PLANNER_PACKET.json (repeatable)",
    )
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    roots = [
        (path if path.is_absolute() else root / path)
        for path in (args.packet_root or [Path("workspace")])
    ]
    report = build_report(roots, root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "PLANNER_LANE_COMPARISON.json", report)
    totals = report["totals"]
    print(
        f"comparable packets: {totals['comparable_packet_count']}/"
        f"{totals['packet_count']}; identical seed and parameters: "
        f"{totals['identical_seed_and_parameters_count']}; identical parameters: "
        f"{totals['identical_profile_and_parameters_count']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
