#!/usr/bin/env python3
"""Compute marginal segmentation yield per roll and recommend effort allocation.

The campaign has no stopping rule.  Every failure routes to "this is not an
absence", which is epistemically correct and operationally paralysing: a roll
that has never produced geometry this pipeline can check keeps occupying a GPU
slot forever.

This script measures, per roll, from evidence that already exists:

* grow attempts and measured area added per attempt;
* ``NO_SEED`` rate by cause, where the rejection diagnostics were materialized;
* the ``POST_FIT_RELATION_GUARD`` passed/failed relation split, where an
  evaluation exists;

and applies a frozen, ink-blind effort-allocation policy to emit ``CONTINUE``,
``PAUSE_PENDING_REVIEW``, or ``NEEDS_NEW_APPROACH`` per roll.

Hard constraint, enforced by the policy profile, by this code, and by the
tests: the recommendation is about **where to spend the next GPU slot**.  It is
never a statement about what the roll contains.  A pause does not say there is
no ink.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
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
from helena_build_recipe_yield import collect_attempts  # noqa: E402


POLICY_RELATIVE = Path(
    "framework/profiles/01-segmentation/segmentation-effort-allocation-policy-v1-1.0.0.json"
)
GUARD_ARTEFACT = "AB_EVALUATION.json"
SEED_SCREEN = "SEED_SCREEN.json"
REJECTION_SCHEMA = "campaignx.seed_candidate_rejection_diagnostics.v1"
DRAIN_SCHEMA = "campaignx.vast_drain_final_receipt.v1"

CONTINUE = "CONTINUE"
PAUSE = "PAUSE_PENDING_REVIEW"
NEEDS_NEW_APPROACH = "NEEDS_NEW_APPROACH"


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_policy(root: Path, override: Path | None) -> dict[str, Any]:
    path = override or (root / POLICY_RELATIVE)
    policy = load_json(path)
    if policy.get("schema") != "campaignx.effort_allocation_policy.v1":
        raise RuntimeError(f"not an effort-allocation policy profile: {path}")
    if policy.get("ink_blind") is not True:
        raise RuntimeError("effort-allocation policy must be ink-blind")
    return policy


def collect_guard_evaluations(roots: list[Path], root: Path) -> dict[str, list[dict[str, Any]]]:
    """Every POST_FIT_RELATION_GUARD split found, keyed by sample."""

    found: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for search_root in roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob(GUARD_ARTEFACT)):
            payload = load_json(path)
            sample = payload.get("sample_id")
            guard = payload.get("guard")
            if not sample or not isinstance(guard, dict):
                continue
            passed = guard.get("passed_count")
            failed = guard.get("failed_count")
            if not isinstance(passed, int) or not isinstance(failed, int):
                continue
            key = relative(path.resolve(), root)
            found[str(sample)][key] = {
                "evaluation_path": key,
                "generated_at_utc": payload.get("generated_at_utc"),
                "evaluation_status": payload.get("status"),
                "scope": payload.get("scope"),
                "guard_passed_count": passed,
                "guard_failed_count": failed,
                "guard_relation_count": passed + failed,
                "guard_pass_rate": (passed / (passed + failed)) if passed + failed else None,
            }
    return {
        sample: sorted(
            rows.values(),
            key=lambda row: (str(row["generated_at_utc"]), row["evaluation_path"]),
        )
        for sample, rows in found.items()
    }


def collect_no_seed_evidence(roots: list[Path], root: Path) -> dict[str, Any]:
    """The NO_SEED cause breakdown, and an explicit note where it is missing."""

    diagnostics: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    screens_without_breakdown = 0
    for search_root in roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob(SEED_SCREEN)):
            payload = load_json(path)
            row = payload.get("rejection_diagnostics")
            if not isinstance(row, dict) or row.get("schema") != REJECTION_SCHEMA:
                screens_without_breakdown += 1
                continue
            task = path.parent / "CLAIMED_TASK.json"
            diagnostics.append(
                {
                    "screen_path": relative(path.resolve(), root),
                    "sample_id": (
                        load_json(task).get("sample_id") if task.is_file() else None
                    ),
                    "raw_candidate_count": payload.get("raw_candidate_count"),
                    "eligible_candidate_count": payload.get("eligible_candidate_count"),
                    "rejected_candidate_count": row.get("rejected_candidate_count"),
                    "rejection_counts": row.get("rejection_counts"),
                }
            )
        for path in sorted(search_root.rglob("*.json")):
            if path.name != "VAST_DRAIN_FINAL_RECEIPT.json":
                continue
            payload = load_json(path)
            if payload.get("schema") != DRAIN_SCHEMA:
                continue
            aggregate.append(
                {
                    "receipt_path": relative(path.resolve(), root),
                    "run_id": payload.get("run_id"),
                    "task_counts": payload.get("task_counts"),
                }
            )
    return {
        "cause_breakdown_receipts": diagnostics,
        "cause_breakdown_receipt_count": len(diagnostics),
        "seed_screens_without_cause_breakdown": screens_without_breakdown,
        "aggregate_outcome_receipts": aggregate,
        "coverage_note": (
            "campaignx.seed_candidate_rejection_diagnostics.v1 postdates most historical "
            "runs, so a per-roll NO_SEED rate by cause is UNAVAILABLE wherever no such "
            "receipt exists; UNAVAILABLE is never scored as zero"
        ),
    }


def catalog_rows(database: Path) -> dict[str, dict[str, Any]]:
    if not database.is_file():
        return {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        surfaces = connection.execute(
            "SELECT sample_id, area_cm2 FROM campaign_surfaces"
        ).fetchall()
        try:
            overlaps = connection.execute(
                "SELECT campaign_surface_id FROM campaign_public_aabb_overlap_warnings"
            ).fetchall()
        except sqlite3.Error:
            overlaps = []
    per_sample: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"surface_count": 0, "gross_area_cm2": 0.0, "public_overlap_surface_count": 0}
    )
    for sample, area in surfaces:
        entry = per_sample[str(sample)]
        entry["surface_count"] += 1
        entry["gross_area_cm2"] += float(area)
    overlapping = {str(row[0]) for row in overlaps}
    for surface_id in overlapping:
        parts = surface_id.split(":")
        if len(parts) >= 2:
            per_sample[parts[1]]["public_overlap_surface_count"] += 1
    return dict(per_sample)


def recommend(
    policy: dict[str, Any],
    *,
    attempt_count: int,
    guard: dict[str, Any] | None,
    area_per_attempt: float | None,
) -> dict[str, Any]:
    thresholds = policy["thresholds"]
    causes: list[str] = []
    if attempt_count < int(thresholds["minimum_attempts_before_any_pause"]):
        return {
            "recommendation": CONTINUE,
            "primary_cause": "INSUFFICIENT_EFFORT_SPENT",
            "causes": ["INSUFFICIENT_EFFORT_SPENT"],
        }
    if guard is None:
        return {
            "recommendation": NEEDS_NEW_APPROACH,
            "primary_cause": "NO_RELATION_GUARD_EVALUATION",
            "causes": ["NO_RELATION_GUARD_EVALUATION"],
        }
    if guard["guard_relation_count"] < int(thresholds["minimum_guard_relation_count"]):
        return {
            "recommendation": PAUSE,
            "primary_cause": "GUARD_EVIDENCE_TOO_THIN",
            "causes": ["GUARD_EVIDENCE_TOO_THIN"],
        }
    rate = guard["guard_pass_rate"]
    if rate is not None and rate < float(
        thresholds["needs_new_approach_maximum_guard_pass_rate"]
    ):
        return {
            "recommendation": NEEDS_NEW_APPROACH,
            "primary_cause": "GUARD_PASS_RATE_BELOW_NEW_APPROACH_FLOOR",
            "causes": ["GUARD_PASS_RATE_BELOW_NEW_APPROACH_FLOOR"],
        }
    if rate is not None and rate < float(thresholds["continue_minimum_guard_pass_rate"]):
        causes.append("GUARD_PASS_RATE_BELOW_CONTINUE_FLOOR")
    if area_per_attempt is not None and area_per_attempt < float(
        thresholds["minimum_unique_area_per_attempt_cm2"]
    ):
        causes.append("MARGINAL_AREA_BELOW_FLOOR")
    if causes:
        return {"recommendation": PAUSE, "primary_cause": causes[0], "causes": causes}
    return {"recommendation": CONTINUE, "primary_cause": "CONTINUE", "causes": ["CONTINUE"]}


def build_report(
    roots: list[Path], root: Path, database: Path, policy: dict[str, Any]
) -> dict[str, Any]:
    attempts = collect_attempts(roots, root)
    guards = collect_guard_evaluations(roots, root)
    catalog = catalog_rows(database)
    no_seed = collect_no_seed_evidence(roots, root)
    no_seed_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in no_seed["cause_breakdown_receipts"]:
        if row["sample_id"]:
            no_seed_by_sample[str(row["sample_id"])].append(row)

    per_sample_attempts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        if row["sample_id"]:
            per_sample_attempts[str(row["sample_id"])].append(row)

    samples = sorted(set(per_sample_attempts) | set(catalog) | set(guards))
    if not samples:
        raise RuntimeError("no roll evidence found under the requested roots")

    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_attempts = per_sample_attempts.get(sample, [])
        attempt_count = len(sample_attempts)
        measured = [
            row["area_cm2"] for row in sample_attempts if row["area_cm2"] is not None
        ]
        catalogued = catalog.get(sample, {})
        gross_area = catalogued.get("gross_area_cm2")
        area_per_attempt = (
            (gross_area / attempt_count)
            if gross_area is not None and attempt_count
            else (sum(measured) / attempt_count if measured and attempt_count else None)
        )
        history = guards.get(sample, [])
        latest = history[-1] if history else None
        decision = recommend(
            policy,
            attempt_count=attempt_count,
            guard=latest,
            area_per_attempt=area_per_attempt,
        )
        rows.append(
            {
                "sample_id": sample,
                "attempt_count": attempt_count,
                "measured_attempt_count": len(measured),
                "catalogued_surface_count": catalogued.get("surface_count"),
                "catalogued_gross_area_cm2": gross_area,
                "surface_retention_fraction": (
                    catalogued["surface_count"] / attempt_count
                    if catalogued.get("surface_count") is not None and attempt_count
                    else None
                ),
                "area_per_attempt_cm2": area_per_attempt,
                "public_overlap_surface_count": catalogued.get(
                    "public_overlap_surface_count"
                ),
                "relation_guard": latest,
                "relation_guard_evaluation_count": len(history),
                "relation_guard_history": history,
                "no_seed_cause_breakdown": no_seed_by_sample.get(sample, "UNAVAILABLE"),
                **decision,
            }
        )

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["recommendation"]] += 1
    return {
        "schema": "campaignx.campaign_budget_review.v1",
        "generated_at_utc": utc_now(),
        "framework_version": framework_version(root),
        "decides": policy["decides"],
        "never_decides": policy["never_decides"],
        "policy_profile_id": policy["profile_id"],
        "policy_sha256": canonical_hash(policy),
        "policy_thresholds": policy["thresholds"],
        "totals": {
            "roll_count": len(rows),
            "recommendation_counts": dict(sorted(counts.items())),
            "attempt_count": sum(row["attempt_count"] for row in rows),
            "rolls_without_relation_guard_evaluation": sorted(
                row["sample_id"] for row in rows if row["relation_guard"] is None
            ),
        },
        "rolls": rows,
        "no_seed_evidence": no_seed,
        "ink_used": False,
        "out_of_scope": [
            "retiring a roll, deleting its evidence, or closing its scientific question",
            "GPU-hour accounting: the receipts record no wall-clock or device time",
        ],
        "non_claims": list(policy["non_claims"])
        + [
            "this review reallocates effort; it does not reclassify any historical result",
            "an UNAVAILABLE NO_SEED cause breakdown is missing evidence, not a zero rate",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    root = repository_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        action="append",
        default=None,
        help="directory searched recursively for roll evidence (repeatable)",
    )
    parser.add_argument(
        "--catalog-database",
        type=Path,
        default=Path(
            "workspace/catalog/geometry_surface_catalog_v4/GEOMETRY_SURFACE_CATALOG.sqlite"
        ),
    )
    parser.add_argument("--policy-profile", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    roots = [
        (path if path.is_absolute() else root / path)
        for path in (args.evidence_root or [Path("workspace")])
    ]
    database = (
        args.catalog_database
        if args.catalog_database.is_absolute()
        else root / args.catalog_database
    )
    override = None
    if args.policy_profile is not None:
        override = (
            args.policy_profile
            if args.policy_profile.is_absolute()
            else root / args.policy_profile
        )
    policy = load_policy(root, override)
    report = build_report(roots, root, database, policy)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "CAMPAIGN_BUDGET_REVIEW.json", report)
    print(
        f"rolls: {report['totals']['roll_count']}; "
        + json.dumps(report["totals"]["recommendation_counts"], sort_keys=True)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
