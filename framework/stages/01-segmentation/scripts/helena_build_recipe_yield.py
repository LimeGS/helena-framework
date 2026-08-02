#!/usr/bin/env python3
"""Aggregate measured surface area by segmentation recipe.

Planner v2 receives up to twelve regional *failures* (``NO_SEED``,
``GROW_FAILED``, ``DUPLICATE_SURFACE``) and never a success or an obtained
area.  It can therefore avoid repeating a failed recipe but cannot move toward
one that works: a taboo list, not adaptive search.  The missing input is a
yield table.

This script builds that table from evidence that already exists — geometry
recovery growth receipts, fleet growth/finalization receipts, and the geometry
surface catalogue — grouped by ``(seed_region_policy, generations,
step_size)``.  It deliberately does **not** change the planner v2 contract to
consume the table; wiring a yield signal into experiment selection is a
scientific change that needs its own decision.  The product here is the
measurement plus an explicit verdict on whether the data can distinguish
recipes at all.

Area in this corpus is dominated by the fixed output grid and the per-roll
voxel size, so the report stratifies by ``voxel_size_um`` and reports the
within-stratum spread next to the between-recipe spread.  A verdict of
``INSUFFICIENT_TO_DISCRIMINATE`` is a real result and is reported with numbers
rather than replaced by an invented policy.
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
    mean,
    population_stdev,
    repository_root,
    utc_now,
    write_json_atomic,
)


ARCHIVE_RECEIPT_KIND = "campaign_x_phase4_geometry_recovery_v1_growth_receipt"
FLEET_RECEIPT_SCHEMA_PREFIX = "campaignx.segment_fleet_growth_receipt"

# ``worker.py`` reads ``candidate_discovery.seed_region_policy`` with this
# default, so an attempt that omits the key genuinely ran under ``fixed-v1``.
FLEET_DEFAULT_SEED_REGION_POLICY = "fixed-v1"
UNDECLARED_SEED_REGION_POLICY = "UNDECLARED"

# A group needs at least this many measured attempts before its mean is
# compared with another group's.  Two observations of a constant-area grow are
# not a yield estimate.
MINIMUM_COMPARABLE_ATTEMPTS = 5


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def sibling_area(receipt_path: Path) -> tuple[float | None, str | None]:
    """Fleet growth receipts carry no area; the finalizer measures it later."""

    directory = receipt_path.parent
    for name, key in (
        ("ARTIFACT_SET.json", "area_cm2"),
        ("FINALIZATION_RECEIPT.json", None),
    ):
        path = directory / name
        if not path.is_file():
            continue
        payload = load_json(path)
        value = (
            payload.get(key)
            if key is not None
            else payload.get("surface", {}).get("area_cm2")
        )
        if isinstance(value, (int, float)):
            return float(value), name
    return None, None


def fleet_seed_region_policy(receipt_path: Path) -> tuple[str, str]:
    task_path = receipt_path.parent / "CLAIMED_TASK.json"
    if not task_path.is_file():
        return UNDECLARED_SEED_REGION_POLICY, "NO_CLAIMED_TASK"
    discovery = load_json(task_path).get("candidate_discovery") or {}
    declared = discovery.get("seed_region_policy")
    if declared is None:
        return FLEET_DEFAULT_SEED_REGION_POLICY, "WORKER_DEFAULT"
    return str(declared), "CLAIMED_TASK"


def fleet_sample_id(receipt_path: Path) -> str | None:
    for name in ("CLAIMED_TASK.json", "SEGMENTATION_PLAN.json"):
        path = receipt_path.parent / name
        if path.is_file():
            value = load_json(path).get("sample_id")
            if value:
                return str(value)
    return None


def archive_sample_id(receipt: dict[str, Any], receipt_path: Path) -> str | None:
    archived = (receipt.get("archived_surface") or {}).get("path")
    if isinstance(archived, str):
        parts = [part for part in archived.split("/") if part]
        if len(parts) >= 2:
            return parts[-2]
    parts = receipt_path.parts
    for index, part in enumerate(parts):
        if part == "surfaces" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def collect_attempts(roots: list[Path], root: Path) -> list[dict[str, Any]]:
    """One row per grow attempt that left a growth receipt."""

    attempts: dict[str, dict[str, Any]] = {}
    for search_root in roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob("GROWTH_RECEIPT.json")):
            receipt = load_json(path)
            profile = receipt.get("profile") or {}
            generations = profile.get("generations")
            step_size = profile.get("step_size")
            voxel_size = profile.get("voxelsize")
            key = relative(path.resolve(), root)
            if receipt.get("kind") == ARCHIVE_RECEIPT_KIND:
                area = receipt.get("area_cm2")
                row = {
                    "receipt_path": key,
                    "source": "GEOMETRY_RECOVERY_ARCHIVE",
                    "sample_id": archive_sample_id(receipt, path),
                    "seed_region_policy": receipt.get("plan_query_scope")
                    or UNDECLARED_SEED_REGION_POLICY,
                    "seed_region_policy_source": (
                        "PLAN_QUERY_SCOPE"
                        if receipt.get("plan_query_scope")
                        else "UNDECLARED"
                    ),
                    "generations": generations,
                    "step_size": step_size,
                    "voxel_size_um": voxel_size,
                    "status": receipt.get("status"),
                    "area_cm2": float(area) if isinstance(area, (int, float)) else None,
                    "area_source": "GROWTH_RECEIPT" if isinstance(area, (int, float)) else None,
                }
            elif str(receipt.get("schema", "")).startswith(FLEET_RECEIPT_SCHEMA_PREFIX):
                area, area_source = sibling_area(path)
                policy, policy_source = fleet_seed_region_policy(path)
                status = receipt.get("status")
                if status is None:
                    status = "GROW_SUCCEEDED" if receipt.get("exit_code") == 0 else "GROW_FAILED"
                row = {
                    "receipt_path": key,
                    "source": "SEGMENT_FLEET",
                    "sample_id": fleet_sample_id(path),
                    "seed_region_policy": policy,
                    "seed_region_policy_source": policy_source,
                    "generations": generations,
                    "step_size": step_size,
                    "voxel_size_um": voxel_size,
                    "status": status,
                    "area_cm2": area,
                    "area_source": area_source,
                }
            else:
                continue
            attempts[key] = row
    return sorted(attempts.values(), key=lambda row: row["receipt_path"])


def catalog_cross_check(database: Path) -> dict[str, Any] | None:
    if not database.is_file():
        return None
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT sample_id, area_cm2, profile_json FROM campaign_surfaces"
        ).fetchall()
    recipes: dict[str, int] = defaultdict(int)
    areas: list[float] = []
    for _sample, area, profile_json in rows:
        profile = json.loads(profile_json) if profile_json else {}
        recipes[
            f"generations={profile.get('generations')},step_size={profile.get('step_size')}"
        ] += 1
        areas.append(float(area))
    return {
        "database": str(database.name),
        "catalogued_surface_count": len(rows),
        "catalogued_gross_area_cm2": sum(areas),
        "distinct_grow_parameter_cells": dict(sorted(recipes.items())),
        "distinct_area_value_count": len({round(value, 3) for value in areas}),
    }


def group_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        groups[(row["seed_region_policy"], row["generations"], row["step_size"])].append(row)
    output: list[dict[str, Any]] = []
    for (policy, generations, step_size), rows in groups.items():
        measured = [row for row in rows if row["area_cm2"] is not None]
        areas = [row["area_cm2"] for row in measured]
        strata: dict[Any, list[float]] = defaultdict(list)
        for row in measured:
            strata[row["voxel_size_um"]].append(row["area_cm2"])
        voxel_strata = []
        for voxel, values in sorted(strata.items(), key=lambda item: str(item[0])):
            group_mean = mean(values)
            voxel_strata.append(
                {
                    "voxel_size_um": voxel,
                    "measured_attempt_count": len(values),
                    "mean_area_cm2": group_mean,
                    "minimum_area_cm2": min(values),
                    "maximum_area_cm2": max(values),
                    "relative_range": (
                        (max(values) - min(values)) / group_mean
                        if group_mean
                        else None
                    ),
                }
            )
        status_counts: dict[str, int] = defaultdict(int)
        source_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            status_counts[str(row["status"])] += 1
            source_counts[str(row["source"])] += 1
        output.append(
            {
                "recipe": {
                    "seed_region_policy": policy,
                    "generations": generations,
                    "step_size": step_size,
                },
                "recipe_key": f"{policy}|generations={generations}|step_size={step_size}",
                "execution_harness_counts": dict(sorted(source_counts.items())),
                "attempt_count": len(rows),
                "measured_attempt_count": len(measured),
                "unmeasured_attempt_count": len(rows) - len(measured),
                "status_counts": dict(sorted(status_counts.items())),
                "sample_ids": sorted({row["sample_id"] for row in rows if row["sample_id"]}),
                "total_area_cm2": sum(areas) if areas else None,
                "mean_area_cm2": mean(areas),
                "stdev_area_cm2": population_stdev(areas),
                "minimum_area_cm2": min(areas) if areas else None,
                "maximum_area_cm2": max(areas) if areas else None,
                "area_per_attempt_cm2": (sum(areas) / len(rows)) if areas else None,
                "voxel_size_strata": voxel_strata,
                "comparable": len(measured) >= MINIMUM_COMPARABLE_ATTEMPTS,
            }
        )
    return sorted(output, key=lambda row: row["recipe_key"])


def harness_of(group: dict[str, Any]) -> str | None:
    """The single execution harness behind a group, or ``None`` when mixed."""

    harnesses = [
        name for name, count in group["execution_harness_counts"].items() if count
    ]
    return harnesses[0] if len(harnesses) == 1 else None


def within_harness_comparisons(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare recipes only inside one execution harness.

    Across harnesses the seed-region policy is perfectly collinear with which
    code grew the surface, so a between-harness difference cannot be read as a
    recipe effect at all.  The only interpretable comparison is between two
    policies executed by the same harness.
    """

    by_harness: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        harness = harness_of(group)
        if harness is not None and group["comparable"]:
            by_harness[harness].append(group)
    comparisons: list[dict[str, Any]] = []
    for harness, members in sorted(by_harness.items()):
        means = [group["mean_area_cm2"] for group in members]
        stdevs = [
            group["stdev_area_cm2"]
            for group in members
            if group["stdev_area_cm2"] is not None
        ]
        between = (max(means) - min(means)) if len(means) >= 2 else None
        pooled = mean(stdevs) if stdevs else None
        if len(members) < 2:
            verdict = "SINGLE_COMPARABLE_RECIPE"
        elif between is None or pooled is None or between <= pooled:
            verdict = "INSUFFICIENT_TO_DISCRIMINATE"
        else:
            verdict = "SEPARATION_OBSERVED_NOT_CAUSAL"
        comparisons.append(
            {
                "execution_harness": harness,
                "comparable_recipe_keys": sorted(
                    group["recipe_key"] for group in members
                ),
                "comparable_recipe_count": len(members),
                "between_recipe_mean_spread_cm2": between,
                "pooled_within_recipe_stdev_cm2": pooled,
                "verdict": verdict,
                "dominant_recipe": None,
            }
        )
    return comparisons


def assess(groups: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [group for group in groups if group["comparable"]]
    means = [group["mean_area_cm2"] for group in comparable]
    stdevs = [
        group["stdev_area_cm2"] for group in comparable if group["stdev_area_cm2"] is not None
    ]
    between = (max(means) - min(means)) if len(means) >= 2 else None
    pooled = mean(stdevs) if stdevs else None
    grow_parameter_cells = {
        (group["recipe"]["generations"], group["recipe"]["step_size"]) for group in groups
    }
    stratified_ranges = [
        stratum["relative_range"]
        for group in comparable
        for stratum in group["voxel_size_strata"]
        if stratum["relative_range"] is not None and stratum["measured_attempt_count"] >= 2
    ]
    if len(comparable) < 2:
        verdict = "INSUFFICIENT_TO_DISCRIMINATE"
        reason = (
            f"only {len(comparable)} recipe group reaches the "
            f"{MINIMUM_COMPARABLE_ATTEMPTS}-attempt comparability floor"
        )
    elif between is None or pooled is None:
        verdict = "INSUFFICIENT_TO_DISCRIMINATE"
        reason = "no group has enough measured attempts to estimate spread"
    elif between <= pooled:
        verdict = "INSUFFICIENT_TO_DISCRIMINATE"
        reason = (
            f"the largest difference between recipe means ({between:.6f} cm2) is not "
            f"larger than the pooled within-recipe spread ({pooled:.6f} cm2)"
        )
    else:
        verdict = "SEPARATION_OBSERVED_NOT_CAUSAL"
        reason = (
            f"recipe means differ by {between:.6f} cm2 against a pooled within-recipe "
            f"spread of {pooled:.6f} cm2; the recipes were not randomly assigned, so "
            "this is an association, not an effect"
        )
    leader = (
        max(comparable, key=lambda group: group["mean_area_cm2"])["recipe_key"]
        if verdict == "SEPARATION_OBSERVED_NOT_CAUSAL"
        else None
    )
    harnesses = {harness_of(group) for group in comparable}
    confounded = len(harnesses) > 1 or None in harnesses
    within = within_harness_comparisons(groups)
    return {
        "verdict": verdict,
        "reason": reason,
        "dominant_recipe": None,
        "highest_mean_recipe_key": leader,
        "seed_region_policy_confounded_with_execution_harness": confounded,
        "within_execution_harness_comparisons": within,
        "interpretable_recipe_comparison_count": sum(
            1 for row in within if row["comparable_recipe_count"] >= 2
        ),
        "comparable_group_count": len(comparable),
        "group_count": len(groups),
        "between_recipe_mean_spread_cm2": between,
        "pooled_within_recipe_stdev_cm2": pooled,
        "distinct_grow_parameter_cells": sorted(
            f"generations={generations},step_size={step_size}"
            for generations, step_size in grow_parameter_cells
        ),
        "grow_parameter_axis_explored": len(grow_parameter_cells) > 1,
        "maximum_within_voxel_stratum_relative_range": (
            max(stratified_ranges) if stratified_ranges else None
        ),
        "minimum_comparable_attempts": MINIMUM_COMPARABLE_ATTEMPTS,
    }


def build_report(roots: list[Path], root: Path, database: Path) -> dict[str, Any]:
    attempts = collect_attempts(roots, root)
    if not attempts:
        raise RuntimeError("no growth receipts found under the requested roots")
    groups = group_rows(attempts)
    assessment = assess(groups)
    return {
        "schema": "campaignx.segmentation_recipe_yield.v1",
        "generated_at_utc": utc_now(),
        "framework_version": framework_version(root),
        "question": (
            "Grouped by (seed_region_policy, generations, step_size), does any "
            "segmentation recipe yield more measured surface area per attempt?"
        ),
        "grouping_key": ["seed_region_policy", "generations", "step_size"],
        "method": [
            "one row per grow attempt that left a GROWTH_RECEIPT.json",
            "archive attempts take seed_region_policy from plan_query_scope",
            "fleet attempts take it from CLAIMED_TASK.candidate_discovery, defaulting "
            "to the worker default fixed-v1",
            "fleet area is read from the finalizer artefact, which measures it after the grow",
            "area is additionally stratified by voxel_size_um because the grow writes a "
            "fixed output grid",
        ],
        "totals": {
            "attempt_count": len(attempts),
            "measured_attempt_count": sum(
                1 for row in attempts if row["area_cm2"] is not None
            ),
            "distinct_seed_region_policy_count": len(
                {row["seed_region_policy"] for row in attempts}
            ),
            "sample_id_count": len({row["sample_id"] for row in attempts if row["sample_id"]}),
            "total_measured_area_cm2": sum(
                row["area_cm2"] for row in attempts if row["area_cm2"] is not None
            ),
        },
        "assessment": assessment,
        "groups": groups,
        "attempts": attempts,
        "attempt_set_sha256": canonical_hash(
            [row["receipt_path"] for row in attempts]
        ),
        "catalog_cross_check": catalog_cross_check(database),
        "ink_used": False,
        "consumed_by_planner_v2": False,
        "out_of_scope": [
            "changing the planner v2 packet contract to consume this table; that is a "
            "separate scientific decision",
            "GPU-hour cost per attempt: the receipts record no wall-clock or device time",
        ],
        "non_claims": [
            "measured area is candidate geometry, never a validated physical sheet",
            "area is not ink, text, or First Letters evidence",
            "recipes were not randomly assigned to cells, so any difference between "
            "groups is an association and not a measured effect of the recipe",
            "a difference between groups run by different execution harnesses is not a "
            "seed-region-policy effect; read within_execution_harness_comparisons",
            "an INSUFFICIENT_TO_DISCRIMINATE verdict is a statement about this evidence, "
            "not a statement that recipes cannot matter",
            "a missing group is unobserved, never a measured zero",
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
        help="directory searched recursively for GROWTH_RECEIPT.json (repeatable)",
    )
    parser.add_argument(
        "--catalog-database",
        type=Path,
        default=Path("workspace/catalog/geometry_surface_catalog_v4/GEOMETRY_SURFACE_CATALOG.sqlite"),
        help="geometry surface catalogue used only as an independent cross-check",
    )
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
    report = build_report(roots, root, database)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "SEGMENTATION_RECIPE_YIELD.json", report)
    assessment = report["assessment"]
    print(
        f"attempts: {report['totals']['attempt_count']}; recipe groups: "
        f"{assessment['group_count']} ({assessment['comparable_group_count']} comparable); "
        f"verdict: {assessment['verdict']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
