#!/usr/bin/env python3
"""Validate and compare a frozen paired VC3D seed-selection benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPEC_KEYS = {
    "schema",
    "benchmark_id",
    "frozen_at_utc",
    "execution_scope",
    "baseline",
    "closed_loop",
    "minimum_cells",
    "maximum_cells",
    "minimum_scrolls",
    "minimum_relative_yield_improvement",
    "maximum_relative_reviewer_rate_regression",
    "maximum_incremental_compute_wall_hours_per_cell",
    "maximum_new_incorrect_lamina_rate_upper_bound",
    "paired_test_alpha",
    "minimum_pairs_per_scroll",
    "review_protocol_id",
    "cells",
}
ARM_KEYS = {"policy_version", "planner", "seed_probe_mode"}
CELL_KEYS = {"cell_id", "sample_id", "independence_block_id"}
RESULT_KEYS = {"schema", "benchmark_id", "started_at_utc", "records"}
RECORD_KEYS = {
    "cell_id",
    "sample_id",
    "independence_block_id",
    "arm",
    "policy_version",
    "planner",
    "seed_probe_mode",
    "source_content_lock_sha256",
    "candidate_set_sha256",
    "vc3d_binary_sha256",
    "full_grow_profile_sha256",
    "full_grow_envelope_sha256",
    "rng_seed",
    "worker_tier",
    "compute_device",
    "review_protocol_id",
    "reviewer_blinded",
    "canonical_area_cm2",
    "usable_nonduplicate_single_lamina_area_cm2",
    "compute_wall_hours",
    "reviewer_minutes",
    "geometry_rejected",
    "incorrect_lamina",
    "abstained",
    "geometry_qc_passed",
    "physical_qc_passed",
    "dedup_passed",
    "single_lamina_confirmed",
    "canonical_output_invariant_ok",
    "lease_invariant_ok",
    "replay_invariant_ok",
    "budget_invariant_ok",
}
IDENTITY_FIELDS = (
    "sample_id",
    "independence_block_id",
    "source_content_lock_sha256",
    "candidate_set_sha256",
    "vc3d_binary_sha256",
    "full_grow_profile_sha256",
    "full_grow_envelope_sha256",
    "rng_seed",
    "worker_tier",
    "compute_device",
    "review_protocol_id",
)
INVARIANT_FIELDS = (
    "canonical_output_invariant_ok",
    "lease_invariant_ok",
    "replay_invariant_ok",
    "budget_invariant_ok",
)
SHA_FIELDS = (
    "source_content_lock_sha256",
    "candidate_set_sha256",
    "vc3d_binary_sha256",
    "full_grow_profile_sha256",
    "full_grow_envelope_sha256",
)


class BenchmarkError(ValueError):
    """The benchmark contract is incomplete or internally inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"cannot read {path}: {error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise BenchmarkError(f"{field} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BenchmarkError(f"{field} is not a valid timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise BenchmarkError(f"{field} must be UTC")
    return parsed


def exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{field} must be an object")
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise BenchmarkError(
            f"{field} keys differ from the contract; missing={missing}, extra={extra}"
        )
    return value


def finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite and non-negative"
        raise BenchmarkError(f"{field} must be {qualifier}")
    return result


def require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkError(f"{field} must be boolean")
    return value


def validate_arm(value: Any, field: str) -> dict[str, Any]:
    arm = exact_keys(value, ARM_KEYS, field)
    if not all(isinstance(arm[key], str) and arm[key] for key in ARM_KEYS):
        raise BenchmarkError(f"{field} identity fields must be non-empty strings")
    return arm


def validate_spec(value: Any) -> dict[str, Any]:
    spec = exact_keys(value, SPEC_KEYS, "spec")
    if spec["schema"] != "campaignx.seed_probe_benchmark_spec.v1":
        raise BenchmarkError("unsupported benchmark spec schema")
    if not isinstance(spec["benchmark_id"], str) or not spec["benchmark_id"]:
        raise BenchmarkError("benchmark_id must be a non-empty string")
    parse_utc(spec["frozen_at_utc"], "frozen_at_utc")
    if spec["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise BenchmarkError(
            "steering benchmark must declare ISOLATED_NONPRODUCTION"
        )
    baseline = validate_arm(spec["baseline"], "baseline")
    closed = validate_arm(spec["closed_loop"], "closed_loop")
    if baseline["seed_probe_mode"] != "off":
        raise BenchmarkError("baseline seed_probe_mode must be off")
    if closed["seed_probe_mode"] != "select":
        raise BenchmarkError("closed_loop seed_probe_mode must be select")
    if baseline["planner"] != "deterministic-v2" or closed["planner"] != "deterministic-v2":
        raise BenchmarkError(
            "production approval comparison must hold planner at deterministic-v2"
        )
    if baseline["policy_version"] == closed["policy_version"]:
        raise BenchmarkError("arm policy versions must be distinct")

    minimum = int(spec["minimum_cells"])
    maximum = int(spec["maximum_cells"])
    scrolls = int(spec["minimum_scrolls"])
    if not (40 <= minimum <= maximum <= 60):
        raise BenchmarkError("cell bounds must satisfy 40 <= minimum <= maximum <= 60")
    if scrolls < 3:
        raise BenchmarkError("minimum_scrolls must be at least 3")
    for field in (
        "minimum_relative_yield_improvement",
        "maximum_relative_reviewer_rate_regression",
        "maximum_incremental_compute_wall_hours_per_cell",
    ):
        finite_number(spec[field], field)
    harm_bound = finite_number(
        spec["maximum_new_incorrect_lamina_rate_upper_bound"],
        "maximum_new_incorrect_lamina_rate_upper_bound",
        positive=True,
    )
    if harm_bound > 1:
        raise BenchmarkError(
            "maximum_new_incorrect_lamina_rate_upper_bound must be at most 1"
        )
    alpha = finite_number(spec["paired_test_alpha"], "paired_test_alpha", positive=True)
    if alpha > 0.10:
        raise BenchmarkError("paired_test_alpha must be at most 0.10")
    minimum_pairs_per_scroll = spec["minimum_pairs_per_scroll"]
    if (
        isinstance(minimum_pairs_per_scroll, bool)
        or not isinstance(minimum_pairs_per_scroll, int)
        or minimum_pairs_per_scroll < 5
    ):
        raise BenchmarkError("minimum_pairs_per_scroll must be an integer >= 5")
    if not isinstance(spec["review_protocol_id"], str) or not spec["review_protocol_id"]:
        raise BenchmarkError("review_protocol_id must be a non-empty string")

    cells = spec["cells"]
    if not isinstance(cells, list):
        raise BenchmarkError("cells must be a list")
    if not minimum <= len(cells) <= maximum:
        raise BenchmarkError("pre-registered cell count is outside the frozen bounds")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    seen_blocks: set[str] = set()
    for index, raw in enumerate(cells):
        cell = exact_keys(raw, CELL_KEYS, f"cells[{index}]")
        if not all(isinstance(cell[key], str) and cell[key] for key in CELL_KEYS):
            raise BenchmarkError(f"cells[{index}] fields must be non-empty strings")
        if cell["cell_id"] in seen:
            raise BenchmarkError(f"duplicate cell_id {cell['cell_id']}")
        if cell["independence_block_id"] in seen_blocks:
            raise BenchmarkError(
                "every cell must name a distinct pre-registered "
                f"independence_block_id; duplicate {cell['independence_block_id']}"
            )
        seen.add(cell["cell_id"])
        seen_blocks.add(cell["independence_block_id"])
        normalized.append(dict(cell))
    if len({row["sample_id"] for row in normalized}) < scrolls:
        raise BenchmarkError("pre-registered cohort has too few scrolls")
    return {**spec, "cells": normalized}


def validate_record(
    raw: Any,
    index: int,
    spec: dict[str, Any],
    cells: dict[str, dict[str, str]],
) -> dict[str, Any]:
    record = exact_keys(raw, RECORD_KEYS, f"records[{index}]")
    cell_id = record["cell_id"]
    if cell_id not in cells:
        raise BenchmarkError(f"results contain unregistered cell {cell_id!r}")
    if record["sample_id"] != cells[cell_id]["sample_id"]:
        raise BenchmarkError(f"{cell_id} sample_id differs from the frozen spec")
    if (
        record["independence_block_id"]
        != cells[cell_id]["independence_block_id"]
    ):
        raise BenchmarkError(
            f"{cell_id} independence_block_id differs from the frozen spec"
        )
    arm_name = record["arm"]
    if arm_name not in {"baseline", "closed_loop"}:
        raise BenchmarkError(f"{cell_id} has unknown arm {arm_name!r}")
    arm = spec[arm_name]
    for field in ARM_KEYS:
        if record[field] != arm[field]:
            raise BenchmarkError(
                f"{cell_id}/{arm_name} {field} differs from the frozen arm"
            )
    if record["review_protocol_id"] != spec["review_protocol_id"]:
        raise BenchmarkError(f"{cell_id}/{arm_name} review protocol changed")
    if require_bool(record["reviewer_blinded"], "reviewer_blinded") is not True:
        raise BenchmarkError(f"{cell_id}/{arm_name} review was not blinded")
    for field in SHA_FIELDS:
        value = record[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BenchmarkError(f"{cell_id}/{arm_name} {field} is not lowercase SHA-256")
    for field in (
        "rng_seed",
        "worker_tier",
        "compute_device",
    ):
        if not isinstance(record[field], str) or not record[field]:
            raise BenchmarkError(f"{cell_id}/{arm_name} {field} must be non-empty")
    if record["compute_device"] not in {"cpu", "cuda"}:
        raise BenchmarkError(f"{cell_id}/{arm_name} compute_device is unsupported")

    canonical = finite_number(
        record["canonical_area_cm2"], f"{cell_id}/{arm_name} canonical_area_cm2"
    )
    usable = finite_number(
        record["usable_nonduplicate_single_lamina_area_cm2"],
        f"{cell_id}/{arm_name} usable area",
    )
    compute = finite_number(
        record["compute_wall_hours"],
        f"{cell_id}/{arm_name} compute_wall_hours",
        positive=True,
    )
    reviewer = finite_number(
        record["reviewer_minutes"], f"{cell_id}/{arm_name} reviewer_minutes"
    )
    if usable > canonical:
        raise BenchmarkError(f"{cell_id}/{arm_name} usable area exceeds canonical area")

    boolean_fields = {
        field: require_bool(record[field], f"{cell_id}/{arm_name} {field}")
        for field in (
            "geometry_rejected",
            "incorrect_lamina",
            "abstained",
            "geometry_qc_passed",
            "physical_qc_passed",
            "dedup_passed",
            "single_lamina_confirmed",
            *INVARIANT_FIELDS,
        )
    }
    usable_authorized = all(
        boolean_fields[field]
        for field in (
            "geometry_qc_passed",
            "physical_qc_passed",
            "dedup_passed",
            "single_lamina_confirmed",
        )
    )
    if usable > 0 and not usable_authorized:
        raise BenchmarkError(
            f"{cell_id}/{arm_name} reports usable area without all QC/dedup/lamina gates"
        )
    if boolean_fields["incorrect_lamina"] and boolean_fields["single_lamina_confirmed"]:
        raise BenchmarkError(
            f"{cell_id}/{arm_name} cannot be both incorrect and confirmed single-lamina"
        )
    if boolean_fields["abstained"] and (canonical > 0 or usable > 0):
        raise BenchmarkError(
            f"{cell_id}/{arm_name} abstained but reports canonical or usable area"
        )
    if boolean_fields["geometry_rejected"] and (
        canonical > 0
        or usable > 0
        or boolean_fields["geometry_qc_passed"]
    ):
        raise BenchmarkError(
            f"{cell_id}/{arm_name} geometry rejection contradicts its area/QC fields"
        )
    return {
        **record,
        "canonical_area_cm2": canonical,
        "usable_nonduplicate_single_lamina_area_cm2": usable,
        "compute_wall_hours": compute,
        "reviewer_minutes": reviewer,
        **boolean_fields,
    }


def validate_results(
    value: Any, spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    results = exact_keys(value, RESULT_KEYS, "results")
    if results["schema"] != "campaignx.seed_probe_benchmark_results.v1":
        raise BenchmarkError("unsupported benchmark results schema")
    if results["benchmark_id"] != spec["benchmark_id"]:
        raise BenchmarkError("results benchmark_id differs from spec")
    if parse_utc(results["started_at_utc"], "started_at_utc") <= parse_utc(
        spec["frozen_at_utc"], "frozen_at_utc"
    ):
        raise BenchmarkError("benchmark results started before the spec was frozen")
    raw_records = results["records"]
    if not isinstance(raw_records, list):
        raise BenchmarkError("records must be a list")
    cells = {
        row["cell_id"]: {
            "sample_id": row["sample_id"],
            "independence_block_id": row["independence_block_id"],
        }
        for row in spec["cells"]
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_records):
        record = validate_record(raw, index, spec, cells)
        key = (record["cell_id"], record["arm"])
        if key in records:
            raise BenchmarkError(f"duplicate result record {key}")
        records[key] = record
    expected = {
        (cell_id, arm)
        for cell_id in cells
        for arm in ("baseline", "closed_loop")
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise BenchmarkError(
            f"results are not a complete paired cohort; missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    return results, records


def safe_relative(candidate: float, baseline: float) -> float | None:
    return (candidate / baseline) - 1.0 if baseline > 0 else None


def exact_one_sided_sign_p_value(wins: int, trials: int) -> float:
    """P[X >= wins] for X~Binomial(trials, 0.5), without optional libraries."""

    if not 0 <= wins <= trials:
        raise BenchmarkError("paired sign-test counts are invalid")
    return sum(math.comb(trials, value) for value in range(wins, trials + 1)) / (
        2**trials
    )


def zero_event_upper_bound(trials: int, alpha: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound when zero harms are seen."""

    if trials < 1 or not 0 < alpha < 1:
        raise BenchmarkError("safety-bound inputs are invalid")
    return 1.0 - alpha ** (1.0 / trials)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = sum(row["usable_nonduplicate_single_lamina_area_cm2"] for row in rows)
    canonical = sum(row["canonical_area_cm2"] for row in rows)
    compute = sum(row["compute_wall_hours"] for row in rows)
    reviewer = sum(row["reviewer_minutes"] for row in rows)
    return {
        "cell_count": len(rows),
        "canonical_area_cm2": canonical,
        "usable_nonduplicate_single_lamina_area_cm2": usable,
        "compute_wall_hours": compute,
        "reviewer_minutes": reviewer,
        "usable_area_per_compute_wall_hour": usable / compute,
        "reviewer_minutes_per_usable_cm2": reviewer / usable if usable > 0 else None,
        "geometry_rejection_rate": sum(row["geometry_rejected"] for row in rows)
        / len(rows),
        "incorrect_lamina_rate": sum(row["incorrect_lamina"] for row in rows)
        / len(rows),
        "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
    }


def check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    detail: str,
    evidence: Any,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
            "evidence": evidence,
        }
    )


def compare(
    spec_value: Any,
    results_value: Any,
    *,
    spec_hash: str,
    results_hash: str,
) -> dict[str, Any]:
    spec = validate_spec(spec_value)
    results, paired = validate_results(results_value, spec)
    cells = [row["cell_id"] for row in spec["cells"]]

    identity_mismatches: list[dict[str, Any]] = []
    invariant_violations: list[dict[str, Any]] = []
    for cell_id in cells:
        baseline = paired[(cell_id, "baseline")]
        closed = paired[(cell_id, "closed_loop")]
        differing = [
            field for field in IDENTITY_FIELDS if baseline[field] != closed[field]
        ]
        if differing:
            identity_mismatches.append(
                {"cell_id": cell_id, "differing_fields": differing}
            )
        for arm, row in (("baseline", baseline), ("closed_loop", closed)):
            failed = [field for field in INVARIANT_FIELDS if not row[field]]
            if failed:
                invariant_violations.append(
                    {"cell_id": cell_id, "arm": arm, "failed": failed}
                )

    by_arm = {
        arm: summarize([paired[(cell_id, arm)] for cell_id in cells])
        for arm in ("baseline", "closed_loop")
    }
    baseline_summary = by_arm["baseline"]
    closed_summary = by_arm["closed_loop"]
    yield_change = safe_relative(
        closed_summary["usable_area_per_compute_wall_hour"],
        baseline_summary["usable_area_per_compute_wall_hour"],
    )
    baseline_reviewer = baseline_summary["reviewer_minutes_per_usable_cm2"]
    closed_reviewer = closed_summary["reviewer_minutes_per_usable_cm2"]
    reviewer_change = (
        safe_relative(closed_reviewer, baseline_reviewer)
        if closed_reviewer is not None and baseline_reviewer is not None
        else None
    )
    incremental_compute = (
        closed_summary["compute_wall_hours"]
        - baseline_summary["compute_wall_hours"]
    ) / len(cells)
    lamina_regression = (
        closed_summary["incorrect_lamina_rate"]
        - baseline_summary["incorrect_lamina_rate"]
    )

    paired_counts = {
        "closed_loop_margin_wins": 0,
        "ties_or_below_margin": 0,
        "baseline_raw_wins": 0,
    }
    for cell_id in cells:
        baseline = paired[(cell_id, "baseline")]
        closed = paired[(cell_id, "closed_loop")]
        baseline_rate = (
            baseline["usable_nonduplicate_single_lamina_area_cm2"]
            / baseline["compute_wall_hours"]
        )
        closed_rate = (
            closed["usable_nonduplicate_single_lamina_area_cm2"]
            / closed["compute_wall_hours"]
        )
        required_rate = baseline_rate * (
            1.0 + spec["minimum_relative_yield_improvement"]
        )
        if closed_rate > required_rate and not math.isclose(
            closed_rate, required_rate, rel_tol=1e-12, abs_tol=1e-12
        ):
            paired_counts["closed_loop_margin_wins"] += 1
        else:
            paired_counts["ties_or_below_margin"] += 1
        if closed_rate < baseline_rate and not math.isclose(
            baseline_rate, closed_rate, rel_tol=1e-12, abs_tol=1e-12
        ):
            paired_counts["baseline_raw_wins"] += 1

    sign_p_value = exact_one_sided_sign_p_value(
        paired_counts["closed_loop_margin_wins"], len(cells)
    )
    new_lamina_harms = [
        cell_id
        for cell_id in cells
        if paired[(cell_id, "closed_loop")]["incorrect_lamina"]
        and not paired[(cell_id, "baseline")]["incorrect_lamina"]
    ]
    safety_upper_bound = (
        zero_event_upper_bound(len(cells), spec["paired_test_alpha"])
        if not new_lamina_harms
        else 1.0
    )

    rows_by_scroll: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"baseline": [], "closed_loop": []}
    )
    for row in paired.values():
        rows_by_scroll[row["sample_id"]][row["arm"]].append(row)
    per_scroll: dict[str, dict[str, Any]] = {}
    for sample, arms in sorted(rows_by_scroll.items()):
        summaries = {
            arm: summarize(arms[arm])
            for arm in ("baseline", "closed_loop")
        }
        per_scroll[sample] = {
            **summaries,
            "paired_cell_count": len(arms["baseline"]),
            "relative_yield_change": safe_relative(
                summaries["closed_loop"]["usable_area_per_compute_wall_hour"],
                summaries["baseline"]["usable_area_per_compute_wall_hour"],
            ),
            "relative_reviewer_rate_change": (
                safe_relative(
                    summaries["closed_loop"][
                        "reviewer_minutes_per_usable_cm2"
                    ],
                    summaries["baseline"][
                        "reviewer_minutes_per_usable_cm2"
                    ],
                )
                if summaries["closed_loop"][
                    "reviewer_minutes_per_usable_cm2"
                ]
                is not None
                and summaries["baseline"][
                    "reviewer_minutes_per_usable_cm2"
                ]
                is not None
                else None
            ),
        }

    checks: list[dict[str, Any]] = []
    check(
        checks,
        "MATCHED_EXECUTION_IDENTITY",
        not identity_mismatches,
        "paired arms preserve source, candidates, binary, profile, envelope, RNG, worker tier, device, and review protocol",
        {"mismatches": identity_mismatches},
    )
    check(
        checks,
        "INVARIANTS",
        not invariant_violations,
        "canonical output, lease, replay, and budget invariants hold",
        {"violations": invariant_violations},
    )
    check(
        checks,
        "YIELD_IMPROVEMENT",
        yield_change is not None
        and yield_change >= spec["minimum_relative_yield_improvement"],
        "closed-loop usable area per compute wall-hour meets the frozen superiority margin",
        {
            "relative_change": yield_change,
            "minimum": spec["minimum_relative_yield_improvement"],
        },
    )
    check(
        checks,
        "PAIRED_SUPERIORITY",
        sign_p_value <= spec["paired_test_alpha"],
        "the exact paired sign test rejects chance at the frozen one-sided alpha",
        {
            **paired_counts,
            "paired_cell_count": len(cells),
            "one_sided_p_value": sign_p_value,
            "maximum_alpha": spec["paired_test_alpha"],
        },
    )
    check(
        checks,
        "REVIEWER_RATE",
        reviewer_change is not None
        and reviewer_change <= spec["maximum_relative_reviewer_rate_regression"],
        "reviewer minutes per usable square centimetre stays within the frozen margin",
        {
            "relative_change": reviewer_change,
            "maximum": spec["maximum_relative_reviewer_rate_regression"],
        },
    )
    check(
        checks,
        "COMPUTE_BUDGET",
        incremental_compute
        <= spec["maximum_incremental_compute_wall_hours_per_cell"],
        "incremental compute wall-hours per cell stays within budget",
        {
            "incremental_per_cell": incremental_compute,
            "maximum": spec[
                "maximum_incremental_compute_wall_hours_per_cell"
            ],
        },
    )
    check(
        checks,
        "NEW_INCORRECT_LAMINA_SAFETY",
        not new_lamina_harms
        and safety_upper_bound
        <= spec["maximum_new_incorrect_lamina_rate_upper_bound"],
        "no paired cell introduces a new incorrect lamina and the exact upper bound stays within the frozen margin",
        {
            "new_harm_count": len(new_lamina_harms),
            "new_harm_cell_ids": new_lamina_harms,
            "one_sided_upper_bound": safety_upper_bound,
            "alpha": spec["paired_test_alpha"],
            "net_absolute_rate_change": lamina_regression,
            "maximum": spec[
                "maximum_new_incorrect_lamina_rate_upper_bound"
            ],
        },
    )
    check(
        checks,
        "SCROLL_COVERAGE",
        len(per_scroll) >= spec["minimum_scrolls"]
        and all(
            row["paired_cell_count"] >= spec["minimum_pairs_per_scroll"]
            for row in per_scroll.values()
        ),
        "every pre-registered scroll has the minimum paired sample",
        {
            "scroll_count": len(per_scroll),
            "minimum_scrolls": spec["minimum_scrolls"],
            "minimum_pairs_per_scroll": spec["minimum_pairs_per_scroll"],
            "pairs_by_scroll": {
                sample: row["paired_cell_count"]
                for sample, row in per_scroll.items()
            },
        },
    )
    regressed_scrolls = [
        sample
        for sample, row in per_scroll.items()
        if row["relative_yield_change"] is None
        or row["relative_yield_change"] < 0
    ]
    reviewer_regressed_scrolls = [
        sample
        for sample, row in per_scroll.items()
        if row["relative_reviewer_rate_change"] is None
        or row["relative_reviewer_rate_change"]
        > spec["maximum_relative_reviewer_rate_regression"]
    ]
    check(
        checks,
        "SCROLL_NONREGRESSION",
        not regressed_scrolls and not reviewer_regressed_scrolls,
        "yield and reviewer rate stay within their frozen margin on every registered scroll",
        {
            "yield_regressed_scrolls": regressed_scrolls,
            "reviewer_rate_regressed_scrolls": reviewer_regressed_scrolls,
        },
    )

    approved = all(row["status"] == "PASS" for row in checks)
    receipt: dict[str, Any] = {
        "schema": "campaignx.seed_probe_benchmark_decision.v1",
        "benchmark_id": spec["benchmark_id"],
        "status": "APPROVED_SELECT" if approved else "NOT_APPROVED",
        "execution_scope": spec["execution_scope"],
        "spec_sha256": spec_hash,
        "results_sha256": results_hash,
        "paired_cell_count": len(cells),
        "scroll_count": len(per_scroll),
        "authorized_sample_ids": sorted(per_scroll),
        "arms": {
            "baseline": spec["baseline"],
            "closed_loop": spec["closed_loop"],
        },
        "metrics": {
            "by_arm": by_arm,
            "relative_yield_change": yield_change,
            "relative_reviewer_rate_change": reviewer_change,
            "incremental_compute_wall_hours_per_cell": incremental_compute,
            "absolute_incorrect_lamina_rate_change": lamina_regression,
            "paired_counts": paired_counts,
            "paired_sign_test_one_sided_p_value": sign_p_value,
            "new_incorrect_lamina_harm_count": len(new_lamina_harms),
            "new_incorrect_lamina_rate_upper_bound": safety_upper_bound,
            "per_scroll": per_scroll,
        },
        "checks": checks,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "non_claims": [
            "approval is a rollout decision, not proof that any seed is correct",
            "TIFXYZ geometry certification is not independent physical-lamina truth",
            "results outside this frozen cohort are unmeasured",
            "independence_block_id is a preregistered experimental assertion, "
            "not an automatically measured distance",
        ],
    }
    receipt["receipt_sha256"] = content_sha256(receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compare a frozen paired deterministic-v2/off versus deterministic-v2/select benchmark."
    )
    result.add_argument("--spec", required=True, type=Path)
    result.add_argument("--results", required=True, type=Path)
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        spec_value = load_json(arguments.spec)
        results_value = load_json(arguments.results)
        receipt = compare(
            spec_value,
            results_value,
            spec_hash=content_sha256(spec_value),
            results_hash=content_sha256(results_value),
        )
    except BenchmarkError as error:
        receipt = {
            "schema": "campaignx.seed_probe_benchmark_decision.v1",
            "status": "NOT_APPROVED",
            "error": str(error),
            "generated_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        receipt["receipt_sha256"] = content_sha256(receipt)
    if arguments.output:
        write_json_atomic(arguments.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "APPROVED_SELECT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
