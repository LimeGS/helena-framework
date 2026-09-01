"""Ink-blind task budgets derived from candidate-coverage evidence."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import heapq
import json
import math
import os
import re
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from scipy.stats import beta

from .common import content_sha256, stable_id, utc_now


BUDGET_SCHEMA = "campaignx.first_letters_task_budget.v1"
BUDGET_SANITIZED_SCHEMA = "campaignx.first_letters_task_budget.sanitized.v1"
CAP_SCHEMA = "campaignx.first_letters_compute_cap.v1"
ELIGIBLE_POPULATION_SCHEMA = "campaignx.first_letters_eligible_population.v1"
PREFLIGHT_SCHEMA = "campaignx.segment_candidate_coverage_preflight.sanitized.v1"
SAMPLED_INFERENCE_ASSUMPTION = (
    "MODEL_BASED_BINOMIAL_EXCHANGEABILITY_NOT_DESIGN_BASED"
)
NO_SEED_SCIENTIFIC_CAUSAL_LABELS = frozenset({
    "NO_M7_CANDIDATES",
    "CT_MATERIAL_SUPPORT_REJECTED",
    "MALFORMED_COORDINATE_OR_SCORE",
    "INSUFFICIENT_CELL_INTERIOR_CLEARANCE",
    "INSUFFICIENT_VOLUME_INTERIOR_CLEARANCE",
})


def load_campaign_policy_profile() -> dict[str, Any]:
    """Load the checked-in campaign policy whose hash admissions freeze."""
    path = (
        Path(__file__).resolve().parents[3]
        / "profiles/01-segmentation/"
        "first-letters-campaign-decision-policy-1.2.0.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    _validated_policy(value)
    return value


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _receipt_hash(value: dict[str, Any]) -> str:
    return content_sha256({
        key: row for key, row in value.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })


def _public_hash(value: dict[str, Any]) -> str:
    return content_sha256({
        key: row for key, row in value.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })


def _preflight_hash(value: dict[str, Any]) -> str:
    return content_sha256({
        key: row for key, row in value.items()
        if key not in {
            "generated_at_utc", "receipt_sha256",
            "evidence_status", "evidence_status_reason",
        }
    })


def _detection_probability(population: int, successes: int, draws: int) -> float:
    if draws <= 0 or successes <= 0 or population <= 0:
        return 0.0
    draws = min(draws, population)
    if successes >= population or draws > population - successes:
        return 1.0
    no_hit = 1.0
    failures = population - successes
    for index in range(draws):
        no_hit *= (failures - index) / (population - index)
    return 1.0 - no_hit


def _minimum_draws(population: int, successes: int, target: float) -> int:
    if not 0 < successes <= population:
        raise ValueError("a positive finite population is required for a task budget")
    if not 0 < target < 1:
        raise ValueError("target detection probability must be between zero and one")

    # Advance the hypergeometric miss probability by one draw at a time in log
    # space.  This is O(N), whereas recomputing the full product for every
    # candidate n is O(N^2) at the policy's 262,144-cell ceiling.
    log_miss = 0.0
    log_threshold = math.log1p(-target)
    candidate = population
    for draws in range(1, population + 1):
        remaining = population - draws + 1
        if remaining <= successes:
            candidate = draws
            break
        log_miss += math.log1p(-successes / remaining)
        if log_miss <= log_threshold:
            candidate = draws
            break

    # Floating point only locates the boundary.  The returned boundary is
    # verified with exact integers so a close comparison cannot add/drop a task.
    target_fraction = Fraction(str(target))

    def exact_met(draws: int) -> bool:
        if draws <= 0:
            return False
        if draws > population - successes:
            return True
        if successes <= draws:
            miss_numerator = math.comb(population - draws, successes)
            miss_denominator = math.comb(population, successes)
        else:
            miss_numerator = math.comb(population - successes, draws)
            miss_denominator = math.comb(population, draws)
        return (
            (miss_denominator - miss_numerator) * target_fraction.denominator
            >= miss_denominator * target_fraction.numerator
        )

    while candidate > 1 and exact_met(candidate - 1):
        candidate -= 1
    while candidate < population and not exact_met(candidate):
        candidate += 1
    return candidate


def _validated_policy(policy: object) -> tuple[dict[str, Any], float, float]:
    if (not isinstance(policy, dict)
            or policy.get("schema") != "campaignx.first_letters_campaign_policy.v1"
            or not isinstance(policy.get("profile_id"), str)):
        raise ValueError("campaign policy is unsupported")
    budget = policy.get("task_budget")
    if not isinstance(budget, dict):
        raise ValueError("campaign policy has no task budget")
    target = budget.get("target_detection_probability")
    sampled = budget.get("sampled_preflight")
    population_scan = budget.get("population_scan")
    if (not isinstance(target, (int, float)) or isinstance(target, bool)
            or not 0 < float(target) < 1
            or not isinstance(sampled, dict)
            or not isinstance(population_scan, dict)
            or population_scan.get("mode") !=
                "EXACT_DETERMINISTIC_STREAMING_GEOMETRY_SCAN"
            or not isinstance(population_scan.get("hard_grid_cell_limit"), int)
            or isinstance(population_scan.get("hard_grid_cell_limit"), bool)
            or not 1 <= population_scan["hard_grid_cell_limit"] <= 1_000_000
            or population_scan.get("limit_exceeded") !=
                "CONTROL_INCOMPLETE_REQUIRE_COARSER_FROZEN_GRID"
            or sampled.get("lower_bound") != "ONE_SIDED_CLOPPER_PEARSON"
            or sampled.get("lower_tail_alpha") != 0.05):
        raise ValueError("campaign policy task-budget statistics are unsupported")
    if sampled.get("inference_assumption") != SAMPLED_INFERENCE_ASSUMPTION:
        raise ValueError(
            "campaign policy sampled inference assumption is unsupported")
    return policy, float(target), float(sampled["lower_tail_alpha"])


def _validated_preflight(value: object) -> tuple[dict[str, Any], int, int, int, str]:
    if not isinstance(value, dict) or value.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("candidate preflight schema is unsupported")
    if value.get("evidence_status") != "CURRENT":
        raise ValueError("candidate preflight evidence must be CURRENT")
    if value.get("receipt_sha256") != _preflight_hash(value):
        raise ValueError("candidate preflight sanitized hash is invalid")
    private_hash = value.get("private_receipt_sha256")
    if (not isinstance(private_hash, str) or len(private_hash) != 64
            or any(character not in "0123456789abcdef" for character in private_hash)):
        raise ValueError("candidate preflight private hash is invalid")
    funnel = value.get("funnel")
    design = value.get("sampling_design")
    bins = value.get("spatial_bins")
    bindings = value.get("bindings")
    gates = value.get("gates")
    if not all(isinstance(row, dict) for row in (funnel, design, bindings, gates)):
        raise ValueError("candidate preflight budget inputs are incomplete")
    if not isinstance(bins, list) or not all(isinstance(row, dict) for row in bins):
        raise ValueError("candidate preflight spatial evidence is incomplete")
    failed = funnel.get("cells_failed_source")
    if not _nonnegative_integer(failed) or failed != funnel.get("source_errors"):
        raise ValueError("candidate preflight source-error counts are invalid")
    if failed:
        raise ValueError("candidate preflight source errors cannot enter a task budget")
    measurement = value.get("measurement_kind")
    if measurement not in {"CENSUS", "ESTIMATE"}:
        raise ValueError("candidate preflight measurement is incomplete")
    expected_design = (
        "complete-eligible-grid-census-v1"
        if measurement == "CENSUS"
        else "deterministic-golden-coprime-rank1-grid-sample-v1"
    )
    if (design.get("measurement_kind") != measurement
            or design.get("name") != expected_design):
        raise ValueError(
            "candidate preflight sampling-design assumptions do not match "
            "the frozen budget model")
    trials = funnel.get("cells_surveyed_successfully")
    if not _nonnegative_integer(trials):
        raise ValueError("candidate preflight trial count is invalid")
    usable = sum(row.get("usable_candidate_cells", -1) for row in bins)
    if not _nonnegative_integer(usable) or usable > trials:
        raise ValueError("candidate preflight usable-cell count is invalid")
    population = (funnel.get("geometrically_eligible_cells")
                  if measurement == "CENSUS"
                  else funnel.get("geometrically_eligible_cells_estimate"))
    if not _nonnegative_integer(population) or trials > population:
        raise ValueError("candidate preflight eligible population is invalid")
    if measurement == "CENSUS" and trials != population:
        raise ValueError("candidate preflight census did not survey its population")
    return value, population, usable, trials, measurement


def _validated_cap(value: object, *, mission_id: str, sample_id: str) -> dict[str, Any]:
    required = {"schema", "cap_id", "mission_id", "sample_id", "maximum_tasks"}
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") != CAP_SCHEMA
            or not isinstance(value.get("cap_id"), str) or not value["cap_id"]
            or value.get("mission_id") != mission_id
            or value.get("sample_id") != sample_id
            or not _nonnegative_integer(value.get("maximum_tasks"))
            or value["maximum_tasks"] > 4096):
        raise ValueError("frozen compute cap is invalid or names another mission/sample")
    return value


def _execution_bindings(
    preflight: dict[str, Any], policy: dict[str, Any],
) -> dict[str, Any]:
    bindings = preflight["bindings"]
    result = {
        key: copy.deepcopy(bindings[key]) for key in (
            "sample_id", "mission_id", "source_snapshot_id",
            "p0_artifact_id", "p0_artifact_sha256",
            "p0_selection_version", "p0_selection_sha256",
            "catalog_snapshot_sha256",
            "source_content_lock_sha256", "ct_sha256", "m7_sha256",
            "m7_uri_sha256",
            "coordinate_frame", "voxel_size_um", "shape_xyz",
            "grid_version", "policy_version", "provider", "m7_threshold",
            "grid_step", "query_radius", "parallelism", "maximum_cells",
            "selection_strategy", "candidate_selection_policy",
            "seed_region_policy", "code_revision",
        )
    }
    result["gates"] = copy.deepcopy(preflight["gates"])
    queue_policy = policy["task_budget"].get("queue_execution")
    if not isinstance(queue_policy, dict):
        raise ValueError("campaign policy has no frozen queue execution")
    packet_limit = preflight["gates"].get("packet_candidate_limit")
    if (not isinstance(packet_limit, int) or isinstance(packet_limit, bool)
            or not 1 <= packet_limit <= 100):
        raise ValueError("candidate preflight packet limit is invalid")
    from .generator import DEFAULT_ENVELOPE  # noqa: PLC0415
    queue_execution = copy.deepcopy(queue_policy)
    queue_execution["parameter_envelope"] = {
        **copy.deepcopy(DEFAULT_ENVELOPE),
        "maximum_candidate_count": packet_limit,
    }
    result["queue_execution"] = queue_execution
    return result


def _expected_cell_geometry(
    cell_id: str, execution: dict[str, Any], gates: dict[str, Any],
) -> tuple[dict[str, int], list[list[int]]]:
    """Derive the exact frozen grid geometry named by a probability-prefix id."""
    match = re.fullmatch(r"r(\d{5})c(\d{5})a(\d{5})", str(cell_id))
    shape = execution.get("shape_xyz")
    step = execution.get("grid_step")
    radius = execution.get("query_radius")
    volume_margin = gates.get("volume_clearance")
    if (match is None or not isinstance(shape, list) or len(shape) != 3
            or not all(isinstance(value, int) and not isinstance(value, bool)
                       and value > 0 for value in shape)
            or not isinstance(step, int) or isinstance(step, bool) or step < 1
            or not isinstance(radius, int) or isinstance(radius, bool) or radius < 1
            or not isinstance(volume_margin, int)
            or isinstance(volume_margin, bool) or volume_margin < 1):
        raise ValueError("campaign budget frozen grid geometry is invalid")
    margin = max(radius, volume_margin)
    axes: list[list[int]] = []
    for size in shape:
        centers = list(range(margin, size - margin, step))
        if centers:
            last = size - margin - 1
            if last - centers[-1] >= step // 2:
                centers.append(last)
        axes.append(centers)
    indices = [int(value) for value in match.groups()]
    if any(index >= len(axes[axis]) for axis, index in enumerate(indices)):
        raise ValueError("campaign budget prefix cell is outside the frozen grid")
    values = [axes[axis][index] for axis, index in enumerate(indices)]
    center = dict(zip("xyz", values, strict=True))
    bounds = [
        [value - radius for value in values],
        [value + radius for value in values],
    ]
    return center, bounds


def probability_prefix_order_seed(
    preflight: dict[str, Any], policy: dict[str, Any], compute_cap: dict[str, Any],
) -> str:
    """Return the outcome-independent rank seed used by the streaming scan."""
    policy, _target, _alpha = _validated_policy(policy)
    preflight, _population, _observed, _trials, _measurement = (
        _validated_preflight(preflight)
    )
    bindings = preflight["bindings"]
    _validated_cap(
        compute_cap, mission_id=bindings["mission_id"],
        sample_id=bindings["sample_id"],
    )
    return content_sha256({
        "schema": "campaignx.first_letters_probability_prefix_seed.v1",
        "execution_bindings": _execution_bindings(preflight, policy),
        "campaign_policy_sha256": content_sha256(policy),
    })


class EligiblePopulationAccumulator:
    """Hash an exact population stream while retaining only its bounded top rank."""

    def __init__(self, *, order_seed_sha256: str, prefix_limit: int) -> None:
        if (not isinstance(order_seed_sha256, str)
                or len(order_seed_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in order_seed_sha256)):
            raise ValueError("eligible population order seed is invalid")
        if (not isinstance(prefix_limit, int) or isinstance(prefix_limit, bool)
                or prefix_limit < 0):
            raise ValueError("eligible population prefix limit is invalid")
        self.order_seed_sha256 = order_seed_sha256
        self.prefix_limit = prefix_limit
        self.count = 0
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._heap: list[tuple[int, str]] = []
        self._retained_ranks: dict[int, str] = {}
        self._finished = False

    def observe(self, cell_id: str) -> None:
        if self._finished:
            raise ValueError("eligible population scan is already complete")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("eligible population cell id is invalid")
        if self.count:
            self._digest.update(b",")
        self._digest.update(json.dumps(
            cell_id, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
        self.count += 1
        if self.prefix_limit < 1:
            return
        rank = int(content_sha256({
            "seed": self.order_seed_sha256, "cell_id": cell_id,
        }), 16)
        if rank in self._retained_ranks:
            raise ValueError("eligible population SHA256 rank collision")
        row = (-rank, cell_id)
        if len(self._heap) < self.prefix_limit:
            heapq.heappush(self._heap, row)
            self._retained_ranks[rank] = cell_id
        elif rank < -self._heap[0][0]:
            removed = heapq.heapreplace(self._heap, row)
            self._retained_ranks.pop(-removed[0])
            self._retained_ranks[rank] = cell_id

    def finish(self) -> dict[str, Any]:
        if self._finished:
            raise ValueError("eligible population scan is already complete")
        self._finished = True
        self._digest.update(b"]")
        ranked = [
            row[1] for row in sorted(
                self._heap,
                key=lambda item: (
                    content_sha256({
                        "seed": self.order_seed_sha256, "cell_id": item[1],
                    }),
                    item[1],
                ),
            )
        ]
        return {
            "schema": ELIGIBLE_POPULATION_SCHEMA,
            "scan_order": "GRID_PRODUCT_XYZ_INDEX_V1",
            "eligible_population_cells": self.count,
            "population_order_sha256": self._digest.hexdigest(),
            "order_seed_sha256": self.order_seed_sha256,
            "ranked_prefix_capacity": self.prefix_limit,
            "ranked_prefix_cell_ids": ranked,
            "ranked_prefix_sha256": content_sha256(ranked),
        }


def summarize_eligible_population(
    eligible_cell_ids: list[str], *, order_seed_sha256: str, prefix_limit: int,
) -> dict[str, Any]:
    """Build the compact population proof used by pure/unit callers."""
    if (not isinstance(eligible_cell_ids, list)
            or len(set(eligible_cell_ids)) != len(eligible_cell_ids)):
        raise ValueError("current eligible cell population is missing or duplicate")
    accumulator = EligiblePopulationAccumulator(
        order_seed_sha256=order_seed_sha256, prefix_limit=prefix_limit)
    for cell_id in eligible_cell_ids:
        accumulator.observe(cell_id)
    return accumulator.finish()


def _validated_eligible_population(
    value: object, *, order_seed_sha256: str, prefix_limit: int,
) -> tuple[int, list[str], str]:
    required = {
        "schema", "scan_order", "eligible_population_cells",
        "population_order_sha256", "order_seed_sha256",
        "ranked_prefix_capacity", "ranked_prefix_cell_ids",
        "ranked_prefix_sha256",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") != ELIGIBLE_POPULATION_SCHEMA
            or value.get("scan_order") != "GRID_PRODUCT_XYZ_INDEX_V1"
            or value.get("order_seed_sha256") != order_seed_sha256
            or value.get("ranked_prefix_capacity") != prefix_limit
            or not _nonnegative_integer(value.get("eligible_population_cells"))):
        raise ValueError("eligible population proof is invalid or out of scope")
    population = value["eligible_population_cells"]
    prefix = value.get("ranked_prefix_cell_ids")
    expected_prefix_count = min(population, prefix_limit)
    if (not isinstance(prefix, list) or len(prefix) != expected_prefix_count
            or len(prefix) != len(set(prefix))
            or any(not isinstance(cell_id, str) or not cell_id for cell_id in prefix)
            or value.get("ranked_prefix_sha256") != content_sha256(prefix)):
        raise ValueError("eligible population ranked prefix is invalid")
    population_sha = value.get("population_order_sha256")
    if (not isinstance(population_sha, str) or len(population_sha) != 64
            or any(character not in "0123456789abcdef"
                   for character in population_sha)):
        raise ValueError("eligible population scan hash is invalid")
    return population, list(prefix), population_sha


def derive_task_budget(
    preflight: dict[str, Any], policy: dict[str, Any], compute_cap: dict[str, Any],
    *, manual_task_count: int | None = None,
    manual_lower_reason: str | None = None,
    eligible_cell_ids: list[str] | None = None,
    eligible_population: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive one content-bound budget from current ink-blind evidence."""
    policy, target, alpha = _validated_policy(policy)
    preflight, preflight_population, observed, trials, measurement = (
        _validated_preflight(preflight)
    )
    bindings = preflight["bindings"]
    mission_id, sample_id = bindings.get("mission_id"), bindings.get("sample_id")
    if (not isinstance(mission_id, str) or not mission_id
            or not isinstance(sample_id, str) or not sample_id
            or preflight.get("sample_id") != sample_id):
        raise ValueError("candidate preflight mission/sample scope is invalid")
    cap = _validated_cap(compute_cap, mission_id=mission_id, sample_id=sample_id)
    policy_sha = content_sha256(policy)
    cap_sha = content_sha256(cap)
    execution_bindings = _execution_bindings(preflight, policy)
    order_seed_sha = probability_prefix_order_seed(preflight, policy, cap)
    if eligible_population is not None and eligible_cell_ids is not None:
        raise ValueError("provide one eligible population proof, not two")
    if eligible_population is None:
        if eligible_cell_ids is None:
            raise ValueError("current eligible cell population is missing")
        eligible_population = summarize_eligible_population(
            eligible_cell_ids, order_seed_sha256=order_seed_sha,
            prefix_limit=cap["maximum_tasks"],
        )
    population, population_prefix, population_order_sha = (
        _validated_eligible_population(
            eligible_population, order_seed_sha256=order_seed_sha,
            prefix_limit=cap["maximum_tasks"],
        )
    )
    if measurement == "CENSUS" and population != preflight_population:
        raise ValueError(
            "census eligible cell population is missing, duplicate, or drifted")
    if observed > 0 and population == 0:
        raise ValueError(
            "current eligible cell population contradicts positive preflight evidence")

    # This seed deliberately excludes K, candidate coordinates, result hashes,
    # and the preflight receipt hash.  The resulting permutation is frozen from
    # source/policy scope before candidate outcomes can influence it.  The
    # compute cap is deliberately excluded: caps truncate one nested order.
    lower_bound: float | None = None
    upper_bound: float | None = None
    conservative_successes = observed
    probability_model = "EXACT_HYPERGEOMETRIC_WITHOUT_REPLACEMENT"
    inference_assumption: str | None = None
    decision = "CONTINUE"
    allowed_next_actions = ["QUEUE_WITH_BOUND_BUDGET"]

    if observed == 0:
        requested: int | None = None
        planned = 0
        achieved = 0.0
        if measurement == "CENSUS":
            decision = "DO_NOT_QUEUE_CURRENT_SOURCE"
            allowed_next_actions = ["CHANGE_CANDIDATE_SOURCE_OR_POLICY"]
        else:
            inference_assumption = SAMPLED_INFERENCE_ASSUMPTION
            upper_bound = (float(beta.ppf(1.0 - alpha, 1, trials))
                           if trials else 1.0)
            decision = "MORE_PREFLIGHT_OR_NEW_SOURCE_REQUIRED"
            allowed_next_actions = [
                "RUN_LARGER_PREFLIGHT", "CHANGE_CANDIDATE_SOURCE"]
        manual_lower = False
    else:
        if measurement == "ESTIMATE":
            inference_assumption = SAMPLED_INFERENCE_ASSUMPTION
            lower_bound = float(beta.ppf(alpha, observed, trials - observed + 1))
            conservative_successes = max(1, math.floor(population * lower_bound))
            probability_model = "CONSERVATIVE_CLOPPER_PEARSON_TO_FINITE_POPULATION"
        requested = _minimum_draws(population, conservative_successes, target)
        automatic = min(requested, cap["maximum_tasks"], population)
        if automatic == 0:
            if manual_task_count is not None:
                raise ValueError(
                    "manual task count cannot exceed a zero frozen compute cap")
            planned = 0
            manual_lower = False
            achieved = 0.0
            decision = "NO_COMPUTE_AUTHORIZED"
            allowed_next_actions = ["INCREASE_FROZEN_COMPUTE_CAP"]
        elif manual_task_count is not None:
            if (not isinstance(manual_task_count, int)
                    or isinstance(manual_task_count, bool)
                    or not 1 <= manual_task_count <= automatic):
                raise ValueError("manual task count must be a positive lower budget")
            planned = manual_task_count
        else:
            planned = automatic
        if automatic:
            manual_lower = (
                manual_task_count is not None and manual_task_count < automatic)
            if manual_lower and not str(manual_lower_reason or "").strip():
                raise ValueError(
                    "a manual lower task budget requires an explicit reason")
            achieved = _detection_probability(
                population, conservative_successes, planned)
    selected_prefix = population_prefix[:planned]
    queue_selection = {
        "sampling": "WITHOUT_REPLACEMENT",
        "ordering": "SERVER_DERIVED_SHA256_PERMUTATION_V1",
        "selection_strategy": "probability-prefix-v1",
        "order_seed_sha256": order_seed_sha,
        "population_order_sha256": population_order_sha,
        "population_scan_order": "GRID_PRODUCT_XYZ_INDEX_V1",
        "prefix_cell_ids": selected_prefix,
        "prefix_sha256": content_sha256(selected_prefix),
    }
    receipt: dict[str, Any] = {
        "schema": BUDGET_SCHEMA,
        "decision": decision,
        "mission_id": mission_id,
        "sample_id": sample_id,
        "preflight_receipt_sha256": preflight["private_receipt_sha256"],
        "preflight_sanitized_receipt_sha256": preflight["receipt_sha256"],
        "campaign_policy_profile_id": policy["profile_id"],
        "campaign_policy_sha256": policy_sha,
        "compute_cap": copy.deepcopy(cap),
        "compute_cap_sha256": cap_sha,
        "compute_cap_tasks": cap["maximum_tasks"],
        "population_scan_hard_grid_cell_limit": policy["task_budget"][
            "population_scan"]["hard_grid_cell_limit"],
        "measurement_kind": measurement,
        "preflight_eligible_population_cells": preflight_population,
        "eligible_population_cells": population,
        "successful_preflight_trials": trials,
        "observed_usable_cells": observed,
        "prevalence_lower_confidence_bound": lower_bound,
        "prevalence_upper_confidence_bound": upper_bound,
        "conservative_population_usable_cells": conservative_successes,
        "target_detection_probability": target,
        "requested_task_count": requested,
        "planned_task_count": planned,
        "manual_lower_budget": manual_lower,
        "manual_lower_reason": (
            str(manual_lower_reason).strip() if manual_lower else None),
        "planned_sampling_percentage": (
            100.0 * planned / population if population else 0.0),
        "achieved_detection_probability": achieved,
        "target_detection_probability_met": achieved >= target,
        "probability_model": probability_model,
        "sampling_inference_assumption": inference_assumption,
        "queue_selection": queue_selection,
        "execution_bindings": execution_bindings,
        "allowed_next_actions": allowed_next_actions,
        "non_claim": (
            "A bounded candidate result is not evidence of surface, ink, text, "
            "or letter absence. Estimated-preflight confidence bounds assume "
            "binomial exchangeability and are not a design-based guarantee."),
        "generated_at_utc": utc_now(),
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def sanitize_task_budget_receipt(private: dict[str, Any]) -> dict[str, Any]:
    """Project aggregate budget evidence without exposing the frozen cell order."""
    if (not isinstance(private, dict) or private.get("schema") != BUDGET_SCHEMA
            or private.get("receipt_sha256") != _receipt_hash(private)):
        raise ValueError("task-budget private receipt hash or schema is invalid")
    public = copy.deepcopy(private)
    private_sha = public.pop("receipt_sha256")
    selection = dict(public.get("queue_selection") or {})
    prefix = selection.pop("prefix_cell_ids", None)
    if not isinstance(prefix, list):
        raise ValueError("task-budget private receipt has no frozen selection prefix")
    selection["prefix_count"] = len(prefix)
    public["queue_selection"] = selection
    public["schema"] = BUDGET_SANITIZED_SCHEMA
    public["private_receipt_sha256"] = private_sha
    public["receipt_sha256"] = _public_hash(public)
    return public


def _validated_starvation_policy(
    policy: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy, _target, _alpha = _validated_policy(policy)
    stopping = policy.get("candidate_starvation")
    if (not isinstance(stopping, dict)
            or stopping.get("output_schema") !=
                "campaignx.first_letters_campaign_decision.v1"
            or stopping.get("minimum_scientific_terminal_attempts") != 8
            or stopping.get("evaluation_block_size") != 8
            or stopping.get("pause_no_m7_count") != 7):
        raise ValueError("campaign starvation policy is unsupported")
    return policy, stopping


def _decision_receipt_hash(value: dict[str, Any]) -> str:
    return content_sha256({
        key: row for key, row in value.items()
        if key not in {"generated_at_utc", "receipt_sha256"}
    })


def _scientific_terminal_attempt(
    row: dict[str, Any], *, mission_id: str, policy_version: str,
) -> tuple[str, str | None] | None:
    if (row.get("mission_id") != mission_id
            or row.get("policy_version") != policy_version):
        return None
    if row.get("_malformed_attempt_identity") is True:
        return "EXCLUDED", "CONFIGURATION_BLOCK:MALFORMED_ATTEMPT_IDENTITY"
    state = row.get("state")
    result = row.get("result")
    failure_class = (
        result.get("failure_class") if isinstance(result, dict) else None)
    excluded_states = {
        "CANCELLED": "CANCELLED",
        "POLICY_REJECTED": "CONFIGURATION_BLOCK",
        "LEASE_EXPIRED": "LEASE_EXHAUSTION",
        "LEASE_EXHAUSTED": "LEASE_EXHAUSTION",
        "FINALIZATION_FAILED": "PUBLICATION_FAILURE",
        "BLOCKED_SOURCE_UNAVAILABLE": "SOURCE_FAILURE",
        "BLOCKED_PROBE_ARTIFACT_UNAVAILABLE": "SOURCE_FAILURE",
        "PROBE_TECHNICAL_FAILURE": "WORKER_FAILURE",
        "GROW_FAILED": "WORKER_FAILURE",
        "FIXTURE_ONLY": "FIXTURE_ONLY",
    }
    if state != "NO_SEED":
        scientific_states = {
            "ARCHIVED", "QC_PENDING", "DUPLICATE_SURFACE",
            "PROBE_REVIEW_PENDING", "PROBE_REJECTED_ALL",
        }
        if state in scientific_states:
            if failure_class is not None:
                return (
                    "EXCLUDED",
                    "CONFIGURATION_BLOCK:INCONSISTENT_TERMINAL_FAILURE_CLASS:"
                    f"{state}:{failure_class}",
                )
            return "OTHER_SCIENTIFIC", None
        if state in excluded_states:
            canonical = excluded_states[str(state)]
            if failure_class != canonical:
                return (
                    "EXCLUDED",
                    "CONFIGURATION_BLOCK:INCONSISTENT_TERMINAL_FAILURE_CLASS:"
                    f"{state}:{failure_class}",
                )
            return "EXCLUDED", canonical
        active_states = {
            "PENDING", "CLAIMED", "PLANNING", "PROBING", "LOCKED_READY",
            "RUNNING", "UPLOADED", "FINALIZING", "RETRY_ON_LARGER_GPU",
        }
        if state in active_states or not row.get("terminal_at_utc"):
            return None
        return (
            "EXCLUDED",
            f"CONFIGURATION_BLOCK:UNKNOWN_TERMINAL_STATE:{state}",
        )
    if failure_class is not None:
        return (
            "EXCLUDED",
            "CONFIGURATION_BLOCK:INCONSISTENT_NO_SEED_FAILURE_CLASS:"
            f"{failure_class}",
        )
    if not isinstance(result, dict):
        return "EXCLUDED", "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS"
    diagnosis = result.get("no_seed_causal_diagnosis")
    if not isinstance(diagnosis, dict):
        return "EXCLUDED", "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS"
    digest = diagnosis.get("diagnosis_sha256")
    core = {
        key: value for key, value in diagnosis.items()
        if key not in {"generated_at_utc", "diagnosis_sha256"}
    }
    raw = diagnosis.get("m7_raw_candidate_count")
    cause_counts = diagnosis.get("cause_counts")
    primary_causes = diagnosis.get("primary_causes")
    valid_cause_counts = (
        isinstance(cause_counts, dict)
        and all(isinstance(key, str) and key
                and isinstance(value, int) and not isinstance(value, bool)
                and value >= 0 for key, value in cause_counts.items())
    )
    expected_primary = (
        sorted(key for key, value in cause_counts.items() if value > 0)
        if valid_cause_counts else None
    )
    if (diagnosis.get("schema") !=
            "campaignx.no_seed_causal_diagnosis.v1"
            or diagnosis.get("status") != "NO_SEED"
            or diagnosis.get("task_id") != row.get("task_id")
            or diagnosis.get("attempt_id") != row.get("attempt_id")
            or diagnosis.get("ink_used") is not False
            or result.get("status") != "NO_SEED"
            or result.get("ink_used") is not False
            or digest != content_sha256(core)
            or result.get("no_seed_causal_diagnosis_sha256") != digest
            or not isinstance(raw, int) or isinstance(raw, bool) or raw < 0
            or result.get("raw_candidate_count") != raw
            or not valid_cause_counts
            or primary_causes != expected_primary
            or result.get("no_seed_cause_counts") != cause_counts
            or result.get("primary_causes") != primary_causes):
        return "EXCLUDED", "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS"
    causal_labels = set(cause_counts)
    if causal_labels - NO_SEED_SCIENTIFIC_CAUSAL_LABELS:
        return (
            "EXCLUDED",
            "CONFIGURATION_BLOCK:UNKNOWN_NO_SEED_CAUSAL_LABEL",
        )
    if NO_SEED_SCIENTIFIC_CAUSAL_LABELS - causal_labels:
        return (
            "EXCLUDED",
            "CONFIGURATION_BLOCK:MISSING_NO_SEED_CAUSAL_LABEL",
        )
    if cause_counts["NO_M7_CANDIDATES"] != (1 if raw == 0 else 0):
        return "EXCLUDED", "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS"
    stage_counts = {
        key: diagnosis.get(key) for key in (
            "ct_support_input_candidate_count",
            "ct_support_retained_candidate_count",
            "ct_support_rejected_candidate_count",
            "post_ct_candidate_count",
            "eligible_after_clearance_count",
        )
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in stage_counts.values()):
        return "EXCLUDED", "CONFIGURATION_BLOCK:MALFORMED_CAUSAL_DIAGNOSIS"
    ct_input = stage_counts["ct_support_input_candidate_count"]
    ct_retained = stage_counts["ct_support_retained_candidate_count"]
    ct_rejected = stage_counts["ct_support_rejected_candidate_count"]
    post_ct = stage_counts["post_ct_candidate_count"]
    eligible = stage_counts["eligible_after_clearance_count"]
    clearance_rejections = sum(
        value for key, value in cause_counts.items()
        if key not in {"NO_M7_CANDIDATES", "CT_MATERIAL_SUPPORT_REJECTED"}
    )
    impossible = (
        ct_input != raw
        or ct_retained > ct_input
        or ct_rejected != ct_input - ct_retained
        or post_ct != ct_retained
        or eligible != 0
        or cause_counts.get("CT_MATERIAL_SUPPORT_REJECTED") != ct_rejected
        or result.get("post_ct_candidate_count") != post_ct
        or result.get("usable_candidate_count") != 0
        or (raw == 0 and any(
            value != 0 for key, value in cause_counts.items()
            if key != "NO_M7_CANDIDATES"))
        or (post_ct > 0 and clearance_rejections < post_ct)
    )
    if impossible:
        return (
            "EXCLUDED",
            "CONFIGURATION_BLOCK:IMPOSSIBLE_NO_SEED_CAUSAL_COUNTS",
        )
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return ("NO_M7_CANDIDATES" if raw == 0 else "OTHER_SCIENTIFIC", None)
    return "EXCLUDED", "CONFIGURATION_BLOCK:MISSING_RAW_M7_COUNT"


def derive_campaign_decision_receipts(
    attempts: list[dict[str, Any]], admissions: list[dict[str, Any]],
    policy: dict[str, Any], *, mission_id: str, policy_version: str,
) -> list[dict[str, Any]]:
    """Derive immutable decisions at complete scientific-terminal blocks."""
    policy, stopping = _validated_starvation_policy(policy)
    if (not isinstance(attempts, list)
            or not isinstance(mission_id, str) or not mission_id
            or not isinstance(policy_version, str) or not policy_version):
        raise ValueError("campaign decision scope or attempts are invalid")
    scoped = [copy.deepcopy(row) for row in attempts
              if (isinstance(row, dict)
                  and row.get("mission_id") == mission_id
                  and row.get("policy_version") == policy_version)]
    latest_by_task: dict[str, dict[str, Any]] = {}
    superseded_attempt_ids: list[str] = []
    active_states = {
        "PENDING", "CLAIMED", "PLANNING", "PROBING", "LOCKED_READY",
        "RUNNING", "UPLOADED", "FINALIZING", "RETRY_ON_LARGER_GPU",
    }
    for position, row in enumerate(scoped):
        task_id = str(row.get("task_id") or "")
        attempt_number = row.get("attempt_number")
        if (not task_id or not isinstance(attempt_number, int)
                or isinstance(attempt_number, bool) or attempt_number < 1):
            if row.get("state") in active_states:
                continue
            row["_malformed_attempt_identity"] = True
            latest_by_task[f"__malformed_attempt_identity__:{position}"] = row
            continue
        previous = latest_by_task.get(task_id)
        rank = (
            attempt_number,
            str(row.get("terminal_at_utc") or ""),
            str(row.get("attempt_id") or ""),
        )
        if previous is None:
            latest_by_task[task_id] = row
            continue
        previous_rank = (
            previous.get("attempt_number"),
            str(previous.get("terminal_at_utc") or ""),
            str(previous.get("attempt_id") or ""),
        )
        if rank > previous_rank:
            superseded_attempt_ids.append(str(previous.get("attempt_id") or ""))
            latest_by_task[task_id] = row
        else:
            superseded_attempt_ids.append(str(row.get("attempt_id") or ""))
    superseded_attempt_ids = sorted(
        attempt_id for attempt_id in superseded_attempt_ids if attempt_id)
    ordered = sorted(
        latest_by_task.values(),
        key=lambda row: (
            str(row.get("terminal_at_utc") or ""),
            str(row.get("attempt_id") or ""),
        ),
    )
    ordered_admissions = sorted(
        (
            copy.deepcopy(row) for row in admissions
            if (isinstance(row, dict)
                and row.get("mission_id") == mission_id
                and (row.get("execution_bindings") or {}).get(
                    "policy_version") == policy_version)
        ),
        key=lambda row: (
            str(row.get("registered_at_utc") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("receipt_sha256") or ""),
        ),
    )
    authorities_by_sha: dict[str, dict[str, Any]] = {}
    for admission in ordered_admissions:
        authority = {
            key: value for key, value in admission.items()
            if key != "registered_at_utc"
        }
        digest = admission.get("admission_sha256")
        if (isinstance(digest, str)
                and digest == content_sha256({
                    key: value for key, value in authority.items()
                    if key != "admission_sha256"
                })):
            authorities_by_sha[digest] = authority

    classified: list[tuple[dict[str, Any], str, str | None]] = []
    scientific: list[tuple[dict[str, Any], str]] = []
    for row in ordered:
        classification = _scientific_terminal_attempt(
            row, mission_id=mission_id, policy_version=policy_version)
        if classification is None:
            continue
        if classification[0] != "EXCLUDED":
            envelope = row.get("campaign_budget")
            authority = (
                authorities_by_sha.get(envelope.get("admission_sha256"))
                if isinstance(envelope, dict) else None
            )
            if (authority is None
                    or not campaign_budget_task_matches_admission(row, authority)):
                classification = (
                    "EXCLUDED",
                    "CONFIGURATION_BLOCK:UNBOUND_CAMPAIGN_BUDGET_ADMISSION",
                )
            else:
                row["_governing_admission_sha256"] = authority[
                    "admission_sha256"]
                row["_governing_budget_receipt_sha256"] = authority[
                    "receipt_sha256"]
        classified.append((row, classification[0], classification[1]))
        if classification[0] == "EXCLUDED":
            continue
        scientific.append((row, classification[0]))
    completed: list[dict[str, Any] | None] = []
    for admission in ordered_admissions:
        approved = admission.get("approved_task_count")
        prefix = admission.get("prefix_cell_ids")
        authority = {
            key: value for key, value in admission.items()
            if key != "registered_at_utc"
        }
        if (admission.get("schema") !=
                "campaignx.first_letters_task_budget_admission.v1"
                or admission.get("admission_sha256") != content_sha256({
                    key: value for key, value in authority.items()
                    if key != "admission_sha256"
                })
                or not isinstance(approved, int) or isinstance(approved, bool)
                or approved < 1 or not isinstance(prefix, list)
                or len(prefix) != approved):
            completed.append(None)
            continue
        by_rank: dict[int, tuple[dict[str, Any], str]] = {}
        for row, classification in scientific:
            envelope = row.get("campaign_budget")
            if (not isinstance(envelope, dict)
                    or envelope.get("receipt_sha256") !=
                        admission.get("receipt_sha256")
                    or row.get("sample_id") != admission.get("sample_id")):
                continue
            rank = envelope.get("selection_rank")
            if (not campaign_budget_task_matches_admission(row, authority)
                    or not isinstance(rank, int) or isinstance(rank, bool)
                    or rank in by_rank):
                continue
            by_rank[rank] = (row, classification)
        if set(by_rank) != set(range(approved)):
            completed.append(None)
            continue
        ranked = [by_rank[rank] for rank in range(approved)]
        completed.append({
            "sample_id": admission["sample_id"],
            "budget_receipt_sha256": admission["receipt_sha256"],
            "attempts": ranked,
            "zero_raw_m7": all(
                classification == "NO_M7_CANDIDATES"
                for _row, classification in ranked),
        })
    pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    pair_index = 0
    for index, (first, second) in enumerate(
        zip(completed, completed[1:]), start=1,
    ):
        if (first is not None and second is not None
                and first["sample_id"] != second["sample_id"]
                and first["zero_raw_m7"] and second["zero_raw_m7"]):
            pair = first, second
            pair_index = index
            break
    trigger_rows = (
        [row for item in pair for row, _classification in item["attempts"]]
        if pair is not None else []
    )
    cross_scroll_cutoff = (
        max((str(row.get("terminal_at_utc") or ""),
             str(row.get("attempt_id") or "")) for row in trigger_rows)
        if trigger_rows else None
    )
    block_size = int(stopping["evaluation_block_size"])
    receipts: list[dict[str, Any]] = []

    def bound_attempt(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "attempt_id": row["attempt_id"],
            "task_id": row["task_id"],
            "sample_id": row["sample_id"],
            "admission_sha256": row["_governing_admission_sha256"],
            "budget_receipt_sha256": row[
                "_governing_budget_receipt_sha256"],
        }

    for start in range(0, len(scientific) - block_size + 1, block_size):
        block = scientific[start:start + block_size]
        lower_key = (
            (str(scientific[start - 1][0].get("terminal_at_utc") or ""),
             str(scientific[start - 1][0].get("attempt_id") or ""))
            if start else None
        )
        upper_key = (
            str(block[-1][0].get("terminal_at_utc") or ""),
            str(block[-1][0].get("attempt_id") or ""),
        )
        if cross_scroll_cutoff is not None and cross_scroll_cutoff <= upper_key:
            break
        excluded = []
        for row, classification, reason in classified:
            key = (str(row.get("terminal_at_utc") or ""),
                   str(row.get("attempt_id") or ""))
            if (classification == "EXCLUDED"
                    and (lower_key is None or key > lower_key)
                    and key <= upper_key):
                excluded.append({
                    "attempt_id": row.get("attempt_id"),
                    "task_id": row.get("task_id"),
                    "reason": reason,
                })
        no_m7 = [row for row, classification in block
                 if classification == "NO_M7_CANDIDATES"]
        trigger_governing_admissions = sorted({
            row["_governing_admission_sha256"] for row in no_m7
        })
        ambiguous_trigger_authority = (
            len(no_m7) >= int(stopping["pause_no_m7_count"])
            and len(trigger_governing_admissions) != 1
        )
        control_incomplete_reasons = (
            ["AMBIGUOUS_TRIGGER_GOVERNING_ADMISSIONS"]
            if ambiguous_trigger_authority else []
        )
        evidence_incomplete = ambiguous_trigger_authority or any(
            ":" in str(row.get("reason") or "") for row in excluded)
        paused = (
            not evidence_incomplete
            and len(no_m7) >= int(stopping["pause_no_m7_count"])
        )
        decision = (
            "CONTROL_INCOMPLETE" if evidence_incomplete
            else "PAUSE_CANDIDATE_STARVATION" if paused else "CONTINUE"
        )
        receipt: dict[str, Any] = {
            "schema": stopping["output_schema"],
            "decision": decision,
            "evidence_status": (
                "INCOMPLETE" if evidence_incomplete else "COMPLETE"),
            "mission_id": mission_id,
            "policy_version": policy_version,
            "campaign_policy_profile_id": policy["profile_id"],
            "campaign_policy_sha256": content_sha256(policy),
            "evaluation_kind": "SCIENTIFIC_TERMINAL_BLOCK",
            "evaluation_index": len(receipts) + 1,
            "scientific_terminal_attempt_ids": [
                row["attempt_id"] for row, _classification in block],
            "scientific_terminal_attempts": [
                bound_attempt(row) for row, _classification in block],
            "trigger_governing_admission_sha256s": (
                trigger_governing_admissions),
            "governing_admission_sha256s": (
                trigger_governing_admissions if paused else []),
            "no_m7_numerator": len(no_m7),
            "scientific_terminal_denominator": len(block),
            "excluded_attempt_count": len(excluded),
            "excluded_attempts": excluded,
            "control_incomplete_reasons": control_incomplete_reasons,
            "superseded_attempt_ids": superseded_attempt_ids,
            "trigger_attempt_ids": (
                [row["attempt_id"] for row in no_m7] if paused else []),
            "completed_zero_raw_m7_scrolls": [],
            "allowed_next_actions": (
                ["CREATE_MATERIALLY_CHANGED_VERSIONED_STRATEGY",
                 "CLOSE_CAMPAIGN"]
                if paused else ["QUEUE_NEXT_BOUND_WAVE", "CLOSE_CAMPAIGN"]),
            "pause_effect": (
                "BLOCK_NEW_P1_WITHOUT_CANCELLING_OR_REPRIORITIZING_EXISTING_WORK"
                if paused else None),
            "non_claim": (
                "Candidate starvation is not evidence of surface, ink, text, "
                "or letter absence."),
            "generated_at_utc": utc_now(),
        }
        if evidence_incomplete:
            receipt["allowed_next_actions"] = [
                "REPAIR_OR_REPLAY_CAUSAL_EVIDENCE", "CLOSE_CAMPAIGN"]
        receipt["receipt_sha256"] = _decision_receipt_hash(receipt)
        receipts.append(receipt)
        if paused:
            break
    if receipts and receipts[-1]["decision"] == "PAUSE_CANDIDATE_STARVATION":
        return receipts

    if pair is not None:
        receipt = {
            "schema": stopping["output_schema"],
            "decision": "PAUSE_CANDIDATE_STARVATION",
            "evidence_status": "COMPLETE",
            "mission_id": mission_id,
            "policy_version": policy_version,
            "campaign_policy_profile_id": policy["profile_id"],
            "campaign_policy_sha256": content_sha256(policy),
            "evaluation_kind": "CONSECUTIVE_ZERO_RAW_M7_SCROLL_BUDGETS",
            "evaluation_index": pair_index,
            "scientific_terminal_attempt_ids": [
                row["attempt_id"] for row in trigger_rows],
            "scientific_terminal_attempts": [
                bound_attempt(row) for row in trigger_rows],
            "trigger_governing_admission_sha256s": sorted({
                row["_governing_admission_sha256"] for row in trigger_rows
            }),
            "governing_admission_sha256s": sorted({
                row["_governing_admission_sha256"]
                for row, _classification in pair[1]["attempts"]
            }),
            "no_m7_numerator": len(trigger_rows),
            "scientific_terminal_denominator": len(trigger_rows),
            "excluded_attempt_count": 0,
            "excluded_attempts": [],
            "control_incomplete_reasons": [],
            "superseded_attempt_ids": superseded_attempt_ids,
            "trigger_attempt_ids": [row["attempt_id"] for row in trigger_rows],
            "completed_zero_raw_m7_scrolls": [
                {"sample_id": item["sample_id"],
                 "budget_receipt_sha256": item["budget_receipt_sha256"]}
                for item in pair],
            "allowed_next_actions": [
                "CREATE_MATERIALLY_CHANGED_VERSIONED_STRATEGY",
                "CLOSE_CAMPAIGN"],
            "pause_effect": (
                "BLOCK_NEW_P1_WITHOUT_CANCELLING_OR_REPRIORITIZING_EXISTING_WORK"),
            "non_claim": (
                "Candidate starvation is not evidence of surface, ink, text, "
                "or letter absence."),
            "generated_at_utc": utc_now(),
        }
        receipt["receipt_sha256"] = _decision_receipt_hash(receipt)
        receipts.append(receipt)
    return receipts


def _resume_causal_value(admission: dict[str, Any], field: str) -> object:
    execution = admission.get("execution_bindings") or {}
    queue = execution.get("queue_execution") or {}
    gates = execution.get("gates") or {}
    values = {
        "m7_source": {
            "m7_sha256": execution.get("m7_sha256"),
            "m7_uri_sha256": execution.get("m7_uri_sha256"),
        },
        "calibrated_m7_threshold": execution.get("m7_threshold"),
        "grid_version": execution.get("grid_version"),
        "discovery_provider": execution.get("provider"),
        "authorized_seed_probe_mode": queue.get("seed_probe_mode"),
        "evidence_backed_clearance_policy": {
            key: copy.deepcopy(gates.get(key)) for key in (
                "cell_clearance", "volume_clearance",
                "candidate_interior_clearance", "ct_material_support_gate",
            )
        },
    }
    if field not in values:
        raise ValueError(
            f"{field!r} is not a material candidate-starvation causal input")
    return values[field]


def campaign_resume_material_evidence_sha256(
    prior_admission: dict[str, Any], new_admission: dict[str, Any], field: str,
) -> str:
    """Derive evidence for directly comparable causal changes from authority.

    Threshold, seed-probe, and clearance changes need external scientific
    validators and therefore deliberately have no hash-only fallback here.
    """
    if field not in {"m7_source", "grid_version", "discovery_provider"}:
        raise ValueError(
            f"{field!r} requires a verified external evidence validator")
    if (field == "m7_source"
            and prior_admission.get("sample_id") !=
                new_admission.get("sample_id")):
        raise ValueError(
            "m7_source is scroll-local and can only resume the same sample; "
            "cross-sample discovery requires a global grid or provider change")
    prior_value = _resume_causal_value(prior_admission, field)
    new_value = _resume_causal_value(new_admission, field)
    evidence = {
        "schema": "campaignx.first_letters_resume_material_evidence.v1",
        "field": field,
        "prior_admission_sha256": prior_admission.get("admission_sha256"),
        "new_admission_sha256": new_admission.get("admission_sha256"),
        "prior_budget_receipt_sha256": prior_admission.get("receipt_sha256"),
        "new_budget_receipt_sha256": new_admission.get("receipt_sha256"),
        "prior_preflight_receipt_sha256": prior_admission.get(
            "preflight_receipt_sha256"),
        "new_preflight_receipt_sha256": new_admission.get(
            "preflight_receipt_sha256"),
        "prior_preflight_sanitized_receipt_sha256": prior_admission.get(
            "preflight_sanitized_receipt_sha256"),
        "new_preflight_sanitized_receipt_sha256": new_admission.get(
            "preflight_sanitized_receipt_sha256"),
        "prior_value_sha256": content_sha256(
            {"field": field, "value": prior_value}),
        "new_value_sha256": content_sha256(
            {"field": field, "value": new_value}),
    }
    required_hashes = [
        value for key, value in evidence.items()
        if key.endswith("_sha256")
    ]
    if any(not isinstance(value, str)
           or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in required_hashes):
        raise ValueError(
            "campaign resume material evidence lacks its exact preflight/budget pair")
    return content_sha256(evidence)


def validate_authorized_seed_probe_mode_evidence(
    value: object, *, prior_admission: dict[str, Any],
    new_admission: dict[str, Any], prior_decision: dict[str, Any],
    policy: dict[str, Any], benchmark_authorization_v2: dict[str, Any],
) -> dict[str, Any]:
    """Validate retained benchmark-v2 evidence for a Task 5 mode resume."""

    required = {
        "schema", "prior_decision_receipt_sha256", "prior_admission_sha256",
        "new_admission_sha256", "predecessor_policy_file_sha256",
        "successor_policy_file_sha256", "old_seed_probe_mode",
        "new_seed_probe_mode", "unchanged_source_grid_sha256",
        "benchmark_authorization_v2", "review_owner", "allow_unvalidated",
        "evidence_sha256",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") !=
                "campaignx.first_letters_authorized_seed_probe_mode_evidence.v1"
            or value.get("allow_unvalidated") is not False):
        raise ValueError("authorized seed-probe evidence differs from its closed schema")
    expected_hash = content_sha256({
        key: row for key, row in value.items() if key != "evidence_sha256"
    })
    if value.get("evidence_sha256") != expected_hash:
        raise ValueError("authorized seed-probe evidence hash is invalid")
    prior_mode = _resume_causal_value(prior_admission, "authorized_seed_probe_mode")
    new_mode = _resume_causal_value(new_admission, "authorized_seed_probe_mode")
    if prior_mode is None:
        prior_mode = prior_admission.get("seed_probe_mode")
    if new_mode is None:
        new_mode = new_admission.get("seed_probe_mode")
    prior_source_grid = prior_admission.get("source_grid_sha256")
    new_source_grid = new_admission.get("source_grid_sha256")
    if (value["prior_decision_receipt_sha256"] != prior_decision.get("receipt_sha256")
            or value["prior_admission_sha256"] != prior_admission.get("admission_sha256")
            or value["new_admission_sha256"] != new_admission.get("admission_sha256")
            or value["predecessor_policy_file_sha256"] !=
                policy.get("predecessor_profile_file_sha256")
            or value["successor_policy_file_sha256"] != policy.get("profile_file_sha256")
            or value["old_seed_probe_mode"] != prior_mode
            or value["new_seed_probe_mode"] != new_mode
            or prior_mode == new_mode
            or new_mode != "select"
            or value["unchanged_source_grid_sha256"] != prior_source_grid
            or prior_source_grid != new_source_grid
            or value["benchmark_authorization_v2"] != benchmark_authorization_v2
            or benchmark_authorization_v2.get("schema") !=
                "campaignx.seed_probe_benchmark_authorization.v2"
            or not isinstance(value["review_owner"], str)
            or not value["review_owner"].strip()):
        raise ValueError("authorized seed-probe evidence authority drift")
    return copy.deepcopy(value)


def _campaign_resume_authorization_evidence(
    value: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "campaignx.first_letters_resume_authorization_evidence.v1",
        **{key: value.get(key) for key in (
            "mission_id", "prior_sample_id", "new_sample_id",
            "prior_policy_version", "new_policy_version",
            "prior_admission_sha256", "new_admission_sha256",
            "prior_decision_receipt_sha256", "material_changes",
            "authorized_by", "authentication_context",
        )},
    }


def bind_campaign_resume_authorization_principal(
    proposal: dict[str, Any], *, authorized_by: str,
    session_fingerprint_sha256: str,
    request_method: str, request_path: str,
) -> dict[str, Any]:
    """Replace caller-authored identity with the authenticated panel context."""
    if (not isinstance(proposal, dict)
            or proposal.get("schema") !=
                "campaignx.first_letters_campaign_resume_authorization.v1"
            or proposal.get("authorization_sha256") != content_sha256({
                key: value for key, value in proposal.items()
                if key != "authorization_sha256"
            })
            or not isinstance(authorized_by, str) or not authorized_by.strip()
            or re.fullmatch(r"[0-9a-f]{64}", session_fingerprint_sha256) is None
            or request_method != "POST"
            or request_path != "/api/segmentation/runs"):
        raise ValueError(
            "campaign resume proposal or authenticated panel context is invalid")
    bound = copy.deepcopy(proposal)
    for key in (
        "authorization_sha256", "authorization_evidence_sha256",
        "authentication_context",
    ):
        bound.pop(key, None)
    bound["authorized_by"] = authorized_by.strip()
    bound["authentication_context"] = {
        "mechanism": "HELENA_AUTHENTICATED_PANEL_SESSION",
        "principal": authorized_by.strip(),
        "session_fingerprint_sha256": session_fingerprint_sha256,
        "request_method": request_method,
        "request_path": request_path,
    }
    bound["authorization_evidence_sha256"] = content_sha256(
        _campaign_resume_authorization_evidence(bound))
    bound["authorization_sha256"] = content_sha256(bound)
    return bound


def _validate_pause_trigger_admission_provenance(
    decision: dict[str, Any], *, pause_no_m7_count: int,
) -> None:
    attempts = decision.get("scientific_terminal_attempts")
    trigger_ids = decision.get("trigger_attempt_ids")
    trigger_admissions = decision.get(
        "trigger_governing_admission_sha256s")
    governing_admissions = decision.get("governing_admission_sha256s")
    if (not isinstance(attempts, list) or not attempts
            or not isinstance(trigger_ids, list) or not trigger_ids
            or any(not isinstance(attempt_id, str) or not attempt_id
                   for attempt_id in trigger_ids)
            or len(set(trigger_ids)) != len(trigger_ids)):
        raise ValueError("campaign pause trigger admission provenance is invalid")
    attempts_by_id: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        attempt_id = attempt.get("attempt_id") if isinstance(attempt, dict) else None
        admission_sha256 = (
            attempt.get("admission_sha256")
            if isinstance(attempt, dict) else None
        )
        if (not isinstance(attempt_id, str) or not attempt_id
                or attempt_id in attempts_by_id
                or re.fullmatch(r"[0-9a-f]{64}", str(admission_sha256 or ""))
                    is None
                or not isinstance(attempt.get("sample_id"), str)
                or not attempt["sample_id"]):
            raise ValueError(
                "campaign pause trigger admission provenance is invalid")
        attempts_by_id[attempt_id] = attempt
    if any(attempt_id not in attempts_by_id for attempt_id in trigger_ids):
        raise ValueError("campaign pause trigger admission provenance is invalid")
    derived_trigger_admissions = sorted({
        attempts_by_id[attempt_id]["admission_sha256"]
        for attempt_id in trigger_ids
    })
    if (trigger_admissions != derived_trigger_admissions
            or not isinstance(governing_admissions, list)):
        raise ValueError("campaign pause trigger admission provenance is invalid")

    evaluation_kind = decision.get("evaluation_kind")
    if evaluation_kind == "SCIENTIFIC_TERMINAL_BLOCK":
        if (len(trigger_ids) < pause_no_m7_count
                or len(derived_trigger_admissions) != 1):
            raise ValueError(
                "campaign pause trigger admission provenance is invalid")
        expected_governing = derived_trigger_admissions
    elif evaluation_kind == "CONSECUTIVE_ZERO_RAW_M7_SCROLL_BUDGETS":
        completed = decision.get("completed_zero_raw_m7_scrolls")
        if (not isinstance(completed, list) or len(completed) != 2
                or any(not isinstance(item, dict) for item in completed)
                or completed[0].get("sample_id") == completed[1].get("sample_id")):
            raise ValueError(
                "campaign pause trigger admission provenance is invalid")
        completing_sample = completed[1].get("sample_id")
        expected_governing = sorted({
            attempts_by_id[attempt_id]["admission_sha256"]
            for attempt_id in trigger_ids
            if attempts_by_id[attempt_id]["sample_id"] == completing_sample
        })
        if len(expected_governing) != 1:
            raise ValueError(
                "campaign pause trigger admission provenance is invalid")
    else:
        raise ValueError("campaign pause trigger admission provenance is invalid")
    if governing_admissions != expected_governing:
        raise ValueError("campaign pause trigger admission provenance is invalid")


def validate_campaign_resume_authorization(
    value: object, *, prior_admission: dict[str, Any],
    new_admission: dict[str, Any], prior_decision: dict[str, Any],
    policy: dict[str, Any],
    authoritative_attempts: list[dict[str, Any]] | None = None,
    registered_admissions: list[dict[str, Any]] | None = None,
    trusted_authorization_sha256s: set[str] | None = None,
    external_material_evidence_by_field: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind a paused policy's replacement to evidence of a causal change."""
    policy, stopping = _validated_starvation_policy(policy)
    resume = stopping.get("resume_requires")
    if (not isinstance(resume, dict)
            or resume.get("new_policy_version") is not True
            or resume.get("planner_only_change_is_material_for_no_m7") is not False
            or not isinstance(resume.get("at_least_one_material_change"), list)):
        raise ValueError("campaign resume policy is unsupported")
    required = {
        "schema", "mission_id", "prior_sample_id", "new_sample_id",
        "prior_policy_version", "new_policy_version", "prior_admission_sha256",
        "new_admission_sha256", "prior_decision_receipt_sha256",
        "material_changes", "authorized_by", "authentication_context",
        "authorization_evidence_sha256", "authorization_sha256",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") !=
                "campaignx.first_letters_campaign_resume_authorization.v1"
            or value.get("authorization_sha256") != content_sha256({
                key: row for key, row in value.items()
                if key != "authorization_sha256"
            })):
        raise ValueError("campaign resume authorization hash or schema is invalid")
    if (not isinstance(trusted_authorization_sha256s, set)
            or value["authorization_sha256"] not in
                trusted_authorization_sha256s):
        raise ValueError(
            "campaign resume authorization has no trusted principal attestation")
    expected_authorization_evidence = content_sha256(
        _campaign_resume_authorization_evidence(value))
    authentication_context = value.get("authentication_context")
    if (not isinstance(value.get("authorized_by"), str)
            or not value["authorized_by"].strip()
            or not isinstance(authentication_context, dict)
            or set(authentication_context) != {
                "mechanism", "principal", "session_fingerprint_sha256",
                "request_method", "request_path",
            }
            or authentication_context.get("mechanism") !=
                "HELENA_AUTHENTICATED_PANEL_SESSION"
            or authentication_context.get("principal") != value["authorized_by"]
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(authentication_context.get(
                    "session_fingerprint_sha256") or ""),
            ) is None
            or authentication_context.get("request_method") != "POST"
            or authentication_context.get("request_path") !=
                "/api/segmentation/runs"
            or value.get("authorization_evidence_sha256") !=
                expected_authorization_evidence):
        raise ValueError("campaign resume authorization evidence is incomplete")
    prior_execution = prior_admission.get("execution_bindings") or {}
    new_execution = new_admission.get("execution_bindings") or {}
    mission_id = prior_admission.get("mission_id")
    if (mission_id != new_admission.get("mission_id")
            or value.get("mission_id") != mission_id
            or value.get("prior_sample_id") != prior_admission.get("sample_id")
            or value.get("new_sample_id") != new_admission.get("sample_id")
            or value.get("prior_policy_version") !=
                prior_execution.get("policy_version")
            or value.get("new_policy_version") != new_execution.get("policy_version")
            or value.get("prior_policy_version") == value.get("new_policy_version")
            or value.get("prior_admission_sha256") !=
                prior_admission.get("admission_sha256")
            or value.get("new_admission_sha256") !=
                new_admission.get("admission_sha256")):
        raise ValueError(
            "campaign resume authorization names another scope, policy, or admission")
    if (prior_decision.get("schema") != stopping["output_schema"]
            or prior_decision.get("decision") != "PAUSE_CANDIDATE_STARVATION"
            or prior_decision.get("mission_id") != mission_id
            or prior_decision.get("policy_version") !=
                prior_execution.get("policy_version")
            or prior_decision.get("receipt_sha256") !=
                _decision_receipt_hash(prior_decision)
            or value.get("prior_decision_receipt_sha256") !=
                prior_decision.get("receipt_sha256")):
        raise ValueError("campaign resume authorization is not bound to the pause")
    if (not isinstance(authoritative_attempts, list)
            or not isinstance(registered_admissions, list)):
        raise ValueError(
            "campaign resume lacks authoritative persisted pause provenance")
    authoritative_prior_admissions = []
    for authority in registered_admissions:
        if not isinstance(authority, dict):
            continue
        canonical = {
            key: row for key, row in authority.items()
            if key != "registered_at_utc"
        }
        if (canonical.get("admission_sha256") ==
                prior_admission.get("admission_sha256")):
            authoritative_prior_admissions.append(canonical)
    canonical_prior = {
        key: row for key, row in prior_admission.items()
        if key != "registered_at_utc"
    }
    if (len(authoritative_prior_admissions) != 1
            or authoritative_prior_admissions[0] != canonical_prior):
        raise ValueError(
            "campaign resume lacks authoritative persisted pause provenance")
    authoritative_receipts = derive_campaign_decision_receipts(
        authoritative_attempts,
        registered_admissions,
        policy,
        mission_id=str(mission_id),
        policy_version=str(prior_execution.get("policy_version") or ""),
    )
    authoritative_pauses = [
        receipt for receipt in authoritative_receipts
        if (receipt.get("decision") == "PAUSE_CANDIDATE_STARVATION"
            and receipt.get("receipt_sha256") ==
                prior_decision.get("receipt_sha256"))
    ]
    stable_prior = {
        key: row for key, row in prior_decision.items()
        if key != "generated_at_utc"
    }
    if (len(authoritative_pauses) != 1
            or {
                key: row for key, row in authoritative_pauses[0].items()
                if key != "generated_at_utc"
            } != stable_prior):
        raise ValueError(
            "campaign resume is not bound to the authoritative persisted pause")
    authoritative_pause = authoritative_pauses[0]
    governing_admissions = authoritative_pause.get(
        "governing_admission_sha256s")
    trigger_governing_admissions = authoritative_pause.get(
        "trigger_governing_admission_sha256s")
    _validate_pause_trigger_admission_provenance(
        authoritative_pause,
        pause_no_m7_count=int(stopping["pause_no_m7_count"]),
    )
    if (not isinstance(governing_admissions, list)
            or len(governing_admissions) != 1
            or any(not isinstance(digest, str) for digest in governing_admissions)
            or not isinstance(trigger_governing_admissions, list)
            or not trigger_governing_admissions
            or any(not isinstance(digest, str)
                   for digest in trigger_governing_admissions)
            or not set(governing_admissions).issubset(
                set(trigger_governing_admissions))
            or prior_admission.get("admission_sha256") not in
                governing_admissions):
        raise ValueError(
            "campaign resume prior admission did not govern the active pause")
    allowed = set(resume["at_least_one_material_change"])
    changes = value.get("material_changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("campaign resume requires a material causal change")
    seen: set[str] = set()
    for change in changes:
        if (not isinstance(change, dict)
                or set(change) != {
                    "field", "prior_value_sha256", "new_value_sha256",
                    "evidence_sha256",
                }):
            raise ValueError("campaign resume material-change evidence is invalid")
        field = change.get("field")
        if not isinstance(field, str) or field not in allowed or field in seen:
            raise ValueError(
                "campaign resume field is not a unique material causal input; "
                "planner-only changes cannot resume NO_M7 starvation")
        seen.add(field)
        expected_prior = content_sha256({
            "field": field,
            "value": _resume_causal_value(prior_admission, field),
        })
        expected_new = content_sha256({
            "field": field,
            "value": _resume_causal_value(new_admission, field),
        })
        if field == "authorized_seed_probe_mode":
            external = (external_material_evidence_by_field or {}).get(field)
            if not isinstance(external, dict):
                raise ValueError(
                    "campaign resume material change lacks retained seed-probe evidence")
            benchmark = external.get("benchmark_authorization_v2")
            validated_external = validate_authorized_seed_probe_mode_evidence(
                external,
                prior_admission=prior_admission,
                new_admission=new_admission,
                prior_decision=prior_decision,
                policy=policy,
                benchmark_authorization_v2=benchmark,
            )
            expected_evidence = validated_external["evidence_sha256"]
        else:
            try:
                expected_evidence = campaign_resume_material_evidence_sha256(
                    prior_admission, new_admission, field)
            except ValueError as exc:
                raise ValueError(
                    "campaign resume material change lacks a verified evidence "
                    "validator") from exc
        if (change.get("prior_value_sha256") != expected_prior
                or change.get("new_value_sha256") != expected_new
                or expected_prior == expected_new
                or change.get("evidence_sha256") != expected_evidence):
            raise ValueError(
                "campaign resume material change is unchanged or lacks bound evidence")
    return copy.deepcopy(value)


def derive_campaign_active_policy_chain(
    admissions: list[dict[str, Any]], decisions: list[dict[str, Any]],
    authorizations: list[dict[str, Any]], *, mission_id: str,
) -> dict[str, Any] | None:
    """Return the single authorized policy chain and its active immutable gate."""
    scoped_admissions = [
        row for row in admissions
        if isinstance(row, dict) and row.get("mission_id") == mission_id
    ]
    if not scoped_admissions:
        return None
    policies: list[str] = []
    for row in scoped_admissions:
        policy_version = (row.get("execution_bindings") or {}).get(
            "policy_version")
        if not isinstance(policy_version, str) or not policy_version:
            raise ValueError("campaign admission has no policy version")
        if policy_version not in policies:
            policies.append(policy_version)
    successor_by_prior: dict[str, str] = {}
    predecessor_by_new: dict[str, str] = {}
    authorization_by_prior: dict[str, dict[str, Any]] = {}
    for row in authorizations:
        if not isinstance(row, dict) or row.get("mission_id") != mission_id:
            continue
        digest = row.get("authorization_sha256")
        if (digest != content_sha256({
                key: value for key, value in row.items()
                if key != "authorization_sha256"
            })):
            raise ValueError("campaign successor authorization hash is invalid")
        prior = row.get("prior_policy_version")
        new = row.get("new_policy_version")
        if (not isinstance(prior, str) or not prior
                or not isinstance(new, str) or not new or prior == new
                or prior not in policies or new not in policies
                or prior in successor_by_prior
                or new in predecessor_by_new):
            raise ValueError("campaign successor policy chain forks or cycles")
        successor_by_prior[prior] = new
        predecessor_by_new[new] = prior
        authorization_by_prior[prior] = row
    roots = [policy for policy in policies if policy not in predecessor_by_new]
    if len(roots) != 1:
        raise ValueError("campaign has no single active successor policy chain")
    chain: list[str] = []
    current = roots[0]
    while current not in chain:
        chain.append(current)
        successor = successor_by_prior.get(current)
        if successor is None:
            break
        current = successor
    else:
        raise ValueError("campaign successor policy chain contains a cycle")
    if set(policies) != set(chain):
        raise ValueError("campaign contains an unauthorized policy branch")
    active_policy = chain[-1]
    scoped_decisions = [
        row for row in decisions
        if isinstance(row, dict)
        and row.get("mission_id") == mission_id
        and row.get("policy_version") == active_policy
    ]
    blocking = next((
        row for row in scoped_decisions
        if row.get("decision") == "CONTROL_INCOMPLETE"
    ), None)
    if blocking is None:
        blocking = next((
            row for row in scoped_decisions
            if row.get("decision") == "PAUSE_CANDIDATE_STARVATION"
        ), None)
    return {
        "mission_id": mission_id,
        "policy_chain": chain,
        "active_policy_version": active_policy,
        "active_blocking_decision": copy.deepcopy(blocking),
        "successor_authorizations": copy.deepcopy(authorization_by_prior),
    }


def derive_campaign_active_decision(
    attempts: list[dict[str, Any]], admissions: list[dict[str, Any]],
    decisions: list[dict[str, Any]], authorizations: list[dict[str, Any]],
    policy: dict[str, Any], *, mission_id: str,
) -> dict[str, Any] | None:
    """Derive the live successor-policy gate without mutating receipt history."""
    policy, stopping = _validated_starvation_policy(policy)
    chain = derive_campaign_active_policy_chain(
        admissions, decisions, authorizations, mission_id=mission_id)
    if chain is None:
        return None
    blocking = chain["active_blocking_decision"]
    if blocking is not None:
        return copy.deepcopy(blocking)
    active_policy = chain["active_policy_version"]
    latest_by_task: dict[str, dict[str, Any]] = {}
    active_states = {
        "PENDING", "CLAIMED", "PLANNING", "PROBING", "LOCKED_READY",
        "RUNNING", "UPLOADED", "FINALIZING", "RETRY_ON_LARGER_GPU",
    }
    for position, original in enumerate(attempts):
        if (not isinstance(original, dict)
                or original.get("mission_id") != mission_id
                or original.get("policy_version") != active_policy):
            continue
        row = copy.deepcopy(original)
        task_id = row.get("task_id")
        attempt_number = row.get("attempt_number")
        if (not isinstance(task_id, str) or not task_id
                or not isinstance(attempt_number, int)
                or isinstance(attempt_number, bool) or attempt_number < 1):
            if row.get("state") not in active_states and row.get("terminal_at_utc"):
                row["_malformed_attempt_identity"] = True
                latest_by_task[f"__malformed__:{position}"] = row
            continue
        previous = latest_by_task.get(task_id)
        rank = (attempt_number, str(row.get("terminal_at_utc") or ""),
                str(row.get("attempt_id") or ""))
        previous_rank = (
            (previous.get("attempt_number"),
             str(previous.get("terminal_at_utc") or ""),
             str(previous.get("attempt_id") or ""))
            if previous is not None else None
        )
        if previous_rank is None or rank > previous_rank:
            latest_by_task[task_id] = row
    ordered = sorted(latest_by_task.values(), key=lambda row: (
        str(row.get("terminal_at_utc") or ""),
        str(row.get("attempt_id") or ""),
    ))
    classified: list[tuple[dict[str, Any], str, str | None]] = []
    scientific: list[tuple[dict[str, Any], str]] = []
    for row in ordered:
        result = _scientific_terminal_attempt(
            row, mission_id=mission_id, policy_version=active_policy)
        if result is None:
            continue
        classified.append((row, result[0], result[1]))
        if result[0] != "EXCLUDED":
            scientific.append((row, result[0]))
    block_size = int(stopping["evaluation_block_size"])
    completed_count = (len(scientific) // block_size) * block_size
    partial = scientific[completed_count:]
    lower_key = (
        (str(scientific[completed_count - 1][0].get("terminal_at_utc") or ""),
         str(scientific[completed_count - 1][0].get("attempt_id") or ""))
        if completed_count else None
    )
    excluded = []
    for row, classification, reason in classified:
        key = (str(row.get("terminal_at_utc") or ""),
               str(row.get("attempt_id") or ""))
        if (classification == "EXCLUDED"
                and (lower_key is None or key > lower_key)):
            excluded.append({
                "attempt_id": row.get("attempt_id"),
                "task_id": row.get("task_id"),
                "reason": reason,
            })
    no_m7 = [row for row, result in partial
             if result == "NO_M7_CANDIDATES"]
    state: dict[str, Any] = {
        "schema": "campaignx.first_letters_campaign_active_decision.v1",
        "decision": "CONTINUE",
        "evidence_status": "IN_PROGRESS",
        "mission_id": mission_id,
        "policy_version": active_policy,
        "policy_chain": copy.deepcopy(chain["policy_chain"]),
        "evaluation_kind": "ACTIVE_SCIENTIFIC_TERMINAL_BLOCK",
        "evaluation_index": completed_count // block_size + 1,
        "scientific_terminal_attempt_count": len(partial),
        "scientific_terminal_attempt_ids": [
            row.get("attempt_id") for row, _result in partial],
        "no_m7_numerator": len(no_m7),
        "scientific_terminal_denominator": block_size,
        "excluded_attempt_count": len(excluded),
        "excluded_attempts": excluded,
        "trigger_attempt_ids": [],
        "allowed_next_actions": ["QUEUE_NEXT_BOUND_WAVE", "CLOSE_CAMPAIGN"],
        "active_state_source": "SERVER_DERIVED_LIVE_POLICY_CHAIN",
        "non_claim": (
            "Candidate starvation is not evidence of surface, ink, text, "
            "or letter absence."),
    }
    state["state_sha256"] = content_sha256(state)
    return state


def validate_task_budget_receipt_pair(
    private: object, public: object,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one immutable private/sanitized task-budget pair."""
    if not isinstance(private, dict) or not isinstance(public, dict):
        raise ValueError("task-budget receipt pair must contain JSON objects")
    if (private.get("schema") != BUDGET_SCHEMA
            or private.get("receipt_sha256") != _receipt_hash(private)):
        raise ValueError("task-budget private receipt hash or schema is invalid")
    if (public.get("schema") != BUDGET_SANITIZED_SCHEMA
            or public.get("private_receipt_sha256") != private["receipt_sha256"]
            or public.get("receipt_sha256") != _public_hash(public)):
        raise ValueError("task-budget sanitized receipt hash or binding is invalid")
    if sanitize_task_budget_receipt(private) != public:
        raise ValueError("task-budget sanitized projection is invalid")
    return private, public


def _write_staged_json(
    directory: Path, value: dict[str, Any], *, mode: int,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".task-budget-", dir=directory)
    path = Path(name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def persist_task_budget_receipt_pair(
    private_path: Path, public_path: Path,
    private: dict[str, Any], public: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Create both receipt halves once under a cross-process mission lock."""
    private_path, public_path = Path(private_path), Path(public_path)
    if private_path == public_path:
        raise ValueError("private and sanitized task-budget paths must differ")
    validate_task_budget_receipt_pair(private, public)
    lock_path = private_path.parent / ".task-budget-publication.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        private_exists, public_exists = private_path.exists(), public_path.exists()
        if private_exists and public_exists:
            existing_private = json.loads(private_path.read_text(encoding="utf-8"))
            existing_public = json.loads(public_path.read_text(encoding="utf-8"))
            validate_task_budget_receipt_pair(existing_private, existing_public)
            if existing_private["receipt_sha256"] != private["receipt_sha256"]:
                raise ValueError(
                    "task-budget receipt pair already exists with different content")
            return {"private_receipt": existing_private,
                    "sanitized_receipt": existing_public}
        if private_exists:
            existing_private = json.loads(private_path.read_text(encoding="utf-8"))
            existing_public = sanitize_task_budget_receipt(existing_private)
            if existing_private["receipt_sha256"] != private["receipt_sha256"]:
                raise ValueError(
                    "task-budget private receipt already exists with different content")
            staged_public = _write_staged_json(
                public_path.parent, existing_public, mode=0o644)
            try:
                os.link(staged_public, public_path)
            finally:
                staged_public.unlink(missing_ok=True)
            validate_task_budget_receipt_pair(existing_private, existing_public)
            return {"private_receipt": existing_private,
                    "sanitized_receipt": existing_public}
        if public_exists:
            existing_public = json.loads(public_path.read_text(encoding="utf-8"))
            if (existing_public.get("schema") != BUDGET_SANITIZED_SCHEMA
                    or existing_public.get("receipt_sha256") !=
                        _public_hash(existing_public)):
                raise ValueError("task-budget sanitized receipt is invalid")
            recovered_private = copy.deepcopy(private)
            recovered_private["generated_at_utc"] = existing_public.get(
                "generated_at_utc")
            validate_task_budget_receipt_pair(recovered_private, existing_public)
            staged_private = _write_staged_json(
                private_path.parent, recovered_private, mode=0o600)
            try:
                os.link(staged_private, private_path)
            finally:
                staged_private.unlink(missing_ok=True)
            return {"private_receipt": recovered_private,
                    "sanitized_receipt": existing_public}
        staged_private = _write_staged_json(
            private_path.parent, private, mode=0o600)
        staged_public = _write_staged_json(
            public_path.parent, public, mode=0o644)
        private_published = False
        try:
            os.link(staged_private, private_path)
            private_published = True
            os.link(staged_public, public_path)
        except BaseException:
            if private_published:
                private_path.unlink(missing_ok=True)
            raise
        finally:
            staged_private.unlink(missing_ok=True)
            staged_public.unlink(missing_ok=True)
        return {"private_receipt": private, "sanitized_receipt": public}


def validate_task_budget_for_queue(
    receipt: object, *, mission_id: str, sample_id: str,
    preflight_receipt_sha256: str, policy_sha256: str,
    requested_tasks: int, execution_bindings: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a queue request is exactly the frozen budget."""
    if (not isinstance(receipt, dict) or receipt.get("schema") != BUDGET_SCHEMA
            or receipt.get("receipt_sha256") != _receipt_hash(receipt)):
        raise ValueError("task-budget receipt hash or schema is invalid")
    if (receipt.get("decision") != "CONTINUE"
            or receipt.get("mission_id") != mission_id
            or receipt.get("sample_id") != sample_id
            or receipt.get("preflight_receipt_sha256") != preflight_receipt_sha256
            or receipt.get("campaign_policy_sha256") != policy_sha256):
        raise ValueError("task-budget scope, policy, or preflight binding is invalid")
    planned = receipt.get("planned_task_count")
    if (not _nonnegative_integer(planned) or planned < 1
            or requested_tasks != planned):
        raise ValueError("requested task count differs from the frozen task budget")
    if execution_bindings != receipt.get("execution_bindings"):
        raise ValueError("queue execution bindings differ from the task budget")
    return receipt


def admit_p1_creation(
    mission_manifest: dict[str, Any], *, creation_path: str,
    budget_private: dict[str, Any] | None = None,
    budget_public: dict[str, Any] | None = None,
    mission_id: str | None = None, sample_id: str | None = None,
    preflight_receipt_sha256: str | None = None,
    policy_sha256: str | None = None,
    requested_tasks: int | None = None,
    execution_bindings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """One explicit authority shared by every server-side P1 creation path."""
    from framework.contracts import mission as mission_contract  # noqa: PLC0415

    controlled = mission_contract.is_first_letters_discovery_manifest(
        mission_manifest)
    if not controlled:
        return None
    if creation_path != "bootstrap":
        raise ValueError(
            f"controlled campaign P1 path {creation_path!r} is not authorized "
            "by the probability-prefix discovery budget")
    if not isinstance(budget_private, dict) or not isinstance(budget_public, dict):
        raise ValueError("controlled bootstrap requires a budget receipt pair")
    if (not isinstance(mission_id, str) or not isinstance(sample_id, str)
            or not isinstance(preflight_receipt_sha256, str)
            or not isinstance(policy_sha256, str)
            or not isinstance(requested_tasks, int)
            or not isinstance(execution_bindings, dict)):
        raise ValueError("controlled bootstrap budget admission is incomplete")
    validate_task_budget_receipt_pair(budget_private, budget_public)
    admitted = validate_task_budget_for_queue(
        budget_private, mission_id=mission_id, sample_id=sample_id,
        preflight_receipt_sha256=preflight_receipt_sha256,
        policy_sha256=policy_sha256, requested_tasks=requested_tasks,
        execution_bindings=execution_bindings,
    )
    if (mission_manifest.get("mission_id") != mission_id
            or mission_manifest.get("campaign_policy_sha256") != policy_sha256
            or mission_manifest.get("campaign_policy_id") !=
                admitted.get("campaign_policy_profile_id")
            or mission_manifest.get("deployed_revision") !=
                execution_bindings.get("code_revision")):
        raise ValueError(
            "controlled mission identity differs from the admitted task budget")
    selection = admitted.get("queue_selection") or {}
    prefix = selection.get("prefix_cell_ids")
    if (not isinstance(prefix, list)
            or len(prefix) != admitted.get("planned_task_count")):
        raise ValueError("controlled task budget has no exact selection prefix")
    admission = {
        "schema": "campaignx.first_letters_task_budget_admission.v1",
        "mission_id": mission_id,
        "sample_id": sample_id,
        "receipt_sha256": admitted["receipt_sha256"],
        "preflight_receipt_sha256": admitted["preflight_receipt_sha256"],
        "preflight_sanitized_receipt_sha256": admitted[
            "preflight_sanitized_receipt_sha256"],
        "approved_task_count": admitted["planned_task_count"],
        "order_seed_sha256": selection.get("order_seed_sha256"),
        "population_order_sha256": selection.get("population_order_sha256"),
        "prefix_sha256": selection.get("prefix_sha256"),
        "prefix_cell_ids": copy.deepcopy(prefix),
        "execution_bindings": copy.deepcopy(execution_bindings),
    }
    admission["admission_sha256"] = content_sha256(admission)
    return admission


def bind_campaign_budget_to_tasks(
    tasks: list[dict[str, Any]], admission: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bind the exact probability prefix and rank onto each generated task."""
    prefix = admission.get("prefix_cell_ids")
    approved = admission.get("approved_task_count")
    if (admission.get("schema") !=
            "campaignx.first_letters_task_budget_admission.v1"
            or not isinstance(prefix, list)
            or not isinstance(approved, int) or isinstance(approved, bool)
            or approved < 1 or len(prefix) != approved
            or len(prefix) != len(set(prefix))
            or len(tasks) != approved):
        raise ValueError("campaign budget admission or generated task count is invalid")
    if [task.get("cell_id") for task in tasks] != prefix:
        raise ValueError("generated tasks differ from the frozen probability prefix")
    common = {
        key: copy.deepcopy(admission[key]) for key in (
            "schema", "mission_id", "sample_id", "receipt_sha256",
            "preflight_receipt_sha256",
            "preflight_sanitized_receipt_sha256",
            "approved_task_count", "order_seed_sha256",
            "population_order_sha256", "prefix_sha256", "prefix_cell_ids",
            "execution_bindings", "admission_sha256",
        )
    }
    bound = []
    for rank, task in enumerate(tasks):
        if (task.get("mission_id") != admission["mission_id"]
                or task.get("sample_id") != admission["sample_id"]):
            raise ValueError("generated task scope differs from campaign budget")
        bound.append({
            **copy.deepcopy(task),
            "campaign_budget": {**copy.deepcopy(common), "selection_rank": rank},
        })
    return bound


def campaign_budget_task_matches_admission(
    task: dict[str, Any], admission: dict[str, Any],
) -> bool:
    """Return whether an already-persisted task belongs to this authority."""
    envelope = task.get("campaign_budget")
    if not isinstance(envelope, dict):
        return False
    common = {key: value for key, value in envelope.items()
              if key != "selection_rank"}
    rank = envelope.get("selection_rank")
    prefix = admission.get("prefix_cell_ids")
    return (
        common == admission
        and isinstance(rank, int) and not isinstance(rank, bool)
        and isinstance(prefix, list) and 0 <= rank < len(prefix)
        and task.get("cell_id") == prefix[rank]
        and task.get("mission_id") == admission.get("mission_id")
        and task.get("sample_id") == admission.get("sample_id")
    )


def _validated_task9_discovery_gate(value: object, *, mission_id: str) -> dict[str, Any]:
    """Validate the retained Task 9 gate without treating caller bytes as authority."""

    if not isinstance(value, dict):
        raise ValueError("TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED")
    gate = copy.deepcopy(value)
    gate_sha = gate.pop("gate_sha256", None)
    required = {
        "schema", "mission_id", "readiness_sha256", "control_binding_sha256",
        "wave_receipt_sha256", "policy_chain_sha256", "deployed_revision",
        "allow_unvalidated",
    }
    if (set(gate) != required
            or gate.get("schema") !=
                "campaignx.first_letters_task9_discovery_gate.v1"
            or gate.get("mission_id") != mission_id
            or gate.get("allow_unvalidated") is not False
            or gate_sha != content_sha256(gate)):
        raise ValueError("TASK9_CURRENT_CONTROL_AND_WAVE_AUTHORITY_REQUIRED")
    return copy.deepcopy(value)


def validate_promotion_child_admission(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("promotion-child admission must be an object")
    admission = copy.deepcopy(value)
    digest = admission.pop("admission_sha256", None)
    if (admission.get("schema") !=
            "campaignx.first_letters_promotion_child_admission.v1"
            or admission.get("allow_unvalidated") is not False
            or admission.get("statistical_rank_delta") != 0
            or admission.get("scientific_denominator_limit") != 1
            or admission.get("discovery_namespace") !=
                "NONCANONICAL_DISCOVERY"
            or digest != content_sha256(admission)):
        raise ValueError("promotion-child admission hash/contract is invalid")
    return copy.deepcopy(value)


def authorize_promotion_child(
    *, parent_task: dict[str, Any], registered_budget_admission: dict[str, Any],
    active_policy_chain: dict[str, Any],
    benchmark_authorization_v2: dict[str, Any],
    discovery_receipt: dict[str, Any], selected_candidate: dict[str, Any],
    normal_growth_lock: dict[str, Any], task9_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Authorize one fresh ordinary child for an existing sampled opportunity."""

    gate = _validated_task9_discovery_gate(
        task9_gate, mission_id=str(parent_task.get("mission_id"))
    )
    if (not isinstance(registered_budget_admission, dict)
            or registered_budget_admission.get("schema") !=
                "campaignx.first_letters_task_budget_admission.v1"
            or registered_budget_admission.get("admission_sha256") !=
                content_sha256({
                    key: row for key, row in registered_budget_admission.items()
                    if key != "admission_sha256"
                })):
        raise ValueError("registered Task 5 budget admission is invalid")
    rank = parent_task.get("selection_rank")
    prefix = registered_budget_admission.get("prefix_cell_ids")
    if (not isinstance(rank, int) or isinstance(rank, bool)
            or not isinstance(prefix, list) or not 0 <= rank < len(prefix)
            or parent_task.get("cell_id") != prefix[rank]
            or parent_task.get("mission_id") !=
                registered_budget_admission.get("mission_id")
            or parent_task.get("sample_id") !=
                registered_budget_admission.get("sample_id")
            or parent_task.get("campaign_budget_admission_sha256") !=
                registered_budget_admission.get("admission_sha256")):
        raise ValueError("parent task does not own the registered budget rank")
    execution = registered_budget_admission.get("execution_bindings") or {}
    for field in (
        "source_snapshot_id", "grid_version", "p0_artifact_id",
        "p0_artifact_sha256", "catalog_snapshot_sha256",
    ):
        if execution.get(field) != parent_task.get(field):
            raise ValueError(f"parent task {field} differs from its budget authority")
    successor = active_policy_chain.get("active_policy_version")
    if (successor != "first-letters-search@1.1.0"
            or active_policy_chain.get("paused") is not False
            or gate.get("policy_chain_sha256") !=
                active_policy_chain.get("policy_chain_sha256")):
        raise ValueError("promotion successor policy is inactive, paused, or unbound")
    from .seed_probe import (  # noqa: PLC0415
        coordinate_sha256_v1,
        validate_first_letters_normal_growth_lock,
        validate_seed_probe_benchmark_receipt_v2,
        validate_task6_coordinate,
    )
    if (not isinstance(benchmark_authorization_v2, dict)
            or benchmark_authorization_v2 !=
                validate_seed_probe_benchmark_receipt_v2(
                    benchmark_authorization_v2.get("decision"),
                    execution_manifest=benchmark_authorization_v2.get(
                        "execution_manifest")
                )):
        raise ValueError("promotion benchmark-v2 retained bytes/hash drift")
    receipt = copy.deepcopy(discovery_receipt)
    receipt_sha = receipt.pop("receipt_sha256", None)
    if (receipt.get("schema") !=
            "campaignx.first_letters_discovery_receipt.v1"
            or receipt.get("mission_id") != parent_task.get("mission_id")
            or receipt.get("parent_task_id") != parent_task.get("task_id")
            or receipt.get("parent_attempt_id") != parent_task.get("attempt_id")
            or receipt.get("namespace") != "NONCANONICAL_DISCOVERY"
            or receipt.get("allow_unvalidated") is not False
            or receipt_sha != content_sha256(receipt)):
        raise ValueError("discovery receipt is invalid or outside the parent")
    coordinate = validate_task6_coordinate(
        selected_candidate.get("promotion_coordinate_ct_l0_xyz"),
        require_integral=True,
    )
    coordinate_sha = selected_candidate.get("promotion_coordinate_sha256")
    if (coordinate_sha != coordinate_sha256_v1(coordinate)
            or selected_candidate.get("raw_coordinate_ct_l0_xyz") != coordinate
            or selected_candidate.get("raw_coordinate_sha256") != coordinate_sha
            or selected_candidate.get("coordinate_admission_state") !=
                "PROMOTABLE_INTEGRAL_COORDINATE_V1"
            or not isinstance(selected_candidate.get("ct_terminal_sha256"), str)
            or not isinstance(selected_candidate.get("clearance_terminal_sha256"), str)):
        raise ValueError("selected promotion candidate is not CT/clearance admissible")
    normal_lock = validate_first_letters_normal_growth_lock(normal_growth_lock)
    if (normal_lock.get("source_snapshot_id") != parent_task.get("source_snapshot_id")
            or normal_lock.get("promotion_coordinate_ct_l0_xyz") != coordinate
            or normal_lock.get("promotion_coordinate_sha256") != coordinate_sha
            or normal_lock.get("deployed_revision") != gate.get("deployed_revision")):
        raise ValueError("normal full-grow lock differs from promotion authority")
    opportunity_id = stable_id("first-letters-opportunity", {
        "admission_sha256": registered_budget_admission["admission_sha256"],
        "selection_rank": rank,
    })
    child_task_id = stable_id("first-letters-promotion-child", {
        "scientific_opportunity_id": opportunity_id,
        "successor_policy_version": successor,
        "promotion_coordinate_sha256": coordinate_sha,
    })
    core = {
        "schema": "campaignx.first_letters_promotion_child_admission.v1",
        "mission_id": parent_task["mission_id"],
        "sample_id": parent_task["sample_id"],
        "scientific_opportunity_id": opportunity_id,
        "parent_task_id": parent_task["task_id"],
        "parent_attempt_id": parent_task["attempt_id"],
        "parent_budget_admission_sha256": registered_budget_admission[
            "admission_sha256"],
        "registered_budget_admission": copy.deepcopy(
            registered_budget_admission
        ),
        "selection_rank": rank,
        "cell_id": parent_task["cell_id"],
        "source_snapshot_id": parent_task["source_snapshot_id"],
        "grid_version": parent_task["grid_version"],
        "p0_artifact_id": parent_task["p0_artifact_id"],
        "p0_artifact_sha256": parent_task["p0_artifact_sha256"],
        "catalog_snapshot_sha256": parent_task["catalog_snapshot_sha256"],
        "parent_policy_version": parent_task["policy_version"],
        "successor_policy_version": successor,
        "child_task_id": child_task_id,
        "discovery_receipt": copy.deepcopy(discovery_receipt),
        "discovery_receipt_sha256": discovery_receipt["receipt_sha256"],
        "discovery_artifact_sha256": discovery_receipt["artifact_sha256"],
        "discovery_namespace": "NONCANONICAL_DISCOVERY",
        "selected_candidate": copy.deepcopy(selected_candidate),
        "raw_coordinate_ct_l0_xyz": copy.deepcopy(coordinate),
        "raw_coordinate_sha256": coordinate_sha,
        "promotion_coordinate_ct_l0_xyz": copy.deepcopy(coordinate),
        "promotion_coordinate_sha256": coordinate_sha,
        "benchmark_authorization_v2": copy.deepcopy(
            benchmark_authorization_v2
        ),
        "benchmark_authorization_sha256": benchmark_authorization_v2[
            "authorization_sha256"],
        "active_policy_chain": copy.deepcopy(active_policy_chain),
        "task9_gate": gate,
        "normal_growth_lock": normal_lock,
        "statistical_rank_delta": 0,
        "scientific_denominator_limit": 1,
        "allow_unvalidated": False,
    }
    return validate_promotion_child_admission({
        **core, "admission_sha256": content_sha256(core)
    })


def build_promotion_child_task(admission: dict[str, Any]) -> dict[str, Any]:
    admission = validate_promotion_child_admission(admission)
    coordinate = admission["promotion_coordinate_ct_l0_xyz"]
    growth = admission["normal_growth_lock"]["growth_envelope"]
    parameter_envelope = {
        "generations": copy.deepcopy(growth["generations"]),
        "step_size": copy.deepcopy(growth["step_size"]),
        "min_area_cm": {
            "minimum": 0.0, "maximum": 0.0, "default": 0.0,
        },
        "use_cuda": {"allowed": [False], "default": False},
    }
    budget = {
        **copy.deepcopy(admission["registered_budget_admission"]),
        "selection_rank": admission["selection_rank"],
    }
    return {
        "schema": "campaignx.first_letters_promotion_child_task.v1",
        "task_id": admission["child_task_id"],
        "child_task_id": admission["child_task_id"],
        "mission_id": admission["mission_id"],
        "sample_id": admission["sample_id"],
        "scientific_opportunity_id": admission["scientific_opportunity_id"],
        "parent_task_id": admission["parent_task_id"],
        "parent_attempt_id": admission["parent_attempt_id"],
        "selection_rank": admission["selection_rank"],
        "statistical_rank_delta": 0,
        "source_snapshot_id": admission["source_snapshot_id"],
        "grid_version": admission["grid_version"],
        "cell_id": admission["cell_id"],
        "policy_version": admission["successor_policy_version"],
        "successor_policy_version": admission["successor_policy_version"],
        "p0_artifact_id": admission["p0_artifact_id"],
        "p0_artifact_sha256": admission["p0_artifact_sha256"],
        "catalog_snapshot_sha256": admission["catalog_snapshot_sha256"],
        "bounds_xyz": [copy.deepcopy(coordinate), copy.deepcopy(coordinate)],
        "center_xyz": {"x": coordinate[0], "y": coordinate[1], "z": coordinate[2]},
        "priority": 1.0,
        "parameter_envelope": parameter_envelope,
        "normal_growth_lock": copy.deepcopy(admission["normal_growth_lock"]),
        "promotion_coordinate_ct_l0_xyz": copy.deepcopy(coordinate),
        "promotion_coordinate_sha256": admission["promotion_coordinate_sha256"],
        "promotion_child_admission_sha256": admission["admission_sha256"],
        "campaign_budget": budget,
        "resource_requirements": {
            "gpu_required": False, "minimum_vram_gb": 0.0,
            "seed_probe_required": False,
        },
        "fresh_start": True,
        "allow_unvalidated": False,
    }


def validate_promotion_child_task(
    child_task: dict[str, Any], *, admission: dict[str, Any],
    registered_budget_admission: dict[str, Any],
    authoritative_source_snapshot: dict[str, Any],
) -> None:
    admission = validate_promotion_child_admission(admission)
    if child_task != build_promotion_child_task(admission):
        raise ValueError("promotion child differs from immutable admission")
    if (registered_budget_admission !=
            admission["registered_budget_admission"]
            or registered_budget_admission.get("admission_sha256") !=
                admission["parent_budget_admission_sha256"]
            or authoritative_source_snapshot.get("source_snapshot_id") !=
                admission["source_snapshot_id"]
            or child_task.get("allow_unvalidated") is not False):
        raise ValueError("promotion child source or budget authority drift")


def classify_discovery_scientific_opportunity(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify a parent/child opportunity once, excluding platform retries."""

    if not isinstance(rows, list) or not rows:
        return {"state": "CONTROL_INCOMPLETE", "denominator_contribution": 0}
    scientific = [row for row in rows
                  if row.get("scientific_terminal") is True
                  and row.get("platform_excluded") is not True]
    if len(scientific) > 1:
        return {"state": "CONTROL_INCOMPLETE", "denominator_contribution": 0}
    states = {row.get("state") for row in rows}
    known = {
        "PROBE_REVIEW_PENDING", "DISCOVERY_ABSTAINED_NO_UNIQUE_WINNER",
        "DISCOVERY_REJECTED_CANDIDATES", "DISCOVERY_PROMOTED", "PENDING",
        "CLAIMED", "PLANNING", "RUNNING", "GROW_FAILED", "NO_SEED",
        "ARCHIVED", "QC_PENDING", "DUPLICATE_SURFACE",
    }
    if not states <= known:
        return {"state": "CONTROL_INCOMPLETE", "denominator_contribution": 0}
    if scientific:
        return {"state": "SCIENTIFIC_TERMINAL", "denominator_contribution": 1}
    return {"state": "OPEN", "denominator_contribution": 0}


def validate_campaign_budget_task_batch(
    tasks: list[dict[str, Any]], existing_payloads: list[dict[str, Any]] | None = None,
    *, registered_admission: dict[str, Any] | None = None,
    authoritative_snapshot: dict[str, Any] | None = None,
) -> str | None:
    """Validate one batch plus prior rows while the store holds its write lock."""
    envelopes = [task.get("campaign_budget") for task in tasks]
    if not any(envelope is not None for envelope in envelopes):
        return None
    if any(not isinstance(envelope, dict) for envelope in envelopes):
        raise ValueError("campaign budget envelope must be present on every task")
    first = envelopes[0]
    assert isinstance(first, dict)
    common_fields = {
        "schema", "mission_id", "sample_id", "receipt_sha256",
        "preflight_receipt_sha256", "preflight_sanitized_receipt_sha256",
        "approved_task_count", "order_seed_sha256", "population_order_sha256",
        "prefix_sha256", "prefix_cell_ids", "execution_bindings",
        "admission_sha256",
    }
    if set(first) != common_fields | {"selection_rank"}:
        raise ValueError("campaign budget envelope is incomplete")
    common = {field: first.get(field) for field in common_fields}
    prefix = common["prefix_cell_ids"]
    approved = common["approved_task_count"]
    receipt = common["receipt_sha256"]
    if (common["schema"] !=
            "campaignx.first_letters_task_budget_admission.v1"
            or not isinstance(receipt, str)
            or len(receipt) != 64
            or not isinstance(approved, int) or isinstance(approved, bool)
            or approved < 1
            or not isinstance(prefix, list) or len(prefix) != approved
            or len(prefix) != len(set(prefix))
            or common["prefix_sha256"] != content_sha256(prefix)):
        raise ValueError("campaign budget envelope prefix or receipt is invalid")
    if (not isinstance(registered_admission, dict)
            or registered_admission.get("admission_sha256") != content_sha256({
                key: value for key, value in registered_admission.items()
                if key != "admission_sha256"
            })
            or common != registered_admission):
        raise ValueError(
            "campaign budget admission is not registered or differs from signed authority")
    execution = common.get("execution_bindings")
    queue_execution = (
        execution.get("queue_execution") if isinstance(execution, dict) else None)
    gates = execution.get("gates") if isinstance(execution, dict) else None
    if not isinstance(queue_execution, dict) or not isinstance(gates, dict):
        raise ValueError("campaign budget execution bindings are incomplete")
    if not isinstance(authoritative_snapshot, dict):
        raise ValueError("campaign budget authoritative source snapshot is unavailable")
    source_lock = authoritative_snapshot.get("source_content_lock")
    expected_source = {
        "source_snapshot_id": execution.get("source_snapshot_id"),
        "sample_id": execution.get("sample_id"),
        "ct_sha256": execution.get("ct_sha256"),
        "m7_sha256": execution.get("m7_sha256"),
        "coordinate_frame": execution.get("coordinate_frame"),
        "voxel_size_um": execution.get("voxel_size_um"),
        "shape_xyz": execution.get("shape_xyz"),
        "m7_threshold": execution.get("m7_threshold"),
        "source_content_lock_sha256": execution.get(
            "source_content_lock_sha256"),
        "m7_uri_sha256": execution.get("m7_uri_sha256"),
    }
    actual_source = {
        "source_snapshot_id": authoritative_snapshot.get("source_snapshot_id"),
        "sample_id": authoritative_snapshot.get("sample_id"),
        "ct_sha256": authoritative_snapshot.get("ct_sha256"),
        "m7_sha256": authoritative_snapshot.get("m7_sha256"),
        "coordinate_frame": authoritative_snapshot.get("coordinate_frame"),
        "voxel_size_um": authoritative_snapshot.get("voxel_size_um"),
        "shape_xyz": authoritative_snapshot.get("shape_xyz"),
        "m7_threshold": authoritative_snapshot.get("m7_threshold"),
        "source_content_lock_sha256": (
            content_sha256(source_lock) if isinstance(source_lock, dict) else None),
        "m7_uri_sha256": hashlib.sha256(
            str(authoritative_snapshot.get("m7_uri") or "").encode("utf-8")
        ).hexdigest(),
    }
    if actual_source != expected_source:
        raise ValueError(
            "campaign budget registered source snapshot differs from signed authority")

    observed: dict[int, tuple[str, str, str, str]] = {}
    batch_ranks: set[int] = set()

    def include(task: dict[str, Any], *, existing: bool) -> None:
        envelope = task.get("campaign_budget")
        if not isinstance(envelope, dict):
            if existing:
                return
            raise ValueError("campaign budget envelope is missing")
        if existing and (
            envelope.get("mission_id") != common["mission_id"]
            or envelope.get("sample_id") != common["sample_id"]
        ):
            return
        if envelope.get("receipt_sha256") != receipt:
            if existing:
                existing_policy = (
                    envelope.get("execution_bindings") or {}
                ).get("policy_version")
                current_policy = (
                    common.get("execution_bindings") or {}
                ).get("policy_version")
                if existing_policy != current_policy:
                    return
                raise ValueError(
                    "controlled mission/sample/policy is already bound to "
                    "another task budget receipt")
            raise ValueError("mixed campaign budget receipts in one task batch")
        if ({field: envelope.get(field) for field in common_fields} != common
                or set(envelope) != common_fields | {"selection_rank"}):
            raise ValueError("campaign budget task envelopes disagree")
        rank = envelope.get("selection_rank")
        cell_id = task.get("cell_id")
        if (not isinstance(rank, int) or isinstance(rank, bool)
                or not 0 <= rank < approved
                or cell_id != prefix[rank]
                or task.get("mission_id") != common["mission_id"]
                or task.get("sample_id") != common["sample_id"]):
            raise ValueError("campaign budget selection rank is outside its prefix")

        exact_task_fields = {
            "mission_id": execution.get("mission_id"),
            "sample_id": execution.get("sample_id"),
            "source_snapshot_id": execution.get("source_snapshot_id"),
            "catalog_snapshot_sha256": execution.get(
                "catalog_snapshot_sha256"),
            "grid_version": execution.get("grid_version"),
            "policy_version": execution.get("policy_version"),
            "p0_selection_version": execution.get("p0_selection_version"),
            "p0_selection_sha256": execution.get("p0_selection_sha256"),
            "p0_artifact_id": execution.get("p0_artifact_id"),
            "p0_artifact_sha256": execution.get("p0_artifact_sha256"),
        }
        if any(task.get(field) != expected
               for field, expected in exact_task_fields.items()):
            raise ValueError(
                "campaign budget task provenance differs from execution binding")
        if task.get("parameter_envelope") != queue_execution.get(
                "parameter_envelope"):
            raise ValueError(
                "campaign budget task parameter envelope differs from execution binding")
        if (task.get("candidate_selection_policy") !=
                execution.get("candidate_selection_policy")
                or task.get("planner") != queue_execution.get("planner")
                or task.get("planner_model") != queue_execution.get("planner_model")
                or task.get("candidate_rank", 1) != queue_execution.get("candidate_rank", 1)
                or task.get("reconsider_covered", False) != queue_execution.get("reconsider_covered", False)):
            raise ValueError(
                "campaign budget task planner policy differs from execution binding")
        discovery = task.get("candidate_discovery")
        if not isinstance(discovery, dict):
            raise ValueError("campaign budget task has no candidate discovery binding")
        region = discovery.get("region")
        expected_radius = {
            axis: execution.get("query_radius") for axis in "xyz"}
        expected_gate = gates.get("ct_material_support_gate")
        expected_center, expected_bounds = _expected_cell_geometry(
            str(cell_id), execution, gates)
        if (discovery.get("provider") != execution.get("provider")
                or task.get("center_xyz") != expected_center
                or task.get("bounds_xyz") != expected_bounds
                or not isinstance(region, dict)
                or region.get("center") != expected_center
                or discovery.get("prediction_uri") !=
                    authoritative_snapshot.get("m7_uri")
                or discovery.get("prediction_space") !=
                    queue_execution.get("prediction_space")
                or discovery.get("minimum_separation_voxels") !=
                    queue_execution.get("minimum_separation_voxels")
                or discovery.get("m7_threshold") != execution.get("m7_threshold")
                or region.get("radius") != expected_radius
                or discovery.get("max_candidates") !=
                    queue_execution["parameter_envelope"].get(
                        "maximum_candidate_count")
                or discovery.get("minimum_cell_interior_clearance_voxels") !=
                    gates.get("candidate_interior_clearance")
                or discovery.get("minimum_volume_interior_clearance_voxels") !=
                    gates.get("volume_clearance")
                or discovery.get("seed_region_policy") !=
                    execution.get("seed_region_policy")
                or discovery.get("recenter_probe_max_candidates") !=
                    queue_execution.get("recenter_probe_max_candidates")
                or discovery.get("recenter_radius_xyz") !=
                    queue_execution.get("recenter_radius_xyz")
                or discovery.get("ct_material_support_gate") != expected_gate):
            raise ValueError(
                "campaign budget task discovery knobs differ from execution binding")
        expected_probe_mode = queue_execution.get("seed_probe_mode")
        probe = task.get("seed_probe")
        if expected_probe_mode == "off":
            if probe is not None:
                raise ValueError(
                    "campaign budget task seed probe differs from execution binding")
        elif (not isinstance(probe, dict)
                or probe.get("mode") != expected_probe_mode
                or (probe.get("probe_parameters") or {}).get("top_k") !=
                    queue_execution.get("seed_probe_top_k")
                or (probe.get("probe_parameters") or {}).get("generations") !=
                    queue_execution.get("seed_probe_generations")):
            raise ValueError(
                "campaign budget task seed probe differs from execution binding")
        identity = (
            str(cell_id), str(task.get("source_snapshot_id") or ""),
            str(task.get("grid_version") or ""),
            str(task.get("policy_version") or ""),
        )
        previous = observed.get(rank)
        if previous is not None and previous != identity:
            raise ValueError(
                "campaign budget selection rank has conflicting task identity")
        if not existing and rank in batch_ranks:
            raise ValueError("campaign budget task batch repeats a selection rank")
        observed[rank] = identity
        if not existing:
            batch_ranks.add(rank)

    for existing in existing_payloads or []:
        include(existing, existing=True)
    for task in tasks:
        include(task, existing=False)
    if len(observed) > approved:
        raise ValueError("campaign task count exceeds its approved budget")
    return receipt
