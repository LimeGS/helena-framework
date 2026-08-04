"""Deterministic closed-loop seed trials for Stage 01.

The probe is deliberately not a second planner and not a surface.  It executes
the same small, frozen VC3D experiment for the first K candidates, measures the
result with the existing TIFXYZ geometry gate, and either identifies one
unambiguous winner or abstains.

Probe bytes and rows live in their own namespace.  They must never be inserted
into ``surfaces``, ``artifact_sets`` or downstream QC.  Only the later full grow
is canonical and it still passes normal finalization, deduplication, geometry
and physical QC.
"""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .artifact_store import open_artifact_store
from .common import (
    artifact_manifest,
    content_sha256,
    file_sha256,
    read_json,
    utc_now,
    write_json_atomic,
)
from .executor import InsufficientGpuMemoryError
from .finalizer import certify_surface_geometry, inspect_tifxyz
from .planner import candidate_rank_key


PROBE_REQUIRED_ARTIFACTS = (
    "x.tif",
    "y.tif",
    "z.tif",
    "generations.tif",
    "meta.json",
)
PROBE_PROFILE_ID = "vc3d-m7-probe-v1"
PROBE_PROFILE_SHA256 = (
    "219a0208224e92239b58e03a9f1ad3780cd49fa9151485898ae69600c9d43f33"
)
PROBE_EVALUATION_PROFILE_ID = "tifxyz-geometry-gate-probe-v1"
PROBE_EVALUATION_PROFILE_SHA256 = (
    "918899b622604bb4c7c1a42b9d961e24832856fd7940f25af7dabb7ca2c858b6"
)
PROBE_TERMINAL_TRIAL_STATES = frozenset(
    {"SUCCEEDED", "REJECTED", "UNMEASURED", "FAILED"}
)
PROBE_DECISION_ACTIONS = frozenset(
    {"CONTINUE_WINNER", "HUMAN_REVIEW", "REJECT_ALL"}
)
SOURCE_CONTENT_LOCK_SCHEMA = "campaignx.source_content_lock.v1"
BENCHMARK_SPEC_SCHEMA = "campaignx.seed_probe_benchmark_spec.v1"
BENCHMARK_DECISION_SCHEMA = "campaignx.seed_probe_benchmark_decision.v1"

# How far down m7's candidate ordering a probe may look. Mirrored by the fleet
# CLI's argparse choices, the panel's request model and the launcher contract
# it serves; a test compares them, because this bound has been wrong in four
# places at once.
TOP_K_MAXIMUM = 20
BENCHMARK_AUTHORIZATION_SCHEMA = (
    "campaignx.seed_probe_benchmark_authorization.v1"
)
BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA = (
    "campaignx.seed_probe_benchmark_execution_authorization.v1"
)
BENCHMARK_EXECUTION_SCHEMA = "campaignx.seed_probe_benchmark_execution.v1"
BENCHMARK_RNG_PROTOCOL = (
    "sha256-benchmark-spec-sample-cell-prefix16-v1"
)
# Canonical SHA-256 of generator.SEED_PROBE_CONTINUATION_ENVELOPE.  Keeping the
# digest here avoids an import cycle (generator imports this module) while
# making any future envelope drift fail closed in the benchmark builder.
BENCHMARK_FULL_GROW_ENVELOPE_SHA256 = (
    "aa5cf6b030e3b8f2d5f90009c332f4e22e120fd1d8f87cb6f6f11aa66aba8990"
)
BENCHMARK_REQUIRED_CHECK_IDS = frozenset(
    {
        "MATCHED_EXECUTION_IDENTITY",
        "INVARIANTS",
        "YIELD_IMPROVEMENT",
        "PAIRED_SUPERIORITY",
        "REVIEWER_RATE",
        "COMPUTE_BUDGET",
        "NEW_INCORRECT_LAMINA_SAFETY",
        "SCROLL_COVERAGE",
        "SCROLL_NONREGRESSION",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIR_RNG_RE = re.compile(r"^[0-9a-f]{16}$")


class ProbeWinnerMaterializationError(RuntimeError):
    """A retained winner could not be verified into the continuation sandbox."""

    def __init__(
        self,
        *,
        probe_run_id: str,
        decision: dict[str, Any],
        winner_trial_id: str,
        artifact_uri: str,
        error: Exception,
    ) -> None:
        super().__init__(
            "retained probe winner could not be materialized: "
            f"{type(error).__name__}: {str(error)[:1000]}"
        )
        self.probe_run_id = probe_run_id
        self.decision = copy.deepcopy(decision)
        self.winner_trial_id = winner_trial_id
        self.artifact_uri = artifact_uri


def default_seed_probe_policy(
    *,
    mode: str,
    top_k: int = 2,
    generations: int = 12,
    step_size: int = 12,
    use_cuda: bool = False,
    maximum_attempts_per_candidate: int = 2,
    benchmark_authorization: dict[str, Any] | None = None,
    review_owner: str | None = None,
) -> dict[str, Any]:
    """Return the complete immutable v1 policy stored on every probe task."""

    value = {
        "schema": "campaignx.seed_probe_policy.v1",
        "policy_id": "seed-probe-v1",
        "mode": str(mode).lower(),
        "top_k": int(top_k),
        "probe_profile_id": PROBE_PROFILE_ID,
        "probe_profile_sha256": PROBE_PROFILE_SHA256,
        "probe_parameters": {
            "generations": int(generations),
            "step_size": int(step_size),
            "min_area_cm": 0.0,
            "use_cuda": bool(use_cuda),
        },
        "maximum_total_probe_generations": (
            int(top_k)
            * int(generations)
            * int(maximum_attempts_per_candidate)
        ),
        "maximum_attempts_per_candidate": int(maximum_attempts_per_candidate),
        "evaluation_profile_id": PROBE_EVALUATION_PROFILE_ID,
        "evaluation_profile_sha256": PROBE_EVALUATION_PROFILE_SHA256,
        "decision_policy_id": "unique-geometry-certified-v1",
        "inconclusive_action": "HUMAN_REVIEW",
        "ink_used": False,
        **(
            {"benchmark_authorization": benchmark_authorization}
            if benchmark_authorization is not None
            else {}
        ),
        **({"review_owner": review_owner} if review_owner is not None else {}),
    }
    return normalize_seed_probe_policy(value)


def _require_lowercase_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def normalize_seed_probe_benchmark_authorization(
    value: Any,
) -> dict[str, Any]:
    """Validate the compact approval bound into a select task.

    The task intentionally carries no local receipt path.  It carries the
    benchmark identity and canonical decision digest so queued work remains
    portable and auditable after the bootstrap host disappears.
    """

    expected = {
        "schema",
        "benchmark_id",
        "decision_receipt_sha256",
        "spec_sha256",
        "results_sha256",
        "paired_cell_count",
        "scroll_count",
        "authorized_sample_ids",
        "execution_scope",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark authorization differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported seed probe benchmark authorization schema")
    benchmark_id = value["benchmark_id"]
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("benchmark authorization benchmark_id must be non-empty")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "benchmark authorization must come from ISOLATED_NONPRODUCTION"
        )
    for field in (
        "decision_receipt_sha256",
        "spec_sha256",
        "results_sha256",
    ):
        _require_lowercase_sha256(value[field], field)
    for field, minimum, maximum in (
        ("paired_cell_count", 40, 60),
        ("scroll_count", 3, None),
    ):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"benchmark authorization {field} must be an integer")
        if raw < minimum or (maximum is not None and raw > maximum):
            bounds = (
                f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
            )
            raise ValueError(
                f"benchmark authorization {field} must be {bounds}"
            )
    authorized_sample_ids = value["authorized_sample_ids"]
    if (
        not isinstance(authorized_sample_ids, list)
        or any(
            not isinstance(sample_id, str) or not sample_id
            for sample_id in authorized_sample_ids
        )
        or authorized_sample_ids != sorted(set(authorized_sample_ids))
        or len(authorized_sample_ids) != value["scroll_count"]
    ):
        raise ValueError(
            "benchmark authorization authorized_sample_ids must be the "
            "sorted unique approved scroll cohort"
        )
    return copy.deepcopy(value)


def normalize_seed_probe_benchmark_execution_authorization(
    value: Any,
) -> dict[str, Any]:
    """Validate a pre-result authority for only the frozen causal cohort."""

    expected = {
        "schema",
        "benchmark_id",
        "benchmark_spec_sha256",
        "execution_scope",
        "arm",
        "baseline_policy_version",
        "policy_version",
        "planner",
        "seed_probe_mode",
        "cells",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark execution authorization differs from v1 "
            f"contract: expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA:
        raise ValueError(
            "unsupported seed probe benchmark execution authorization schema"
        )
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "benchmark execution authorization must be ISOLATED_NONPRODUCTION"
        )
    if (
        value["arm"] != "closed_loop"
        or value["planner"] != "deterministic-v2"
        or value["seed_probe_mode"] != "select"
    ):
        raise ValueError(
            "benchmark execution authorization must name the "
            "deterministic-v2/select closed_loop arm"
        )
    for field in (
        "benchmark_id",
        "baseline_policy_version",
        "policy_version",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"benchmark execution {field} must be non-empty")
    if value["baseline_policy_version"] == value["policy_version"]:
        raise ValueError(
            "benchmark execution arm policy versions must be distinct"
        )
    _require_lowercase_sha256(
        value["benchmark_spec_sha256"], "benchmark_spec_sha256"
    )
    cells = value["cells"]
    if not isinstance(cells, list) or not 40 <= len(cells) <= 60:
        raise ValueError(
            "benchmark execution authorization must bind 40..60 cells"
        )
    seen: set[tuple[str, str]] = set()
    seen_blocks: set[str] = set()
    normalized_cells: list[dict[str, str]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or set(cell) != {
            "cell_id",
            "sample_id",
            "independence_block_id",
        }:
            raise ValueError(
                f"benchmark execution cells[{index}] differs from v1 contract"
            )
        if any(
            not isinstance(cell[field], str) or not cell[field]
            for field in (
                "cell_id",
                "sample_id",
                "independence_block_id",
            )
        ):
            raise ValueError(
                f"benchmark execution cells[{index}] identity is incomplete"
            )
        identity = (cell["sample_id"], cell["cell_id"])
        if identity in seen:
            raise ValueError(
                "benchmark execution authorization contains duplicate cells"
            )
        block = cell["independence_block_id"]
        if block in seen_blocks:
            raise ValueError(
                "benchmark execution authorization contains duplicate "
                "independence blocks"
            )
        seen.add(identity)
        seen_blocks.add(block)
        normalized_cells.append(dict(cell))
    if len({cell["sample_id"] for cell in normalized_cells}) < 3:
        raise ValueError(
            "benchmark execution authorization must bind at least three scrolls"
        )
    return {**copy.deepcopy(value), "cells": normalized_cells}


def validate_seed_probe_benchmark_spec(value: Any) -> dict[str, Any]:
    """Validate a frozen causal spec before its select arm has any results."""

    expected = {
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
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark spec differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_SPEC_SCHEMA:
        raise ValueError("unsupported seed probe benchmark spec schema")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError("benchmark spec must be ISOLATED_NONPRODUCTION")
    if not isinstance(value["benchmark_id"], str) or not value[
        "benchmark_id"
    ].strip():
        raise ValueError("benchmark spec benchmark_id must be non-empty")
    frozen_at = value["frozen_at_utc"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("Z"):
        raise ValueError("benchmark spec frozen_at_utc must be UTC")
    try:
        parsed_at = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("benchmark spec frozen_at_utc is invalid") from error
    if parsed_at.utcoffset() != timezone.utc.utcoffset(parsed_at):
        raise ValueError("benchmark spec frozen_at_utc must be UTC")

    expected_arm_keys = {"policy_version", "planner", "seed_probe_mode"}
    for name in ("baseline", "closed_loop"):
        arm = value[name]
        if not isinstance(arm, dict) or set(arm) != expected_arm_keys:
            raise ValueError(f"benchmark spec {name} arm differs from v1 contract")
        if any(not isinstance(arm[field], str) or not arm[field] for field in arm):
            raise ValueError(f"benchmark spec {name} arm identity is incomplete")
    if (
        value["baseline"]["planner"] != "deterministic-v2"
        or value["baseline"]["seed_probe_mode"] != "off"
        or value["closed_loop"]["planner"] != "deterministic-v2"
        or value["closed_loop"]["seed_probe_mode"] != "select"
        or value["baseline"]["policy_version"]
        == value["closed_loop"]["policy_version"]
    ):
        raise ValueError(
            "benchmark spec must compare distinct deterministic-v2/off and "
            "deterministic-v2/select arms"
        )
    for field, minimum, maximum in (
        ("minimum_cells", 40, 60),
        ("maximum_cells", 40, 60),
        ("minimum_scrolls", 3, None),
    ):
        raw = value[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"benchmark spec {field} must be an integer")
        if raw < minimum or (maximum is not None and raw > maximum):
            raise ValueError(f"benchmark spec {field} is outside v1 bounds")
    if value["minimum_cells"] > value["maximum_cells"]:
        raise ValueError("benchmark spec cell bounds are inverted")
    for field in (
        "minimum_relative_yield_improvement",
        "maximum_relative_reviewer_rate_regression",
        "maximum_incremental_compute_wall_hours_per_cell",
        "maximum_new_incorrect_lamina_rate_upper_bound",
    ):
        raw = value[field]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise ValueError(f"benchmark spec {field} must be non-negative")
    safety_upper_bound = float(
        value["maximum_new_incorrect_lamina_rate_upper_bound"]
    )
    if not 0.0 < safety_upper_bound <= 1.0:
        raise ValueError(
            "benchmark spec "
            "maximum_new_incorrect_lamina_rate_upper_bound must be in (0, 1]"
        )
    paired_alpha = value["paired_test_alpha"]
    if (
        isinstance(paired_alpha, bool)
        or not isinstance(paired_alpha, (int, float))
        or not math.isfinite(float(paired_alpha))
        or not 0.0 < float(paired_alpha) <= 0.10
    ):
        raise ValueError(
            "benchmark spec paired_test_alpha must be in (0, 0.10]"
        )
    minimum_pairs = value["minimum_pairs_per_scroll"]
    if (
        isinstance(minimum_pairs, bool)
        or not isinstance(minimum_pairs, int)
        or minimum_pairs < 5
    ):
        raise ValueError(
            "benchmark spec minimum_pairs_per_scroll must be an integer >= 5"
        )
    if not isinstance(value["review_protocol_id"], str) or not value[
        "review_protocol_id"
    ]:
        raise ValueError("benchmark spec review_protocol_id must be non-empty")
    cells = value["cells"]
    if (
        not isinstance(cells, list)
        or not value["minimum_cells"] <= len(cells) <= value["maximum_cells"]
    ):
        raise ValueError("benchmark spec cohort is outside its frozen bounds")

    authorization = {
        "schema": BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA,
        "benchmark_id": value["benchmark_id"],
        "benchmark_spec_sha256": content_sha256(value),
        "execution_scope": value["execution_scope"],
        "arm": "closed_loop",
        "baseline_policy_version": value["baseline"]["policy_version"],
        "policy_version": value["closed_loop"]["policy_version"],
        "planner": value["closed_loop"]["planner"],
        "seed_probe_mode": value["closed_loop"]["seed_probe_mode"],
        "cells": cells,
    }
    normalized = normalize_seed_probe_benchmark_execution_authorization(
        authorization
    )
    if len({cell["sample_id"] for cell in normalized["cells"]}) < value[
        "minimum_scrolls"
    ]:
        raise ValueError("benchmark spec cohort has too few scrolls")
    pairs_by_scroll: dict[str, int] = {}
    for cell in normalized["cells"]:
        pairs_by_scroll[cell["sample_id"]] = (
            pairs_by_scroll.get(cell["sample_id"], 0) + 1
        )
    if any(
        count < value["minimum_pairs_per_scroll"]
        for count in pairs_by_scroll.values()
    ):
        raise ValueError(
            "benchmark spec cohort has too few pairs on at least one scroll"
        )
    return normalized


def load_seed_probe_benchmark_spec(path: Path) -> dict[str, Any]:
    """Load a preregistered isolated spec for its causal select arm."""

    try:
        value = read_json(Path(path))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read seed probe benchmark spec: {error}") from error
    return validate_seed_probe_benchmark_spec(value)


def benchmark_pair_rng_seed(
    authorization: dict[str, Any],
    *,
    sample_id: str,
    cell_id: str,
) -> str:
    """Derive the common pair seed from preregistered, arm-neutral identity."""

    normalized = normalize_seed_probe_benchmark_execution_authorization(
        authorization
    )
    identity = {
        "schema": "campaignx.seed_probe_benchmark_pair_rng.v1",
        "benchmark_id": normalized["benchmark_id"],
        "benchmark_spec_sha256": normalized["benchmark_spec_sha256"],
        "sample_id": str(sample_id),
        "cell_id": str(cell_id),
    }
    return content_sha256(identity)[:16]


def normalize_seed_probe_benchmark_execution(
    value: Any,
    *,
    sample_id: str | None = None,
    cell_id: str | None = None,
    policy_version: str | None = None,
    planner: str | None = None,
    seed_probe_mode: str | None = None,
    parameter_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one arm/cell execution contract and optional task context."""

    expected = {
        "schema",
        "execution_scope",
        "benchmark_id",
        "benchmark_spec_sha256",
        "authorization",
        "authorization_sha256",
        "arm",
        "policy_version",
        "planner",
        "seed_probe_mode",
        "sample_id",
        "cell_id",
        "pair_rng_seed",
        "rng_protocol",
        "full_grow_envelope_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark_execution differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_EXECUTION_SCHEMA:
        raise ValueError("unsupported seed probe benchmark_execution schema")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "benchmark_execution must be ISOLATED_NONPRODUCTION"
        )
    authorization = normalize_seed_probe_benchmark_execution_authorization(
        value["authorization"]
    )
    if (
        value["authorization_sha256"] != content_sha256(authorization)
        or value["benchmark_id"] != authorization["benchmark_id"]
        or value["benchmark_spec_sha256"]
        != authorization["benchmark_spec_sha256"]
    ):
        raise ValueError(
            "benchmark_execution is not bound to its preregistered "
            "authorization"
        )
    _require_lowercase_sha256(
        value["authorization_sha256"], "authorization_sha256"
    )
    _require_lowercase_sha256(
        value["benchmark_spec_sha256"], "benchmark_spec_sha256"
    )

    arm = value["arm"]
    if arm == "baseline":
        expected_policy = authorization["baseline_policy_version"]
        expected_mode = "off"
    elif arm == "closed_loop":
        expected_policy = authorization["policy_version"]
        expected_mode = "select"
    else:
        raise ValueError(
            "benchmark_execution arm must be baseline or closed_loop"
        )
    if (
        value["policy_version"] != expected_policy
        or value["planner"] != "deterministic-v2"
        or value["seed_probe_mode"] != expected_mode
    ):
        raise ValueError(
            "benchmark_execution differs from its preregistered arm"
        )

    execution_sample = value["sample_id"]
    execution_cell = value["cell_id"]
    if any(
        not isinstance(item, str) or not item
        for item in (execution_sample, execution_cell)
    ):
        raise ValueError(
            "benchmark_execution sample_id and cell_id must be non-empty"
        )
    if not any(
        row["sample_id"] == execution_sample
        and row["cell_id"] == execution_cell
        for row in authorization["cells"]
    ):
        raise ValueError(
            "benchmark_execution task is outside the preregistered cohort"
        )
    expected_seed = benchmark_pair_rng_seed(
        authorization,
        sample_id=execution_sample,
        cell_id=execution_cell,
    )
    if (
        not isinstance(value["pair_rng_seed"], str)
        or _PAIR_RNG_RE.fullmatch(value["pair_rng_seed"]) is None
        or value["pair_rng_seed"] != expected_seed
    ):
        raise ValueError(
            "benchmark_execution pair_rng_seed does not match its frozen cell"
        )
    if value["rng_protocol"] != BENCHMARK_RNG_PROTOCOL:
        raise ValueError("unsupported benchmark_execution RNG protocol")
    if (
        value["full_grow_envelope_sha256"]
        != BENCHMARK_FULL_GROW_ENVELOPE_SHA256
    ):
        raise ValueError(
            "benchmark_execution must use the common resume-compatible "
            "full-grow envelope"
        )
    if parameter_envelope is not None and content_sha256(
        parameter_envelope
    ) != value["full_grow_envelope_sha256"]:
        raise ValueError(
            "benchmark_execution full-grow envelope hash does not match "
            "the task envelope"
        )

    contextual = {
        "sample_id": sample_id,
        "cell_id": cell_id,
        "policy_version": policy_version,
        "planner": planner,
        "seed_probe_mode": seed_probe_mode,
    }
    for field, expected_value in contextual.items():
        if (
            expected_value is not None
            and value[field] != expected_value
        ):
            raise ValueError(
                f"benchmark_execution {field} differs from its task"
            )
    return {
        **copy.deepcopy(value),
        "authorization": authorization,
    }


def build_seed_probe_benchmark_execution(
    authorization: dict[str, Any],
    *,
    arm: str,
    sample_id: str,
    cell_id: str,
    parameter_envelope: dict[str, Any],
) -> dict[str, Any]:
    """Build the only explicit-RNG contract accepted by a benchmark task."""

    normalized = normalize_seed_probe_benchmark_execution_authorization(
        authorization
    )
    if arm == "baseline":
        policy_version = normalized["baseline_policy_version"]
        seed_probe_mode = "off"
    elif arm == "closed_loop":
        policy_version = normalized["policy_version"]
        seed_probe_mode = "select"
    else:
        raise ValueError("benchmark execution arm must be baseline or closed_loop")
    value = {
        "schema": BENCHMARK_EXECUTION_SCHEMA,
        "execution_scope": "ISOLATED_NONPRODUCTION",
        "benchmark_id": normalized["benchmark_id"],
        "benchmark_spec_sha256": normalized["benchmark_spec_sha256"],
        "authorization": normalized,
        "authorization_sha256": content_sha256(normalized),
        "arm": arm,
        "policy_version": policy_version,
        "planner": "deterministic-v2",
        "seed_probe_mode": seed_probe_mode,
        "sample_id": str(sample_id),
        "cell_id": str(cell_id),
        "pair_rng_seed": benchmark_pair_rng_seed(
            normalized,
            sample_id=str(sample_id),
            cell_id=str(cell_id),
        ),
        "rng_protocol": BENCHMARK_RNG_PROTOCOL,
        "full_grow_envelope_sha256": content_sha256(parameter_envelope),
    }
    return normalize_seed_probe_benchmark_execution(
        value,
        sample_id=str(sample_id),
        cell_id=str(cell_id),
        policy_version=policy_version,
        planner="deterministic-v2",
        seed_probe_mode=seed_probe_mode,
        parameter_envelope=parameter_envelope,
    )


def validate_seed_probe_benchmark_execution_task(
    task: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate a task's explicit execution contract or historical absence."""

    if task.get("execution_rng_seed") is not None:
        raise ValueError(
            "execution_rng_seed is not a standalone override; use an exact "
            "isolated benchmark_execution contract"
        )
    raw = task.get("benchmark_execution")
    if raw is None:
        return None
    policy = task.get("seed_probe")
    mode = (
        str(policy.get("mode"))
        if isinstance(policy, dict)
        else "off"
    )
    return normalize_seed_probe_benchmark_execution(
        raw,
        sample_id=str(task.get("sample_id") or ""),
        cell_id=str(task.get("cell_id") or ""),
        policy_version=str(task.get("policy_version") or ""),
        planner=str(task.get("planner") or ""),
        seed_probe_mode=mode,
        parameter_envelope=task.get("parameter_envelope"),
    )


def validate_seed_probe_benchmark_receipt(
    value: Any,
) -> dict[str, Any]:
    """Validate an exact causal approval and return its compact task binding."""

    expected = {
        "schema",
        "benchmark_id",
        "status",
        "execution_scope",
        "spec_sha256",
        "results_sha256",
        "paired_cell_count",
        "scroll_count",
        "authorized_sample_ids",
        "arms",
        "metrics",
        "checks",
        "generated_at_utc",
        "non_claims",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            "seed probe benchmark receipt differs from v1 contract: "
            f"expected {sorted(expected)}, got {got}"
        )
    if value["schema"] != BENCHMARK_DECISION_SCHEMA:
        raise ValueError("unsupported seed probe benchmark decision schema")
    if value["status"] != "APPROVED_SELECT":
        raise ValueError("seed probe benchmark receipt is not APPROVED_SELECT")
    if value["execution_scope"] != "ISOLATED_NONPRODUCTION":
        raise ValueError(
            "seed probe select approval must come from ISOLATED_NONPRODUCTION"
        )
    benchmark_id = value["benchmark_id"]
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("benchmark receipt benchmark_id must be non-empty")
    for field in ("spec_sha256", "results_sha256", "receipt_sha256"):
        _require_lowercase_sha256(value[field], field)

    arms = value["arms"]
    if not isinstance(arms, dict) or set(arms) != {"baseline", "closed_loop"}:
        raise ValueError("benchmark receipt arms differ from v1 contract")
    expected_arm_keys = {"policy_version", "planner", "seed_probe_mode"}
    for name in ("baseline", "closed_loop"):
        arm = arms[name]
        if not isinstance(arm, dict) or set(arm) != expected_arm_keys:
            raise ValueError(f"benchmark receipt {name} arm differs from v1 contract")
        if any(not isinstance(arm[field], str) or not arm[field] for field in arm):
            raise ValueError(f"benchmark receipt {name} arm identity is incomplete")
    if (
        arms["baseline"]["planner"] != "deterministic-v2"
        or arms["baseline"]["seed_probe_mode"] != "off"
        or arms["closed_loop"]["planner"] != "deterministic-v2"
        or arms["closed_loop"]["seed_probe_mode"] != "select"
        or arms["baseline"]["policy_version"]
        == arms["closed_loop"]["policy_version"]
    ):
        raise ValueError(
            "benchmark receipt must compare distinct deterministic-v2/off and "
            "deterministic-v2/select arms"
        )

    checks = value["checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError("benchmark receipt checks must be a non-empty list")
    seen_checks: set[str] = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) != {
            "check_id",
            "status",
            "detail",
            "evidence",
        }:
            raise ValueError(
                f"benchmark receipt checks[{index}] differs from v1 contract"
            )
        check_id = check["check_id"]
        if not isinstance(check_id, str) or not check_id:
            raise ValueError(f"benchmark receipt checks[{index}] has no check_id")
        if check_id in seen_checks:
            raise ValueError(f"benchmark receipt has duplicate check {check_id}")
        seen_checks.add(check_id)
        if check["status"] != "PASS":
            raise ValueError(f"benchmark receipt check {check_id} did not PASS")
        if not isinstance(check["detail"], str) or not check["detail"]:
            raise ValueError(f"benchmark receipt check {check_id} has no detail")
    if seen_checks != BENCHMARK_REQUIRED_CHECK_IDS:
        raise ValueError(
            "benchmark receipt check set differs from v1 approval contract"
        )
    if not isinstance(value["metrics"], dict):
        raise ValueError("benchmark receipt metrics must be an object")
    per_scroll = value["metrics"].get("per_scroll")
    if not isinstance(per_scroll, dict) or any(
        not isinstance(sample_id, str) or not sample_id
        for sample_id in per_scroll
    ):
        raise ValueError(
            "benchmark receipt metrics.per_scroll must bind the approved "
            "scroll cohort"
        )
    if not isinstance(value["non_claims"], list) or not all(
        isinstance(item, str) for item in value["non_claims"]
    ):
        raise ValueError("benchmark receipt non_claims must be a string list")
    generated_at = value["generated_at_utc"]
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise ValueError("benchmark receipt generated_at_utc must be UTC")
    try:
        parsed_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "benchmark receipt generated_at_utc is invalid"
        ) from error
    if parsed_at.utcoffset() != timezone.utc.utcoffset(parsed_at):
        raise ValueError("benchmark receipt generated_at_utc must be UTC")

    authorization = {
        "schema": BENCHMARK_AUTHORIZATION_SCHEMA,
        "benchmark_id": benchmark_id,
        "decision_receipt_sha256": value["receipt_sha256"],
        "spec_sha256": value["spec_sha256"],
        "results_sha256": value["results_sha256"],
        "paired_cell_count": value["paired_cell_count"],
        "scroll_count": value["scroll_count"],
        "authorized_sample_ids": value["authorized_sample_ids"],
        "execution_scope": value["execution_scope"],
    }
    normalized_authorization = normalize_seed_probe_benchmark_authorization(
        authorization
    )
    if normalized_authorization["authorized_sample_ids"] != sorted(per_scroll):
        raise ValueError(
            "benchmark receipt authorized_sample_ids differs from its "
            "per-scroll evidence"
        )
    body = {
        key: item for key, item in value.items() if key != "receipt_sha256"
    }
    if content_sha256(body) != value["receipt_sha256"]:
        raise ValueError("benchmark receipt canonical SHA-256 does not match")
    return normalized_authorization


def load_seed_probe_benchmark_receipt(path: Path) -> dict[str, Any]:
    """Load one explicit receipt path and return its validated task binding."""

    try:
        value = read_json(Path(path))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read seed probe benchmark receipt: {error}") from error
    return validate_seed_probe_benchmark_receipt(value)


def normalize_seed_probe_policy(value: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact v1 policy instead of accepting ignored knobs."""

    if not isinstance(value, dict):
        raise ValueError("seed_probe must be an object")
    expected = {
        "schema",
        "policy_id",
        "mode",
        "top_k",
        "probe_profile_id",
        "probe_profile_sha256",
        "probe_parameters",
        "maximum_total_probe_generations",
        "maximum_attempts_per_candidate",
        "evaluation_profile_id",
        "evaluation_profile_sha256",
        "decision_policy_id",
        "inconclusive_action",
        "ink_used",
    }
    actual = set(value)
    optional = {"benchmark_authorization", "review_owner"}
    if not expected.issubset(actual) or actual.difference(expected) - optional:
        raise ValueError(
            "seed_probe fields differ from v1 contract: "
            f"expected {sorted(expected)} with conditional "
            "benchmark_authorization/review_owner, "
            f"got {sorted(value)}"
        )
    if value["schema"] != "campaignx.seed_probe_policy.v1":
        raise ValueError("unsupported seed probe schema")
    if value["policy_id"] != "seed-probe-v1":
        raise ValueError("unsupported seed probe policy")
    mode = str(value["mode"]).lower()
    if mode not in {"shadow", "select"}:
        raise ValueError("seed probe mode must be shadow or select")
    top_k = int(value["top_k"])
    # The fourth place this bound lived. Three made the probe a tie-break
    # between m7's best candidates, and the first shadow run ever executed
    # measured 8 of 8 ELIGIBLE at that depth -- a 0% rejection rate against a
    # 34% break-even, so probing paid for nothing. Whether m7's ordering holds
    # at rank 20 is the open question, and this is what made it unaskable.
    #
    # maximum_attempts below is a different number and stays at 3: it bounds
    # retries per candidate, not how far down the ordering to look.
    if not 1 <= top_k <= TOP_K_MAXIMUM:
        raise ValueError(f"seed probe top_k must be between 1 and {TOP_K_MAXIMUM}")
    if mode == "select" and top_k < 2:
        raise ValueError(
            "seed probe select mode requires at least two candidates; "
            "top_k=1 is a geometry preflight, not a comparison"
        )
    if value["probe_profile_id"] != PROBE_PROFILE_ID:
        raise ValueError("unsupported seed probe profile")
    if value["probe_profile_sha256"] != PROBE_PROFILE_SHA256:
        raise ValueError("seed probe profile SHA-256 does not match v1")
    parameters = value["probe_parameters"]
    if not isinstance(parameters, dict) or set(parameters) != {
        "generations",
        "step_size",
        "min_area_cm",
        "use_cuda",
    }:
        raise ValueError("seed probe parameters differ from the v1 profile")
    generations = int(parameters["generations"])
    step_size = int(parameters["step_size"])
    if not 10 <= generations <= 20:
        raise ValueError("seed probe generations must be between 10 and 20")
    if not 8 <= step_size <= 15:
        raise ValueError("seed probe step_size must be between 8 and 15")
    if float(parameters["min_area_cm"]) != 0.0:
        raise ValueError("seed probe min_area_cm is frozen at 0")
    if parameters["use_cuda"] is not False:
        raise ValueError("seed probe v1 use_cuda is frozen at false")
    maximum_attempts = int(value["maximum_attempts_per_candidate"])
    if not 1 <= maximum_attempts <= 3:
        raise ValueError("maximum probe attempts per candidate must be 1..3")
    maximum_total = int(value["maximum_total_probe_generations"])
    required_total = top_k * generations * maximum_attempts
    if maximum_total != required_total or maximum_total > 180:
        raise ValueError(
            "maximum_total_probe_generations must equal top_k * generations "
            "* maximum_attempts_per_candidate and may not exceed 180"
        )
    if value["evaluation_profile_id"] != PROBE_EVALUATION_PROFILE_ID:
        raise ValueError("unsupported seed probe evaluation profile")
    if (
        value["evaluation_profile_sha256"]
        != PROBE_EVALUATION_PROFILE_SHA256
    ):
        raise ValueError("seed probe evaluation profile SHA-256 does not match v1")
    if value["decision_policy_id"] != "unique-geometry-certified-v1":
        raise ValueError("unsupported seed probe decision policy")
    if value["inconclusive_action"] != "HUMAN_REVIEW":
        raise ValueError("seed probe v1 must send inconclusive evidence to review")
    if value["ink_used"] is not False:
        raise ValueError("seed probes must be ink-blind")
    authorization = value.get("benchmark_authorization")
    if authorization is not None and mode != "select":
        raise ValueError(
            "benchmark authorization may only be bound to seed probe select mode"
        )
    normalized_authorization = None
    review_owner = value.get("review_owner")
    if authorization is not None:
        authorization_schema = (
            authorization.get("schema")
            if isinstance(authorization, dict)
            else None
        )
        if authorization_schema == BENCHMARK_AUTHORIZATION_SCHEMA:
            normalized_authorization = (
                normalize_seed_probe_benchmark_authorization(authorization)
            )
        elif authorization_schema == BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA:
            normalized_authorization = (
                normalize_seed_probe_benchmark_execution_authorization(
                    authorization
                )
            )
        else:
            raise ValueError(
                "unsupported seed probe benchmark authorization schema"
            )
    if (
        isinstance(normalized_authorization, dict)
        and normalized_authorization.get("schema")
        == BENCHMARK_AUTHORIZATION_SCHEMA
    ):
        if (
            not isinstance(review_owner, str)
            or not review_owner.strip()
            or len(review_owner) > 200
            or any(ord(character) < 32 for character in review_owner)
        ):
            raise ValueError(
                "production-approved seed probe select requires a non-empty "
                "review_owner"
            )
        review_owner = review_owner.strip()
    elif review_owner is not None:
        raise ValueError(
            "review_owner is only valid with production benchmark authorization"
        )
    return {
        **value,
        "mode": mode,
        "top_k": top_k,
        "probe_parameters": {
            "generations": generations,
            "step_size": step_size,
            "min_area_cm": 0.0,
            "use_cuda": parameters["use_cuda"],
        },
        "maximum_total_probe_generations": maximum_total,
        "maximum_attempts_per_candidate": maximum_attempts,
        **(
            {"benchmark_authorization": normalized_authorization}
            if normalized_authorization is not None
            else {}
        ),
        **({"review_owner": review_owner} if review_owner is not None else {}),
    }


def verify_probe_evaluation_profile() -> dict[str, Any]:
    """Fail closed if deployed gate code differs from the frozen probe profile."""

    profile_path = (
        Path(__file__).resolve().parent
        / "profiles/tifxyz-geometry-gate-probe-v1.json"
    )
    if file_sha256(profile_path) != PROBE_EVALUATION_PROFILE_SHA256:
        raise RuntimeError("seed probe evaluation profile bytes have drifted")
    profile = read_json(profile_path)
    repository = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "framework").is_dir()
        ),
        None,
    )
    if repository is None:
        raise RuntimeError("cannot locate repository root for probe gate audit")
    for field in ("gate", "integrity_dependency"):
        descriptor = profile[field]
        implementation = repository / descriptor["path"]
        if file_sha256(implementation) != descriptor["sha256"]:
            raise RuntimeError(
                f"seed probe {field} implementation differs from its "
                "frozen evaluation profile"
            )
    return profile


def verify_probe_growth_profile() -> dict[str, Any]:
    """Fail closed if the deployed VC3D probe profile bytes drift."""

    profile_path = (
        Path(__file__).resolve().parent / "profiles/vc3d-m7-probe-v1.json"
    )
    if file_sha256(profile_path) != PROBE_PROFILE_SHA256:
        raise RuntimeError("seed probe VC3D profile bytes have drifted")
    profile = read_json(profile_path)
    if (
        profile.get("profile_id") != PROBE_PROFILE_ID
        or profile.get("noncanonical") is not True
        or profile.get("ink_used") is not False
    ):
        raise RuntimeError("seed probe VC3D profile contract is invalid")
    return profile


def normalize_source_content_lock(source: dict[str, Any]) -> dict[str, Any]:
    """Validate that select mode reads immutable, content-verified inputs.

    A digest beside a mutable URI is not a lock: the grower would still read
    whatever bytes happen to live at that URI.  This contract therefore binds
    the digest, the URI actually read, and a version identifier embedded in
    that URI.  Source intake is responsible for producing the verification
    receipt; the probe path only consumes an exact frozen receipt.
    """

    if not isinstance(source, dict):
        raise ValueError("seed probe source must be an object")
    lock = source.get("source_content_lock")
    expected = {
        "schema",
        "status",
        "verification_method",
        "verified_at_utc",
        "ct_uri",
        "ct_sha256",
        "ct_version_id",
        "m7_uri",
        "m7_sha256",
        "m7_version_id",
    }
    if not isinstance(lock, dict) or set(lock) != expected:
        raise ValueError(
            "seed probe select mode requires an exact source_content_lock v1"
        )
    if (
        lock["schema"] != SOURCE_CONTENT_LOCK_SCHEMA
        or lock["status"] != "VERIFIED_IMMUTABLE"
    ):
        raise ValueError("seed probe source content lock is not verified immutable")
    fixture = str(source.get("ct_uri", "")).startswith("fixture://") and str(
        source.get("m7_uri", "")
    ).startswith("fixture://")
    expected_method = (
        "fixture-sha256-v1"
        if fixture
        else "immutable-uri-manifest-sha256-v1"
    )
    if lock["verification_method"] != expected_method:
        raise ValueError("seed probe source lock uses an unsupported verifier")
    verified_at = str(lock["verified_at_utc"])
    try:
        verified_time = datetime.fromisoformat(
            verified_at.replace("Z", "+00:00")
        )
    except ValueError:
        verified_time = None
    if (
        len(verified_at) < 20
        or not verified_at.endswith("Z")
        or verified_time is None
        or verified_time.utcoffset() != timezone.utc.utcoffset(verified_time)
    ):
        raise ValueError("seed probe source lock needs a UTC verification time")
    for prefix in ("ct", "m7"):
        uri = str(source.get(f"{prefix}_uri") or "")
        digest = str(source.get(f"{prefix}_sha256") or "")
        version_id = str(lock[f"{prefix}_version_id"] or "")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"seed probe select mode needs a lowercase {prefix} SHA-256"
            )
        if (
            lock[f"{prefix}_uri"] != uri
            or lock[f"{prefix}_sha256"] != digest
        ):
            raise ValueError(
                f"seed probe {prefix} lock does not match the source snapshot"
            )
        if len(version_id) < 8 or version_id not in uri:
            raise ValueError(
                f"seed probe {prefix} URI does not embed its immutable version"
            )
    return copy.deepcopy(lock)


def frozen_probe_candidates(
    candidates: list[dict[str, Any]], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Take a stable, bounded candidate set without changing legacy ranking."""

    normalized = normalize_seed_probe_policy(policy)
    ordered = sorted(candidates, key=candidate_rank_key)
    frozen: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ordered[: normalized["top_k"]], start=1):
        if any(key not in candidate for key in ("candidate_id", "x", "y", "z")):
            raise ValueError("seed probe candidate lacks exact id/XYZ")
        frozen.append(
            {
                "candidate_rank": rank,
                "candidate_id": str(candidate["candidate_id"]),
                "x": int(candidate["x"]),
                "y": int(candidate["y"]),
                "z": int(candidate["z"]),
                "score": float(candidate.get("score", 0.0)),
                "cell_interior_clearance_voxels": candidate.get(
                    "cell_interior_clearance_voxels"
                ),
                "volume_interior_clearance_voxels": candidate.get(
                    "volume_interior_clearance_voxels"
                ),
                "source": candidate.get("source"),
            }
        )
    if not frozen:
        raise ValueError("a seed probe run requires at least one candidate")
    return frozen


def executor_fingerprint(executor: Any) -> dict[str, Any]:
    """Bind probe reuse to the implementation that produced the bytes."""

    binary = getattr(executor, "binary", None)
    if binary is not None:
        path = Path(binary)
        if not path.is_file():
            raise FileNotFoundError(f"VC3D grow binary is unavailable: {path}")
        return {
            "executor": "vc3d",
            "binary_sha256": file_sha256(path),
        }
    return {
        "executor": type(executor).__name__,
        "fixture_only": type(executor).__name__ == "FixtureGrowExecutor",
    }


def build_probe_locked_plan(
    task: dict[str, Any],
    trial: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-planner plan used for one identical micro-experiment."""

    normalized = normalize_seed_probe_policy(policy)
    benchmark_execution = validate_seed_probe_benchmark_execution_task(task)
    candidate = trial["candidate"]
    return {
        "schema": "campaignx.seed_probe_locked_plan.v1",
        "status": "LOCKED_READY",
        "task_id": task["task_id"],
        # Stable logical trial identity, not retry identity.  The executor uses
        # this to derive VC_GROWPATCH_RNG_SEED, so recovery is byte-repeatable.
        "attempt_id": trial["probe_trial_id"],
        "probe_run_id": trial["probe_run_id"],
        "probe_trial_id": trial["probe_trial_id"],
        "source_snapshot_id": task["source"]["source_snapshot_id"],
        "sample_id": task["sample_id"],
        "cell": {
            "cell_id": task["cell_id"],
            "bounds_xyz": task["bounds_xyz"],
            "center_xyz": task["center_xyz"],
        },
        "source": {
            "source_snapshot_id": task["source"]["source_snapshot_id"],
            "ct_uri": task["source"]["ct_uri"],
            "ct_sha256": task["source"].get("ct_sha256"),
            "m7_uri": task["source"]["m7_uri"],
            "m7_sha256": task["source"].get("m7_sha256"),
            "shape_xyz": task["source"]["shape_xyz"],
            "voxel_size_um": task["source"]["voxel_size_um"],
            "coordinate_frame": task["source"].get(
                "coordinate_frame", "ct_l0_xyz"
            ),
            **(
                {
                    "source_content_lock": normalize_source_content_lock(
                        task["source"]
                    )
                }
                if normalized["mode"] == "select"
                else {}
            ),
        },
        "selected_seed": {
            key: candidate[key] for key in ("candidate_id", "x", "y", "z")
        },
        "profile_id": normalized["probe_profile_id"],
        "profile_sha256": normalized["probe_profile_sha256"],
        "parameters": dict(normalized["probe_parameters"]),
        "policy_sha256": content_sha256(normalized),
        **(
            {
                "benchmark_execution": benchmark_execution,
                "parameter_envelope_sha256": benchmark_execution[
                    "full_grow_envelope_sha256"
                ],
            }
            if benchmark_execution is not None
            else {}
        ),
        "noncanonical": True,
        "ink_used": False,
        "non_claims": [
            "this plan can only produce a noncanonical seed probe",
            "a seed probe is not physical-sheet or correct-lamina acceptance",
        ],
    }


def validate_probe_locked_plan(
    *,
    task: dict[str, Any],
    trial: dict[str, Any],
    policy: dict[str, Any],
    locked_plan: dict[str, Any],
) -> dict[str, Any]:
    """Return the only plan the ledger may accept for this logical trial."""

    expected = build_probe_locked_plan(task, trial, policy)
    if locked_plan != expected:
        raise ValueError(
            "probe locked plan differs from the authoritative task/trial/policy plan"
        )
    return expected


def expected_probe_artifact_uri(
    *,
    namespace_identity: dict[str, Any],
    sample_id: str,
    probe_run_id: str,
    probe_trial_id: str,
    artifact_sha256: str,
) -> str:
    """Derive the only URI a persisted namespace may publish for this trial."""

    components = tuple(
        str(value)
        for value in (
            sample_id,
            probe_run_id,
            probe_trial_id,
            artifact_sha256,
        )
    )
    if any(
        not value or "/" in value or "\\" in value or value in {".", ".."}
        for value in components
    ):
        raise ValueError("probe artifact identity contains an unsafe path component")
    backend = (
        namespace_identity.get("backend")
        if isinstance(namespace_identity, dict)
        and namespace_identity.get("schema")
        == "campaignx.seed_probe_namespace.v1"
        else None
    )
    if backend == "local" and set(namespace_identity) == {
        "schema",
        "backend",
        "probe_root",
    }:
        root = Path(str(namespace_identity["probe_root"]))
        if not root.is_absolute():
            raise ValueError("local probe namespace identity is invalid")
        return str(root.joinpath(*components).resolve())
    if backend == "s3" and set(namespace_identity) == {
        "schema",
        "backend",
        "bucket",
        "probe_prefix",
    }:
        bucket = str(namespace_identity["bucket"])
        prefix = str(namespace_identity["probe_prefix"]).strip("/")
        if not bucket or not prefix:
            raise ValueError("probe S3 namespace identity is invalid")
        expected_path = "/".join((prefix, *components))
        return f"s3://{bucket}/{expected_path}"
    raise ValueError("probe executor fingerprint has no valid artifact namespace")


def validate_probe_artifact_uri(
    *,
    artifact_uri: str,
    namespace_identity: dict[str, Any],
    sample_id: str,
    probe_run_id: str,
    probe_trial_id: str,
    artifact_sha256: str,
) -> None:
    """Accept only the exact local/S3 noncanonical publication namespace."""

    parsed = urlparse(str(artifact_uri))
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("probe artifact URI may not contain parameters")
    expected = expected_probe_artifact_uri(
        namespace_identity=namespace_identity,
        sample_id=sample_id,
        probe_run_id=probe_run_id,
        probe_trial_id=probe_trial_id,
        artifact_sha256=artifact_sha256,
    )
    if str(artifact_uri) != expected:
        raise ValueError(
            "probe artifact URI is outside the configured publication namespace"
        )


def evaluate_probe_surface(
    surface_dir: Path,
    *,
    task: dict[str, Any],
    probe_run_id: str,
    probe_trial_id: str,
    probe_artifact_set_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure a probe with the normal gate and map it to a probe verdict."""

    inspection = inspect_tifxyz(
        surface_dir, float(task["source"]["voxel_size_um"])
    )
    geometry = certify_surface_geometry(surface_dir)
    state = str(geometry.get("geometry_qc_state") or "GEOMETRY_UNMEASURED")
    complete = bool(geometry.get("measurement_complete"))
    resolution_limited = geometry.get("resolution_limited")
    if (
        state == "GEOMETRY_CERTIFIED"
        and complete
        and resolution_limited is False
    ):
        verdict = "ELIGIBLE"
    elif state.startswith("GEOMETRY_REJECTED_") and complete:
        verdict = "REJECTED"
    else:
        verdict = "UNMEASURED"
    # The geometry gate records the local directory it inspected.  That is
    # useful in an operator receipt but is deliberately not scientific
    # evidence: a crash retry writes identical bytes under a new attempt
    # directory.  Bind the decision to reproducible measurements, not that
    # transient path (or a path-bearing exception string).
    stable_geometry = stable_probe_geometry(geometry)
    evaluation = {
        "schema": "campaignx.seed_probe_evaluation.v1",
        "probe_run_id": probe_run_id,
        "probe_trial_id": probe_trial_id,
        "probe_artifact_set_id": probe_artifact_set_id,
        "profile_id": "tifxyz-geometry-gate-probe-v1",
        "profile_sha256": PROBE_EVALUATION_PROFILE_SHA256,
        "verdict": verdict,
        "geometry_qc_state": state,
        "measurement_complete": complete,
        "resolution_limited": resolution_limited,
        "inspection": {
            key: inspection[key]
            for key in (
                "shape",
                "finite_coordinate_count",
                "valid_triangle_count",
                "bbox_xyz",
                "area_cm2",
            )
        },
        "geometry_certification_sha256": content_sha256(stable_geometry),
        "ink_used": False,
        "non_claims": [
            "ELIGIBLE means only that this micro-patch passed the frozen geometry gate",
            "the geometry gate does not prove that a patch follows the correct physical lamina",
        ],
    }
    return evaluation, geometry


def stable_probe_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Strip only execution-location fields from geometry evidence."""

    return {
        key: value
        for key, value in geometry.items()
        if key not in {"surface_dir", "surface_name", "error"}
    }


def validate_probe_completion_evidence(
    *,
    evaluation: dict[str, Any],
    geometry: dict[str, Any],
    growth_receipt: dict[str, Any],
    artifact_uri: str,
    artifact_manifest: dict[str, Any],
    task_id: str,
    probe_run_id: str,
    probe_trial_id: str,
    probe_artifact_set_id: str,
    locked_plan_sha256: str,
    sample_id: str,
    namespace_identity: dict[str, Any],
    policy: dict[str, Any],
) -> str:
    """Cross-bind every receipt before the ledger can assign a verdict."""

    expected_evaluation_keys = {
        "schema",
        "probe_run_id",
        "probe_trial_id",
        "probe_artifact_set_id",
        "profile_id",
        "profile_sha256",
        "verdict",
        "geometry_qc_state",
        "measurement_complete",
        "resolution_limited",
        "inspection",
        "geometry_certification_sha256",
        "ink_used",
        "non_claims",
    }
    if (
        not isinstance(evaluation, dict)
        or set(evaluation) != expected_evaluation_keys
        or evaluation.get("schema")
        != "campaignx.seed_probe_evaluation.v1"
        or evaluation.get("probe_run_id") != probe_run_id
        or evaluation.get("probe_trial_id") != probe_trial_id
        or evaluation.get("probe_artifact_set_id")
        != probe_artifact_set_id
        or evaluation.get("profile_id") != policy["evaluation_profile_id"]
        or evaluation.get("profile_sha256")
        != policy["evaluation_profile_sha256"]
        or evaluation.get("ink_used") is not False
    ):
        raise ValueError("probe evaluation is not bound to this frozen trial")
    inspection = evaluation.get("inspection")
    if not isinstance(inspection, dict) or set(inspection) != {
        "shape",
        "finite_coordinate_count",
        "valid_triangle_count",
        "bbox_xyz",
        "area_cm2",
    }:
        raise ValueError("probe evaluation inspection has unexpected fields")
    if (
        evaluation["geometry_qc_state"]
        != geometry.get("geometry_qc_state")
        or evaluation["measurement_complete"]
        is not geometry.get("measurement_complete")
        or evaluation["resolution_limited"]
        is not geometry.get("resolution_limited")
        or evaluation["geometry_certification_sha256"]
        != content_sha256(stable_probe_geometry(geometry))
    ):
        raise ValueError("probe evaluation does not match geometry evidence")
    state = str(evaluation["geometry_qc_state"])
    complete = evaluation["measurement_complete"] is True
    resolution_limited = evaluation["resolution_limited"]
    expected_verdict = (
        "ELIGIBLE"
        if state == "GEOMETRY_CERTIFIED"
        and complete
        and resolution_limited is False
        else "REJECTED"
        if state.startswith("GEOMETRY_REJECTED_") and complete
        else "UNMEASURED"
    )
    if evaluation.get("verdict") != expected_verdict:
        raise ValueError("probe verdict contradicts its geometry state")
    if (
        growth_receipt.get("schema")
        != "campaignx.segment_fleet_growth_receipt.v1"
        or growth_receipt.get("status") != "GROW_SUCCEEDED"
        or growth_receipt.get("task_id") != task_id
        or growth_receipt.get("attempt_id") != probe_trial_id
        or growth_receipt.get("locked_plan_sha256") != locked_plan_sha256
        or growth_receipt.get("ink_used") is not False
    ):
        raise ValueError("probe growth receipt is not bound to the locked trial")
    if (
        artifact_manifest.get("probe_run_id") != probe_run_id
        or artifact_manifest.get("probe_trial_id") != probe_trial_id
        or artifact_manifest.get("locked_plan_sha256")
        != locked_plan_sha256
    ):
        raise ValueError("probe artifact manifest is not bound to the locked trial")
    artifact_sha = str(artifact_manifest.get("artifact_sha256") or "")
    validate_probe_artifact_uri(
        artifact_uri=artifact_uri,
        namespace_identity=namespace_identity,
        sample_id=sample_id,
        probe_run_id=probe_run_id,
        probe_trial_id=probe_trial_id,
        artifact_sha256=artifact_sha,
    )
    return expected_verdict


def decide_probe_run(run: dict[str, Any]) -> dict[str, Any]:
    """Apply the categorical v1 decision matrix; no weighted pseudo-score."""

    policy = normalize_seed_probe_policy(run["policy"])
    trials = sorted(run["trials"], key=lambda row: int(row["candidate_rank"]))
    if not trials:
        raise ValueError("cannot decide an empty probe run")
    if any(str(row["state"]) not in PROBE_TERMINAL_TRIAL_STATES for row in trials):
        raise ValueError("all probe trials must be terminal before decision")
    outcomes = [
        {
            "probe_trial_id": str(row["probe_trial_id"]),
            "candidate_id": str(row["candidate"]["candidate_id"]),
            "candidate_rank": int(row["candidate_rank"]),
            "state": str(row["state"]),
            "verdict": (
                str(row["evaluation"]["verdict"])
                if isinstance(row.get("evaluation"), dict)
                else None
            ),
            "geometry_qc_state": (
                row["evaluation"].get("geometry_qc_state")
                if isinstance(row.get("evaluation"), dict)
                else None
            ),
            "evaluation_sha256": (
                content_sha256(row["evaluation"])
                if isinstance(row.get("evaluation"), dict)
                else None
            ),
        }
        for row in trials
    ]
    eligible = [row for row in outcomes if row["verdict"] == "ELIGIBLE"]
    rejected = [row for row in outcomes if row["verdict"] == "REJECTED"]
    inconclusive = [
        row
        for row in outcomes
        if row["verdict"] not in {"ELIGIBLE", "REJECTED"}
    ]
    winner: str | None = None
    if len(outcomes) < 2:
        action = "HUMAN_REVIEW"
        reason = "INSUFFICIENT_CANDIDATES_FOR_COMPARISON"
    elif inconclusive:
        action = "HUMAN_REVIEW"
        reason = "AT_LEAST_ONE_PROBE_WAS_UNMEASURED_OR_FAILED"
    elif len(eligible) == 1 and len(rejected) == len(outcomes) - 1:
        action = "CONTINUE_WINNER"
        winner = eligible[0]["probe_trial_id"]
        reason = "EXACTLY_ONE_GEOMETRY_ELIGIBLE_PROBE"
    elif len(eligible) > 1:
        action = "HUMAN_REVIEW"
        reason = "MULTIPLE_GEOMETRY_ELIGIBLE_PROBES"
    elif len(rejected) == len(outcomes):
        action = "REJECT_ALL"
        reason = "ALL_PROBED_CANDIDATES_GEOMETRY_REJECTED"
    else:
        action = "HUMAN_REVIEW"
        reason = "INCONCLUSIVE_PROBE_EVIDENCE"
    evidence_set_sha256 = content_sha256(outcomes)
    policy_sha256 = content_sha256(policy)
    decision = {
        "schema": "campaignx.seed_probe_decision.v1",
        "probe_run_id": str(run["probe_run_id"]),
        "policy_id": policy["decision_policy_id"],
        "policy_sha256": policy_sha256,
        "evidence_set_sha256": evidence_set_sha256,
        "action": action,
        "winner_trial_id": winner,
        "reason": reason,
        "trial_outcomes": outcomes,
        "ink_used": False,
        "non_claims": [
            "a probe winner is not a correct-sheet or correct-lamina certificate",
            "REJECT_ALL applies only to the candidates this bounded run probed",
        ],
    }
    if action not in PROBE_DECISION_ACTIONS:
        raise RuntimeError("seed probe decision escaped its frozen action set")
    return decision


def seed_probe_resume_envelope(
    parameter_envelope: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Return the only full-grow envelope that can resume this select probe."""

    normalized = normalize_seed_probe_policy(policy)
    if not isinstance(parameter_envelope, dict):
        raise ValueError("full growth parameter_envelope must be an object")
    envelope = copy.deepcopy(parameter_envelope)
    if normalized["mode"] != "select":
        return envelope
    if not isinstance(envelope.get("parameters"), dict):
        raise ValueError("full growth envelope must contain parameters")
    parameters = envelope["parameters"]
    probe_parameters = normalized["probe_parameters"]
    generation_rule = dict(parameters.get("generations") or {})
    if generation_rule.get("type") != "integer":
        raise ValueError("full growth generations must be an integer envelope")
    reached_generations = int(probe_parameters["generations"])
    minimum_generations = max(
        int(generation_rule.get("minimum", 0)),
        reached_generations + 1,
    )
    maximum_generations = int(generation_rule.get("maximum", -1))
    if minimum_generations > maximum_generations:
        raise ValueError(
            "full growth envelope has no generation target after the probe"
        )
    generation_rule["minimum"] = minimum_generations
    default_generations = int(
        generation_rule.get("default", minimum_generations)
    )
    generation_rule["default"] = min(
        maximum_generations,
        max(minimum_generations, default_generations),
    )
    parameters["generations"] = generation_rule
    for name in ("step_size", "min_area_cm", "use_cuda"):
        if name not in parameters:
            raise ValueError(
                f"full growth envelope lacks resume-compatibility parameter {name}"
            )
        frozen = probe_parameters[name]
        rule = parameters[name]
        if "minimum" in rule and frozen < rule["minimum"]:
            raise ValueError(f"probe {name} is below the full envelope")
        if "maximum" in rule and frozen > rule["maximum"]:
            raise ValueError(f"probe {name} is above the full envelope")
        if "const" in rule and frozen != rule["const"]:
            raise ValueError(f"probe {name} conflicts with the full envelope")
        parameters[name] = {
            "type": "boolean" if isinstance(frozen, bool) else (
                "integer" if isinstance(frozen, int) else "number"
            ),
            "const": frozen,
            "default": frozen,
        }
    return envelope


def validate_seed_probe_task_contract(task: dict[str, Any]) -> dict[str, Any]:
    """Validate source and continuation inputs before any candidate bytes are read."""

    policy = normalize_seed_probe_policy(task.get("seed_probe"))
    benchmark_execution = validate_seed_probe_benchmark_execution_task(task)
    source = task.get("source")
    if not isinstance(source, dict):
        raise ValueError("seed probe task has no bound source snapshot")
    source_m7_uri = str(source.get("m7_uri") or "")
    if not source_m7_uri:
        raise ValueError("seed probe source has no m7_uri")

    lock: dict[str, Any] | None = None
    if policy["mode"] == "select":
        lock = normalize_source_content_lock(source)
        seed_probe_resume_envelope(task.get("parameter_envelope"), policy)
        authorization = policy.get("benchmark_authorization")
        if authorization is None:
            if benchmark_execution is not None:
                raise ValueError(
                    "benchmark_execution requires its preregistered "
                    "authorization in the select policy"
                )
            if lock.get("verification_method") != "fixture-sha256-v1":
                raise ValueError(
                    "seed probe select on a non-fixture source requires a "
                    "production approval or preregistered isolated benchmark "
                    "authorization"
                )
        elif (
            authorization.get("schema")
            == BENCHMARK_EXECUTION_AUTHORIZATION_SCHEMA
        ):
            identity = {
                "sample_id": str(task.get("sample_id") or ""),
                "cell_id": str(task.get("cell_id") or ""),
            }
            if not any(
                cell["sample_id"] == identity["sample_id"]
                and cell["cell_id"] == identity["cell_id"]
                for cell in authorization["cells"]
            ):
                raise ValueError(
                    "seed probe benchmark task is outside the preregistered "
                    "isolated cohort"
                )
            if (
                task.get("policy_version")
                != authorization["policy_version"]
                or task.get("planner") != authorization["planner"]
            ):
                raise ValueError(
                    "seed probe benchmark task differs from its preregistered "
                    "closed_loop arm"
                )
            if (
                benchmark_execution is None
                or benchmark_execution["arm"] != "closed_loop"
                or benchmark_execution["authorization"] != authorization
            ):
                raise ValueError(
                    "preregistered select requires its exact isolated "
                    "benchmark_execution binding"
                )
        else:
            if str(task.get("sample_id") or "") not in authorization[
                "authorized_sample_ids"
            ]:
                raise ValueError(
                    "production-approved select task sample is outside the "
                    "authorized non-regressing scroll cohort"
                )
            if benchmark_execution is not None:
                raise ValueError(
                    "a production-approved select task may not carry an "
                    "isolated benchmark_execution contract"
                )

    discovery = task.get("candidate_discovery")
    if isinstance(discovery, dict):
        provider = str(discovery.get("provider") or "vc3d-mcp")
        if provider == "vc3d-mcp":
            if str(discovery.get("prediction_uri") or "") != source_m7_uri:
                raise ValueError(
                    "seed probe candidate discovery URI does not match the "
                    "content-locked m7 URI"
                )
            if str(discovery.get("prediction_space") or "ct_l0_xyz") != "ct_l0_xyz":
                raise ValueError(
                    "seed probe candidate discovery must use ct_l0_xyz"
                )
        elif policy["mode"] == "select":
            raise ValueError(
                "seed probe select requires vc3d-mcp candidate discovery"
            )
    elif policy["mode"] == "select":
        fixture_candidates = task.get("recorded_candidates")
        if not (
            isinstance(lock, dict)
            and lock.get("verification_method") == "fixture-sha256-v1"
            and isinstance(fixture_candidates, list)
            and fixture_candidates
        ):
            raise ValueError(
                "seed probe select requires source-bound m7 candidate discovery"
            )
    return policy


def _resume_compatible_task(
    task: dict[str, Any],
    *,
    decision: dict[str, Any],
    winner: dict[str, Any],
    materialized_surface: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Constrain a full planner packet to the winner and compatible topology."""

    normalized = normalize_seed_probe_policy(policy)
    value = copy.deepcopy(task)
    value["parameter_envelope"] = seed_probe_resume_envelope(
        value["parameter_envelope"], normalized
    )
    reached_generations = int(
        normalized["probe_parameters"]["generations"]
    )
    value["resume_from"] = str(materialized_surface)
    value["resume_artifact"] = {
        "schema": "campaignx.seed_probe_resume_artifact.v1",
        "probe_run_id": decision["probe_run_id"],
        "probe_trial_id": winner["probe_trial_id"],
        "probe_artifact_set_id": winner["probe_artifact_set_id"],
        "artifact_uri": winner["artifact_uri"],
        "manifest_sha256": winner["manifest_sha256"],
        "materialized_path": str(materialized_surface),
        "reached_generations": reached_generations,
        "noncanonical": True,
    }
    value["seed_probe_decision"] = decision
    return value


def build_probe_continuation_contract(
    *,
    task_id: str,
    continuation_attempt_id: str,
    probe_run_id: str,
    source_snapshot_id: str,
    decision_id: str,
    decision_sha256: str,
    winner_trial_id: str,
    winner_probe_artifact_set_id: str,
    winner_manifest_sha256: str,
    winner_artifact_uri: str,
    winner_candidate: dict[str, Any],
    locked_plan: dict[str, Any],
) -> dict[str, Any]:
    """Bind a canonical continuation to stable winner evidence."""

    if (
        locked_plan.get("schema") != "campaignx.segmentation_locked_plan.v2"
        or locked_plan.get("status") != "LOCKED_READY"
        or locked_plan.get("task_id") != task_id
        or locked_plan.get("attempt_id") != continuation_attempt_id
        or locked_plan.get("source_snapshot_id") != source_snapshot_id
        or locked_plan.get("ink_used") is not False
    ):
        raise ValueError(
            "probe continuation is not an exact locked v2 plan for its task"
        )
    decision = locked_plan.get("seed_probe_decision")
    if (
        not isinstance(decision, dict)
        or decision.get("schema") != "campaignx.seed_probe_decision.v1"
        or decision.get("probe_run_id") != probe_run_id
        or decision.get("action") != "CONTINUE_WINNER"
        or decision.get("winner_trial_id") != winner_trial_id
        or decision.get("ink_used") is not False
        or content_sha256(decision) != decision_sha256
        or locked_plan.get("seed_probe_decision_sha256")
        != decision_sha256
    ):
        raise ValueError(
            "locked continuation does not carry the persisted probe decision"
        )
    winner_outcomes = [
        row
        for row in decision.get("trial_outcomes", [])
        if isinstance(row, dict)
        and row.get("probe_trial_id") == winner_trial_id
    ]
    candidate_id = str(winner_candidate.get("candidate_id") or "")
    if (
        len(winner_outcomes) != 1
        or winner_outcomes[0].get("candidate_id") != candidate_id
        or winner_outcomes[0].get("verdict") != "ELIGIBLE"
    ):
        raise ValueError("persisted winner and continuation candidate differ")

    expected_seed = {
        "candidate_id": candidate_id,
        **{
            axis: float(winner_candidate[axis])
            for axis in "xyz"
        },
    }
    selected_seed = locked_plan.get("selected_seed")
    if (
        not isinstance(selected_seed, dict)
        or selected_seed.get("candidate_id") != expected_seed["candidate_id"]
        or any(
            float(selected_seed.get(axis)) != expected_seed[axis]
            for axis in "xyz"
        )
    ):
        raise ValueError(
            "locked continuation did not select the exact probe winner"
        )

    resume = locked_plan.get("resume_artifact")
    stable_resume = {
        "probe_run_id": probe_run_id,
        "probe_trial_id": winner_trial_id,
        "probe_artifact_set_id": winner_probe_artifact_set_id,
        "artifact_uri": winner_artifact_uri,
        "manifest_sha256": winner_manifest_sha256,
    }
    if (
        not isinstance(resume, dict)
        or any(resume.get(key) != value for key, value in stable_resume.items())
        or resume.get("noncanonical") is not True
        or not isinstance(resume.get("reached_generations"), int)
        or int(resume["reached_generations"]) < 1
        or locked_plan.get("resume_from") != resume.get("materialized_path")
    ):
        raise ValueError(
            "locked continuation does not carry the exact winner artifact"
        )
    parameters = locked_plan.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("locked continuation has no full-grow parameters")
    topology_parameters = {}
    for name in ("step_size", "min_area_cm", "use_cuda"):
        if name not in parameters:
            raise ValueError(
                f"locked continuation has no topology parameter {name}"
            )
        topology_parameters[name] = parameters[name]
    return {
        "schema": "campaignx.seed_probe_continuation_contract.v1",
        "continuation_task_id": task_id,
        "source_snapshot_id": source_snapshot_id,
        "probe_run_id": probe_run_id,
        "decision_id": decision_id,
        "decision_sha256": decision_sha256,
        "winner_trial_id": winner_trial_id,
        "winner_probe_artifact_set_id": winner_probe_artifact_set_id,
        "winner_seed": expected_seed,
        "resume_artifact": {
            **stable_resume,
            "reached_generations": int(resume["reached_generations"]),
            "noncanonical": True,
        },
        "full_growth_profile_id": str(locked_plan.get("profile_id") or ""),
        "topology_parameters": topology_parameters,
        "ink_used": False,
    }


def validate_probe_continuation_contract(
    contract: dict[str, Any],
    locked_plan: dict[str, Any],
    *,
    task_id: str,
    attempt_id: str,
) -> None:
    """Reject finalization unless its plan still matches the promoted winner."""

    if (
        not isinstance(contract, dict)
        or contract.get("schema")
        != "campaignx.seed_probe_continuation_contract.v1"
        or contract.get("continuation_task_id") != task_id
        or contract.get("ink_used") is not False
        or locked_plan.get("task_id") != task_id
        or locked_plan.get("attempt_id") != attempt_id
        or locked_plan.get("source_snapshot_id")
        != contract.get("source_snapshot_id")
        or locked_plan.get("ink_used") is not False
    ):
        raise RuntimeError(
            "final grow is not bound to the probe continuation task"
        )
    decision = locked_plan.get("seed_probe_decision")
    if (
        not isinstance(decision, dict)
        or content_sha256(decision) != contract.get("decision_sha256")
        or locked_plan.get("seed_probe_decision_sha256")
        != contract.get("decision_sha256")
        or decision.get("probe_run_id") != contract.get("probe_run_id")
        or decision.get("winner_trial_id")
        != contract.get("winner_trial_id")
        or decision.get("action") != "CONTINUE_WINNER"
    ):
        raise RuntimeError(
            "final grow lost the selected probe decision binding"
        )
    winner_seed = contract.get("winner_seed")
    selected_seed = locked_plan.get("selected_seed")
    if (
        not isinstance(winner_seed, dict)
        or not isinstance(selected_seed, dict)
        or selected_seed.get("candidate_id")
        != winner_seed.get("candidate_id")
        or any(
            float(selected_seed.get(axis)) != float(winner_seed.get(axis))
            for axis in "xyz"
        )
    ):
        raise RuntimeError("final grow did not use the selected probe seed")
    expected_resume = contract.get("resume_artifact")
    actual_resume = locked_plan.get("resume_artifact")
    stable_keys = (
        "probe_run_id",
        "probe_trial_id",
        "probe_artifact_set_id",
        "artifact_uri",
        "manifest_sha256",
        "reached_generations",
        "noncanonical",
    )
    if (
        not isinstance(expected_resume, dict)
        or not isinstance(actual_resume, dict)
        or any(
            actual_resume.get(key) != expected_resume.get(key)
            for key in stable_keys
        )
        or locked_plan.get("resume_from")
        != actual_resume.get("materialized_path")
    ):
        raise RuntimeError(
            "final grow did not resume the selected probe artifact"
        )
    if (
        locked_plan.get("profile_id")
        != contract.get("full_growth_profile_id")
    ):
        raise RuntimeError("final grow changed the continuation profile")
    parameters = locked_plan.get("parameters")
    topology = contract.get("topology_parameters")
    if (
        not isinstance(parameters, dict)
        or not isinstance(topology, dict)
        or any(parameters.get(name) != value for name, value in topology.items())
    ):
        raise RuntimeError(
            "final grow changed probe-compatible topology parameters"
        )


def validate_probe_finalization_authority(
    *,
    task_payload: dict[str, Any],
    seed_probe_required: bool,
    probe_run: dict[str, Any] | None,
    probe_decision: dict[str, Any] | None,
    promotion: dict[str, Any] | None,
    locked_plan: dict[str, Any],
    replay: bool,
) -> None:
    """Make the persisted select decision—not plan omission—the authority.

    A caller-controlled locked plan cannot opt a select-mode task out of its
    probe stop.  Shadow tasks continue through the ordinary lane, while select
    tasks may finalize only after one persisted CONTINUE_WINNER decision has
    been bound to the exact promotion and continuation plan.
    """

    policy_value = task_payload.get("seed_probe")
    policy = (
        normalize_seed_probe_policy(policy_value)
        if policy_value is not None
        else None
    )
    if bool(seed_probe_required) != (policy is not None):
        raise RuntimeError(
            "authoritative task probe requirement and policy disagree"
        )
    mode = policy["mode"] if policy is not None else None
    plan_decision = locked_plan.get("seed_probe_decision")
    has_plan_winner = (
        isinstance(plan_decision, dict)
        and plan_decision.get("action") == "CONTINUE_WINNER"
    )

    if mode != "select":
        if promotion is not None or has_plan_winner:
            raise RuntimeError(
                "only an authoritative select-mode task may finalize a "
                "probe-selected continuation"
            )
        return

    if not isinstance(probe_run, dict) or not isinstance(
        probe_decision, dict
    ):
        raise RuntimeError(
            "select-mode finalization requires a persisted probe decision"
        )
    run_policy = normalize_seed_probe_policy(probe_run.get("policy"))
    expected_run_state = "PROMOTED" if replay else "CONTINUING"
    if (
        run_policy["mode"] != "select"
        or content_sha256(run_policy) != content_sha256(policy)
        or probe_run.get("state") != expected_run_state
    ):
        raise RuntimeError(
            "select-mode probe run is not the authoritative continuation"
        )

    receipt = probe_decision.get("receipt")
    receipt_sha256 = probe_decision.get("receipt_sha256")
    winner_trial_id = probe_decision.get("winner_trial_id")
    if (
        not isinstance(receipt, dict)
        or content_sha256(receipt) != receipt_sha256
        or receipt.get("probe_run_id") != probe_run.get("probe_run_id")
        or receipt.get("action") != "CONTINUE_WINNER"
        or receipt.get("winner_trial_id") != winner_trial_id
        or probe_decision.get("action") != "CONTINUE_WINNER"
        or not isinstance(winner_trial_id, str)
        or not winner_trial_id
    ):
        raise RuntimeError(
            "select-mode finalization is stopped by its persisted probe "
            "decision"
        )
    if (
        promotion is None
        or promotion.get("decision_id") != probe_decision.get("decision_id")
        or promotion.get("winner_trial_id") != winner_trial_id
    ):
        raise RuntimeError(
            "select-mode winner has no exact persisted promotion"
        )
    if (
        not has_plan_winner
        or content_sha256(plan_decision) != receipt_sha256
        or locked_plan.get("seed_probe_decision_sha256") != receipt_sha256
        or plan_decision.get("probe_run_id") != probe_run.get("probe_run_id")
        or plan_decision.get("winner_trial_id") != winner_trial_id
    ):
        raise RuntimeError(
            "final grow omitted or changed the authoritative select decision"
        )


class SeedProbeCoordinator:
    """Run/recover probe trials while the parent worker keeps its task lease."""

    def __init__(
        self,
        store: Any,
        grow_executor: Any,
        artifact_root: Path | str,
        worker_id: str,
        lease_seconds: int,
        worker_capabilities: dict[str, Any],
    ):
        self.store = store
        self.grow_executor = grow_executor
        self.artifact_store = open_artifact_store(artifact_root)
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.worker_capabilities = worker_capabilities

    def run(
        self,
        task: dict[str, Any],
        candidates: list[dict[str, Any]],
        attempt_dir: Path,
    ) -> dict[str, Any]:
        policy = validate_seed_probe_task_contract(task)
        verify_probe_growth_profile()
        verify_probe_evaluation_profile()
        frozen = frozen_probe_candidates(candidates, policy)
        fingerprint = {
            **executor_fingerprint(self.grow_executor),
            "probe_namespace": self.artifact_store.probe_namespace_identity(),
        }
        probe_run = self.store.prepare_probe_run(
            task,
            frozen,
            policy,
            fingerprint,
        )
        probe_root = attempt_dir / "seed-probe"
        probe_root.mkdir(exist_ok=True)
        write_json_atomic(probe_root / "PROBE_POLICY.json", policy)
        write_json_atomic(probe_root / "PROBE_CANDIDATES.json", frozen)

        while probe_run.get("decision") is None:
            trial = self.store.claim_probe_trial(
                task["task_id"],
                task["attempt_id"],
                task["lease_token"],
                probe_run["probe_run_id"],
                self.worker_id,
                self.lease_seconds,
                self.worker_capabilities,
            )
            if trial is None:
                snapshot = self.store.probe_run(probe_run["probe_run_id"])
                if all(
                    row["state"] in PROBE_TERMINAL_TRIAL_STATES
                    for row in snapshot["trials"]
                ):
                    decision = decide_probe_run(snapshot)
                    self.store.record_probe_decision(
                        task["task_id"],
                        task["attempt_id"],
                        task["lease_token"],
                        probe_run["probe_run_id"],
                        decision,
                    )
                    probe_run = self.store.probe_run(probe_run["probe_run_id"])
                    break
                raise RuntimeError(
                    "seed probe run has unfinished trials but none can be claimed"
                )

            trial_dir = (
                probe_root
                / trial["probe_trial_id"]
                / trial["probe_attempt_id"]
            )
            trial_dir.mkdir(parents=True, exist_ok=False)
            plan = build_probe_locked_plan(task, trial, policy)
            write_json_atomic(trial_dir / "PROBE_LOCKED_PLAN.json", plan)
            self.store.transition_probe_trial(
                task["task_id"],
                task["attempt_id"],
                task["lease_token"],
                trial["probe_trial_id"],
                trial["probe_attempt_id"],
                trial["lease_token"],
                "RUNNING",
                locked_plan=plan,
            )
            try:
                grown = self.grow_executor.execute(plan, trial_dir)
                surface_dir = Path(grown["surface_dir"])
                files = artifact_manifest(surface_dir, PROBE_REQUIRED_ARTIFACTS)
                manifest = {
                    "schema": "campaignx.seed_probe_artifact_set.v1",
                    "probe_run_id": probe_run["probe_run_id"],
                    "probe_trial_id": trial["probe_trial_id"],
                    "locked_plan_sha256": content_sha256(plan),
                    "files": files,
                    "artifact_sha256": content_sha256(files),
                    "noncanonical": True,
                    "ink_used": False,
                }
                write_json_atomic(trial_dir / "PROBE_ARTIFACT_SET.json", manifest)
                artifact_set_id = self.store.reserve_probe_artifact(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    manifest,
                )
                published = self.artifact_store.publish_probe(
                    surface_dir,
                    task["sample_id"],
                    probe_run["probe_run_id"],
                    trial["probe_trial_id"],
                    artifact_set_id,
                    manifest,
                )
                evaluation, geometry = evaluate_probe_surface(
                    surface_dir,
                    task=task,
                    probe_run_id=probe_run["probe_run_id"],
                    probe_trial_id=trial["probe_trial_id"],
                    probe_artifact_set_id=artifact_set_id,
                )
                write_json_atomic(
                    trial_dir / "PROBE_GEOMETRY_CERTIFICATION.json", geometry
                )
                write_json_atomic(
                    trial_dir / "PROBE_EVALUATION.json", evaluation
                )
                self.store.complete_probe_trial(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    artifact_set_id,
                    published["artifact_uri"],
                    grown["receipt"],
                    evaluation,
                    geometry,
                )
            except InsufficientGpuMemoryError as error:
                self.store.fail_probe_trial(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    {
                        "status": "RETRY_ON_LARGER_GPU",
                        "growth_receipt": error.receipt,
                        "error_type": type(error).__name__,
                        "ink_used": False,
                    },
                    retryable=True,
                )
                raise
            except Exception as error:
                failure = {
                    "status": "PROBE_TECHNICAL_FAILURE",
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                    "generated_at_utc": utc_now(),
                    "ink_used": False,
                    "non_claim": "an operational failure is not evidence against a seed",
                }
                write_json_atomic(trial_dir / "PROBE_FAILURE.json", failure)
                self.store.fail_probe_trial(
                    task["task_id"],
                    task["attempt_id"],
                    task["lease_token"],
                    trial["probe_trial_id"],
                    trial["probe_attempt_id"],
                    trial["lease_token"],
                    failure,
                    retryable=True,
                )
            probe_run = self.store.probe_run(probe_run["probe_run_id"])

        decision = probe_run.get("decision")
        if not isinstance(decision, dict):
            raise RuntimeError("seed probe run reached no immutable decision")
        write_json_atomic(probe_root / "PROBE_DECISION.json", decision)
        if policy["mode"] == "shadow":
            return {
                "status": "PROBE_SHADOW_COMPLETE",
                "probe_run_id": probe_run["probe_run_id"],
                "decision": decision,
                "planner_candidates": candidates,
                "planner_task": task,
            }
        if decision["action"] != "CONTINUE_WINNER":
            return {
                "status": (
                    "PROBE_REJECTED_ALL"
                    if decision["action"] == "REJECT_ALL"
                    else "PROBE_REVIEW_PENDING"
                ),
                "probe_run_id": probe_run["probe_run_id"],
                "decision": decision,
            }
        winner = next(
            row
            for row in probe_run["trials"]
            if row["probe_trial_id"] == decision["winner_trial_id"]
        )
        materialized = probe_root / "winner-resume"
        try:
            self.artifact_store.materialize_probe(
                winner["artifact_uri"],
                materialized,
                winner["manifest"],
            )
        except Exception as error:
            raise ProbeWinnerMaterializationError(
                probe_run_id=probe_run["probe_run_id"],
                decision=decision,
                winner_trial_id=winner["probe_trial_id"],
                artifact_uri=str(winner["artifact_uri"]),
                error=error,
            ) from error
        planner_candidate = next(
            row
            for row in candidates
            if row["candidate_id"] == winner["candidate"]["candidate_id"]
        )
        planner_task = _resume_compatible_task(
            task,
            decision=decision,
            winner=winner,
            materialized_surface=materialized,
            policy=policy,
        )
        return {
            "status": "PROBE_WINNER",
            "probe_run_id": probe_run["probe_run_id"],
            "decision": decision,
            "planner_candidates": [planner_candidate],
            "planner_task": planner_task,
            "winner": winner,
        }
