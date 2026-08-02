from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .common import content_sha256


# Only outcomes that say something about seed availability, the attempted
# geometry recipe, or spatial redundancy may steer a later geometry attempt.
# Provider outages, OOMs, finalizer/runtime failures, and downstream ink/QC
# outcomes are deliberately excluded.
ACTIONABLE_SEGMENTATION_HISTORY_STATES = frozenset(
    {
        "NO_SEED",
        "GROW_FAILED",
        "DUPLICATE_SURFACE",
        "GEOMETRY_REJECTED_BRIDGE",
        "GEOMETRY_REJECTED_LAMINA_SWITCH",
        "GEOMETRY_REJECTED_DISTORTION",
        "GEOMETRY_REJECTED_COVERAGE",
    }
)


def bounds_overlap(left: list[list[float]], right: list[list[float]]) -> bool:
    """Return whether two inclusive XYZ AABBs overlap."""

    if len(left) != 2 or len(right) != 2:
        raise ValueError("regional history bounds must contain low/high XYZ rows")
    if any(len(row) != 3 for row in (*left, *right)):
        raise ValueError("regional history bounds must be three-dimensional")
    return all(
        float(left[0][axis]) <= float(right[1][axis])
        and float(right[0][axis]) <= float(left[1][axis])
        for axis in range(3)
    )


def _plan_value(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def build_regional_attempt_history(
    task: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    *,
    limit: int = 12,
) -> dict[str, Any]:
    """Build the small, ink-blind failure packet exposed to planner v2.

    Store adapters supply immutable attempt rows. This function performs the
    same filtering and normalization for SQLite and PostgreSQL, intentionally
    omitting raw error strings, logs, downstream QC, and any ink-derived data.
    """

    if limit < 1 or limit > 32:
        raise ValueError("regional attempt history limit must be between 1 and 32")
    task_bounds = task["bounds_xyz"]
    attempts: list[dict[str, Any]] = []
    for raw in rows:
        state = str(raw.get("state", ""))
        # Finalization deliberately keeps the attempt/task state QC_PENDING so
        # old API/status contracts remain stable; the orthogonal geometry axis
        # lives in the attempt result.  Promote only hard geometry rejections
        # into this geometry-only history packet. Physical/ink QC is still
        # excluded.
        result = raw.get("result")
        geometry_state = (
            str(result.get("geometry_qc_state"))
            if isinstance(result, dict)
            and result.get("geometry_qc_state") is not None
            else ""
        )
        if (
            state == "QC_PENDING"
            and geometry_state in ACTIONABLE_SEGMENTATION_HISTORY_STATES
        ):
            state = geometry_state
        if state not in ACTIONABLE_SEGMENTATION_HISTORY_STATES:
            continue
        bounds = raw.get("bounds_xyz")
        if not isinstance(bounds, list) or not bounds_overlap(task_bounds, bounds):
            continue
        plan = _plan_value(raw.get("locked_plan"))
        selected_seed = None
        profile_id = None
        parameters = None
        recipe_sha256 = None
        if plan is not None:
            selected = plan.get("selected_seed")
            if isinstance(selected, dict) and all(axis in selected for axis in "xyz"):
                selected_seed = {
                    key: selected[key]
                    for key in ("candidate_id", "x", "y", "z")
                    if key in selected
                }
            profile_id = plan.get("profile_id")
            parameters = plan.get("parameters") if isinstance(plan.get("parameters"), dict) else None
            if selected_seed is not None and profile_id is not None and parameters is not None:
                recipe_sha256 = content_sha256(
                    {
                        "seed_xyz": [selected_seed[axis] for axis in "xyz"],
                        "profile_id": profile_id,
                        "parameters": parameters,
                    }
                )
        attempts.append(
            {
                "history_id": str(raw["attempt_id"]),
                "task_id": str(raw["task_id"]),
                "cell_id": str(raw["cell_id"]),
                "policy_version": str(raw["policy_version"]),
                "outcome": state,
                "same_cell": str(raw["cell_id"]) == str(task["cell_id"]),
                "bounds_xyz": bounds,
                "selected_seed": selected_seed,
                "profile_id": profile_id,
                "parameters": parameters,
                "recipe_sha256": recipe_sha256,
                "updated_at_utc": str(raw.get("updated_at") or ""),
            }
        )
        if len(attempts) >= limit:
            break
    state_counts = Counter(row["outcome"] for row in attempts)
    body = {
        "schema": "campaignx.segmentation_regional_attempt_history.v1",
        "task_id": task["task_id"],
        "source_snapshot_id": task["source"]["source_snapshot_id"],
        "region": {
            "cell_id": task["cell_id"],
            "bounds_xyz": task_bounds,
            "matching_rule": "same-source-inclusive-aabb-overlap-v1",
        },
        "selection": {
            "included_outcomes": sorted(ACTIONABLE_SEGMENTATION_HISTORY_STATES),
            "excluded_classes": [
                "provider_or_source_outage",
                "gpu_capacity_failure",
                "planner_contract_failure",
                "finalizer_or_storage_failure",
                "downstream_qc_or_ink_outcome",
            ],
            "maximum_attempts": limit,
        },
        "attempt_count": len(attempts),
        "outcome_counts": dict(sorted(state_counts.items())),
        "attempts": attempts,
        "ink_used": False,
    }
    return {**body, "history_sha256": content_sha256(body)}
