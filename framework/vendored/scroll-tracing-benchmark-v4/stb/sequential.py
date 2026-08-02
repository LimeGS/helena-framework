"""V4 autoregressive same-surface tracing contract and scorer.

Labels are held in a separate reference object so inference code never receives
them. A valid trace must be both relation-correct and state-continuous: the ID
selected at step t must be the anchor ID at step t+1.
"""
from dataclasses import dataclass

import numpy as np

SAME = 0
ADJACENT_WRAP = 1
UNRELATED = 2


@dataclass(frozen=True)
class SequentialTraceTask:
    trace_id: str
    anchors: np.ndarray
    candidates: np.ndarray
    candidate_valid: np.ndarray
    anchor_ids: np.ndarray
    candidate_ids: np.ndarray
    continuity_tolerance: float = 1e-6

    def __post_init__(self):
        anchors = np.asarray(self.anchors)
        candidates = np.asarray(self.candidates)
        valid = np.asarray(self.candidate_valid)
        if anchors.ndim != 2 or anchors.shape[1] != 3:
            raise ValueError("anchors must be (steps, 3)")
        if not len(anchors):
            raise ValueError("a trace must contain at least one step")
        if candidates.ndim != 3 or candidates.shape[0] != len(anchors) or candidates.shape[2] != 3:
            raise ValueError("candidates must be (steps, candidates, 3)")
        if valid.shape != candidates.shape[:2]:
            raise ValueError("candidate_valid must match candidates[:2]")
        if np.asarray(self.anchor_ids).shape != (len(anchors),):
            raise ValueError("anchor_ids must contain one ID per step")
        if np.asarray(self.candidate_ids).shape != candidates.shape[:2]:
            raise ValueError("candidate_ids must align with candidates")
        if not np.isfinite(anchors).all():
            raise ValueError("anchors must be finite")
        if not np.isfinite(candidates[valid]).all():
            raise ValueError("valid candidates must be finite")
        if not np.isfinite(self.continuity_tolerance) or self.continuity_tolerance < 0:
            raise ValueError("continuity_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class SequentialTraceReference:
    relation_labels: np.ndarray

    def validate_for(self, task):
        labels = np.asarray(self.relation_labels)
        if labels.shape != task.candidates.shape[:2]:
            raise ValueError("relation_labels must align with task candidates")
        if not np.isin(labels[task.candidate_valid], (SAME, ADJACENT_WRAP, UNRELATED)).all():
            raise ValueError("valid relation labels must be 0=same, 1=adjacent, 2=unrelated")


def score_trace(task, reference, selected_indices):
    """Score one genuinely chained trace.

    Any non-SAME selection, invalid index, abstention, or state-continuity
    violation ends survival. Later rows are not counted as valid decisions,
    preventing independent sampled decisions from masquerading as a trace.
    """
    reference.validate_for(task)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.shape != (len(task.anchors),):
        raise ValueError("selected_indices must contain one entry per step; -1 means abstain")

    counts = {"same": 0, "adjacent_wrap": 0, "unrelated": 0}
    active = True
    first_failure = None
    failure_kind = None
    decisions = 0
    for step, index in enumerate(selected):
        if not active:
            break
        if index < 0:
            first_failure, failure_kind, active = step, "abstain", False
            break
        if index >= task.candidates.shape[1] or not task.candidate_valid[step, index]:
            first_failure, failure_kind, active = step, "invalid_candidate", False
            break
        relation = int(reference.relation_labels[step, index])
        decisions += 1
        key = ("same", "adjacent_wrap", "unrelated")[relation]
        counts[key] += 1
        if relation != SAME:
            first_failure, failure_kind, active = step, key, False
            break
        if step + 1 < len(selected):
            selected_id = task.candidate_ids[step, index]
            coordinate_error = np.linalg.norm(
                task.candidates[step, index] - task.anchors[step + 1]
            )
            if (selected_id != task.anchor_ids[step + 1]
                    or coordinate_error > task.continuity_tolerance):
                first_failure, failure_kind, active = step, "continuity_violation", False
                break

    steps = len(selected)
    survived = steps if first_failure is None else first_failure
    return {
        "trace_id": task.trace_id,
        "steps": steps,
        "decisions": decisions,
        "first_failure_step": first_failure,
        "failure_kind": failure_kind,
        "survived_steps": survived,
        "survival_fraction": survived / steps if steps else float("nan"),
        "completed": bool(first_failure is None and steps > 0),
        "valid_completed": bool(first_failure is None and counts["same"] == steps),
        **counts,
        "total_relation_errors": counts["adjacent_wrap"] + counts["unrelated"],
    }


def summarize_traces(rows):
    if not rows:
        raise ValueError("at least one trace score is required")
    n = len(rows)
    steps = sum(r["steps"] for r in rows)
    return {
        "traces": n,
        "valid_completion_pct": 100.0 * sum(r["valid_completed"] for r in rows) / n,
        "mean_survival_fraction": float(np.mean([r["survival_fraction"] for r in rows])),
        "adjacent_wrap_errors_per_1000_total_steps": 1000.0 * sum(r["adjacent_wrap"] for r in rows) / steps,
        "unrelated_errors_per_1000_total_steps": 1000.0 * sum(r["unrelated"] for r in rows) / steps,
        "total_relation_errors_per_1000_total_steps": 1000.0 * sum(r["total_relation_errors"] for r in rows) / steps,
    }
