#!/usr/bin/env python3
"""Evaluate the deterministic R6 geometry policy on frozen local V2 bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
sys.path.insert(0, str(ROOT / "phase2" / "src"))

from campaign_x_phase2.constraint_graph import ConstraintGraph, Edge  # noqa: E402
from campaign_x_phase2.contracts import (  # noqa: E402
    constraints_to_point_collection,
    point_collection_to_constraints,
)
from campaign_x_phase2.direct_geometry_policy import (  # noqa: E402
    agreement_decision,
    swap_geometry_row,
)


CLASS_COUNTS = {"SAME": 600, "ADJACENT": 100, "UNRELATED": 900}

# `ece_max: 0.10` used to live here. It was removed because it was not an
# expected-calibration-error gate at all.
#
# The metric was computed as `_ece(np.ones(len(pair_ids)), correct)`: every
# confidence was the literal 1.0, so every sample landed in the last bin of a
# 15-bin reliability diagram and the whole sum collapsed to the single term
# `1.0 * |mean(correct) - 1.0|`, i.e. exactly `|accuracy - 1.0|`. A gate of
# "ECE <= 0.10" was therefore a gate of "accuracy >= 90%" wearing a calibration
# name, and the reported R6 value of 0.0 was only a restatement of
# `relation_accuracy == 1.0`.
#
# This is not a cosmetic rename, because the same metric NAME was load-bearing
# earlier in the same experiment series. R0 (0.058966), R1 (0.063847) and R2
# (0.062115) were each terminated by an "ECE" gate of <= 0.05 -- but there the
# quantity was a genuine bootstrap-UCB calibration error over a probabilistic
# classifier's real predicted probabilities. R6 keeps the name and reports
# 0.0 against a loosened <= 0.10, while feeding the estimator a constant. The
# estimand changed silently between rounds of one series; the two numbers are
# not comparable and must never be tabulated in the same column.
# See docs/SERIES_MULTIPLICITY_STAGE05_R0_R6.json.
#
# The accuracy is still reported, under a name that says what it is
# (`accuracy_on_all_pairs`), and it is still gated -- by the pre-existing
# `same_precision`, `same_recall` and `adjacent_as_same` gates, which
# constrain the same predictions per class instead of in one pooled number.
GATES = {
    "same_precision_min": 0.95,
    "same_recall_min": 0.50,
    "adjacent_as_same_max": 0.05,
    "relative_sign_accuracy_min": 0.90,
    "correct_accepted_sign_recall_min": 0.30,
    "accepted_sign_rows_min": 1,
    "accepted_graph_edges_min": 1,
    "pointcollections_roundtrip_min": 1.0,
    "candidate_recall_k12_min": 0.95,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _by_pair(rows: list[dict[str, Any]], *, role: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in result:
            raise ValueError(f"{role} pair identity differs")
        result[pair_id] = row
    return result


# `_ece` was deleted together with the `ece` gate. Its only caller passed a
# constant confidence of 1.0, which made it an alias for `|accuracy - 1.0|`
# rather than a calibration statistic. Reinstating a real calibration gate
# requires a real confidence -- see `_decision_confidence` below -- plus a
# reliability analysis that this evaluator does not perform.


def _decision_confidence(decision: Any) -> float:
    """Return the conservative geometric band margin behind one decision.

    Both geometry streams already compute a margin (the signed distance from
    the measurement to the nearest frozen band boundary,
    `direct_geometry_policy.py:91,107,119`); the evaluator previously discarded
    both and wrote the literal `1.0` into every emitted constraint. The
    conservative choice is the smaller of the two independent margins.

    This is NOT a probability and NOT a calibrated confidence. It is a raw
    band margin whose units differ per relation: for SAME it is
    `1.35 - path_ratio` and spans (0, 1.35]; for ADJACENT it is the minimum of
    four normal/separation/tangential slacks and is bounded above by 0.15.
    Ordering it across relations is meaningless. It is propagated because a
    real per-decision quantity is strictly more informative than a constant,
    and because a constant is what disguised the accuracy gate as calibration
    in the first place.

    Only agreed SAME/ADJACENT decisions reach this function, so both margins
    are finite: the infinite margins belong to UNRELATED-by-distinct-regions
    (+inf) and AMBIGUOUS (-inf), which are never emitted as constraints.
    """

    margins = (float(decision.topology.margin), float(decision.ray.margin))
    if not all(math.isfinite(value) for value in margins):
        raise ValueError("accepted decision has a non-finite band margin")
    return min(margins)


def _node_id(xyz: Any) -> str:
    vector = np.asarray(xyz, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("graph endpoint must be finite XYZ")
    payload = json.dumps(
        [float(value) for value in vector],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "r6-xyz:" + hashlib.sha256(payload).hexdigest()


def evaluate(
    *,
    topology_path: Path,
    ray_path: Path,
    truth_path: Path,
    sample_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_rows = read_jsonl(truth_path)
    sample_rows = read_jsonl(sample_path)
    truth = _by_pair(truth_rows, role="truth")
    sample = _by_pair(sample_rows, role="sample")
    pair_ids = [row["pair_id"] for row in truth_rows]
    selected_ids = set(pair_ids)
    topology_all = _by_pair(read_jsonl(topology_path), role="topology")
    rays_all = _by_pair(read_jsonl(ray_path), role="ray")
    topology = {
        pair_id: topology_all[pair_id]
        for pair_id in pair_ids
        if pair_id in topology_all
    }
    rays = {
        pair_id: rays_all[pair_id]
        for pair_id in pair_ids
        if pair_id in rays_all
    }
    if (
        len(pair_ids) != 1600
        or selected_ids != set(topology)
        or selected_ids != set(rays)
        or selected_ids != set(sample)
    ):
        raise ValueError("R6 input pair inventories differ")
    observed_counts = Counter(str(row.get("relation")) for row in truth_rows)
    if dict(observed_counts) != CLASS_COUNTS:
        raise ValueError("R6 truth class counts differ")

    predictions: list[str] = []
    signs: list[int] = []
    agreements: list[bool] = []
    swap_checks: list[bool] = []
    confidences: list[float] = []
    graph = ConstraintGraph()
    constraints: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        top = topology[pair_id]
        ray = rays[pair_id]
        decision = agreement_decision(top, ray)
        predictions.append(decision.relation)
        signs.append(decision.relative_sign_ab)
        agreements.append(decision.agreed)
        swapped = agreement_decision(
            swap_geometry_row(top, topology=True),
            swap_geometry_row(ray, topology=False),
        )
        swap_checks.append(
            swapped.relation == decision.relation
            and (
                decision.relation != "ADJACENT"
                or swapped.relative_sign_ab == -decision.relative_sign_ab
            )
        )
        if not decision.agreed or decision.relation == "UNRELATED":
            continue
        delta = 0 if decision.relation == "SAME" else decision.relative_sign_ab
        endpoint_a = _node_id(top["endpoint_a_xyz_l0"])
        endpoint_b = _node_id(top["endpoint_b_xyz_l0"])
        confidence = _decision_confidence(decision)
        confidences.append(confidence)
        constraint = {
            "constraint_id": f"r6:{pair_id}",
            "type": (
                "SAME_WINDING"
                if decision.relation == "SAME"
                else "RELATIVE_WINDING"
            ),
            "endpoint_a_xyz_l0": list(top["endpoint_a_xyz_l0"]),
            "endpoint_b_xyz_l0": list(top["endpoint_b_xyz_l0"]),
            "winding_delta": delta,
            # Uncalibrated geometric band margin, not a probability. See
            # `_decision_confidence`. Consumers must not compare it across
            # relation types nor read it as P(correct).
            "confidence": confidence,
            "confidence_semantics": "UNCALIBRATED_GEOMETRIC_BAND_MARGIN",
        }
        if graph.add(
            Edge(
                constraint["constraint_id"],
                endpoint_a,
                endpoint_b,
                delta,
                confidence,
            )
        ):
            constraints.append(constraint)

    expected = np.asarray([row["relation"] for row in truth_rows], dtype=str)
    predicted = np.asarray(predictions, dtype=str)
    expected_sign = np.asarray(
        [int(row["relative_sign_ab"]) for row in truth_rows],
        dtype=np.int64,
    )
    predicted_sign = np.asarray(signs, dtype=np.int64)
    agreed_array = np.asarray(agreements, dtype=bool)
    same_truth = expected == "SAME"
    same_pred = predicted == "SAME"
    adjacent_truth = expected == "ADJACENT"
    correct = predicted == expected
    same_true_positive = int(np.sum(same_truth & same_pred))
    same_precision = same_true_positive / max(1, int(same_pred.sum()))
    same_recall = same_true_positive / int(same_truth.sum())
    adjacent_as_same = float(np.mean(same_pred[adjacent_truth]))
    accepted_sign = adjacent_truth & agreed_array & (predicted == "ADJACENT")
    sign_correct = (
        predicted_sign[accepted_sign] == expected_sign[accepted_sign]
    )
    candidate_retrieved = np.asarray(
        [bool(sample[pair_id]["candidate_k12_retrieved"]) for pair_id in pair_ids],
        dtype=bool,
    )
    candidate_recall = float(candidate_retrieved[adjacent_truth].mean())
    point_collection = constraints_to_point_collection(
        constraints,
        collection_name="campaign-x:r6-direct-geometry-local-functional",
    )
    decoded = point_collection_to_constraints(point_collection)
    redecoded = point_collection_to_constraints(
        constraints_to_point_collection(decoded, collection_name="roundtrip")
    )
    roundtrip = 1.0 if decoded == redecoded else 0.0
    metrics = {
        "pair_count": len(pair_ids),
        "class_counts": CLASS_COUNTS,
        "topology_ray_agreement_rate": float(np.mean(agreements)),
        "relation_accuracy": float(correct.mean()),
        "same_precision": same_precision,
        "same_recall": same_recall,
        "adjacent_as_same": adjacent_as_same,
        # Formerly reported as "ece". It was never a calibration error: the
        # confidence handed to the estimator was the constant 1.0, so the value
        # equalled |accuracy - 1.0| exactly. Reported here under its true name,
        # and no longer gated. See the GATES comment above.
        "accuracy_on_all_pairs": float(correct.mean()),
        "confidence_semantics": "UNCALIBRATED_GEOMETRIC_BAND_MARGIN",
        "accepted_confidence_min": (
            float(min(confidences)) if confidences else 0.0
        ),
        "accepted_confidence_median": (
            float(np.median(np.asarray(confidences, dtype=np.float64)))
            if confidences
            else 0.0
        ),
        "accepted_relative_sign_accuracy": (
            float(sign_correct.mean()) if sign_correct.size else 0.0
        ),
        "correct_accepted_sign_recall": float(
            sign_correct.sum() / int(adjacent_truth.sum())
        ),
        "accepted_sign_rows": int(accepted_sign.sum()),
        "accepted_graph_edges": int(len(constraints)),
        "same_graph_edges": int(np.sum(same_pred)),
        "relative_graph_edges": int(accepted_sign.sum()),
        "graph": graph.summary(),
        "pointcollections_roundtrip": roundtrip,
        "candidate_recall_at_12": candidate_recall,
        "physical_swap_consistency": float(np.mean(swap_checks)),
    }
    gate_results = {
        "same_precision": same_precision >= GATES["same_precision_min"],
        "same_recall": same_recall >= GATES["same_recall_min"],
        "adjacent_as_same": adjacent_as_same <= GATES["adjacent_as_same_max"],
        "relative_sign_accuracy": (
            metrics["accepted_relative_sign_accuracy"]
            >= GATES["relative_sign_accuracy_min"]
        ),
        "correct_accepted_sign_recall": (
            metrics["correct_accepted_sign_recall"]
            >= GATES["correct_accepted_sign_recall_min"]
        ),
        "accepted_sign_rows": (
            metrics["accepted_sign_rows"] >= GATES["accepted_sign_rows_min"]
        ),
        "accepted_graph_edges": (
            metrics["accepted_graph_edges"] >= GATES["accepted_graph_edges_min"]
        ),
        "pointcollections_roundtrip": (
            roundtrip >= GATES["pointcollections_roundtrip_min"]
        ),
        "candidate_recall_at_12": (
            candidate_recall >= GATES["candidate_recall_k12_min"]
        ),
        "topology_ray_exact_agreement": math.isclose(
            metrics["topology_ray_agreement_rate"], 1.0
        ),
        "physical_swap_consistency": math.isclose(
            metrics["physical_swap_consistency"], 1.0
        ),
    }
    passed = all(gate_results.values())
    policy = {
        "kind": "campaign_x_phase2_r6_direct_geometry_policy_v1",
        "status": "FROZEN_R6_DIRECT_GEOMETRY_POLICY",
        "scope": "LOCAL_FUNCTIONAL_ONLY",
        "learned_model": False,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
        "decision": (
            "emit only exact topology/ray baseline agreement after the frozen "
            "65-variant robust eligibility filter"
        ),
        "bands": {
            "same_path_ratio_max": 1.35,
            "normal_alignment_abs_min": 0.85,
            "adjacent_signed_separation_spacing": [0.55, 1.45],
            "adjacent_tangential_spacing_max": 1.50,
            "unrelated_signed_separation_spacing_min": 2.50,
            "unrelated_tangential_spacing_min": 3.00,
        },
        "sign_rule": "sign(dot(endpoint_b-endpoint_a, oriented_normal_a))",
        "abstain_on_disagreement_or_ambiguity": True,
        "h0_input_count": 0,
        "h1_input_count": 0,
        "r5_predictions_used_for_decision": False,
        "post_hoc_to_local_holdout_v2_labels": True,
        "required_amendment": "phase2/PHASE2_CONTRACT_AMENDMENT_020.md",
        "implementation": {
            "path": "phase2/src/campaign_x_phase2/direct_geometry_policy.py",
            "sha256": sha256_file(
                ROOT / "phase2" / "src" / "campaign_x_phase2"
                / "direct_geometry_policy.py"
            ),
        },
    }
    result = {
        "kind": "campaign_x_phase2_r6_local_functional_result_v1",
        "status": (
            "PASSED_R6_LOCAL_FUNCTIONAL"
            if passed
            else "FAILED_R6_LOCAL_FUNCTIONAL"
        ),
        "scope": "LOCAL_FUNCTIONAL_ONLY",
        "metrics": metrics,
        "gates": GATES,
        "gate_results": gate_results,
        "passed": passed,
        "input_sha256": {
            topology_path.name: sha256_file(topology_path),
            ray_path.name: sha256_file(ray_path),
            truth_path.name: sha256_file(truth_path),
            sample_path.name: sha256_file(sample_path),
        },
        "h0_input_count": 0,
        "h1_input_count": 0,
        "independent_h1_validated": False,
        "external_generalization_claim": False,
        "first_letters_eligible": passed,
        "validation_scope": (
            "LOCAL_PIPELINE_CONTINUATION_ONLY" if passed else "BLOCKED"
        ),
        "historical_r5_terminal_preserved": True,
        "series_multiplicity": {
            "declaration": "docs/SERIES_MULTIPLICITY_STAGE05_R0_R6.json",
            "series_id": "stage05-relation-v2-R0-R6",
            "round_id": "R6",
            "unbiased_estimator": False,
            "note": (
                "R6 is the terminal round of an eight-round adaptive series in "
                "which each round was designed from the previous round's "
                "failure mode. No metric in this receipt is an unbiased "
                "estimate of out-of-sample performance, and no multiplicity "
                "correction (family-wise, Bonferroni or otherwise) has been "
                "applied to any of them."
            ),
        },
        "limitations": [
            "The policy is evaluated against deterministic geometry-reference labels.",
            "This is not an independent H1 result.",
            "The policy requires both topology and ray geometry plus the frozen robust filter.",
            (
                "The functional metrics are CIRCULAR, not measurements. The "
                "policy's decision bands (2.50 * spacing, 3.00 * spacing, "
                "mutual_nearest_non_incident, distinct_regions) are the same "
                "literals used by the assigner that generated the labels "
                "(local_holdout_v1_observation_universe.py:220-230 vs "
                "direct_geometry_policy.py:114-129). A value of 1.0 here shows "
                "that a reimplementation reproduces the label generator; it "
                "does not show that the relation is recovered correctly."
            ),
            (
                "The former 'ece' metric was not a calibration error: it was "
                "computed with a constant confidence of 1.0 and therefore "
                "equalled |accuracy - 1.0|. It has been renamed to "
                "accuracy_on_all_pairs and its gate has been removed."
            ),
            (
                "'confidence' on emitted constraints is an uncalibrated "
                "geometric band margin, not a probability, and its scale "
                "differs between SAME and ADJACENT."
            ),
        ],
    }
    return policy, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--ray", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args()
    policy, result = evaluate(
        topology_path=args.topology,
        ray_path=args.ray,
        truth_path=args.truth,
        sample_path=args.sample,
    )
    atomic_json(args.policy_output, policy)
    atomic_json(args.result_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
