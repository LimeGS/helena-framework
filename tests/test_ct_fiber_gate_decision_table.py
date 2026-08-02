"""FIX-10.2 — decision table for the frozen CT/fibre localization gate v3.

WHAT THIS COVERS
----------------
The two functions that decide whether a candidate is queued for orthogonal CT
review, and which the audit found had **no test at all** (audit 3.5, T-2):

* ``apply_ct_fiber_gate.compare``   — the four comparison operators
* ``apply_ct_fiber_gate.apply_rule`` — the 8-term conjunction, the
  emitted ``decision`` label and the ordered ``failed_features`` list

executed against the real frozen profile
``framework/profiles/validation/ct-fiber-localization-gate-v3-candidate-coverage.json``
and the real calibration feature vectors recovered from
``workspace/campaigns/campaign-x-2026/findings/ct-gate-v3-validation/runtime/controls/gate-v3/CT_FIBER_GATE_DECISIONS.json``
(33 candidates, each carrying its 8 measured feature values).

It replays all 33 frozen decisions, measures the **joint recall of the 8-term
conjunction** (previously unmeasured — only the per-term pass counts were ever
recorded), pins the fibrous-confound and PHerc268 boundary-artifact rejections,
and proves every one of the 8 thresholds is load bearing.

WHAT THIS DOES *NOT* COVER
--------------------------
Feature *extraction* (``extract_ct_fiber_features.py``) is covered by
``tests/test_ct_coverage_features.py``; this file starts from feature vectors.
The gate's CLI (``main``) is not exercised.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evidence import needs_campaign_evidence


ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = (
    ROOT / "framework/stages/04-validation/scripts/apply_ct_fiber_gate.py"
)
V3_PROFILE = (
    ROOT
    / "framework/profiles/validation/ct-fiber-localization-gate-v3-candidate-coverage.json"
)
CONTROL_DECISIONS = (
    ROOT
    / "workspace/campaigns/campaign-x-2026/findings/ct-gate-v3-validation"
    / "runtime/controls/gate-v3/CT_FIBER_GATE_DECISIONS.json"
)

RETAIN = "CT_CANDIDATE_COVERAGE_SAFE_SURFACE_LOCALIZED_RETAIN_FOR_INK_REVIEW"
DOWNRANK = "CT_INVALID_OR_DIFFUSE_DOWNRANK_REVIEWABLE_NON_NEGATIVE"

# Requirement order in the frozen profile; ``failed_features`` follows it.
REQUIREMENT_ORDER = (
    "candidate_bbox_nonzero_fraction",
    "central_slice_center_nonzero",
    "central_slice_nonzero_fraction",
    "central_slice_zero_distance_ratio",
    "depth_profile_peak_count",
    "depth_profile_top3_fraction",
    "depth_profile_entropy",
    "argmax_depth_p90_p10_span",
)


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "helena_ct_fiber_gate_decision_table", GATE_SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_gate()


def rule() -> dict:
    return json.loads(V3_PROFILE.read_text(encoding="utf-8"))


def control_decisions() -> list[dict]:
    return json.loads(CONTROL_DECISIONS.read_text(encoding="utf-8"))


def feature_row(decision: dict) -> dict[str, str]:
    """Recover the CSV-shaped feature row the gate consumed."""

    return {check["feature"]: str(check["value"]) for check in decision["checks"]}


def control_rows() -> list[tuple[dict, dict[str, str]]]:
    return [(item, feature_row(item)) for item in control_decisions()]


def positive_control_row() -> dict[str, str]:
    """The first PHerc0139 source-locked positive control (candidate G01)."""

    for decision, row in control_rows():
        if decision["class"] == "PUBLIC_POSITIVE_CONTROL" and decision["retained"]:
            return row
    raise AssertionError("no retained public positive control in the frozen set")


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "operator", "threshold", "expected"),
    [
        (0.95, ">=", 0.95, True),
        (0.9499, ">=", 0.95, False),
        (0.95, ">", 0.95, False),
        (8.0, "<=", 8.0, True),
        (8.0001, "<=", 8.0, False),
        (8.0, "<", 8.0, False),
        (7.9, "<", 8.0, True),
    ],
)
def test_compare_is_inclusive_exactly_where_the_operator_says(
    value, operator, threshold, expected
):
    assert GATE.compare(value, operator, threshold) is expected


def test_compare_rejects_an_unsupported_operator():
    with pytest.raises(ValueError, match="unsupported operator"):
        GATE.compare(1.0, "==", 1.0)


def test_every_operator_used_by_the_frozen_profile_is_supported():
    for requirement in rule()["requirements"]:
        GATE.compare(0.0, str(requirement["operator"]), 0.0)


# ---------------------------------------------------------------------------
# apply_rule() — the three specified decision-table vectors
# ---------------------------------------------------------------------------


@needs_campaign_evidence
def test_positive_control_vector_is_retained():
    result = GATE.apply_rule(positive_control_row(), rule())

    assert result["retained"] is True
    assert result["failed_features"] == []
    assert result["decision"] == RETAIN
    assert len(result["checks"]) == 8
    assert all(check["passed"] for check in result["checks"])


@needs_campaign_evidence
def test_fibrous_confound_vector_is_downranked():
    row = positive_control_row()
    row["depth_profile_peak_count"] = "12"
    row["depth_profile_entropy"] = "0.99"

    result = GATE.apply_rule(row, rule())

    assert result["retained"] is False
    assert result["decision"] == DOWNRANK
    assert result["failed_features"] == [
        "depth_profile_peak_count",
        "depth_profile_entropy",
    ]


@needs_campaign_evidence
def test_pherc268_boundary_artifact_fails_exactly_the_coverage_requirement():
    """The documented post-hoc artifact must fail on coverage and nothing else."""

    evidence = rule()["calibration_evidence"]["pherc268_posthoc_boundary_candidate"]
    assert evidence["candidate_bbox_nonzero_fraction"] == 0.938602

    row = positive_control_row()
    row["candidate_bbox_nonzero_fraction"] = str(
        evidence["candidate_bbox_nonzero_fraction"]
    )

    result = GATE.apply_rule(row, rule())

    assert result["retained"] is False
    assert result["decision"] == DOWNRANK
    assert result["failed_features"] == ["candidate_bbox_nonzero_fraction"]


@needs_campaign_evidence
def test_apply_rule_raises_when_a_required_feature_is_missing():
    row = positive_control_row()
    del row["depth_profile_entropy"]

    with pytest.raises(KeyError):
        GATE.apply_rule(row, rule())


# ---------------------------------------------------------------------------
# Replay of the 33 frozen calibration decisions
# ---------------------------------------------------------------------------


@needs_campaign_evidence
def test_frozen_profile_reproduces_every_recorded_control_decision():
    current = rule()
    mismatches = []
    for decision, row in control_rows():
        replayed = GATE.apply_rule(row, current)
        if (
            replayed["retained"] != decision["retained"]
            or replayed["failed_features"] != decision["failed_features"]
            or replayed["decision"] != decision["decision"]
        ):
            mismatches.append(
                (decision["group_id"], decision["candidate_id"], replayed, decision)
            )

    assert mismatches == [], f"{len(mismatches)} frozen decisions no longer replay"


@needs_campaign_evidence
def test_failed_features_preserve_the_profile_requirement_order():
    order = [str(item["feature"]) for item in rule()["requirements"]]
    assert order == list(REQUIREMENT_ORDER)
    for decision, row in control_rows():
        failed = GATE.apply_rule(row, rule())["failed_features"]
        assert failed == [name for name in order if name in set(failed)]


# ---------------------------------------------------------------------------
# The number nobody had measured: joint recall of the 8-term conjunction
# ---------------------------------------------------------------------------


def joint_recall_report() -> dict:
    current = rule()
    per_term_pass: dict[str, int] = {name: 0 for name in REQUIREMENT_ORDER}
    by_group: dict[str, list[bool]] = {}
    by_class: dict[str, list[bool]] = {}
    for decision, row in control_rows():
        result = GATE.apply_rule(row, current)
        for check in result["checks"]:
            if check["passed"]:
                per_term_pass[str(check["feature"])] += 1
        by_group.setdefault(str(decision["group_id"]), []).append(result["retained"])
        by_class.setdefault(str(decision["class"]), []).append(result["retained"])
    total = sum(len(values) for values in by_group.values())
    return {
        "total_calibration_candidates": total,
        "joint_recall_all_calibration_controls": sum(
            sum(values) for values in by_group.values()
        )
        / total,
        "joint_recall_by_group": {
            key: sum(values) / len(values) for key, values in sorted(by_group.items())
        },
        "joint_recall_by_class": {
            key: sum(values) / len(values) for key, values in sorted(by_class.items())
        },
        "per_term_recall": {
            name: count / total for name, count in per_term_pass.items()
        },
    }


@needs_campaign_evidence
def test_joint_recall_of_the_eight_term_conjunction_is_measured_and_pinned(capsys):
    """The gate is a conjunction of 8 terms; only per-term counts were recorded.

    FINDING (audit criterion (c) — declared, but the operational consequence
    was never extracted).  Over the 33 candidates the profile itself calls its
    calibration controls, the joint recall of the conjunction is **0.4545**,
    not 1.0:

    * ``PHerc0139-public-positive`` (source-locked full positive): 15/15 = 1.000
    * ``PHerc172-public-positive``  (coverage diagnostic only):     0/18 = 0.000

    Every one of the 18 PHerc172 rejections is on the depth terms
    (``depth_profile_entropy`` alone, or together with
    ``depth_profile_top3_fraction``) — the terms inherited unchanged from v1.
    The conjunction's recall is therefore measured on exactly one scroll.  The
    profile pre-declares PHerc172 as ``COVERAGE_DIAGNOSTIC_ONLY``, so this is
    not a contradiction, but the gate has **zero measured sensitivity on the
    second published positive scroll** and that has never been stated as a
    number.
    """

    report = joint_recall_report()
    print(json.dumps(report, indent=2, sort_keys=True))

    assert report["total_calibration_candidates"] == 33
    assert report["joint_recall_all_calibration_controls"] == pytest.approx(
        15 / 33, abs=1e-12
    )
    assert report["joint_recall_by_group"] == {
        "PHerc0139-public-positive": pytest.approx(1.0),
        "PHerc172-public-positive": pytest.approx(0.0),
    }
    assert report["joint_recall_by_class"] == {
        "PUBLIC_POSITIVE_CONTROL": pytest.approx(1.0),
        "COVERAGE_DIAGNOSTIC_CONTROL": pytest.approx(0.0),
    }
    # The conjunction is strictly worse than its weakest term: no single term
    # rejects more than 18/33, yet the product rejects 18/33 on the depth pair.
    assert report["per_term_recall"]["depth_profile_entropy"] == pytest.approx(
        15 / 33, abs=1e-12
    )
    assert report["per_term_recall"]["depth_profile_top3_fraction"] == pytest.approx(
        25 / 33, abs=1e-12
    )
    for name in (
        "candidate_bbox_nonzero_fraction",
        "central_slice_center_nonzero",
        "central_slice_nonzero_fraction",
        "central_slice_zero_distance_ratio",
        "depth_profile_peak_count",
        "argmax_depth_p90_p10_span",
    ):
        assert report["per_term_recall"][name] == pytest.approx(1.0), (
            f"{name} rejects nothing in the calibration set"
        )

    captured = capsys.readouterr()
    assert "joint_recall_all_calibration_controls" in captured.out


@needs_campaign_evidence
def test_source_locked_positive_control_recall_is_complete():
    """The one group the profile calls full-positive evidence must be 15/15."""

    report = joint_recall_report()
    declared = rule()["calibration_evidence"]["source_locked_full_positive_controls"]
    assert declared == ["PHerc0139-public-positive"]
    assert report["joint_recall_by_group"]["PHerc0139-public-positive"] == 1.0
    assert (
        rule()["calibration_evidence"][
            "source_locked_full_positive_control_candidate_count"
        ]
        == 15
    )


@needs_campaign_evidence
def test_six_of_the_eight_terms_never_reject_a_calibration_control():
    """Six of the eight terms have never rejected a single calibration control.

    Their thresholds are therefore unvalidated against any measured negative in
    this set; only the two depth terms discriminate.  Pinned so that a future
    calibration extension has to update this number deliberately.
    """

    report = joint_recall_report()
    inert = sorted(
        name for name, value in report["per_term_recall"].items() if value == 1.0
    )
    assert len(inert) == 6
    assert "depth_profile_entropy" not in inert
    assert "depth_profile_top3_fraction" not in inert


# ---------------------------------------------------------------------------
# Mutation sensitivity — every threshold must be load bearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(8))
@needs_campaign_evidence
def test_tightening_any_single_threshold_breaks_the_positive_controls(index: int):
    current = rule()
    requirement = current["requirements"][index]
    feature = str(requirement["feature"])
    positives = [
        row
        for decision, row in control_rows()
        if decision["class"] == "PUBLIC_POSITIVE_CONTROL"
    ]
    assert positives, "no positive controls to mutate against"

    values = [float(row[feature]) for row in positives]
    mutated = copy.deepcopy(current)
    if str(requirement["operator"]) in (">=", ">"):
        mutated["requirements"][index]["threshold"] = max(values) + 1e-6
    else:
        mutated["requirements"][index]["threshold"] = min(values) - 1e-6

    retained_before = sum(
        GATE.apply_rule(row, current)["retained"] for row in positives
    )
    retained_after = sum(
        GATE.apply_rule(row, mutated)["retained"] for row in positives
    )

    assert retained_before == len(positives)
    assert retained_after < retained_before, (
        f"threshold for {feature} is inert: mutating it changed no decision"
    )


@needs_campaign_evidence
def test_relaxing_the_coverage_threshold_would_readmit_the_pherc268_artifact():
    """0.95 is exactly what excludes the audited boundary artifact at 0.938602."""

    evidence = rule()["calibration_evidence"]["pherc268_posthoc_boundary_candidate"]
    row = positive_control_row()
    row["candidate_bbox_nonzero_fraction"] = str(
        evidence["candidate_bbox_nonzero_fraction"]
    )

    assert GATE.apply_rule(row, rule())["retained"] is False

    relaxed = rule()
    relaxed["requirements"][0]["threshold"] = 0.93
    assert str(relaxed["requirements"][0]["feature"]) == (
        "candidate_bbox_nonzero_fraction"
    )
    assert GATE.apply_rule(row, relaxed)["retained"] is True


def test_the_frozen_thresholds_are_exactly_the_audited_values():
    thresholds = {
        str(item["feature"]): float(item["threshold"])
        for item in rule()["requirements"]
    }
    assert thresholds == {
        "candidate_bbox_nonzero_fraction": 0.95,
        "central_slice_center_nonzero": 1.0,
        "central_slice_nonzero_fraction": 0.95,
        "central_slice_zero_distance_ratio": 1.0,
        "depth_profile_peak_count": 8.0,
        "depth_profile_top3_fraction": 0.12,
        "depth_profile_entropy": 0.982,
        "argmax_depth_p90_p10_span": 30.0,
    }
    assert rule()["policy"]["changing_any_threshold_requires_a_new_version"] is True
