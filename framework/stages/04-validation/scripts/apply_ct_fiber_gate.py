#!/usr/bin/env python3
"""Apply a frozen transparent CT depth-localization gate to feature rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == ">":
        return value > threshold
    raise ValueError(f"unsupported operator: {operator}")


def apply_rule(
    row: dict[str, str],
    rule: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for requirement in rule["requirements"]:
        feature = str(requirement["feature"])
        value = float(row[feature])
        threshold = float(requirement["threshold"])
        passed = compare(value, str(requirement["operator"]), threshold)
        checks.append(
            {
                "feature": feature,
                "value": value,
                "operator": requirement["operator"],
                "threshold": threshold,
                "passed": passed,
            }
        )
    retained = all(check["passed"] for check in checks)
    return {
        "retained": retained,
        "decision": (
            rule["decision"]["all_requirements_pass"]
            if retained
            else rule["decision"]["any_requirement_fails"]
        ),
        "failed_features": [
            str(check["feature"]) for check in checks if not check["passed"]
        ],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    feature_path = args.features.resolve()
    rule_path = args.rule.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rule = json.loads(rule_path.read_text())
    with feature_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("feature table is empty")

    decisions: list[dict[str, Any]] = []
    for row in rows:
        result = apply_rule(row, rule)
        decisions.append(
            {
                "group_id": row["group_id"],
                "class": row["class"],
                "candidate_id": row["candidate_id"],
                **result,
            }
        )

    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for item in decisions:
        key = "retained" if item["retained"] else "downranked"
        by_group[str(item["group_id"])][key] += 1
        by_class[str(item["class"])][key] += 1

    decision_path = output / "CT_FIBER_GATE_DECISIONS.json"
    decision_path.write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "kind": "campaign_x_phase4_ct_fiber_gate_evaluation_v1",
        "status": "COMPLETED",
        "generated_at_utc": utc_now(),
        "features": str(feature_path),
        "features_sha256": sha256_file(feature_path),
        "rule": str(rule_path),
        "rule_sha256": sha256_file(rule_path),
        "row_count": len(rows),
        "retained_count": sum(item["retained"] for item in decisions),
        "downranked_count": sum(not item["retained"] for item in decisions),
        "by_group": {
            key: dict(value) for key, value in sorted(by_group.items())
        },
        "by_class": {
            key: dict(value) for key, value in sorted(by_class.items())
        },
        "artifacts": {
            "decisions": decision_path.name,
            "decisions_sha256": sha256_file(decision_path),
        },
        "interpretation": (
            "The frozen rule was applied without threshold changes. A retained "
            "component is only queued for orthogonal CT review; a downranked "
            "component is only deprioritized as depth-diffuse. Neither decision "
            "accepts ink, rejects a scroll, or establishes letter identity."
        ),
    }
    receipt_path = output / "CT_FIBER_GATE_EVALUATION.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
