#!/usr/bin/env python3
"""Validate a frozen CT gate on independent controls, failing closed by scroll."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    policy = manifest["policy"]
    development_groups = set(
        map(str, manifest.get("threshold_development_groups", []))
    )
    development_scrolls = set(
        map(str, manifest.get("threshold_development_scrolls", []))
    )
    controls = list(manifest["controls"])
    independent = [
        row
        for row in controls
        if str(row["group_id"]) not in development_groups
        and str(row["scroll_id"]) not in development_scrolls
    ]
    leaked_groups = [
        str(row["group_id"])
        for row in controls
        if str(row["group_id"]) in development_groups
    ]
    excluded_development_scrolls = sorted(
        {
            str(row["scroll_id"])
            for row in controls
            if str(row["scroll_id"]) in development_scrolls
        }
    )
    positives_by_scroll: dict[str, list[bool]] = defaultdict(list)
    confounds_by_scroll: dict[str, list[bool]] = defaultdict(list)
    for row in independent:
        expected = str(row["expected_class"])
        retained = bool(row["retained"])
        if expected == "POSITIVE":
            positives_by_scroll[str(row["scroll_id"])].append(retained)
        elif expected == "CONFOUND":
            confounds_by_scroll[str(row["scroll_id"])].append(not retained)
        else:
            raise ValueError(f"unsupported expected_class: {expected}")

    recalls = {
        scroll: sum(values) / len(values)
        for scroll, values in sorted(positives_by_scroll.items())
    }
    confound_rejections = {
        scroll: sum(values) / len(values)
        for scroll, values in sorted(confounds_by_scroll.items())
    }
    reasons: list[str] = []
    if leaked_groups:
        reasons.append("THRESHOLD_DEVELOPMENT_GROUP_LEAKAGE")
    if len(positives_by_scroll) < int(policy["minimum_independent_positive_scrolls"]):
        reasons.append("INSUFFICIENT_INDEPENDENT_POSITIVE_SCROLLS")
    if len(confounds_by_scroll) < int(policy["minimum_independent_confound_scrolls"]):
        reasons.append("INSUFFICIENT_INDEPENDENT_CONFOUND_SCROLLS")
    minimum_recall = float(policy["minimum_positive_recall_per_scroll"])
    if any(value < minimum_recall for value in recalls.values()):
        reasons.append("POSITIVE_RECALL_BELOW_GATE")
    status = (
        "INDEPENDENT_MULTISCROLL_BENCHMARK_PASSED"
        if not reasons
        else "BLOCKED_INSUFFICIENT_OR_FAILED_INDEPENDENT_CONTROLS"
    )
    return {
        "status": status,
        "blocking_reasons": reasons,
        "independent_positive_scroll_count": len(positives_by_scroll),
        "independent_confound_scroll_count": len(confounds_by_scroll),
        "positive_recall_by_scroll": recalls,
        "confound_rejection_by_scroll": confound_rejections,
        "leaked_group_ids": sorted(leaked_groups),
        "excluded_threshold_development_scrolls": excluded_development_scrolls,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_reference = str(args.manifest)
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite multiscroll benchmark receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {
        "schema": "campaignx.ct_gate_multiscroll_validation.v1",
        "generated_at_utc": utc_now(),
        "manifest": {"path": manifest_reference, "sha256": sha256(manifest_path)},
        **evaluate_manifest(manifest),
        "non_claims": [
            "a blocked benchmark does not estimate recall",
            "no threshold tuning or reroll is performed",
            "not accepted ink, text, letters, or First Letters",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
