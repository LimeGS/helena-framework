#!/usr/bin/env python3
"""Preflight and evaluate the frozen MULTISCROLL_TRANSFER_V1 benchmark.

The benchmark is deliberately fail-closed. Model predictions may locate
examples, but only independently certified labels may enter evaluation.
PHerc0139 and every other threshold-development scroll are excluded by scroll
identity, not merely by an exact component/group identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_ID = "MULTISCROLL_TRANSFER_V1"
ALLOWED_BENCHMARK_IDS = {
    "MULTISCROLL_TRANSFER_V1",
    "MULTISCROLL_TRANSFER_V2",
    "MULTISCROLL_TRANSFER_V3",
}
ALLOWED_LABEL_AUTHORITIES = {
    "PUBLIC_EXPERT_LABEL",
    "PUBLIC_CURATED_SURFACE_LABEL",
    "HUMAN_CT_ADJUDICATION",
    "DIRECTLY_VISIBLE_VOLUMETRIC_INK_ADJUDICATION",
}
PROHIBITED_LABEL_AUTHORITIES = {
    "MODEL_PREDICTION",
    "PSEUDO_LABEL",
    "SELF_CONSISTENCY",
    "MODEL_ASSISTED_UNADJUDICATED",
}
ALLOWED_V4_TIERS = {
    "TIER_A_V3_RETAINED_REVIEW",
    "TIER_B_SHADOW_REVIEW",
    "TIER_C_EXTEND_OR_RESEGMENT",
}


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


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a two-sided Wilson score interval."""

    if total <= 0:
        return None
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def grouped_bootstrap_interval(
    rows: list[dict[str, Any]],
    success_key: str,
    *,
    iterations: int,
    seed: int,
) -> list[float] | None:
    """Bootstrap component recall by resampling complete surface groups."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["surface_group_id"])].append(row)
    group_ids = sorted(grouped)
    if not group_ids:
        return None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        sample_rows = [row for group_id in sampled for row in grouped[group_id]]
        values.append(
            sum(bool(row[success_key]) for row in sample_rows) / len(sample_rows)
        )
    values.sort()
    lower = values[int(math.floor(0.025 * (len(values) - 1)))]
    upper = values[int(math.ceil(0.975 * (len(values) - 1)))]
    return [lower, upper]


def _validate_source(source: dict[str, Any], reasons: list[str], prefix: str) -> None:
    authority = str(source.get("label_authority", ""))
    if authority in PROHIBITED_LABEL_AUTHORITIES:
        reasons.append(f"{prefix}:PROHIBITED_LABEL_AUTHORITY:{authority}")
    elif authority not in ALLOWED_LABEL_AUTHORITIES:
        reasons.append(f"{prefix}:UNSUPPORTED_LABEL_AUTHORITY:{authority or 'MISSING'}")
    assets = source.get("assets", [])
    if not assets:
        reasons.append(f"{prefix}:MISSING_SOURCE_ASSETS")
    for index, asset in enumerate(assets):
        if not str(asset.get("uri", "")):
            reasons.append(f"{prefix}:ASSET_{index}_MISSING_URI")
        digest = str(asset.get("sha256", ""))
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            reasons.append(f"{prefix}:ASSET_{index}_INVALID_SHA256")
    if authority == "PUBLIC_CURATED_SURFACE_LABEL":
        required_roles = {"INK_LABEL", "SUPERVISION_MASK", "SURFACE_CT"}
        roles = {str(asset.get("role", "")) for asset in assets}
        missing_roles = sorted(required_roles - roles)
        for role in missing_roles:
            reasons.append(f"{prefix}:MISSING_CURATED_LABEL_ASSET:{role}")
        if source.get("prediction_used_as_ground_truth") is not False:
            reasons.append(f"{prefix}:CURATED_LABEL_PREDICTION_POLICY_MISSING")
        if not str(source.get("coordinate_frame_id", "")):
            reasons.append(f"{prefix}:MISSING_CURATED_LABEL_COORDINATE_FRAME")


def preflight_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check whether a manifest is eligible to be frozen and executed."""

    reasons: list[str] = []
    warnings: list[str] = []
    if manifest.get("benchmark_id") not in ALLOWED_BENCHMARK_IDS:
        reasons.append("WRONG_BENCHMARK_ID")
    if manifest.get("status") not in {"DRAFT_SOURCE_AUDIT", "FROZEN_BEFORE_RESULTS"}:
        reasons.append("INVALID_MANIFEST_STATUS")

    policy = manifest.get("policy", {})
    development_scrolls = set(map(str, manifest.get("development_scrolls", [])))
    sources = list(manifest.get("scroll_sources", []))
    source_by_scroll: dict[str, dict[str, Any]] = {}
    eligible_positive_scrolls: list[str] = []
    eligible_confound_scrolls: list[str] = []
    for source in sources:
        scroll_id = str(source.get("scroll_id", ""))
        if not scroll_id:
            reasons.append("SOURCE_WITHOUT_SCROLL_ID")
            continue
        if scroll_id in source_by_scroll:
            reasons.append(f"DUPLICATE_SCROLL_SOURCE:{scroll_id}")
            continue
        source_by_scroll[scroll_id] = source
        role = str(source.get("benchmark_role", ""))
        label_authority = str(source.get("label_authority", ""))
        certified_positive = int(source.get("certified_positive_components", 0))
        certified_confound = int(source.get("certified_confound_components", 0))
        surface_groups = int(source.get("independent_surface_groups", 0))

        if scroll_id in development_scrolls and role != "DEVELOPMENT_ONLY":
            reasons.append(f"DEVELOPMENT_SCROLL_NOT_EXCLUDED:{scroll_id}")
        if role == "EVALUATION":
            _validate_source(source, reasons, scroll_id)
            if label_authority in ALLOWED_LABEL_AUTHORITIES:
                if (
                    certified_positive
                    >= int(policy.get("minimum_positive_components_per_scroll", 50))
                    and surface_groups
                    >= int(policy.get("minimum_surface_groups_per_scroll", 5))
                ):
                    eligible_positive_scrolls.append(scroll_id)
                if (
                    certified_confound
                    >= int(policy.get("minimum_confound_components_per_scroll", 50))
                    and surface_groups
                    >= int(policy.get("minimum_surface_groups_per_scroll", 5))
                ):
                    eligible_confound_scrolls.append(scroll_id)
        elif role == "CANDIDATE_SOURCE_ONLY":
            warnings.append(f"NOT_EVALUATION_READY:{scroll_id}")
        elif role != "DEVELOPMENT_ONLY":
            reasons.append(f"INVALID_BENCHMARK_ROLE:{scroll_id}:{role or 'MISSING'}")

    minimum_positive_scrolls = int(
        policy.get("minimum_independent_positive_scrolls", 3)
    )
    minimum_confound_scrolls = int(
        policy.get("minimum_independent_confound_scrolls", 2)
    )
    if len(eligible_positive_scrolls) < minimum_positive_scrolls:
        reasons.append("INSUFFICIENT_INDEPENDENT_POSITIVE_SCROLLS")
    if len(eligible_confound_scrolls) < minimum_confound_scrolls:
        reasons.append("INSUFFICIENT_INDEPENDENT_CONFOUND_SCROLLS")

    controls = list(manifest.get("controls", []))
    if (
        manifest.get("status") == "FROZEN_BEFORE_RESULTS"
        and not controls
        and not manifest.get("controls_file")
    ):
        reasons.append("FROZEN_MANIFEST_WITHOUT_CONTROLS")
    if manifest.get("status") == "DRAFT_SOURCE_AUDIT" and controls:
        reasons.append("DRAFT_SOURCE_AUDIT_MUST_NOT_CONTAIN_RESULTS")

    return {
        "status": (
            "READY_TO_FREEZE"
            if not reasons
            else "BLOCKED_SOURCE_LABELS_OR_CONTRACT_INCOMPLETE"
        ),
        "blocking_reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
        "eligible_positive_scrolls": sorted(eligible_positive_scrolls),
        "eligible_confound_scrolls": sorted(eligible_confound_scrolls),
        "source_scroll_count": len(source_by_scroll),
        "development_scrolls": sorted(development_scrolls),
    }


def _control_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["surface_group_id"]), str(row["component_id"])


def _validate_controls(
    manifest: dict[str, Any], controls: list[dict[str, Any]]
) -> list[str]:
    reasons: list[str] = []
    development_scrolls = set(map(str, manifest.get("development_scrolls", [])))
    evaluation_scrolls = {
        str(source["scroll_id"])
        for source in manifest.get("scroll_sources", [])
        if source.get("benchmark_role") == "EVALUATION"
    }
    seen: set[tuple[str, str]] = set()
    for row in controls:
        prefix = f"{row.get('scroll_id', 'MISSING')}:{row.get('component_id', 'MISSING')}"
        key = _control_key(row)
        if key in seen:
            reasons.append(f"DUPLICATE_COMPONENT:{key[0]}:{key[1]}")
        seen.add(key)
        scroll_id = str(row.get("scroll_id", ""))
        if scroll_id in development_scrolls:
            reasons.append(f"DEVELOPMENT_SCROLL_IN_EVALUATION:{scroll_id}")
        if scroll_id not in evaluation_scrolls:
            reasons.append(f"{prefix}:SCROLL_NOT_DECLARED_FOR_EVALUATION")
        expected = str(row.get("expected_class", ""))
        if expected not in {"POSITIVE", "CONFOUND"}:
            reasons.append(f"{prefix}:INVALID_EXPECTED_CLASS")
        if expected == "CONFOUND" and str(row.get("confound_subtype", "")) not in {
            "FIBER",
            "EDGE",
            "CRACK",
            "DIFFUSE_SIGNAL",
            "OTHER_ADJUDICATED_NON_INK",
        }:
            reasons.append(f"{prefix}:MISSING_OR_INVALID_CONFOUND_SUBTYPE")
        _validate_source(row["label_source"], reasons, prefix)
        coordinate = row.get("ct_coordinate_xyz")
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 3
            or any(not math.isfinite(float(value)) for value in coordinate)
        ):
            reasons.append(f"{prefix}:INVALID_CT_COORDINATE")
        if not str(row.get("slice_order", "")):
            reasons.append(f"{prefix}:MISSING_SLICE_ORDER")
        decision_sources = row.get("decision_sources", {})
        for decision_name in ("v3", "v4"):
            decision_source = decision_sources.get(decision_name)
            if not isinstance(decision_source, dict):
                reasons.append(f"{prefix}:MISSING_{decision_name.upper()}_SOURCE")
                continue
            _validate_source(
                {
                    "label_authority": "PUBLIC_EXPERT_LABEL",
                    "assets": [decision_source],
                },
                reasons,
                f"{prefix}:{decision_name.upper()}_DECISION",
            )
        if not isinstance(row.get("v3_retained"), bool):
            reasons.append(f"{prefix}:MISSING_V3_DECISION")
        tier = str(row.get("v4_tier", ""))
        if tier not in ALLOWED_V4_TIERS:
            reasons.append(f"{prefix}:INVALID_V4_TIER:{tier or 'MISSING'}")
        if row.get("v4_not_discarded") is not True:
            reasons.append(f"{prefix}:V4_PRESERVATION_VIOLATION")
        voxel = row.get("voxel_size_um")
        if (
            not isinstance(voxel, list)
            or len(voxel) != 3
            or any(float(value) <= 0 for value in voxel)
        ):
            reasons.append(f"{prefix}:INVALID_VOXEL_SIZE")
        if not str(row.get("scanner_domain", "")):
            reasons.append(f"{prefix}:MISSING_SCANNER_DOMAIN")
    return reasons


def _metric_payload(
    rows: list[dict[str, Any]],
    success_key: str,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    successes = sum(bool(row[success_key]) for row in rows)
    groups = len({str(row["surface_group_id"]) for row in rows})
    return {
        "components": len(rows),
        "surface_groups": groups,
        "successes": successes,
        "rate": _ratio(successes, len(rows)),
        "component_wilson_95": wilson_interval(successes, len(rows)),
        "surface_group_bootstrap_95": grouped_bootstrap_interval(
            rows,
            success_key,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
    }


def evaluate_benchmark(
    manifest: dict[str, Any], controls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate frozen v3 and v4 outputs on the same certified controls."""

    preflight = preflight_manifest(manifest)
    reasons = list(preflight["blocking_reasons"])
    if manifest.get("status") != "FROZEN_BEFORE_RESULTS":
        reasons.append("BENCHMARK_NOT_FROZEN_BEFORE_RESULTS")
    reasons.extend(_validate_controls(manifest, controls))
    policy = manifest["policy"]
    bootstrap_iterations = int(policy.get("bootstrap_iterations", 2000))
    bootstrap_seed = int(policy.get("bootstrap_seed", 20260723))

    by_scroll: dict[str, dict[str, Any]] = {}
    for scroll_id in sorted({str(row["scroll_id"]) for row in controls}):
        scroll_rows = [row for row in controls if str(row["scroll_id"]) == scroll_id]
        positives = [row for row in scroll_rows if row["expected_class"] == "POSITIVE"]
        confounds = [row for row in scroll_rows if row["expected_class"] == "CONFOUND"]
        for row in scroll_rows:
            row["_v3_positive_success"] = bool(row["v3_retained"])
            row["_v4_review_success"] = row["v4_tier"] in {
                "TIER_A_V3_RETAINED_REVIEW",
                "TIER_B_SHADOW_REVIEW",
            }
            row["_v4_preserved_success"] = bool(row["v4_not_discarded"])
            row["_v3_confound_downrank"] = not bool(row["v3_retained"])
            row["_v4_confound_downrank"] = row["v4_tier"] != "TIER_A_V3_RETAINED_REVIEW"

        tiers = Counter(str(row["v4_tier"]) for row in scroll_rows)
        by_scroll[scroll_id] = {
            "positive_v3_tier_a_recall": _metric_payload(
                positives,
                "_v3_positive_success",
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            ),
            "positive_v4_direct_review_recall": _metric_payload(
                positives,
                "_v4_review_success",
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed + 1,
            ),
            "positive_v4_preservation_recall": _metric_payload(
                positives,
                "_v4_preserved_success",
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed + 2,
            ),
            "confound_v3_downrank_rate": _metric_payload(
                confounds,
                "_v3_confound_downrank",
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed + 3,
            ),
            "confound_v4_downrank_or_extension_rate": _metric_payload(
                confounds,
                "_v4_confound_downrank",
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed + 4,
            ),
            "v4_tier_counts": dict(sorted(tiers.items())),
            "scanner_domains": sorted(
                {str(row["scanner_domain"]) for row in scroll_rows}
            ),
            "voxel_sizes_um": sorted(
                {tuple(map(float, row["voxel_size_um"])) for row in scroll_rows}
            ),
        }

    minimum_components = int(policy["minimum_positive_components_per_scroll"])
    minimum_confounds = int(policy["minimum_confound_components_per_scroll"])
    minimum_groups = int(policy["minimum_surface_groups_per_scroll"])
    minimum_recall = float(policy["minimum_positive_recall_per_scroll"])
    evaluated_positive_scrolls = 0
    evaluated_confound_scrolls = 0
    for scroll_id, metrics in by_scroll.items():
        positive = metrics["positive_v4_direct_review_recall"]
        preserved = metrics["positive_v4_preservation_recall"]
        confound = metrics["confound_v3_downrank_rate"]
        if positive["components"]:
            evaluated_positive_scrolls += 1
            if positive["components"] < minimum_components:
                reasons.append(f"{scroll_id}:INSUFFICIENT_POSITIVE_COMPONENTS")
            if positive["surface_groups"] < minimum_groups:
                reasons.append(f"{scroll_id}:INSUFFICIENT_POSITIVE_SURFACE_GROUPS")
            if positive["rate"] < minimum_recall:
                reasons.append(f"{scroll_id}:V4_DIRECT_REVIEW_RECALL_BELOW_GATE")
            if preserved["rate"] != 1.0:
                reasons.append(f"{scroll_id}:V4_PRESERVATION_BELOW_100_PERCENT")
        if confound["components"]:
            evaluated_confound_scrolls += 1
            if confound["components"] < minimum_confounds:
                reasons.append(f"{scroll_id}:INSUFFICIENT_CONFOUND_COMPONENTS")
            if confound["surface_groups"] < minimum_groups:
                reasons.append(f"{scroll_id}:INSUFFICIENT_CONFOUND_SURFACE_GROUPS")

    if evaluated_positive_scrolls < int(policy["minimum_independent_positive_scrolls"]):
        reasons.append("INSUFFICIENT_EVALUATED_POSITIVE_SCROLLS")
    if evaluated_confound_scrolls < int(policy["minimum_independent_confound_scrolls"]):
        reasons.append("INSUFFICIENT_EVALUATED_CONFOUND_SCROLLS")
    benchmark_id = str(manifest.get("benchmark_id", BENCHMARK_ID))
    return {
        "status": (
            f"{benchmark_id}_PASSED"
            if not reasons
            else f"{benchmark_id}_BLOCKED_OR_FAILED"
        ),
        "blocking_or_failure_reasons": sorted(set(reasons)),
        "metrics_by_scroll": by_scroll,
        "evaluated_component_count": len(controls),
        "evaluated_positive_scroll_count": evaluated_positive_scrolls,
        "evaluated_confound_scroll_count": evaluated_confound_scrolls,
        "interpretation": {
            "v3_tier_a": "strict historical priority gate",
            "v4_direct_review": "Tier A plus Tier B; immediately reviewable",
            "v4_preservation": "Tier A, B, or C; Tier C requests more support",
            "confound_metrics": "routing efficiency, never proof of absence",
        },
    }


def _load_controls(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    reference = manifest.get("controls_file")
    if not reference:
        return list(manifest.get("controls", []))
    controls_path = (manifest_path.parent / str(reference)).resolve()
    expected = str(manifest.get("controls_file_sha256", ""))
    if sha256(controls_path) != expected:
        raise RuntimeError("controls file SHA-256 does not match frozen manifest")
    return json.loads(controls_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["preflight", "evaluate"], default="preflight"
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite benchmark receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.mode == "preflight":
        result = preflight_manifest(manifest)
    else:
        result = evaluate_benchmark(
            manifest, _load_controls(manifest, manifest_path)
        )
    receipt = {
        "schema": "campaignx.multiscroll_transfer_benchmark.v1",
        "benchmark_id": BENCHMARK_ID,
        "mode": args.mode,
        "generated_at_utc": utc_now(),
        "manifest": {
            "path": str(args.manifest),
            "sha256": sha256(manifest_path),
        },
        **result,
        "non_claims": [
            "model predictions are never ground truth",
            "a blocked source audit does not estimate recall",
            "Tier C requests more support and is not a negative label",
            "no ink, text, letters, or First Letters are accepted automatically",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
