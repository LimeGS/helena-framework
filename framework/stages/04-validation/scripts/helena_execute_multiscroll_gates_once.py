#!/usr/bin/env python3
"""Execute frozen CT gate v3 and shadow router v4 once on frozen controls.

This executor is deliberately fail-closed:

* labels and CT patches must already be frozen by the independent
  materialization stage;
* profile and input hashes are verified before an execution claim is written;
* an existing claim, result, or non-empty output directory prevents reruns;
* v3 and v4 see exactly the same controls and CT tensor;
* every v4 route preserves the candidate (Tier A, B, or C).

The script evaluates transfer.  It never accepts ink, text, letters, or First
Letters and never changes either profile after observing results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from helena_multiscroll_transfer_v1 import (
    evaluate_benchmark,
    preflight_manifest,
)
from helena_route_ct_gate_shadow_v4 import route_decision
from apply_ct_fiber_gate import apply_rule
from extract_ct_fiber_features import (
    candidate_bbox_nonzero_fraction,
    extract_features,
)
from extract_ct_fiber_features_physical import (
    extract_physical_depth_features,
    resolve_half_window_um,
)


CENTRAL_SLICE = 32


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


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_assets_for_scroll(
    controls: list[dict[str, Any]], scroll_id: str
) -> list[dict[str, str]]:
    assets: dict[tuple[str, str], dict[str, str]] = {}
    for row in controls:
        if str(row["scroll_id"]) != scroll_id:
            continue
        for asset in row["label_source"]["assets"]:
            key = (str(asset["role"]), str(asset["sha256"]))
            assets[key] = {
                "role": str(asset["role"]),
                "uri": str(asset["uri"]),
                "sha256": str(asset["sha256"]),
            }
    return [assets[key] for key in sorted(assets)]


def build_manifest(
    controls: list[dict[str, Any]],
    *,
    benchmark_id: str,
    controls_file: str,
    controls_file_sha256: str,
    v3_profile: Path,
    v4_profile: Path,
    materialization_receipt: Path,
    manifest_role: str,
) -> dict[str, Any]:
    scroll_sources: list[dict[str, Any]] = []
    for scroll_id in sorted({str(row["scroll_id"]) for row in controls}):
        rows = [row for row in controls if str(row["scroll_id"]) == scroll_id]
        positives = sum(row["expected_class"] == "POSITIVE" for row in rows)
        confounds = sum(row["expected_class"] == "CONFOUND" for row in rows)
        groups = len({str(row["surface_group_id"]) for row in rows})
        coordinate_frames = sorted(
            {str(row["label_source"]["coordinate_frame_id"]) for row in rows}
        )
        scroll_sources.append(
            {
                "scroll_id": scroll_id,
                "benchmark_role": "EVALUATION",
                "label_authority": "PUBLIC_CURATED_SURFACE_LABEL",
                "certified_positive_components": positives,
                "certified_confound_components": confounds,
                "independent_surface_groups": groups,
                "assets": _source_assets_for_scroll(controls, scroll_id),
                "prediction_used_as_ground_truth": False,
                "coordinate_frame_id": "|".join(coordinate_frames),
            }
        )
    return {
        "schema": "campaignx.multiscroll_transfer_manifest.v1",
        "benchmark_id": benchmark_id,
        "status": "FROZEN_BEFORE_RESULTS",
        "frozen_at_utc": utc_now(),
        "manifest_role": manifest_role,
        "development_scrolls": ["PHerc0139", "PHerc0172"],
        "policy": {
            "minimum_independent_positive_scrolls": 3,
            "minimum_independent_confound_scrolls": 2,
            "minimum_positive_components_per_scroll": 50,
            "minimum_confound_components_per_scroll": 50,
            "minimum_surface_groups_per_scroll": 5,
            "minimum_positive_recall_per_scroll": 0.95,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260723,
            "no_silent_discard": True,
            "threshold_changes_after_freeze_prohibited": True,
        },
        "scroll_sources": scroll_sources,
        "controls_file": controls_file,
        "controls_file_sha256": controls_file_sha256,
        "frozen_profiles": {
            "v3": {"path": str(v3_profile), "sha256": sha256(v3_profile)},
            "v4": {"path": str(v4_profile), "sha256": sha256(v4_profile)},
        },
        "materialization": {
            "path": str(materialization_receipt),
            "sha256": sha256(materialization_receipt),
        },
        "prediction_used_as_ground_truth": False,
    }


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": str(row["surface_group_id"]),
        "class": str(row["expected_class"]),
        "candidate_id": str(row["component_id"]),
    }


def execute(
    *,
    materialized_root: Path,
    v3_profile_path: Path,
    v4_profile_path: Path,
    output_root: Path,
    benchmark_id: str = "MULTISCROLL_TRANSFER_V1",
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("refusing to rerun or overwrite one-shot evaluation")
    output_root.mkdir(parents=True, exist_ok=True)

    materialization_path = materialized_root / "MATERIALIZATION_RECEIPT.json"
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if (
        materialization.get("status")
        != "OFFICIAL_CONTROLS_FROZEN_BEFORE_V3_V4"
    ):
        raise RuntimeError("materialization receipt is not frozen before v3/v4")
    if materialization.get("benchmark_id") != benchmark_id:
        raise RuntimeError("materialization benchmark id does not match execution")
    if materialization["gate_visibility"] != {
        "v3_executed": False,
        "v4_executed": False,
        "v3_v4_outputs_read_during_label_creation": False,
    }:
        raise RuntimeError("gate visibility contract was violated")

    tensor_path = materialized_root / materialization["artifacts"]["patch_tensor"]
    controls_path = (
        materialized_root / materialization["artifacts"]["frozen_controls"]
    )
    if sha256(tensor_path) != materialization["artifacts"]["patch_tensor_sha256"]:
        raise RuntimeError("frozen CT tensor hash mismatch")
    if (
        sha256(controls_path)
        != materialization["artifacts"]["frozen_controls_sha256"]
    ):
        raise RuntimeError("frozen control hash mismatch")
    frozen_controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if len(frozen_controls) != 300:
        raise RuntimeError("expected exactly 300 frozen controls")

    v3_profile = json.loads(v3_profile_path.read_text(encoding="utf-8"))
    v4_profile = json.loads(v4_profile_path.read_text(encoding="utf-8"))
    if v3_profile.get("kind") != "campaignx.ct_surface_localization_gate.v3":
        raise RuntimeError("wrong v3 profile")
    if v4_profile.get("kind") not in {
        "campaignx.ct_fiber_shadow_router_profile.v4",
        "campaignx.ct_fiber_supported_window_router_profile.v4_1",
    }:
        raise RuntimeError("wrong v4 profile")

    preexecution_manifest = build_manifest(
        frozen_controls,
        benchmark_id=benchmark_id,
        controls_file=str(controls_path),
        controls_file_sha256=sha256(controls_path),
        v3_profile=v3_profile_path,
        v4_profile=v4_profile_path,
        materialization_receipt=materialization_path,
        manifest_role="PREEXECUTION_LABEL_AND_POLICY_FREEZE",
    )
    preexecution_manifest_path = (
        output_root / "FROZEN_PREEXECUTION_MANIFEST.json"
    )
    write_json(preexecution_manifest_path, preexecution_manifest)
    preexecution_preflight = preflight_manifest(preexecution_manifest)
    write_json(
        output_root / "PREEXECUTION_PREFLIGHT.json",
        preexecution_preflight,
    )
    if preexecution_preflight["status"] != "READY_TO_FREEZE":
        raise RuntimeError(
            "preexecution manifest failed preflight: "
            + ",".join(preexecution_preflight["blocking_reasons"])
        )

    preclaim = {
        "schema": "campaignx.multiscroll_transfer_execution_preclaim.v1",
        "benchmark_id": benchmark_id,
        "status": "READY_FOR_SINGLE_EXECUTION",
        "created_at_utc": utc_now(),
        "inputs": {
            "materialization_receipt_sha256": sha256(materialization_path),
            "patch_tensor_sha256": sha256(tensor_path),
            "frozen_controls_sha256": sha256(controls_path),
            "v3_profile_sha256": sha256(v3_profile_path),
            "v4_profile_sha256": sha256(v4_profile_path),
            "preexecution_manifest_sha256": sha256(
                preexecution_manifest_path
            ),
        },
        "counts": {
            "controls": len(frozen_controls),
            "scrolls": len({row["scroll_id"] for row in frozen_controls}),
            "positives": sum(
                row["expected_class"] == "POSITIVE" for row in frozen_controls
            ),
            "confounds": sum(
                row["expected_class"] == "CONFOUND" for row in frozen_controls
            ),
        },
        "rerun_allowed": False,
    }
    preclaim["content_sha256"] = canonical_sha256(
        {key: value for key, value in preclaim.items() if key != "content_sha256"}
    )
    preclaim_path = output_root / "PRECLAIM.json"
    write_json(preclaim_path, preclaim)

    claim = {
        "schema": "campaignx.multiscroll_transfer_execution_claim.v1",
        "benchmark_id": benchmark_id,
        "status": "CLAIMED_SINGLE_EXECUTION",
        "claimed_at_utc": utc_now(),
        "preclaim_sha256": sha256(preclaim_path),
        "attempt": 1,
        "maximum_attempts": 1,
    }
    claim["content_sha256"] = canonical_sha256(
        {key: value for key, value in claim.items() if key != "content_sha256"}
    )
    write_json(output_root / "EXECUTION_CLAIM.json", claim)

    tensor = np.load(tensor_path, mmap_mode="r")
    if list(tensor.shape) != materialization["patch_shape_n_z_y_x"]:
        raise RuntimeError("frozen CT tensor shape mismatch")

    v3_feature_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    v3_decisions: list[dict[str, Any]] = []
    v4_routes: list[dict[str, Any]] = []
    physical_config = dict(v4_profile["physical_depth_sampling"])
    coverage_features = set(v4_profile["routing"]["coverage_features"])
    minimum_coverage = float(
        physical_config["minimum_window_coverage_fraction"]
    )

    for row in frozen_controls:
        patch = np.asarray(tensor[int(row["patch_tensor_index"])])
        x0, y0, x1, y1 = map(int, row["analysis_bbox_xyxy"])
        identity = _identity(row)
        features: dict[str, Any] = {
            **identity,
            "candidate_bbox_nonzero_fraction": candidate_bbox_nonzero_fraction(
                patch,
                central_slice=CENTRAL_SLICE,
                source_bbox_xyxy=(x0, y0, x1, y1),
            ),
            **extract_features(patch, central_slice=CENTRAL_SLICE),
        }
        v3_feature_rows.append(features)
        decision = {**identity, **apply_rule(features, v3_profile)}
        v3_decisions.append(decision)

        voxel_um = float(row["voxel_size_um"][0])
        # Shared with the production extractor so the validated path and the
        # production path cannot diverge again.
        half_window_um = resolve_half_window_um(
            physical_config,
            depth_slices=patch.shape[0],
            central_slice=CENTRAL_SLICE,
            voxel_um=voxel_um,
        )
        physical = {
            **identity,
            "candidate_bbox_nonzero_fraction": features[
                "candidate_bbox_nonzero_fraction"
            ],
            **extract_physical_depth_features(
                patch,
                central_slice=CENTRAL_SLICE,
                voxel_um=voxel_um,
                half_window_um=half_window_um,
                canonical_step_um=float(physical_config["canonical_step_um"]),
                top_energy_band_um=float(physical_config["top_energy_band_um"]),
                central_band_half_width_um=float(
                    physical_config["central_band_half_width_um"]
                ),
                argmax_near_central_um=float(
                    physical_config["argmax_near_central_um"]
                ),
                peak_relative_height=float(
                    physical_config["peak_relative_height"]
                ),
            ),
        }
        physical_rows.append(physical)
        v4_routes.append(
            route_decision(
                decision,
                coverage_features=coverage_features,
                physical_features={key: str(value) for key, value in physical.items()},
                minimum_window_coverage_fraction=minimum_coverage,
            )
        )

    v3_features_path = output_root / "V3_FEATURES.csv"
    with v3_features_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(v3_feature_rows[0]))
        writer.writeheader()
        writer.writerows(v3_feature_rows)
    physical_path = output_root / "V4_PHYSICAL_FEATURES.csv"
    with physical_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(physical_rows[0]))
        writer.writeheader()
        writer.writerows(physical_rows)
    v3_decisions_path = output_root / "V3_DECISIONS.json"
    v4_routes_path = output_root / "V4_ROUTES.json"
    write_json(v3_decisions_path, v3_decisions)
    write_json(v4_routes_path, v4_routes)

    routes_by_component = {
        str(row["candidate_id"]): row for row in v4_routes
    }
    decisions_by_component = {
        str(row["candidate_id"]): row for row in v3_decisions
    }
    evaluated_controls: list[dict[str, Any]] = []
    for row in frozen_controls:
        component_id = str(row["component_id"])
        route = routes_by_component[component_id]
        decision = decisions_by_component[component_id]
        evaluated_controls.append(
            {
                **row,
                "decision_sources": {
                    "v3": {
                        "uri": v3_decisions_path.as_uri(),
                        "sha256": sha256(v3_decisions_path),
                    },
                    "v4": {
                        "uri": v4_routes_path.as_uri(),
                        "sha256": sha256(v4_routes_path),
                    },
                },
                "v3_retained": bool(decision["retained"]),
                "v4_tier": str(route["shadow_tier"]),
                "v4_not_discarded": bool(route["not_discarded"]),
            }
        )
    evaluated_controls_path = output_root / "FROZEN_EVALUATED_CONTROLS.json"
    write_json(evaluated_controls_path, evaluated_controls)
    manifest = build_manifest(
        evaluated_controls,
        benchmark_id=benchmark_id,
        controls_file=evaluated_controls_path.name,
        controls_file_sha256=sha256(evaluated_controls_path),
        v3_profile=v3_profile_path,
        v4_profile=v4_profile_path,
        materialization_receipt=materialization_path,
        manifest_role="POSTEXECUTION_EVALUATION_ENVELOPE",
    )
    manifest["preexecution_manifest"] = {
        "path": preexecution_manifest_path.name,
        "sha256": sha256(preexecution_manifest_path),
    }
    manifest_path = output_root / "EVALUATION_MANIFEST.json"
    write_json(manifest_path, manifest)
    preflight = preflight_manifest(manifest)
    write_json(output_root / "PREFLIGHT.json", preflight)
    if preflight["status"] != "READY_TO_FREEZE":
        raise RuntimeError(
            "frozen manifest failed preflight: "
            + ",".join(preflight["blocking_reasons"])
        )

    benchmark = evaluate_benchmark(manifest, evaluated_controls)
    benchmark_path = output_root / f"{benchmark_id}_RESULT.json"
    write_json(benchmark_path, benchmark)
    receipt = {
        "schema": "campaignx.multiscroll_transfer_one_shot_execution.v1",
        "benchmark_id": benchmark_id,
        "status": "ONE_SHOT_EXECUTION_COMPLETE",
        "completed_at_utc": utc_now(),
        "benchmark_status": benchmark["status"],
        "rerun_performed": False,
        "profile_thresholds_changed": False,
        "control_count": len(evaluated_controls),
        "v3_retained_count": sum(row["retained"] for row in v3_decisions),
        "v4_tier_counts": dict(
            sorted(Counter(row["shadow_tier"] for row in v4_routes).items())
        ),
        "artifacts": {
            path.name: sha256(path)
            for path in [
                preclaim_path,
                preexecution_manifest_path,
                output_root / "PREEXECUTION_PREFLIGHT.json",
                output_root / "EXECUTION_CLAIM.json",
                v3_features_path,
                v3_decisions_path,
                physical_path,
                v4_routes_path,
                evaluated_controls_path,
                manifest_path,
                output_root / "PREFLIGHT.json",
                benchmark_path,
            ]
        },
        "non_claims": [
            "No control outcome accepts ink, text, letters, or First Letters.",
            "Model predictions were not used as benchmark ground truth.",
            "Tier C is an extension request, never a negative decision.",
        ],
    }
    receipt["content_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "content_sha256"}
    )
    write_json(output_root / "EXECUTION_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--v3-profile", type=Path, required=True)
    parser.add_argument("--v4-profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--benchmark-id",
        choices=[
            "MULTISCROLL_TRANSFER_V1",
            "MULTISCROLL_TRANSFER_V2",
            "MULTISCROLL_TRANSFER_V3",
        ],
        default="MULTISCROLL_TRANSFER_V1",
    )
    args = parser.parse_args()
    receipt = execute(
        materialized_root=args.materialized_root.resolve(),
        v3_profile_path=args.v3_profile.resolve(),
        v4_profile_path=args.v4_profile.resolve(),
        output_root=args.output_root.resolve(),
        benchmark_id=args.benchmark_id,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
