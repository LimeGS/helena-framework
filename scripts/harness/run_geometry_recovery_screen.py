#!/usr/bin/env python3
"""Render and six-replica screen a locked geometry-first recovery batch.

This is a diagnostic router.  It never promotes a detector activation to ink,
text, or a First Letters claim.  Each completed row has enough hashes and
counts to be rerun or audited independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


MODEL_FAMILY = "timesformer_GP_scroll1"
REQUIRED_TIFF_COUNT = 65
MIN_FREE_GIB = 4
NO_COMMON_VALID_ERROR = "RuntimeError: screening maps have no common valid pixels"
DEFAULT_SUMMARY_NAME = "GEOMETRY_RECOVERY_V2_SCREEN_EXECUTION.json"
DEFAULT_SCREENING_LABEL = "robust_gp_scroll1_v1"
V1_BATCH_KIND = "campaign_x_phase4_geometry_first_recovery_v1_batch_plan"
V2_BATCH_KIND = "campaign_x_phase4_geometry_first_recovery_v2_batch_plan"
V3_BATCH_KIND = "campaign_x_phase4_geometry_first_recovery_v3_batch_plan"
V4_BATCH_KIND = "campaign_x_phase4_geometry_first_recovery_v4_batch_plan"
ARCHIVE_AWARE_CLEARANCE_VOXELS = 256.0
SCREENING_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_screen_configuration(
    checkpoint: Path,
    *,
    expected_checkpoint_sha256: str | None,
    screening_label: str,
    comparability_classification: str,
) -> dict[str, str | None]:
    """Fail closed on model drift and make non-comparable screens explicit.

    A geometry plan is intentionally ink blind, so its selected surface can be
    screened by a newly published public model.  That new result must never be
    blended with a historical model series merely because the model family has
    the same name.  This small receipt binds the exact weights and exposes that
    distinction to every downstream router.
    """
    if not SCREENING_LABEL_PATTERN.fullmatch(screening_label):
        raise RuntimeError("screening label must be a plain safe path component")
    if comparability_classification not in {
        "COMPARABLE_FROZEN_SERIES",
        "NONCOMPARABLE_CURRENT_PUBLIC_MODEL",
        "EXPLORATORY_UNVERIFIED_MODEL",
    }:
        raise RuntimeError("screen comparability classification is not recognized")
    actual = sha256(checkpoint)
    if expected_checkpoint_sha256 is not None:
        expected = expected_checkpoint_sha256.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("expected checkpoint SHA256 must be a lowercase 64-hex digest")
        if actual != expected:
            raise RuntimeError("checkpoint hash differs from the locked screen configuration")
    if comparability_classification == "COMPARABLE_FROZEN_SERIES" and expected_checkpoint_sha256 is None:
        raise RuntimeError("a comparable frozen screen requires an explicit checkpoint SHA256")
    return {
        "checkpoint_sha256": actual,
        "expected_checkpoint_sha256": expected_checkpoint_sha256,
        "screening_label": screening_label,
        "comparability_classification": comparability_classification,
    }


def validate_locked_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the immutable selected batch, accepting only known locked recipes.

    Later versions do not relax the V1 screen contract: they add frozen
    archive/public/historical-geometry clearances before screening a surface.
    """
    if plan.get("status") != "LOCKED_READY_PILOT":
        raise RuntimeError("screen requires a locked geometry recovery batch plan")
    if plan.get("selection_rule", {}).get("ink_used") is not False:
        raise RuntimeError("geometry recovery batch must be ink blind")
    kind = plan.get("kind")
    planned = plan.get("selected_pilot")
    if kind == V1_BATCH_KIND:
        pass
    elif kind == V2_BATCH_KIND:
        rule = plan.get("selection_rule", {})
        if float(rule.get("recovery_archive_clearance_voxels", 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS:
            raise RuntimeError("archive-aware geometry recovery needs the locked archive clearance")
        if not isinstance(planned, list) or any(
            row.get("state") != "SELECTED_FOR_ARCHIVE_AWARE_PILOT"
            or float(row.get("combined_novelty_gap_voxels", 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS
            for row in planned
        ):
            raise RuntimeError("archive-aware selected pilot is incomplete or below its locked clearance")
    elif kind == V3_BATCH_KIND:
        rule = plan.get("selection_rule", {})
        if (
            float(rule.get("recovery_archive_clearance_voxels", 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS
            or float(rule.get("public_archive_clearance_voxels", 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS
        ):
            raise RuntimeError("known-surface-aware geometry recovery needs both locked archive clearances")
        public_snapshot = plan.get("public_archive_snapshot")
        if not isinstance(public_snapshot, dict) or int(public_snapshot.get("surface_count", 0)) < 0:
            raise RuntimeError("known-surface-aware geometry recovery lacks its public snapshot")
        if not isinstance(planned, list) or any(
            row.get("state") != "SELECTED_FOR_PUBLIC_AND_ARCHIVE_AWARE_PILOT"
            or float(row.get("combined_novelty_gap_voxels", 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS
            for row in planned
        ):
            raise RuntimeError("known-surface-aware selected pilot is incomplete or below locked clearance")
    elif kind == V4_BATCH_KIND:
        rule = plan.get("selection_rule", {})
        required_clearances = (
            "recovery_archive_clearance_voxels",
            "public_archive_clearance_voxels",
            "historical_growth_clearance_voxels",
        )
        if any(float(rule.get(key, 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS for key in required_clearances):
            raise RuntimeError("all-known-geometry recovery needs every locked clearance")
        if not isinstance(planned, list) or any(
            row.get("state") != "SELECTED_FOR_ALL_KNOWN_GEOMETRY_PILOT"
            or float(row.get("combined_novelty_gap_voxels", 0.0)) < ARCHIVE_AWARE_CLEARANCE_VOXELS
            for row in planned
        ):
            raise RuntimeError("all-known-geometry selected pilot is incomplete or below locked clearance")
    else:
        raise RuntimeError("screen requires a supported locked geometry recovery batch plan")
    if not isinstance(planned, list) or not planned:
        raise RuntimeError("screen has no locked selected pilot")
    return planned


def tiffs(path: Path) -> tuple[list[Path], str]:
    return ordered_tiff_files(path, require_numeric=True, allow_empty=True)


def run_logged(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        return int(subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode)


def screen(
    *, plan_path: Path, eligible_path: Path, runtime: Path, renderer: Path,
    checkpoint: Path, inference: Path, analysis: Path, execute: bool,
    sample_ids: set[str] | None = None,
    summary_name: str = DEFAULT_SUMMARY_NAME,
    screening_label: str = DEFAULT_SCREENING_LABEL,
    expected_checkpoint_sha256: str | None = None,
    comparability_classification: str = "EXPLORATORY_UNVERIFIED_MODEL",
) -> dict[str, Any]:
    plan = load(plan_path)
    planned = validate_locked_plan(plan)
    selected = [row for row in planned if sample_ids is None or str(row["sample_id"]) in sample_ids]
    if sample_ids is not None and {str(row["sample_id"]) for row in selected} != sample_ids:
        raise RuntimeError("requested sample is not present in the locked plan")
    if not selected:
        raise RuntimeError("requested sub-batch is empty")
    eligible = {str(row["sample_id"]): row for row in load(eligible_path)["entries"]}
    if not set(str(row["sample_id"]) for row in selected) <= set(eligible):
        raise RuntimeError("batch references an ineligible scroll")
    for artifact in (renderer, checkpoint, inference, analysis):
        if not artifact.is_file():
            raise RuntimeError(f"screen artifact is missing: {artifact}")
    model_provenance = validate_screen_configuration(
        checkpoint,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        screening_label=screening_label,
        comparability_classification=comparability_classification,
    )
    if Path(summary_name).name != summary_name or not summary_name.endswith(".json"):
        raise RuntimeError("summary name must be a plain .json filename")
    runtime.mkdir(parents=True, exist_ok=True)
    receipts = []
    for item in sorted(selected, key=lambda row: str(row["sample_id"])):
        sample_id, seed_id = str(item["sample_id"]), str(item["seed_id"])
        base = runtime / sample_id / seed_id
        surface = base / "surface"
        tiff_dir = base / "tiffs"
        screening = base / screening_label
        analysis_dir = screening / "analysis"
        analysis_path = analysis_dir / "INK_STABILITY_ANALYSIS.json"
        entry = eligible[sample_id]
        row = {"sample_id": sample_id, "seed_id": seed_id, "state": "PENDING"}
        if not all((surface / name).is_file() for name in ("x.tif", "y.tif", "z.tif", "meta.json")):
            row.update({"state": "FAILED", "error": "surface TIFXYZ is incomplete"})
            receipts.append(row)
            continue
        if not execute:
            dry_run_tiffs, dry_run_ordering = tiffs(tiff_dir)
            row.update({
                "state": "DRY_RUN",
                "tiff_count": len(dry_run_tiffs),
                "slice_ordering": dry_run_ordering,
            })
            receipts.append(row)
            continue
        if shutil.disk_usage(runtime).free < MIN_FREE_GIB * 1024**3:
            row.update({"state": "BLOCKED_DISK_GUARD", "error": f"less than {MIN_FREE_GIB} GiB free"})
            receipts.append(row)
            break
        if len(tiffs(tiff_dir)[0]) == 0:
            command = [
                str(renderer), "--segmentation", str(surface), "--volume", str(runtime / "ct_metadata_cache" / f"{sample_id}.zarr"),
                "--remote-url", str(entry["ct_uri"]), "--prefetch-remote", "--scale", "1", "--group-idx", "0", "--auto-crop",
                "--flatten", "--flatten-iterations", "10", "--num-slices", "65", "--slice-step", "1", "--cache-gb", "4", "--timeout", "90",
                "--voxel-size", str(entry["voxel_size_um"]), "--voxel-unit", "micrometer", "--tif-output", str(tiff_dir), "--log-path", str(base / "ct-render.log"),
            ]
            row["render_exit_code"] = run_logged(command, base / "ct-render.stdout.log")
        rendered, slice_ordering = tiffs(tiff_dir)
        count = len(rendered)
        row["tiff_count"] = count
        row["slice_ordering"] = slice_ordering
        if count != REQUIRED_TIFF_COUNT:
            row.update({"state": "FAILED", "error": "renderer did not produce exactly 65 TIFFs"})
            receipts.append(row)
            continue
        receipt_path = screening / "INK_SCREENING_RECEIPT.json"
        if not receipt_path.is_file():
            command = [
                "python3", str(inference), "--sample-id", sample_id, "--tiff-dir", str(tiff_dir), "--checkpoint", str(checkpoint),
                "--model-family", MODEL_FAMILY, "--output", str(screening), "--depth-centers", "25,32,39", "--tiling-offsets", "0,8",
                "--frames", "26", "--source-pixel-um", str(entry["voxel_size_um"]), "--training-pixel-um", "7.91",
                "--source-slice-um", str(entry["voxel_size_um"]), "--training-slice-um", "7.91", "--tile-size", "64", "--stride", "16",
                "--batch-size", "128", "--min-valid-ratio", "0.60", "--device", "cuda",
            ]
            row["inference_exit_code"] = run_logged(command, base / "robust-inference.stdout.log")
        if not receipt_path.is_file():
            row.update({"state": "FAILED", "error": "six-replica inference receipt is absent"})
            receipts.append(row)
            continue
        if not analysis_path.is_file():
            command = [
                "python3", str(analysis), "--sample-id", sample_id, "--screening-dir", str(screening), "--tiff-dir", str(tiff_dir),
                "--output", str(analysis_dir), "--source-center", "32", "--source-pixel-um", str(entry["voxel_size_um"]),
                "--training-pixel-um", "7.91", "--glyph-threshold", "0.5", "--hotspots", "12", "--crop-size", "384",
            ]
            row["analysis_exit_code"] = run_logged(command, base / "robust-analysis.stdout.log")
        if not analysis_path.is_file():
            analysis_log = base / "robust-analysis.stdout.log"
            no_common = (
                receipt_path.is_file()
                and analysis_log.is_file()
                and NO_COMMON_VALID_ERROR in analysis_log.read_text(encoding="utf-8", errors="replace")
                and count == REQUIRED_TIFF_COUNT
            )
            if no_common:
                row.update({
                    "state": "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
                    "screening_outcome": "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
                    "manual_review_route": "NO_CROSS_REPLICA_CT_SUPPORT",
                    "analysis_log": str(analysis_log),
                    "analysis_log_sha256": sha256(analysis_log),
                    "error": NO_COMMON_VALID_ERROR,
                })
                receipts.append(row)
                continue
            row.update({"state": "FAILED", "error": "stability analysis is absent"})
            receipts.append(row)
            continue
        payload = load(analysis_path)
        screening_summary = payload.get("text_like_screening", {})
        row.update({
            "state": "COMPLETED_DIAGNOSTIC_ONLY",
            "analysis": str(analysis_path), "analysis_sha256": sha256(analysis_path),
            "screening_receipt": str(receipt_path), "screening_receipt_sha256": sha256(receipt_path),
            "screening_outcome": screening_summary.get("screening_outcome"),
            "glyph_like_candidate_count": screening_summary.get("glyph_like_candidate_count"),
            "rows_with_at_least_four_candidates": screening_summary.get("rows_with_at_least_four_candidates"),
            "manual_review_route": payload.get("manual_review_routing", {}).get("route"),
        })
        receipts.append(row)
    output = {
        "kind": "campaign_x_phase4_geometry_recovery_v1_screen_execution",
        "generated_at_utc": utc_now(),
        "status": "DRY_RUN" if not execute else "COMPLETED_DIAGNOSTIC_ONLY" if len(receipts) == len(selected) and all(row["state"] in {"COMPLETED_DIAGNOSTIC_ONLY", "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS"} for row in receipts) else "PARTIAL_OR_FAILED",
        "plan_sha256": sha256(plan_path), "planned_selected_count": len(planned),
        "executed_sample_ids": sorted(str(row["sample_id"]) for row in selected),
        "selected_count": len(selected), "completed_count": sum(row["state"] == "COMPLETED_DIAGNOSTIC_ONLY" for row in receipts),
        "renderer_sha256": sha256(renderer), "checkpoint_sha256": sha256(checkpoint), "inference_sha256": sha256(inference), "analysis_sha256": sha256(analysis),
        "model_provenance": model_provenance,
        "receipts": receipts,
        "policy": ["six replica maps are a router only", "no activation is accepted as ink or letters", "no-common-valid CT is terminal insufficiency, not a negative", "all text-like positives still require orthogonal CT"],
    }
    (runtime / summary_name).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--screening-label", default=DEFAULT_SCREENING_LABEL)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument(
        "--comparability-classification",
        choices=("COMPARABLE_FROZEN_SERIES", "NONCOMPARABLE_CURRENT_PUBLIC_MODEL", "EXPLORATORY_UNVERIFIED_MODEL"),
        default="EXPLORATORY_UNVERIFIED_MODEL",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = screen(
        plan_path=args.plan,
        eligible_path=args.eligible,
        runtime=args.runtime,
        renderer=args.renderer,
        checkpoint=args.checkpoint,
        inference=args.inference,
        analysis=args.analysis,
        execute=args.execute,
        sample_ids=set(args.sample) or None,
        summary_name=args.summary_name,
        screening_label=args.screening_label,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        comparability_classification=args.comparability_classification,
    )
    print(json.dumps({key: output[key] for key in ("status", "selected_count", "completed_count")}, sort_keys=True))
    return 0 if output["status"] in {"DRY_RUN", "COMPLETED_DIAGNOSTIC_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
