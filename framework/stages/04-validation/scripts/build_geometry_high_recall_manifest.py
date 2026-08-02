#!/usr/bin/env python3
"""Freeze a high-recall CT-router manifest from a geometry-recovery screen.

The strict geometry-recovery screen is intentionally a high-specificity
channel.  This adapter turns its *already generated* six-map evidence into an
additive high-recall channel without changing the strict result, rerunning the
ink model, or treating any component as ink.  It is deliberately fail-closed:
every screen receipt, strict analysis, and six raw maps must agree before a
manifest can be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_KIND = "campaign_x_phase4_high_recall_router_manifest_v1"
SUMMARY_KIND = "campaign_x_phase4_geometry_recovery_v1_screen_execution"
ANALYSIS_KIND = "campaign_x_phase4_ink_stability_analysis_v1"
SCREENING_KIND = "campaign_x_phase4_timesformer_private_screening_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return value


def relative_to(base: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def resolve_under(root: Path, raw: str, label: str) -> Path:
    path = Path(raw)
    # Screen receipts from an in-tree runner may contain either a path relative
    # to the worktree or the runner's absolute in-tree path.  Both refer to
    # the same immutable artifact; accept the latter only after the identical
    # containment check.  An arbitrary absolute path remains fail-closed.
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes root: {raw}") from error
    return resolved


def canonical_map_rows(
    *, screening_dir: Path, analysis: dict[str, Any], screening: dict[str, Any]
) -> list[dict[str, Any]]:
    analysis_maps = analysis.get("input", {}).get("maps")
    runs = screening.get("inference", {}).get("runs")
    if not isinstance(analysis_maps, list) or len(analysis_maps) != 6:
        raise RuntimeError("strict analysis must bind exactly six replica maps")
    if not isinstance(runs, list) or len(runs) != 6:
        raise RuntimeError("screening receipt must bind exactly six inference runs")
    analysis_by_name: dict[str, dict[str, Any]] = {}
    for row in analysis_maps:
        if not isinstance(row, dict):
            raise RuntimeError("analysis map row is not an object")
        name = str(row.get("file", ""))
        if not name or Path(name).name != name or name in analysis_by_name:
            raise RuntimeError("analysis has unsafe or duplicate map name")
        analysis_by_name[name] = row
    run_by_name: dict[str, dict[str, Any]] = {}
    for row in runs:
        if not isinstance(row, dict):
            raise RuntimeError("screening inference row is not an object")
        name = str(row.get("npy", ""))
        if not name or Path(name).name != name or name in run_by_name:
            raise RuntimeError("screening has unsafe or duplicate map name")
        run_by_name[name] = row
    if set(analysis_by_name) != set(run_by_name):
        raise RuntimeError("analysis and screening replica names differ")

    rows: list[dict[str, Any]] = []
    coordinates: set[tuple[int, int]] = set()
    shape: list[int] | None = None
    for name in sorted(analysis_by_name):
        analysis_row = analysis_by_name[name]
        run = run_by_name[name]
        path = screening_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"replica map is missing: {path}")
        digest = sha256_file(path)
        if digest != str(analysis_row.get("sha256", "")):
            raise RuntimeError(f"analysis hash mismatch for {name}")
        if digest != str(run.get("npy_sha256", "")):
            raise RuntimeError(f"screening hash mismatch for {name}")
        depth = int(analysis_row.get("depth_center", -1))
        offset = int(analysis_row.get("tiling_offset", -1))
        if depth != int(run.get("depth_center_source_index", -2)):
            raise RuntimeError(f"depth mismatch for {name}")
        if offset != int(run.get("tiling_offset", -2)):
            raise RuntimeError(f"offset mismatch for {name}")
        current = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if current.ndim != 2 or not np.issubdtype(current.dtype, np.number):
                raise RuntimeError(f"replica is not a two-dimensional numeric map: {name}")
            current_shape = [int(current.shape[0]), int(current.shape[1])]
        finally:
            mmap = getattr(current, "_mmap", None)
            if mmap is not None:
                mmap.close()
        if shape is None:
            shape = current_shape
        elif shape != current_shape:
            raise RuntimeError("replica-map shapes differ")
        coordinates.add((depth, offset))
        rows.append(
            {
                "file": name,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "shape_y_x": current_shape,
                "depth_center": depth,
                "tiling_offset": offset,
            }
        )
    depths = {depth for depth, _ in coordinates}
    offsets = {offset for _, offset in coordinates}
    if len(coordinates) != 6 or len(depths) != 3 or len(offsets) != 2:
        raise RuntimeError("replicas are not exactly one 3-depth by 2-offset grid")
    if set(coordinates) != {(depth, offset) for depth in depths for offset in offsets}:
        raise RuntimeError("replica grid is incomplete")
    return rows


def completed_receipts_for_recovery(
    summary: dict[str, Any], *, allow_disk_guard_subset: bool
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return terminal receipts, optionally from one tightly defined disk stop.

    This is deliberately not a general partial-batch escape hatch.  It permits
    only a completed prefix interrupted by one ``BLOCKED_DISK_GUARD`` receipt
    and exactly one planned-but-unstarted sample.  That structure is the V7
    resource failure: it cannot silently hide a model/render error or select a
    favorable subset after seeing inference output.
    """
    status = str(summary.get("status", ""))
    receipts = summary.get("receipts")
    selected_count = int(summary.get("selected_count", -1))
    completed_count = int(summary.get("completed_count", -1))
    if not isinstance(receipts, list) or selected_count < 1:
        raise RuntimeError("screen summary receipt count disagrees with selected_count")
    if status == "COMPLETED_DIAGNOSTIC_ONLY":
        if len(receipts) != selected_count:
            raise RuntimeError("screen summary is incomplete")
        terminal_states = {
            "COMPLETED_DIAGNOSTIC_ONLY",
            "CT_INSUFFICIENT_NO_COMMON_VALID_PIXELS",
        }
        if any(
            not isinstance(row, dict) or row.get("state") not in terminal_states
            for row in receipts
        ):
            raise RuntimeError("nonterminal sample in screen summary")
        completed = [
            row
            for row in receipts
            if row.get("state") == "COMPLETED_DIAGNOSTIC_ONLY"
        ]
        if len(completed) != completed_count:
            raise RuntimeError("screen summary completed receipt count mismatch")
        if not completed:
            raise RuntimeError("screen summary has no CT-supported sample to route")
        return completed, None
    if not allow_disk_guard_subset:
        raise RuntimeError("screen execution summary is not terminal")
    if status != "PARTIAL_OR_FAILED":
        raise RuntimeError("partial recovery requires PARTIAL_OR_FAILED status")
    completed = [row for row in receipts if isinstance(row, dict) and row.get("state") == "COMPLETED_DIAGNOSTIC_ONLY"]
    blocked = [row for row in receipts if isinstance(row, dict) and row.get("state") == "BLOCKED_DISK_GUARD"]
    if len(completed) != completed_count or not completed:
        raise RuntimeError("partial recovery completed receipt count mismatch")
    if len(blocked) != 1 or str(blocked[0].get("error", "")) != "less than 4 GiB free":
        raise RuntimeError("partial recovery requires exactly one disk-guard block")
    if len(receipts) != completed_count + 1 or selected_count != completed_count + 2:
        raise RuntimeError("partial recovery must have exactly one blocked and one unstarted sample")
    planned = summary.get("executed_sample_ids")
    if not isinstance(planned, list) or len(planned) != selected_count:
        raise RuntimeError("partial recovery lacks the frozen planned sample list")
    completed_ids = {str(row.get("sample_id", "")) for row in completed}
    blocked_ids = {str(row.get("sample_id", "")) for row in blocked}
    unstarted = [str(sample_id) for sample_id in planned if str(sample_id) not in completed_ids | blocked_ids]
    if len(unstarted) != 1 or not unstarted[0]:
        raise RuntimeError("partial recovery cannot identify exactly one unstarted sample")
    return completed, {
        "kind": "DISK_GUARD_COMPLETED_SUBSET_V1",
        "original_status": status,
        "original_selected_count": selected_count,
        "completed_count": completed_count,
        "blocked_disk_guard": {"sample_id": str(blocked[0].get("sample_id", "")), "seed_id": str(blocked[0].get("seed_id", "")), "error": str(blocked[0].get("error", ""))},
        "unstarted_sample_ids": unstarted,
        "no_model_reinference": True,
        "strict_gate_unchanged": True,
    }


def build(
    root: Path, summary_path: Path, output: Path, *, allow_disk_guard_subset: bool = False
) -> dict[str, Any]:
    root = root.resolve()
    summary_path = resolve_under(root, str(summary_path), "screen execution summary")
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite manifest: {output}")
    summary = read_json(summary_path, "screen execution summary")
    if summary.get("kind") != SUMMARY_KIND:
        raise RuntimeError("unexpected screen execution summary kind")
    receipts, partial_recovery = completed_receipts_for_recovery(
        summary, allow_disk_guard_subset=allow_disk_guard_subset
    )
    checkpoint = str(summary.get("checkpoint_sha256", ""))
    if len(checkpoint) != 64:
        raise RuntimeError("screen summary lacks a checkpoint SHA-256")

    windows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for receipt in sorted(receipts, key=lambda row: str(row.get("sample_id", ""))):
        if not isinstance(receipt, dict):
            raise RuntimeError("screen receipt is not an object")
        if receipt.get("state") != "COMPLETED_DIAGNOSTIC_ONLY":
            raise RuntimeError("nonterminal sample in screen summary")
        if int(receipt.get("tiff_count", -1)) != 65:
            raise RuntimeError("high-recall CT review requires exactly 65 TIFF slices")
        sample_id = str(receipt.get("sample_id", "")).strip()
        seed_id = str(receipt.get("seed_id", "")).strip()
        if not sample_id or not seed_id or (sample_id, seed_id) in identities:
            raise RuntimeError("invalid or duplicate sample/seed identity")
        identities.add((sample_id, seed_id))
        analysis_path = resolve_under(root, str(receipt.get("analysis", "")), "analysis")
        if sha256_file(analysis_path) != str(receipt.get("analysis_sha256", "")):
            raise RuntimeError(f"screen summary analysis hash mismatch: {seed_id}")
        analysis = read_json(analysis_path, "strict analysis")
        if analysis.get("kind") != ANALYSIS_KIND or analysis.get("sample_id") != sample_id:
            raise RuntimeError(f"strict analysis identity mismatch: {seed_id}")
        screening_path = resolve_under(root, str(receipt.get("screening_receipt", "")), "screening receipt")
        if sha256_file(screening_path) != str(receipt.get("screening_receipt_sha256", "")):
            raise RuntimeError(f"screen summary screening hash mismatch: {seed_id}")
        screening = read_json(screening_path, "screening receipt")
        if screening.get("kind") != SCREENING_KIND:
            raise RuntimeError(f"unexpected screening receipt kind: {seed_id}")
        screening_dir = screening_path.parent
        maps = canonical_map_rows(
            screening_dir=screening_dir, analysis=analysis, screening=screening
        )
        shape = maps[0]["shape_y_x"]
        windows.append(
            {
                "scroll_id": sample_id,
                "window_id": seed_id,
                "screening_dir": relative_to(output.parent, screening_dir),
                "strict_gate_analysis": relative_to(output.parent, analysis_path),
                "provenance": {
                    "source_screen_execution": relative_to(root, summary_path),
                    "source_screen_execution_sha256": sha256_file(summary_path),
                    "sample_id": sample_id,
                    "surface_id": seed_id,
                    "source_crop_xyxy": [0, 0, int(shape[1]), int(shape[0])],
                    "model_family": "timesformer_GP_scroll1",
                    "checkpoint_sha256": checkpoint,
                    "replica_grid": {"map_count": 6, "maps": maps},
                },
            }
        )
    manifest = {
        "kind": MANIFEST_KIND,
        "status": "READY_FOR_HIGH_RECALL_CT_ROUTING",
        "generated_at_utc": utc_now(),
        "scope": "ADDITIVE_GEOMETRY_RECOVERY_HIGH_RECALL_V1",
        "source_screen_execution": relative_to(root, summary_path),
        "source_screen_execution_sha256": sha256_file(summary_path),
        "strict_gate_unchanged": True,
        "policy": {
            "no_model_reinference": True,
            "no_strict_gate_rerun_or_threshold_change": True,
            "single_line_signals_are_retainable_for_CT_routing": True,
            "queued_components_are_not_ink_letters_or_readings": True,
        },
        "window_count": len(windows),
        "windows": windows,
    }
    if partial_recovery is not None:
        manifest["partial_recovery"] = partial_recovery
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-disk-guard-subset",
        action="store_true",
        help="allow only the documented one-blocked/one-unstarted disk-guard subset",
    )
    args = parser.parse_args()
    manifest = build(
        args.root,
        args.screen_summary,
        args.output,
        allow_disk_guard_subset=args.allow_disk_guard_subset,
    )
    print(json.dumps({"status": manifest["status"], "window_count": manifest["window_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
