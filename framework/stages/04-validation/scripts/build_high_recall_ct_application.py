#!/usr/bin/env python3
"""Adapt one high-recall router queue to the frozen Phase 4 CT fiber gate.

The high-recall router intentionally emits model-space bounding boxes, not ink
decisions.  This builder groups those boxes by robust window, creates the
strict-analysis-compatible adapter files consumed by the existing CT feature
extractor, and freezes one application spec.  It never changes the router
ranking or accepts/rejects ink.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.contracts.slice_order import ordered_tiff_files  # noqa: E402


ROUTER_KIND = "campaign_x_phase4_high_recall_ct_router_v1"
ROUTER_STATUS = "COMPLETED_DIAGNOSTIC_ROUTING_ONLY"
SPEC_KIND = "campaign_x_phase4_ct_fiber_gate_target_application_spec_v1"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise RuntimeError(f"path escapes Helena Framework root: {path}") from error


def target_voxel_um(root: Path, scroll_id: str) -> float:
    """Resolve voxel size from frozen target metadata or the source catalogue."""
    target_ids = [scroll_id]
    if scroll_id == "PHerc800":
        target_ids.append("PHerc0800")
    for target_id in target_ids:
        lock_path = root / "phase4" / "targets" / target_id / "TARGET_LOCK.json"
        if lock_path.is_file():
            return float(read_json(lock_path)["voxel_size_um"])

    readiness_path = (
        root / "phase4" / "target_readiness" / "TARGET_READINESS.json"
    )
    if readiness_path.is_file():
        targets = read_json(readiness_path).get("targets")
        if not isinstance(targets, list):
            raise RuntimeError("target-readiness targets must be a list")
        for row in targets:
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", ""))
            if sample_id in target_ids:
                voxel_um = float(row["voxel_size_um"])
                if voxel_um <= 0:
                    raise RuntimeError(
                        f"invalid target-readiness voxel size for {scroll_id}"
                    )
                return voxel_um

    # Reusable Stage 01 runs do not require the legacy Phase 4 target-lock
    # hierarchy.  Their authoritative CT metadata is the frozen eligible
    # volume catalogue used by the renderer.  Accept the refactored path
    # first and the historical path second, while still requiring one exact,
    # positive sample match.
    eligible_paths = [
        root / "workspace" / "catalog" / "eligible_volumes.json",
        root / "phase0" / "eligible_volumes.json",
    ]
    for eligible_path in eligible_paths:
        if not eligible_path.is_file():
            continue
        entries = read_json(eligible_path).get("entries")
        if not isinstance(entries, list):
            raise RuntimeError("eligible-volume entries must be a list")
        matches = [
            row
            for row in entries
            if isinstance(row, dict) and str(row.get("sample_id", "")) in target_ids
        ]
        if len(matches) != 1:
            if matches:
                raise RuntimeError(
                    f"eligible-volume catalogue has duplicate rows for {scroll_id}"
                )
            continue
        voxel_um = float(matches[0]["voxel_size_um"])
        if voxel_um <= 0:
            raise RuntimeError(
                f"invalid eligible-volume voxel size for {scroll_id}"
            )
        return voxel_um
    raise FileNotFoundError(
        f"no frozen voxel-size metadata for {scroll_id}"
    )


def validate_router(
    router: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if router.get("kind") != ROUTER_KIND:
        raise RuntimeError(f"unexpected router kind: {router.get('kind')!r}")
    if router.get("status") != ROUTER_STATUS:
        raise RuntimeError("high-recall router is not terminal")
    windows = router.get("windows")
    queue = router.get("ct_review_queue")
    if not isinstance(windows, list) or not windows:
        raise RuntimeError("router windows must be a non-empty list")
    if not isinstance(queue, list) or not queue:
        raise RuntimeError("router CT review queue must be a non-empty list")
    if int(router.get("window_count", -1)) != len(windows):
        raise RuntimeError("router window_count mismatch")
    if int(router.get("ct_review_queue_count", -1)) != len(queue):
        raise RuntimeError("router queue count mismatch")

    by_window: dict[str, dict[str, Any]] = {}
    all_candidates: dict[str, dict[str, Any]] = {}
    for window in windows:
        if not isinstance(window, dict):
            raise RuntimeError("router window row is not an object")
        window_id = str(window.get("window_id", ""))
        if not window_id or window_id in by_window:
            raise RuntimeError(f"invalid or duplicate window_id: {window_id!r}")
        shape = window.get("shape_y_x")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or min(map(int, shape)) < 1
        ):
            raise RuntimeError(f"invalid analysis shape for {window_id}")
        components = window.get("components")
        if not isinstance(components, list):
            raise RuntimeError(f"window components are not a list: {window_id}")
        for component in components:
            candidate_id = str(component.get("candidate_id", ""))
            if not candidate_id or candidate_id in all_candidates:
                raise RuntimeError(
                    f"invalid or duplicate router candidate: {candidate_id!r}"
                )
            all_candidates[candidate_id] = component
        by_window[window_id] = window

    queued_ids: set[str] = set()
    for queued in queue:
        if not isinstance(queued, dict):
            raise RuntimeError("queued component is not an object")
        candidate_id = str(queued.get("candidate_id", ""))
        if not candidate_id or candidate_id in queued_ids:
            raise RuntimeError(
                f"invalid or duplicate queued candidate: {candidate_id!r}"
            )
        queued_ids.add(candidate_id)
        source = all_candidates.get(candidate_id)
        if source is None or source != {
            key: queued[key] for key in source if key in queued
        }:
            # Quota routing adds rank/reason fields, but every original field
            # must remain byte-logically equal.
            if source is None or any(queued.get(key) != value for key, value in source.items()):
                raise RuntimeError(
                    f"queued candidate drifted from window component: {candidate_id}"
                )
        if str(queued.get("window_id", "")) not in by_window:
            raise RuntimeError(f"queued candidate references unknown window: {candidate_id}")
    return by_window, queue


def build(
    *,
    root: Path,
    metadata_root: Path | None = None,
    router_path: Path,
    output: Path,
    gate_freeze: Path,
    patch_radius_um: float,
    central_slice: int,
    voxel_um: float | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    metadata_root = (metadata_root or root).resolve()
    router_path = router_path.resolve()
    output = output.resolve()
    gate_freeze = gate_freeze.resolve()
    if output.exists():
        raise RuntimeError(
            "refusing to reuse high-recall CT adapter output directory"
        )
    if patch_radius_um <= 0:
        raise RuntimeError("patch radius must be positive")
    if central_slice < 0:
        raise RuntimeError("central slice must be non-negative")
    if voxel_um is not None and voxel_um <= 0:
        raise RuntimeError("explicit voxel size must be positive")
    if not gate_freeze.is_file():
        raise FileNotFoundError(gate_freeze)

    router = read_json(router_path)
    by_window, queue = validate_router(router)
    queue_by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in queue:
        queue_by_window[str(item["window_id"])].append(item)

    spec_path = output / "HIGH_RECALL_CT_FIBER_SPEC.json"
    receipt_path = output / "HIGH_RECALL_CT_ADAPTER_RECEIPT.json"
    if spec_path.exists() or receipt_path.exists():
        raise RuntimeError("refusing to overwrite high-recall CT adapter evidence")

    analyses: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for window_id in sorted(queue_by_window):
        window = by_window[window_id]
        scroll_id = str(window["scroll_id"])
        screening_dir = Path(str(window["screening_dir"])).resolve()
        tiff_dir = screening_dir.parent / "tiffs"
        tiffs, slice_ordering = ordered_tiff_files(tiff_dir)
        if len(tiffs) != 65:
            raise RuntimeError(
                f"{window_id} requires exactly 65 TIFF slices, found {len(tiffs)}"
            )
        resolved_voxel_um = (
            float(voxel_um)
            if voxel_um is not None
            else target_voxel_um(metadata_root, scroll_id)
        )

        candidates: list[dict[str, Any]] = []
        for item in queue_by_window[window_id]:
            y0, x0, y1, x1 = map(int, item["bbox_y0_x0_y1_x1"])
            if not (0 <= y0 < y1 <= int(window["shape_y_x"][0])):
                raise RuntimeError(f"candidate Y bounds outside model map: {item['candidate_id']}")
            if not (0 <= x0 < x1 <= int(window["shape_y_x"][1])):
                raise RuntimeError(f"candidate X bounds outside model map: {item['candidate_id']}")
            candidates.append(
                {
                    "candidate_id": str(item["candidate_id"]),
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "high_recall_routing_score": float(item["routing_score"]),
                    "quota_reasons": list(item["quota_reasons"]),
                    "source_component_rank_in_window": int(
                        item["component_rank_in_window"]
                    ),
                }
            )

        analysis_path = output / "analyses" / f"{window_id}.json"
        analysis = {
            "kind": "campaign_x_phase4_high_recall_ct_adapter_analysis_v1",
            "status": "ADAPTED_FOR_FROZEN_CT_GATE_ONLY",
            "generated_at_utc": utc_now(),
            "sample_id": scroll_id,
            "window_id": window_id,
            "input": {
                "shape_y_x": list(map(int, window["shape_y_x"])),
                "router_receipt": relative_to_root(router_path, root),
                "router_receipt_sha256": sha256_file(router_path),
                "screening_directory": relative_to_root(screening_dir, root),
            },
            "text_like_screening": {
                "screening_outcome": "HIGH_RECALL_COMPONENTS_REQUIRE_CT_REVIEW",
                "candidate_count": len(candidates),
                "candidates": candidates,
            },
            "interpretation": (
                "Schema adapter only. Every component remains an unverified "
                "high-recall model activation."
            ),
        }
        write_json(analysis_path, analysis)
        analyses.append(
            {
                "window_id": window_id,
                "sample_id": scroll_id,
                "path": relative_to_root(analysis_path, root),
                "sha256": sha256_file(analysis_path),
                "candidate_count": len(candidates),
            }
        )
        groups.append(
            {
                "group_id": window_id,
                "class": "UNKNOWN_TARGET_HIGH_RECALL",
                "tiff_directory": relative_to_root(tiff_dir, root),
                "slice_ordering": slice_ordering,
                "analysis": relative_to_root(analysis_path, root),
                "central_slice": central_slice,
                "voxel_um": resolved_voxel_um,
                "voxel_um_source": (
                    "frozen_surface_qc_input"
                    if voxel_um is not None
                    else "frozen_target_metadata"
                ),
            }
        )

    spec = {
        "kind": SPEC_KIND,
        "status": "FROZEN_BEFORE_TARGET_FEATURE_EXTRACTION",
        "frozen_at_utc": utc_now(),
        "application_name": "gp_scroll1_all_shortlist_high_recall_v1",
        "source_router_receipt": relative_to_root(router_path, root),
        "source_router_receipt_sha256": sha256_file(router_path),
        "patch_radius_um": patch_radius_um,
        "groups": groups,
        "policy": {
            "gate_freeze": relative_to_root(gate_freeze, metadata_root),
            "gate_freeze_sha256": sha256_file(gate_freeze),
            "router_scores_and_quotas_unchanged": True,
            "no_threshold_change_after_target_features": True,
            "retained_components_still_require_orthogonal_ct_review": True,
            "downranked_components_do_not_prove_absence": True,
            "no_automatic_ink_or_letter_acceptance": True,
        },
    }
    write_json(spec_path, spec)
    receipt = {
        "kind": "campaign_x_phase4_high_recall_ct_adapter_receipt_v1",
        "status": "READY_FOR_FROZEN_CT_FEATURE_EXTRACTION",
        "generated_at_utc": utc_now(),
        "router": {
            "path": relative_to_root(router_path, root),
            "sha256": sha256_file(router_path),
            "window_count": int(router["window_count"]),
            "queue_count": int(router["ct_review_queue_count"]),
        },
        "adapted_window_count": len(groups),
        "adapted_candidate_count": sum(row["candidate_count"] for row in analyses),
        "analyses": analyses,
        "spec": {
            "path": relative_to_root(spec_path, root),
            "sha256": sha256_file(spec_path),
        },
        "explicit_non_claims": [
            "not accepted ink",
            "not a letter",
            "not a First Letters claim",
            "not evidence of absence when the CT gate downranks a component",
        ],
    }
    write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help=(
            "root containing frozen target/eligible-volume metadata; defaults "
            "to --root for legacy in-tree runs"
        ),
    )
    parser.add_argument("--router-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-freeze", type=Path)
    parser.add_argument("--patch-radius-um", type=float, default=200.0)
    parser.add_argument("--central-slice", type=int, default=32)
    parser.add_argument(
        "--voxel-um",
        type=float,
        help=(
            "authoritative frozen voxel size supplied by the surface-QC input; "
            "when omitted, resolve it from --metadata-root"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    gate = (
        args.gate_freeze.resolve()
        if args.gate_freeze
        else root
        / "phase4"
        / "ct_fiber_benchmark_v1"
        / "CT_FIBER_GATE_FREEZE.json"
    )
    receipt = build(
        root=root,
        metadata_root=args.metadata_root,
        router_path=args.router_receipt,
        output=args.output,
        gate_freeze=gate,
        patch_radius_um=args.patch_radius_um,
        central_slice=args.central_slice,
        voxel_um=args.voxel_um,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
