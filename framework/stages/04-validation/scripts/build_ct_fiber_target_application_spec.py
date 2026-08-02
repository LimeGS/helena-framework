#!/usr/bin/env python3
"""Freeze a CT fiber-gate application spec from a completed robust batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPLETED_STATUSES = {
    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE",
    "COMPLETED_DIAGNOSTIC_ONLY",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_groups(
    *,
    root: Path,
    batch_root: Path,
    receipt: dict[str, Any],
    central_slice: int,
    robust_root_name: str = "robust_windows_v1",
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in receipt["results"]:
        rank = int(result["global_rank"])
        sample_id = str(result["sample_id"])
        surface_id = str(result["surface_id"])
        group_id = f"rank-{rank:02d}-{sample_id}-{surface_id}"
        if group_id in seen:
            raise RuntimeError(f"duplicate group_id: {group_id}")
        seen.add(group_id)
        lock = read_json(
            root / "phase4" / "targets" / sample_id / "TARGET_LOCK.json"
        )
        groups.append(
            {
                "group_id": group_id,
                "class": "UNKNOWN_TARGET",
                "tiff_directory": str(
                    (
                        batch_root
                        / robust_root_name
                        / group_id
                        / "tiffs"
                    ).relative_to(root)
                ),
                "analysis": str(
                    Path(result["analysis"]).resolve().relative_to(root)
                ),
                "central_slice": central_slice,
                "voxel_um": float(lock["voxel_size_um"]),
            }
        )
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--batch-receipt", type=Path, required=True)
    parser.add_argument("--gate-freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--application-name", required=True)
    parser.add_argument(
        "--robust-root-name",
        default="robust_windows_v1",
    )
    parser.add_argument("--central-slice", type=int, default=32)
    parser.add_argument("--patch-radius-um", type=float, default=200.0)
    args = parser.parse_args()

    root = args.root.resolve()
    batch_root = (
        args.batch_root.resolve()
        if args.batch_root
        else root / "phase4" / "expanded_candidate_surface_screen_v1"
    )
    receipt_path = args.batch_receipt.resolve()
    gate_freeze = (
        args.gate_freeze.resolve()
        if args.gate_freeze
        else root
        / "phase4"
        / "ct_fiber_benchmark_v1"
        / "CT_FIBER_GATE_FREEZE.json"
    )
    receipt = read_json(receipt_path)
    if receipt.get("status") not in COMPLETED_STATUSES:
        raise RuntimeError("robust batch receipt is not complete")
    if int(receipt["completed_count"]) != len(receipt["results"]):
        raise RuntimeError("robust batch receipt result count mismatch")
    if not gate_freeze.is_file():
        raise FileNotFoundError(gate_freeze)

    payload = {
        "kind": "campaign_x_phase4_ct_fiber_gate_target_application_spec_v1",
        "status": "FROZEN_BEFORE_TARGET_FEATURE_EXTRACTION",
        "frozen_at_utc": utc_now(),
        "application_name": args.application_name,
        "source_batch_receipt": str(receipt_path.relative_to(root)),
        "source_batch_receipt_sha256": sha256_file(receipt_path),
        "patch_radius_um": args.patch_radius_um,
        "groups": build_groups(
            root=root,
            batch_root=batch_root,
            receipt=receipt,
            central_slice=args.central_slice,
            robust_root_name=args.robust_root_name,
        ),
        "policy": {
            "gate_freeze": str(gate_freeze.relative_to(root)),
            "gate_freeze_sha256": sha256_file(gate_freeze),
            "no_threshold_change_after_target_features": True,
            "retained_components_still_require_orthogonal_ct_review": True,
            "no_target_acceptance_from_this_application": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "group_count": len(payload["groups"]),
                "status": payload["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
