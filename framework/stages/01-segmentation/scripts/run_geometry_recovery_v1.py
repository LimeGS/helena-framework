#!/usr/bin/env python3
"""Run a small, locked ink-blind VC3D geometry-recovery pilot safely.

The input plan is created locally by ``build_geometry_recovery_v1.py``.
This runner is deliberately one-worker only: a seed expansion can consume a
large remote cache, and every terminal outcome receives a receipt rather than
being silently retried or reclassified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BINARY = Path("/workspace/villa-phase3/build-phase3-gcc13/bin/vc_grow_seg_from_seed")
REQUIRED = ("x.tif", "y.tif", "z.tif", "meta.json")
MINIMUM_AREA_CM2 = 0.25
DEFAULT_SUMMARY_NAME = "GEOMETRY_RECOVERY_V1_EXECUTION.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def complete_surface(path: Path) -> bool:
    return all((path / filename).is_file() and (path / filename).stat().st_size > 0 for filename in REQUIRED)


def profile(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "seed",
        "generations": 35,
        "step_size": 20,
        "min_area_cm": 0.0,
        "use_cuda": False,
        "voxelsize": float(item["voxel_size_um"]),
    }


def run(
    plan_path: Path,
    output_root: Path,
    binary: Path,
    execute: bool,
    archive_surface_root: Path | None = None,
    sample_ids: set[str] | None = None,
    summary_name: str = DEFAULT_SUMMARY_NAME,
) -> dict[str, Any]:
    plan = load(plan_path)
    if plan.get("status") not in {"LOCKED_READY", "LOCKED_READY_PILOT"}:
        raise RuntimeError("geometry-recovery plan is not locked")
    if plan.get("selection_rule", {}).get("ink_used") is not False:
        raise RuntimeError("geometry-recovery plan must be ink blind")
    planned = plan.get("selected_pilot", [])
    if not planned:
        raise RuntimeError("locked plan has no selected seed")
    selected = [item for item in planned if sample_ids is None or str(item["sample_id"]) in sample_ids]
    if sample_ids is not None and {str(item["sample_id"]) for item in selected} != sample_ids:
        raise RuntimeError("requested sample is not present in the locked plan")
    if not selected:
        raise RuntimeError("requested sub-batch is empty")
    if len(selected) > 13:
        raise RuntimeError("pilot may contain at most one seed per target")
    if len({item["sample_id"] for item in selected}) != len(selected):
        raise RuntimeError("pilot must contain no more than one seed per target")
    if not binary.is_file():
        raise RuntimeError(f"VC3D grow binary is unavailable: {binary}")
    if Path(summary_name).name != summary_name or not summary_name.endswith(".json"):
        raise RuntimeError("summary name must be a plain .json filename")
    # Only when something will actually be grown. The headroom is here so a
    # grow does not die halfway and leave a truncated surface behind; a dry run
    # writes a profile and a receipt, which is kilobytes. Checking it
    # unconditionally made plan validation fail on a machine that was merely
    # low on disk -- and it failed as "less than 4 GiB free before geometry
    # pilot", which reads like the plan was rejected.
    if execute:
        probe = output_root.parent if output_root.exists() else output_root.parents[1]
        free = shutil.disk_usage(probe).free
        if free < 4 * 1024**3:
            raise RuntimeError(
                f"less than 4 GiB free before geometry pilot: {free / 1024**3:.1f} GiB at {probe}")
    plan_hash = sha256(plan_path)
    receipts = []
    archive_aware_kind = "campaign_x_phase4_geometry_first_recovery_v2_batch_plan"
    public_archive_aware_kind = "campaign_x_phase4_geometry_first_recovery_v3_batch_plan"
    all_known_geometry_kind = "campaign_x_phase4_geometry_first_recovery_v4_batch_plan"
    for item in selected:
        state = item.get("state")
        if state == "SELECTED_FOR_ARCHIVE_AWARE_PILOT":
            if plan.get("kind") != archive_aware_kind:
                raise RuntimeError(f"archive-aware selection has wrong plan kind: {item['seed_id']}")
            if float(item.get("combined_novelty_gap_voxels", -1.0)) < 256.0:
                raise RuntimeError(f"archive-aware selection fails novelty clearance: {item['seed_id']}")
        elif state == "SELECTED_FOR_PUBLIC_AND_ARCHIVE_AWARE_PILOT":
            if plan.get("kind") != public_archive_aware_kind:
                raise RuntimeError(f"public-and-archive-aware selection has wrong plan kind: {item['seed_id']}")
            if float(item.get("combined_novelty_gap_voxels", -1.0)) < 256.0:
                raise RuntimeError(f"public-and-archive-aware selection fails novelty clearance: {item['seed_id']}")
            if float(plan.get("selection_rule", {}).get("public_archive_clearance_voxels", 0.0)) < 256.0:
                raise RuntimeError(f"public archive clearance is not frozen: {item['seed_id']}")
        elif state == "SELECTED_FOR_ALL_KNOWN_GEOMETRY_PILOT":
            if plan.get("kind") != all_known_geometry_kind:
                raise RuntimeError(f"all-known-geometry selection has wrong plan kind: {item['seed_id']}")
            if float(item.get("combined_novelty_gap_voxels", -1.0)) < 256.0:
                raise RuntimeError(f"all-known-geometry selection fails novelty clearance: {item['seed_id']}")
            selection_rule = plan.get("selection_rule", {})
            for key in (
                "recovery_archive_clearance_voxels",
                "public_archive_clearance_voxels",
                "historical_growth_clearance_voxels",
            ):
                if float(selection_rule.get(key, 0.0)) < 256.0:
                    raise RuntimeError(f"all-known-geometry clearance is not frozen ({key}): {item['seed_id']}")
        elif state != "SELECTED_FOR_ONE_SURFACE_PILOT":
            raise RuntimeError(f"unselected item in pilot: {item['seed_id']}")
        coordinate = item.get("candidate_coordinate_xyz_l0")
        if not isinstance(coordinate, dict) or not all(axis in coordinate for axis in "xyz"):
            raise RuntimeError(f"missing fixed seed coordinate: {item['seed_id']}")
        seed_id = str(item["seed_id"])
        base = output_root / str(item["sample_id"]) / seed_id
        surface = base / "surface"
        config = base / "profile.json"
        log = base / "grow.log"
        receipt_path = base / "GROWTH_RECEIPT.json"
        base.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(profile(item), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            str(binary), "--volume", str(item["prediction_uri"]), "--target-dir", str(surface),
            "--params", str(config), "--seed", *(str(int(coordinate[axis])) for axis in "xyz"),
            "--segment-name", seed_id,
        ]
        reused = complete_surface(surface)
        exit_code: int | None = None
        error: str | None = None
        if execute and not reused:
            env = os.environ.copy()
            env["VC_GROWPATCH_RNG_SEED"] = hashlib.sha256(seed_id.encode()).hexdigest()[:16]
            try:
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=1800, check=False)
                exit_code = int(result.returncode)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        elif execute:
            exit_code = 0
        area = None
        files: dict[str, dict[str, Any]] = {}
        if complete_surface(surface):
            meta = load(surface / "meta.json")
            area = float(meta.get("area_cm2", 0.0))
            files = {filename: {"sha256": sha256(surface / filename), "size_bytes": (surface / filename).stat().st_size} for filename in REQUIRED}
        status = (
            "DRY_RUN" if not execute else "PASSED" if exit_code == 0 and error is None and area is not None and area >= MINIMUM_AREA_CM2 else "FAILED"
        )
        archived_surface: dict[str, Any] | None = None
        if status == "PASSED" and archive_surface_root is not None:
            archive = archive_surface_root / str(item["sample_id"]) / seed_id
            archive.mkdir(parents=True, exist_ok=True)
            archived_files: dict[str, dict[str, Any]] = {}
            for filename in REQUIRED:
                source, destination = surface / filename, archive / filename
                source_hash = sha256(source)
                if destination.is_file():
                    if sha256(destination) != source_hash:
                        raise RuntimeError(f"refusing to overwrite non-matching archived surface: {destination}")
                else:
                    shutil.copy2(source, destination)
                archived_files[filename] = {"sha256": source_hash, "size_bytes": source.stat().st_size}
            archived_surface = {"path": str(archive), "files": archived_files}
        receipt = {
            "kind": "campaign_x_phase4_geometry_recovery_v1_growth_receipt",
            "generated_at_utc": utc_now(),
            "status": status,
            "plan_sha256": plan_hash,
            "plan_query_scope": plan.get("query_scope"),
            "seed_id": seed_id,
            "sample_id": item["sample_id"],
            "seed_coordinate_xyz_l0": coordinate,
            "novelty_gap_to_prior_surface_voxels": item["novelty_gap_to_prior_surface_voxels"],
            "profile": profile(item),
            "profile_sha256": sha256(config),
            "command": command,
            "exit_code": exit_code,
            "reused_complete_surface": reused,
            "area_cm2": area,
            "minimum_area_cm2": MINIMUM_AREA_CM2,
            "files": files,
            "archived_surface": archived_surface,
            "log_sha256": sha256(log) if log.is_file() else None,
            "error": error,
            "ink_used": False,
            "non_claims": ["a grown surface is not accepted physical geometry", "no ink or First Letters claim is made"],
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipts.append(receipt)
    summary = {
        "kind": "campaign_x_phase4_geometry_recovery_v1_execution",
        "generated_at_utc": utc_now(),
        "status": "DRY_RUN" if not execute else "PASSED" if all(item["status"] == "PASSED" for item in receipts) else "FAILED",
        "plan_sha256": plan_hash,
        "planned_selected_count": len(planned),
        "executed_sample_ids": sorted(str(item["sample_id"]) for item in selected),
        "receipt_count": len(receipts),
        "passed_count": sum(item["status"] == "PASSED" for item in receipts),
        "receipts": receipts,
        "next_gate": "ORTHOGONAL_CT_AND_REPLICA_SCREENING",
    }
    (output_root / summary_name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--archive-surface-root", type=Path)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = run(
        args.plan,
        args.output_root,
        args.binary,
        args.execute,
        args.archive_surface_root,
        set(args.sample) or None,
        args.summary_name,
    )
    print(json.dumps({key: payload[key] for key in ("status", "receipt_count", "passed_count", "next_gate")}, sort_keys=True))
    return 0 if payload["status"] in {"DRY_RUN", "PASSED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
