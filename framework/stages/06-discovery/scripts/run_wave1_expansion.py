#!/usr/bin/env python3
"""Execute and audit a locked Phase 4 surface-expansion plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PLAN = Path("/workspace/campaign-x-phase4/phase4/wave1/WAVE1_EXPANSION_PLAN.json")
DEFAULT_BINARY = Path("/workspace/villa-phase3/build-phase3-gcc13/bin/vc_grow_seg_from_seed")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def surface_files(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ("x.tif", "y.tif", "z.tif", "generations.tif", "meta.json"):
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"missing surface file: {path}")
        result[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return result


def measured_area(directory: Path) -> float:
    payload = load(directory / "meta.json")
    return float(payload["area_cm2"])


def command_for(binary: Path, target: dict[str, Any], surface: dict[str, Any]) -> list[str]:
    output = Path(surface["remote_output_surface"])
    profile = output.parent.parent / "profiles" / f"{surface['seed_id']}.json"
    return [
        str(binary),
        "--volume",
        target["surface_prediction_uri"],
        "--target-dir",
        str(output),
        "--params",
        str(profile),
        "--resume",
        surface["remote_source_surface"],
        "--resume-opt",
        surface["resume_optimization"],
        "--segment-name",
        "expanded",
    ]


def run_one(
    binary: Path,
    plan_sha256: str,
    target: dict[str, Any],
    surface: dict[str, Any],
) -> dict[str, Any]:
    source = Path(surface["remote_source_surface"])
    output = Path(surface["remote_output_surface"])
    output.mkdir(parents=True, exist_ok=True)
    command = command_for(binary, target, surface)
    log_path = output.parent / "grow.log"
    receipt_path = output.parent / "EXPANSION_RECEIPT.json"
    started = utc_now()
    reused = False
    exit_code = 0
    if (output / "meta.json").is_file():
        reused = True
        log_path.touch(exist_ok=True)
    else:
        env = os.environ.copy()
        env["VC_GROWPATCH_RNG_SEED"] = str(surface["rng_seed"])
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=1800,
                check=False,
            )
        exit_code = int(process.returncode)

    error = None
    files: dict[str, dict[str, Any]] = {}
    area = 0.0
    try:
        files = surface_files(output)
        area = measured_area(output)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    passed = (
        exit_code == 0
        and error is None
        and area >= float(surface["minimum_expanded_area_gate_cm2"])
    )
    receipt = {
        "kind": "campaign_x_phase4_surface_expansion_receipt_v1",
        "generated_at_utc": utc_now(),
        "started_at_utc": started,
        "status": "PASSED" if passed else "FAILED",
        "sample_id": target["sample_id"],
        "seed_id": surface["seed_id"],
        "plan_sha256": plan_sha256,
        "algorithm": "VC3D vc_grow_seg_from_seed resume",
        "command": command,
        "rng_seed": surface["rng_seed"],
        "profile_sha256": sha256(Path(command[command.index("--params") + 1])),
        "source_files": surface_files(source),
        "output_files": files,
        "initial_area_cm2": surface["initial_area_cm2"],
        "measured_expanded_area_cm2": area,
        "minimum_expanded_area_gate_cm2": surface["minimum_expanded_area_gate_cm2"],
        "exit_code": exit_code,
        "reused_existing_pass": reused,
        "log_path": str(log_path),
        "log_sha256": sha256(log_path),
        "error": error,
        "ink_used": False,
        "topology_claimed": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def execute(plan_path: Path, binary: Path, selected: set[str], jobs: int) -> dict[str, Any]:
    plan = load(plan_path)
    if plan["status"] != "LOCKED_READY":
        raise RuntimeError("surface-expansion plan is not LOCKED_READY")
    targets = [
        target for target in plan["targets"] if not selected or target["sample_id"] in selected
    ]
    if selected and selected != {target["sample_id"] for target in targets}:
        missing = sorted(selected - {target["sample_id"] for target in targets})
        raise RuntimeError(f"unknown selected targets: {missing}")
    tasks = [
        (target, surface)
        for target in targets
        for surface in target["surfaces"]
    ]
    plan_hash = sha256(plan_path)
    receipts: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [
            pool.submit(run_one, binary, plan_hash, target, surface)
            for target, surface in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            receipts.append(future.result())
    receipts.sort(key=lambda item: (item["sample_id"], item["seed_id"]))
    passed = all(item["status"] == "PASSED" for item in receipts)
    summary = {
        "kind": str(plan["kind"]).replace("_plan_", "_execution_"),
        "generated_at_utc": utc_now(),
        "status": "PASSED" if passed else "FAILED",
        "plan_path": str(plan_path),
        "plan_sha256": plan_hash,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "jobs": jobs,
        "surface_count": len(receipts),
        "passed_surface_count": sum(item["status"] == "PASSED" for item in receipts),
        "targets": sorted({item["sample_id"] for item in receipts}),
        "receipts": receipts,
        "ink_used": False,
        "next_gate": "PHYSICAL_INDEPENDENCE_AND_SAFE_RELATIONS",
    }
    output_name = plan_path.name.replace("_PLAN.json", "_EXECUTION.json")
    if output_name == plan_path.name:
        raise RuntimeError(f"plan filename cannot derive receipt name: {plan_path.name}")
    output = plan_path.parent / output_name
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    result = execute(args.plan, args.binary, set(args.target), args.jobs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
