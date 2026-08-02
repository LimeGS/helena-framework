#!/usr/bin/env python3
"""Build deterministic M0 manifests and enforce Phase 2 input contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
PHASE2 = ROOT / "phase2"
WORKSPACE = ROOT.parent
sys.path.insert(0, str(PHASE2 / "src"))

from campaign_x_phase2.contracts import assert_uri_allowed  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def build() -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    phase0 = ROOT / "phase0"
    phase1 = ROOT / "phase1"
    locked_inputs = [
        phase0 / "eligible_volumes.json",
        phase0 / "coordinate_contracts" / "ct_l0_xyz.json",
        phase0 / "target_contamination_ledger.json",
        phase1 / "PHASE1_AUTOMATED_CLOSEOUT.json",
        phase1 / "AUTOMATED_PROVISIONAL_TARGETS.json",
        PHASE2 / "TOP20_AUTOMATED_PORTFOLIO.json",
        PHASE2 / "PRIORITY_SCORE_V1.json",
        PHASE2 / "configs" / "phase2.json",
    ]
    for path in locked_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    eligible = read_json(phase0 / "eligible_volumes.json")
    contamination = read_json(phase0 / "target_contamination_ledger.json")
    for entry in eligible["entries"]:
        assert_uri_allowed(entry["sample_id"], entry["ct_uri"], contamination["entries"])

    portfolio = read_json(PHASE2 / "TOP20_AUTOMATED_PORTFOLIO.json")
    if portfolio["candidate_count"] != 20 or len(portfolio["candidates"]) != 20:
        raise ValueError("top-20 portfolio cardinality mismatch")

    source_lock = {
        "kind": "campaign_x_phase2_source_lock_v1",
        "generated_at_utc": now,
        "sources": {
            "campaign_x": {"path": str(ROOT), "commit": git_head(ROOT)},
            "villa": {
                "repository": "https://github.com/ScrollPrize/villa.git",
                "path": str(ROOT / "vendor" / "villa"),
                "commit": git_head(ROOT / "vendor" / "villa"),
            },
            "geowrap_ssl_lab": {
                "path": str(WORKSPACE / "geowrap-ssl-lab"),
                "commit": git_head(WORKSPACE / "geowrap-ssl-lab"),
            },
            "scroll_tracing_benchmark": {
                "path": str(WORKSPACE / "release" / "scroll-tracing-benchmark-gh"),
                "commit": git_head(WORKSPACE / "release" / "scroll-tracing-benchmark-gh"),
            },
            "campaign_framework": {
                "path": str(WORKSPACE / "release" / "helena-framework"),
                "commit": git_head(WORKSPACE / "release" / "helena-framework"),
            },
        },
    }
    data_manifest = {
        "kind": "campaign_x_phase2_data_manifest_v1",
        "generated_at_utc": now,
        "locked_inputs": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in locked_inputs
        ],
        "eligible_volume_count": len(eligible["entries"]),
        "portfolio_count": len(portfolio["candidates"]),
        "portfolio_seed_ids": [item["seed_id"] for item in portfolio["candidates"]],
        "paris4_validation": {
            "status": "BLOCKED_VALIDATION_DATA",
            "reason": "The public PHercParis4 bucket exposes volumes and representations, but the verified_patches, winding PointCollections, fibers, and tracks referenced by official fit_spiral.py were not found in the public listing.",
            "substitution_allowed": False,
        },
    }
    coordinate_contract = {
        "kind": "campaign_x_phase2_coordinate_contract_v1",
        "internal": "native L0 XYZ in voxels",
        "storage": "OME-Zarr ZYX",
        "official_pointcollections_p": "XYZ; official loader reverses p to obtain ZYX",
        "scaling": "forbidden unless an explicit source manifest declares a level transform",
        "round_trip_required": True,
    }
    contract = {
        "kind": "campaign_x_phase2_contract_v1",
        "approved_at_utc": now,
        "plan": "phase2/PHASE2_PLAN.md",
        "config": "phase2/configs/phase2.json",
        "budget_max_usd": 30.0,
        "claims_forbidden": True,
        "target_mode": "SHADOW_ONLY",
        "validation_status": "BLOCKED_VALIDATION_DATA",
    }
    state_path = PHASE2 / "RUN_STATE.json"
    if state_path.is_file():
        run_state = read_json(state_path)
        run_state["updated_at_utc"] = now
        run_state.setdefault("milestones", {})["M0"] = {
            "status": "PASSED_WITH_VALIDATION_BLOCKER",
            "blocker": "PARIS4_GROUND_TRUTH_NOT_PUBLICLY_LOCATED",
        }
    else:
        run_state = {
            "kind": "campaign_x_phase2_run_state_v1",
            "updated_at_utc": now,
            "overall": "RUNNING_ENGINEERING_VALIDATION_BLOCKED",
            "milestones": {
                "M0": {"status": "PASSED_WITH_VALIDATION_BLOCKER", "blocker": "PARIS4_GROUND_TRUTH_NOT_PUBLICLY_LOCATED"},
                **{f"M{index}": {"status": "PENDING"} for index in range(1, 9)},
            },
        }
    write_json(PHASE2 / "SOURCE_LOCK.json", source_lock)
    write_json(PHASE2 / "DATA_MANIFEST.json", data_manifest)
    write_json(PHASE2 / "COORDINATE_CONTRACT.json", coordinate_contract)
    write_json(PHASE2 / "CONTAMINATION_RULES.json", contamination)
    write_json(PHASE2 / "PHASE2_CONTRACT.json", contract)
    write_json(state_path, run_state)
    return run_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="build and validate M0 manifests")
    parser.parse_args()
    state = build()
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
