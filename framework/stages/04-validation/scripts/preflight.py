#!/usr/bin/env python3
"""Fail-closed preflight for Helena Framework Phase 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
EXPECTED_SAMPLES = {
    "PHerc125", "PHerc191", "PHerc211", "PHerc257", "PHerc268",
    "PHerc358", "PHerc800", "PHerc813", "PHerc826", "PHerc1203",
    "PHerc1218", "PHerc1447", "PHerc1545",
}
EXPECTED_R61_AREA = 0.7005780483573333
EXPECTED_CHECKPOINT = "13fc568e9fc90954e5d3b9db623ff7d0a4ce24facab173fdb71d618c23e26cd4"
EXPECTED_VILLA = "05dcf0349356bc833670d61e5eca00be58376e35"
FORBIDDEN_PHERC1203_TOKEN = "20260319130212-2.403um-0.2m-77keV"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_if_changed(path: Path, value: dict[str, Any]) -> None:
    stable = {key: item for key, item in value.items() if key != "generated_at_utc"}
    if path.is_file():
        previous = load(path)
        if {key: item for key, item in previous.items() if key != "generated_at_utc"} == stable:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_checks(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}

    source_lock = load(root / "phase4/SOURCE_LOCK.json")
    closeout = load(root / "phase3/fast_ab_existing_surfaces/R61_AB_V1/R61_CLOSEOUT.json")
    guard = load(root / "phase3/fast_ab_existing_surfaces/R61_AB_V1/POST_FIT_RELATION_GUARD.json")
    eligible = load(root / "phase0/eligible_volumes.json")
    contamination = load(root / "phase0/target_contamination_ledger.json")
    ranking = load(root / "phase2/PRIORITY_SCORE_V1.json")
    model_lock = load(root / "phase4/MODEL_LOCK.json")
    cost = load(root / "phase4/COST_LEDGER.json")
    state = load(root / "phase4/RUN_STATE.json")
    environment = load(root / "phase4/ENVIRONMENT_RECEIPT.json")

    for name, locked in source_lock["inputs"].items():
        path = root / locked["path"]
        ok = path.is_file() and sha256(path) == locked["sha256"]
        checks[f"source_hash:{name}"] = ok
        if not ok:
            errors.append(f"Source lock mismatch: {name}")

    r61_checkpoint = closeout["artifacts"]["r61_checkpoint"]["sha256"]
    r61_area = closeout["metrics"]["official_satisfied_area_fraction"]["r61"]
    checks["r61_status"] = closeout["status"] == "PASSED_R61_AGGREGATE_AB_WITH_LOCAL_GUARD"
    checks["r61_checkpoint"] = r61_checkpoint == EXPECTED_CHECKPOINT
    checks["r61_area"] = abs(float(r61_area) - EXPECTED_R61_AREA) <= 1e-15
    checks["villa_commit"] = closeout["official_villa_commit"] == EXPECTED_VILLA
    checks["guard_active"] = (
        guard["status"] == "ACTIVE_LOCAL_GUARD"
        and guard["policy"]["correct_sign_required"] is True
        and guard["policy"]["maximum_absolute_winding_error"] == 0.5
    )

    entries = eligible["entries"]
    samples = {entry["sample_id"] for entry in entries}
    checks["eligible_13_exact"] = len(entries) == 13 and samples == EXPECTED_SAMPLES
    checks["eligible_targets_allowed"] = all(entry["target_allowed"] is True for entry in entries)

    ledger_entries = contamination["entries"]
    ledger_samples = {entry["sample_id"] for entry in ledger_entries}
    checks["contamination_ledger_13_exact"] = len(ledger_entries) == 13 and ledger_samples == EXPECTED_SAMPLES
    p1203 = next(entry for entry in entries if entry["sample_id"] == "PHerc1203")
    p1203_ledger = next(entry for entry in ledger_entries if entry["sample_id"] == "PHerc1203")
    checks["pherc1203_eligible_scan_only"] = (
        "9.362um" in p1203["ct_uri"]
        and FORBIDDEN_PHERC1203_TOKEN in p1203["higher_resolution_sibling_uri"]
        and FORBIDDEN_PHERC1203_TOKEN in p1203_ledger["higher_resolution_sibling_uri"]
    )

    ranked = ranking["ranked_candidates"]
    ranked_scrolls = {item["scroll_id"] for item in ranked}
    checks["candidate_count_78"] = len(ranked) == 78
    checks["candidate_scrolls_12"] = len(ranked_scrolls) == 12
    checks["pherc257_absent_from_candidates"] = "PHerc257" not in ranked_scrolls

    checks["geometry_lock"] = (
        model_lock["geometry"]["status"] == "LOCKED"
        and model_lock["geometry"]["champion_reference_checkpoint_sha256"] == EXPECTED_CHECKPOINT
        and model_lock["geometry"]["cross_scroll_checkpoint_reuse_allowed"] is False
        and model_lock["geometry"]["target_specific_fit_required"] is True
    )
    checks["ink_fail_closed"] = (
        model_lock["ink"]["status"] == "NOT_YET_LOCKED"
        and model_lock["ink"]["execution_allowed"] is False
        and state["ink_execution_allowed"] is False
    )
    checks["budget_valid"] = (
        cost["hard_ceiling_usd"] == 25.0
        and cost["spent_usd"] >= 0
        and cost["committed_usd"] >= 0
        and cost["spent_usd"] + cost["committed_usd"] < cost["hard_ceiling_usd"]
    )
    checks["environment_ready"] = (
        environment["status"] == "PASSED"
        and environment["vast"]["instance_id"] == 45010090
        and environment["vast"]["actual_status"] == "running"
        and environment["vast"]["gpu"] == "NVIDIA GeForce RTX 3090"
        and environment["vast"]["gpu_memory_free_mib"] >= 20000
        and environment["vast"]["workspace_free_bytes"] >= 50_000_000_000
        and environment["official_villa_commit"] == EXPECTED_VILLA
        and environment["r61_checkpoint_sha256"] == EXPECTED_CHECKPOINT
        and environment["remote_phase3_archive_present"] is True
        and environment["s3_receipt_status"] == "COMPLETE"
    )

    for name, ok in checks.items():
        if not ok and not any(name in error for error in errors):
            errors.append(f"Failed check: {name}")

    return {
        "kind": "campaign_x_phase4_preflight_receipt_v1",
        "generated_at_utc": utc_now(),
        "status": "PASSED" if not errors else "FAILED",
        "gate": "P4_0_PREFLIGHT",
        "checks": checks,
        "errors": errors,
        "champion": {
            "official_villa_commit": closeout["official_villa_commit"],
            "r61_checkpoint_sha256": r61_checkpoint,
            "official_satisfied_area_fraction": r61_area,
            "guard_status": guard["status"],
        },
        "inventory": {
            "eligible_target_count": len(entries),
            "candidate_count": len(ranked),
            "candidate_scroll_count": len(ranked_scrolls),
        },
        "ink_execution_allowed": False,
        "private_candidate_policy": "PRIVATE_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "phase4/PREFLIGHT_RECEIPT.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    receipt = run_checks(args.root.resolve())
    if not args.no_write:
        write_if_changed(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
