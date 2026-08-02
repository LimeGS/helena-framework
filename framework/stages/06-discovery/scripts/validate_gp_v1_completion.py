#!/usr/bin/env python3
"""Fail-closed readiness check for the complete GP Scroll1 v1 pipeline.

Exit code 75 means the immutable upstream pipeline is still legitimately
incomplete.  Any contradictory completed artifact exits nonzero instead of
being treated as something a watcher may wait through.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


WAITING_EXIT = 75
COMPLETE_ROBUST_STATUSES = {
    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE",
    "COMPLETED_DIAGNOSTIC_ONLY",
}


class StillRunning(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_or_wait(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise StillRunning(f"{label} not present")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def validate(
    *,
    coarse_path: Path,
    v1_ranking_path: Path,
    robust_path: Path,
    ct_evaluation_path: Path,
    viewer_manifest_path: Path,
    checkpoint_path: Path,
    gate_freeze_path: Path,
    orchestrator_log_path: Path,
    expected_tasks: int,
    expected_windows: int | None,
    maximum_windows: int | None = None,
) -> dict[str, Any]:
    coarse = read_or_wait(coarse_path, "coarse receipt")
    coarse_status = coarse.get("status")
    if coarse_status != "COMPLETED_PRIORITIZATION_ONLY":
        if coarse_status in {"RUNNING_OR_PARTIAL", "PENDING", None}:
            raise StillRunning(f"coarse status is {coarse_status!r}")
        raise RuntimeError(f"unexpected coarse status: {coarse_status!r}")
    if int(coarse.get("task_count", -1)) != expected_tasks:
        raise RuntimeError("coarse task count changed")
    if int(coarse.get("completed_count", -1)) != expected_tasks:
        raise RuntimeError("complete coarse receipt has incomplete count")
    if int(coarse.get("failed_count", -1)) != 0:
        raise RuntimeError("complete coarse receipt contains failures")

    ranking = read_or_wait(v1_ranking_path, "v1 ranking")
    if ranking.get("kind") != (
        "campaign_x_phase4_expanded_surface_global_ranking_v1"
    ):
        raise RuntimeError("unexpected v1 ranking kind")
    if ranking.get("status") != "COMPLETED_PRIORITIZATION_ONLY":
        raise RuntimeError("v1 ranking is not complete")
    if ranking.get("source_batch_receipt_sha256") != sha256_file(coarse_path):
        raise RuntimeError("v1 ranking coarse hash mismatch")
    rows = ranking.get("global_priority")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("v1 ranking selected no windows")
    selected_count = len(rows)
    if expected_windows is not None and selected_count != expected_windows:
        raise RuntimeError("v1 ranking selected count mismatch")
    if maximum_windows is not None and not 1 <= selected_count <= maximum_windows:
        raise RuntimeError("v1 ranking selected count exceeds frozen maximum")
    if [int(row.get("global_rank", -1)) for row in rows] != list(
        range(1, selected_count + 1)
    ):
        raise RuntimeError("v1 ranking is not contiguous")

    robust = read_or_wait(robust_path, "v1 robust receipt")
    robust_count = int(robust.get("completed_count", -1))
    if robust_count < selected_count:
        if robust_count < 0 or robust_count > selected_count:
            raise RuntimeError("v1 robust progress count is invalid")
        raise StillRunning(
            f"v1 robust progress is {robust_count}/{selected_count}"
        )
    if robust_count != selected_count:
        raise RuntimeError("v1 robust completed count mismatch")
    if robust.get("status") not in COMPLETE_ROBUST_STATUSES:
        raise RuntimeError("v1 robust terminal status is invalid")
    if robust.get("selected_global_ranks") != list(
        range(1, selected_count + 1)
    ):
        raise RuntimeError("v1 robust ranks mismatch")
    if robust.get("global_ranking_sha256") != sha256_file(v1_ranking_path):
        raise RuntimeError("v1 robust ranking hash mismatch")
    checkpoint = robust.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("v1 robust checkpoint binding missing")
    if checkpoint.get("sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("v1 robust checkpoint hash mismatch")
    results = robust.get("results")
    if not isinstance(results, list) or len(results) != selected_count:
        raise RuntimeError("v1 robust result count mismatch")

    gate = read_or_wait(ct_evaluation_path, "v1 CT gate evaluation")
    if gate.get("kind") != "campaign_x_phase4_ct_fiber_gate_evaluation_v1":
        raise RuntimeError("unexpected v1 CT gate kind")
    if gate.get("status") != "COMPLETED":
        raise StillRunning("v1 CT gate evaluation is not complete")
    if gate.get("rule_sha256") != sha256_file(gate_freeze_path):
        raise RuntimeError("v1 CT gate rule hash mismatch")
    features = Path(str(gate.get("features", "")))
    if not features.is_file():
        raise RuntimeError("v1 CT feature ledger is missing")
    if gate.get("features_sha256") != sha256_file(features):
        raise RuntimeError("v1 CT feature ledger hash mismatch")

    if not viewer_manifest_path.is_file():
        raise StillRunning("v1 viewer manifest not present")
    if not orchestrator_log_path.is_file():
        raise StillRunning("v1 orchestrator log not present")
    if "COMPLETE: GP-Scroll1 shortlist pipeline" not in (
        orchestrator_log_path.read_text(errors="replace")
    ):
        raise StillRunning("v1 orchestrator has not emitted its final marker")

    return {
        "status": "READY_FOR_V2_DELTA",
        "coarse_task_count": expected_tasks,
        "v1_selected_count": selected_count,
        "frozen_maximum_windows": maximum_windows,
        "v1_robust_completed_count": robust_count,
        "coarse_receipt_sha256": sha256_file(coarse_path),
        "v1_ranking_sha256": sha256_file(v1_ranking_path),
        "v1_robust_receipt_sha256": sha256_file(robust_path),
        "v1_ct_evaluation_sha256": sha256_file(ct_evaluation_path),
        "v1_viewer_manifest_sha256": sha256_file(viewer_manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-receipt", type=Path, required=True)
    parser.add_argument("--v1-ranking", type=Path, required=True)
    parser.add_argument("--v1-robust-receipt", type=Path, required=True)
    parser.add_argument("--v1-ct-evaluation", type=Path, required=True)
    parser.add_argument("--v1-viewer-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-freeze", type=Path, required=True)
    parser.add_argument("--orchestrator-log", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=87)
    parser.add_argument("--expected-windows", type=int)
    parser.add_argument("--maximum-windows", type=int, default=48)
    args = parser.parse_args()
    try:
        result = validate(
            coarse_path=args.coarse_receipt.resolve(),
            v1_ranking_path=args.v1_ranking.resolve(),
            robust_path=args.v1_robust_receipt.resolve(),
            ct_evaluation_path=args.v1_ct_evaluation.resolve(),
            viewer_manifest_path=args.v1_viewer_manifest.resolve(),
            checkpoint_path=args.checkpoint.resolve(),
            gate_freeze_path=args.gate_freeze.resolve(),
            orchestrator_log_path=args.orchestrator_log.resolve(),
            expected_tasks=args.expected_tasks,
            expected_windows=args.expected_windows,
            maximum_windows=args.maximum_windows,
        )
    except StillRunning as error:
        print(json.dumps({"status": "WAITING_FOR_V1", "reason": str(error)}))
        return WAITING_EXIT
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
