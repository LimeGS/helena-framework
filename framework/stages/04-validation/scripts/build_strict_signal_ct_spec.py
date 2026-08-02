#!/usr/bin/env python3
"""Freeze an immediate CT diagnostic for one completed strict ink signal.

This is deliberately separate from the all-window high-recall router.  It
never changes its queue or thresholds; it merely permits an already strict,
completed signal to receive the same frozen depth-localization screen while
the independent complement batch continues to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

_STAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists()) / "framework/stages"
for _stage_scripts in _STAGE_ROOT.glob("*/scripts"):
    _stage_scripts_text = str(_stage_scripts)
    if _stage_scripts_text not in sys.path:
        sys.path.insert(0, _stage_scripts_text)
ROOT = _STAGE_ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any

from framework.contracts.slice_order import ordered_tiff_files
from build_high_recall_ct_application import target_voxel_um


STRICT_OUTCOME = "POTENTIAL_TEXT_LIKE_SIGNAL_REQUIRES_CT_REVIEW"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as error:
        raise RuntimeError(f"path escapes root: {path}") from error


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(*, root: Path, robust_shard: Path, seed_id: str, output: Path, gate: Path) -> dict[str, Any]:
    root, robust_shard, output, gate = (path.resolve() for path in (root, robust_shard, output, gate))
    if output.exists():
        raise RuntimeError(f"refusing to overwrite CT signal output: {output}")
    if not gate.is_file():
        raise FileNotFoundError(gate)
    shard = read_json(robust_shard)
    if shard.get("kind") != "campaign_x_phase4_coverage_surface_v2_robust_shard_v1":
        raise RuntimeError("unexpected robust shard kind")
    task = next((row for row in shard.get("tasks", []) if row.get("seed_id") == seed_id), None)
    if not isinstance(task, dict):
        raise RuntimeError(f"strict signal is absent from shard: {seed_id}")
    if task.get("status") != "ROBUST_COMPLETED" or task.get("screening_outcome") != STRICT_OUTCOME:
        raise RuntimeError("task is not a completed strict CT-review signal")

    analysis_path = Path(str(task["analysis"])).resolve()
    receipt_path = Path(str(task["screening_receipt"])).resolve()
    if sha256_file(analysis_path) != str(task["analysis_sha256"]):
        raise RuntimeError("strict analysis hash mismatch")
    if sha256_file(receipt_path) != str(task["screening_receipt_sha256"]):
        raise RuntimeError("screening receipt hash mismatch")
    analysis = read_json(analysis_path)
    receipt = read_json(receipt_path)
    if analysis.get("status") != "COMPLETED_DIAGNOSTIC_ONLY" or receipt.get("status") != "COMPLETED_DIAGNOSTIC_ONLY":
        raise RuntimeError("strict signal provenance is not terminal")
    screening = analysis.get("text_like_screening", {})
    candidates = screening.get("candidates")
    if screening.get("screening_outcome") != STRICT_OUTCOME or not isinstance(candidates, list) or not candidates:
        raise RuntimeError("strict signal has no immutable candidate inventory")
    if int(screening.get("glyph_like_candidate_count", -1)) != len(candidates):
        raise RuntimeError("strict candidate inventory count mismatch")

    tiff_dir = analysis_path.parent.parent.parent / "tiffs"
    tiffs, slice_ordering = ordered_tiff_files(tiff_dir)
    if len(tiffs) != 65:
        raise RuntimeError(f"strict CT signal requires 65 TIFFs, found {len(tiffs)}")
    sample_id = str(task["sample_id"])
    spec = {
        "kind": "campaign_x_phase4_strict_signal_ct_spec_v1",
        "status": "FROZEN_BEFORE_TARGET_FEATURE_EXTRACTION",
        "frozen_at_utc": utc_now(),
        "scope": "POST_HOC_STRICT_SIGNAL_DIAGNOSTIC_ONLY",
        "patch_radius_um": 200.0,
        "groups": [{
            "group_id": seed_id,
            "class": "POST_HOC_STRICT_SIGNAL_DIAGNOSTIC",
            "tiff_directory": relative_to_root(tiff_dir, root),
            "slice_ordering": slice_ordering,
            "analysis": relative_to_root(analysis_path, root),
            "central_slice": 32,
            "voxel_um": target_voxel_um(root, sample_id),
        }],
        "source": {
            "robust_shard": relative_to_root(robust_shard, root),
            "robust_shard_sha256": sha256_file(robust_shard),
            "seed_id": seed_id,
            "analysis": relative_to_root(analysis_path, root),
            "analysis_sha256": sha256_file(analysis_path),
            "screening_receipt": relative_to_root(receipt_path, root),
            "screening_receipt_sha256": sha256_file(receipt_path),
            "candidate_count": len(candidates),
        },
        "policy": {
            "gate_freeze": relative_to_root(gate, root),
            "gate_freeze_sha256": sha256_file(gate),
            "new_model_inference": False,
            "router_queue_modified": False,
            "threshold_change_after_observation": False,
            "retained_components_still_require_orthogonal_ct_review": True,
            "no_automatic_ink_or_letter_acceptance": True,
        },
    }
    analysis_copy = {**analysis, "text_like_screening": {**screening, "candidates": candidates}}
    output.mkdir(parents=True)
    analysis_out = output / "STRICT_SIGNAL_ANALYSIS.json"
    write_json(analysis_out, analysis_copy)
    # The extractor consumes the copied analysis so all CT inputs are contained
    # in this application directory; the hash above still binds its source.
    spec["groups"][0]["analysis"] = relative_to_root(analysis_out, root)
    spec_path = output / "STRICT_SIGNAL_CT_SPEC.json"
    write_json(spec_path, spec)
    result = {
        "kind": "campaign_x_phase4_strict_signal_ct_spec_receipt_v1",
        "status": "READY_FOR_FROZEN_CT_FEATURE_EXTRACTION",
        "scope": spec["scope"],
        "generated_at_utc": utc_now(),
        "spec": {"path": relative_to_root(spec_path, root), "sha256": sha256_file(spec_path)},
        "analysis": {"path": relative_to_root(analysis_out, root), "sha256": sha256_file(analysis_out)},
        "candidate_count": len(candidates),
        "explicit_non_claims": ["not accepted ink", "not a letter", "not a First Letters claim"],
    }
    write_json(output / "STRICT_SIGNAL_CT_RECEIPT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--robust-shard", type=Path, required=True)
    parser.add_argument("--seed-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-freeze", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    gate = args.gate_freeze or root / "phase4" / "ct_fiber_benchmark_v1" / "CT_FIBER_GATE_FREEZE.json"
    print(json.dumps(build(root=root, robust_shard=args.robust_shard, seed_id=args.seed_id, output=args.output, gate=gate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
