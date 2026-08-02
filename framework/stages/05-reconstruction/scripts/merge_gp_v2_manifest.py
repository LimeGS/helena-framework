#!/usr/bin/env python3
"""Merge reused v1 and newly computed v2-delta robust evidence by provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FREEZE_KIND = "campaign_x_phase4_gp_scroll1_v2_delta_freeze_v1"
ROBUST_KIND = "campaign_x_phase4_expanded_window_robust_batch_v1"
GATE_KIND = "campaign_x_phase4_ct_fiber_gate_evaluation_v1"
MERGE_KIND = "campaign_x_phase4_gp_scroll1_v2_merged_evidence_manifest_v1"
COMPLETE_ROBUST_STATUSES = {
    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE",
    "COMPLETED_DIAGNOSTIC_ONLY",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def write_exact(path: Path, value: dict[str, Any], *, dry_run: bool) -> None:
    content = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary artifact blocks write: {temporary}")
    temporary.write_bytes(content)
    temporary.replace(path)


def result_identity(result: dict[str, Any]) -> dict[str, Any]:
    crop = result.get("source_crop_xyxy")
    if not isinstance(crop, list) or len(crop) != 4:
        raise RuntimeError("robust result has invalid source crop")
    return {
        "sample_id": str(result.get("sample_id", "")),
        "surface_id": str(result.get("surface_id", "")),
        "source_crop_xyxy": [int(value) for value in crop],
    }


def validate_robust(
    path: Path,
    *,
    expected_count: int,
    expected_ranking_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    require_file(path, "robust receipt")
    receipt = read_json(path)
    if receipt.get("kind") != ROBUST_KIND:
        raise RuntimeError(f"unexpected robust receipt kind: {path}")
    if receipt.get("status") not in COMPLETE_ROBUST_STATUSES:
        raise RuntimeError(f"robust receipt is not complete: {path}")
    if int(receipt.get("completed_count", -1)) != expected_count:
        raise RuntimeError(f"robust completed_count mismatch: {path}")
    if receipt.get("selected_global_ranks") != list(
        range(1, expected_count + 1)
    ):
        raise RuntimeError(f"robust rank list mismatch: {path}")
    if receipt.get("global_ranking_sha256") != expected_ranking_sha256:
        raise RuntimeError(f"robust ranking hash mismatch: {path}")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"robust checkpoint missing: {path}")
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise RuntimeError(f"robust checkpoint hash mismatch: {path}")
    results = receipt.get("results")
    if not isinstance(results, list) or len(results) != expected_count:
        raise RuntimeError(f"robust results mismatch: {path}")
    by_rank: dict[int, dict[str, Any]] = {}
    for result in results:
        rank = int(result.get("global_rank", -1))
        if rank in by_rank:
            raise RuntimeError(f"duplicate robust result rank {rank}: {path}")
        analysis = Path(str(result.get("analysis", "")))
        require_file(analysis, f"analysis for robust rank {rank}")
        if result.get("analysis_sha256") != sha256_file(analysis):
            raise RuntimeError(f"analysis hash mismatch at robust rank {rank}")
        by_rank[rank] = result
    return receipt, by_rank


def validate_gate(
    path: Path,
    *,
    gate_sha256: str,
) -> dict[str, Any]:
    require_file(path, "CT gate evaluation")
    value = read_json(path)
    if value.get("kind") != GATE_KIND or value.get("status") != "COMPLETED":
        raise RuntimeError(f"CT gate evaluation is not complete: {path}")
    if value.get("rule_sha256") != gate_sha256:
        raise RuntimeError(f"CT gate rule hash mismatch: {path}")
    features = Path(str(value.get("features", "")))
    require_file(features, "CT gate feature ledger")
    if value.get("features_sha256") != sha256_file(features):
        raise RuntimeError(f"CT gate feature hash mismatch: {path}")
    return value


def build_manifest(
    *,
    freeze_path: Path,
    v1_robust_receipt_path: Path,
    v1_ct_evaluation_path: Path,
    delta_robust_receipt_path: Path | None,
    delta_ct_evaluation_path: Path | None,
) -> dict[str, Any]:
    require_file(freeze_path, "v2 delta freeze")
    freeze = read_json(freeze_path)
    if freeze.get("kind") != FREEZE_KIND:
        raise RuntimeError("unexpected v2 delta freeze kind")
    if freeze.get("status") != "FROZEN_BEFORE_V2_DELTA_EXECUTION":
        raise RuntimeError("v2 delta freeze has unexpected status")
    selection = freeze.get("selection")
    if not isinstance(selection, dict):
        raise RuntimeError("v2 delta freeze selection missing")
    entries = selection.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("v2 delta freeze entries missing")
    expected_v2 = int(selection.get("expected_v2_count", -1))
    reused_count = int(selection.get("reused_v1_exact_count", -1))
    delta_count = int(selection.get("delta_compute_count", -1))
    if expected_v2 < 1 or reused_count < 0 or delta_count < 0:
        raise RuntimeError("invalid v2 delta counts")
    if reused_count + delta_count != expected_v2 or len(entries) != expected_v2:
        raise RuntimeError("v2 delta freeze count invariant failed")

    v1_binding = freeze.get("v1_robust_receipt")
    checkpoint = freeze.get("checkpoint")
    gate = freeze.get("ct_fiber_gate")
    delta_ranking = freeze.get("delta_ranking")
    if not all(
        isinstance(value, dict)
        for value in (v1_binding, checkpoint, gate, delta_ranking)
    ):
        raise RuntimeError("v2 delta freeze bindings are incomplete")
    for label, binding in (
        ("checkpoint", checkpoint),
        ("CT gate freeze", gate),
        ("delta ranking", delta_ranking),
    ):
        bound_path = Path(str(binding.get("path", "")))
        require_file(bound_path, label)
        if sha256_file(bound_path) != binding.get("sha256"):
            raise RuntimeError(f"{label} no longer matches freeze")
    for label in ("coarse_receipt", "v1_ranking", "v2_ranking"):
        binding = freeze.get(label)
        if not isinstance(binding, dict):
            raise RuntimeError(f"{label} binding missing")
        bound_path = Path(str(binding.get("path", "")))
        require_file(bound_path, label)
        if sha256_file(bound_path) != binding.get("sha256"):
            raise RuntimeError(f"{label} no longer matches freeze")
    if sha256_file(v1_robust_receipt_path) != v1_binding["sha256"]:
        raise RuntimeError("v1 robust receipt no longer matches freeze")

    v1_ranking = freeze.get("v1_ranking")
    if not isinstance(v1_ranking, dict):
        raise RuntimeError("v1 ranking binding missing")
    _, v1_results = validate_robust(
        v1_robust_receipt_path,
        expected_count=int(v1_ranking["selected_count"]),
        expected_ranking_sha256=str(v1_ranking["sha256"]),
        checkpoint_sha256=str(checkpoint["sha256"]),
    )
    v1_gate = validate_gate(
        v1_ct_evaluation_path,
        gate_sha256=str(gate["sha256"]),
    )

    delta_results: dict[int, dict[str, Any]] = {}
    delta_gate: dict[str, Any] | None = None
    if delta_count:
        if delta_robust_receipt_path is None or delta_ct_evaluation_path is None:
            raise RuntimeError("delta robust and CT artifacts are required")
        _, delta_results = validate_robust(
            delta_robust_receipt_path,
            expected_count=delta_count,
            expected_ranking_sha256=str(delta_ranking["sha256"]),
            checkpoint_sha256=str(checkpoint["sha256"]),
        )
        delta_gate = validate_gate(
            delta_ct_evaluation_path,
            gate_sha256=str(gate["sha256"]),
        )
    elif delta_robust_receipt_path is not None or delta_ct_evaluation_path is not None:
        raise RuntimeError("unexpected delta artifacts for an empty delta")

    merged_entries: list[dict[str, Any]] = []
    seen_v2_ranks: set[int] = set()
    for entry in entries:
        v2_rank = int(entry.get("v2_global_rank", -1))
        if v2_rank in seen_v2_ranks:
            raise RuntimeError(f"duplicate v2 rank in freeze: {v2_rank}")
        seen_v2_ranks.add(v2_rank)
        provenance = entry.get("provenance")
        if provenance == "REUSED_V1_EXACT_WINDOW":
            source_rank = int(entry["v1_global_rank"])
            result = v1_results[source_rank]
            source_receipt = v1_robust_receipt_path
            source_gate = v1_ct_evaluation_path
            gate_value = v1_gate
        elif provenance == "COMPUTE_V2_EXACT_SET_DIFFERENCE":
            source_rank = int(entry["delta_global_rank"])
            result = delta_results[source_rank]
            assert delta_robust_receipt_path is not None
            assert delta_ct_evaluation_path is not None
            assert delta_gate is not None
            source_receipt = delta_robust_receipt_path
            source_gate = delta_ct_evaluation_path
            gate_value = delta_gate
        else:
            raise RuntimeError(f"unknown provenance: {provenance!r}")
        if result_identity(result) != entry["identity"]:
            raise RuntimeError(f"merged identity mismatch at v2 rank {v2_rank}")
        merged_entries.append(
            {
                **entry,
                "source_robust_global_rank": source_rank,
                "source_robust_receipt": str(source_receipt),
                "source_robust_receipt_sha256": sha256_file(source_receipt),
                "source_analysis": result["analysis"],
                "source_analysis_sha256": result["analysis_sha256"],
                "robust_screening_outcome": result.get("screening_outcome"),
                "robust_route": result.get("route"),
                "glyph_like_candidate_count": result.get(
                    "glyph_like_candidate_count"
                ),
                "row_band_count": result.get("row_band_count"),
                "source_ct_gate_evaluation": str(source_gate),
                "source_ct_gate_evaluation_sha256": sha256_file(source_gate),
                "ct_retained_component_count_batch": int(
                    gate_value["retained_count"]
                ),
                "ct_downranked_component_count_batch": int(
                    gate_value["downranked_count"]
                ),
            }
        )
    if seen_v2_ranks != set(range(1, expected_v2 + 1)):
        raise RuntimeError("merged v2 ranks are not complete and contiguous")

    return {
        "kind": MERGE_KIND,
        "status": "COMPLETED_MERGED_DIAGNOSTIC_ONLY",
        "freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "summary": {
            "v2_selected_count": expected_v2,
            "reused_v1_exact_count": reused_count,
            "computed_v2_delta_count": delta_count,
            "merged_entry_count": len(merged_entries),
        },
        "entries": merged_entries,
        "policy": [
            "every v2 selection has exactly one robust-evidence provenance",
            "v1 reuse is restricted to exact physical window identity",
            "CT counts are batch context and not per-window ink decisions",
            "near-overlap windows remain independent evidence",
            "no merged entry automatically accepts ink or letters",
        ],
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not a First Letters submission claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--v1-robust-receipt", type=Path, required=True)
    parser.add_argument("--v1-ct-evaluation", type=Path, required=True)
    parser.add_argument("--delta-robust-receipt", type=Path)
    parser.add_argument("--delta-ct-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        freeze_path=args.freeze.resolve(),
        v1_robust_receipt_path=args.v1_robust_receipt.resolve(),
        v1_ct_evaluation_path=args.v1_ct_evaluation.resolve(),
        delta_robust_receipt_path=(
            args.delta_robust_receipt.resolve()
            if args.delta_robust_receipt
            else None
        ),
        delta_ct_evaluation_path=(
            args.delta_ct_evaluation.resolve()
            if args.delta_ct_evaluation
            else None
        ),
    )
    write_exact(args.output.resolve(), manifest, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "status": (
                    "DRY_RUN_VALIDATED"
                    if args.dry_run
                    else manifest["status"]
                ),
                **manifest["summary"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
