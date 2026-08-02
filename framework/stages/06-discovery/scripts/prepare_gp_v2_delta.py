#!/usr/bin/env python3
"""Freeze the exact-compute delta between GP Scroll1 v1 and v2 rankings.

The v2 ranking is additive.  Windows that are byte-for-byte identical in
physical source coordinates reuse the already completed v1 robust result.
Only exact set-difference windows are adapted to the canonical ranking schema
consumed by ``run_expanded_robust_windows.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


V1_RANKING_KIND = "campaign_x_phase4_expanded_surface_global_ranking_v1"
V2_RANKING_KIND = "campaign_x_phase4_expanded_surface_global_ranking_v2"
ROBUST_KIND = "campaign_x_phase4_expanded_window_robust_batch_v1"
COMPLETE_ROBUST_STATUSES = {
    "COMPLETED_WITH_RAW_CT_REVIEW_QUEUE",
    "COMPLETED_DIAGNOSTIC_ONLY",
}
DELTA_KIND = "campaign_x_phase4_gp_scroll1_v2_delta_ranking_v1"
FREEZE_KIND = "campaign_x_phase4_gp_scroll1_v2_delta_freeze_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def crop_tuple(row: dict[str, Any]) -> tuple[int, int, int, int]:
    raw = row.get("source_crop_xyxy")
    if not isinstance(raw, list) or len(raw) != 4:
        raise RuntimeError("window source_crop_xyxy must contain four integers")
    values = tuple(int(value) for value in raw)
    if list(values) != raw:
        raise RuntimeError(f"non-integral source crop: {raw!r}")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid source crop: {raw!r}")
    return values


def identity(row: dict[str, Any]) -> tuple[str, str, tuple[int, int, int, int]]:
    sample_id = str(row.get("sample_id", ""))
    surface_id = str(row.get("surface_id", ""))
    if not sample_id or not surface_id:
        raise RuntimeError("window identity requires sample_id and surface_id")
    return sample_id, surface_id, crop_tuple(row)


def identity_json(
    key: tuple[str, str, tuple[int, int, int, int]]
) -> dict[str, Any]:
    return {
        "sample_id": key[0],
        "surface_id": key[1],
        "source_crop_xyxy": list(key[2]),
    }


def validate_ranked_rows(
    rows: Any,
    *,
    rank_key: str,
    expected_count: int,
    label: str,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]]]:
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError(
            f"{label} must contain exactly {expected_count} windows"
        )
    expected_ranks = list(range(1, expected_count + 1))
    actual_ranks = [int(row.get(rank_key, -1)) for row in rows]
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            f"{label} ranks are not contiguous: {actual_ranks!r}"
        )
    by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = identity(row)
        if key in by_identity:
            raise RuntimeError(f"duplicate exact window in {label}: {key!r}")
        by_identity[key] = row
    return rows, by_identity


def validate_v1_robust_receipt(
    receipt: dict[str, Any],
    *,
    receipt_path: Path,
    v1_ranking_path: Path,
    v1_rows: list[dict[str, Any]],
    checkpoint_path: Path,
) -> dict[int, dict[str, Any]]:
    if receipt.get("kind") != ROBUST_KIND:
        raise RuntimeError("unexpected v1 robust receipt kind")
    if receipt.get("status") not in COMPLETE_ROBUST_STATUSES:
        raise RuntimeError("v1 robust receipt is not complete")
    expected_count = len(v1_rows)
    if int(receipt.get("completed_count", -1)) != expected_count:
        raise RuntimeError("v1 robust receipt completed_count mismatch")
    expected_ranks = list(range(1, expected_count + 1))
    if receipt.get("selected_global_ranks") != expected_ranks:
        raise RuntimeError("v1 robust receipt rank list mismatch")
    if receipt.get("global_ranking_sha256") != sha256_file(v1_ranking_path):
        raise RuntimeError("v1 robust receipt ranking hash mismatch")
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("v1 robust receipt checkpoint missing")
    if checkpoint.get("sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("v1 robust receipt checkpoint hash mismatch")
    results = receipt.get("results")
    if not isinstance(results, list) or len(results) != expected_count:
        raise RuntimeError("v1 robust result count mismatch")
    by_rank: dict[int, dict[str, Any]] = {}
    for expected_row, result in zip(v1_rows, results, strict=True):
        rank = int(result.get("global_rank", -1))
        if rank in by_rank:
            raise RuntimeError(f"duplicate robust result rank: {rank}")
        if rank != int(expected_row["global_rank"]):
            raise RuntimeError("v1 robust result ordering mismatch")
        if identity(result) != identity(expected_row):
            raise RuntimeError(
                f"v1 robust result identity mismatch at rank {rank}"
            )
        analysis = Path(str(result.get("analysis", "")))
        if not analysis.is_file():
            raise FileNotFoundError(
                f"v1 robust analysis missing at rank {rank}: {analysis}"
            )
        if result.get("analysis_sha256") != sha256_file(analysis):
            raise RuntimeError(
                f"v1 robust analysis hash mismatch at rank {rank}"
            )
        by_rank[rank] = result
    # Bind the path too, so a copied-but-renamed receipt is explicit.
    require_file(receipt_path, "v1 robust receipt")
    return by_rank


def prepare_artifacts(
    *,
    coarse_receipt_path: Path,
    v1_ranking_path: Path,
    v2_ranking_path: Path,
    v1_robust_receipt_path: Path,
    checkpoint_path: Path,
    gate_freeze_path: Path,
    delta_ranking_path: Path,
    freeze_path: Path,
    expected_selected_count: int | None,
    script_root: Path,
    maximum_selected_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path, label in (
        (coarse_receipt_path, "coarse receipt"),
        (v1_ranking_path, "v1 ranking"),
        (v2_ranking_path, "v2 ranking"),
        (v1_robust_receipt_path, "v1 robust receipt"),
        (checkpoint_path, "checkpoint"),
        (gate_freeze_path, "gate freeze"),
    ):
        require_file(path, label)
    if expected_selected_count is not None and expected_selected_count < 1:
        raise ValueError("expected_selected_count must be positive")
    if maximum_selected_count is not None and maximum_selected_count < 1:
        raise ValueError("maximum_selected_count must be positive")
    if expected_selected_count is None and maximum_selected_count is None:
        raise ValueError("an exact or maximum selected count is required")

    coarse_sha = sha256_file(coarse_receipt_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    gate_sha = sha256_file(gate_freeze_path)
    v1_sha = sha256_file(v1_ranking_path)
    v2_sha = sha256_file(v2_ranking_path)

    coarse = read_json(coarse_receipt_path)
    if coarse.get("status") != "COMPLETED_PRIORITIZATION_ONLY":
        raise RuntimeError("coarse receipt is not complete")
    if int(coarse.get("completed_count", -1)) != int(
        coarse.get("task_count", -2)
    ):
        raise RuntimeError("coarse receipt count mismatch")
    if int(coarse.get("failed_count", -1)) != 0:
        raise RuntimeError("coarse receipt contains failed tasks")

    v1 = read_json(v1_ranking_path)
    if v1.get("kind") != V1_RANKING_KIND:
        raise RuntimeError("unexpected v1 ranking kind")
    if v1.get("status") != "COMPLETED_PRIORITIZATION_ONLY":
        raise RuntimeError("v1 ranking is not complete")
    if v1.get("source_batch_receipt_sha256") != coarse_sha:
        raise RuntimeError("v1 ranking coarse-receipt hash mismatch")
    raw_v1_rows = v1.get("global_priority")
    if not isinstance(raw_v1_rows, list) or not raw_v1_rows:
        raise RuntimeError("v1 ranking selected no windows")
    if (
        expected_selected_count is not None
        and len(raw_v1_rows) != expected_selected_count
    ):
        raise RuntimeError("v1 ranking selected count mismatch")
    if (
        maximum_selected_count is not None
        and len(raw_v1_rows) > maximum_selected_count
    ):
        raise RuntimeError("v1 ranking exceeds frozen maximum")
    v1_rows, v1_by_identity = validate_ranked_rows(
        raw_v1_rows,
        rank_key="global_rank",
        expected_count=len(raw_v1_rows),
        label="v1 ranking",
    )

    v2 = read_json(v2_ranking_path)
    if v2.get("kind") != V2_RANKING_KIND:
        raise RuntimeError("unexpected v2 ranking kind")
    if v2.get("status") != "COMPLETED_PRIORITIZATION_ONLY":
        raise RuntimeError("v2 ranking is not complete")
    if v2.get("source_batch_receipt_sha256") != coarse_sha:
        raise RuntimeError("v2 ranking coarse-receipt hash mismatch")
    binding = v2.get("frozen_v1_audit_binding")
    if not isinstance(binding, dict) or binding.get("sha256") != v1_sha:
        raise RuntimeError("v2 ranking does not bind the exact v1 ranking")
    raw_v2_rows = v2.get("global_priority_v2")
    if not isinstance(raw_v2_rows, list) or not raw_v2_rows:
        raise RuntimeError("v2 ranking selected no windows")
    if (
        expected_selected_count is not None
        and len(raw_v2_rows) != expected_selected_count
    ):
        raise RuntimeError("v2 ranking selected count mismatch")
    if (
        maximum_selected_count is not None
        and len(raw_v2_rows) > maximum_selected_count
    ):
        raise RuntimeError("v2 ranking exceeds frozen maximum")
    v2_rows, _ = validate_ranked_rows(
        raw_v2_rows,
        rank_key="global_rank_v2",
        expected_count=len(raw_v2_rows),
        label="v2 ranking",
    )

    v1_robust = read_json(v1_robust_receipt_path)
    v1_results = validate_v1_robust_receipt(
        v1_robust,
        receipt_path=v1_robust_receipt_path,
        v1_ranking_path=v1_ranking_path,
        v1_rows=v1_rows,
        checkpoint_path=checkpoint_path,
    )

    delta_rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for v2_row in v2_rows:
        key = identity(v2_row)
        v2_rank = int(v2_row["global_rank_v2"])
        base_entry: dict[str, Any] = {
            "v2_global_rank": v2_rank,
            "identity": identity_json(key),
            "v2_selection_lane": v2_row.get("global_selection_lane_v2"),
            "v2_global_score": float(v2_row["global_score_v2"]),
        }
        if key in v1_by_identity:
            v1_row = v1_by_identity[key]
            v1_rank = int(v1_row["global_rank"])
            result = v1_results[v1_rank]
            entries.append(
                {
                    **base_entry,
                    "provenance": "REUSED_V1_EXACT_WINDOW",
                    "v1_global_rank": v1_rank,
                    "v1_analysis": result["analysis"],
                    "v1_analysis_sha256": result["analysis_sha256"],
                }
            )
            continue

        delta_rank = len(delta_rows) + 1
        adapted = dict(v2_row)
        adapted["global_rank"] = delta_rank
        adapted["v2_global_rank"] = v2_rank
        adapted["score"] = float(v2_row["global_score_v2"])
        adapted["delta_provenance"] = "COMPUTE_V2_EXACT_SET_DIFFERENCE"
        delta_rows.append(adapted)
        entries.append(
            {
                **base_entry,
                "provenance": "COMPUTE_V2_EXACT_SET_DIFFERENCE",
                "delta_global_rank": delta_rank,
            }
        )

    delta_ranking: dict[str, Any] = {
        "kind": DELTA_KIND,
        "status": "COMPLETED_PRIORITIZATION_ONLY",
        "source_batch_receipt": str(coarse_receipt_path),
        "source_batch_receipt_sha256": coarse_sha,
        "source_v1_ranking": str(v1_ranking_path),
        "source_v1_ranking_sha256": v1_sha,
        "source_v2_ranking": str(v2_ranking_path),
        "source_v2_ranking_sha256": v2_sha,
        "selection": {
            "v2_selected_count": len(v2_rows),
            "reused_v1_exact_count": len(v2_rows) - len(delta_rows),
            "delta_compute_count": len(delta_rows),
            "identity_fields": [
                "sample_id",
                "surface_id",
                "source_crop_xyxy",
            ],
        },
        "global_priority": delta_rows,
        "policy": [
            "only byte-exact source-crop identities reuse v1 robust evidence",
            "near overlaps are not deduplicated",
            "delta ranks are contiguous solely for the canonical robust runner",
            "v2_global_rank remains attached to every delta row",
            "no ranking output accepts ink, letters, or a First Letters claim",
        ],
    }
    delta_bytes = canonical_bytes(delta_ranking)

    script_names = (
        "prepare_gp_v2_delta.py",
        "merge_gp_v2_manifest.py",
        "validate_gp_v1_completion.py",
        "run_gp_v2_delta_after_v1.sh",
        "rank_coarse_ink_windows_v2.py",
        "rank_expanded_candidate_windows_v2.py",
        "run_expanded_robust_windows.py",
        "postprocess_robust_batch.sh",
    )
    script_hashes: dict[str, str] = {}
    for name in script_names:
        candidates = [script_root / name]
        candidates.extend(
            script_root / "framework" / "stages" / stage / "scripts" / name
            for stage in (
                "01-segmentation",
                "02-flattening",
                "03-ink",
                "04-validation",
                "05-reconstruction",
                "06-discovery",
            )
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        require_file(path, f"pipeline script {name}")
        script_hashes[name] = sha256_file(path)

    freeze: dict[str, Any] = {
        "kind": FREEZE_KIND,
        "status": "FROZEN_BEFORE_V2_DELTA_EXECUTION",
        "coarse_receipt": {
            "path": str(coarse_receipt_path),
            "sha256": coarse_sha,
        },
        "v1_ranking": {
            "path": str(v1_ranking_path),
            "sha256": v1_sha,
            "selected_count": len(v1_rows),
        },
        "v2_ranking": {
            "path": str(v2_ranking_path),
            "sha256": v2_sha,
            "selected_count": len(v2_rows),
        },
        "v1_robust_receipt": {
            "path": str(v1_robust_receipt_path),
            "sha256": sha256_file(v1_robust_receipt_path),
            "completed_count": len(v1_results),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
        },
        "ct_fiber_gate": {
            "path": str(gate_freeze_path),
            "sha256": gate_sha,
        },
        "delta_ranking": {
            "path": str(delta_ranking_path),
            "sha256": sha256_bytes(delta_bytes),
            "selected_count": len(delta_rows),
        },
        "selection": {
            "expected_v2_count": len(v2_rows),
            "frozen_maximum_selected_count": maximum_selected_count,
            "reused_v1_exact_count": len(v2_rows) - len(delta_rows),
            "delta_compute_count": len(delta_rows),
            "entries": entries,
        },
        "script_hashes": script_hashes,
        "policy": [
            "v1 evidence is immutable and reused only for exact identities",
            "v2 delta output is additive and uses isolated namespaces",
            (
                "a dynamic free-space guard is mandatory before robust delta "
                "compute: max(10 GiB, 8 GiB + 0.5 GiB per delta window)"
            ),
            "no files are deleted to satisfy the free-space guard",
            "the merged result remains diagnostic and requires raw-CT review",
        ],
        "explicit_non_claims": [
            "not automatic ink acceptance",
            "not automatic letter acceptance",
            "not a First Letters submission claim",
        ],
    }
    return delta_ranking, freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-receipt", type=Path, required=True)
    parser.add_argument("--v1-ranking", type=Path, required=True)
    parser.add_argument("--v2-ranking", type=Path, required=True)
    parser.add_argument("--v1-robust-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-freeze", type=Path, required=True)
    parser.add_argument("--delta-ranking-output", type=Path, required=True)
    parser.add_argument("--freeze-output", type=Path, required=True)
    parser.add_argument("--expected-selected-count", type=int)
    parser.add_argument("--maximum-selected-count", type=int, default=48)
    parser.add_argument(
        "--script-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    delta, freeze = prepare_artifacts(
        coarse_receipt_path=args.coarse_receipt.resolve(),
        v1_ranking_path=args.v1_ranking.resolve(),
        v2_ranking_path=args.v2_ranking.resolve(),
        v1_robust_receipt_path=args.v1_robust_receipt.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        gate_freeze_path=args.gate_freeze.resolve(),
        delta_ranking_path=args.delta_ranking_output.resolve(),
        freeze_path=args.freeze_output.resolve(),
        expected_selected_count=args.expected_selected_count,
        script_root=args.script_root.resolve(),
        maximum_selected_count=args.maximum_selected_count,
    )
    write_exact(
        args.delta_ranking_output.resolve(),
        delta,
        dry_run=args.dry_run,
    )
    write_exact(args.freeze_output.resolve(), freeze, dry_run=args.dry_run)
    print(
        json.dumps(
            {
                "status": "DRY_RUN_VALIDATED" if args.dry_run else "FROZEN",
                "v2_selected_count": freeze["selection"]["expected_v2_count"],
                "reused_v1_exact_count": freeze["selection"][
                    "reused_v1_exact_count"
                ],
                "delta_compute_count": freeze["selection"][
                    "delta_compute_count"
                ],
                "delta_ranking": str(args.delta_ranking_output.resolve()),
                "freeze": str(args.freeze_output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
