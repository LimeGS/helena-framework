#!/usr/bin/env python3
"""Build a separate recall-oriented v2 global ranking from coarse maps.

The script waits for no work and performs no inference.  It consumes a
completed coarse batch receipt, invokes the additive per-sample v2 ranker, and
writes only v2-named artifacts.  The frozen v1 ranking may be bound by hash for
audit but is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCALE_CONTRACT_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "04-validation" / "scripts"
)
if str(_SCALE_CONTRACT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCALE_CONTRACT_SCRIPTS))

# FIX-09: the normalized grid pitch is the training pixel size, and it is
# declared by the ink lane profile.  Import the contract that already resolves
# it and fails closed on a disagreement instead of restating the number.
from analyze_ink_stability import (  # noqa: E402
    DEFAULT_INK_PROFILE,
    resolve_training_pixel_um,
)


SCHEMA_VERSION = "campaign_x_phase4_expanded_surface_global_ranking_v2"
GLOBAL_LEGACY_RESCUE_FRACTION = 0.20


def normalized_pixel_um() -> str:
    resolved, _ = resolve_training_pixel_um(
        profile_path=DEFAULT_INK_PROFILE, requested=None
    )
    return str(resolved)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def global_score(window: dict[str, Any]) -> float:
    """Normalize within sample before comparing non-calibrated scroll maps."""

    return float(
        0.70 * float(window["candidate_percentile_v2"])
        + 0.20 * float(window["candidate_percentile_legacy_v1"])
        + 0.10 * float(window["features"]["priority_score_v2"])
    )


def select_global_with_rescue(
    windows: list[dict[str, Any]],
    *,
    limit: int,
    max_per_sample: int,
    rescue_fraction: float = GLOBAL_LEGACY_RESCUE_FRACTION,
) -> list[dict[str, Any]]:
    """Select global windows with scroll diversity and a raw-v1 rescue lane."""

    if limit < 1 or max_per_sample < 1:
        raise ValueError("selection limits must be positive")
    if not 0.0 <= rescue_fraction <= 0.5:
        raise ValueError("rescue_fraction must be between 0 and 0.5")
    for window in windows:
        window["global_score_v2"] = global_score(window)

    primary_target = max(0, limit - math.ceil(limit * rescue_fraction))
    rescue_target = min(limit, math.ceil(limit * rescue_fraction))
    primary_order = sorted(
        windows,
        key=lambda item: (
            -float(item["global_score_v2"]),
            item["sample_id"],
            item["surface_id"],
            item["source_crop_xyxy"],
        ),
    )
    legacy_order = sorted(
        windows,
        key=lambda item: (
            -float(item["candidate_percentile_legacy_v1"]),
            -float(item["features"]["legacy_v1_audit"]["score"]),
            item["sample_id"],
            item["surface_id"],
            item["source_crop_xyxy"],
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    by_sample: dict[str, int] = {}

    def add(window: dict[str, Any], lane: str) -> bool:
        sample_id = str(window["sample_id"])
        if (
            id(window) in selected_ids
            or by_sample.get(sample_id, 0) >= max_per_sample
        ):
            return False
        window["global_selection_lane_v2"] = lane
        selected.append(window)
        selected_ids.add(id(window))
        by_sample[sample_id] = by_sample.get(sample_id, 0) + 1
        return True

    for window in primary_order:
        if len(selected) >= primary_target:
            break
        add(window, "V2_PRIMARY")
    rescue_added = 0
    for window in legacy_order:
        if rescue_added >= rescue_target or len(selected) >= limit:
            break
        if add(window, "LEGACY_V1_RECALL_RESCUE"):
            rescue_added += 1
    for window in primary_order:
        if len(selected) >= limit:
            break
        add(window, "V2_PRIMARY_FILL")

    for rank, window in enumerate(selected, start=1):
        window["global_rank_v2"] = rank
    return selected


def validate_safe_name(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe {label}: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.70)
    parser.add_argument("--global-top-n", type=int, default=48)
    parser.add_argument("--max-per-sample", type=int, default=4)
    parser.add_argument("--global-legacy-rescue-fraction", type=float, default=0.20)
    parser.add_argument("--per-sample-legacy-rescue-fraction", type=float, default=0.25)
    parser.add_argument(
        "--batch-receipt-name",
        default="EXPANDED_SURFACE_BATCH_RECEIPT.json",
    )
    parser.add_argument("--screening-name", default="coarse_screen_v1")
    parser.add_argument(
        "--per-sample-ranking-name",
        default="coarse_ranking_v2",
    )
    parser.add_argument(
        "--ranking-output-name",
        default="GLOBAL_COARSE_WINDOW_RANKING_V2.json",
    )
    parser.add_argument(
        "--v1-global-ranking",
        type=Path,
        help="Optional frozen v1 ranking to bind by hash for side-by-side audit.",
    )
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()

    for value, label in (
        (args.batch_receipt_name, "batch receipt name"),
        (args.screening_name, "screening name"),
        (args.per_sample_ranking_name, "per-sample ranking name"),
        (args.ranking_output_name, "ranking output name"),
    ):
        validate_safe_name(value, label)
    if args.per_sample_ranking_name == "coarse_ranking_v1":
        raise ValueError("v2 refuses to overwrite the v1 per-sample namespace")
    if args.ranking_output_name == "GLOBAL_COARSE_WINDOW_RANKING.json":
        raise ValueError("v2 refuses to overwrite the v1 global ranking")

    root = args.root.resolve()
    batch_root = (
        args.batch_root.resolve()
        if args.batch_root
        else root / "phase4" / "expanded_candidate_surface_screen_v1"
    )
    batch_receipt = batch_root / args.batch_receipt_name
    batch = read_json(batch_receipt)
    if batch.get("status") != "COMPLETED_PRIORITIZATION_ONLY":
        raise RuntimeError("expanded-surface coarse batch is not complete")
    if batch.get("screening_name") not in (None, args.screening_name):
        raise RuntimeError("batch receipt screening namespace mismatch")

    v1_binding: dict[str, Any] | None = None
    if args.v1_global_ranking:
        v1_path = args.v1_global_ranking.resolve()
        if not v1_path.is_file():
            raise RuntimeError(f"v1 global ranking does not exist: {v1_path}")
        v1_receipt = read_json(v1_path)
        if v1_receipt.get("kind") != "campaign_x_phase4_expanded_surface_global_ranking_v1":
            raise RuntimeError("v1 audit binding has unexpected kind")
        v1_binding = {
            "path": str(v1_path),
            "sha256": sha256_file(v1_path),
            "status": v1_receipt.get("status"),
        }

    sample_filter = set(args.sample_id or [])
    windows: list[dict[str, Any]] = []
    ranking_receipts: list[dict[str, Any]] = []
    for sample_root in sorted(batch_root.glob("PHerc*")):
        sample_id = sample_root.name
        if sample_filter and sample_id not in sample_filter:
            continue
        lock_path = root / "phase4" / "targets" / sample_id / "TARGET_LOCK.json"
        lock = read_json(lock_path)
        output = sample_root / args.per_sample_ranking_name
        subprocess.run(
            [
                "python3",
                str(root / "scripts" / "rank_coarse_ink_windows_v2.py"),
                "--sample-id",
                sample_id,
                "--root",
                str(sample_root),
                "--output",
                str(output),
                "--screening-name",
                args.screening_name,
                "--source-pixel-um",
                str(float(lock["voxel_size_um"])),
                "--normalized-pixel-um",
                normalized_pixel_um(),
                "--size-mm",
                "20",
                "--downsample",
                "8",
                "--step-fraction",
                "0.25",
                "--minimum-valid-ratio",
                str(args.minimum_valid_ratio),
                "--top-n",
                "16",
                "--max-per-surface",
                "3",
                "--legacy-rescue-fraction",
                str(args.per_sample_legacy_rescue_fraction),
                "--allow-smaller-fit",
            ],
            check=True,
        )
        receipt_path = output / "COARSE_INK_WINDOW_RANKING_V2.json"
        receipt = read_json(receipt_path)
        if receipt.get("kind") != "campaign_x_phase4_coarse_ink_window_ranking_v2":
            raise RuntimeError(f"unexpected v2 receipt kind: {receipt_path}")
        ranking_receipts.append(
            {
                "sample_id": sample_id,
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "eligible_window_count": receipt["search"]["eligible_window_count"],
                "selected_window_count": len(receipt["ranked_windows"]),
            }
        )
        for window in receipt["ranked_windows"]:
            enriched = dict(window)
            enriched["sample_id"] = sample_id
            enriched["coarse_ranking_v2_receipt"] = str(receipt_path)
            windows.append(enriched)

    selected = select_global_with_rescue(
        windows,
        limit=args.global_top_n,
        max_per_sample=args.max_per_sample,
        rescue_fraction=args.global_legacy_rescue_fraction,
    )
    output_path = batch_root / args.ranking_output_name
    write_json(
        output_path,
        {
            "kind": SCHEMA_VERSION,
            "status": "COMPLETED_PRIORITIZATION_ONLY",
            "generated_at_utc": utc_now(),
            "source_batch_receipt": str(batch_receipt),
            "source_batch_receipt_sha256": sha256_file(batch_receipt),
            "frozen_v1_audit_binding": v1_binding,
            "search": {
                "minimum_valid_ratio": args.minimum_valid_ratio,
                "screening_name": args.screening_name,
                "per_sample_ranking_name": args.per_sample_ranking_name,
                "global_top_n": args.global_top_n,
                "max_per_sample": args.max_per_sample,
                "global_legacy_rescue_fraction": (
                    args.global_legacy_rescue_fraction
                ),
                "per_sample_legacy_rescue_fraction": (
                    args.per_sample_legacy_rescue_fraction
                ),
                "all_ranked_window_count": len(windows),
                "selected_window_count": len(selected),
            },
            "global_score": {
                "within_sample_v2_percentile": 0.70,
                "within_sample_legacy_v1_percentile": 0.20,
                "absolute_v2_score": 0.10,
                "reason": (
                    "within-sample percentiles dominate because coarse model "
                    "probabilities are not calibrated across scrolls"
                ),
            },
            "per_sample_ranking_receipts": ranking_receipts,
            "global_priority_v2": selected,
            "policy": [
                "v2 is a separate additive ranking and does not modify v1",
                "all features and penalty contributions are emitted per window",
                "fiber and saturation penalties are bounded and never hard reject",
                "legacy-v1 recall rescue is reserved per sample and globally",
                "scroll diversity is enforced by max_per_sample",
                "coarse ranking only allocates robust compute",
                "every advanced window requires six-replica and raw-CT review",
            ],
            "explicit_non_claims": [
                "not automatic ink acceptance",
                "not automatic letter acceptance",
                "not a First Letters submission claim",
                "not calibrated probability of ink",
                "not evidence that an unselected window lacks ink",
            ],
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "candidate_count": len(windows),
                "selected_count": len(selected),
                "sample_count": len(ranking_receipts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
