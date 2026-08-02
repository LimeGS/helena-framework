#!/usr/bin/env python3
"""Rank physical windows from the expanded-surface coarse screening batch."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_diverse(
    windows: list[dict[str, Any]], *, limit: int, max_per_sample: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_sample: dict[str, int] = {}
    for window in sorted(
        windows,
        key=lambda item: (
            -float(item["score"]),
            item["sample_id"],
            item["surface_id"],
            item["source_crop_xyxy"],
        ),
    ):
        sample_id = str(window["sample_id"])
        if by_sample.get(sample_id, 0) >= max_per_sample:
            continue
        selected.append(window)
        by_sample[sample_id] = by_sample.get(sample_id, 0) + 1
        if len(selected) >= limit:
            break
    for rank, window in enumerate(selected, start=1):
        window["global_rank"] = rank
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--minimum-valid-ratio", type=float, default=0.70)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--max-per-sample", type=int, default=2)
    parser.add_argument(
        "--batch-receipt-name",
        default="EXPANDED_SURFACE_BATCH_RECEIPT.json",
    )
    parser.add_argument(
        "--screening-name",
        default="coarse_screen_v1",
    )
    parser.add_argument(
        "--per-sample-ranking-name",
        default="coarse_ranking_v1",
    )
    parser.add_argument(
        "--ranking-output-name",
        default="GLOBAL_COARSE_WINDOW_RANKING.json",
    )
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args()
    for value, label in (
        (args.batch_receipt_name, "batch receipt name"),
        (args.screening_name, "screening name"),
        (args.per_sample_ranking_name, "per-sample ranking name"),
        (args.ranking_output_name, "ranking output name"),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"unsafe {label}: {value!r}")

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

    sample_filter = set(args.sample_id or [])
    windows: list[dict[str, Any]] = []
    ranking_receipts: list[dict[str, Any]] = []
    for sample_root in sorted(batch_root.glob("PHerc*")):
        sample_id = sample_root.name
        if sample_filter and sample_id not in sample_filter:
            continue
        lock = read_json(root / "phase4" / "targets" / sample_id / "TARGET_LOCK.json")
        output = sample_root / args.per_sample_ranking_name
        subprocess.run(
            [
                "python3",
                str(root / "scripts" / "rank_coarse_ink_windows.py"),
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
                "12",
                "--max-per-surface",
                "2",
                "--allow-smaller-fit",
            ],
            check=True,
        )
        receipt_path = output / "COARSE_INK_WINDOW_RANKING.json"
        receipt = read_json(receipt_path)
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
            enriched["coarse_ranking_receipt"] = str(receipt_path)
            windows.append(enriched)

    selected = select_diverse(
        windows,
        limit=args.global_top_n,
        max_per_sample=args.max_per_sample,
    )
    write_json(
        batch_root / args.ranking_output_name,
        {
            "kind": "campaign_x_phase4_expanded_surface_global_ranking_v1",
            "status": "COMPLETED_PRIORITIZATION_ONLY",
            "generated_at_utc": utc_now(),
            "source_batch_receipt": str(batch_receipt),
            "source_batch_receipt_sha256": sha256_file(batch_receipt),
            "search": {
                "minimum_valid_ratio": args.minimum_valid_ratio,
                "screening_name": args.screening_name,
                "global_top_n": args.global_top_n,
                "max_per_sample": args.max_per_sample,
                "all_ranked_window_count": len(windows),
                "selected_window_count": len(selected),
            },
            "per_sample_ranking_receipts": ranking_receipts,
            "global_priority": selected,
            "policy": [
                "scores compare physically normalized coarse maps",
                "at most two windows per target scroll enter global priority",
                "coarse ranking only allocates robust compute",
                "every advanced window requires the frozen six-replica screen",
            ],
            "explicit_non_claims": [
                "not automatic ink acceptance",
                "not automatic letter acceptance",
                "not a First Letters submission claim",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
