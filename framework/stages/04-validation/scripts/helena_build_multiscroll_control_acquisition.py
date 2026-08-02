#!/usr/bin/env python3
"""Build a deterministic, label-blind control-acquisition queue.

Official prediction layers are used only to choose public regions to inspect.
They never populate expected_class or become benchmark ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SCROLLS = ("PHerc1667", "PHercParis4", "PHerc0814")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _origins_for_type(segment: dict[str, Any], data_type: str) -> list[str]:
    paths: list[str] = []
    for item in segment.get("data", []):
        if item.get("type") != data_type:
            continue
        for origin in item.get("origins", []):
            path = str(origin.get("path", ""))
            if path:
                paths.append(path)
    return sorted(set(paths))


def _stable_order(scroll_id: str, segment_id: str) -> str:
    return hashlib.sha256(f"{scroll_id}:{segment_id}".encode()).hexdigest()


def build_queue(
    metadata: dict[str, Any],
    *,
    scrolls: tuple[str, ...],
    regions_per_scroll: int,
    metadata_sha256: str,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    inventory: dict[str, Any] = {}
    for scroll_id in scrolls:
        scroll = metadata.get("samples", {}).get(scroll_id)
        if not isinstance(scroll, dict):
            inventory[scroll_id] = {"status": "MISSING_FROM_METADATA"}
            continue
        candidates: list[tuple[str, dict[str, Any], list[str], list[str]]] = []
        for segment_id, segment in scroll.get("segments", {}).items():
            detections = _origins_for_type(segment, "ink-detection")
            previews = _origins_for_type(segment, "ink-detection-downsampled")
            if detections and previews:
                candidates.append((str(segment_id), segment, detections, previews))
        candidates.sort(key=lambda row: _stable_order(scroll_id, row[0]))
        selected = candidates[:regions_per_scroll]
        inventory[scroll_id] = {
            "public_segment_count": len(scroll.get("segments", {})),
            "segments_with_prediction_and_preview": len(candidates),
            "selected_region_count": len(selected),
        }
        for index, (segment_id, segment, detections, previews) in enumerate(
            selected, start=1
        ):
            tasks.append(
                {
                    "task_id": (
                        f"MULTISCROLL_TRANSFER_V1:{scroll_id}:"
                        f"{index:02d}:{segment_id}"
                    ),
                    "scroll_id": scroll_id,
                    "segment_id": segment_id,
                    "surface_group_id": f"{scroll_id}:{segment_id}",
                    "status": "PENDING_INDEPENDENT_CT_ADJUDICATION",
                    "selection": {
                        "method": "SHA256_ORDER_OVER_PUBLIC_SEGMENTS_WITH_PREDICTIONS",
                        "selection_key": _stable_order(scroll_id, segment_id),
                        "label_blind": True,
                        "prediction_used_only_as_candidate_locator": True,
                    },
                    "surface": {
                        "long_id": segment.get("long_id"),
                        "original_volume_id": segment.get("original_volume_id"),
                        "width": segment.get("properties", {}).get("width"),
                        "height": segment.get("properties", {}).get("height"),
                    },
                    "candidate_locator_assets": {
                        "ink_detection_paths": detections,
                        "downsampled_preview_paths": previews,
                    },
                    "required_work": [
                        "sample proposed high-response and low-response locations without assigning labels",
                        "render registered orthogonal CT at every proposed location",
                        "adjudicate positive ink or named confound from CT/expert evidence without viewing gate outcomes",
                        "record CT xyz, voxel size, slice order, scanner domain and source hashes",
                        "continue until at least 50 positives and 50 confounds exist across at least five surface groups",
                    ],
                    "prohibited": [
                        "copying a model prediction into expected_class",
                        "viewing v3 or v4 decisions before labels are frozen",
                        "changing the selected regions after observing gate results",
                    ],
                }
            )
    return {
        "schema": "campaignx.multiscroll_control_acquisition_queue.v1",
        "benchmark_id": "MULTISCROLL_TRANSFER_V1",
        "status": "REGIONS_SELECTED_LABELS_NOT_YET_CERTIFIED",
        "generated_at_utc": utc_now(),
        "source_metadata_sha256": metadata_sha256,
        "selection_policy": {
            "scrolls": list(scrolls),
            "regions_per_scroll": regions_per_scroll,
            "method": "SHA256_ORDER_OVER_PUBLIC_SEGMENTS_WITH_PREDICTIONS",
            "reroll_allowed": False,
        },
        "inventory": inventory,
        "tasks": tasks,
        "task_count": len(tasks),
        "non_claims": [
            "selected regions are not positive controls",
            "model predictions are candidate locators only",
            "no expected class exists until independent adjudication is frozen",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regions-per-scroll", type=int, default=5)
    parser.add_argument("--scroll", action="append", dest="scrolls")
    args = parser.parse_args()
    if args.regions_per_scroll < 5:
        raise ValueError("at least five regions per scroll are required")
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite acquisition queue")
    metadata_path = args.metadata.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    queue = build_queue(
        metadata,
        scrolls=tuple(args.scrolls or DEFAULT_SCROLLS),
        regions_per_scroll=args.regions_per_scroll,
        metadata_sha256=sha256(metadata_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": queue["status"], "task_count": queue["task_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
