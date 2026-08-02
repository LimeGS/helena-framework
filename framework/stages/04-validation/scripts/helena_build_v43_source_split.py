#!/usr/bin/env python3
"""Freeze the official-label development/V4 source split for CT router v4.3.

The split is made from source identity only.  It never imports a router,
prediction, score, or prior transfer result:

* development adds PHerc0139 and two PHerc0841 surfaces;
* MULTISCROLL_TRANSFER_V4 reserves PHerc0009B, PHerc0500P2 and PHercMAN5;
* PHerc0841/w00 is reserved for a later benchmark and is not consumed here.

Spatial partitions provide enough label-selection strata on a large surface,
but v4.3 model validation groups by the complete official surface identity,
not by these partitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from helena_build_official_ink_label_plan import (
    BUCKET_ID,
    canonical_sha256,
    list_bucket,
    resolve_surface,
    utc_now,
)


DEVELOPMENT_SOURCES = {
    "PHerc0139": ("ink/0139", None, 1),
    "PHerc0841": ("ink/841", "auto_grown_", 3),
}
V4_SOURCES = {
    "PHerc0009B": ("ink/0009b", None, 5),
    "PHerc0500P2": ("ink/0500p2", None, 5),
    "PHercMAN5": ("ink/man5", None, 5),
}


def build_split_plan(
    benchmark_id: str,
    source_spec: dict[str, tuple[str, str | None, int]],
) -> dict[str, Any]:
    surfaces: list[dict[str, Any]] = []
    for scroll_id, (prefix, required_prefix, partitions) in sorted(
        source_spec.items()
    ):
        directories = sorted(
            str(entry["path"])
            for entry in list_bucket(prefix)
            if entry.get("type") == "directory"
            and (
                required_prefix is None
                or Path(str(entry["path"])).name.startswith(required_prefix)
            )
        )
        for directory in directories:
            surface = resolve_surface(
                scroll_id,
                directory,
                list_bucket(directory),
            )
            surface["region_partition_count"] = partitions
            surface["model_validation_group"] = (
                f"{scroll_id}:{surface['official_surface_id']}"
            )
            surfaces.append(surface)

    by_scroll: dict[str, dict[str, Any]] = {}
    for scroll_id in sorted(source_spec):
        rows = [row for row in surfaces if row["scroll_id"] == scroll_id]
        ready = [row for row in rows if row["status"] == "READY"]
        by_scroll[scroll_id] = {
            "source_surface_count": len(rows),
            "ready_surface_count": len(ready),
            "prospective_region_group_count": sum(
                int(row["region_partition_count"]) for row in ready
            ),
            "complete_surface_validation_group_count": len(ready),
            "meets_selection_minimum_five_groups": sum(
                int(row["region_partition_count"]) for row in ready
            )
            >= 5,
        }
    status = (
        "OFFICIAL_LABEL_SOURCES_READY"
        if surfaces
        and all(
            row["ready_surface_count"] > 0
            and row["meets_selection_minimum_five_groups"]
            for row in by_scroll.values()
        )
        else "BLOCKED_INSUFFICIENT_ALIGNED_GROUPS"
    )
    payload: dict[str, Any] = {
        "schema": "campaignx.v43_official_source_split.v1",
        "benchmark_id": benchmark_id,
        "status": status,
        "generated_at_utc": utc_now(),
        "bucket": f"hf://buckets/{BUCKET_ID}/",
        "split_policy": {
            "source_identity_only": True,
            "router_outputs_visible": False,
            "predictions_are_ground_truth": False,
            "model_validation_unit": "COMPLETE_OFFICIAL_SURFACE",
            "region_partitions_are_selection_strata_only": True,
        },
        "by_scroll": by_scroll,
        "surfaces": surfaces,
        "non_claims": [
            "No v3, v4.1, v4.2, or v4.3 result influenced this split.",
            "Predictions are not labels.",
            "No ink, text, letters, or First Letters are accepted automatically.",
        ],
    }
    payload["content_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    return payload


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite source split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--v4-output", type=Path, required=True)
    args = parser.parse_args()
    development = build_split_plan(
        "CT_PRIORITY_ROUTER_V43_DEVELOPMENT",
        DEVELOPMENT_SOURCES,
    )
    v4 = build_split_plan("MULTISCROLL_TRANSFER_V4", V4_SOURCES)
    _write_new(args.development_output.resolve(), development)
    _write_new(args.v4_output.resolve(), v4)
    print(
        json.dumps(
            {
                "development": development["by_scroll"],
                "v4": v4["by_scroll"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (
        0
        if development["status"] == v4["status"] == "OFFICIAL_LABEL_SOURCES_READY"
        else 2
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
