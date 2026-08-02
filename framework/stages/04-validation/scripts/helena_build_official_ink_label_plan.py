#!/usr/bin/env python3
"""Build a fail-closed, coordinate-aligned public ink-label source plan.

The Scroll Prize public ``scrollprize/datasets`` bucket contains curated
surface-conditioned CT, ink labels, supervision masks and TIFXYZ coordinates.
This program inventories those assets without downloading the large arrays.
It deliberately ignores every ``preds`` directory: predictions can locate
examples, but they are not benchmark truth.

The output is a prospective input plan for ``MULTISCROLL_TRANSFER_V1``.  A
source is READY only when the same directory contains:

* a surface CT Zarr;
* an ink-label TIFF;
* a supervision-mask TIFF;
* x/y/z TIFXYZ maps;
* metadata tying the assets to one surface.

PHerc0814 currently has one curated surface.  It is split prospectively into
five non-overlapping spatial regions, as permitted by the benchmark contract;
the other domains use one independent group per curated surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


BENCHMARK_ID = "MULTISCROLL_TRANSFER_V1"
BUCKET_ID = "scrollprize/datasets"
DEFAULT_SOURCES = {
    "PHerc1667": "ink/1667",
    "PHercParis4": "ink/phercparis4",
    "PHerc0814": "ink/814",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def list_bucket(path: str) -> list[dict[str, Any]]:
    command = [
        "hf",
        "buckets",
        "list",
        f"{BUCKET_ID}/{path}",
        "--json",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected Hugging Face listing for {path}")
    return payload


def _file(
    entries: list[dict[str, Any]],
    *,
    exact_name: str | None = None,
    suffixes: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    files = [entry for entry in entries if entry.get("type") == "file"]
    if exact_name is not None:
        matches = [
            entry for entry in files if Path(str(entry["path"])).name == exact_name
        ]
    else:
        matches = [
            entry
            for entry in files
            if Path(str(entry["path"])).name.endswith(suffixes)
        ]
    return sorted(matches, key=lambda entry: str(entry["path"]))[-1] if matches else None


def _directory(
    entries: list[dict[str, Any]],
    *,
    allowed_suffix: str,
    prohibited_tokens: tuple[str, ...],
) -> dict[str, Any] | None:
    matches = [
        entry
        for entry in entries
        if entry.get("type") == "directory"
        and str(entry["path"]).endswith(allowed_suffix)
        and not any(token in str(entry["path"]) for token in prohibited_tokens)
    ]
    return sorted(matches, key=lambda entry: str(entry["path"]))[0] if matches else None


def _asset(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    payload = {
        "uri": f"hf://buckets/{BUCKET_ID}/{entry['path']}",
        "path": str(entry["path"]),
        "kind": str(entry["type"]),
    }
    if entry.get("size") is not None:
        payload["size_bytes"] = int(entry["size"])
    if entry.get("xet_hash"):
        payload["xet_hash"] = str(entry["xet_hash"])
    return payload


def resolve_surface(
    scroll_id: str,
    directory: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    label = _file(
        entries,
        suffixes=("_inklabels_v2.tif", "_inklabels.tif"),
    )
    supervision = _file(
        entries,
        suffixes=("_supervision_mask_v2.tif", "_supervision_mask.tif"),
    )
    surface = _directory(
        entries,
        allowed_suffix=".zarr",
        prohibited_tokens=(
            "inklabels",
            "supervision",
            "validation",
            "max_",
            "/preds",
        ),
    )
    assets = {
        "metadata": _asset(_file(entries, exact_name="meta.json")),
        "ink_label": _asset(label),
        "supervision_mask": _asset(supervision),
        "surface_ct": _asset(surface),
        "tifxyz_x": _asset(_file(entries, exact_name="x.tif")),
        "tifxyz_y": _asset(_file(entries, exact_name="y.tif")),
        "tifxyz_z": _asset(_file(entries, exact_name="z.tif")),
    }
    missing = sorted(name for name, asset in assets.items() if asset is None)
    name = Path(directory).name
    region_partitions = 5 if scroll_id == "PHerc0814" else 1
    return {
        "scroll_id": scroll_id,
        "source_directory": f"hf://buckets/{BUCKET_ID}/{directory}/",
        "official_surface_id": name,
        "label_authority": "PUBLIC_CURATED_SURFACE_LABEL",
        "prediction_assets_used_as_truth": False,
        "region_partition_count": region_partitions,
        "status": "READY" if not missing else "BLOCKED_MISSING_ALIGNED_ASSETS",
        "missing_assets": missing,
        "assets": assets,
    }


def build_plan(
    *,
    lister: Callable[[str], list[dict[str, Any]]] = list_bucket,
    sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    sources = sources or DEFAULT_SOURCES
    surfaces: list[dict[str, Any]] = []
    for scroll_id, prefix in sorted(sources.items()):
        directories = sorted(
            str(entry["path"])
            for entry in lister(prefix)
            if entry.get("type") == "directory"
        )
        for directory in directories:
            surfaces.append(resolve_surface(scroll_id, directory, lister(directory)))

    by_scroll: dict[str, dict[str, Any]] = {}
    for scroll_id in sorted(sources):
        selected = [row for row in surfaces if row["scroll_id"] == scroll_id]
        ready = [row for row in selected if row["status"] == "READY"]
        groups = sum(int(row["region_partition_count"]) for row in ready)
        by_scroll[scroll_id] = {
            "source_surface_count": len(selected),
            "ready_surface_count": len(ready),
            "prospective_region_group_count": groups,
            "meets_minimum_five_groups": groups >= 5,
        }
    status = (
        "OFFICIAL_LABEL_SOURCES_READY"
        if all(row["meets_minimum_five_groups"] for row in by_scroll.values())
        else "BLOCKED_INSUFFICIENT_ALIGNED_GROUPS"
    )
    plan = {
        "schema": "campaignx.multiscroll_official_ink_label_plan.v1",
        "benchmark_id": BENCHMARK_ID,
        "status": status,
        "generated_at_utc": utc_now(),
        "bucket": f"hf://buckets/{BUCKET_ID}/",
        "label_policy": {
            "authority": "PUBLIC_CURATED_SURFACE_LABEL",
            "predictions_are_ground_truth": False,
            "positive_definition": (
                "ink_label > 0 inside the official supervision mask"
            ),
            "confound_definition": (
                "ink_label == 0 inside the official supervision mask; hard "
                "confounds are selected without reading v3/v4 outcomes"
            ),
            "minimum_groups_per_scroll": 5,
        },
        "by_scroll": by_scroll,
        "surfaces": surfaces,
        "non_claims": [
            "This inventory does not yet select benchmark components.",
            "No prediction is treated as a label.",
            "No v3/v4 output was read.",
            "No ink, text, letters, or First Letters are accepted automatically.",
        ],
    }
    plan["content_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "content_sha256"}
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite official label source plan")
    plan = build_plan()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if plan["status"] == "OFFICIAL_LABEL_SOURCES_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
