#!/usr/bin/env python3
"""Compute a transparent PHerc 0139 length estimate from committed geometry.

This is an assumption-explicit model, not a measurement of the complete
scroll. It deliberately reports no mesh area or total surface coverage: those
need a versioned per-mesh area calculation from the source meshes.

Usage:
    python scripts/estimate_extent.py
    python scripts/estimate_extent.py --out /tmp/scroll_extent_estimate.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VX_UM = 2.399
PITCH_MM = 0.17
OUTER_RADII_MM = (25.0, 30.0)
CORE_RADII_MM = (3.0, 5.0)


def wrap_token(segment: str) -> str:
    match = re.search(r"-(w\d+|title)_?", segment)
    if not match:
        raise ValueError(f"cannot parse wrap token from {segment!r}")
    return match.group(1)


def modeled_length_m(outer_mm: float, core_mm: float) -> tuple[int, float]:
    """Arithmetic-spiral approximation using one circumference per wrap."""
    turns = round((outer_mm - core_mm) / PITCH_MM)
    radii = [core_mm + i * PITCH_MM for i in range(turns)]
    return turns, sum(2 * math.pi * radius for radius in radii) / 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data/scroll_extent_estimate.json"))
    args = parser.parse_args()

    radial = json.loads((ROOT / "data/wrap_radial.json").read_text())
    body_radii_mm = [
        stats["r_mean"] * VX_UM / 1000
        for segment, stats in radial["segments"].items()
        if wrap_token(segment) != "title"
    ]
    if len(body_radii_mm) != 37:
        raise ValueError(f"expected 37 body wraps, found {len(body_radii_mm)}")

    atlas_length_m = sum(2 * math.pi * radius for radius in body_radii_mm) / 1000
    models = []
    for outer_mm in OUTER_RADII_MM:
        for core_mm in CORE_RADII_MM:
            turns, length_m = modeled_length_m(outer_mm, core_mm)
            models.append({
                "outer_radius_mm": outer_mm,
                "core_radius_mm": core_mm,
                "pitch_mm_per_turn": PITCH_MM,
                "modeled_wraps": turns,
                "modeled_length_m": round(length_m, 3),
                "atlas_body_length_fraction": round(atlas_length_m / length_m, 4),
            })

    result = {
        "purpose": "assumption-explicit length model; not a complete-scroll measurement",
        "source": "data/wrap_radial.json",
        "voxel_size_um": VX_UM,
        "atlas_body_wrap_count": len(body_radii_mm),
        "atlas_body_circumference_sum_m": round(atlas_length_m, 3),
        "assumptions": {
            "outer_radius_mm": list(OUTER_RADII_MM),
            "core_radius_mm": list(CORE_RADII_MM),
            "pitch_mm_per_turn": PITCH_MM,
            "length_formula": "sum(2*pi*(core_radius_mm + i*pitch_mm_per_turn))",
        },
        "models": models,
        "limits": [
            "No per-mesh area is computed here.",
            "No total surface fraction is inferred here.",
            "Whole-scroll radii and pitch are assumptions, not observations from every wrap.",
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
