#!/usr/bin/env python3
"""Replay the committed full-calibration counts from real-data measurements.

``calibration_stats*.json`` are the frozen per-window measurements produced
from the real rasters. This script applies the shipped v0.3 gates to those
measurements, joins them to the current role-aware registry, and prints the
release-report counts without pretending that this lightweight replay is a
fresh execution of the image-processing code. Use ``validate_tool.py`` for
the much slower full code sweep.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import card_config


def _load_measurements() -> dict[str, dict]:
    with (HERE / "calibration_stats.json").open() as handle:
        primary = {row["id"]: row for row in json.load(handle)}
    with (HERE / "calibration_stats3.json").open() as handle:
        otsu = json.load(handle)
    for ident, row in primary.items():
        row["otsu"] = otsu[ident]["otsu"]
    return primary


def verdict(row: dict) -> bool:
    return (
        row["e_frac"] >= card_config.E_FRAC_MIN
        and row["areafrac_K05"] >= card_config.STRUCTURE_AREAFRAC_MIN
        and row["otsu"] >= card_config.OTSU_SEP_MIN
    )


def summary() -> dict[str, dict]:
    with (HERE / "calibration_set.json").open() as handle:
        windows = json.load(handle)["windows"]
    measurements = _load_measurements()
    if {row["id"] for row in windows} != set(measurements):
        raise ValueError("calibration registry and frozen measurements do not name the same windows")

    buckets = defaultdict(lambda: {"passed": 0, "total": 0})
    for window in windows:
        key = f"{window['role']}/{window['split']}/{window['label']}"
        buckets[key]["total"] += 1
        buckets[key]["passed"] += int(verdict(measurements[window["id"]]))
    return dict(sorted(buckets.items()))


def main() -> int:
    for key, counts in summary().items():
        print(f"{key}: {counts['passed']}/{counts['total']} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
