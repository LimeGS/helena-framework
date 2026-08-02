#!/usr/bin/env python3
"""Offline integrity checks for the published PHerc 0139 atlas release.

Checks the committed data, plate sets and viewer manifest without downloading
external data. Run after regenerating the viewer:

    python scripts/validate_release.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BODY = [f"w{i:03d}" for i in range(59, 22, -1)]
EXPECTED_ORDER = EXPECTED_BODY + ["title"]
EXPECTED_READINGS = {"w025", "w034", "w047", "w049"}


def load_json(path: Path):
    return json.loads(path.read_text())


def viewer_entries() -> list[dict]:
    text = (ROOT / "viewer/index.html").read_text()
    marker = "const PLATES = "
    start = text.index(marker) + len(marker)
    end = text.index("];", start) + 1
    return json.loads(text[start:end])


def check(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS {message}")
    else:
        print(f"FAIL {message}")
        failures.append(message)


def main() -> None:
    failures: list[str] = []
    radial = load_json(ROOT / "data/wrap_radial.json")
    estimate = load_json(ROOT / "data/scroll_extent_estimate.json")
    high = load_json(ROOT / "data/review_0139_human.json")["decisions"]
    band = load_json(ROOT / "data/review_band_0139.json")["decisions"]
    entries = viewer_entries()

    check(len(radial["segments"]) == 38, "radial snapshot has 38 segments", failures)
    check(estimate["atlas_body_wrap_count"] == 37, "extent estimate uses 37 body wraps", failures)
    check(len(entries) == 38, "viewer has 38 entries", failures)
    check([entry["wrap"] for entry in entries] == EXPECTED_ORDER,
          "viewer order is w059 through w023, then title", failures)
    check({entry["wrap"] for entry in entries if entry.get("reading")} == EXPECTED_READINGS,
          "viewer labels exactly the four wraps with published phrase readings", failures)

    clear = sum(decision["rating"] == 1 for decision in high + band)
    check(len(high) == 63 and len(band) == 73 and clear == 107,
          "review data has 63 high-band + 73 relaxed-band decisions and 107 clear windows", failures)
    check(sum(entry["n_clear"] for entry in entries) == clear,
          "viewer clear-window totals match review JSON", failures)

    for directory, suffix in (("plates", ".png"), ("plates_photo", "_photo.png"),
                              ("plates_villa", "_villa.png")):
        files = list((ROOT / directory).glob(f"*{suffix}"))
        check(len(files) == 38, f"{directory} contains 38 plate files", failures)

    expected_keys = {"src", "src_photo", "src_villa", "boxes", "size_px"}
    check(all(expected_keys <= entry.keys() for entry in entries),
          "viewer entries use relative sources and SVG box metadata", failures)
    check(all(not value.startswith("data:") for entry in entries
              for key, value in entry.items() if key.startswith("src")),
          "viewer does not embed image payloads", failures)
    viewer_dir = ROOT / "viewer"
    check(all((viewer_dir / entry[key]).resolve().is_file()
              for entry in entries for key in ("src", "src_photo", "src_villa")),
          "every viewer source path resolves to a committed image", failures)
    check(all(all(0 <= x <= entry["size_px"][0] and 0 <= y <= entry["size_px"][1]
                  and size > 0 and x + size <= entry["size_px"][0]
                  and y + size <= entry["size_px"][1]
                  for x, y, size in entry["boxes"])
              for entry in entries),
          "all review boxes lie inside their source plate", failures)
    index_size = (ROOT / "viewer/index.html").stat().st_size
    check(index_size < 1_000_000, "viewer index is below 1 MB", failures)
    check("submitted 2026-06-27" in (ROOT / "README.md").read_text(),
          "README uses the arXiv submission date", failures)
    check("plates_photo/" in (ROOT / "LICENSE").read_text() and
          "plates_villa/" in (ROOT / "LICENSE").read_text(),
          "license note covers all derivative plate styles", failures)

    if failures:
        print(f"\n{len(failures)} release check(s) failed.")
        raise SystemExit(1)
    print("\nAll release checks passed.")


if __name__ == "__main__":
    main()
