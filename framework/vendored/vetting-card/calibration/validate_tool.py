#!/usr/bin/env python3
"""End-to-end validation of the v0.2 verdicts THROUGH the real tool
(vet_map.run_all_checks on the real maps), not through the measurement
scripts: the implementation must reproduce the calibration numbers.

Checks, in order:
  1. The five windows from the 2026-07-11 discovery diagnostic all PASS.
  2. Fit population through the tool: held-out TPR/FPR match the frozen
     result (55/56 positives pass; 1/45 clean negatives pass).
  3. Scroll 4 external validation is reported separately (27/36 dense-wrap
     positives pass; 0/60 lacunae negatives pass).
  4. Synthetic families through the tool: white noise, blurred noise
     (sigma 12/20/30), synthetic fiber lines -> 0 passes; blank -> degenerate.
  5. Timing: one 0139 window and one large-S1-panel window, default path.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vet_map import normalize01, run_all_checks

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_C = {}


def get_raw(rel):
    if rel not in _C:
        _C.clear()
        _C[rel] = np.array(Image.open(os.path.join(ROOT, rel)).convert("L"), dtype=np.float64)
    return _C[rel]


def verdict(w):
    raw = get_raw(w["map"])
    r = run_all_checks(raw, (w["x"], w["y"], w["x"] + w["win"], w["y"] + w["win"]), w["px_um"])
    return r["overall"]["pass"]


def main():
    ws = json.load(open(os.path.join(HERE, "calibration_set.json")))["windows"]
    by_id = {w["id"]: w for w in ws}

    # 1. the five from the diagnostic
    diag_ids = [i for i in by_id
                if any(seg in i for seg in ("w044_2026011522", "w043_2026011217",
                                             "w041_2026010816", "w058_2026021020"))
                and by_id[i]["group"] == "0139_clear"]
    ok = 0
    for i in sorted(diag_ids):
        p = verdict(by_id[i])
        ok += p
        print(f"  diag {i.split('_y')[0][-20:]}: {'PASS' if p else 'FAIL'}")
    print(f"1) original diagnostic: {ok}/{len(diag_ids)} pass (0 before)")

    # 2. the complete set per split (map-sorted for the cache)
    ws_sorted = sorted(ws, key=lambda w: w["map"])
    results = {}
    for w in ws_sorted:
        results[w["id"]] = verdict(w)
    for split in ("train", "test"):
        pos = [w for w in ws if w["split"] == split and w["label"] == "positive"]
        neg = [w for w in ws if w["split"] == split and w["label"] == "negative"
               and w["group"] != "s1_implied_neg"]
        tp = sum(results[w["id"]] for w in pos)
        fp = sum(results[w["id"]] for w in neg)
        print(f"2) {split}: TPR {tp}/{len(pos)}  FPR {fp}/{len(neg)}")

    for label in ("positive", "negative"):
        rows = [w for w in ws if w["split"] == "external_validation" and w["label"] == label]
        passed = sum(results[w["id"]] for w in rows)
        print(f"3) external Scroll 4 {label}: {passed}/{len(rows)} pass")

    # 3. tool-generated synthetics
    rng = np.random.default_rng(0)
    fam = {
        "white": lambda: rng.random((512, 512)),
        "smooth12": lambda: ndimage.gaussian_filter(rng.random((512, 512)), 12),
        "smooth20": lambda: ndimage.gaussian_filter(rng.random((512, 512)), 20),
        "smooth30": lambda: ndimage.gaussian_filter(rng.random((512, 512)), 30),
        "fiber_lines": lambda: (0.5 + 0.4 * np.sin(2 * np.pi * np.arange(512)[:, None]
                                / rng.integers(6, 14)) + 0.3 * rng.random((512, 512))),
    }
    for name, gen in fam.items():
        passes = 0
        for _ in range(20):
            raw = normalize01(gen()) * 255.0
            r = run_all_checks(raw, (0, 0, 512, 512), 19.2)
            passes += r["overall"]["pass"]
        print(f"4) synthetic {name}: {passes}/20 pass (0 expected)")
    blank = run_all_checks(np.full((512, 512), 7.0), (0, 0, 512, 512), 19.2)
    print(f"4) blank: degenerate gate = {'OK' if not blank['overall']['pass'] else 'MAL'}")

    # 4. timing
    w0 = next(w for w in ws_sorted if w["group"] == "0139_clear")
    t = time.time(); verdict(w0); t0139 = time.time() - t
    wS1 = next(w for w in ws_sorted if w["group"] == "s1_confirmed")
    get_raw(wS1["map"])  # precarga fuera del reloj
    t = time.time(); verdict(wS1); tS1 = time.time() - t
    print(f"5) timing default: 0139 {t0139:.2f}s | panel S1 grande {tS1:.2f}s (objetivo <5s)")


if __name__ == "__main__":
    main()
