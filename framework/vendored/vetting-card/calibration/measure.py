#!/usr/bin/env python3
"""Compute raw + candidate statistics for every calibration window.

For each window in calibration_set.json, against the map normalized the
same way vet_map does (map-level min-max to [0,1]):

  e_raw      current letter_energy statistic: mean squared DoG response in
             the bbox (padded-crop DoG, exactly vet_map.bandpass_energy).
  e_frac     contrast-invariant candidate: e_raw / patch variance.
  n_K10      current structure statistic at K=1.0: count of connected
             components with area inside card_config's letter band.
  n_K05      same at K=0.5 (lower adaptive threshold).
  dens_K10   n_K10 normalized per letter-cell: count / (bbox_area / hi^2).
  areafrac_K10  fraction of bbox pixels in in-band components (K=1.0).
  patch_mean, patch_std   context.

Parallel over windows (each loads only its padded crop lazily via a
per-process map cache). Output: calibration_stats.json.

Run from the vetting-card root:
    python calibration/measure.py [--set calibration/calibration_set.json]
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import card_config
from vet_map import bandpass_energy, normalize01

_MAP_CACHE = {}
_ROOT = None


def get_map01(rel):
    if rel not in _MAP_CACHE:
        raw = np.array(Image.open(os.path.join(_ROOT, rel)).convert("L"), dtype=np.float64)
        _MAP_CACHE.clear()          # keep at most one map per worker
        _MAP_CACHE[rel] = normalize01(raw)
    return _MAP_CACHE[rel]


def structure_stats(map01, bbox, px_um, k):
    x0, y0, x1, y1 = bbox
    patch = map01[y0:y1, x0:x1]
    thr = patch.mean() + k * patch.std()
    labeled, n = ndimage.label(patch > thr)
    lo_a, hi_a = card_config.component_area_px_range(px_um)
    if n == 0:
        return 0, 0.0, []
    sizes = np.bincount(labeled.ravel())[1:]
    inband = (sizes >= lo_a) & (sizes <= hi_a)
    count = int(inband.sum())
    areafrac = float(sizes[inband].sum() / patch.size)
    return count, areafrac, sizes.tolist()


def one(w):
    m = get_map01(w["map"])
    bbox = (w["x"], w["y"], w["x"] + w["win"], w["y"] + w["win"])
    px_um = w["px_um"]
    sig_lo, sig_hi = card_config.bandpass_sigmas(px_um)
    x0, y0, x1, y1 = bbox
    patch = m[y0:y1, x0:x1]
    e_raw = bandpass_energy(m, bbox, sig_lo, sig_hi)
    var = float(patch.var())
    n10, af10, sizes10 = structure_stats(m, bbox, px_um, 1.0)
    n05, af05, _ = structure_stats(m, bbox, px_um, 0.5)
    _, hi_scale = card_config.letter_scale_px_range(px_um)
    cells = (patch.size) / (hi_scale ** 2)
    return {
        "id": w["id"], "group": w["group"], "label": w["label"], "split": w["split"],
        "px_um": px_um, "win": w["win"],
        "e_raw": e_raw,
        "e_frac": e_raw / var if var > 0 else 0.0,
        "n_K10": n10, "n_K05": n05,
        "dens_K10": n10 / cells if cells > 0 else 0.0,
        "dens_K05": n05 / cells if cells > 0 else 0.0,
        "areafrac_K10": af10, "areafrac_K05": af05,
        "patch_mean": float(patch.mean()), "patch_std": float(patch.std()),
        "sizes_K10_top": sorted(sizes10, reverse=True)[:5],
    }


def init(root):
    global _ROOT
    _ROOT = root


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default=os.path.join(HERE, "calibration_set.json"))
    ap.add_argument("--project-root", default=os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    ap.add_argument("--out", default=os.path.join(HERE, "calibration_stats.json"))
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    ws = json.load(open(args.set))["windows"]
    ws.sort(key=lambda w: w["map"])          # group by map -> cache hits
    with Pool(args.jobs, initializer=init, initargs=(args.project_root,)) as pool:
        stats = pool.map(one, ws, chunksize=8)
    json.dump(stats, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(stats)} filas")


if __name__ == "__main__":
    main()
