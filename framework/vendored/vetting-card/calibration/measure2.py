#!/usr/bin/env python3
"""Extended candidate statistics (round 2): stroke-scale DoG bands, text-row
periodicity, and component-shape stats. Same windows as measure.py; output
calibration_stats2.json. Selection happens on the TRAIN split only."""
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
from vet_map import normalize01

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_C = {}


def get_map01(rel):
    if rel not in _C:
        _C.clear()
        _C[rel] = normalize01(np.array(Image.open(os.path.join(ROOT, rel)).convert("L"), dtype=np.float64))
    return _C[rel]


def dog_efrac(patch, s1, s2):
    lo = ndimage.gaussian_filter(patch, s1, mode="mirror")
    hi = ndimage.gaussian_filter(patch, s2, mode="mirror")
    v = patch.var()
    return float(((lo - hi) ** 2).mean() / v) if v > 0 else 0.0


def row_periodicity(patch, px_um):
    prof = patch.mean(axis=1)
    prof = prof - prof.mean()
    if prof.std() < 1e-9:
        return 0.0
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac /= ac[0]
    lo = max(2, int(1500 / px_um))
    hi = min(len(ac) - 1, int(5000 / px_um))
    return float(ac[lo:hi + 1].max()) if hi > lo else 0.0


def comp_shape(patch, px_um, k):
    thr = patch.mean() + k * patch.std()
    lab, n = ndimage.label(patch > thr)
    if n == 0:
        return 0.0, 0.0
    lo_a, hi_a = card_config.component_area_px_range(px_um)
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.where((sizes >= lo_a) & (sizes <= hi_a))[0] + 1
    if len(keep) == 0:
        return 0.0, 0.0
    elong = []
    objs = ndimage.find_objects(lab)
    for i in keep[:60]:
        sl = objs[i - 1]
        ys, xs = np.nonzero(lab[sl] == i)
        if len(ys) < 8:
            continue
        cov = np.cov(np.vstack([ys, xs]))
        ev = np.linalg.eigvalsh(cov)
        elong.append(10.0 if ev[0] <= 1e-9 else float(np.sqrt(ev[1] / ev[0])))
    if not elong:
        return 0.0, 0.0
    e = np.array(elong)
    return float(np.median(e)), float((e >= 2.0).mean())


def one(w):
    m = get_map01(w["map"])
    x0, y0, win = w["x"], w["y"], w["win"]
    patch = m[y0:y0 + win, x0:x0 + win]
    px = w["px_um"]
    L, H = card_config.letter_scale_px_range(px)
    out = {"id": w["id"]}
    for name, (a, b) in {"cur": (0.25 * L, 0.5 * H), "stroke": (0.10 * L, 0.30 * L),
                         "mid": (0.20 * L, 0.60 * L), "wide": (0.15 * L, 0.60 * H)}.items():
        a = max(0.5, a)
        out["ef_" + name] = dog_efrac(patch, a, max(a * 1.5, b))
    out["rowper"] = row_periodicity(patch, px)
    for k, tag in ((0.5, "K05"), (1.0, "K10")):
        med_e, frac_e = comp_shape(patch, px, k)
        out["elong_med_" + tag] = med_e
        out["elong_frac_" + tag] = frac_e
    return out


def main():
    ws = json.load(open(os.path.join(HERE, "calibration_set.json")))["windows"]
    ws.sort(key=lambda w: w["map"])
    with Pool(6) as p:
        rows = p.map(one, ws, chunksize=8)
    json.dump(rows, open(os.path.join(HERE, "calibration_stats2.json"), "w"), indent=1)
    print("stats2:", len(rows))


if __name__ == "__main__":
    main()
