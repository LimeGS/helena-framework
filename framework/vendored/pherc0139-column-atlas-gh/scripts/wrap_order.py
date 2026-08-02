#!/usr/bin/env python3
"""Compute the physical wrap order of PHerc 0139 from the public meshes.

Downloads every segment's TIFXYZ mesh (meta.json + x/y/z.tif) from the
Vesuvius Challenge open-data bucket (anonymous), then:

1. Subsamples 20k valid points from each mesh's 3D point cloud.
2. Fits the scroll axis: first principal component of the union of all
   subsampled points (~760k points).
3. Per segment, computes the MEAN RADIAL DISTANCE of its point cloud to
   that axis. That is the wrap radius.

METHOD NOTE (the pitfall that costs you a day): do NOT use the radius of
the mesh's centroid. Each segment is a full turn of the scroll, so its
centroid collapses onto the axis regardless of the wrap's true radius
(we measured Spearman 0.28 = noise with centroids, 0.9993 with mean
radial distance of the cloud).

Output: wrap_radial.json (axis, origin, per-segment radial stats) and a
printed table sorted by radius.

Usage:
    python scripts/wrap_order.py [--work WORK_DIR] [--scroll PHerc0139]

Requires: numpy, pillow, boto3, scipy (for the Spearman check).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vesuvius_data import ls, download, read_tifxyz  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="work_meshes", help="local cache dir for meshes")
    ap.add_argument("--scroll", default="PHerc0139")
    ap.add_argument("--out", default="data/wrap_radial.json",
                    help="committed radial-table path (default: data/wrap_radial.json)")
    ap.add_argument("--subsample", type=int, default=20000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    subs, _ = ls(f"{args.scroll}/segments/", delimiter="/")
    segs = sorted(s.split("/")[-2] for s in subs)
    print(f"{len(segs)} segmentos en {args.scroll}")

    clouds, labels, vol_refs = [], [], set()
    for i, seg in enumerate(segs):
        msubs, _ = ls(f"{args.scroll}/segments/{seg}/mesh/", delimiter="/")
        tdirs = [s for s in msubs if "tifxyz" in s]
        if not tdirs:
            print(f"  {seg}: sin tifxyz, salteado")
            continue
        tdir = tdirs[0]
        vol_refs.add(tdir.rstrip("/").split("/")[-1].split("-on-")[-1])
        local = os.path.join(args.work, seg)
        os.makedirs(local, exist_ok=True)
        for fn in ["meta.json", "x.tif", "y.tif", "z.tif"]:
            download(tdir + fn, os.path.join(local, fn))
        d = read_tifxyz(local)
        pts = d["coords"][d["valid"]]
        idx = rng.choice(len(pts), size=min(args.subsample, len(pts)), replace=False)
        clouds.append(pts[idx])
        labels.append(seg)
        print(f"  {i+1}/{len(segs)} {seg} ok ({len(pts)} pts validos)")

    # Every mesh must reference the same volume for the
    # coordinates to be comparable. In PHerc0139 (2026-07) they are.
    print("volumenes referenciados:", vol_refs)
    if len(vol_refs) > 1:
        print("WARNING: more than one volume -- radii are NOT comparable across groups")

    allpts = np.concatenate(clouds)
    mu = allpts.mean(axis=0)
    _, S, Vt = np.linalg.svd(allpts - mu, full_matrices=False)
    axis = Vt[0]
    print("global variance per component:", (S**2 / np.sum(S**2)).round(4))

    out = {}
    for seg, pts in zip(labels, clouds):
        v = pts - mu
        along = v @ axis
        perp = v - np.outer(along, axis)
        r = np.linalg.norm(perp, axis=1)
        out[seg] = {
            "r_mean": float(r.mean()), "r_median": float(np.median(r)),
            "r_p10": float(np.percentile(r, 10)), "r_p90": float(np.percentile(r, 90)),
            "along_mean": float(along.mean()),
            "along_min": float(along.min()), "along_max": float(along.max()),
        }
    json.dump({"axis": axis.tolist(), "origin": mu.tolist(), "segments": out},
              open(args.out, "w"), indent=1)

    # tabla + Spearman (cuerpo wNNN solamente)
    body = []
    for seg, s in out.items():
        m = re.search(r"-(w\d+)_", seg)
        if m:
            body.append((int(m.group(1)[1:]), s["r_mean"], seg))
    body.sort()
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr([b[0] for b in body], [b[1] for b in body])
        print(f"\nSpearman wrap# vs r_mean: rho={rho:.4f} p={p:.2e}")
    except ImportError:
        print("\n(scipy no instalado: sin Spearman)")
    print(f"{'w#':>4} {'r_mean_vx':>10} {'r_mean_mm(2.399um/vx)':>22}")
    for w, r, seg in body:
        print(f"{w:4d} {r:10.0f} {r*2.399/1000:22.2f}")
    print(f"\nescrito: {args.out}")


if __name__ == "__main__":
    main()
