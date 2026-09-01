#!/usr/bin/env python3
"""Compare two probability maps of the same surface, pixel for pixel.

Two P5 runs that differ in exactly one input should be compared on more than
their summary statistics. A p99 that moves by 0.01 says almost nothing on its
own: two maps can share every percentile and disagree everywhere, and two maps
can be bit-identical while a receipt claims they came from different inputs.

So this reports three things that a percentile cannot fake:

  identical      the digests match, which means the change under test did not
                 reach the output at all
  correlation    Pearson r over the pixels valid in both. r=1.0000 on runs that
                 were supposed to differ is the signature of a parameter that
                 was accepted and then ignored -- it has happened here before
  disagreement   where the two maps actually differ, and by how much

  compare_two_maps.py a/probability.npy b/probability.npy --label-a "voxel 1.0"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    sha_a, sha_b = digest(args.a), digest(args.b)
    A, B = np.load(args.a).astype(np.float64), np.load(args.b).astype(np.float64)

    report: dict = {
        "a": {"path": str(args.a), "label": args.label_a, "sha256": sha_a,
              "shape": list(A.shape)},
        "b": {"path": str(args.b), "label": args.label_b, "sha256": sha_b,
              "shape": list(B.shape)},
        "identical_bytes": sha_a == sha_b,
    }
    print(f"{args.label_a}: {A.shape} {sha_a[:16]}")
    print(f"{args.label_b}: {B.shape} {sha_b[:16]}")

    if sha_a == sha_b:
        print("\\nIDENTICAL. Whatever was changed did not reach the output.")
    elif A.shape != B.shape:
        # Different geometry is a real answer, not a failure to compare: it
        # means the change moved the render, so there is no pixel to pair.
        print(f"\\nDifferent shapes, so there is no per-pixel comparison to make. "
              f"That is itself the finding: the change altered the render's own "
              f"geometry, not just its values.")
        report["comparable"] = False
    else:
        valid = np.isfinite(A) & np.isfinite(B)
        report["valid_pixels"] = int(valid.sum())
        a, b = A[valid], B[valid]
        if a.size and a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            report["pearson_r"] = r
            print(f"\\nPearson r = {r:.6f} over {a.size:,} pixels valid in both")
            if r > 0.99995:
                print("  r is 1.0000 to four places: these are the same map. A "
                      "parameter that changes nothing was accepted and ignored.")
        difference = np.abs(a - b)
        report["max_abs_difference"] = float(difference.max())
        report["mean_abs_difference"] = float(difference.mean())
        report["fraction_differing"] = float((difference > 1e-6).mean())
        print(f"  max |difference|  {difference.max():.6f}")
        print(f"  mean |difference| {difference.mean():.6f}")
        print(f"  pixels differing  {(difference > 1e-6).mean() * 100:.2f}%")

    for name, array in ((args.label_a, A), (args.label_b, B)):
        finite = array[np.isfinite(array)]
        if finite.size:
            stats = {"p50": float(np.percentile(finite, 50)),
                     "p99": float(np.percentile(finite, 99)),
                     "max": float(finite.max()), "min": float(finite.min())}
            report.setdefault("statistics", {})[name] = stats
            print(f"\\n{name}: p50={stats['p50']:.4f} p99={stats['p99']:.4f} "
                  f"range {stats['min']:.4f}..{stats['max']:.4f}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\\n")
        print(f"\\n{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
