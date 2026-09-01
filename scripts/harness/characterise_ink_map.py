#!/usr/bin/env python3
"""What a probability map looks like, beyond the number it reports.

Every ink lane emits a float per pixel and a p99, and those two facts have
turned out to say almost nothing about whether the map read the papyrus. Two
maps of the same surface can share every percentile and correlate at r=0.02.
A map whose brightest region is the edge of the render still passes liveness.

So this measures the properties that distinguish a detector reading a surface
from a detector reporting its own texture:

  distribution   floor, median, p99, max. The floor is diagnostic on its own:
                 lanes trained with BCE label smoothing sit near 0.25 for
                 confident no-ink, so a median close to the floor across
                 unrelated surfaces means the lane is emitting its prior.

  edge           how much of the brightest 1% sits within a band of the render
                 boundary, against where that map's pixels sit in general.
                 1.0x is no preference; a detector tracing the boundary is
                 reporting the boundary.

  structure      the brightest 1% as connected components. Ink is strokes:
                 elongated, many pixels, few components. Fibre and speckle
                 confounds are many small round ones. Reported as the median
                 component size and the share of the top 1% living in
                 components of at least `--stroke-px` pixels.

  anisotropy     whether that structure has a direction. Papyrus fibre runs one
                 way, so a detector following fibre produces oriented
                 structure; text does too, but along the writing line rather
                 than the grain, and either is more informative than isotropic
                 speckle. Measured as the eigenvalue ratio of the gradient
                 structure tensor over the bright pixels.

  tiling         energy at the inference tile pitch. A map that shows its own
                 tile grid is reporting the tiling, and nothing on a papyrus
                 has that periodicity.

None of these decide whether there is ink. They decide whether a map is worth
asking that question about.

  characterise_ink_map.py a/probability.npy b/mean_probability.npy --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


MIN_BASELINE_PCT = 2.0


def edge_enrichment(values, valid, distance, band, percentile):
    """Enrichment of the top slice near an edge, and whether it can be believed.

    The ratio is unreadable when almost no pixel is near the edge: one control
    map produced 7.2% against a 0.1% baseline, which prints as 88.9x and means
    nothing -- a handful of pixels either way moves it by tens. So the baseline
    travels with the number and a ratio computed on less than 2% is returned as
    unreliable rather than as a large figure.
    """
    near = valid & (distance <= band)
    top = valid & (values >= np.percentile(values[valid], percentile))
    share_top = 100.0 * (top & near).sum() / max(top.sum(), 1)
    share_all = 100.0 * near.sum() / max(valid.sum(), 1)
    ratio = share_top / max(share_all, 1e-9)
    return {"top_pct": round(share_top, 1), "baseline_pct": round(share_all, 1),
            "enrichment": round(ratio, 2),
            "reliable": bool(share_all >= MIN_BASELINE_PCT)}


def structure(values, valid, percentile, stroke_px):
    top = valid & (values >= np.percentile(values[valid], percentile))
    labels, count = ndimage.label(top)
    if count == 0:
        return {"components": 0}
    sizes = np.bincount(labels.ravel())[1:]
    big = sizes >= stroke_px
    return {
        "components": int(count),
        "median_component_px": int(np.median(sizes)),
        "largest_component_px": int(sizes.max()),
        "share_in_components_over_threshold_pct":
            round(100.0 * sizes[big].sum() / max(sizes.sum(), 1), 1),
        "threshold_px": stroke_px,
    }


def anisotropy(values, valid):
    """Eigenvalue ratio of the gradient structure tensor: 1.0 is isotropic."""
    filled = np.where(valid, values, float(np.nanmedian(values[valid])))
    gy, gx = np.gradient(ndimage.gaussian_filter(filled, 2.0))
    weight = valid.astype(float)
    jxx = float((gx * gx * weight).sum())
    jyy = float((gy * gy * weight).sum())
    jxy = float((gx * gy * weight).sum())
    trace, det = jxx + jyy, jxx * jyy - jxy * jxy
    root = max(trace * trace / 4.0 - det, 0.0) ** 0.5
    high, low = trace / 2.0 + root, trace / 2.0 - root
    return round(high / max(low, 1e-12), 2)


def tiling(values, valid, pitch):
    """Ratio of spectral energy at the tile pitch to its neighbourhood."""
    filled = np.where(valid, values, float(np.nanmedian(values[valid])))
    row = filled.mean(axis=0) - filled.mean()
    spectrum = np.abs(np.fft.rfft(row)) ** 2
    if len(row) < pitch * 3:
        return None
    k = max(1, round(len(row) / pitch))
    lo, hi = max(1, k - 2), min(len(spectrum), k + 3)
    around = np.concatenate([spectrum[max(1, k - 12):lo], spectrum[hi:k + 13]])
    if not around.size:
        return None
    return round(float(spectrum[lo:hi].max() / max(around.mean(), 1e-12)), 1)


def characterise(path: Path, band: int, percentile: float,
                 stroke_px: int, pitch: int) -> dict:
    array = np.load(path).astype(np.float64)
    valid = np.isfinite(array) & (array > 0)
    if valid.sum() < 1000:
        return {"path": str(path), "error": "no usable valid region"}
    values = array[valid]

    height, width = array.shape
    rows, columns = np.mgrid[0:height, 0:width]
    to_array = np.minimum.reduce(
        [rows, columns, height - 1 - rows, width - 1 - columns])

    return {
        "path": str(path),
        "name": path.parent.name,
        "shape": list(array.shape),
        "valid_pct": round(100.0 * valid.sum() / array.size, 1),
        "floor_p1": round(float(np.percentile(values, 1)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p99": round(float(np.percentile(values, 99)), 4),
        "max": round(float(values.max()), 4),
        "edge_mask_x": edge_enrichment(
            array, valid, ndimage.distance_transform_edt(valid), band, percentile),
        "edge_array_x": edge_enrichment(array, valid, to_array, band, percentile),
        "structure": structure(array, valid, percentile, stroke_px),
        "anisotropy": anisotropy(array, valid),
        "tiling_x": tiling(array, valid, pitch),
    }


def edge_cell(edge: dict) -> str:
    """`--` where the baseline is too small for the ratio to mean anything."""
    return f"{edge['enrichment']:.1f}x" if edge["reliable"] else "  --"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("maps", nargs="+", type=Path)
    parser.add_argument("--band", type=int, default=40)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--stroke-px", type=int, default=50)
    parser.add_argument("--tile-pitch", type=int, default=64)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rows = [characterise(p, args.band, args.percentile, args.stroke_px,
                         args.tile_pitch) for p in args.maps]
    header = (f"{'map':<34} {'floor':>6} {'p50':>6} {'p99':>6} "
              f"{'edge':>7} {'comp':>6} {'med':>5} {'big%':>6} {'aniso':>6} {'tile':>6}")
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['path'][:33]:<34} {row['error']}")
            continue
        st = row["structure"]
        print(f"{row['name'][:33]:<34} {row['floor_p1']:>6.3f} {row['p50']:>6.3f} "
              f"{row['p99']:>6.3f} {edge_cell(row['edge_mask_x']):>7} "
              f"{st.get('components', 0):>6} {st.get('median_component_px', 0):>5} "
              f"{st.get('share_in_components_over_threshold_pct', 0):>5.1f}% "
              f"{row['anisotropy']:>6.2f} "
              f"{(str(row['tiling_x']) + 'x') if row['tiling_x'] else '   -':>6}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"\n{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
