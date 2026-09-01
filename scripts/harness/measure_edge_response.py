#!/usr/bin/env python3
"""How much of a probability map's brightest 1% sits at the edge of the render.

A detector that lights up along the boundary of what was rendered is reporting
the boundary. The map still has a high p99, still passes liveness, and still
looks like a result -- the number does not say where it came from.

The measure is a ratio, not a fraction, because the fraction alone is
meaningless: a narrow render is mostly edge, so 20% of anything falls there. It
compares where the top 1% sits against where the map's own valid pixels sit, so
1.0x is "no preference for the edge" whatever the geometry.

Two edges are measured because a map has two, and a detector can find either:

  mask   the boundary of the valid region -- where the surface stopped
  array  the boundary of the raster -- where the render was cut off

  measure_edge_response.py run-a/probability.npy run-b/probability.npy
  measure_edge_response.py --band 40 --json out.json map.npy

Measured on PHerc826 with this: the TimeSformer lane put 23.9% of its brightest
1% within 40 px of the mask edge against a 10.1% baseline -- 2.4x -- on the
surface with the highest p99 of the whole scroll. The 9 um lane came in under
1.0x on all five, which is its own finding: a map that ignores the geometry
entirely is not reading the surface either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def enrichment(values: np.ndarray, valid: np.ndarray, near: np.ndarray,
               percentile: float) -> tuple[float, float, float]:
    """Share of the top slice near the edge, the baseline share, and the ratio."""
    top = valid & (values >= np.percentile(values[valid], percentile))
    share_top = 100.0 * (top & near).sum() / max(top.sum(), 1)
    share_all = 100.0 * near.sum() / max(valid.sum(), 1)
    return share_top, share_all, share_top / max(share_all, 1e-9)


def measure(path: Path, band: int, percentile: float) -> dict:
    array = np.load(path).astype(np.float64)
    valid = np.isfinite(array) & (array > 0)
    if valid.sum() < 1000:
        return {"path": str(path), "error": "no usable valid region"}

    # Distance to the nearest invalid pixel: the edge of the surface as
    # rendered, whatever shape it is.
    to_mask = ndimage.distance_transform_edt(valid)
    # Distance to the nearest raster boundary: where the render was cut off.
    # A map whose valid region fills the frame has no mask edge at all, and
    # this is the only edge it has.
    height, width = array.shape
    rows, columns = np.mgrid[0:height, 0:width]
    to_array = np.minimum.reduce(
        [rows, columns, height - 1 - rows, width - 1 - columns])

    result: dict = {"path": str(path), "shape": list(array.shape),
                    "valid_pixels": int(valid.sum()), "band_px": band,
                    "percentile": percentile}
    for name, distance in (("mask", to_mask), ("array", to_array)):
        top, base, ratio = enrichment(array, valid, valid & (distance <= band),
                                      percentile)
        result[name] = {"top_share_pct": round(top, 2),
                        "baseline_pct": round(base, 2),
                        "enrichment": round(ratio, 2)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("maps", nargs="+", type=Path)
    parser.add_argument("--band", type=int, default=40,
                        help="how many pixels counts as 'at the edge' (default 40, "
                             "about 375 um at 9.362 um per pixel)")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    print(f"{'map':<34} {'mask':>18} {'array':>18}")
    print(f"{'':<34} {'top / base = x':>18} {'top / base = x':>18}")
    results = []
    for path in args.maps:
        row = measure(path, args.band, args.percentile)
        results.append(row)
        if "error" in row:
            print(f"{path.parent.name[:33]:<34} {row['error']}")
            continue
        cells = []
        for name in ("mask", "array"):
            cell = row[name]
            cells.append(f"{cell['top_share_pct']:>5.1f}/{cell['baseline_pct']:>4.1f}"
                         f"={cell['enrichment']:>4.1f}x")
        print(f"{path.parent.name[:33]:<34} {cells[0]:>18} {cells[1]:>18}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\n{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
