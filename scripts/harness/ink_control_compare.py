#!/usr/bin/env python3
"""What our render and our ink map are worth, against the community's own.

    scripts/harness/ink_control_compare.py --stack DIR --map probability.npy

Two comparisons, because they answer different questions and only one of them
is about the model:

* our rendered layer stack against their published surface volume for the same
  segment. This says whether P4 sampled the same ground at the same depth. It
  is the control on the control: if it fails, nothing downstream means anything.
* our probability map against their published ink map, on the window the first
  comparison has already pinned. This says whether the detector sees what their
  recipe saw.

Both references are open data from PHerc 0139 segment 20250108000000-w025, over
one CT volume, and the ink map's filename names the mesh and the recipe that
produced it. Nothing here is registered, warped or fitted: the render lands on
their pixel grid by construction, because the window is cut on mesh-cell
boundaries and rendered at scale 1.0.

Measured 2026-07-28, in the order the control found things:

    ink maps  r = 0.079   the lane routed through the generic runner
              r = 0.092   through the recipe's own runner (@1.1.0)
              r = 0.885   and with the render's normal direction corrected

    renders   r = 0.9815 on the middle layer -- but layer by layer in
              descending order, our 0 against their 85 and our 62 against their
              23, each at r = 0.99. The renderer traverses the normal the other
              way on this mesh, which --flip-normals corrects.

Two faults, and only the second one moved the number. The first was real: the
recipe ships its own runner and the profile named the generic one, which
normalises clip(0,200)/200 and resamples the depth axis where the recipe divides
by 255 and takes consecutive layers. The second was the sign of the normal: a
depth-reversed slab is a correct render of the far side of the sheet first, and
an ink model handed one produces something unrelated -- which is what 0.079 was.

At 0.885, in the "as is" orientation only (every other transform falls to noise)
and with matching marginals -- ours marks 0.340 of pixels above 0.5 and theirs
0.351 -- this pipeline reproduces the community's map with the community's
model. What remains between 0.885 and 1.0 is unaccounted for: their published
map may carry test-time augmentation, an ensemble or post-processing that its
filename does not name, and our render differs from theirs by interpolation at
r = 0.99 rather than 1.0.

Non-claims
----------
* The window is chosen on their ink map, so this is a positive control and not
  a survey.
* A correlation with a published map is not a reading.
* A recipe name is not a checkpoint: a published map may come from a different
  checkpoint, an ensemble, or post-processing the name does not carry, and this
  comparison cannot separate that from a real disagreement.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tifffile

OPEN_DATA = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SEGMENT = "PHerc0139/segments/20250108000000-w025_2025010863"
SURFACE_VOLUME = (f"{OPEN_DATA}/{SEGMENT}/surface-volumes/"
                  "2.399um-0.22m-78keV-volume-20260102150214.zarr/0")
INK_MAP = (f"{OPEN_DATA}/{SEGMENT}/ink-detection/PHerc0139-20250108000000-2.399um-"
           "0.22m-78keV-volume-20260102150214-20260417190342-"
           "new_canon_autoresearch_recipe-tile256-stride128.tif")
SURFACE_DEPTH, CHUNK = 109, 128
# The window this control renders, in mesh cells, and the mesh grid's scale.
WINDOW_ROW, WINDOW_COL, WINDOW_CELLS, CELL = 1126, 972, 102, 20


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def their_surface_window(cy: int, cx: int, chunks: int) -> np.ndarray:
    """Their rendered surface volume, assembled from raw zarr chunks.

    The array is uncompressed uint8 with one chunk per (z, y, x) cell, so each
    object is exactly depth*128*128 bytes and needs no zarr client.
    """
    window = np.zeros((SURFACE_DEPTH, chunks * CHUNK, chunks * CHUNK), dtype=np.uint8)
    for iy in range(chunks):
        for ix in range(chunks):
            with urllib.request.urlopen(
                    f"{SURFACE_VOLUME}/0/{cy + iy}/{cx + ix}", timeout=120) as response:
                raw = response.read()
            window[:, iy * CHUNK:(iy + 1) * CHUNK, ix * CHUNK:(ix + 1) * CHUNK] = (
                np.frombuffer(raw, dtype=np.uint8).reshape(SURFACE_DEPTH, CHUNK, CHUNK))
    return window


def compare_renders(stack: Path, chunks: int = 4) -> dict:
    """Our layers against theirs, and at which of their depths ours sits."""
    ours = np.stack([tifffile.imread(path)
                     for path in sorted(stack.glob("*.tif"), key=lambda p: int(p.stem))])
    origin_y, origin_x = WINDOW_ROW * CELL, WINDOW_COL * CELL
    cy, cx = (origin_y + CHUNK - 1) // CHUNK, (origin_x + CHUNK - 1) // CHUNK
    theirs = their_surface_window(cy, cx, chunks)
    offset_y, offset_x = cy * CHUNK - origin_y, cx * CHUNK - origin_x
    size = chunks * CHUNK
    ours_window = ours[:, offset_y:offset_y + size, offset_x:offset_x + size]
    middle = ours_window[len(ours) // 2]
    by_depth = sorted(((pearson(middle, theirs[z]), z) for z in range(SURFACE_DEPTH)),
                      reverse=True)
    best_r, best_z = by_depth[0]
    return {"our_layers": int(len(ours)), "their_layers": SURFACE_DEPTH,
            "our_middle_matches_their_layer": int(best_z),
            "pearson_r": best_r,
            "one_layer_away": float(max(pearson(middle, theirs[best_z - 1]),
                                        pearson(middle, theirs[best_z + 1]))
                                    if 0 < best_z < SURFACE_DEPTH - 1 else float("nan")),
            "worst_of_their_layers": float(by_depth[-1][0])}


def compare_maps(probability: Path) -> dict:
    """Our map against theirs, under every axis convention a renderer can differ by."""
    ours = np.load(probability).astype(np.float64)
    origin_y, origin_x = WINDOW_ROW * CELL, WINDOW_COL * CELL
    size = WINDOW_CELLS * CELL
    # Read it whole first: a compressed TIFF is not readable from a stream that
    # cannot seek, and 750 MB of uint8 is cheaper than being clever about it.
    with urllib.request.urlopen(INK_MAP, timeout=600) as response:
        theirs_full = tifffile.imread(io.BytesIO(response.read()))
    theirs = theirs_full[origin_y:origin_y + size,
                         origin_x:origin_x + size].astype(np.float64) / 255.0
    del theirs_full
    views = {"as is": ours, "flip ud": ours[::-1], "flip lr": ours[:, ::-1],
             "rot180": ours[::-1, ::-1], "transpose": ours.T,
             "rot90": np.rot90(ours), "rot270": np.rot90(ours, 3),
             "transpose flipped": ours.T[::-1]}
    scores = {label: pearson(view, theirs) for label, view in views.items()}
    return {"pearson_r_by_orientation": scores,
            "best": max(scores.items(), key=lambda item: abs(item[1])),
            "ours_fraction_above_half": float((ours > 0.5).mean()),
            "theirs_fraction_above_half": float((theirs > 0.5).mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path,
                        help="our rendered layer stack, as .tif per layer")
    parser.add_argument("--map", type=Path, help="our probability.npy")
    parser.add_argument("--receipt", type=Path, default=None)
    arguments = parser.parse_args()
    if not arguments.stack and not arguments.map:
        parser.error("pass --stack, --map, or both")

    report: dict = {"schema": "campaignx.ink_control_comparison.v1",
                    "reference_segment": SEGMENT,
                    "window_mesh_cells": {"row": WINDOW_ROW, "col": WINDOW_COL,
                                          "cells": WINDOW_CELLS, "scale": CELL}}
    if arguments.stack:
        report["renders"] = compare_renders(arguments.stack)
        print("render vs their published surface volume:")
        for key, value in report["renders"].items():
            print(f"  {key}: {value}")
    if arguments.map:
        report["ink_maps"] = compare_maps(arguments.map)
        print("\nink map vs their published ink map:")
        for label, score in report["ink_maps"]["pearson_r_by_orientation"].items():
            print(f"  {label:18s} r = {score:+.4f}")
        print(f"  ours marks {report['ink_maps']['ours_fraction_above_half']:.3f} "
              f"of pixels above 0.5, theirs "
              f"{report['ink_maps']['theirs_fraction_above_half']:.3f}")
    report["non_claims"] = [
        "the window is chosen on their ink map, so this is a positive control",
        "a correlation with a published map is not a reading",
        "a recipe name is not a checkpoint: an ensemble or post-processing the "
        "name does not carry looks the same as a disagreement from here",
    ]
    if arguments.receipt:
        arguments.receipt.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\nwrote {arguments.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
