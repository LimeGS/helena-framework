#!/usr/bin/env python3
"""Regenerate the column plates from the OFFICIAL ds8 ink maps.

For each wrap in reading order (outer to inner, w059 -> w023, then the
title as control), downloads the segment's official ds8 ink-detection map
from the open-data bucket and writes a full-resolution plate:
contrast stretch to the 2-98 percentile band, NOTHING else (no denoising,
no content edits, no resampling).

Reading order comes from wrap_radial.json (see wrap_order.py): the wNNN
sequence is strongly geometry-consistent with radial order (Spearman 0.9993),
though it is not a strict sort of the per-wrap radius estimates. Scrolls are read
unrolling outer -> inner, and within a plate the greek runs left to right
(verified: continuous text lines across the full plate width + non-mirrored
letterforms + the officially-read title belongs to this same render family).

Usage:
    python scripts/make_plates.py --order data/wrap_radial.json \
        [--out plates] [--work work_maps]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vesuvius_data import ls, download  # noqa: E402

def reading_from_radial(path):
    """Outer-to-inner wraps derived from P8's measured mean radii."""
    try:
        payload = json.loads(open(path).read())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read measured wrap order {path}: {exc}") from exc
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(segments, dict) or not segments:
        raise RuntimeError("measured wrap order has no segments")
    measured = []
    seen = set()
    for segment, statistics in segments.items():
        match = re.search(r"-(w\d+|title)_", str(segment))
        if not match:
            continue
        wrap = match.group(1)
        if wrap in seen:
            raise RuntimeError(f"duplicate wrap {wrap} in measured order")
        seen.add(wrap)
        try:
            radius = float(statistics["r_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"wrap {wrap} has no numeric r_mean") from exc
        if not math.isfinite(radius):
            raise RuntimeError(f"wrap {wrap} has non-finite r_mean")
        measured.append((radius, wrap))
    if not measured:
        raise RuntimeError("measured wrap order contains no named wraps")
    measured.sort(key=lambda item: (-item[0], item[1]))
    return [wrap for _radius, wrap in measured]


def stretch(a):
    lo, hi = np.percentile(a, [2, 98])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="plates")
    ap.add_argument("--work", default="work_maps")
    ap.add_argument("--scroll", default="PHerc0139")
    ap.add_argument("--order", required=True,
                    help="P8 wrap_radial.json; plates follow descending r_mean")
    args = ap.parse_args()
    if os.path.isdir(args.out) and os.listdir(args.out):
        raise RuntimeError(
            f"plate output {args.out} is not empty; use a new evidence directory")
    os.makedirs(args.out, exist_ok=True)
    reading = reading_from_radial(args.order)
    ordering_sha256 = hashlib.sha256(open(args.order, "rb").read()).hexdigest()

    subs, _ = ls(f"{args.scroll}/segments/", delimiter="/")
    segs = sorted(s.split("/")[-2] for s in subs)
    by_wrap = {}
    for seg in segs:
        for w in reading:
            if f"-{w}_" in seg:
                by_wrap[w] = seg

    missing = [wrap for wrap in reading if wrap not in by_wrap]
    if missing:
        raise RuntimeError(f"measured wraps absent from public segments: {missing}")

    plates = []
    for i, w in enumerate(reading):
        seg = by_wrap[w]
        _, keys = ls(f"{args.scroll}/segments/{seg}/ink-detection/downsampled/", delimiter="/")
        ds8 = [k for k, _ in keys if k.endswith("-ds8.jpg")]
        if not ds8:
            raise RuntimeError(
                f"measured wrap {w} has no official ds8 map; a partial plate "
                "set cannot be called successful")
        local = os.path.join(args.work, os.path.basename(ds8[0]))
        download(ds8[0], local)
        M = np.array(Image.open(local).convert("L"))
        filename = f"{i+1:02d}_{w}.png"
        destination = os.path.join(args.out, filename)
        Image.fromarray(stretch(M)).save(destination, optimize=True)
        payload = open(destination, "rb").read()
        plates.append({
            "file": filename,
            "wrap": w,
            "source_segment": seg,
            "source_map": ds8[0],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "width": int(M.shape[1]),
            "height": int(M.shape[0]),
        })
        print(f"{i+1:02d} {w} <- {seg} ({M.shape[1]}x{M.shape[0]}px)")

    manifest = {
        "schema": "campaignx.p9_plate_set.v1",
        "status": "PASS",
        "sample_id": args.scroll,
        "ordering_sha256": ordering_sha256,
        "plate_count": len(plates),
        "plates": plates,
    }
    temporary = os.path.join(args.out, ".PLATE_MANIFEST.json.tmp")
    with open(temporary, "w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, os.path.join(args.out, "PLATE_MANIFEST.json"))


if __name__ == "__main__":
    main()
