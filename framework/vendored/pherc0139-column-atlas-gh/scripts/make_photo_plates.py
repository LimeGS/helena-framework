#!/usr/bin/env python3
"""Regenerate the publication-style plates (papyrus texture + ink composited).

Same reading order and same official ds8 ink maps as make_plates.py, but
instead of showing the ink map white-on-black, this composites it onto the
CT surface texture as a light papyrus background. Two recipes, sharing one
texture fetch per wrap:

    plates_photo/  out = paper_texture * (1 - ALPHA * ink)            (ours)
    plates_villa/  out = tint*opacity + paper*(1-opacity), opacity =
                   smoothstep(ink, VILLA_OPACITY_LO, VILLA_OPACITY_HI)
                   * VILLA_OPACITY_SCALE                 (verified villa recipe)

plates_photo is our own approximation, in the visual spirit of the paper's
Fig. 4 panel b ("ink-enhanced signal shown in black against the papyrus
texture") but not a byte-verified match to it -- no formula is published in
the paper. plates_villa reproduces, byte-for-byte against the published
config, the ink-bake recipe in ScrollPrize/villa's
foundation/scroll-unwrap-pipeline (docs/RENDER-STYLE.md, configs/global.yaml
`[ink]`): tint (16,16,16) neutral carbon black, opacity_lo=0.42,
opacity_hi=0.78, opacity_scale=0.88, no glow. That pipeline composites onto
its own mesh-rendered RGBA texture, which we don't have access to here, so
plates_villa still uses our same zarr-slice papyrus background as the base
-- the ink formula is verified, the base texture is still an approximation.

The texture is the segment's surface-volume zarr (level-3 multiscale,
mid-depth slice). Those chunks are raw uint8 with no compressor, so we pull
only the three mid layers of each chunk with HTTP Range requests rather
than downloading the whole volume.

Usage:
    python scripts/make_photo_plates.py [--out plates_photo]
        [--out-villa plates_villa] [--work work_maps]

Requires: numpy, pillow, boto3.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import boto3
from botocore import UNSIGNED
from botocore.config import Config

Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vesuvius_data import ls, download  # noqa: E402

BUCKET = "vesuvius-challenge-open-data"
ALPHA = 0.88
# Real villa ink-bake recipe (verified byte-for-byte, 2026-07-13, against
# ScrollPrize/villa foundation/scroll-unwrap-pipeline docs/RENDER-STYLE.md +
# configs/global.yaml [ink]): opacity = smoothstep(ink, lo, hi) * scale,
# composited as tint*opacity + base*(1-opacity). Tint is neutral carbon
# black, never crushed to pure #000.
VILLA_OPACITY_LO = 0.42
VILLA_OPACITY_HI = 0.78
VILLA_OPACITY_SCALE = 0.88
VILLA_TINT = 16.0  # (16,16,16) = #101010; single channel, base is achromatic
# Every segmented wrap in physical reading order (outer -> inner, w059 -> w023;
# rho=0.9993 vs radius), with the title (subscriptio, innermost) last as control.
READING = ["w059", "w058", "w057", "w056", "w055", "w054", "w053", "w052",
           "w051", "w050", "w049", "w048", "w047", "w046", "w045", "w044",
           "w043", "w042", "w041", "w040", "w039", "w038", "w037", "w036",
           "w035", "w034", "w033", "w032", "w031", "w030", "w029", "w028",
           "w027", "w026", "w025", "w024", "w023", "title"]


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def fetch_texture(cli, seg, work):
    """Mid-depth slice of the segment's surface-volume, level-3 multiscale."""
    subs, _ = ls(f"PHerc0139/segments/{seg}/surface-volumes/", delimiter="/")
    if not subs:
        return None
    zarr = subs[0]
    za = json.load(open(download(zarr + "3/.zarray", os.path.join(work, f"{seg}_l3.zarray"))))
    if za["compressor"] is not None or za["dtype"] != "|u1":
        return None  # this fast path assumes raw uint8 chunks
    zs, ys, xs = za["shape"]
    _, ycs, xcs = za["chunks"]
    ny, nx = math.ceil(ys / ycs), math.ceil(xs / xcs)
    z_mid, n_z = zs // 2, 3
    z0 = max(0, z_mid - 1)
    layer = ycs * xcs
    start, end = z0 * layer, (z0 + n_z) * layer - 1
    out = np.zeros((ys, xs), dtype=np.float32)

    def get(yx):
        cy, cx = yx
        try:
            r = cli.get_object(Bucket=BUCKET, Key=f"{zarr}3/0/{cy}/{cx}",
                               Range=f"bytes={start}-{end}")
            return cy, cx, np.frombuffer(r["Body"].read(), dtype=np.uint8) \
                .reshape(n_z, ycs, xcs).mean(axis=0)
        except Exception:
            return cy, cx, None

    with ThreadPoolExecutor(16) as ex:
        for cy, cx, block in ex.map(get, [(a, b) for a in range(ny) for b in range(nx)]):
            if block is None:
                continue
            y0, x0 = cy * ycs, cx * xcs
            h, w = min(ycs, ys - y0), min(xcs, xs - x0)
            out[y0:y0 + h, x0:x0 + w] = block[:h, :w]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="plates_photo")
    ap.add_argument("--out-villa", default="plates_villa")
    ap.add_argument("--work", default="work_maps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.out_villa, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    cli = boto3.client("s3", region_name="us-east-1",
                       config=Config(signature_version=UNSIGNED, max_pool_connections=32))

    subs, _ = ls("PHerc0139/segments/", delimiter="/")
    allsegs = sorted(s.split("/")[-2] for s in subs)
    by_wrap = {w: seg for w in READING for seg in allsegs if f"-{w}_" in seg}

    for i, w in enumerate(READING):
        seg = by_wrap[w]
        _, keys = ls(f"PHerc0139/segments/{seg}/ink-detection/downsampled/", delimiter="/")
        ds8 = [k for k, _ in keys if k.endswith("-ds8.jpg")]
        if not ds8:
            print(f"{w}: no ds8 map, skipped")
            continue
        ink = np.array(Image.open(download(ds8[0], os.path.join(args.work, os.path.basename(ds8[0]))))
                       .convert("L"), dtype=np.float32)
        tex = fetch_texture(cli, seg, args.work)
        if tex is None or tex.shape != ink.shape:
            print(f"{w}: texture unavailable/misaligned, flat background")
            tex = np.full_like(ink, 160.0)
        valid = tex > 1
        lo, hi = np.percentile(tex[valid], [2, 98]) if valid.any() else (0, 255)
        paper = 105 + np.clip((tex - lo) / max(hi - lo, 1), 0, 1) * 135
        ink01 = np.clip(ink / 255.0, 0, 1)

        comp = paper * (1 - ALPHA * ink01)
        comp[~valid] = 246
        Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(
            f"{args.out}/{i+1:02d}_{w}_photo.png", optimize=True)

        opacity = smoothstep(ink01, VILLA_OPACITY_LO, VILLA_OPACITY_HI) * VILLA_OPACITY_SCALE
        villa = VILLA_TINT * opacity + paper * (1 - opacity)
        villa[~valid] = 246
        Image.fromarray(np.clip(villa, 0, 255).astype(np.uint8)).save(
            f"{args.out_villa}/{i+1:02d}_{w}_villa.png", optimize=True)

        print(f"{i+1:02d} {w} <- {seg}")


if __name__ == "__main__":
    main()
