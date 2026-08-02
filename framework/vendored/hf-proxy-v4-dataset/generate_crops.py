#!/usr/bin/env python3
"""Regenerate proxy_v4 / proxy_s2s3_v1 training crops from train_labels.jsonl.

train_labels.jsonl stores *coordinates into the official/self-rendered maps*,
not pixels, so this script is what turns a labels row back into the exact
160x160 crop the classifier trains on. The preprocessing below is copied
verbatim from the model card's "How to use" section and from
the project's own train_proxy_v4.py / train_proxy_s2s3_v1.py -- do not "clean
it up", the scores are only reproducible if the recipe is bit-for-bit
identical.

Two map families, two lookup rules:

  scroll == "s1"    -> panel_or_segment is one of the 16 official ds8
                       PHercParis4 (Scroll 1) ink-detection panels
                       (w010-w100 series, June 2026 upload), a plain
                       single-channel image. Window is always 512x512
                       (WIN=512 for these ds8 maps, the scale the
                       index and training used throughout) and gets resized
                       straight to 160x160.

  scroll == "s2s3"  -> panel_or_segment is "s3p1"/"s3p2" (two self-rendered
                       Scroll 3 mosaics, stored as float32 .npy, values in
                       [0,1]) or "s2-<segid>" (an individual self-rendered
                       Scroll 2 fragment .npy under s2small_harvest/{A,B}/).
                       These are the project's OWN custom renders, not
                       something published on the official S3 bucket -- see
                       "Known limitation" in DATASET_CARD.md. Window size
                       ("win") varies per row (physical ~1cm at that mosaic's
                       px/um scale) and where win < 160 target px the crop is
                       resized to "cls_px" px and PADDED onto a black 160x160
                       canvas rather than stretched (see README "Do not
                       stretch undersized crops").

Usage:
    python generate_crops.py \\
        --labels train_labels.jsonl \\
        --maps-dir /path/to/s1_official_panels \\
        --s2s3-dir /path/to/s2s3_harvest \\
        --out ./crops \\
        [--split train] [--download]

--maps-dir defaults to look for "<panel_or_segment>.jpg" files (the official
ds8 ink maps, one per panel). If a panel is missing and --download is
passed, it is fetched from the public anonymous S3 bucket
(vesuvius-challenge-open-data) using the key layout

    PHercParis4/segments/<panel>/ink-detection/downsampled/<...>-ds8.jpg

The download is source-locked to the exact 2.4 um / volume-20260411134726
render family used by the index and training run. It refuses to download if
that source is absent or ambiguous; some panel prefixes also publish a
different 1.129 um ds8 render. Crops are written as lossless PNG: the
original training scripts cropped in memory and never wrote intermediate
files, so lossless storage is what reproduces their pixels exactly.

--s2s3-dir expects the project's internal harvest layout: "<panel>_mosaic.npy" for s3p1/s3p2, and
"s2small_harvest/{A,B}/p<segid>.npy" for s2-<segid> rows. These are large
(hundreds of MB) project-internal render artifacts, not redistributed with
this dataset -- S2/S3 crop regeneration requires access to them separately
(see DATASET_CARD.md). Rows are skipped with a warning if --s2s3-dir is not
given.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SZ = 160  # classifier input side, both scroll families


def stretch_to_01(crop_f32: np.ndarray) -> np.ndarray:
    """Percentile [2, 99.5] stretch over 'active' (>10) pixels -- identical to
    the model card's score_window() and to crop_arr() in
    train_proxy_v4.py / train_proxy_s2s3_v1.py."""
    active = crop_f32[crop_f32 > 10]
    if len(active) > 50:
        lo, hi = np.percentile(active, [2, 99.5])
        return np.clip((crop_f32 - lo) / max(hi - lo, 1e-6), 0, 1)
    return crop_f32 / 255.0


def crop_s1(im: np.ndarray, y: int, x: int, win: int) -> Image.Image:
    c = im[y:y + win, x:x + win].astype(np.float32)
    c01 = stretch_to_01(c)
    img = Image.fromarray((c01 * 255).astype(np.uint8))
    return img.resize((SZ, SZ), Image.BILINEAR)


def crop_s2s3(mosaic01: np.ndarray, y: int, x: int, win: int, cls_px: int) -> Image.Image:
    """mosaic01: float array already in [0,1] (as stored in the project's
    *_mosaic.npy / s2small_harvest .npy files)."""
    c255 = (mosaic01[y:y + win, x:x + win] * 255).astype(np.float32)
    c01 = stretch_to_01(c255)
    img = Image.fromarray((c01 * 255).astype(np.uint8)).resize((cls_px, cls_px), Image.BILINEAR)
    if cls_px == SZ:
        return img
    canvas = Image.new("L", (SZ, SZ), 0)
    canvas.paste(img, ((SZ - cls_px) // 2, (SZ - cls_px) // 2))
    return canvas


def load_s1_panel(maps_dir: str, panel: str, download: bool) -> np.ndarray:
    path = os.path.join(maps_dir, panel + ".jpg")
    if not os.path.exists(path):
        if download:
            _try_download_panel(panel, path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"official panel image not found: {path}\n"
                f"Point --maps-dir at a directory containing '{panel}.jpg', "
                f"or pass --download to attempt an S3 fetch."
            )
    return np.array(Image.open(path).convert("L"))


S3_BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
S1_DS8_SOURCE = "2.4um-0.22m-78keV-volume-20260411134726"


def _select_s1_ds8_key(keys: list[str], prefix: str) -> str:
    """Select the one ds8 map from the exact render family used by proxy_v4.

    A panel prefix may contain both 1.129 um and 2.4 um ds8 maps. Selecting
    the first lexicographic key is therefore not reproducible.
    """
    ds8_keys = sorted(key for key in keys if key.endswith("-ds8.jpg"))
    matches = [key for key in ds8_keys if S1_DS8_SOURCE in key]
    if len(matches) != 1:
        available = ", ".join(ds8_keys) if ds8_keys else "(none)"
        raise RuntimeError(
            f"{prefix}: expected exactly one ds8 map containing "
            f"{S1_DS8_SOURCE!r}, found {len(matches)}; "
            f"available ds8 keys: {available}"
        )
    return matches[0]


def _try_download_panel(panel: str, dest: str) -> None:
    """Fetch one official Scroll 1 (PHercParis4) panel ds8 ink map from the
    public anonymous bucket, source-locked to the render family used by the
    published index:

        PHercParis4/segments/<panel>/ink-detection/downsampled/<...>-ds8.jpg
    """
    import re
    import urllib.parse
    import urllib.request

    prefix = f"PHercParis4/segments/{panel}/ink-detection/downsampled/"
    list_url = f"{S3_BASE}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
    try:
        with urllib.request.urlopen(list_url, timeout=60) as resp:
            xml = resp.read().decode()
    except Exception as e:
        raise RuntimeError(f"bucket listing failed for {panel}: {e}") from e
    keys = re.findall(r"<Key>([^<]+)</Key>", xml)
    selected_key = _select_s1_ds8_key(keys, prefix)
    src = f"{S3_BASE}/{urllib.parse.quote(selected_key)}"
    print(f"  [download] fetching {selected_key} -> {dest}", file=sys.stderr)
    urllib.request.urlretrieve(src, dest)


def load_s2s3_mosaic(cache: dict, s2s3_dir: str, panel: str) -> np.ndarray:
    if panel in cache:
        return cache[panel]
    if panel in ("s3p1", "s3p2"):
        path = os.path.join(s2s3_dir, f"{panel}_mosaic.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} -- see --s2s3-dir in --help")
        arr = np.load(path).astype(np.float32)
    else:
        segid = panel.replace("s2-", "")
        matches = glob.glob(os.path.join(s2s3_dir, "s2small_harvest", "?", f"p{segid}.npy"))
        if not matches:
            raise FileNotFoundError(
                f"missing s2small_harvest/*/p{segid}.npy under {s2s3_dir} -- see --s2s3-dir in --help"
            )
        arr = np.load(matches[0]).astype(np.float32)
    cache[panel] = arr
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="train_labels.jsonl")
    ap.add_argument("--maps-dir", default=None, help="dir with official S1 panel <panel>.jpg files")
    ap.add_argument("--s2s3-dir", default=None, help="dir with s3p1_mosaic.npy / s3p2_mosaic.npy / s2small_harvest/")
    ap.add_argument("--out", default="./crops")
    ap.add_argument("--split", default=None, choices=["train", "excluded"],
                     help="only generate crops for this split (default: all rows)")
    ap.add_argument("--download", action="store_true",
                     help="best-effort S3 fetch of missing S1 panels (see module docstring)")
    ap.add_argument("--limit", type=int, default=None, help="generate only the first N matching rows (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = [json.loads(l) for l in open(args.labels) if l.strip()]
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]

    s1_cache, s2s3_cache = {}, {}
    n_ok = n_skip = 0
    for i, r in enumerate(rows):
        out_path = os.path.join(args.out, f"{r['id']}.png")
        if os.path.exists(out_path):
            n_ok += 1
            continue
        try:
            if r["scroll"] == "s1":
                if not args.maps_dir:
                    raise FileNotFoundError("--maps-dir is required for scroll==s1 rows")
                if r["panel_or_segment"] not in s1_cache:
                    s1_cache[r["panel_or_segment"]] = load_s1_panel(args.maps_dir, r["panel_or_segment"], args.download)
                im = s1_cache[r["panel_or_segment"]]
                img = crop_s1(im, r["y"], r["x"], r["win"])
            elif r["scroll"] == "s2s3":
                if not args.s2s3_dir:
                    raise FileNotFoundError("--s2s3-dir is required for scroll==s2s3 rows")
                mosaic = load_s2s3_mosaic(s2s3_cache, args.s2s3_dir, r["panel_or_segment"])
                img = crop_s2s3(mosaic, r["y"], r["x"], r["win"], r.get("cls_px", SZ))
            else:
                raise ValueError(f"unknown scroll {r['scroll']!r}")
            img.save(out_path)
            n_ok += 1
        except Exception as e:
            print(f"  SKIP {r['id']}: {e}", file=sys.stderr)
            n_skip += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} ({n_ok} ok, {n_skip} skipped)", file=sys.stderr)

    print(f"done: {n_ok} crops written to {args.out} ({n_skip} skipped)")


if __name__ == "__main__":
    main()
