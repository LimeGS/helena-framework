"""Transport human ink labels onto a re-render without downloading full PPMs.

Ink labels live on a segment's (u, v) grid. The segment's PPM maps that grid
to 3D coordinates in a legacy volume frame, and published transforms map that
frame into newer (e.g. 2.4 um) volumes. Re-rendering the same PPM against the
new volume therefore yields a stack whose pixels are already aligned with the
original labels — transported ground truth for free.

The obstacle is size: Scroll 1 segment PPMs reach 15 GB. This tool picks
label-dense windows first and then HTTP-Range fetches only the PPM rows those
windows need, so the transfer cost is a few hundred MB per window.

Each window is packaged as an .npz (img: DxSxS uint8, label: SxS, mask: SxS)
plus a QC overlay JPEG (render mid-layer with the label boundary drawn on
top) for visual verification before the window enters any training set.

Usage:
    python label_transport.py --seg 20231005123336 \
        [--label labels/20231005123336_inklabels.png] \
        [--size 1280 --max-windows 4]

Status note: the windowing/fetch path is exercised; the downstream renderer
call may need adaptation to your rendering setup (see --render-cmd).
"""
import argparse
import os
import subprocess
import urllib.request

import numpy as np
from PIL import Image

from wedge_extract import ppm_header_str

Image.MAX_IMAGE_PIXELS = None
BASE = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths"


def ppm_header(seg):
    req = urllib.request.Request(f"{BASE}/{seg}/{seg}.ppm",
                                 headers={"Range": "bytes=0-399"})
    head = urllib.request.urlopen(req, timeout=30).read()
    i = head.find(b"<>\n")
    f = dict(l.split(": ") for l in head[:i].decode().strip().splitlines())
    return int(f["width"]), int(f["height"]), i + 3


def fetch_rows(seg, off, W, y0, nrows, out_path, batch=64):
    """Fetch [y0, y0+nrows) PPM rows via HTTP Range, in row batches.

    Single giant Range requests (>~1 GB) are unreliable on this server
    (IncompleteRead); batching by rows with retries is."""
    with open(out_path, "wb") as f:
        f.write(ppm_header_str(W, nrows).encode())
        for r0 in range(0, nrows, batch):
            r1 = min(r0 + batch, nrows)
            start = off + (y0 + r0) * W * 48
            end = off + (y0 + r1) * W * 48 - 1
            want = (r1 - r0) * W * 48
            for attempt in range(4):
                req = urllib.request.Request(
                    f"{BASE}/{seg}/{seg}.ppm",
                    headers={"Range": f"bytes={start}-{end}"})
                try:
                    buf = urllib.request.urlopen(req, timeout=300).read()
                    if len(buf) == want:
                        break
                except Exception:
                    pass
                import time; time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"rows {y0+r0}-{y0+r1} unfetchable after retries")
            f.write(buf)


def pick_windows(lab, size, max_windows, min_density=0.01):
    """Rank size x size windows by ink density (16x-downsampled scan) and
    greedily pick the densest ones that do not overlap each other."""
    lb = (lab > 127).astype(np.float32)
    ds = lb[::16, ::16]
    s16 = size // 16
    step = max(s16 // 2, 1)
    cands = []
    for y in range(0, ds.shape[0] - s16, step):
        for x in range(0, ds.shape[1] - s16, step):
            cands.append((ds[y:y + s16, x:x + s16].mean(), y * 16, x * 16))
    cands.sort(reverse=True)
    picked = []
    for d, y, x in cands:
        if d < min_density or len(picked) >= max_windows:
            break
        if all(abs(y - py) >= size or abs(x - px) >= size for _, py, px in picked):
            picked.append((d, y, x))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--label", default=None,
                    help="path to the segment's ink label PNG")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--max-windows", type=int, default=4)
    ap.add_argument("--render-cmd", default="python render_scroll3.py",
                    help="command that renders a banded PPM to a layer stack. "
                         "This repo ships Scroll 2/3 renderers; for Scroll 1 "
                         "adapt the volume constants (five lines, see README)")
    ap.add_argument("--out", default="transported")
    ap.add_argument("--tmp", default="tmp_bands")
    a = ap.parse_args()
    S = a.size
    lab_path = a.label or f"labels/{a.seg}_inklabels.png"
    lab = np.array(Image.open(lab_path).convert("L"), np.uint8)
    W, H, off = ppm_header(a.seg)
    if lab.shape != (H, W):
        lab = np.array(Image.open(lab_path).convert("L").resize((W, H),
                                                                Image.NEAREST),
                       np.uint8)
    print(f"{a.seg}: PPM {W}x{H}, label ink={(lab > 127).mean():.4f}", flush=True)

    picked = pick_windows(lab, S, a.max_windows)
    print(f"windows: {[(round(d, 3), y, x) for d, y, x in picked]}", flush=True)

    os.makedirs(a.tmp, exist_ok=True)
    od = f"{a.out}/{a.seg}"
    os.makedirs(od, exist_ok=True)
    for wi, (d, y0, x0) in enumerate(picked):
        y0 = min(y0, H - S)
        x0 = min(x0, W - S)
        band_ppm = f"{a.tmp}/{a.seg}_w{wi}.ppm"
        fetch_rows(a.seg, off, W, y0, S, band_ppm)
        rdir = f"{a.tmp}/{a.seg}_w{wi}"
        r = subprocess.run(a.render_cmd.split() +
                           ["--ppm", band_ppm, "--out", rdir, "--layers", "32"],
                           capture_output=True, text=True)
        if "DONE" not in r.stdout:
            print(f"win{wi} RENDER_FAILED: {r.stdout[-200:]} {r.stderr[-200:]}",
                  flush=True)
            continue
        vol = np.memmap(f"{rdir}/stack.u8", dtype=np.uint8, mode="r",
                        shape=(32, S, W))
        img = np.asarray(vol[:, :, x0:x0 + S])
        l = lab[y0:y0 + S, x0:x0 + S]
        m = (img[16] > 0).astype(np.uint8)
        np.savez_compressed(f"{od}/win{wi}_y{y0}_x{x0}.npz",
                            img=img, label=l, mask=m)
        # QC overlay: mid layer with label boundary in red
        from scipy.ndimage import binary_dilation
        lbb = l > 127
        edge = binary_dilation(lbb, iterations=2) & ~lbb
        ov = np.stack([img[16]] * 3, -1)
        ov[edge] = [255, 60, 30]
        q = Image.fromarray(ov)
        q.thumbnail((1200, 1200))
        q.save(f"{od}/qc_w{wi}.jpg", quality=80)
        subprocess.run(["rm", "-rf", rdir, band_ppm])
        print(f"PACKED {a.seg} win{wi} dens={d:.3f} valid={m.mean():.2f}",
              flush=True)
    print(f"SEG_DONE {a.seg}", flush=True)


if __name__ == "__main__":
    main()
