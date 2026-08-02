#!/usr/bin/env python3
"""Rebuild viewer/index.html from the plates + the human review.

Reads the full-resolution plates (plates/ ink, plates_photo/ and
plates_villa/ composites), the human window review
(data/review_0139_human.json for the >=0.9 pass, plus
data/review_band_0139.json for the relaxed 0.6-0.9 pass if present) and the
wrap geometry (data/wrap_radial.json), and regenerates the viewer:

  - one PLATES[] entry per segmented wrap, in physical reading order
    (outer -> inner, w059 -> w023, title last as control);
  - each entry carries repository-relative paths to all three full-resolution
    styles. Amber review boxes are rendered by the page as an SVG overlay on
    the ink-map style; no images are embedded in index.html;
  - per-wrap metadata: n_clear (# rating==1 windows), max_score, radius and
    review state. A reviewed wrap with no clear window remains distinguishable
    from one that has not yet been reviewed.

The page chrome (head/style/script) is preserved verbatim from the current
viewer/index.html, and the four published phrase-reading labels are carried
over from it byte-for-byte; only the PLATES array is swapped. Idempotent.

Usage:
    python scripts/build_viewer.py
"""
import json
import os
import re
from collections import defaultdict

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

VX_UM = 2.399           # CT voxel size (um) -> radius in mm


def wtok(seg):
    m = re.search(r"-(w\d+|title)_?", seg)
    return m.group(1) if m else None


def main():
    rev = json.load(open(os.path.join(ROOT, "data/review_0139_human.json")))
    decisions = list(rev["decisions"])
    # fold in the relaxed-threshold pass (0.6<=score<0.9, human-reviewed) if present
    band_path = os.path.join(ROOT, "data/review_band_0139.json")
    if os.path.exists(band_path):
        decisions += json.load(open(band_path))["decisions"]
    wr = json.load(open(os.path.join(ROOT, "data/wrap_radial.json")))["segments"]

    vpath = os.path.join(ROOT, "viewer/index.html")
    html = open(vpath, encoding="utf-8").read()
    marker = "const PLATES = "
    i0 = html.index(marker)
    arr_end = html.index("];", i0) + 1   # through the closing ']' , before the ';'
    i1 = arr_end + 1                      # past the ';' , for the tail splice
    # carry the published phrase-reading labels over verbatim (Greek + diacritics)
    prev = {e["wrap"]: e.get("reading", "")
            for e in json.loads(html[i0 + len(marker):arr_end])}

    # per-wrap aggregation from the raw review
    agg = defaultdict(lambda: {"n_clear": 0, "max_score": None, "boxes": []})
    reviewed = set()  # any wrap a human actually looked at (any verdict)
    for d in decisions:
        w = wtok(d["segment"])
        reviewed.add(w)
        e = agg[w]
        e["max_score"] = d["score"] if e["max_score"] is None else max(e["max_score"], d["score"])
        if d["rating"] == 1:
            e["n_clear"] += 1
            e["boxes"].append((d["x"], d["y"], d["win"]))

    seg_by_w, r_by_w = {}, {}
    for seg, st in wr.items():
        w = wtok(seg)
        seg_by_w[w] = seg
        r_by_w[w] = round(st["r_mean"] * VX_UM / 1000, 2)

    body = sorted((w for w in seg_by_w if w != "title"),
                  key=lambda w: int(w[1:]), reverse=True)
    order = body + ["title"]

    plates = []
    for i, w in enumerate(order, 1):
        seg = seg_by_w[w]
        ink_name = f"{i:02d}_{w}.png"
        with Image.open(os.path.join(ROOT, "plates", ink_name)) as ink:
            W, H = ink.size

        nc = agg[w]["n_clear"]
        plates.append({
            "pos": i, "wrap": w, "seg": seg, "r_mm": r_by_w[w],
            "n_clear": nc, "reviewed": w in reviewed,
            "max_score": round(agg[w]["max_score"], 4) if nc > 0 else None,
            "src": f"../plates/{ink_name}", "size_px": [W, H],
            "reading": prev.get(w, ""), "boxes": agg[w]["boxes"],
            "src_photo": f"../plates_photo/{i:02d}_{w}_photo.png",
            "src_villa": f"../plates_villa/{i:02d}_{w}_villa.png",
        })
        print(f"{i:02d} {w:6s} {W}x{H} n_clear={nc} "
              f"boxes={len(agg[w]['boxes'])} max={plates[-1]['max_score']}")

    new_html = (html[:i0] + marker
                + json.dumps(plates, ensure_ascii=False) + ";" + html[i1:])
    open(vpath, "w", encoding="utf-8").write(new_html)
    kb = len(new_html.encode("utf-8")) / 1024
    print(f"\nwrote {vpath}: {len(plates)} plates, {kb:.1f} KiB")


if __name__ == "__main__":
    main()
