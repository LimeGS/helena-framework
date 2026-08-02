#!/usr/bin/env python3
"""Recompute every published bucket and denominator from the shipped index.

Input: full_index_complete.json (125,298 rows of {panel, y, x, v3}).
NOTE on the score key: it is named "v3" for legacy reasons (the field name
predates the final checkpoint) but every score in this file was produced by
proxy_v4.pt. Gold = v3 >= 0.9.

Buckets, as documented in the model card's Coverage section:
  - June wrap-series panels (2026-06-23 upload, names ...-wAAA-BBB),
    excluding the two `_jordi` alternate-processing panels.
  - July series (2026-07-01 upload): exactly one panel (w104-106) covers a
    wrap range whose June segment has no published ink map, so it joins the
    deduplicated union; the other 25 re-cover indexed wraps (redundant).
  - Classic-lineage panels (no wrap range in the name): 23 panels that
    reduce to 18 physical surfaces by name lineage (offset-0 re-render,
    _v14, _copy, _v2_flatboi, _v8 are re-renders); representatives count,
    re-renders are redundant. Kept OUTSIDE the wrap-series denominator
    (physical overlap with the panel series unresolved).

Run: python summarize_index.py  (from this directory)
"""
import collections
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = 0.9

CLASSIC_RERENDERS = {
    "20260602225659-5753_0",                    # offset-0 re-render of 20230702185753
    "20260602230115-20230702185753_v14",        # re-process of 20230702185753
    "20260603193311-5753_-1_copy",              # duplicate upload of 5753_-1
    "20260603222816-20231005123336_v2_flatboi", # re-flattening of 20231005123336
    "20260604223808-20231210121321_v8",         # re-process of 20231210121321
}
JULY_UNIQUE = "20260701183141-w104-106"  # June's w104-106 segment has no published ink map


def main():
    rows = json.load(open(os.path.join(HERE, "full_index_complete.json")))
    wr = lambda p: re.search(r"w(\d+)-(\d+)", p)

    assert len(rows) == 125298, len(rows)
    assert len({(r["panel"], r["y"], r["x"]) for r in rows}) == len(rows), "filas duplicadas"
    gold = [r for r in rows if r["v3"] >= GOLD]
    assert len(gold) == 15386, len(gold)
    panels = {r["panel"] for r in rows}
    assert len(panels) == 78, len(panels)

    june = {p for p in panels if wr(p) and p.startswith("202606") and "_jordi" not in p}
    jordi = {p for p in panels if "_jordi" in p}
    july = {p for p in panels if wr(p) and not p.startswith("202606")}
    classic = {p for p in panels if not wr(p)}
    assert (len(june), len(jordi), len(july), len(classic)) == (27, 2, 26, 23)

    g = collections.Counter()
    for r in gold:
        p = r["panel"]
        if p in june or p == JULY_UNIQUE:
            g["union"] += 1
            if int(wr(p).group(1)) >= 38:
                g["union_beyond_w037"] += 1
        elif p in july:
            g["july_redundant"] += 1
        elif p in jordi:
            g["jordi_redundant"] += 1
        elif p in CLASSIC_RERENDERS:
            g["classic_rerender"] += 1
        else:
            g["classic_representative"] += 1

    assert g["union"] == 5678, g
    assert g["union_beyond_w037"] == 4424, g
    assert g["classic_representative"] == 2460, g
    redundant = g["classic_rerender"] + g["jordi_redundant"] + g["july_redundant"]
    assert (g["classic_rerender"], g["jordi_redundant"], g["july_redundant"]) == (224, 739, 6285)
    assert g["union"] + g["classic_representative"] + redundant == 15386

    print(f"rows {len(rows)} | panels {len(panels)} | gold(>= {GOLD}) {len(gold)}")
    print(f"deduplicated wrap-series union : {g['union']}")
    print(f"  beyond w037 (proxy frontier) : {g['union_beyond_w037']}"
          f"  ({100 * g['union_beyond_w037'] / g['union']:.1f}%)")
    print(f"classic representatives (18 surfaces): {g['classic_representative']}")
    print(f"redundant re-renders          : {redundant}"
          f"  (= {g['classic_rerender']} classic + {g['jordi_redundant']} jordi + {g['july_redundant']} july)")
    print("all published numbers reproduced OK")


if __name__ == "__main__":
    main()
