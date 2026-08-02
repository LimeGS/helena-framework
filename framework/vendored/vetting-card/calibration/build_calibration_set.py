#!/usr/bin/env python3
"""Build the real-data calibration set for the vetting-card thresholds.

This is the labeled set card_config.py declared it lacked ("no such labeled
set exists yet"). Every window is on an OFFICIAL ds8 ink map and carries a
human or heuristic label with explicit provenance:

POSITIVES (human-verified letterforms):
  - 54 PHerc 0139 windows rated "clear text" one-by-one by this project's
    reviewer (rating 1 in review_0139_human.json; published in the
    herculaneum-legibility-index repo).
  - 120 Scroll 1 windows proposed by the legibility model and individually
    confirmed by the reviewer (round=model_proposed in train_labels.jsonl;
    published in the HF dataset repo).

HELD-OUT / REPORT-ONLY:
  - 9 PHerc 0139 windows rated "possible" (rating 2) — ambiguous by
    design, never used to fit or score; reported for context only.

SCROLL 4 / PHerc 1667 (the fully-read scroll — ground truth):
  - Positives: proxy_v4 gold windows on wraps mapped to densely-transcribed
    columns (papyrological transcription, wrap-level ground truth).
  - Negatives: low-score windows on wraps confirmed as lacunae in the
    published reading — the strongest "no text here" label available.

NEGATIVES:
  - 50 Scroll 1 GPU-verified fiber windows (fiber_negatives_50.jsonl,
    published; heuristic-ranked, human-spot-checked class).
  - 87 Scroll 1 implied negatives (round=uncertainty_sampled_implied in
    train_labels.jsonl — browsed by the reviewer and not flagged).
  - Up to 100 PHerc 0139 background windows: legibility score <= 0.05 AND
    center at least 2 window-sides away from every gold (>=0.9) window in
    the same segment (heuristic label, flagged as such).

Split: fixed-seed 70/30 train/test stratified by source group. The five
windows from the 2026-07-11 discovery diagnostic (vetting_diag_20260711)
are FORCED into the test split so "the original failures now pass" is an
out-of-sample statement.

Output: calibration_set.json (coordinates + labels + px_um only, no
pixels — same registry-by-recipe philosophy as the published datasets).

Run from the vetting-card root:
    python calibration/build_calibration_set.py \
        [--project-root ../..] [--out calibration/calibration_set.json]
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random

# The five windows evaluated in the 2026-07-11 discovery diagnostic
# (data/index_s5_0139/vetting_diag_20260711.log): forced into test split.
DIAG_KEYS = {
    ("20260115000000-w044_2026011522", 1904, 1632),   # timing run + DENSO
    ("20260112000000-w043_2026011217", None, None),   # DENSO (first clear in seg)
    ("20260108000000-w041_2026010816", None, None),   # RALO
    ("20260210000000-w058_2026021020", None, None),   # RALO
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "calibration_set.json"))
    ap.add_argument(
        "--reference-splits", default=os.path.join(os.path.dirname(__file__), "calibration_set.json"),
        help="frozen registry whose fit-group train/test assignments are preserved when available",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = os.path.abspath(args.project_root)

    frozen_splits = {}
    reference = os.path.abspath(args.reference_splits)
    if os.path.exists(reference):
        reference_data = json.load(open(reference))
        frozen_splits = {
            row["id"]: row["split"]
            for row in reference_data.get("windows", [])
            if row.get("split") in {"train", "test"}
        }

    rows = []

    # ---- 0139: human review ------------------------------------------------
    rev = json.load(open(f"{root}/data/index_s5_0139/review_0139_human.json"))
    maps_0139 = f"{root}/data/index_s5_0139/maps/pherc0139"
    for d in rev["decisions"]:
        seg = d["segment"]
        jpgs = glob.glob(f"{maps_0139}/{seg}/*ds8.jpg")
        if not jpgs:
            continue
        label = {1: "positive", 2: "ambiguous"}[d["rating"]]
        rows.append({
            "id": f"0139_clear_{seg}_y{d['y']}_x{d['x']}" if d["rating"] == 1
                  else f"0139_possible_{seg}_y{d['y']}_x{d['x']}",
            "group": "0139_clear" if d["rating"] == 1 else "0139_possible",
            "label": label,
            "map": os.path.relpath(jpgs[0], root),
            "x": d["x"], "y": d["y"], "win": d["win"],
            "px_um": 9830.0 / d["win"],
            "source": "review_0139_human.json (single human reviewer, published)",
        })

    # ---- S1: labels del dataset publicado ----------------------------------
    labels = [json.loads(l) for l in
              open(f"{root}/release/hf-proxy-v4-dataset/train_labels.jsonl") if l.strip()]
    panels_dir = f"{root}/data/letters/s1_atlas"
    for r in labels:
        if r["scroll"] != "s1":
            continue
        if r["round"] == "model_proposed":
            grp, lab = "s1_confirmed", "positive"
        elif r["round"] == "uncertainty_sampled_implied":
            grp, lab = "s1_implied_neg", "negative"
        else:
            continue
        rows.append({
            "id": f"s1_{grp}_{r['panel_or_segment']}_y{r['y']}_x{r['x']}",
            "group": grp, "label": lab,
            "map": os.path.relpath(f"{panels_dir}/{r['panel_or_segment']}.jpg", root),
            "x": r["x"], "y": r["y"], "win": r["win"], "px_um": 19.2,
            "source": f"train_labels.jsonl round={r['round']} (published)",
        })

    fib = [json.loads(l) for l in
           open(f"{root}/release/hf-proxy-v4-dataset/fiber_negatives_50.jsonl") if l.strip()]
    for r in fib:
        rows.append({
            "id": f"s1_fiber_{r['panel_or_segment']}_y{r['y']}_x{r['x']}",
            "group": "s1_fiber_neg", "label": "negative",
            "map": os.path.relpath(f"{panels_dir}/{r['panel_or_segment']}.jpg", root),
            "x": r["x"], "y": r["y"], "win": r["win"], "px_um": 19.2,
            "source": "fiber_negatives_50.jsonl (published)",
        })

    # ---- 0139 background negatives (heuristic) ------------------------------
    idx = json.load(open(f"{root}/data/index_s5_0139/index_0139.json"))
    by_seg = collections.defaultdict(list)
    for w in idx:
        by_seg[w["segment"]].append(w)
    rng = random.Random(args.seed)
    bg = []
    for seg, ws in sorted(by_seg.items()):
        jpgs = glob.glob(f"{maps_0139}/{seg}/*ds8.jpg")
        if not jpgs:
            continue
        golds = [(w["y"], w["x"], w["WIN"]) for w in ws if w["score"] >= 0.9]
        for w in ws:
            if w["score"] > 0.05 or w.get("clamped"):
                continue
            far = all(max(abs(w["y"] - gy), abs(w["x"] - gx)) >= 2 * gw
                      for gy, gx, gw in golds) if golds else True
            if far:
                bg.append((seg, jpgs[0], w))
    rng.shuffle(bg)
    for seg, jpg, w in bg[:100]:
        rows.append({
            "id": f"0139_bg_{seg}_y{w['y']}_x{w['x']}",
            "group": "0139_bg_neg", "label": "negative",
            "map": os.path.relpath(jpg, root),
            "x": w["x"], "y": w["y"], "win": w["WIN"], "px_um": 9830.0 / w["WIN"],
            "source": "index_0139.json score<=0.05, >=2 win from any gold (heuristic)",
        })

    # ---- Scroll 4 / PHerc 1667: GROUND-TRUTH-anchored (the read scroll) -----
    # Positives: proxy_v4 gold windows on wraps whose mapped column is densely
    # transcribed by papyrologists (w028-w034 = cols 11-17, 105-159 letters —
    # from paper_anchor_align.json + s4_lines.json). Negatives: low-score
    # windows on wraps confirmed as lacunae / lost-margin in the published
    # reading (w011, w037-w041), which scored 0% gold. Wrap-level ground
    # truth (papyrological transcription), not one reviewer's eye. Maps are
    # ds8 at 19.19 um/px, WIN=512 (S4_V4_VALIDATION.md).
    s4_scores = f"{root}/data/ocr_align/s4_proxy_v4_scores.json"
    s4_maps = f"{root}/data/ocr_align/wrap_ds8"
    if os.path.exists(s4_scores):
        DENSE = {"w028", "w029", "w030", "w031", "w032", "w033", "w034"}
        LACUNAE = {"w011", "w037", "w038", "w039", "w040", "w041"}
        S4_WIN, S4_PX = 512, 9830.0 / 512.0  # 19.20 um/px
        sc = json.load(open(s4_scores))
        for w in sc:
            jpg = f"{s4_maps}/{w['wrap']}.jpg"
            if not os.path.exists(jpg):
                continue
            if w["wrap"] in DENSE and w["score"] >= 0.9:
                rows.append({
                    "id": f"s4_dense_{w['wrap']}_y{w['y']}_x{w['x']}",
                    "group": "s4_dense_pos", "label": "positive",
                    "map": os.path.relpath(jpg, root),
                    "x": w["x"], "y": w["y"], "win": S4_WIN, "px_um": S4_PX,
                    "source": "s4_proxy_v4 gold in papyrologist-dense wrap (w028-034, >=100 transcribed letters)",
                })
        # negatives: subsample the lacunae low-score windows to ~60
        lac = [w for w in sc if w["wrap"] in LACUNAE and w["score"] <= 0.05
               and os.path.exists(f"{s4_maps}/{w['wrap']}.jpg")]
        rng_s4 = random.Random(args.seed + 4)
        rng_s4.shuffle(lac)
        for w in lac[:60]:
            rows.append({
                "id": f"s4_lac_{w['wrap']}_y{w['y']}_x{w['x']}",
                "group": "s4_lacunae_neg", "label": "negative",
                "map": os.path.relpath(f"{s4_maps}/{w['wrap']}.jpg", root),
                "x": w["x"], "y": w["y"], "win": S4_WIN, "px_um": S4_PX,
                "source": "s4_proxy_v4 score<=0.05 on published-lacunae wrap (w011,w037-041)",
            })

    # ---- role + split -------------------------------------------------------
    # The published registry, not Python's salted hash(), is the source of
    # truth for the historical fitting split. New fit examples use a stable
    # SHA-256-derived group seed. Scroll 4 is external validation only; it
    # must never appear as a training row merely because generic split code
    # happened to run over it.
    for row in rows:
        group = row["group"]
        if group in {"0139_clear", "s1_confirmed", "s1_fiber_neg", "0139_bg_neg"}:
            row["role"] = "fit"
        elif group.startswith("s4_"):
            row["role"] = "external_validation"
        elif group == "s1_implied_neg":
            row["role"] = "context_only"
        else:
            row["role"] = "report_only"

    # ---- split 70/30 stratified within the fitting population --------------
    first_clear_in_seg = {}
    for r in rows:
        if r["group"] == "0139_clear":
            first_clear_in_seg.setdefault(r["map"].split("/")[-2], r["id"])
    forced_test = set()
    for seg, y, x in DIAG_KEYS:
        for r in rows:
            if r["group"] != "0139_clear" or seg not in r["map"]:
                continue
            if y is None or (r["y"] == y and r["x"] == x):
                forced_test.add(r["id"])
                break
    by_group = collections.defaultdict(list)
    for r in rows:
        if r["role"] == "fit":
            by_group[r["group"]].append(r)
    for grp, lst in by_group.items():
        lst.sort(key=lambda r: r["id"])
        stable_offset = int.from_bytes(hashlib.sha256(grp.encode("utf-8")).digest()[:4], "big")
        rng2 = random.Random(args.seed + stable_offset)
        rng2.shuffle(lst)
        n_test = max(1, round(0.3 * len(lst)))
        test_ids = set(r["id"] for r in lst[:n_test]) | {i for i in forced_test
                                                         if any(r["id"] == i for r in lst)}
        for r in lst:
            r["split"] = frozen_splits.get(r["id"], "test" if r["id"] in test_ids else "train")
    for r in rows:
        if r["role"] == "external_validation":
            r["split"] = "external_validation"
        elif r["role"] == "context_only":
            r["split"] = "context_only"
        elif r["role"] == "report_only":
            r["split"] = "report_only"

    counts = collections.Counter((r["group"], r["split"]) for r in rows)
    out = {
        "schema_version": "v2", "built": "2026-07-12", "seed": args.seed,
        "counts": {f"{g}/{s}": n for (g, s), n in sorted(counts.items())},
        "windows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: {len(rows)} ventanas")
    for k, v in out["counts"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
