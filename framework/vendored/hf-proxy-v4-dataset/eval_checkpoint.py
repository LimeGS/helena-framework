#!/usr/bin/env python3
"""Frozen evaluator for the published proxy_v4.pt.

Deterministically reconstructs the validation split the training recipe
defines (the three held-out panels: labeled rows routed by panel, the
val-panel fiber negatives, and the seed-0 procedural background negatives)
and scores the published checkpoint on it.

Expected output, verified 2026-07-11 against the released proxy_v4.pt:

    val: 81 positives / 123 negatives
    AUROC 0.9853457794

Usage:
    python eval_checkpoint.py --checkpoint /path/to/proxy_v4.pt \
        --maps-dir /path/to/s1_official_panels --crops-dir ./crops

Crops for the val rows must exist (run generate_crops.py on both
train_labels.jsonl and fiber_negatives_50.jsonl first); background-negative
crops are generated on the fly from --maps-dir, same as train.py.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train import (VAL_PANELS, WIN_S1, SZ, CropDataset, auroc, load_rows,
                   sample_background_negatives)
from torch.utils.data import DataLoader
import torch.nn as nn

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="proxy_v4.pt")
    ap.add_argument("--labels", default=os.path.join(HERE, "train_labels.jsonl"))
    ap.add_argument("--fiber", default=os.path.join(HERE, "fiber_negatives_50.jsonl"))
    ap.add_argument("--maps-dir", required=True)
    ap.add_argument("--crops-dir", default="./crops")
    args = ap.parse_args()

    def crop_path(r):
        p = os.path.join(args.crops_dir, f"{r['id']}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing crop {p} -- run generate_crops.py first")
        return p

    val_items = []
    pos_by_panel = {}
    for r in load_rows(args.labels):
        if r["scroll"] != "s1" or r["split"] != "train":
            continue
        if r["label"] == "positive":
            pos_by_panel.setdefault(r["panel_or_segment"], []).append((r["y"], r["x"]))
        if r["panel_or_segment"] in VAL_PANELS:
            val_items.append((crop_path(r), 1 if r["label"] == "positive" else 0, r["weight"]))
    for r in load_rows(args.fiber):
        if r["panel_or_segment"] in VAL_PANELS:
            val_items.append((crop_path(r), 0, r["weight"]))

    for panel, y, x in sample_background_negatives(args.maps_dir, pos_by_panel, seed=0):
        if panel not in VAL_PANELS:
            continue
        crop_out = os.path.join(args.crops_dir, f"bg_{panel}_y{y}_x{x}.png")
        if not os.path.exists(crop_out):
            im = np.array(Image.open(os.path.join(args.maps_dir, panel + ".jpg")).convert("L"))
            c = im[y:y + WIN_S1, x:x + WIN_S1].astype(np.float32)
            active = c[c > 10]
            if len(active) > 50:
                lo, hi = np.percentile(active, [2, 99.5])
                c = np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                c = c / 255.0
            Image.fromarray((c * 255).astype(np.uint8)).resize((SZ, SZ), Image.BILINEAR).save(crop_out)
        val_items.append((crop_out, 0, 1.0))

    n_pos = sum(l for _, l, _ in val_items)
    print(f"val: {n_pos} positives / {len(val_items) - n_pos} negatives")

    m = torchvision.models.resnet18()
    m.fc = nn.Linear(512, 1)
    m.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    m.eval()

    ys, ss = [], []
    with torch.no_grad():
        for xb, yb, _ in DataLoader(CropDataset(val_items, aug=False), batch_size=64):
            ss += list(torch.sigmoid(m(xb).squeeze(1)).numpy())
            ys += list(yb.numpy())
    print(f"AUROC {auroc(ys, ss):.10f}")


if __name__ == "__main__":
    main()
