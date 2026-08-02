#!/usr/bin/env python3
"""Consolidated training script for proxy_v4 / proxy_s2s3_v1.

This reproduces the actual recipe used to train the shipped checkpoints
(the published proxy_v4.pt / proxy_s2s3_v1.pt), consolidated from the
project's own data/letters/s1_atlas/train_proxy_v4.py and
train_proxy_s2s3_v1.py. Architecture, loss, optimizer, spatial validation
split, and per-example weighting are copied as-is; only the data-loading
layer changed, to work off train_labels.jsonl + a crops directory (produced
by generate_crops.py) instead of raw panel images.

Two modes:

  --mode s1     Train proxy_v4 from an ImageNet-initialized ResNet-18, using
                the S1 rows of train_labels.jsonl (541 positives, weight
                1.0; 87 implied negatives, weight 0.7), spatially held out on
                3 panels (w046-052, w059-063, w085-088) -- identical to
                train_proxy_v4.py's VAL_PANELS.

  --mode s2s3   Fine-tune proxy_s2s3_v1 from a proxy_v4-shaped checkpoint
                (--init-checkpoint), adding the S2/S3 rows of
                train_labels.jsonl as extra negatives (23 human-reviewed
                noise, weight 1.0), at a lower learning rate -- identical to
                train_proxy_s2s3_v1.py. NOTE: the 169 "implicit" S2/S3
                negatives (weight 0.7) described in the model card cannot be
                regenerated from the files in this release -- see
                VERIFICATION.md / DATASET_CARD.md "Known limitation". This
                mode will therefore fine-tune on fewer negatives than the
                original run and should not be expected to reproduce
                proxy_s2s3_v1.pt bit-for-bit.

The original recipe also mixes in two negative sources that are NOT human
labels and so are not in train_labels.jsonl (see DATASET_CARD.md
"What's not included"):

  1. 50 GPU-verified-fiber windows (top-50 of the score-sorted full-grid
     candidate ranking, weight 1.0). These SHIP with this release as
     fiber_negatives_50.jsonl -- generate their crops and pass
     --extra-negatives fiber_negatives_50.jsonl to reproduce the full
     proxy_v4 recipe (12 of the 50 fall in the validation panels and are
     routed there, as in the original run).
  2. Procedural background negatives: a fixed-seed, content-thresholded
     random sample of ~30 windows per panel that contains a positive mark
     (weight 1.0). This needs no stored labels -- it's regenerated directly
     from the raw map images. Pass --maps-dir (the same directory given to
     generate_crops.py for S1) and it is included automatically; pass
     --no-background to disable it and train on the labeled rows alone.

Usage:
    python train.py --mode s1 \\
        --labels train_labels.jsonl --crops-dir ./crops \\
        --maps-dir /path/to/s1_official_panels \\
        --extra-negatives fiber_negatives_50.jsonl \\
        --out proxy_v4_reproduced.pt

    python train.py --mode s2s3 \\
        --labels train_labels.jsonl --crops-dir ./crops \\
        --init-checkpoint proxy_v4_reproduced.pt \\
        --out proxy_s2s3_v1_reproduced.pt
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None

SZ = 160
WIN_S1 = 512
VAL_PANELS = {
    "20260623144224-w046-052",
    "20260623145652-w059-063",
    "20260623154006-w085-088",
}


# --------------------------------------------------------------- model ----

def build_model() -> nn.Module:
    """torchvision resnet18(), fc -> Linear(512, 1). Matches proxy_v4.pt /
    proxy_s2s3_v1.pt state_dict exactly (122 tensors)."""
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(512, 1)
    return m


# --------------------------------------------------------------- data -----

class CropDataset(Dataset):
    """items: list of (crop_path, label:int, weight:float). Crops are
    pre-generated 160x160 uint8 grayscale PNGs from generate_crops.py;
    replicated to 3 channels here, same as README's score_window()."""

    def __init__(self, items, aug: bool):
        self.items = items
        self.aug = aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, weight = self.items[i]
        a = np.array(Image.open(path).convert("L"), np.float32) / 255.0
        if self.aug:
            if random.random() < 0.5:
                a = a[:, ::-1].copy()
            if random.random() < 0.5:
                a = a[::-1, :].copy()
            a = np.rot90(a, random.randrange(4)).copy()
        t = torch.from_numpy(a)[None].repeat(3, 1, 1)
        return t, torch.tensor(float(label)), torch.tensor(float(weight))


def load_rows(labels_path: str):
    return [json.loads(l) for l in open(labels_path) if l.strip()]


def sample_background_negatives(maps_dir: str, pos_by_panel: dict, seed: int = 0,
                                 win: int = WIN_S1, quota: int = 30, tries: int = 600,
                                 radius: int = 1024, content_thresh: float = 0.3):
    """Reproduces train_proxy_v4.py's neg_bg sampling exactly: fixed-seed RNG,
    one panel image loaded at a time, `quota` accepted windows per panel that
    has >=1 positive mark, rejecting windows within `radius` px of any
    positive or with <content_thresh active-pixel coverage.

    Returns list of (panel, y, x) triples. Needs the raw panel .jpg files
    (same directory as --maps-dir for generate_crops.py), NOT the crops dir,
    since acceptance depends on raw pixel content.
    """
    rng = random.Random(seed)
    out = []
    for panel in sorted(pos_by_panel):
        path = os.path.join(maps_dir, panel + ".jpg")
        im = np.array(Image.open(path).convert("L"))
        H, W = im.shape
        got, n_tries = 0, 0
        while got < quota and n_tries < tries:
            n_tries += 1
            y = rng.randrange(0, H - win, 256)
            x = rng.randrange(0, W - win, 256)
            if any(abs(y - py) <= radius and abs(x - px) <= radius for py, px in pos_by_panel[panel]):
                continue
            if (im[y:y + win, x:x + win] > 10).mean() < content_thresh:
                continue
            out.append((panel, y, x))
            got += 1
    return out


def auroc(y, s) -> float:
    """Rank-sum AUROC, no sklearn dependency -- identical formula to the one
    in train_proxy_v4.py / train_proxy_s2s3_v1.py."""
    y = np.asarray(y)
    order = np.argsort(s)
    y_sorted = y[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.arange(1, len(y_sorted) + 1)
    return float((ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------- main -----

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["s1", "s2s3"], required=True)
    ap.add_argument("--labels", default="train_labels.jsonl")
    ap.add_argument("--crops-dir", default="./crops")
    ap.add_argument("--maps-dir", default=None, help="raw S1 panel .jpg dir, for procedural background negatives")
    ap.add_argument("--no-background", action="store_true", help="disable procedural background negatives")
    ap.add_argument("--extra-negatives", default=None,
                     help="optional jsonl (same row schema) of extra negative windows + matching crops in --crops-dir")
    ap.add_argument("--init-checkpoint", default=None, help="required for --mode s2s3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="default: 6 (s1) / 4 (s2s3)")
    ap.add_argument("--lr", type=float, default=None, help="default: 3e-4 (s1) / 1e-4 (s2s3)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "s2s3" and not args.init_checkpoint:
        ap.error("--mode s2s3 requires --init-checkpoint (a proxy_v4-shaped checkpoint)")

    epochs = args.epochs or (6 if args.mode == "s1" else 4)
    lr = args.lr or (3e-4 if args.mode == "s1" else 1e-4)
    out_path = args.out or (f"proxy_v4_reproduced.pt" if args.mode == "s1" else "proxy_s2s3_v1_reproduced.pt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.labels)
    s1_rows = [r for r in rows if r["scroll"] == "s1" and r["split"] == "train"]
    s2s3_rows = [r for r in rows if r["scroll"] == "s2s3" and r["split"] == "train"]

    def crop_path(r):
        p = os.path.join(args.crops_dir, f"{r['id']}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing crop {p} -- run generate_crops.py first")
        return p

    # ---- S1 positives + implied negatives (identical to both v4 and s2s3 fine-tune) ----
    train_items, val_items = [], []
    pos_by_panel = {}
    for r in s1_rows:
        label = 1 if r["label"] == "positive" else 0
        item = (crop_path(r), label, r["weight"])  # keep raw weight; (1+0.5*y) applied in loss
        if r["panel_or_segment"] in VAL_PANELS:
            val_items.append(item)
        else:
            train_items.append(item)
        if r["label"] == "positive":
            pos_by_panel.setdefault(r["panel_or_segment"], []).append((r["y"], r["x"]))

    n_labeled_train = len(train_items)

    if not args.no_background:
        # Both train_proxy_v4.py and train_proxy_s2s3_v1.py rebuild this same
        # S1 background-negative set (the s2s3 fine-tune keeps seeing S1
        # background examples so it doesn't forget the base task). Note the
        # two original scripts don't draw identical windows even at the same
        # seed=0 -- train_proxy_s2s3_v1.py advances its RNG with an earlier
        # `rng.shuffle(neg_s2s3)` call first, so its background sample is a
        # different draw from the same seed than train_proxy_v4.py's. This
        # script does not attempt to replicate that exact call ordering; it
        # only guarantees a shape-correct, seeded, reproducible-with-itself
        # background sample.
        if not args.maps_dir:
            raise SystemExit("--maps-dir is required for procedural background negatives "
                              "(or pass --no-background to skip them, but AUROC will not match the "
                              "documented recipe)")
        bg = sample_background_negatives(args.maps_dir, pos_by_panel, seed=args.seed)
        n_bg_train = n_bg_val = 0
        for panel, y, x in bg:
            crop_id = f"bg_{panel}_y{y}_x{x}"
            crop_out = os.path.join(args.crops_dir, f"{crop_id}.png")
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
            item = (crop_out, 0, 1.0)
            if panel in VAL_PANELS:
                val_items.append(item)
                n_bg_val += 1
            else:
                train_items.append(item)
                n_bg_train += 1
        print(f"background negatives: {n_bg_train} train / {n_bg_val} val (seed={args.seed})")

    if args.extra_negatives:
        extra = load_rows(args.extra_negatives)
        n_ex_val = 0
        for r in extra:
            item = (crop_path(r), 0, r.get("weight", 1.0))
            # same spatial routing as every other row: val panels go to val,
            # exactly as the original recipe did with its 50 fiber negatives
            # (12 of which fall in VAL_PANELS and shape the reported AUROC)
            if r["panel_or_segment"] in VAL_PANELS:
                val_items.append(item)
                n_ex_val += 1
            else:
                train_items.append(item)
        print(f"extra negatives from {args.extra_negatives}: {len(extra)} ({n_ex_val} to val)")

    # ---- S2/S3 negatives (only used in s2s3 fine-tune mode, train split only) ----
    if args.mode == "s2s3":
        for r in s2s3_rows:
            train_items.append((crop_path(r), 0, r["weight"]))
        print(f"S2/S3 negatives added to train: {len(s2s3_rows)} "
              f"(NOTE: original recipe used 192 -- see DATASET_CARD.md 'Known limitation')")

    print(f"labeled S1 rows: {n_labeled_train} train (spatial holdout: {sorted(VAL_PANELS)})")
    print(f"final: {len(train_items)} train ({sum(l for _, l, _ in train_items)} positive) "
          f"| {len(val_items)} val ({sum(l for _, l, _ in val_items)} positive)")

    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model()
    if args.mode == "s2s3":
        sd = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(sd)
        print(f"initialized from {args.init_checkpoint}")
    model = model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    train_loader = DataLoader(CropDataset(train_items, aug=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CropDataset(val_items, aug=False), batch_size=64, shuffle=False) if val_items else None

    best_auroc = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            opt.zero_grad()
            out = model(xb).squeeze(1)
            loss = (bce(out, yb) * wb * (1 + 0.5 * yb)).mean()
            loss.backward()
            opt.step()

        if val_loader is None:
            torch.save(model.state_dict(), out_path)
            print(f"epoch {ep + 1}: no val panels present in this crop set, saved unconditionally")
            continue

        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                s = torch.sigmoid(model(xb.to(dev)).squeeze(1)).cpu().numpy()
                ss += list(s)
                ys += list(yb.numpy())
        a = auroc(ys, ss)
        print(f"epoch {ep + 1}: AUROC val (held-out panels) {a:.3f}")
        if a > best_auroc:
            best_auroc = a
            torch.save(model.state_dict(), out_path)

    print(f"done. best val AUROC {best_auroc:.3f} -> {out_path}")


if __name__ == "__main__":
    main()
