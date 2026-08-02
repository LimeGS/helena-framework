"""Inference adapter for the PHerc.1667 ResNet3D family (HF, trust_remote_code).
Input geometry (B,1,62,256,256); normalization uint8 clip[0,200]/255 exactly
as in the paper -- plain /255 shifts output means measurably. Gaussian-blended
tiling, stride 128, tile validity threshold 0.75.
Usage: python infer_resnet3d_1667.py --stack stack.u8 --shape 62,H,W \
         --model models/iteration-5 --out pred.png
"""
import argparse, os
import numpy as np


def preprocess_tile(tile):
    """The paper's normalization: uint8 clip [0,200], then /255 (NOT /200)."""
    return np.clip(tile.astype(np.float32), 0, 200) / 255.0


def gaussian_kernel(n, sigma_frac=0.25):
    ax = np.arange(n) - (n - 1) / 2
    g = np.exp(-(ax**2) / (2 * (n * sigma_frac) ** 2))
    return np.outer(g, g).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--shape", required=True)       # "62,H,W"
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--min-valid", type=float, default=0.75)
    a = ap.parse_args()

    import torch
    D, H, W = map(int, a.shape.split(","))
    vol = np.memmap(a.stack, dtype=np.uint8, mode="r", shape=(D, H, W))
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True)
    model = model.cuda().eval()

    T, S = a.tile, a.stride
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    gk = gaussian_kernel(T)
    mid = vol[D // 2]
    coords = [(y, x) for y in range(0, H - T + 1, S) for x in range(0, W - T + 1, S)
              if (np.asarray(mid[y:y+T, x:x+T]) > 0).mean() >= a.min_valid]
    print(f"{len(coords)} tiles", flush=True)
    if not coords:
        raise SystemExit("no tiles pass --min-valid; check the stack and shape")
    buf, pos = [], []
    with torch.no_grad():
        for i, (y, x) in enumerate(coords):
            buf.append(preprocess_tile(np.asarray(vol[:, y:y+T, x:x+T])))
            pos.append((y, x))
            if len(buf) == a.batch or i == len(coords) - 1:
                xb = torch.from_numpy(np.stack(buf))[:, None].cuda()
                with torch.autocast("cuda"):
                    out = model(pixel_values=xb)
                prob = torch.sigmoid(out.logits.float())
                prob = torch.nn.functional.interpolate(prob, size=(T, T), mode="bilinear")
                pb = prob[:, 0].cpu().numpy()
                for (yy, xx), p in zip(pos, pb):
                    acc[yy:yy+T, xx:xx+T] += p * gk
                    cnt[yy:yy+T, xx:xx+T] += gk
                buf, pos = [], []
            if i % 500 == 0:
                print(f"tile {i}/{len(coords)}", flush=True)
    res = np.where(cnt > 0, acc / np.maximum(cnt, 1e-6), 0)
    res[np.asarray(mid) == 0] = 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    from PIL import Image
    Image.fromarray((res * 255).astype(np.uint8)).save(a.out)
    print(f"INFER1667_DONE {a.out} mean {res[res>0].mean()*255:.1f}", flush=True)


if __name__ == "__main__":
    main()
