"""Batched volumetric sweep for infer_dino3d: producer threads prefetch 256^3
patches from S3 into a queue while the GPU runs batched forwards, then paints
max-probability along each surface normal onto the flat PPM grid. Removes the
fetch-bound idle that a naive per-cell loop suffers.
"""
import argparse, os, queue, threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from infer_dino3d import (load_transform_fix2mov, parse_ppm_header,
                          fetch_chunk, norm_pminmax, build_model)

CH, P, STRIDE = 128, 256, 224
VOLSHAPE = (33592, 15761, 15761)


def fetch_patch(corner):
    ids = [(corner[0] // CH + dz, corner[1] // CH + dy, corner[2] // CH + dx)
           for dz in range(P // CH) for dy in range(P // CH) for dx in range(P // CH)]
    vol = np.zeros((P, P, P), np.uint8)
    with ThreadPoolExecutor(8) as ex:
        for key, arr in ex.map(fetch_chunk, ids):
            if arr is None:
                continue
            oz = key[0]*CH - corner[0]; oy = key[1]*CH - corner[1]; ox = key[2]*CH - corner[2]
            vol[oz:oz+CH, oy:oy+CH, ox:ox+CH] = arr
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="models1667/dino3d/ckpt_78k_fullsup.pth")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--halfdepth", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    R, t = load_transform_fix2mov()
    w, h, dim, off = parse_ppm_header(a.ppm)
    mm = np.memmap(a.ppm, dtype=np.float64, mode="r", offset=off).reshape(h, w, dim)
    blk = np.asarray(mm)
    pos_f, nrm_f = blk[..., :3], blk[..., 3:6]
    valid = (pos_f > 0).all(-1)
    pos = pos_f @ R.T + t
    nrm = nrm_f @ R.T
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    pz = pos[..., ::-1]; nz = nrm[..., ::-1]                 # xyz -> zyx
    valid &= np.all((pz >= 0) & (pz < np.array(VOLSHAPE)), axis=-1)

    cell = np.floor_divide(pz.astype(np.int64), STRIDE)
    ids = cell[..., 0]*10**10 + cell[..., 1]*10**5 + cell[..., 2]
    ids[~valid] = -1
    uniq = np.unique(ids[ids >= 0])
    # one fetch corner per cell
    corners = {}
    for u in uniq:
        sel = np.argwhere(ids == u)
        czyx = pz[sel[:, 0], sel[:, 1]].mean(0)
        c = [int(min(max(round(v) - P//2, 0), VOLSHAPE[i]-P)) for i, v in enumerate(czyx)]
        corners[u] = [x - x % CH for x in c]
    print(f"{valid.sum()} valid px, {len(uniq)} cells", flush=True)

    net = build_model(a.ckpt, "cuda")
    flat = np.full((h, w), -1.0, np.float32)
    offs = np.arange(-a.halfdepth, a.halfdepth + 1)

    q = queue.Queue(maxsize=a.workers * 2)
    SENTINEL = object()

    def producer(chunk_of_us):
        for u in chunk_of_us:
            vol = fetch_patch(corners[u])
            q.put((u, vol))

    # shard cells across producer threads
    lst = list(uniq)
    parts = [lst[i::a.workers] for i in range(a.workers)]
    threads = [threading.Thread(target=producer, args=(p,), daemon=True) for p in parts if p]
    for th in threads:
        th.start()

    def drain_done():
        return all(not th.is_alive() for th in threads) and q.empty()

    done = 0
    buf = []
    total = len(uniq)
    while done < total:
        try:
            item = q.get(timeout=120)
        except queue.Empty:
            if drain_done():
                break
            continue
        buf.append(item)
        if len(buf) >= a.batch or (drain_done() and buf):
            us = [b[0] for b in buf]
            x = np.stack([norm_pminmax(b[1]) for b in buf])[:, None]   # (B,1,256,256,256)
            xt = torch.from_numpy(x).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xt)
            logits = out["ink"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits.float())[:, 0].cpu().numpy()    # (B,256,256,256)
            for bi, u in enumerate(us):
                sel = np.argwhere(ids == u)
                corner = np.array(corners[u])
                pzs = pz[sel[:, 0], sel[:, 1]]; nzs = nz[sel[:, 0], sel[:, 1]]
                samp = pzs[:, None, :] + offs[None, :, None] * nzs[:, None, :]
                C = np.round(samp).astype(np.int64) - corner
                ok = np.all((C >= 0) & (C < P), axis=-1)
                pv = np.zeros(C.shape[:2], np.float32)
                okf = ok.reshape(-1); Cf = C.reshape(-1, 3)[okf]
                pv.reshape(-1)[okf] = prob[bi][Cf[:, 0], Cf[:, 1], Cf[:, 2]]
                flat[sel[:, 0], sel[:, 1]] = pv.max(1)
            done += len(buf)
            buf = []
            if done % 60 < a.batch:
                print(f"cell {done}/{total} qsize={q.qsize()}", flush=True)

    np.save(a.out + ".npy", flat)
    from PIL import Image
    vis = np.clip(flat, 0, 1); vis[flat < 0] = 0
    Image.fromarray((vis*255).astype(np.uint8)).save(a.out + ".png")
    got = flat[flat >= 0]
    print(f"SWEEP_FAST_DONE {a.out} coverage={len(got)/flat.size*100:.1f}% "
          f"mean={got.mean():.4f} frac>.5={(got > .5).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
