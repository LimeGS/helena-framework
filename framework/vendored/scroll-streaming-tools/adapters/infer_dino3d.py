"""Adapter for scrollprize/ink_3d_dino_guided: volumetric nnU-Net (142M),
256^3 raw patches, no flattening. Builds the network from the checkpoint's
embedded config via the vesuvius package, loads the EMA weights (verified
0 missing / 0 unexpected), normalizes with percentile 1/99 min-max over valid
voxels. Includes patch-group evaluation with an air control.
"""
import argparse, json, os, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOLKEY = "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr"
CH = 128
VOLSHAPE = (33592, 15761, 15761)
GRID = tuple(-(-s // CH) for s in VOLSHAPE)
P = 256


def load_transform_fix2mov():
    with urllib.request.urlopen(f"{BUCKET}/{VOLKEY}/transform.json", timeout=60) as f:
        t = json.load(f)
    M = np.array(t["transformation_matrix"])
    Ah = np.eye(4); Ah[:3, :3] = M[:, :3]; Ah[:3, 3] = M[:, 3]
    inv = np.linalg.inv(Ah)
    return inv[:3, :3], inv[:3, 3]


def parse_ppm_header(path):
    with open(path, "rb") as f:
        head = f.read(400)
    sep = head.find(b"<>\n")
    fields = dict(l.split(": ") for l in head[:sep].decode().strip().splitlines())
    return int(fields["width"]), int(fields["height"]), int(fields["dim"]), sep + 3


def sample_ppm_centers(path, k, R, t):
    w, h, dim, off = parse_ppm_header(path)
    mm = np.memmap(path, dtype=np.float64, mode="r", offset=off).reshape(h, w, dim)
    xs = np.linspace(w * 0.08, w * 0.92, k).astype(int)
    out = []
    for x in xs:
        col = mm[:, x, :3]
        val = np.where((col > 0).all(-1))[0]
        if not len(val):
            continue
        y = val[len(val) // 2]
        p_fix = col[y]                      # (x,y,z) frame fijo
        p_mov = p_fix @ R.T + t             # (x,y,z) frame mov (2.4um)
        out.append((float(p_mov[2]), float(p_mov[1]), float(p_mov[0])))  # zyx
    return out


def fetch_chunk(key):
    url = f"{BUCKET}/{VOLKEY}/0/{key[0]}/{key[1]}/{key[2]}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return key, np.frombuffer(r.read(), np.uint8).reshape(CH, CH, CH)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return key, None
            import time; time.sleep(2 * (attempt + 1))
        except Exception:
            import time; time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"unrecoverable chunk {key}")


def fetch_patch(czyx):
    corner = [min(max(int(c) - P // 2, 0), VOLSHAPE[i] - P) for i, c in enumerate(czyx)]
    corner = [c - c % CH for c in corner]                    # align to 128-chunk grid
    ids = [(corner[0] // CH + dz, corner[1] // CH + dy, corner[2] // CH + dx)
           for dz in range(P // CH) for dy in range(P // CH) for dx in range(P // CH)]
    vol = np.zeros((P, P, P), np.uint8)
    with ThreadPoolExecutor(8) as ex:
        for key, arr in ex.map(fetch_chunk, ids):
            if arr is None:
                continue
            oz = key[0] * CH - corner[0]; oy = key[1] * CH - corner[1]; ox = key[2] * CH - corner[2]
            vol[oz:oz+CH, oy:oy+CH, ox:ox+CH] = arr
    return vol, corner


def norm_pminmax(vol):
    x = vol.astype(np.float32)
    valid = x[x > 0]
    if len(valid) < 1000:
        return np.zeros_like(x)
    lo, hi = np.percentile(valid, 1.0), np.percentile(valid, 99.0)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


class Mgr:
    pass


def build_model(ck_path, dev):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mgr = Mgr()
    mgr.targets = cfg["targets"]
    mgr.train_patch_size = cfg["patch_size"]
    mgr.train_batch_size = 1
    mgr.in_channels = cfg.get("in_channels", 1)
    mgr.model_config = cfg.get("model_config") or {}
    mgr.autoconfigure = mgr.model_config.get("autoconfigure", True)
    mgr.enable_deep_supervision = False
    mgr.spacing = [1, 1, 1]
    mgr.model_name = cfg.get("wandb_run_name", "dino3d")
    mgr.op_dims = 3
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    net = NetworkFromConfig(mgr)
    sd = ck["ema_model"]
    sd = { (k[10:] if k.startswith("_orig_mod.") else k).removeprefix("module."): v
           for k, v in sd.items() }
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"state: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing[:3]:
        print("  missing[:3]:", missing[:3], flush=True)
    if unexpected[:3]:
        print("  unexpected[:3]:", unexpected[:3], flush=True)
    assert len(missing) == 0, "state_dict does not match the built architecture"
    net.eval().to(dev)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="out_dino3d")
    ap.add_argument("--ckpt", default="models1667/dino3d/ckpt_78k_fullsup.pth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    R, t = load_transform_fix2mov()
    groups = {
        "prime1": sample_ppm_centers("ppm/20240716140050.ppm", a.k, R, t),
        "wedge": sample_ppm_centers("ppm/wedge_w40_z5200.ppm", a.k, R, t),
        "air": [(1000.0, 1000.0, 1000.0), (32000.0, 14000.0, 14000.0)],
    }
    net = build_model(a.ckpt, dev)
    from PIL import Image
    rows = []
    for g, centers in groups.items():
        for i, c in enumerate(centers):
            vol, corner = fetch_patch(c)
            x = norm_pminmax(vol)
            xt = torch.from_numpy(x)[None, None].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xt)
            logits = out["ink"] if isinstance(out, dict) else out
            prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
            m, p99, frac = prob.mean(), np.percentile(prob, 99), (prob > 0.5).mean()
            rows.append((g, i, corner, m, p99, frac))
            print(f"{g}[{i}] corner={corner} mean={m:.4f} p99={p99:.4f} frac>.5={frac:.4f}", flush=True)
            mid = P // 2
            ct = (x[mid] * 255).astype(np.uint8)
            pr = (prob[mid] * 255).astype(np.uint8)
            panel = np.concatenate([ct, np.full((P, 8), 255, np.uint8), pr], axis=1)
            Image.fromarray(panel).save(f"{a.out}/{g}_{i:02d}.png")
    print("== summary ==", flush=True)
    for g in groups:
        ms = [r[3] for r in rows if r[0] == g]
        fr = [r[5] for r in rows if r[0] == g]
        if ms:
            print(f"{g}: mean(prob)={np.mean(ms):.4f} frac>.5={np.mean(fr):.4f} n={len(ms)}", flush=True)
    print("DINO3D_E13_DONE", flush=True)


if __name__ == "__main__":
    main()
