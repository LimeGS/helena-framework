"""Scroll 2 (PHercParis3) variant of the chunk-gather renderer, targeting
the April-2026 2.400 um rescan. Only the volume key, shape and matrix-derived
spacing differ from render_scroll3.py -- diff the two files to adapt the
renderer to any new masked OME-Zarr release.
"""
import argparse, json, os, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOLKEY = "PHercParis3/volumes/20260427095331-2.400um-0.2m-78keV-masked.zarr"
NOMINAL_SPACING = 7.91 / 2.400   # voxel-size ratio, for reference only
CH = 128
VOLSHAPE = (68417, 42403, 42403)          # z,y,x from the .zarray metadata
GRID = tuple(-(-s // CH) for s in VOLSHAPE)


def load_transform():
    with urllib.request.urlopen(f"{BUCKET}/{VOLKEY}/transform.json", timeout=60) as f:
        t = json.load(f)
    M = np.array(t["transformation_matrix"])
    Ah = np.eye(4); Ah[:3, :3] = M[:, :3]; Ah[:3, 3] = M[:, 3]
    inv = np.linalg.inv(Ah)                # published matrix is mov->fix; invert
    fix = np.array(t["fixed_landmarks"]); mov = np.array(t["moving_landmarks"])
    err = np.abs(fix @ inv[:3, :3].T + inv[:3, 3] - mov).mean()
    print(f"transform fix->mov landmark_err={err:.1f}", flush=True)
    assert err < 60, "transform does not match landmarks"
    # matrix-derived isotropic scale: moving voxels per fixed voxel.
    # this differs from the nominal voxel-size ratio (see README) and is the
    # correct per-layer step for reproducing legacy layer spacing.
    scale = float(np.cbrt(abs(np.linalg.det(inv[:3, :3]))))
    return inv[:3, :3], inv[:3, 3], scale


def parse_ppm_header(path):
    with open(path, "rb") as f:
        head = f.read(400)
    sep = head.find(b"<>\n")
    assert sep > 0, f"no PPM header sentinel in first 400 bytes of {path}"
    fields = dict(l.split(": ") for l in head[:sep].decode().strip().splitlines())
    return int(fields["width"]), int(fields["height"]), int(fields["dim"]), sep + 3


def fetch_chunk(key):
    url = f"{BUCKET}/{VOLKEY}/0/{key[0]}/{key[1]}/{key[2]}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                buf = r.read()
            return key, np.frombuffer(buf, np.uint8).reshape(CH, CH, CH)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return key, None           # empty chunk (fill 0)
            import time; time.sleep(2 * (attempt + 1))
        except Exception:
            import time; time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"unrecoverable chunk {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stripe", type=int, default=640)
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--pilot-stripe", type=int, default=-1)
    ap.add_argument("--max-gb", type=float, default=14.0)
    ap.add_argument("--layers", type=int, default=26)
    ap.add_argument("--spacing", type=float, default=None,
                    help="layer step in target voxels; default: derived from "
                         "the transform matrix (legacy spacing). use 1.0 for "
                         "native-resolution steps")
    a = ap.parse_args()

    w, h, dim, off = parse_ppm_header(a.ppm)
    print(f"ppm {w}x{h}", flush=True)
    R, t, mat_spacing = load_transform()
    if a.spacing is None:
        a.spacing = mat_spacing
    print(f"layer spacing: {a.spacing:.4f} target voxels "
          f"(matrix-derived {mat_spacing:.4f}, nominal {NOMINAL_SPACING:.4f})", flush=True)
    mm = np.memmap(a.ppm, dtype=np.float64, mode="r", offset=off).reshape(h, w, dim)

    os.makedirs(f"{a.out}/layers", exist_ok=True)
    # resume guard: a checkpointed run must be resumed with identical geometry,
    # otherwise stripe indices and the memmap layout silently mean different data
    metapath = f"{a.out}/render_meta.json"
    meta = {"layers": a.layers, "stripe": a.stripe, "w": w, "h": h,
            "spacing": round(a.spacing, 6)}
    if os.path.exists(metapath):
        prev = json.load(open(metapath))
        assert prev == meta, f"resume geometry mismatch: {prev} vs {meta}"
    else:
        json.dump(meta, open(metapath, "w"))
    outpath = f"{a.out}/stack.u8"
    expected = a.layers * h * w
    if os.path.exists(outpath):
        assert os.path.getsize(outpath) == expected, \
            f"stack.u8 is {os.path.getsize(outpath)} bytes, expected {expected}"
    out = np.memmap(outpath, dtype=np.uint8, mode="r+" if os.path.exists(outpath) else "w+",
                    shape=(a.layers, h, w))
    ckpath = f"{a.out}/done.json"
    done = set(json.load(open(ckpath))) if os.path.exists(ckpath) else set()

    stripes = list(range(0, w, a.stripe))
    pending = [i for i in range(len(stripes))
               if i not in done and (a.pilot_stripe < 0 or i == a.pilot_stripe)]
    print(f"{len(stripes)} stripes, {len(pending)} pending", flush=True)
    total_bytes = 0

    for i, x0 in enumerate(stripes):
        if i not in pending:
            continue
        x1 = min(x0 + a.stripe, w)
        blk = np.asarray(mm[:, x0:x1, :])
        pos, nrm = blk[..., :3], blk[..., 3:6]
        valid = (pos > 0).all(-1)
        p = pos @ R.T + t
        n = nrm @ R.T
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        # sample coords (layers,H,S,3), zyx order
        ks = (np.arange(a.layers) - (a.layers - 1) / 2) * a.spacing
        C = np.round(p[None] + ks[:, None, None, None] * n[None]).astype(np.int64)
        C = C[..., [2, 1, 0]]
        inb = valid[None] & np.all((C >= 0) & (C < np.array(VOLSHAPE)), axis=-1)
        cid = C // CH
        lin = (cid[..., 0] * GRID[1] + cid[..., 1]) * GRID[2] + cid[..., 2]
        lin_v = lin[inb]
        uniq = np.unique(lin_v)
        gb = len(uniq) * CH**3 / 1e9
        print(f"stripe {i}: {len(uniq)} chunks = {gb:.2f} GB", flush=True)
        if gb > a.max_gb:
            print(f"ABORT stripe: budget {gb:.1f} GB > cap {a.max_gb}", flush=True)
            raise SystemExit(2)
        keys = [(int(u // (GRID[1] * GRID[2])), int((u // GRID[2]) % GRID[1]), int(u % GRID[2]))
                for u in uniq]
        store = {}
        with ThreadPoolExecutor(a.conc) as ex:
            for key, arr in ex.map(fetch_chunk, keys):
                kl = (key[0] * GRID[1] + key[1]) * GRID[2] + key[2]
                store[kl] = arr
        total_bytes += len(uniq) * CH**3
        res = np.zeros(C.shape[:-1], np.uint8)
        offs = C - cid * CH
        flat_lin = lin.reshape(-1); flat_off = offs.reshape(-1, 3)
        flat_in = inb.reshape(-1); flat_res = res.reshape(-1)
        order_idx = np.argsort(flat_lin, kind="stable")
        sl = flat_lin[order_idx]
        bounds = np.searchsorted(sl, uniq)
        bounds = np.append(bounds, len(sl))
        for ui, u in enumerate(uniq):
            seg = order_idx[bounds[ui]:bounds[ui + 1]]
            seg = seg[flat_in[seg]]
            if not len(seg):
                continue
            arr = store[u]
            if arr is None:
                continue
            o = flat_off[seg]
            flat_res[seg] = arr[o[:, 0], o[:, 1], o[:, 2]]
        out[:, :, x0:x1] = res
        del store
        done.add(i)
        json.dump(sorted(done), open(ckpath, "w"))
        print(f"stripe {i} OK (cumulative {total_bytes/1e9:.1f} GB)", flush=True)

    if a.pilot_stripe < 0 and len(done) == len(stripes):
        from PIL import Image
        for k in range(a.layers):
            Image.fromarray(np.asarray(out[k])).save(f"{a.out}/layers/{k:02d}.tif")
        seg = os.path.basename(a.out)
        Image.fromarray(((np.asarray(out[a.layers//2]) > 0) * 255).astype(np.uint8)).save(f"{a.out}/{seg}_mask.png")
        print("RENDER_DONE", flush=True)
    else:
        print("PILOT_STRIPE_DONE", flush=True)


if __name__ == "__main__":
    main()
