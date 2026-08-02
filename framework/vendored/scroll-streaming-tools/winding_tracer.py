"""Radial ray-casting winding tracer over the community surface-prediction
zarr for Scroll 3 (quarter scale). Decompresses the mask once to SHM_PATH,
estimates the scroll axis per slice, casts (z, theta) rays and indexes every
sheet crossing into r(z, theta, winding). --pilot validates against a
ground-truth segment mesh (median radial error 3.42 vox in our runs).
Usage: python winding_tracer.py [--pilot] [--procs 48] [--ntheta 1440]
"""
import argparse, os
import multiprocessing as mp
import numpy as np

# --- local setup: adjust these three for your copy of the surface predictions ---
SHM = "/dev/shm/surface_mask.u8"   # decompressed mask cache (see ensure_shm)
SHAPE = (8398, 3941, 3941)         # quarter-scale surface-prediction zarr shape
MAXW = 120                         # max windings recorded per ray


def ensure_shm():
    if os.path.exists(SHM) and os.path.getsize(SHM) == np.prod(SHAPE):
        return
    import zarr
    print("decompressing mask to shm (once)...", flush=True)
    z = zarr.open("preds/surface.zarr", mode="r")
    arr = z["0"] if hasattr(z, "keys") and "0" in z else z
    out = np.memmap(SHM, dtype=np.uint8, mode="w+", shape=SHAPE)
    B = 256
    for z0 in range(0, SHAPE[0], B):
        out[z0:z0+B] = arr[z0:z0+B]
        if (z0 // B) % 8 == 0:
            print(f"  shm {z0}/{SHAPE[0]}", flush=True)
    out.flush()
    print("shm ready", flush=True)


_G = {}

def _init(axis):
    _G["m"] = np.memmap(SHM, dtype=np.uint8, mode="r", shape=SHAPE)
    _G["axis"] = axis


def _trace_zband(args):
    z0, z1, ntheta, rmax = args
    m, axis = _G["m"], _G["axis"]
    thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    cs, sn = np.cos(thetas), np.sin(thetas)
    rr = np.arange(20, rmax, 1.0)
    out = np.full((z1-z0, ntheta, MAXW), -1, np.float32)
    for zi in range(z0, z1):
        cy, cx = axis[zi]
        sl = m[zi]
        ys = np.clip(np.round(cy + rr[None, :]*sn[:, None]).astype(np.int32), 0, SHAPE[1]-1)
        xs = np.clip(np.round(cx + rr[None, :]*cs[:, None]).astype(np.int32), 0, SHAPE[2]-1)
        hits = sl[ys, xs] > 0                      # (ntheta, len(rr))
        d = np.diff(hits.astype(np.int8), axis=1)
        for ti in range(ntheta):
            starts = np.where(d[ti] == 1)[0]
            ends = np.where(d[ti] == -1)[0]
            n = 0
            for s in starts:
                e = ends[ends > s]
                # run occupies rr[s+1] .. rr[e[0]] inclusive (diff offsets by one)
                r0 = rr[s + 1] if s + 1 < len(rr) else rr[s]
                mid = (r0 + rr[e[0]]) / 2 if len(e) else r0 + 1.0
                if n < MAXW:
                    out[zi-z0, ti, n] = mid
                    n += 1
    return z0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--procs", type=int, default=54)
    ap.add_argument("--ntheta", type=int, default=720)
    ap.add_argument("--zband", type=int, default=64)
    a = ap.parse_args()
    ensure_shm()
    m = np.memmap(SHM, dtype=np.uint8, mode="r", shape=SHAPE)

    # per-slice axis (mask centroid, subsampled + smoothed)
    print("estimating axis...", flush=True)
    axis = np.zeros((SHAPE[0], 2), np.float32)
    for z in range(0, SHAPE[0], 16):
        sl = m[z, ::4, ::4]
        ys, xs = np.nonzero(sl)
        if len(ys) > 100:
            axis[z] = (ys.mean()*4, xs.mean()*4)
    # interpolate gaps and smooth
    for c in range(2):
        v = axis[::16, c]
        good = v > 0
        idx = np.arange(len(v))
        v = np.interp(idx, idx[good], v[good]) if good.any() else v
        k = np.ones(9)/9
        v = np.convolve(v, k, mode="same")
        axis[:, c] = np.interp(np.arange(SHAPE[0]), np.arange(0, SHAPE[0], 16), v)
    np.save("axis.npy", axis)
    print("axis ready; sample z=3500:", axis[3500], flush=True)

    if a.pilot:
        pts = np.load("mesh_pts_quarter.npz")["zyx"]
        z0, z1 = int(pts[:, 0].min()), int(pts[:, 0].max())+1
        print(f"pilot z {z0}..{z1}", flush=True)
    else:
        z0, z1 = 0, SHAPE[0]

    rmax = 1900
    bands = [(b, min(b+a.zband, z1), a.ntheta, rmax) for b in range(z0, z1, a.zband)]
    ctx = mp.get_context("fork")   # workers touch only memmap+numpy: fork is safe and avoids re-reading
    R = np.full((z1-z0, a.ntheta, MAXW), -1, np.float32)
    with ctx.Pool(a.procs, initializer=_init, initargs=(axis,)) as pool:
        done = 0
        for zb, out in pool.imap_unordered(_trace_zband, bands):
            R[zb-z0:zb-z0+out.shape[0]] = out
            done += 1
            if done % 5 == 0:
                print(f"{done}/{len(bands)} z-bands", flush=True)
    np.save("rmap_pilot.npy" if a.pilot else "rmap_full.npy", R)
    print("rmap saved", R.shape, flush=True)

    if a.pilot:
        # metric: radial distance from each mesh point to nearest winding
        pts = np.load("mesh_pts_quarter.npz")["zyx"]
        sel = np.random.default_rng(0).choice(len(pts), 30000, replace=False)
        errs = []
        for p in pts[sel]:
            zi = int(round(p[0])) - z0
            if not (0 <= zi < R.shape[0]):
                continue
            cy, cx = axis[int(round(p[0]))]
            dy, dx = p[1]-cy, p[2]-cx
            r = np.hypot(dy, dx)
            th = (np.arctan2(dy, dx) % (2*np.pi)) / (2*np.pi) * a.ntheta
            ws = R[zi, int(th) % a.ntheta]
            ws = ws[ws > 0]
            if len(ws):
                errs.append(np.abs(ws - r).min())
        errs = np.array(errs)
        print(f"PILOT: n={len(errs)} | median radial err {np.median(errs):.2f} vox "
              f"| p90 {np.percentile(errs,90):.2f} | <2vox: {(errs<2).mean()*100:.1f}%", flush=True)
    print("TRACER_DONE", flush=True)


if __name__ == "__main__":
    main()
