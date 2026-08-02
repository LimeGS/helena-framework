"""Extract a virgin wedge -- any (winding, z-band) region -- as a synthetic
PPM in the fixed legacy frame, so the renderer treats unsegmented scroll
territory like a normal segment. Re-traces the requested band, refines each
crossing to the run centroid, emits positions + radial normals through the
published transform. Normals are radial approximations: good for detection
experiments, not final reading renders.
Usage: python wedge_extract.py --z0 5200 --z1 5600 --winding 40 --out w40.ppm
"""
import argparse, json, os, urllib.request
import numpy as np

SHM = "/dev/shm/surface_mask.u8"
SHAPE = (8398, 3941, 3941)


def ppm_header_str(width, height):
    """Header understood by the renderers' parse_ppm_header (dim=6 float64)."""
    return (f"width: {width}\nheight: {height}\ndim: 6\n"
            f"ordered: true\ntype: double\nversion: 1\n<>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z0", type=int, required=True)
    ap.add_argument("--z1", type=int, required=True)
    ap.add_argument("--winding", type=int, required=True)
    ap.add_argument("--ntheta", type=int, default=2880)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = np.memmap(SHM, dtype=np.uint8, mode="r", shape=SHAPE)
    axis = np.load("axis.npy")
    url = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
           "PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr/transform.json")
    t = json.load(urllib.request.urlopen(url, timeout=60))
    M = np.array(t["transformation_matrix"])          # moving (full-res xyz) -> fixed
    A, b = M[:, :3], M[:, 3]

    thetas = np.linspace(0, 2*np.pi, a.ntheta, endpoint=False)
    cs, sn = np.cos(thetas), np.sin(thetas)
    rr = np.arange(20, 1900, 1.0)
    H = a.z1 - a.z0
    pos = np.zeros((H, a.ntheta, 3))                  # fix frame
    nrm = np.zeros((H, a.ntheta, 3))
    got = 0
    for zi in range(a.z0, a.z1):
        cy, cx = axis[zi]
        sl = m[zi]
        ys = np.clip(np.round(cy + rr[None, :]*sn[:, None]).astype(np.int32), 0, SHAPE[1]-1)
        xs = np.clip(np.round(cx + rr[None, :]*cs[:, None]).astype(np.int32), 0, SHAPE[2]-1)
        hits = sl[ys, xs] > 0
        d = np.diff(hits.astype(np.int8), axis=1)
        for ti in range(a.ntheta):
            starts = np.where(d[ti] == 1)[0]
            ends = np.where(d[ti] == -1)[0]
            runs = []
            for s in starts:
                e = ends[ends > s]
                # run occupies rr[s+1] .. rr[e[0]] inclusive (diff offsets by one)
                r0 = rr[s + 1] if s + 1 < len(rr) else rr[s]
                runs.append((r0, rr[e[0]] if len(e) else r0 + 2))
            if len(runs) <= a.winding:
                continue
            r0, r1 = runs[a.winding]
            r = (r0 + r1) / 2                          # run centroid = refined crossing
            # tracer works at quarter scale; moving-frame full-res = quarter * 4
            y_m = (cy + r*sn[ti]) * 4.0
            x_m = (cx + r*cs[ti]) * 4.0
            z_m = zi * 4.0
            p_mov = np.array([x_m, y_m, z_m])          # (x,y,z) order, as landmarks
            n_mov = np.array([cs[ti], sn[ti], 0.0])    # radial normal
            pos[zi-a.z0, ti] = p_mov @ A.T + b
            nrm[zi-a.z0, ti] = n_mov @ A.T
            got += 1
    nn = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.where(nn > 1e-6, nrm/np.maximum(nn, 1e-6), 0)
    print(f"wedge: {H}x{a.ntheta} | coverage {got/(H*a.ntheta)*100:.1f}%", flush=True)
    data = np.concatenate([pos, nrm], axis=-1)
    with open(a.out, "wb") as f:
        f.write(ppm_header_str(a.ntheta, H).encode())
        f.write(data.astype(np.float64).tobytes())
    print(f"WEDGE_DONE {a.out} {os.path.getsize(a.out)//2**20} MiB", flush=True)


if __name__ == "__main__":
    main()
