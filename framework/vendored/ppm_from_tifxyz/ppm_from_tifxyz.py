"""tifxyz -> PPM bridge: converts a segment's official mesh
(mesh/{x,y,z}.tif at scale 0.05, in the original volume's frame) into a standard
PPM, so the SAME segment can be re-rendered through this pipeline and a "render
gap" told apart from a "physical gap".

Normals: cross product of the position gradients (dP/du x dP/dv).

Usage:
    python3 ppm_from_tifxyz.py --mesh-dir seg_mesh --scale 0.05 \
        --width 9660 --height 11720 --out seg.ppm
"""
import argparse

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", required=True)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", default=None, help="u0,v0,U,V in tif-grid coordinates (for windows)")
    ap.add_argument("--upsample", type=float, default=None, help="factor total tif->salida (default: 1/scale)")
    a = ap.parse_args()

    xyz = []
    for c in ("x", "y", "z"):
        im = np.array(Image.open(f"{a.mesh_dir}/{c}.tif"), np.float32)
        xyz.append(im)
    P = np.stack(xyz, -1)
    if a.crop:
        u0, v0, U, V = map(int, a.crop.split(","))
        P = P[v0:v0+V, u0:u0+U]
    valid_small = (P > -0.5).all(-1)
    print(f"malla {P.shape} valid={valid_small.mean():.2f}", flush=True)

    # upsample bilineal
    f = a.upsample if a.upsample else 1.0 / a.scale
    H, W = int(P.shape[0] * f), int(P.shape[1] * f)
    Pt = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        Pt[..., k] = np.array(
            Image.fromarray(P[..., k]).resize((W, H), Image.BILINEAR))
    vmask = np.array(Image.fromarray(valid_small.astype(np.uint8) * 255)
                     .resize((W, H), Image.NEAREST)) > 127
    Pt[~vmask] = 0

    # normals from the gradient
    du = np.gradient(Pt, axis=1)
    dv = np.gradient(Pt, axis=0)
    N = np.cross(du, dv)
    nn = np.linalg.norm(N, axis=-1, keepdims=True)
    N = np.where(nn > 1e-6, N / np.maximum(nn, 1e-6), 0)
    N[~vmask] = 0

    data = np.concatenate([Pt, N], -1).astype(np.float64)
    header = (f"width: {W}\nheight: {H}\ndim: 6\n"
              f"ordered: true\ntype: double\nversion: 1\n<>\n")
    with open(a.out, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())
    print(f"PPM_OK {a.out} ({W}x{H})", flush=True)


if __name__ == "__main__":
    main()
